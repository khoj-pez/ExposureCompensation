import random
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn

from PIL import Image
import numpy as np
from tqdm import tqdm

from models.mlp import FCNet
from models.pe import PE
from models.grid import DenseNet, InstantNGP
from models.exp import ExposureCorrectedNet

def adjust_exposure(image, exposure):
    image = image * exposure
    image = torch.clamp(image, 0, 1)
    return image

def get_coords(res, normalize = False):
    x = y = torch.arange(res)
    xx, yy = torch.meshgrid(x, y)
    coords = torch.stack([xx, yy], dim=-1)
    if normalize:
        coords = coords / (res - 1)
    return coords

def get_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

def reconstruct_from_patches(patches, original_image_size):
    patch_size = patches[0].shape[1]
    reconstructed_image = torch.zeros((original_image_size[0], original_image_size[1], 3), device=patches[0].device)
    
    idx = 0
    for i in range(0, original_image_size[0], patch_size):
        for j in range(0, original_image_size[1], patch_size):
            reconstructed_image[i:i+patch_size, j:j+patch_size] = patches[idx]
            idx += 1
            
    return reconstructed_image

def tensor_to_image(tensor):
    img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    return img

def get_overlapping_patches(img_np, patch_size, overlap=0):
    patches = []
    stride = patch_size - overlap
    for i in range(0, img_np.shape[0] - patch_size + 1, stride):
        for j in range(0, img_np.shape[1] - patch_size + 1, stride):
            patch = img_np[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return patches

class ImageDataset(Dataset):
    def __init__(self, image_path, image_size, device = torch.device('cpu')):
        self.image = Image.open(image_path).resize(image_size)
        self.rgb_vals_gt = torch.from_numpy(np.array(self.image)).reshape(-1, 3).to(device)
        self.rgb_vals_gt = self.rgb_vals_gt.float() / 255

        self.patches = get_overlapping_patches(np.array(self.image), 64)
        self.exposures = torch.tensor([random.uniform(1, 2) for _ in self.patches]).unsqueeze(-1).to(device)
        self.patches = [adjust_exposure(torch.from_numpy(patch).float().to(device)/255, exposure) for patch, exposure in zip(self.patches, self.exposures)]
        self.reconstructed_image = tensor_to_image(reconstruct_from_patches(self.patches, image_size))

        # self.latent_codes = nn.ParameterList([nn.Parameter(torch.randn(256, 1).to(device)) for _ in self.patches])
        self.latent_scalars = nn.ParameterList([nn.Parameter(torch.tensor(random.uniform(1, 2)).to(device)) for _ in self.patches])
        
        self.rgb_vals = torch.from_numpy(np.array(self.reconstructed_image)).reshape(-1, 3).to(device)
        self.rgb_vals = self.rgb_vals.float() / 255
        # self.coords = get_coords(image_size[0], normalize=True).reshape(-1, 2).to(device)
        # self.latent_codes_for_all_coords = torch.stack([self.get_latent_code(coord).squeeze(-1) for coord in self.coords], dim=0)
        self.coords = get_coords(image_size[0], normalize=True).reshape(-1, 2).to(device)
        self.latent_codes_for_all_coords = torch.stack([self.get_latent_code(coord) for coord in self.coords], dim=0).unsqueeze(-1)

    def get_latent_code(self, coord):
        patch_size = 64

        denormalized_coord = coord * (self.image.size[0] - 1)

        x_idx = (denormalized_coord[0] // patch_size).long()
        y_idx = (denormalized_coord[1] // patch_size).long()

        # Compute number of patches along the width
        patches_along_width = self.image.size[0] // patch_size
        
        patch_index = x_idx * patches_along_width + y_idx
        
        # return self.latent_codes[patch_index]
        return self.latent_scalars[patch_index]


    def __len__(self):
        return len(self.rgb_vals)
    def __getitem__(self, idx):
        coordinates = self.coords[idx]
        # latent_code = self.get_latent_code(coordinates).squeeze(-1)
        latent_code = self.get_latent_code(coordinates).unsqueeze(-1)
        # return self.coords[idx], self.rgb_vals[idx], latent_code
        return self.coords[idx], self.rgb_vals[idx], self.rgb_vals_gt[idx], latent_code

class Trainer:
    def __init__(self, image_path, image_size, model_type = 'mlp', use_pe = True, device = torch.device('cpu')):
        self.dataset = ImageDataset(image_path, image_size, device)
        self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=True)

        self.model = ExposureCorrectedNet().to(device)

        lr = 1e-3

        # parameters_to_optimize = list(self.model.parameters()) + list(self.dataset.latent_codes)
        parameters_to_optimize = list(self.model.parameters()) + list(self.dataset.latent_scalars)
        self.optimizer = torch.optim.Adam(parameters_to_optimize, lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.nepochs = 200

    def run(self):
        pbar = tqdm(range(self.nepochs))
        for epoch in pbar:
            self.model.train()
            for coords, rgb_vals, rgb_vals_gt, latent_code in self.dataloader:
                self.optimizer.zero_grad()
                color, pred = self.model(coords, latent_code)
                loss = self.criterion(pred, rgb_vals)
                # loss_color = self.criterion(color, rgb_vals_gt)
                # combined_loss = loss + loss_color
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                coords = self.dataset.coords
                latent_codes = self.dataset.latent_codes_for_all_coords
                color, pred = self.model(coords, latent_codes)
                gt = self.dataset.rgb_vals_gt
                color = color.squeeze(-1)
                psnr = get_psnr(color, gt)
            pbar.set_description(f'Epoch: {epoch}, PSNR: {psnr:.2f}')
            color = color.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
            color = (color * 255).astype(np.uint8)
            pred = pred.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
            pred = (pred * 255).astype(np.uint8)
            gt = self.dataset.rgb_vals.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
            gt = (gt * 255).astype(np.uint8)
            save_image = np.hstack([gt, color, pred])
            save_image = Image.fromarray(save_image)
            #save_image.save(f'output_{epoch}.png')
            self.visualize(np.array(save_image), text = '# params: {}, PSNR: {:.2f}'.format(self.get_num_params(), psnr))

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def visualize(self, image, text):
        save_image = np.ones((300, 256*3, 3), dtype=np.uint8) * 255
        img_start = (300 - 256)
        save_image[img_start:img_start + 256, :, :] = image
        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        position = (100, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(save_image, text, position, font, scale, color, thickness)
        cv2.imshow('image', save_image)
        cv2.waitKey(1)



if __name__ == '__main__':
    image_path = 'image4.jpg'
    image_size = (256, 256)
    device = torch.device('cpu')

    trainer = Trainer(image_path, image_size, device)
    print('# params: {}'.format(trainer.get_num_params()))
    trainer.run()
