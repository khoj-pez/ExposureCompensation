import torch
import torch.nn as nn

from models.mlp import FCNet

class ExposureNet(nn.Module):
    # def __init__(self, latent_dim=256, width=256, num_layers=3):
    def __init__(self, width=256, num_layers=3):
        super(ExposureNet, self).__init__()
        # layers = [nn.Linear(latent_dim, width), nn.ReLU()]
        layers = [nn.Linear(1, width), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 9))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return out.view(-1, 3, 3)

class ExposureCorrectedNet(nn.Module):
    def __init__(self):
        super(ExposureCorrectedNet, self).__init__()
        self.color = FCNet()
        self.matrix = ExposureNet()

    def forward(self, coord, latent_code):
        color = self.color(coord).unsqueeze(-1) # (N, 3, 1)
        transformation_matrix = self.matrix(latent_code) # (N, 3, 3)
        exposed_color = torch.bmm(transformation_matrix, color).squeeze() # (N, 3)
        color = color.squeeze() # (N, 3)
        return color, exposed_color
