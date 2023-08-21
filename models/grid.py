import torch
import torch.nn as nn
import numpy as np

from models.mlp import MLP
def bilinear_interpolation(res, grid, points, grid_type):
    PRIMES = [1, 265443567, 805459861]

    # Get the dimensions of the grid
    grid_size, feat_size = grid.shape
    points = points[None]
    _, N, _ = points.shape
    # Get the x and y coordinates of the four nearest points for each input point
    x = points[:, :, 0] * (res - 1)
    y = points[:, :, 1] * (res - 1)

    x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
    y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()

    x2 = torch.clip(x1 + 1, 0, res - 1).int()
    y2 = torch.clip(y1 + 1, 0, res - 1).int()

    # Compute the weights for each of the four points
    w1 = (x2 - x) * (y2 - y)
    w2 = (x - x1) * (y2 - y)
    w3 = (x2 - x) * (y - y1)
    w4 = (x - x1) * (y - y1)

    if grid_type == "NGLOD":
        # Interpolate the values for each point
        id1 = (x1 + y1 * res).long()
        id2 = (y1 * res + x2).long()
        id3 = (y2 * res + x1).long()
        id4 = (y2 * res + x2).long()

    elif grid_type == "HASH":
        npts = res**2
        if npts > grid_size:
            id1 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1])) % grid_size
            id2 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1])) % grid_size
            id3 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1])) % grid_size
            id4 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1])) % grid_size
        else:
            id1 = (x1 + y1 * res).long()
            id2 = (y1 * res + x2).long()
            id3 = (y2 * res + x1).long()
            id4 = (y2 * res + x2).long()
    else:
        print("NOT IMPLEMENTED")

    values = (
        torch.einsum("ab,abc->abc", w1, grid[(id1).long()])
        + torch.einsum("ab,abc->abc", w2, grid[(id2).long()])
        + torch.einsum("ab,abc->abc", w3, grid[(id3).long()])
        + torch.einsum("ab,abc->abc", w4, grid[(id4).long()])
    )
    return values[0]
class DenseGrid(nn.Module):
    def __init__(self, base_lod, num_lods, feature_dim = 4):
        super(DenseGrid, self).__init__()
        self.feature_dim = feature_dim
        self.code_book = nn.ParameterList([])
        self.LODS = [2 ** L for L in range(base_lod, base_lod + num_lods)]
        self.init_feature_maps()
    def forward(self, x):
        #x: (B, 2) between 0 and 1
        #out: (B, feature_dim)

        feats = []
        for i, res in enumerate(self.LODS):
            current_feature_map = self.code_book[i]
            features = bilinear_interpolation(res, current_feature_map, x, grid_type="NGLOD")
            feats.append(features)
        out = torch.cat(feats, dim=-1)
        return out
    def init_feature_maps(self):
        for L in self.LODS:
            feature_map = torch.zeros((L * L, self.feature_dim))
            feature_map = nn.Parameter(feature_map)
            nn.init.normal_(feature_map, mean=0, std=0.1)
            self.code_book.append(feature_map)

class HashGrid(nn.Module):
    def __init__(self, min_lod = 16, max_lod = 1024, num_lods = 10, feature_dim = 4, bandwidth = 13):
        super(HashGrid, self).__init__()
        self.feature_dim = feature_dim
        self.code_book = nn.ParameterList([])
        self.code_book_size = 2 ** bandwidth
        b = np.exp((np.log(max_lod) - np.log(min_lod)) / (num_lods - 1))
        self.LODS = [int(1 + np.floor(min_lod * (b ** l))) for l in range(num_lods)]
        self.init_feature_maps()
    def init_feature_maps(self):
        for L in self.LODS:
            feature_map = torch.zeros((min(L * L, self.code_book_size), self.feature_dim))
            feature_map = nn.Parameter(feature_map)
            nn.init.normal_(feature_map, mean=0, std=0.1)
            self.code_book.append(feature_map)
    def forward(self, x):
        #x: (B, 2) between 0 and 1
        #out: (B, feature_dim)

        feats = []
        for i, res in enumerate(self.LODS):
            current_feature_map = self.code_book[i]
            features = bilinear_interpolation(res, current_feature_map, x, grid_type="HASH")
            feats.append(features)
        out = torch.cat(feats, dim=-1)
        return out


class InstantNGP(nn.Module):
    def __init__(self):
        super(InstantNGP, self).__init__()
        self.grid = HashGrid()
        self.mlp = MLP(40, 3, 64, 1)
    def forward(self, x):
        x = self.grid(x)
        x = self.mlp(x)
        return x

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.grid = DenseGrid(4, 5)
        self.mlp = MLP(20, 3, 64, 1)
    def forward(self, x):
        x = self.grid(x)
        x = self.mlp(x)
        return x