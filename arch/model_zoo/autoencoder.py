from .common import MultiViewLinearBnReLU, LinearBNReLU
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self,
                 dims: list,
                 activate: str = "ReLU",
                 **kwargs):
        super(Encoder, self).__init__()
        self.net = MultiViewLinearBnReLU(dims, 
                                         activate=activate,
                                         **kwargs)
        
    def forward(self, xs):
        return self.net(xs)
    

class Decoder(nn.Module):
    def __init__(self, 
                 dims: list,
                 activate: str = "ReLU",
                 **kwargs):
        super(Decoder, self).__init__()
        self.net = MultiViewLinearBnReLU(dims,
                                         activate=activate,
                                         **kwargs)
        
    def forward(self, xs):
        return self.net(xs)
    

class AutoEncoder(nn.Module):
    def __init__(self, 
                 encoder: dict,
                 decoder: dict,
                 fusion: dict):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder(**encoder)
        self.dec = Decoder(**decoder)
        self.fusion = nn.Sequential(
            LinearBNReLU(**fusion),
            nn.Softmax(dim=-1),
        )

    def forward(self, xs: list, mask_matrix: torch.Tensor=None):
        latent = self.enc(xs)
        cat_latent = torch.concat(latent, dim=1)
        ws = self.fusion(cat_latent)   # n x v
        mask_matrix = torch.stack(mask_matrix)
        matrix = mask_matrix.squeeze().transpose(0, 1)  # n x v
        w = matrix * (ws * matrix).sum(dim=0, keepdim=True) / matrix.sum(dim=0, keepdim=True)
        ws = ws / torch.abs(w).sum(dim=-1, keepdim=True)
        z = 0
        for w, h in zip(ws.T, latent):
            w = w.unsqueeze(1)
            z += w * h
            
        mz = [z for _ in xs]
        fusion2recon = self.dec(mz)
        out = self.dec(latent)
        return out, fusion2recon, latent, z
