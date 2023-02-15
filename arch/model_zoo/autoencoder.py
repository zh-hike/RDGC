from .common import MultiViewLinearBnReLU
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
                 decoder: dict):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder(**encoder)
        self.dec = Decoder(**decoder)

    def forward(self, xs: list, mask_matrix: torch.Tensor=None):
        latent = self.enc(xs)
        out = self.dec(latent)
        return out, latent
