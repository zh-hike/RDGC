from kmeans_pytorch import kmeans
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.util import HiddenPrints


class CtrLoss:
    def __init__(self,  num_classes: int, sigma: int = 0.7, taup: int = 0.8, taus: int = 0.8):
        self.sigma = sigma
        self.taus = taus
        self.taup = taup
        self.num_classes = num_classes
        self.cri = nn.CrossEntropyLoss()

    def __call__(self, fusion_latent: torch.Tensor, **kwargs):
        # centers: cxd
        # fusion_latent: nxd
        with HiddenPrints():
            _, centers = kmeans(X=fusion_latent, num_clusters=self.num_classes, device=torch.device('cuda:0'))
        # fusion_latent = F.normalize(fusion_latent, dim=-1)
        # centers = F.normalize(centers, dim=-1)
        P = torch.mm(fusion_latent, centers.cuda().T) / self.taup
        P = torch.exp(P)
        P = P / P.sum(dim=-1, keepdim=True)
        sim_PL = torch.mm(P, P.T)
        WL = torch.where(sim_PL > self.sigma, sim_PL, torch.zeros_like(sim_PL, device=torch.device('cuda:0')))

        WS = torch.mm(fusion_latent, fusion_latent.T) / self.taus
        # sim_WS = torch.exp(sim_WS)
        # WS = sim_WS / sim_WS.sum(dim=-1, keepdim=True)
        
        loss = self.cri(WS, WL.detach())
        return {'CtrLoss': loss}

