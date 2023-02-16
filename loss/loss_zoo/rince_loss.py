from ..util import get_adjacency_matrix
import torch.nn as nn


class RINCELoss:
    def __init__(self, q=0.7, lambd=1):
        self.q = 0.7
        self.lambd = 1
        self.cri = nn.CrossEntropyLoss()


    def __call__(self, latents: list, miss_matrixs: list, **kwargs) -> dict:
        loss = 0
        for latent, miss_matrix in zip(latents, miss_matrixs):
            A, sim_matrix = get_adjacency_matrix(latent, miss_matrix)
            loss += self.cri(sim_matrix, A.detach())
        return {'RINCELoss': loss}

        