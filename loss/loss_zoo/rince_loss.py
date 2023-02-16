from ..util import get_adjacency_matrix

class RINCELoss:
    def __init__(self, q=0.7, lambd=1):
        self.q = 0.7
        self.lambd = 1

    def forward(self, latents: list, miss_matrixs: list) -> dict:
        
        