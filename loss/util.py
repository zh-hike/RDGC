import torch
import torch.nn.functional as F


def get_adjacency_matrix(h, miss_matrix=None, topk=3):
    miss_matrix = miss_matrix.squeeze()
    h = F.normalize(h, p=2, dim=-1)
    n, _ = h.shape
    sim_matrix = torch.mm(h, h.T)
    indices = torch.topk(sim_matrix, topk, dim=1).indices
    A = torch.zeros((n, n), device="cuda:0")
    A.scatter_(1, indices, torch.ones_like(A))
    A = torch.where(A + A.T > 0, torch.ones_like(A, device="cuda:0"), torch.zeros_like(A, device="cuda:0"))   # nxn
    miss_matrix = torch.mm(miss_matrix.unsqueeze(1), miss_matrix.unsqueeze(0))  # nxn
    A = A * miss_matrix
    return A, sim_matrix