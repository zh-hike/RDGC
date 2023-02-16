import torch.nn as nn
from torch.nn import ReLU, Sigmoid, Softmax
import torch

class LinearBNReLU(nn.Module):
    def __init__(self, dims: list, 
                       dropout: float = 0,
                       activate: str = "ReLU",
                       pre_activate: str = None,
                       pre_bn: bool = False,  # or int
                       out_activate: str = None):
        super(LinearBNReLU, self).__init__()
        """
        单视图数据的前向
        """
        net = nn.ModuleList([])
        cur_dim = dims[0]
        if pre_bn:
            net.append(nn.BatchNorm1d(pre_bn))
        if pre_activate:
            net.append(eval(pre_activate)())

        for i, dim in enumerate(dims[1:-1]):                
            net.append(nn.Linear(cur_dim, dim))
            net.append(nn.BatchNorm1d(cur_dim))
            net.append(nn.Dropout(dropout))
            if activate:
                net.append(eval(activate)())
            cur_dim = dim
            
        net.append(nn.Linear(cur_dim, dims[-1]))
        if out_activate:
            net.append(eval(out_activate)())
        self.layer = nn.Sequential(*net)

    def forward(self, x: torch.Tensor):
        return self.layer(x)
    

class MultiViewLinearBnReLU(nn.Module):
    def __init__(self, 
                 mul_dims: list,
                 dropout: float = 0,
                 activate: str = "ReLU",
                 pre_activate: str = None,
                 pre_bn: bool = False,  # or int
                 out_activate: str = None):
        super(MultiViewLinearBnReLU, self).__init__()
        self.nets = nn.ModuleList([])
        for dim in mul_dims:
            self.nets.append(LinearBNReLU(dim, 
                                          dropout=dropout,
                                          pre_activate=pre_activate,
                                          pre_bn=pre_bn,  # or int
                                          activate=activate,
                                          out_activate=out_activate))
            
    def forward(self, xs: list):
        outs = []
        for x, net in zip(xs, self.nets):
            outs.append(net(x))

        return outs

            
