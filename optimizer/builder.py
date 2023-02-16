from torch.optim import SGD, Adam
import copy

def build_optimizer(config, net):
    cfg = copy.deepcopy(config)
    name = cfg.pop('name')
    optimizer = eval(name)(net.parameters(), **cfg)
    return optimizer