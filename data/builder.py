from .dataset import *
import copy
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

def build_dataloader(config):
    cfg = copy.deepcopy(config)
    dataset_cfg = cfg.pop('Dataset')
    dataname = dataset_cfg.pop('name')
    dataset = eval(dataname)(**dataset_cfg)

    dataloader_cfg = cfg.pop('DataLoader')
    sampler_cfg = dataloader_cfg.pop('sampler')
    sampler_name = sampler_cfg.pop('name')
    sampler = eval(sampler_name)(dataset, **sampler_cfg)

    dataloader = DataLoader(dataset, sampler=sampler, **dataloader_cfg)
    return dataloader