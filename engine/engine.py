import os
from arch.builder import build_arch
from data.builder import build_dataloader
from loss.builder import build_loss
from optimizer.builder import build_optimizer
from .pretrain import train_epoch as pretrain_epoch
import torch


class Engine:
    def __init__(self, cfg, mode='pretrain'):
        self.cfg = cfg
        self.pretrain_epochs = self.cfg['Global'].get('pretrain_epochs')
        self.output = self.cfg['Global'].get('output', './output')
        self.epochs = self.cfg['Global'].get('epochs')
        
        os.makedirs(self.output, exist_ok=True)
        self.mode = mode

        # Arch
        self.model = build_arch(self.cfg['Arch']).cuda()
        print(self.model)
        # optimizer
        self.optimizer = build_optimizer(self.cfg['Optimizer'], self.model)


        # dataloader
        self.dataloader = build_dataloader(self.cfg['Data'])

        # loss
        self.loss_func = build_loss(self.cfg['Loss'])

        if self.cfg['Global'].get('pretrained_model'):
            self.load(self.cfg['Global'].get('pretrained_model'))
    
    def save(self):
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(self.output, f'{self.mode}.pt'))

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def pretrain(self):
        for epoch in range(self.pretrain_epochs):
            pretrain_epoch(self)
        self.save()

    def train(self):
        for epoch in range(self.epochs):
            pretrain_epoch(self)
            