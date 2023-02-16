import os
from arch.builder import build_arch
from data.builder import build_dataloader
from loss.builder import build_loss
from metric.builder import build_metric
from optimizer.builder import build_optimizer
from .pretrain import train_epoch as pretrain_epoch
import torch
from utils.util import predict


class Engine:
    def __init__(self, cfg, mode='pretrain'):
        self.cfg = cfg
        self.pretrain_epochs = self.cfg['Global'].get('pretrain_epochs')
        self.output = self.cfg['Global'].get('output', './output')
        self.epochs = self.cfg['Global'].get('epochs')
        self.num_classes = self.cfg['Global'].get('num_classes')
        os.makedirs(self.output, exist_ok=True)
        self.mode = mode
        self.eval_interval = self.cfg['Global'].get('eval_interval')

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
    
        # metric
        self.metric_func = build_metric(self.cfg['Metric'], self.num_classes)

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
        for epoch in range(1, self.epochs+1):
            loss_dict, output, targets = pretrain_epoch(self)
            fusion_latent = output['fusion_latent']
            loss = loss_dict['loss'].item()
            if self.eval_interval > 0 and epoch % self.eval_interval == 0:
                pred = predict(fusion_latent, num_classes=self.num_classes)
                metric = self.metric_func(pred, targets.cpu().detach())
                print(loss_dict, metric)
