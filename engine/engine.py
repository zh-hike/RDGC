import os
from arch.builder import build_arch

class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pretrain_epochs = self.cfg['Global'].get('pretrain_epochs')
        self.output = self.cfg['Global'].get('output', './output')
        self.epochs = self.cfg['Global'].get('epochs')
        os.makedirs(self.output, exist_ok=True)

        # Arch
        self.model = build_arch(self.cfg['Arch'])
        print(self.model)


    def pretrain(self):
        pass

    def train(self):
        pass