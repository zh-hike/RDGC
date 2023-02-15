from .dataset import *
import copy


def build_dataloader(config: dict):   
    """
    config为字典格式，具体见./config/*.yaml里的Data的参数
    返回 dataloader
    """
    cfg = copy.deepcopy(config)
