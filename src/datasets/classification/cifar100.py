import os

from typing import Optional
from .dataset_cls import DatasetCls
from ...utils.config_parser import DatasetConfig


class Cifar100(DatasetCls):

    SUBDIRS = ["data"]
    
    NUM_CLASSES = 100
    NUM_IMAGES = 60000

    def __init__(self, dataset_config: DatasetConfig, save_path: Optional[str]=None):
        save = os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits", "cifar100")
        super().__init__(dataset_config, save)

    def __len__(self) -> int:
        return self.NUM_IMAGES
