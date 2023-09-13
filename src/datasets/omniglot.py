import os
import torch

from typing import Callable, List, Optional
from PIL.Image import Image as PilImgType
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import Omniglot
from torchvision.transforms import transforms

from .dataset import CustomDataset
from ..utils.tools import Tools, Logger
from ..utils.config_parser import DatasetConfig
from ...config.consts import SubsetsDict, General as _CG


class CustomOmniglot(CustomDataset):

    label_to_idx = { f"{i}": i for i in range(1623) }
    idx_to_label = Tools.invert_dict(label_to_idx)

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
        self.train_dataset = Omniglot(
            os.path.dirname(self.dataset_config.dataset_path),
            background=True,
            transform=transforms.ToTensor(),
            download=True
        )
        
        self._max_train_label = self._get_max_train_label()

        target_transform = lambda x: self._add_constant(x)
        self.test_dataset = Omniglot(
            os.path.dirname(self.dataset_config.dataset_path),
            background=False,
            transform=transforms.ToTensor(),
            target_transform=target_transform,
            download=True
        )

        self.dataset = ConcatDataset([self.train_dataset, self.test_dataset])

        self.subsets_dict: SubsetsDict = self.split_torch_dataset()

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return 1623 * 20

    @property
    def augment_strategy(self):
        return self._augment_strategy
    
    @augment_strategy.setter
    def augment_strategy(self, val):
        self._augment_strategy = None

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        return []
    
    def get_label_list(self) -> List[int]:
        return []
    
    def split_torch_dataset(self) -> SubsetsDict:
        train_set = Subset(self.train_dataset, [i for i in range(len(self.train_dataset))])
        test_set = Subset(self.test_dataset, [i for i in range(len(self.test_dataset))])

        train_str, val_str, test_str = _CG.DEFAULT_SUBSETS
        
        return { train_str: train_set, val_str: None, test_str: test_set }   # type: ignore

    def _get_max_train_label(self) -> int:
        loader = DataLoader(self.train_dataset, batch_size=1)
        max_overall = 0
        for _, label in loader:
            if torch.max(label).item() > max_overall:
                max_overall = torch.max(label).item()

        return int(max_overall)
    
    def _add_constant(self, target):
        return target + self._max_train_label + 1
