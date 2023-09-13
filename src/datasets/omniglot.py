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

    N_IMG_PER_CLASS = 20
    N_CLASSES_TRAIN = 964
    N_CLASSES_TEST = 659

    label_to_idx = { f"{i}": i for i in range(N_CLASSES_TRAIN + N_CLASSES_TEST) }
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

        self.full_dataset = ConcatDataset([self.train_dataset, self.test_dataset])
        self.subsets_dict: SubsetsDict = self.split_torch_dataset()

        self.label_list = self.get_label_list()

    def __getitem__(self, index):
        return self.full_dataset[index]
    
    def __len__(self):
        return (CustomOmniglot.N_CLASSES_TRAIN + CustomOmniglot.N_CLASSES_TEST) * CustomOmniglot.N_IMG_PER_CLASS

    @property
    def augment_strategy(self):
        return self._augment_strategy
    
    @augment_strategy.setter
    def augment_strategy(self, val):
        self._augment_strategy = None

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        return []
    
    def get_label_list(self) -> List[int]:
        # this is called in super(), but the dataset has not been initialized yet. Fill the value after super() has finished (bad practice)
        if not hasattr(self, "full_dataset"):
            return []
        
        r = torch.arange(CustomOmniglot.N_CLASSES_TRAIN + CustomOmniglot.N_CLASSES_TEST).reshape(-1, 1)
        return r.expand(-1, CustomOmniglot.N_IMG_PER_CLASS).flatten().tolist()
    
    def split_torch_dataset(self) -> SubsetsDict:
        train_set = Subset(self, [i for i in range(len(self.train_dataset))])
        test_set = Subset(self, [i for i in range(len(self.train_dataset), len(self.train_dataset) + len(self.test_dataset))])

        train_str, val_str, test_str = _CG.DEFAULT_SUBSETS
        
        Logger.instance().debug(f"Splitting omniglot: [{len(train_set)}, {len(test_set)}]")
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
