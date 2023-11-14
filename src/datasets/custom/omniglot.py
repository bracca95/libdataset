import os
import torch

from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Omniglot
from torchvision.transforms import transforms

from ..dataset import DatasetWrapper, DatasetLauncher
from ...utils.tools import Tools, Logger
from ...utils.config_parser import DatasetConfig
from ....config.consts import General as _CG


class OmniglotWrapper(DatasetWrapper):

    N_IMG_PER_CLASS = 20
    N_CLASSES_TRAIN = 964
    N_CLASSES_TEST = 659

    label_to_idx = { f"{i}": i for i in range(N_CLASSES_TRAIN + N_CLASSES_TEST) }
    idx_to_label = Tools.invert_dict(label_to_idx)

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__()
        self.dataset_config = dataset_config

        self._label_list = self.get_label_list()

        self.background_dataset = Omniglot(
            os.path.dirname(dataset_config.dataset_path),
            background=True,
            transform=transforms.Compose([transforms.ToTensor(), DatasetLauncher.normalize_or_identity(dataset_config)]),
            download=True
        )
        
        # for the test set, start labelling from the train set's highest label value+1
        self._max_train_label = self._get_max_train_label()
        target_transform = lambda x: self._add_constant(x)
        
        self.evaluation_dataset = Omniglot(
            os.path.dirname(dataset_config.dataset_path),
            background=False,
            transform=transforms.Compose([transforms.ToTensor(), DatasetLauncher.normalize_or_identity(dataset_config)]),
            target_transform=target_transform,
            download=True
        )

        self._val_dataset = None

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        return []
    
    def get_label_list(self) -> List[int]:
        r = torch.arange(self.N_CLASSES_TRAIN + self.N_CLASSES_TEST).reshape(-1, 1)
        return r.expand(-1, self.N_IMG_PER_CLASS).flatten().tolist()
    
    def split_dataset(self, split_ratios: List[float]=[.8]):
        ...

    def _get_max_train_label(self) -> int:
        loader = DataLoader(self.train_dataset, batch_size=1)
        max_overall = 0
        for _, label in loader:
            if torch.max(label).item() > max_overall:
                max_overall = torch.max(label).item()

        return int(max_overall)
    
    def _add_constant(self, target):
        return target + self._max_train_label + 1
    
    @property
    def image_list(self) -> List[str]:
        return self._image_list
    
    @image_list.setter
    def image_list(self, value):
        self._image_list = value

    @property
    def label_list(self) -> List[int]:
        return self._label_list
    
    @label_list.setter
    def label_list(self, value):
        self._label_list = value

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset
    
    @train_dataset.setter
    def train_dataset(self, value: Dataset):
        self._train_dataset = value

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset
    
    @test_dataset.setter
    def test_dataset(self, value: Dataset):
        self._test_dataset = value

    @property
    def val_dataset(self) -> Optional[Dataset]:
        return self._val_dataset
    
    @val_dataset.setter
    def val_dataset(self, value: Optional[Dataset]):
        self._val_dataset = value
