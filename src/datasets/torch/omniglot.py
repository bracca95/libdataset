import os

from typing import Optional
from torch.utils.data import Dataset
from torchvision.datasets import Omniglot
from torchvision.transforms import transforms

from ..dataset import DatasetWrapper
from ...utils.config_parser import DatasetConfig


class OmniglotWrapper(DatasetWrapper):

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__()
        self.dataset_config = dataset_config
        
        self._train_dataset = Omniglot(
            os.path.dirname(dataset_config.dataset_path),
            background=True,
            transform=transforms.Compose([transforms.ToTensor(), self.normalize_or_identity(dataset_config)]),
            download=True,
        )

        self._test_dataset = Omniglot(
            os.path.dirname(dataset_config.dataset_path),
            background=False,
            transform=transforms.Compose([transforms.ToTensor(), self.normalize_or_identity(dataset_config)]),
            download=True
        )

        self._val_dataset = None

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