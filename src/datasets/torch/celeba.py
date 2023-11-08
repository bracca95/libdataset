import os

from typing import Optional
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import transforms, Compose

from ..dataset import DatasetWrapper
from ...utils.config_parser import DatasetConfig


class CelebaWrapper(DatasetWrapper):

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__()
        self.dataset_config = dataset_config
        
        self._train_dataset = CelebA(
            os.path.dirname(self.dataset_config.dataset_path),
            split="train",
            target_type="bbox",
            transform=self.load_image(self.dataset_config),
            download=False
        )

        self._test_dataset = CelebA(
            os.path.dirname(self.dataset_config.dataset_path),
            split="test",
            target_type="bbox",
            transform=self.load_image(self.dataset_config),
            download=False
        )

        self._val_dataset = CelebA(
            os.path.dirname(self.dataset_config.dataset_path),
            split="val",
            target_type="bbox",
            transform=self.load_image(self.dataset_config),
            download=False
        )

    @staticmethod
    def load_image(dataset_config: DatasetConfig) -> Compose:
        return transforms.Compose([
                transforms.Resize((dataset_config.image_size, dataset_config.image_size)),
                transforms.ToTensor(),
                DatasetWrapper.normalize_or_identity(dataset_config)
            ])

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