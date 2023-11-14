import os

from typing import Optional
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import transforms, Compose

from ..dataset import DatasetLauncher
from ...utils.config_parser import DatasetConfig


class CelebaWrapper(Dataset):

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__()
        self.dataset_config = dataset_config
        
        self.train_dataset = CelebA(
            os.path.dirname(self.dataset_config.dataset_path),
            split="train",
            target_type="bbox",
            transform=self.load_image(self.dataset_config),
            download=False
        )

        self.test_dataset = CelebA(
            os.path.dirname(self.dataset_config.dataset_path),
            split="test",
            target_type="bbox",
            transform=self.load_image(self.dataset_config),
            download=False
        )

        self.val_dataset = CelebA(
            os.path.dirname(self.dataset_config.dataset_path),
            split="valid",
            target_type="bbox",
            transform=self.load_image(self.dataset_config),
            download=False
        )

    @staticmethod
    def load_image(dataset_config: DatasetConfig) -> Compose:
        return transforms.Compose([
                transforms.CenterCrop((dataset_config.crop_size, dataset_config.crop_size)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((dataset_config.image_size, dataset_config.image_size)),
                transforms.ToTensor(),
                DatasetLauncher.normalize_or_identity(dataset_config)
            ])
