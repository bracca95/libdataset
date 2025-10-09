import os
import torch

from PIL import Image
from glob import glob
from torchvision.transforms import transforms
from typing import Optional, List, Tuple
from .dataset_cls import DatasetCls
from ..dataset import DatasetWrapper, DatasetLauncher
from ...utils.config_parser import DatasetConfig


class Cifar10(DatasetCls):
    """CIFAR100
    
    SeeAlso:
    [Kaggle link](https://www.kaggle.com/datasets/ayush1220/cifar10)
    """

    NUM_CLASSES = 100
    NUM_IMAGES = 60000
    TRAIN_IMAGES = 50000
    TEST_IMAGES = NUM_IMAGES - TRAIN_IMAGES

    def __init__(self, dataset_config: DatasetConfig, save_split: Optional[None]=None):
        super().__init__(dataset_config, save_split)

    def __len__(self) -> int:
        return self.NUM_IMAGES

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        avail_ext = ("jpeg", "jpg", "png", "JPG", "JPG", "JPEG")
        train_images = glob(os.path.join(self.dataset_config.dataset_path, "train", "*", "*"))
        test_images = glob(os.path.join(self.dataset_config.dataset_path, "test", "*", "*"))
        
        images = list(filter(lambda x: x.endswith(avail_ext), train_images + test_images))
        return images
    
    def load_image(self, path: str, augment: Optional[List[str]]) -> torch.Tensor:
        repeat: int = self.dataset_config.augment_times      # type: ignore .non-null checked in config parser
        
        conversion = DatasetLauncher.rgb_or_l(self.dataset_config.dataset_type, self.dataset_config.dataset_mean)
        
        img_pil = Image.open(path).convert(conversion)
        img_size = self.dataset_config.image_size

        # basic operations: always performed
        basic_transf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            DatasetLauncher.normalize_or_identity(self.dataset_config)
        ])

        img_pil = basic_transf(img_pil)
        return img_pil
    
    def split_dataset(self, save_path: str) -> Tuple[DatasetLauncher, Optional[DatasetLauncher], DatasetLauncher]:
        train_images = self._image_list[:self.TRAIN_IMAGES]
        train_labels = self._label_list[:self.TRAIN_IMAGES]
        test_images = self._image_list[self.TRAIN_IMAGES:]
        test_labels = self._label_list[self.TRAIN_IMAGES:]
        val_images = test_images.copy()
        val_labels = test_labels.copy()

        return self.get_launchers(train_images, train_labels, val_images, val_labels, test_images, test_labels)