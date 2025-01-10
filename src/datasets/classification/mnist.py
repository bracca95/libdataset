import os

from glob import glob
from copy import deepcopy
from typing import List, Optional, Tuple

from .dataset_cls import DatasetCls
from ..dataset import DatasetLauncher
from ...utils.downloader import Download
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger


class Mnist(DatasetCls):

    SUBDIRS = ["MNIST", "raw"]
    TRAIN_CLASSES = 10
    TRAIN_IMAGES = 60000
    TEST_IMAGES = 10000

    def __init__(self, dataset_config: DatasetConfig):
        Download.download_mnist(root=dataset_config.dataset_path, subdirs=self.SUBDIRS, version=self.SUBDIRS[0])
        super().__init__(dataset_config)

    def __len__(self) -> int:
        return self.TRAIN_IMAGES

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        avail_ext = ("jpeg", "jpg", "png", "JPG", "JPG", "JPEG")
        
        train_images = glob(os.path.join(self.dataset_config.dataset_path, *self.SUBDIRS, "train", "*", "*"))
        train_images = list(filter(lambda x: x.endswith(avail_ext), train_images))
        test_images = glob(os.path.join(self.dataset_config.dataset_path, *self.SUBDIRS, "test", "*", "*"))
        test_images = list(filter(lambda x: x.endswith(avail_ext), test_images))
        
        return train_images + test_images

    def split_dataset(self, save_path: str) -> Tuple[DatasetLauncher, Optional[DatasetLauncher], DatasetLauncher]:
        train_images = self.image_list[:self.TRAIN_IMAGES]
        train_labels = self.label_list[:self.TRAIN_IMAGES]
        
        test_images = self.image_list[self.TRAIN_IMAGES:]
        test_labels = self.label_list[self.TRAIN_IMAGES:]

        val_images = deepcopy(test_images)
        val_labels = deepcopy(test_labels)

        # create DatasetLauncher with augmentation for training if required
        train_dataset = DatasetLauncher(train_images, train_labels, augment=None, load_img_callback=self.load_image)
        val_dataset = DatasetLauncher(val_images, val_labels, augment=None, load_img_callback=self.load_image)
        test_dataset = DatasetLauncher(test_images, test_labels, augment=None, load_img_callback=self.load_image)
        
        # fill info dict
        train_dataset.set_info(train_labels, self.idx_to_label)
        val_dataset.set_info(val_labels, self.idx_to_label)
        test_dataset.set_info(test_labels, self.idx_to_label)

        # avoid using validation dataset if 0.0 is specified in the config.dataset.dataset_splits
        if len(self.dataset_config.dataset_splits) == 3:
            if self.dataset_config.dataset_splits[1] < 0.1:
                Logger.instance().warning(f"Overriding validation set: empty! No validation will be performed.")
                val_dataset = None
                
        return train_dataset, val_dataset, test_dataset


class FashionMnist(Mnist):

    SUBDIRS = ["FashionMNIST", "raw"]

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)