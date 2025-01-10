import os
import torch

from PIL import Image
from glob import glob
from copy import deepcopy
from typing import Optional, Tuple, Set, List
from torchvision.transforms import transforms

from .dataset_fsl import FewShotDataset
from ..dataset import DatasetLauncher
from ...utils.tools import Logger, Tools
from ...utils.downloader import Download
from ...utils.config_parser import DatasetConfig


class MetaMnist(FewShotDataset):

    SUBDIRS = ["MNIST", "raw"]
    TRAIN_CLASSES = 10
    TRAIN_IMAGES = 60000
    TEST_IMAGES = 10000

    def __init__(self, dataset_config: DatasetConfig):
        Download.download_mnist(dataset_config.dataset_path, self.SUBDIRS, version=self.SUBDIRS[0])
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        avail_ext = ("jpeg", "jpg", "png", "JPG", "JPG", "JPEG")
        
        train_images = glob(os.path.join(self.dataset_config.dataset_path, *self.SUBDIRS, "train", "*", "*"))
        train_images = list(filter(lambda x: x.endswith(avail_ext), train_images))
        test_images = glob(os.path.join(self.dataset_config.dataset_path, *self.SUBDIRS, "test", "*", "*"))
        test_images = list(filter(lambda x: x.endswith(avail_ext), test_images))
        
        return train_images + test_images

    def load_image(self, path: str, augment: Optional[List[str]]) -> torch.Tensor:
        repeat: int = self.dataset_config.augment_times      # type: ignore .non-null checked in config parser
        
        conversion = "RGB"
        if self.dataset_config.dataset_mean is not None and len(self.dataset_config.dataset_mean) == 1:
            conversion = "L"
        
        img_pil = Image.open(path).convert(conversion)

        # basic operations: always performed
        basic_transf = transforms.Compose([
            transforms.Resize((self.dataset_config.image_size, self.dataset_config.image_size)),
            transforms.ToTensor(),
            DatasetLauncher.normalize_or_identity(self.dataset_config)
        ])

        return basic_transf(img_pil)

    def get_label_list(self) -> List[int]:
        if not self.image_list:
            self.image_list = self.get_image_list(None)
        
        label_list = [os.path.basename(os.path.dirname(img_path)) for img_path in self.image_list]
        label_set = list(dict.fromkeys(label_list))     # not a set, but preserves the order
        self.label_to_idx = { val: i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        return [self.label_to_idx[l] for l in label_list]
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        if self.dataset_config.dataset_splits == [1.0, 0.0, 0.0]:
            class_train = set(list(self.idx_to_label.values()))
            class_val = set()
            class_test = set()
        elif self.dataset_config.dataset_splits == [0.0, 0.0, 1.0]:
            class_train = set()
            class_val = set(list(self.idx_to_label.values()))
            class_test = deepcopy(class_val)
        else:
            Logger.instance().warning(f"Meta {self.SUBDIRS[0]} must be used in 100% train/100% test. You are not")
            Logger.instance().warning(f"Fallback to 5 classes used in train and 5 classes for test for N-way K-shot")

            class_train = set(list(self.idx_to_label.values())[:5])
            class_test = set(list(self.idx_to_label.values())[5:])
            class_val = deepcopy(class_test)

            if self.dataset_config.dataset_splits[1] < 0.1:
                Logger.instance().warning("No validation will be performed")
                class_val = set()

        return class_train, class_val, class_test

    def expected_length(self):
        return self.TRAIN_IMAGES + self.TEST_IMAGES


class MetaFashionMnist(MetaMnist):

    SUBDIRS = ["FashionMNIST", "raw"]

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)