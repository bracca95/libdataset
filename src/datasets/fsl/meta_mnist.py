import os
import torch

from PIL import Image
from glob import glob
from copy import deepcopy
from typing import Optional, Tuple, Set, List
from torchvision.transforms import transforms

from .dataset_fsl import FewShotDataset
from ..dataset import DatasetLauncher
from ...imgproc import RandomProjection
from ...utils.tools import Logger, Tools
from ...utils.downloader import Download
from ...utils.config_parser import DatasetConfig


class Mnist2Fashion(FewShotDataset):
    """Meta version of MNIST/FashionMNIST

    This dataset merges MNIST and FashionMNIST to build a meta-dataset that trains on the train split of the first and
    test on the unseen classes of the second.
    """

    SUBDIRS_TRAIN = ["MNIST", "raw"]
    SUBDIRS_TEST = ["FashionMNIST", "raw"]
    TRAIN_CLASSES = 10          # MNIST
    TEST_CLASSES = 10           # FashionMNIST
    TRAIN_IMAGES = 60000        # MNIST's train split
    TEST_IMAGES = 10000         # FashionMNIST's test split

    def __init__(self, dataset_config: DatasetConfig):
        self.test_path = os.path.join(os.path.dirname(dataset_config.dataset_path), self.SUBDIRS_TEST[0])
        Download.download_mnist(dataset_config.dataset_path, self.SUBDIRS_TRAIN, version=self.SUBDIRS_TRAIN[0])
        Download.download_mnist(self.test_path, self.SUBDIRS_TEST, version=self.SUBDIRS_TEST[0])
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        avail_ext = ("jpeg", "jpg", "png", "JPG", "JPG", "JPEG")
        
        train_images = glob(os.path.join(self.dataset_config.dataset_path, *self.SUBDIRS_TRAIN, "train", "*", "*"))
        train_images = list(filter(lambda x: x.endswith(avail_ext), train_images))
        test_images = glob(os.path.join(self.test_path, *self.SUBDIRS_TEST, "test", "*", "*"))
        test_images = list(filter(lambda x: x.endswith(avail_ext), test_images))
        
        return train_images + test_images

    def get_label_list(self) -> List[int]:
        if not self.image_list:
            self.image_list = self.get_image_list(None)
        
        # MNIST and FashionMNIST have the same labels, differentiate them by adding the dataset name
        label_list_train: List[str] = list()
        for img_path in self.image_list[:self.TRAIN_IMAGES]:
            label_list_train.append(f"{self.SUBDIRS_TRAIN[0]}_{os.path.basename(os.path.dirname(img_path))}")

        label_list_test: List[str] = list()
        for img_path in self.image_list[self.TRAIN_IMAGES:]:
            label_list_test.append(f"{self.SUBDIRS_TEST[0]}_{os.path.basename(os.path.dirname(img_path))}")

        label_list = label_list_train + label_list_test
        
        label_set = list(dict.fromkeys(label_list))     # not a set, but preserves the order
        self.label_to_idx = { val: i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        return [self.label_to_idx[l] for l in label_list]

    def load_image(self, path: str, augment: Optional[List[str]]) -> torch.Tensor:
        repeat: int = self.dataset_config.augment_times      # type: ignore .non-null checked in config parser
        
        conversion = "RGB"
        if self.dataset_config.dataset_mean is not None and len(self.dataset_config.dataset_mean) == 1:
            conversion = "L"
        
        img_pil = Image.open(path).convert(conversion)
        img_size = self.dataset_config.image_size

        # basic operations: always performed
        basic_transf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            DatasetLauncher.normalize_or_identity(self.dataset_config)
        ])

        img_pil = basic_transf(img_pil)

        # if augmentation required
        if augment is not None and "projection" in augment:
            aug_matrix = torch.randn(img_size**2, img_size**2) * (1 / img_size)**0.5     # same as GPICL
            augment_function = transforms.Compose([RandomProjection(aug_matrix)])
            img_pil = augment_function(img_pil)

        return img_pil
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        # the split is fixed: MNIST to train, FashionMNIST (test split) to test
        class_train = set([l for l in self.idx_to_label.values() if l.split("_")[0] == self.SUBDIRS_TRAIN[0]])
        class_test = set([l for l in self.idx_to_label.values() if l.split("_")[0] == self.SUBDIRS_TEST[0]])
        class_val = deepcopy(class_test)

        if self.dataset_config.dataset_splits[1] < 0.1:
            Logger.instance().warning("No validation will be performed")
            class_val = set()

        return class_train, class_val, class_test

    def expected_length(self):
        return self.TRAIN_IMAGES + self.TEST_IMAGES


class Fashion2Mnist(Mnist2Fashion):

    SUBDIRS_TRAIN = ["FashionMNIST", "raw"]
    SUBDIRS_TEST = ["MNIST", "raw"]

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)