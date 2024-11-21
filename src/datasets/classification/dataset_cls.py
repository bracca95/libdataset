import os
import torch

from PIL import Image
from glob import glob
from typing import Optional, Tuple, List
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from ..dataset import DatasetWrapper, DatasetLauncher
from ...utils.tools import Logger, Tools
from ...utils.config_parser import DatasetConfig
from ....config.consts import General as _CG


class DatasetCls(DatasetWrapper):
    """Dataset for Standard Supervised Classification tasks

    This is normally accomplished by ImageFolder in torch, but it would require a precise structure and the labels
    must also indicate if the sample is meant to be used in train, val or test. This superclass is designed to 
    accomplish the same task when these conditions are not met. Mind that the split is random and might change when the
    source code is modified.
    """

    SUBDIRS = []

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__()
        self.dataset_config = dataset_config
        
        self._image_list = self.get_image_list(None)
        self._label_list = self.get_label_list()
        self._train_dataset, self._val_dataset, self._test_dataset = self.split_dataset()

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        avail_ext = ("jpeg", "jpg", "png", "JPG", "JPG", "JPEG")
        images = glob(os.path.join(self.dataset_config.dataset_path, *self.SUBDIRS, "*", "*"))
        images = list(filter(lambda x: x.endswith(avail_ext), images))
        
        return images
    
    def get_label_list(self) -> List[int]:
        if not self._image_list:
            self._image_list = self.get_image_list()
        
        label_list = [os.path.basename(os.path.dirname(img_path)) for img_path in self._image_list]
        label_set = set(label_list)
        self.label_to_idx = { val: i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        return [self.label_to_idx[l] for l in label_list]
    
    def load_image(self, path: str, augment: Optional[List[str]]) -> torch.Tensor:
        repeat: int = self.dataset_config.augment_times      # type: ignore .non-null checked in config parser
        img_pil = Image.open(path).convert("RGB")

        # basic operations: always performed
        basic_transf = transforms.Compose([
            transforms.Resize((self.dataset_config.image_size, self.dataset_config.image_size)),
            transforms.ToTensor(),
            DatasetLauncher.normalize_or_identity(self.dataset_config)
        ])

        return basic_transf(img_pil)

    def split_dataset(self) -> Tuple[DatasetLauncher, Optional[DatasetLauncher], DatasetLauncher]:
        """Random split"""

        split_ratios = self.dataset_config.dataset_splits
        
        # shuffle
        indices = list(range(len(self._image_list)))
        indices = torch.randperm(len(self._image_list)).tolist()
        image_shuffled = [self._image_list[i] for i in indices]
        label_shuffled = [self._label_list[i] for i in indices]

        # compute ratios, then split
        train_size = int(split_ratios[0] * len(self._image_list))
        val_size = int(split_ratios[1] * len(self._image_list))
        test_size = len(self._image_list) - (train_size + val_size)
        
        # create the three sets of image lists
        train_images = image_shuffled[:train_size]
        val_images = image_shuffled[train_size:train_size + val_size]
        test_images = image_shuffled[train_size + val_size:]

        # do the same for labels
        train_labels = label_shuffled[:train_size]
        val_labels = label_shuffled[train_size:train_size + val_size]
        test_labels = label_shuffled[train_size + val_size:]

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
    