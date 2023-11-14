import os
import torch

from PIL import Image
from glob import glob
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from ..dataset import DatasetWrapper, DatasetLauncher
from ...utils.tools import Logger, Tools
from ...utils.config_parser import DatasetConfig
from ....config.consts import General as _CG


class MiniImagenet(DatasetWrapper):
    """MiniImagenet

    This class takes for granted that:
        * each class is represented by a directory (100 classes)
        * each directory contains 600 samples
        * all the (100) class directories are located under the same root 

    SeeAlso:
        [main page](https://github.com/fiveai/on-episodes-fsl)
        [splits](https://github.com/mileyan/simple_shot/tree/master/split/mini)
        [download](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE)
    """

    N_IMG_PER_CLASS = 600
    N_CLASSES_TRAIN = 64
    N_CLASSES_VAL = 16
    N_CLASSES_TEST = 20

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__()
        self.dataset_config = dataset_config
        self._image_list = self.get_image_list(None)
        self._label_list = self.get_label_list()
        self._train_dataset, self._val_dataset, self._test_dataset = self.split_dataset()

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        return glob(os.path.join(self.dataset_config.dataset_path, "n*", "*JPEG"))
    
    def get_label_list(self) -> List[int]:
        if self.image_list is None:
            self.image_list = self.get_image_list(None)

        # extract unique label values from dirname
        label_set = sorted(set([os.path.basename(os.path.dirname(image_name)) for image_name in self.image_list]))
        self.label_to_idx = { val: i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        # use mapping to return an int for the corresponding str label
        return [self.label_to_idx[os.path.basename(os.path.dirname(image_name))] for image_name in self.image_list]
    
    def load_image(self, path: str, augment: bool) -> torch.Tensor:
        img_pil = Image.open(path).convert("RGB")

        if augment:
            # TODO implement
            pass
        
        # resize
        img_pil = transforms.Resize((self.dataset_config.image_size, self.dataset_config.image_size))(img_pil)

        # rescale [0-255](int) to [0-1](float)
        img = transforms.ToTensor()(img_pil)

        # normalize
        img = DatasetLauncher.normalize_or_identity(self.dataset_config)(img)

        return img
    
    def split_dataset(self, split_ratios: List[float]=[.8]) -> Tuple[DatasetLauncher, Optional[DatasetLauncher], DatasetLauncher]:
        # get split
        tensor_labels = torch.tensor(self.label_list, dtype=torch.int)
        class_train, class_val, class_test = self._parse_split_csv(None)
        
        # select the images where indices corresponds to train/val/test classes
        def select_img_lbl(class_set: set) -> Tuple[List[str], List[int]]:
            label_ints = torch.tensor([self.label_to_idx[c] for c in class_set], dtype=torch.int)
            indices = torch.where(tensor_labels == label_ints.unsqueeze(1))[1]
            images = [self.image_list[i] for i in indices]
            labels = [self.label_list[i] for i in indices]
            return images, labels
        
        # create DatasetLauncher with augmentation for training if required
        augment = True if self.dataset_config.augment_online is not None else False
        train_dataset = DatasetLauncher(*select_img_lbl(class_train), augment, load_img_callback=self.load_image)
        val_dataset = DatasetLauncher(*select_img_lbl(class_val), augment=False, load_img_callback=self.load_image)
        test_dataset = DatasetLauncher(*select_img_lbl(class_test), augment=False, load_img_callback=self.load_image)
        
        # fill info dict
        train_dataset.set_info(select_img_lbl(class_train)[1], self.idx_to_label)
        val_dataset.set_info(select_img_lbl(class_val)[1], self.idx_to_label)
        test_dataset.set_info(select_img_lbl(class_test)[1], self.idx_to_label)

        # avoid using validation dataset if 0.0 is specified in the config.dataset.dataset_splits
        if len(self.dataset_config.dataset_splits) == 3:
            if self.dataset_config.dataset_splits[1] < 0.1:
                Logger.instance().warning(f"Overriding validation set: empty! No validation will be performed.")
                val_dataset = None
                
        return train_dataset, val_dataset, test_dataset
    
    def _expected_length(self) -> int:
        return (self.N_CLASSES_TRAIN + self.N_CLASSES_TEST + self.N_CLASSES_VAL) * self.N_IMG_PER_CLASS
    
    def _parse_split_csv(self, path: Optional[str]):
        import pandas as pd

        if path is None:
            path = os.path.join(os.path.dirname(self.dataset_config.dataset_path), "preprocessing", "miniimagenet", "ultimate_split")
            Logger.instance().debug(f"No path to csv split provided, trying to use default at {path}")
        
        path = Tools.validate_path(path)

        def get_class_set(split_name: str):
            split_path = Tools.validate_path(os.path.join(path, f"{split_name}.csv"))
            df = pd.read_csv(split_path)
            return set(df["label"].values)
        
        return get_class_set("train"), get_class_set("val"), get_class_set("test")
    
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