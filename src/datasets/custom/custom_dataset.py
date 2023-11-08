from __future__ import annotations

import os
import torch
import random

from PIL import Image
from PIL.Image import Image as PilImgType
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data import Subset
from torchvision.transforms import transforms

from ..dataset import DatasetWrapper
from ...imgproc import Processing
from ...utils.tools import Tools, Logger
from ...utils.config_parser import DatasetConfig
from ....config.consts import SubsetsDict
from ....config.consts import General as _CG


@dataclass
class SubsetInfo:
    """SubsetInfo dataclass

    The aim of this class is to wrap the output of torch.utils.data.random_split, so that it contains the information
    related to the number of samples that belong to the split subsets. If you are creating your own dataset and then
    you want to use pytorch's random_split method, this is the way to go not to lose information. Mind that subset
    is a tuple containing THE WHOLE dataset and the indexes of the values that are in the subset.

    name (str): name of the split set { 'train', 'val', 'test' }
    info_dict (dict): dict with (key, val) = (class name, number of instances per class)

    SeeAlso:
        [PyTorch's Subset documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset)
    """
    
    name: str
    info_dict: dict


class DatasetLauncher(Dataset):

    def __init__(self, image_list: List[str], label_list: List[int], load_img_callback: Callable[[str], torch.Tensor]):
        super().__init__()
        self.image_list = image_list
        self.label_list = label_list
        self.load_img_callback = load_img_callback

        self.dataset_info: Optional[SubsetInfo] = None

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]
        curr_label_batch = self.label_list[index]
        
        return self.load_img_callback(curr_img_batch), curr_label_batch

    def __len__(self):
        return len(self.label_list)

    def set_info(self, split_name: str, info_dict: dict):
        Logger.instance().debug(f"{split_name}: {info_dict}")
        self.dataset_info = SubsetInfo(split_name, info_dict)

class CustomDataset(DatasetWrapper):

    label_to_idx = {}
    idx_to_label = Tools.invert_dict(label_to_idx)

    AUG_DIR = "img_augment"

    @abstractproperty
    def augment_strategy(self):
        ...

    @abstractmethod
    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        ...

    @abstractmethod
    def get_label_list(self) -> List[int]:
        ...

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        
        self.dataset_aug_path: str = os.path.join(os.path.dirname(self.dataset_config.dataset_path), self.AUG_DIR)
        self.filt: List[str] = list(self.label_to_idx.keys())

        if self.dataset_config.augment_offline is not None:
            self.augment_dataset(50, self.augment_strategy)

        self.image_list: Optional[List[str]] = self.get_image_list(self.filt)
        self.label_list: Optional[List[int]] = self.get_label_list()

        split_ratios = self.dataset_config.dataset_splits
        self._train_dataset, self._val_dataset, self._test_dataset = self.split_dataset(split_ratios)

    def split_dataset(self, split_ratios: List[float]=[.8]) -> Tuple[Dataset, Optional[Dataset], Dataset]:
        """Split a dataset into train, (val), test

        Split a dataset into two or three parts, trying to account for a balanced number of samples for each class.

        This is not the best approach to use for few-shot learning, as it includes all the classes in all the splits.
        We would like to have different classes for train/val/test splits instead. However, since the number of classes
        for our glass dataset is very low, we train with all possibile combinations. 
        Use fsl_split for proper FSL split.

        Args:
            split_ratios (List[float]=[.8]): if len == 1: train/test, if len == 3 train/val/test else exception

        Returns:
            train/(val)/test set (Tuple[Dataset, Optional[Dataset], Dataset])

        Raises:
            ValueError if `label_list` has not been computed yet
            ValueError if split_ratios != {1, 3}

        """

        if self.label_list is None:
            raise ValueError(f"Cannot split dataset if the length of labels is unknown")

        train_str, val_str, test_str = _CG.DEFAULT_SUBSETS
        
        # split ratios must be a list of length 3
        if type(split_ratios) is float:
            split_ratios = [split_ratios, 0.0, 1.0 - split_ratios]
        elif len(split_ratios) == 1:
            split_ratios = [split_ratios[0], 0.0, 1.0 - split_ratios[0]]
        elif len(split_ratios) == 3:
            split_ratios = split_ratios
        else:
            raise ValueError(f"split_ratios argument accepts either a list of 1 value (train,test) or 3 (train,val,test)")

        # the length of unique labels is equal to idx_to_label because I have already filtered them
        label_count = { c: self.label_list.count(c) for c in self.idx_to_label.keys() }

        label_count_log = { self.idx_to_label[c]: self.label_list.count(c) for c in self.idx_to_label.keys() }
        Logger.instance().debug(f"Population: {label_count_log}")
        
        # save positions of the labels for each class
        class_indices = { label: [] for label in set(self.label_list) }
        for pos, label in enumerate(self.label_list):
            class_indices[label].append(pos)
        
        # init dicts to store the split data
        train_indices = {}
        val_indices = {}
        test_indices = {}

        # split each class into three parts based on percentages
        for label, count in label_count.items():
            indices = class_indices[label]
            random.shuffle(indices)  # Shuffle the indices for randomness

            train_size = int(split_ratios[0] * count)
            val_size = int(split_ratios[1] * count)
            test_size = count - train_size - val_size

            train_indices[label] = indices[:train_size]
            val_indices[label] = indices[train_size:train_size + val_size]
            test_indices[label] = indices[train_size + val_size:]

        # flatten the dictionaries to get the final selected indices for each split
        selected_train_indices = [index for indices in train_indices.values() for index in indices]
        selected_val_indices = [index for indices in val_indices.values() for index in indices]
        selected_test_indices = [index for indices in test_indices.values() for index in indices]

        # get the image paths and labels for each split and prepare datasets
        train_images = [self.image_list[i] for i in selected_train_indices]
        train_labels = [self.label_list[i] for i in selected_train_indices]
        train_dataset = DatasetLauncher(train_images, train_labels, self.load_image)
        train_dataset.set_info(train_str, {self.idx_to_label[k]: len(train_indices[k]) for k in train_indices.keys()})

        val_dataset = None
        if len(selected_val_indices) > 0:
            val_images = [self.image_list[i] for i in selected_val_indices]
            val_labels = [self.label_list[i] for i in selected_val_indices]
            val_dataset = DatasetLauncher(val_images, val_labels, self.load_image)
            val_dataset.set_info(val_str, {self.idx_to_label[k]: len(val_indices[k]) for k in val_indices.keys()})

        test_images = [self.image_list[i] for i in selected_test_indices]
        test_labels = [self.label_list[i] for i in selected_test_indices]
        test_dataset = DatasetLauncher(test_images, test_labels, self.load_image)
        test_dataset.set_info(test_str, {self.idx_to_label[k]: len(test_indices[k]) for k in test_indices.keys()})

        return train_dataset, val_dataset, test_dataset
    
    def load_image(self, path: str) -> torch.Tensor:
        img_pil = Image.open(path).convert("RGB")
        
        # resize
        img_pil = transforms.Resize((self.dataset_config.image_size, self.dataset_config.image_size))(img_pil)

        # rescale [0-255](int) to [0-1](float)
        img = transforms.ToTensor()(img_pil)

        # normalize
        img = self.normalize_or_identity(self.dataset_config)(img)

        return img
    
    def augment_dataset(self, iters: int, augment_func: Optional[Callable[[PilImgType], PilImgType]]):
        """Perform offline augmentation
        
        Increase the number of available samples with augmentation techniques, if required in config. Offline
        augmentation can work on a limited set of classes; indeed, it should be used if there are not enough samples
        for each class.

        Args:
            iters (int): number of augmentation iteration for the same image.
            augment_fuct (Callable): function used to augment.
        """

        if augment_func is None:
            Logger.instance().debug("No augment function specified: return")
            return
        
        Logger.instance().debug("increasing the number of images...")
        
        if os.path.exists(self.dataset_aug_path):
            if len(os.listdir(self.dataset_aug_path)) > 0:
                Logger.instance().warning("the dataset has already been augmented")
                return
        else:
            os.makedirs(self.dataset_aug_path)
        
        image_list = self.get_image_list(self.dataset_config.augment_offline)
        Processing.store_augmented_images(image_list, self.dataset_aug_path, iters, augment_func)

        Logger.instance().debug("dataset augmentation completed")

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset
    
    @train_dataset.setter
    def train_dataset(self, value):
        self._train_dataset = value

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset
    
    @test_dataset.setter
    def test_dataset(self, value):
        self._test_dataset = value

    @property
    def val_dataset(self) -> Optional[Dataset]:
        return self._val_dataset
    
    @val_dataset.setter
    def val_dataset(self, value):
        self._val_dataset = value