from __future__ import annotations

import os
import torch

from PIL import Image
from PIL.Image import Image as PilImgType
from abc import abstractmethod, abstractproperty
from typing import Optional, List, Tuple, Callable
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from ..dataset import DatasetWrapper, DatasetLauncher
from ...imgproc import Processing
from ...utils.tools import Tools, Logger
from ...utils.config_parser import DatasetConfig
from ....config.consts import General as _CG


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

        self._image_list: List[str] = self.get_image_list(self.filt)
        self._label_list: List[int] = self.get_label_list()

        split_ratios = self.dataset_config.dataset_splits
        self._train_dataset, self._val_dataset, self._test_dataset = self.split_dataset(split_ratios)

    def split_dataset(self, split_ratios: List[float]=[.8]) -> Tuple[DatasetLauncher, Optional[DatasetLauncher], DatasetLauncher]:
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

        if len(self.label_list) == 0:
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
        
        # count the number of elements for each class
        label_tensor = torch.tensor(self.label_list, dtype=torch.int)
        populat = { self.idx_to_label[k.item()]: (label_tensor == k).sum().item() for k in torch.unique(label_tensor) }
        Logger.instance().debug(f"Population: {populat}")

        # dictionary to map class labels to a list of indices (nonzero() operation)
        class_indices = { label: (label_tensor == label).nonzero().squeeze() for label in torch.unique(label_tensor) }

        # split each class into three parts based on percentages
        split_indices = {}
        for label, indices in class_indices.items():
            permuted_indices = indices[torch.randperm(len(indices))]

            train_size = int(split_ratios[0] * len(permuted_indices))
            val_size = int(split_ratios[1] * len(permuted_indices))
            test_size = len(permuted_indices) - (train_size + val_size)
            
            split_indices[label.item()] = torch.split(permuted_indices, [train_size, val_size, test_size])

        # concatenate the selected indices to get the final indices for each split
        selected_train_indices = torch.cat([split_indices[label.item()][0] for label in torch.unique(label_tensor)], dim=0).tolist()
        selected_val_indices = torch.cat([split_indices[label.item()][1] for label in torch.unique(label_tensor)], dim=0).tolist()
        selected_test_indices = torch.cat([split_indices[label.item()][2] for label in torch.unique(label_tensor)], dim=0).tolist()

        augment = True if self.dataset_config.augment_online is not None else False
        # get the image paths and labels for each split and prepare datasets
        train_images = [self.image_list[i] for i in selected_train_indices]
        train_labels = [self.label_list[i] for i in selected_train_indices]
        train_dataset = DatasetLauncher(train_images, train_labels, augment, load_img_callback=self.load_image)
        train_dataset.set_info(train_labels, self.idx_to_label)

        val_dataset = None
        if len(selected_val_indices) > 0:
            val_images = [self.image_list[i] for i in selected_val_indices]
            val_labels = [self.label_list[i] for i in selected_val_indices]
            val_dataset = DatasetLauncher(val_images, val_labels, augment=False, load_img_callback=self.load_image)
            val_dataset.set_info(val_labels, self.idx_to_label)

        test_images = [self.image_list[i] for i in selected_test_indices]
        test_labels = [self.label_list[i] for i in selected_test_indices]
        test_dataset = DatasetLauncher(test_images, test_labels, augment=False, load_img_callback=self.load_image)
        test_dataset.set_info(test_labels, self.idx_to_label)

        return train_dataset, val_dataset, test_dataset
    
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