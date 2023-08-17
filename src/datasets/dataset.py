from __future__ import annotations

import os
import torch

from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, List, Callable
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data import Subset

from ..imgproc import Processing
from ..utils.tools import Tools, Logger
from ..utils.config_parser import Config
from ...config.consts import SubsetsDict
from ...config.consts import General as _GC


@dataclass
class SubsetInfo:
    """SubsetInfo dataclass

    The aim of this class is to wrap the output of torch.utils.data.random_split, so that it contains the information
    related to the number of samples that belong to the split subsets. If you are creating your own dataset and then
    you want to use pytorch's random_split method, this is the way to go not to lose information. Mind that subset
    is a tuple containing THE WHOLE dataset and the indexes of the values that are in the subset.

    name (str): name of the split set { 'train', 'val', 'test' }
    subset (Optional[Subset]): one element of the list outputted by torch.utils.data.random_split
    info_dict (Optional): dict with (key, val) = (class name, number of instances per class)

    SeeAlso:
        [PyTorch's Subset documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset)
    """
    
    name: str
    subset: Optional[Subset]
    info_dict: Optional[dict]


class CustomDataset(ABC, Dataset):

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

    def __init__(self, dataset_path: str, aug_off: Optional[List[str]], aug_on: Optional[List[str]], crop_size: int, img_size: int, split_ratios: List[float]):
        Dataset().__init__()
        self.dataset_path = dataset_path
        self.dataset_aug_path: str = os.path.join(os.path.dirname(self.dataset_path), self.AUG_DIR)
        self.filt: List[str] = list(self.label_to_idx.keys())

        self.augment_online: Optional[List[str]] = aug_on
        self.augment_offline: Optional[List[str]] = aug_off
        if self.augment_offline is not None:
            self.augment_dataset(50, eval(f"Processing.{self.augment_strategy}"))

        self.image_list: Optional[List[str]] = self.get_image_list(self.filt)
        self.label_list: Optional[List[int]] = self.get_label_list()

        self.crop_size: int = crop_size
        self.img_size: Optional[int] = img_size
        self.in_dim = self.img_size
        self.out_dim = len(self.label_to_idx)

        self.mean: Optional[float] = None
        self.std: Optional[float] = None

        self.subsets_dict: SubsetsDict = self.split_dataset(split_ratios)

    def __getitem__(self, index):
        return Dataset.__getitem__(self, index)
    
    def __len__(self):
        """ https://github.com/pytorch/pytorch/issues/25247#issuecomment-525380635
        No `def __len__(self)` default?
        See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        in pytorch/torch/utils/data/sampler.py 
        """
        return Dataset.__len__(self)

    def split_dataset(self, split_ratios: List[float]=[.8]) -> SubsetsDict:
        """Split a dataset into train, (val), test
        
        Wrap torch.utils.data.random_split with a TypedDict that includes:
        { 'train': torch.utils.data.Subset, 'val': torch.utils.data.Subset, 'test': torch.utils.data.Subset }
        The validation subset is None if not required

        Args:
            split_ratios (List[float]=[.8]): if len == 1: train/test, if len == 3 train/val/test else exception

        Returns:
            SubsetsDict (TypedDict)

        Raises:
            ValueError if split_ratios != {1, 3}

        """
        
        if type(split_ratios) is float:
            split_ratios = [split_ratios]
        
        train_len = int(len(self) * split_ratios[0])
        if len(split_ratios) == 1:
            split_lens = [train_len, len(self) - train_len]
        elif len(split_ratios) == 3:
            val_len = int(len(self) * split_ratios[1])
            split_lens = [train_len, val_len, len(self) - (train_len + val_len)]
        else:
            raise ValueError(f"split_ratios argument accepts either a list of 1 value (train,test) or 3 (train,val,test)")

        subsets = random_split(self, split_lens)
        val_set = subsets[1] if len(split_lens) == 3 and len(subsets[1]) > 0 else None

        train_str, val_str, test_str = _GC.DEFAULT_SUBSETS
        Logger.instance().debug(f"Splitting dataset: {split_lens}")
        
        return { train_str: subsets[0], val_str: val_set, test_str: subsets[-1] }   # type: ignore

    def get_subset_info(self, subset_str_id: str) -> Optional[SubsetInfo]:
        """Wrap subset into SubsetInfo structure (holds more information)

        Args:
            subset_str_id (str): { 'train', 'val', 'test' }

        Returns:
            SubsetInfo if the Subset is present (validation dataset can be None)

        Raises:
            ValueError if `subset_str_id` is not in { 'train', 'val', 'test' }
        """

        if subset_str_id not in _GC.DEFAULT_SUBSETS:
            raise ValueError(f"TrainTest::get_subset_info: only accept 'train', 'val', 'test'")
        
        if self.subsets_dict is None:
            return None
        
        if self.subsets_dict[subset_str_id] is None:
            info_dict = None
        else:
            subset_labels = [self[idx][1] for idx in self.subsets_dict[subset_str_id].indices]
            classes = list(set(subset_labels))
            info_dict = { self.idx_to_label[i]: subset_labels.count(i) for i in classes }
            Logger.instance().debug(f"{subset_str_id} has {len(classes)} classes: {info_dict}")
        
        return SubsetInfo(subset_str_id, self.subsets_dict[subset_str_id], info_dict)
    
    def augment_dataset(self, iters: int, augment_func: Optional[Callable[[List[str], str, int], None]]):
        """Perform offline augmentation
        
        Increase the number of available samples with augmentation techniques, if required in config. Offline
        augmentation can work on a limited set of classes; indeed, it should be used if there are not enough samples
        for each class.

        Args:
            iters (int): number of augmentation iteration for the same image.
            augment_fuct (Callable): function used to augment. (image list, output directory, number of iterations)
        """

        if augment_func is None:
            Logger.instance.debug("No augment function specified: return")
            return
        
        Logger.instance().debug("increasing the number of images...")
        
        if os.path.exists(self.dataset_aug_path):
            if len(os.listdir(self.dataset_aug_path)) > 0:
                Logger.instance().warning("the dataset has already been augmented")
                return
        else:
            os.makedirs(self.dataset_aug_path)
        
        image_list = self.get_image_list(self.augment_offline)
        augment_func(image_list, self.dataset_aug_path, iters)

        Logger.instance().debug("dataset augmentation completed")

    @staticmethod
    def compute_mean_std(dataset: CustomDataset, config: Config):
        # https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/31
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        mean = 0.0
        for batch in dataloader:
            images, _ = batch
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(dataloader.dataset)

        var = 0.0
        pixel_count = 0
        for batch in dataloader:
            images, _ = batch
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1))**2).sum([0,2])
            pixel_count += images.nelement()
        std = torch.sqrt(var / pixel_count)

        if any(map(lambda x: torch.isnan(x), mean)) or any(map(lambda x: torch.isnan(x), std)):
            raise ValueError("mean or std are none")

        config.dataset_mean = mean.tolist()
        config.dataset_std = std.tolist()
        config.serialize(os.getcwd(), "config/config.json")
        Logger.instance().warning(f"Mean: {mean}, std: {std}. Run the program again.")