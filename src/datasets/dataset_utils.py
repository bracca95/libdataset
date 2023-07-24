import os
import torch

from typing import Optional, List
from dataclasses import dataclass
from torch.utils.data import Subset, Dataset, DataLoader, random_split

from src.utils.tools import Logger
from src.utils.config_parser import Config
from src.datasets.staple_dataset import CustomDataset
from src.datasets.defectviews import GlassOpt, GlassOptBckg, GlassOptTricky, BubblePoint, QPlusV1, QPlusV2
from config.consts import SubsetsDict
from config.consts import General as _GC


@dataclass
class SubsetInfo:
    """SubsetInfo dataclass

    The aim of this class is to wrap the output of torch.utils.data.random_split, so that it contains the information
    related to the number of samples that belong to the split subsets. If you are creating your own dataset and then
    you want to use pytorch's random_split method, this is the way to go not to lose information.

    name (str): name of the split set { 'train', 'val', 'test' }
    subset (Optional[Subset]): one element of the list outputted by torch.utils.data.random_split
    info_dict (Optional): dict with (key, val) = (class name, number of instances per class)
    """
    
    name: str
    subset: Optional[Subset]
    info_dict: Optional[dict]


class DatasetBuilder:

    @staticmethod
    def load_dataset(config: Config) -> CustomDataset:
        if config.dataset_type == "opt6":
            return GlassOpt(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size)
        elif config.dataset_type == "opt_bckg":
            return GlassOptBckg(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size)
        elif config.dataset_type == "opt_tricky":
            return GlassOptTricky(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size)
        elif config.dataset_type == "qplusv1":
            return QPlusV1(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size)
        elif config.dataset_type == "qplusv2":
            return QPlusV2(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size)
        elif config.dataset_type == "binary":
            return BubblePoint(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size)
        else:
            raise ValueError("values allowed: {`opt6`, `opt_bckg`, `binary`, `qplusv1`, `qplusv2`} for dataset_type")
        
    @staticmethod
    def compute_mean_std(dataset: Dataset, config: Config):
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

    @staticmethod
    def split_dataset(dataset: Dataset, split_ratios: List[float]=[.8]) -> SubsetsDict:
        """Split a dataset into train, (val), test
        
        Wrap torch.utils.data.random_split with a TypedDict that includes:
        {'train': torch.utils.data.torch.utils.data.Subset, 'val': torch.utils.data.Subset, 'test': torch.utils.data.Subset}
        The validation subset is None if not required

        Args:
            dataset (torch.utils.data.Dataset)
            split_ratios (List[float]=[.8]): if len == 1: train/test, if len == 3 train/val/test else exception

        Returns:
            SubsetsDict (TypedDict)

        Raises:
            ValueError if split_ratios != {1, 3}

        """
        
        if type(split_ratios) is float:
            split_ratios = [split_ratios]
        
        train_len = int(len(dataset) * split_ratios[0])
        if len(split_ratios) == 1:
            split_lens = [train_len, len(dataset) - train_len]
        elif len(split_ratios) == 3:
            val_len = int(len(dataset) * split_ratios[1])
            split_lens = [train_len, val_len, len(dataset) - (train_len + val_len)]
        else:
            raise ValueError(f"split_ratios argument accepts either a list of 1 value (train,test) or 3 (train,val,test)")

        subsets = random_split(dataset, split_lens)
        val_set = subsets[1] if len(split_lens) == 3 and len(subsets[1]) > 0 else None

        train_str, val_str, test_str = _GC.DEFAULT_SUBSETS
        Logger.instance().debug(f"Splitting dataset: {split_lens}")
        
        return { train_str: subsets[0], val_str: val_set, test_str: subsets[-1] }   # type: ignore