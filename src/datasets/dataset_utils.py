from typing import Optional
from dataclasses import dataclass
from torch.utils.data import Subset

from src.utils.config_parser import Config
from src.datasets.staple_dataset import CustomDataset
from src.datasets.defectviews import DefectViews, BubblePoint


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
        if config.dataset_type == "all":
            return DefectViews(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size)
        elif config.dataset_type == "binary":
            return BubblePoint(config.dataset_path, config.augment_online, config.crop_size, config.image_size)
        else:
            raise ValueError("either `all` or `binary` for dataset_type")


