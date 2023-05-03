from typing import Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, Subset

from src.utils.config_parser import Config
from src.datasets.defectviews import DefectViews, BubblePoint


@dataclass
class SubsetInfo:
    name: str
    subset: Optional[Subset]
    info_dict: Optional[dict]


class CustomDatasetLoader:

    @staticmethod
    def load_dataset(config: Config) -> Dataset:
        if config.dataset_type == "all":
            return DefectViews(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size)
        elif config.dataset_type == "binary":
            return BubblePoint(config.dataset_path, config.augment_online, config.crop_size)
        else:
            raise ValueError("either `all` or `binary` for dataset_type")


