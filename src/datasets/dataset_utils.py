from typing import Union
from torch.utils.data import Dataset

from .dataset import DatasetWrapper
from .fsl.mvtec_adac import MvtecAdac, MetaMvtec
from ..utils.config_parser import DatasetConfig
from ..utils.tools import Logger


class DatasetBuilder:

    @staticmethod
    def load_dataset(dataset_config: DatasetConfig) -> Union[DatasetWrapper, Dataset]:
        if dataset_config.dataset_type == "mvtec":
            return MvtecAdac(dataset_config)
        elif dataset_config.dataset_type == "meta_mvtec":
            return MetaMvtec(dataset_config, None)
        else:
            raise ValueError(
                "values allowed: {`mvtec`, `meta_mvtec`}"
            )
