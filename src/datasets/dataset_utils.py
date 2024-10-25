from typing import Union
from torch.utils.data import Dataset

from .dataset import DatasetWrapper
from .fsl.dagm_adac import DagmAdac, MetaDagm
from .fsl.mvtec_adac import MvtecAdac, MetaMvtec
from .fsl.cifar import CifarFs
from .fsl.miniimagenet import MiniImagenet
from ..utils.config_parser import DatasetConfig
from ..utils.tools import Logger


class DatasetBuilder:

    @staticmethod
    def load_dataset(dataset_config: DatasetConfig) -> Union[DatasetWrapper, Dataset]:
        if dataset_config.dataset_type == "mvtec":
            Logger.instance().debug(f"Loading dataset MVTec")
            return MvtecAdac(dataset_config)
        elif dataset_config.dataset_type == "meta_mvtec":
            Logger.instance().debug(f"Loading dataset meta-MVTec")
            return MetaMvtec(dataset_config, None)
        elif dataset_config.dataset_type == "dagm":
            Logger.instance().debug(f"Loading dataset DAGM")
            return DagmAdac(dataset_config)
        elif dataset_config.dataset_type == "meta_dagm":
            Logger.instance().debug(f"Loading dataset meta-DAGM")
            return MetaDagm(dataset_config, None)
        
        
        # debug
        elif dataset_config.dataset_type == "miniimagenet":
            Logger.instance().debug(f"Loading dataset miniimagenet")
            return MiniImagenet(dataset_config)
        elif dataset_config.dataset_type == "cifar_fs":
            Logger.instance().debug(f"Loading dataset cifar_fs")
            return CifarFs(dataset_config)
        else:
            raise ValueError(
                "values allowed: {`mvtec`, `meta_mvtec`, `dagm`, `meta_dagm`, `miniimagenet`, `cifar_fs`}"
            )
