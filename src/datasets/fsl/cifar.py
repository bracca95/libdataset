import os
import sys

from glob import glob
from typing import List, Set, Tuple, Optional
from torch.utils.data import Dataset

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG


class CifarFs(FewShotDataset):
    """CIFAR-FS

    Train/val/test are already are already split in the dataset I downloaded: check the link. This version of the
    dataset can only be used with these three default splits; write another class to use different combinations.

    SeeAlso:
        [main page](https://github.com/bertinetto/r2d2)
        [downlaod](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view?usp=sharing)
    """

    N_IMG_PER_CLASS = 600
    N_CLASSES_TRAIN = 64
    N_CLASSES_VAL = 16
    N_CLASSES_TEST = 20
    DIR_DATA = "data"
    DIR_SPLITS = "splits"
    DIR_BERTINETTO = "bertinetto"

    def __init__(self, dataset_config: DatasetConfig):
        self.data_dir, self.label_dir = self.__check_dataset_struct(dataset_config)
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        return glob(os.path.join(self.data_dir, "*", "*png"))
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        def get_class_set(split_name: str):
            split_path = Tools.validate_path(os.path.join(self.label_dir, f"{split_name}.txt"))
            with open(split_path, "r") as f:
                labels = [l.strip() for l in f if l.strip()]
            
            return set(labels)

        return get_class_set("train"), get_class_set("val"), get_class_set("test")

    def expected_length(self) -> int:
        return (CifarFs.N_CLASSES_TRAIN + CifarFs.N_CLASSES_TEST + CifarFs.N_CLASSES_VAL) * CifarFs.N_IMG_PER_CLASS
    
    @staticmethod
    def __check_dataset_struct(dataset_config: DatasetConfig) -> Tuple[str, str]:
        try:
            data_dir = Tools.validate_path(os.path.join(dataset_config.dataset_path, CifarFs.DIR_DATA))
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"`data` folder is supposed to be found in the dataset root directory. {fnf}")
            sys.exit(-1)

        try:
            label_dir = os.path.join(dataset_config.dataset_path, CifarFs.DIR_SPLITS, CifarFs.DIR_BERTINETTO)
            label_dir = Tools.validate_path(label_dir)
        except FileNotFoundError as fnf:
            Logger.instance().warning(f"{label_dir} not found, looking for $PROJ/splits/cifar_fs")
            label_dir = os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits", "cifar_fs")
            if not os.path.exists(label_dir):
                raise FileNotFoundError(f"Local {label_dir} was not provided")
            
        return data_dir, label_dir