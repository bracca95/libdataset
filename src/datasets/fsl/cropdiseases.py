import os
import torch

from glob import glob
from typing import List, Set, Tuple, Optional

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG


class CropDiseases(FewShotDataset):
    """PlantVillage Dataset AKA CropDiseases

    The orginal dataset train/test split was not meant to be used as meta-learning/FSL dataset. Here, we use the whole
    dataset (38 classes only) for validation purposes. This class does not provide any training/validation images.
    The total number of elements should be 54305.

    SeeAlso:
        [dataset](https://github.com/spMohanty/PlantVillage-Dataset)
    """

    N_CLASSES = 38
    N_CLASS_TRAIN = 0
    N_CLASS_VAL = 0
    N_CLASS_TEST = 38
    N_IMAGES = 54305

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        avail_ext = ("jpeg", "jpg", "png", "JPG")
        img_list = glob(os.path.join(self.dataset_config.dataset_path, "*", "*"))
        img_list = list(filter(lambda x: x.endswith(avail_ext), img_list))
        return img_list
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        ds_path = self.dataset_config.dataset_path
        class_test = os.listdir(ds_path)
        class_test = set([e for e in class_test if os.path.isdir(os.path.join(ds_path, e)) and not e.startswith(".")])
        return set(), set(), class_test

    def expected_length(self) -> int:
        return self.N_IMAGES