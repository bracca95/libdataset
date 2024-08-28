import os
import torch

from glob import glob
from typing import List, Set, Tuple, Optional

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG

class Metainat(FewShotDataset):
    """Meta-iNat dataset

    SeeAlso:
        [FSL dataset](https://github.com/daviswer/fewshotlocal?tab=readme-ov-file#setup)
    """

    N_CLASSES = 1135
    N_CLASS_TRAIN = 908
    N_CLASS_VAL = 0
    N_CLASS_TEST = 227
    N_IMAGES = 243986
    TRAIN_DIR = "train"
    TEST_DIR = "test"

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.train_dir_path = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, self.TRAIN_DIR))
        self.test_dir_path = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, self.TEST_DIR))
        
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        train_img_list = glob(os.path.join(self.train_dir_path, "*", "*.png"))
        test_img_list = glob(os.path.join(self.test_dir_path, "*", "*.png"))
        img_list = train_img_list + test_img_list
        
        return img_list
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        cls_train = list(filter(lambda x: os.path.isdir(os.path.join(self.train_dir_path, x)), os.listdir(self.train_dir_path)))
        cls_test = list(filter(lambda x: os.path.isdir(os.path.join(self.test_dir_path, x)), os.listdir(self.test_dir_path)))
        
        class_train = set(cls_train)
        class_val = set()
        class_test = set(cls_test)
        
        return class_train, class_val, class_test

    def expected_length(self) -> int:
        return self.N_IMAGES