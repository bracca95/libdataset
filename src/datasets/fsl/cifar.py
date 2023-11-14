import os

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

    META_TRAIN = "meta_train"
    META_TEST = "meta_test"
    META_VAL = "meta_val"

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.train_dir = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, self.META_TRAIN))
        self.val_dir = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, self.META_VAL))
        self.test_dir = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, self.META_TEST))
        self._check_meta_split()
        
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        img_train = glob(os.path.join(self.train_dir, "*", "*.png"))
        img_val = glob(os.path.join(self.val_dir, "*", "*.png"))
        img_test = glob(os.path.join(self.test_dir, "*", "*.png"))

        if not len(img_train) + len(img_val) + len(img_test) == self.expected_length():
            raise ValueError(f"There should be 600 images in all the 100 CIFAR classes.")

        return img_train + img_val + img_test
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        class_train = set(filter(lambda x: os.path.isdir(os.path.join(self.train_dir, x)), os.listdir(self.train_dir)))
        class_val = set(filter(lambda x: os.path.isdir(os.path.join(self.val_dir, x)), os.listdir(self.val_dir)))
        class_test = set(filter(lambda x: os.path.isdir(os.path.join(self.test_dir, x)), os.listdir(self.test_dir)))

        if not len(class_train) == CifarFs.N_CLASSES_TRAIN or \
           not len(class_val) == CifarFs.N_CLASSES_VAL or \
           not len(class_test) == CifarFs.N_CLASSES_TEST:
            raise ValueError(f"Some class is missing in cifar-fs directory or it has been filtered.")
        
        return class_train, class_val, class_test

    def expected_length(self) -> int:
        return (CifarFs.N_CLASSES_TRAIN + CifarFs.N_CLASSES_TEST + CifarFs.N_CLASSES_VAL) * CifarFs.N_IMG_PER_CLASS
    
    def _check_meta_split(self):
        if not set([self.META_TRAIN, self.META_TEST]) <= set(os.listdir(self.dataset_config.dataset_path)):
            msg = f"'meta_train' and/or 'meta_test' dir/s missing in {self.dataset_config.dataset_path}"
            Logger.instance().error(msg)
            raise ValueError(msg)
