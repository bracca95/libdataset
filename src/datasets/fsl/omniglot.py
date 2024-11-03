import os
import torch

from glob import glob
from typing import List, Tuple, Set, Optional
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Omniglot
from torchvision.transforms import transforms

from .dataset_fsl import FewShotDataset
from ..dataset import DatasetWrapper, DatasetLauncher
from ...utils.tools import Tools, Logger
from ...utils.config_parser import DatasetConfig
from ....config.consts import General as _CG


class OmniglotWrapper(FewShotDataset):

    N_IMG_PER_CLASS = 20
    N_CLASSES_TRAIN = 964
    N_CLASSES_TEST = 659
    DATA_DIR = "omniglot-py"
    TRAIN_DIR = "images_background"
    TEST_DIR = "images_evaluation"

    def __init__(self, dataset_config: DatasetConfig):
        self.data_dir = os.path.join(dataset_config.dataset_path, self.DATA_DIR)
        self._check_integrity(self.data_dir)
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        train = glob(os.path.join(self.data_dir, self.TRAIN_DIR, "*", "character*", "*png"))
        test = glob(os.path.join(self.data_dir, self.TEST_DIR, "*", "character*", "*png"))
        return train + test
    
    def get_label_list(self) -> List[int]:
        """
        Omniglot starts counting labels from 1, so 1 to 1623. For index to label, we use 0 to 1622 instead.
        Thus idx_to_label will be 0: "1" and label_to_idx "1": 0.
        I can sort the labels since they are ordered (see function `_check_labels`).
        """

        if self.image_list is None:
            self.image_list = self.get_image_list(None)

        label_set = sorted(set([os.path.basename(image_name).rsplit("_", 1)[0] for image_name in self.image_list]))

        # extract unique label values from dirname
        self.label_to_idx = { f"{val}": i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        # use mapping to return an int for the corresponding str label
        return [self.label_to_idx[os.path.basename(image_name.rsplit("_", 1)[0])] for image_name in self.image_list]
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        train_end = self.N_CLASSES_TRAIN + 1
        test_end = self.N_CLASSES_TRAIN + self.N_CLASSES_TEST + 1
        class_train = set([f"{x:04d}" for x in range(1, train_end)])
        class_test = set([f"{x:04d}" for x in range(train_end, test_end)])

        return class_train, set(), class_test
    
    def expected_length(self) -> int:
        return (self.N_CLASSES_TRAIN + self.N_CLASSES_TEST) * self.N_IMG_PER_CLASS
    
    @staticmethod
    def _check_labels(train_or_test: List[str]):
        """
        Omniglot labels go from 1 to 1623 (they do not start from 0). The label is embedded in the filename, just before
        the last "_" character. This function proves that labels up until "964" belong to the evaluation and the 
        remaining are left to the evaluation set (hence, they are ordered)

        Args:
            train_or_test (List[str]): train list or test list
        """

        classes = set([os.path.basename(name).rsplit("_", 1)[0] for name in train_or_test])
        classes_int = [int(l) for l in classes]
        _min = min(classes_int)
        _max = max(classes_int)

        # train_set_int == [i for i in range(1, 965)] -> True

    @staticmethod
    def _check_integrity(path: str):
        """Omniglot integrity, given torch dataset

        This method prevents any other check on the number of images, if performed immediately.
        Number of train images 19280: 20 people per character, so 19280/20 = 964 characters (N_CLASSES_TRAIN)
        Number of test images 13180: 20 people per character, so 13180/20 = 659 characters (N_CLASSES_TEST)
        """

        train = Omniglot(
            os.path.dirname(path),
            background=True,
            transform=transforms.ToTensor(),
            download=True
        )

        test = Omniglot(
            os.path.dirname(path),
            background=False,
            transform=transforms.ToTensor(),
            download=True
        )