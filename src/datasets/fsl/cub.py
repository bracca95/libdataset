import os
import torch

from glob import glob
from typing import List, Set, Tuple, Optional

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG

class Cub(FewShotDataset):

    """CUB 200 2011

    The orginal dataset train/test split does not account for a validation set, and most importantly it does not split
    train and test classes: we want to classify unseen classes, not unseen instances!

    Not all classes have the same number of samples: 144/200 classes have exactly 60 samples. The remaining classes
    have less samples.

    SeeAlso:
        [FSL dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)
    """

    N_CLASSES = 200
    N_IMAGES = 11788

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
        
    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        try:
            img_dir = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, "images"))
        except FileNotFoundError as fnf:
            msg = f"A Directory `images` (with all the subdirs of the classes) must be contained in the path " + \
                  f"specified in config.json. You chose {self.dataset_config.dataset_path} but it looks wrong."
            Logger.instance().critical(f"{fnf}\n{msg}")
            raise FileNotFoundError(msg)
        
        images = glob(os.path.join(img_dir, "*", "*.jpg"))

        # check
        if not len(images) == self.N_IMAGES:
            msg = f"The number of images ({len(images)}) should be {self.N_IMAGES}"
            Logger.instance().critical(msg)
            raise ValueError(msg)
        
        return images
    
    def expected_length(self) -> int:
        return self.N_IMAGES
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """Random split based on the number of classes
        
        This function provides a better implementation for splitting a FSL dataset, so that the classes in 
        train/val/test splits do not intersect. Call this method inside the overwritten version of split_dataset().
        """

        split_ratios = self.dataset_config.dataset_splits

        if len(split_ratios) == 1:
            split_ratios = [split_ratios[0], 1.0 - split_ratios[0]]
        if len(split_ratios) == 2 or len(split_ratios) > 3:
            raise ValueError(f"split_ratios argument accepts either a list of 1 value (train,test) or 3 (train,val,test)")

        n_classes = len(self.label_to_idx.keys())
        shuffled_idxs = torch.randperm(n_classes)
        split_points = [int(n_classes * ratio) for ratio in split_ratios]

        # if no validation is used, set eof_split to 0 so that the last index starts with the end of train
        if len(split_points) == 3:
            eof_split = split_points[1]
            class_val_idx = set((shuffled_idxs[split_points[0]:split_points[0] + split_points[1]]).tolist())
        else:
            eof_split = 0
            class_val_idx = set()
        class_val = set([self.idx_to_label[c] for c in class_val_idx])

        # Split the numbers into two or three sets
        class_train_idx = set((shuffled_idxs[:split_points[0]]).tolist())
        class_test_idx = set((shuffled_idxs[split_points[0] + eof_split:]).tolist())

        class_train = set([self.idx_to_label[c] for c in class_train_idx])
        class_test = set([self.idx_to_label[c] for c in class_test_idx])

        if not len(shuffled_idxs) == len(class_train) + len(class_val) + len(class_test):
            raise ValueError(f"The number of classes {shuffled_idxs} does not match the split.")
        
        return class_train, class_val, class_test