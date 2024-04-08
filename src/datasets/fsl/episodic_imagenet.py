import os
import sys
import pandas as pd

from glob import glob
from typing import List, Set, Tuple, Optional
from torch.utils.data import Dataset

from .dataset_fsl import FewShotDataset
from ...utils.tools import Logger, Tools
from ...utils.config_parser import DatasetConfig
from ....config.consts import General as _CG


class EpisodicImagenet(FewShotDataset):
    """EpisodicImagenet

    Provide a meta-learning split for imagenet so that the training set classes do not overlap the val/test classes
    from miniimagenet. Use miniimagenet to val/test splits. The expected input folder (root) should be at ILSVRC
    (included) and it must contain the following subfolders: 'ILSVER/Data/CLS-LOC/train'.

    SeeAlso:
        [repo](https://github.com/bracca95/imagenet2mini-labels)
    """

    MINI_N_IMG_PER_CLASS = 600
    MINI_N_CLASSES_VAL = 16
    MINI_N_CLASSES_TEST = 20
    ILSVRC_N_TRAIN_IMGS = 1234487

    def __init__(self, dataset_config: DatasetConfig):
        # must be init!!
        self.root_ilsvrc: str = str()
        self.root_mini: str = str()
        self.split_dir: str = os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits")
        self.train_labels: Set[str] = set()

        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        ## imagenet train ##

        # 1. manage error: missing "clean" version of imagenet-1k and/or train labels
        try:
            split_1k = Tools.validate_path(os.path.join(self.split_dir, "episodic_imagenet"))
            if not os.path.exists(os.path.join(split_1k, "ilsvrc_cleaned.txt")):
                raise FileNotFoundError(
                    f"File 'ilsvrc_cleaned.txt should exist. Check https://github.com/bracca95/imagenet2mini-labels"
                )
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"{fnf.args}")
            sys.exit(-1)
        
        # 2. manage error: imagenet-1k directory structure
        rltv_err: str = str()
        full_err: str = str(f"Structure for imagenet-1k should be 'ILSVER/Data/CLS-LOC/train'. Class folders located there")
        self.root_ilsvrc = os.path.join(self.dataset_config.dataset_path, "Data", "CLS-LOC", "train")
        if not "Data" in os.listdir(self.dataset_config.dataset_path):
            msg = f"Missin 'Data' directory inside ILSVRC folder"
            rltv_err = msg
        elif not "CLS-LOC" in os.listdir(os.path.join(self.dataset_config.dataset_path, "Data")):
            msg = f"Missin 'CLS-LOC' directory inside ILSVRC folder"
            rltv_err = msg
        elif not "train" in os.listdir(os.path.join(self.dataset_config.dataset_path, "Data", "CLS-LOC")):
            msg = f"Missin 'train' directory inside ILSVRC folder"
            rltv_err = msg
        elif not any(fold.startswith('n') for fold in os.listdir(self.root_ilsvrc)):
            msg = f"directories in 'train' should start with 'n' (class symbolic name)"
            rltv_err = msg
        else: pass

        # return one of the errors if any occurred
        if not rltv_err == str():
            full_err += full_err
            Logger.instance().error(full_err)
            raise ValueError(full_err)

        ## miniimagenet val/test ##
        # 3. manage error: miniimagenet is missing
        try:
            self.root_mini = os.path.join(os.path.dirname(self.dataset_config.dataset_path), "miniimagenet")
            self.root_mini = Tools.validate_path(self.root_mini)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"Where is 'miniimagenet' dataset? Put is at the same level as ILSVRC {fnf.args}")
            sys.exit(-1)

        # everything should be ok up to this point
        
        # read imagenet-1k images stripped of imagenet trainig set, also get unique train labels (required later)
        with open(os.path.join(split_1k, "ilsvrc_cleaned.txt"), "r") as f:
            ilsvrc_img_paths: List[str] = list()
            for l in f:
                if l.strip():
                    label = l.strip().split("_")[0]
                    ilsvrc_img_paths.append(os.path.join(self.root_ilsvrc, label, l.strip()))
                    self.train_labels.add(label)

        # filter training images on miniimagenet
        mini_img_paths = glob(os.path.join(self.root_mini, "n*", "*JPEG"))
        old_train_mini_csv = Tools.validate_path(os.path.join(self.split_dir, "miniimagenet", "train.csv"))
        df_train = pd.read_csv(old_train_mini_csv)
        old_train_mini = set(df_train["label"].values)
        mini_img_paths = list(filter(
            lambda x: not any(os.path.basename(os.path.dirname(x)) == c for c in old_train_mini), mini_img_paths
        ))

        return ilsvrc_img_paths + mini_img_paths
    
    def expected_length(self) -> int:
        return self.ILSVRC_N_TRAIN_IMGS + (self.MINI_N_CLASSES_TEST + self.MINI_N_CLASSES_VAL) * self.MINI_N_IMG_PER_CLASS
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        split_1k = Tools.validate_path(os.path.join(self.split_dir, "episodic_imagenet"))
        split_mini = Tools.validate_path(os.path.join(self.split_dir, "miniimagenet"))

        def get_class_set(split_name: str):
            # use imagenet train
            if split_name == "train":
                return self.train_labels
            
            # use mini-imagenet val/test
            split_path = Tools.validate_path(os.path.join(split_mini, f"{split_name}.csv"))
            df = pd.read_csv(split_path)
            return set(df["label"].values)
        
        return get_class_set("train"), get_class_set("val"), get_class_set("test")
