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


class EpisodicImagenet1k(FewShotDataset):
    """EpisodicImagenet

    Use all the training samples included in the training set of imagenet-1k to replicate the results of CAML. They were
    not supposed to use all the classes if the test were to be made on miniimagenet because the images are the same, but
    we need this to replicate.
    The val and test set are still extracted from miniimagenet.

    SeeAlso:
        - [CAML](https://arxiv.org/abs/2310.10971)
    """

    ILSVRC_N_IMGS = 1281167

    def __init__(self, dataset_config: DatasetConfig):
        # must be init!!
        self.root_ilsvrc: str = str()
        self.split_dir: str = os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits")
        self.train_labels: Set[str] = set()

        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        ## imagenet train ##
        
        # 2. manage error: imagenet-1k directory structure
        rltv_err: str = str()
        full_err: str = str(f"Structure for imagenet-1k should be 'ILSVER/Data/CLS-LOC/train'. Class folders located there")
        self.root_ilsvrc = os.path.join(self.dataset_config.dataset_path, "Data", "CLS-LOC", "train")
        if not "Data" in os.listdir(self.dataset_config.dataset_path):
            msg = f"Missing 'Data' directory inside ILSVRC folder"
            rltv_err = msg
        elif not "CLS-LOC" in os.listdir(os.path.join(self.dataset_config.dataset_path, "Data")):
            msg = f"Missing 'CLS-LOC' directory inside ILSVRC folder"
            rltv_err = msg
        elif not "train" in os.listdir(os.path.join(self.dataset_config.dataset_path, "Data", "CLS-LOC")):
            msg = f"Missing 'train' directory inside ILSVRC folder"
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
        # 2. manage error: miniimagenet is missing
        try:
            self.root_mini = os.path.join(os.path.dirname(self.dataset_config.dataset_path), "miniimagenet")
            self.root_mini = Tools.validate_path(self.root_mini)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"Where is 'miniimagenet' dataset? Put is at the same level as ILSVRC {fnf.args}")
            sys.exit(-1)
        
        # read imagenet-1k images: they also include miniimagenet val/test
        ilsvrc_img_paths = glob(os.path.join(self.root_ilsvrc, "n*", "*JPEG"))
        self.train_labels = set(list(map(lambda x: os.path.basename(os.path.dirname(x)), ilsvrc_img_paths)))

        return ilsvrc_img_paths
    
    def expected_length(self) -> int:
        return self.ILSVRC_N_IMGS
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
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
