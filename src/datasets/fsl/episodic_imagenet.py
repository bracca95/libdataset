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
    
    VAL_DATASET = "miniimagenet"
    N_VAL_IMAGES = MINI_N_IMG_PER_CLASS * MINI_N_CLASSES_VAL
    N_TEST_IMAGES = MINI_N_IMG_PER_CLASS * MINI_N_CLASSES_TEST

    def __init__(self, dataset_config: DatasetConfig):
        # must be init!!
        self.root_ilsvrc: str = str()
        self.root_val: str = str()
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

        ## val/test ##
        # 3. manage error: val/test is missing
        self.root_val = self._check_val_test_existance()

        # everything should be ok up to this point
        
        # read imagenet-1k images stripped of imagenet trainig set, also get unique train labels (required later)
        with open(os.path.join(split_1k, "ilsvrc_cleaned.txt"), "r") as f:
            ilsvrc_img_paths: List[str] = list()
            for l in f:
                if l.strip():
                    label = l.strip().split("_")[0]
                    ilsvrc_img_paths.append(os.path.join(self.root_ilsvrc, label, l.strip()))
                    self.train_labels.add(label)

        val_test_paths = self._get_val_test_imgs()
        return ilsvrc_img_paths + val_test_paths
    
    def expected_length(self) -> int:
        return self.ILSVRC_N_TRAIN_IMGS + self.N_VAL_IMAGES + self.N_TEST_IMAGES
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        split_val_ds = Tools.validate_path(os.path.join(self.split_dir, "miniimagenet"))

        def get_class_set(split_name: str):
            # use imagenet train
            if split_name == "train":
                return self.train_labels
            
            # use mini-imagenet val/test
            split_path = Tools.validate_path(os.path.join(split_val_ds, f"{split_name}.csv"))
            df = pd.read_csv(split_path)
            return set(df["label"].values)
        
        return get_class_set("train"), get_class_set("val"), get_class_set("test")
    
    def _check_val_test_existance(self):
        try:
            root_val_test = os.path.join(os.path.dirname(self.dataset_config.dataset_path), self.VAL_DATASET)
            root_val_test = Tools.validate_path(root_val_test)
        except FileNotFoundError as fnf:
            Logger.instance().critical(
                f"Where is {self.VAL_DATASET} dataset? Put is at the same level as ILSVRC {fnf.args}"
            )
            sys.exit(-1)

        return root_val_test
    
    def _get_val_test_imgs(self):
        # filter training images on miniimagenet
        mini_img_paths = glob(os.path.join(self.root_val, "n*", "*JPEG"))
        old_train_mini_csv = Tools.validate_path(os.path.join(self.split_dir, "miniimagenet", "train.csv"))
        df_train = pd.read_csv(old_train_mini_csv)
        old_train_mini = set(df_train["label"].values)
        mini_img_paths = list(filter(
            lambda x: not any(os.path.basename(os.path.dirname(x)) == c for c in old_train_mini), mini_img_paths
        ))

        return mini_img_paths

####
# different validation sets
####
class EpisodicImagenetValCifar(EpisodicImagenet):

    CIFAR_N_IMG_PER_CLASS = 600
    CIFAR_N_CLASSES_VAL = 16
    CIFAR_N_CLASSES_TEST = 20

    VAL_DATASET = "cifar100"
    N_VAL_IMAGES = CIFAR_N_IMG_PER_CLASS * CIFAR_N_CLASSES_VAL
    N_TEST_IMAGES = CIFAR_N_IMG_PER_CLASS * CIFAR_N_CLASSES_TEST

    DIR_DATA = "data"

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        split_val_ds = Tools.validate_path(os.path.join(self.split_dir, "cifar_fs"))

        def get_class_set(split_name: str):
            # use imagenet train
            if split_name == "train":
                return self.train_labels
            
            split_path = Tools.validate_path(os.path.join(split_val_ds, f"{split_name}.txt"))
            with open(split_path, "r") as f:
                labels = [l.strip() for l in f if l.strip()]
            
            return set(labels)

        return get_class_set("train"), get_class_set("val"), get_class_set("test")

    def _get_val_test_imgs(self):
        split_val_ds = Tools.validate_path(os.path.join(self.split_dir, "cifar_fs"))
        
        val_test_labels = []
        for split_name in ["val", "test"]:
            split_path = Tools.validate_path(os.path.join(split_val_ds, f"{split_name}.txt"))
            with open(split_path, "r") as f:
                labels = [l.strip() for l in f if l.strip()]
            val_test_labels.extend(labels)
        
        data_dir = os.path.join(self._check_val_test_existance(), self.DIR_DATA)

        val_test_img_paths = []
        for class_name in val_test_labels:
            val_test_img_paths.extend(glob(os.path.join(data_dir, class_name, "*png")))
        
        return val_test_img_paths


class EpisodicImagenetValCub(EpisodicImagenet):

    CUB_N_CLASS_VAL = 30
    CUB_N_CLASS_TEST = 30
    CUB_N_VAL_TEST_IMAGES = 3549
    IMG_DIR = "images"

    VAL_DATASET = f"CUB{os.sep}CUB_200_2011"

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        obj = Tools.read_json(os.path.join(self.split_dir, "cub", "cu_birds_splits.json"))
        
        class_val = set(obj.get("valid"))
        class_test = set(obj.get("test"))
        
        return self.train_labels, class_val, class_test
    
    def expected_length(self) -> int:
        return self.ILSVRC_N_TRAIN_IMGS + self.CUB_N_VAL_TEST_IMAGES

    def _get_val_test_imgs(self):
        obj = Tools.read_json(os.path.join(self.split_dir, "cub", "cu_birds_splits.json"))
        
        class_val = set(obj.get("valid"))
        class_test = set(obj.get("test"))
        val_test_labels = class_val | class_test

        data_dir = os.path.join(self._check_val_test_existance(), self.IMG_DIR)

        val_test_img_paths = []
        for class_name in val_test_labels:
            val_test_img_paths.extend(glob(os.path.join(data_dir, class_name, "*jpg")))
        
        return val_test_img_paths
    

class EpisodicImagenetValAircraft(EpisodicImagenet):

    AIRCRAFT_N_CLASS_VAL = 15
    AIRCRAFT_N_CLASS_TEST = 15
    AIRCRAFT_N_CLASS_IMG = 100
    DIR_DATA = "data"
    DIR_SPLITS = "splits"

    VAL_DATASET = "aircraft"
    N_VAL_IMAGES = AIRCRAFT_N_CLASS_VAL * AIRCRAFT_N_CLASS_IMG
    N_TEST_IMAGES = AIRCRAFT_N_CLASS_TEST * AIRCRAFT_N_CLASS_IMG

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        obj = Tools.read_json(os.path.join(self.split_dir, "aircraft", "aircraft_splits.json"))
        
        class_val = set(obj.get("valid"))
        class_test = set(obj.get("test"))
        
        return self.train_labels, class_val, class_test

    def _get_val_test_imgs(self):
        obj = Tools.read_json(os.path.join(self.split_dir, "aircraft", "aircraft_splits.json"))
        
        class_val = set(obj.get("valid"))
        class_test = set(obj.get("test"))
        val_test_labels = class_val | class_test

        try:
            data_dir = Tools.validate_path(os.path.join(self._check_val_test_existance(), self.DIR_DATA, self.DIR_SPLITS))
        except FileNotFoundError as fnf:
            Logger.instance().critical(
                f"Aircraft has not been preprocessed. Before running this, set the following parameters:\n" +
                f"'dataset_path': '/path/to/aircraft'\n" +
                f"'dataset_type': 'aircraft'"
            )

        val_test_img_paths = []
        for class_name in val_test_labels:
            val_test_img_paths.extend(glob(os.path.join(data_dir, class_name, "*jpg")))
        
        return val_test_img_paths