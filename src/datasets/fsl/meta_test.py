import os
import sys
import shutil
import pandas as pd

from glob import glob
from typing import List, Set, Tuple, Optional

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig, read_from_json, write_to_json
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG


class MetaTest(FewShotDataset):

    N_IMAGES = 0

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
    

class CropDiseases(MetaTest):
    """PlantVillage Dataset AKA CropDiseases

    The orginal dataset train/test split was not meant to be used as meta-learning/FSL dataset. Here, we use the whole
    dataset (38 classes only) for validation purposes. This class does not provide any training/validation images.
    The total number of elements should be 54305.

    SeeAlso:
        [dataset](https://github.com/spMohanty/PlantVillage-Dataset)
    """

    N_CLASS_TEST = 38
    N_IMAGES = 54305

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)


class EuroSat(MetaTest):
    """EuroSAT RGB

    SeeAlso:
        [download](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)
    """
    
    N_CLASS_TEST = 10
    N_IMAGES = 27000

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)


class Isic(MetaTest):
    """ISIC Melanoma 2018

    ISIC melanoma dataset has been released several times and with different purposes (segmentation, detection,
    classification). In this case, we are using the whole trainin/val/test splits from Task 3: classification of the
    2018 version dataset (follow the link below with these instructions).

    The dataset requires some pre-processing to be adapted to few-shot tasks, so ensure to have the following structure:
    /ISIC
    |___ISIC2018_Task3_Training_Input
    |   |___ISIC_0029292.jpg ...
    |___ISIC2018_Task3_Training_GroundTruth
    |   |___ISIC2018_Task3_Training_GroundTruth.csv
    |___ISIC2018_Task3_Validation_Input
    |   |___ISIC_0029292.jpg ...
    |___ISIC2018_Task3_Validation_GroundTruth
    |   |___ISIC2018_Task3_Validation_GroundTruth.csv
    |___ISIC2018_Task3_Test_Input
    |   |___ISIC_0029292.jpg ...
    |___ISIC2018_Task3_Test_GroundTruth
    |   |___ISIC2018_Task3_Test_GroundTruth.csv
    
    SeeAlso:
        [dataset](https://challenge.isic-archive.com/data/)
    """
    
    N_CLASS_TEST = 7
    N_IMAGES = 11720

    ROOT_FOLD = "ISIC2018_Task3_"
    SUBFOLD_TRAIN = "Training"
    SUBFOLD_VAL = "Validation"
    SUBFOLD_TEST = "Test"
    LEAF_INPUT = "_Input"
    LEAF_GT = "_GroundTruth"

    DIR_IMAGES = "images"

    def __init__(self, dataset_config: DatasetConfig):
        self.__init_dataset(dataset_config)
        
        dataset_config.dataset_path = os.path.join(dataset_config.dataset_path, self.DIR_IMAGES)
        super().__init__(dataset_config)

    def __init_dataset(self, dataset_config: DatasetConfig):
        img_dir = os.path.join(os.path.join(dataset_config.dataset_path, self.DIR_IMAGES))
        if os.path.exists(img_dir) and len(os.listdir(img_dir)) == self.N_CLASS_TEST:
            if len(glob(os.path.join(img_dir, "*", "*jpg"))) == self.expected_length():
                Logger.instance().debug(f"ISIC dataset has already been pre-processed")
                return
        
        # pre-processing not performed yet
        # check correct folder existance
        try:
            ds_path = dataset_config.dataset_path
            d_train = Tools.validate_path(os.path.join(ds_path, f"{self.ROOT_FOLD}{self.SUBFOLD_TRAIN}{self.LEAF_INPUT}"))
            gt_train = Tools.validate_path(os.path.join(ds_path, f"{self.ROOT_FOLD}{self.SUBFOLD_TRAIN}{self.LEAF_GT}"))
            d_val = Tools.validate_path(os.path.join(ds_path, f"{self.ROOT_FOLD}{self.SUBFOLD_VAL}{self.LEAF_INPUT}"))
            gt_val = Tools.validate_path(os.path.join(ds_path, f"{self.ROOT_FOLD}{self.SUBFOLD_VAL}{self.LEAF_GT}"))
            d_test = Tools.validate_path(os.path.join(ds_path, f"{self.ROOT_FOLD}{self.SUBFOLD_TEST}{self.LEAF_INPUT}"))
            gt_test = Tools.validate_path(os.path.join(ds_path, f"{self.ROOT_FOLD}{self.SUBFOLD_TEST}{self.LEAF_GT}"))
        except FileNotFoundError as fnf:
            Logger.instance().error(fnf)
            sys.exit(-1)

        csv_train = os.path.join(ds_path, gt_train, f"{self.ROOT_FOLD}{self.SUBFOLD_TRAIN}{self.LEAF_GT}.csv")
        csv_val = os.path.join(ds_path, gt_val, f"{self.ROOT_FOLD}{self.SUBFOLD_VAL}{self.LEAF_GT}.csv")
        csv_test = os.path.join(ds_path, gt_test, f"{self.ROOT_FOLD}{self.SUBFOLD_TEST}{self.LEAF_GT}.csv")
        
        img_lbl_train = self._get_img_label(dataset_config, csv_train)
        img_lbl_val = self._get_img_label(dataset_config, csv_val)
        img_lbl_test = self._get_img_label(dataset_config, csv_test)

        self._move_to_class_folder(d_train, img_dir, img_lbl_train)
        self._move_to_class_folder(d_val, img_dir, img_lbl_val)
        self._move_to_class_folder(d_test, img_dir, img_lbl_test)

    def _get_img_label(self, dataset_config: DatasetConfig, csv_path: str) -> List[Tuple[str, str]]:
        # read class names
        df = pd.read_csv(csv_path)
        classes = set(df.columns.values)
        classes.remove('image')

        # create a directory for each class
        for c in classes:
            os.makedirs(os.path.join(dataset_config.dataset_path, self.DIR_IMAGES, c), exist_ok=True)

        # link every image with its own class
        image_class_pairs = []
        for index, row in df.iterrows():
            float_cols = row.drop(labels='image')
            class_name = float_cols.index[float_cols.values > 0][0]
            image_class_pairs.append((f"{row['image']}.jpg", class_name))

        return image_class_pairs
    
    @staticmethod
    def _move_to_class_folder(src_root: str, dst_root: str, img_label_list: List[Tuple[str, str]]):
        for tup in img_label_list:
            img_name = tup[0]
            c = tup[1]
            
            img_src = os.path.join(src_root, img_name)
            img_dst = os.path.join(dst_root, c, img_name)

            shutil.move(img_src, img_dst)