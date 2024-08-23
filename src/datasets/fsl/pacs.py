import os
import re
import time
import shutil
import pandas as pd

from glob import glob
from typing import List, Set, Tuple, Optional

from .meta_test import MetaTest
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG


class Pacs(MetaTest):
    """PACS

    PACS is an image dataset for domain generalization. It consists of four domains, namely:
    Photo (1,670 images), Art Painting (2,048 images), Cartoon (2,344 images) and Sketch (3,929 images). 
    Each domain contains seven categories.

    SeeAlso:
        [FSL dataset](https://paperswithcode.com/dataset/pacs)
        [download](https://www.kaggle.com/datasets/nickfratto/pacs-dataset)
    """

    N_IMAGES = 9991
    
    CSV_FILE = "label.csv"
    COL_FILE = "file"
    COL_OBJECT = "object"
    COL_DOMAIN = "domain"
    
    DATA_DIR = f"pacs_data{os.sep}pacs_data"

    LABEL_COL = str()

    def __init__(self, dataset_config: DatasetConfig):
        self.df = self._preprocess(dataset_config)
        super().__init__(dataset_config)
    
    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        avail_ext = ("jpeg", "jpg", "png", "JPG")
        images = glob(os.path.join(self.dataset_config.dataset_path, self.DATA_DIR, "*", "*", "*"))
        images = list(filter(lambda x: x.endswith(avail_ext), images))
        
        return images

    def get_label_list(self) -> List[int]:
        if self.image_list is None:
            self.image_list = self.get_image_list(None)

        # extract unique label values from dirname
        label_set = set(self.df[self.LABEL_COL].to_list())
        self.label_to_idx = { val: i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        # use mapping to return an int for the corresponding str label
        return [self.label_to_idx[l] for l in self.df[self.LABEL_COL]]
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        return set(), set(), set(self.label_to_idx.keys())
    
    def expected_length(self) -> int:
        return self.N_IMAGES

    def _preprocess(self, dataset_config: DatasetConfig) -> pd.DataFrame:
        elems = os.listdir(dataset_config.dataset_path)
        if self.CSV_FILE in elems and not "dct2_images" in elems:
            Logger.instance().debug("PACS dataset already processed")
            return pd.read_csv(os.path.join(dataset_config.dataset_path, self.CSV_FILE))
        
        # remove directory handling exception
        try:
            shutil.rmtree(os.path.join(dataset_config.dataset_path, "dct2_images"))
        except FileNotFoundError as fnf:
            Logger.instance().debug(f"Trying to delete dct2_images folder yet not present. Do not worry though.")
        
        # crawl all possible image extension
        avail_ext = ("jpeg", "jpg", "png", "JPG")
        images = glob(os.path.join(dataset_config.dataset_path, self.DATA_DIR, "*", "*", "*"))
        images = list(filter(lambda x: x.endswith(avail_ext), images))

        # define the two labels set
        objects = [os.path.basename(os.path.dirname(i)) for i in images]
        domains = [os.path.basename(os.path.dirname(os.path.dirname(i))) for i in images]
        
        # write csv to manage labels
        df = pd.DataFrame({
            self.COL_FILE: images,
            self.COL_OBJECT: objects,
            self.COL_DOMAIN: domains
        })

        df.to_csv(os.path.join(dataset_config.dataset_path, self.CSV_FILE))
        return df


class PacsObject(Pacs):

    LABEL_COL = Pacs.COL_OBJECT

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
    

class PacsDomain(Pacs):

    LABEL_COL = Pacs.COL_DOMAIN

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
