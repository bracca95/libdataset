import os
import re
import time
import pandas as pd

from glob import glob
from difflib import SequenceMatcher
from typing import List, Set, Tuple, Optional

from .meta_test import MetaTest
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG


class WikiArt(MetaTest):
    """WikiArt dataset (wrapper)

    The WikiArt dataset originally contains 1119 artists, 11 genres and 27 styles. Some use 129 artist, but most of them
    are grouped under "0", which does not make any sense. We further process it to removes some corrupted filenames
    and the artists that do not have at least 10 paintings. Finally, we have 966 unique artists.

    SeeAlso:
        [download](https://www.kaggle.com/datasets/steubk/wikiart)
    """

    N_IMAGES = 80091 # (80675, 81444) (<10 samples artist, corrupted filenames)
    COL_FILE = "file"
    COL_ARTIST = "artist"
    COL_GENRE = "genre"
    COL_STYLE = "style"
    CLS_FILE = "wclasses.csv"
    LABEL_COL = str()

    def __init__(self, dataset_config: DatasetConfig):
        self.df = self._preprocess(dataset_config)
        super().__init__(dataset_config)
    
    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        img_list = [os.path.join(self.dataset_config.dataset_path, p) for p in self.df[self.COL_FILE]]
        return img_list

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
        pattern = r'/(?P<name>[^/_]+)_'
        
        df = pd.read_csv(os.path.join(dataset_config.dataset_path, self.CLS_FILE))
        artist_list = [re.search(pattern, path).group("name") for path in df[self.COL_FILE]]
        df[self.COL_ARTIST] = artist_list

        # fix incorrect encoding (two options)
        real_files = glob(os.path.join(dataset_config.dataset_path, "*", "*jpg"))
        image_list = [os.path.join(dataset_config.dataset_path, p) for p in df[self.COL_FILE]]
        missing = set(image_list) - set(real_files)
        df = df[~df[self.COL_FILE].isin([m.replace(dataset_config.dataset_path + os.sep, "") for m in missing])]

        # keep only artists with at least 10 entries
        artist_counts = df[self.COL_ARTIST].value_counts()
        artists_to_keep = artist_counts[artist_counts >= 10].index
        df = df[df[self.COL_ARTIST].isin(artists_to_keep)]

        return df


class WikiArtArtist(WikiArt):

    LABEL_COL = WikiArt.COL_ARTIST

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
    

class WikiArtGenre(WikiArt):

    LABEL_COL = WikiArt.COL_GENRE

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)


class WikiArtStyle(WikiArt):

    LABEL_COL = WikiArt.COL_STYLE

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

