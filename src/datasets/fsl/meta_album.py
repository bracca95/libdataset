import os
import openml
import random
import numpy as np
import pandas as pd

from glob import glob
from typing import Optional, List, Set, Tuple

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG


class MetaAlbum(FewShotDataset):
    """Few-shot set Meta Album

    The idea is to use a bunch of dataset to train a model. The exact dataset to download is defined by `did`: if no
    did is provided, the program will read the config file and use the first element of the list (this is usually
    done at the beginning, to load a default dataset [44334]). Remind to always specify another `dataset_id` later.

    [Meta Album](https://meta-album.github.io/)
    """

    OPENML_DATASET_LEAF = f"{os.sep}".join("org/openml/www/datasets".split("/"))
    DIR_IMAGES = "images"
    NAME_PRFX = "Meta_Album_"

    # dataframe info
    COL_FILENAME = "FILE_NAME"
    COL_CATEGORY = "CATEGORY"

    LOWER_BOUND = 10

    def __init__(self, dataset_config: DatasetConfig, did: Optional[int]=None):
        if did is None:
            Logger.instance().warning(
                f"Be aware that you are selecting the first id of the list! [{dataset_config.dataset_id[0]}]"
            )
        
        self.did = did if did is not None else dataset_config.dataset_id[0]     #type: ignore (checked by the parser)

        # set cache directory to store info
        cache_dir_root = dataset_config.dataset_path
        openml.config.set_root_cache_directory(cache_dir_root)
        self.curr_dataset_path = os.path.join(cache_dir_root, self.OPENML_DATASET_LEAF, str(self.did))
        
        # load dataset (if already downloaded, do not use openml API)
        if str(self.did) not in os.listdir(os.path.join(cache_dir_root, self.OPENML_DATASET_LEAF)):
            Logger.instance().warning(f"meta_album dataset {str(self.did)} is going to be downloaded")
            dataset = openml.datasets.get_dataset(self.did, download_all_files=True)
            self.df_meta_album, self.y, _, _ = dataset.get_data()
        else:
            Logger.instance().debug(f"OpenML dataset {self.did} found, no need to download")
            croissant = Tools.read_json(os.path.join(self.curr_dataset_path, f"dataset_{str(self.did)}_croissant.json"))
            if croissant["name"].startswith(self.NAME_PRFX):
                img_folder_name = croissant["name"][len(self.NAME_PRFX):]
            img_folder = Tools.validate_path(os.path.join(self.curr_dataset_path, img_folder_name))
            self.df_meta_album = pd.read_csv(os.path.join(img_folder, "labels.csv"))

        self.df_meta_album = self._check_meta_album_fsl()

        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        # the directory containing the images has the same name as the .zip file, so remove the last 4 characters
        leaf_dirname = list(filter(lambda x: x.endswith(".zip"), os.listdir(self.curr_dataset_path)))[0][:-4]
        avail_ext = ("jpeg", "jpg", "png", "PNG", "JPG")
        img_list = glob(os.path.join(self.curr_dataset_path, leaf_dirname, self.DIR_IMAGES, "*"))
        img_list = list(filter(lambda x: x.endswith(avail_ext), img_list))

        # check for classes that have less than lower_bound images and remove them from the list
        if len(self.df_meta_album[self.COL_FILENAME]) < len(img_list):
            img_names_set = set([os.path.basename(img) for img in img_list])
            df_filename_set = set(self.df_meta_album[self.COL_FILENAME].values)
            missing = img_names_set - df_filename_set

            for i, img in enumerate(img_list):
                if os.path.basename(img) in missing:
                    img_list.pop(i)

        return img_list
    
    def get_label_list(self) -> List[int]:
        if self.image_list is None:
            self.image_list = self.get_image_list(None)

        df: pd.DataFrame = self.df_meta_album.copy()

        # order classes as image_list, not dataframe
        img_names = list(map(lambda x: os.path.basename(x), self.image_list))
        df.set_index(self.COL_FILENAME, inplace=True)
        ordered_classes = [df.loc[image_name, self.COL_CATEGORY] for image_name in img_names]

        # ERROR MANAGER: there are wrong type values in the COL_CATEGORY as some are seen as pd.Series
        if any(type(oc) is pd.Series for oc in ordered_classes):
            Logger.instance().warning(f"Dataset {self.did} has wrong values for the category column (pd.Series)")
            for i, _ in enumerate(ordered_classes):
                if type(ordered_classes[i]) is pd.Series:
                    ordered_classes[i] = list(ordered_classes[i])[0]

        # label to idx and vice versa to be compliant to FewShotDataset
        self.label_to_idx = { val: i for i, val in enumerate(set(ordered_classes)) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        # use mapping to return an int for the corresponding str label
        return [self.label_to_idx[c] for c in ordered_classes]
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        Logger.instance().info(f"Number of images for dataset {self.did}: {len(self.image_list)}")
        all_classes = set(list(self.df_meta_album[self.COL_CATEGORY]))

        # check if there is enough space for a validation set (at least 5 classes), otherwise split train/val
        n_cls_val = int(np.floor(self.dataset_config.dataset_splits[1] * len(all_classes)))
        n_cls_test = int(np.floor(self.dataset_config.dataset_splits[2] * len(all_classes)))

        # test only
        if int(np.ceil(self.dataset_config.dataset_splits[2])) == 1:
            return set(), set(), all_classes

        if n_cls_val < 5:
            class_val = set(list(all_classes)[:5])
            class_train = all_classes - class_val
            Logger.instance().warning(f"Not enough classes for valid in dataset {self.did}: using {len(class_train)}")
            return class_train, class_val, set()

        # default behaviour instead
        class_val = set(list(all_classes)[:n_cls_val])
        class_test = set(list(set(all_classes - class_val))[n_cls_val : (n_cls_val + n_cls_test)])
        class_train = all_classes - class_val - class_test
        Logger.instance().info(f"train/val/test split classes: {len(class_train)}, {len(class_val)}, {len(class_test)}")
        
        return class_train, class_val, class_test

    def expected_length(self):
        return len(self.df_meta_album)

    def _check_meta_album_fsl(self) -> pd.DataFrame:
        # count elements per class and remove those below lower bound
        class_counts = self.df_meta_album.groupby(self.COL_CATEGORY)[self.COL_FILENAME].count()
        filt_classes = class_counts[class_counts < self.LOWER_BOUND].index.tolist()
        df_filtered = self.df_meta_album[~self.df_meta_album[self.COL_CATEGORY].isin(filt_classes)]

        if len(filt_classes) > 0:
            filt_dict = []
            tot_filt = 0
            for fc in filt_classes:
                filt_dict.append(f'{fc}: {class_counts[fc]}')
                tot_filt += class_counts[fc]
            
            Logger.instance().warning(
                f"Dataset {self.did} has classes that do not reach the min number of samples ({self.LOWER_BOUND}): \n" +
                f"{filt_dict}\n" +
                f"Total number of classes that will be be removed {len(filt_dict)}. " +
                f"Total number of elements that will be removed: {tot_filt}"
            )

        return df_filtered