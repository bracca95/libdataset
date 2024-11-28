import os
import openml
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

    DOMAINS = {
        44241: 0, 44313: 0, 44275: 0,
        44285: 0, 44298: 0, 44305: 0,
        44320: 0, 44331: 0, 44338: 0,
        44238: 1, 44248: 1, 44276: 1,
        44282: 1, 44292: 1, 44306: 1,
        44317: 1, 44326: 1, 44340: 1,
        44239: 2, 44249: 2, 44272: 2,
        44283: 2, 44293: 2, 44302: 2,
        44318: 2, 44327: 2, 44335: 2,
        44242: 3, 44314: 3, 44273: 3,
        44286: 3, 44299: 3, 44303: 3,
        44321: 3, 44332: 3, 44336: 3,
        44237: 4, 44312: 4, 44278: 4,
        44281: 4, 44297: 4, 44308: 4,
        44316: 4, 44330: 4, 44342: 4,
        44246: 5, 44315: 5, 44277: 5,
        44290: 5, 44300: 5, 44307: 5,
        44324: 5, 44333: 5, 44341: 5,
        44245: 6, 44251: 6, 44279: 6,
        44289: 6, 44295: 6, 44309: 6,
        44323: 6, 44329: 6, 44343: 6,
        44244: 7, 44250: 7, 44274: 7,
        44288: 7, 44294: 7, 44304: 7,
        44322: 7, 44328: 7, 44337: 7,
        44240: 8, 44247: 8, 44271: 8,
        44284: 8, 44291: 8, 44301: 8,
        44319: 8, 44325: 8, 44334: 8,
        44243: 9, 44252: 9, 44280: 9,
        44287: 9, 44296: 9, 44310: 9
    }

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

        # since you filtered the dataframe to have a lower bound of images per class, you might have images that exceed
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

        # ensure to pick the correct label according to the image name
        img_names = list(map(lambda x: os.path.basename(x), self.image_list))
        df.set_index(self.COL_FILENAME, inplace=True)
        ordered_classes = [df.loc[image_name, self.COL_CATEGORY] for image_name in img_names]

        # ERROR MANAGER: there are wrong type values in the COL_CATEGORY as some are seen as pd.Series (Extended)
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
        
        # pay attention: we ensure that the elements are unique by set, then convert to list to have a precise order
        all_classes: List[str] = sorted(set(self.df_meta_album[self.COL_CATEGORY].tolist()))
        n_avail_cls = len(all_classes)

        # test only
        if (1.0 - _CG.EPS) < self.dataset_config.dataset_splits[2] < (1.0 + _CG.EPS):
            return set(), set(), set(all_classes)

        # get the desired number of classes (minimum always 5, None if float split == 0.0)
        req_n_train: Optional[int] = self.get_n_classes_via_splits(self.dataset_config.dataset_splits[0], n_avail_cls)
        req_n_val: Optional[int] = self.get_n_classes_via_splits(self.dataset_config.dataset_splits[1], n_avail_cls)
        req_n_test: Optional[int] = self.get_n_classes_via_splits(self.dataset_config.dataset_splits[2], n_avail_cls)

        # if the required number of classes is larger than the number of available, reduce the amount of largest group
        req_list: List[Optional[int]] = [req_n_train, req_n_val, req_n_test]
        req_n_cls: int = sum([r for r in req_list if r is not None])
        if req_n_cls > n_avail_cls:
            Logger.instance().warning(f"Trying to reduce the largest split")
            
            # get argmax of the largest split
            values = [val if val is not None else float('-inf') for val in req_list]
            argmax = values.index(max(values))
            
            # get the other two
            all_indexes = set([0, 1, 2])
            all_indexes.remove(argmax)
            argmin_1, argmin_2 = list(all_indexes)

            # remove elements from the largest split
            req_list[argmax] = n_avail_cls - sum([r for r in [req_list[argmin_1], req_list[argmin_2]] if r is not None])

        # get class names
        class_train = all_classes[:req_list[0]] if req_list[0] is not None else set()
        class_test = all_classes[-req_list[2]:] if req_list[2] is not None else set()
        
        if req_list[1] is None:
            class_val = set()
        elif req_list[0] is not None and req_list[2] is not None:
            class_val = all_classes[req_list[0] : req_list[0] + req_list[1]]
        elif req_list[0] is not None and req_list[2] is None:
            class_val = all_classes[req_list[0] : req_list[0] + req_list[1]]
        elif req_list[2] is not None and req_list[0] is None:
            class_val = all_classes[-(req_list[2]+req_list[1]) : -req_list[2]]
        else:
            raise ValueError(f"Something went wrong while splitting")

        Logger.instance().info(f"train/val/test split classes: {len(class_train)}, {len(class_val)}, {len(class_test)}")
        
        return set(class_train), set(class_val), set(class_test)

    def expected_length(self):
        return len(self.df_meta_album)

    def _check_meta_album_fsl(self) -> pd.DataFrame:
        # ERROR MANAGER: there might be .DS_Store counted as an image!
        self.df_meta_album = self.df_meta_album[self.df_meta_album[self.COL_FILENAME] != '.DS_Store']

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