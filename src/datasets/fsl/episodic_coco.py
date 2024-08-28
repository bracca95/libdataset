import os
import sys
import shutil
import pandas as pd

from tqdm import tqdm
from glob import glob
from typing import List, Set, Tuple, Optional
from torch.utils.data import Dataset

from .dataset_fsl import FewShotDataset
from ...utils.tools import Logger, Tools
from ...utils.config_parser import DatasetConfig
from ....config.consts import CustomDatasetConsts as _CDC


class EpisodicCoco(FewShotDataset):
    """EpisodicCoco

    COCO is an object detection dataset, hence more labels appear within the same image. CAML decided to use it as a
    classification dataset by using the same image across more labels. A pre-processing step is required to save the
    images n times, where n is the number of labels that occur for that particular image. The original COCO dataset
    (refer to kaggle link download below) must be present at the first iteration, so that the images can be saved in a
    more appropriate torch ImageFolder version.
    We will use this dataset for the training part only, so val/test splits are not required.

    How to use:
        - if it is the first time:
            1. Provide /path/to/coco2017 folder in `dataset_path` (dataset downloaded from kaggle)
            2. Wait for completion and run the program again with /path/to/episodic_coco
        - otherwise:
            1. directly run with /path/to/episodic_coco

    SeeAlso:
        - [mscoco](https://paperswithcode.com/dataset/coco)
        - [download](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
    """

    N_IMGS = 117266
    N_CLASSES = 80
    NAME_COCO2017 = "coco2017"
    NAME_EP_COCO = "episodic_coco"
    DIR_ANNOTATIONS = "annotations"
    DIR_DATA = "train2017"
    FILE_ANNO = "instances_train2017.json"
    FILL = 12   # number of fixed digits that define a coco image filename

    def __init__(self, dataset_config: DatasetConfig):
        self.root_coco = self.__init_dataset(dataset_config)
        self.did = _CDC.EpisodicCoco
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        img_paths = glob(os.path.join(self.root_coco, "*", "*jpg"))
        return img_paths
    
    def expected_length(self) -> int:
        return self.N_IMGS
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        # use it entirely as a training set
        train_set = set(os.listdir(self.root_coco))
        if not len(train_set) == self.N_CLASSES:
            raise ValueError(f"There should be 80 classes in the dataset folder. Found {len(train_set)}")
        
        return train_set, set(), set()
    
    def __init_dataset(self, dataset_config: DatasetConfig) -> str:
        if os.path.basename(dataset_config.dataset_path) not in (self.NAME_COCO2017, self.NAME_EP_COCO):
            raise ValueError(
                f"The root directory containing COCO must be named either 'coco2017' or 'episodic_coco'. " +
                f"You wrote {os.path.basename(dataset_config.dataset_path)}"
            )
        
        # root for episodic coco
        root_ep_coco = os.path.join(os.path.dirname(dataset_config.dataset_path), self.NAME_EP_COCO)
        os.makedirs(root_ep_coco, exist_ok=True)

        # if dataset_name == "coco2017" but "episodic_coco" exists ask to change the config and run the program again
        if os.path.basename(dataset_config.dataset_path) == self.NAME_COCO2017:
            if self.__ep_coco_exists(root_ep_coco):
                raise ValueError(f"You choose 'coco2017' but 'episodic_coco' exists. Use that to run the program")
        
        # if dataset_name == "episodic_coco", then check not empty and return config
        if os.path.basename(dataset_config.dataset_path) == self.NAME_EP_COCO:
            if self.__ep_coco_exists(root_ep_coco):
                return dataset_config.dataset_path
            raise FileNotFoundError(f"The directory is empty or there are not more than 3 files (strange).")
        
        # if dataset_name == "coco2017", then save images in the correct format
        if not os.path.exists(os.path.join(dataset_config.dataset_path, self.DIR_DATA)):
            raise ValueError(f"{os.path.join(dataset_config.dataset_path, self.DIR_DATA)} not found.")
        if not os.path.exists(os.path.join(dataset_config.dataset_path, self.DIR_ANNOTATIONS)):
            raise ValueError(f"{os.path.join(dataset_config.dataset_path, self.DIR_ANNOTATIONS)} not found.")
        
        Logger.instance().warning(
            f"You need at least 60 GB of free space to store episodic_coco. Starting now, it will take some time"
        )

        file_anno = self._read_file_anno(dataset_config)
        
        for seg in tqdm(file_anno["annotations"], total=len(file_anno["annotations"])):
            label = str(seg["category_id"])
            os.makedirs(os.path.join(root_ep_coco, label), exist_ok=True)
            compose_filename = f"{str(seg['image_id']).zfill(self.FILL)}.jpg"
            shutil.copy(
                os.path.join(dataset_config.dataset_path, self.DIR_DATA, compose_filename),
                os.path.join(root_ep_coco, label, compose_filename)
            )
        
        msg = f"Files have been copied to {root_ep_coco}. Run the program again with this config in `dataset_path`"
        Logger.instance().debug(msg)
        raise SystemExit(msg) 

    def __ep_coco_exists(self, root_ep_coco: str) -> bool:
        if not os.path.exists(root_ep_coco):
            return False
        
        try:
            rand_folder = os.listdir(root_ep_coco)[3]    # show class folders
            if len(os.listdir(os.path.join(root_ep_coco, rand_folder))) > 3:     # show images in class
                return True  # at least not empty
        except IndexError as ie:
            Logger.instance().warning(f"episodic coco is empty")
            return False

        return False
    
    def _read_file_anno(self, dataset_config: DatasetConfig) -> dict:
        try:
            file_anno = Tools.read_json(os.path.join(dataset_config.dataset_path, self.DIR_ANNOTATIONS, self.FILE_ANNO))
        except FileNotFoundError as fnf:
            Logger.instance().warning(f"{fnf}. Searching in split dir...")
            # search in split dir
            split_dir = os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits", self.NAME_EP_COCO)
            split_file = os.path.join(split_dir, self.FILE_ANNO)
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"{split_file} does not exist!")
            file_anno = Tools.read_json(split_file)

        return file_anno