import os

from PIL import Image
from tqdm import tqdm
from glob import glob
from typing import Optional, List, Set, Tuple

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG


class DagmAdac(FewShotDataset):
    """Training set of few-shot DAGM

    This class is meant to be used when comparing the supervised learning approach with later adaptation to new tasks
    to few-shot techniques. We still need to separate training classes from val and test, but we only need the train
    split here. Use MetaDagm when testing the adaptation performance.
    I cannot do the usual split, otherwise the cross-entropy loss would search for a larger label space (the one that
    includes all the labels) during the training phase: therefore, I strip the training part.

    [DAGM](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
    """

    DATA_DIR = "DAGM_KaggleUpload"
    SUBDIR_TRAIN = "Train"
    SUBDIR_TEST = "Test"
    SUBDIR_LABELS = "Label"
    DEFECTIVE = "defective"
    PRISTINE = "pristine"
    SPLIT_FILE = os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits", "dagm", "dagm_test")

    def __init__(self, dataset_config: DatasetConfig, did: Optional[int]=None):
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        data_dir = os.path.join(self.dataset_config.dataset_path, self.DATA_DIR)

        # filters
        avail_ext = ("jpeg", "jpg", "png", "PNG", "JPG")    # keep
        test_classes = self.read_split(self.SPLIT_FILE)     # remove
        
        # find all and filter (1: classX, 2: Train/Test, 3: .PNG)
        img_list = glob(os.path.join(data_dir, "*", "*", "*"))
        img_list = list(filter(lambda x: x.endswith(avail_ext), img_list))
        remove = list(map(lambda x: any([a in x for a in test_classes]), img_list))
        keep = [not boolean for boolean in remove]
        train_img_list = [element for element, keep in zip(img_list, keep) if keep]
        
        return train_img_list
    
    def get_label_list(self) -> List[int]:
        if self.image_list is None:
            self.image_list = self.get_image_list(None)

        # extract unique label values from dirname
        label_list = []
        for img_name in self.image_list:
            class_name = img_name.rsplit(os.sep, -1)[-3]    # ClassX
            name, ext = os.path.basename(img_name).split(".")
            label_name = f"{name}_label.{ext}"
            label_path = os.path.join(os.path.dirname(img_name), self.SUBDIR_LABELS, label_name)
            if os.path.exists(label_path):
                label_list.append(f"{class_name}_{self.DEFECTIVE}")
            else:
                label_list.append(f"{class_name}_{self.PRISTINE}")
        
        label_set = sorted(set(label_list))
        self.label_to_idx = { val: i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        # use mapping to return an int for the corresponding str label
        return [self.label_to_idx[name] for name in label_list]
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        train_classes = set(self.label_to_idx.keys())
        return train_classes, set(), set()

    def expected_length(self):
        raise NotImplementedError

    @staticmethod
    def read_split(file_path: str) -> List[str]:
        test_classes = []
        with open(file_path) as f:
            for line in f:
                test_classes.extend(line.replace(",", "").replace(";", "").split())

        # preserve the order
        unique_list = []
        [unique_list.append(item) for item in test_classes if item not in unique_list]

        return unique_list


class MetaDagm(DagmAdac):
    """Few-shot version for MVTec dataset

    The idea is to use the original test(!) sub-classes of DAGM to define a few-shot learning scenario, where you can
    learn different types of anomalies and then transfer the knowledge to unkown classes.

    [DAGM](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
    """

    IDS = {
        0: "Class1",
        1: "Class2",
        2: "Class3",
        3: "Class4",
        4: "Class5",
        5: "Class6",
        6: "Class7",
        7: "Class8",
        8: "Class9",
        9: "Class10"
    }

    def __init__(self, dataset_config: DatasetConfig, did: Optional[int]=None):
        self.did = self.check(dataset_config, did)
        self.curr_class = self.IDS[self.did]
        super().__init__(dataset_config, did)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        data_dir = os.path.join(self.dataset_config.dataset_path, self.DATA_DIR, self.curr_class)

        # filters
        avail_ext = ("jpeg", "jpg", "png", "PNG", "JPG")
        
        # find all and filter (1: Train/Test, 2: .PNG)
        img_list = glob(os.path.join(data_dir, "*", "*"))
        img_list = list(filter(lambda x: x.endswith(avail_ext), img_list))
        
        return img_list
    
    def get_label_list(self) -> List[int]:
        if self.image_list is None:
            self.image_list = self.get_image_list(None)

        # extract unique label values from dirname
        label_list = []
        for img_name in self.image_list:
            name, ext = os.path.basename(img_name).split(".")
            label_name = f"{name}_label.{ext}"
            label_path = os.path.join(os.path.dirname(img_name), self.SUBDIR_LABELS, label_name)
            if os.path.exists(label_path):
                label_list.append(self.DEFECTIVE)
            else:
                label_list.append(self.PRISTINE)
        
        label_set = sorted(set(label_list))
        self.label_to_idx = { val: i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        # use mapping to return an int for the corresponding str label
        return [self.label_to_idx[name] for name in label_list]

    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        train_classes, test_classes = [], []
        if self.dataset_config.dataset_splits[0] > sum(self.dataset_config.dataset_splits[1:]):
            train_classes = self.label_to_idx.keys()
        else:
            test_classes = self.label_to_idx.keys()
        
        return set(train_classes), set(test_classes), set(test_classes)

    def check(self, dataset_config: DatasetConfig, did: Optional[int]) -> int:
        if did is None:
            if not dataset_config.dataset_id: # null or empty
                all_values = self.IDS.values()
                test_values = self.read_split(DagmAdac.SPLIT_FILE)
                train_values = [item for item in all_values if item not in test_values]
                
                first_key = train_values[0]
                msg = f"You are selecting the first ID `{first_key}`. This message should appear only once on top."
                Logger.instance().warning(msg)

                invert_ids_dict = Tools.invert_dict(self.IDS)
                return invert_ids_dict[first_key]

            if len(dataset_config.dataset_id) == 1:
                curr_id = dataset_config.dataset_id[0]
                Logger.instance().warning(f"You are probably calling the dataloader (id = {curr_id})")
                return curr_id
            
            else:
                raise ValueError(f"Unknown condition for dataset_id")

        return did