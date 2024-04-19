import os
import re
import shutil

from glob import glob
from typing import List, Set, Tuple, Optional

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG

class Fungi(FewShotDataset):
    """Fungi FSL split

    2018 Fungi Classification for the FGVCx competition. Over 100,000 fungi images of nearly 1,500 wild mushrooms 
    species. The FSL splits are taken from meta-dataset repository (train/val of the original dataset).

    SeeAlso:
        [download (train/val)](https://github.com/visipedia/fgvcx_fungi_comp?tab=readme-ov-file#data)
        [split](https://github.com/google-research/meta-dataset/blob/main/meta_dataset/dataset_conversion/splits/fungi_splits.json)
    """

    N_CLASS_TRAIN = 994
    N_CLASS_VAL = 200
    N_CLASS_TEST = 200
    N_IMAGES = 89760
    IMG_DIR = "images"

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.img_dir_path = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, self.IMG_DIR))
        self.split_dir = Tools.validate_path(os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits", "fungi"))
        self.__adapt_dirnames()

        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        # fuck it's broken
        img_list = []
        for root, dirs, files in os.walk(self.img_dir_path):
            for file in files:
                if file.endswith(".JPG"):
                    img_path = os.path.join(root, file)
                    img_list.append(img_path)
        
        return img_list
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        obj = Tools.read_json(os.path.join(self.split_dir, "fungi_splits.json"))
        pattern = re.compile(r'^\d+\.') # initial number and "."

        def get_class_set(split_name: str):
            og_names = obj.get(split_name)
            result = [pattern.sub('', s) for s in og_names]
            return set(result)
        
        return get_class_set("train"), get_class_set("valid"), get_class_set("test")

    def expected_length(self) -> int:
        return self.N_IMAGES
    
    def __adapt_dirnames(self):
        items = os.listdir(self.img_dir_path)
    
        # if any directory does not start with a digit processing has already been done
        if any(not os.path.isdir(os.path.join(self.img_dir_path, item)) or not item[0].isdigit() for item in items):
            Logger.instance().debug("Fungi has already been pre-processed (names)")
            return
        
        # rename
        for item in items:
            item_path = os.path.join(self.img_dir_path, item)
            if os.path.isdir(item_path):
                new_name = re.sub(r'^\d+\_', '', item)
                new_name = new_name.replace('_', ' ')
                new_path = os.path.join(self.img_dir_path, new_name)
                shutil.move(item_path, new_path)

        Logger.instance().debug(f"Fungi has been renamed!")

    # def __init__(self, dataset_config: DatasetConfig):
    #     super().__init__(dataset_config)
        
    # def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
    #     try:
    #         img_dir = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, "images"))
    #     except FileNotFoundError as fnf:
    #         msg = f"A Directory `images` (with all the subdirs of the classes) must be contained in the path " + \
    #               f"specified in config.json. You chose {self.dataset_config.dataset_path} but it looks wrong."
    #         Logger.instance().critical(f"{fnf}\n{msg}")
    #         raise FileNotFoundError(msg)
        
    #     images = glob(os.path.join(img_dir, "*", "*.jpg"))

    #     # check
    #     if not len(images) == self.N_IMAGES:
    #         msg = f"The number of images ({len(images)}) should be {self.N_IMAGES}"
    #         Logger.instance().critical(msg)
    #         raise ValueError(msg)
        
    #     return images
    
    # def expected_length(self) -> int:
    #     return self.N_IMAGES
    
    # def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
    #     """Random split based on the number of classes
        
    #     This function provides a better implementation for splitting a FSL dataset, so that the classes in 
    #     train/val/test splits do not intersect. Call this method inside the overwritten version of split_dataset().
    #     """

    #     split_ratios = self.dataset_config.dataset_splits

    #     if len(split_ratios) == 1:
    #         split_ratios = [split_ratios[0], 1.0 - split_ratios[0]]
    #     if len(split_ratios) == 2 or len(split_ratios) > 3:
    #         raise ValueError(f"split_ratios argument accepts either a list of 1 value (train,test) or 3 (train,val,test)")

    #     n_classes = len(self.label_to_idx.keys())
    #     shuffled_idxs = torch.randperm(n_classes)
    #     split_points = [int(n_classes * ratio) for ratio in split_ratios]

    #     # if no validation is used, set eof_split to 0 so that the last index starts with the end of train
    #     if len(split_points) == 3:
    #         eof_split = split_points[1]
    #         class_val_idx = set((shuffled_idxs[split_points[0]:split_points[0] + split_points[1]]).tolist())
    #     else:
    #         eof_split = 0
    #         class_val_idx = set()
    #     class_val = set([self.idx_to_label[c] for c in class_val_idx])

    #     # Split the numbers into two or three sets
    #     class_train_idx = set((shuffled_idxs[:split_points[0]]).tolist())
    #     class_test_idx = set((shuffled_idxs[split_points[0] + eof_split:]).tolist())

    #     class_train = set([self.idx_to_label[c] for c in class_train_idx])
    #     class_test = set([self.idx_to_label[c] for c in class_test_idx])

    #     if not len(shuffled_idxs) == len(class_train) + len(class_val) + len(class_test):
    #         raise ValueError(f"The number of classes {shuffled_idxs} does not match the split.")
        
    #     return class_train, class_val, class_test