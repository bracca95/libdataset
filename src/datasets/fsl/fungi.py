import os
import re
import shutil

from glob import glob
from typing import List, Set, Tuple, Optional

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import CustomDatasetConsts as _CDC

class Fungi(FewShotDataset):
    """Fungi FSL split

    2018 Fungi Classification for the FGVCx competition. Over 100,000 fungi images of nearly 1,500 wild mushrooms 
    species. The FSL splits are taken from meta-dataset repository (train/val of the original dataset).

    SeeAlso:
        [download (train/val)](https://github.com/visipedia/fgvcx_fungi_comp?tab=readme-ov-file#data)
        [split](https://github.com/google-research/meta-dataset/blob/main/meta_dataset/dataset_conversion/splits/fungi_splits.json)
    """

    N_CLASS_TRAIN = 926 #994
    N_CLASS_VAL = 192
    N_CLASS_TEST = 187
    N_IMAGES = 88952 #89618 #89760
    IMG_DIR = "images"

    def __init__(self, dataset_config: DatasetConfig):
        self.did = _CDC.Fungi
        self.dataset_config = dataset_config
        self.img_dir_path = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, self.IMG_DIR))
        self.split_dir = Tools.validate_path(os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits", "fungi"))
        self.__adapt_dirnames()
        self.sparse = self._get_sparse_classes(self.img_dir_path)

        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        # fuck it's broken
        img_list = []
        for c in os.listdir(self.img_dir_path):
            if c not in self.sparse:
                img_list.extend(glob(os.path.join(self.img_dir_path, c, "*JPG")))
        return img_list
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        obj = Tools.read_json(os.path.join(self.split_dir, "fungi_splits.json"))
        pattern = re.compile(r'^\d+\.') # initial number and "."

        def get_class_set(split_name: str, sparse):
            # fungi annotation is wrong: two classes appear twice! (994 vs 992)
            og_names = obj.get(split_name)
            result = [pattern.sub('', s) for s in og_names]
            return set(result) - sparse
        
        return get_class_set("train", self.sparse), get_class_set("valid", self.sparse), get_class_set("test", self.sparse)

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