import os

from glob import glob
from typing import List, Set, Tuple, Optional

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG

class Dtd(FewShotDataset):
    """DTD dataset

    DTD is a texture database, consisting of 5640 images, organized according to a list of 47 terms (categories) 
    inspired from human perception. There are 120 images for each category. Image sizes range between 300x300 and 
    640x640, and the images contain at least 90% of the surface representing the category attribute. 
    The images were collected from Google and Flickr by entering our proposed attributes and related terms as search 
    queries. The images were annotated using Amazon Mechanical Turk in several iterations.

    SeeAlso:
        [FSL dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
        [split](https://github.com/google-research/meta-dataset/blob/main/meta_dataset/dataset_conversion/splits/dtd_splits.json)
    """

    N_CLASSES = 47
    N_CLASS_TRAIN = 33
    N_CLASS_VAL = 7
    N_CLASS_TEST = 7
    N_IMAGES = 5640
    N_IMAGES_PER_CLASS = 120
    IMG_DIR = "images"

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.img_dir_path = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, self.IMG_DIR))
        self.split_dir = Tools.validate_path(os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits", "dtd"))
        
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        img_list = glob(os.path.join(self.img_dir_path, "*", "*.jpg"))
        return img_list
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        obj = Tools.read_json(os.path.join(self.split_dir, "dtd_splits.json"))
        
        class_train = set(obj.get("train"))
        class_val = set(obj.get("valid"))
        class_test = set(obj.get("test"))
        
        return class_train, class_val, class_test

    def expected_length(self) -> int:
        return self.N_IMAGES
