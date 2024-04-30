import os

from PIL import Image
from glob import glob
from typing import List, Set, Tuple, Optional

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG

class Aircraft(FewShotDataset):
    """FGVC Aircraft

    From their website: "The dataset contains 10,200 images of aircraft, with 100 images for each of 102 different 
    aircraft model variants". Nevertheless, if you download the dataset from their official website (link below), you
    will get 100 classes and 10000 images.

    The file that defines these classes is called 'variants.txt', under the folder 'data'. Unfortunately, following
    this parent class, we need to know the class of an image from its parent directory. This dataset is provided with
    all the images under 'data/images' instead. For this reason, a script will create the 100 directories and move
    all the data in the corresponding folder.

    Also, if you download the annotations from scratch, you need to ensure to replace '/' character with '_' otherwise
    it is read as subfolder when creating splits!

    Here, we use all the images (og train/val/test) for the meta-learning task, with our splits borrowed from github. 

    SeeAlso:
        [FSL dataset](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
        [split](https://github.com/google-research/meta-dataset/blob/main/meta_dataset/dataset_conversion/splits/cu_birds_splits.json)
    """

    N_CLASSES = 100
    N_CLASS_TRAIN = 70
    N_CLASS_VAL = 15
    N_CLASS_TEST = 15
    N_IMAGES = 10000
    
    DIR_DATA = "data"
    DIR_IMGS = "images"
    DIR_SPLITS = "splits"

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.img_dir = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, self.DIR_DATA, self.DIR_IMGS))
        self.split_imgs_dir = os.path.join(self.dataset_config.dataset_path, self.DIR_DATA, self.DIR_SPLITS)
        self.json_split = os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits", "aircraft")
        self.__copy_to_folders()
        
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        img_list = glob(os.path.join(self.split_imgs_dir, "*", "*.jpg"))
        return img_list
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        obj = Tools.read_json(os.path.join(self.json_split, "aircraft_splits.json"))
        
        class_train = set(obj.get("train"))
        class_val = set(obj.get("valid"))
        class_test = set(obj.get("test"))
        
        return class_train, class_val, class_test

    def expected_length(self) -> int:
        return self.N_IMAGES

    def __copy_to_folders(self):
        os.makedirs(self.split_imgs_dir, exist_ok=True)
        
        # no directory in it
        try:
            num_images = Tools.recursive_count(self.split_imgs_dir)
        except FileNotFoundError as fnf:
            Logger.instance().warning(f"{fnf}. Populating..")
            num_images = 0

        # correct number of images
        if num_images == self.N_IMAGES:
            Logger.instance().debug(f"Aircraft dataset ok in splits folder")
            return
        
        # wrong number of images
        Logger.instance().warning(f"Only found {num_images}/{self.N_IMAGES}. Copying Aircraft again..")
        data_root = os.path.join(self.dataset_config.dataset_path, self.DIR_DATA)
        for split_name in ["train", "val", "test"]:
            with open(os.path.join(data_root, f"images_variant_{split_name}.txt")) as f:
                for line in f:
                    if line.strip():
                        filename, variant = line.strip().split(" ", 1)
                        variant = variant.replace("/", "_")
                        os.makedirs(os.path.join(self.split_imgs_dir, variant), exist_ok=True)
                        full_old_filename = os.path.join(self.img_dir, f"{filename}.jpg")
                        full_new_filename = os.path.join(self.split_imgs_dir, variant, f"{filename}.jpg")
                        
                        # remove 20 pixels from the bottom (copyright info)
                        img = Image.open(full_old_filename)
                        w, h = img.size
                        crop = img.crop((0, 0, w, h - 20))
                        crop.save(full_new_filename)
        
        Logger.instance().debug(f"Aircraft has been copied!")