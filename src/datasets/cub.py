# https://www.vision.caltech.edu/datasets/cub_200_2011/

import os
import torch

from PIL import Image
from glob import glob
from typing import List, Optional
from torchvision import transforms
from torch.utils.data import Subset

from .dataset import CustomDataset
from ..utils.config_parser import DatasetConfig
from ..utils.tools import Logger, Tools
from ...config.consts import SubsetsDict, General as _CG

class Cub(CustomDataset):

    """CUB 200 2011

    The orginal dataset train/test split does not account for a validation set, and most importantly it does not split
    train and test classes: we want to classify unseen classes, not unseen instances!

    Not all classes have the same number of samples: 144/200 classes have exactly 60 samples. The remaining classes
    have less samples.

    SeeAlso:
        [FSL dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)
    """

    N_CLASSES = 200
    N_IMAGES = 11788

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        # self.augment_strategy = None/Processing.[] # should you need to augment, put the method here
        super().__init__(dataset_config)

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]
        curr_label_batch = self.label_list[index]
        
        return self.load_image(curr_img_batch), curr_label_batch
    
    def __len__(self):
        return self.N_IMAGES

    @property
    def augment_strategy(self):
        return self._augment_strategy
    
    @augment_strategy.setter
    def augment_strategy(self, val):
        # assign the real value if needed
        self._augment_strategy = None

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        try:
            img_dir = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, "images"))
        except FileNotFoundError as fnf:
            msg = f"A Directory `images` (with all the subdirs of the classes) must be contained in the path " + \
                  f"specified in config.json. You chose {self.dataset_config.dataset_path} but it looks wrong."
            Logger.instance().critical(f"{fnf}\n{msg}")
            raise FileNotFoundError(msg)
        
        images = glob(os.path.join(img_dir, "*", "*.jpg"))

        # check
        if not len(images) == self.N_IMAGES:
            msg = f"The number of images ({len(images)}) should be {self.N_IMAGES}"
            Logger.instance().critical(msg)
            raise ValueError(msg)
        
        return images
    
    def get_label_list(self) -> List[int]:
        if self.image_list is None:
            self.image_list = self.get_image_list(None)

        # extract unique label values from dirname
        label_set = sorted(set([os.path.basename(os.path.dirname(image_name)) for image_name in self.image_list]))
        self.label_to_idx = { val: i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        # use mapping to return an int for the corresponding str label
        return [self.label_to_idx[os.path.basename(os.path.dirname(image_name))] for image_name in self.image_list]
    
    def load_image(self, path: str) -> torch.Tensor:
        img_pil = Image.open(path).convert("RGB")
        
        # resize
        img_pil = transforms.Resize((self.dataset_config.image_size, self.dataset_config.image_size))(img_pil)

        # rescale [0-255](int) to [0-1](float)
        img = transforms.ToTensor()(img_pil)

        # normalize
        img = self.normalize_or_identity(self.dataset_config)(img)

        return img # type: ignore
    
    def split_dataset(self, split_ratios: List[float]=[.8]) -> SubsetsDict:
        return self.fsl_split(self, self.get_label_list(), self.N_CLASSES, self.dataset_config.dataset_splits)