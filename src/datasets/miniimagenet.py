import os
import torch

from PIL import Image
from glob import glob
from typing import List, Optional
from torch.utils.data import Subset
from torchvision.transforms import transforms

from .dataset import CustomDataset
from ..utils.tools import Logger, Tools
from ..utils.config_parser import DatasetConfig
from ...config.consts import SubsetsDict, General as _CG


class MiniImageNet(CustomDataset):

    N_IMG_PER_CLASS = 600
    N_CLASSES_TRAIN = 64
    N_CLASSES_VAL = 16
    N_CLASSES_TEST = 20

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        # self.augment_strategy = None/Processing.[] # should you need to augment, put the method here
        super().__init__(dataset_config)

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]
        curr_label_batch = self.label_list[index]
        
        return self.load_image(curr_img_batch), curr_label_batch
    
    def __len__(self):
        return (MiniImageNet.N_CLASSES_TRAIN + MiniImageNet.N_CLASSES_TEST + MiniImageNet.N_CLASSES_VAL) * MiniImageNet.N_IMG_PER_CLASS

    @property
    def augment_strategy(self):
        return self._augment_strategy
    
    @augment_strategy.setter
    def augment_strategy(self, val):
        # assign the real value if needed
        self._augment_strategy = None

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        return glob(os.path.join(self.dataset_config.dataset_path, "n*", "*JPEG"))
    
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
        # ignore split_ratios for this dataset

        if not len(self.image_list) == len(self):
            raise ValueError(f"MiniImagenet must have 600*100 images. You have {len(self.image_list)}")
        
        train_set = Subset(self, [i for i in range(
            0, 
            self.N_IMG_PER_CLASS * self.N_CLASSES_TRAIN
            )])
        
        val_set = Subset(self, [i for i in range(
            self.N_IMG_PER_CLASS * self.N_CLASSES_TRAIN, 
            (self.N_IMG_PER_CLASS * self.N_CLASSES_TRAIN) + (self.N_IMG_PER_CLASS * self.N_CLASSES_VAL)
            )])
        
        test_set = Subset(self, [i for i in range(
            (self.N_IMG_PER_CLASS * self.N_CLASSES_TRAIN) + (self.N_IMG_PER_CLASS * self.N_CLASSES_VAL), 
            len(self)
            )])

        train_str, val_str, test_str = _CG.DEFAULT_SUBSETS
        
        Logger.instance().debug(f"Splitting miniimagenet: [{len(train_set)}, {len(val_set)}, {len(test_set)}]")

        # avoid using validation dataset if 0.0 is specified in the config.dataset.dataset_splits
        if len(self.dataset_config.dataset_splits) == 3:
            if self.dataset_config.dataset_splits[1] < 0.1:
                Logger.instance().warning(f"Overriding validation set: empty! No validation will be performed.")
                val_set = None
                
        return { train_str: train_set, val_str: val_set, test_str: test_set }   # type: ignore