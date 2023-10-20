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


class CifarFs(CustomDataset):
    """CIFAR-FS

    Train/val/test are already are already split in the dataset I downloaded: check the link. This version of the
    dataset can only be used with these three default splits; write another class to use different combinations.

    SeeAlso:
        [main page](https://github.com/bertinetto/r2d2)
        [drive folder](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view?usp=sharing)
    """

    N_IMG_PER_CLASS = 600
    N_CLASSES_TRAIN = 64
    N_CLASSES_VAL = 16
    N_CLASSES_TEST = 20

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        # self.augment_strategy = None/Processing.[] # should you need to augment, put the method here

        self.train_dir = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, "meta_train"))
        self.val_dir = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, "meta_val"))
        self.test_dir = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, "meta_test"))

        super().__init__(dataset_config)

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]
        curr_label_batch = self.label_list[index]
        
        return self.load_image(curr_img_batch), curr_label_batch
    
    def __len__(self):
        return self.__expected_length()
    
    @property
    def augment_strategy(self):
        return self._augment_strategy
    
    @augment_strategy.setter
    def augment_strategy(self, val):
        # assign the real value if needed
        self._augment_strategy = None

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        img_train = glob(os.path.join(self.train_dir, "*", "*.png"))
        img_val = glob(os.path.join(self.val_dir, "*", "*.png"))
        img_test = glob(os.path.join(self.test_dir, "*", "*.png"))

        if not len(img_train) + len(img_val) + len(img_test) == self.__expected_length():
            raise ValueError(f"There should be 600 images in all the 100 CIFAR classes.")

        return img_train + img_val + img_test
    
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
        class_train = list(filter(lambda x: os.path.isdir(os.path.join(self.train_dir, x)), os.listdir(self.train_dir)))
        class_val = list(filter(lambda x: os.path.isdir(os.path.join(self.val_dir, x)), os.listdir(self.val_dir)))
        class_test = list(filter(lambda x: os.path.isdir(os.path.join(self.test_dir, x)), os.listdir(self.test_dir)))

        if not len(class_train) == CifarFs.N_CLASSES_TRAIN or \
           not len(class_val) == CifarFs.N_CLASSES_VAL or \
           not len(class_test) == CifarFs.N_CLASSES_TEST:
            raise ValueError(f"Some class is missing in cifar-fs directory or it has been filtered.")
        
        # get labels (int) that are in the train/val/test splits
        label_train = torch.LongTensor([self.label_to_idx[c] for c in class_train])
        label_val = torch.LongTensor([self.label_to_idx[c] for c in class_val])
        label_test = torch.LongTensor([self.label_to_idx[c] for c in class_test])

        # find all the occurrences
        idxs_train = torch.where(torch.LongTensor(self.label_list).unsqueeze(0) == label_train.unsqueeze(1))[1]
        idxs_val = torch.where(torch.LongTensor(self.label_list).unsqueeze(0) == label_val.unsqueeze(1))[1]
        idxs_test = torch.where(torch.LongTensor(self.label_list).unsqueeze(0) == label_test.unsqueeze(1))[1]

        # create subsets
        subset_train = Subset(self, idxs_train.tolist())
        subset_val = Subset(self, idxs_val.tolist())
        subset_test = Subset(self, idxs_test.tolist())

        train_str, val_str, test_str = _CG.DEFAULT_SUBSETS
        return { train_str: subset_train, val_str: subset_val, test_str: subset_test }   # type:ignore

    def __expected_length(self) -> int:
        return (CifarFs.N_CLASSES_TRAIN + CifarFs.N_CLASSES_TEST + CifarFs.N_CLASSES_VAL) * CifarFs.N_IMG_PER_CLASS