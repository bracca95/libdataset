import os
import torch

from PIL import Image
from glob import glob
from typing import List, Tuple, Optional
from torchvision import transforms
from torch.utils.data import Dataset

from ..dataset import DatasetWrapper, DatasetLauncher
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG

class Cub(DatasetWrapper):

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
        super().__init__()
        self.dataset_config = dataset_config
        self._image_list = self.get_image_list(None)
        self._label_list = self.get_label_list()
        self._train_dataset, self._val_dataset, self._test_dataset = self.split_dataset(self.dataset_config.dataset_splits)

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
    
    def load_image(self, path: str, augment: bool) -> torch.Tensor:
        img_pil = Image.open(path).convert("RGB")
        
        if augment:
            # TODO implement
            pass

        # resize
        img_pil = transforms.Resize((self.dataset_config.image_size, self.dataset_config.image_size))(img_pil)

        # rescale [0-255](int) to [0-1](float)
        img = transforms.ToTensor()(img_pil)

        # normalize
        img = DatasetLauncher.normalize_or_identity(self.dataset_config)(img)

        return img
    
    def _expected_length(self) -> int:
        return self.N_IMAGES
    
    def split_dataset(self, split_ratios: List[float]=[.8]) -> Tuple[DatasetLauncher, Optional[DatasetLauncher], DatasetLauncher]:
        """Random split based on the number of classes
        
        This function provides a better implementation for splitting a FSL dataset, so that the classes in 
        train/val/test splits do not intersect. Call this method inside the overwritten version of split_dataset().
        """

        if len(split_ratios) == 1:
            split_ratios = [split_ratios[0], 1.0 - split_ratios[0]]
        if len(split_ratios) == 2 or len(split_ratios) > 3:
            raise ValueError(f"split_ratios argument accepts either a list of 1 value (train,test) or 3 (train,val,test)")

        n_classes = len(self.label_to_idx.keys())
        shuffled_idxs = torch.randperm(n_classes)
        split_points = [int(n_classes * ratio) for ratio in split_ratios]

        # if no validation is used, set eof_split to 0 so that the last index starts with the end of train
        if len(split_points) == 3:
            eof_split = split_points[1]
            class_val_idx = set((shuffled_idxs[split_points[0]:split_points[0] + split_points[1]]).tolist())
        else:
            eof_split = 0
            class_val_idx = set()
        class_val = set([self.idx_to_label[c] for c in class_val_idx])

        # Split the numbers into two or three sets
        class_train_idx = set((shuffled_idxs[:split_points[0]]).tolist())
        class_test_idx = set((shuffled_idxs[split_points[0] + eof_split:]).tolist())

        class_train = set([self.idx_to_label[c] for c in class_train_idx])
        class_test = set([self.idx_to_label[c] for c in class_test_idx])

        if not len(shuffled_idxs) == len(class_train) + len(class_val) + len(class_test):
            raise ValueError(f"The number of classes {shuffled_idxs} does not match the split.")

        # get all class labels as tensor
        tensor_labels = torch.tensor(self.label_list, dtype=torch.int)

        # select the images where indices corresponds to train/val/test classes
        def select_img_lbl(class_set: set) -> Tuple[List[str], List[int]]:
            label_ints = torch.tensor([self.label_to_idx[c] for c in class_set], dtype=torch.int)
            indices = torch.where(tensor_labels == label_ints.unsqueeze(1))[1]
            images = [self.image_list[i] for i in indices]
            labels = [self.label_list[i] for i in indices]
            return images, labels
        
        # create DatasetLauncher with augmentation for training if required
        augment = True if self.dataset_config.augment_online is not None else False
        train_dataset = DatasetLauncher(*select_img_lbl(class_train), augment, load_img_callback=self.load_image)
        val_dataset = DatasetLauncher(*select_img_lbl(class_val), augment=False, load_img_callback=self.load_image)
        test_dataset = DatasetLauncher(*select_img_lbl(class_test), augment=False, load_img_callback=self.load_image)
        
        # fill info dict
        train_dataset.set_info(select_img_lbl(class_train)[1], self.idx_to_label)
        val_dataset.set_info(select_img_lbl(class_val)[1], self.idx_to_label)
        test_dataset.set_info(select_img_lbl(class_test)[1], self.idx_to_label)

        # avoid using validation dataset if 0.0 is specified in the config.dataset.dataset_splits
        if len(self.dataset_config.dataset_splits) == 3:
            if self.dataset_config.dataset_splits[1] < 0.1:
                Logger.instance().warning(f"Overriding validation set: empty! No validation will be performed.")
                val_dataset = None
                
        return train_dataset, val_dataset, test_dataset
    
    @property
    def image_list(self) -> List[str]:
        return self._image_list
    
    @image_list.setter
    def image_list(self, value):
        self._image_list = value

    @property
    def label_list(self) -> List[int]:
        return self._label_list
    
    @label_list.setter
    def label_list(self, value):
        self._label_list = value

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset
    
    @train_dataset.setter
    def train_dataset(self, value: Dataset):
        self._train_dataset = value

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset
    
    @test_dataset.setter
    def test_dataset(self, value: Dataset):
        self._test_dataset = value

    @property
    def val_dataset(self) -> Optional[Dataset]:
        return self._val_dataset
    
    @val_dataset.setter
    def val_dataset(self, value: Optional[Dataset]):
        self._val_dataset = value