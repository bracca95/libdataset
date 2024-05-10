import os
import torch

from abc import abstractmethod
from PIL import Image
from torch import Tensor
from typing import Optional, Tuple, Set, List, Callable
from PIL.Image import Image as PilImgType
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from ..dataset import DatasetWrapper, DatasetLauncher
from ...imgproc import Processing
from ...utils.tools import Logger, Tools
from ...utils.config_parser import DatasetConfig
from ....config.consts import General as _CG


class FewShotDataset(DatasetWrapper):
    """FSL datasets managed with ImageFolder

    Current available datasets (in my hosts):
        - CIFAR-FS
        - miniImagenet
        - CUB (originally not intended for FSL, I provided my splits)
    
    SeeAlso:
        [CIFAR-FS main page](https://github.com/bertinetto/r2d2)
        [CIFAR-FS download](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view?usp=sharing)
        [miniImagenet main page](https://github.com/fiveai/on-episodes-fsl)
        [miniImagenet download](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE)
        [miniImagenet split](https://github.com/mileyan/simple_shot/tree/master/split/mini)
        [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/))
    """

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__()
        self.dataset_config = dataset_config
        
        self._image_list = self.get_image_list(None)
        self._label_list = self.get_label_list()
        self._train_dataset, self._val_dataset, self._test_dataset = self.split_dataset(self.split_method)
    
    @abstractmethod
    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        ...

    @abstractmethod
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        ...

    @abstractmethod
    def expected_length(self):
        ...

    def get_label_list(self) -> List[int]:
        if self.image_list is None:
            self.image_list = self.get_image_list(None)

        # extract unique label values from dirname
        label_set = sorted(set([os.path.basename(os.path.dirname(image_name)) for image_name in self.image_list]))
        self.label_to_idx = { val: i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        # use mapping to return an int for the corresponding str label
        return [self.label_to_idx[os.path.basename(os.path.dirname(image_name))] for image_name in self.image_list]

    def load_image(self, path: str, augment: Optional[List[str]]) -> torch.Tensor:
        repeat: int = self.dataset_config.augment_times      # type: ignore .non-null checked in config parser
        img_pil = Image.open(path).convert("RGB")

        img_list = []
        if augment is not None and "dataset" in [a.lower() for a in augment]:
            img_list = self.ssl_augment_basic(img_pil, self.dataset_config, (2*repeat)-1, strong=True, weak=False)
            img_list.insert(0, img_pil)

        if augment is not None and "support" in [a.lower() for a in augment]:
            img_list = self.ssl_augment_basic(img_pil, self.dataset_config, repeat, strong=True, weak=False)

        if augment is not None and "umtra" in [a.lower() for a in augment]:
            repeat = 5
            img_list = self.ssl_augment_basic(img_pil, self.dataset_config, (2*repeat)-1, strong=True, weak=False)
            img_list.insert(0, img_pil)

        # as in traditional SSL benchmarks
        # in the original PsCo paper one augmentation is strong the other is weak
        if augment is not None and "psco" in [a.lower() for a in augment]:
            img_list = self.ssl_augment_basic(img_pil, self.dataset_config, repeat+1, strong=True, weak=True)
        
        # basic operations: always performed
        basic_transf = transforms.Compose([
            transforms.Resize((self.dataset_config.image_size, self.dataset_config.image_size)),
            transforms.ToTensor(),
            DatasetLauncher.normalize_or_identity(self.dataset_config)
        ])

        # basic case
        if len(img_list) < 1:
            return basic_transf(img_pil)
        
        # return augmented stacked images
        augmented_images = [basic_transf(curr_img) for curr_img in img_list]
        augmented_images = torch.stack(augmented_images, dim=0)

        return augmented_images
    
    def split_dataset(self, split_method: Callable) -> Tuple[DatasetLauncher, Optional[DatasetLauncher], DatasetLauncher]:
        # get split
        class_train, class_val, class_test = split_method()
        
        tensor_labels = torch.tensor(self.label_list, dtype=torch.int)
        
        # select the images where indices corresponds to train/val/test classes
        def select_img_lbl(class_set: set) -> Tuple[List[str], List[int]]:
            label_ints = torch.tensor([self.label_to_idx[c] for c in class_set], dtype=torch.int)
            indices = torch.where(tensor_labels == label_ints.unsqueeze(1))[1]
            images = [self.image_list[i] for i in indices]
            labels = [self.label_list[i] for i in indices]
            return images, labels
        
        # create DatasetLauncher with augmentation for training if required
        augment = self.dataset_config.augment_online
        train_dataset = DatasetLauncher(*select_img_lbl(class_train), augment, load_img_callback=self.load_image)
        val_dataset = DatasetLauncher(*select_img_lbl(class_val), augment=None, load_img_callback=self.load_image)
        test_dataset = DatasetLauncher(*select_img_lbl(class_test), augment=None, load_img_callback=self.load_image)
        
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
    
    @staticmethod
    def ssl_augment_basic(x: PilImgType, dataset_config: DatasetConfig, n: int, strong: bool, weak: bool) -> List[Tensor]:
        img_size = dataset_config.image_size
        
        transform_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 0.8)), # ConditionalRandomCrop(64)
            Processing.rotate_lambda(deg=60, p=1.0),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(3),
            transforms.RandomAffine(degrees=0, shear=[-45, 45, -45, 45])
        ]

        weak_transform_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 0.8)),
            transforms.RandomHorizontalFlip(p=1.0),
        ]

        if strong and weak:  # psco
            random_transforms = [
                transforms.Compose([transforms.RandomChoice(transform_list) for _ in range(3)])
                for _ in range(n//2)
            ]
            random_transforms.extend([transforms.Compose(weak_transform_list) for _ in range(n//2)])
        elif strong and not weak:
            # select three augmentations for each image (one strong augmented version is returned)
            random_transforms = [
                transforms.Compose([transforms.RandomChoice(transform_list) for _ in range(3)])
                for _ in range(n)
            ]
        else:
            random_transforms = [transforms.RandomChoice(transform_list) for _ in range(n)]
        
        return [aug_method(x) for aug_method in random_transforms]

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