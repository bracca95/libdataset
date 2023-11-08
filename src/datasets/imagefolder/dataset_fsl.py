import os

from typing import Optional
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from ..dataset import DatasetWrapper
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
        [CIFAR-FS downlaod](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view?usp=sharing)
        [miniImagenet main page](https://github.com/fiveai/on-episodes-fsl)
        [miniImagenet download](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE)
        [miniImagenet split](https://github.com/mileyan/simple_shot/tree/master/split/mini)
        [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/))
    """

    META_TRAIN = "meta_train"
    META_TEST = "meta_test"
    META_VAL = "meta_val"

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__()
        self.dataset_config = dataset_config
        self._check_meta_split()
        
        self._train_dataset = ImageFolder(
            os.path.join(self.dataset_config.dataset_path, self.META_TRAIN),
            transform=transforms.Compose([transforms.ToTensor(), self.normalize_or_identity(dataset_config)]),
            target_transform=None,
        )

        self._test_dataset = ImageFolder(
            os.path.join(self.dataset_config.dataset_path, self.META_TEST),
            transform=transforms.Compose([transforms.ToTensor(), self.normalize_or_identity(dataset_config)]),
            target_transform=None
        )

        self._val_dataset = None
        if self.META_VAL in os.listdir(self.dataset_config.dataset_path):
            self._val_dataset = ImageFolder(
                os.path.join(self.dataset_config.dataset_path, self.META_VAL),
                transform=transforms.Compose([transforms.ToTensor(), self.normalize_or_identity(dataset_config)]),
                target_transform=None
            )

    def _check_meta_split(self):
        if not set([self.META_TRAIN, self.META_TEST]) <= set(os.listdir(self.dataset_config.dataset_path)):
            msg = f"'meta_train' and/or 'meta_test' dir/s missing in {self.dataset_config.dataset_path}"
            Logger.instance().error(msg)
            raise ValueError(msg)

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset
    
    @train_dataset.setter
    def train_dataset(self, value):
        self._train_dataset = value

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset
    
    @test_dataset.setter
    def test_dataset(self, value):
        self._test_dataset = value

    @property
    def val_dataset(self) -> Optional[Dataset]:
        return self._val_dataset
    
    @val_dataset.setter
    def val_dataset(self, value):
        self._val_dataset = value