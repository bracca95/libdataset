import os
import torch

from abc import ABC, abstractmethod
from torch import Tensor
from typing import Tuple, List, Callable, Optional
from PIL.Image import Image as PilImgType
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from ..imgproc import Processing
from ..utils.tools import Logger
from ..utils.config_parser import DatasetConfig


class DatasetWrapper(ABC):
    """Wrapper for an entire dataset

    This interface is meant to wrap an entire dataset with its splits. Torch Datasets are usually kept separated, but
    I want to wrap them in a unique class in order to be able to split as I wish and keep track of the label list and
    image paths for all the samples (it may be required in tasks like FSL).
    """

    @property
    @abstractmethod
    def image_list(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def label_list(self) -> List[int]:
        ...

    @property
    @abstractmethod
    def train_dataset(self) -> Dataset:
        ...

    @property
    @abstractmethod
    def test_dataset(self) -> Dataset:
        ...

    @property
    @abstractmethod
    def val_dataset(self) -> Optional[Dataset]:
        ...

    @staticmethod
    def sample_augment(x: PilImgType, dataset_config: DatasetConfig, strong: bool) -> Tensor:
        img_size = dataset_config.image_size
        
        # define transformations that can fit both RGB and L images
        transform_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 0.8)), # ConditionalRandomCrop(64)
            Processing.rotate_lambda(deg=60, p=1.0),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.GaussianBlur(3),
            transforms.RandomAffine(degrees=0, shear=[-45, 45, -45, 45])
        ]

        # transformations for RGB images only
        transform_rgb_list = [
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ]

        # add RGB transformations when the image has 3 channels
        if x.mode == "RGB":
            transform_list.extend(transform_rgb_list)

        # select 3 augmentations if strong, 1 if not
        n = 3 if strong else 1
        random_transforms = transforms.Compose([transforms.RandomChoice(transform_list) for _ in range(n)])
        
        return random_transforms(x)


class DatasetLauncher(Dataset):
    """One split of DatasetWrapper

    This class is meant to replicate a single torch Dataset, but it is able to keep track of all the indices and image
    paths. Torch structures like ImageFolder or other subclasses of Dataset do not always allow for that.
    """

    def __init__(
            self,
            image_list: List[str],
            label_list: List[int],
            augment: Optional[List[str]],
            load_img_callback: Callable[[str, Optional[List[str]]], torch.Tensor]
    ):
        super().__init__()
        self.image_list = image_list
        self.label_list = label_list
        self.augment = augment
        self.load_img_callback = load_img_callback

        self.info_dict: Optional[dict] = None

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]
        curr_label_batch = self.label_list[index]
        
        return self.load_img_callback(curr_img_batch, self.augment), curr_label_batch

    def __len__(self):
        return len(self.label_list)

    def set_info(self, label_list: List[int], idx_to_label: dict):
        t_label = torch.tensor(label_list, dtype=torch.int)
        info = { idx_to_label[value.item()]: (t_label == value).sum().item() for value in torch.unique(t_label) }
        self.info_dict = info

    @staticmethod
    def compute_mean_std(dataset: Dataset, ds_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if "imagenet" in ds_type:
            Logger.instance().debug(f"Dataset type is {ds_type}: imagenet mean/std selected")
            return torch.Tensor([0.485, 0.456, 0.406]), torch.Tensor([0.229, 0.224, 0.225])

        # https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/31
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        mean = 0.0
        for batch in dataloader:
            images, _ = batch
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(dataloader.dataset)

        var = 0.0
        pixel_count = 0
        for batch in dataloader:
            images, _ = batch
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1))**2).sum([0,2])
            pixel_count += images.nelement()
        std = torch.sqrt(var / pixel_count)

        if any(map(lambda x: torch.isnan(x), mean)) or any(map(lambda x: torch.isnan(x), std)):
            raise ValueError("mean or std are none")

        Logger.instance().warning(f"Mean: {mean}, std: {std}. Run the program again.")
        return mean, std
    
    @staticmethod
    def normalize_or_identity(dataset_config: DatasetConfig) -> torch.nn.Module:
        """Normalize image/s by dataset's mean and std

        If config allows normalization, first check if the mean and std values are available to perform the it. If not,
        returns the original image via identity function. This case can either be an error in configuration or it is 
        necessary when this method is called during the computation of the dataset's mean and std (load_image method).

        Args:
            dataset_config (..utils.config_parser.DatasetConfig)

        Returns:
            torch.nn.Module: Normalize module (with computed mean/std) or identity function
        """

        if dataset_config.normalize:
            if dataset_config.dataset_mean is not None and dataset_config.dataset_std is not None:
                normalize = transforms.Normalize(
                    torch.Tensor(dataset_config.dataset_mean),
                    torch.Tensor(dataset_config.dataset_std)
                )
                return normalize

        return torch.nn.Identity()
    
    @staticmethod
    def save_sample_image_batch(dataset: Dataset, outfolder: str):
        if "sample_batch.png" in os.listdir(outfolder):
            return
        
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        example_data, _ = next(iter(loader))
        img_grid = make_grid(example_data)

        grid_pil = transforms.ToPILImage()(img_grid)
        grid_pil.save(os.path.join(outfolder, "sample_batch.png"))
