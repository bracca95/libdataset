import os
import torch

from PIL import Image
from glob import glob
from typing import Optional, Union, List
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from src.imgproc import Processing
from src.utils.config_parser import Config
from src.utils.tools import Logger, Tools
from config.consts import SubsetsDict
from config.consts import General as _GC


class DefectViews(Dataset):

    label_to_idx = {
        "bubble": 0, 
        "point": 1,
        "break": 2,
        "dirt": 3,
        "mark": 4,
        "scratch": 5
    }

    idx_to_label = Tools.invert_dict(label_to_idx)

    AUG_DIR = "img_augment"

    def __init__(self, dataset_path: str, aug_off: bool, aug_on: bool, crop_size: int, img_size: Optional[int] = None, filt: Optional[List[str]] = None):
        self.dataset_path: str = dataset_path
        self.dataset_aug_path: str = os.path.join(os.path.dirname(self.dataset_path), self.AUG_DIR)
        self.filt: Optional[List[str]] = filt

        self.augment_online: bool = aug_on
        self.augment_offline: bool = aug_off
        if self.augment_offline:
            self.augment_dataset()

        self.image_list: Optional[List[str]] = self.get_image_list(self.filt)
        self.label_list: Optional[List[int]] = self.get_label_list()

        self.crop_size: int = crop_size
        self.img_size: Optional[int] = img_size
        self.in_dim = self.img_size if self.img_size is not None else self.crop_size
        self.out_dim = len(self.label_to_idx)

        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        extra_image_list = [f for f in glob(os.path.join(self.dataset_aug_path, "*.png"))] if self.augment_offline else []
        image_list = [f for f in glob(os.path.join(self.dataset_path, "*.png"))]
        
        image_list = image_list + extra_image_list
        
        if filt is not None:
            filenames = list(map(lambda x: os.path.basename(x), image_list))
            image_list = list(filter(lambda x: Tools.check_string(x.rsplit("_")[0], filt, True, False), filenames))
            image_list = list(map(lambda x: os.path.join(self.dataset_path, x), image_list))
        
        if not all(map(lambda x: x.endswith(".png"), image_list)) or image_list == []:
            raise ValueError("incorrect image list. Check the provided path for your dataset.")

        return image_list

    def get_label_list(self) -> List[int]:
        if self.image_list is None:
            self.get_image_list(self.filt)

        filenames = list(map(lambda x: os.path.basename(x), self.image_list))
        label_list = list(map(lambda x: x.rsplit("_")[0], filenames))
       
        Logger.instance().debug(f"Labels used: {set(label_list)}")
        Logger.instance().debug(f"Number of images per class: { {i: label_list.count(i) for i in set(label_list)} }")

        return [self.label_to_idx[defect] for defect in label_list]
    
    def augment_dataset(self):
        Logger.instance().debug("increasing the number of images...")
        
        if os.path.exists(self.dataset_aug_path):
            if len(os.listdir(self.dataset_aug_path)) > 0:
                Logger.instance().warning("the dataset has already been augmented")
                return
        else:
            os.makedirs(self.dataset_aug_path)
        
        image_list = self.get_image_list(["break", "mark"]) # "break", "mark", "scratch"
        Processing.store_augmented_images(image_list, self.dataset_aug_path)

        Logger.instance().debug("dataset augmentatio completed")
        

    def load_image(self, path: str) -> torch.Tensor:
        # about augmentation https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch
        img_pil = Image.open(path).convert("L")

        # crop
        img_pil = Processing.crop_no_padding(img_pil, self.crop_size, path)
        
        # resize (if required)
        if self.img_size is not None:
            img_pil = transforms.Resize((self.img_size, self.img_size))(img_pil)

        # rescale [0-255](int) to [0-1](float)
        img = transforms.ToTensor()(img_pil)

        # normalize
        if self.mean is not None and self.std is not None:
            img = transforms.Normalize(self.mean, self.std)(img)

        return img # type: ignore
    
    @staticmethod
    def compute_mean_std(dataset: Dataset, config: Config):
        # https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/31
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

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

        config.dataset_mean = mean.tolist()
        config.dataset_std = std.tolist()
        config.serialize(os.getcwd(), "config/config.json")
        Logger.instance().warning(f"Mean: {mean}, std: {std}. Run the program again.")

    @staticmethod
    def split_dataset(dataset: Dataset, split_ratios: Union[float, List[float]]=[.8]) -> SubsetsDict:
        if type(split_ratios) is float:
            split_ratios = [split_ratios]
        
        if len(split_ratios) != 1 and len(split_ratios) != 3:
            raise ValueError(f"split_ratios argument accepts either a list of 1 value (train,test) or 3 (train,val,test)")
        
        train_len = int(len(dataset) * split_ratios[0])
        if len(split_ratios) == 1:
            split_lens = [train_len, len(dataset) - train_len]
        else:
            val_len = train_len - int(len(dataset) * split_ratios[1])
            split_lens = [train_len, val_len, len(dataset) - (train_len + val_len)]

        subsets = random_split(dataset, split_lens)
        val_set = None if len(subsets) == 2 else subsets[1]

        train_str, val_str, test_str = _GC.DEFAULT_SUBSETS
        
        return { train_str: subsets[0], val_str: val_set, test_str: subsets[-1] }   # type: ignore

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]
        curr_label_batch = self.label_list[index]
        
        return self.load_image(curr_img_batch), curr_label_batch

    def __len__(self):
        return len(self.image_list) # type: ignore


class BubblePoint(DefectViews):

    label_to_idx = {
        "bubble": 0,
        "point": 1
    }

    idx_to_label = Tools.invert_dict(label_to_idx)

    def __init__(self, dataset_path: str, aug_on: bool, crop_size: int, img_size: Optional[int] = None):
        super().__init__(dataset_path, aug_off=False, aug_on=aug_on, crop_size=crop_size, img_size=img_size, filt=["bubble", "point"])
