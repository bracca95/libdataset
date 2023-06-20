import os
import torch

from PIL import Image
from glob import glob
from typing import Optional, List
from torchvision import transforms

from src.imgproc import Processing
from src.utils.tools import Logger, Tools
from src.datasets.staple_dataset import CustomDataset


class GlassOpt(CustomDataset):

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
    NO_CROP = ["scratch", "break", "mark"]

    # https://stackoverflow.com/a/42583719
    split_name = staticmethod(lambda x: os.path.basename(x).rsplit("_", -1)[0])

    def __init__(self, dataset_path: str, aug_off: Optional[List[str]], aug_on: Optional[List[str]], crop_size: int, img_size: int):
        self.dataset_path: str = dataset_path
        self.dataset_aug_path: str = os.path.join(os.path.dirname(self.dataset_path), self.AUG_DIR)
        self.filt: List[str] = list(self.label_to_idx.keys())

        self.augment_online: Optional[List[str]] = aug_on
        self.augment_offline: Optional[List[str]] = aug_off
        if self.augment_offline is not None:
            self.augment_dataset(50)

        self.image_list: Optional[List[str]] = self.get_image_list(self.filt)
        self.label_list: Optional[List[int]] = self.get_label_list()

        self.crop_size: int = crop_size
        self.img_size: Optional[int] = img_size
        self.in_dim = self.img_size
        self.out_dim = len(self.label_to_idx)

        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def get_image_list(self, filt: List[str]) -> List[str]:
        """Read all the filenames in the dataset directory

        Read and return all the filenames in the dataset directory. If filt is not None, it will read only the specified
        elements (by class names - `self.label_to_idx` as an example). It will read also the directory containing
        augmented images, if initialized.

        Args:
            filt (Optional[List[str]]): if not None, restrict the defect classes by name

        Returns:
            List[str] containing the full path to the images
        """
        
        extra_image_list = glob(os.path.join(self.dataset_aug_path, "*.png")) if self.augment_offline else list()
        image_list = glob(os.path.join(self.dataset_path, "*.png"))
        image_list = image_list + extra_image_list
        
        image_list = list(filter(lambda x: Tools.check_string(self.split_name(x), filt, True, False), image_list))
        
        if not all(map(lambda x: x.endswith(".png"), image_list)) or image_list == []:
            raise ValueError("incorrect image list. Check the provided path for your dataset.")
        
        Logger.instance().info("Got image list")

        return image_list

    def get_label_list(self) -> List[int]:
        """Get all the label names, for each image name

        Returns:
            List[str] with the name of the labels
        """
        
        if self.image_list is None:
            self.get_image_list(self.filt)

        label_list = list(map(lambda x: self.split_name(x), self.image_list))
       
        Logger.instance().debug(f"Labels used: {set(label_list)}")
        Logger.instance().debug(f"Number of images per class: { {i: label_list.count(i) for i in set(label_list)} }")

        return [self.label_to_idx[defect] for defect in label_list]
    
    def augment_dataset(self, iters: int):
        """Perform offline augmentation
        
        Increase the number of available samples with augmentation techniques, if required in config. Offline
        augmentation can work on a limited set of classes; indeed, it should be used if there are not enough samples
        for each class.

        iters (int): number of augmentation iteration for the same image
        """
        
        Logger.instance().debug("increasing the number of images...")
        
        if os.path.exists(self.dataset_aug_path):
            if len(os.listdir(self.dataset_aug_path)) > 0:
                Logger.instance().warning("the dataset has already been augmented")
                return
        else:
            os.makedirs(self.dataset_aug_path)
        
        image_list = self.get_image_list(self.augment_offline)
        Processing.store_augmented_images(image_list, self.dataset_aug_path, iters)

        Logger.instance().debug("dataset augmentatio completed")
        

    def load_image(self, path: str) -> torch.Tensor:
        """Load image as tensor

        This will be used in the trainloader only. Read the image, crop it, resize it (if required in config), make it
        a tensor and normalize (if required).

        Args:
            path (str): full path to the image

        Returns:
            torch.Tensor of the modified image

        See Also:
            https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch
            https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch
            https://stackoverflow.com/a/72642001
        """
        
        img_pil = Image.open(path).convert("L")

        # crop
        # augmented images are already square-shaped, do not crop!
        # scratches, breaks and marks are likely not to be square-shaped, so cropping will lose information
        # CHECK: I could crop only bubbles, since they are the only ones that have to preserve proportions strictly
        if self.dataset_aug_path not in path and not Tools.check_string(os.path.basename(path), self.NO_CROP, False, False):
            img_pil = Processing.crop_no_padding(img_pil, self.crop_size, path)
        
        # resize
        img_pil = transforms.Resize((self.img_size, self.img_size))(img_pil)

        # rescale [0-255](int) to [0-1](float)
        img = transforms.ToTensor()(img_pil)

        # normalize
        if self.mean is not None and self.std is not None:
            img = transforms.Normalize(self.mean, self.std)(img)

        return img # type: ignore

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]
        curr_label_batch = self.label_list[index]
        
        return self.load_image(curr_img_batch), curr_label_batch

    def __len__(self):
        return len(self.image_list) # type: ignore


class GlassOptBckg(GlassOpt):

    label_to_idx = {
        "background": 0,
        "bubble": 1, 
        "point": 2,
        "break": 3,
        "dirt": 4,
        "mark": 5,
        "scratch": 6
    }

    idx_to_label = Tools.invert_dict(label_to_idx)
    NO_CROP = ["scratch", "break", "mark", "background"]

    def __init__(self, dataset_path: str, aug_off: Optional[List[str]], aug_on: Optional[List[str]], crop_size: int, img_size: int):
        super().__init__(dataset_path, aug_off=aug_off, aug_on=aug_on, crop_size=crop_size, img_size=img_size)


class QPlusV1(GlassOpt):

    label_to_idx = {
        "mark": 0,
        "glass_id": 1,
        "dirt": 2,
        "point": 3,
        "scratch_light": 4,
        "halo": 5,
        "scratch_multi": 6,
        "dust": 7,
        "coating": 8,
        "scratch_heavy": 9,
        "dirt_halo": 10
    }

    idx_to_label = Tools.invert_dict(label_to_idx)
    NO_CROP = ["mark", "glass_id", "dirt", "dirt_halo", "point", "scratch_light", "scratch_heavy", "scratch_multi", "dust", "halo", "coating"]
    split_name = staticmethod(lambda x: os.path.basename(x).rsplit("_did", 1)[0])

    def __init__(self, dataset_path: str, aug_off: Optional[List[str]], aug_on: Optional[List[str]], crop_size: int, img_size: int):
        super().__init__(dataset_path, aug_off=None, aug_on=None, crop_size=crop_size, img_size=img_size)


class QPlusV2(GlassOpt):

    label_to_idx = {
        "mark": 0,
        "glass_id": 1,
        "dirt": 2,
        "point": 3,
        "scratch_light": 4,
        "halo": 5,
        "scratch_multi": 6,
        "dust": 7,
        "coating": 8,
        "scratch_heavy": 9,
        "dirt_halo": 10,
        "bubble": 11,
        "bubble_hole": 12
    }

    idx_to_label = Tools.invert_dict(label_to_idx)
    NO_CROP = list(label_to_idx.keys())
    split_name = staticmethod(lambda x: os.path.basename(x).rsplit("_did", 1)[0])

    def __init__(self, dataset_path: str, aug_off: Optional[List[str]], aug_on: Optional[List[str]], crop_size: int, img_size: int):
        super().__init__(dataset_path, aug_off=None, aug_on=None, crop_size=crop_size, img_size=img_size)


class BubblePoint(GlassOpt):

    label_to_idx = {
        "bubble": 0,
        "point": 1
    }

    idx_to_label = Tools.invert_dict(label_to_idx)

    def __init__(self, dataset_path: str, aug_off: Optional[List[str]], aug_on: Optional[List[str]], crop_size: int, img_size: int):
        super().__init__(dataset_path, aug_off=None, aug_on=aug_on, crop_size=crop_size, img_size=img_size)
