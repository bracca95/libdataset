import os
import torch
import torchvision

from PIL import Image
from glob import glob
from typing import Optional, Union, List
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.imgproc import Processing
from src.utils.config_parser import Config
from src.utils.tools import Logger, Tools


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
        extra_image_list = [f for f in glob(os.path.join(os.path.dirname(self.dataset_path), self.AUG_DIR, "*.png"))] if self.augment_offline else []
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
        dataset_augment_dir = os.path.join(os.path.dirname(self.dataset_path), self.AUG_DIR)
        
        if os.path.exists(dataset_augment_dir):
            Logger.instance().warning("the dataset has already been augmented")
            return
        
        os.makedirs(dataset_augment_dir)
        image_list = self.get_image_list(["break", "mark"]) # "break", "mark", "scratch"
        Processing.store_augmented_images(image_list, dataset_augment_dir)

        Logger.instance().debug("dataset augmentatio completed")
        

    def load_image(self, path: str) -> torch.Tensor:
        # about augmentation https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch
        img_pil = Image.open(path).convert("L")

        # crop
        # TODO: the else condition fits well for almost square-shaped images but not for scratches or breaks
        if img_pil.size[0] * img_pil.size[1] < self.crop_size * self.crop_size:
            m = min(img_pil.size)
            centercrop = transforms.Compose([transforms.CenterCrop((m, m))])
            resize = transforms.Compose([transforms.Resize((self.crop_size, self.crop_size))])
            
            img_pil = centercrop(img_pil)
            img_pil = resize(img_pil)

            Logger.instance().debug(f"image size for {os.path.basename(path)} is less than required. Upscaling.")
        else:
            centercrop = transforms.Compose([transforms.CenterCrop((self.crop_size, self.crop_size))])
            img_pil = centercrop(img_pil)
        
        # resize (if required)
        if self.img_size is not None:
            resize = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])
            img_pil = resize(img_pil)

        # rescale [0-255](int) to [0-1](float)
        totensor = transforms.Compose([transforms.ToTensor()])
        img = totensor(img_pil)

        # normalize
        if self.mean is not None and self.std is not None:
            normalize = transforms.Normalize(self.mean, self.std)
            img = normalize(img)

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


class MNIST:

    label_to_idx = { str(i): i for i in range(10) }

    def __init__(self, root_dir: str, crop_size: Optional[int], img_size: Optional[int]):
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        
        self.in_dim = 28
        self.out_dim = 10
        if crop_size is None and img_size is None:
            self.in_dim = 28
        if crop_size is not None and img_size is None:
            self.in_dim = crop_size
        if crop_size is None and img_size is not None:
            self.in_dim = img_size
        if crop_size is not None and img_size is not None:
            self.in_dim = img_size
        
        if crop_size is not None:
            Logger.instance().warning("Cropping MNIST!!")
            self.transform = transforms.Compose([
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor()
            ])
        
        if img_size is not None:
            Logger.instance().warning("Reshaping MNIST!!")
            self.transform = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor()
            ])

        if crop_size is None and img_size is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
    
    def get_train_test(self):
        train = torchvision.datasets.MNIST(root=self.root_dir, 
                                          train=True, 
                                          transform=self.transform,  
                                          download=True)
        
        test = torchvision.datasets.MNIST(root=self.root_dir, 
                                          train=False, 
                                          transform=self.transform)
        
        return train, test


class TTSet:
    def __init__(self, 
                 tt_set: Union[Dataset, torchvision.datasets.MNIST],
                 indexes: Optional[List[int]] = None,
                 dataset: Optional[Dataset] = None
                ):
        self.tt_set = tt_set
        self.indexes = indexes
        self.dataset = dataset
        self.classes: Optional[List[str]] = self.__get_classes()
        self.elem_per_class: Optional[dict] = self.__get_elem_per_class()

    def __get_classes(self):
        if self.indexes is None and self.dataset is None:
            return self.tt_set.classes
        else:
            train_label_list = [self.dataset[idx][1] for idx in self.indexes]
            classes = set(train_label_list)
            Logger.instance.debug(f"split dataset: {classes}")
            
            return list(classes)

    def __get_elem_per_class(self):
        if self.indexes is None and self.dataset is None:
            return None
        
        if self.classes is None:
            self.__get_classes()

        di = {self.dataset.idx_to_label[i]: self.classes.count(i) for i in self.classes}
        Logger.instance().debug(f"number of elements per class: {di}")

        return di
    