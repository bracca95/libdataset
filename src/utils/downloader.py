import os
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from typing import Optional, List
from torchvision.datasets import MNIST as torch_MNIST
from torchvision.datasets import FashionMNIST as torch_FashionMNIST

from .tools import Tools, Logger



class Download:

    @staticmethod
    def save_images(img_array: np.ndarray, label_array: np.ndarray, dirs: List[str]) -> None:
        counter = 0
        
        for idx, img in zip(label_array, img_array):
            img_name = f"{counter:05d}.png"
            pil_image = Image.fromarray(img.astype(np.uint8), mode='L')
            pil_image.save(f"{os.path.join(*dirs, str(idx), img_name)}")
            counter += 1

    @staticmethod
    def download_mnist(root: str, subdirs: Optional[List[str]], version: str="MNIST"):
        if subdirs is None:
            subdirs = list()

        if os.path.exists(os.path.join(root, *subdirs, "train", str(0))):
            Logger.instance().debug(f"{version} already downloaded and images are saved to png")
            return

        # choose dataset
        dataset = torch_MNIST if version == "MNIST" else torch_FashionMNIST
        
        # the meta test split is not specified in config, so the directory might not be created
        if not os.path.exists(root):
            Logger.instance().warning(f"Creating {root}, if you plan to use {version} as test set (meta), it's ok.")
            os.makedirs(root)

        # download from torchvision
        Logger.instance().debug(f"Downloading {version}")
        train_dataset = dataset(root=root, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dataset(root=root, train=False, transform=transforms.ToTensor(), download=True)

        # to numpy
        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()

        # save
        Logger.instance().debug(f"Saving torch {version} to png")
        os.makedirs(os.path.join(root, *subdirs, "train"), exist_ok=True)
        os.makedirs(os.path.join(root, *subdirs, "test"), exist_ok=True)

        # prepare train and test directories
        for i in range(10):
            os.makedirs(os.path.join(root, *subdirs, "train", str(i)), exist_ok=True)
            os.makedirs(os.path.join(root, *subdirs, "test", str(i)), exist_ok=True)

        # save train, test images
        Download.save_images(train_images, train_labels, dirs=[root, *subdirs, "train"])
        Download.save_images(test_images, test_labels, dirs=[root, *subdirs, "test"])

        Logger.instance().debug(f"{version} has been saved!")
