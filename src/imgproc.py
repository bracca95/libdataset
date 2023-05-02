import os
import torch

from PIL import Image
from tqdm import tqdm
from typing import List
from torchvision import transforms


class Processing:

    ITERS = 10
    offline_transforms = transforms.RandomOrder([
        transforms.RandomCrop((28), pad_if_needed=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)
    ])
    
    @staticmethod
    def store_augmented_images(img_list: List[str], new_dir: str):
        for img_path in tqdm(img_list):
            img_pil = Image.open(img_path).convert("L")
            img_filename, img_ext = os.path.basename(img_path).rsplit(".")

            for it in range(Processing.ITERS):
                img_aug = Processing.offline_transforms(img_pil)
                new_filename = f"{img_filename}_{it}.{img_ext}"
                img_aug.save(os.path.join(new_dir, new_filename))
