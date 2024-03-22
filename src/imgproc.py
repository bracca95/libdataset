import os
import torch

from PIL import Image
from PIL.Image import Image as PilImgType
from tqdm import tqdm
from typing import Optional, List, Callable
from torchvision.transforms import transforms

from .utils.tools import Logger


class ConditionalRandomCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: PilImgType):
        if min(img.size) >= self.size:
            return transforms.RandomCrop(self.size)(img)
        else:
            return img


class Processing:

    offline_transforms = transforms.RandomOrder([
        transforms.RandomCrop((256, 256), pad_if_needed=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)
    ])
    offline_transforms_v2 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        #transforms.RandomRotation(180, expand=True)
    ])

    @staticmethod
    def crop_no_padding(img: PilImgType, crop_size: int, path: Optional[str]=None) -> PilImgType:
        """Crop ensuring that the output image is not padded

        Avoid padding if the image size is smaller than required size. The image is first cropped by its shortest side
        and eventually resized to the specified crop size.
        Mind that this fits well for almost-square-shaped images like bubbles and points, but not for scratches, breaks
        or marks, which are likely to be rectangular-shaped.

        Args:
            img (Image)
            crop_size (int)
            path (Optional[str]): if specified, the image name is logged

        Returns:
            cropped Image
        """
        
        if img.size[0] * img.size[1] < crop_size * crop_size:
            m = min(img.size)
            center_and_resize = transforms.Compose([
                transforms.CenterCrop((m, m)),
                transforms.Resize((crop_size, crop_size))
            ])
            
            img = center_and_resize(img)
            if path is not None:
                Logger.instance().info(f"image size for {os.path.basename(path)} is less than required. Upscaling.")
        else:
            # the else condition fits well for almost square-shaped images but not for scratches or breaks
            img = transforms.CenterCrop((crop_size, crop_size))(img)

        return img

    @staticmethod
    def rotate_image(img: PilImgType, prob: float) -> PilImgType:
        """Rotate a PIL Image with multiples of 90 degrees with a given probability.

        Args:
            img (Image): A PIL Image of shape (C, H, W).
            prob (float): A float value between 0.0 and 1.0 representing the probability of rotation.

        Returns:
            The same Image rotated by 0, 90, 180 or 270 degrees.
        """

        if torch.rand(1) < prob:
            n_rot = torch.randint(1, 4, (1,)).item()  # random integer between 1 and 3
            img = img.rotate(90 * n_rot, expand=True)
        
        return img
    
    @staticmethod
    def rotate_lambda(deg: int, p: float=0.5) -> torch.nn.Module:
        if torch.rand(1) < p:
            return transforms.RandomRotation(degrees=deg)
        
        return torch.nn.Identity()
    
    @staticmethod
    def store_augmented_images(img_list: List[str], new_dir: str, iters: int, aug_fun: Callable[[PilImgType], PilImgType]):
        """Save the augmented images in the specified folder

        Args:
            img_list (List[str]): the images that have to be augmented
            new_dir (str): output directory
            iters (int): number of iterations for the same image
        """
        
        for img_path in tqdm(img_list):
            img_pil = Image.open(img_path).convert("L")
            img_filename, img_ext = os.path.basename(img_path).rsplit(".")

            # remove the following line when random rotation is used instead
            img_pil = Processing.rotate_image(img_pil, prob=0.5)
            for it in range(iters):
                img_aug = aug_fun(img_pil)
                new_filename = f"{img_filename}_{it}.{img_ext}"
                img_aug.save(os.path.join(new_dir, new_filename))
