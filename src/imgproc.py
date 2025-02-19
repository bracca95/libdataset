import os
import torch

from PIL import Image
from PIL.Image import Image as PilImgType
from tqdm import tqdm
from torch import Tensor
from typing import Optional, List, Callable, Tuple
from torchvision.transforms import transforms
from torchvision.transforms import functional as func_t

from .utils.tools import Logger


class ConditionalRandomCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: PilImgType):
        if min(img.size) >= self.size:
            return transforms.RandomCrop(self.size)(img)
        else:
            return img


class RandomProjection:
    """Random Projection

    Init a matrix that performs any random projection, than flatten the image that you want to augment so that
    $\\mathbb{R}^{N_x \\times N_x} \\cdot \\mathbb{R}^{N_x}$ and finally reshape back to the original image size.
    """
    
    def __init__(self, matrix):
        self.matrix = matrix
        
    def __call__(self, x):
        bs, chans, h, w = x.size()
        
        if not h * w == self.matrix.size(0):
            raise ValueError(f"Projection matrix size {self.matrix.shape} must match flattened image size {h * w}.")
    
        x_flat = x.view(bs, chans, -1)
        x_augment = torch.matmul(x_flat, self.matrix.T)
        x_back = x_augment.view(bs, chans, h, w)
        
        return x_back


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
    def sample_augment(x: PilImgType, img_size: int, strong: bool) -> Tensor:
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
    def rotate_image(img: PilImgType, angle: int, zero_deg: bool, prob: float=1.0) -> Tuple[PilImgType, int]:
        """Rotate a PIL Image with multiples of 'angle' degrees with a given probability.

        Args:
            img (Image): A PIL Image of shape (C, H, W).
            angle (int): rotate by (multiplier of) an angle: must be divider of 360
            zero_deg (bool): include zero degree rotation (no rotation)
            prob (float=1.0): A float value between 0.0 and 1.0 representing the probability of rotation.

        Returns:
            The same Image rotated by angle * n_rot (ranomly sampled)

        Raises:
            ValueError if the angle is not a divider of 360
        """

        if not 360 % angle == 0:
            raise ValueError(f"'angle' must be divider of 360 ({angle})")

        divider = 360 // angle
        start = 0 if zero_deg else 1

        n_rot = 0
        if torch.rand(1) < prob:
            n_rot = int(torch.randint(start, divider, (1,)).item())
            img = img.rotate(angle * n_rot, expand=True)
        
        return img, n_rot
    
    @staticmethod
    def rotate_tensor(x: Tensor, angle: int, zero_deg: bool, prob: float=1.0) -> Tuple[Tensor, int]:
        """Rotate a Tensor (batch) with multiples of 'angle' degrees with a given probability.

        Args:
            x (Tensor): A Tensor of shape (N, C, H, W).
            angle (int): rotate by (multiplier of) an angle: must be divider of 360
            zero_deg (bool): include zero degree rotation (no rotation)
            prob (float=1.0): A float value between 0.0 and 1.0 representing the probability of rotation.

        Returns:
            The same Tensor rotated by angle * n_rot (ranomly sampled)

        Raises:
            ValueError if the angle is not a divider of 360
        """

        if not 360 % angle == 0:
            raise ValueError(f"'angle' must be divider of 360 ({angle})")

        divider = 360 // angle
        start = 0 if zero_deg else 1

        n_rot = 0
        if torch.rand(1) < prob:
            n_rot = int(torch.randint(start, divider, (1,)).item())
            x = func_t.rotate(x, angle=90 * n_rot)
        
        return x, n_rot
    
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
            img_pil, _ = Processing.rotate_image(img_pil, angle=90, zero_deg=False, prob=0.5)
            for it in range(iters):
                img_aug = aug_fun(img_pil)
                new_filename = f"{img_filename}_{it}.{img_ext}"
                img_aug.save(os.path.join(new_dir, new_filename))
