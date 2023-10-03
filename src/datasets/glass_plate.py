from __future__ import annotations

import os
import torch
import pandas as pd

from PIL import Image
from glob import glob
from functools import reduce
from itertools import groupby
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import transforms

from .dataset import CustomDataset
from ..utils.tools import Logger, Tools
from ..utils.config_parser import DatasetConfig
from ...config.consts import SubsetsDict
from ...config.consts import General as _CG
from ...config.consts import BboxFileHeader as _CH


@dataclass
class Bbox:
    defect_class: str
    min_x: int
    max_x: int
    min_y: int
    max_y: int


class SinglePlate:
    """Single glass plate class"""

    PATCH_SIZE = 60
    PATCH_STRIDE = 50
    
    ## uncomment to choose which (2 cls vs 4 cls)

    ## 4 cls
    # idx_to_label = {
    #     0: "bubble",
    #     1: "scratch_heavy",
    #     2: "point",
    #     3: "dirt"
    # }

    ## 2 cls
    # idx_to_label = {
    #     0: "bubble",
    #     1: "scratch_heavy"
    # }

    ## bubbles only
    idx_to_label = {
        0: "bubble"
    }

    label_to_idx = Tools.invert_dict(idx_to_label)

    def __init__(self, ch_1: str, ch_2: str):
        self.ch_1 = ch_1
        self.ch_2 = ch_2

        # TODO make as args
        self.patch_list = self.create_patch_list(patch_w=self.PATCH_SIZE, patch_h=self.PATCH_SIZE, stride=self.PATCH_STRIDE)
        self.patch_list_defects: List[Patch] = list()

    def create_patch_list(self, patch_w: int, patch_h: int, stride: int) -> List[Patch]:
        """Create a list of patches for a single glass plate

        Use a sliding window approach to create fixed size image patch where defects are mapped from the world frame
        (absolute values found in the csv) to the body frame (local patch). The full image is open to check its size.
        There might be faster method to infer it.

        Args:
            patch_w (int): desired patch width (columns)
            patch_h (int): desired patch height (rows)
            stride (int): stride for the sliding window

        Returns:
            List[Patch]
        """

        img_w, img_h = Image.open(self.ch_1).size
        return self.sliding_window(self, img_w, img_h, patch_w, patch_h, stride)
    
    def locate_defects(self, df: pd.DataFrame, filt: Optional[List[str]]):
        """Locate defects in the patch

        Starting from the world frame location, convert it to the body frame. Since there are two channels, which have
        slightly different bounding boxes (depending on the intensity for each channel), wrap them to assign one
        bound box for each two-channel image. Do this for every patch.

        Args:
            df (pd.DataFrame): the dataframe grouped by defects for that specific plate.
            filt (Optional[List[str]]): classes to select. If None, use all defect classes
        """

        # wrap
        for _, group in df.groupby(_CH.COL_ID_DEFECT):
            defect_id = group[_CH.COL_ID_DEFECT].tolist()[0]
            defect_class = group[_CH.COL_CLASS_KEY].tolist()[0]

            # skip
            if defect_class is not None and defect_class not in filt: 
                Logger.instance().info(f"skipping defect class {defect_class}: not included in {filt}")
                continue
            
            bbox_min_x = min(group[_CH.COL_BBOX_MIN_X])
            bbox_max_x = max(group[_CH.COL_BBOX_MAX_X])
            bbox_min_y = min(group[_CH.COL_BBOX_MIN_Y])
            bbox_max_y = max(group[_CH.COL_BBOX_MAX_Y])

            if bbox_min_x == bbox_max_x:
                Logger.instance().warning(f"min/max x overlap in defect {defect_id}: increasing bb width")
                bbox_min_x -= 1
                bbox_max_x += 1

            if bbox_min_y == bbox_max_y:
                Logger.instance().warning(f"min/max y overlap in defect {defect_id}: increasing height")
                bbox_min_y -= 1
                bbox_max_y += 1

            # check if there is a defect in a patch
            for patch in self.patch_list:
                if bbox_min_x > patch.start_w and bbox_min_y > patch.start_h \
                and bbox_max_x < patch.start_w + patch.w and bbox_max_y < patch.start_h + patch.h \
                and defect_class in filt:
                    abs_bbox = Bbox(defect_class, bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y)
                    patch.map_defect_locally(abs_bbox)
                    self.patch_list_defects.append(patch)
                    break # keep only one patch with the same defect, do not duplicate
                    # debug: can save image here

    def tolist(self):
        return [self.ch_1, self.ch_2]
    
    @staticmethod
    def sliding_window(plate: SinglePlate, img_w: int, img_h: int, patch_w: int, patch_h: int, stride: int) -> List[Patch]:
        # the image size is reduced. Consider enhancing the method by padding
        patch_list: List[Patch] = list()
        for i in range(0, img_h - patch_h + 1, stride):
            for j in range(0, img_w - patch_w + 1, stride):
                patch = Patch(plate, j, i, patch_w, patch_h)
                patch_list.append(patch)
            
            # include last column patch for each line
            patch = Patch(plate, img_w - patch_w, i, patch_w, patch_h)
            patch_list.append(patch)

        # include last row patches
        for j in range(0, img_w - patch_w + 1, stride):
            patch = Patch(plate, j, img_h - patch_h, patch_w, patch_h)
            patch_list.append(patch)

        patch = Patch(plate, img_w - patch_w, img_h - patch_h, patch_w, patch_h)
        patch_list.append(patch)

        return patch_list
    
    @staticmethod
    def __debug_save_image_patch(patch: Patch):
        # call this in SinglePlate::locate_defects
        from PIL import ImageDraw
        img = Image.open(patch.plate_paths.ch_1).convert("RGB")
        crop = img.crop((patch.start_w, patch.start_h, patch.start_w + patch.w, patch.start_h + patch.h))
        draw = ImageDraw.Draw(crop)
        draw.rectangle([patch.defects[0].min_x, patch.defects[0].min_y, patch.defects[0].max_x, patch.defects[0].max_y], (255, 0, 0))
        crop.save("output/patch.png")

    @staticmethod
    def __save_exact_defect(defect_id: int, patch: Patch):
        # call this in SinglePlate::locate_defects
        img_1 = Image.open(patch.plate_paths.ch_1).convert("L")
        img_2 = Image.open(patch.plate_paths.ch_2).convert("L")

        patch_coords = (patch.start_w, patch.start_h, patch.start_w + patch.w, patch.start_h + patch.h)
        defect_coords = (patch.defects[0].min_x, patch.defects[0].min_y, patch.defects[0].max_x, patch.defects[0].max_y)
        
        patch_1 = img_1.crop(patch_coords)
        patch_2 = img_2.crop(patch_coords)

        defect_1 = patch_1.crop(defect_coords)
        defect_2 = patch_2.crop(defect_coords)

        try:
            defect_1.save(f"output/{patch.defects[0].defect_class}_did_{defect_id}_vid_1.png")
            defect_2.save(f"output/{patch.defects[0].defect_class}_did_{defect_id}_vid_2.png")
        except SystemError:
            Logger.instance().error(f"There is an error in the bounding box, check values: {patch.defects[0]}")
        except AttributeError:
            Logger.instance().error(f"There is an error in the bounding box, check values: {patch.defects[0]}")

class Patch:

    def __init__(self, plate_paths: SinglePlate, start_w: int, start_h: int, w: int, h: int):
        self.plate_paths = plate_paths
        self.start_w = start_w
        self.start_h = start_h
        self.w = w
        self.h = h
        
        self.defects: Optional[List[Bbox]] = None

    def map_defect_locally(self, abs_bbox: Bbox):
        """Map defect on a single patch

        Starting from the world frame location, convert it to the body frame. Since there are two channels, which have
        slightly different bounding boxes (depending on the intensity for each channel), wrap them to assign one
        bound box for each two-channel image.

        Args:
            abs_bbox (Bbox): the values read from the csv file
        """

        rel_bbox_min_x = abs_bbox.min_x - self.start_w
        rel_bbox_max_x = rel_bbox_min_x + (abs_bbox.max_x - abs_bbox.min_x) # self.start_w + self.w - abs_bbox.max_x
        rel_bbox_min_y = abs_bbox.min_y - self.start_h
        rel_bbox_max_y = rel_bbox_min_y + (abs_bbox.max_y - abs_bbox.min_y)

        relative_bbox = Bbox(abs_bbox.defect_class, rel_bbox_min_x, rel_bbox_max_x, rel_bbox_min_y, rel_bbox_max_y)
        
        if self.defects is None:
            self.defects = [relative_bbox]
            return
        
        self.defects.append(relative_bbox)

class GlassPlate(TorchDataset):

    TEST_ONE = False

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config

        # read csv by filtering classes and returning
        self._dataset_csv = "/media/lorenzo/M/datasets/dataset_opt/2.2_dataset_opt/bounding_boxes.csv"
        self._df = self.parse_csv(filt=None)

        # get names of each channel, grouping by plate
        all_dataset_images = glob(os.path.join(self.dataset_config.dataset_path, "202*", "*.png")) # CHECK: only channel 1
        plate_name_list = set(map(lambda x: x.rsplit("_", 1)[0], all_dataset_images))
        
        # additional check: all images named in the df must be present in the image directory
        if any(map(lambda x: x not in set(all_dataset_images), self._df[_CH.COL_IMG_NAME].to_list())):
            # TODO discard those instead of raising exception
            raise ValueError(f"There dataframe at {self._dataset_csv} contains image path references that do not exist")

        # group df for same plate (names)
        df = self._df.copy()
        df["plate_group"] = df[_CH.COL_IMG_NAME].str.extract(r'(.+)_')[0]
        df = df.groupby("plate_group").agg(list).reset_index()

        # find defects in each plate
        self.patches_with_defects = []
        for plate_name in plate_name_list:
            plate = SinglePlate(f"{plate_name}_1.png", f"{plate_name}_2.png")
            filtered_df = df[df[_CH.COL_IMG_NAME].apply(lambda x: all(item in x for item in plate.tolist()))]
            if filtered_df.empty: continue
            filtered_df = filtered_df.explode(list(self._df.columns), ignore_index=True).drop(columns=["plate_group"])
            plate.locate_defects(filtered_df, filt=list(SinglePlate.label_to_idx.keys()))
            self.patches_with_defects.extend(plate.patch_list_defects)

        Logger.instance().debug(f"There are {len(self.patches_with_defects)} patches that contain defects (any).")

        ## save for YOLO part
        if len(self.dataset_config.dataset_splits) < 3:
            raise ValueError(f"YOLO need train/val/test splits: set config to have three splits.")
        train_list, val_list, test_list = self._train_test_split(self.dataset_config.dataset_splits)
        ord_train_list = self._sort_by_plate_filename(train_list)
        ord_val_list = self._sort_by_plate_filename(val_list)
        ord_test_list = self._sort_by_plate_filename(test_list)

        # save yolo format patches here
        for idx, k in enumerate(ord_train_list):
            self.__save_yolo_format(idx, ord_train_list[k], "train", self.dataset_config.image_size)
        for idx, k in enumerate(ord_val_list):
            self.__save_yolo_format(idx, ord_val_list[k], "val", self.dataset_config.image_size)
        for idx, k in enumerate(ord_test_list):
            self.__save_yolo_format(idx, ord_test_list[k], "test", self.dataset_config.image_size)

        # TODO self.subsets_dict: SubsetsDict = self.split_dataset(self.dataset_config.dataset_splits)
        
        super().__init__()

    def __getitem__(self, idx):
        image_batch = self._load_batch_patch(self.patches_with_defects[idx])
        label_batch = self._load_batch_defect_labels(self.patches_with_defects[idx])
        
        return image_batch, label_batch
    
    def __len__(self):
        return len(self.patches_with_defects)

    def _load_batch_patch(self, patch: Patch) -> torch.Tensor:
        full_img_pil_1 = Image.open(patch.plate_paths.ch_1).convert("L")
        img_pil_1 = full_img_pil_1.crop((patch.start_w, patch.start_h, patch.start_w + patch.w, patch.start_h + patch.h))
        del full_img_pil_1
        
        full_img_pil_2 = Image.open(patch.plate_paths.ch_2).convert("L")
        img_pil_2 = full_img_pil_2.crop((patch.start_w, patch.start_h, patch.start_w + patch.w, patch.start_h + patch.h))
        del full_img_pil_2

        # rescale [0-255](int) to [0-1](float)
        img_1 = transforms.ToTensor()(img_pil_1)
        img_2 = transforms.ToTensor()(img_pil_2)

        img = torch.stack([img_1, img_2], dim=0).squeeze(1)

        # if use only one channel for test
        if self.TEST_ONE:
            del img
            img = img_1.clone()

        # normalize
        img = CustomDataset.normalize_or_identity(self.dataset_config)(img)

        return img # type: ignore
    
    def _load_batch_defect_labels(self, patch: Patch):
        # TODO FIX pytorch does not like variable length lists as labels
        return patch.defects

    def parse_csv(self, filt: Optional[List[str]]=None) -> pd.DataFrame:
        """Read bounding boxes csv (parser)
        
        Reads the bounding_boxes.csv file correctly:
            1. separator is ";",
            2. index_col is broken,
            3. remove NaN entries from image names and bboxes
            4. bboxes from float to int
            5. replace full path
        At the end, the df is returned ordered following the `filt` argument, if specified.

        Args:
            filt (Optional[List[str]]): list with classes of defects that must be accounted

        Returns:
            pd.DataFrame
        """
        
        df = pd.read_csv(self._dataset_csv, sep=";", index_col=False)

        # check if all the required columns exist
        col_values = [getattr(_CH, attr) for attr in dir(_CH) if attr.startswith("COL")]
        if not set(col_values) == set(df.columns):
            raise ValueError(f"Some columns are not present. Check:\ndf: {df.columns}\nyours: {col_values}")

        # remove wrong values (NaN)
        to_int_cols = [_CH.COL_BBOX_MIN_X, _CH.COL_BBOX_MAX_X, _CH.COL_BBOX_MIN_Y, _CH.COL_BBOX_MAX_Y]
        not_be_nan_cols = [_CH.COL_IMG_NAME] + to_int_cols
        df = df.dropna(subset=not_be_nan_cols)

        # bbox to int
        df[to_int_cols] = df[to_int_cols].astype(int)

        # replace path
        df[_CH.COL_IMG_NAME] = df[_CH.COL_IMG_NAME].apply(lambda x: os.path.join(self.dataset_config.dataset_path, f"{os.sep}".join(x.rsplit(os.sep, -1)[-2:])))

        # class filters
        if filt is not None:
            defect_classes = list(self.get_defect_class(df))
            selected_columns = set()
            for col in set(filt):
                # check if wrong names were inserted in the config.json file
                if col not in defect_classes:
                    Logger.instance().warning(f"No defect named {col}")
                else:    
                    selected_columns.add(col)

            df = df.loc[df[_CH.COL_CLASS_KEY].isin(selected_columns)]

        return self.order_by(df, filt)

    def get_defect_class(self, df: pd.DataFrame):
        return set(df[_CH.COL_CLASS_KEY].unique())
    
    def _train_test_split(self, ratios: List[float]) -> Tuple[List[Patch], List[Patch], List[Patch]]:
        # mandatory to have defects in train, so pass `self.patches_with_defects` or a subset containing defects
        filt = list(SinglePlate.idx_to_label.values())
        n_defect: dict = self.n_defect_per_class(self.patches_with_defects, filt=filt)
        
        ds_len = reduce(lambda a, b: a+b, list(n_defect.values()))
        splits = { k: [
                int(v * ratios[0]),
                int(v * ratios[1]),
                v-(int(v * ratios[0]) + int(v * ratios[1]))
            ] for k, v in n_defect.items() 
        }

        train_list_defect: List[Patch] = list()
        val_list_defect: List[Patch] = list()
        test_list_defect: List[Patch] = list()
        for k, v in splits.items():
            count_val = 0
            count_test = 0
            for patch in self.patches_with_defects:
                if any(map(lambda x: x.defect_class == k, patch.defects)):
                    if count_val < v[1]:
                        val_list_defect.append(patch)
                        count_val += 1
                    elif count_test < v[2]:
                        test_list_defect.append(patch)
                        count_test += 1
                    else:
                        train_list_defect.append(patch)

        return train_list_defect, val_list_defect, test_list_defect

    def n_defect_per_class(self, patches_w_defects: List[Patch], filt: Optional[List[str]]) -> dict:
        filter_patches = patches_w_defects.copy()
        cls_filt = set(self._df[_CH.COL_CLASS_KEY].tolist())

        # check if at least one element of the one specified (filt) is in the defect list for each pat
        if filt is not None:
            cls_filt = set(filt)
            filter_patches = list(filter(lambda x: len(set([d.defect_class for d in x.defects]) & (cls_filt)) > 0, patches_w_defects))
        
        # init defect dictionary
        numel_defect = { c: 0 for c in cls_filt }
        for patch in filter_patches:
            for d in patch.defects:
                for f in cls_filt:
                    if d.defect_class == f:
                        numel_defect[f] += 1

        return numel_defect

    def get_one_view_per_channel(self, df: pd.DataFrame, order_by: Optional[List[str]]=None) -> Optional[pd.DataFrame]:
        """Return one view for each channel

        Sometimes multiple views are linked to the same image channel. REMOVE the ones that exceeds.
        This method also orders the outcome df, if specified.

        Args:
            df (pd.DataFrame): may apply on differently filtered dataframes, not necessarily the main of this class
            order_by (Optional[list[str]]): if you want to return a csv with a different order.

        Returns:
            Optional[pd.DataFrame]
        """
        
        # locate the 3rd, 4th, ... occurrence (view) for a defect and remove from db
        # https://stackoverflow.com/a/70168878
        out = df[df.groupby(_CH.COL_ID_DEFECT).cumcount().le(1)]

        return self.order_by(out, order_by)

    @staticmethod
    def order_by(df: pd.DataFrame, order_by: Optional[List[str]] = None) -> pd.DataFrame:
        """Order by one/more different column/s
    
        Args:
            df (pd.DataFrame): the input csv
            order_by (lis[str]): the selected columns

        Returns:
            pd.DataFame
            
        """

        if order_by is None or len(order_by) == 0:
            Logger.instance().debug("Order by: nothing to do!")
            return df
        
        if any(list(map(lambda x: x not in df.columns, order_by))):
            Logger.instance().error(f"sort by: {order_by}")
            raise ValueError("Check config file: some columns may not be present")
            
        Logger.instance().debug(f"sort by: {order_by}")
        return df.sort_values(order_by, ascending=[True] * len(order_by))
    
    @staticmethod
    def __save_yolo_format(plate_idx: int, plate_patch_list: List[Patch], split: str, img_size: int=640):
        parent_plate_ch1 = plate_patch_list[0].plate_paths.ch_1
        parent_plate_ch2 = plate_patch_list[0].plate_paths.ch_2

        image_folder_path = os.path.join(os.getcwd(), "output", split, "images")
        label_folder_path = os.path.join(os.getcwd(), "output", split, "labels")

        if not os.path.exists(image_folder_path): os.makedirs(image_folder_path)
        if not os.path.exists(label_folder_path): os.makedirs(label_folder_path)

        # save image file (.png)
        img_1 = Image.open(parent_plate_ch1).convert("L")
        img_2 = Image.open(parent_plate_ch2).convert("L")

        for patch_idx, patch in enumerate(plate_patch_list):
            patch_basename = f"plate_{plate_idx}_patch_{patch_idx}"
            # check if a patch has defects (safe check)
            if not patch.defects:
                Logger.instance().warning(f"No defects in selected patch")
                continue

            # filter patches that have specific defects (SinglePlate class attributes)
            filtered_defects = []
            for d in patch.defects:
                if d.defect_class in list(SinglePlate.label_to_idx.keys()):
                    filtered_defects.append(d)

            # if a patch has any of those specific defects, then save the image patch
            if len(filtered_defects) > 0:
                patch_filename = os.path.join(image_folder_path, f"{patch_basename}.png")
                crop_1 = img_1.crop((patch.start_w, patch.start_h, patch.start_w + patch.w, patch.start_h + patch.h))
                crop_2 = img_2.crop((patch.start_w, patch.start_h, patch.start_w + patch.w, patch.start_h + patch.h))
                
                ## alpha channel
                img_merge = Image.new("LA", crop_1.size)
                img_merge.paste(crop_1, (0, 0))
                img_merge.paste(crop_2, (0, 0), crop_2)
                
                ## RGB
                # img_merge = Image.merge("RGB", (crop_1, crop_2, crop_2))
                
                (img_merge.resize((img_size,img_size), resample=Image.BILINEAR)).save(patch_filename)
        
                # writing annotations
                with open(os.path.join(label_folder_path, f"{patch_basename}.txt"), "a") as f:
                    for didx, defect in enumerate(filtered_defects):
                        x = float((defect.max_x + defect.min_x) / 2) / float(SinglePlate.PATCH_SIZE)
                        y = float((defect.max_y + defect.min_y) / 2) / float(SinglePlate.PATCH_SIZE)
                        w = float((defect.max_x - defect.min_x) / float(SinglePlate.PATCH_SIZE))
                        h = float((defect.max_y - defect.min_y) / float(SinglePlate.PATCH_SIZE))
                        line = f"{SinglePlate.label_to_idx[defect.defect_class]} {x} {y} {w} {h}"
                        if didx == 0:
                            f.writelines(line)
                        else:
                            f.writelines(f"\n{line}")

    @staticmethod
    def _sort_by_plate_filename(patch_list: List[Patch]):
        objs = patch_list.copy()
        objs.sort(key=lambda x: x.plate_paths.ch_1)
        return { key: list(group) for key, group in groupby(objs, key=lambda x: x.plate_paths.ch_1) }

    @staticmethod
    def compute_mean_std(dataset: Union[CustomDataset, TorchDataset]) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = CustomDataset.compute_mean_std(dataset)
        return mean, std