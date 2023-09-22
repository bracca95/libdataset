import os
import pandas as pd

from PIL import Image
from glob import glob
from typing import Optional, List, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset

from ..imgproc import Processing
from ..utils.tools import Tools, Logger
from ..utils.config_parser import DatasetConfig
from ...config.consts import SubsetsDict
from ...config.consts import General as _GC


@dataclass
class PlateImagePaths:
    ch_1: str
    ch_2: str

    def tolist(self):
        return [self.ch_1, self.ch_2]


@dataclass
class Patch:
    plate_paths: PlateImagePaths
    start_w: int
    start_h: int
    w: int
    h: int


class GlassPlate(Dataset):

    COL_PIECE_ID = "#id_piece"
    COL_PIECE_CODE = "#piece_code"
    COL_IMG_NAME = "#img_name"
    COL_PATH_TO_RUNLIST_MASK = "#path_to_runlist_mask"
    COL_CLASS_KEY = "#class_key"
    COL_BBOX_MIN_X = "#bbox.min.x"
    COL_BBOX_MIN_Y = "#bbox.min.y"
    COL_BBOX_MAX_X = "#bbox.max.x"
    COL_BBOX_MAX_Y = "#bbox.max.y"
    COL_ID_DEFECT = "#id_defect"
    COL_ID_CAM = "#id_cam"
    COL_ID_CHANNEL = "#id_channel"
    COL_SOURCE = "#source"
    COL_DEFECT_IS_VALIDATED = "#defect_is_validated"
    COL_SEVERITY = "#severity"
    COL_FAREA = "#fArea"
    COL_FAVGINT = "#fAvgInt"
    COL_FELONG = "#fElong"
    COL_VSCORE = "#vScore"

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config

        # read csv by filtering classes and returning
        self._dataset_csv = "/media/lorenzo/M/datasets/dataset_opt/2.2_dataset_opt/bounding_boxes.csv"
        self._df = self.parse_csv(filt=None)

        all_dataset_images = glob(os.path.join(self.dataset_config.dataset_path, "202*", "*.png")) # CHECK: only channel 1
        plate_names = set(list(map(lambda x: x.rsplit("_", 1)[0], all_dataset_images)))
        self.plate_list = list(map(lambda x: PlateImagePaths(f"{x}_1.png", f"{x}_2.png"), plate_names))

        self.patch_list: List[Patch] = self.create_patch_list(416, 416, 400)
        
        # additional check: all images named in the df must be present in the image directory
        if any(map(lambda x: x not in set(all_dataset_images), self._df[self.COL_IMG_NAME].to_list())):
            # TODO discard those instead of raising exception
            raise ValueError(f"There dataframe at {self._dataset_csv} contains image path references that do not exist")
        
        grouped = self._df.groupby(self.COL_ID_DEFECT)
        for gname, gdata in grouped:
            for patch in self.patch_list:
                if not gdata[self.COL_IMG_NAME].tolist() == patch.plate_paths.tolist(): continue
                
        
        super().__init__()

        
        # group df by id_defect and create a function load_image that reads the whole image, splits into patches (416x416)
        # then computes the relative shift for bounding boxes and this info will create a 
        # PlatePatch object(img_portion, ref_sys, img_name)

    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    def __len__(self):
        pass

    def create_patch_list(self, patch_w: int, patch_h: int, stride: int):
        patch_list: List[Patch] = list()
        for plate in self.plate_list:
            img_w, img_h = Image.open(plate.ch_1).size
            patch_list.extend(self.sliding_window(plate, img_w, img_h, patch_w, patch_h, stride))
    
        return patch_list

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
        col_values = [getattr(self, attr) for attr in dir(self) if attr.startswith("COL")]
        if not set(col_values) == set(df.columns):
            raise ValueError(f"Some columns are not present. Check:\ndf: {df.columns}\nyours: {col_values}")

        # remove wrong values (NaN)
        to_int_cols = [self.COL_BBOX_MIN_X, self.COL_BBOX_MAX_X, self.COL_BBOX_MIN_Y, self.COL_BBOX_MAX_Y]
        not_be_nan_cols = [self.COL_IMG_NAME] + to_int_cols
        df = df.dropna(subset=not_be_nan_cols)

        # bbox to int
        df[to_int_cols] = df[to_int_cols].astype(int)

        # replace path
        df[self.COL_IMG_NAME] = df[self.COL_IMG_NAME].apply(lambda x: os.path.join(self.dataset_config.dataset_path, f"{os.sep}".join(x.rsplit(os.sep, -1)[-2:])))

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

            df = df.loc[df[self.COL_CLASS_KEY].isin(selected_columns)]

        return self.order_by(df, filt)

    def get_defect_class(self, df: pd.DataFrame):
        return set(df[self.COL_CLASS_KEY].unique())

    def n_defect_per_class(self, df: pd.DataFrame, classes: Optional[List[str]]):
        """Print the number of defects for each class.

        The classes specified in the config/config.json file are searched in the dataframe and, for each of them, the
        number of defects is returned. If no classes are provided, all the defect classes in the dataframe are taken 
        into account.

        Args:
            df (pd.DataFrame): may apply on differently filtered dataframes, not necessarily the main of this class
            classes (Optional[List[str]]): list of defect defined in the config/config.json file.
            
        """

        defect_classes = list(self.get_defect_class(df))
        selected_classes = set()

        if classes is None or len(classes) == 0:
            # consider all the classes found in df
            selected_classes = set(defect_classes).copy()
            Logger.instance().info(f"defects: \n{df[self.COL_CLASS_KEY].value_counts()}")
        else:
            # check if wrong names were inserted in the config.json file
            req_classes = set(classes)
            for col in req_classes:
                if col not in defect_classes:
                    Logger.instance().warning(f"No defect named {col}")
                else:    
                    selected_classes.add(col)

            Logger.instance().info("defects: \n" +\
                f"{df.loc[df[self.COL_CLASS_KEY].isin(selected_classes), self.COL_CLASS_KEY].value_counts()}"
            )

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
        out = df[df.groupby(self.COL_ID_DEFECT).cumcount().le(1)]

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
    def sliding_window(plate: PlateImagePaths, img_w: int, img_h: int, patch_w: int, patch_h: int, stride: int) -> List[Patch]:
        patch_list: List[Patch] = list()
        for i in range(0, img_h - patch_h + 1, stride):
            for j in range(0, img_w - patch_w + 1, stride):
                patch = Patch(plate, j, i, patch_w, patch_h)
                #patch = image_tensor[:, i:i + patch_size[0], j:j + patch_size[1]]
                patch_list.append(patch)

        return patch_list

    # @staticmethod
    # def read_img():
    #     # sel is patch_list[len(path_list) // 2], the selected path
    #     img.crop((sel.start_x, sel.start_y, sel.start_x+416, sel.start_y+416)).save(os.path.join(os.getcwd(), "output", "patch.png"))