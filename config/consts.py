# about TypedDict https://stackoverflow.com/a/64938100
import torch
from typing import TypeVar, TypedDict

T = TypeVar("T")
PlatePathsDict = TypedDict("PlatePathsDict", { "ch_1": str, "ch_2": str })


class CustomDatasetConsts:
    EpisodicCoco = 11111
    Fungi = 22222


class General:
    DEFAULT_BOOL = False
    DEFAULT_INT = 0
    DEFAULT_STR = ""
    DEFAULT_LIST = []
    DEFAULT_SUBSETS = ["train", "val", "test"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPS = 0.001


class DatasetConst:
    CONFIG_DATASET_PATH = "dataset_path"
    CONFIG_DATASET_TYPE = "dataset_type"
    CONFIG_DATASET_ID = "dataset_id"
    CONFIG_DATASET_SPLITS = "dataset_splits"
    CONFIG_NORMALIZE = "normalize"
    CONFIG_AUGMENT_ONLINE = "augment_online"
    CONFIG_AUGMENT_OFFLINE = "augment_offline"
    CONFIG_AUGMENT_TIMES = "augment_times"
    CONFIG_CROP_SIZE = "crop_size"
    CONFIG_IMAGE_SIZE = "image_size"
    CONFIG_DATASET_MEAN = "dataset_mean"
    CONFIG_DATASET_STD = "dataset_std"