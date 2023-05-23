# about TypedDict https://stackoverflow.com/a/64938100
import torch
from torch.utils.data import Subset
from typing import Optional, TypeVar, TypedDict

T = TypeVar("T")
SubsetsDict = TypedDict("SubsetsDict", { "train": Subset, "val": Optional[Subset], "test": Subset })


class General:
    DEFAULT_BOOL = False
    DEFAULT_INT = 0
    DEFAULT_STR = ""
    DEFAULT_LIST = []
    DEFAULT_SUBSETS = ["train", "val", "test"]
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    EPS = 0.001


class ConfigConst:
    CONFIG_DATASET_PATH = "dataset_path"
    CONFIG_DATASET_TYPE = "dataset_type"
    CONFIG_DATASET_SPLITS = "dataset_splits"
    CONFIG_AUGMENT_ONLINE = "augment_online"
    CONFIG_AUGMENT_OFFLINE = "augment_offline"
    CONFIG_BATCH_SIZE = "batch_size"
    CONFIG_EPOCHS = "epochs"
    CONFIG_CROP_SIZE = "crop_size"
    CONFIG_IMAGE_SIZE = "image_size"
    CONFIG_DATASET_MEAN = "dataset_mean"
    CONFIG_DATASET_STD = "dataset_std"
    CONFIG_FSL = "fsl"


class FSLConsts:
    FSL_EPISODES = "episodes"
    FSL_TRAIN_N_WAY = "train_n_way"
    FSL_TRAIN_K_SHOT_S = "train_k_shot_s"
    FSL_TRAIN_K_SHOT_Q = "train_k_shot_q"
    FSL_TEST_N_WAY = "test_n_way"
    FSL_TEST_K_SHOT_S = "test_k_shot_s"
    FSL_TEST_K_SHOT_Q = "test_k_shot_q"