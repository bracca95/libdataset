# inspired by quicktype.io

import os
import sys
import json

from functools import reduce
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any, Callable, Iterable, Type, cast

from src.utils.tools import Tools, Logger
from config.consts import T
from config.consts import General as _CG
from config.consts import ConfigConst as _CC
from config.consts import FSLConsts as _CFSL

def from_bool(x: Any) -> bool:
    Tools.check_instance(x, bool)
    return x

def from_int(x: Any) -> int:
    Tools.check_instance(x, int)
    return x

def from_float(x: Any) -> float:
    Tools.check_instance(x, float)
    return x

def from_str(x: Any) -> str:
    Tools.check_instance(x, str)
    return x

def from_none(x: Any) -> Any:
    Tools.check_instance(x, None)
    return x

def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    Tools.check_instance(x, list)
    return [f(y) for y in x]

def from_union(fs: Iterable[Any], x: Any):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    raise TypeError(f"{x} should be one out of {[type(f.__name__) for f in fs]}")


def to_class(c: Type[T], x: Any) -> dict:
    Tools.check_instance(x, c)
    return cast(Any, x).serialize()


@dataclass
class Fsl:
    episodes: int
    train_n_way: int
    train_k_shot_s: int
    train_k_shot_q: int
    test_n_way: int
    test_k_shot_s: int
    test_k_shot_q: int

    @classmethod
    def deserialize(cls, obj: Any) -> 'Fsl':
        
        try:
            Tools.check_instance(obj, dict)
            episodes = from_int(obj.get(_CFSL.FSL_EPISODES))
            train_n_way = from_int(obj.get(_CFSL.FSL_TRAIN_N_WAY))
            train_k_shot_s = from_int(obj.get(_CFSL.FSL_TRAIN_K_SHOT_S))
            train_k_shot_q = from_int(obj.get(_CFSL.FSL_TRAIN_K_SHOT_Q))
            test_n_way = from_int(obj.get(_CFSL.FSL_TEST_N_WAY))
            test_k_shot_s = from_int(obj.get(_CFSL.FSL_TEST_K_SHOT_S))
            test_k_shot_q = from_int(obj.get(_CFSL.FSL_TEST_K_SHOT_Q))
            
        except TypeError as te:
            Logger.instance().error(te.args)
            sys.exit(-1)

        Logger.instance().debug(f"episodes: {episodes}, train_n_way: {train_n_way}, train_k_shot_s: {train_k_shot_s}, " +
                                f"train_k_shot_q: {train_k_shot_q}, test_n_way: {test_n_way}, " +
                                f"test_k_shot_s: {test_k_shot_s}, test_k_shot_q: {test_k_shot_q}")
        return Fsl(episodes, train_n_way, train_k_shot_s, train_k_shot_q, test_n_way, test_k_shot_s, test_k_shot_q)

    def serialize(self) -> dict:
        result: dict = {}
        
        result[_CFSL.FSL_EPISODES] = from_int(self.episodes)
        result[_CFSL.FSL_TRAIN_N_WAY] = from_int(self.train_n_way)
        result[_CFSL.FSL_TRAIN_K_SHOT_S] = from_int(self.train_k_shot_s)
        result[_CFSL.FSL_TRAIN_K_SHOT_Q] = from_int(self.train_k_shot_q)
        result[_CFSL.FSL_TEST_N_WAY] = from_int(self.test_n_way)
        result[_CFSL.FSL_TEST_K_SHOT_S] = from_int(self.test_k_shot_s)
        result[_CFSL.FSL_TEST_K_SHOT_Q] = from_int(self.test_k_shot_q)

        Logger.instance().info(f"ObjectList serialized: {result}")
        return result


@dataclass
class Config:
    dataset_path: str = _CG.DEFAULT_STR
    dataset_type: str = _CG.DEFAULT_STR
    dataset_splits: List[float] = field(default_factory=list)
    batch_size: int = _CG.DEFAULT_INT
    epochs: int = _CG.DEFAULT_INT
    crop_size: int = _CG.DEFAULT_INT
    image_size: Optional[int] = None
    augment_online: Optional[List[str]] = None
    augment_offline: Optional[List[str]] = None
    dataset_mean: Optional[List[float]] = None
    dataset_std: Optional[List[float]] = None
    fsl: Optional[Fsl] = None

    @classmethod
    def deserialize(cls, str_path: str) -> 'Config':
        obj = Tools.read_json(str_path)
        
        try:
            dataset_path = Tools.validate_path(obj.get(_CC.CONFIG_DATASET_PATH))
        except (FileNotFoundError, ValueError) as fnf:
            Logger.instance().error(fnf.args)
            dataset_path = input("insert dataset path: ")
            while not os.path.exists(dataset_path):
                dataset_path = input("insert dataset path: ")
                
        try:
            dataset_type = from_str(obj.get(_CC.CONFIG_DATASET_TYPE))
            dataset_splits = from_list(lambda x: from_float(x), obj.get(_CC.CONFIG_DATASET_SPLITS))
            batch_size = from_int(obj.get(_CC.CONFIG_BATCH_SIZE))
            epochs = from_int(obj.get(_CC.CONFIG_EPOCHS))
            crop_size = from_int(obj.get(_CC.CONFIG_CROP_SIZE))
            image_size = from_union([from_none, from_int], obj.get(_CC.CONFIG_IMAGE_SIZE))
            augment_online = from_union([lambda x: from_list(from_str, x), from_none], obj.get(_CC.CONFIG_AUGMENT_ONLINE))
            augment_offline = from_union([lambda x: from_list(from_str, x), from_none], obj.get(_CC.CONFIG_AUGMENT_OFFLINE))
            dataset_mean = from_union([lambda x: from_list(from_float, x), from_none], obj.get(_CC.CONFIG_DATASET_MEAN))
            dataset_std = from_union([lambda x: from_list(from_float, x), from_none], obj.get(_CC.CONFIG_DATASET_STD))
            fsl = from_union([Fsl.deserialize, from_none], obj.get(_CC.CONFIG_FSL))
        except TypeError as te:
            Logger.instance().critical(te.args)
            sys.exit(-1)

        if augment_online is not None:
            if len(augment_online) == 0:
                augment_online = None

        if augment_offline is not None:
            if len(augment_offline) == 0:
                augment_offline = None

        if len(dataset_splits) not in (1, 3):
            raise ValueError("`dataset_splits` must have len == 1 (train/test) or len == 3 (train/val/test)")
        
        if len(dataset_splits) == 3:
            if 1.0-_CG.EPS < reduce(lambda a,b: a+b, dataset_splits) < 1.0+_CG.EPS:
                pass
            else:
                raise ValueError("the sum for dataset_splits must be 1")
        
        Logger.instance().info(f"Config deserialized: " +
            f"dataset_path: {dataset_path}, dataset_type: {dataset_type}, dataset_splits: {dataset_splits}, " +
            f"augment_online: {augment_online}, augment_offline: {augment_offline}, batch_size {batch_size}, epochs: {epochs}, " +
            f"dataset mean: {dataset_mean}, dataset_std: {dataset_std}, crop_size: {crop_size}, image_size: {image_size}, " +
            f"fsl: {fsl}")
        
        return Config(dataset_path, dataset_type, dataset_splits, batch_size, epochs, crop_size, image_size, augment_online, augment_offline, dataset_mean, dataset_std, fsl)

    def serialize(self, directory: str, filename: str):
        result: dict = {}
        dire = None

        try:
            dire = Tools.validate_path(directory)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"{fnf.args}")
            sys.exit(-1)
        
        # if you do not want to write null values, add a field to result if and only if self.field is not None
        result[_CC.CONFIG_DATASET_PATH] = from_str(self.dataset_path)
        result[_CC.CONFIG_DATASET_TYPE] = from_str(self.dataset_type)
        result[_CC.CONFIG_DATASET_SPLITS] = from_list(lambda x: from_float(x), self.dataset_splits)
        result[_CC.CONFIG_BATCH_SIZE] = from_int(self.batch_size)
        result[_CC.CONFIG_EPOCHS] = from_int(self.epochs)
        result[_CC.CONFIG_CROP_SIZE] = from_int(self.crop_size)
        result[_CC.CONFIG_IMAGE_SIZE] = from_union([from_none, from_int], self.image_size)
        result[_CC.CONFIG_AUGMENT_ONLINE] = from_union([lambda x: from_list(from_str, x), from_none], self.augment_online)
        result[_CC.CONFIG_AUGMENT_OFFLINE] = from_union([lambda x: from_list(from_str, x), from_none], self.augment_offline)
        result[_CC.CONFIG_DATASET_MEAN] = from_union([lambda x: from_list(from_float, x), from_none], self.dataset_mean)
        result[_CC.CONFIG_DATASET_STD] = from_union([lambda x: from_list(from_float, x), from_none], self.dataset_std)
        result[_CC.CONFIG_FSL] = from_union([lambda x: to_class(Fsl, x), from_none], self.fsl)

        with open(os.path.join(dire, filename), "w") as f:
            json_dict = json.dumps(result, indent=4)
            f.write(json_dict)

        Logger.instance().info("Config serialized")


def config_to_json(x: Config) -> Any:
    return to_class(Config, x)
