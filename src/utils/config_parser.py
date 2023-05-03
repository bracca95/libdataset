# inspired by quicktype.io

import os
import sys
import json

from dataclasses import dataclass
from typing import Optional, Union, List, Any, Callable, Iterable, Type, cast

from src.utils.tools import Tools, Logger
from config.consts import T
from config.consts import General as _CG
from config.consts import ConfigConst as _CC

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
class Config:
    dataset_path: str = _CG.DEFAULT_STR
    dataset_type: str = _CG.DEFAULT_STR
    batch_size: int = _CG.DEFAULT_INT
    epochs: int = _CG.DEFAULT_INT
    crop_size: int = _CG.DEFAULT_INT
    image_size: Optional[int] = None
    augment_online: Optional[List[str]] = None
    augment_offline: Optional[List[str]] = None
    dataset_mean: Optional[List[float]] = None
    dataset_std: Optional[List[float]] = None

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
            batch_size = from_int(obj.get(_CC.CONFIG_BATCH_SIZE))
            epochs = from_int(obj.get(_CC.CONFIG_EPOCHS))
            crop_size = from_int(obj.get(_CC.CONFIG_CROP_SIZE))
            image_size = from_union([from_none, from_int], obj.get(_CC.CONFIG_IMAGE_SIZE))
            augment_online = from_union([lambda x: from_list(from_str, x), from_none], obj.get(_CC.CONFIG_AUGMENT_ONLINE))
            augment_offline = from_union([lambda x: from_list(from_str, x), from_none], obj.get(_CC.CONFIG_AUGMENT_OFFLINE))
            dataset_mean = from_union([lambda x: from_list(from_float, x), from_none], obj.get(_CC.CONFIG_DATASET_MEAN))
            dataset_std = from_union([lambda x: from_list(from_float, x), from_none], obj.get(_CC.CONFIG_DATASET_STD))
        except TypeError as te:
            Logger.instance().critical(te.args)
            sys.exit(-1)

        if augment_online is not None:
            if len(augment_online) == 0:
                augment_online = None

        if augment_offline is not None:
            if len(augment_offline) == 0:
                augment_offline = None
        
        Logger.instance().info(f"Config deserialized: " +
            f"dataset_path: {dataset_path}, dataset_type: {dataset_type}, augment_online: {augment_online}, " +
            f"augment_offline: {augment_offline}, batch_size {batch_size}, epochs: {epochs}, " +
            f"dataset mean: {dataset_mean}, dataset_std: {dataset_std}, crop_size: {crop_size}, image_size: {image_size}")
        
        return Config(dataset_path, dataset_type, batch_size, epochs, crop_size, image_size, augment_online, augment_offline, dataset_mean, dataset_std)

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
        result[_CC.CONFIG_BATCH_SIZE] = from_int(self.batch_size)
        result[_CC.CONFIG_EPOCHS] = from_int(self.epochs)
        result[_CC.CONFIG_CROP_SIZE] = from_int(self.crop_size)
        result[_CC.CONFIG_IMAGE_SIZE] = from_union([from_none, from_int], self.image_size)
        result[_CC.CONFIG_AUGMENT_ONLINE] = from_union([lambda x: from_list(from_str, x), from_none], self.augment_online)
        result[_CC.CONFIG_AUGMENT_OFFLINE] = from_union([lambda x: from_list(from_str, x), from_none], self.augment_offline)
        result[_CC.CONFIG_DATASET_MEAN] = from_union([lambda x: from_list(from_float, x), from_none], self.dataset_mean)
        result[_CC.CONFIG_DATASET_STD] = from_union([lambda x: from_list(from_float, x), from_none], self.dataset_std)

        with open(os.path.join(dire, filename), "w") as f:
            json_dict = json.dumps(result, indent=4)
            f.write(json_dict)

        Logger.instance().info("Config serialized")


def config_to_json(x: Config) -> Any:
    return to_class(Config, x)
