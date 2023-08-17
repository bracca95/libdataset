from .dataset import CustomDataset
from .defectviews import GlassOpt, GlassOptBckg, GlassOptTricky, BubblePoint, QPlusV1, QPlusV2
from ..utils.config_parser import Config
from ..utils.tools import Logger


class DatasetBuilder:

    @staticmethod
    def load_dataset(config: Config) -> CustomDataset:
        if config.dataset_type == "opt6":
            Logger.instance().info("Loading dataset GlassOpt (type CustomDataset)")
            return GlassOpt(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size, config.dataset_splits)
        elif config.dataset_type == "opt_bckg":
            Logger.instance().info("Loading dataset GlassOptBckg (type GlassOpt)")
            return GlassOptBckg(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size, config.dataset_splits)
        elif config.dataset_type == "opt_tricky":
            Logger.instance().info("Loading dataset GlassOptTricky (type GlassOpt)")
            return GlassOptTricky(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size, config.dataset_splits)
        elif config.dataset_type == "qplusv1":
            Logger.instance().info("Loading dataset QPlusV1 (type GlassOpt)")
            return QPlusV1(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size, config.dataset_splits)
        elif config.dataset_type == "qplusv2":
            Logger.instance().info("Loading dataset QPlusV2 (type GlassOpt)")
            return QPlusV2(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size, config.dataset_splits)
        elif config.dataset_type == "binary":
            Logger.instance().info("Loading dataset BubblePoint (type GlassOpt)")
            return BubblePoint(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size, config.image_size, config.dataset_splits)
        else:
            raise ValueError("values allowed: {`opt6`, `opt_bckg`, `binary`, `qplusv1`, `qplusv2`} for dataset_type")
