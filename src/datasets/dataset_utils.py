from .dataset import DatasetWrapper
from .custom.glass_plate import GlassPlate, GlassPlateTrainYolo, GlassPlateTestYolo
from .custom.defectviews import GlassOpt, GlassOptBckg, GlassOptTricky, GlassOptDouble, GlassOptDoubleInference, BubblePoint, QPlusV1, QPlusV2, QPlusDouble
from .torch.omniglot import OmniglotWrapper
from .torch.celeba import CelebaWrapper
from .imagefolder.dataset_fsl import FewShotDataset
from ..utils.config_parser import DatasetConfig
from ..utils.tools import Logger


class DatasetBuilder:

    @staticmethod
    def load_dataset(dataset_config: DatasetConfig) -> DatasetWrapper:
        if dataset_config.dataset_type == "opt6":
            Logger.instance().info("Loading dataset GlassOpt (type DatasetWrapper)")
            return GlassOpt(dataset_config)
        elif dataset_config.dataset_type == "opt_bckg":
            Logger.instance().info("Loading dataset GlassOptBckg (type GlassOpt)")
            return GlassOptBckg(dataset_config)
        elif dataset_config.dataset_type == "opt_tricky":
            Logger.instance().info("Loading dataset GlassOptTricky (type GlassOpt)")
            return GlassOptTricky(dataset_config)
        elif dataset_config.dataset_type == "opt_double":
            Logger.instance().info("Loading dataset GlassOptDouble (type GlassOpt)")
            return GlassOptDouble(dataset_config)
        elif dataset_config.dataset_type == "opt_double_inference":
            Logger.instance().info("Loading dataset GlassOptDoubleInference (type GlassOptDouble)")
            return GlassOptDoubleInference(dataset_config)
        elif dataset_config.dataset_type == "qplus_double":
            Logger.instance().info("Loading dataset QPlusDouble (type GlassOptDouble)")
            return QPlusDouble(dataset_config)
        elif dataset_config.dataset_type == "qplusv1":
            Logger.instance().info("Loading dataset QPlusV1 (type GlassOpt)")
            return QPlusV1(dataset_config)
        elif dataset_config.dataset_type == "qplusv2":
            Logger.instance().info("Loading dataset QPlusV2 (type GlassOpt)")
            return QPlusV2(dataset_config)
        elif dataset_config.dataset_type == "binary":
            Logger.instance().info("Loading dataset BubblePoint (type GlassOpt)")
            return BubblePoint(dataset_config)
        elif dataset_config.dataset_type == "omniglot":
            Logger.instance().info("Loading dataset Omniglot (type DatasetWrapper)")
            return OmniglotWrapper(dataset_config)
        elif dataset_config.dataset_type == "celeba":
            Logger.instance().info("Loading dataset Omniglot (type DatasetWrapper)")
            return CelebaWrapper(dataset_config)
        elif dataset_config.dataset_type in ("miniimagenet", "cub", "cifar_fs"):
            Logger.instance().info("Loading dataset Mini Imagenet (type DatasetWrapper)")
            return FewShotDataset(dataset_config)
        else:
            raise ValueError(
                "values allowed: {`opt6`, `opt_bckg`, `opt_double`, `opt_double_inference`, `binary`, `qplusv1`, " +
                "`qplusv2`, `qplus_double`, `omniglot`, `miniimagenet`, `opt_yolo_train`, `opt_yolo_test`, `cub`, " +
                "`cifar_fs`, `celeba`} for dataset_type"
            )
        
class YoloDatasetBuilder:

    @staticmethod
    def load_dataset(dataset_config: DatasetConfig) -> GlassPlate:
        if dataset_config.dataset_type == "opt_yolo_train":
            Logger.instance().info("Loading dataset GlassPlateTrainYolo (type GlassPlate)")
            return GlassPlateTrainYolo(dataset_config)
        elif dataset_config.dataset_type == "opt_yolo_test":
            Logger.instance().info("Loading dataset GlassPlateTestYolo (type GlassPlate)")
            return GlassPlateTestYolo(dataset_config)
        else:
            raise ValueError(
            "values allowed: {`opt6`, `opt_bckg`, `opt_double` `binary`, `qplusv1`, `qplusv2`, " +
            "`qplus_double` `omniglot`, `miniimagenet`, `opt_yolo_train`, `opt_yolo_test`} for dataset_type"
        )
