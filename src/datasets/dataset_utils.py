from typing import Union
from torch.utils.data import Dataset

from .dataset import DatasetWrapper
from .fsl.omniglot import OmniglotWrapper
from .fsl.episodic_imagenet import EpisodicImagenet, EpisodicImagenetValCifar, EpisodicImagenetValCub, EpisodicImagenetValAircraft
from .fsl.episodic_imagenet1k import EpisodicImagenet1k
from .fsl.episodic_coco import EpisodicCoco
from .fsl.miniimagenet import MiniImagenet
from .fsl.cifar import CifarFs
from .fsl.cub import Cub
from .fsl.dtd import Dtd
from .fsl.pacs import PacsObject, PacsDomain
from .fsl.fungi import Fungi
from .fsl.aircraft import Aircraft
from .fsl.meta_inat import Metainat
from .fsl.meta_test import CropDiseases, EuroSat, Isic
from .fsl.meta_album import MetaAlbum
from .fsl.full_meta_album import FullMetaAlbum
from .fsl.wikiart import WikiArtArtist, WikiArtGenre, WikiArtStyle
from .torch.celeba import CelebaWrapper
from .classification.meta_album_csl import MetaAlbumCls
from ..utils.config_parser import DatasetConfig
from ..utils.tools import Logger


class DatasetBuilder:

    @staticmethod
    def load_dataset(dataset_config: DatasetConfig) -> Union[DatasetWrapper, Dataset]:
        if dataset_config.dataset_type == "omniglot":
            Logger.instance().debug("Loading dataset Omniglot (type FewShotDataset)")
            return OmniglotWrapper(dataset_config)
        elif dataset_config.dataset_type == "episodic_imagenet":
            Logger.instance().debug("Loading dataset EpisodicImagenet (type FewShotDataset)")
            return EpisodicImagenet(dataset_config)
        elif dataset_config.dataset_type == "episodic_imagenet_val_cifar":
            Logger.instance().debug("Loading dataset EpisodicImagenetValCifar (type FewShotDataset)")
            return EpisodicImagenetValCifar(dataset_config)
        elif dataset_config.dataset_type == "episodic_imagenet_val_cub":
            Logger.instance().debug("Loading dataset EpisodicImagenetValCub (type FewShotDataset)")
            return EpisodicImagenetValCub(dataset_config)
        elif dataset_config.dataset_type == "episodic_imagenet_val_aircraft":
            Logger.instance().debug("Loading dataset EpisodicImagenetValAircraft (type FewShotDataset)")
            return EpisodicImagenetValAircraft(dataset_config)
        elif dataset_config.dataset_type == "episodic_imagenet1k":
            Logger.instance().debug("Loading dataset EpisodicImagenet1k (type FewShotDataset)")
            return EpisodicImagenet1k(dataset_config)
        elif dataset_config.dataset_type == "episodic_coco":
            Logger.instance().debug("Loading dataset EpisodicCoco (type FewShotDataset)")
            return EpisodicCoco(dataset_config)
        elif dataset_config.dataset_type == "miniimagenet":
            Logger.instance().debug("Loading dataset Mini Imagenet (type FewShotDataset)")
            return MiniImagenet(dataset_config)
        elif dataset_config.dataset_type == "cifar_fs":
            Logger.instance().debug("Loading dataset CIFAR-FS (type FewShotDataset)")
            return CifarFs(dataset_config)
        elif dataset_config.dataset_type == "cub":
            Logger.instance().debug("Loading dataset CUB (type FewShotDataset)")
            return Cub(dataset_config)
        elif dataset_config.dataset_type == "fungi":
            Logger.instance().debug("Loading dataset Fungi (type FewShotDataset)")
            return Fungi(dataset_config)
        elif dataset_config.dataset_type == "aircraft":
            Logger.instance().debug("Loading dataset Aircraft (type FewShotDataset)")
            return Aircraft(dataset_config)
        elif dataset_config.dataset_type == "meta_inat":
            Logger.instance().debug("Loading dataset Meta-iNat (type FewShotDataset)")
            return Metainat(dataset_config)
        elif dataset_config.dataset_type == "meta_album":
            Logger.instance().debug("Loading dataset MetaAlbum (type FewShotDataset)")
            return MetaAlbum(dataset_config)
        elif dataset_config.dataset_type == "full_meta_album":
            Logger.instance().debug("Loading dataset FullMetaAlbum (type FullMetalJacket)")
            return FullMetaAlbum(dataset_config)
        elif dataset_config.dataset_type == "dtd":
            Logger.instance().debug("Loading dataset Dtd (type FewShotDataset)")
            return Dtd(dataset_config)
        elif dataset_config.dataset_type == "cropdiseases":
            Logger.instance().debug("Loading dataset CropDiseases (type MetaTest)")
            return CropDiseases(dataset_config)
        elif dataset_config.dataset_type == "eurosat":
            Logger.instance().debug("Loading dataset EuroSat (type MetaTest)")
            return EuroSat(dataset_config)
        elif dataset_config.dataset_type == "isic":
            Logger.instance().debug("Loading dataset Isic (type MetaTest)")
            return Isic(dataset_config)
        elif dataset_config.dataset_type == "wikiart_artist":
            Logger.instance().debug("Loading dataset WikiArt-Artist (type MetaTest)")
            return WikiArtArtist(dataset_config)
        elif dataset_config.dataset_type == "wikiart_genre":
            Logger.instance().debug("Loading dataset WikiArt-Genre (type MetaTest)")
            return WikiArtGenre(dataset_config)
        elif dataset_config.dataset_type == "wikiart_style":
            Logger.instance().debug("Loading dataset WikiArt-Style (type MetaTest)")
            return WikiArtStyle(dataset_config)
        elif dataset_config.dataset_type == "pacs_object":
            Logger.instance().debug("Loading dataset PACS-Object (type MetaTest)")
            return PacsObject(dataset_config)
        elif dataset_config.dataset_type == "pacs_domain":
            Logger.instance().debug("Loading dataset PACS-Domain (type MetaTest)")
            return PacsDomain(dataset_config)
        elif dataset_config.dataset_type == "celeba":
            Logger.instance().debug("Loading dataset CelebA (type Dataset)")
            return CelebaWrapper(dataset_config)
        elif dataset_config.dataset_type == "meta_album_cls":
            Logger.instance().debug("Loading dataset MetaAlbumCls (type DatasetCls)")
            return MetaAlbumCls(dataset_config)
        else:
            raise ValueError(
                "values allowed: {`omniglot`, `episodic_imagenet`, `episodic_imagenet1k`, `episodic_coco`, " +
                "`miniimagenet`, `cub`, `fungi`, `aircraft`, `meta_inat`, `meta_album`, `cropdiseases`, `eurosat`, " +
                "`isic`, `dtd`, `cifar_fs`, `celeba`, `wikiart` {_artist, _genre, _style}, `pacs` {_object, _domain} " +
                "`meta_album_cls` for dataset_type.\n" +
                "`episodic_imagenet` can also be run with other evaluation datasets: append " +
                "(_val_cifar, _val_cub, _val_aircraft)"
            )
