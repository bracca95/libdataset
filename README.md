# A repo to manage them all: libdataset

## Dataset
The idea is to have a unique repository to manage dataset for deep learning projects. Your own dataset can be built by implementing the abstract class `DatasetWrapper` in `src.dataset.dataset.py`

## Dataset Config
Edit the `config/config.json` file to start

```
"dataset_path": string,
"dataset_type": {`omniglot`, `episodic_imagenet`, `episodic_imagenet1k`, `episodic_coco`, `miniimagenet`, `cub`, `fungi`, `aircraft`, `meta_inat`, `meta_album`, `cropdiseases`, `eurosat`, `isic`, `dtd`, `cifar_fs`, `celeba`, `episodic_imagenet` can also be run with other evaluation datasets: append (_val_cifar, _val_cub, val_aircraft)},
"dataset_id": Optional[List[int]] (param for MetaAlbum datasets only)
"dataset_splits": List[float] (1 for train/test (e.g. [0.8]), 3 for train/val/test),
"normalize": bool,
"crop_size": int,
"image_size": int (after crop, reshape can be applied),
"augment_online": Optional[List[str]] (classes for online augmentation),
"augment_offline": Optional[List[str]] (classes for offline augmentation),
"augment_times": Optional[int] (number of times to apply augmentations)
"dataset_mean": Optional[List[float]] (Grayscale/RGB),
"dataset_std": Optional[List[float]] (Grayscale/RGB)
```

If `dataset_mean` and `dataset_std` are set to null, the program will compute them and then it will quit the execution.
Run the program again to train your model.