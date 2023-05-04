# Baseline Project for Defect Views dataset

## Dataset
Your own dataset can be built by inheriting from the abstract class `CustomDataset` in `src.dataset.staple_dataset`

## Config
```
"dataset_path": string,
"dataset_type": {'all' (all defects), 'binary' (bubble vs points)},
"batch_size": int,
"epochs": int,
"crop_size": int (suggested 28),
"image_size": int (after crop, reshape can be applied),
"augment_online": List[str] (classes that undergo online augmentation),
"augment_offline": List[str] (classes that undergo offline augmentation),
"dataset_mean": List[float] (Grayscale/RGB)
"dataset_std": List[float] (Grayscale/RGB)
```
