# Baseline Project for Defect Views dataset

## Dataset
Your own dataset can be built by inheriting from the abstract class `CustomDataset` in `src.dataset.dataset`

## Dataset Config
Edit the `config/config.json` file to start

```
"dataset_path": string,
"dataset_type": {`opt6`, `opt_bckg`, `opt_double` `binary`, `qplusv1`, `qplusv2`, `qplus_double` `omniglot`, `miniimagenet`, `opt_yolo_train`, `opt_yolo_test`, `cub`, `cifar_fs`},
"dataset_splits": List[float] (1 for train/test (e.g. [0.8]), 3 for train/val/test),
"normalize": bool,
"crop_size": int,
"image_size": int (after crop, reshape can be applied),
"augment_online": Optional[List[str]] (classes for online augmentation),
"augment_offline": Optional[List[str]] (classes for offline augmentation),
"dataset_mean": Optional[List[float]] (Grayscale/RGB),
"dataset_std": Optional[List[float]] (Grayscale/RGB)
```

If `dataset_mean` and `dataset_std` are set to null, the program will compute them and then it will quit the execution.
Run the program again to train your model.