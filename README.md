# Baseline Project for Defect Views dataset

## Dataset
Your own dataset can be built by inheriting from the abstract class `CustomDataset` in `src.dataset.staple_dataset`

## Config
Edit the `config/config.json` file to start

```
"dataset_path": string,
"dataset_type": {'all' (all defects), 'binary' (bubble vs points)},
"dataset_splits": List[float] (1 for train/test (e.g. [0.8]), 3 for train/val/test),
"batch_size": int,
"epochs": int,
"crop_size": int (suggested 28),
"image_size": Optional[int] (after crop, reshape can be applied),
"augment_online": Optional[List[str]] (classes for online augmentation),
"augment_offline": Optional[List[str]] (classes for offline augmentation),
"dataset_mean": Optional[List[float]] (Grayscale/RGB),
"dataset_std": Optional[List[float]] (Grayscale/RGB)
```

If `dataset_mean` and `dataset_std` are set to null, the program will compute them and then it will quit the execution.
Run the program again to train your model.

## Model
Models are supposed to be put in `src/models`. You can use ad additional subfolder; suppose that you need different
implementations of few-show learning frameworks, such as ProtoNet and Siamese Network, you can create a subfolder `FSL`
in the models directory, then put your `.py` file there.

## Train/Test routines
Train and test routines have to be inherited from the `TrainTest` abstract class in `src/train_test_routine.py`.