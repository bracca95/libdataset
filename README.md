# A repo to manage them all: libdataset

## Dataset
The idea is to have a unique repository to manage dataset for deep learning projects. Add your own by extending `DatasetWrapper` in `src.dataset.dataset.py`. The `DatasetCls` in `src.datasets.dataset_cls.py` shows an example. More few-shot datasets are added as subclass of `FewShotDataset` in `src.datasets.fsl.dataset_fsl`.

## Dataset Config
Edit the `config/config.json` file to start

```
"dataset_path": string,
"dataset_type": {`omniglot`, `episodic_imagenet`, `episodic_imagenet1k`, `episodic_coco`, `miniimagenet`, `cub`, `fungi`, `aircraft`, `meta_inat`, `meta_album`, `cropdiseases`, `eurosat`, `isic`, `dtd`, `cifar_fs`, `celeba`, `wikiart` {_artist, _genre, _style}, `pacs` {_object, _domain}, `meta_album_cls`, `mnist`, `fashion_mnist`, `mnist2fashion`, `fashion2mnist` for dataset_type. `episodic_imagenet` can also be run with other evaluation datasets: append (_val_cifar, _val_cub, val_aircraft)},
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

## Online Augmentations
Configuring augmentations (`augment_online`) might be tricky:
* meta-datasets make use of "support" + "query", but can use "strong" as well.
* classification datasets use typical sample augmentations: "sample_weak", "sample_strong", "sample_rot_45", "sample_rot_90".
* meta-mnist and omniglot are exceptions: they use classification augmentations even if they are meta-datasets.

## Requirements
This program has been tested with python 3.10.x

Install the required dependencies with:
```python
pip install -r requirements.txt
```


## Usage
1. Create a context batch sampler by following [this example](https://github.com/bracca95/CAMeLU/blob/main/src/samplers/context_sampler.py).

2. Load a torch dataloader as `torch.utils.data.DataLoader`
```python
@staticmethod
def init_loader(config: Config, dataset_wrapper: DatasetWrapper split_set: str) -> Optional[DataLoader]:
        current_dataset = getattr(dataset_wrapper, f"{split_set}_dataset")
        
        if current_dataset is None:
            return None

        sampler = CtxBatchSampler(
            labels=current_dataset.label_list,
            classes_per_it=config.context.n_way,
            n_samples_cls=config.context.k_shot + config.context.k_query,
            iterations=config.context.episodes
        )
    
        return DataLoader(
            current_dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers
        )
```

3. Follow the main example in `unit_test/test_module.py`
