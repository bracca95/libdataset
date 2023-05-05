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

## Models
[ProtoNet implementation used](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch).

# References
```bib
@article{DBLP:journals/corr/SnellSZ17,
  author    = {Jake Snell and
               Kevin Swersky and
               Richard S. Zemel},
  title     = {Prototypical Networks for Few-shot Learning},
  journal   = {CoRR},
  volume    = {abs/1703.05175},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.05175},
  archivePrefix = {arXiv},
  eprint    = {1703.05175},
  timestamp = {Wed, 07 Jun 2017 14:41:38 +0200},
  biburl    = {http://dblp.org/rec/bib/journals/corr/SnellSZ17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```
