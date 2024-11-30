from typing import Optional, List, Set, Tuple

from .dataset_fsl import FewShotDataset
from .meta_album import MetaAlbum
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG


class FullMetaAlbum(FewShotDataset):
    """Few-shot set Meta Album of all the albums

    This is a wrapper around meta-album datasets to use all the classes as if they were part of a unique collection.
    This is tested on the micro and mini releases of meta-album.

    [Meta Album](https://meta-album.github.io/)
    """

    def __init__(self, dataset_config: DatasetConfig):
        if not dataset_config.dataset_id:
            raise ValueError(f"FullMetaAlbum requires `dataset_id` to be defined")

        self.album_list = [MetaAlbum(dataset_config, did) for did in dataset_config.dataset_id]
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        img_list = []
        for album in self.album_list:
            img_list.extend(album.get_image_list(None))

        return img_list
    
    def get_label_list(self) -> List[int]:
        duplicate_labels: Set[str] = set()
        label_list: List[str] = []
        
        for album in self.album_list:
            # find duplicates (debugging and info purposes)
            curr_unique_labels = list(album.label_to_idx.keys())
            Tools.add_elems_to_set(duplicate_labels, *list(set(curr_unique_labels) & set(label_list)))
            
            # add all the labels
            curr_label_name = [album.idx_to_label[c] for c in album.label_list]
            label_list.extend(curr_label_name)
        
        if len(duplicate_labels) > 0:
            msg = f"{len(duplicate_labels)} duplicate labels among albums: {duplicate_labels}.\n" + \
                  f"Total labels are {len(set(label_list))} instead of {len(set(label_list)) + len(duplicate_labels)}"
            Logger.instance().warning(msg)
        
        self.label_to_idx = { val: i for i, val in enumerate(set(label_list)) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        return [self.label_to_idx[c] for c in label_list]
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        train_labels, val_labels, test_labels = [], [], []
        
        for album in self.album_list:
            train_curr, val_curr, test_curr = album.split_method()
            train_labels.extend(list(train_curr))
            val_labels.extend(list(val_curr))
            test_labels.extend(list(test_curr))

        set_train = set(train_labels)
        set_val = set(val_labels)
        set_test = set(test_labels)

        # find possible intersections (same-name label from one album that was in train, might be val for another album)
        if len(set_train & set_val) > 0:
            set_train = set_train - (set_train & set_val)
            set_val = set_val | (set_train & set_val)

        if len(set_train & set_test) > 0:
            set_train = set_train - (set_train & set_test)
            set_test = set_test | (set_train & set_test)

        if len(set_val & set_test) > 0:
            set_test = set_test - (set_val & set_test)
            set_val = set_val | (set_val & set_test)

        return set_train, set_val, set_test

    def expected_length(self):
        return len(self.image_list)
    