import os

from PIL import Image
from tqdm import tqdm
from glob import glob
from typing import Optional, List, Set, Tuple

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG


class MvtecAdac(FewShotDataset):
    """Training set of few-shot Adac

    This class is meant to be used when comparing the supervised learning approach with later adaptation to new tasks
    to few-shot techniques. We still need to separate training classes from val and test, but we only need the train
    split here. Use MetaMvtec when testing the adaptation performance.
    I cannot do the usual split, otherwise the cross-entropy loss would search for a larger label space (the one that
    includes all the labels) during the training phase: therefore, I strip the training part.

    [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
    """

    DATA_DIR = "data"
    SUBDIR_TEST = "test"
    SPLIT_FILE = os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits", "mvtec", "mvtec_test.json")

    def __init__(self, dataset_config: DatasetConfig, did: Optional[int]=None):
        self.pre_process(dataset_config)
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        data_dir = os.path.join(self.dataset_config.dataset_path, self.DATA_DIR)

        # class filter: use only the classes defined in split/mvtec_adac/mvtec_test.json
        split_dict = Tools.read_json(self.SPLIT_FILE)
        use_classes = split_dict.keys()
        all_classes = os.listdir(data_dir)
        remove_classes = list(set(all_classes) - set(use_classes))

        # extension filter
        avail_ext = ("jpeg", "jpg", "png", "PNG", "JPG")
        
        # find all and filter
        img_list = glob(os.path.join(data_dir, "*", self.SUBDIR_TEST, "*", "*"))
        img_list = list(filter(lambda x: x.endswith(avail_ext), img_list))
        img_list = [img for img in img_list if all(c not in img for c in remove_classes)]

        # find test defect classes
        split_dict = Tools.read_json(self.SPLIT_FILE)
        test_classes = [f"{k}_{v}" for k, v_list in split_dict.items() for v in v_list]

        # adapt image names to defect class names
        parsed_img_list = [os.path.dirname(img_name.removeprefix(data_dir)) for img_name in img_list]
        parsed_img_list = [s.replace(os.sep, "_") for s in parsed_img_list]
        parsed_img_list = [s.replace("_test_", "_").removeprefix("_") for s in parsed_img_list]

        # find indices to remove
        remove_indices = set()
        for test_def_class in set(test_classes):
            for i, parsed_img in enumerate(parsed_img_list):
                if test_def_class == parsed_img:
                    remove_indices.add(i)

        # retrieve images to keep
        keep_indices = set(range(len(parsed_img_list))) - remove_indices
        train_img_list = list(map(img_list.__getitem__, keep_indices))
        
        return train_img_list
    
    def get_label_list(self) -> List[int]:
        data_dir = os.path.join(self.dataset_config.dataset_path, self.DATA_DIR)

        if self.image_list is None:
            self.image_list = self.get_image_list(None)

        # extract unique label values from dirname
        parsed_img_list = [os.path.dirname(img_name.removeprefix(data_dir)) for img_name in self.image_list]
        parsed_img_list = [s.replace(os.sep, "_") for s in parsed_img_list]
        parsed_img_list = [s.replace("_test_", "_").removeprefix("_") for s in parsed_img_list]
        label_set = sorted(set(parsed_img_list))
        self.label_to_idx = { val: i for i, val in enumerate(label_set) }
        self.idx_to_label = Tools.invert_dict(self.label_to_idx)

        # use mapping to return an int for the corresponding str label
        return [self.label_to_idx[name] for name in parsed_img_list]
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        train_classes = set(self.label_to_idx.keys())
        return train_classes, set(), set()

    def expected_length(self):
        raise NotImplementedError

    def pre_process(self, dataset_config: DatasetConfig):
        """Augment the number of classes
        
        Since the number of defect classes might be too low, we want to increase the overall amount of training classes
        by creating three new variations (rotation with 90, 180, 270). The original defect classes used for training
        are the ones that are not specified in the split/mvtec_adac/mvtec_test.json file.
        """
        
        # quick check that pre-processing has already taken place (not totally correct)
        data_dir = os.path.join(dataset_config.dataset_path, self.DATA_DIR)
        if any(map(lambda x: "90" in x, glob(os.path.join(data_dir, "*", self.SUBDIR_TEST, "*")))):
            Logger.instance().debug(f"MVTec pre-processing has already been done")
            return
        
        Logger.instance().debug("Augmenting MVTec dataset")
        
        split_dict = Tools.read_json(self.SPLIT_FILE)
        avail_ext = ("jpeg", "jpg", "png", "PNG", "JPG")

        # for every background class (k), we have v defect class for test
        for k, v in tqdm(split_dict.items(), total=len(split_dict.keys())):
            curr_bckg_path = os.path.join(data_dir, k, self.SUBDIR_TEST)
            
            # look all the defect classes in each directory (background) and remove the test classes to get the train
            def_all = [s for s in os.listdir(os.path.join(curr_bckg_path)) if os.path.isdir(os.path.join(curr_bckg_path, s))]
            def_train = list(set(def_all) - set(v))
            def_class_imgs = [img for tr_dir in def_train for img in glob(os.path.join(curr_bckg_path, tr_dir, "*"))]
            def_class_imgs = list(filter(lambda x: x.endswith(avail_ext), def_class_imgs))

            # augment with these angles to form new train classes
            for angle in (90, 180, 270):
                for img_path in def_class_imgs:
                    img_pil = Image.open(img_path).convert("RGB")
                    rot_image = img_pil.rotate(angle, expand=True)

                    # save in a new directory
                    class_dir, filename = img_path.rsplit(os.sep, -1)[-2:]
                    new_dir = os.path.join(curr_bckg_path, f"{class_dir}_{str(angle)}")
                    new_img_path = os.path.join(new_dir, filename)
                    os.makedirs(new_dir, exist_ok=True)
                    rot_image.save(os.path.join(new_dir, new_img_path))

        Logger.instance().debug("MVTec pre-processing has finished to augment the classes!")


class MetaMvtec(MvtecAdac):
    """Few-shot version for MVTec dataset

    The idea is to use the original test(!) sub-classes of MVTec to define a few-shot learning scenario, where you can
    learn different types of anomalies and then transfer the knowledge to unkown classes.

    [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
    """

    IDS = {
        0: "bottle",
        1: "cable",
        2: "capsule",
        3: "carpet",
        4: "grid",
        5: "hazelnut",
        6: "leather",
        7: "metal_nut",
        8: "pill",
        9: "screw",
        10: "tile",
        11: "toothbrush",
        12: "transistor",
        13: "wood",
        14: "zipper"
    }

    def __init__(self, dataset_config: DatasetConfig, did: Optional[int]=None):
        self.did = self.check(dataset_config, did)
        self.curr_class = self.IDS[self.did]
        super().__init__(dataset_config, did)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        data_dir = os.path.join(self.dataset_config.dataset_path, self.DATA_DIR, self.curr_class)
        avail_ext = ("jpeg", "jpg", "png", "PNG", "JPG")
        img_list = glob(os.path.join(data_dir, self.SUBDIR_TEST, "*", "*"))
        img_list = list(filter(lambda x: x.endswith(avail_ext), img_list))

        return img_list

    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        split_dict = Tools.read_json(self.SPLIT_FILE)
        test_classes = [f"{self.curr_class}_{v}" for v in split_dict[self.curr_class]]
        train_classes = list(set(self.label_to_idx.keys()) - set(test_classes))
        
        return set(train_classes), set(test_classes), set(test_classes)

    def check(self, dataset_config: DatasetConfig, did: Optional[int]) -> int:
        if did is None:
            if not dataset_config.dataset_id: # null or empty
                split_dict = Tools.read_json(self.SPLIT_FILE)
                first_key, first_val = next(iter(split_dict.items()))
                msg = f"You are selecting the first ID `{first_key}`. This message should appear only once on top."
                Logger.instance().warning(msg)

                invert_ids_dict = Tools.invert_dict(self.IDS)
                return invert_ids_dict[first_key]

            if len(dataset_config.dataset_id) == 1:
                curr_id = dataset_config.dataset_id[0]
                Logger.instance().warning(f"You are probably calling the dataloader (id = {curr_id})")
                return curr_id
            
            else:
                raise ValueError(f"Unknown condition for dataset_id")

        return did