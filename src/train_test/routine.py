import torch

from abc import ABC, abstractmethod
from torch import nn
from typing import Optional, Deque
from collections import deque

from src.utils.tools import Logger
from src.utils.config_parser import Config
from src.models.model import Model
from src.datasets.staple_dataset import CustomDataset
from src.datasets.dataset_utils import SubsetInfo
from config.consts import SubsetsDict, General


class TrainTest(ABC):

    train_str, val_str, test_str = General.DEFAULT_SUBSETS
    
    def __init__(self, model: Model, dataset: CustomDataset, subsets_dict: Optional[SubsetsDict]=None):
        self.model = model
        self.dataset = dataset
        self.subsets_dict = subsets_dict

        self.train_info: Optional[SubsetInfo] = self.get_subset_info(self.train_str)
        self.val_info: Optional[SubsetInfo] = self.get_subset_info(self.val_str)
        self.test_info: Optional[SubsetInfo] = self.get_subset_info(self.test_str)

        self.acc_var: Deque[float] = deque(maxlen=10)

    @abstractmethod
    def train(self, config: Config):
        ...

    @abstractmethod
    def test(self, config: Config, model_path: str):
        ...

    def get_subset_info(self, subset_str_id: str) -> Optional[SubsetInfo]:
        """Wrap subset into SubsetInfo structure (holds more information)

        Args:
            subset_str_id (str): { 'train', 'val', 'test' }

        Returns:
            SubsetInfo if the Subset is present (validation dataset can be None)

        Raises:
            ValueError if `subset_str_id` is not in { 'train', 'val', 'test' }
        """

        if subset_str_id not in General.DEFAULT_SUBSETS:
            raise ValueError(f"TrainTest::get_subset_info: only accept 'train', 'val', 'test'")
        
        if self.subsets_dict is None:
            return None
        
        if self.subsets_dict[subset_str_id] is None:
            info_dict = None
        else:
            subset_labels = [self.dataset[idx][1] for idx in self.subsets_dict[subset_str_id].indices]
            classes = list(set(subset_labels))
            info_dict = { self.dataset.idx_to_label[i]: subset_labels.count(i) for i in classes }
            Logger.instance().debug(f"{subset_str_id} has {len(classes)} classes: {info_dict}")
        
        return SubsetInfo(subset_str_id, self.subsets_dict[subset_str_id], info_dict)
    

    def check_stop_conditions(self, curr_acc: float, limit: float = 0.985, eps: float = 0.001) -> bool:
        if curr_acc < limit:
            return False
        
        if not len(self.acc_var) == self.acc_var.maxlen:
            self.acc_var.append(curr_acc)
            return False
        
        self.acc_var.popleft()
        self.acc_var.append(curr_acc)

        acc_var = torch.Tensor(list(self.acc_var))
        if torch.max(acc_var) - torch.min(acc_var) > 2 * eps:
            return False
        
        Logger.instance().warning(f"Raised stop iteration: last {len(self.acc_var)} increment below {2 * eps}.")
        return True


class TrainTestExample(TrainTest):

    def __init__(self, model: Model, dataset: CustomDataset, subsets_dict: Optional[SubsetsDict]=None):
        super().__init__(model, dataset, subsets_dict)

    def train(self):
        Logger.instance().debug("train example")

    def test(self):
        Logger.instance().debug("test example")