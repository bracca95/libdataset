from abc import ABC, abstractmethod
from typing import Optional, List
from torch.utils.data import Dataset


class CustomDataset(ABC, Dataset):

    label_to_idx = {}

    @abstractmethod
    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        ...

    @abstractmethod
    def get_label_list(self) -> List[int]:
        ...

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    # def __len__(self):
        """ https://github.com/pytorch/pytorch/issues/25247#issuecomment-525380635
        No `def __len__(self)` default?
        See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        in pytorch/torch/utils/data/sampler.py 
        """
    #     return super(CustomDataset, self).__len__()