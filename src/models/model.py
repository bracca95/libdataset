import torch
from torch import nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    @staticmethod
    def get_output_size(model: nn.Module, x: torch.Tensor):
        with torch.no_grad():
            output = model(x)

        # assuming a flat tensor so that shape = (batch_size, feature_vector)
        return output.shape[-1]