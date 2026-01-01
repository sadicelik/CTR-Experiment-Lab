import numpy as np
import torch
from torch import nn


class LinearLayer(nn.Module):
    def __init__(self, field_dims, output_dim: int = 1):
        super().__init__()
        # One dimensional Embedding(1396615, 1)
        self.embedding = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        nn.init.xavier_uniform_(self.embedding.weight.data)  # weight init

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of size ``(batch_size, num_fields)`` eg. ``(512,23)``
        """
        adjusted_x = x + self.offsets
        return torch.sum(self.embedding(adjusted_x), dim=1) + self.bias
