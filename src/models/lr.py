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
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.embedding(x), dim=1) + self.bias


class LogisticRegression(nn.Module):
    """
    Logistic Regression (LR) model.

    **Reference:**
    https://static.googleusercontent.com/media/research.google.com/tr//pubs/archive/41159.pdf
    """

    def __init__(self, field_dims):
        super().__init__()
        self.linear = LinearLayer(field_dims)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of size ``(batch_size, num_fields)`` eg. ``(512,23)``

        Returns
        -------
        self.linear(x).squeeze(1) : torch.Tensor
            Tensor of size ``(batch_size, 1)`` squuezed for loss calculation \\
            Eg. ``torch.Size([512, 1]`` to ``torch.Size([512,]``
        """
        return self.linear(x).squeeze(1)
