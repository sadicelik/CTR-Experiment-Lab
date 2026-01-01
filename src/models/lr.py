import torch
from torch import nn

from .layers.linear import LinearLayer


class LogisticRegression(nn.Module):
    """
    Logistic Regression (LR) model.

    References
    ----------
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
