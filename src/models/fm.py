import torch
from torch import nn

from .layers.core import EmbeddingLayer
from .layers.interaction import FMLayer
from .layers.linear import LinearLayer


class FM(nn.Module):
    """
    Initializes a Factorization Machine model.

    Parameters
    ----------
    field_dims : tuple of int
        Number of fields in the input data.
    embed_dim : int, optional
        Dimension of the embedding space. Defaults to 16.
    reduce_sum : bool, optional
        Whether to reduce the sum of the FM output. Defaults to True.

    References
    ----------
    **[1]:** https://ieeexplore.ieee.org/document/5694074

    """

    def __init__(self, field_dims, embed_dim: int = 16, reduce_sum: bool = True):
        super().__init__()
        self.fm = FMLayer(reduce_sum)
        self.linear = LinearLayer(field_dims)
        self.embedding = EmbeddingLayer(field_dims, embed_dim)

    def forward(self, x: torch.Tensor):
        fm_full = self.linear(x) + self.fm(self.embedding(x))
        logits = fm_full.squeeze(1)

        return logits  # pass to BCEWithLogitsLoss
