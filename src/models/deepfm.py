import torch
from torch import nn

from .layers.core import EmbeddingLayer, MultiLayerPerceptronLayer
from .layers.interaction import FMLayer
from .layers.linear import LinearLayer


class DeepFM(nn.Module):
    """
    DeepFM is a factorization machine based neural network for CTR Prediction.

    Parameters
    ----------
    field_dims : tuple of int
        Number of fields in the input data.
    embed_dim : int, optional
        Dimension of the embedding space. Defaults to 16.
    hidden_dims : tuple of int, optional
        Dimensions of the hidden layers in the MLP. Defaults to (300, 300, 300).
    dropout : float, optional
        Dropout rate of the MLP. Defaults to 0.2.
    output_layer : bool, optional
        Whether to include an output layer in the MLP. Defaults to True.

    References
    ----------
    **[1]:** Guo H, Tang R, Ye Y, et al. DeepFM: A factorization-Machine based Neural Network for CTR Prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
    """

    def __init__(
        self,
        field_dims,
        embed_dim: int = 16,
        hidden_dims: tuple[int] = (400, 400, 400),
        dropout: float = 0.6,
        output_layer: bool = True,
        reduce_sum: bool = True,
    ):
        super().__init__()
        self.fm_second_order = FMLayer(reduce_sum)
        self.fm_linear = LinearLayer(field_dims)
        self.embedding = EmbeddingLayer(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptronLayer(
            self.embed_output_dim, hidden_dims, dropout, output_layer
        )

    def forward(self, x: torch.Tensor):
        fm_part = self.fm_second_order(self.embedding(x)) + self.fm_linear(x)
        deep_part = self.mlp(self.embedding(x).view(-1, self.embed_output_dim))
        logits = (fm_part + deep_part).squeeze(1)

        return logits  # pass to BCEWithLogitsLoss
