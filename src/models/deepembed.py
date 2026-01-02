import torch
from torch import nn

from .layers.core import EmbeddingLayer, MultiLayerPerceptronLayer


class DeepEmbed(nn.Module):
    """
    A basic MLP model with feature embeddings for CTR predictions.

    **From paper:**
    - Embedding size is set as 16.
    - For the DNN layers, the number of hidden layers is set as [300, 300, 300].
    - Activation function RELU.
    - Dropout and regularizations are available in the source code for FINT.
    """

    def __init__(
        self,
        field_dims,
        embed_dim: int = 16,
        hidden_dims: tuple[int] = (300, 300, 300),
        dropout: float = 0.2,
        output_layer: bool = True,
    ):
        super().__init__()
        # Embedding layer for each categorical feature (e.g., user_id or ad_id)
        self.embedding = EmbeddingLayer(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim  # (23 * 16)
        # MLP layer
        self.mlp = MultiLayerPerceptronLayer(
            self.embed_output_dim, hidden_dims, dropout, output_layer
        )

    def forward(self, x: torch.Tensor):
        embed_x = self.embedding(x)
        # Flatten 3D tensor to 2D, (batch_size, embed_output_dim) (512, 23 * 16)
        logits = self.mlp(embed_x.view(-1, self.embed_output_dim)).squeeze(1)
        return logits  # pass to BCEWithLogitsLoss
