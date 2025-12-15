import numpy as np
import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self, field_dims, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        nn.init.xavier_uniform_(self.embedding.weight.data)  # weight init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of size ``(batch_size, num_fields)`` eg. ``(512,23)``

        Returns
        -------
        self.embedding(x) : torch.Tensor
            Tensor of size ``(batch_size, num_fields, embed_dim)` eg. ``(512,23,16) ``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MultiLayerPerceptronLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int],
        dropout: float = 0.2,
        output_layer: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))  # binary CTR output

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


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
            self.embed_output_dim, hidden_dims, dropout, output_layer=output_layer
        )

    def forward(self, x: torch.Tensor):
        embed_x = self.embedding(x)
        # Flatten 3D tensor to 2D, (batch_size, embed_output_dim) (512, 23 * 16)
        logits = self.mlp(embed_x.view(-1, self.embed_output_dim)).squeeze(1)
        return logits  # pass to BCEWithLogitsLoss
