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
        adjusted_x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(adjusted_x)


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
