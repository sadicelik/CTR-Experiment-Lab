import torch
from torch import nn


class MLP(nn.Module):
    """
    A similar neural network architechture presented in paper for FINT.

    https://github.com/zhishan01/FINT/blob/master/models/layers.py
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(256, 128, 64),
        dropout: float = 0.2,
        output_layer: bool = True,
    ) -> None:
        super().__init__()

        layers = []
        prev = input_dim

        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            prev = hidden_dim

        if output_layer:
            layers.append(torch.nn.Linear(prev, 1))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x).squeeze(1)
        return logits  # pass to BCEWithLogitsLoss
