import numpy as np
import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self, field_dims, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        nn.init.xavier_uniform_(self.embedding.weight.data)  # weight init

    def forward(self, x: torch.Tensor):
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


class FieldAwareInteractionLayer(nn.Module):
    """
    **From Paper:**
    - This layer aims to promote field-aware interaction between features to
    explore more possible combinations.
    - For each feature/field create an embedding for other features.
    - Lets say we have 22 features
    - STEP 1: Each interaction layer conducts two steps, computing Hadamard product
    - STEP 2: Channelwise weighted sum pooling.
    - STEP Extra: To avoid training collapse, we also add residual connection
        in each field-aware interaction layer.
    """

    def __init__(self, num_fields: int, embed_dim: int):
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim

        # Linear on the LAST dimension (F+1 -> F)
        self.proj = nn.Linear(num_fields + 1, num_fields, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, vi: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        vi : torch.Tensor, shape = (B, F, D)
            Input tensor from previous layer
        v0 : torch.Tensor, shape = (B, F, D)
            Original imput tensor, ie, embeddings vector

        Returns
        -------
        out: torch.Tensor, shape = (B, F, D)
            Output tensor of field aware interaction
        """
        B, F, D = v0.shape
        assert F == self.num_fields and D == self.embed_dim, "Shape mismatch."

        # Pad a field of ones along the field axis -> (B, F+1, D)
        ones = torch.ones(B, 1, D, dtype=v0.dtype, device=v0.device)
        x = torch.cat([v0, ones], dim=1)
        # Move field axis to last dim: (B, D, F+1)
        x = x.transpose(1, 2)
        # Dense (last-dim projection): (B, D, F+1) -> (B, D, F)
        context = self.proj(x)
        # Back to (B, F, D)
        context = context.transpose(1, 2)

        # Element-wise product
        out = vi * context

        return out


class FieldAwareInteractionBlock(nn.Module):
    def __init__(self, num_fields, embed_dim: int, layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(
            [FieldAwareInteractionLayer(num_fields, embed_dim) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of size ``(batch_size, num_fields, embed_dim)`` eg. ``(512,23,16)``

        Returns
        ---------
        vi : torch.Tensor
            Tensor of size ``(batch_size, num_fields, embed_dim)` eg. ``(512,23,16)``
        """
        v0 = x
        vi = x

        # Field-aware interaction stack
        for layer in self.layers:
            vi = layer(vi, v0)
        return vi


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


class FINT(nn.Module):
    """
    The FINT model described in the paper for CTR predictions.

    **From paper:**
    - Embedding size is set as 16.
    - For the DNN layers, the number of hidden layers is set as [300, 300, 300].
    - Activation function RELU.
    - Dropout and regularizations are available in the source code for FINT.

    **Reference:** https://arxiv.org/pdf/2107.01999 \\
    **Reference:** https://github.com/zhishan01/FINT/tree/master
    """

    def __init__(
        self,
        field_dims,
        embed_dim: int = 16,
        fint_layers: int = 1,
        hidden_dims=(300, 300, 300),
        dropout: float = 0.2,
    ):
        super().__init__()
        # Embedding layer
        self.embedding = EmbeddingLayer(field_dims, embed_dim)
        self.num_fields = len(field_dims)
        self.output_dim = self.num_fields * embed_dim  # (23 * 16)

        # Field-aware interaction block
        self.fint_block = FieldAwareInteractionBlock(
            self.num_fields,
            embed_dim,
            layers=fint_layers,
        )

        # MLP layer
        self.mlp = MultiLayerPerceptronLayer(
            self.output_dim, hidden_dims, dropout, output_layer=True
        )

    def forward(self, x: torch.Tensor):
        # Embedding layer
        embed_x = self.embedding(x)
        # Field-aware interaction block
        fint_out = self.fint_block(embed_x)
        # MLP head to out probs
        logits = self.mlp(fint_out.view(-1, self.output_dim)).squeeze(1)
        return logits  # pass to BCEWithLogitsLoss
