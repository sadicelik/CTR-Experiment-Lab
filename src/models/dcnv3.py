import torch
from torch import nn

from .layers.core import EmbeddingLayer


class MultiHeadFeatureEmbedding(nn.Module):
    """
    Embedding layer with multi head proposed in the paper.

    Parameters
    ----------
    field_dims : tuple of int
        Number of fields in the input data.
    embedding_dim : int
        Dimension of the embedding space. Defaults to 16.
    num_heads : int
        Number of heads in the multi-head mechanism. Defaults to 2.
    """

    def __init__(self, field_dims, embedding_dim: int, num_heads: int = 2):
        super(MultiHeadFeatureEmbedding, self).__init__()
        self.num_heads = num_heads
        # TODO: May change the Embedding layer later, similar to FuxiCTR
        self.embedding_layer = EmbeddingLayer(field_dims, embedding_dim)

    def forward(self, x: torch.Tensor):
        feature_emb = self.embedding_layer(x)  # B × F × D = 512 × 23 × 16
        multihead_feature_emb = torch.tensor_split(feature_emb, self.num_heads, dim=-1)
        multihead_feature_emb = torch.stack(
            multihead_feature_emb, dim=1
        )  # B × H × F × D/H
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(
            multihead_feature_emb, 2, dim=-1
        )  # B × H × F × D/2H
        multihead_feature_emb1, multihead_feature_emb2 = multihead_feature_emb1.flatten(
            start_dim=2
        ), multihead_feature_emb2.flatten(
            start_dim=2
        )  # B × H × FD/2H; B × H × FD/2H
        multihead_feature_emb = torch.cat(
            [multihead_feature_emb1, multihead_feature_emb2], dim=-1
        )
        return multihead_feature_emb  # B × H × FD/H


class LCN(nn.Module):
    """Linear Cross Network"""

    def __init__(
        self,
        input_dim,
        num_cross_layers: int = 3,
        layer_norm: bool = True,
        batch_norm: bool = True,
        net_dropout: float = 0.1,
        num_heads: int = 1,
    ):
        super(LCN, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()

        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)

        self.masker = nn.ReLU()
        self.sfc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor):
        x0 = x
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x0 * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        logit = self.sfc(x)

        return logit


class ECN(nn.Module):
    """Exponential Cross Network"""

    def __init__(
        self,
        input_dim,
        num_cross_layers: int = 3,
        layer_norm: bool = True,
        batch_norm: bool = False,
        net_dropout: float = 0.1,
        num_heads: int = 1,
    ):
        super(ECN, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()

        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)

        self.masker = nn.ReLU()
        self.dfc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor):
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        logit = self.dfc(x)

        return logit


class DCNv3(nn.Module):
    def __init__(
        self,
        field_dims,
        embedding_dim: int = 10,
        num_deep_cross_layers: int = 4,
        num_shallow_cross_layers: int = 4,
        deep_net_dropout: float = 0.1,
        shallow_net_dropout: float = 0.3,
        layer_norm: bool = True,
        batch_norm: bool = False,
        num_heads: int = 1,
        **kwargs,
    ):
        super(DCNv3, self).__init__()
        self.embedding_layer = MultiHeadFeatureEmbedding(
            field_dims, embedding_dim * num_heads, num_heads
        )
        # TODO: May be bugged
        self.embed_output_dim = len(field_dims) * embedding_dim
        self.LCN = LCN(
            input_dim=self.embed_output_dim,
            num_cross_layers=num_shallow_cross_layers,
            net_dropout=shallow_net_dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            num_heads=num_heads,
        )
        self.ECN = ECN(
            input_dim=self.embed_output_dim,
            num_cross_layers=num_deep_cross_layers,
            net_dropout=deep_net_dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor):
        feature_emb = self.embedding_layer(x)
        slogit = self.LCN(feature_emb).mean(dim=1)
        dlogit = self.ECN(feature_emb).mean(dim=1)
        logits = ((dlogit + slogit) * 0.5).squeeze(1)

        return logits  # pass to BCEWithLogitsLoss

    def add_loss(self):
        pass
