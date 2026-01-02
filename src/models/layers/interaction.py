import torch
from torch import nn


class FMLayer(nn.Module):
    """
    Factorization Machine models pairwise (order-2) feature interactions
    without linear term and bias.

    References
    ----------
    **[1]:** https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """

    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor):
        square_of_sum = torch.pow(torch.sum(x, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(x**2, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square

        if self.reduce_sum:
            return 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        else:
            return cross_term
