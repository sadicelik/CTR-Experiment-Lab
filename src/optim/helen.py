import math

import torch


class Helen(torch.optim.Optimizer):
    """
    Helen optimizer.

    Parameters
    ----------

    Reference
    ---------
    https://arxiv.org/abs/2403.00798

    """

    def __init__(
        self,
        lr_embed=1e-3,
        lr_net=1e-3,
        rho=0.05,
        net_pert=True,
        bound=0.3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        adaptive=False,
        **kwargs
    ):
        pass

    def first_step():
        pass

    def second_step():
        pass
