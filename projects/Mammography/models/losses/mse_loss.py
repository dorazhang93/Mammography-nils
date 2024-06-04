# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from mmengine.model import BaseModule
from torch import nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MSELoss(BaseModule):
    """Loss function for CAE.

    Compute the align loss and the main loss.

    Args:
        lambd (float): The weight for the align loss.
    """

    def __init__(self,) -> None:
        super().__init__()
        self.loss_mse = nn.MSELoss()

    def forward(
            self, logits: torch.Tensor, target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function of CAE Loss.

        Args:
            logits (torch.Tensor): The outputs from the decoder.
            target (torch.Tensor): The targets generated by dalle.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The main loss and align loss.
        """
        if target.dim() == 1:
            target = target.view(-1, 1)
        loss = self.loss_mse(logits,target)

        return loss
