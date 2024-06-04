# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, List
from mmpretrain.structures import DataSample
import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .cls_head import BinaryClsHead


@MODELS.register_module()
class LymphNodeClsHead(BinaryClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 loss_weight:float,
                 task_idx: int,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LymphNodeClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.num_classes = num_classes
        self.loss_weight=loss_weight
        self.task_idx=task_idx
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)[:,self.task_idx]
        cls_score = pre_logits.view(pre_logits.size(0),-1)
        return cls_score
