# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class RegressionHead(BaseModule):
    """regression head.

    Args:
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='MSELoss', loss_weight=1.0)``.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 loss: dict = dict(type='MSELoss'),
                 init_cfg: Optional[dict] = None):
        super(RegressionHead, self).__init__(init_cfg=init_cfg)

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor],data_samples: List[DataSample]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The ClsHead doesn't have the final classification head,
        # just return the unpacked inputs.
        return pre_logits

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats,data_samples)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target)
        losses['loss'] = loss*self.loss_weight

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        cls_score = self(feats,data_samples)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """

        pred_scores = cls_score

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score in zip(data_samples, pred_scores):
            if data_sample is None:
                data_sample = DataSample()
            data_sample.set_pred_score(score)
            out_data_samples.append(data_sample)
        return out_data_samples
