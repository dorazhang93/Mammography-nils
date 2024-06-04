from mmpretrain.registry import METRICS
from mmengine.evaluator import BaseMetric
from typing import List, Sequence, Dict
import torch
import torch.nn.functional as F

@METRICS.register_module()
class BinaryCrossEntropy(BaseMetric):
    def process(self, data_batch: Sequence[Dict], data_samples: Sequence[Dict]):
        """ The processed results should be stored in ``self.results``, which will
            be used to computed the metrics when all batches have been processed.
            `data_batch` stores the batch data from dataloader,
            and `data_samples` stores the batch outputs from model.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            if 'pred_score' in data_sample:
                result['pred_score'] = data_sample['pred_score'].cpu()
            else:
                result['pred_label'] = data_sample['pred_label'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()
            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """ Compute the metrics from processed results and returns the evaluation results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        metrics = {}
        # concat
        target = torch.cat([res['gt_label'] for res in results])
        if 'pred_score' in results[0]:
            logits = torch.stack([res['pred_score'] for res in results])
            if target.dim() == 1:
                target = target.view(-1, 1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target.float(),  # only accepts float type tensor
                reduction='none'
                )
            metrics['loss'] = loss.mean()
        else:
            # If only label in the `pred_label`.
            raise ValueError("Only labels found in the pred_label. Invalid input for cross entropy calculation")
        return metrics
