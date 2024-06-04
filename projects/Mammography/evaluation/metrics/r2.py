from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS
from typing import List, Sequence, Dict, Union
import torch
import sklearn.metrics as skm
import numpy as np
import mmengine
import scipy



@METRICS.register_module()
class R2(BaseMetric):
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
        target = torch.cat([res['gt_label'] for res in results]).numpy()
        if 'pred_score' in results[0]:
            if results[0]['pred_score'].shape[0]==1:
                pred = np.squeeze(torch.stack([res['pred_score'] for res in results]).numpy())
            else:
                raise ValueError("Invalid shape")
            metrics['r2_score'] = torch.tensor(skm.r2_score(target, pred))
        else:
            # If only label in the `pred_label`.
            raise ValueError("Only labels found in the pred_label. Invalid input for cross entropy calculation")

        return metrics

