from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS
from typing import List, Sequence, Dict, Union
import torch
import sklearn.metrics as skm
import numpy as np
import mmengine
import scipy



@METRICS.register_module()
class AUC(BaseMetric):
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
        target = torch.cat([res['gt_label'] for res in results]).numpy()
        # print(target)
        if 'pred_score' in results[0]:
            if results[0]['pred_score'].shape[0]==1:
                # probility after sigmoid
                logits = np.squeeze(torch.stack([res['pred_score'] for res in results]).numpy())
                pred = scipy.special.expit(logits)
                # print("pred_score prediction",pred)
            else:
                # logits before softmax
                logits = np.squeeze(torch.stack([res['pred_score'] for res in results]).numpy())
                # print("AUC logits", logits)
                pred = scipy.special.softmax(logits, axis=1)
                # print("probs",pred)
                pred = pred[:,1]
                # print("positive probs",pred)

            metrics['roc_auc'] = torch.tensor(skm.roc_auc_score(target, pred))
            metrics['pr_auc'] = torch.tensor(skm.average_precision_score(target, pred))
        else:
            # If only label in the `pred_label`.
            raise ValueError("Only labels found in the pred_label. Invalid input for cross entropy calculation")

        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence]
    ):

        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        print(pred),print(target)
        roc=torch.tensor(skm.roc_auc_score(target, pred))
        pr=torch.tensor(skm.average_precision_score(target, pred))
        return roc,pr
