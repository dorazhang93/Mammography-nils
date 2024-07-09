import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import zero
import os
import argparse
from data_loader import load_clinical_data, load_clinical_data_test
from utils import IndexLoader, format_seconds, get_lr,make_summary, dump_stats, backup_output, calculate_metrics
# %%
class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_layers: ty.List[int],
        dropout: float,
        d_out: int,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x):
        x = x.float()

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


def parse_args():
    parser=argparse.ArgumentParser(description='test a mlp model')
    parser.add_argument('--data-root', default='/home/',help="path to dataset")
    parser.add_argument('--ckpt-path', default='',help="path to ckpt")
    parser.add_argument('--work-dir', default='/home/',help="the dir to save logs and predictions")
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--mode', default='clinical-only', type=str,help='clinical-only mode or clinical-mammo mode')
    parser.add_argument('--site',default='malmo',type=str,help="the site of independent test set")
    args=parser.parse_args()
    return args

# %%
if __name__=='__main__':
    zero.set_randomness(0)
    args=parse_args()
    dataset_dir = args.data_root
    output=Path(args.work_dir)
    output.mkdir(parents=True,exist_ok=True)

    stats: ty.Dict[str, ty.Any] = {
        'dataset': dataset_dir,
        'algorithm': Path(__file__).stem,
    }
    timer = zero.Timer()
    timer.run()

    X, Y,fortnr = load_clinical_data_test(dataset_dir, mode=args.mode)
    X={k: torch.as_tensor(v) for k,v in X.items()}
    Y={k: torch.as_tensor(v) for k,v in Y.items()}

    device = torch.device('cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
    if device.type != 'cpu':
        X = {k: v.to(device) for k, v in X.items()}
        Y = {k: v.to(device) for k, v in Y.items()}

    Y = {k: v.float() for k, v in Y.items()}

    eval_batch_size=32

    loss_fn=F.binary_cross_entropy_with_logits

    model=MLP(
        d_in=X['test'].shape[1],
        d_layers=[128],
        d_out=1,
        dropout=0.2
    ).to(device)


    checkpoint_path = Path(args.ckpt_path) / 'checkpoint.pth'


    @torch.no_grad()
    def evaluate(parts):
        model.eval()
        metrics = {}
        predictions = {}
        for part in parts:
            predictions[part] = (
                torch.cat(
                    [
                        model(
                            X[part][idx]
                        )
                        for idx in IndexLoader(
                        len(X[part]),
                        eval_batch_size,
                        False,
                        device,
                    )
                    ]
                )
            )
            loss = loss_fn(predictions[part],Y[part])
            predictions[part]=predictions[part].cpu()
            metrics[part] = calculate_metrics(
                Y[part].numpy(),  # type: ignore[code]
                predictions[part].numpy(),  # type: ignore[code]
                'logits',
            )
            metrics[part]['loss']=loss.cpu().numpy().item()

        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', make_summary(part_metrics))
        return metrics, predictions

    def save_test_prediction(preds,labels,fortnrs,output):
        assert len(preds)==len(labels)
        assert len(preds)==len(fortnrs)
        data=[]
        for i in range(len(preds)):
            data.append({'fortnr':fortnrs[i],
                         'pred_score': torch.tensor([preds[i]]),
                         'gt_label': torch.tensor([labels[i]])})
        with open(output / f'test_{args.site}.pkl', 'wb') as file:
            pkl.dump(data,file)
        return

    # %%
    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    stats['metrics'], predictions = evaluate(['test'])
    for k, v in predictions.items():
        save_test_prediction(v, Y['test'], fortnr['test'], output)
    stats['time'] = format_seconds(timer())
    print('Done!')










