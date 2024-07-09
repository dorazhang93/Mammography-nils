import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zero
import os
import argparse
import pickle as pkl
from data_loader import load_clinical_data
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
    parser=argparse.ArgumentParser(description='train a mlp')
    parser.add_argument('--data-root', default='/home/*',help="path to dataset")
    parser.add_argument('--work-dir', default='/home/*',help="the dir to save logs and models")
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--max-epoch',default=100, type=int)
    parser.add_argument('--lr',default=0.01,type=float)
    parser.add_argument('--weight-decay',default=1e-04,type=float)
    parser.add_argument('--mode', default='clinical-only', type=str,help='clinical-only mode or clinical-mammo mode')
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

    X, Y, fortnr = load_clinical_data(dataset_dir, mode=args.mode)
    X={k: torch.as_tensor(v) for k,v in X.items()}
    Y={k: torch.as_tensor(v) for k,v in Y.items()}

    device = torch.device('cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
    if device.type != 'cpu':
        X = {k: v.to(device) for k, v in X.items()}
        Y = {k: v.to(device) for k, v in Y.items()}
    #TODO softmax
    Y = {k: v.float() for k, v in Y.items()}

    train_size=len(Y['train'])
    print("number of train sample", train_size)
    batch_size=args.batch_size
    max_epoch=args.max_epoch
    epoch_size = math.ceil(train_size / batch_size)
    eval_batch_size=32

    #TODO softmax
    loss_fn=F.binary_cross_entropy_with_logits
    # loss_fn=F.cross_entropy

    model=MLP(
        d_in=X['train'].shape[1],
        d_layers=[128],
    #TODO softmax
        d_out=1,
        dropout=0.2
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    stream = zero.Stream(IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(16)# patientce=16
    training_log = {'train': [], 'val': [], 'test': []}
    timer = zero.Timer()
    checkpoint_path = output / 'checkpoint.pth'


    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch} | {format_seconds(timer())} | {output}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': get_lr(optimizer),
                    'batch_size': batch_size,
                    'epoch_size': epoch_size,
                }.items()
            )
        )


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


    def save_checkpoint(final):
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stream': stream.state_dict(),
                'random_state': zero.get_random_state(),
                **{
                    x: globals()[x]
                    for x in [
                        'progress',
                        'stats',
                        'timer',
                        'training_log',
                    ]
                },
            },
            checkpoint_path,
        )
        dump_stats(stats, output, final)
        backup_output(output)


    # %%
    timer.run()
    for epoch in stream.epochs(max_epoch):
        print_epoch_info()

        model.train()
        epoch_losses = []
        for batch_idx in epoch:
            optimizer.zero_grad()

            loss = loss_fn(
                model(X['train'][batch_idx]),
                Y['train'][batch_idx],
                )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach())
        # scheduler.step()
        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log['train'].extend(epoch_losses)
        print(f'[train] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

        metrics, predictions = evaluate(['val','test'])
        for k, v in metrics.items():
            training_log[k].append(v)
        progress.update(1/metrics['val']['loss'])

        if progress.success:
            print('New best epoch!')
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            save_checkpoint(False)
            for k, v in predictions.items():
                np.save(output / f'p_{k}.npy', v)

        elif progress.fail:
            break

    # %%
    def save_test_prediction(preds,labels,fortnrs,output):
        assert len(preds)==len(labels)
        assert len(preds)==len(fortnrs)
        data=[]
        for i in range(len(preds)):
            data.append({'fortnr':fortnrs[i],
                         'pred_score': torch.tensor([preds[i]]),
                         'gt_label': torch.tensor([labels[i]])})
        with open(output / 'test.pkl', 'wb') as file:
            pkl.dump(data,file)
        return

    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    stats['metrics'], predictions = evaluate(['val','test'])
    for k, v in predictions.items():
        np.save(output / f'p_{k}.npy', v)
        if k=='test':
            save_test_prediction(v,Y['test'],fortnr['test'],output)
    stats['time'] = format_seconds(timer())
    save_checkpoint(True)
    print('Done!')










