import json
from pathlib import Path
import datetime
import os
import shutil

import time
import typing as ty

import torch.optim as optim
import numpy as np
import torch
import math
import zero
import scipy.special
import sklearn.metrics as skm
PROJECT_DIR="/home/*/*/Projects/Mammography/mmpretrain_temp/mmpretrain"

class IndexLoader:
    def __init__(
        self, train_size: int, batch_size: int, shuffle: bool, device: torch.device
    ) -> None:
        self._train_size = train_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._device = device

    def __len__(self) -> int:
        return math.ceil(self._train_size / self._batch_size)

    def __iter__(self):
        indices = list(
            zero.iloader(self._train_size, self._batch_size, shuffle=self._shuffle)
        )
        return iter(torch.cat(indices).to(self._device).split(self._batch_size))




def format_seconds(seconds: float) -> str:
    return str(datetime.timedelta(seconds=round(seconds)))

def get_lr(optimizer: optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']


def calculate_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    classification_mode: str,
) -> ty.Dict[str, float]:
    pos_num=sum(y)
    neg_num=len(y)-pos_num
    pos_fraction= pos_num/neg_num
    labels = None

    if classification_mode == 'probs':
        probs = prediction
    elif classification_mode == 'logits':
        probs = (scipy.special.expit(prediction))
    else:
        assert classification_mode == 'labels'
        probs = None
        labels = prediction
    if labels is None:
        labels = (np.round(probs).astype('int64'))
    #TODO softmax
    # probs=scipy.special.softmax(prediction, axis=1)
    # probs=probs[:,1]+probs[:,2]
    # labels = (np.round(probs).astype('int64'))
    # y=np.clip(y,0,1)

    result = skm.classification_report(y, labels, output_dict=True)  # type: ignore[code]
    result['roc_auc'] = skm.roc_auc_score(y, probs)  # type: ignore[code]
    result['pr_auc'] = skm.average_precision_score(y,probs)
    result['pos_fraction'] = pos_fraction
    return result  # type: ignore[code]

def make_summary(metrics: ty.Dict[str, ty.Any]) -> str:
    precision = 3
    summary = {}
    for k, v in metrics.items():
        if k.isdigit():
            continue
        k = {
            'loss': 'loss',
            'accuracy': 'acc',
            'roc_auc': 'roc_auc',
            'macro avg': 'm',
            'weighted avg': 'w',
        }.get(k, k)
        if isinstance(v, float):
            v = round(v, precision)
            summary[k] = v
        else:
            v = {
                {'precision': 'p', 'recall': 'r', 'f1-score': 'f1', 'support': 's'}.get(
                    x, x
                ): round(v[x], precision)
                for x in v
            }
            for item in v.items():
                summary[k + item[0]] = item[1]

    s = [f'loss = {summary.pop("loss"):.3f}']
    for k, v in summary.items():
        if k not in ['mp', 'mr', 'wp', 'wr']:  # just to save screen space
            s.append(f'{k} = {v}')
    return ' | '.join(s)

def dump_json(x: ty.Any, path: ty.Union[Path, str], *args, **kwargs) -> None:
    Path(path).write_text(json.dumps(x, *args, **kwargs) + '\n')


def dump_stats(stats: dict, output_dir: Path, final: bool = False) -> None:
    dump_json(stats, output_dir / 'stats.json', indent=4)
    json_output_path = os.environ.get('JSON_OUTPUT_FILE')
    if final:
        output_dir.joinpath('DONE').touch()
        if json_output_path:
            try:
                key = str(output_dir.relative_to(PROJECT_DIR))
            except ValueError:
                pass
            else:
                json_output_path = Path(json_output_path)
                try:
                    json_data = json.loads(json_output_path.read_text())
                except (FileNotFoundError, json.decoder.JSONDecodeError):
                    json_data = {}
                json_data[key] = stats
                json_output_path.write_text(json.dumps(json_data))
            shutil.copyfile(
                json_output_path,
                os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
            )

def backup_output(output_dir: Path) -> None:
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output_dir.relative_to(PROJECT_DIR)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output_dir = dir_ / relative_output_dir
        prev_backup_output_dir = new_output_dir.with_name(new_output_dir.name + '_prev')
        new_output_dir.parent.mkdir(exist_ok=True, parents=True)
        if new_output_dir.exists():
            new_output_dir.rename(prev_backup_output_dir)
        shutil.copytree(output_dir, new_output_dir)
        if prev_backup_output_dir.exists():
            shutil.rmtree(prev_backup_output_dir)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        pass
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')
