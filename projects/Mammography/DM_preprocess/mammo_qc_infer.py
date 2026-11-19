"""Inference-only copy of the mammo-qc abnormality detector, for Python 3.8 / torch 1.10.

Why a copy instead of ``import mammo_qc``: the mammo-qc package was developed with
Python 3.13 / torch 2.15 and declares ``requires-python >= 3.10``, ``torch >= 2.4``,
``pydicom >= 3.0``, ``numpy >= 1.26`` and ``pandas >= 2.1``, so it cannot be installed in
this environment. Only what inference needs is kept here, with the changes required for
torch 1.10:

* ``torch.load(..., weights_only=False)`` -> ``weights_only`` does not exist in torch 1.10
  (TypeError). The checkpoint holds only tensors, plain dicts/lists and an OrderedDict
  (no numpy objects), so a plain ``torch.load`` is safe here.
* Inference runs in float32. The training config has ``amp: true`` (bf16 autocast);
  that is a training setting, and fp32 inference avoids depending on autocast/bf16
  support in torch 1.10.
* Class names are taken from the checkpoint (8 object / 4 quality / 4 fov classes for
  ``enb0_imagenet_refine_omidb``), not from a hard-coded taxonomy.

The image preprocessing is a line-for-line copy of ``mammo_qc/preprocess.py``
(load -> MONOCHROME1 inversion -> flat-edge background to zero -> min-max 0-1 ->
INTER_AREA resize to work_size (512) -> INTER_AREA resize to input_size (256)).
It must stay identical to training: the QC model must NOT receive the windowed /
cropped images of the breast preprocessing.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TASKS = ("object", "quality", "fov")
STANDARD_INDEX = 0  # index 0 of every head is the "standard" class


# --------------------------------------------------------------------------- #
# Preprocessing (identical to mammo_qc/preprocess.py, DICOM path only)
# --------------------------------------------------------------------------- #
def _to_gray(arr: np.ndarray, multiframe: bool = False) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if multiframe:
            warnings.warn("multi-frame image: using the first frame")
            return arr[0]
        if arr.shape[-1] in (3, 4):
            return arr[..., :3].astype(np.float32).mean(-1)
        if arr.shape[0] in (3, 4):
            return arr[:3].astype(np.float32).mean(0)
        warnings.warn(f"3-D array {arr.shape}: using the first slice")
        return arr[0]
    raise ValueError(f"unsupported image shape {arr.shape}")


def read_dicom_raw(path) -> np.ndarray:
    """Stored pixel values as float32, MONOCHROME1 inverted (as in training)."""
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array
    pi = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    multiframe = arr.ndim == 3 and pi.startswith("MONOCHROME")
    arr = _to_gray(arr, multiframe=multiframe).astype(np.float32)
    if pi == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def _flat_edge_background(arr: np.ndarray, band: float = 0.02, flat_tol: float = 0.03) -> Optional[float]:
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return None
    w = arr.shape[1]
    b = max(1, int(round(w * band)))
    best = None
    for strip in (arr[:, :b], arr[:, w - b:]):
        p10, p50, p90 = np.percentile(strip, [10, 50, 90])
        spread = (p90 - p10) / (hi - lo)
        if spread < flat_tol and (best is None or spread < best[0]):
            best = (spread, float(p50))
    return None if best is None else best[1]


def background_to_zero(arr: np.ndarray) -> np.ndarray:
    bg = _flat_edge_background(arr)
    if bg is None:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if (bg - lo) / (hi - lo) > 0.5:
        arr = hi - arr
        bg = hi - bg
    return np.maximum(arr - bg, 0.0)


def normalize01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def qc_input_from_dicom(path, data_cfg: dict) -> np.ndarray:
    """DICOM file -> float32 model input of shape (input_size, input_size) in [0, 1]."""
    arr = read_dicom_raw(path)
    if data_cfg.get("dicom_background_to_zero", True):
        arr = background_to_zero(arr)
    arr = normalize01(arr)
    s = int(data_cfg["work_size"])
    arr = cv2.resize(arr, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
    n = int(data_cfg["input_size"])
    return cv2.resize(arr, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)


def save_png01(path, img: np.ndarray) -> None:
    """Same PNG encoding as mammo_qc.preprocess.save_png."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img=(np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    thumb = cv2.resize(img, (257, 257), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), thumb)


class QCDataset(torch.utils.data.Dataset):
    """Returns (image [1, H, W], index, error message or '')."""

    def __init__(self, paths: List[str], data_cfg: dict):
        self.paths = paths
        self.data_cfg = data_cfg
        self.size = int(data_cfg["input_size"])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            x = qc_input_from_dicom(self.paths[i], self.data_cfg)
            err = ""
        except Exception as e:  # noqa: BLE001 - an unreadable file must not stop the run
            x = np.zeros((self.size, self.size), np.float32)
            err = f"{type(e).__name__}: {e}"[:300]
        return torch.from_numpy(np.ascontiguousarray(x))[None], i, err


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MultiTaskNet(nn.Module):
    """Same module layout as mammo_qc.model.MultiTaskNet, so the state dict loads strictly."""

    def __init__(self, backbone: str, classes: Dict[str, Tuple[str, ...]], dropout: float = 0.0):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, in_chans=3)
        n = self.backbone.num_features
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Dropout(dropout), nn.Linear(n, len(classes[t]))) for t in TASKS
        })
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        f = self.backbone(x)
        return {t: head(f) for t, head in self.heads.items()}


class QCModel:
    """Loaded detector + its data config, class names and uncertainty thresholds."""

    def __init__(self, checkpoint: str, device: torch.device, thresholds: Optional[str] = None):
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:  # torch < 1.13: no weights_only argument
            ckpt = torch.load(checkpoint, map_location="cpu")
        self.classes: Dict[str, Tuple[str, ...]] = {t: tuple(c) for t, c in ckpt["classes"].items()}
        self.cfg = ckpt["cfg"]
        self.data_cfg = self.cfg["data"]
        self.net = MultiTaskNet(self.cfg["model"]["backbone"], self.classes,
                                float(self.cfg["model"].get("dropout", 0.0)))
        self.net.load_state_dict(ckpt["model"], strict=True)
        self.net.to(device).eval()
        self.device = device

        thr_path = Path(thresholds) if thresholds else Path(checkpoint).parent / "thresholds.json"
        self.tau: Optional[Dict[str, float]] = None
        if thr_path.exists():
            self.tau = json.loads(thr_path.read_text())["tau"]
        logger.info("QC model: %s (%s, %s epochs), classes %s, tau %s",
                    checkpoint, self.cfg["model"]["backbone"], ckpt.get("epochs_trained"),
                    {t: len(c) for t, c in self.classes.items()}, self.tau)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        out = self.net(x.to(self.device).float())
        return {t: torch.softmax(out[t].float(), dim=1).cpu().numpy() for t in TASKS}

    def describe(self, probs: Dict[str, np.ndarray], b: int,
                 standard_threshold: Optional[float] = None) -> dict:
        """Per-image result row: probabilities, predicted class, uncertainty, abnormal flag."""
        row: dict = {}
        p_std, nonstd, unc = 1.0, [], []
        for t in TASKS:
            p = probs[t][b]
            for c, v in zip(self.classes[t], p):
                row[f"prob_{t}_{c}"] = float(v)
            k = int(p.argmax())
            row[f"pred_{t}"] = self.classes[t][k]
            maxp = float(p.max())
            ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(len(p)))
            row[f"maxprob_{t}"], row[f"entropy_{t}"] = maxp, ent
            row[f"uncertain_{t}"] = int(self.tau is not None and maxp < self.tau[t])
            p_std *= float(p[STANDARD_INDEX])
            if k != STANDARD_INDEX:
                nonstd.append((t, self.classes[t][k]))
            if row[f"uncertain_{t}"]:
                unc.append(t)
        row["p_standard"] = p_std
        if standard_threshold is not None:
            abnormal = p_std < standard_threshold
        else:
            abnormal = bool(nonstd)

        # normal with flag that pred_object=needle & pred_fov=full_breast
        if row["pred_object"]=="needle" and row["pred_fov"]=="full_breast":
            row["normal_flag"]=True
        else:
            row["normal_flag"]=False

        row["abnormal"] = int(abnormal)
        row["abnormality"] = "; ".join(f"{t}={c}" for t, c in nonstd) if nonstd else (
            "low_p_standard" if abnormal else "")
        row["uncertain"] = int(bool(unc))
        row["_nonstandard"] = nonstd  # internal: used to place thumbnails, dropped before saving
        return row

    def columns(self) -> List[str]:
        return ([f"prob_{t}_{c}" for t in TASKS for c in self.classes[t]]
                + [f"{k}_{t}" for k in ("pred", "maxprob", "entropy", "uncertain") for t in TASKS]
                + ["p_standard", "abnormal", "abnormality", "uncertain"])
