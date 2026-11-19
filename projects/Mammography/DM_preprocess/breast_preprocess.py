"""Source-agnostic breast preprocessing for images that passed mammo-QC.

Follows the step order of the harmonisation layer (dm_preprocess_v2.py /
``mammodiff.data.preprocess``), with one configuration for every image: the source
dataset is unknown for an unseen cohort, so there is no per-source config.

For every DICOM, in order:
  1. read stored pixel values; invert MONOCHROME1, and also invert when a flat image
     edge is bright (wrong/missing tag after anonymisation - same rule as mammo-QC)
  2. resample to the canonical pixel spacing (skipped and flagged if spacing is unknown)
  3. build a 0-1 segmentation proxy (background level -> 0, 99.9th percentile -> 1)
  4. segment the breast: threshold -> opening -> largest component -> fill holes
  5. check the breast-area fraction (breast pixels / all pixels of the uncropped image).
     Outside ``breast_frac_range``: redo the segmentation with flat-edge detection
     (rows / columns whose 5th-95th percentile spread is below ``flat_tol`` are
     background). Still outside the range: flag ``breast_frac_flag = "out_of_range"``.
  6. suppress artefacts: every pixel outside the breast mask -> 0
  8. crop to the breast bounding box plus a fixed physical air margin
  9. intensity window: 0.5th-99.5th percentile of breast pixels (``foreground``) or the
     full stored bit range (``native``) -> float32 in [0, 1], background = 0

Output resolution is the canonical spacing, not a training resolution (the model
decides that), exactly as in dm_preprocess_v2.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from mammo_qc_infer import _flat_edge_background, _to_gray


@dataclass
class PreprocessConfig:
    target_spacing_mm: float = 0.1          # canonical pixel spacing after resampling
    valid_spacing_mm: Tuple[float, float] = (0.02, 0.3)   # tag values outside this are ignored
    seg_threshold: float = 0.05             # step 4: proxy > this (0-1) is foreground
    seg_open_mm: float = 1.5                # step 4: opening kernel; removes thin lines and noise
    breast_frac_range: Tuple[float, float] = (0.05, 0.9)  # step 5
    flat_tol: float = 0.05                  # step 5: p95 - p5 of a row/column below this = flat
    flat_fg_tol: float = 0.05               # step 5: |proxy - background level| above this = breast
    flat_gap_frac: float = 0.01             # step 5: gaps up to this fraction are bridged
    flat_close_mm: float = 8.0              # step 5: closing that smooths the redo mask's skin line
    margin_mm: float = 5.0                  # step 8: air margin around the breast box
    orientation: str = "left"               # step 7: "left", "right" or "none"
    quantize_mode: str = "foreground"       # step 9: "foreground" or "native"
    window_percentiles: Tuple[float, float] = (0.5, 99.5)


@dataclass
class ProcessRecord:
    uid: str
    success: bool = False
    failure_stage: Optional[str] = None
    failure_message: Optional[str] = None
    photometric: Optional[str] = None
    inverted: Optional[str] = None          # "tag", "edge" or None
    bits_stored: Optional[int] = None
    input_shape: Optional[Tuple[int, int]] = None
    native_spacing_mm: Optional[float] = None
    spacing_tag: Optional[str] = None
    spacing_unknown: bool = False
    output_spacing_mm: Optional[float] = None
    resampled_shape: Optional[Tuple[int, int]] = None
    seg_method: Optional[str] = None        # "threshold" or "flat_edge"
    breast_frac_threshold: Optional[float] = None
    breast_frac_flat_edge: Optional[float] = None
    breast_frac: Optional[float] = None     # fraction of the method finally used
    breast_frac_flag: Optional[str] = None  # "ok", "ok_after_redo", "out_of_range"
    chest_wall_side: Optional[str] = None
    was_flipped: bool = False
    crop_box: Optional[Tuple[int, int, int, int]] = None   # x0, y0, x1, y1 in resampled frame
    output_shape: Optional[Tuple[int, int]] = None
    window_lo: Optional[float] = None
    window_hi: Optional[float] = None
    clipped_low_frac: Optional[float] = None
    clipped_high_frac: Optional[float] = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class PreprocessError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


# --------------------------------------------------------------------------- #
# Mask helpers
# --------------------------------------------------------------------------- #
def _odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def largest_component_filled(binary: np.ndarray) -> np.ndarray:
    """Largest 8-connected foreground component with interior holes filled (bool)."""
    b = binary.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n <= 1:
        return np.zeros(binary.shape, bool)
    mask = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    # holes = background components that do not touch the image border
    n2, lab2 = cv2.connectedComponents((~mask).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]]))
    holes = ~np.isin(lab2, border) & ~mask
    return mask | holes


def _morph(binary: np.ndarray, k: int, op: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(binary.astype(np.uint8), op, kernel).astype(bool)


def _open(binary: np.ndarray, k: int) -> np.ndarray:
    return _morph(binary, k, cv2.MORPH_OPEN)


def _longest_run(flags: np.ndarray, max_gap: int) -> Optional[Tuple[int, int]]:
    """[start, end) of the longest run of True, bridging gaps of up to max_gap False values."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]]) + 1
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def segment_threshold(proxy: np.ndarray, cfg: PreprocessConfig, px_mm: float) -> np.ndarray:
    binary = proxy > cfg.seg_threshold
    binary = _open(binary, _odd(cfg.seg_open_mm / px_mm))
    return largest_component_filled(binary)


def segment_flat_edge(proxy: np.ndarray, cfg: PreprocessConfig, px_mm: float) -> np.ndarray:
    """Rows/columns with a flat 5th-95th percentile profile are background.

    The breast lies inside the longest run of non-flat columns x non-flat rows. The
    background level is the median of the flat region; inside the box, pixels that
    differ from it by more than ``flat_fg_tol`` are breast. Percentiles make the
    test robust to burned-in text and small markers in the background.
    """
    h, w = proxy.shape
    p5c, p95c = np.percentile(proxy, [5, 95], axis=0)
    p5r, p95r = np.percentile(proxy, [5, 95], axis=1)
    cols = _longest_run((p95c - p5c) >= cfg.flat_tol, int(cfg.flat_gap_frac * w))
    rows = _longest_run((p95r - p5r) >= cfg.flat_tol, int(cfg.flat_gap_frac * h))
    if cols is None or rows is None:
        return np.zeros(proxy.shape, bool)
    (x0, x1), (y0, y1) = cols, rows
    outside = np.ones(proxy.shape, bool)
    outside[y0:y1, x0:x1] = False
    bg = float(np.median(proxy[outside])) if outside.any() else float(proxy.min())
    binary = np.zeros(proxy.shape, bool)
    binary[y0:y1, x0:x1] = np.abs(proxy[y0:y1, x0:x1] - bg) > cfg.flat_fg_tol
    binary = _open(binary, _odd(cfg.seg_open_mm / px_mm))
    mask = largest_component_filled(binary)
    return largest_component_filled(_morph(mask, _odd(cfg.flat_close_mm / px_mm), cv2.MORPH_CLOSE))


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
class BreastPreprocessor:
    def __init__(self, cfg: Optional[PreprocessConfig] = None):
        self.cfg = cfg or PreprocessConfig()

    # step 1 ----------------------------------------------------------------
    def _read(self, ds, rec: ProcessRecord) -> np.ndarray:
        pi = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        rec.photometric = pi
        if not pi.startswith("MONOCHROME"):
            raise PreprocessError("decode", f"unsupported PhotometricInterpretation {pi!r}")
        arr = ds.pixel_array
        arr = _to_gray(arr, multiframe=arr.ndim == 3).astype(np.float32)
        rec.bits_stored = int(getattr(ds, "BitsStored", 0) or 0) or None
        rec.input_shape = (int(arr.shape[0]), int(arr.shape[1]))
        if pi == "MONOCHROME1":
            arr = arr.max() - arr
            rec.inverted = "tag"
        bg = _flat_edge_background(arr)
        lo, hi = float(arr.min()), float(arr.max())
        if bg is not None and hi > lo and (bg - lo) / (hi - lo) > 0.5:
            arr = hi - arr
            rec.inverted = None if rec.inverted == "tag" else "edge"
            rec.extras["bright_edge_inverted"] = True
        return arr

    # step 2 ----------------------------------------------------------------
    def _spacing(self, ds, rec: ProcessRecord) -> Optional[float]:
        lo, hi = self.cfg.valid_spacing_mm
        for tag in ("ImagerPixelSpacing", "PixelSpacing"):
            val = getattr(ds, tag, None)
            try:
                s = float(val[0]) if val is not None else None
            except (TypeError, ValueError, IndexError):
                s = None
            if s is not None and np.isfinite(s) and lo <= s <= hi:
                rec.native_spacing_mm, rec.spacing_tag = s, tag
                return s
        rec.spacing_unknown = True
        return None

    def _resample(self, arr: np.ndarray, spacing: Optional[float], rec: ProcessRecord) -> np.ndarray:
        target = self.cfg.target_spacing_mm
        if spacing is None or abs(spacing / target - 1) < 1e-3:
            rec.output_spacing_mm = target if spacing is not None else None
            return arr
        scale = spacing / target
        h, w = arr.shape
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        rec.output_spacing_mm = target
        return cv2.resize(arr, size, interpolation=interp)

    # steps 3-5 -------------------------------------------------------------
    @staticmethod
    def _proxy(arr: np.ndarray) -> np.ndarray:
        bg = _flat_edge_background(arr)
        lo = float(arr.min()) if bg is None else bg
        hi = float(np.percentile(arr, 99.9))
        if hi <= lo:
            hi = float(arr.max())
        if hi <= lo:
            raise PreprocessError("segment", "constant image")
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def _segment(self, proxy: np.ndarray, px_mm: float, rec: ProcessRecord) -> np.ndarray:
        lo, hi = self.cfg.breast_frac_range
        mask = segment_threshold(proxy, self.cfg, px_mm)
        frac = float(mask.mean())
        rec.breast_frac_threshold = round(frac, 4)
        rec.seg_method, rec.breast_frac, rec.breast_frac_flag = "threshold", round(frac, 4), "ok"
        if lo <= frac <= hi:
            return mask
        mask2 = segment_flat_edge(proxy, self.cfg, px_mm)
        frac2 = float(mask2.mean())
        rec.breast_frac_flat_edge = round(frac2, 4)
        if mask2.any():
            mask, frac = mask2, frac2
            rec.seg_method, rec.breast_frac = "flat_edge", round(frac2, 4)
        rec.breast_frac_flag = "ok_after_redo" if lo <= frac <= hi else "out_of_range"
        if not mask.any():
            raise PreprocessError("segment", "no breast found")
        return mask

    # steps 6-9 -------------------------------------------------------------
    def _orient(self, arr, mask, rec):
        w = mask.shape[1]
        left, right = int(mask[:, : w // 2].sum()), int(mask[:, w // 2:].sum())
        rec.chest_wall_side = "left" if left >= right else "right"
        target = self.cfg.orientation
        if target in ("left", "right") and rec.chest_wall_side != target:
            rec.was_flipped = True
            return arr[:, ::-1].copy(), mask[:, ::-1].copy()
        return arr, mask

    def _crop(self, arr, mask, px_mm, rec):
        ys, xs = np.nonzero(mask)
        m = int(round(self.cfg.margin_mm / px_mm))
        h, w = mask.shape
        x0, x1 = max(0, xs.min() - m), min(w, xs.max() + 1 + m)
        y0, y1 = max(0, ys.min() - m), min(h, ys.max() + 1 + m)
        rec.crop_box = (int(x0), int(y0), int(x1), int(y1))
        return arr[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    def _window(self, arr, mask, rec) -> np.ndarray:
        vals = arr[mask]
        if self.cfg.quantize_mode == "foreground":
            lo, hi = (float(v) for v in np.percentile(vals, self.cfg.window_percentiles))
        elif self.cfg.quantize_mode == "native":
            lo = 0.0
            hi = float(2 ** rec.bits_stored - 1) if rec.bits_stored else float(arr.max())
        else:
            raise ValueError(f"unknown quantize_mode {self.cfg.quantize_mode!r}")
        if hi <= lo:
            raise PreprocessError("window", f"empty intensity window [{lo}, {hi}]")
        rec.window_lo, rec.window_hi = lo, hi
        rec.clipped_low_frac = round(float((vals < lo).mean()), 5)
        rec.clipped_high_frac = round(float((vals > hi).mean()), 5)
        out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        out[~mask] = 0.0
        return out

    # ------------------------------------------------------------------------
    def process_dicom(self, ds, uid: str) -> Tuple[Optional[np.ndarray], ProcessRecord]:
        """Return (float32 image in [0, 1] or None, record). Never raises on bad images."""
        rec = ProcessRecord(uid=uid)
        stage = "decode"
        try:
            arr = self._read(ds, rec)
            stage = "resample"
            spacing = self._spacing(ds, rec)
            arr = self._resample(arr, spacing, rec)
            rec.resampled_shape = (int(arr.shape[0]), int(arr.shape[1]))
            px_mm = rec.output_spacing_mm or self.cfg.target_spacing_mm  # unknown: assume canonical
            stage = "segment"
            mask = self._segment(self._proxy(arr), px_mm, rec)
            # stage = "orient"
            # arr, mask = self._orient(arr, mask, rec)
            stage = "crop"
            arr, mask = self._crop(arr, mask, px_mm, rec)
            stage = "window"
            out = self._window(arr, mask, rec)
        except PreprocessError as e:
            rec.failure_stage, rec.failure_message = e.stage, str(e)
            return None, rec
        except Exception as e:  # noqa: BLE001 - one bad image must not stop the cohort
            rec.failure_stage, rec.failure_message = stage, f"{type(e).__name__}: {e}"[:300]
            return None, rec
        rec.output_shape = (int(out.shape[0]), int(out.shape[1]))
        rec.success = True
        return out, rec
