#!/usr/bin/env python
"""Build the JSON meta file of a test cohort: mammo-QC first, then breast preprocessing.

    python generate_json_meta.py \\
        --input-csv /data/cohort/meta.csv \\
        --output-folder /data/cohort/derived \\
        --qc-checkpoint /models/mammo-qc/enb0_imagenet_refine_omidb/final.pt

Input CSV (paths relative to the CSV's folder, or absolute):
    required  dicom_path, patient_id
    optional  LNM (P/N or 1/0), Tsize, plus any other columns (kept in the QC table)

Stage 1 - pixel-level quality control (mammo-QC, EfficientNet-B0)
    Every DICOM is classified by the three heads (object / quality / field of view).
    An image is standard only if object = breast_only, quality = FFDM and
    fov = full_breast; everything else is abnormal and is excluded from Stage 2.
      qc/qc_results.csv            one row per image: probabilities, predicted classes,
                                   abnormal flag, `abnormality` (e.g. "object=implant")
      qc/abnormal_images.csv       the abnormal and unreadable rows only
      qc/thumbnails/normal/        every standard image (256 x 256, QC model input)
      qc/thumbnails/abnormal/<task>/<class>/   every abnormal image, per non-standard head

Stage 2 - breast preprocessing of the standard images (breast_preprocess.py)
    resample -> segment -> breast-fraction check (flat-edge redo) -> suppress artefacts
    -> orient -> crop with air margin -> intensity window.
      preprocessed/png/            8-bit PNG
      preprocessed/npz/            float32 in [0, 1], key "image"
      preprocessed/thumbnails/     256 x 256 PNG of the processed image
      preprocessed/thumbnails_breast_frac_out_of_range/   copies, for inspection
      preprocessed/process_log.jsonl, preprocessed/preprocess_results.csv

Final outputs
      <json-name> (default test.json)   {"data_list": [{img_path, npz_path, patient_id,
                                          image_id, gt_label: {N, tumor_size}, ...}]}
      run_summary.json, pipeline.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from breast_preprocess import BreastPreprocessor, PreprocessConfig  # noqa: E402

logger = logging.getLogger("generate_json_meta")
MISSING = -999  # value of a missing ground-truth label in the JSON (as before)


# --------------------------------------------------------------------------- #
# Arguments and set-up
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = ap.add_argument_group("input / output")
    io.add_argument("--input-csv", required=True, help="CSV with dicom_path, patient_id [, LNM, Tsize]")
    io.add_argument("--output-folder", required=True)
    io.add_argument("--json-name", default="test.json")
    io.add_argument("--thumbnail-size", type=int, default=257)

    qc = ap.add_argument_group("stage 1: mammo-QC")
    qc.add_argument("--qc-checkpoint", required=True, help="mammo-qc final.pt")
    qc.add_argument("--qc-thresholds", default=None,
                    help="thresholds.json for the `uncertain` columns (default: next to the checkpoint)")
    qc.add_argument("--standard-threshold", type=float, default=None,
                    help="abnormal if p_standard < this; default: argmax rule of the three heads")
    qc.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    qc.add_argument("--qc-batch-size", type=int, default=32)
    qc.add_argument("--qc-workers", type=int, default=4, help="DataLoader workers (DICOM decoding)")
    qc.add_argument("--reuse-qc", action="store_true",
                    help="skip stage 1 and read qc/qc_results.csv of an earlier run")
    qc.add_argument("--qc-only", action="store_true", help="stop after stage 1")

    pp = ap.add_argument_group("stage 2: breast preprocessing (same settings for every image)")
    d = PreprocessConfig()
    pp.add_argument("--workers", type=int, default=4, help="processes; 0 = run in this process")
    pp.add_argument("--target-spacing", type=float, default=d.target_spacing_mm, help="mm per pixel")
    pp.add_argument("--margin-mm", type=float, default=d.margin_mm)
    pp.add_argument("--orientation", choices=["left", "right", "none"], default=d.orientation,
                    help="side of the chest wall after preprocessing; none = keep as acquired")
    pp.add_argument("--quantize-mode", choices=["foreground", "native"], default=d.quantize_mode)
    pp.add_argument("--breast-frac-range", type=float, nargs=2, default=list(d.breast_frac_range),
                    metavar=("LO", "HI"))
    pp.add_argument("--seg-threshold", type=float, default=d.seg_threshold)
    pp.add_argument("--flat-tol", type=float, default=d.flat_tol,
                    help="p95 - p5 spread (0-1 scale) below which a row/column is flat background")
    return ap.parse_args(argv)


def setup_logging(out_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "pipeline.log", mode="a")):
        h.setFormatter(fmt)
        root.addHandler(h)


def safe(s) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(s))


def encode_lnm(value) -> Optional[int]:
    """P/N or 1/0 -> 1/0; anything else -> None."""
    if pd.isna(value):
        return None
    v = str(value).strip().upper()
    if v in ("P", "1", "1.0", "POS", "POSITIVE"):
        return 1
    if v in ("N", "0", "0.0", "NEG", "NEGATIVE"):
        return 0
    logger.warning("unrecognised LNM value %r -> treated as missing", value)
    return None


def load_cohort(csv_path: Path) -> pd.DataFrame:
    """Read the input CSV, resolve DICOM paths and give every image a stable id."""
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} does not exist")
    df = pd.read_csv(csv_path)
    missing = [c for c in ("dicom_path", "patient_id") if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path}: missing column(s) {missing}")
    df = df.reset_index(drop=True)
    data_root = csv_path.resolve().parent
    df["dicom_path"] = df["dicom_path"].astype(str)
    df["abs_path"] = [str(data_root / p) for p in df["dicom_path"]]   # absolute paths stay absolute
    df["uid"] = [hashlib.sha1(f"{pid}|{p}".encode()).hexdigest()[:16]
                 for pid, p in zip(df["patient_id"], df["dicom_path"])]
    df["image_id"] = [f"{safe(pid)}_{uid}" for pid, uid in zip(df["patient_id"], df["uid"])]
    df["gt_N"] = [encode_lnm(v) for v in df["LNM"]] if "LNM" in df.columns else None
    df["gt_tumor_size"] = (pd.to_numeric(df["Tsize"], errors="coerce") if "Tsize" in df.columns else np.nan)
    dup = df["uid"].duplicated()
    if dup.any():
        logger.warning("%d duplicated (patient_id, dicom_path) rows dropped", int(dup.sum()))
        df = df[~dup].reset_index(drop=True)
    logger.info("cohort: %d images of %d patients from %s", len(df), df["patient_id"].nunique(), csv_path)
    return df


# --------------------------------------------------------------------------- #
# Stage 1: mammo-QC
# --------------------------------------------------------------------------- #
def run_qc(df: pd.DataFrame, args, qc_dir: Path) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    from mammo_qc_infer import QCDataset, QCModel, save_png01

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("stage 1: mammo-QC on %s (torch %s)", device, torch.__version__)
    model = QCModel(args.qc_checkpoint, device, args.qc_thresholds)
    if int(model.data_cfg["input_size"]) != args.thumbnail_size:
        logger.warning("QC thumbnails are saved at the model input size %s, not %s",
                       model.data_cfg["input_size"], args.thumbnail_size)

    thumb_dir = qc_dir / "thumbnails"
    shutil.rmtree(thumb_dir, ignore_errors=True)  # a re-run must not mix with old thumbnails
    loader = DataLoader(QCDataset(df["abs_path"].tolist(), model.data_cfg),
                        batch_size=args.qc_batch_size, shuffle=False, num_workers=args.qc_workers,
                        pin_memory=device.type == "cuda")
    rows: List[dict] = []
    t0 = time.time()
    for x, idx, errs in loader:
        probs = model.predict(x)
        for b, i in enumerate(idx.tolist()):
            meta = df.iloc[i]
            row = {"uid": meta["uid"], "error": errs[b]}
            if errs[b]:
                row["qc_status"] = "unreadable"
                logger.warning("unreadable: %s (%s)", meta["dicom_path"], errs[b])
                rows.append(row)
                continue
            row.update(model.describe(probs, b, args.standard_threshold))
            nonstd = row.pop("_nonstandard")
            row["qc_status"] = "normal" if not row['abnormal'] else \
                "normal_flag" if row["normal_flag"] else \
                    "abnormal_flag" if row["abnormal_flag"] else "abnormal"

            img, name = x[b, 0].numpy(), meta["image_id"] + ".png"
            if not row["abnormal"]:
                save_png01(thumb_dir / "normal" / name, img)
            else:
                for task, cls in (nonstd or [("p_standard", "low_p_standard")]):
                    save_png01(thumb_dir / "abnormal" / task / cls / name, img)
            if row["normal_flag"]:# normal cases with only needle detected are flagged but still kept for nodal prediction
                save_png01(thumb_dir / "normal_flaged" / name, img)
            if row["abnormal_flag"]:# abnormal cases with breast_only, full_breast and not FFDM
                save_png01(thumb_dir / "abnormal_flaged" / name, img)
            rows.append(row)
        if len(rows) % 500 < len(idx):
            logger.info("  QC %d/%d images (%.1f img/s)", len(rows), len(df), len(rows) / (time.time() - t0))

    res = pd.DataFrame(rows)
    keep = [c for c in df.columns if c not in ("abs_path", "gt_N", "gt_tumor_size")]
    cols = keep + ["qc_status"] + model.columns() + ["error"]
    table = df[keep].merge(res, on="uid", how="left").reindex(columns=cols)
    table.to_csv(qc_dir / "qc_results.csv", index=False)
    table[table["qc_status"] != "normal"].to_csv(qc_dir / "abnormal_images.csv", index=False)
    logger.info("stage 1 done in %.0f s -> %s", time.time() - t0, qc_dir / "qc_results.csv")
    return table


def qc_summary(table: pd.DataFrame) -> dict:
    status = table["qc_status"].value_counts().to_dict()
    abn = table[table["qc_status"] == "abnormal"]
    by_class = Counter(part.strip() for s in abn["abnormality"].fillna("") for part in s.split(";") if part.strip())
    return {"n_images": int(len(table)), "status": {k: int(v) for k, v in status.items()},
            "abnormal_by_class": dict(by_class.most_common()),
            "n_uncertain": int(table.get("uncertain", pd.Series(dtype=float)).fillna(0).sum())}


# --------------------------------------------------------------------------- #
# Stage 2: breast preprocessing (one image per call, runs in worker processes)
# --------------------------------------------------------------------------- #
def _init_worker() -> None:
    cv2.setNumThreads(1)


def preprocess_one(row: dict, cfg: PreprocessConfig, out_dir: str, thumb_size: int) -> dict:
    import pydicom

    out = Path(out_dir)
    try:
        ds = pydicom.dcmread(row["abs_path"], force=True)
    except Exception as e:  # noqa: BLE001
        return {"uid": row["uid"], "success": False, "failure_stage": "decode",
                "failure_message": f"{type(e).__name__}: {e}"[:300]}
    image, rec = BreastPreprocessor(cfg).process_dicom(ds, row["uid"])
    result = rec.to_dict()
    if image is None:
        return result

    name = row["image_id"]
    png_path, npz_path = out / "png" / f"{name}.png", out / "npz" / f"{name}.npz"
    img8 = np.round(image * 255.0).astype(np.uint8)
    cv2.imwrite(str(png_path), img8)
    np.savez_compressed(npz_path, image=image.astype(np.float32),
                        pixel_spacing_mm=np.float32(rec.output_spacing_mm or np.nan))
    thumb = cv2.resize(img8, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "thumbnails" / f"{name}.png"), thumb)
    if rec.breast_frac_flag == "out_of_range":
        cv2.imwrite(str(out / "thumbnails_breast_frac_out_of_range" / f"{name}.png"), thumb)
    result.update(png_path=str(png_path), npz_path=str(npz_path))
    return result


def run_preprocessing(normal: pd.DataFrame, cfg: PreprocessConfig, args, pp_dir: Path) -> pd.DataFrame:
    for sub in ("png", "npz", "thumbnails", "thumbnails_breast_frac_out_of_range"):
        shutil.rmtree(pp_dir / sub, ignore_errors=True)
        (pp_dir / sub).mkdir(parents=True)
    logger.info("stage 2: preprocessing %d standard images with %d worker(s); config %s",
                len(normal), args.workers, cfg)
    rows = normal[["uid", "image_id", "abs_path"]].to_dict(orient="records")
    results: List[dict] = []
    t0 = time.time()

    def collect(res: dict) -> None:
        results.append(res)
        log_fh.write(json.dumps(res, default=str) + "\n")
        if not res.get("success"):
            logger.warning("preprocessing failed %s: [%s] %s", res["uid"],
                           res.get("failure_stage"), res.get("failure_message"))
        elif res.get("breast_frac_flag") == "out_of_range":
            logger.warning("breast fraction out of range %s: %.3f (%s)", res["uid"],
                           res["breast_frac"], res["seg_method"])
        if len(results) % 200 == 0:
            logger.info("  preprocessed %d/%d (%.1f img/s)", len(results), len(rows),
                        len(results) / (time.time() - t0))

    with open(pp_dir / "process_log.jsonl", "w") as log_fh:
        if args.workers > 0:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
                futures = [pool.submit(preprocess_one, r, cfg, str(pp_dir), args.thumbnail_size) for r in rows]
                for fut in as_completed(futures):
                    collect(fut.result())
        else:
            for r in rows:
                collect(preprocess_one(r, cfg, str(pp_dir), args.thumbnail_size))

    from breast_preprocess import ProcessRecord

    cols = list(ProcessRecord(uid="").to_dict()) + ["png_path", "npz_path"]
    res = pd.DataFrame(results).reindex(columns=cols)  # same columns even if every image failed
    table = normal[["uid", "patient_id", "dicom_path"]].merge(res, on="uid", how="left")
    table.to_csv(pp_dir / "preprocess_results.csv", index=False)
    logger.info("stage 2 done in %.0f s -> %s", time.time() - t0, pp_dir / "preprocess_results.csv")
    return table


def preprocess_summary(pp: pd.DataFrame) -> dict:
    ok = pp[pp["success"] == True]  # noqa: E712 - column may hold NaN
    return {
        "n_input": int(len(pp)),
        "n_ok": int(len(ok)),
        "failures_by_stage": {k: int(v) for k, v in pp.loc[pp["success"] != True, "failure_stage"]
                              .fillna("unknown").value_counts().items()},
        "seg_method": {k: int(v) for k, v in ok["seg_method"].value_counts().items()},
        "breast_frac_flag": {k: int(v) for k, v in ok["breast_frac_flag"].value_counts().items()},
        "breast_frac_quantiles": {str(q): round(float(ok["breast_frac"].quantile(q)), 4)
                                  for q in (0.01, 0.05, 0.5, 0.95, 0.99)} if len(ok) else {},
        "spacing_unknown": int(ok["spacing_unknown"].fillna(False).astype(bool).sum()),
        "flipped": int(ok["was_flipped"].fillna(False).astype(bool).sum()),
        "inverted": {str(k): int(v) for k, v in ok["inverted"].fillna("none").value_counts().items()},
    }


# --------------------------------------------------------------------------- #
# JSON meta file
# --------------------------------------------------------------------------- #
def build_meta(df: pd.DataFrame, pp: pd.DataFrame) -> Dict[str, list]:
    ok = pp[pp["success"] == True].set_index("uid")  # noqa: E712
    data_list = []
    for _, r in df[df["uid"].isin(ok.index)].iterrows():
        p = ok.loc[r["uid"]]
        n = r["gt_N"]
        ts = r["gt_tumor_size"]
        data_list.append({
            "img_path": p["png_path"],
            "npz_path": p["npz_path"],
            "patient_id": str(r["patient_id"]),
            "image_id": r["uid"],
            "dicom_path": r["dicom_path"],
            "gt_label": {"N": MISSING if n is None or pd.isna(n) else int(n),
                         "tumor_size": MISSING if pd.isna(ts) else float(ts)},
            "clinic_vars": None,
            "breast_frac": float(p["breast_frac"]),
            "breast_frac_flag": p["breast_frac_flag"],
        })
    return {"data_list": data_list}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_folder).resolve()  # absolute paths in the JSON
    qc_dir, pp_dir = out_dir / "qc", out_dir / "preprocessed"
    for d in (qc_dir, pp_dir):
        d.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    logger.info("=" * 70)
    logger.info("generate_json_meta: %s", " ".join(sys.argv))

    df = load_cohort(Path(args.input_csv))

    # Stage 1 ---------------------------------------------------------------
    if args.reuse_qc:
        qc_table = pd.read_csv(qc_dir / "qc_results.csv")
        if not set(df["uid"]) <= set(qc_table["uid"]):
            raise SystemExit("--reuse-qc: qc_results.csv does not cover every image of the input CSV")
        logger.info("stage 1 skipped: reusing %s", qc_dir / "qc_results.csv")
    else:
        qc_table = run_qc(df, args, qc_dir)
    summary = {"qc": qc_summary(qc_table)}
    logger.info("QC: %s", summary["qc"]["status"])
    for cls, n in summary["qc"]["abnormal_by_class"].items():
        logger.info("  abnormal %-30s %d", cls, n)
    if args.qc_only:
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        return 0

    # Stage 2 ---------------------------------------------------------------
    included_flags=["normal","normal_flag","abnormal_flag"]
    logger.info("Images included for stage 2 preprocessing: %s", ", ".join(included_flags))
    normal_uids = set(qc_table.loc[qc_table["qc_status"].isin(included_flags), "uid"])
    normal = df[df["uid"].isin(normal_uids)].reset_index(drop=True)
    if normal.empty:
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        logger.error("no standard images left after QC; nothing to preprocess")
        return 1
    cfg = PreprocessConfig(target_spacing_mm=args.target_spacing, margin_mm=args.margin_mm,
                           orientation=args.orientation, quantize_mode=args.quantize_mode,
                           breast_frac_range=tuple(args.breast_frac_range),
                           seg_threshold=args.seg_threshold, flat_tol=args.flat_tol)
    pp = run_preprocessing(normal, cfg, args, pp_dir)
    summary["preprocessing"] = preprocess_summary(pp)

    # JSON ------------------------------------------------------------------
    meta = build_meta(df, pp)
    json_path = out_dir / args.json_name
    json_path.write_text(json.dumps(meta, indent=1))
    summary["json"] = {"path": str(json_path), "n_images": len(meta["data_list"]),
                       "n_patients": len({d["patient_id"] for d in meta["data_list"]})}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    s = summary["preprocessing"]
    logger.info("preprocessing: %d/%d ok; segmentation %s; breast fraction %s; failures %s",
                s["n_ok"], s["n_input"], s["seg_method"], s["breast_frac_flag"], s["failures_by_stage"])
    logger.info("DONE: %d of %d input images in %s (%d patients)", summary["json"]["n_images"],
                len(df), json_path, summary["json"]["n_patients"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
