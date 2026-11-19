# QC → breast preprocessing → JSON meta (test cohort)

```bash
python generate_json_meta.py \
    --input-csv   /data/cohort/meta.csv \
    --output-folder /data/cohort/derived \
    --qc-checkpoint /models/mammo-qc/runs/enb0_imagenet_refine_omidb/final.pt \
    --qc-workers 8 --workers 8
    
```


| File | Role |
|---|---|
| `generate_json_meta.py` | Driver: stage 1 QC, stage 2 preprocessing, JSON, logging |
| `mammo_qc_infer.py` | Inference-only copy of mammo-qc, runs on Python 3.8 / torch 1.10.1 |
| `breast_preprocess.py` | Source-agnostic preprocessing (dm_preprocess_v2 step order) |
| `requirements_py38.txt` | Tested versions and install order |
| `tests/make_phantoms.py` | Synthetic DICOMs for a smoke test (not mammograms) |

Copy `final.pt` and `thresholds.json` (same folder) from `mammo-qc-code/runs/enb0_imagenet_refine_omidb/`.
`last.pt` is not needed.

## Stage 1: mammo-QC (all images)

Standard = object `breast_only` and quality `FFDM` and fov `full_breast` (argmax of each head),
or `p_standard >= --standard-threshold` if given. Everything else is abnormal and is not preprocessed.

- `qc/qc_results.csv`: all images. Columns `qc_status` (normal / abnormal / unreadable),
  `abnormality` (e.g. `object=implant; fov=spot_view`), all class probabilities, `p_standard`,
  max-probability, entropy and `uncertain` per head (recorded only; no uncertain folder).
- `qc/abnormal_images.csv`: abnormal and unreadable rows.
- `qc/thumbnails/normal/` and `qc/thumbnails/abnormal/<task>/<class>/`: every image, 256 × 256,
  exactly as the model saw it. An image with two non-standard heads appears in both folders.

After reviewing, re-run only stage 2 with `--reuse-qc`; edit `qc_status` in `qc_results.csv` to override.

## Stage 2: preprocessing (standard images only)

1. Read stored values; invert MONOCHROME1, or when a flat image edge is bright.
2. Resample to `--target-spacing` (0.1 mm) from ImagerPixelSpacing / PixelSpacing.
   No valid tag: not resampled, `spacing_unknown = True`.
3. Segmentation proxy: background level → 0, 99.9th percentile → 1.
4. Threshold (0.05) → opening (1.5 mm) → largest component → fill holes.
5. `breast_frac` = breast pixels / all pixels of the uncropped, resampled image.
   Outside `--breast-frac-range` (0.10–0.75): redo with flat-edge detection (rows/columns whose
   5th–95th percentile spread < `--flat-tol` are background). Still outside: `breast_frac_flag = out_of_range`
   and a thumbnail copy in `preprocessed/thumbnails_breast_frac_out_of_range/`. These images are kept in the JSON.
6. Pixels outside the mask → 0.

8. Crop to the breast box + 5 mm air.
9. Window 0.5–99.5th percentile of breast pixels → [0, 1].

Outputs: `preprocessed/png/` (uint8 = round(x × 255)), `preprocessed/npz/` (float32, key `image`,
plus `pixel_spacing_mm`), `preprocessed/thumbnails/` (256 × 256), `process_log.jsonl` and
`preprocess_results.csv` (one record per image, including failures).

## JSON (`test.json`)

```json
{"data_list": [{"img_path": ".../png/P001_<uid>.png", "npz_path": ".../npz/P001_<uid>.npz",
  "patient_id": "P001", "image_id": "<uid>", "dicom_path": "...",
  "gt_label": {"N": 1, "tumor_size": 22.0}, "clinic_vars": null,
  "breast_frac": 0.33, "breast_frac_flag": "ok"}]}
```

`N`: LNM P/N or 1/0 → 1/0. Missing labels are −999, as before. `<uid>` = sha1(patient_id|dicom_path)[:16].
