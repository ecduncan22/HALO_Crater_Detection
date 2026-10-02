# HALO Crater Detection: Master Project Document

> This is the main tracking document for the project. It records what we're building, what the model expects, the decisions we've made, what's still open, and where each part stands.
> **Update it whenever a decision is made or a milestone moves.** The newest entries in the Decision Log and Run Log go at the top.

| | |
|---|---|
| **Repo** | https://github.com/ecduncan22/HALO_Crater_Detection |
| **Owner** | Erik Duncan |
| **Started** | 2026-10-02 |
| **Last updated** | 2026-10-02 |
| **Current phase** | Phase 0: Project setup |

---

## 1. Goal

Build a repo that:

1. **Takes in VHR imagery** (Planet SkySat or Vantor) as a single `.tif` or a directory of them, read either locally or from an **S3 bucket**.
2. **Preprocesses** the imagery so it matches the data type, band order, resolution and normalization the crater detection models were trained on.
3. **Runs inference** with the trained U-Net models and writes **crater detection polygons** as shapefiles (GeoPackage optional).
4. **Provides a dashboard** to view full scenes with their detections. This is a quick visual check, built last.
5. **Evaluates** the models against the ground-truth validation samples. Key question: *is the Vantor model (more labels, longer training) good enough on SkySat imagery, or do we need the separate SkySat model?*

**Deployment target:** the repo is cloned onto an **Azure VM (CPU only for now)**. The data stays in an **S3 bucket** hosted elsewhere.

---

## 2. Status at a glance

| Phase | What | Status |
|---|---|---|
| 0 | Project setup: master doc, repo skeleton, Python environment | 🟡 In progress |
| 1 | Port notebook inference into a clean module and reproduce the original results (parity test) | ⚪ Not started |
| 2 | Sensor-specific preprocessing (SkySat, Vantor) | ⚪ Not started |
| 3 | Scalable full-scene inference (windowed, overlap, CPU-tuned) | ⚪ Not started |
| 4 | Post-processing and shapefile output | ⚪ Not started |
| 5 | Run on the two test scenes (visual check) | ⚪ Not started |
| 6 | Evaluation on validation samples, SkySat model vs. Vantor model | ⚪ Not started |
| 7 | Cloud: VM provisioning, S3 read/write, batch runs | ⚪ Not started |
| 8 | Dashboard | ⚪ Not started |

Legend: ⚪ not started · 🟡 in progress · 🟢 done · 🔴 blocked

---

## 3. Data and model inventory

| Item | Location | Notes |
|---|---|---|
| Original training code (Brandt et al. 2020 U-Net lineage) | `Z:\Base_Unet_Notebook\UNET_notebooks` | Notebooks 1 (preprocessing), 2 (training), 3 (raster analysis/inference), plus `core/`, `oldcore/`, `config/` |
| Training labels (shapefiles) | `Z:\2014_Thesis\Training_Shapefiles` | ⚠️ Not yet readable by Claude. Needs to be connected via the folder picker. |
| Training imagery | `Z:\2014_Imagery_Reprocessing\2010\RGB` | ⚠️ Not yet readable. Needed to confirm GSD, bit depth and band order. |
| Validation imagery and GT labels | `Z:\Full_Ukraine_Project\Master_Validation_2023\Validation_Samples_2024` | ⚠️ Not yet readable. Could be copied locally because the Z: network drive is very slow. |
| Model #1 | `C:\Projects\UMD\HALO_2026\models\2022_crater_model.h5` | 415 MB, dated 2022-06-29. **Which model is this (SkySat or Vantor)? Still to confirm.** |
| Model #2 | n/a | ⏳ To be provided |
| Test scene: SkySat | n/a | ⏳ To be uploaded |
| Test scene: Vantor | n/a | ⏳ To be uploaded |

**Rule:** imagery, model weights and outputs never go into git (see `.gitignore`). The repo holds code, configs and docs only.

---

## 4. What the model expects (training specification)

This section was reconstructed from the original code on 2026-10-02. Items marked **(verify)** still need to be checked against the training imagery itself.

### 4.1 Model

| Property | Value | Source |
|---|---|---|
| Architecture | U-Net, 64 base filters, 5 levels (64→1024), BatchNorm, MaxPool, UpSampling | `core/UNet.py` |
| Framework | TensorFlow / Keras, saved as full `.h5` (Keras 2.4.0) | `.h5` attributes |
| Input | `(256, 256, 3)` float32, channels-last | `.h5` model config |
| Output | `(256, 256, 1)` sigmoid probability per pixel (semantic segmentation, not boxes) | `.h5` model config |
| Loss | Weighted Tversky (α=0.6, β=0.4). Boundary pixels weighted 10×. | `core/losses.py` |
| Custom objects needed to load | `tversky, dice_coef, dice_loss, accuracy, specificity, sensitivity` (or load with `compile=False`) | notebook 3 |

### 4.2 Input preparation

| Step | What training/inference did | Implication for us |
|---|---|---|
| **Bands** | 3 channels: raster bands 1, 2, 3 of the RGB training imagery, in file order. The SkySat training run listed channels as `Red, Green, Blue`. Old inference code built WorldView inputs as **R, G, B** (4-band: B=1, G=2, R=3; 8-band: B=2, G=3, R=5). | Model input order is **R, G, B** **(verify against the training imagery)**. SkySat Analytic (B,G,R,N) and Vantor multispectral products need reordering. |
| **Normalization** | **Per-channel z-score:** `(x − mean) / (std + 1e-8)`. Training: always z-scored per band over the whole training area, then re-z-scored per 256-px patch 40% of the time. Inference: z-scored **per 256×256 tile, per channel**. | We reproduce inference exactly: per-tile, per-channel z-score. Because of this, absolute bit depth (8 vs 11 vs 16 bit) matters less. **Contrast/stretch and colour balance still matter** (e.g. visual vs. analytic products). |
| **Bit depth** | Unknown **(verify)**. Training imagery folder is `RGB`, which suggests 8-bit visual-type products. | Decide whether we feed analytic (linear) or visual (stretched) products. Evaluation will tell us which works better. |
| **Ground sample distance (GSD)** | Unknown **(verify)** | **Highest-risk unknown.** Crater size in pixels must match training. Resample inputs to the training GSD if they differ. |
| **Tile size** | 256 × 256 | Fixed by the model. |
| **Tile overlap at inference** | Config said stride 196, but the run actually used **stride 256 (no overlap)**, merging with pixel-wise **MAX**. | We'll use overlap (e.g. stride 192) to avoid seam artifacts, then compare against no overlap. |
| **Edges / nodata** | Edge tiles zero-padded *after* normalization. Nodata pixels were *included* in mean/std. | Improvement: mask nodata out of the mean/std so black borders don't skew tiles. |
| **Augmentation (training only)** | Flips, crop ≤10%, linear contrast 0.3–1.2, piecewise affine, perspective | Explains some robustness to contrast. Not used at inference. |

### 4.3 Post-processing in the original code

- Threshold probability at **0.5**, giving a binary mask.
- `cv2.findContours` (RETR_TREE) turns the mask into polygons, with holes kept; coordinates are converted with the raster transform.
- Shapefile attributes: `id`, area (field was named `canopy`, inherited from the tree-counting code).
- Output CRS = the input image CRS.

### 4.4 Notes and risks from the code review

- The training script set `training_frames = validation_frames = testing_frames` (all the same areas), so **the validation metrics logged during training are not held-out**. The independent validation set (Section 3) is essential for honest numbers.
- The SkySat training notebook loaded only **one training frame** (`Red_0.png`).
- The original inference kept the full-scene probability mask in memory as float32. For a 35k × 42k scene that's about 6 GB. We'll write results tile by tile instead.
- The `.h5` is 415 MB because it includes the optimizer state. A weights-only or ONNX export would be smaller and possibly faster on CPU.
- Keras 2.4-era `.h5` files load most reliably with **TensorFlow ≤ 2.15** (Keras 2). TF ≥ 2.16 defaults to Keras 3. Pin TF 2.15, or use `tf-keras`.

---

## 5. Sensor notes

To be confirmed against the actual test scenes when they arrive.

| | SkySat | Vantor |
|---|---|---|
| Likely products | Ortho Visual (3-band RGB, 8-bit) or Ortho Analytic SR (4-band B,G,R,NIR, 16-bit) | Pansharpened RGB/4-band, or 8-band multispectral (WorldView / Legion) |
| Nominal GSD | ~0.5 m ortho | ~0.3 m pansharpened |
| Band mapping to model R,G,B | Visual: 1,2,3 · Analytic: 3,2,1 | 4-band: 3,2,1 · 8-band: 5,3,2 |
| Model to use | SkySat model, and test whether the Vantor model is good enough | Vantor model |

Band mappings and target GSD will live in per-sensor config files (`configs/sensors/*.yaml`), not in code.

---

## 6. Proposed architecture

### 6.1 Pipeline

```
input (.tif | directory | s3://...)
   │
   ▼
[io]          list scenes → fetch to local disk (S3) → open with rasterio
   │
   ▼
[preprocess]  sensor config → select/reorder bands to R,G,B → mask nodata
              → resample to training GSD (if needed)
   │
   ▼
[inference]   windowed tiling (256 px, overlap) → per-tile per-channel z-score
              → batched model.predict (CPU) → merge (MAX) → probability GeoTIFF (COG)
   │
   ▼
[postprocess] threshold → polygonize → filter (min area, etc.) → attributes
              (area_m2, equiv_diam_m, mean_prob, max_prob, scene_id, model)
              → Shapefile (+ GeoPackage)
   │
   ▼
[outputs]     local outputs/ dir → optionally upload back to S3
   │
   ├─► [evaluate]  compare to GT polygons → precision / recall / F1 (object level)
   └─► [dashboard] scene + detections viewer
```

### 6.2 Proposed repo layout

```
HALO_Crater_Detection/
├── MASTER_PROJECT.md          ← this document
├── README.md                  ← install + quick start
├── pyproject.toml / requirements.txt
├── configs/
│   ├── default.yaml           ← tile size, stride, threshold, batch size, paths
│   ├── sensors/{skysat,vantor}.yaml
│   └── models.yaml            ← model registry: name → file, sensor, training GSD, checksum
├── src/halo_craters/
│   ├── io/                    ← local + S3 read/write
│   ├── preprocess/            ← band mapping, nodata, resampling, normalization
│   ├── inference/             ← model loading, tiling, merging
│   ├── postprocess/           ← polygonize, filter, write vectors
│   ├── evaluate/              ← metrics vs ground truth
│   ├── dashboard/
│   └── cli.py                 ← `halo-craters run --input ... --sensor skysat --model vantor`
├── scripts/                   ← VM setup, model download
├── tests/                     ← incl. parity test vs original notebook output
├── data/      (gitignored)
├── models/    (gitignored)
└── outputs/   (gitignored)
```

### 6.3 Software stack (proposed)

- Python 3.10 or 3.11, in a virtual environment with a pinned `requirements.txt`
- `tensorflow-cpu==2.15.*` (Keras 2, loads the legacy `.h5`). Option to try ONNX Runtime later for faster CPU inference.
- `rasterio`, `numpy`, `geopandas`, `shapely>=2`, `pyogrio`, `opencv-python-headless`
- `boto3` for S3
- `pyyaml` for configs, `tqdm` for progress bars
- Dashboard: to be decided (likely `leafmap`/`folium` with COG tiles, or a static HTML report)

---

## 7. Cloud and infrastructure plan

### 7.1 Storage (S3, accessed from Azure)

- The VM reads from the S3 bucket with **AWS credentials** (an access key pair or IAM user scoped to the bucket). These are stored on the VM in `~/.aws/credentials` or as environment variables. **Never committed to git.**
- Approach: **download each scene to the VM's local disk → process → upload results.** This is simpler and more reliable than streaming reads (`/vsis3/`), and lets us re-run without re-downloading.
- Cost note: AWS charges **egress** when data leaves S3 for Azure. This is negligible for a few scenes but worth knowing at scale.
- Bucket layout to agree on, for example:
  `s3://<bucket>/imagery/{skysat,vantor}/...`, `s3://<bucket>/models/`, `s3://<bucket>/outputs/<run_id>/`

### 7.2 Models

**Recommendation:** keep the weights out of git and store them in the **S3 bucket under `models/`**, so there's one source of truth. A `scripts/download_models.sh` pulls them to `models/` on the VM. `configs/models.yaml` records each model's filename, sensor, training GSD and **SHA-256 checksum**, so we always know exactly which weights produced a result. A one-time manual upload to the VM also works as a fallback.

### 7.3 VM sizing (initial recommendation, to be confirmed by benchmark)

A CPU U-Net at 256×256 with 64 base filters is compute-heavy. A full scene can be tens of thousands of tiles, so **cores matter most, then RAM.**

| Option | vCPU / RAM | Notes |
|---|---|---|
| `Standard_D16s_v5` | 16 / 64 GiB | Balanced general-purpose. **Suggested starting point.** |
| `Standard_D16ds_v5` | 16 / 64 GiB + local temp SSD | Same as above, plus fast local scratch for staging scenes |
| `Standard_F16s_v2` | 16 / 32 GiB | Compute-optimized, cheaper. RAM could be tight. |

- OS: **Ubuntu 22.04 or 24.04 LTS**
- Disk: Premium SSD data disk, **≥ 256–512 GB** (scenes, probability rasters, outputs)
- **Before provisioning:** we'll benchmark tiles/second on one test scene locally, then size the VM from that measurement rather than guessing.
- Cost control: deallocate (stop) the VM when idle. You pay for compute only while it's running.

---

## 8. Evaluation plan

- **Test scenes (1 SkySat, 1 Vantor):** visual check of detections overlaid on imagery.
- **Validation samples** (`Validation_Samples_2024`, with GT): object-level metrics.
  - Match predicted polygons to GT by IoU ≥ 0.3–0.5 (threshold to be agreed) or by centroid-in-polygon
  - Report precision, recall, F1, counts per sample, and pixel-level IoU
- **Experiment matrix:**

| Imagery | Model | Purpose |
|---|---|---|
| SkySat | SkySat model | Baseline |
| SkySat | Vantor model | **Is the Vantor model sufficient for SkySat?** |
| Vantor | Vantor model | Baseline |
| (optional) SkySat resampled to Vantor GSD | Vantor model | Does matching resolution help? |

- Also sweep: probability threshold (0.3–0.7), tile overlap, analytic vs. visual product.

---

## 9. Open questions

| # | Question | Owner | Status |
|---|---|---|---|
| Q1 | Which model is `2022_crater_model.h5`: SkySat or Vantor? When will the second model be available? | Erik | Open |
| Q2 | GSD, bit depth and band order of the training imagery (`Z:\2014_Imagery_Reprocessing\2010\RGB`) and the training imagery for each model | Claude, once the folder is connected | Open |
| Q3 | Connect the training labels, training imagery and validation folders to the session via the folder picker (or copy them locally, since the Z: drive is slow) | Erik | Open |
| Q4 | Which SkySat / Vantor product types will we receive (visual vs. analytic, 3/4/8-band, pansharpened)? | Erik | Open |
| Q5 | S3 bucket: name, region, who owns it, and do we get read-only or read/write access (for writing outputs)? | Erik | Open |
| Q6 | Azure subscription / resource group / region for the VM | Erik | Open |
| Q7 | Typical crater size range (m) and minimum size worth reporting, for the post-processing filter | Erik | Open |
| Q8 | Output format: shapefile only, or also GeoPackage / GeoJSON? Required attribute fields? | Erik | Open |

---

## 10. Decision log

*(newest first)*

| Date | Decision | Why |
|---|---|---|
| 2026-10-02 | Model weights stay out of git. They'll be stored in the S3 bucket `models/` prefix, with checksums in `configs/models.yaml`. | Large binaries don't belong in git. One source of truth, and every result traces back to exact weights. |
| 2026-10-02 | Inference reproduces the original **per-tile, per-channel z-score** normalization with **256 px** tiles and **R,G,B** input. | Matches how the model was trained and run (see Section 4). |
| 2026-10-02 | Phase 1 is a **parity test** against the original notebook before any improvements. | So we know any later change in results comes from a deliberate change, not a porting bug. |
| 2026-10-02 | Dashboard deferred to the final phase. | Erik: it's a quick visual check, not core functionality. |
| 2026-10-02 | Target runtime: CPU-only Azure VM. Data stays in S3. | Project constraints. |
| 2026-10-02 | This document lives at the repo root as `MASTER_PROJECT.md`, with a copy in the Claude project. | Versioned with the code and visible on the VM and GitHub. |

---

## 11. Run log

*(newest first. One row per meaningful run.)*

| Date | Scene(s) | Sensor | Model | Config / commit | Result / notes |
|---|---|---|---|---|---|
| | | | | | |

---

## 12. Changelog

- **2026-10-02:** Document created. Training spec reconstructed from `Base_Unet_Notebook` code and the `.h5` metadata.
