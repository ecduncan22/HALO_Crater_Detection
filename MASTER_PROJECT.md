# HALO Crater Detection: Master Project Document

> This is the main tracking document for the project. It records what we're building, what the model expects, the decisions we've made, what's still open, and where each part stands.
> **Update it whenever a decision is made or a milestone moves.** The newest entries in the Decision Log and Run Log go at the top.

| | |
|---|---|
| **Repo** | https://github.com/ecduncan22/HALO_Crater_Detection |
| **Owner** | Erik Duncan |
| **Started** | 2026-10-02 |
| **Last updated** | 2026-10-02 |
| **Current phase** | Phase 6 evaluation under way. Now: full P001 scene on Erik's GPU. Next: ensembling, VM |

---

## 1. Goal

Build a repo that:

1. **Takes in VHR imagery** (Planet SkySat or Vantor) as a single `.tif` or a directory of them, read either locally or from an **S3 bucket**.
2. **Preprocesses** the imagery so it matches the data type, band order, resolution and normalization the crater detection models were trained on.
3. **Runs inference** with the trained U-Net models and writes **crater detection polygons** as shapefiles (and GeoPackage).
4. **Provides a dashboard** to view full scenes with their detections. This is a quick visual check, built last.
5. **Evaluates** the models against the ground-truth validation samples. Key question: *is the Vantor model (more labels, longer training) good enough on SkySat imagery, or do we need the separate SkySat model?*

**Deployment target:** the repo is cloned onto an **Azure VM (CPU only for now)**. The data stays in an **S3 bucket** hosted elsewhere.

---

## 2. Status at a glance

| Phase | What | Status |
|---|---|---|
| 0 | Project setup: master doc, repo skeleton, Python environment | 🟢 Done |
| 1 | Port notebook inference into a clean package; parity check vs. original detections | 🟢 Done (see Section 8.1) |
| 2 | Sensor-specific preprocessing (band maps, resolution matching, nodata) | 🟡 Band maps and resampling built; per-sensor defaults to be decided from evaluation |
| 3 | Stable, scalable full-scene inference (streaming, overlap, **tile-offset/flip ensembling**) | 🟡 Streaming + overlap built; ensembling next |
| 4 | Post-processing and shapefile/GeoPackage output | 🟢 Built (strip-wise polygonize, attributes) |
| 5 | Run on the two test scenes + full GT_2 P001 (visual check) | 🟡 GPU route ready (`scripts/windows/`); P001 full-scene run pending on Erik's workstation |
| 6 | Evaluation on validation samples; SkySat model vs. Vantor model | 🟡 Strata-4 sample 14494 + 4 windows of GT_2 P001 (full-res Vantor) evaluated |
| 7 | Cloud: VM provisioning, S3 read/write, batch runs | ⚪ Not started |
| 8 | Dashboard | ⚪ Not started |

Legend: ⚪ not started · 🟡 in progress · 🟢 done · 🔴 blocked

---

## 3. Data and model inventory

| Item | Location | Notes |
|---|---|---|
| Original training code (Brandt et al. 2020 U-Net lineage) | `Z:\Base_Unet_Notebook\UNET_notebooks` | Notebooks 1 (preprocessing), 2 (training), 3 (raster analysis/inference), plus `core/`, `oldcore/` (what training actually imported), `config/` |
| Training labels | `Z:\2014_Thesis\Training_Shapefiles\Crater_Labels.shp` | 18,472 crater polygons, EPSG:32637. Equivalent diameter: median **9.4 m**, 5–95% **4.8–17.1 m**, 99% 21 m |
| Training imagery | `Z:\2014_Imagery_Reprocessing\2010\RGB` | WorldView, 2014-09-05, pansharpened, **uint8, 3 bands stored B,G,R, 0.324 m**, 16384² tiles, UTM 37N |
| Validation samples (all strata) | `Z:\Full_Ukraine_Project\Master_Validation_2023\Validation_Samples_2024` | Strata 1–4 + test. Network drive is slow. |
| Validation Strata 4 (local copy) | `C:\Projects\UMD\HALO_2026\GT\val_samples_2024\strata_4` | 15 samples. Each has: AOI polygon (1 km²), clipped images (Vantor 2022, **uint16, B,G,R, 0.5 m**), `_detected_craters.shp` (original model output, after post-processing, **union of several acquisitions**), `_marked_craters.shp` (**GT**), `_crater_location_points.shp` |
| **GT_2 (full-res Vantor)** | `C:\Projects\UMD\HALO_2026\GT\GT_2` | `imagery/22AUG01081633-…_P001_pansharpened_RGB.tif` (**uint16, B,G,R, 0.25 m**, 74700×64712 px, 38.7 GB, strip layout, ~300 km²) and `…_P007_…` (uploading). `shp/Craters.shp`: 10,314 GT polygons (9,978 inside P001), EPSG:32637, median diameter 8.8 m (5–95%: 3.8–16 m) |
| Other GT | `C:\Projects\UMD\HALO_2026\GT\GT_1` | `MC_In_Grid.shp` + six 2014 reference images (6.9 GB). Not yet used. |
| **Model: `vantor_2022`** | `C:\Projects\UMD\HALO_2026\models\2022_crater_model.h5` | Trained **only on the 2014 `2010\RGB` imagery** (confirmed by Erik). 415 MB (with optimizer state). SHA-256 `e3ce27b7…a916` |
| Model: SkySat | n/a | ⏳ To be provided |
| Test scene: SkySat | `C:\Projects\UMD\HALO_2026\data\skysat_scene\pansharpened_udm2\20220702_074811_ssc12_u0001_pansharpened.tif` | 2022-07-02, **uint16, 4 bands, B,G,R,NIR** (Planet standard order; file tags say R,G,B; to confirm), **0.5 m**, 13637×32405 px, tiled + overviews, 1.1 GB |
| Test scene: Vantor | `C:\Projects\UMD\HALO_2026\data\vantor_scene\22SEP08082313-M1BS-506895608030_01_P004_pansharpened_RGB.tif` | 2022-09-08, **uint16, B,G,R, 0.25 m**, 73155×70455 px, **39 GB, strip layout (1 row/strip)** |

**Rule:** imagery, model weights and outputs never go into git (see `.gitignore`). The repo holds code, configs and docs only.

---

## 4. What the model expects (training specification)

Reconstructed from the original code (`oldcore/` is what training imported), the `.h5` metadata and the training imagery itself. Verified 2026-10-02.

### 4.1 Model

| Property | Value | Source |
|---|---|---|
| Architecture | U-Net, 64 base filters, 5 levels (64→1024), BatchNorm, MaxPool, UpSampling; 34.5 M parameters | `UNet.py`, `.h5` |
| Framework | TensorFlow / Keras, full `.h5` (Keras 2.4.0). Loads fine in **TF 2.15.1** with `compile=False` | tested |
| Input | `(256, 256, 3)` float32, channels-last | `.h5` config |
| Output | `(256, 256, 1)` sigmoid probability per pixel (semantic segmentation) | `.h5` config |
| Loss | Weighted Tversky (α=0.6, β=0.4), boundary pixels weighted 10× | `losses.py` |
| CPU speed | ~1.3 s per 256² tile on 2 cores (dev container). Scales roughly with cores. | measured |

### 4.2 Input preparation

| Step | Training / original inference | What we do |
|---|---|---|
| **Bands** | Training fed **file order** of the training imagery, which is **Blue, Green, Red**. All Vantor `_RGB` products we have also store **B,G,R** (confirmed by Erik), so the original pipeline always fed B,G,R. | `configs/models.yaml` declares `input_bands: [blue, green, red]`; `configs/sensors.yaml` maps names → file bands per sensor (`vantor` = B,G,R → file bands 1,2,3). |
| **Normalization** | Per-channel z-score `(x − mean)/(std + 1e-8)`. Training: per training area, plus per-patch 40% of the time. Inference: **per 256×256 tile, per channel.** | Same, per tile. Optionally excludes nodata from the stats (`mask_nodata`, default on). |
| **Bit depth** | Training imagery is **uint8**. | Optional scene-level 2–98% stretch to 0–255 (`rescale: percentile`). Tested: **no benefit** on sample 14494 (Section 8.1). Default off. |
| **GSD** | Training imagery **0.324 m**. | `target_gsd_m` resamples on the fly (bilinear). **Matching 0.324 m matters a lot** (Section 8.1). |
| **Tiling** | 256 px; the notebook actually ran stride 256 (no overlap), MAX merge, zero-padded edges. | Stride 192 default with **center-keep** merge (each pixel from the tile where it's most central); last tile shifted to the edge instead of padded. `--parity` reproduces the notebook. |
| **Augmentation** | Flips, crop ≤10%, linear contrast 0.3–1.2, piecewise affine, perspective (training only). | n/a. Flips make test-time flip-averaging a natural fit. |

### 4.3 Post-processing

- Original: threshold **0.5** → `cv2.findContours` → polygons (holes kept), image CRS, `id` + area attribute.
- Ours: threshold 0.5 → 8-connected components → polygons with exact pixel boundaries. Done in overlapping strips, so scenes of any size work. Attributes: `crater_id, n_pix, mean_prob, max_prob, area_m2, diam_m, source_im, model, sensor`. Optional `min_area_m2` filter.
- The original `_detected_craters.shp` in the validation samples **also had extra post-processing**: forest/urban masking, merging across images. Not reproduced yet (see Q9).

### 4.4 Notes and risks

- **The model's output depends heavily on tile placement.** On identical input, stride 256 vs. stride 192 agree on only ~60% of craters (F1 0.59), and missed craters aren't borderline (our max probability < 0.1 in 89% of them). Cause: per-tile normalization plus edge effects. **Planned fix:** average probabilities over several tile offsets and flips (test-time augmentation). This should make results stable and probably more accurate. It's the top item for Phase 3.
- Training used `training = validation = test` frames, so the training-time metrics weren't held out. Only the validation samples give honest numbers.
- The original full-scene inference held a float32 mask in memory (~6 GB for 35k×42k). We stream row strips instead, so memory stays bounded by one strip.
- The `.h5` includes optimizer state (415 MB). A weights-only copy is 138 MB and gives identical predictions. ONNX export is still an option for faster CPU inference.

---

## 5. Sensor notes (verified on the actual files)

| | SkySat test scene | Vantor test scene | Vantor validation clips | Training imagery |
|---|---|---|---|---|
| dtype | uint16 | uint16 | uint16 | uint8 |
| Bands in file | **B, G, R, NIR** (to confirm) | **B, G, R** | **B, G, R** | **B, G, R** |
| GSD | 0.5 m | 0.25 m | 0.5 m | 0.324 m |
| `--sensor` | `skysat` | `vantor` | `vantor` | `worldview_2014_training` |
| Crater (9.4 m) in pixels | 19 px | 38 px | 19 px | 29 px |

⚠️ Don't trust colour-interpretation tags in these files, and don't judge band order by eye on **raw 16-bit** data. Raw DN is naturally highest in blue, so a *correctly* ordered raw image looks bluish and a swapped one can look 'natural'. That misled the first check of the Vantor files (corrected 2026-10-02 by Erik: they are B,G,R). Quicklooks now use the sensor band map plus a per-band stretch (`scripts/quicklook.py --sensor vantor`) to show true colour.

---

## 6. Architecture (as built)

### 6.1 Pipeline

```
input (.tif | directory)                        [S3: Phase 7]
   │
   ▼
config        configs/{default,sensors,models}.yaml → band order, GSD, tiling, threshold
   │
   ▼
inference     open (optionally resampled to target GSD via WarpedVRT)
              → stream one row-strip of tiles at a time
              → per tile: nodata mask → [optional 8-bit stretch] → per-channel z-score
              → batched U-Net predict (CPU) → center-keep merge
              → uint8 probability Cloud-Optimized GeoTIFF
   │
   ▼
postprocess   threshold → connected components (overlapping strips) → polygons + attributes
              → GeoPackage + Shapefile, run.json (config, model, commit, timings)
   │
   ├─► evaluate   one-to-one IoU matching vs. reference polygons → P / R / F1
   └─► dashboard  (Phase 8)
```

### 6.2 Repo layout

```
HALO_Crater_Detection/
├── MASTER_PROJECT.md            ← this document
├── README.md                    ← install + quick start
├── pyproject.toml               ← package + pinned key deps; `halo-craters` CLI
├── configs/
│   ├── default.yaml             ← tiling, threshold, GSD, rescale, output formats
│   ├── sensors.yaml             ← band name → file band index per sensor
│   └── models.yaml              ← model registry (path, sha256, input bands, native GSD)
├── src/halo_craters/
│   ├── config.py  tiling.py  preprocess.py  model.py
│   ├── inference.py  postprocess.py  evaluate.py  pipeline.py  cli.py
├── scripts/compare_sample.py    ← compare runs vs. a validation sample's originals + GT
├── tests/                       ← tiling coverage, normalization parity, strip polygonize
├── data/  models/  outputs/     (gitignored)
```

### 6.3 Usage

```bash
halo-craters run path/to/scene_or_dir --sensor vantor --model vantor_2022 --target-gsd 0.324 --out outputs/
halo-craters evaluate outputs/<scene>/<scene>_craters.gpkg sample_marked_craters.shp --aoi sample.shp
```

### 6.4 Software stack

Python 3.11 venv · `tensorflow-cpu==2.15.1` · `numpy<2` · rasterio · geopandas/pyogrio · shapely 2 · scipy · pyyaml · tqdm · (boto3 for S3)

---

## 7. Cloud and infrastructure plan

### 7.1 Storage (S3, accessed from Azure)

- The VM reads S3 with **AWS credentials** scoped to the bucket, stored in `~/.aws/credentials` or environment variables. **Never committed to git.**
- Approach: **download each scene to the VM's local disk → process → upload results.** This also matters because some products (e.g. the 39 GB Vantor scene) use a strip layout that's slow to read remotely.
- Cost note: AWS charges **egress** for data leaving S3 to Azure.
- Proposed bucket layout: `s3://<bucket>/imagery/{skysat,vantor}/…`, `s3://<bucket>/models/`, `s3://<bucket>/outputs/<run_id>/`

### 7.2 Models

Weights stay out of git. They'll be stored in the bucket under `models/`, and `configs/models.yaml` records each model's SHA-256. A `scripts/download_models.sh` will come in Phase 7.

### 7.3 VM sizing (to confirm by benchmark)

Measured: ~1.3 s/tile on 2 cores. Rough tile counts with stride 192 (×4 with a 4-way ensemble):

| Scene | At native GSD | At 0.324 m |
|---|---|---|
| SkySat test (0.5 m, 13.6k×32.4k) | ~12k tiles | ~29k tiles |
| Vantor test (0.25 m, 73k×70k) | ~140k tiles | ~84k tiles |

On 16 cores (~0.16 s/tile if scaling holds), a resampled SkySat scene takes ~1.3 h, and the Vantor scene ~4 h per ensemble member. **CPU is workable for occasional scenes but slow for routine ensembled runs.** Keep a GPU VM, or ONNX/int8 optimization, in mind. Suggested start: `Standard_D16s_v5` (16 vCPU / 64 GiB), Ubuntu 22.04/24.04, ≥512 GB Premium SSD. Deallocate when idle.

---

## 8. Evaluation

### 8.1 Phase 1 parity check: validation sample 14494 (Strata 4)

AOI 1 km². GT = 559 craters. Original `_detected_craters` in the AOI = 989, a **union of two acquisitions**: 22JUN13 (520) and 22JUN09 (459). Input: the 22JUN13 clip (0.5 m, uint16). Matching: one-to-one, IoU ≥ 0.1. Runs are scored against the original detections *from the same acquisition* and against GT.

| Run | GSD | Bands fed | n | vs. original (P / R) | vs. GT P | vs. GT R | vs. GT F1 |
|---|---|---|---|---|---|---|---|
| **ORIGINAL (22JUN13 detections)** | full-res strips | ? | **520** | n/a | **0.51** | **0.47** | **0.49** |
| parity, file order = **B,G,R (native)** | 0.5 | 1,2,3 | 143 | 0.55 / 0.15 | 0.77 | 0.20 | 0.31 |
| parity, reversed = R,G,B | 0.5 | 3,2,1 | 210 | 0.44 / 0.18 | 0.64 | 0.24 | 0.35 |
| **parity, B,G,R (native)** | **0.324** | **1,2,3** | 340 | 0.49 / 0.32 | 0.66 | 0.40 | 0.50 |
| parity, reversed = R,G,B | 0.324 | 3,2,1 | **533** | 0.36 / 0.37 | 0.49 | **0.47** | **0.48** |
| parity, R,G,B + 8-bit stretch | 0.324 | 3,2,1 | 401 | 0.44 / 0.34 | 0.58 | 0.42 | 0.49 |
| parity, B,G,R + 8-bit stretch | 0.324 | 1,2,3 | 295 | 0.52 / 0.30 | 0.73 | 0.38 | 0.50 |
| default (stride 192, center-keep, nodata masked), R,G,B | 0.324 | 3,2,1 | 412 | n/a | 0.58 | 0.43 | 0.49 |

**Conclusions**
1. **Resolution is the dominant factor.** At 0.5 m the model misses most craters (R≈0.2). Resampling to the 0.324 m training GSD more than doubles recall.
2. **At 0.324 m our pipeline matches the original run's numbers** (closest count with the reversed order, closest accuracy with either): 533 vs. 520 detections, the same median size (61 vs. 64 m²), and the same accuracy vs. GT (F1 0.48 vs. 0.49). So results are **not "very different."**
3. **Individual polygons only partly coincide** (~37% one-to-one) because (a) the original ran on the full-resolution source strips, not these resampled 0.5 m clips, and (b) the model is very sensitive to tile placement (~60% agreement between two tilings of the *same* input). Exact polygon parity would require the original source scenes (Q10).
4. B,G,R (native) vs. R,G,B: similar F1. Native B,G,R gives higher precision, reversed R,G,B higher recall. 8-bit stretching gave no clear gain. *(Labels corrected 2026-10-02: originally reported with the band names swapped.)*
5. Georegistration: detections (ours and original) sit within ~2 m of GT on 22JUN13. The 22JUN09 clip is offset ~6 m from GT, which hurts its scores (it's also listed in `raster_errors.txt`).

*Caveat: this is one sample. Results must be confirmed on all 15 Strata-4 samples.*

### 8.2 GT_2: full-resolution Vantor scene P001 (2022-08-01, 0.25 m)

The full scene would take more than a day on the 2-core dev container, so four 1 km × 1 km windows (4000×4000 px) were cut from P001 with their GT. Defaults otherwise: stride 192, center-keep, nodata masked. Matching IoU ≥ 0.1. Scripts: `scripts/evaluate_windows.py`, `scripts/quicklook.py`.

| Window | Lower-left (UTM 37N) | GT n | GT median diam. | Character |
|---|---|---|---|---|
| w1_dense | 435373, 5373041 | 526 | 10.1 m | Crater field in cultivated land |
| w2_dense_south | 435373, 5363041 | 249 | **5.4 m (42% < 5 m)** | Small craters; woodland, farm buildings |
| w3_medium | 435373, 5369041 | 99 | 11.2 m | Fields + rough grassland + bright chalky ground |
| w4_sparse | 424373, 5372041 | 20 | 12.4 m | Fields, few craters |

**Results (0.324 m, bands fed reversed = R,G,B, from before the band-order correction; no masking, no min-area):**

| Window | Pred | P | R | F1 |
|---|---|---|---|---|
| w1_dense | 652 | 0.65 | **0.81** | **0.72** |
| w2_dense_south | 108 | 0.35 | 0.15 | 0.21 |
| w3_medium | 178 | 0.45 | **0.81** | 0.58 |
| w4_sparse | 44 | 0.36 | 0.80 | 0.50 |
| **Pooled** | 982 | 0.57 | 0.63 | 0.60 |
| Pooled, min-area 30 m² | 838 | 0.65 | 0.61 | 0.63 |

**Settings comparison (pooled):**

| Setting | w1 + w3: P / R / F1 | w2 (small craters): P / R / F1 |
|---|---|---|
| 0.324 m, reversed R,G,B | 0.61 / 0.81 / **0.70** | 0.35 / 0.15 / 0.21 |
| **0.324 m, native B,G,R (now the default)** | 0.66 / 0.77 / **0.71** | not run |
| native 0.25 m, reversed R,G,B | 0.52 / 0.84 / 0.64 | 0.37 / **0.33** / **0.34** |

**Recall by GT diameter (0.324 m, reversed R,G,B, all 4 windows):** < 5 m: 0.06 · 5–8 m: 0.45 · 8–12 m: 0.81 · 12–16 m: 0.82 · > 16 m: 0.69

**Conclusions**
1. **On full-resolution imagery the model works well for craters ≥ 8 m:** recall ≈ 0.8 in crater fields, F1 ≈ 0.7 where the land is mostly cultivated.
2. **Small craters (< 5–8 m) are largely missed.** They're rare in the training labels (5th percentile 4.8 m). Running at native 0.25 m doubles small-crater recall but adds many false positives elsewhere. A size-aware setup (e.g. combining both resolutions) is worth testing; retraining with small craters would be the real fix.
3. **False positives cluster in woodland, on building roofs, and in rough grassland / bright chalky bare ground.** Woodland and roofs are clear errors that masking would remove (Q9). Many detections in the rough grassland look like pale circular disturbances very similar to labelled craters, so **some may be unlabelled craters** (Q12). Precision may be understated there.
4. **Band order is close to a wash** (native B,G,R F1 0.71 vs. reversed R,G,B 0.70). Native gives more precision, reversed more recall. We use the native B,G,R, matching training and the original pipeline.
5. A **min-area filter of ~30 m²** (≈ 6 m diameter) raises precision by ~0.08 at a recall cost of ~0.015. **Adopted: 25 m² default** (Q7).
6. Compared with the 0.5 m validation clips (Section 8.1, F1 ≈ 0.49), full-resolution input is clearly better. **Production should use full-resolution imagery, not resampled clips.**

### 8.3 Planned evaluation

- Run all Strata-4 samples (then other strata) with a fixed config. Report P/R/F1 per sample and pooled.
- **Experiment matrix:**

| Imagery | Model | Purpose |
|---|---|---|
| SkySat | SkySat model | Baseline |
| SkySat (resampled to 0.324 m) | Vantor model | **Is the Vantor model sufficient for SkySat?** |
| Vantor | Vantor model | Baseline |

- Sweeps: target GSD (0.3–0.4), threshold (0.3–0.7), band order, **tile-offset/flip ensembling**, min-area filter.
- Note: original detections had forest/urban masks applied. Our raw output will have more false positives in those areas until a masking step exists (Q9).

---

## 9. Open questions

| # | Question | Owner | Status |
|---|---|---|---|
| Q1 | Which model is `2022_crater_model.h5`? | Erik | ✅ Vantor model, trained only on `2010\RGB` |
| Q2 | Training imagery GSD / bit depth / band order | Claude | ✅ 0.324 m, uint8, B,G,R |
| Q3 | Connect labels, training imagery and validation folders | Erik | ✅ Done (Strata 4 copied locally) |
| Q4 | Which SkySat / Vantor product types will we receive in production? Are they always like the test scenes? | Erik | Open |
| Q5 | S3 bucket: name, region, owner, read-only or read/write? | Erik | Open |
| Q6 | Azure subscription / resource group / region for the VM | Erik | Open |
| Q7 | Minimum crater size worth reporting (for `min_area_m2`) | Erik | ✅ **25 m²** (≈ 5.6 m diameter), now the default |
| Q8 | Output format: shapefile + GeoPackage OK? Required attribute fields? | Erik | Open |
| Q9 | Forest/urban masking: which masks were used for the validation detections, and should the repo apply them? | Erik | 🟡 Erik has a **field-boundary layer** to clip detections to fields. Whether to apply it is up to the **end user**. Plan: optional `--clip-to` step. |
| Q10 | Are the original full-resolution 22JUN13/22JUN09 source strips available (for exact parity)? Not essential. | Erik | Open |
| Q11 | When will the SkySat model be available, and what was it trained on (GSD, bands, bit depth)? | Erik | Open |
| Q12 | Was GT_2 `Craters.shp` labelled **exhaustively** across the whole P001/P007 scenes, or only in some land types/areas (e.g. cultivated fields)? Affects how to read precision. | Erik | ✅ GT is likely **incomplete** (misses a fair number of craters). Treat precision as a lower bound; don't over-tune for it. |
| Q13 | Run the full P001 scene: on a bigger machine (Erik's workstation or a trial VM)? ~78k tiles at 0.324 m (≈ 3.5 h on 16 cores, ≈ 28 h on 2). | Erik | ✅ Run on **Erik's workstation GPU** via `scripts/windows/` (TF 2.10.1 GPU env), then develop the VM. |

---

## 10. Decision log

*(newest first)*

| Date | Decision | Why |
|---|---|---|
| 2026-10-02 | Use Erik's Windows workstation GPU for full-scene runs during development: conda env `halo-gpu` with TF 2.10.1 (last native-Windows GPU release), double-click scripts in `scripts/windows/`, logs in `_scratch/`. TF made an install extra (`[cpu]` / `[gpu]`). | Full P001 is ~78k tiles: too slow on 2 cores. Erik's GPU is available now; the VM comes after. |
| 2026-10-02 | **Min-area default 25 m².** Field-boundary clipping optional, pending the end user. | Erik. Raises precision at little recall cost. |
| 2026-10-02 | Vantor sensor map corrected to **B,G,R** (file bands 1,2,3); quicklooks show true colour (R,G,B display, per-band stretch). | Erik confirmed the files are B,G,R. The earlier visual check was misled by raw 16-bit colour. |
| 2026-10-02 | Keep **0.324 m** as the working default for the Vantor model. Feed native B,G,R; reversed order is statistically equivalent. | GT_2 windows: 0.324 m beats native 0.25 m on F1 (0.70 vs 0.64) except for very small craters. |
| 2026-10-02 | The Vantor model is fed **B,G,R** (training file order) by default, mapped by band name per sensor. | Training imagery stores B,G,R and training fed file order. |
| 2026-10-02 | Resample inputs to the model's **native 0.324 m** (`--target-gsd 0.324`). To become the per-model default after multi-sample confirmation. | Recall roughly doubles vs. 0.5 m on sample 14494, matching the original run's accuracy. |
| 2026-10-02 | Phase 1 parity accepted at the **aggregate** level (count, size, accuracy vs. GT), not polygon-for-polygon. | Original inputs aren't available here, and the model is tile-placement sensitive (Section 8.1). |
| 2026-10-02 | Next priority: **test-time ensembling** (multiple tile offsets + flips, averaged probabilities). | Removes the tile-placement instability. Likely improves accuracy. |
| 2026-10-02 | Default tiling: stride 192 with center-keep merge, shifted edge tiles, nodata excluded from tile stats. `--parity` keeps the notebook behaviour available. | Avoids tile-edge and padding artifacts and keeps memory bounded. |
| 2026-10-02 | Probability output as uint8 COG; vectors as GeoPackage + Shapefile; a `run.json` per scene. | Small, viewable, and every result is traceable to config + commit + model. |
| 2026-10-02 | TensorFlow pinned to `tensorflow-cpu==2.15.1`, Python 3.10/3.11. | Loads the Keras-2.4 `.h5` reliably. |
| 2026-10-02 | Model weights stay out of git; stored in the S3 bucket `models/` prefix with checksums in `configs/models.yaml`. | One source of truth, traceable results. |
| 2026-10-02 | Dashboard deferred to the final phase. | Erik: it's a quick visual check. |
| 2026-10-02 | Target runtime: CPU-only Azure VM; data in S3. | Project constraints. |
| 2026-10-02 | This document lives at the repo root, with a copy in the Claude project. | Versioned with the code. |

---

## 11. Run log

*(newest first)*

| Date | Scene(s) | Sensor | Model | Config / commit | Result / notes |
|---|---|---|---|---|---|
| 2026-10-02 | GT_2 P001, 4 × 1 km² windows | vantor | vantor_2022 | 0.324 m reversed R,G,B (all); native 0.25 m reversed (w1–w3); 0.324 m native B,G,R (w1, w3); v0.1.0 | Pooled F1 0.60 (w1 0.72). Small craters missed. FPs in woodland/roofs/rough ground (Section 8.2). ~3 min per window at 0.324 m on 2 cores. |
| 2026-10-02 | Strata 4 / sample_14494, 22JUN13 + 22JUN09 clips | vantor | vantor_2022 | 9 variants (Section 8.1), v0.1.0 | Best: 0.324 m → 533 (reversed) / 340 (native) craters, F1 vs GT 0.48 / 0.50 (original 0.49). Dev container, 2 cores: ~1–2 min per variant. |

---

## 12. Changelog

- **2026-10-02 (d):** Band-order correction (Vantor files are B,G,R; result tables relabelled). Min-area default 25 m². GPU-ready: TF install extras, `--log-file`, periodic progress logging, BigTIFF-safe outputs, Windows GPU scripts + `docs/windows_gpu.md`. Answered Q7, Q12, Q13; updated Q9.
- **2026-10-02 (c):** GT_2 full-resolution evaluation (Section 8.2). Added `scripts/evaluate_windows.py` and `scripts/quicklook.py`. Added Q12, Q13.
- **2026-10-02 (b):** Phase 1. Built the `halo_craters` package and CLI. Verified the training imagery spec (uint8, B,G,R, 0.324 m), sensor band orders and crater size distribution. Ran the parity investigation on sample 14494. Answered Q1–Q3, added Q9–Q11.
- **2026-10-02 (a):** Document created. Training spec reconstructed from `Base_Unet_Notebook` code and `.h5` metadata.
