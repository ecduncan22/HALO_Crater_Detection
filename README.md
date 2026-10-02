# HALO Crater Detection

Applies trained U-Net crater detection models to very-high-resolution (VHR) satellite imagery from **Planet SkySat** and **Vantor**, and writes crater detections as polygons (GeoPackage + Shapefile).

Pipeline: **input imagery → sensor-specific preprocessing → tiled U-Net inference → polygonization → vectors**

Designed to run on a CPU-only Azure VM with imagery stored in an S3 bucket.
See **[MASTER_PROJECT.md](MASTER_PROJECT.md)** for the project plan, model specification, decisions, evaluation results and status.

## Install

Python 3.10 or 3.11 (TensorFlow 2.15 is required to load the Keras-2 `.h5` models).

```bash
git clone https://github.com/ecduncan22/HALO_Crater_Detection.git
cd HALO_Crater_Detection
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Put the model weights in `models/` (not in git). The paths and checksums are listed in `configs/models.yaml`.

## Run

```bash
# one scene (or a directory of .tif files)
halo-craters run /path/to/scene.tif --sensor vantor --model vantor_2022 --target-gsd 0.324 --out outputs/

# SkySat scene with the Vantor model
halo-craters run /path/to/skysat.tif --sensor skysat --model vantor_2022 --target-gsd 0.324 --out outputs/

# compare against reference polygons
halo-craters evaluate outputs/<scene>/<scene>_craters.gpkg reference.shp --aoi aoi.shp --iou 0.1
```

Each scene gets `outputs/<scene>/` with `*_craters.gpkg`, `*_craters.shp`, `*_prob.tif` (uint8 probability, Cloud-Optimized GeoTIFF) and `run.json` (config, model, commit, timings).

Useful options: `--bands 3,2,1` (override band order), `--stride`, `--threshold`, `--min-area`, `--rescale percentile`, `--parity` (reproduce the original notebook's tiling), `--tag` (keep experiment outputs apart). Sensor band layouts live in `configs/sensors.yaml`; run defaults are in `configs/default.yaml`.

## Repository rules

- **No data, model weights, or outputs in git.** These live in the S3 bucket and in the local, gitignored `data/`, `models/` and `outputs/` folders.
- **No credentials in git.** AWS keys go in `~/.aws/credentials` or environment variables on the VM.
