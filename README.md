# HALO Crater Detection

Applies trained U-Net crater detection models to very-high-resolution (VHR) satellite imagery from **Planet SkySat** and **Vantor**, and writes crater detections as shapefiles.

Pipeline: **input imagery → sensor-specific preprocessing → tiled U-Net inference → polygonization → shapefile (+ dashboard)**

Designed to run on a CPU-only Azure VM with imagery stored in an S3 bucket.

> 🚧 Early development. See **[MASTER_PROJECT.md](MASTER_PROJECT.md)** for the project plan, model specification, decisions and status.

## Repository rules

- **No data, model weights, or outputs in git.** These live in the S3 bucket and in the local, gitignored `data/`, `models/` and `outputs/` folders.
- **No credentials in git.** AWS keys go in `~/.aws/credentials` or environment variables on the VM.

## Quick start

Coming in Phase 1.
