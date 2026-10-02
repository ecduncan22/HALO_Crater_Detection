# Running on a Windows workstation GPU

TensorFlow 2.10.1 is the last release with **native Windows GPU** support. It needs
CUDA 11.2 + cuDNN 8.1, which conda-forge provides inside the environment, so no
system-wide CUDA install is required. It loads the project's Keras-2 `.h5` models.

**Requirements:** an NVIDIA GPU with a recent driver (`nvidia-smi` works), and conda
([Miniforge](https://conda-forge.org/download/) recommended).

## 1. One-time setup

Double-click `scripts\windows\setup_gpu_env.bat`. It:

- prints the GPU (`nvidia-smi`)
- creates the conda env `halo-gpu` (Python 3.10, CUDA 11.2, cuDNN 8.1, numpy 1.23, rasterio 1.3, geopandas …)
- installs `tensorflow==2.10.1` and this repo (editable, so `git pull` updates the code)
- prints `SETUP_OK` if TensorFlow sees the GPU

Log: `_scratch\gpu_setup.log`

## 2. Run detection

Double-click `scripts\windows\run_detection_gpu.bat` (default: the GT_2 P001 scene →
`C:\Projects\UMD\HALO_2026\outputs\gt2_full`). For another scene:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\run_detection_gpu.ps1 `
    -Image D:\scenes\scene.tif -Out D:\outputs -Sensor vantor -TargetGsd 0.324 -BatchSize 32
```

Progress is logged every ~30 s to `_scratch\gpu_run.log` (tiles done, tiles/s, ETA);
the last line is `== RUN_FINISHED exit=0` on success.

## Linux / WSL2 / Azure GPU VMs

Use TensorFlow 2.15 instead: `pip install -e ".[gpu]"` (installs `tensorflow[and-cuda]==2.15.1`).
