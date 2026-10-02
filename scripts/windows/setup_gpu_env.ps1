# One-time setup of a GPU Python environment ("halo-gpu") on Windows.
#
# Uses TensorFlow 2.10.1, the last release with native Windows GPU support,
# with CUDA 11.2 / cuDNN 8.1 from conda-forge. Requires an NVIDIA GPU + driver
# and a conda installation (Miniforge recommended: https://conda-forge.org/download/).
#
# Run by double-clicking scripts\windows\setup_gpu_env.bat
# Log: _scratch\gpu_setup.log (in the repo folder)

$ErrorActionPreference = "Continue"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "_scratch"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$Log = Join-Path $LogDir "gpu_setup.log"
Start-Transcript -Path $Log -Force | Out-Null

function Find-Conda {
    $c = Get-Command conda -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $roots = @("$env:USERPROFILE\miniforge3", "$env:USERPROFILE\mambaforge", "$env:USERPROFILE\anaconda3",
               "$env:USERPROFILE\miniconda3", "$env:LOCALAPPDATA\miniforge3", "$env:LOCALAPPDATA\anaconda3",
               "$env:LOCALAPPDATA\miniconda3", "C:\ProgramData\miniforge3", "C:\ProgramData\anaconda3",
               "C:\ProgramData\miniconda3", "C:\miniforge3", "C:\anaconda3", "C:\miniconda3")
    foreach ($r in $roots) { if (Test-Path "$r\Scripts\conda.exe") { return "$r\Scripts\conda.exe" } }
    return $null
}

$ok = $true
try {
    Write-Host "== HALO GPU setup  $(Get-Date -Format s)"
    Write-Host "== Repo: $Repo"
    Write-Host "== GPU (nvidia-smi):"
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) { & nvidia-smi 2>&1 | Out-Host } else { Write-Host "nvidia-smi not found on PATH" }

    $conda = Find-Conda
    if (-not $conda) {
        Write-Host "ERROR: conda not found. Install Miniforge (https://conda-forge.org/download/), then run this again."
        $ok = $false
        return
    }
    Write-Host "== conda: $conda"
    & $conda --version 2>&1 | Out-Host

    $envs = (& $conda env list 2>&1) -join "`n"
    if ($envs -match "(?m)^halo-gpu\s") {
        Write-Host "== Environment 'halo-gpu' already exists; updating packages in place"
    } else {
        Write-Host "== Creating environment 'halo-gpu' (this can take 5-15 minutes)"
        & $conda create -n halo-gpu -y -c conda-forge --override-channels `
            python=3.10 cudatoolkit=11.2 cudnn=8.1.0 "numpy=1.23" "rasterio=1.3" geopandas pyogrio `
            "shapely>=2" scipy pyyaml tqdm pip 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: conda create failed"; $ok = $false; return }
    }

    Write-Host "== Installing TensorFlow 2.10.1 (GPU) and the halo-craters package"
    & $conda run --no-capture-output -n halo-gpu python -m pip install "tensorflow==2.10.1" "numpy==1.23.*" 2>&1 | Out-Host
    & $conda run --no-capture-output -n halo-gpu python -m pip install --no-deps -e "$Repo" 2>&1 | Out-Host

    Write-Host "== Verifying"
    $check = @"
import tensorflow as tf, rasterio, geopandas, numpy
print('tensorflow', tf.__version__, '| numpy', numpy.__version__, '| rasterio', rasterio.__version__, '| GDAL', rasterio.__gdal_version__, '| geopandas', geopandas.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
import halo_craters; print('halo_craters', halo_craters.__version__)
print('SETUP_OK' if gpus else 'SETUP_NO_GPU')
"@
    & $conda run --no-capture-output -n halo-gpu python -c $check 2>&1 | Out-Host
}
catch {
    Write-Host "ERROR: $_"
    $ok = $false
}
finally {
    Write-Host "== Finished $(Get-Date -Format s)  ok=$ok"
    Stop-Transcript | Out-Null
}
