# Run crater detection on the GPU (Windows), using the "halo-gpu" environment
# created by setup_gpu_env.ps1.
#
# Default: the full GT_2 P001 Vantor scene. Edit the parameters below, or call e.g.
#   powershell -ExecutionPolicy Bypass -File run_detection_gpu.ps1 -Image D:\x.tif -Out D:\out
#
# Run by double-clicking scripts\windows\run_detection_gpu.bat
# Log: _scratch\gpu_run.log (in the repo folder)

param(
    [string]$Image    = "C:\Projects\UMD\HALO_2026\GT\GT_2\imagery\22AUG01081633-M1BS-507349043020_01_P001_pansharpened_RGB.tif",
    [string]$Out      = "C:\Projects\UMD\HALO_2026\outputs\gt2_full",
    [string]$Model    = "C:\Projects\UMD\HALO_2026\models\2022_crater_model.h5",
    [string]$Sensor   = "vantor",
    [string]$ModelName = "vantor_2022",
    [double]$TargetGsd = 0.324,
    [int]$BatchSize   = 32
)

$ErrorActionPreference = "Continue"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "_scratch"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$Log = Join-Path $LogDir "gpu_run.log"
"" | Set-Content -Path $Log -Encoding utf8
Add-Content -Path $Log -Value "== HALO GPU run $(Get-Date -Format s)  image=$Image  out=$Out  model=$Model"

$conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $conda) {
    foreach ($r in @("$env:USERPROFILE\miniforge3", "$env:USERPROFILE\mambaforge", "$env:USERPROFILE\anaconda3",
                     "$env:USERPROFILE\miniconda3", "$env:LOCALAPPDATA\miniforge3", "$env:LOCALAPPDATA\anaconda3",
                     "$env:LOCALAPPDATA\miniconda3", "C:\ProgramData\miniforge3", "C:\ProgramData\anaconda3",
                     "C:\ProgramData\miniconda3", "C:\miniforge3", "C:\anaconda3", "C:\miniconda3")) {
        if (Test-Path "$r\Scripts\conda.exe") { $conda = "$r\Scripts\conda.exe"; break }
    }
}
if (-not $conda) { Add-Content $Log "ERROR: conda not found"; Write-Host "conda not found"; exit 1 }

New-Item -ItemType Directory -Force $Out | Out-Null
Write-Host "Running detection - progress is logged to $Log"
& $conda run --no-capture-output -n halo-gpu halo-craters --log-file "$Log" run "$Image" `
    --sensor $Sensor --model $ModelName --model-path "$Model" `
    --target-gsd $TargetGsd --batch-size $BatchSize --out "$Out"
$code = $LASTEXITCODE
Add-Content -Path $Log -Value "== RUN_FINISHED exit=$code $(Get-Date -Format s)"
Write-Host "Finished with exit code $code"
