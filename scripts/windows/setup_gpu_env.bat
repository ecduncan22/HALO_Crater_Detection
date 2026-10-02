@echo off
rem Double-click to create the "halo-gpu" conda environment (one time).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_gpu_env.ps1"
echo.
echo Log written to _scratch\gpu_setup.log in the repo folder.
pause
