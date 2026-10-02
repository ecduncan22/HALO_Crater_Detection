@echo off
rem Double-click to run crater detection on the GPU (default: GT_2 P001 scene).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_detection_gpu.ps1" %*
echo.
echo Log written to _scratch\gpu_run.log in the repo folder.
pause
