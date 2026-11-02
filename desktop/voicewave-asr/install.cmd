@echo off
setlocal EnableExtensions
cd /d "%~dp0"
where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Conda not found. Run from Miniconda Prompt.
  pause
  exit /b 1
)

for %%I in ("%~dp0..\VoiceWaveASREnv") do set "ENV_DIR=%%~fI"
set "PYTHON=%ENV_DIR%\python.exe"
set "PIP_CONSTRAINT=" 

if not exist "%PYTHON%" (
  echo Creating local Conda environment in "%ENV_DIR%"...
  call conda create --prefix "%ENV_DIR%" python=3.11 -y
  if errorlevel 1 goto :error
)

"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo Installing CUDA 12 libraries for Faster-Whisper...
  "%PYTHON%" -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
  if errorlevel 1 goto :error
)

echo.
echo Core packages installed in "%ENV_DIR%".
echo For local Paraformer, run install_paraformer.cmd.
pause
exit /b 0

:error
echo [ERROR] Installation failed.
pause
exit /b 1
