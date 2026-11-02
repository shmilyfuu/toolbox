@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Use the Python interpreter from this project's existing prefix environment.
for %%I in ("%~dp0..\VoiceWaveASREnv\python.exe") do set "PYTHON=%%~fI"
if not exist "%PYTHON%" (
  echo [ERROR] Python environment missing: "%PYTHON%"
  echo Expected: "%PYTHON%"
  pause
  exit /b 1
)

rem Avoid applying constraints from another project's pip environment.
set "PIP_CONSTRAINT="

echo [1/4] Interpreter:
"%PYTHON%" -c "import sys; print(sys.executable); print(sys.version)"
if errorlevel 1 goto :error

echo.
echo [2/4] Installing matching CUDA 12.8 PyTorch and TorchAudio builds...
"%PYTHON%" -m pip install "torch==2.7.1+cu128" "torchaudio==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :error

echo.
echo [3/4] Installing FunASR and ModelScope...
"%PYTHON%" -m pip install "funasr>=1.3.7,<2" "modelscope>=1.15" "numpy>=1.26,<2"
if errorlevel 1 goto :error

echo.
echo [4/4] Checking imports, GPU and package compatibility...
"%PYTHON%" -c "import sys, torch, torchaudio; from funasr import AutoModel; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('TorchAudio:', torchaudio.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if errorlevel 1 goto :error
"%PYTHON%" -m pip check
if errorlevel 1 goto :error

echo.
echo Installation verified. You can start VoiceWave ASR with your existing command.
pause
exit /b 0

:error
echo.
echo [ERROR] Installation or verification failed in "%PYTHON%".
pause
exit /b 1
