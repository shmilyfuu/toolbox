@echo off
setlocal EnableExtensions
cd /d "%~dp0"
for %%I in ("%~dp0..\VoiceWaveASREnv\python.exe") do set "PYTHON=%%~fI"
if not exist "%PYTHON%" (
  echo [ERROR] Python environment missing: "%PYTHON%"
  pause
  exit /b 1
)
"%PYTHON%" main.py
if errorlevel 1 pause
