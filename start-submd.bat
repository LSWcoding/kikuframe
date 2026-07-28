@echo off
setlocal
cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo [KikuFrame] Python 3.12 was not found. Install Python 3.12 and try again.
  pause
  exit /b 1
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo [KikuFrame] FFmpeg was not found in PATH. Install FFmpeg and try again.
  pause
  exit /b 1
)

where deno >nul 2>&1
if errorlevel 1 (
  echo [KikuFrame] Deno was not found in PATH. Install Deno and try again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e . --quiet
.venv\Scripts\submd.exe ui

if errorlevel 1 pause
endlocal
