@echo off
rem PDF Translator launcher for Windows.
rem First run: creates a private Python environment (.venv) and installs everything
rem (this downloads PyTorch and is several GB, but only happens once).
rem Later runs: start the program straight away.

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" goto launch

echo ==========================================================
echo  First-time setup. This downloads PyTorch and the other
echo  libraries (several GB). It only happens once.
echo ==========================================================
echo.

where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")

%PY% -m venv .venv
if errorlevel 1 goto failed

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed

rem PyTorch with CUDA support (also runs on PCs without an NVIDIA GPU, using the CPU)
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto failed

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
echo Setup complete.
echo.

:launch
rem pythonw.exe starts the window without a console
start "" ".venv\Scripts\pythonw.exe" gui.py
exit /b 0

:failed
echo.
echo Setup failed. Read the messages above, then press a key to close.
pause >nul
exit /b 1
