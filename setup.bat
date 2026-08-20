@echo off
REM Setup script for XRFLab (Windows): venv, deps, sample data, desktop shortcut
setlocal EnableExtensions
cd /d "%~dp0"

echo ==========================================
echo XRFLab Application Setup
echo ==========================================
echo.

echo Checking Python version...
python --version
if errorlevel 1 (
    echo Python was not found. Install Python 3.12 from python.org and enable PATH.
    exit /b 1
)
echo.

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 exit /b 1
echo Virtual environment created
echo.

echo Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1
echo Virtual environment activated
echo.

echo Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo Dependencies installed
echo.

echo Generating sample XRF spectra...
python -m utils.sample_data
echo Sample data generated
echo.

echo Installing desktop shortcut...
python -m utils.desktop_shortcut
if errorlevel 1 (
    echo Desktop shortcut skipped - create it later from Help - Install Desktop Shortcut
) else (
    echo Desktop shortcut installed
)
echo.

echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To run the application:
echo   1. Activate the virtual environment:
echo      venv\Scripts\activate
echo   2. Run the application:
echo      python main.py
echo.
echo Or double-click the XRFLab icon on your Desktop.
echo.
echo Sample data is available in: sample_data\
echo.
endlocal
