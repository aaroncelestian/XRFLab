#!/bin/bash
# Setup script for XRFLab (macOS / Linux): venv, deps, sample data, desktop shortcut

set -e

echo "=========================================="
echo "XRFLab Application Setup"
echo "=========================================="
echo ""

# Always run from the repo root (directory containing this script)
cd "$(dirname "$0")"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Generate sample data
echo "Generating sample XRF spectra..."
python -m utils.sample_data
echo "✓ Sample data generated"
echo ""

# Desktop shortcut (uses this venv's python)
echo "Installing desktop shortcut..."
if python -m utils.desktop_shortcut; then
    echo "✓ Desktop shortcut installed"
else
    echo "⚠ Desktop shortcut skipped (you can create it later from Help → Install Desktop Shortcut)"
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the application:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo "  2. Run the application:"
echo "     python main.py"
echo ""
echo "Or double-click the XRFLab icon on your Desktop."
echo ""
echo "Sample data is available in: sample_data/"
echo ""
