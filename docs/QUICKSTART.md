# Quick Start Guide

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Generate Sample Data

Create sample XRF spectra for testing:

```bash
python -m utils.sample_data
```

This creates three sample files in `sample_data/`:
- `steel_sample.txt` - Steel alloy (Fe, Cr, Ni, Mn)
- `brass_sample.txt` - Brass alloy (Cu, Zn)  
- `mineral_sample.txt` - Mineral sample (Ca, Ti, Fe)

## Run the Application

```bash
python main.py
```

## Quick Test Workflow

1. **Launch the application**
   ```bash
   python main.py
   ```

2. **Load a sample spectrum**
   - Click `File → Open Spectrum` (or press `Ctrl+O`)
   - Navigate to `sample_data/`
   - Select `steel_sample.txt`

3. **Explore the interface**
   - **Left Panel**: Select elements (Fe, Cr, Ni, Mn for steel)
   - **Center Panel**: Interactive spectrum plot with crosshair
   - **Right Panel**: Results display (will show data after fitting)

4. **Try the controls**
   - Use mouse wheel to zoom
   - Click and drag to pan
   - Toggle `View → Logarithmic Y-axis`
   - Toggle `View → Show Grid`

## Keyboard Shortcuts

- `Ctrl+O` - Open spectrum
- `Ctrl+S` - Save project
- `Ctrl+F` - Fit spectrum (Phase 2)
- `Ctrl+Q` - Quantification (Phase 3)
- `Ctrl+Q` (menu) - Quit application

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

### xraylib Installation Issues

On some systems, xraylib may need special installation:

**macOS**:
```bash
brew install xraylib
pip install xraylib
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install libxrl-dev
pip install xraylib
```

**Windows**:
```bash
pip install xraylib
```

If issues persist, xraylib can be temporarily skipped for Phase 1 testing.

### Display Issues

If the GUI doesn't display correctly:
- Ensure you have a display server running
- Try setting: `export QT_QPA_PLATFORM=xcb` (Linux)
- Update graphics drivers

## Next Steps

After testing Phase 1:
- Phase 2 will add spectrum fitting and peak identification
- Phase 3 will add fundamental parameters quantification
- Phase 4 will add advanced features and polish

## Getting Help

Check the main README.md for:
- Full feature list
- Project structure
- Development roadmap
- Contributing guidelines
