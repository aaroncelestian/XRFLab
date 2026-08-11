# XRFLab

Desktop application for X-ray fluorescence (XRF) spectrum analysis: interactive fitting, detector/tube calibration, batch processing, and area-normalized semi-quantification. Fundamental-parameters / fisx tools are available under **Calibration → Standards**.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PySide6](https://img.shields.io/badge/PySide6-6.6+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Documentation

- **[Tutorial](docs/TUTORIAL.md)** — blanks, FWHM, standards, single spectrum, batch
- **[Quick Start Guide](docs/QUICKSTART.md)** — get started in 5 minutes
- **[Fitting Guide](docs/FITTING_GUIDE.md)** — spectrum fitting
- **[Calibration Workflow](docs/CALIBRATION_WORKFLOW.md)** — FWHM / standards
- **[Changelog](docs/CHANGELOG.md)** — version history
- **[Project Specifications](docs/starter.MD)** — original requirements

## Features

### Implemented
- Modern **PySide6** GUI with PyQtGraph spectrum + residuals display
- Interactive **periodic table** element selection (118 elements)
- **Spectrum fitting** with SNIP/polynomial backgrounds and Gaussian/Voigt (and related) peak shapes
- Tube lines, Compton scatter, and tube-profile soft constraints
- **FWHM calibration**, tube profiles, and standards calibration tabs
- **Semi-quant**: area-normalized relative intensities (not FP weight %)
- **Batch analysis** using the same fitter contract as the Analysis tab
- Multi-format I/O (TXT, CSV, MCA, HDF5, EMSA)
- Session model with injectable detector / instrument state
- Optional **fisx** / xraylib FP helpers for standards-based work

### Clarifications
- Analysis **Semi-Quant** = peak-area relative intensities normalized to 100%. It is **not** fundamental-parameters concentrations.
- Use **Calibration → Standards** for instrument-calibrated / FP-style intensity work.
- Project open/save, energy-axis dialog, dark theme, and report generation are not implemented yet (menus for unfinished items are hidden).

## Installation

### Requirements
- Python 3.9 or higher
- pip

### Setup

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
# or: pip install -e ".[dev]"
```

Optional: `./setup.sh` creates the venv, installs deps, and generates sample data.

## Usage

```bash
python main.py
```

Generate sample spectra:

```bash
python -m utils.sample_data
```

Run core tests:

```bash
pytest
```

### Typical Analysis workflow

1. **File → Open Spectrum** (`Ctrl+O`)
2. **Peak Find → Find Peaks + Auto-ID**
3. Review **Elements** (uncheck false IDs / add missing)
4. **Fitting → Fit Spectrum** (`Ctrl+F`)
5. **Results → Semi-Quant** for relative intensities (or Calibration → Standards for calibrated work)

### Calibration

Use the **Calibration** tab (or Tools menu):
- **FWHM** — detector resolution vs energy
- **Tube Profiles** — per-kV scatter line ratios
- **Standards** — intensity / instrument calibration (FP/fisx)

## Project Structure

```
XRFLab/
├── main.py                 # GUI entry point
├── matplotlib_config.py    # Shared matplotlib rcParams for CLI plots
├── pyproject.toml          # Package metadata + pytest config
├── requirements.txt
├── core/                   # Qt-free analysis engine
│   ├── spectrum.py
│   ├── fitting.py          # SpectrumFitter + semi-quant
│   ├── peak_fitting.py
│   ├── session.py          # AnalysisSession document model
│   ├── instrument_state.py # DetectorModel / InstrumentState
│   ├── batch_processing.py
│   ├── fwhm_calibration.py
│   ├── calibration.py
│   └── ...
├── ui/                     # PySide6 panels
├── utils/                  # I/O, sample data, updater
├── tests/                  # pytest suite
├── docs/
├── sample_data/
└── resources/
```

## Technology Stack

- **PySide6** / **PyQtGraph** — GUI and interactive plots
- **NumPy / SciPy / Pandas** — numerics and tables
- **xraylib** / **fisx** — X-ray physics databases and FP
- **matplotlib** — offline calibration/CLI plots (`matplotlib_config.py`)
- **h5py / openpyxl** — HDF5 and Excel I/O

## Roadmap (next)

- Project / session file persistence (`.xrfp`)
- Wire FP concentrations into Analysis when standards prerequisites are met
- Energy-axis calibration UI
- Report generation
- Collapse remaining orphan/parallel calibration loaders

## License

MIT License — see LICENSE file for details.
