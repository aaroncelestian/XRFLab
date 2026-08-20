# XRFLab

Desktop application for X-ray fluorescence (XRF) spectrum analysis: interactive fitting, detector/tube calibration, batch processing, area-normalized semi-quantification, and standardless fundamental-parameters composition from a single spectrum. Multi-standard / fisx tools are available under **Calibration → Standards**.

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
- **FP composition**: standardless wt% from one fitted spectrum, with a matrix model (measured / oxide / carbonate / hydroxide) and live H₂O, OH, and CO₂ knobs for unmeasurable light elements
- **Batch analysis** using the same fitter contract as the Analysis tab
- **Mapping tab** for Oxford INCA / Horiba XGT `.ipj` projects: **Maps** (element maps, RGB, correlations, drawn intensity profiles) and **Line scan** (collected line / multipoint ROI profiles and area-normalized semi-quant)
- Multi-format I/O (TXT, CSV, MCA, HDF5, EMSA, IPJ)
- Session model with injectable detector / instrument state
- Optional **fisx** / xraylib FP helpers for standards-based work

### Clarifications
- Analysis **Semi-Quant** = peak-area relative intensities normalized to 100%. It is **not** fundamental-parameters concentrations.
- Analysis **FP Composition** = standardless wt% from fitted peak areas, using a matrix model plus optional H₂O / OH / CO₂. Those knobs are user assumptions, not measured concentrations.
- Use **Calibration → Standards** for instrument-calibrated / multi-standard intensity work.
- Mapping **Line scan** semi-quant is only for collected line / multipoint spectra. A transect drawn on a map is an intensity profile, not a fit.
- Project open/save, energy-axis dialog, dark theme, and report generation are not implemented yet (menus for unfinished items are hidden).

## Installation

XRFLab is a Python desktop app. Install **Python**, clone or copy the project, create a **virtual environment**, then install the packages in `requirements.txt`.

**Recommended:** Python **3.10, 3.11, or 3.12** (64-bit). Python 3.9 may work; 3.13+ is untested. Use a venv so XRFLab’s packages do not mix with other software.

Put the project in a simple local folder (`~/XRFLab` or `C:\XRFLab`). Cloud-synced paths (iCloud, OneDrive, Dropbox) and paths with unusual characters can break Qt plugins or compiled libraries.

---

### macOS

#### 1. Install Python

Use either method. Confirm in **Terminal** (Applications → Utilities → Terminal):

```bash
python3 --version
```

You want `Python 3.10` or newer.

**Option A — python.org (simplest)**

1. Download the macOS 64-bit installer from [https://www.python.org/downloads/macos/](https://www.python.org/downloads/macos/).
2. Run it and finish the installer.
3. If `python3` is still not found, open a **new** Terminal window.

**Option B — Homebrew** (if you already use [Homebrew](https://brew.sh))

```bash
brew install python@3.12
python3 --version
```

Apple Silicon (M1/M2/M3/M4) and Intel Macs both work. You do **not** need Rosetta for this app.

#### 2. Get the XRFLab folder

If you have Git:

```bash
cd ~
git clone https://github.com/aaroncelestian/XRFLab.git
cd XRFLab
```

Or download/unzip the project and:

```bash
cd /path/to/XRFLab
```

#### 3. Create a virtual environment and install packages

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The prompt should start with `(venv)`. Stay in that environment for every later command.

One-shot alternative (same steps plus sample spectra):

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

If `xraylib` fails to install from pip, try:

```bash
brew install tschoonj/tap/xraylib
pip install xraylib
```

Or use the Conda method in [Alternative: Conda](#alternative-conda-macos-and-windows) below.

#### 4. Launch

```bash
source venv/bin/activate   # skip if already active
python main.py
```

Optional sample spectra:

```bash
python -m utils.sample_data
```

After the window opens: **Help → Install Desktop Shortcut** adds an XRFLab icon to your Desktop.

Each new Terminal session:

```bash
cd /path/to/XRFLab
source venv/bin/activate
python main.py
```

---

### Windows

Use **Command Prompt** or **PowerShell**. These steps use Command Prompt.

#### 1. Install Python

1. Download the **Windows 64-bit** installer from [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/).
2. Run it.
3. Check **Add python.exe to PATH** (bottom of the first screen). This is required.
4. Click **Install Now**.
5. Close and reopen Command Prompt.

Check:

```bat
python --version
pip --version
```

You want `Python 3.10` or newer. If Windows opens the Microsoft Store instead of Python, turn off **Settings → Apps → Advanced app settings → App execution aliases** for `python.exe` / `python3.exe`, or reinstall Python with PATH enabled.

The **Windows Store** Python build is not recommended (Qt and scientific wheels often fail).

#### 2. Get the XRFLab folder

If you have [Git for Windows](https://git-scm.com/download/win):

```bat
cd %USERPROFILE%
git clone https://github.com/aaroncelestian/XRFLab.git
cd XRFLab
```

Or unzip the project and:

```bat
cd C:\path\to\XRFLab
```

#### 3. Create a virtual environment and install packages

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The prompt should start with `(venv)`. Stay in that environment for every later command.

If `xraylib` fails on pip, use the Conda method below. Installing a C compiler to build xraylib from source is not required when wheels or conda-forge packages are available.

#### 4. Launch

```bat
venv\Scripts\activate
python main.py
```

Optional sample spectra:

```bat
python -m utils.sample_data
```

After the window opens: **Help → Install Desktop Shortcut** adds an XRFLab icon to your Desktop.

Each new Command Prompt session:

```bat
cd C:\path\to\XRFLab
venv\Scripts\activate
python main.py
```

In PowerShell, activation is:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, run once (as Administrator if needed):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

### Alternative: Conda (macOS and Windows)

Use this if you already have Anaconda/Miniconda, or if `pip install xraylib` fails. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is enough.

```bash
cd /path/to/XRFLab
conda create -n xrflab python=3.12 -y
conda activate xrflab
conda install -c conda-forge xraylib -y
pip install -r requirements.txt
python main.py
```

If pip then tries to reinstall xraylib and errors, install the rest without it:

```bash
pip install PySide6 pyqtgraph numpy scipy pandas h5py openpyxl fisx matplotlib olefile
```

`conda activate xrflab` replaces `source venv/bin/activate` / `venv\Scripts\activate`. Do not mix a `venv` folder and this conda env.

---

### Confirm the install

From the activated environment, in the XRFLab folder:

```bash
python -c "import PySide6, pyqtgraph, numpy, scipy, pandas, xraylib, fisx; print('OK')"
python main.py
```

Optional tests (after `pip install pytest`):

```bash
pytest
```

---

### Troubleshooting

| Problem | What to try |
| --- | --- |
| `python` / `python3` not found | Reinstall Python and enable PATH (Windows). Open a **new** terminal. On macOS use `python3`. |
| `No module named PySide6` (or similar) | The venv is not active, or you installed packages for a different Python. Activate, then `pip install -r requirements.txt` again. |
| `xraylib` pip error | Use conda-forge (`conda install -c conda-forge xraylib`) or, on macOS, `brew install tschoonj/tap/xraylib` then `pip install xraylib`. |
| Window never appears / Qt plugin error | Install from a local disk path, not iCloud/OneDrive. On Windows use 64-bit Python from python.org, not the Store. |
| `pip` not recognized (Windows) | `python -m pip install -r requirements.txt` |
| macOS “Python is from an unidentified developer” | System Settings → Privacy & Security → Open Anyway, or right-click the Python installer → Open. |
| PowerShell `Activate.ps1` is disabled | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt. |
| Want a Desktop icon | Launch once, then **Help → Install Desktop Shortcut**. |

---

## Usage

With the virtual environment (or conda env) **activated**, from the XRFLab folder:

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
5. **Results → Semi-Quant** for relative intensities, or **FP Composition** for matrix-based wt% (tune H₂O / OH / CO₂ as needed)

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
│   ├── matrix_model.py     # Oxide/carbonate/hydroxide + H2O/OH/CO2
│   ├── fp_quantification.py  # Standardless FP wt% from one spectrum
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
