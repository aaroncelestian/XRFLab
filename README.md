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
- **Mapping tab** for Oxford INCA / Horiba XGT `.ipj` projects: **Maps** (element maps, RGB, correlations, drawn intensity profiles) and **Line scan** (collected line / multipoint ROI profiles and area-normalized semi-quant). **Merge IPJs…** combines multipoint/line-scan series from many `.ipj` files into one project (spectra named `filename_sample_site_spectrum`); save as `.xrfp`
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

**Recommended path:** [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) (macOS and Windows). It gives you Python plus a reliable build of **xraylib**, which is the dependency that most often fails with plain `pip`.

You do **not** need the full Anaconda distribution. Miniconda is smaller and enough for XRFLab. If you already have Anaconda or Miniforge, you can skip the Miniconda installer and start at step 2.

**Python version:** 3.10, 3.11, or 3.12 (64-bit). Prefer **3.12**. Python 3.9 may work; 3.13+ is untested.

Put the project in a simple local folder (`~/XRFLab` or `C:\XRFLab`). Cloud-synced paths (iCloud, OneDrive, Dropbox) can break Qt plugins or compiled libraries.

---

### Recommended: Miniconda (macOS and Windows)

#### 1. Install Miniconda

1. Download the installer for your OS from [Miniconda](https://www.anaconda.com/download/success) (choose **Miniconda**, not Anaconda).
2. Run the installer.
   - **macOS:** Apple Silicon (M1/M2/M3/M4) and Intel both work; pick the matching installer. You do not need Rosetta.
   - **Windows:** Use the 64-bit installer. Allow it to initialize conda for Command Prompt / PowerShell when asked.
3. Close and reopen your terminal (macOS **Terminal**, or Windows **Anaconda Prompt** / Command Prompt / PowerShell).

Check:

```bash
conda --version
```

#### 2. Get the XRFLab folder

**macOS / Linux (Terminal):**

```bash
cd ~
git clone https://github.com/aaroncelestian/XRFLab.git
cd XRFLab
```

**Windows (Command Prompt or Anaconda Prompt):**

```bat
cd %USERPROFILE%
git clone https://github.com/aaroncelestian/XRFLab.git
cd XRFLab
```

Or download/unzip the project and `cd` into that folder. On Windows without Git, use [Git for Windows](https://git-scm.com/download/win) or unzip from GitHub.

#### 3. Create the environment and install packages

Same commands on macOS and Windows (in the XRFLab folder):

```bash
conda create -n xrflab python=3.12 -y
conda activate xrflab
conda install -c conda-forge xraylib -y
pip install -r requirements.txt
python -m utils.sample_data
python -m utils.desktop_shortcut
```

That last command puts an **XRFLab** icon on your Desktop (macOS `.app`, Windows `.lnk`). You can recreate it later with **Help → Install Desktop Shortcut**.

The prompt should show `(xrflab)`. Stay in that environment for every later command.

If `pip install -r requirements.txt` tries to rebuild `xraylib` and errors, install the remaining packages without it:

```bash
pip install PySide6 pyqtgraph numpy scipy pandas h5py openpyxl fisx matplotlib olefile pytest
```

Do **not** also create a `venv` folder inside this conda env. Use one environment at a time: either conda `xrflab` or a pip venv (see below), not both.

#### 4. Launch

```bash
conda activate xrflab
cd /path/to/XRFLab
python main.py
```

On Windows, if `conda activate` is not found in a normal Command Prompt, open **Anaconda Prompt** (or “Miniconda Prompt”) from the Start menu and run the same commands there.

Or double-click the **XRFLab** Desktop icon.

Each new session:

```bash
conda activate xrflab
cd /path/to/XRFLab
python main.py
```

---

### Confirm the install

With `(xrflab)` active, from the XRFLab folder:

```bash
python -c "import PySide6, pyqtgraph, numpy, scipy, pandas, xraylib, fisx; print('OK')"
python main.py
```

Optional tests:

```bash
pytest
```

---

### Alternative: python.org + venv (no conda)

Use this if you already manage Python yourself and prefer a lightweight install. If `xraylib` fails under pip, switch to the Miniconda path above.

#### macOS

1. Install Python 3.12 from [python.org](https://www.python.org/downloads/macos/) or Homebrew (`brew install python@3.12`).
2. Open Terminal and run:

```bash
cd ~
git clone https://github.com/aaroncelestian/XRFLab.git
cd XRFLab
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m utils.sample_data
python -m utils.desktop_shortcut
python main.py
```

One-shot helper (venv + deps + sample data + Desktop shortcut):

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
python main.py
```

If `xraylib` fails:

```bash
brew install tschoonj/tap/xraylib
pip install xraylib
```

#### Windows

1. Install **Windows 64-bit** Python from [python.org](https://www.python.org/downloads/windows/).
2. Check **Add python.exe to PATH**, then finish the installer and open a **new** Command Prompt.
3. Avoid the Microsoft Store Python build (Qt and scientific wheels often fail). If `python` opens the Store, disable app execution aliases for `python.exe` / `python3.exe`.

```bat
cd %USERPROFILE%
git clone https://github.com/aaroncelestian/XRFLab.git
cd XRFLab
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m utils.sample_data
python -m utils.desktop_shortcut
python main.py
```

One-shot helper (venv + deps + sample data + Desktop shortcut):

```bat
setup.bat
venv\Scripts\activate
python main.py
```

In PowerShell, activate with `.\venv\Scripts\Activate.ps1`. If scripts are blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Each new session: `source venv/bin/activate` (macOS) or `venv\Scripts\activate` (Windows), then `python main.py` — or use the Desktop icon.

---

### Troubleshooting

| Problem | What to try |
| --- | --- |
| `conda` not found | Reopen the terminal, or use **Anaconda Prompt** / **Miniconda Prompt** (Windows). Re-run the Miniconda installer and allow shell initialization. |
| `conda activate` fails on Windows | Use Anaconda/Miniconda Prompt, or run `conda init cmd.exe` / `conda init powershell`, then open a new window. |
| `No module named PySide6` (or similar) | The env is not active. Run `conda activate xrflab` (or `source venv/bin/activate`), then reinstall with `pip install -r requirements.txt`. |
| `xraylib` pip error | Prefer Miniconda: `conda install -c conda-forge xraylib`. On macOS pip-only installs, try `brew install tschoonj/tap/xraylib` then `pip install xraylib`. |
| Window never appears / Qt plugin error | Install from a local disk path, not iCloud/OneDrive. On Windows avoid Store Python; use Miniconda or python.org 64-bit. |
| `python` not found (venv path) | Open a new terminal; on macOS use `python3`. On Windows enable PATH or use `python -m pip …`. |
| macOS “unidentified developer” | System Settings → Privacy & Security → Open Anyway. |
| Want a Desktop icon | Run `python -m utils.desktop_shortcut` with the env active, or use **Help → Install Desktop Shortcut**. |

---

## Usage

With the conda env (or venv) **activated**, from the XRFLab folder:

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
