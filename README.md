# XRF Fundamental Parameters Analysis

A professional desktop application for X-ray fluorescence (XRF) spectroscopy data analysis using the fundamental parameters method.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PySide6](https://img.shields.io/badge/PySide6-6.6+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

### Current Implementation (Phase 1)
- ✅ **Modern GUI** with PySide6 (Qt6)
- ✅ **High-performance plotting** using PyQtGraph
- ✅ **Interactive spectrum display** with crosshair and zoom/pan
- ✅ **Element selection panel** with searchable tree widget
- ✅ **Results display** with statistics and quantification table
- ✅ **Multiple file format support** (TXT, CSV, MCA, HDF5)
- ✅ **Professional styling** with custom Qt stylesheet
- ✅ **Sample data generator** for testing

### Planned Features (Future Phases)
- 🔄 **Spectrum fitting engine** with background modeling (SNIP, polynomial)
- 🔄 **Peak identification** and element suggestion
- 🔄 **Fundamental parameters quantification** using xraylib
- 🔄 **Matrix corrections** and secondary fluorescence
- 🔄 **Energy calibration** tools
- 🔄 **Batch processing** capabilities
- 🔄 **Report generation** and export

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python main.py
```

### Generating Sample Data

To create sample XRF spectra for testing:

```bash
python -m utils.sample_data
```

This will create a `sample_data/` directory with synthetic spectra:
- `steel_sample.txt` - Steel alloy (Fe, Cr, Ni, Mn)
- `brass_sample.txt` - Brass alloy (Cu, Zn)
- `mineral_sample.txt` - Mineral sample (Ca, Ti, Fe)

### Loading Spectra

1. Click **File → Open Spectrum** or use `Ctrl+O`
2. Select a spectrum file (supports .txt, .csv, .mca, .h5/.hdf5)
3. The spectrum will be displayed in the center panel

### Element Selection

1. Use the search box to filter elements
2. Check elements you want to analyze
3. Click "Fit Spectrum" to perform analysis (coming in Phase 2)

### Viewing Results

- **Fit Statistics**: Chi-squared, R-squared values
- **Quantification Results**: Element concentrations with errors
- **Identified Peaks**: List of detected peaks with energies

## Project Structure

```
FpXrF/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── starter.MD             # Detailed project specifications
│
├── ui/                    # User interface modules
│   ├── __init__.py
│   ├── main_window.py     # Main application window
│   ├── spectrum_widget.py # Spectrum plotting widget
│   ├── element_panel.py   # Element selection panel
│   └── results_panel.py   # Results display panel
│
├── core/                  # Core analysis modules
│   ├── __init__.py
│   ├── spectrum.py        # Spectrum data class
│   ├── fitting.py         # Fitting algorithms (planned)
│   ├── quantification.py  # FP quantification (planned)
│   └── database.py        # xraylib interface (planned)
│
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── io_handler.py      # File I/O for various formats
│   ├── sample_data.py     # Sample data generator
│   └── calculations.py    # Helper calculations (planned)
│
└── resources/             # Application resources
    ├── styles.qss         # Qt stylesheet
    └── icons/             # Application icons (planned)
```

## Technology Stack

### Core Framework
- **PySide6** - Qt6 bindings for Python (GUI framework)
- **PyQtGraph** - High-performance scientific plotting

### Scientific Libraries
- **xraylib** - X-ray cross-section databases and atomic parameters
- **NumPy** - Numerical computations
- **SciPy** - Scientific algorithms
- **Pandas** - Data management and export

### File Formats
- **h5py** - HDF5 file support
- **openpyxl** - Excel export support

## Development Roadmap

### Phase 1: Basic Infrastructure ✅
- [x] Main window with menu bar and panels
- [x] PyQtGraph spectrum display
- [x] File I/O for common formats
- [x] Element selection interface
- [x] Results display panel
- [x] Professional styling

### Phase 2: Core Analysis (In Progress)
- [ ] Integrate xraylib for fundamental parameters
- [ ] Background modeling (SNIP algorithm)
- [ ] Peak fitting (Gaussian/Voigt profiles)
- [ ] Automatic peak identification
- [ ] Element suggestion

### Phase 3: Quantification
- [ ] Fundamental parameters calculations
- [ ] Matrix corrections
- [ ] Secondary fluorescence effects
- [ ] Calibration methods
- [ ] Standardless analysis

### Phase 4: Polish & Advanced Features
- [ ] Dark/light theme toggle
- [ ] Progress bars for long operations
- [ ] High-quality plot export
- [ ] Report generation
- [ ] Batch processing
- [ ] Database of reference spectra

## Contributing

This is a research/educational project. Contributions, suggestions, and bug reports are welcome!

## References

- **xraylib**: X-ray fluorescence cross-section library
- **PyMca**: Open-source XRF analysis software (reference implementation)
- **PyXRF**: Python-based XRF analysis toolkit

## License

MIT License - see LICENSE file for details

## Author

Developed for XRF spectroscopy research and education.

## Acknowledgments

- xraylib developers for the fundamental parameters database
- PyQtGraph team for the excellent plotting library
- Qt/PySide6 for the GUI framework
