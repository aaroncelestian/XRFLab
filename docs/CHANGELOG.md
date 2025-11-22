# Changelog

## Version 1.1.0 - Periodic Table Update (2024-11-22)

### ✨ New Features

#### Interactive Periodic Table
- **Replaced tree list with full periodic table** for element selection
- **118 elements** from Hydrogen to Hassium
- **Color-coded by element groups**:
  - Alkali metals, alkaline earth metals
  - Transition metals, post-transition metals
  - Metalloids, nonmetals, halogens, noble gases
  - Lanthanides and actinides
- **Interactive buttons** with hover effects and selection highlighting
- **Visual legend** showing element group colors

#### Quick Selection Tools
- **Select All** - Select all elements in periodic table
- **Clear All** - Deselect all elements
- **Common XRF** - Auto-select 35 commonly analyzed XRF elements

#### Enhanced UX
- **Tooltips** showing element name and atomic number
- **Visual feedback** with color changes on hover and selection
- **Scrollable layout** for smaller screens
- **Professional appearance** matching scientific software standards

### 🔧 Improvements
- Fixed Qt warning about toolbar object name
- Cleaner element selection code
- Better signal handling for element changes
- More intuitive interface for XRF users

### 📝 Documentation
- Added `PERIODIC_TABLE_UPDATE.md` - Detailed feature guide
- Added `CHANGELOG.md` - This file
- Updated visual guides

### 🐛 Bug Fixes
- Fixed QMainWindow::saveState() warning for toolbar
- Improved element selection signal propagation

---

## Version 1.0.0 - Initial Release (2024-11-22)

### 🎉 Initial Features

#### Core Application
- PySide6/Qt6 based GUI framework
- Three-panel layout (elements, spectrum, results)
- Menu bar with File, Analysis, View, Tools, Help
- Toolbar with quick-access buttons
- Status bar with messages

#### Spectrum Display
- High-performance PyQtGraph plotting
- Interactive crosshair with energy/counts display
- Logarithmic Y-axis support
- Grid toggle
- Zoom and pan capabilities
- Residuals plot
- Peak markers support

#### Element Selection (Original)
- Tree-based element list
- Searchable interface
- Organized by categories (Light, Transition, Heavy)
- Select All / Clear All buttons

#### Sample Information
- Sample name, type, thickness inputs
- Experimental parameters (excitation, current, time, detector, angle)

#### Fitting Controls
- Background method selection (SNIP, Polynomial, Linear, None)
- Peak shape selection (Gaussian, Voigt, Pseudo-Voigt)
- Escape peaks toggle
- Pile-up correction toggle
- Fit button

#### Results Display
- Fit statistics (χ², R², χ²ᵣ, iterations)
- Quantification table (Element, Concentration, Error, Line)
- Total concentration with color coding
- Identified peaks list
- Export button

#### File I/O
- Load formats: TXT, CSV, MCA, HDF5
- Save formats: TXT, CSV, HDF5
- Export: CSV, Excel (XLSX)
- Automatic format detection

#### Core Data Structures
- Spectrum class with full functionality
- Energy calibration
- ROI extraction
- Normalization
- Rebinning and smoothing

#### Utilities
- Sample data generator
- Synthetic XRF spectra creation
- Pre-built sample files (steel, brass, mineral)

#### Styling
- Modern Qt stylesheet
- Professional color scheme
- Consistent UI elements
- Hover effects and visual feedback

#### Documentation
- Comprehensive README
- Quick start guide
- Project summary
- Layout guide
- Setup script

---

## Roadmap

### Version 1.2.0 - Phase 2 (Planned)
- [ ] xraylib integration for fundamental parameters
- [ ] Background modeling (SNIP algorithm)
- [ ] Peak fitting engine (Gaussian/Voigt)
- [ ] Automatic peak identification
- [ ] Element suggestion based on peaks

### Version 2.0.0 - Phase 3 (Planned)
- [ ] Fundamental parameters quantification
- [ ] Matrix corrections
- [ ] Secondary fluorescence effects
- [ ] Calibration methods
- [ ] Standardless analysis

### Version 3.0.0 - Phase 4 (Planned)
- [ ] Dark/light theme toggle
- [ ] Progress bars for long operations
- [ ] Batch processing
- [ ] Report generation (PDF)
- [ ] Database of reference spectra
- [ ] Machine learning peak identification

---

## Contributors

Developed for XRF spectroscopy research and education.

## License

MIT License
