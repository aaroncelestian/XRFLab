Here are comprehensive instructions for Claude 4.5 in Windsurf to build a beautiful, professional XRF fundamental parameters analysis application:

---

## Project Instructions: XRF Fundamental Parameters Analysis Application

### Project Overview
Create a professional desktop application for X-ray fluorescence (XRF) fundamental parameters data analysis using PySide6 for the GUI framework. The application should have a modern, scientific aesthetic with excellent data visualization capabilities.

### Technology Stack
**Core Framework:**
- PySide6 (Qt6 for Python) - for open-source GUI development with professional appearance
- PyQtGraph - for high-performance, interactive scientific plotting (better than matplotlib for real-time data)

**Scientific Libraries:**
- xraylib - for X-ray cross-section databases and atomic parameters
- XrayDB - supplementary database for additional X-ray reference data
- numpy, scipy - numerical computations
- pandas - data management

**Reference Implementations:**
- Study PyMca's open-source implementation for fundamental parameters algorithms
- Reference PyXRF's architecture for spectrum fitting approaches

### Application Architecture

**Main Window Structure:**
1. **Menu Bar** with standard scientific application menus:
   - File (Open Spectrum, Save Project, Export Results, Recent Files)
   - Analysis (Fit Spectrum, Quantification, Background Settings)
   - View (Toggle panels, Plot settings, Themes)
   - Tools (Calibration, Element Database, Preferences)
   - Help (Documentation, About)

2. **Toolbar** with quick-access icons for common operations

3. **Main Layout** (use QSplitter for resizable panels):
   - **Left Panel** (30% width):
     - Sample information input fields
     - Element selection tree/list with checkboxes
     - Experimental parameters (excitation energy, detector specs, geometry)
     - Fitting parameters controls
   
   - **Center Panel** (50% width):
     - PyQtGraph PlotWidget for spectrum display (semi-log scale)
     - Interactive crosshair for energy identification
     - Zoom/pan capabilities
     - Multiple spectrum overlay capability
     - Residuals plot below main spectrum
   
   - **Right Panel** (20% width):
     - Results table showing quantified elements
     - Fit statistics (chi-squared, R-squared)
     - Peak identification list
     - Export options

4. **Bottom Status Bar** with processing status and cursor position

### Key Features to Implement

**Phase 1 - Basic Infrastructure:**
1. Create main window with tabbed/docked interface
2. Implement file I/O for common XRF formats (.txt, .csv, .mca, HDF5)
3. Set up PyQtGraph plotting with:
   - Logarithmic Y-axis option
   - Interactive region selection
   - Energy calibration display
   - Peak markers

**Phase 2 - Core Analysis:**
1. Integrate xraylib for fundamental parameters:
   - Photoionization cross-sections
   - Fluorescence yields
   - Line energies and intensities
   - Mass attenuation coefficients

2. Implement spectrum fitting engine:
   - Background modeling (SNIP algorithm, polynomial)
   - Peak fitting (Gaussian/Voigt profiles)
   - Escape peak handling
   - Pile-up correction

3. Element identification:
   - Automatic peak finding
   - Element suggestion based on detected peaks
   - K, L, M line series recognition

**Phase 3 - Quantification:**
1. Fundamental parameters quantification:
   - Matrix correction calculations
   - Secondary fluorescence effects
   - Absorption corrections
   - Geometric factor handling

2. Calibration options:
   - Standardless analysis
   - Internal standard method
   - Reference material calibration

**Phase 4 - Polish & UX:**
1. Modern styling:
   - Dark/light theme toggle using Qt Style Sheets (QSS)
   - Custom color schemes for plots
   - Consistent iconography (use Qt built-in icons or Material Design icons)

2. Performance optimization:
   - Use QThread for long-running calculations
   - Progress bars for fitting operations
   - Responsive UI during computation

3. Data export:
   - High-quality plot export (PNG, SVG, PDF)
   - Results to CSV/Excel
   - Report generation

### Design Guidelines

**Visual Aesthetics:**
- Use a professional scientific color palette (matplotlib's 'viridis' or 'plasma' for data)
- Clean, minimalist interface with clear visual hierarchy
- Proper spacing and alignment using Qt layouts
- Tooltips on all controls for user guidance

**Plotting Best Practices:**
- Use PyQtGraph's PlotWidget for main spectrum display
- Enable antialiasing for smooth line rendering
- Provide grid lines (toggleable)
- Show proper axis labels with units (Energy [keV], Counts)
- Color-code fitted vs. measured spectra
- Display background and individual peak contributions

**Code Organization:**
```
xrf_analysis/
├── main.py                 # Application entry point
├── ui/
│   ├── main_window.py     # Main window class
│   ├── spectrum_widget.py # Custom spectrum display widget
│   ├── element_panel.py   # Element selection panel
│   └── results_panel.py   # Results display panel
├── core/
│   ├── spectrum.py        # Spectrum data class
│   ├── fitting.py         # Fitting algorithms
│   ├── quantification.py  # FP quantification engine
│   └── database.py        # xraylib/XrayDB interface
├── utils/
│   ├── io_handler.py      # File I/O utilities
│   └── calculations.py    # Helper calculations
└── resources/
    ├── icons/             # Application icons
    └── styles.qss         # Qt Style Sheet
```

### Initial Implementation Steps

1. **Set up project structure** with virtual environment
2. **Install dependencies:**
   ```bash
   pip install PySide6 pyqtgraph xraylib numpy scipy pandas h5py
   ```

3. **Create basic main window** with menu bar and central widget
4. **Implement PyQtGraph PlotWidget** for spectrum display
5. **Add file loading** for simple text format spectra
6. **Test basic plotting** with sample XRF data

### Advanced Features (Future)

- Batch processing multiple spectra
- Spectral imaging/mapping visualization
- Database of reference spectra
- Machine learning peak identification
- Remote database integration
- Collaboration features (export/share configurations)

### Testing Approach

- Create unit tests for fitting algorithms
- Use sample XRF spectra from public databases for validation
- Compare quantification results with PyMca output
- Test with various spectrum formats

### Documentation

- Inline docstrings for all functions
- User manual with screenshots
- Example workflows
- Theory documentation for FP method

---

**Start with Phase 1** and build incrementally. Focus on creating a solid, extensible architecture that can accommodate future features. Prioritize user experience and visual polish from the beginning, as scientific users appreciate tools that are both powerful and pleasant to use.