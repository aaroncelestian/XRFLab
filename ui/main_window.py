"""
Main window for XRF Fundamental Parameters Analysis Application
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QToolBar, QStatusBar, QMessageBox, QFileDialog,
    QTabWidget
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence, QIcon

from ui.spectrum_widget import SpectrumWidget
from ui.element_panel import ElementPanel
from ui.results_panel import ResultsPanel
from ui.standards_panel import StandardsPanel
from ui.fwhm_calibration_panel import FWHMCalibrationPanel
from utils.io_handler import IOHandler
from core.fitting import SpectrumFitter


class MainWindow(QMainWindow):
    """Main application window with menu, toolbar, and panels"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRFLab - Fundamental Parameters Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.io_handler = IOHandler()
        self.fitter = SpectrumFitter()
        self.current_spectrum = None
        self.fit_result = None
        self.settings = QSettings()
        
        # Setup UI
        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self._create_central_widget()
        self._create_status_bar()
        self._load_stylesheet()
        
        # Restore window state
        self._restore_settings()
    
    def _create_actions(self):
        """Create all menu and toolbar actions"""
        # File actions
        self.open_action = QAction("&Open Spectrum...", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.setStatusTip("Open an XRF spectrum file")
        self.open_action.triggered.connect(self.open_spectrum)
        
        self.save_project_action = QAction("&Save Project...", self)
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_action.setStatusTip("Save current project")
        self.save_project_action.triggered.connect(self.save_project)
        
        self.export_results_action = QAction("&Export Results...", self)
        self.export_results_action.setStatusTip("Export analysis results")
        self.export_results_action.triggered.connect(self.export_results)
        
        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.setStatusTip("Exit application")
        self.exit_action.triggered.connect(self.close)
        
        # Analysis actions
        self.fit_spectrum_action = QAction("&Fit Spectrum", self)
        self.fit_spectrum_action.setShortcut("Ctrl+F")
        self.fit_spectrum_action.setStatusTip("Fit the current spectrum")
        self.fit_spectrum_action.triggered.connect(self.fit_spectrum)
        
        self.quantify_action = QAction("&Quantification", self)
        self.quantify_action.setShortcut("Ctrl+Q")
        self.quantify_action.setStatusTip("Perform quantitative analysis")
        self.quantify_action.triggered.connect(self.quantify)
        
        self.background_action = QAction("&Background Settings...", self)
        self.background_action.setStatusTip("Configure background removal")
        self.background_action.triggered.connect(self.configure_background)
        
        # View actions
        self.toggle_log_action = QAction("&Logarithmic Y-axis", self)
        self.toggle_log_action.setCheckable(True)
        self.toggle_log_action.setChecked(True)
        self.toggle_log_action.setStatusTip("Toggle logarithmic Y-axis")
        self.toggle_log_action.triggered.connect(self.toggle_log_scale)
        
        self.toggle_grid_action = QAction("Show &Grid", self)
        self.toggle_grid_action.setCheckable(True)
        self.toggle_grid_action.setChecked(True)
        self.toggle_grid_action.setStatusTip("Toggle grid display")
        self.toggle_grid_action.triggered.connect(self.toggle_grid)
        
        self.toggle_theme_action = QAction("&Dark Theme", self)
        self.toggle_theme_action.setCheckable(True)
        self.toggle_theme_action.setStatusTip("Toggle dark/light theme")
        self.toggle_theme_action.triggered.connect(self.toggle_theme)
        
        # Tools actions
        self.calibration_action = QAction("Energy &Calibration...", self)
        self.calibration_action.setStatusTip("Calibrate energy axis")
        self.calibration_action.triggered.connect(self.calibrate_energy)
        
        self.element_db_action = QAction("&Element Database...", self)
        self.element_db_action.setStatusTip("View element database")
        self.element_db_action.triggered.connect(self.show_element_database)
        
        # Help actions
        self.about_action = QAction("&About", self)
        self.about_action.setStatusTip("About this application")
        self.about_action.triggered.connect(self.show_about)
    
    def _create_menus(self):
        """Create menu bar and menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.export_results_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        analysis_menu.addAction(self.fit_spectrum_action)
        analysis_menu.addAction(self.quantify_action)
        analysis_menu.addSeparator()
        analysis_menu.addAction(self.background_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.toggle_log_action)
        view_menu.addAction(self.toggle_grid_action)
        view_menu.addSeparator()
        view_menu.addAction(self.toggle_theme_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction(self.calibration_action)
        tools_menu.addAction(self.element_db_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.about_action)
    
    def _create_toolbar(self):
        """Create toolbar with quick-access buttons"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolbar")  # Set object name to avoid warning
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_project_action)
        toolbar.addSeparator()
        toolbar.addAction(self.fit_spectrum_action)
        toolbar.addAction(self.quantify_action)
        toolbar.addSeparator()
        toolbar.addAction(self.toggle_log_action)
    
    def _create_central_widget(self):
        """Create the main layout with tabs for Analysis and Calibration"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Analysis tab (main interface)
        analysis_tab = self._create_analysis_tab()
        self.tab_widget.addTab(analysis_tab, "Analysis")
        
        # Standards tab (intensity calibration using known concentrations)
        self.standards_panel = StandardsPanel()
        self.standards_panel.calibration_complete.connect(self.on_calibration_applied)
        self.tab_widget.addTab(self.standards_panel, "Standards")
        
        # FWHM Calibration tab (detector resolution calibration)
        self.fwhm_calibration_panel = FWHMCalibrationPanel()
        self.fwhm_calibration_panel.calibration_complete.connect(self.on_fwhm_calibration_applied)
        self.tab_widget.addTab(self.fwhm_calibration_panel, "FWHM Calibration")
        
        layout.addWidget(self.tab_widget)
    
    def _create_analysis_tab(self):
        """Create the analysis tab with improved layout"""
        analysis_widget = QWidget()
        layout = QHBoxLayout(analysis_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main horizontal splitter (left panel | right side)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Element selection and parameters
        self.element_panel = ElementPanel()
        main_splitter.addWidget(self.element_panel)
        
        # Right side - Vertical splitter (spectrum on top | results below)
        right_splitter = QSplitter(Qt.Vertical)
        
        # Top: Spectrum display with residuals
        self.spectrum_widget = SpectrumWidget()
        right_splitter.addWidget(self.spectrum_widget)
        
        # Bottom: Results panels in horizontal layout
        results_widget = QWidget()
        results_layout = QHBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(5)
        
        # Create results panel (will be split into three sections)
        self.results_panel = ResultsPanel()
        results_layout.addWidget(self.results_panel)
        
        right_splitter.addWidget(results_widget)
        
        # Set sizes for vertical splitter (70% spectrum, 30% results)
        right_splitter.setSizes([700, 300])
        
        main_splitter.addWidget(right_splitter)
        
        # Set initial sizes for horizontal splitter (30% left, 70% right)
        main_splitter.setSizes([400, 1000])
        
        layout.addWidget(main_splitter)
        
        # Connect signals
        self.element_panel.elements_changed.connect(self.on_elements_changed)
        self.element_panel.fit_requested.connect(self.fit_spectrum)
        self.element_panel.element_clicked.connect(self.on_element_clicked)
        
        return analysis_widget
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _load_stylesheet(self):
        """Load and apply Qt stylesheet"""
        try:
            with open("resources/styles.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            # Use default styling if stylesheet not found
            pass
    
    def _restore_settings(self):
        """Restore window settings from previous session"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def _save_settings(self):
        """Save window settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
    
    # Action handlers
    def open_spectrum(self):
        """Open an XRF spectrum file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open XRF Spectrum",
            "",
            "All Supported (*.txt *.csv *.mca *.h5 *.hdf5);;Text Files (*.txt);;CSV Files (*.csv);;MCA Files (*.mca);;HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        
        if file_path:
            try:
                spectrum = self.io_handler.load_spectrum(file_path)
                self.current_spectrum = spectrum
                self.spectrum_widget.set_spectrum(spectrum)
                self.calibration_panel.set_spectrum(spectrum)  # Pass to calibration panel
                
                # Auto-populate experimental parameters from spectrum metadata
                if hasattr(spectrum, 'metadata') and spectrum.metadata:
                    self.element_panel.update_from_spectrum_metadata(spectrum.metadata)
                
                self.status_bar.showMessage(f"Loaded: {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Spectrum",
                    f"Failed to load spectrum:\n{str(e)}"
                )
    
    def save_project(self):
        """Save current project"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "XRF Project (*.xrfp);;All Files (*)"
        )
        
        if file_path:
            try:
                # TODO: Implement project saving
                self.status_bar.showMessage(f"Saved: {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving Project",
                    f"Failed to save project:\n{str(e)}"
                )
    
    def export_results(self):
        """Export analysis results"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            try:
                results = self.results_panel.get_results()
                self.io_handler.export_results(results, file_path)
                self.status_bar.showMessage(f"Exported: {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Exporting Results",
                    f"Failed to export results:\n{str(e)}"
                )
    
    def fit_spectrum(self):
        """Fit the current spectrum"""
        if self.current_spectrum is None:
            QMessageBox.warning(
                self,
                "No Spectrum",
                "Please load a spectrum first."
            )
            return
        
        self.status_bar.showMessage("Fitting spectrum...", 0)
        
        try:
            # Get selected elements
            elements = self.element_panel.get_selected_elements()
            
            # Get fitting parameters
            fit_params = self.element_panel.get_fitting_params()
            background_method = fit_params['background_method'].lower()
            peak_shape = fit_params['peak_shape'].lower()
            
            # Get experimental parameters
            exp_params = self.element_panel.get_experimental_params()
            
            # Perform fitting (pass all parameters including tube lines and experimental params)
            self.fit_result = self.fitter.fit_spectrum(
                energy=self.current_spectrum.energy,
                counts=self.current_spectrum.counts,
                elements=elements,
                background_method=background_method,
                peak_shape=peak_shape,
                auto_find_peaks=True,
                tube_element=fit_params.get('tube_element', 'Rh'),
                excitation_kv=fit_params.get('excitation_kv', 50.0),
                include_tube_lines=fit_params.get('include_tube_lines', True),
                experimental_params=exp_params
            )
            
            # Update spectrum display
            self.spectrum_widget.set_fitted_spectrum(self.fit_result.fitted_spectrum)
            self.spectrum_widget.set_background(self.fit_result.background)
            
            # Update results panel
            self.results_panel.set_fit_statistics(self.fit_result.statistics)
            self.results_panel.set_peaks(self.fit_result.peaks)
            
            # Perform quantification
            exp_params = self.element_panel.get_experimental_params()
            concentrations = self.fitter.quantify_elements(
                self.fit_result.peaks, exp_params
            )
            self.results_panel.set_quantification(concentrations)
            
            self.status_bar.showMessage(
                f"Fitting complete: {len(self.fit_result.peaks)} peaks fitted, "
                f"χ²ᵣ = {self.fit_result.statistics['reduced_chi_squared']:.2f}",
                5000
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Fitting Error",
                f"An error occurred during fitting:\n{str(e)}"
            )
            self.status_bar.showMessage("Fitting failed", 5000)
    
    def quantify(self):
        """Perform quantitative analysis"""
        if self.current_spectrum is None:
            QMessageBox.warning(
                self,
                "No Spectrum",
                "Please load a spectrum first."
            )
            return
        
        self.status_bar.showMessage("Performing quantification...", 0)
        # TODO: Implement quantification
        self.status_bar.showMessage("Quantification complete", 5000)
    
    def configure_background(self):
        """Configure background removal settings"""
        # TODO: Implement background configuration dialog
        pass
    
    def toggle_log_scale(self, checked):
        """Toggle logarithmic Y-axis"""
        self.spectrum_widget.set_log_scale(checked)
    
    def toggle_grid(self, checked):
        """Toggle grid display"""
        self.spectrum_widget.set_grid(checked)
    
    def toggle_theme(self, checked):
        """Toggle between dark and light theme"""
        # TODO: Implement theme switching
        pass
    
    def calibrate_energy(self):
        """Open energy calibration dialog"""
        # TODO: Implement calibration dialog
        pass
    
    def show_element_database(self):
        """Show element database viewer"""
        # TODO: Implement element database viewer
        pass
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About XRFLab",
            "<h3>XRFLab</h3>"
            "<p>Version 1.0.0</p>"
            "<p>A professional application for X-ray fluorescence spectroscopy analysis "
            "using fundamental parameters method.</p>"
            "<p>Built with PySide6, PyQtGraph, and xraylib.</p>"
        )
    
    def on_elements_changed(self, elements):
        """Handle element selection changes"""
        # TODO: Update spectrum display with selected elements
        pass
    
    def on_element_clicked(self, symbol, z):
        """Handle element click - show emission lines on spectrum"""
        # Clear existing markers
        self.spectrum_widget.clear_peak_markers()
        
        # Show emission lines for clicked element
        self.spectrum_widget.show_element_lines(symbol, z)
        
        self.status_bar.showMessage(f"Showing emission lines for {symbol} (Z={z})", 3000)
    
    def on_fwhm_calibration_applied(self, fwhm_calibration):
        """Handle FWHM calibration being applied"""
        # Update the Standards panel with the FWHM calibration
        self.standards_panel.update_fwhm_status(fwhm_calibration)
        
        # Show status message
        if fwhm_calibration.model_type == 'detector':
            fwhm_0_ev = fwhm_calibration.parameters['fwhm_0'] * 1000
            epsilon_ev = fwhm_calibration.parameters['epsilon'] * 1000
            self.status_bar.showMessage(
                f"FWHM calibration applied to Standards: FWHM₀={fwhm_0_ev:.1f} eV, "
                f"ε={epsilon_ev:.2f} eV/keV (R²={fwhm_calibration.r_squared:.4f})",
                5000
            )
        else:
            self.status_bar.showMessage(
                f"FWHM calibration applied to Standards: {fwhm_calibration.model_type} model "
                f"(R²={fwhm_calibration.r_squared:.4f})",
                5000
            )
    
    def on_calibration_applied(self, calibration_result):
        """Handle calibration being applied"""
        self.status_bar.showMessage(
            f"Calibration applied: FWHM₀={calibration_result.fwhm_0*1000:.1f} eV, "
            f"ε={calibration_result.epsilon*1000:.2f} eV",
            5000
        )
        
        # Switch back to analysis tab
        self.tab_widget.setCurrentIndex(0)
    
    def closeEvent(self, event):
        """Handle window close event"""
        self._save_settings()
        event.accept()
