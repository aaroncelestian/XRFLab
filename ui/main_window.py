"""
Main window for XRF Fundamental Parameters Analysis Application
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QToolBar, QStatusBar, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence, QIcon

from ui.spectrum_widget import SpectrumWidget
from ui.element_panel import ElementPanel
from ui.results_panel import ResultsPanel
from utils.io_handler import IOHandler


class MainWindow(QMainWindow):
    """Main application window with menu, toolbar, and panels"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRF Fundamental Parameters Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.io_handler = IOHandler()
        self.current_spectrum = None
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
        """Create the main layout with three panels"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Element selection and parameters
        self.element_panel = ElementPanel()
        main_splitter.addWidget(self.element_panel)
        
        # Center panel - Spectrum display
        self.spectrum_widget = SpectrumWidget()
        main_splitter.addWidget(self.spectrum_widget)
        
        # Right panel - Results
        self.results_panel = ResultsPanel()
        main_splitter.addWidget(self.results_panel)
        
        # Set initial sizes (30%, 50%, 20%)
        main_splitter.setSizes([420, 700, 280])
        
        layout.addWidget(main_splitter)
        
        # Connect signals
        self.element_panel.elements_changed.connect(self.on_elements_changed)
        self.element_panel.fit_requested.connect(self.fit_spectrum)
    
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
        # TODO: Implement spectrum fitting
        self.status_bar.showMessage("Fitting complete", 5000)
    
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
            "About XRF Analysis",
            "<h3>XRF Fundamental Parameters Analysis</h3>"
            "<p>Version 1.0.0</p>"
            "<p>A professional application for X-ray fluorescence spectroscopy analysis "
            "using fundamental parameters method.</p>"
            "<p>Built with PySide6, PyQtGraph, and xraylib.</p>"
        )
    
    def on_elements_changed(self, elements):
        """Handle element selection changes"""
        # TODO: Update spectrum display with selected elements
        pass
    
    def closeEvent(self, event):
        """Handle window close event"""
        self._save_settings()
        event.accept()
