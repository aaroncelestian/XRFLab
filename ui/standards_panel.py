"""
Standards Calibration Panel UI

This panel focuses on intensity calibration using reference standards with known concentrations.
FWHM parameters are taken from the FWHM Calibration tab and held fixed during optimization.
The goal is to match calculated intensities to measured intensities for accurate quantification.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QPushButton, QLabel, QLineEdit, QTextEdit,
                               QFileDialog, QProgressBar, QMessageBox, QSplitter,
                               QCheckBox, QDoubleSpinBox, QListWidget, QListWidgetItem,
                               QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget)
from PySide6.QtCore import Qt, Signal, QThread, QStandardPaths
from pathlib import Path
import pyqtgraph as pg
import numpy as np
import json
import csv
from typing import Dict, List

from core.calibration import InstrumentCalibrator, CalibrationResult
from ui.concentration_entry_dialog import ConcentrationEntryDialog
from utils.io_handler import IOHandler


# Built-in standards library (empty by default - users load their own)
STANDARDS_LIBRARY = {}


class CalibrationWorker(QThread):
    """Worker thread for running calibration"""
    finished = Signal(object)  # CalibrationResult
    progress = Signal(str)  # Progress message
    
    def __init__(self, calibrator, energy, counts, concentrations, excitation_energy, 
                 experimental_params=None, use_measured_intensities=True, bg_params=None):
        super().__init__()
        self.calibrator = calibrator
        self.energy = energy
        self.counts = counts
        self.concentrations = concentrations
        self.excitation_energy = excitation_energy
        self.experimental_params = experimental_params
        self.use_measured_intensities = use_measured_intensities
        self.bg_params = bg_params or {}
    
    def run(self):
        """Run calibration in background thread"""
        try:
            self.progress.emit("Starting intensity calibration...")
            self.progress.emit("Note: FWHM parameters are fixed from FWHM Calibration")
            result = self.calibrator.calibrate(
                self.energy,
                self.counts,
                self.concentrations,
                self.excitation_energy,
                use_measured_intensities=self.use_measured_intensities,
                experimental_params=self.experimental_params,
                bg_params=self.bg_params
            )
            self.finished.emit(result)
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            result = CalibrationResult(
                fwhm_0=0.050,
                epsilon=0.0015,
                voigt_gamma_ratio=0.15,
                efficiency_params={},
                chi_squared=float('inf'),
                r_squared=0.0,
                success=False,
                message=str(e)
            )
            self.finished.emit(result)


class StandardsPanel(QWidget):
    """Panel for intensity calibration using reference standards with known concentrations"""
    
    calibration_complete = Signal(object)  # CalibrationResult
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.calibrator = InstrumentCalibrator()
        self.io_handler = IOHandler()
        self.current_spectrum = None
        self.reference_concentrations = None
        self.calibration_result = None
        self.worker = None
        self.selected_standards = []  # List of standards to use
        self.standards_data = {}  # Dict of {standard_name: {spectrum, concentrations, loaded}}
        
        self._init_ui()
        
        # Try to load saved calibration on startup
        self._auto_load_calibration()
    
    @staticmethod
    def get_default_calibration_path():
        """Get the default path for saving/loading Standards calibration"""
        # Use application data directory
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not app_data:
            # Fallback to home directory
            app_data = str(Path.home() / ".xrflab")
        
        # Create directory if it doesn't exist
        cal_dir = Path(app_data) / "calibrations"
        cal_dir.mkdir(parents=True, exist_ok=True)
        
        return cal_dir / "standards_calibration.json"
    
    def _init_ui(self):
        """Initialize the user interface with sub-tabs"""
        layout = QVBoxLayout(self)
        
        # Create splitter for controls and plot
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Tabbed interface for compact layout
        left_tab_widget = QTabWidget()
        left_tab_widget.setMaximumWidth(700)  # Same as Analysis tab
        
        # Tab 1: Standards Selection
        standards_tab = self._create_standards_selection_tab()
        left_tab_widget.addTab(standards_tab, "Standards")
        
        # Tab 2: Calibration & Output
        calibration_tab = self._create_calibration_tab()
        left_tab_widget.addTab(calibration_tab, "Calibration")
        
        splitter.addWidget(left_tab_widget)
        
        # Right side: Spectrum comparison plot (keep as is)
        plot_widget = self._create_plot_widget()
        splitter.addWidget(plot_widget)
        
        # Set initial sizes for horizontal splitter (50% left, 50% right)
        splitter.setSizes([600, 600])
        
        layout.addWidget(splitter)
    
    def _create_standards_selection_tab(self):
        """Create Standards Selection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        
        # FWHM status group
        fwhm_group = self._create_fwhm_status_group()
        layout.addWidget(fwhm_group)
        
        # Standards library group
        library_group = self._create_standards_library_group()
        layout.addWidget(library_group)
        
        # Selected standards group
        selected_group = self._create_selected_standards_group()
        layout.addWidget(selected_group)
        
        layout.addStretch()
        return widget
    
    def _create_calibration_tab(self):
        """Create Calibration & Output tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        
        # Calibration controls
        controls_group = self._create_controls_group()
        layout.addWidget(controls_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results display
        results_group = self._create_results_group()
        layout.addWidget(results_group, stretch=1)
        
        return widget
    
    def _create_fwhm_status_group(self):
        """Create FWHM calibration status display"""
        group = QGroupBox("FWHM Calibration Status")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)
        
        # Status label
        self.fwhm_status_label = QLabel(
            "<b>⚠️ No FWHM calibration loaded</b><br>"
            "Please run FWHM Calibration first (FWHM Calibration tab)"
        )
        self.fwhm_status_label.setWordWrap(True)
        self.fwhm_status_label.setStyleSheet("color: #cc6600;")
        layout.addWidget(self.fwhm_status_label)
        
        # Info text
        info = QLabel(
            "<small>FWHM parameters (FWHM₀, ε) are fixed during intensity calibration. "
            "This ensures detector resolution is accurately modeled while optimizing "
            "intensity scaling factors.</small>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        return group
    
    def _create_standards_library_group(self):
        """Create standards library selection"""
        group = QGroupBox("Standards Library")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)
        
        # Info
        info = QLabel(
            "<b>Select reference standards for calibration</b><br>"
            "Choose one or more standards with known concentrations"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Standards list (scrollable)
        self.standards_list = QListWidget()
        self.standards_list.setMinimumHeight(120)
        self.standards_list.setMaximumHeight(200)
        self.standards_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.standards_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        for name, info in STANDARDS_LIBRARY.items():
            item = QListWidgetItem(f"{name} - {info['description']}")
            item.setData(Qt.UserRole, name)
            self.standards_list.addItem(item)
        layout.addWidget(self.standards_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Standard")
        add_btn.clicked.connect(self._add_standard)
        btn_layout.addWidget(add_btn)
        
        load_custom_btn = QPushButton("Load Custom...")
        load_custom_btn.clicked.connect(self._load_custom_standard)
        btn_layout.addWidget(load_custom_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return group
    
    def _create_selected_standards_group(self):
        """Create selected standards display"""
        group = QGroupBox("Selected Standards")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)
        
        # Table of selected standards (scrollable)
        self.selected_table = QTableWidget()
        self.selected_table.setColumnCount(3)
        self.selected_table.setHorizontalHeaderLabels(["Standard", "Status", "Actions"])
        self.selected_table.horizontalHeader().setStretchLastSection(True)
        self.selected_table.setMinimumHeight(100)
        self.selected_table.setMaximumHeight(200)
        self.selected_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.selected_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout.addWidget(self.selected_table)
        
        # Info
        info = QLabel(
            "<small>Load spectrum and concentration data for each standard before calibration</small>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        return group
    
    def _create_controls_group(self):
        """Create calibration control buttons"""
        group = QGroupBox("Calibration")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)
        
        # Background method selection
        bg_method_layout = QHBoxLayout()
        bg_method_layout.addWidget(QLabel("Background Method:"))
        
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(["AsLS (Recommended)", "SNIP", "Polynomial", "Linear", "None"])
        self.bg_method_combo.setCurrentIndex(0)  # Default to AsLS
        self.bg_method_combo.currentIndexChanged.connect(self._on_bg_method_changed)
        self.bg_method_combo.setToolTip(
            "AsLS: Asymmetric Least Squares (best for XRF)\n"
            "SNIP: Statistics-sensitive Non-linear Iterative Peak-clipping\n"
            "Polynomial: Polynomial fit\n"
            "Linear: Simple linear baseline"
        )
        bg_method_layout.addWidget(self.bg_method_combo)
        bg_method_layout.addStretch()
        layout.addLayout(bg_method_layout)
        
        # AsLS parameters (default)
        self.als_params_widget = QWidget()
        als_layout = QHBoxLayout(self.als_params_widget)
        als_layout.setContentsMargins(0, 0, 0, 0)
        
        als_layout.addWidget(QLabel("λ (smoothness):"))
        self.als_lam_spin = QDoubleSpinBox()
        self.als_lam_spin.setRange(1e3, 1e7)
        self.als_lam_spin.setValue(1e5)
        self.als_lam_spin.setDecimals(0)
        self.als_lam_spin.setSingleStep(1e4)
        self.als_lam_spin.setToolTip("Smoothness: 10³ to 10⁷ (higher = smoother)")
        als_layout.addWidget(self.als_lam_spin)
        
        als_layout.addWidget(QLabel("p (asymmetry):"))
        self.als_p_spin = QDoubleSpinBox()
        self.als_p_spin.setRange(0.001, 0.05)
        self.als_p_spin.setValue(0.01)
        self.als_p_spin.setDecimals(3)
        self.als_p_spin.setSingleStep(0.001)
        self.als_p_spin.setToolTip("Asymmetry: 0.001 to 0.05 (lower = tighter fit)")
        als_layout.addWidget(self.als_p_spin)
        
        als_layout.addStretch()
        layout.addWidget(self.als_params_widget)
        
        # SNIP parameters (hidden by default)
        self.snip_params_widget = QWidget()
        snip_layout = QHBoxLayout(self.snip_params_widget)
        snip_layout.setContentsMargins(0, 0, 0, 0)
        
        snip_layout.addWidget(QLabel("Iterations:"))
        self.snip_iter_spin = QDoubleSpinBox()
        self.snip_iter_spin.setRange(5, 100)
        self.snip_iter_spin.setValue(20)
        self.snip_iter_spin.setDecimals(0)
        self.snip_iter_spin.setSingleStep(5)
        self.snip_iter_spin.setToolTip("Number of iterations (higher = smoother)")
        snip_layout.addWidget(self.snip_iter_spin)
        snip_layout.addStretch()
        layout.addWidget(self.snip_params_widget)
        self.snip_params_widget.setVisible(False)
        
        # Apply background button
        apply_bg_layout = QHBoxLayout()
        self.apply_bg_btn = QPushButton("Preview Background")
        self.apply_bg_btn.setToolTip("Preview background subtraction with current parameters")
        self.apply_bg_btn.setEnabled(False)
        apply_bg_layout.addWidget(self.apply_bg_btn)
        apply_bg_layout.addStretch()
        layout.addLayout(apply_bg_layout)
        
        # Main buttons
        btn_layout = QHBoxLayout()
        
        self.calibrate_btn = QPushButton("Run Intensity Calibration")
        self.calibrate_btn.clicked.connect(self._run_calibration)
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setToolTip("Optimize intensity scaling to match known concentrations")
        btn_layout.addWidget(self.calibrate_btn)
        
        self.apply_btn = QPushButton("Apply Calibration")
        self.apply_btn.clicked.connect(self._apply_calibration)
        self.apply_btn.setEnabled(False)
        btn_layout.addWidget(self.apply_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Save/Load
        save_load_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Calibration...")
        self.save_btn.clicked.connect(self._save_calibration)
        self.save_btn.setEnabled(False)
        save_load_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("Load Calibration...")
        self.load_btn.clicked.connect(self._load_calibration)
        save_load_layout.addWidget(self.load_btn)
        
        save_load_layout.addStretch()
        layout.addLayout(save_load_layout)
        
        return group
    
    def _create_results_group(self):
        """Create results display group"""
        group = QGroupBox("Calibration Output")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)
        
        # Progress output
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setMaximumHeight(80)
        self.terminal_output.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "font-family: 'Courier New', monospace; font-size: 10pt; }"
        )
        self.terminal_output.setPlainText("Ready for calibration...")
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.terminal_output)
        
        # Results summary
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(120)
        self.results_text.setMinimumHeight(100)
        self.results_text.setPlainText("No calibration results yet")
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_text)
        
        return group
    
    def _create_plot_widget(self):
        """Create spectrum comparison plot"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot with two subplots
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')
        
        # Top plot: Measured vs Calculated
        self.spectrum_plot = self.plot_widget.addPlot(row=0, col=0)
        self.spectrum_plot.setLabel('left', 'Counts', color='k')
        self.spectrum_plot.setLabel('bottom', 'Energy (keV)', color='k')
        self.spectrum_plot.setTitle('Intensity Calibration Fit', color='k')
        self.spectrum_plot.addLegend()
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.measured_curve = self.spectrum_plot.plot(
            pen=pg.mkPen('#00008B', width=2), name='Measured'
        )
        self.calculated_curve = self.spectrum_plot.plot(
            pen=pg.mkPen('r', width=2, style=Qt.DashLine), name='Calculated'
        )
        self.background_curve = self.spectrum_plot.plot(
            pen=pg.mkPen('#FFA500', width=1, style=Qt.DotLine), name='Background'
        )
        
        # Bottom plot: Residuals
        self.residual_plot = self.plot_widget.addPlot(row=1, col=0)
        self.residual_plot.setLabel('left', 'Residuals (σ)', color='k')
        self.residual_plot.setLabel('bottom', 'Energy (keV)', color='k')
        self.residual_plot.setTitle('Fit Residuals', color='k')
        self.residual_plot.showGrid(x=True, y=True, alpha=0.3)
        self.residual_plot.addLine(y=0, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        
        self.residual_curve = self.residual_plot.plot(
            pen=None, symbol='o', symbolSize=5, symbolBrush='b'
        )
        
        layout.addWidget(self.plot_widget)
        
        return widget
    
    def update_fwhm_status(self, fwhm_calibration):
        """Update FWHM status when calibration is applied"""
        if fwhm_calibration:
            # Update the calibrator with the FWHM calibration
            self.calibrator.fwhm_calibration = fwhm_calibration
            # Get calibration date
            cal_date = fwhm_calibration.calibration_date
            if cal_date:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(cal_date)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    date_str = "Unknown"
            else:
                date_str = "Unknown"
            
            if fwhm_calibration.model_type == 'detector':
                fwhm_0_ev = fwhm_calibration.parameters['fwhm_0'] * 1000
                epsilon_ev = fwhm_calibration.parameters['epsilon'] * 1000
                status_text = (
                    f"<b>✓ FWHM Calibration Active</b><br>"
                    f"FWHM₀ = {fwhm_0_ev:.1f} eV<br>"
                    f"ε = {epsilon_ev:.2f} eV/keV<br>"
                    f"R² = {fwhm_calibration.r_squared:.4f}<br>"
                    f"<small>Calibrated: {date_str}</small><br>"
                    f"<small>Auto-saved and will persist between sessions</small>"
                )
                self.fwhm_status_label.setStyleSheet("color: green;")
            else:
                status_text = (
                    f"<b>✓ FWHM Calibration Active</b><br>"
                    f"Model: {fwhm_calibration.model_type}<br>"
                    f"R² = {fwhm_calibration.r_squared:.4f}<br>"
                    f"<small>Calibrated: {date_str}</small><br>"
                    f"<small>Auto-saved and will persist between sessions</small>"
                )
                self.fwhm_status_label.setStyleSheet("color: green;")
            
            self.fwhm_status_label.setText(status_text)
            
            # Update calibrator with FWHM calibration
            self.calibrator = InstrumentCalibrator(fwhm_calibration=fwhm_calibration)
        else:
            self.fwhm_status_label.setText(
                "<b>⚠️ No FWHM calibration loaded</b><br>"
                "Please run FWHM Calibration first (FWHM Calibration tab)"
            )
            self.fwhm_status_label.setStyleSheet("color: #cc6600;")
    
    def _add_standard(self):
        """Add selected standard to calibration list"""
        current_item = self.standards_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a standard from the library.")
            return
        
        standard_name = current_item.data(Qt.UserRole)
        
        # Check if already added
        if standard_name in self.standards_data:
            QMessageBox.information(self, "Already Added", f"{standard_name} is already in the list.")
            return
        
        # Add to table
        row = self.selected_table.rowCount()
        self.selected_table.insertRow(row)
        
        self.selected_table.setItem(row, 0, QTableWidgetItem(standard_name))
        self.selected_table.setItem(row, 1, QTableWidgetItem("Not loaded"))
        
        # Add load button
        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(lambda: self._load_standard_data(standard_name, row))
        self.selected_table.setCellWidget(row, 2, load_btn)
        
        # Initialize data storage
        self.standards_data[standard_name] = {"loaded": False}
    
    def _load_custom_standard(self):
        """Load custom standard and add to library"""
        from PySide6.QtWidgets import QInputDialog
        
        # Ask for standard name
        standard_name, ok = QInputDialog.getText(
            self,
            "New Standard Name",
            "Enter a name for your new standard:\n"
            "(This will be added to the standards library)",
            text="My Standard"
        )
        
        if not ok or not standard_name.strip():
            return
        
        standard_name = standard_name.strip()
        
        # Check if already exists in library
        if standard_name in STANDARDS_LIBRARY:
            reply = QMessageBox.question(
                self,
                "Standard Exists",
                f"A standard named '{standard_name}' already exists in the library.\n\n"
                "Do you want to replace it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # Load the data first
        spectrum_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select XRF SPECTRUM File for {standard_name}",
            "",
            "All Supported (*.txt *.csv *.mca);;Text Files (*.txt);;CSV Files (*.csv);;MCA Files (*.mca)"
        )
        
        if not spectrum_path:
            return
        
        try:
            spectrum = self.io_handler.load_spectrum(spectrum_path)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Spectrum",
                f"Failed to load XRF spectrum file:\n{str(e)}\n\n"
                f"⚠️ Make sure you selected the SPECTRUM file (XRF data with energy/counts),\n"
                f"NOT the concentration CSV file.\n\n"
                f"You'll be asked for the concentration file in the next step."
            )
            return
        
        # Load or enter concentrations
        concentrations = self._load_or_enter_concentrations(standard_name)
        
        if not concentrations:
            return
        
        # Add to library
        STANDARDS_LIBRARY[standard_name] = {
            "description": "User-defined standard",
            "matrix": "custom",
            "use_case": "Custom calibration"
        }
        
        # Add to library list widget
        item = QListWidgetItem(f"{standard_name} - User-defined standard")
        item.setData(Qt.UserRole, standard_name)
        self.standards_list.addItem(item)
        
        # Store the data
        self.standards_data[standard_name] = {
            'spectrum': spectrum,
            'concentrations': concentrations,
            'loaded': True
        }
        
        # Add to selected standards table
        row = self.selected_table.rowCount()
        self.selected_table.insertRow(row)
        
        # Standard name
        name_item = QTableWidgetItem(standard_name)
        self.selected_table.setItem(row, 0, name_item)
        
        # Status
        status_item = QTableWidgetItem("✓ Loaded")
        status_item.setForeground(Qt.green)
        self.selected_table.setItem(row, 1, status_item)
        
        # Add load button
        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(lambda: self._load_standard_data(standard_name, row))
        self.selected_table.setCellWidget(row, 2, load_btn)
        
        # Enable calibration if we have standards
        self._check_ready_for_calibration()
        
        QMessageBox.information(
            self,
            "Standard Added to Library",
            f"Successfully added '{standard_name}' to the standards library!\n\n"
            f"Spectrum: {Path(spectrum_path).name}\n"
            f"Elements: {len(concentrations)}\n"
            f"Total concentration: {sum(concentrations.values()):.2f} wt%\n\n"
            f"This standard is now available in the library for future use."
        )
    
    def _load_standard_data(self, standard_name, row):
        """Load spectrum and concentration data for a standard"""
        # Step 1: Load spectrum file
        spectrum_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Spectrum File for {standard_name}",
            "",
            "All Supported (*.txt *.csv *.mca);;Text Files (*.txt);;CSV Files (*.csv);;MCA Files (*.mca)"
        )
        
        if not spectrum_path:
            return
        
        try:
            spectrum = self.io_handler.load_spectrum(spectrum_path)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Spectrum",
                f"Failed to load spectrum:\n{str(e)}"
            )
            return
        
        # Step 2: Load or enter concentrations
        concentrations = self._load_or_enter_concentrations(standard_name)
        
        if not concentrations:
            return
        
        # Store the data
        self.standards_data[standard_name] = {
            'spectrum': spectrum,
            'concentrations': concentrations,
            'loaded': True
        }
        
        # Update table status
        status_item = QTableWidgetItem("✓ Loaded")
        status_item.setForeground(Qt.green)
        self.selected_table.setItem(row, 1, status_item)
        
        # Enable calibration if we have standards
        self._check_ready_for_calibration()
        
        QMessageBox.information(
            self,
            "Standard Loaded",
            f"Successfully loaded {standard_name}:\n\n"
            f"Spectrum: {Path(spectrum_path).name}\n"
            f"Elements: {len(concentrations)}\n"
            f"Total concentration: {sum(concentrations.values()):.2f} wt%"
        )
    
    def _load_or_enter_concentrations(self, standard_name):
        """Load concentrations from CSV or enter manually"""
        # Ask user if they have a CSV file
        reply = QMessageBox.question(
            self,
            "Concentration Data",
            f"Do you have a CSV file with element concentrations for {standard_name}?\n\n"
            "CSV format should have columns: Element, Concentration\n"
            "Example:\n"
            "  Si, 32.5\n"
            "  Al, 10.2\n"
            "  Fe, 5.8",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        
        if reply == QMessageBox.Cancel:
            return None
        elif reply == QMessageBox.Yes:
            return self._load_concentrations_from_csv()
        else:
            return self._enter_concentrations_manually(standard_name)
    
    def _load_concentrations_from_csv(self):
        """Load concentrations from CSV file"""
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Concentration CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not csv_path:
            return None
        
        try:
            concentrations = {}
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                
                # Read first row to detect format
                first_row = next(reader, None)
                if not first_row:
                    return None
                
                # Detect column indices
                element_col = None
                conc_col = None
                
                # Check if first row is header
                header = [col.lower().strip() for col in first_row]
                
                # Look for element/symbol column
                for i, col in enumerate(header):
                    if 'symbol' in col or col == 'element':
                        element_col = i
                        break
                
                # Look for concentration column
                for i, col in enumerate(header):
                    if 'concentration' in col or 'conc' in col:
                        conc_col = i
                        break
                
                # If we found headers, use them
                if element_col is not None and conc_col is not None:
                    # Process data rows
                    for row in reader:
                        if len(row) > max(element_col, conc_col):
                            element = row[element_col].strip()
                            try:
                                conc_str = row[conc_col].strip()
                                if conc_str:
                                    conc = float(conc_str)
                                    # Convert mg/kg to wt% if needed (mg/kg / 10000 = wt%)
                                    if conc > 100:  # Likely mg/kg
                                        conc = conc / 10000.0
                                    if conc > 0:
                                        concentrations[element] = conc
                            except (ValueError, IndexError):
                                continue
                else:
                    # No header found, assume simple format: Element, Concentration
                    # Try to parse first row as data
                    try:
                        element = first_row[0].strip()
                        conc = float(first_row[1])
                        if conc > 100:  # Likely mg/kg
                            conc = conc / 10000.0
                        if conc > 0:
                            concentrations[element] = conc
                    except (ValueError, IndexError):
                        pass  # First row was header, skip it
                    
                    # Process remaining rows
                    for row in reader:
                        if len(row) >= 2:
                            element = row[0].strip()
                            try:
                                conc = float(row[1])
                                if conc > 100:  # Likely mg/kg
                                    conc = conc / 10000.0
                                if conc > 0:
                                    concentrations[element] = conc
                            except ValueError:
                                continue
            
            if not concentrations:
                QMessageBox.warning(
                    self,
                    "No Data",
                    "No valid concentration data found in CSV file.\n\n"
                    "Expected format:\n"
                    "- With headers: Symbol, Concentration (or similar)\n"
                    "- Without headers: Element, Concentration\n"
                    "- Concentrations in wt% or mg/kg"
                )
                return None
            
            return concentrations
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading CSV",
                f"Failed to load CSV file:\n{str(e)}"
            )
            return None
    
    def _enter_concentrations_manually(self, standard_name):
        """Enter concentrations manually via dialog"""
        from PySide6.QtWidgets import QDialog
        dialog = ConcentrationEntryDialog(standard_name, self)
        
        if dialog.exec() == QDialog.Accepted:
            return dialog.get_concentrations()
        
        return None
    
    def _on_bg_method_changed(self, index):
        """Handle background method selection change"""
        method = self.bg_method_combo.currentText()
        
        # Hide all parameter widgets
        self.als_params_widget.setVisible(False)
        self.snip_params_widget.setVisible(False)
        
        # Show relevant parameters
        if "AsLS" in method:
            self.als_params_widget.setVisible(True)
        elif "SNIP" in method:
            self.snip_params_widget.setVisible(True)
    
    def _check_ready_for_calibration(self):
        """Check if ready to run calibration"""
        has_loaded_standards = any(
            data.get("loaded", False) for data in self.standards_data.values()
        )
        has_fwhm = self.calibrator.fwhm_calibration is not None
        
        self.calibrate_btn.setEnabled(has_loaded_standards and has_fwhm)
    
    def _auto_load_calibration(self):
        """Automatically load saved calibration on startup"""
        cal_path = self.get_default_calibration_path()
        
        if cal_path.exists():
            try:
                self.calibration_result = CalibrationResult.load(str(cal_path))
                
                # Enable buttons
                self.apply_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                
                # Display results
                self._display_calibration_results(self.calibration_result)
                
                # Update terminal output
                self.terminal_output.append(f"✓ Loaded saved Standards calibration from {cal_path}")
                
                # Auto-apply
                self.calibration_complete.emit(self.calibration_result)
                
            except Exception as e:
                # Silently fail - no calibration available
                self.terminal_output.append("No saved Standards calibration found (this is normal on first run)")
    
    def _auto_save_calibration(self):
        """Automatically save calibration to default location"""
        if self.calibration_result is None:
            return
        
        try:
            cal_path = self.get_default_calibration_path()
            self.calibration_result.save(str(cal_path))
            self.terminal_output.append(f"✓ Auto-saved Standards calibration to {cal_path}")
        except Exception as e:
            self.terminal_output.append(f"⚠ Auto-save failed: {str(e)}")
    
    def _display_calibration_results(self, result):
        """Display calibration results in the results text box"""
        if result and result.success:
            # Get calibration date
            cal_date = result.calibration_date
            if cal_date:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(cal_date)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    date_str = "Unknown"
            else:
                date_str = "Unknown"
            
            results_html = f"""
            <b>Standards Calibration Loaded</b><br><br>
            <b>Intensity Parameters:</b><br>
            Intensity Scale: {result.efficiency_params.get('intensity_scale', 'N/A'):.2f}<br>
            Rh Scatter Scale: {result.efficiency_params.get('rh_scatter_scale', 'N/A'):.4f}<br><br>
            <b>Fit Quality:</b><br>
            R² = {result.r_squared:.4f}<br>
            χ² = {result.chi_squared:.2f}<br><br>
            <small>Calibrated: {date_str}</small><br>
            <small>Auto-saved and will persist between sessions</small>
            """
            self.results_text.setHtml(results_html)
    
    def _run_calibration(self):
        """Run intensity calibration using multiple standards"""
        # Check if we have loaded standards
        loaded_standards = [name for name, data in self.standards_data.items() 
                          if data.get('loaded', False)]
        
        if not loaded_standards:
            QMessageBox.warning(
                self,
                "No Standards Loaded",
                "Please load at least one standard before running calibration.\n\n"
                "Click 'Load Data' for each standard you want to use."
            )
            return
        
        # Check if FWHM calibration is available
        if self.calibrator.fwhm_calibration is None:
            reply = QMessageBox.question(
                self,
                "No FWHM Calibration",
                "No FWHM calibration is loaded. This may affect calibration quality.\n\n"
                "Do you want to continue anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # Show progress
        self.terminal_output.append(f"\n{'='*50}")
        self.terminal_output.append(f"Starting calibration with {len(loaded_standards)} standard(s):")
        for name in loaded_standards:
            n_elements = len(self.standards_data[name]['concentrations'])
            self.terminal_output.append(f"  • {name}: {n_elements} elements")
        self.terminal_output.append(f"{'='*50}\n")
        
        # For now, show a detailed message about what will be implemented
        QMessageBox.information(
            self,
            "Multi-Standard Calibration",
            f"Calibration will be performed using {len(loaded_standards)} standard(s):\n\n"
            + "\n".join([f"• {name}" for name in loaded_standards]) + "\n\n"
            "The calibration will:\n"
            "1. Fit each standard spectrum with fixed FWHM\n"
            "2. Extract peak intensities for all elements\n"
            "3. Optimize intensity scaling factors\n"
            "4. Calculate detector efficiency curve\n"
            "5. Determine scatter peak parameters\n\n"
            "Full implementation coming soon..."
        )
        
        self.terminal_output.append("Calibration ready to run with loaded standards.")
        self.terminal_output.append("Full implementation in progress...\n")
        
        # TODO: Implement actual calibration
        # This will involve:
        # 1. For each standard:
        #    - Fit spectrum with fixed FWHM from FWHM calibration
        #    - Extract peak intensities
        # 2. Combine all standards data
        # 3. Optimize global parameters (intensity scale, efficiency, etc.)
        # 4. Create CalibrationResult with all parameters
        
        # When calibration completes, auto-save it
        # self._auto_save_calibration()
    
    def _apply_calibration(self):
        """Apply calibration"""
        if self.calibration_result is None:
            QMessageBox.warning(self, "No Calibration", "Please run calibration first.")
            return
        
        # Auto-save when applying
        self._auto_save_calibration()
        
        # Emit signal
        self.calibration_complete.emit(self.calibration_result)
        
        QMessageBox.information(
            self,
            "Calibration Applied",
            "Standards calibration has been applied and saved.\n\n"
            "This calibration will be automatically loaded next time you open the app."
        )
    
    def _save_calibration(self):
        """Save calibration to file"""
        if self.calibration_result is None:
            QMessageBox.warning(self, "No Calibration", "Please run calibration first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Standards Calibration",
            str(Path.home() / "standards_calibration.json"),
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                self.calibration_result.save(file_path)
                QMessageBox.information(
                    self,
                    "Calibration Saved",
                    f"Standards calibration saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save calibration:\n{str(e)}"
                )
    
    def _load_calibration(self):
        """Load calibration from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Standards Calibration",
            str(Path.home()),
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                self.calibration_result = CalibrationResult.load(file_path)
                
                # Enable buttons
                self.apply_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                
                # Display results
                self._display_calibration_results(self.calibration_result)
                
                QMessageBox.information(
                    self,
                    "Calibration Loaded",
                    f"Standards calibration loaded from:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Failed to load calibration:\n{str(e)}"
                )
