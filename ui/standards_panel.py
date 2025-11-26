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
                               QComboBox, QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, Signal, QThread
from pathlib import Path
import pyqtgraph as pg
import numpy as np
import json
from typing import Dict, List

from core.calibration import InstrumentCalibrator, CalibrationResult


# Built-in standards library
STANDARDS_LIBRARY = {
    "NIST SRM 610": {
        "description": "Trace Elements in Glass",
        "file": "standards/NIST_SRM_610.csv",
        "matrix": "glass",
        "use_case": "General trace element analysis"
    },
    "NIST SRM 612": {
        "description": "Trace Elements in Glass",
        "file": "standards/NIST_SRM_612.csv",
        "matrix": "glass",
        "use_case": "Low-level trace elements"
    },
    "NIST SRM 1400": {
        "description": "Bone Ash",
        "file": "standards/NIST_SRM_1400.csv",
        "matrix": "bone",
        "use_case": "Biological samples"
    },
    "Custom Standard": {
        "description": "User-defined standard",
        "file": None,
        "matrix": "custom",
        "use_case": "Custom applications"
    }
}


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
        self.current_spectrum = None
        self.reference_concentrations = None
        self.calibration_result = None
        self.worker = None
        self.selected_standards = []  # List of standards to use
        self.standards_data = {}  # Dict of {standard_name: {spectrum, concentrations}}
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create splitter for controls and plot
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # FWHM status group
        fwhm_group = self._create_fwhm_status_group()
        controls_layout.addWidget(fwhm_group)
        
        # Standards library group
        library_group = self._create_standards_library_group()
        controls_layout.addWidget(library_group)
        
        # Selected standards group
        selected_group = self._create_selected_standards_group()
        controls_layout.addWidget(selected_group)
        
        # Calibration controls
        controls_group = self._create_controls_group()
        controls_layout.addWidget(controls_group)
        
        # Results display
        results_group = self._create_results_group()
        controls_layout.addWidget(results_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        controls_layout.addStretch()
        
        # Right side: Spectrum comparison plot
        plot_widget = self._create_plot_widget()
        
        splitter.addWidget(controls_widget)
        splitter.addWidget(plot_widget)
        splitter.setSizes([400, 800])
        
        layout.addWidget(splitter)
    
    def _create_fwhm_status_group(self):
        """Create FWHM calibration status display"""
        group = QGroupBox("FWHM Calibration Status")
        layout = QVBoxLayout(group)
        
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
        """Load custom standard"""
        QMessageBox.information(
            self,
            "Custom Standard",
            "Custom standard loading will be implemented.\n\n"
            "You'll be able to:\n"
            "1. Load spectrum file\n"
            "2. Load or enter concentrations\n"
            "3. Save as a new standard in the library"
        )
    
    def _load_standard_data(self, standard_name, row):
        """Load spectrum and concentration data for a standard"""
        # For now, just show a dialog
        QMessageBox.information(
            self,
            "Load Standard Data",
            f"Loading data for {standard_name}:\n\n"
            f"1. Select spectrum file (.txt, .mca, etc.)\n"
            f"2. Select concentration file (.csv)\n\n"
            f"This will be implemented to load actual data."
        )
        
        # Update status
        self.selected_table.setItem(row, 1, QTableWidgetItem("✓ Loaded"))
        self.standards_data[standard_name]["loaded"] = True
        
        # Enable calibration if we have standards
        self._check_ready_for_calibration()
    
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
    
    def _run_calibration(self):
        """Run intensity calibration"""
        QMessageBox.information(
            self,
            "Run Calibration",
            "Intensity calibration will optimize:\n\n"
            "• Intensity scaling factors\n"
            "• Detector efficiency curve\n"
            "• Scatter peak intensities\n\n"
            "FWHM parameters are held fixed from FWHM Calibration.\n\n"
            "Implementation in progress..."
        )
    
    def _apply_calibration(self):
        """Apply calibration"""
        pass
    
    def _save_calibration(self):
        """Save calibration to file"""
        pass
    
    def _load_calibration(self):
        """Load calibration from file"""
        pass
