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
        # {name: {concentrations, spectra: [{path, name, spectrum}], loaded}}
        self.standards_data = {}
        self._spot_plot_curves = []
        
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
        
        # Standards list (add / reload / remove)
        standards_group = self._create_standards_group()
        layout.addWidget(standards_group, stretch=1)
        
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
    
    def _create_standards_group(self):
        """Create standards list; each standard can hold multiple spot spectra"""
        group = QGroupBox("Standards")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)
        
        info = QLabel(
            "Add a standard once (certified concentrations), then load one or "
            "more spot spectra to check variance. Multi-select files when prompted."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        self.selected_table = QTableWidget()
        self.selected_table.setColumnCount(3)
        self.selected_table.setHorizontalHeaderLabels(["Standard", "Spots", "Elements"])
        header = self.selected_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.selected_table.setMinimumHeight(100)
        self.selected_table.setMaximumHeight(160)
        self.selected_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.selected_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.selected_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.selected_table.itemSelectionChanged.connect(self._on_standard_selection_changed)
        layout.addWidget(self.selected_table)
        
        spots_label = QLabel("Spot spectra for selected standard:")
        layout.addWidget(spots_label)
        
        self.spots_list = QListWidget()
        self.spots_list.setMinimumHeight(80)
        self.spots_list.setToolTip("Individual measurement spots on the selected standard")
        layout.addWidget(self.spots_list, stretch=1)
        
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Standard")
        add_btn.setToolTip(
            "Create a new standard: name, spot spectrum file(s), then concentrations"
        )
        add_btn.clicked.connect(self._add_standard)
        btn_layout.addWidget(add_btn)
        
        add_spectra_btn = QPushButton("Add Spectra…")
        add_spectra_btn.setToolTip(
            "Add more spot spectra to the selected standard (same concentrations)"
        )
        add_spectra_btn.clicked.connect(self._add_spectra_to_selected)
        btn_layout.addWidget(add_spectra_btn)
        
        remove_spot_btn = QPushButton("Remove Spot")
        remove_spot_btn.setToolTip("Remove the selected spot spectrum")
        remove_spot_btn.clicked.connect(self._remove_selected_spot)
        btn_layout.addWidget(remove_spot_btn)
        
        remove_btn = QPushButton("Remove Standard")
        remove_btn.setToolTip("Remove the selected standard and all its spectra")
        remove_btn.clicked.connect(self._remove_selected_standard)
        btn_layout.addWidget(remove_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
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
    
    def _spectrum_file_filter(self):
        return (
            "All Supported (*.txt *.csv *.mca);;"
            "Text Files (*.txt);;CSV Files (*.csv);;MCA Files (*.mca)"
        )
    
    def _pick_spectrum_files(self, title):
        """Multi-select spectrum files; returns list of paths"""
        paths, _ = QFileDialog.getOpenFileNames(
            self, title, "", self._spectrum_file_filter()
        )
        return paths or []
    
    def _load_spectra_from_paths(self, paths):
        """Load spectrum objects from file paths. Returns (entries, errors)."""
        entries = []
        errors = []
        for path in paths:
            try:
                spectrum = self.io_handler.load_spectrum(path)
                entries.append({
                    'path': path,
                    'name': Path(path).name,
                    'spectrum': spectrum,
                })
            except Exception as e:
                errors.append(f"{Path(path).name}: {e}")
        return entries, errors
    
    def _add_standard(self):
        """Add a new standard: name → one or more spot spectra → concentrations"""
        from PySide6.QtWidgets import QInputDialog
        
        standard_name, ok = QInputDialog.getText(
            self,
            "Add Standard",
            "Name for this standard\n(e.g. NIST 2586):",
            text="My Standard"
        )
        
        if not ok or not standard_name.strip():
            return
        
        standard_name = standard_name.strip()
        
        if standard_name in self.standards_data:
            reply = QMessageBox.question(
                self,
                "Standard Exists",
                f"'{standard_name}' is already in the list.\n\n"
                "Add more spot spectra to it?\n"
                "(Concentrations stay the same.)",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._add_spectra_to_standard(standard_name)
            return
        
        paths = self._pick_spectrum_files(
            f"Select Spot Spectrum File(s) for {standard_name}\n"
            "(select multiple files for replicate spots)"
        )
        if not paths:
            return
        
        entries, errors = self._load_spectra_from_paths(paths)
        if errors:
            QMessageBox.warning(
                self,
                "Some Files Failed",
                "Could not load:\n" + "\n".join(errors)
            )
        if not entries:
            QMessageBox.critical(
                self,
                "Error Loading Spectra",
                "No spectrum files could be loaded.\n\n"
                "Select measured XRF spectra (energy/counts), "
                "not the concentration CSV."
            )
            return
        
        concentrations = self._load_or_enter_concentrations(standard_name)
        if not concentrations:
            return
        
        self.standards_data[standard_name] = {
            'concentrations': concentrations,
            'spectra': entries,
            'loaded': True,
        }
        
        self._upsert_standard_row(standard_name)
        self._select_standard_row(standard_name)
        self._check_ready_for_calibration()
        
        QMessageBox.information(
            self,
            "Standard Added",
            f"Added '{standard_name}'.\n\n"
            f"Spot spectra: {len(entries)}\n"
            f"Elements: {len(concentrations)}\n"
            f"Total concentration: {sum(concentrations.values()):.2f} wt%\n\n"
            "Use Add Spectra… to load more spots on this standard."
        )
    
    def _add_spectra_to_selected(self):
        """Add more spot spectra to the currently selected standard"""
        name = self._selected_standard_name()
        if not name:
            QMessageBox.information(
                self,
                "No Selection",
                "Select a standard in the list, then click Add Spectra…"
            )
            return
        self._add_spectra_to_standard(name)
    
    def _add_spectra_to_standard(self, standard_name):
        """Append spot spectra to an existing standard"""
        if standard_name not in self.standards_data:
            return
        
        paths = self._pick_spectrum_files(
            f"Add Spot Spectra to {standard_name}"
        )
        if not paths:
            return
        
        # Skip duplicates by path
        existing = {
            entry['path'] for entry in self.standards_data[standard_name]['spectra']
        }
        new_paths = [p for p in paths if p not in existing]
        if not new_paths:
            QMessageBox.information(
                self,
                "Already Loaded",
                "All selected files are already loaded for this standard."
            )
            return
        
        entries, errors = self._load_spectra_from_paths(new_paths)
        if errors:
            QMessageBox.warning(
                self,
                "Some Files Failed",
                "Could not load:\n" + "\n".join(errors)
            )
        if not entries:
            return
        
        self.standards_data[standard_name]['spectra'].extend(entries)
        self.standards_data[standard_name]['loaded'] = True
        
        self._upsert_standard_row(standard_name)
        self._refresh_spots_list(standard_name)
        self._plot_standard_spots(standard_name)
        self._check_ready_for_calibration()
        
        n = len(self.standards_data[standard_name]['spectra'])
        QMessageBox.information(
            self,
            "Spectra Added",
            f"Added {len(entries)} spot(s) to '{standard_name}'.\n"
            f"Total spots: {n}"
        )
    
    def _upsert_standard_row(self, standard_name):
        """Insert or update a row for this standard in the table"""
        data = self.standards_data.get(standard_name)
        if not data:
            return
        
        row = self._find_standard_row(standard_name)
        if row is None:
            row = self.selected_table.rowCount()
            self.selected_table.insertRow(row)
            self.selected_table.setItem(row, 0, QTableWidgetItem(standard_name))
        
        n_spots = len(data.get('spectra', []))
        spots_item = QTableWidgetItem(str(n_spots))
        spots_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if n_spots > 0:
            spots_item.setForeground(Qt.green)
        self.selected_table.setItem(row, 1, spots_item)
        
        n_elem = len(data.get('concentrations', {}))
        elem_item = QTableWidgetItem(str(n_elem))
        elem_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_table.setItem(row, 2, elem_item)
    
    def _find_standard_row(self, standard_name):
        """Return table row index for a standard name, or None"""
        for row in range(self.selected_table.rowCount()):
            item = self.selected_table.item(row, 0)
            if item and item.text() == standard_name:
                return row
        return None
    
    def _select_standard_row(self, standard_name):
        """Select a standard row and refresh its spot list"""
        row = self._find_standard_row(standard_name)
        if row is not None:
            self.selected_table.selectRow(row)
    
    def _selected_standard_name(self):
        """Return name of currently selected standard, or None"""
        rows = {index.row() for index in self.selected_table.selectedIndexes()}
        if len(rows) != 1:
            return None
        item = self.selected_table.item(next(iter(rows)), 0)
        return item.text() if item else None
    
    def _on_standard_selection_changed(self):
        """Refresh spot list and plot when a standard is selected"""
        name = self._selected_standard_name()
        self._refresh_spots_list(name)
        if name:
            self._plot_standard_spots(name)
    
    def _refresh_spots_list(self, standard_name):
        """Populate the spot spectra list for a standard"""
        self.spots_list.clear()
        if not standard_name or standard_name not in self.standards_data:
            return
        
        for i, entry in enumerate(self.standards_data[standard_name]['spectra'], start=1):
            item = QListWidgetItem(f"Spot {i}: {entry['name']}")
            item.setData(Qt.UserRole, entry['path'])
            item.setToolTip(entry['path'])
            self.spots_list.addItem(item)
    
    def _remove_selected_spot(self):
        """Remove the selected spot spectrum from the current standard"""
        standard_name = self._selected_standard_name()
        if not standard_name:
            QMessageBox.information(
                self, "No Selection", "Select a standard first."
            )
            return
        
        spot_item = self.spots_list.currentItem()
        if not spot_item:
            QMessageBox.information(
                self, "No Spot Selected", "Select a spot spectrum to remove."
            )
            return
        
        path = spot_item.data(Qt.UserRole)
        spectra = self.standards_data[standard_name]['spectra']
        self.standards_data[standard_name]['spectra'] = [
            e for e in spectra if e['path'] != path
        ]
        
        if not self.standards_data[standard_name]['spectra']:
            self.standards_data[standard_name]['loaded'] = False
        
        self._upsert_standard_row(standard_name)
        self._refresh_spots_list(standard_name)
        self._plot_standard_spots(standard_name)
        self._check_ready_for_calibration()
    
    def _remove_selected_standard(self):
        """Remove the currently selected standard and all its spectra"""
        name = self._selected_standard_name()
        if not name:
            QMessageBox.information(
                self,
                "No Selection",
                "Select a standard in the list, then click Remove Standard."
            )
            return
        
        row = self._find_standard_row(name)
        self.standards_data.pop(name, None)
        if row is not None:
            self.selected_table.removeRow(row)
        
        self.spots_list.clear()
        self._clear_spot_plot()
        self._check_ready_for_calibration()
    
    def _clear_spot_plot(self):
        """Clear overlay curves for spot spectra"""
        for curve in self._spot_plot_curves:
            try:
                self.spectrum_plot.removeItem(curve)
            except Exception:
                pass
        self._spot_plot_curves.clear()
        self.measured_curve.setData([], [])
        self.calculated_curve.setData([], [])
        self.background_curve.setData([], [])
        self.residual_curve.setData([], [])
    
    def _plot_standard_spots(self, standard_name):
        """Overlay spot spectra and show mean for variance check"""
        self._clear_spot_plot()
        
        data = self.standards_data.get(standard_name)
        if not data or not data.get('spectra'):
            self.spectrum_plot.setTitle('Intensity Calibration Fit', color='k')
            return
        
        entries = data['spectra']
        n = len(entries)
        self.spectrum_plot.setTitle(
            f'{standard_name}: {n} spot{"s" if n != 1 else ""}',
            color='k'
        )
        
        # Light overlays for each spot
        for i, entry in enumerate(entries):
            spec = entry['spectrum']
            color = pg.intColor(i, hues=max(n, 1), values=1, maxValue=200)
            curve = self.spectrum_plot.plot(
                spec.energy,
                spec.counts,
                pen=pg.mkPen(color, width=1),
                name=f"Spot {i + 1}" if n <= 8 else None,
            )
            self._spot_plot_curves.append(curve)
        
        # Mean spectrum (bold)
        mean_spec = self._mean_spectrum(entries)
        if mean_spec is not None:
            self.measured_curve.setData(mean_spec.energy, mean_spec.counts)
            self.measured_curve.opts['name'] = 'Mean'
    
    def _mean_spectrum(self, entries):
        """Average counts across spot spectra (requires matching energy grids)"""
        if not entries:
            return None
        
        from core.spectrum import Spectrum
        
        ref = entries[0]['spectrum']
        counts_stack = []
        for entry in entries:
            spec = entry['spectrum']
            if len(spec.energy) != len(ref.energy) or not np.allclose(
                spec.energy, ref.energy, rtol=0, atol=1e-6
            ):
                # Different grids — skip averaging; caller still has overlays
                return None
            counts_stack.append(spec.counts)
        
        mean_counts = np.mean(np.vstack(counts_stack), axis=0)
        return Spectrum(
            energy=ref.energy.copy(),
            counts=mean_counts,
            live_time=float(np.mean([e['spectrum'].live_time for e in entries])),
            real_time=float(np.mean([e['spectrum'].real_time for e in entries])),
            metadata={'averaged_from': len(entries)},
        )
    
    def get_standard_mean_spectrum(self, standard_name):
        """Return mean spectrum for a standard, or first spot if grids differ"""
        data = self.standards_data.get(standard_name)
        if not data or not data.get('spectra'):
            return None
        mean = self._mean_spectrum(data['spectra'])
        if mean is not None:
            return mean
        return data['spectra'][0]['spectrum']
    
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
            data.get("loaded", False) and data.get("spectra")
            for data in self.standards_data.values()
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
        loaded_standards = [
            name for name, data in self.standards_data.items()
            if data.get('loaded', False) and data.get('spectra')
        ]
        
        if not loaded_standards:
            QMessageBox.warning(
                self,
                "No Standards Loaded",
                "Please add at least one standard with spot spectra before "
                "running calibration.\n\n"
                "Click Add Standard, then Add Spectra… for more spots."
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
        summary_lines = []
        for name in loaded_standards:
            n_elements = len(self.standards_data[name]['concentrations'])
            n_spots = len(self.standards_data[name].get('spectra', []))
            line = f"  • {name}: {n_spots} spot(s), {n_elements} elements"
            self.terminal_output.append(line)
            summary_lines.append(f"• {name}: {n_spots} spot(s), {n_elements} elements")
        self.terminal_output.append(f"{'='*50}\n")
        
        # Multi-standard calibration is not implemented yet — be honest with the user
        QMessageBox.information(
            self,
            "Multi-Standard Calibration",
            f"{len(loaded_standards)} standard(s) are loaded:\n\n"
            + "\n".join(summary_lines) + "\n\n"
            "Multi-standard optimization is not available yet.\n"
            "Use a single standard with known concentrations for now.\n"
            "This action will be enabled when multi-standard support lands."
        )
        
        self.terminal_output.append(
            "Multi-standard calibration is not implemented yet — no run performed.\n"
        )
        return
        
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
