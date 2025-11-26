"""
FWHM Calibration Panel UI

This panel provides a user interface for calibrating detector resolution (FWHM vs Energy)
using pure element standards. The calibrated FWHM model is used throughout the application
for accurate peak fitting and quantitative analysis.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QPushButton, QLabel, QTextEdit, QFileDialog, 
                               QProgressBar, QMessageBox, QSplitter, QComboBox)
from PySide6.QtCore import Qt, Signal, QThread, QStandardPaths
from pathlib import Path
import pyqtgraph as pg
import numpy as np
import json
from datetime import datetime

from core.fwhm_calibration import FWHMCalibration, load_fwhm_calibration
from calibrate_peak_shape import PeakShapeCalibrator


class FWHMCalibrationWorker(QThread):
    """Worker thread for running FWHM calibration"""
    finished = Signal(object, object)  # (FWHMCalibration, measurements_list)
    progress = Signal(str)  # Progress message
    error = Signal(str)  # Error message
    
    def __init__(self, data_dir, model_type='detector', remove_outliers=True):
        super().__init__()
        self.data_dir = data_dir
        self.model_type = model_type
        self.remove_outliers = remove_outliers
    
    def run(self):
        """Run FWHM calibration in background thread"""
        try:
            self.progress.emit("Creating calibrator...")
            calibrator = PeakShapeCalibrator(Path(self.data_dir))
            
            self.progress.emit("Processing standard files...")
            calibrator.process_all_files()
            
            if len(calibrator.measurements) < 3:
                self.error.emit("Not enough measurements for calibration! Need at least 3 peaks.")
                return
            
            self.progress.emit(f"Found {len(calibrator.measurements)} peaks. Fitting model...")
            
            # Fit resolution model
            results = calibrator.fit_resolution_model(
                remove_outliers=self.remove_outliers,
                model=self.model_type
            )
            
            # Convert to FWHMCalibration object
            fwhm_cal = FWHMCalibration(
                model_type=results['model'],
                parameters={k: v for k, v in results.items() 
                           if not k.endswith('_err') and k not in ['model', 'r_squared', 'rmse', 'aic', 'bic']},
                parameter_errors={k[:-4]: v for k, v in results.items() if k.endswith('_err')},
                r_squared=results['r_squared'],
                rmse=results['rmse'],
                aic=results['aic'],
                bic=results['bic'],
                n_peaks=len(calibrator.measurements),
                energy_range=(
                    min(m.energy for m in calibrator.measurements),
                    max(m.energy for m in calibrator.measurements)
                ),
                calibration_date=datetime.now().isoformat()
            )
            
            self.progress.emit("Calibration complete!")
            # Return both calibration and measurements for plotting
            self.finished.emit(fwhm_cal, calibrator.measurements)
            
        except Exception as e:
            self.error.emit(f"Calibration failed: {str(e)}")
            import traceback
            traceback.print_exc()


class FWHMCalibrationPanel(QWidget):
    """Panel for FWHM calibration using pure element standards"""
    
    calibration_complete = Signal(object)  # FWHMCalibration
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fwhm_calibration = None
        self.measurements = None
        self.worker = None
        self.data_dir = None
        
        self._init_ui()
        
        # Try to load saved calibration on startup
        self._auto_load_calibration()
    
    @staticmethod
    def get_default_calibration_path():
        """Get the default path for saving/loading FWHM calibration"""
        # Use application data directory
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not app_data:
            # Fallback to home directory
            app_data = str(Path.home() / ".xrflab")
        
        # Create directory if it doesn't exist
        cal_dir = Path(app_data) / "calibrations"
        cal_dir.mkdir(parents=True, exist_ok=True)
        
        return cal_dir / "fwhm_calibration.json"
    
    def _init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create splitter for controls and plot
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # Data directory group
        data_group = self._create_data_group()
        controls_layout.addWidget(data_group)
        
        # Model selection group
        model_group = self._create_model_group()
        controls_layout.addWidget(model_group)
        
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
        
        # Right side: Calibration plot
        plot_widget = self._create_plot_widget()
        
        splitter.addWidget(controls_widget)
        splitter.addWidget(plot_widget)
        splitter.setSizes([350, 850])
        
        layout.addWidget(splitter)
    
    def _create_data_group(self):
        """Create data directory selection group"""
        group = QGroupBox("Reference Spectra")
        layout = QVBoxLayout(group)
        
        # Info label
        info_text = QLabel(
            "<b>Select directory containing pure element standard spectra</b><br>"
            "Use pure element standards (Fe, Cu, Ti, Zn, Mg, cubic zirconia) to calibrate detector resolution."
        )
        info_text.setWordWrap(True)
        layout.addWidget(info_text)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Data Directory:"))
        self.data_dir_label = QLabel("No directory selected")
        self.data_dir_label.setStyleSheet("color: gray;")
        dir_layout.addWidget(self.data_dir_label, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_data_dir)
        dir_layout.addWidget(browse_btn)
        
        layout.addLayout(dir_layout)
        
        # Expected files info
        info = QLabel(
            "<small>Expected files: Fe.txt, Cu.txt, Ti.txt, Zn.txt, Mg.txt, cubic zirconia.txt<br>"
            "Each file should contain pure element XRF spectrum data.</small>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        return group
    
    def _create_model_group(self):
        """Create model selection group"""
        group = QGroupBox("Calibration")
        layout = QVBoxLayout(group)
        
        # Model type selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Type:"))
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "detector (Standard Physics Model)",
            "linear",
            "quadratic",
            "exponential",
            "power"
        ])
        self.model_combo.setCurrentIndex(0)
        self.model_combo.setToolTip(
            "detector: FWHM(E) = √(FWHM₀² + 2.355² × ε × E)\n"
            "linear: FWHM(E) = a + b×E\n"
            "quadratic: FWHM(E) = a + b×E + c×E²\n"
            "exponential: FWHM(E) = a × exp(b×E)\n"
            "power: FWHM(E) = a × E^b"
        )
        model_layout.addWidget(self.model_combo, 1)
        layout.addLayout(model_layout)
        
        # Note about detector model
        note = QLabel(
            "<small><b>Recommended:</b> Use 'detector' model for standard SDD detectors. "
            "This is the physics-based model that describes detector resolution.</small>"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        
        return group
    
    def _create_controls_group(self):
        """Create calibration control buttons"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.calibrate_btn = QPushButton("Run Calibration")
        self.calibrate_btn.clicked.connect(self._run_calibration)
        self.calibrate_btn.setEnabled(False)
        btn_layout.addWidget(self.calibrate_btn)
        
        self.apply_btn = QPushButton("Apply Calibration")
        self.apply_btn.clicked.connect(self._apply_calibration)
        self.apply_btn.setEnabled(False)
        self.apply_btn.setToolTip("Apply this calibration to the Instrument Calibrator")
        btn_layout.addWidget(self.apply_btn)
        
        layout.addLayout(btn_layout)
        
        # Save/Load buttons
        save_load_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Calibration...")
        self.save_btn.clicked.connect(self._save_calibration)
        self.save_btn.setEnabled(False)
        save_load_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("Load Calibration...")
        self.load_btn.clicked.connect(self._load_calibration)
        save_load_layout.addWidget(self.load_btn)
        
        layout.addLayout(save_load_layout)
        
        return group
    
    def _create_results_group(self):
        """Create results display group"""
        group = QGroupBox("Calibration Output")
        layout = QVBoxLayout(group)
        
        # Progress output
        self.progress_output = QTextEdit()
        self.progress_output.setReadOnly(True)
        self.progress_output.setMaximumHeight(80)
        self.progress_output.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "font-family: 'Courier New', monospace; font-size: 10pt; }"
        )
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.progress_output)
        
        # Results summary
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        self.results_text.setMinimumHeight(100)
        self.results_text.setPlainText("No calibration results yet")
        layout.addWidget(QLabel("Calibration Results:"))
        layout.addWidget(self.results_text)
        
        return group
    
    def _create_plot_widget(self):
        """Create FWHM vs Energy plot"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')
        
        # Top plot: FWHM vs Energy
        self.fwhm_plot = self.plot_widget.addPlot(row=0, col=0)
        self.fwhm_plot.setLabel('left', 'FWHM (eV)', color='k')
        self.fwhm_plot.setLabel('bottom', 'Energy (keV)', color='k')
        self.fwhm_plot.setTitle('Detector Resolution Calibration', color='k')
        self.fwhm_plot.addLegend()
        self.fwhm_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Scatter plot for measurements
        self.measurement_scatter = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen('k', width=1), brush=pg.mkBrush(0, 0, 139, 150)
        )
        self.fwhm_plot.addItem(self.measurement_scatter)
        
        # Fitted curve
        self.fitted_curve = self.fwhm_plot.plot(
            pen=pg.mkPen('r', width=2), name='Fitted Model'
        )
        
        # Bottom plot: Residuals
        self.residual_plot = self.plot_widget.addPlot(row=1, col=0)
        self.residual_plot.setLabel('left', 'Residual (eV)', color='k')
        self.residual_plot.setLabel('bottom', 'Energy (keV)', color='k')
        self.residual_plot.setTitle('Fit Residuals', color='k')
        self.residual_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Zero line
        self.residual_plot.addLine(y=0, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        
        # Residual scatter
        self.residual_scatter = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen('k', width=1), brush=pg.mkBrush(0, 0, 139, 150)
        )
        self.residual_plot.addItem(self.residual_scatter)
        
        layout.addWidget(self.plot_widget)
        
        return widget
    
    def _browse_data_dir(self):
        """Browse for data directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            str(Path.home()),
            QFileDialog.ShowDirsOnly
        )
        
        if dir_path:
            self.data_dir = Path(dir_path)
            self.data_dir_label.setText(str(self.data_dir))
            self.data_dir_label.setStyleSheet("color: black;")
            self.calibrate_btn.setEnabled(True)
    
    def _run_calibration(self):
        """Run FWHM calibration"""
        if not self.data_dir:
            QMessageBox.warning(self, "No Data Directory", "Please select a data directory first.")
            return
        
        # Get model type
        model_text = self.model_combo.currentText()
        model_type = model_text.split()[0]  # Extract first word
        
        # Clear previous results
        self.progress_output.clear()
        self.results_text.clear()
        
        # Disable buttons
        self.calibrate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Create and start worker
        self.worker = FWHMCalibrationWorker(
            str(self.data_dir),
            model_type=model_type,
            remove_outliers=True
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_calibration_complete)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def _on_progress(self, message):
        """Handle progress update"""
        self.progress_output.append(message)
    
    def _on_error(self, message):
        """Handle error"""
        self.progress_bar.setVisible(False)
        self.calibrate_btn.setEnabled(True)
        self.progress_output.append(f"ERROR: {message}")
        QMessageBox.critical(self, "Calibration Error", message)
    
    def _on_calibration_complete(self, fwhm_cal, measurements):
        """Handle calibration completion"""
        self.progress_bar.setVisible(False)
        self.calibrate_btn.setEnabled(True)
        self.fwhm_calibration = fwhm_cal
        self.measurements = measurements
        
        # Enable buttons
        self.apply_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # Display results
        self._display_results(fwhm_cal)
        
        # Update plot
        self._update_plot(fwhm_cal, measurements)
        
        # Auto-save the new calibration
        self._auto_save_calibration()
        
        QMessageBox.information(
            self, 
            "Calibration Complete",
            f"FWHM calibration successful and saved!\n\n"
            f"Model: {fwhm_cal.model_type}\n"
            f"R² = {fwhm_cal.r_squared:.4f}\n"
            f"RMSE = {fwhm_cal.rmse*1000:.1f} eV\n\n"
            f"Click 'Apply Calibration' to use it in the Standards tab."
        )
    
    def _display_results(self, fwhm_cal):
        """Display calibration results"""
        if fwhm_cal.model_type == 'detector':
            fwhm_0_ev = fwhm_cal.parameters['fwhm_0'] * 1000
            epsilon_ev = fwhm_cal.parameters['epsilon'] * 1000
            fwhm_0_err_ev = fwhm_cal.parameter_errors.get('fwhm_0', 0) * 1000
            epsilon_err_ev = fwhm_cal.parameter_errors.get('epsilon', 0) * 1000
            
            results_html = f"""
            <b>Calibration Successful!</b><br><br>
            <b>Model:</b> Detector (Physics-based)<br>
            <b>Equation:</b> FWHM(E) = √(FWHM₀² + 2.355² × ε × E)<br><br>
            <b>Parameters:</b><br>
            FWHM₀ = {fwhm_0_ev:.1f} ± {fwhm_0_err_ev:.1f} eV<br>
            ε = {epsilon_ev:.2f} ± {epsilon_err_ev:.2f} eV/keV<br><br>
            <b>Fit Quality:</b><br>
            R² = {fwhm_cal.r_squared:.4f}<br>
            RMSE = {fwhm_cal.rmse*1000:.1f} eV<br>
            AIC = {fwhm_cal.aic:.1f}<br>
            BIC = {fwhm_cal.bic:.1f}<br>
            Peaks used: {fwhm_cal.n_peaks}<br><br>
            <b>FWHM Predictions:</b><br>
            """
            
            # Add predictions at key energies
            for energy in [1.5, 3.0, 6.0, 10.0, 15.0]:
                fwhm_pred = fwhm_cal.predict_fwhm(energy) * 1000
                results_html += f"{energy:.1f} keV → {fwhm_pred:.1f} eV<br>"
        else:
            # Generic display for other models
            results_html = f"""
            <b>Calibration Successful!</b><br><br>
            <b>Model:</b> {fwhm_cal.model_type}<br><br>
            <b>Parameters:</b><br>
            """
            for key, value in fwhm_cal.parameters.items():
                error = fwhm_cal.parameter_errors.get(key, 0)
                results_html += f"{key} = {value:.6f} ± {error:.6f}<br>"
            
            results_html += f"""
            <br><b>Fit Quality:</b><br>
            R² = {fwhm_cal.r_squared:.4f}<br>
            RMSE = {fwhm_cal.rmse*1000:.1f} eV<br>
            Peaks used: {fwhm_cal.n_peaks}<br>
            """
        
        self.results_text.setHtml(results_html)
    
    def _update_plot(self, fwhm_cal, measurements):
        """Update plot with calibration results"""
        if not measurements:
            return
        
        # Extract measurement data
        energies = np.array([m.energy for m in measurements])
        fwhms_ev = np.array([m.fwhm * 1000 for m in measurements])  # Convert to eV
        
        # Plot measured points
        self.measurement_scatter.setData(
            x=energies,
            y=fwhms_ev,
            symbol='o',
            symbolSize=10,
            symbolBrush=pg.mkBrush(0, 0, 139, 150),
            symbolPen=pg.mkPen('k', width=1)
        )
        
        # Generate fitted curve
        e_min, e_max = energies.min(), energies.max()
        e_range = e_max - e_min
        e_model = np.linspace(e_min - 0.1 * e_range, e_max + 0.1 * e_range, 200)
        fwhm_model_ev = fwhm_cal.predict_fwhm_array(e_model) * 1000  # Convert to eV
        
        self.fitted_curve.setData(x=e_model, y=fwhm_model_ev)
        
        # Calculate residuals
        fwhm_predicted_ev = fwhm_cal.predict_fwhm_array(energies) * 1000
        residuals_ev = fwhms_ev - fwhm_predicted_ev
        
        # Plot residuals
        self.residual_scatter.setData(
            x=energies,
            y=residuals_ev,
            symbol='o',
            symbolSize=10,
            symbolBrush=pg.mkBrush(0, 0, 139, 150),
            symbolPen=pg.mkPen('k', width=1)
        )
        
        # Auto-range plots
        self.fwhm_plot.autoRange()
        self.residual_plot.autoRange()
    
    def _plot_fitted_curve_only(self, fwhm_cal):
        """Plot fitted curve without measurement points (for loaded calibrations)"""
        # Clear measurement points
        self.measurement_scatter.clear()
        self.residual_scatter.clear()
        
        # Use energy range from calibration
        e_min, e_max = fwhm_cal.energy_range
        e_model = np.linspace(e_min, e_max, 200)
        fwhm_model_ev = fwhm_cal.predict_fwhm_array(e_model) * 1000  # Convert to eV
        
        self.fitted_curve.setData(x=e_model, y=fwhm_model_ev)
        
        # Auto-range plots
        self.fwhm_plot.autoRange()
        self.residual_plot.autoRange()
    
    def _auto_load_calibration(self):
        """Automatically load saved calibration on startup"""
        cal_path = self.get_default_calibration_path()
        
        if cal_path.exists():
            try:
                self.fwhm_calibration = load_fwhm_calibration(str(cal_path))
                self.measurements = None
                
                # Enable buttons
                self.apply_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                
                # Display results
                self._display_results(self.fwhm_calibration)
                
                # Update plot (show fitted curve only)
                self._plot_fitted_curve_only(self.fwhm_calibration)
                
                # Update progress output
                self.progress_output.append(f"✓ Loaded saved calibration from {cal_path}")
                
                # Auto-apply to Standards panel
                self.calibration_complete.emit(self.fwhm_calibration)
                
            except Exception as e:
                # Silently fail - no calibration available
                self.progress_output.append(f"No saved calibration found (this is normal on first run)")
    
    def _auto_save_calibration(self):
        """Automatically save calibration to default location"""
        if self.fwhm_calibration is None:
            return
        
        try:
            cal_path = self.get_default_calibration_path()
            self.fwhm_calibration.save(str(cal_path))
            self.progress_output.append(f"✓ Auto-saved calibration to {cal_path}")
        except Exception as e:
            self.progress_output.append(f"⚠ Auto-save failed: {str(e)}")
    
    def _apply_calibration(self):
        """Apply calibration to instrument calibrator"""
        if self.fwhm_calibration is None:
            QMessageBox.warning(self, "No Calibration", "Please run calibration first.")
            return
        
        # Auto-save when applying
        self._auto_save_calibration()
        
        # Emit signal to notify main window
        self.calibration_complete.emit(self.fwhm_calibration)
        
        QMessageBox.information(
            self,
            "Calibration Applied",
            "FWHM calibration has been applied and saved.\n\n"
            "The calibrated FWHM model will be used for peak fitting and quantification.\n\n"
            "This calibration will be automatically loaded next time you open the app."
        )
    
    def _save_calibration(self):
        """Save calibration to file"""
        if self.fwhm_calibration is None:
            QMessageBox.warning(self, "No Calibration", "Please run calibration first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FWHM Calibration",
            str(Path.home() / "fwhm_calibration.json"),
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                self.fwhm_calibration.save(file_path)
                QMessageBox.information(
                    self,
                    "Calibration Saved",
                    f"FWHM calibration saved to:\n{file_path}"
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
            "Load FWHM Calibration",
            str(Path.home()),
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                self.fwhm_calibration = load_fwhm_calibration(file_path)
                self.measurements = None  # No measurements when loading from file
                
                # Enable buttons
                self.apply_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                
                # Display results
                self._display_results(self.fwhm_calibration)
                
                # Update plot (show fitted curve only, no measurements)
                self._plot_fitted_curve_only(self.fwhm_calibration)
                
                QMessageBox.information(
                    self,
                    "Calibration Loaded",
                    f"FWHM calibration loaded from:\n{file_path}\n\n"
                    f"Note: Measurement points not available from saved file."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Failed to load calibration:\n{str(e)}"
                )
