"""
Instrument calibration panel UI
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QPushButton, QLabel, QLineEdit, QTextEdit,
                               QFileDialog, QProgressBar, QMessageBox, QSplitter)
from PySide6.QtCore import Qt, Signal, QThread
from pathlib import Path
import pyqtgraph as pg
import numpy as np
from core.calibration import InstrumentCalibrator, CalibrationResult


class CalibrationWorker(QThread):
    """Worker thread for running calibration"""
    finished = Signal(object)  # CalibrationResult
    progress = Signal(str)  # Progress message
    
    def __init__(self, calibrator, energy, counts, concentrations, excitation_energy, experimental_params=None, use_measured_intensities=True):
        super().__init__()
        self.calibrator = calibrator
        self.energy = energy
        self.counts = counts
        self.concentrations = concentrations
        self.excitation_energy = excitation_energy
        self.experimental_params = experimental_params
        self.use_measured_intensities = use_measured_intensities
    
    def run(self):
        """Run calibration in background thread"""
        try:
            self.progress.emit("Starting calibration...")
            result = self.calibrator.calibrate(
                self.energy,
                self.counts,
                self.concentrations,
                self.excitation_energy,
                use_measured_intensities=self.use_measured_intensities,
                experimental_params=self.experimental_params
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


class CalibrationPanel(QWidget):
    """Panel for instrument calibration using reference standards"""
    
    calibration_complete = Signal(object)  # CalibrationResult
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.calibrator = InstrumentCalibrator()
        self.current_spectrum = None
        self.reference_concentrations = None
        self.calibration_result = None
        self.worker = None
        
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
        
        # Reference spectrum group
        spectrum_group = self._create_spectrum_group()
        controls_layout.addWidget(spectrum_group)
        
        # Reference concentrations group
        conc_group = self._create_concentrations_group()
        controls_layout.addWidget(conc_group)
        
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
        splitter.setSizes([350, 850])  # Give more space to plot
        
        layout.addWidget(splitter)
    
    def _create_spectrum_group(self):
        """Create reference spectrum selection group"""
        group = QGroupBox("Reference Spectrum")
        layout = QVBoxLayout(group)
        
        # Info label (no file selection - must load from main window)
        info_text = QLabel(
            "<b>Note:</b> Load spectrum from main window (File → Open Spectrum)<br>"
            "This ensures metadata is properly extracted for calibration."
        )
        info_text.setWordWrap(True)
        layout.addWidget(info_text)
        
        # Status label
        self.spectrum_info_label = QLabel("No spectrum loaded")
        self.spectrum_info_label.setStyleSheet("color: gray; margin-top: 10px;")
        layout.addWidget(self.spectrum_info_label)
        
        return group
    
    def _create_concentrations_group(self):
        """Create reference concentrations selection group"""
        group = QGroupBox("Reference Concentrations")
        layout = QVBoxLayout(group)
        
        # File selection
        file_layout = QHBoxLayout()
        self.conc_path_edit = QLineEdit()
        self.conc_path_edit.setPlaceholderText("Select NIST CSV file...")
        self.conc_path_edit.setReadOnly(True)
        file_layout.addWidget(self.conc_path_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_concentrations)
        file_layout.addWidget(browse_btn)
        
        layout.addLayout(file_layout)
        
        # Concentrations info
        self.conc_info_label = QLabel("No concentrations loaded")
        layout.addWidget(self.conc_info_label)
        
        return group
    
    def _create_controls_group(self):
        """Create calibration control buttons"""
        group = QGroupBox("Calibration")
        layout = QHBoxLayout(group)
        
        # Calibration method checkbox
        self.use_fisx_checkbox = QCheckBox("Use fisx FP for calibration (recommended)")
        self.use_fisx_checkbox.setChecked(True)  # Default to fisx (physically correct)
        self.use_fisx_checkbox.setToolTip(
            "Checked: Optimize FWHM using fisx-calculated intensities (recommended)\n"
            "  - Uses known concentrations + physics to calculate expected intensities\n"
            "  - No circular dependency\n\n"
            "Unchecked: Optimize FWHM using measured peak intensities (faster but less accurate)\n"
            "  - Requires peak fitting first (circular dependency)"
        )
        layout.addWidget(self.use_fisx_checkbox)
        
        self.calibrate_btn = QPushButton("Run Calibration")
        self.calibrate_btn.clicked.connect(self._run_calibration)
        self.calibrate_btn.setEnabled(False)
        layout.addWidget(self.calibrate_btn)
        
        self.apply_btn = QPushButton("Apply Calibration")
        self.apply_btn.clicked.connect(self._apply_calibration)
        self.apply_btn.setEnabled(False)
        layout.addWidget(self.apply_btn)
        
        self.save_btn = QPushButton("Save Calibration...")
        self.save_btn.clicked.connect(self._save_calibration)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("Load Calibration...")
        self.load_btn.clicked.connect(self._load_calibration)
        layout.addWidget(self.load_btn)
        
        layout.addStretch()
        
        return group
    
    def _create_results_group(self):
        """Create results display group"""
        group = QGroupBox("Calibration Results")
        layout = QVBoxLayout(group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(100)  # Reduced from 150
        self.results_text.setMinimumHeight(80)
        self.results_text.setPlainText("No calibration results yet")
        layout.addWidget(self.results_text)
        
        return group
    
    def _create_plot_widget(self):
        """Create spectrum comparison plot"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot with two subplots
        self.plot_widget = pg.GraphicsLayoutWidget()
        
        # Top plot: Measured vs Calculated
        self.spectrum_plot = self.plot_widget.addPlot(row=0, col=0)
        self.spectrum_plot.setLabel('left', 'Counts')
        self.spectrum_plot.setLabel('bottom', 'Energy (keV)')
        self.spectrum_plot.setTitle('Calibration Fit')
        self.spectrum_plot.addLegend()
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Plot curves
        self.measured_curve = self.spectrum_plot.plot(
            pen=pg.mkPen('b', width=2), name='Measured'
        )
        self.calculated_curve = self.spectrum_plot.plot(
            pen=pg.mkPen('r', width=2, style=Qt.DashLine), name='Calculated'
        )
        
        # Bottom plot: Residuals
        self.residual_plot = self.plot_widget.addPlot(row=1, col=0)
        self.residual_plot.setLabel('left', 'Residuals')
        self.residual_plot.setLabel('bottom', 'Energy (keV)')
        self.residual_plot.showGrid(x=True, y=True, alpha=0.3)
        self.residual_plot.addLine(y=0, pen=pg.mkPen('k', style=Qt.DashLine))
        
        self.residual_curve = self.residual_plot.plot(
            pen=pg.mkPen('g', width=1)
        )
        
        # Link x-axes
        self.residual_plot.setXLink(self.spectrum_plot)
        
        # Set row heights: spectrum plot gets 2x height of residuals plot
        # This makes residuals 50% less tall (1/3 vs 2/3 of total height)
        self.plot_widget.ci.layout.setRowStretchFactor(0, 2)  # Spectrum plot
        self.plot_widget.ci.layout.setRowStretchFactor(1, 1)  # Residuals plot
        
        layout.addWidget(self.plot_widget)
        
        return widget
    
    def _browse_spectrum(self):
        """Browse for reference spectrum file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Spectrum",
            "",
            "Spectrum Files (*.txt *.dat *.csv *.mca);;All Files (*)"
        )
        
        if file_path:
            self._load_spectrum(file_path)
    
    def _browse_concentrations(self):
        """Browse for reference concentrations CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Concentrations CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self._load_concentrations(file_path)
    
    def _load_spectrum(self, file_path):
        """Load reference spectrum"""
        try:
            from utils.io_handler import IOHandler
            io_handler = IOHandler()
            self.current_spectrum = io_handler.load_spectrum(file_path)
            
            self.spectrum_path_edit.setText(file_path)
            self.spectrum_info_label.setText(
                f"Loaded: {len(self.current_spectrum.energy)} channels, "
                f"{self.current_spectrum.energy[0]:.2f}-{self.current_spectrum.energy[-1]:.2f} keV"
            )
            
            self._check_ready()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load spectrum:\n{str(e)}")
    
    def _load_concentrations(self, file_path):
        """Load reference concentrations from CSV"""
        try:
            self.reference_concentrations = InstrumentCalibrator.load_reference_concentrations(file_path)
            
            self.conc_path_edit.setText(file_path)
            self.conc_info_label.setText(
                f"Loaded: {len(self.reference_concentrations)} elements"
            )
            
            self._check_ready()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load concentrations:\n{str(e)}")
    
    def _check_ready(self):
        """Check if ready to run calibration"""
        ready = (self.current_spectrum is not None and 
                self.reference_concentrations is not None)
        self.calibrate_btn.setEnabled(ready)
    
    def _run_calibration(self):
        """Run instrument calibration"""
        if self.current_spectrum is None or self.reference_concentrations is None:
            return
        
        # Get excitation energy from metadata
        excitation_energy = float(self.current_spectrum.metadata.get('excitation_energy', 50.0))
        
        # Disable buttons during calibration
        self.calibrate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Get experimental parameters from spectrum metadata
        experimental_params = {}
        if hasattr(self.current_spectrum, 'metadata') and self.current_spectrum.metadata:
            experimental_params = {
                'incident_angle': self.current_spectrum.metadata.get('incident_angle', 45.0),
                'takeoff_angle': self.current_spectrum.metadata.get('takeoff_angle', 45.0),
                'tube_current': self.current_spectrum.metadata.get('tube_current', 1.0),
                'tube_element': self.current_spectrum.metadata.get('tube_element', 'Rh'),
            }
        
        # Print experimental parameters for verification
        print("Updated experimental parameters from spectrum metadata:")
        print(f"  Excitation: {excitation_energy} keV")
        print(f"  Current: {experimental_params.get('tube_current', 'N/A')} mA")
        print(f"  Live time: {self.current_spectrum.live_time} s")
        print(f"  Incident angle: {experimental_params.get('incident_angle', 'N/A')}°")
        
        # Run calibration in background thread
        use_measured = not self.use_fisx_checkbox.isChecked()  # Inverted: checked = use fisx
        self.worker = CalibrationWorker(
            self.calibrator,
            self.current_spectrum.energy,
            self.current_spectrum.counts,
            self.reference_concentrations,
            excitation_energy,
            experimental_params,
            use_measured_intensities=use_measured
        )
        self.worker.finished.connect(self._on_calibration_finished)
        self.worker.progress.connect(self._on_calibration_progress)
        self.worker.start()
    
    def _on_calibration_progress(self, message):
        """Handle calibration progress updates"""
        print(message)
    
    def _on_calibration_finished(self, result: CalibrationResult):
        """Handle calibration completion"""
        self.calibration_result = result
        
        # Re-enable buttons
        self.calibrate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update plot with calibration results
        if result.success and self.current_spectrum is not None:
            self._update_calibration_plot(result)
        
        # Display results
        if result.success:
            results_text = (
                f"<b>Calibration Successful!</b><br><br>"
                f"<b>Optimized Parameters:</b><br>"
                f"FWHM₀: {result.fwhm_0:.4f} keV ({result.fwhm_0*1000:.1f} eV)<br>"
                f"ε (epsilon): {result.epsilon:.6f} keV<br>"
                f"Voigt γ/σ ratio: {result.voigt_gamma_ratio:.3f}<br><br>"
                f"<b>Fit Quality:</b><br>"
                f"R²: {result.r_squared:.4f}<br>"
                f"χ²: {result.chi_squared:.2f}<br><br>"
                f"<b>FWHM at key energies:</b><br>"
            )
            
            # Calculate FWHM at common energies
            import numpy as np
            test_energies = [1.74, 3.69, 6.40, 8.05]  # Si, Ca, Fe, Cu
            for e in test_energies:
                fwhm = np.sqrt(result.fwhm_0**2 + 2.35 * result.epsilon * e)
                results_text += f"{e:.2f} keV: {fwhm*1000:.1f} eV<br>"
            
            self.results_text.setHtml(results_text)
            
            self.apply_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            
            QMessageBox.information(
                self,
                "Calibration Complete",
                f"Calibration successful!\nR² = {result.r_squared:.4f}"
            )
        else:
            self.results_text.setHtml(
                f"<b>Calibration Failed</b><br><br>"
                f"Error: {result.message}"
            )
            QMessageBox.warning(
                self,
                "Calibration Failed",
                f"Calibration failed:\n{result.message}"
            )
    
    def _apply_calibration(self):
        """Apply calibration to peak fitter"""
        if self.calibration_result is None or not self.calibration_result.success:
            return
        
        # Update PeakFitter class variables
        from core.peak_fitting import PeakFitter
        PeakFitter.FWHM_0 = self.calibration_result.fwhm_0
        PeakFitter.EPSILON = self.calibration_result.epsilon
        PeakFitter.VOIGT_GAMMA_RATIO = self.calibration_result.voigt_gamma_ratio
        PeakFitter.USE_CALIBRATED_SHAPES = True  # Enable fixed-shape fitting
        
        # Emit signal
        self.calibration_complete.emit(self.calibration_result)
        
        QMessageBox.information(
            self,
            "Applied",
            "Calibration applied! Peak shapes are now fixed.\n"
            "Only intensity and position will be refined during fitting."
        )
    
    def _save_calibration(self):
        """Save calibration to file"""
        if self.calibration_result is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            "instrument_calibration.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                self.calibrator.save_calibration(self.calibration_result, file_path)
                QMessageBox.information(self, "Saved", f"Calibration saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
    
    def _load_calibration(self):
        """Load calibration from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Calibration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                self.calibration_result = self.calibrator.load_calibration(file_path)
                self._on_calibration_finished(self.calibration_result)
                
                # Auto-apply loaded calibration
                if self.calibration_result.success:
                    from core.peak_fitting import PeakFitter
                    PeakFitter.FWHM_0 = self.calibration_result.fwhm_0
                    PeakFitter.EPSILON = self.calibration_result.epsilon
                    PeakFitter.VOIGT_GAMMA_RATIO = self.calibration_result.voigt_gamma_ratio
                    PeakFitter.USE_CALIBRATED_SHAPES = True
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load:\n{str(e)}")
    
    def set_spectrum(self, spectrum):
        """Set spectrum from main window"""
        self.current_spectrum = spectrum
        if spectrum:
            self.spectrum_info_label.setText(
                f"✓ Loaded from main window: {len(spectrum.energy)} channels, "
                f"{spectrum.energy[0]:.2f}-{spectrum.energy[-1]:.2f} keV"
            )
            self.spectrum_info_label.setStyleSheet("color: green; margin-top: 10px;")
            self._check_ready()
            
            # Plot measured spectrum
            self.measured_curve.setData(spectrum.energy, spectrum.counts)
    
    def _update_calibration_plot(self, result: CalibrationResult):
        """Update plot with calibration results"""
        if self.current_spectrum is None or self.reference_concentrations is None:
            return
        
        try:
            print("Updating calibration plot...")
            # Get experimental parameters from spectrum metadata
            excitation_energy = float(self.current_spectrum.metadata.get('excitation_energy', 50.0))
            experimental_params = {}
            if hasattr(self.current_spectrum, 'metadata') and self.current_spectrum.metadata:
                experimental_params = {
                    'incident_angle': self.current_spectrum.metadata.get('incident_angle', 45.0),
                    'takeoff_angle': self.current_spectrum.metadata.get('takeoff_angle', 45.0),
                    'tube_element': self.current_spectrum.metadata.get('tube_element', 'Rh'),
                }
            
            print(f"Calculating element data with fisx...")
            # Calculate spectrum with optimized parameters
            element_data = self.calibrator._prepare_element_data(
                self.reference_concentrations,
                excitation_energy,
                experimental_params
            )
            print(f"Got {len(element_data)} element lines")
            
            # Use FWHM_0, epsilon, and intensity scale
            # Note: intensity_scale is stored in efficiency_params for now
            intensity_scale = result.efficiency_params.get('intensity_scale', 1.0) if result.efficiency_params else 1.0
            params = np.array([
                result.fwhm_0,
                result.epsilon,
                intensity_scale
            ])
            
            calculated_spectrum = self.calibrator._calculate_spectrum(
                self.current_spectrum.energy,
                element_data,
                params
            )
            
            # Scale calculated to match measured peak height for visualization
            if np.max(calculated_spectrum) > 0:
                scale_factor = np.max(self.current_spectrum.counts) / np.max(calculated_spectrum)
                calculated_spectrum_scaled = calculated_spectrum * scale_factor
            else:
                calculated_spectrum_scaled = calculated_spectrum
            
            # Calculate residuals
            residuals = self.current_spectrum.counts - calculated_spectrum_scaled
            
            # Calculate fit quality with fisx-calculated spectrum
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((self.current_spectrum.counts - np.mean(self.current_spectrum.counts))**2)
            r_squared_fisx = 1 - (ss_res / ss_tot)
            chi_squared_fisx = ss_res / len(self.current_spectrum.counts)
            
            print(f"Fit quality with fisx-calculated spectrum:")
            print(f"  R² = {r_squared_fisx:.4f}")
            print(f"  χ² = {chi_squared_fisx:.2f}")
            
            # Update plots
            self.measured_curve.setData(
                self.current_spectrum.energy,
                self.current_spectrum.counts
            )
            self.calculated_curve.setData(
                self.current_spectrum.energy,
                calculated_spectrum_scaled
            )
            self.residual_curve.setData(
                self.current_spectrum.energy,
                residuals
            )
            
            # Auto-range
            self.spectrum_plot.autoRange()
            self.residual_plot.autoRange()
            
            # Update results text with fisx fit quality
            if hasattr(self, 'calibration_result') and self.calibration_result:
                result = self.calibration_result
                results_text = (
                    f"<b>Calibration Successful!</b><br><br>"
                    f"<b>Optimized Parameters:</b><br>"
                    f"FWHM₀: {result.fwhm_0:.4f} keV ({result.fwhm_0*1000:.1f} eV)<br>"
                    f"ε (epsilon): {result.epsilon:.6f} keV<br><br>"
                    f"<b>Fit Quality (with fisx FP):</b><br>"
                    f"R²: {r_squared_fisx:.4f}<br>"
                    f"χ²: {chi_squared_fisx:.2f}<br>"
                )
                self.results_text.setHtml(results_text)
            
        except Exception as e:
            print(f"Error updating calibration plot: {e}")
