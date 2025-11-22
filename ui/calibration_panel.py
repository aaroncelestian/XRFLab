"""
Instrument calibration panel UI
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QPushButton, QLabel, QLineEdit, QTextEdit,
                               QFileDialog, QProgressBar, QMessageBox)
from PySide6.QtCore import Qt, Signal, QThread
from pathlib import Path
from core.calibration import InstrumentCalibrator, CalibrationResult


class CalibrationWorker(QThread):
    """Worker thread for running calibration"""
    finished = Signal(object)  # CalibrationResult
    progress = Signal(str)  # Progress message
    
    def __init__(self, calibrator, energy, counts, concentrations, excitation_energy):
        super().__init__()
        self.calibrator = calibrator
        self.energy = energy
        self.counts = counts
        self.concentrations = concentrations
        self.excitation_energy = excitation_energy
    
    def run(self):
        """Run calibration in background thread"""
        try:
            self.progress.emit("Starting calibration...")
            result = self.calibrator.calibrate(
                self.energy,
                self.counts,
                self.concentrations,
                self.excitation_energy
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
        
        # Instructions
        instructions = QLabel(
            "<b>Instrument Calibration</b><br>"
            "Use a reference standard (e.g., NIST SRM) to calibrate detector parameters.<br>"
            "1. Load reference spectrum<br>"
            "2. Load reference concentrations (CSV)<br>"
            "3. Run calibration<br>"
            "4. Apply or save calibration"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Reference spectrum group
        spectrum_group = self._create_spectrum_group()
        layout.addWidget(spectrum_group)
        
        # Reference concentrations group
        conc_group = self._create_concentrations_group()
        layout.addWidget(conc_group)
        
        # Calibration controls
        controls_group = self._create_controls_group()
        layout.addWidget(controls_group)
        
        # Results display
        results_group = self._create_results_group()
        layout.addWidget(results_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
    
    def _create_spectrum_group(self):
        """Create reference spectrum selection group"""
        group = QGroupBox("Reference Spectrum")
        layout = QVBoxLayout(group)
        
        # File selection
        file_layout = QHBoxLayout()
        self.spectrum_path_edit = QLineEdit()
        self.spectrum_path_edit.setPlaceholderText("Select reference spectrum file...")
        self.spectrum_path_edit.setReadOnly(True)
        file_layout.addWidget(self.spectrum_path_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_spectrum)
        file_layout.addWidget(browse_btn)
        
        layout.addLayout(file_layout)
        
        # Spectrum info
        self.spectrum_info_label = QLabel("No spectrum loaded")
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
        self.results_text.setMaximumHeight(150)
        self.results_text.setPlainText("No calibration results yet")
        layout.addWidget(self.results_text)
        
        return group
    
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
        
        # Run calibration in background thread
        self.worker = CalibrationWorker(
            self.calibrator,
            self.current_spectrum.energy,
            self.current_spectrum.counts,
            self.reference_concentrations,
            excitation_energy
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
            self.spectrum_path_edit.setText("Current spectrum")
            self.spectrum_info_label.setText(
                f"Loaded: {len(spectrum.energy)} channels, "
                f"{spectrum.energy[0]:.2f}-{spectrum.energy[-1]:.2f} keV"
            )
            self._check_ready()
