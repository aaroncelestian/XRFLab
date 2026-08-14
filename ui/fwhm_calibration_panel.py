"""
FWHM Calibration Panel UI

Calibrate detector resolution (FWHM vs energy) from pure-element standards.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QTextEdit, QFileDialog,
    QProgressBar, QMessageBox, QSplitter, QComboBox,
    QFormLayout, QSizePolicy, QFrame,
)
from PySide6.QtCore import Qt, Signal, QThread, QStandardPaths
from PySide6.QtGui import QFont
from pathlib import Path
import pyqtgraph as pg
import numpy as np
from datetime import datetime

from core.fwhm_calibration import FWHMCalibration, load_fwhm_calibration
from calibrate_peak_shape import PeakShapeCalibrator


class FWHMCalibrationWorker(QThread):
    """Worker thread for running FWHM calibration"""
    finished = Signal(object, object)  # (FWHMCalibration, measurements_list)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, data_dir, model_type='detector', remove_outliers=True, tube_kv=None):
        super().__init__()
        self.data_dir = data_dir
        self.model_type = model_type
        self.remove_outliers = remove_outliers
        self.tube_kv = tube_kv

    def run(self):
        try:
            self.progress.emit("Creating calibrator...")
            calibrator = PeakShapeCalibrator(
                Path(self.data_dir), tube_kv=self.tube_kv
            )

            self.progress.emit("Processing standard files...")
            calibrator.process_all_files()

            if len(calibrator.measurements) < 3:
                self.error.emit(
                    "Not enough measurements for calibration! Need at least 3 peaks."
                )
                return

            self.progress.emit(
                f"Found {len(calibrator.measurements)} peaks. Fitting model..."
            )

            results = calibrator.fit_resolution_model(
                remove_outliers=self.remove_outliers,
                model=self.model_type,
            )

            fwhm_cal = FWHMCalibration(
                model_type=results['model'],
                parameters={
                    k: v for k, v in results.items()
                    if not k.endswith('_err')
                    and k not in ['model', 'r_squared', 'rmse', 'aic', 'bic']
                },
                parameter_errors={
                    k[:-4]: v for k, v in results.items() if k.endswith('_err')
                },
                r_squared=results['r_squared'],
                rmse=results['rmse'],
                aic=results['aic'],
                bic=results['bic'],
                n_peaks=len(calibrator.measurements),
                energy_range=(
                    min(m.energy for m in calibrator.measurements),
                    max(m.energy for m in calibrator.measurements),
                ),
                calibration_date=datetime.now().isoformat(),
            )

            self.progress.emit("Calibration complete!")
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
        self._auto_load_calibration()

    @staticmethod
    def get_default_calibration_path():
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not app_data:
            app_data = str(Path.home() / ".xrflab")

        cal_dir = Path(app_data) / "calibrations"
        cal_dir.mkdir(parents=True, exist_ok=True)
        return cal_dir / "fwhm_calibration.json"

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        controls = QWidget()
        controls.setMinimumWidth(280)
        controls.setMaximumWidth(360)
        col = QVBoxLayout(controls)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)

        col.addWidget(self._create_data_group())
        col.addWidget(self._create_kv_group())
        col.addWidget(self._create_model_group())
        col.addWidget(self._create_controls_group())
        col.addWidget(self._create_results_group(), stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(4)
        col.addWidget(self.progress_bar)

        splitter.addWidget(controls)
        splitter.addWidget(self._create_plot_widget())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

        layout.addWidget(splitter)

    def _create_data_group(self):
        group = QGroupBox("Standards")
        row = QHBoxLayout(group)
        row.setContentsMargins(8, 10, 8, 8)
        row.setSpacing(6)

        self.data_dir_label = QLabel("No folder selected")
        self.data_dir_label.setStyleSheet("color: #888;")
        self.data_dir_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.data_dir_label.setToolTip(
            "Folder of pure-element spectra (Fe.txt, Cu.txt, Ti.txt, …)\n"
            "used to calibrate detector FWHM vs energy."
        )
        row.addWidget(self.data_dir_label, 1)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(88)
        browse_btn.setToolTip(
            "Select directory with pure element standards\n"
            "(Fe, Cu, Ti, Zn, Mg, cubic zirconia, …)"
        )
        browse_btn.clicked.connect(self._browse_data_dir)
        row.addWidget(browse_btn)

        return group

    def _create_kv_group(self):
        group = QGroupBox("Tube voltage (tag only)")
        row = QHBoxLayout(group)
        row.setContentsMargins(8, 10, 8, 8)
        row.setSpacing(6)
        row.addWidget(QLabel("Standards collected at:"))
        self.tube_kv_combo = QComboBox()
        self.tube_kv_combo.addItem("Mixed / unknown", None)
        for kv in (15.0, 30.0, 50.0):
            self.tube_kv_combo.addItem(f"{kv:g} kV", kv)
        self.tube_kv_combo.setToolTip(
            "Tag FWHM peak measurements with the instrument mode used.\n"
            "FWHM itself is still one FWHM(E) curve — pooling 15/30/50 is fine.\n"
            "Use Tube Profiles tab for per-kV Rh scatter ratios."
        )
        row.addWidget(self.tube_kv_combo, 1)
        return group

    def _create_model_group(self):
        group = QGroupBox("Model")
        row = QHBoxLayout(group)
        row.setContentsMargins(8, 10, 8, 8)
        row.setSpacing(6)

        row.addWidget(QLabel("Type:"))

        self.model_combo = QComboBox()
        self.model_combo.addItem("Detector", "detector")
        self.model_combo.addItem("Linear", "linear")
        self.model_combo.addItem("Quadratic", "quadratic")
        self.model_combo.addItem("Exponential", "exponential")
        self.model_combo.addItem("Power", "power")
        self.model_combo.setCurrentIndex(0)
        self.model_combo.setToolTip(
            "Detector (recommended for SDD):\n"
            "  FWHM(E) = √(FWHM₀² + 2.355² · ε · E)\n\n"
            "Linear: a + b·E\n"
            "Quadratic: a + b·E + c·E²\n"
            "Exponential: a · exp(b·E)\n"
            "Power: a · E^b"
        )
        row.addWidget(self.model_combo, 1)

        return group

    def _create_controls_group(self):
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        row1 = QHBoxLayout()
        row1.setSpacing(6)

        self.calibrate_btn = QPushButton("Run")
        self.calibrate_btn.clicked.connect(self._run_calibration)
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setToolTip("Fit FWHM vs energy from the standards folder")
        row1.addWidget(self.calibrate_btn)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_calibration)
        self.apply_btn.setEnabled(False)
        self.apply_btn.setToolTip("Use this calibration for peak fitting / Standards")
        row1.addWidget(self.apply_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(6)

        self.save_btn = QPushButton("Save…")
        self.save_btn.clicked.connect(self._save_calibration)
        self.save_btn.setEnabled(False)
        row2.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load…")
        self.load_btn.clicked.connect(self._load_calibration)
        row2.addWidget(self.load_btn)
        layout.addLayout(row2)

        return group

    def _create_results_group(self):
        group = QGroupBox("Results")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.status_label = QLabel("No calibration loaded")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        metrics = QFrame()
        form = QFormLayout(metrics)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(4)
        form.setLabelAlignment(Qt.AlignRight)

        mono = QFont("Menlo")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(11)

        self.metric_model = QLabel("—")
        self.metric_fwhm0 = QLabel("—")
        self.metric_epsilon = QLabel("—")
        self.metric_r2 = QLabel("—")
        self.metric_rmse = QLabel("—")
        self.metric_peaks = QLabel("—")

        for lab in (
            self.metric_model, self.metric_fwhm0, self.metric_epsilon,
            self.metric_r2, self.metric_rmse, self.metric_peaks,
        ):
            lab.setFont(mono)

        form.addRow("Model", self.metric_model)
        form.addRow("FWHM₀", self.metric_fwhm0)
        form.addRow("ε", self.metric_epsilon)
        form.addRow("R²", self.metric_r2)
        form.addRow("RMSE", self.metric_rmse)
        form.addRow("Peaks", self.metric_peaks)
        layout.addWidget(metrics)

        self.predictions_label = QLabel("")
        self.predictions_label.setWordWrap(True)
        self.predictions_label.setStyleSheet("color: #444; font-size: 11px;")
        layout.addWidget(self.predictions_label)

        self.progress_output = QTextEdit()
        self.progress_output.setReadOnly(True)
        self.progress_output.setMaximumHeight(72)
        self.progress_output.setPlaceholderText("Log")
        self.progress_output.setStyleSheet(
            "QTextEdit { background-color: #f5f5f5; color: #333; "
            "font-family: Menlo, monospace; font-size: 10px; border: 1px solid #ddd; }"
        )
        layout.addWidget(self.progress_output, stretch=1)

        # Alias for any leftover callers
        self.results_text = self.progress_output

        return group

    def _create_plot_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')

        self.fwhm_plot = self.plot_widget.addPlot(row=0, col=0)
        self.fwhm_plot.setLabel('left', 'FWHM', units='eV', color='k')
        self.fwhm_plot.setLabel('bottom', 'Energy', units='keV', color='k')
        self.fwhm_plot.setTitle('Detector resolution', color='k', size='11pt')
        self.fwhm_plot.addLegend(offset=(10, 10))
        self.fwhm_plot.showGrid(x=True, y=True, alpha=0.25)

        self.measurement_scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen('k', width=1),
            brush=pg.mkBrush(0, 0, 139, 150),
            name='Measured',
        )
        self.fwhm_plot.addItem(self.measurement_scatter)

        self.fitted_curve = self.fwhm_plot.plot(
            pen=pg.mkPen('#c0392b', width=2), name='Model'
        )

        self.residual_plot = self.plot_widget.addPlot(row=1, col=0)
        self.residual_plot.setLabel('left', 'Residual', units='eV', color='k')
        self.residual_plot.setLabel('bottom', 'Energy', units='keV', color='k')
        self.residual_plot.setTitle('Residuals', color='k', size='11pt')
        self.residual_plot.showGrid(x=True, y=True, alpha=0.25)
        self.residual_plot.setXLink(self.fwhm_plot)
        self.residual_plot.addLine(
            y=0, pen=pg.mkPen('#c0392b', width=1, style=Qt.DashLine)
        )

        self.residual_scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen('k', width=1),
            brush=pg.mkBrush(0, 0, 139, 150),
        )
        self.residual_plot.addItem(self.residual_scatter)

        self.plot_widget.ci.layout.setRowStretchFactor(0, 3)
        self.plot_widget.ci.layout.setRowStretchFactor(1, 1)

        layout.addWidget(self.plot_widget)
        return widget

    def _browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Standards Directory",
            str(Path.home()),
            QFileDialog.ShowDirsOnly,
        )
        if dir_path:
            self.data_dir = Path(dir_path)
            self.data_dir_label.setText(self.data_dir.name)
            self.data_dir_label.setToolTip(str(self.data_dir))
            self.data_dir_label.setStyleSheet("color: #222;")
            self.calibrate_btn.setEnabled(True)

    def _run_calibration(self):
        if not self.data_dir:
            QMessageBox.warning(
                self, "No Data Directory", "Please select a data directory first."
            )
            return

        model_type = self.model_combo.currentData() or "detector"

        self.progress_output.clear()
        self._clear_metrics()

        self.calibrate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.worker = FWHMCalibrationWorker(
            str(self.data_dir),
            model_type=model_type,
            remove_outliers=True,
            tube_kv=self.tube_kv_combo.currentData(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_calibration_complete)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, message):
        self.progress_output.append(message)
        self.status_label.setText(message)

    def _on_error(self, message):
        self.progress_bar.setVisible(False)
        self.calibrate_btn.setEnabled(True)
        self.progress_output.append(f"ERROR: {message}")
        self.status_label.setText("Calibration failed")
        self.status_label.setStyleSheet("color: #b00020;")
        QMessageBox.critical(self, "Calibration Error", message)

    def _on_calibration_complete(self, fwhm_cal, measurements):
        self.progress_bar.setVisible(False)
        self.calibrate_btn.setEnabled(True)
        self.fwhm_calibration = fwhm_cal
        self.measurements = measurements

        self.apply_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        self._display_results(fwhm_cal)
        self._update_plot(fwhm_cal, measurements)
        self._auto_save_calibration()

        QMessageBox.information(
            self,
            "Calibration Complete",
            f"Saved and ready to apply.\n\n"
            f"R² = {fwhm_cal.r_squared:.4f}   "
            f"RMSE = {fwhm_cal.rmse * 1000:.1f} eV",
        )

    def _clear_metrics(self):
        self.metric_model.setText("—")
        self.metric_fwhm0.setText("—")
        self.metric_epsilon.setText("—")
        self.metric_r2.setText("—")
        self.metric_rmse.setText("—")
        self.metric_peaks.setText("—")
        self.predictions_label.setText("")
        self.status_label.setText("Running…")
        self.status_label.setStyleSheet("color: #666;")

    def _display_results(self, fwhm_cal):
        self.status_label.setText("Calibration ready")
        self.status_label.setStyleSheet("color: #1b7a3d;")

        self.metric_model.setText(fwhm_cal.model_type)
        self.metric_r2.setText(f"{fwhm_cal.r_squared:.4f}")
        self.metric_rmse.setText(f"{fwhm_cal.rmse * 1000:.1f} eV")
        self.metric_peaks.setText(str(fwhm_cal.n_peaks))

        if fwhm_cal.model_type == 'detector':
            fwhm_0_ev = fwhm_cal.parameters['fwhm_0'] * 1000
            epsilon_ev = fwhm_cal.parameters['epsilon'] * 1000
            fwhm_0_err = fwhm_cal.parameter_errors.get('fwhm_0', 0) * 1000
            eps_err = fwhm_cal.parameter_errors.get('epsilon', 0) * 1000
            self.metric_fwhm0.setText(f"{fwhm_0_ev:.1f} ± {fwhm_0_err:.1f} eV")
            self.metric_epsilon.setText(f"{epsilon_ev:.2f} ± {eps_err:.2f} eV/keV")
            tip = "FWHM(E) = √(FWHM₀² + 2.355² · ε · E)"
            self.metric_fwhm0.setToolTip(tip)
            self.metric_epsilon.setToolTip(tip)
        else:
            items = list(fwhm_cal.parameters.items())
            if items:
                k0, v0 = items[0]
                e0 = fwhm_cal.parameter_errors.get(k0, 0)
                self.metric_fwhm0.setText(f"{k0} = {v0:.4g} ± {e0:.3g}")
            if len(items) > 1:
                k1, v1 = items[1]
                e1 = fwhm_cal.parameter_errors.get(k1, 0)
                self.metric_epsilon.setText(f"{k1} = {v1:.4g} ± {e1:.3g}")
            else:
                self.metric_epsilon.setText("—")

        preds = [
            f"{e:g}→{fwhm_cal.predict_fwhm(e) * 1000:.0f}"
            for e in (1.5, 3.0, 6.0, 10.0, 15.0)
        ]
        self.predictions_label.setText("FWHM (eV):  " + "   ".join(preds))

    def _update_plot(self, fwhm_cal, measurements):
        if not measurements:
            return

        energies = np.array([m.energy for m in measurements])
        fwhms_ev = np.array([m.fwhm * 1000 for m in measurements])

        self.measurement_scatter.setData(
            x=energies,
            y=fwhms_ev,
            symbol='o',
            symbolSize=10,
            symbolBrush=pg.mkBrush(0, 0, 139, 150),
            symbolPen=pg.mkPen('k', width=1),
        )

        e_min, e_max = energies.min(), energies.max()
        e_range = max(e_max - e_min, 1e-6)
        e_model = np.linspace(e_min - 0.1 * e_range, e_max + 0.1 * e_range, 200)
        self.fitted_curve.setData(
            x=e_model, y=fwhm_cal.predict_fwhm_array(e_model) * 1000
        )

        residuals_ev = fwhms_ev - fwhm_cal.predict_fwhm_array(energies) * 1000
        self.residual_scatter.setData(
            x=energies,
            y=residuals_ev,
            symbol='o',
            symbolSize=10,
            symbolBrush=pg.mkBrush(0, 0, 139, 150),
            symbolPen=pg.mkPen('k', width=1),
        )

        self.fwhm_plot.autoRange()
        self.residual_plot.autoRange()

    def _plot_fitted_curve_only(self, fwhm_cal):
        self.measurement_scatter.clear()
        self.residual_scatter.clear()

        e_min, e_max = fwhm_cal.energy_range
        e_model = np.linspace(e_min, e_max, 200)
        self.fitted_curve.setData(
            x=e_model, y=fwhm_cal.predict_fwhm_array(e_model) * 1000
        )
        self.fwhm_plot.autoRange()
        self.residual_plot.autoRange()

    def _auto_load_calibration(self):
        cal_path = self.get_default_calibration_path()
        if not cal_path.exists():
            return

        try:
            self.fwhm_calibration = load_fwhm_calibration(str(cal_path))
            self.measurements = None
            self.apply_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self._display_results(self.fwhm_calibration)
            self._plot_fitted_curve_only(self.fwhm_calibration)
            self.progress_output.append(f"Loaded {cal_path.name}")
            self.progress_output.setToolTip(str(cal_path))
            self.calibration_complete.emit(self.fwhm_calibration)
        except Exception:
            self.progress_output.append("No saved calibration")

    def _auto_save_calibration(self):
        if self.fwhm_calibration is None:
            return
        try:
            cal_path = self.get_default_calibration_path()
            self.fwhm_calibration.save(str(cal_path))
            self.progress_output.append(f"Saved {cal_path.name}")
            self.progress_output.setToolTip(str(cal_path))
        except Exception as e:
            self.progress_output.append(f"Auto-save failed: {e}")

    def _apply_calibration(self):
        if self.fwhm_calibration is None:
            QMessageBox.warning(self, "No Calibration", "Please run calibration first.")
            return

        self._auto_save_calibration()
        self.calibration_complete.emit(self.fwhm_calibration)
        QMessageBox.information(
            self,
            "Calibration Applied",
            "FWHM calibration applied for peak fitting and Standards.",
        )

    def _save_calibration(self):
        if self.fwhm_calibration is None:
            QMessageBox.warning(self, "No Calibration", "Please run calibration first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FWHM Calibration",
            str(Path.home() / "fwhm_calibration.json"),
            "JSON Files (*.json)",
        )
        if file_path:
            try:
                self.fwhm_calibration.save(file_path)
                QMessageBox.information(
                    self, "Calibration Saved", f"Saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", f"Failed to save calibration:\n{e}"
                )

    def _load_calibration(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load FWHM Calibration",
            str(Path.home()),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        try:
            self.fwhm_calibration = load_fwhm_calibration(file_path)
            self.measurements = None
            self.apply_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self._display_results(self.fwhm_calibration)
            self._plot_fitted_curve_only(self.fwhm_calibration)
            self._auto_save_calibration()
            self.calibration_complete.emit(self.fwhm_calibration)
            self.progress_output.append(f"Loaded {Path(file_path).name}")
            QMessageBox.information(
                self,
                "Calibration Loaded",
                "Loaded, applied, and saved as the default for next startup.",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error", f"Failed to load calibration:\n{e}"
            )

    def restore_calibration(self, calibration) -> None:
        """Install a calibration from a project file without writing AppData."""
        self.fwhm_calibration = calibration
        if calibration is None:
            return
        self.apply_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self._display_results(calibration)
        try:
            self._plot_fitted_curve_only(calibration)
        except Exception:
            pass
