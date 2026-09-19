"""
Standards Calibration Panel UI

Empirical element calibration from certified reference standards:

* Standards tab   – many standards, each with replicate spot spectra and a
                    certified composition. A "Use" checkbox includes/excludes
                    a whole standard. The set is auto-saved between sessions.
* Elements & Fit  – choose which elements to calibrate and how the standard
                    spectra are fitted (line group, background, tube kV).
                    "Fit Standard Spectra" runs in a worker thread.
* Curves tab      – per-element calibration curves (C = f(I)) with R², RMSE
                    and replicate RSD (instrument precision). Individual
                    standards can be excluded per element and the curves are
                    rebuilt instantly without re-fitting spectra.

The resulting StandardsCalibration converts fitted peaks of unknowns to wt%.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal, QThread, QStandardPaths, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QTextEdit, QFileDialog, QProgressBar, QMessageBox, QSplitter, QCheckBox,
    QDoubleSpinBox, QListWidget, QListWidgetItem, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QTabWidget, QDialog, QStackedWidget,
    QInputDialog, QAbstractItemView,
)

from core.calibration import CalibrationResult
from core.fitting import SpectrumFitter
from core.instrument_state import InstrumentState
from core.peak_fitting import PEAK_SHAPE_UI_CHOICES, PEAK_SHAPE_UI_DEFAULT
from core.reference_composition import find_composition_csv, load_composition_csv
from core.standards_calibration import (
    StandardsCalibration, StandardRecord, ElementCurve,
    MODEL_LINEAR, MODEL_THROUGH_ORIGIN, MODEL_QUADRATIC,
    LINE_AUTO, LINE_ALL, LINE_SERIES, element_list_for_fit, suggest_elements,
    load_any_standards_calibration,
)
from ui.concentration_entry_dialog import ConcentrationEntryDialog
from utils.io_handler import IOHandler


MODEL_LABELS = [
    ("Linear (C = a + b·I)", MODEL_LINEAR),
    ("Through origin (C = b·I)", MODEL_THROUGH_ORIGIN),
    ("Quadratic (C = a + b·I + c·I²)", MODEL_QUADRATIC),
]
LINE_LABELS = [
    ("Principal series (all K lines, else L)", LINE_SERIES),
    ("Principal line (Kα, else Lα)", LINE_AUTO),
    ("All fitted lines of element", LINE_ALL),
]
NORMALISE_LABELS = [
    ("Live time → counts/s", "live_time"),
    ("Real time → counts/s", "real_time"),
    ("None (raw counts)", "none"),
]
BACKGROUND_LABELS = [
    ("SNIP", "snip"),
    ("AsLS", "als"),
    ("Polynomial", "polynomial"),
    ("Linear", "linear"),
    ("Adaptive", "adaptive"),
]


class StandardsFitWorker(QThread):
    """Fit every spot spectrum of the enabled standards in the background."""

    finished = Signal(object)          # StandardsCalibration
    failed = Signal(str)
    progress = Signal(str, int, int)   # message, done, total

    def __init__(self, calibration, spectra, fitter, *, fit_kwargs, elements,
                 line_selection, normalise):
        super().__init__()
        self.calibration = calibration
        self.spectra = spectra
        self.fitter = fitter
        self.fit_kwargs = fit_kwargs
        self.elements = elements
        self.line_selection = line_selection
        self.normalise = normalise
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.calibration.fit_spectra(
                self.spectra,
                self.fitter,
                fit_kwargs=self.fit_kwargs,
                elements=self.elements,
                line_selection=self.line_selection,
                normalise=self.normalise,
                progress=lambda m, i, n: self.progress.emit(m, i, n),
                should_stop=lambda: self._stop,
            )
            self.finished.emit(self.calibration)
        except InterruptedError:
            self.failed.emit("Cancelled")
        except Exception as exc:  # pragma: no cover - surfaced in UI
            import traceback
            traceback.print_exc()
            self.failed.emit(str(exc))


class StandardsPanel(QWidget):
    """Panel for empirical element calibration using reference standards."""

    calibration_complete = Signal(object)  # StandardsCalibration (or legacy CalibrationResult)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.io_handler = IOHandler()
        self.fwhm_calibration = None
        self.tube_profile_library = None
        self.calibration: StandardsCalibration = StandardsCalibration()
        # Legacy CalibrationResult objects loaded from old files are kept here
        self.legacy_result: Optional[CalibrationResult] = None
        self.worker: Optional[StandardsFitWorker] = None
        # {standard_name: [{path, name, spectrum}]}
        self.spectra: Dict[str, List[dict]] = {}
        self._spot_plot_curves = []
        self._curve_plot_items = []
        self._updating = False
        self._selected_element: Optional[str] = None

        self._init_ui()
        self._auto_load()

    # ------------------------------------------------------------------ #
    # Compatibility surface used by MainWindow / project files
    # ------------------------------------------------------------------ #
    @property
    def calibration_result(self):
        """Active calibration object (curve-based, or a legacy result)."""
        if self.legacy_result is not None and not self.calibration.success:
            return self.legacy_result
        if self.calibration.success:
            return self.calibration
        return None

    @staticmethod
    def get_default_calibration_path():
        return _app_data_dir() / "standards_calibration.json"

    @staticmethod
    def get_default_standards_set_path():
        return _app_data_dir() / "standards_set.json"

    def update_fwhm_status(self, fwhm_calibration):
        """Show which FWHM model the standard fits will use."""
        self.fwhm_calibration = fwhm_calibration
        if fwhm_calibration:
            cal_date = getattr(fwhm_calibration, "calibration_date", None) or ""
            date_str = cal_date[:16].replace("T", " ") if cal_date else "Unknown"
            if getattr(fwhm_calibration, "model_type", "") == "detector":
                p = fwhm_calibration.parameters
                detail = (
                    f"FWHM₀ = {p['fwhm_0'] * 1000:.1f} eV, "
                    f"ε = {p['epsilon'] * 1000:.2f} eV/keV"
                )
            else:
                detail = f"Model: {fwhm_calibration.model_type}"
            self.fwhm_status_label.setText(
                f"<b>✓ FWHM calibration active</b> — {detail}, "
                f"R² = {fwhm_calibration.r_squared:.4f} "
                f"<small>({date_str})</small>"
            )
            self.fwhm_status_label.setStyleSheet("color: green;")
        else:
            self.fwhm_status_label.setText(
                "<b>⚠️ No FWHM calibration</b> — peak widths will be free. "
                "Calibrate on the FWHM tab first for more stable intensities."
            )
            self.fwhm_status_label.setStyleSheet("color: #cc6600;")
        self._check_ready()

    def set_tube_profile_library(self, library):
        self.tube_profile_library = library

    def restore_calibration(self, result) -> None:
        """Install a calibration from a project file (no AppData write)."""
        if result is None:
            return
        if isinstance(result, StandardsCalibration):
            self._adopt_calibration(result, load_spectra=True)
        else:
            self.legacy_result = result
            self._log(f"Loaded legacy intensity calibration (R² = {getattr(result, 'r_squared', 0):.3f})")
        self._refresh_all()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _init_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        self.left_tabs = QTabWidget()
        self.left_tabs.setMaximumWidth(700)
        self.left_tabs.addTab(self._create_standards_tab(), "Standards")
        self.left_tabs.addTab(self._create_fit_tab(), "Elements && Fit")
        self.left_tabs.addTab(self._create_curves_tab(), "Curves")
        self.left_tabs.currentChanged.connect(self._on_left_tab_changed)
        splitter.addWidget(self.left_tabs)

        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self._create_spectra_plot())
        self.plot_stack.addWidget(self._create_curve_plot())
        splitter.addWidget(self.plot_stack)
        splitter.setSizes([600, 600])
        layout.addWidget(splitter)

    # ---- Standards tab ------------------------------------------------- #
    def _create_standards_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)

        fwhm_group = QGroupBox("FWHM Calibration Status")
        fl = QVBoxLayout(fwhm_group)
        fl.setContentsMargins(5, 8, 5, 5)
        self.fwhm_status_label = QLabel(
            "<b>⚠️ No FWHM calibration</b> — peak widths will be free. "
            "Calibrate on the FWHM tab first for more stable intensities."
        )
        self.fwhm_status_label.setWordWrap(True)
        self.fwhm_status_label.setStyleSheet("color: #cc6600;")
        fl.addWidget(self.fwhm_status_label)
        layout.addWidget(fwhm_group)

        group = QGroupBox("Standards")
        gl = QVBoxLayout(group)
        gl.setContentsMargins(5, 8, 5, 5)
        gl.setSpacing(3)

        info = QLabel(
            "Add each certified standard with all of its replicate spot spectra. "
            "Untick <b>Use</b> to leave a standard out of every curve. The set "
            "(names, spectrum paths, compositions) is saved automatically."
        )
        info.setWordWrap(True)
        gl.addWidget(info)

        self.standards_table = QTableWidget()
        self.standards_table.setColumnCount(4)
        self.standards_table.setHorizontalHeaderLabels(["Use", "Standard", "Spots", "Elements"])
        header = self.standards_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.standards_table.setMinimumHeight(120)
        self.standards_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.standards_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.standards_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.standards_table.itemSelectionChanged.connect(self._on_standard_selection_changed)
        self.standards_table.itemChanged.connect(self._on_standard_item_changed)
        gl.addWidget(self.standards_table, stretch=2)

        gl.addWidget(QLabel("Spot spectra for selected standard:"))
        self.spots_list = QListWidget()
        self.spots_list.setMinimumHeight(70)
        gl.addWidget(self.spots_list, stretch=1)

        row1 = QHBoxLayout()
        b = QPushButton("Add Standard")
        b.setToolTip("Pick spectrum file(s) → name → confirm certified wt% table")
        b.clicked.connect(self._add_standard)
        row1.addWidget(b)
        b = QPushButton("Add Spectra…")
        b.setToolTip("Add more replicate spot spectra to the selected standard")
        b.clicked.connect(self._add_spectra_to_selected)
        row1.addWidget(b)
        b = QPushButton("Edit Composition…")
        b.setToolTip("Edit certified concentrations of the selected standard")
        b.clicked.connect(self._edit_composition)
        row1.addWidget(b)
        row1.addStretch()
        gl.addLayout(row1)

        row2 = QHBoxLayout()
        b = QPushButton("Remove Spot")
        b.clicked.connect(self._remove_selected_spot)
        row2.addWidget(b)
        b = QPushButton("Remove Standard")
        b.clicked.connect(self._remove_selected_standard)
        row2.addWidget(b)
        row2.addStretch()
        b = QPushButton("Save Set…")
        b.setToolTip("Save the standards set (paths + compositions) to a JSON file")
        b.clicked.connect(self._save_standards_set_as)
        row2.addWidget(b)
        b = QPushButton("Load Set…")
        b.setToolTip("Load a standards set JSON (replaces the current list)")
        b.clicked.connect(self._load_standards_set_from)
        row2.addWidget(b)
        gl.addLayout(row2)

        layout.addWidget(group, stretch=1)
        return widget

    # ---- Elements & Fit tab -------------------------------------------- #
    def _create_fit_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)

        el_group = QGroupBox("Elements to calibrate")
        el = QVBoxLayout(el_group)
        el.setContentsMargins(5, 8, 5, 5)
        el.setSpacing(3)
        hint = QLabel(
            "Ticked elements are fitted in every standard spectrum. Fewer "
            "elements → faster fits. <i>Suggest</i> ticks elements certified in "
            "≥2 used standards and reaching ≥100 ppm."
        )
        hint.setWordWrap(True)
        el.addWidget(hint)
        self.elements_list = QListWidget()
        self.elements_list.setFlow(QListWidget.Flow.LeftToRight)
        self.elements_list.setWrapping(True)
        self.elements_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.elements_list.setGridSize(QSize(72, 22))
        self.elements_list.setMinimumHeight(90)
        self.elements_list.setMaximumHeight(160)
        el.addWidget(self.elements_list)
        erow = QHBoxLayout()
        for label, slot in (
            ("Suggest", self._suggest_elements),
            ("All", lambda: self._set_all_elements(True)),
            ("None", lambda: self._set_all_elements(False)),
        ):
            b = QPushButton(label)
            b.clicked.connect(slot)
            erow.addWidget(b)
        erow.addStretch()
        el.addLayout(erow)
        layout.addWidget(el_group)

        set_group = QGroupBox("Fit settings")
        sl = QVBoxLayout(set_group)
        sl.setContentsMargins(5, 8, 5, 5)
        sl.setSpacing(3)

        r = QHBoxLayout()
        r.addWidget(QLabel("Intensity from:"))
        self.line_combo = QComboBox()
        for label, key in LINE_LABELS:
            self.line_combo.addItem(label, key)
        self.line_combo.setToolTip(
            "Which fitted lines are summed to the element intensity.\n"
            "Principal series (default): Kα+Kβ… — with grouped fitting this is the\n"
            "sub-shell amplitude; with released ratios it is robust to an overlap\n"
            "on one line.\n"
            "Principal line: Kα1+Kα2 only.  All: every line, mixing K and L."
        )
        r.addWidget(self.line_combo, stretch=1)
        r.addWidget(QLabel("Normalize:"))
        self.normalise_combo = QComboBox()
        for label, key in NORMALISE_LABELS:
            self.normalise_combo.addItem(label, key)
        r.addWidget(self.normalise_combo, stretch=1)
        sl.addLayout(r)

        r = QHBoxLayout()
        r.addWidget(QLabel("Background:"))
        self.bg_combo = QComboBox()
        for label, key in BACKGROUND_LABELS:
            self.bg_combo.addItem(label, key)
        r.addWidget(self.bg_combo)
        r.addWidget(QLabel("Peak shape:"))
        self.shape_combo = QComboBox()
        for label, key in PEAK_SHAPE_UI_CHOICES:
            self.shape_combo.addItem(label, key)
        self.shape_combo.setCurrentText(PEAK_SHAPE_UI_DEFAULT)
        self.shape_combo.setToolTip(
            "Use the same background and peak shape here and on the Analysis "
            "Fitting tab — intensities must be extracted the same way for the "
            "curves to transfer to unknowns."
        )
        r.addWidget(self.shape_combo)
        self.release_ratios_check = QCheckBox("Release ratios")
        self.release_ratios_check.setChecked(False)
        self.release_ratios_check.setToolTip(
            "Off (default): grouped fit — one amplitude per element sub-shell with "
            "fixed line ratios, all peaks solved jointly.\n"
            "On: legacy per-line sequential fit. Match the Analysis Fitting tab."
        )
        r.addWidget(self.release_ratios_check)
        r.addStretch()
        sl.addLayout(r)

        r = QHBoxLayout()
        r.addWidget(QLabel("Tube:"))
        self.tube_combo = QComboBox()
        self.tube_combo.addItems(["Rh", "W", "Mo", "Ag", "Cu", "Cr"])
        r.addWidget(self.tube_combo)
        r.addWidget(QLabel("kV:"))
        self.kv_spin = QDoubleSpinBox()
        self.kv_spin.setRange(5.0, 100.0)
        self.kv_spin.setDecimals(1)
        self.kv_spin.setValue(50.0)
        self.kv_spin.setToolTip("Excitation voltage; read from spectrum metadata when available")
        r.addWidget(self.kv_spin)
        r.addStretch()
        sl.addLayout(r)

        r = QHBoxLayout()
        r.addWidget(QLabel("Curve model:"))
        self.model_combo = QComboBox()
        for label, key in MODEL_LABELS:
            self.model_combo.addItem(label, key)
        self.model_combo.currentIndexChanged.connect(self._on_curve_settings_changed)
        r.addWidget(self.model_combo, stretch=1)
        self.weighted_check = QCheckBox("Weight by replicate scatter")
        self.weighted_check.setChecked(True)
        self.weighted_check.setToolTip(
            "Weighted least squares: standards whose replicate spots agree "
            "better count more (1/σ²). Counting statistics set a floor."
        )
        self.weighted_check.toggled.connect(self._on_curve_settings_changed)
        r.addWidget(self.weighted_check)
        sl.addLayout(r)
        layout.addWidget(set_group)

        run_group = QGroupBox("Fit standard spectra")
        rl = QVBoxLayout(run_group)
        rl.setContentsMargins(5, 8, 5, 5)
        rl.setSpacing(3)
        row = QHBoxLayout()
        self.fit_btn = QPushButton("Fit Standard Spectra")
        self.fit_btn.setToolTip(
            "Fit every spot spectrum of the used standards with the ticked "
            "elements, then build the calibration curves"
        )
        self.fit_btn.setEnabled(False)
        self.fit_btn.clicked.connect(self._run_fit)
        row.addWidget(self.fit_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_fit)
        row.addWidget(self.cancel_btn)
        row.addStretch()
        rl.addLayout(row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        rl.addWidget(self.progress_bar)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "font-family: 'Courier New', monospace; font-size: 10pt; }"
        )
        self.log_output.setPlainText("Ready.")
        rl.addWidget(self.log_output, stretch=1)
        layout.addWidget(run_group, stretch=1)
        return widget

    # ---- Curves tab ------------------------------------------------------ #
    def _create_curves_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)

        self.curves_summary = QLabel("No calibration curves yet — fit standard spectra first.")
        self.curves_summary.setWordWrap(True)
        layout.addWidget(self.curves_summary)

        el_group = QGroupBox("Element curves (untick Use to drop an element from quantification)")
        el = QVBoxLayout(el_group)
        el.setContentsMargins(5, 8, 5, 5)
        self.curves_table = QTableWidget()
        cols = ["Use", "El", "Line", "Std", "Spec", "Slope wt%/cps", "R²", "RMSE wt%", "RSD %", "Note"]
        self.curves_table.setColumnCount(len(cols))
        self.curves_table.setHorizontalHeaderLabels(cols)
        ch = self.curves_table.horizontalHeader()
        for i in range(len(cols) - 1):
            ch.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        ch.setSectionResizeMode(len(cols) - 1, QHeaderView.ResizeMode.Stretch)
        self.curves_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.curves_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.curves_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.curves_table.itemSelectionChanged.connect(self._on_curve_selection_changed)
        self.curves_table.itemChanged.connect(self._on_curve_item_changed)
        el.addWidget(self.curves_table)
        layout.addWidget(el_group, stretch=3)

        pt_group = QGroupBox("Standards in selected curve (untick Use to exclude an outlier)")
        pl = QVBoxLayout(pt_group)
        pl.setContentsMargins(5, 8, 5, 5)
        self.points_table = QTableWidget()
        pcols = ["Use", "Standard", "Cert. wt%", "Mean cps", "SD cps", "RSD %", "n", "Pred. wt%", "Resid. wt%"]
        self.points_table.setColumnCount(len(pcols))
        self.points_table.setHorizontalHeaderLabels(pcols)
        ph = self.points_table.horizontalHeader()
        ph.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for i in [0] + list(range(2, len(pcols))):
            ph.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.points_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.points_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.points_table.itemChanged.connect(self._on_point_item_changed)
        pl.addWidget(self.points_table)
        layout.addWidget(pt_group, stretch=2)

        row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Calibration")
        self.apply_btn.setToolTip("Use these curves to report wt% on the Analysis tab; auto-saved")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_calibration)
        row.addWidget(self.apply_btn)
        self.save_btn = QPushButton("Save Calibration…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_calibration)
        row.addWidget(self.save_btn)
        b = QPushButton("Load Calibration…")
        b.clicked.connect(self._load_calibration)
        row.addWidget(b)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.setToolTip("Curves, per-standard points and spot intensities")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_csv)
        row.addWidget(self.export_btn)
        row.addStretch()
        layout.addLayout(row)
        return widget

    # ---- plots ------------------------------------------------------------ #
    def _create_spectra_plot(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spectra_plot_widget = pg.GraphicsLayoutWidget()
        self.spectra_plot_widget.setBackground("w")
        self.spectrum_plot = self.spectra_plot_widget.addPlot(row=0, col=0)
        self.spectrum_plot.setLabel("left", "Counts", color="k")
        self.spectrum_plot.setLabel("bottom", "Energy (keV)", color="k")
        self.spectrum_plot.setTitle("Spot spectra", color="k")
        self.spectrum_plot.addLegend()
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.measured_curve = self.spectrum_plot.plot(pen=pg.mkPen("#00008B", width=2), name="Mean")
        layout.addWidget(self.spectra_plot_widget)
        return widget

    def _create_curve_plot(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.curve_plot_widget = pg.GraphicsLayoutWidget()
        self.curve_plot_widget.setBackground("w")
        self.curve_plot = self.curve_plot_widget.addPlot(row=0, col=0)
        self.curve_plot.setLabel("left", "Intensity (counts/s)", color="k")
        self.curve_plot.setLabel("bottom", "Certified concentration (wt%)", color="k")
        self.curve_plot.setTitle("Calibration curve", color="k")
        self.curve_plot.addLegend()
        self.curve_plot.showGrid(x=True, y=True, alpha=0.3)
        self.residual_plot = self.curve_plot_widget.addPlot(row=1, col=0)
        self.residual_plot.setLabel("left", "Predicted − certified (wt%)", color="k")
        self.residual_plot.setLabel("bottom", "Certified concentration (wt%)", color="k")
        self.residual_plot.showGrid(x=True, y=True, alpha=0.3)
        self.residual_plot.addLine(y=0, pen=pg.mkPen("r", width=1, style=Qt.DashLine))
        self.residual_plot.setXLink(self.curve_plot)
        self.curve_plot_widget.ci.layout.setRowStretchFactor(0, 3)
        self.curve_plot_widget.ci.layout.setRowStretchFactor(1, 1)
        layout.addWidget(self.curve_plot_widget)
        return widget

    # ------------------------------------------------------------------ #
    # Standards management
    # ------------------------------------------------------------------ #
    def _spectrum_file_filter(self):
        return (
            "All Supported (*.txt *.csv *.mca);;"
            "Text Files (*.txt);;CSV Files (*.csv);;MCA Files (*.mca)"
        )

    def _pick_spectrum_files(self, title):
        paths, _ = QFileDialog.getOpenFileNames(self, title, "", self._spectrum_file_filter())
        return paths or []

    def _load_spectra_from_paths(self, paths):
        entries, errors = [], []
        for path in paths:
            try:
                spectrum = self.io_handler.load_spectrum(path)
                entries.append({"path": path, "name": Path(path).name, "spectrum": spectrum})
                kv = (spectrum.metadata or {}).get("excitation_energy")
                if kv and not self._kv_from_data:
                    self.kv_spin.setValue(float(kv))
                    self._kv_from_data = True
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        return entries, errors

    _kv_from_data = False

    def _add_standard(self):
        paths = self._pick_spectrum_files(
            "Select spectrum file(s) for this standard (multi-select replicate spots)"
        )
        if not paths:
            return
        suggested = Path(paths[0]).parent.name if len(paths) > 1 else Path(paths[0]).stem
        name, ok = QInputDialog.getText(self, "Standard name", "Name for this standard:", text=suggested)
        if not ok or not name.strip():
            return
        name = name.strip()

        if name in self.calibration.standards:
            reply = QMessageBox.question(
                self, "Standard Exists",
                f"'{name}' is already in the list.\n\nAdd these files as more spot spectra?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._add_spectra_to_standard(name, paths=paths)
            return

        entries, errors = self._load_spectra_from_paths(paths)
        if errors:
            QMessageBox.warning(self, "Some Files Failed", "Could not load:\n" + "\n".join(errors))
        if not entries:
            QMessageBox.critical(
                self, "Error Loading Spectra",
                "No spectrum files could be loaded.\n\nSelect measured XRF spectra "
                "(energy/counts), not the concentration CSV.",
            )
            return

        concentrations = self._confirm_composition(name, paths)
        if not concentrations:
            return

        record = StandardRecord(
            name=name, concentrations=concentrations,
            spectrum_paths=[e["path"] for e in entries], enabled=True,
        )
        self.calibration.add_standard(record)
        self.spectra[name] = entries
        self._after_standards_changed(select=name)

    def _add_spectra_to_selected(self):
        name = self._selected_standard_name()
        if not name:
            QMessageBox.information(self, "No Selection", "Select a standard first.")
            return
        self._add_spectra_to_standard(name)

    def _add_spectra_to_standard(self, name, paths=None):
        record = self.calibration.standards.get(name)
        if record is None:
            return
        if not paths:
            paths = self._pick_spectrum_files(f"Add spot spectra to {name}")
        if not paths:
            return
        new_paths = [p for p in paths if p not in record.spectrum_paths]
        if not new_paths:
            QMessageBox.information(self, "Already Loaded", "All selected files are already in this standard.")
            return
        entries, errors = self._load_spectra_from_paths(new_paths)
        if errors:
            QMessageBox.warning(self, "Some Files Failed", "Could not load:\n" + "\n".join(errors))
        if not entries:
            return
        record.spectrum_paths.extend(e["path"] for e in entries)
        self.spectra.setdefault(name, []).extend(entries)
        self._after_standards_changed(select=name)

    def _edit_composition(self):
        name = self._selected_standard_name()
        if not name:
            QMessageBox.information(self, "No Selection", "Select a standard first.")
            return
        record = self.calibration.standards[name]
        dialog = ConcentrationEntryDialog(
            name, self, concentrations=dict(record.concentrations),
            source_label="Edit certified values (wt%).",
        )
        if dialog.exec() == QDialog.Accepted:
            record.concentrations = dialog.get_concentrations()
            self._after_standards_changed(select=name, rebuild=True)

    def _remove_selected_spot(self):
        name = self._selected_standard_name()
        if not name:
            QMessageBox.information(self, "No Selection", "Select a standard first.")
            return
        item = self.spots_list.currentItem()
        if not item:
            QMessageBox.information(self, "No Spot Selected", "Select a spot spectrum to remove.")
            return
        path = item.data(Qt.UserRole)
        record = self.calibration.standards[name]
        record.spectrum_paths = [p for p in record.spectrum_paths if p != path]
        self.spectra[name] = [e for e in self.spectra.get(name, []) if e["path"] != path]
        self.calibration.intensities.get(name, {}).pop(path, None)
        self._after_standards_changed(select=name, rebuild=True)

    def _remove_selected_standard(self):
        name = self._selected_standard_name()
        if not name:
            QMessageBox.information(self, "No Selection", "Select a standard first.")
            return
        self.calibration.remove_standard(name)
        self.spectra.pop(name, None)
        self.spots_list.clear()
        self._clear_spot_plot()
        self._after_standards_changed(rebuild=True)

    def _confirm_composition(self, name, spectrum_paths):
        found = find_composition_csv(spectrum_paths, standard_name=name)
        initial, source = {}, ""
        if found:
            try:
                initial = load_composition_csv(found)
                source = f"Pre-filled from {found.name} ({len(initial)} elements). Check before accepting."
            except Exception:
                source = f"Could not parse {found.name}; enter values or load another CSV."
        dialog = ConcentrationEntryDialog(name, self, concentrations=initial or None, source_label=source)
        if dialog.exec() == QDialog.Accepted:
            return dialog.get_concentrations()
        return None

    def _after_standards_changed(self, select=None, rebuild=False):
        self._refresh_standards_table()
        if select:
            self._select_standard_row(select)
        self._refresh_elements_list()
        self._check_ready()
        self._auto_save_standards_set()
        if rebuild and self.calibration.has_intensities():
            self._rebuild_curves()

    # ---- standards table --------------------------------------------------- #
    def _refresh_standards_table(self):
        self._updating = True
        try:
            names = list(self.calibration.standards.keys())
            self.standards_table.setRowCount(len(names))
            for row, name in enumerate(names):
                rec = self.calibration.standards[name]
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                use.setCheckState(Qt.Checked if rec.enabled else Qt.Unchecked)
                self.standards_table.setItem(row, 0, use)
                self.standards_table.setItem(row, 1, QTableWidgetItem(name))
                n_loaded = len(self.spectra.get(name, []))
                n_paths = len(rec.spectrum_paths)
                spots = QTableWidgetItem(str(n_loaded) if n_loaded == n_paths else f"{n_loaded}/{n_paths}")
                spots.setTextAlignment(Qt.AlignCenter)
                if n_loaded < n_paths:
                    spots.setForeground(Qt.red)
                    spots.setToolTip("Some spectrum files could not be found")
                self.standards_table.setItem(row, 2, spots)
                el = QTableWidgetItem(str(len(rec.concentrations)))
                el.setTextAlignment(Qt.AlignCenter)
                self.standards_table.setItem(row, 3, el)
        finally:
            self._updating = False

    def _find_standard_row(self, name):
        for row in range(self.standards_table.rowCount()):
            item = self.standards_table.item(row, 1)
            if item and item.text() == name:
                return row
        return None

    def _select_standard_row(self, name):
        row = self._find_standard_row(name)
        if row is not None:
            self.standards_table.selectRow(row)

    def _selected_standard_name(self):
        rows = {i.row() for i in self.standards_table.selectedIndexes()}
        if len(rows) != 1:
            return None
        item = self.standards_table.item(next(iter(rows)), 1)
        return item.text() if item else None

    def _on_standard_selection_changed(self):
        name = self._selected_standard_name()
        self._refresh_spots_list(name)
        if name:
            self._plot_standard_spots(name)

    def _on_standard_item_changed(self, item):
        if self._updating or item.column() != 0:
            return
        name_item = self.standards_table.item(item.row(), 1)
        if not name_item:
            return
        rec = self.calibration.standards.get(name_item.text())
        if rec is None:
            return
        rec.enabled = item.checkState() == Qt.Checked
        self._refresh_elements_list()
        self._check_ready()
        self._auto_save_standards_set()
        if self.calibration.has_intensities():
            self._rebuild_curves()

    def _refresh_spots_list(self, name):
        self.spots_list.clear()
        if not name or name not in self.calibration.standards:
            return
        loaded = {e["path"] for e in self.spectra.get(name, [])}
        for i, path in enumerate(self.calibration.standards[name].spectrum_paths, start=1):
            item = QListWidgetItem(f"Spot {i}: {Path(path).name}" + ("" if path in loaded else "  (missing)"))
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            if path not in loaded:
                item.setForeground(Qt.red)
            self.spots_list.addItem(item)

    # ---- spot spectra plot ------------------------------------------------- #
    def _clear_spot_plot(self):
        for curve in self._spot_plot_curves:
            try:
                self.spectrum_plot.removeItem(curve)
            except Exception:
                pass
        self._spot_plot_curves.clear()
        self.measured_curve.setData([], [])

    def _plot_standard_spots(self, name):
        self._clear_spot_plot()
        entries = self.spectra.get(name, [])
        if not entries:
            self.spectrum_plot.setTitle("Spot spectra", color="k")
            return
        n = len(entries)
        self.spectrum_plot.setTitle(f"{name}: {n} spot{'s' if n != 1 else ''}", color="k")
        for i, entry in enumerate(entries):
            spec = entry["spectrum"]
            color = pg.intColor(i, hues=max(n, 1), values=1, maxValue=200)
            curve = self.spectrum_plot.plot(
                spec.energy, spec.counts, pen=pg.mkPen(color, width=1),
                name=f"Spot {i + 1}" if n <= 8 else None,
            )
            self._spot_plot_curves.append(curve)
        mean = _mean_spectrum(entries)
        if mean is not None:
            self.measured_curve.setData(mean.energy, mean.counts)
        if self.left_tabs.currentIndex() == 0:
            self.plot_stack.setCurrentIndex(0)

    # ------------------------------------------------------------------ #
    # Elements list
    # ------------------------------------------------------------------ #
    def _refresh_elements_list(self):
        previous = {
            self.elements_list.item(i).text(): self.elements_list.item(i).checkState() == Qt.Checked
            for i in range(self.elements_list.count())
        }
        kv = float(self.kv_spin.value())
        candidates = [e["symbol"] for e in element_list_for_fit(self.calibration.standards.values(), excitation_kv=kv)]
        suggested = set(suggest_elements(self.calibration.standards.values(), excitation_kv=kv))
        self.elements_list.clear()
        for sym in candidates:
            item = QListWidgetItem(sym)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checked = previous.get(sym, sym in suggested)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.elements_list.addItem(item)

    def _suggest_elements(self):
        kv = float(self.kv_spin.value())
        suggested = set(suggest_elements(self.calibration.standards.values(), excitation_kv=kv))
        for i in range(self.elements_list.count()):
            item = self.elements_list.item(i)
            item.setCheckState(Qt.Checked if item.text() in suggested else Qt.Unchecked)

    def _set_all_elements(self, state: bool):
        for i in range(self.elements_list.count()):
            self.elements_list.item(i).setCheckState(Qt.Checked if state else Qt.Unchecked)

    def _checked_elements(self) -> List[str]:
        return [
            self.elements_list.item(i).text()
            for i in range(self.elements_list.count())
            if self.elements_list.item(i).checkState() == Qt.Checked
        ]

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def _check_ready(self):
        has = any(
            rec.enabled and self.spectra.get(name)
            for name, rec in self.calibration.standards.items()
        )
        self.fit_btn.setEnabled(bool(has) and self.worker is None)

    def _build_fitter(self) -> SpectrumFitter:
        fitter = SpectrumFitter()
        state = InstrumentState()
        if self.fwhm_calibration is not None:
            state.apply_fwhm_calibration(self.fwhm_calibration)
        state.tube_profile_library = self.tube_profile_library
        fitter.apply_instrument_state(state)
        return fitter

    def _run_fit(self):
        elements = self._checked_elements()
        if not elements:
            QMessageBox.warning(self, "No Elements", "Tick at least one element on the Elements & Fit tab.")
            return
        used = [n for n, r in self.calibration.standards.items() if r.enabled and self.spectra.get(n)]
        if not used:
            QMessageBox.warning(self, "No Standards", "Add at least one standard with spot spectra and tick Use.")
            return
        if self.fwhm_calibration is None:
            reply = QMessageBox.question(
                self, "No FWHM Calibration",
                "No FWHM calibration is loaded; peak widths will be fitted freely, "
                "which makes intensities noisier.\n\nContinue anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        n_spec = sum(len(self.spectra[n]) for n in used)
        self._log("=" * 50)
        self._log(f"Fitting {n_spec} spectra from {len(used)} standard(s), {len(elements)} elements:")
        for n in used:
            self._log(f"  • {n}: {len(self.spectra[n])} spot(s), {len(self.calibration.standards[n].concentrations)} certified elements")

        fit_kwargs = {
            "background_method": self.bg_combo.currentData(),
            "peak_shape": self.shape_combo.currentData(),
            "grouped_lines": not self.release_ratios_check.isChecked(),
            "tube_element": self.tube_combo.currentText(),
            "excitation_kv": float(self.kv_spin.value()),
        }
        cal_copy = copy.deepcopy(self.calibration)
        cal_copy.fwhm_calibration = (
            self.fwhm_calibration.to_dict() if hasattr(self.fwhm_calibration, "to_dict") and self.fwhm_calibration else None
        )
        spectra = {n: [(e["path"], e["spectrum"]) for e in self.spectra[n]] for n in used}

        self.worker = StandardsFitWorker(
            cal_copy, spectra, self._build_fitter(),
            fit_kwargs=fit_kwargs, elements=elements,
            line_selection=self.line_combo.currentData(),
            normalise=self.normalise_combo.currentData(),
        )
        self.worker.progress.connect(self._on_fit_progress)
        self.worker.finished.connect(self._on_fit_finished)
        self.worker.failed.connect(self._on_fit_failed)
        self.progress_bar.setRange(0, n_spec)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.fit_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.worker.start()

    def _cancel_fit(self):
        if self.worker is not None:
            self.worker.stop()
            self._log("Cancelling after the current spectrum…")

    def _on_fit_progress(self, message, done, total):
        self.progress_bar.setValue(done)
        self.log_output.append(f"[{done}/{total}] {message}")

    def _finish_worker(self):
        self.worker = None
        self.progress_bar.setVisible(False)
        self.cancel_btn.setEnabled(False)
        self._check_ready()

    def _on_fit_failed(self, message):
        self._finish_worker()
        self._log(f"✗ {message}")
        if message != "Cancelled":
            QMessageBox.critical(self, "Calibration Failed", message)

    def _on_fit_finished(self, calibration: StandardsCalibration):
        self._finish_worker()
        self.legacy_result = None
        self._adopt_calibration(calibration, load_spectra=False)
        self._log("✓ Spectra fitted; building curves…")
        self._rebuild_curves()
        self._auto_save_calibration()
        self.left_tabs.setCurrentIndex(2)

    def _adopt_calibration(self, calibration: StandardsCalibration, *, load_spectra: bool):
        """Make `calibration` the working object; optionally (re)load spectra from disk."""
        self.calibration = calibration
        if load_spectra:
            for name, rec in calibration.standards.items():
                have = {e["path"] for e in self.spectra.get(name, [])}
                missing = [p for p in rec.spectrum_paths if p not in have and Path(p).exists()]
                if missing:
                    entries, errors = self._load_spectra_from_paths(missing)
                    self.spectra.setdefault(name, []).extend(entries)
                    for err in errors:
                        self._log(f"  ⚠ {name}: {err}")
                for p in rec.spectrum_paths:
                    if not Path(p).exists():
                        self._log(f"  ⚠ {name}: missing file {p}")
        # Restore UI settings from the calibration
        self._updating = True
        try:
            _set_combo_data(self.model_combo, calibration.settings.get("model", MODEL_LINEAR))
            self.weighted_check.setChecked(bool(calibration.settings.get("weighted", True)))
            _set_combo_data(self.line_combo, calibration.settings.get("line_selection", LINE_SERIES))
            _set_combo_data(self.normalise_combo, calibration.settings.get("normalise", "live_time"))
            fs = calibration.fit_settings or {}
            _set_combo_data(self.bg_combo, fs.get("background_method", "snip"))
            _set_combo_data(self.shape_combo, fs.get("peak_shape", "tail_gaussian"))
            self.release_ratios_check.setChecked(not bool(fs.get("grouped_lines", True)))
            if fs.get("tube_element"):
                idx = self.tube_combo.findText(fs["tube_element"])
                if idx >= 0:
                    self.tube_combo.setCurrentIndex(idx)
            if fs.get("excitation_kv"):
                self.kv_spin.setValue(float(fs["excitation_kv"]))
                self._kv_from_data = True
        finally:
            self._updating = False
        self._refresh_standards_table()
        self._refresh_elements_list()
        fitted_elements = set(calibration.fit_settings.get("elements") or [])
        if fitted_elements:
            for i in range(self.elements_list.count()):
                item = self.elements_list.item(i)
                item.setCheckState(Qt.Checked if item.text() in fitted_elements else Qt.Unchecked)
        self._check_ready()

    # ------------------------------------------------------------------ #
    # Curves
    # ------------------------------------------------------------------ #
    def _on_curve_settings_changed(self, *_):
        if self._updating:
            return
        if self.calibration.has_intensities():
            self._rebuild_curves()

    def _rebuild_curves(self):
        try:
            self.calibration.build_curves(
                model=self.model_combo.currentData(),
                weighted=self.weighted_check.isChecked(),
            )
        except Exception as exc:
            self._log(f"✗ Curve build failed: {exc}")
            return
        self._refresh_all()

    def _refresh_all(self):
        self._refresh_curves_table()
        self._refresh_points_table()
        self._plot_selected_curve()
        n_ok = len(self.calibration.fitted_curves())
        has_any = bool(self.calibration.curves)
        self.apply_btn.setEnabled(n_ok > 0 or self.legacy_result is not None)
        self.save_btn.setEnabled(has_any or self.legacy_result is not None)
        self.export_btn.setEnabled(has_any)
        if has_any:
            date = (self.calibration.calibration_date or "")[:16].replace("T", " ")
            n_std = len([s for s in self.calibration.standards.values() if s.enabled])
            self.curves_summary.setText(
                f"<b>{n_ok} usable curve(s)</b> from {n_std} used standard(s) · "
                f"model: {self.model_combo.currentText()} · "
                f"{'weighted' if self.weighted_check.isChecked() else 'unweighted'} · "
                f"fitted {date}"
            )
        elif self.legacy_result is not None:
            self.curves_summary.setText(
                f"Legacy intensity calibration loaded (R² = {getattr(self.legacy_result, 'r_squared', 0):.3f}). "
                "Fit standard spectra to build element curves."
            )
        else:
            self.curves_summary.setText("No calibration curves yet — fit standard spectra first.")

    def _refresh_curves_table(self):
        self._updating = True
        try:
            curves = list(self.calibration.curves.values())
            self.curves_table.setRowCount(len(curves))
            for row, c in enumerate(curves):
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                use.setCheckState(Qt.Checked if c.enabled else Qt.Unchecked)
                self.curves_table.setItem(row, 0, use)
                self.curves_table.setItem(row, 1, QTableWidgetItem(c.element))
                self.curves_table.setItem(row, 2, QTableWidgetItem(c.line_group))
                self.curves_table.setItem(row, 3, _num_item(c.n_standards, "d"))
                self.curves_table.setItem(row, 4, _num_item(c.n_spectra, "d"))
                if c.fitted:
                    self.curves_table.setItem(row, 5, _num_item(c.slope, ".4g"))
                    self.curves_table.setItem(row, 6, _num_item(c.r_squared, ".4f"))
                    self.curves_table.setItem(row, 7, _num_item(c.rmse, ".3g"))
                else:
                    for col in (5, 6, 7):
                        self.curves_table.setItem(row, col, QTableWidgetItem("—"))
                self.curves_table.setItem(row, 8, _num_item(c.mean_rsd_percent, ".1f"))
                note = QTableWidgetItem(c.message)
                self.curves_table.setItem(row, 9, note)
                if not c.fitted:
                    for col in range(self.curves_table.columnCount()):
                        item = self.curves_table.item(row, col)
                        if item:
                            item.setForeground(Qt.gray)
                elif c.message:
                    note.setForeground(Qt.darkYellow)
            # Keep selection on the same element
            if self._selected_element:
                for row, c in enumerate(curves):
                    if c.element == self._selected_element:
                        self.curves_table.selectRow(row)
                        break
                else:
                    self._selected_element = None
            if self._selected_element is None and curves:
                self.curves_table.selectRow(0)
                self._selected_element = curves[0].element
        finally:
            self._updating = False

    def _on_curve_selection_changed(self):
        if self._updating:
            return
        rows = {i.row() for i in self.curves_table.selectedIndexes()}
        if len(rows) != 1:
            return
        item = self.curves_table.item(next(iter(rows)), 1)
        if item:
            self._selected_element = item.text()
            self._refresh_points_table()
            self._plot_selected_curve()
            self.plot_stack.setCurrentIndex(1)

    def _on_curve_item_changed(self, item):
        if self._updating or item.column() != 0:
            return
        el_item = self.curves_table.item(item.row(), 1)
        curve = self.calibration.curves.get(el_item.text()) if el_item else None
        if curve is None:
            return
        curve.enabled = item.checkState() == Qt.Checked
        self._refresh_all()

    def _current_curve(self) -> Optional[ElementCurve]:
        if not self._selected_element:
            return None
        return self.calibration.curves.get(self._selected_element)

    def _refresh_points_table(self):
        curve = self._current_curve()
        self._updating = True
        try:
            points = curve.points if curve else []
            self.points_table.setRowCount(len(points))
            for row, p in enumerate(points):
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                use.setCheckState(Qt.Checked if p.included else Qt.Unchecked)
                self.points_table.setItem(row, 0, use)
                self.points_table.setItem(row, 1, QTableWidgetItem(p.standard))
                self.points_table.setItem(row, 2, _num_item(p.concentration, ".4g"))
                self.points_table.setItem(row, 3, _num_item(p.intensity, ".4g"))
                self.points_table.setItem(row, 4, _num_item(p.intensity_sd, ".3g"))
                self.points_table.setItem(row, 5, _num_item(p.rsd_percent, ".1f"))
                self.points_table.setItem(row, 6, _num_item(p.n_spots, "d"))
                self.points_table.setItem(row, 7, _num_item(p.predicted, ".4g") if p.predicted is not None else QTableWidgetItem("—"))
                self.points_table.setItem(row, 8, _num_item(p.residual, "+.3g") if p.residual is not None else QTableWidgetItem("—"))
                if not p.included:
                    for col in range(self.points_table.columnCount()):
                        item = self.points_table.item(row, col)
                        if item:
                            item.setForeground(Qt.gray)
        finally:
            self._updating = False

    def _on_point_item_changed(self, item):
        if self._updating or item.column() != 0:
            return
        curve = self._current_curve()
        std_item = self.points_table.item(item.row(), 1)
        if curve is None or std_item is None:
            return
        self.calibration.set_point_included(curve.element, std_item.text(), item.checkState() == Qt.Checked)
        self._rebuild_curves()

    # ---- curve plot -------------------------------------------------------- #
    def _clear_curve_plot(self):
        for it in self._curve_plot_items:
            for plot in (self.curve_plot, self.residual_plot):
                try:
                    plot.removeItem(it)
                except Exception:
                    pass
        self._curve_plot_items.clear()
        try:
            self.curve_plot.legend.clear()
        except Exception:
            pass

    def _plot_selected_curve(self):
        self._clear_curve_plot()
        curve = self._current_curve()
        if curve is None:
            self.curve_plot.setTitle("Calibration curve", color="k")
            return
        title = f"{curve.element} {curve.line_group} — "
        if curve.fitted:
            title += f"R² = {curve.r_squared:.4f}, RMSE = {curve.rmse:.3g} wt%, replicate RSD ≈ {curve.mean_rsd_percent:.1f}%"
        else:
            title += curve.message or "not fitted"
        self.curve_plot.setTitle(title, color="k")

        inc = [p for p in curve.points if p.included]
        exc = [p for p in curve.points if not p.included]

        def _scatter(points, brush, pen, name):
            if not points:
                return
            x = np.array([p.concentration for p in points])
            y = np.array([p.intensity for p in points])
            err = np.array([p.intensity_sd for p in points])
            sc = pg.ScatterPlotItem(x=x, y=y, size=9, brush=brush, pen=pen, name=name)
            sc.setData(x=x, y=y, data=[p.standard for p in points])
            sc.setToolTip("Hover a point for its standard")
            self.curve_plot.addItem(sc)
            self._curve_plot_items.append(sc)
            eb = pg.ErrorBarItem(x=x, y=y, top=err, bottom=err, beam=0.0, pen=pen)
            self.curve_plot.addItem(eb)
            self._curve_plot_items.append(eb)
            for p in points:
                label = pg.TextItem(p.standard, color=(80, 80, 80), anchor=(0, 1))
                label.setPos(p.concentration, p.intensity)
                self.curve_plot.addItem(label)
                self._curve_plot_items.append(label)

        _scatter(inc, pg.mkBrush("#1f77b4"), pg.mkPen("#1f77b4"), "Used (±1 SD of spots)")
        _scatter(exc, pg.mkBrush(None), pg.mkPen("#999999"), "Excluded")

        if curve.fitted:
            pts = curve.points
            i_max = max(p.intensity for p in pts) * 1.1 if pts else 1.0
            i_grid = np.linspace(0.0, i_max, 200)
            c_grid = np.array([curve.evaluate(i) for i in i_grid])
            ok = c_grid >= 0
            fit_line = self.curve_plot.plot(c_grid[ok], i_grid[ok], pen=pg.mkPen("r", width=2), name="Fit")
            self._curve_plot_items.append(fit_line)
            if inc:
                x = np.array([p.concentration for p in inc])
                r = np.array([p.residual for p in inc])
                e = np.array([abs(curve.slope) * p.intensity_sem for p in inc])
                sc = pg.ScatterPlotItem(x=x, y=r, size=8, brush=pg.mkBrush("#1f77b4"), pen=pg.mkPen("#1f77b4"))
                self.residual_plot.addItem(sc)
                self._curve_plot_items.append(sc)
                eb = pg.ErrorBarItem(x=x, y=r, top=e, bottom=e, beam=0.0, pen=pg.mkPen("#1f77b4"))
                self.residual_plot.addItem(eb)
                self._curve_plot_items.append(eb)
        self.curve_plot.enableAutoRange()
        self.residual_plot.enableAutoRange()

    def _on_left_tab_changed(self, index):
        if index == 0:
            self.plot_stack.setCurrentIndex(0)
        elif index == 2:
            self.plot_stack.setCurrentIndex(1)

    # ------------------------------------------------------------------ #
    # Apply / save / load
    # ------------------------------------------------------------------ #
    def _apply_calibration(self):
        result = self.calibration_result
        if result is None:
            QMessageBox.warning(self, "No Calibration", "Fit standard spectra first.")
            return
        self._auto_save_calibration()
        self.calibration_complete.emit(result)
        if isinstance(result, StandardsCalibration):
            n = len(result.fitted_curves())
            QMessageBox.information(
                self, "Calibration Applied",
                f"{n} element curve(s) applied. Fitted spectra on the Analysis tab "
                "now report wt% for these elements.\n\n"
                "The calibration is auto-saved and reloaded next time.",
            )
        else:
            QMessageBox.information(self, "Calibration Applied", "Legacy intensity calibration applied.")

    def _save_calibration(self):
        result = self.calibration_result if self.calibration_result is not None else (
            self.calibration if self.calibration.curves else None
        )
        if result is None:
            QMessageBox.warning(self, "No Calibration", "Fit standard spectra first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Standards Calibration",
            str(Path.home() / "standards_calibration.json"), "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            result.save(path)
            QMessageBox.information(self, "Calibration Saved", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save calibration:\n{exc}")

    def _load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Standards Calibration", str(Path.home()), "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            self._load_calibration_file(path)
            QMessageBox.information(self, "Calibration Loaded", f"Loaded from:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load calibration:\n{exc}")

    def _load_calibration_file(self, path):
        import json

        with open(path) as fh:
            data = json.load(fh)
        result = load_any_standards_calibration(data)
        if isinstance(result, StandardsCalibration):
            self.legacy_result = None
            self._adopt_calibration(result, load_spectra=True)
            if result.has_intensities():
                self._rebuild_curves()
            else:
                self._refresh_all()
            self._log(f"✓ Loaded standards calibration from {path}")
        else:
            self.legacy_result = result
            self._refresh_all()
            self._log(f"✓ Loaded legacy intensity calibration from {path}")

    def _export_csv(self):
        if not self.calibration.curves:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Calibration CSV",
            str(Path.home() / "standards_calibration.csv"), "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            self.calibration.export_csv(path)
            QMessageBox.information(self, "Exported", f"Curves and points written to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ---- standards set persistence ------------------------------------------ #
    def _standards_set_dict(self):
        return {
            "type": "standards_set",
            "standards": {k: v.to_dict() for k, v in self.calibration.standards.items()},
        }

    def _auto_save_standards_set(self):
        try:
            import json

            with open(self.get_default_standards_set_path(), "w") as fh:
                json.dump(self._standards_set_dict(), fh, indent=2)
        except Exception as exc:
            self._log(f"⚠ Could not auto-save standards set: {exc}")

    def _save_standards_set_as(self):
        if not self.calibration.standards:
            QMessageBox.information(self, "Nothing to Save", "Add a standard first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Standards Set", str(Path.home() / "standards_set.json"), "JSON Files (*.json)",
        )
        if not path:
            return
        import json

        with open(path, "w") as fh:
            json.dump(self._standards_set_dict(), fh, indent=2)

    def _load_standards_set_from(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Standards Set", str(Path.home()), "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            self._load_standards_set_file(path, replace=True)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load standards set:\n{exc}")

    def _load_standards_set_file(self, path, *, replace: bool):
        import json

        with open(path) as fh:
            data = json.load(fh)
        records = {k: StandardRecord.from_dict(v) for k, v in (data.get("standards") or {}).items()}
        if replace:
            self.calibration = StandardsCalibration()
            self.spectra = {}
        for name, rec in records.items():
            self.calibration.add_standard(rec)
        self._adopt_calibration(self.calibration, load_spectra=True)
        self._after_standards_changed()
        self._refresh_all()
        self._log(f"✓ Loaded {len(records)} standard(s) from {path}")

    # ---- startup ----------------------------------------------------------- #
    def _auto_load(self):
        cal_path = self.get_default_calibration_path()
        set_path = self.get_default_standards_set_path()
        loaded_cal = False
        if cal_path.exists():
            try:
                self._load_calibration_file(str(cal_path))
                loaded_cal = isinstance(self.calibration_result, StandardsCalibration) or bool(self.calibration.standards)
                if self.calibration_result is not None:
                    self.calibration_complete.emit(self.calibration_result)
            except Exception as exc:
                self._log(f"⚠ Could not load saved calibration: {exc}")
        if not loaded_cal and set_path.exists():
            try:
                self._load_standards_set_file(str(set_path), replace=False)
            except Exception as exc:
                self._log(f"⚠ Could not load saved standards set: {exc}")
        if not cal_path.exists() and not set_path.exists():
            self._log("No saved standards yet — click Add Standard to begin.")

    def _auto_save_calibration(self):
        result = self.calibration_result
        if result is None:
            return
        try:
            result.save(str(self.get_default_calibration_path()))
            self._log(f"✓ Auto-saved calibration to {self.get_default_calibration_path()}")
        except Exception as exc:
            self._log(f"⚠ Auto-save failed: {exc}")

    def _log(self, text: str):
        self.log_output.append(text)


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _app_data_dir() -> Path:
    app_data = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    if not app_data:
        app_data = str(Path.home() / ".xrflab")
    cal_dir = Path(app_data) / "calibrations"
    cal_dir.mkdir(parents=True, exist_ok=True)
    return cal_dir


def _num_item(value, fmt) -> QTableWidgetItem:
    try:
        text = format(value, fmt)
    except (TypeError, ValueError):
        text = str(value)
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return item


def _set_combo_data(combo: QComboBox, data):
    for i in range(combo.count()):
        if combo.itemData(i) == data:
            combo.setCurrentIndex(i)
            return


def _mean_spectrum(entries):
    """Average counts across spot spectra (requires matching energy grids)."""
    if not entries:
        return None
    from core.spectrum import Spectrum

    ref = entries[0]["spectrum"]
    stack = []
    for entry in entries:
        spec = entry["spectrum"]
        if len(spec.energy) != len(ref.energy) or not np.allclose(spec.energy, ref.energy, rtol=0, atol=1e-6):
            return None
        stack.append(spec.counts)
    return Spectrum(
        energy=ref.energy.copy(),
        counts=np.mean(np.vstack(stack), axis=0),
        live_time=float(np.mean([e["spectrum"].live_time for e in entries])),
        real_time=float(np.mean([e["spectrum"].real_time for e in entries])),
        metadata={"averaged_from": len(entries)},
    )
