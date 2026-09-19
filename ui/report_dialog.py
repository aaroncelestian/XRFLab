"""Dialog to choose analysis-report sections and composition display mode."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from core.composition import VALUE_RELATIVE, VALUE_WT
from core.report import (
    ALL_SECTIONS,
    SECTION_TITLES,
    ReportContext,
    ReportOptions,
    app_version,
    section_availability,
)


def context_from_main_window(window) -> ReportContext:
    """Collect report data from MainWindow panels and the analysis session."""
    session = window.session
    fwhm_panel = getattr(window, "fwhm_calibration_panel", None)
    tube_panel = getattr(window, "tube_profile_panel", None)
    std_panel = getattr(window, "standards_panel", None)
    batch = getattr(window, "batch_analysis_panel", None)
    comp = getattr(batch, "composition_panel", None) if batch is not None else None
    element_panel = getattr(window, "element_panel", None)

    fit_settings = {}
    if element_panel is not None and hasattr(element_panel, "get_fitting_params"):
        try:
            fit_settings = dict(element_panel.get_fitting_params() or {})
        except Exception:
            fit_settings = {}

    fwhm = None
    measurements = None
    if fwhm_panel is not None:
        fwhm = getattr(fwhm_panel, "fwhm_calibration", None)
        measurements = getattr(fwhm_panel, "measurements", None)
    if fwhm is None:
        fwhm = getattr(session.instrument.detector, "fwhm_calibration", None)

    standards = getattr(session.instrument, "standards_calibration", None)
    if standards is None or not getattr(standards, "curves", None):
        if std_panel is not None:
            panel_cal = getattr(std_panel, "calibration", None)
            if getattr(panel_cal, "curves", None):
                standards = panel_cal
            else:
                standards = getattr(std_panel, "calibration_result", None)

    tube = None
    if tube_panel is not None and hasattr(tube_panel, "get_library"):
        try:
            tube = tube_panel.get_library()
        except Exception:
            tube = None
    if tube is None:
        tube = getattr(session.instrument, "tube_profile_library", None)

    matrix = session.matrix
    results_panel = getattr(window, "results_panel", None)
    if results_panel is not None and hasattr(results_panel, "get_matrix_assumptions"):
        try:
            matrix = results_panel.get_matrix_assumptions()
        except Exception:
            pass

    rows = list(getattr(comp, "rows", None) or [])
    summaries = list(getattr(comp, "summaries", None) or [])

    return ReportContext(
        spectrum=session.spectrum,
        spectrum_path=session.spectrum_path,
        project_path=getattr(window, "_project_path", None),
        fit_result=session.fit_result,
        concentrations=dict(session.concentrations or {}),
        quantification_method=session.quantification_method,
        matrix=matrix,
        fp_result=session.fp_result,
        fwhm_calibration=fwhm,
        fwhm_measurements=measurements,
        tube_library=tube,
        standards_calibration=standards,
        composition_rows=rows,
        composition_summaries=summaries,
        detector=session.instrument.detector,
        excitation_kv=fit_settings.get("excitation_kv"),
        tube_element=fit_settings.get("tube_element"),
        fit_settings=fit_settings,
    )


class ReportDialog(QDialog):
    """Pick report sections, composition mode, and the HTML save path."""

    def __init__(self, parent=None, context: Optional[ReportContext] = None, composition_state: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Export Analysis Report")
        self.setMinimumWidth(520)
        self._context = context or ReportContext()
        self._availability = section_availability(self._context)
        self._checks = {}
        self._init_ui(composition_state or {})

    def _init_ui(self, state: dict) -> None:
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Write a self-contained HTML report of the current session. "
            "Unchecked or unavailable sections are omitted."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        sections = QGroupBox("Sections")
        sec_layout = QVBoxLayout(sections)
        for key in ALL_SECTIONS:
            reason = self._availability.get(key)
            box = QCheckBox(SECTION_TITLES[key])
            if reason:
                box.setChecked(False)
                box.setEnabled(False)
                box.setToolTip(reason)
                box.setText(f"{SECTION_TITLES[key]}  ({reason})")
            else:
                box.setChecked(True)
            self._checks[key] = box
            sec_layout.addWidget(box)
        layout.addWidget(sections)

        display = QGroupBox("Composition display")
        disp = QVBoxLayout(display)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Values"))
        self.value_combo = QComboBox()
        self.value_combo.addItem("FP wt%", VALUE_WT)
        self.value_combo.addItem("Relative intensity", VALUE_RELATIVE)
        source = str(state.get("value_source") or VALUE_RELATIVE)
        idx = self.value_combo.findData(VALUE_WT if source == VALUE_WT else VALUE_RELATIVE)
        if idx >= 0:
            self.value_combo.setCurrentIndex(idx)
        source_row.addWidget(self.value_combo, stretch=1)
        disp.addLayout(source_row)

        mode_row = QHBoxLayout()
        self.oxides_check = QCheckBox("Oxides")
        self.oxides_check.setChecked(bool(state.get("oxides")))
        mode_row.addWidget(self.oxides_check)
        self.fe_combo = QComboBox()
        self.fe_combo.addItems(["FeO", "Fe2O3", "Fe3O4"])
        fe_as = str(state.get("fe_as") or "FeO")
        if fe_as in ("FeO", "Fe2O3", "Fe3O4"):
            self.fe_combo.setCurrentText(fe_as)
        mode_row.addWidget(self.fe_combo)
        self.close_check = QCheckBox("Close to 100%")
        self.close_check.setChecked(bool(state.get("close")))
        mode_row.addWidget(self.close_check)
        mode_row.addStretch()
        disp.addLayout(mode_row)

        self.replicates_check = QCheckBox("Include replicate spectra")
        self.replicates_check.setChecked(True)
        disp.addWidget(self.replicates_check)
        layout.addWidget(display)

        path_box = QGroupBox("Output")
        path_layout = QVBoxLayout(path_box)
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit("xrflab_report.html")
        path_row.addWidget(self.path_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(browse)
        path_layout.addLayout(path_row)
        self.open_check = QCheckBox("Open in browser after export")
        self.open_check.setChecked(True)
        path_layout.addWidget(self.open_check)
        layout.addWidget(path_box)

        buttons = QHBoxLayout()
        buttons.addStretch()
        export_btn = QPushButton("Export")
        export_btn.setDefault(True)
        export_btn.clicked.connect(self._on_export)
        buttons.addWidget(export_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Report",
            self.path_edit.text() or "xrflab_report.html",
            "HTML (*.html);;All Files (*)",
        )
        if path:
            self.path_edit.setText(path)

    def _on_export(self) -> None:
        path = self.save_path()
        if not path:
            self._browse()
            if not self.save_path():
                return
        self.accept()

    def save_path(self) -> str:
        return str(self.path_edit.text() or "").strip()

    def open_after(self) -> bool:
        return bool(self.open_check.isChecked())

    def options(self) -> ReportOptions:
        sections = {
            key for key, box in self._checks.items() if box.isEnabled() and box.isChecked()
        }
        data = self.value_combo.currentData()
        source = data if data in (VALUE_WT, VALUE_RELATIVE) else VALUE_RELATIVE
        return ReportOptions(
            sections=sections,
            value_source=source,
            as_oxides=self.oxides_check.isChecked(),
            fe_as=self.fe_combo.currentText() or "FeO",
            close=self.close_check.isChecked(),
            include_replicates=self.replicates_check.isChecked(),
            app_version=app_version(),
        )
