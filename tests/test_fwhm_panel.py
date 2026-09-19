"""FWHM panel remembers the last foil folder and file ticks."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from ui.fwhm_calibration_panel import FWHMCalibrationPanel


def _app():
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("XRFLabFwhmPanelTest")
    app.setApplicationName("XRFLabFwhmPanelTest")
    return app


def test_fwhm_panel_restores_folder_and_assignments(tmp_path, monkeypatch):
    _app()
    QSettings().clear()
    foils = tmp_path / "foils"
    foils.mkdir()
    (foils / "Fe.txt").write_text("placeholder", encoding="utf-8")
    (foils / "Cu.txt").write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        FWHMCalibrationPanel, "get_default_calibration_path",
        staticmethod(lambda: tmp_path / "missing_fwhm.json"),
    )
    monkeypatch.setattr("ui.fwhm_calibration_panel.example_standards_dir", lambda: foils)

    first = FWHMCalibrationPanel()
    try:
        assert first.data_dir == foils
        by_name = {item.filename: item for item in first._scanned_files}
        assert by_name["Fe.txt"].included
        by_name["Cu.txt"].included = False
        first._save_session()
    finally:
        first.close()
        first.deleteLater()

    second = FWHMCalibrationPanel()
    try:
        assert second.data_dir == foils
        by_name = {item.filename: item for item in second._scanned_files}
        assert by_name["Fe.txt"].included
        assert not by_name["Cu.txt"].included
    finally:
        second.close()
        second.deleteLater()
        QSettings().clear()
