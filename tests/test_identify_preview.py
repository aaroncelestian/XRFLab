"""Selecting an identify-on-plot candidate previews that element's lines."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.element_panel import ElementPanel
from ui.spectrum_widget import SpectrumWidget


def _app():
    return QApplication.instance() or QApplication([])


def _hits():
    return [
        {
            "symbol": "Fe",
            "z": 26,
            "line": "Kα1",
            "line_energy": 6.404,
            "delta_kev": -0.004,
            "delta_ev": -4.0,
            "abs_delta_kev": 0.004,
        },
        {
            "symbol": "Mn",
            "z": 25,
            "line": "Kβ1",
            "line_energy": 6.490,
            "delta_kev": -0.090,
            "delta_ev": -90.0,
            "abs_delta_kev": 0.090,
        },
    ]


def test_identify_list_preview_emits_on_populate_and_selection():
    _app()
    panel = ElementPanel()
    seen = []
    panel.identify_preview_element.connect(lambda symbol, z: seen.append((symbol, z)))

    try:
        panel.set_identify_candidates(6.40, _hits())
        assert seen == [("Fe", 26)]
        assert panel.identify_add_btn.isEnabled()

        panel.identify_candidates_list.setCurrentRow(1)
        assert seen[-1] == ("Mn", 25)
    finally:
        panel.deleteLater()


def test_identify_list_empty_candidates_does_not_preview():
    _app()
    panel = ElementPanel()
    seen = []
    panel.identify_preview_element.connect(lambda symbol, z: seen.append((symbol, z)))

    try:
        panel.set_identify_candidates(1.11, [])
        assert seen == []
        assert not panel.identify_add_btn.isEnabled()
    finally:
        panel.deleteLater()


def test_show_element_lines_adds_markers_for_fe():
    _app()
    widget = SpectrumWidget()
    try:
        widget.show_element_lines("Fe", 26)
        labels = [spec.get("label", "") for spec in widget._peak_marker_specs]
        assert any(label.startswith("Fe-") for label in labels)
    finally:
        widget.deleteLater()
