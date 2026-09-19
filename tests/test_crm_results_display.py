"""CRM Results table must not show negative wt% or a closed-assay total."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.results_panel import ResultsPanel


def _app():
    return QApplication.instance() or QApplication([])


def test_crm_results_show_nq_and_no_sum_as_total():
    _app()
    panel = ResultsPanel()
    try:
        panel.set_quantification({
            "Fe": {
                "concentration": -1.455,
                "error": 5.14,
                "lines": ["Kα1", "Kβ1"],
                "line": "K (Kα+Kβ)",
                "method": "standards_curve",
                "not_quantified": True,
                "quant_flag": "negative",
                "mdc": 1.6,
                "raw_concentration": -1.455,
                "intensity_cps": 12.0,
            },
            "Ca": {
                "concentration": 17.2,
                "error": 2.8,
                "lines": ["Kα1"],
                "line": "K",
                "method": "standards_curve",
                "not_quantified": False,
                "quant_flag": None,
                "mdc": 0.5,
                "raw_concentration": 17.2,
                "intensity_cps": 80.0,
            },
        })
        assert panel.results_table.item(0, 1).text() == "n.q."
        assert "not quantified" in (panel.results_table.item(0, 1).toolTip() or "").lower() \
            or "below" in (panel.results_table.item(0, 1).toolTip() or "").lower() \
            or "intercept" in (panel.results_table.item(0, 1).toolTip() or "").lower()
        assert "17.200" in panel.results_table.item(1, 1).text()
        assert "closed assay" in panel.total_label.text().lower()
        assert "28" not in panel.total_label.text()
        assert "sum of calibrated" not in panel.total_label.text().lower()
    finally:
        panel.deleteLater()
