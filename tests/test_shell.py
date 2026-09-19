"""Smoke test: the console builds and the figure windows host the old widgets."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def test_shell_builds():
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("XRFLabShellTest")
    app.setApplicationName("XRFLabShellTest")

    window = MainWindow()
    try:
        assert window.tab_widget.count() == 4
        assert window.analysis_left_tabs.count() == 5
        assert window.spectrum_widget.parent() is not None
        assert window.batch_analysis_panel.batch_plot_page.parent() is not None
        assert window.mapping_panel.figure_host.parent() is not None
        assert window.charts_window.isHidden() or window.charts_window.isVisible()
        window.analysis_left_tabs.setCurrentIndex(window.TAB_COMPOSITION)
        assert window.analysis_left_tabs.currentIndex() == window.TAB_RESULTS
        window.tab_widget.setCurrentWidget(window.mapping_panel)
        assert window.tab_widget.currentIndex() == 2
        window.tab_widget.setCurrentWidget(window.batch_analysis_panel)
        assert window.tab_widget.currentIndex() == 1
        window.show_fwhm_calibration()
        assert window.tab_widget.currentWidget() is window.calibration_tab
        assert window.calibration_tabs.currentWidget() is window.fwhm_calibration_panel
    finally:
        window.close()
