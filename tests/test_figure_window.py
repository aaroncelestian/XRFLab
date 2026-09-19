"""Figure windows must accept app-quit closes so Python does not linger."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from ui.figure_window import FigureWindow


def _app():
    return QApplication.instance() or QApplication([])


def test_programmatic_close_is_accepted():
    _app()
    window = FigureWindow("Spectrum", "test_quit_programmatic")
    try:
        event = QCloseEvent()
        window.closeEvent(event)
        assert event.isAccepted()
    finally:
        window._shutting_down = True
        window.close()
        window.deleteLater()


def test_user_close_hides_instead_of_destroying():
    _app()
    window = FigureWindow("Charts", "test_quit_user")
    window.show()
    try:
        event = QCloseEvent()
        event.spontaneous = lambda: True  # type: ignore[method-assign]
        window.closeEvent(event)
        assert not event.isAccepted()
        assert not window.isVisible()
    finally:
        window._shutting_down = True
        window.close()
        window.deleteLater()
