#!/usr/bin/env python3
"""
XRF Fundamental Parameters Analysis Application
Main entry point for the application
"""

import os
import sys
from pathlib import Path


def _ensure_qt_plugin_path() -> None:
    """
    Finder / .app launches strip the shell environment, so Qt cannot find
    libqcocoa.dylib unless QT_PLUGIN_PATH is set. Set it before QApplication.
    """
    if os.environ.get("QT_PLUGIN_PATH"):
        return
    try:
        import PySide6

        plugins = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
        if plugins.is_dir():
            os.environ["QT_PLUGIN_PATH"] = str(plugins)
    except Exception:
        pass


_ensure_qt_plugin_path()

from PySide6.QtWidgets import QApplication, QStyleFactory
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor, QIcon
from ui.main_window import MainWindow
from utils.paths import icon_path


def _force_light_mode(app: QApplication) -> None:
    """
    Override OS dark mode on macOS and Windows so UI text stays readable.
    Uses Qt color scheme + Fusion style + an explicit light palette.
    """
    # Qt 6.5+: tell the platform theme to stay in light mode
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    except Exception:
        pass

    # Fusion renders consistently across platforms and ignores native dark chrome
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")

    palette = QPalette()
    window = QColor(245, 245, 245)
    base = QColor(255, 255, 255)
    text = QColor(33, 33, 33)
    disabled = QColor(160, 160, 160)
    highlight = QColor(33, 150, 243)
    button = QColor(224, 224, 224)

    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(249, 249, 249))
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, button)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(51, 51, 51))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(140, 140, 140))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Link, highlight)

    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(200, 200, 200))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(100, 100, 100))

    app.setPalette(palette)


def main():
    """Initialize and run the XRF analysis application"""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("XRFLab")
    app.setOrganizationName("XRFLab")
    app.setApplicationVersion("1.0.0")
    _force_light_mode(app)

    icon_file = icon_path("xrflab.png")
    if icon_file.is_file():
        app.setWindowIcon(QIcon(str(icon_file)))
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
