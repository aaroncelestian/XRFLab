"""Non-modal figure windows for spectrum, map, and chart views."""

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def attach_view_picker(tabs: QTabWidget) -> QComboBox:
    """Replace a tab bar with a combo. Index changes still go through the tabs."""
    tabs.tabBar().hide()
    tabs.setDocumentMode(True)
    combo = QComboBox()
    combo.setObjectName("viewPicker")
    for i in range(tabs.count()):
        combo.addItem(tabs.tabText(i))

    def to_tabs(index: int) -> None:
        if index >= 0 and tabs.currentIndex() != index:
            tabs.setCurrentIndex(index)

    def to_combo(index: int) -> None:
        if index >= 0 and combo.currentIndex() != index:
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

    combo.currentIndexChanged.connect(to_tabs)
    tabs.currentChanged.connect(to_combo)
    if tabs.currentIndex() >= 0:
        combo.setCurrentIndex(tabs.currentIndex())
    return combo


class FigureWindow(QMainWindow):
    """Secondary window. Closing hides it; the hosted widgets stay alive."""

    def __init__(self, title: str, settings_key: str, parent=None, *, offset: int = 0):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setWindowFlags(Qt.Window)
        self.setMinimumSize(480, 360)
        self.resize(960, 680)
        self.move(140 + offset, 80 + offset)

        self._settings = QSettings()
        self._geo_key = f"figure/{settings_key}/geometry"
        self._vis_key = f"figure/{settings_key}/visible"
        self._order: list[str] = []
        self._shutting_down = False

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        picker_row = QHBoxLayout()
        picker_row.setSpacing(8)
        label = QLabel("View")
        label.setObjectName("sectionLabel")
        self._picker = QComboBox()
        self._picker.setObjectName("viewPicker")
        self._picker.currentIndexChanged.connect(self._on_picker)
        picker_row.addWidget(label)
        picker_row.addWidget(self._picker, stretch=1)
        self._picker_row = picker_row
        self._picker_wrap = QWidget()
        self._picker_wrap.setLayout(picker_row)
        self._picker_wrap.hide()
        layout.addWidget(self._picker_wrap)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, stretch=1)
        self.setCentralWidget(central)

    def add_view(self, name: str, widget: QWidget) -> None:
        self._order.append(name)
        self._stack.addWidget(widget)
        self._picker.blockSignals(True)
        self._picker.addItem(name)
        self._picker.blockSignals(False)
        self._picker_wrap.setVisible(self._picker.count() > 1)

    def show_view(self, name: str, *, focus: bool = False) -> None:
        if name in self._order:
            index = self._order.index(name)
            if self._picker.currentIndex() != index:
                self._picker.setCurrentIndex(index)
            else:
                self._stack.setCurrentIndex(index)
        self.show_window(focus=focus)

    def show_window(self, *, focus: bool = False) -> None:
        self._settings.setValue(self._vis_key, True)
        self.show()
        self.raise_()
        if focus:
            self.activateWindow()

    def restore_geometry(self) -> None:
        geo = self._settings.value(self._geo_key)
        if geo:
            self.restoreGeometry(geo)

    def was_visible(self) -> bool:
        value = self._settings.value(self._vis_key)
        if value is None:
            return False
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    def save_geometry(self) -> None:
        self._settings.setValue(self._geo_key, self.saveGeometry())
        self._settings.setValue(self._vis_key, self.isVisible())

    def shutdown(self) -> None:
        """Really close this window — the application is exiting."""
        self._shutting_down = True
        self.save_geometry()
        self.close()

    def _on_picker(self, index: int) -> None:
        if 0 <= index < self._stack.count():
            self._stack.setCurrentIndex(index)

    def closeEvent(self, event):
        # User clicked this window's close box: hide so plots stay alive.
        # Cmd+Q / QApplication.quit() is not spontaneous — must accept or
        # Qt cancels the quit and Python keeps running with no UI.
        if self._shutting_down or not event.spontaneous():
            if not self._shutting_down:
                self.save_geometry()
            event.accept()
            return
        event.ignore()
        self.hide()
        self.save_geometry()
