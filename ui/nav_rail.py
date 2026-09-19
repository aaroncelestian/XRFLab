"""Left-hand lists that switch a stack. Same index API the panels already call."""

from PySide6.QtCore import Qt, QSize, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QWidget,
)


class IndexedStack(QWidget):
    """Vertical list plus pages. Implements the QTabWidget calls this app uses."""

    currentChanged = Signal(int)

    def __init__(self, parent=None, *, rail_width: int = 148, object_name: str = "stepRail"):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._list = QListWidget()
        self._list.setObjectName(object_name)
        self._list.setFixedWidth(rail_width)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setSpacing(2)
        self._list.setFocusPolicy(Qt.NoFocus)
        self._stack = QStackedWidget()
        layout.addWidget(self._list)
        layout.addWidget(self._stack, stretch=1)
        self._list.currentRowChanged.connect(self._on_row)

    def addTab(self, widget: QWidget, title: str) -> None:
        item = QListWidgetItem(title)
        item.setTextAlignment(Qt.AlignCenter)
        item.setSizeHint(QSize(self._list.width(), 40))
        self._list.addItem(item)
        self._stack.addWidget(widget)
        if self._list.currentRow() < 0:
            self._list.setCurrentRow(0)

    def setCurrentIndex(self, index: int) -> None:
        index = int(index)
        if index < 0 or index >= self._stack.count():
            return
        self._list.setCurrentRow(index)
        self._stack.setCurrentIndex(index)

    def currentIndex(self) -> int:
        row = self._list.currentRow()
        return row if row >= 0 else 0

    def setCurrentWidget(self, widget: QWidget) -> None:
        index = self._stack.indexOf(widget)
        if index >= 0:
            self.setCurrentIndex(index)

    def currentWidget(self) -> QWidget:
        return self._stack.currentWidget()

    def count(self) -> int:
        return self._stack.count()

    def widget(self, index: int) -> QWidget:
        return self._stack.widget(index)

    def indexOf(self, widget: QWidget) -> int:
        return self._stack.indexOf(widget)

    def _on_row(self, row: int) -> None:
        if row < 0:
            return
        if self._stack.currentIndex() != row:
            self._stack.setCurrentIndex(row)
        self.currentChanged.emit(row)


class AnalysisSteps(IndexedStack):
    """Analyze steps. A saved Composition index opens Results and scrolls to it."""

    def __init__(self, parent=None):
        super().__init__(parent, rail_width=132, object_name="stepRail")
        self.results_scroll = None
        self.composition_anchor = None

    def setCurrentIndex(self, index: int) -> None:
        index = int(index)
        reveal = index >= 5
        super().setCurrentIndex(4 if reveal else index)
        if reveal:
            QTimer.singleShot(0, self._reveal_composition)

    def _reveal_composition(self) -> None:
        if self.results_scroll is None or self.composition_anchor is None:
            return
        self.results_scroll.ensureWidgetVisible(self.composition_anchor)
