"""Simple collapsible section for dense side panels."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleSection(QWidget):
    """A titled section that can expand/collapse its body."""

    def __init__(self, title: str, parent=None, *, expanded: bool = True):
        super().__init__(parent)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 4, 4, 8)
        self._body_layout.setSpacing(6)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("collapsibleHeader")
        header.setStyleSheet(
            "#collapsibleHeader {"
            "  background: #ececec;"
            "  border: 1px solid #c8c8c8;"
            "  border-radius: 3px;"
            "}"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(4, 2, 4, 2)

        self._toggle = QToolButton()
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: 600; }")
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.toggled.connect(self._on_toggled)
        header_layout.addWidget(self._toggle)
        header_layout.addStretch(1)

        root.addWidget(header)
        root.addWidget(self._body)
        self._body.setVisible(expanded)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    @property
    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def addWidget(self, widget: QWidget) -> None:
        self._body_layout.addWidget(widget)

    def addLayout(self, layout) -> None:
        self._body_layout.addLayout(layout)

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(bool(expanded))

    def _on_toggled(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
