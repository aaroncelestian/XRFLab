"""Non-modal popup that displays a spectrum extracted from a map pixel."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from core.mapping.models import MapSpectrum


class PixelSpectrumPopup(QDialog):
    """Floating spectrum viewer; call set_spectrum() to refresh in place."""

    send_requested = Signal(object, object)  # Spectrum, peak_labels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pixel spectrum")
        self.setWindowFlag(Qt.Tool, True)  # stays above parent, non-modal
        self.setMinimumSize(520, 360)
        self.resize(640, 420)
        self._map_spectrum: Optional[MapSpectrum] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.header = QLabel("Click a map pixel to extract a spectrum")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.setLabel("bottom", "Energy (keV)")
        self.plot.setLabel("left", "Counts")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.curve = self.plot.plot(pen=pg.mkPen("#1f77b4", width=1.5))
        layout.addWidget(self.plot, stretch=1)

        controls = QHBoxLayout()
        self.log_check = QCheckBox("Log Y")
        self.log_check.toggled.connect(self._on_log_toggled)
        controls.addWidget(self.log_check)
        controls.addStretch(1)
        self.send_btn = QPushButton("Send → Analysis")
        self.send_btn.clicked.connect(self._emit_send)
        self.send_btn.setEnabled(False)
        controls.addWidget(self.send_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        controls.addWidget(self.close_btn)
        layout.addLayout(controls)

    def set_spectrum(self, map_spectrum: MapSpectrum) -> None:
        self._map_spectrum = map_spectrum
        sp = map_spectrum.spectrum
        energy = np.asarray(sp.energy, dtype=np.float64)
        counts = np.asarray(sp.counts, dtype=np.float64)
        self.curve.setData(energy, counts)
        self.plot.enableAutoRange(axis="xy")
        x = map_spectrum.x
        y = map_spectrum.y
        xy = f"({int(x)}, {int(y)})" if x is not None and y is not None else ""
        self.setWindowTitle(f"Pixel spectrum {xy}".strip())
        self.header.setText(
            f"{map_spectrum.name} — {sp.total_counts:.0f} total counts  ·  "
            f"{sp.num_channels} channels"
        )
        self.send_btn.setEnabled(True)
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

    def _on_log_toggled(self, checked: bool) -> None:
        self.plot.setLogMode(y=bool(checked))

    def _emit_send(self) -> None:
        if self._map_spectrum is None:
            return
        self.send_requested.emit(
            self._map_spectrum.spectrum,
            self._map_spectrum.peak_labels,
        )
