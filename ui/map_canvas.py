"""Interactive map canvas with line-transect drawing (pyqtgraph)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


class MapCanvas(QWidget):
    """Display a 2D map / overview and optionally draw a transect line."""

    line_drawn = Signal(float, float, float, float)  # x0, y0, x1, y1
    cursor_moved = Signal(float, float, float)  # x, y, value
    pixel_clicked = Signal(float, float)  # x, y (map coords)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.graphics = pg.GraphicsLayoutWidget()
        self.view = self.graphics.addViewBox(lockAspect=True, enableMenu=False)
        self.view.setAspectLocked(True)
        self.image_item = pg.ImageItem()
        self.view.addItem(self.image_item)

        self.line_item = pg.PlotDataItem(pen=pg.mkPen("#ffcc00", width=2))
        self.view.addItem(self.line_item)
        self.spot_scatter = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush(0, 200, 255, 180),
            pen=pg.mkPen(None),
        )
        self.view.addItem(self.spot_scatter)

        layout.addWidget(self.graphics)

        self._data: Optional[np.ndarray] = None
        self._drawing = False
        self._line_mode = False
        self._pick_mode = False
        self._start: Optional[Tuple[float, float]] = None
        self._end: Optional[Tuple[float, float]] = None
        self._rgb_mode = False

        self.proxy = pg.SignalProxy(
            self.image_item.scene().sigMouseClicked,
            rateLimit=60,
            slot=self._on_click,
        )
        self.move_proxy = pg.SignalProxy(
            self.image_item.scene().sigMouseMoved,
            rateLimit=40,
            slot=self._on_move,
        )

    def set_line_mode(self, enabled: bool) -> None:
        self._line_mode = bool(enabled)
        if enabled:
            self._pick_mode = False
        if not enabled:
            self._drawing = False
            self._start = None

    def set_pick_mode(self, enabled: bool) -> None:
        self._pick_mode = bool(enabled)
        if enabled:
            self._line_mode = False
            self._drawing = False

    def clear_line(self) -> None:
        self._start = None
        self._end = None
        self.line_item.setData([], [])

    def set_image(
        self,
        data: np.ndarray,
        *,
        rgb: bool = False,
        auto_levels: bool = True,
    ) -> None:
        """Set map data. For grayscale, expects (H, W); for RGB (H, W, 3)."""
        self._rgb_mode = rgb
        arr = np.asarray(data)
        if rgb:
            if arr.ndim != 3 or arr.shape[2] < 3:
                raise ValueError("RGB image must be HxWx3")
            # pyqtgraph ImageItem expects (W, H, 3) when axisOrder default...
            # Use axisOrder='row-major' so (H,W) displays correctly.
            disp = np.clip(arr[:, :, :3], 0, 1)
            if disp.dtype != np.float64 and disp.max() > 1.5:
                disp = disp.astype(np.float64) / max(float(disp.max()), 1.0)
            self._data = arr[:, :, 0]  # for cursor sampling
            self.image_item.setImage(
                np.ascontiguousarray(disp),
                autoLevels=False,
                levels=(0, 1),
                axisOrder="row-major",
            )
        else:
            self._data = np.asarray(arr, dtype=np.float64)
            levels = None
            if auto_levels and self._data.size:
                lo, hi = np.percentile(self._data, (2, 98))
                if hi <= lo:
                    hi = lo + 1.0
                levels = (lo, hi)
            self.image_item.setImage(
                np.ascontiguousarray(self._data),
                autoLevels=levels is None,
                levels=levels,
                axisOrder="row-major",
            )
        h, w = self._data.shape[:2]
        self.view.setRange(QRectF(0, 0, w, h), padding=0.02)

    def set_spot_markers(self, xs, ys) -> None:
        if xs is None or len(xs) == 0:
            self.spot_scatter.setData([])
            return
        self.spot_scatter.setData(x=list(xs), y=list(ys))

    def set_line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self._start = (x0, y0)
        self._end = (x1, y1)
        self.line_item.setData([x0, x1], [y0, y1])

    def current_line(self) -> Optional[Tuple[float, float, float, float]]:
        if self._start is None or self._end is None:
            return None
        return (*self._start, *self._end)

    def _map_pos(self, scene_pos) -> Optional[Tuple[float, float]]:
        if self.image_item.scene() is None:
            return None
        try:
            p = self.image_item.mapFromScene(scene_pos)
            return float(p.x()), float(p.y())
        except Exception:
            return None

    def _on_click(self, event):
        ev = event[0]
        if not hasattr(ev, "scenePos"):
            return
        pos = self._map_pos(ev.scenePos())
        if pos is None or self._data is None:
            return
        x, y = pos
        h, w = self._data.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        if self._pick_mode:
            self.pixel_clicked.emit(x, y)
            self.spot_scatter.setData(x=[x], y=[y])
            return

        if not self._line_mode:
            return
        if not self._drawing or self._start is None:
            self._start = (x, y)
            self._end = None
            self._drawing = True
            self.line_item.setData([x], [y])
        else:
            self._end = (x, y)
            self._drawing = False
            self.line_item.setData(
                [self._start[0], x], [self._start[1], y]
            )
            self.line_drawn.emit(self._start[0], self._start[1], x, y)

    def _on_move(self, event):
        if self._data is None:
            return
        pos = event[0]
        mapped = self._map_pos(pos)
        if mapped is None:
            return
        x, y = mapped
        h, w = self._data.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            ix, iy = int(x), int(y)
            val = float(self._data[iy, ix])
            self.cursor_moved.emit(x, y, val)
        if self._line_mode and self._drawing and self._start is not None:
            self.line_item.setData([self._start[0], x], [self._start[1], y])
