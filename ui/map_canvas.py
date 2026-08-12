"""Interactive map canvas with single or multi-panel element-map display."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


def _grid_shape(n: int) -> Tuple[int, int]:
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _auto_levels(data: np.ndarray) -> Tuple[float, float]:
    if data.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(data, (2, 98))
    if hi <= lo:
        hi = float(lo) + 1.0
    return float(lo), float(hi)


class MapCanvas(QWidget):
    """Display one map or a grid of element-map subplots with shared tools."""

    line_drawn = Signal(float, float, float, float)  # x0, y0, x1, y1
    cursor_moved = Signal(float, float, float)  # x, y, value
    pixel_clicked = Signal(float, float)  # x, y (map coords)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)

        self._panels: List[dict] = []  # view, image, label, line, spot, data, title
        self._primary_data: Optional[np.ndarray] = None
        self._drawing = False
        self._line_mode = False
        self._pick_mode = False
        self._start: Optional[Tuple[float, float]] = None
        self._end: Optional[Tuple[float, float]] = None
        self._rgb_mode = False
        self._click_proxy = None
        self._move_proxy = None

        self._build_single_panel()
        self._connect_scene()

    # --------------------------------------------------------------- modes
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
        for p in self._panels:
            p["line"].setData([], [])

    def set_spot_markers(self, xs, ys) -> None:
        for p in self._panels:
            if xs is None or len(xs) == 0:
                p["spot"].setData([])
            else:
                p["spot"].setData(x=list(xs), y=list(ys))

    def set_line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self._start = (x0, y0)
        self._end = (x1, y1)
        for p in self._panels:
            p["line"].setData([x0, x1], [y0, y1])

    def current_line(self) -> Optional[Tuple[float, float, float, float]]:
        if self._start is None or self._end is None:
            return None
        return (*self._start, *self._end)

    # -------------------------------------------------------------- display
    def set_image(
        self,
        data: np.ndarray,
        *,
        rgb: bool = False,
        title: str = "",
        auto_levels: bool = True,
    ) -> None:
        """Single-panel mode (overview, RGB, or one map)."""
        self._rgb_mode = rgb
        self._ensure_panel_count(1)
        panel = self._panels[0]
        arr = np.asarray(data)
        if rgb:
            if arr.ndim != 3 or arr.shape[2] < 3:
                raise ValueError("RGB image must be HxWx3")
            disp = arr[:, :, :3]
            maxv = float(np.nanmax(disp)) if disp.size else 1.0
            if maxv > 1.5:
                disp = disp.astype(np.float64) / 255.0
            else:
                disp = disp.astype(np.float64)
            disp = np.clip(disp, 0.0, 1.0)
            self._primary_data = np.asarray(arr[:, :, 0], dtype=np.float64)
            panel["data"] = self._primary_data
            panel["image"].setImage(
                np.ascontiguousarray(disp),
                autoLevels=False,
                levels=(0, 1),
                axisOrder="row-major",
            )
        else:
            self._primary_data = np.asarray(arr, dtype=np.float64)
            panel["data"] = self._primary_data
            levels = _auto_levels(self._primary_data) if auto_levels else None
            panel["image"].setImage(
                np.ascontiguousarray(self._primary_data),
                autoLevels=levels is None,
                levels=levels,
                axisOrder="row-major",
            )
        panel["title"] = title or ""
        panel["label"].setText(panel["title"])
        h, w = self._primary_data.shape[:2]
        panel["view"].setRange(QRectF(0, 0, w, h), padding=0.02)
        self._restore_overlays()

    def set_images(
        self,
        panels: Sequence[Tuple[str, np.ndarray]],
        *,
        auto_levels: bool = True,
    ) -> None:
        """
        Multi-panel subplot mode.

        Args:
            panels: sequence of (title, HxW array)
        """
        items = [(str(t), np.asarray(d, dtype=np.float64)) for t, d in panels if d is not None]
        if not items:
            return
        self._rgb_mode = False
        self._ensure_panel_count(len(items))
        self._primary_data = items[0][1]

        for panel, (title, data) in zip(self._panels, items):
            panel["title"] = title
            panel["data"] = data
            panel["label"].setText(title)
            levels = _auto_levels(data) if auto_levels else None
            panel["image"].setImage(
                np.ascontiguousarray(data),
                autoLevels=levels is None,
                levels=levels,
                axisOrder="row-major",
            )
            h, w = data.shape[:2]
            panel["view"].setRange(QRectF(0, 0, w, h), padding=0.02)

        # Link all viewboxes to the first
        master = self._panels[0]["view"]
        for panel in self._panels[1:]:
            panel["view"].setXLink(master)
            panel["view"].setYLink(master)

        self._restore_overlays()

    # --------------------------------------------------------------- build
    def _clear_layout(self) -> None:
        self.graphics.clear()
        self._panels = []

    def _make_panel(self, row: int, col: int) -> dict:
        label = self.graphics.addLabel("", row=row * 2, col=col)
        view = self.graphics.addViewBox(
            row=row * 2 + 1,
            col=col,
            lockAspect=True,
            enableMenu=False,
        )
        view.setAspectLocked(True)
        image = pg.ImageItem()
        view.addItem(image)
        line = pg.PlotDataItem(pen=pg.mkPen("#ffcc00", width=2))
        view.addItem(line)
        spot = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush(0, 200, 255, 180),
            pen=pg.mkPen(None),
        )
        view.addItem(spot)
        return {
            "view": view,
            "image": image,
            "label": label,
            "line": line,
            "spot": spot,
            "data": None,
            "title": "",
        }

    def _build_single_panel(self) -> None:
        self._clear_layout()
        self._panels = [self._make_panel(0, 0)]

    def _ensure_panel_count(self, n: int) -> None:
        n = max(1, int(n))
        if n == 1 and len(self._panels) == 1:
            return
        self._clear_layout()
        rows, cols = _grid_shape(n)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n:
                    break
                self._panels.append(self._make_panel(r, c))
                idx += 1
        self._connect_scene()

    def _connect_scene(self) -> None:
        scene = self.graphics.scene()
        # Drop old proxies by overwriting refs (GC)
        self._click_proxy = pg.SignalProxy(
            scene.sigMouseClicked, rateLimit=60, slot=self._on_click
        )
        self._move_proxy = pg.SignalProxy(
            scene.sigMouseMoved, rateLimit=40, slot=self._on_move
        )

    def _restore_overlays(self) -> None:
        if self._start is not None and self._end is not None:
            self.set_line(*self._start, *self._end)
        elif self._start is not None:
            for p in self._panels:
                p["line"].setData([self._start[0]], [self._start[1]])

    def _panel_at(self, scene_pos) -> Optional[dict]:
        for panel in self._panels:
            try:
                p = panel["image"].mapFromScene(scene_pos)
                x, y = float(p.x()), float(p.y())
                data = panel["data"]
                if data is None:
                    continue
                h, w = data.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    return panel
            except Exception:
                continue
        # Fallback: first panel with data
        for panel in self._panels:
            if panel["data"] is not None:
                return panel
        return None

    def _map_pos(self, scene_pos, panel: Optional[dict] = None) -> Optional[Tuple[float, float]]:
        panel = panel or self._panel_at(scene_pos)
        if panel is None:
            return None
        try:
            p = panel["image"].mapFromScene(scene_pos)
            return float(p.x()), float(p.y())
        except Exception:
            return None

    def _on_click(self, event):
        ev = event[0]
        if not hasattr(ev, "scenePos"):
            return
        scene_pos = ev.scenePos()
        panel = self._panel_at(scene_pos)
        pos = self._map_pos(scene_pos, panel)
        if pos is None or panel is None or panel["data"] is None:
            return
        x, y = pos
        h, w = panel["data"].shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        if self._pick_mode:
            self.pixel_clicked.emit(x, y)
            self.set_spot_markers([x], [y])
            return

        if not self._line_mode:
            return
        if not self._drawing or self._start is None:
            self._start = (x, y)
            self._end = None
            self._drawing = True
            for p in self._panels:
                p["line"].setData([x], [y])
        else:
            self._end = (x, y)
            self._drawing = False
            for p in self._panels:
                p["line"].setData([self._start[0], x], [self._start[1], y])
            self.line_drawn.emit(self._start[0], self._start[1], x, y)

    def _on_move(self, event):
        scene_pos = event[0]
        panel = self._panel_at(scene_pos)
        pos = self._map_pos(scene_pos, panel)
        if pos is None or panel is None or panel["data"] is None:
            return
        x, y = pos
        h, w = panel["data"].shape[:2]
        if 0 <= x < w and 0 <= y < h:
            val = float(panel["data"][int(y), int(x)])
            self.cursor_moved.emit(x, y, val)
        if self._line_mode and self._drawing and self._start is not None:
            for p in self._panels:
                p["line"].setData([self._start[0], x], [self._start[1], y])
