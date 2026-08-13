"""Interactive map canvas with single or multi-panel element-map display."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsPolygonItem,
    QVBoxLayout,
    QWidget,
)

from core.mapping.profiles import line_band_edges
from core.mapping.regions import circle_outline, polygon_outline, rect_outline


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
    view_clicked = Signal(float, float)  # idle click (not line/pick/region)
    region_drawn = Signal(str, object)  # kind, params

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
        self._region_mode: Optional[str] = None  # rect | circle | poly
        self._region_points: List[Tuple[float, float]] = []
        self._region_outline: Optional[Tuple[list, list]] = None
        self._start: Optional[Tuple[float, float]] = None
        self._end: Optional[Tuple[float, float]] = None
        self._band_width: int = 1
        self._rgb_mode = False
        self._click_proxy = None
        self._move_proxy = None
        self._series_xy: Optional[Tuple[list, list]] = None
        self._series_highlight: Optional[int] = None

        self._build_single_panel()
        self._connect_scene()

    # --------------------------------------------------------------- modes
    def set_line_mode(self, enabled: bool) -> None:
        self._line_mode = bool(enabled)
        if enabled:
            self._pick_mode = False
            self._region_mode = None
            self._region_points = []
        if not enabled:
            self._drawing = False
            self._start = None

    def set_pick_mode(self, enabled: bool) -> None:
        self._pick_mode = bool(enabled)
        if enabled:
            self._line_mode = False
            self._region_mode = None
            self._region_points = []
            self._drawing = False

    def set_region_mode(self, mode: Optional[str]) -> None:
        """mode is 'rect', 'circle', 'poly', or None to stop drawing."""
        self._region_mode = mode if mode else None
        self._region_points = []
        self._drawing = False
        if self._region_mode:
            self._line_mode = False
            self._pick_mode = False

    def clear_region(self) -> None:
        self._region_points = []
        self._region_outline = None
        self._drawing = False
        for p in self._panels:
            if "region" in p:
                p["region"].setData([], [])

    def set_region_outline(self, xs, ys) -> None:
        self._region_outline = (list(xs), list(ys))
        for p in self._panels:
            if "region" in p:
                p["region"].setData(list(xs), list(ys))

    def clear_line(self) -> None:
        self._start = None
        self._end = None
        for p in self._panels:
            p["line"].setData([], [])
            self._set_panel_band(p, None)

    def set_band_width(self, width: int) -> None:
        """Perpendicular averaging width in pixels (1 = center line only)."""
        self._band_width = max(1, int(width))
        self._restore_overlays()

    def set_spot_markers(self, xs, ys) -> None:
        for p in self._panels:
            if xs is None or len(xs) == 0:
                p["spot"].setData([])
            else:
                p["spot"].setData(x=list(xs), y=list(ys))

    def set_series_markers(
        self,
        xs,
        ys,
        *,
        highlight: Optional[int] = None,
        connect: bool = True,
    ) -> None:
        """Overlay a collected line / multipoint series in image coordinates."""
        if xs is None or ys is None or len(xs) == 0:
            self.clear_series_markers()
            return
        self._series_xy = (list(xs), list(ys))
        self._series_highlight = highlight
        self._apply_series_markers(connect=connect)

    def set_series_highlight(self, index: Optional[int]) -> None:
        self._series_highlight = None if index is None else int(index)
        self._apply_series_markers()

    def clear_series_markers(self) -> None:
        self._series_xy = None
        self._series_highlight = None
        for p in self._panels:
            if "series" in p:
                p["series"].setData([])
            if "series_path" in p:
                p["series_path"].setData([], [])
            if "series_hi" in p:
                p["series_hi"].setData([])

    def _apply_series_markers(self, *, connect: bool = True) -> None:
        if self._series_xy is None:
            self.clear_series_markers()
            return
        xs, ys = self._series_xy
        hi = self._series_highlight
        for p in self._panels:
            if "series_path" in p:
                if connect and len(xs) >= 2:
                    p["series_path"].setData(xs, ys)
                else:
                    p["series_path"].setData([], [])
            if "series" in p:
                p["series"].setData(x=xs, y=ys)
            if "series_hi" in p:
                if hi is not None and 0 <= hi < len(xs):
                    p["series_hi"].setData(x=[xs[hi]], y=[ys[hi]])
                else:
                    p["series_hi"].setData([])

    def set_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width: Optional[int] = None,
    ) -> None:
        self._start = (x0, y0)
        self._end = (x1, y1)
        if width is not None:
            self._band_width = max(1, int(width))
        pen = self._transect_pen()
        for p in self._panels:
            p["line"].setPen(pen)
            p["line"].setData([x0, x1], [y0, y1])
            self._set_panel_band(p, (x0, y0, x1, y1))

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
        display: Optional[np.ndarray] = None,
    ) -> None:
        """Single-panel mode (overview, RGB, or one map).

        ``data`` defines the click/line coordinate grid. ``display`` may be a
        finer interpolated image; it is stretched to the original map size.
        """
        self._rgb_mode = rgb
        self._ensure_panel_count(1)
        panel = self._panels[0]
        arr = np.asarray(data)
        shown = np.asarray(display) if display is not None else arr
        if rgb:
            if shown.ndim != 3 or shown.shape[2] < 3:
                raise ValueError("RGB image must be HxWx3")
            disp = shown[:, :, :3]
            maxv = float(np.nanmax(disp)) if disp.size else 1.0
            if maxv > 1.5:
                disp = disp.astype(np.float64) / 255.0
            else:
                disp = disp.astype(np.float64)
            disp = np.clip(disp, 0.0, 1.0)
            if arr.ndim >= 3:
                self._primary_data = np.asarray(arr[:, :, 0], dtype=np.float64)
            else:
                self._primary_data = np.asarray(arr, dtype=np.float64)
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
            shown2 = np.asarray(shown, dtype=np.float64)
            levels = _auto_levels(self._primary_data) if auto_levels else None
            panel["image"].setImage(
                np.ascontiguousarray(shown2),
                autoLevels=levels is None,
                levels=levels,
                axisOrder="row-major",
            )
        panel["title"] = title or ""
        panel["label"].setText(panel["title"])
        h, w = self._primary_data.shape[:2]
        self._fit_image_rect(panel, w, h)
        panel["view"].setRange(QRectF(0, 0, w, h), padding=0.02)
        self._restore_overlays()

    def set_images(
        self,
        panels: Sequence[tuple],
        *,
        auto_levels: bool = True,
    ) -> None:
        """
        Multi-panel subplot mode.

        Args:
            panels: (title, HxW array) or (title, data, display)
        """
        items = []
        for item in panels:
            if item is None:
                continue
            title = str(item[0])
            data = np.asarray(item[1], dtype=np.float64)
            shown = np.asarray(item[2], dtype=np.float64) if len(item) > 2 else data
            items.append((title, data, shown))
        if not items:
            return
        self._rgb_mode = False
        self._ensure_panel_count(len(items))
        self._primary_data = items[0][1]

        for panel, (title, data, shown) in zip(self._panels, items):
            panel["title"] = title
            panel["data"] = data
            panel["label"].setText(title)
            levels = _auto_levels(data) if auto_levels else None
            panel["image"].setImage(
                np.ascontiguousarray(shown),
                autoLevels=levels is None,
                levels=levels,
                axisOrder="row-major",
            )
            h, w = data.shape[:2]
            self._fit_image_rect(panel, w, h)
            panel["view"].setRange(QRectF(0, 0, w, h), padding=0.02)

        # Link all viewboxes to the first
        master = self._panels[0]["view"]
        for panel in self._panels[1:]:
            panel["view"].setXLink(master)
            panel["view"].setYLink(master)

        self._restore_overlays()

    def _fit_image_rect(self, panel: dict, width: int, height: int) -> None:
        """Stretch the (possibly upsampled) image to original map coordinates."""
        img = panel["image"]
        img.resetTransform()
        if hasattr(img, "setRect"):
            img.setRect(QRectF(0.0, 0.0, float(width), float(height)))

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
        dash = pg.mkPen("#ffcc00", width=1.5, style=Qt.DashLine)
        band_a = pg.PlotDataItem(pen=dash)
        band_b = pg.PlotDataItem(pen=dash)
        view.addItem(band_a)
        view.addItem(band_b)
        band_poly = QGraphicsPolygonItem()
        band_poly.setBrush(QBrush(QColor(255, 204, 0, 70)))
        band_poly.setPen(QPen(QColor(255, 180, 0, 210), 0))
        band_poly.setZValue(10)
        view.addItem(band_poly)
        spot = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush(0, 200, 255, 180),
            pen=pg.mkPen(None),
        )
        view.addItem(spot)
        series_path = pg.PlotDataItem(pen=pg.mkPen("#00e5ff", width=1.5))
        series_path.setZValue(20)
        view.addItem(series_path)
        series = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush(0, 220, 255, 210),
            pen=pg.mkPen("#003344", width=0.8),
        )
        series.setZValue(21)
        view.addItem(series)
        series_hi = pg.ScatterPlotItem(
            size=18,
            brush=pg.mkBrush(255, 220, 40, 240),
            pen=pg.mkPen("#ffffff", width=2),
        )
        series_hi.setZValue(22)
        view.addItem(series_hi)
        region = pg.PlotDataItem(
            pen=pg.mkPen("#00e5ff", width=2),
            fillLevel=None,
        )
        view.addItem(region)
        return {
            "view": view,
            "image": image,
            "label": label,
            "line": line,
            "band_a": band_a,
            "band_b": band_b,
            "band_poly": band_poly,
            "spot": spot,
            "series": series,
            "series_path": series_path,
            "series_hi": series_hi,
            "region": region,
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

    def _transect_pen(self) -> QPen:
        """Center-line stroke; grows a little so width changes are obvious."""
        if self._band_width <= 1:
            w = 2.0
        else:
            w = min(2.0 + 0.4 * float(self._band_width), 12.0)
        return pg.mkPen("#ffcc00", width=w)

    def _set_panel_band(
        self, panel: dict, line: Optional[Tuple[float, float, float, float]]
    ) -> None:
        poly_item = panel.get("band_poly")
        if (
            line is None
            or self._band_width <= 1
            or "band_a" not in panel
        ):
            if "band_a" in panel:
                panel["band_a"].setData([], [])
                panel["band_b"].setData([], [])
            if poly_item is not None:
                poly_item.setPolygon(QPolygonF())
            return
        x0, y0, x1, y1 = line
        a0, a1, b0, b1 = line_band_edges((x0, y0), (x1, y1), self._band_width)
        panel["band_a"].setData([a0[0], a1[0]], [a0[1], a1[1]])
        panel["band_b"].setData([b0[0], b1[0]], [b0[1], b1[1]])
        if poly_item is not None:
            poly_item.setPolygon(
                QPolygonF(
                    [
                        QPointF(a0[0], a0[1]),
                        QPointF(a1[0], a1[1]),
                        QPointF(b1[0], b1[1]),
                        QPointF(b0[0], b0[1]),
                    ]
                )
            )

    def _restore_overlays(self) -> None:
        if self._start is not None and self._end is not None:
            self.set_line(*self._start, *self._end)
        elif self._start is not None:
            for p in self._panels:
                p["line"].setData([self._start[0]], [self._start[1]])
                self._set_panel_band(p, None)
        if self._region_outline is not None:
            self.set_region_outline(*self._region_outline)
        if self._series_xy is not None:
            self._apply_series_markers()

    def _local_to_data(self, panel: dict, x: float, y: float) -> Tuple[float, float]:
        """Convert ImageItem local pixels to original map coordinates."""
        data = panel.get("data")
        image = panel["image"].image
        if data is None or image is None:
            return x, y
        ih, iw = image.shape[:2]
        dh, dw = data.shape[:2]
        if iw > 0 and ih > 0 and (iw != dw or ih != dh):
            x = x * dw / iw
            y = y * dh / ih
        return x, y

    def _panel_at(self, scene_pos) -> Optional[dict]:
        for panel in self._panels:
            try:
                p = panel["image"].mapFromScene(scene_pos)
                x, y = self._local_to_data(panel, float(p.x()), float(p.y()))
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
            return self._local_to_data(panel, float(p.x()), float(p.y()))
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

        if self._region_mode:
            self._on_region_click(ev, x, y)
            return

        if not self._line_mode:
            self.view_clicked.emit(x, y)
            return
        if not self._drawing or self._start is None:
            self._start = (x, y)
            self._end = None
            self._drawing = True
            pen = self._transect_pen()
            for p in self._panels:
                p["line"].setPen(pen)
                p["line"].setData([x], [y])
        else:
            self._end = (x, y)
            self._drawing = False
            pen = self._transect_pen()
            for p in self._panels:
                p["line"].setPen(pen)
                p["line"].setData([self._start[0], x], [self._start[1], y])
                self._set_panel_band(p, (self._start[0], self._start[1], x, y))
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
            pen = self._transect_pen()
            for p in self._panels:
                p["line"].setPen(pen)
                p["line"].setData([self._start[0], x], [self._start[1], y])
                self._set_panel_band(p, (self._start[0], self._start[1], x, y))
        if self._region_mode:
            self._preview_region(x, y)

    def _shift_held(self, ev=None) -> bool:
        if ev is not None and hasattr(ev, "modifiers"):
            return bool(ev.modifiers() & Qt.ShiftModifier)
        app = QApplication.instance()
        if app is not None:
            return bool(app.keyboardModifiers() & Qt.ShiftModifier)
        return False

    def _square_corner(
        self, x0: float, y0: float, x: float, y: float, ev=None
    ) -> Tuple[float, float]:
        if not self._shift_held(ev):
            return x, y
        dx, dy = x - x0, y - y0
        side = max(abs(dx), abs(dy))
        if side <= 0:
            return x, y
        return x0 + (side if dx >= 0 else -side), y0 + (side if dy >= 0 else -side)

    def _on_region_click(self, ev, x: float, y: float) -> None:
        button = ev.button() if hasattr(ev, "button") else Qt.LeftButton
        double = bool(ev.double()) if hasattr(ev, "double") else False
        mode = self._region_mode
        if button == Qt.RightButton:
            if mode == "poly" and len(self._region_points) >= 3:
                self._finish_region("poly", list(self._region_points))
            else:
                self._region_points = []
                self._drawing = False
                if self._region_outline is not None:
                    self.set_region_outline(*self._region_outline)
                else:
                    self.set_region_outline([], [])
            return

        if button != Qt.LeftButton:
            return

        if mode == "poly":
            if double and len(self._region_points) >= 3:
                self._finish_region("poly", list(self._region_points))
                return
            if (
                len(self._region_points) >= 3
                and abs(x - self._region_points[0][0]) <= 1.5
                and abs(y - self._region_points[0][1]) <= 1.5
            ):
                self._finish_region("poly", list(self._region_points))
                return
            if not double:
                self._region_points.append((x, y))
                self._drawing = True
                self._preview_region(x, y, ev)
            return

        # rect / circle: two clicks
        if not self._region_points:
            self._region_points = [(x, y)]
            self._drawing = True
            self._preview_region(x, y, ev)
            return
        x0, y0 = self._region_points[0]
        if mode == "rect":
            x, y = self._square_corner(x0, y0, x, y, ev)
            self._finish_region("rect", (x0, y0, x, y))
        else:
            radius = float(np.hypot(x - x0, y - y0))
            self._finish_region("circle", (x0, y0, radius))

    def _preview_region(self, x: float, y: float, ev=None) -> None:
        mode = self._region_mode
        if not mode or not self._region_points:
            return
        if mode == "rect":
            x0, y0 = self._region_points[0]
            x, y = self._square_corner(x0, y0, x, y, ev)
            xs, ys = rect_outline(x0, y0, x, y)
        elif mode == "circle":
            x0, y0 = self._region_points[0]
            radius = float(np.hypot(x - x0, y - y0))
            xs, ys = circle_outline(x0, y0, radius)
        else:
            pts = list(self._region_points) + [(x, y)]
            xs, ys = polygon_outline(pts)
        for p in self._panels:
            if "region" in p:
                p["region"].setData(list(xs), list(ys))

    def _finish_region(self, kind: str, params) -> None:
        from core.mapping.regions import region_outline

        xs, ys = region_outline(kind, params)
        self._region_outline = (list(xs), list(ys))
        self._region_points = []
        self._drawing = False
        self.set_region_outline(xs, ys)
        self.region_drawn.emit(kind, params)
