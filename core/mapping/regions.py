"""Spatial ROI masks for summing a hyperspectral cube over an area."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

Vertex = Tuple[float, float]


def pixel_centers(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Column (x) and row (y) coordinates of pixel centers."""
    yy, xx = np.mgrid[0:height, 0:width]
    return xx.astype(np.float64) + 0.5, yy.astype(np.float64) + 0.5


def rect_mask(
    height: int,
    width: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    """Boolean (H, W) mask of pixels whose centers lie in the axis-aligned box."""
    xa, xb = (float(x0), float(x1)) if x0 <= x1 else (float(x1), float(x0))
    ya, yb = (float(y0), float(y1)) if y0 <= y1 else (float(y1), float(y0))
    xx, yy = pixel_centers(height, width)
    return (xx >= xa) & (xx <= xb) & (yy >= ya) & (yy <= yb)


def circle_mask(
    height: int,
    width: int,
    cx: float,
    cy: float,
    radius: float,
) -> np.ndarray:
    """Boolean (H, W) mask of pixels whose centers lie in the circle."""
    r = max(0.0, float(radius))
    xx, yy = pixel_centers(height, width)
    return (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 <= r * r


def polygon_mask(
    height: int,
    width: int,
    vertices: Sequence[Vertex],
) -> np.ndarray:
    """Boolean (H, W) mask using even-odd fill of the polygon."""
    verts = [(float(x), float(y)) for x, y in vertices]
    if len(verts) < 3:
        return np.zeros((height, width), dtype=bool)
    from matplotlib.path import Path

    if verts[0] != verts[-1]:
        closed = verts + [verts[0]]
    else:
        closed = verts
    path = Path(closed, closed=True)
    xx, yy = pixel_centers(height, width)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return path.contains_points(pts).reshape(height, width)


def region_mask(
    height: int,
    width: int,
    kind: str,
    params,
) -> np.ndarray:
    """Dispatch mask builder. ``params`` matches canvas ``region_drawn`` payload."""
    kind = (kind or "").lower()
    if kind in ("rect", "rectangle", "square"):
        x0, y0, x1, y1 = params
        return rect_mask(height, width, x0, y0, x1, y1)
    if kind in ("circle", "disk"):
        cx, cy, radius = params
        return circle_mask(height, width, cx, cy, radius)
    if kind in ("poly", "polygon"):
        return polygon_mask(height, width, params)
    raise ValueError(f"Unknown region kind: {kind}")


def rect_outline(x0: float, y0: float, x1: float, y1: float) -> Tuple[list, list]:
    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    return xs, ys


def circle_outline(
    cx: float, cy: float, radius: float, n: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * np.pi, n + 1)
    r = max(0.0, float(radius))
    return cx + r * np.cos(t), cy + r * np.sin(t)


def polygon_outline(vertices: Sequence[Vertex]) -> Tuple[list, list]:
    if not vertices:
        return [], []
    xs = [float(p[0]) for p in vertices]
    ys = [float(p[1]) for p in vertices]
    if xs[0] != xs[-1] or ys[0] != ys[-1]:
        xs.append(xs[0])
        ys.append(ys[0])
    return xs, ys


def region_outline(kind: str, params) -> Tuple[list, list]:
    kind = (kind or "").lower()
    if kind in ("rect", "rectangle", "square"):
        x0, y0, x1, y1 = params
        return rect_outline(x0, y0, x1, y1)
    if kind in ("circle", "disk"):
        cx, cy, radius = params
        xs, ys = circle_outline(cx, cy, radius)
        return list(xs), list(ys)
    if kind in ("poly", "polygon"):
        return polygon_outline(params)
    return [], []


def region_label(kind: str, params, n_pixels: int) -> str:
    kind = (kind or "").lower()
    if kind in ("rect", "rectangle", "square"):
        x0, y0, x1, y1 = params
        return (
            f"Rect sum ({x0:.0f},{y0:.0f})–({x1:.0f},{y1:.0f}) "
            f"[{n_pixels} px]"
        )
    if kind in ("circle", "disk"):
        cx, cy, radius = params
        return f"Circle sum r={radius:.1f} @ ({cx:.0f},{cy:.0f}) [{n_pixels} px]"
    if kind in ("poly", "polygon"):
        n_vert = len(params)
        return f"Polygon sum ({n_vert} vertices, {n_pixels} px)"
    return f"Area sum [{n_pixels} px]"
