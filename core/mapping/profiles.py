"""Line-profile extraction from 2D element maps."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from core.mapping.models import ElementMap


def line_distances(
    start: Tuple[float, float],
    end: Tuple[float, float],
    n: int,
) -> np.ndarray:
    """Distance along the transect for n samples (pixel units)."""
    if n <= 0:
        return np.array([])
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if n == 1:
        return np.array([0.0])
    return np.linspace(0.0, length, n)


def _sample_line(
    data: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int,
) -> np.ndarray:
    """Bilinear sample along a line. Coords are (x, y) = (col, row)."""
    h, w = data.shape
    xs = np.linspace(start[0], end[0], n_points)
    ys = np.linspace(start[1], end[1], n_points)
    # Clip to valid range for interpolation
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xs - np.floor(xs)
    wy = ys - np.floor(ys)
    # Keep weights 0 outside
    wx = np.where((xs >= 0) & (xs <= w - 1), wx, 0.0)
    wy = np.where((ys >= 0) & (ys <= h - 1), wy, 0.0)
    v00 = data[y0, x0].astype(np.float64)
    v10 = data[y0, x1].astype(np.float64)
    v01 = data[y1, x0].astype(np.float64)
    v11 = data[y1, x1].astype(np.float64)
    return (
        v00 * (1 - wx) * (1 - wy)
        + v10 * wx * (1 - wy)
        + v01 * (1 - wx) * wy
        + v11 * wx * wy
    )


def extract_line_profile(
    element_map: ElementMap,
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract intensity along a line on an element map.

    Returns:
        distances (pixel units), intensities
    """
    h, w = element_map.shape
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if n_points is None:
        n_points = max(2, int(np.ceil(length)) + 1)
    n_points = max(2, int(n_points))
    dist = line_distances(start, end, n_points)
    values = _sample_line(element_map.data, start, end, n_points)
    return dist, values


def extract_multi_element_profiles(
    maps: Sequence[ElementMap],
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Profile each map along the same transect. Keys are map names."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for m in maps:
        out[m.name] = extract_line_profile(m, start, end, n_points=n_points)
    return out
