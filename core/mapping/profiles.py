"""Line-profile extraction from 2D element maps and hyperspectral cubes."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

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


def perpendicular_unit(
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[float, float]:
    """Unit vector perpendicular to the transect (rotated 90° CCW)."""
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = float(np.hypot(dx, dy))
    if length < 1e-12:
        return 0.0, 1.0
    return -dy / length, dx / length


def band_offsets(width: int) -> np.ndarray:
    """Pixel offsets along the perpendicular for a band of ``width`` samples.

    width=1 → [0]; width=3 → [-1, 0, 1]; width=2 → [-0.5, 0.5].
    """
    w = max(1, int(width))
    return np.arange(w, dtype=np.float64) - (w - 1) / 2.0


def line_band_edges(
    start: Tuple[float, float],
    end: Tuple[float, float],
    width: int,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Four corners of the sampling band: (a0, a1, b0, b1) along the two edges."""
    half = max(int(width), 1) / 2.0
    px, py = perpendicular_unit(start, end)
    a0 = (start[0] + px * half, start[1] + py * half)
    a1 = (end[0] + px * half, end[1] + py * half)
    b0 = (start[0] - px * half, start[1] - py * half)
    b1 = (end[0] - px * half, end[1] - py * half)
    return a0, a1, b0, b1


def _sample_line_center(
    data: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int,
) -> np.ndarray:
    """Bilinear sample along a line. Coords are (x, y) = (col, row)."""
    h, w = data.shape
    xs = np.linspace(start[0], end[0], n_points)
    ys = np.linspace(start[1], end[1], n_points)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xs - np.floor(xs)
    wy = ys - np.floor(ys)
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


def _sample_line(
    data: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int,
    width: int = 1,
) -> np.ndarray:
    """Bilinear sample along a line, averaging a perpendicular band of ``width`` pixels."""
    center = _sample_line_center(data, start, end, n_points)
    offsets = band_offsets(width)
    if offsets.size <= 1:
        return center
    px, py = perpendicular_unit(start, end)
    acc = np.zeros(n_points, dtype=np.float64)
    for off in offsets:
        s = (start[0] + px * off, start[1] + py * off)
        e = (end[0] + px * off, end[1] + py * off)
        acc += _sample_line_center(data, s, e, n_points)
    return acc / float(offsets.size)


def extract_line_profile(
    element_map: ElementMap,
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int | None = None,
    width: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract intensity along a line on an element map.

    ``width`` averages this many pixels perpendicular to the transect
    (1 = center line only).

    Returns:
        distances (pixel units), intensities
    """
    return extract_array_profile(
        element_map.data, start, end, n_points=n_points, width=width
    )


def extract_array_profile(
    data: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int | None = None,
    width: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a (optionally band-averaged) profile from a 2D array."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("profile data must be 2D")
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if n_points is None:
        n_points = max(2, int(np.ceil(length)) + 1)
    n_points = max(2, int(n_points))
    dist = line_distances(start, end, n_points)
    values = _sample_line(data, start, end, n_points, width=width)
    return dist, values


def extract_multi_element_profiles(
    maps: Sequence[ElementMap],
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int | None = None,
    width: int = 1,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Profile each map along the same transect. Keys are map names."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for m in maps:
        out[m.name] = extract_line_profile(
            m, start, end, n_points=n_points, width=width
        )
    return out


def extract_cube_element_profiles(
    cube,
    rois: Sequence[Tuple[str, float, float]],
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int | None = None,
    width: int = 1,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Profile cube ROI maps along a transect.

    Args:
        cube: SpectrumCube (needs ``roi_map_energy``)
        rois: sequence of (label, e0_kev, e1_kev)
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, e0, e1 in rois:
        data = cube.roi_map_energy(float(e0), float(e1))
        out[str(name)] = extract_array_profile(
            data, start, end, n_points=n_points, width=width
        )
    return out
