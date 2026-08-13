"""Display-time map enhancement: neighborhood filters, spatial binning, intensity scale."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.ndimage import gaussian_filter, median_filter, uniform_filter, zoom
except ImportError:  # pragma: no cover
    gaussian_filter = None
    median_filter = None
    uniform_filter = None
    zoom = None


SMOOTH_METHODS = ("none", "mean", "median", "gaussian")
INTENSITY_SCALES = ("linear", "sqrt", "asinh", "log")
BIN_FACTORS = (1, 2, 4)
INTERP_METHODS = ("none", "nearest", "bilinear", "cubic", "quintic")
INTERP_ORDERS = {
    "nearest": 0,
    "bilinear": 1,
    "cubic": 3,
    "quintic": 5,
}


def odd_kernel(size: int) -> int:
    """Clamp to a positive odd kernel (1, 3, 5, …)."""
    n = max(1, int(size))
    if n % 2 == 0:
        n += 1
    return n


def enhance_map(
    data: np.ndarray,
    *,
    smooth: str = "none",
    neighborhood: int = 1,
    bin_factor: int = 1,
    scale: str = "linear",
) -> np.ndarray:
    """
    Return a same-shape copy of ``data`` after optional smooth, block-bin, and scale.

    Spatial binning averages each N×N block, then expands back so map coordinates
    (pick pixel, line tools) stay aligned with the original grid.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("enhance_map expects a 2D array")
    method = (smooth or "none").lower()
    if method not in SMOOTH_METHODS:
        raise ValueError(f"Unknown smooth method: {smooth}")
    scale_name = (scale or "linear").lower()
    if scale_name not in INTENSITY_SCALES:
        raise ValueError(f"Unknown intensity scale: {scale}")
    factor = max(1, int(bin_factor))
    kernel = odd_kernel(neighborhood)

    if method != "none" and kernel > 1:
        arr = _smooth(arr, method, kernel)
    if factor > 1:
        arr = block_bin(arr, factor)
    if scale_name != "linear":
        arr = apply_intensity_scale(arr, scale_name)
    return arr


def upsample_map(
    data: np.ndarray,
    *,
    factor: int = 1,
    method: str = "cubic",
) -> np.ndarray:
    """
    Magnify a map for display (2D or HxWxC).

    Cubic / quintic use spline interpolation (scipy.ndimage.zoom).
    Coordinates stay with the caller — this only changes pixel density.
    """
    arr = np.asarray(data, dtype=np.float64)
    factor = max(1, int(factor))
    name = (method or "none").lower()
    if name not in INTERP_METHODS:
        raise ValueError(f"Unknown interpolation method: {method}")
    if factor == 1 or name == "none":
        return arr
    order = INTERP_ORDERS.get(name, 3)
    if zoom is not None:
        if arr.ndim == 2:
            return zoom(arr, factor, order=order, mode="nearest")
        if arr.ndim == 3:
            z = (float(factor), float(factor)) + (1.0,) * (arr.ndim - 2)
            return zoom(arr, z, order=order, mode="nearest")
        raise ValueError("upsample_map expects 2D or 3D data")
    return _upsample_numpy(arr, factor)


def _upsample_numpy(arr: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor fallback when SciPy is unavailable."""
    if arr.ndim == 2:
        return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)


def _smooth(arr: np.ndarray, method: str, kernel: int) -> np.ndarray:
    if method == "mean":
        if uniform_filter is None:
            return _box_mean_numpy(arr, kernel)
        return uniform_filter(arr, size=kernel, mode="nearest")
    if method == "median":
        if median_filter is None:
            return _box_mean_numpy(arr, kernel)
        return median_filter(arr, size=kernel, mode="nearest")
    if method == "gaussian":
        sigma = max(0.4, (kernel - 1) / 4.0)
        if gaussian_filter is None:
            return _box_mean_numpy(arr, kernel)
        return gaussian_filter(arr, sigma=sigma, mode="nearest")
    return arr


def _box_mean_numpy(arr: np.ndarray, kernel: int) -> np.ndarray:
    """Fallback boxcar if SciPy is unavailable."""
    pad = kernel // 2
    padded = np.pad(arr, pad, mode="edge")
    window = np.ones((kernel, kernel), dtype=np.float64) / (kernel * kernel)
    from numpy.lib.stride_tricks import sliding_window_view

    view = sliding_window_view(padded, (kernel, kernel))
    return np.tensordot(view, window, axes=([2, 3], [0, 1]))


def block_bin(data: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping factor×factor blocks and expand back to input shape."""
    arr = np.asarray(data, dtype=np.float64)
    factor = max(1, int(factor))
    if factor == 1:
        return arr.copy()
    h, w = arr.shape
    nh, nw = h // factor, w // factor
    if nh == 0 or nw == 0:
        return arr.copy()
    trimmed = arr[: nh * factor, : nw * factor]
    binned = trimmed.reshape(nh, factor, nw, factor).mean(axis=(1, 3))
    expanded = np.repeat(np.repeat(binned, factor, axis=0), factor, axis=1)
    if expanded.shape != arr.shape:
        out = np.empty_like(arr)
        out[: expanded.shape[0], : expanded.shape[1]] = expanded
        if expanded.shape[0] < h:
            out[expanded.shape[0] :, : expanded.shape[1]] = expanded[-1]
        if expanded.shape[1] < w:
            out[:, expanded.shape[1] :] = out[:, expanded.shape[1] - 1 : expanded.shape[1]]
        return out
    return expanded


def apply_intensity_scale(data: np.ndarray, scale: str) -> np.ndarray:
    """Compress dynamic range so weak features are visible next to hot pixels."""
    arr = np.asarray(data, dtype=np.float64)
    name = (scale or "linear").lower()
    if name == "linear":
        return arr
    if name == "sqrt":
        return np.sqrt(np.clip(arr, 0.0, None))
    if name == "asinh":
        return np.arcsinh(np.clip(arr, 0.0, None))
    if name == "log":
        return np.log1p(np.clip(arr, 0.0, None))
    raise ValueError(f"Unknown intensity scale: {scale}")


def resize_to(
    data: np.ndarray,
    height: int,
    width: int,
    *,
    order: int = 1,
) -> np.ndarray:
    """Resize 2D or HxWxC array to (height, width) keeping channels."""
    arr = np.asarray(data, dtype=np.float64)
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError("resize_to requires positive height and width")
    if arr.ndim == 2:
        if arr.shape == (height, width):
            return arr.copy()
        zh, zw = height / arr.shape[0], width / arr.shape[1]
        if zoom is not None:
            return zoom(arr, (zh, zw), order=order, mode="nearest")
        return _resize_numpy(arr, height, width)
    if arr.ndim == 3:
        if arr.shape[0] == height and arr.shape[1] == width:
            return arr.copy()
        zh, zw = height / arr.shape[0], width / arr.shape[1]
        if zoom is not None:
            z = (zh, zw) + (1.0,) * (arr.ndim - 2)
            return zoom(arr, z, order=order, mode="nearest")
        out = np.empty((height, width, arr.shape[2]), dtype=np.float64)
        for c in range(arr.shape[2]):
            out[:, :, c] = _resize_numpy(arr[:, :, c], height, width)
        return out
    raise ValueError("resize_to expects 2D or 3D data")


def _resize_numpy(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize when SciPy is unavailable."""
    ys = np.linspace(0, arr.shape[0] - 1, height)
    xs = np.linspace(0, arr.shape[1] - 1, width)
    yi = np.clip(np.round(ys).astype(int), 0, arr.shape[0] - 1)
    xi = np.clip(np.round(xs).astype(int), 0, arr.shape[1] - 1)
    return arr[yi[:, None], xi[None, :]]


def _lut_hot(n: int = 256) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack(
        [np.clip(x * 3.0, 0, 1), np.clip(x * 3.0 - 1.0, 0, 1), np.clip(x * 3.0 - 2.0, 0, 1)]
    )


def _lut_inferno(n: int = 256) -> np.ndarray:
    """Compact inferno-like ramp (black → purple → orange → yellow)."""
    x = np.linspace(0.0, 1.0, n)
    r = np.clip(1.4 * x - 0.15 * np.sin(x * np.pi), 0, 1)
    g = np.clip(x * x * 1.1, 0, 1)
    b = np.clip(0.55 * np.sin(x * np.pi) + 0.15 * (1.0 - x), 0, 1)
    return np.column_stack([r, g, b])


def _lut_cyan(n: int = 256) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack(
        [0.05 * x, 0.65 * x + 0.1 * x * x, np.clip(0.3 + 0.7 * x, 0, 1)]
    )


COLORMAPS = {
    "hot": _lut_hot,
    "inferno": _lut_inferno,
    "cyan": _lut_cyan,
}


def colorize_map(
    data: np.ndarray,
    *,
    percentile: tuple = (2.0, 98.0),
    cmap: str = "hot",
) -> tuple:
    """
    Map a 2D intensity array to RGB in [0, 1] plus an alpha mask in [0, 1].

    Alpha follows scaled intensity so empty pixels stay transparent.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("colorize_map expects a 2D array")
    lut_fn = COLORMAPS.get((cmap or "hot").lower(), _lut_hot)
    lut = lut_fn(256)
    lo, hi = np.percentile(arr, percentile)
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    idx = np.clip((scaled * 255).astype(np.int32), 0, 255)
    rgb = lut[idx]
    return rgb, scaled


def overlay_on_photo(
    photo: np.ndarray,
    overlay_rgb: np.ndarray,
    *,
    alpha: Optional[np.ndarray] = None,
    opacity: float = 0.45,
) -> np.ndarray:
    """
    Alpha-blend an RGB overlay onto an optical photo.

    The overlay is resized to the photo. ``opacity`` is 0–1. If ``alpha`` is
    given (0–1, same shape as the overlay before resize), low-count pixels
    stay more transparent. Returns HxWx3 float in [0, 1].
    """
    photo_arr = np.asarray(photo, dtype=np.float64)
    if photo_arr.ndim == 2:
        photo_arr = np.repeat(photo_arr[:, :, None], 3, axis=2)
    photo_arr = photo_arr[:, :, :3]
    if photo_arr.max(initial=0.0) > 1.5:
        photo_arr = photo_arr / 255.0
    photo_arr = np.clip(photo_arr, 0.0, 1.0)
    h, w = photo_arr.shape[:2]

    over = np.asarray(overlay_rgb, dtype=np.float64)
    if over.ndim != 3 or over.shape[2] < 3:
        raise ValueError("overlay_rgb must be HxWx3")
    over = over[:, :, :3]
    if over.max(initial=0.0) > 1.5:
        over = over / 255.0
    over = np.clip(over, 0.0, 1.0)
    if over.shape[0] != h or over.shape[1] != w:
        over = resize_to(over, h, w, order=1)

    opac = float(np.clip(opacity, 0.0, 1.0))
    if alpha is None:
        a = np.full((h, w), opac, dtype=np.float64)
    else:
        a = np.asarray(alpha, dtype=np.float64)
        if a.shape != (h, w):
            a = resize_to(a, h, w, order=1)
        a = np.clip(a, 0.0, 1.0) * opac
    a3 = a[:, :, None]
    return np.clip(photo_arr * (1.0 - a3) + over * a3, 0.0, 1.0)


def format_acquisition(meta: Optional[dict]) -> str:
    """One- or two-line summary of map live time, dwell, and tube settings."""
    if not meta:
        return ""
    parts: list[str] = []
    live = meta.get("map_live_time_s")
    dwell_ms = meta.get("dwell_ms")
    n_pix = meta.get("n_pixels")
    if live:
        line = f"Map live {live:g} s"
        if dwell_ms is not None:
            if dwell_ms >= 10:
                line += f" · ~{dwell_ms:.1f} ms/pixel"
            elif dwell_ms >= 1:
                line += f" · ~{dwell_ms:.2f} ms/pixel"
            else:
                line += f" · ~{dwell_ms:.2f} ms/pixel"
        if n_pix:
            line += f" ({int(n_pix)} px)"
        parts.append(line)
    tube = []
    kv = meta.get("kv")
    ma = meta.get("ma")
    if kv:
        tube.append(f"{kv:g} kV")
    if ma:
        tube.append(f"{ma:g} mA")
    acquired = meta.get("acquired_at")
    if acquired:
        tube.append(str(acquired)[:10])
    if tube:
        parts.append(" · ".join(tube))
    return "\n".join(parts)
