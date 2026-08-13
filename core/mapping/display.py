"""Display-time map enhancement: neighborhood filters, spatial binning, intensity scale."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None
    median_filter = None
    uniform_filter = None


SMOOTH_METHODS = ("none", "mean", "median", "gaussian")
INTENSITY_SCALES = ("linear", "sqrt", "asinh", "log")
BIN_FACTORS = (1, 2, 4)


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
