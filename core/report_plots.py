"""
Matplotlib figures for the HTML analysis report.

Import matplotlib_config before pyplot so report plots match the rest of
XRFLab. Helpers return PNG bytes (or None if there is nothing to draw).
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

try:
    import matplotlib_config  # noqa: F401
except ImportError:
    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import matplotlib_config  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np


def _png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _as_array(values) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return arr


def plot_spectrum_fit(
    energy,
    counts,
    fitted=None,
    background=None,
) -> Optional[bytes]:
    """Measured spectrum with fitted model and background."""
    e = _as_array(energy)
    y = _as_array(counts)
    if e is None or y is None or e.size != y.size:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(e, y, color="#2c3e50", lw=1.0, label="Measured")
    fit = _as_array(fitted)
    if fit is not None and fit.size == e.size:
        ax.plot(e, fit, color="#c0392b", lw=1.2, label="Fit")
    bg = _as_array(background)
    if bg is not None and bg.size == e.size:
        ax.plot(e, bg, color="#7f8c8d", lw=1.0, ls="--", label="Background")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Spectrum fit")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _png_bytes(fig)


def plot_fit_residuals(energy, residuals) -> Optional[bytes]:
    """Residuals (measured − fitted) versus energy."""
    e = _as_array(energy)
    r = _as_array(residuals)
    if e is None or r is None or e.size != r.size:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.axhline(0.0, color="#c0392b", ls="--", lw=1.0)
    ax.plot(e, r, color="#2c3e50", lw=0.9)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Residual (counts)")
    ax.set_title("Fit residuals")
    fig.tight_layout()
    return _png_bytes(fig)


def plot_fwhm_calibration(calibration, measurements=None) -> Optional[bytes]:
    """FWHM versus energy with model curve and residual panel."""
    if calibration is None or not hasattr(calibration, "predict_fwhm"):
        return None

    energies: List[float] = []
    fwhm_kev: List[float] = []
    for item in measurements or []:
        try:
            energies.append(float(getattr(item, "energy", item["energy"])))
            fwhm_kev.append(float(getattr(item, "fwhm", item["fwhm"])))
        except (TypeError, KeyError, ValueError):
            continue

    er = getattr(calibration, "energy_range", None) or (0.5, 20.0)
    e_min = float(er[0]) if er else 0.5
    e_max = float(er[1]) if er and len(er) > 1 else 20.0
    if energies:
        e_min = min(e_min, min(energies))
        e_max = max(e_max, max(energies))
    if e_max <= e_min:
        e_max = e_min + 1.0
    grid = np.linspace(e_min, e_max, 200)
    try:
        model = np.array([calibration.predict_fwhm(float(x)) for x in grid])
    except Exception:
        return None

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(8.0, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot(grid, model * 1000.0, color="#c0392b", lw=1.6, label="Model")
    if energies:
        ax.scatter(
            energies,
            np.asarray(fwhm_kev) * 1000.0,
            c="#1a2377",
            s=28,
            zorder=3,
            label="Measured",
        )
        pred = np.array([calibration.predict_fwhm(e) for e in energies])
        resid_ev = (np.asarray(fwhm_kev) - pred) * 1000.0
        ax_r.scatter(energies, resid_ev, c="#1a2377", s=28, zorder=3)
    ax_r.axhline(0.0, color="#c0392b", ls="--", lw=1.0)
    ax.set_ylabel("FWHM (eV)")
    ax.set_title("Detector resolution")
    ax.legend(loc="upper left")
    ax_r.set_xlabel("Energy (keV)")
    ax_r.set_ylabel("Residual (eV)")
    fig.tight_layout()
    return _png_bytes(fig)


def plot_predicted_vs_certified(curves: Dict[str, Any]) -> Optional[bytes]:
    """Predicted vs certified wt% for all enabled, fitted curves."""
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    for name, curve in (curves or {}).items():
        if not getattr(curve, "fitted", False) or not getattr(curve, "enabled", True):
            continue
        for point in getattr(curve, "included_points", lambda: [])():
            certified = getattr(point, "concentration", None)
            predicted = getattr(point, "predicted", None)
            if certified is None or predicted is None:
                continue
            xs.append(float(certified))
            ys.append(float(predicted))
            labels.append(str(getattr(curve, "element", name)))
    if not xs:
        return None
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#c0392b", ls="--", lw=1.0, label="1:1")
    ax.scatter(xs, ys, c="#1a2377", s=32, zorder=3)
    ax.set_xlabel("Certified (wt%)")
    ax.set_ylabel("Predicted (wt%)")
    ax.set_title("Standards: predicted vs certified")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _png_bytes(fig)


def plot_calibration_curves(
    curves: Dict[str, Any],
    *,
    max_n: int = 12,
) -> Optional[bytes]:
    """Small-multiples of intensity vs certified wt% with the fitted curve."""
    items: List[Any] = []
    for curve in (curves or {}).values():
        if not getattr(curve, "fitted", False) or not getattr(curve, "enabled", True):
            continue
        pts = list(getattr(curve, "included_points", lambda: [])())
        if pts:
            items.append(curve)
    if not items:
        return None
    items = items[: max(1, int(max_n))]
    n = len(items)
    ncols = 3 if n > 2 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False,
    )
    for idx, curve in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        pts = list(curve.included_points())
        intensities = [float(p.intensity) for p in pts]
        conc = [float(p.concentration) for p in pts]
        ax.scatter(intensities, conc, c="#1a2377", s=22, zorder=3)
        if intensities:
            grid = np.linspace(min(intensities), max(intensities), 80)
            if hasattr(curve, "evaluate"):
                ax.plot(grid, [curve.evaluate(x) for x in grid], color="#c0392b", lw=1.3)
        ax.set_title(f"{curve.element} {getattr(curve, 'line_group', '')}".strip())
        ax.set_xlabel("Intensity (cps)")
        ax.set_ylabel("Certified (wt%)")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.tight_layout()
    return _png_bytes(fig)


def plot_composition_bars(
    series: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    *,
    ylabel: str = "wt%",
    title: str = "Composition",
) -> Optional[bytes]:
    """Grouped bar chart of sample (or spectrum) compositions.

    ``series`` is a list of ``{"name": str, "values": {key: float}}``.
    """
    keys = [str(k) for k in keys if k]
    if not series or not keys:
        return None
    names = [str(item.get("name") or "sample") for item in series]
    n_samp = len(names)
    n_key = len(keys)
    x = np.arange(n_samp)
    width = 0.8 / max(n_key, 1)
    fig_w = max(8.0, min(14.0, 1.4 * n_samp + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, 4.6))
    for i, key in enumerate(keys):
        vals = [
            float((item.get("values") or {}).get(key, 0.0) or 0.0)
            for item in series
        ]
        ax.bar(x + (i - (n_key - 1) / 2.0) * width, vals, width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if n_key <= 16:
        ax.legend(ncols=min(n_key, 4), fontsize=8)
    fig.tight_layout()
    return _png_bytes(fig)
