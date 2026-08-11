"""
Shared matplotlib configuration for XRFLab offline/CLI plots.

Import this module before creating figures in calibration scripts and examples:
    import matplotlib_config  # noqa: F401
    import matplotlib.pyplot as plt
"""

from __future__ import annotations

import matplotlib as mpl

# Readable defaults for scientific spectrum plots
mpl.rcParams.update(
    {
        "figure.figsize": (8.0, 5.5),
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
