"""Grouped (shared-amplitude, fixed-ratio) line fitting."""

from __future__ import annotations

import numpy as np
import pytest

from core.fitting import SpectrumFitter
from core.line_groups import (
    build_line_groups,
    fit_grouped,
    global_shape_bounds,
    shape_params_from_global,
    subshell_of,
)
from core.peak_fitting import PeakFitter


def test_subshell_of_siegbahn_lines():
    assert subshell_of("Kα1") == "K"
    assert subshell_of("Kβ3") == "K"
    assert subshell_of("Lα1") == "L3"
    assert subshell_of("Lβ2") == "L3"
    assert subshell_of("Lβ1") == "L2"
    assert subshell_of("Lγ1") == "L2"
    assert subshell_of("Lβ3") == "L1"
    assert subshell_of("Lγ2") == "L1"
    assert subshell_of("Mα1") == "M"
    assert subshell_of(None) == ""


def test_build_line_groups_partitions_sample_lines_and_singles():
    seeds = [
        {"energy": 6.404, "element": "Fe", "line": "Kα1", "is_tube_line": False, "relative_intensity": 1.0},
        {"energy": 6.391, "element": "Fe", "line": "Kα2", "is_tube_line": False, "relative_intensity": 0.5},
        {"energy": 7.058, "element": "Fe", "line": "Kβ1", "is_tube_line": False, "relative_intensity": 0.13},
        {"energy": 10.552, "element": "Pb", "line": "Lα1", "is_tube_line": False, "relative_intensity": 1.0},
        {"energy": 12.614, "element": "Pb", "line": "Lβ1", "is_tube_line": False, "relative_intensity": 0.6},
        {"energy": 20.216, "element": "Rh", "line": "Kα1", "is_tube_line": True},
        {"energy": 18.8, "element": "Rh", "line": "Compton Kα", "is_tube_line": True, "fixed_fwhm": 0.5},
        {"energy": 3.3, "element": None, "line": None, "is_tube_line": False},
        {"energy": 99.0, "element": "U", "line": "Kα1", "is_tube_line": False},  # out of range
    ]
    groups, singles = build_line_groups(seeds, 0.5, 40.0)
    keys = {g.key: g for g in groups}
    assert set(keys) == {"Fe K", "Pb L3", "Pb L2"}
    fe = keys["Fe K"]
    assert fe.strongest.name == "Kα1"
    ratios = {l.name: l.ratio for l in fe.lines}
    assert ratios["Kα1"] == pytest.approx(1.0)
    assert ratios["Kα2"] == pytest.approx(0.5, rel=0.05)
    assert 0.1 < ratios["Kβ1"] < 0.16
    assert len(singles) == 3  # Rh Kα, Compton, unlabeled


def test_shape_globals_roundtrip():
    for shape in ("gaussian", "tail_gaussian", "hypermet", "voigt", "pseudo_voigt"):
        names, p0, lo, hi = global_shape_bounds(shape)
        assert len(names) == len(p0) == len(lo) == len(hi)
        assert all(l <= v <= h for v, l, h in zip(p0, lo, hi))
        sp = shape_params_from_global(shape, 0.06, p0)
        assert sp["sigma"] == pytest.approx(0.06)
    sp = shape_params_from_global("tail_gaussian", 0.05, [0.1, 3.0])
    assert sp["tail_fraction"] == pytest.approx(0.1)
    assert sp["tail_sigma"] == pytest.approx(0.15)


def _synth(groups_amps, singles_amps, energy, shape="gaussian", noise=True, seed=0):
    """Build counts from group/single amplitudes at detector widths."""
    x = np.asarray(energy, dtype=float)
    y = np.zeros_like(x)
    for g, a in groups_amps:
        for l in g.lines:
            sigma = PeakFitter.calculate_fwhm(l.energy) / 2.355
            y += a * l.ratio * PeakFitter.model_with_defaults(x, 1.0, l.energy, sigma, shape)
    for pos, a in singles_amps:
        sigma = (pos.get("fixed_fwhm") or PeakFitter.calculate_fwhm(pos["energy"])) / 2.355
        y += a * PeakFitter.model_with_defaults(x, 1.0, pos["energy"], sigma, shape)
    bg = 20.0 + 0 * x
    total = y + bg
    if noise:
        rng = np.random.default_rng(seed)
        total = rng.poisson(total).astype(float)
    return total, bg


def test_grouped_fit_resolves_as_ka_pb_la_overlap():
    """As Kα (10.54) sits under Pb Lα (10.55): the As Kβ and Pb Lβ resolve them."""
    PeakFitter.activate(PeakFitter())  # default detector model
    fitter = SpectrumFitter()
    energy = np.arange(0.5, 25.0, 0.01)
    seeds = fitter.build_peak_positions(
        energy, elements=[{"symbol": "As", "z": 33}, {"symbol": "Pb", "z": 82}],
        auto_find_peaks=False, include_tube_lines=False,
    )
    groups, singles = build_line_groups(seeds, energy[0], energy[-1])
    by_key = {g.key: g for g in groups}
    truth = {"As K": 3000.0, "Pb L3": 1200.0, "Pb L2": 900.0, "Pb L1": 250.0}
    gamps = [(by_key[k], a) for k, a in truth.items() if k in by_key]
    counts, bg = _synth(gamps, [], energy)

    res = fit_grouped(energy, counts - bg, counts, groups, singles,
                      shape="gaussian", refine_energy=False)
    for k, a in truth.items():
        if k in by_key:
            assert res.group_amplitudes[k] == pytest.approx(a, rel=0.06), k
    # Per-line peaks carry the group tag and fixed pattern
    as_peaks = [p for p in res.peaks if p.element == "As" and p.line.startswith("K")]
    assert as_peaks and all(p.group == "As K" for p in as_peaks)
    ka1 = next(p for p in as_peaks if p.line == "Kα1")
    kb1 = next(p for p in as_peaks if p.line == "Kβ1")
    assert 0.1 < kb1.amplitude / ka1.amplitude < 0.2


def test_grouped_fit_recovers_energy_offset():
    PeakFitter.activate(PeakFitter())
    fitter = SpectrumFitter()
    energy = np.arange(0.5, 12.0, 0.01)
    seeds = fitter.build_peak_positions(
        energy, elements=[{"symbol": "Fe", "z": 26}, {"symbol": "Ca", "z": 20}],
        auto_find_peaks=False, include_tube_lines=False,
    )
    groups, singles = build_line_groups(seeds, energy[0], energy[-1])
    by_key = {g.key: g for g in groups}
    # Synthesize with every line shifted +20 eV (bad zero)
    shifted = []
    import copy
    for g in groups:
        g2 = copy.deepcopy(g)
        for l in g2.lines:
            l.energy += 0.020
        shifted.append(g2)
    truth = {"Fe K": 8000.0, "Ca K": 4000.0}
    counts, bg = _synth([(g, truth.get(g.key, 0.0)) for g in shifted], [], energy)

    res = fit_grouped(energy, counts - bg, counts, groups, singles,
                      shape="gaussian", refine_energy=True)
    assert res.energy_offset_kev == pytest.approx(0.020, abs=0.004)
    assert res.group_amplitudes["Fe K"] == pytest.approx(8000.0, rel=0.05)
    assert res.group_amplitudes["Ca K"] == pytest.approx(4000.0, rel=0.05)


def test_fit_spectrum_grouped_vs_released_flags_and_peaks():
    """The SpectrumFitter switch produces tagged peaks and mode statistics."""
    PeakFitter.activate(PeakFitter())
    fitter = SpectrumFitter()
    energy = np.arange(0.5, 15.0, 0.01)
    seeds = fitter.build_peak_positions(
        energy, elements=[{"symbol": "Fe", "z": 26}], auto_find_peaks=False,
        include_tube_lines=False,
    )
    groups, singles = build_line_groups(seeds, energy[0], energy[-1])
    fe = next(g for g in groups if g.key == "Fe K")
    counts, _bg = _synth([(fe, 6000.0)], [], energy)

    grouped = fitter.fit_spectrum(
        energy, counts, elements=[{"symbol": "Fe", "z": 26}],
        background_method="linear", peak_shape="gaussian",
        auto_find_peaks=False, include_tube_lines=False, grouped_lines=True,
    )
    assert grouped.statistics["fit_mode"] == "grouped"
    assert "Fe K" in grouped.statistics["line_groups"]
    assert grouped.statistics["line_groups"]["Fe K"] == pytest.approx(6000.0, rel=0.1)
    assert any(p.group == "Fe K" for p in grouped.peaks)

    released = fitter.fit_spectrum(
        energy, counts, elements=[{"symbol": "Fe", "z": 26}],
        background_method="linear", peak_shape="gaussian",
        auto_find_peaks=False, include_tube_lines=False, grouped_lines=False,
    )
    assert released.statistics["fit_mode"] == "released"
    assert all(p.group is None for p in released.peaks)
    # Sequential per-line fit cannot split the unresolved Kα1/Kα2 pair
    # (it lumps them into whichever is fitted first) — the grouped fit can.
    ka_total = sum(p.amplitude for p in released.peaks if p.line in ("Kα1", "Kα2"))
    assert ka_total == pytest.approx(6000.0 * 1.51, rel=0.15)


def test_peak_group_roundtrip_dict():
    from core.peak_fitting import Peak

    p = Peak(energy=6.4, amplitude=1.0, fwhm=0.15, area=1.0, element="Fe",
             line="Kα1", group="Fe K")
    back = Peak.from_dict(p.to_dict())
    assert back.group == "Fe K"
    legacy = Peak.from_dict({"energy": 6.4})
    assert legacy.group is None
