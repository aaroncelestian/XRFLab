"""Standards calibration: principal-series intensities (Kα+Kβ …)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.fitting import SpectrumFitter
from core.peak_fitting import Peak, PeakFitter
from core.standards_calibration import (
    LINE_ALL,
    LINE_AUTO,
    LINE_DEFAULT,
    LINE_SERIES,
    MODEL_THROUGH_ORIGIN,
    StandardRecord,
    StandardsCalibration,
    choose_line_group,
    extract_spot_intensities,
    line_matches_group,
)


def test_series_is_default_and_choose_line_group_modes():
    assert LINE_DEFAULT == LINE_SERIES
    present = {"Kα", "Kβ", "Lα", "Lβ", "Compton"}
    assert choose_line_group(present, LINE_AUTO) == "Kα"
    assert choose_line_group(present, LINE_SERIES) == "K"
    assert choose_line_group({"Lα", "Lβ", "Lγ"}, LINE_SERIES) == "L"
    assert choose_line_group({"Lβ"}, LINE_AUTO) == LINE_ALL  # no α family
    assert choose_line_group({"Compton"}, LINE_SERIES) == LINE_ALL


def test_line_matches_group_series_family_all():
    assert line_matches_group("Kα1", "K")
    assert line_matches_group("Kβ3", "K")
    assert not line_matches_group("Lα1", "K")
    assert line_matches_group("Kα2", "Kα")
    assert not line_matches_group("Kβ1", "Kα")
    assert line_matches_group("Lγ1", LINE_ALL)
    assert not line_matches_group("Compton Kα", LINE_ALL)
    assert not line_matches_group(None, "K")


def _pk(el, line, area, e):
    return Peak(energy=e, amplitude=area, fwhm=0.15, area=area, element=el, line=line)


def test_extract_spot_intensities_sums_whole_series():
    spec = SimpleNamespace(energy=np.linspace(0, 20, 2001), live_time=10.0, real_time=12.0)
    fit = SimpleNamespace(
        background=np.zeros(2001),
        peaks=[
            _pk("Fe", "Kα1", 800, 6.40), _pk("Fe", "Kα2", 400, 6.39),
            _pk("Fe", "Kβ1", 150, 7.06), _pk("Fe", "Lα1", 50, 0.70),
            _pk("Pb", "Lα1", 300, 10.55), _pk("Pb", "Lβ1", 200, 12.61),
            _pk("Pb", "Mα1", 20, 2.35),
            Peak(energy=18.8, amplitude=1, fwhm=0.5, area=999, element="Rh",
                 line="Compton Kα", is_tube_line=True),
        ],
    )
    out = extract_spot_intensities(
        spec, fit, ["Fe", "Pb"], line_groups={"Fe": "K", "Pb": "L"}, normalise="live_time",
    )
    assert out["Fe"].area == pytest.approx(1350)          # Kα1+Kα2+Kβ1, not Lα
    assert out["Fe"].cps == pytest.approx(135.0)
    assert sorted(out["Fe"].lines) == ["Kα1", "Kα2", "Kβ1"]
    assert out["Pb"].area == pytest.approx(500)           # Lα+Lβ, not Mα
    fam = extract_spot_intensities(
        spec, fit, ["Fe"], line_groups={"Fe": "Kα"}, normalise="none",
    )
    assert fam["Fe"].area == pytest.approx(1200)


def _synth_std(energy, fe_amp, ca_amp, fitter, seed):
    """Two-element synthetic standard spectrum with Poisson noise."""
    from core.line_groups import build_line_groups

    seeds = fitter.build_peak_positions(
        energy, elements=[{"symbol": "Fe", "z": 26}, {"symbol": "Ca", "z": 20}],
        auto_find_peaks=False, include_tube_lines=False,
    )
    groups, _ = build_line_groups(seeds, energy[0], energy[-1])
    y = np.full_like(energy, 15.0)
    amps = {"Fe K": fe_amp, "Ca K": ca_amp}
    for g in groups:
        a = amps.get(g.key, 0.0)
        for l in g.lines:
            sigma = PeakFitter.calculate_fwhm(l.energy) / 2.355
            y += a * l.ratio * PeakFitter.model_with_defaults(energy, 1.0, l.energy, sigma, "gaussian")
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        energy=energy, counts=rng.poisson(y).astype(float), live_time=10.0, real_time=11.0,
    )


def test_series_calibration_roundtrip_with_grouped_fit():
    """Fit standards with series intensities, then quantify an unknown."""
    PeakFitter.activate(PeakFitter())
    fitter = SpectrumFitter()
    energy = np.arange(0.5, 12.0, 0.01)

    cal = StandardsCalibration()
    truth = {"S1": (20.0, 5.0), "S2": (40.0, 10.0), "S3": (60.0, 15.0)}  # Fe, Ca wt%
    spectra = {}
    for i, (name, (fe, ca)) in enumerate(truth.items()):
        cal.add_standard(StandardRecord(name=name, concentrations={"Fe": fe, "Ca": ca},
                                        spectrum_paths=[f"{name}.txt"]))
        spectra[name] = [(f"{name}.txt", _synth_std(energy, 100.0 * fe, 60.0 * ca, fitter, i))]

    cal.fit_spectra(
        spectra, fitter, elements=["Fe", "Ca"],
        fit_kwargs={"background_method": "linear", "peak_shape": "gaussian",
                    "include_tube_lines": False, "grouped_lines": True},
        line_selection=LINE_SERIES, normalise="live_time",
    )
    assert cal.line_groups == {"Fe": "K", "Ca": "K"}
    assert cal.fit_settings["grouped_lines"] is True
    for spots in cal.intensities.values():
        for per_el in spots.values():
            assert "Kβ1" in per_el["Fe"].lines and "Kα1" in per_el["Fe"].lines

    cal.build_curves(model=MODEL_THROUGH_ORIGIN, weighted=True)
    assert cal.curves["Fe"].fitted and cal.curves["Fe"].r_squared > 0.999
    assert cal.curves["Fe"].line_group == "K"

    unknown = _synth_std(energy, 100.0 * 30.0, 60.0 * 8.0, fitter, 99)
    res = fitter.fit_spectrum(
        energy, unknown.counts, elements=[{"symbol": "Fe", "z": 26}, {"symbol": "Ca", "z": 20}],
        background_method="linear", peak_shape="gaussian", auto_find_peaks=False,
        include_tube_lines=False, grouped_lines=True,
    )
    q = cal.quantify(res.peaks, unknown.live_time, fit_result=res, energy=energy)
    assert q["Fe"]["concentration"] == pytest.approx(30.0, rel=0.05)
    assert q["Ca"]["concentration"] == pytest.approx(8.0, rel=0.08)
    assert q["Fe"]["line"].startswith("K (")
    assert "Kβ" in q["Fe"]["line"]

    # Serialization keeps the series group
    back = StandardsCalibration.from_dict(cal.to_dict())
    assert back.line_groups["Fe"] == "K"
    assert back.curves["Fe"].line_group == "K"
