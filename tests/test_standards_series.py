"""Standards calibration: principal-series intensities (Kα+Kβ …)."""

from __future__ import annotations

from types import SimpleNamespace
import math

import numpy as np
import pytest

from core.fitting import SpectrumFitter
from core.peak_fitting import Peak, PeakFitter
from core.standards_calibration import (
    LINE_ALL,
    LINE_AUTO,
    LINE_DEFAULT,
    LINE_SERIES,
    MODEL_LINEAR,
    MODEL_THROUGH_ORIGIN,
    ElementCurve,
    SpotIntensity,
    StandardPoint,
    StandardRecord,
    StandardsCalibration,
    annotate_curve_prediction,
    choose_line_group,
    condition_warnings,
    crm_display_concentration,
    extract_spot_intensities,
    line_matches_group,
    recipe_mismatches,
    suggest_outlier_exclusions,
)


def test_series_is_default_and_choose_line_group_modes():
    assert LINE_DEFAULT == LINE_SERIES
    present = {"Kα", "Kβ", "Lα", "Lβ", "Compton"}
    assert choose_line_group(present, LINE_AUTO) == "Kα"
    assert choose_line_group(present, LINE_SERIES) == "K"
    assert choose_line_group({"Lα", "Lβ", "Lγ"}, LINE_SERIES) == "L"
    assert choose_line_group({"Lβ"}, LINE_AUTO) == LINE_ALL  # no α family
    assert choose_line_group({"Compton"}, LINE_SERIES) == LINE_ALL
    # 50 kV barely exceeds the La / Ba K-edge — use L, not noise at ~33 keV
    assert choose_line_group(present, LINE_SERIES, element="La", excitation_kv=50.0) == "L"
    assert choose_line_group(present, LINE_SERIES, element="Ba", excitation_kv=50.0) == "L"
    assert choose_line_group(present, LINE_AUTO, element="La", excitation_kv=50.0) == "Lα"
    # Fe K-edge ~7.1 keV is well excited at 50 kV
    assert choose_line_group(present, LINE_SERIES, element="Fe", excitation_kv=50.0) == "K"
    # Only K fitted → still return K even when overvoltage is low
    assert choose_line_group({"Kα", "Kβ"}, LINE_SERIES, element="La", excitation_kv=50.0) == "K"


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


def _spot(name, path, cps, area=None, err=1.0):
    area = cps * 30.0 if area is None else area
    return SpotIntensity(
        spectrum=name, path=path, live_time=30.0,
        area=area, area_err=err, cps=cps, cps_err=err / 30.0, lines=["Kα1"],
    )


def _cal_with_points(element, pairs):
    """pairs: (standard, wt%, cps)."""
    cal = StandardsCalibration()
    for std, wt, cps in pairs:
        path = f"{std}.txt"
        cal.add_standard(StandardRecord(name=std, concentrations={element: wt},
                                        spectrum_paths=[path]))
        cal.intensities.setdefault(std, {})[path] = {
            element: _spot(f"{std}.txt", path, cps),
        }
    cal.line_groups[element] = "K"
    return cal


def test_negative_slope_is_not_a_fitted_curve():
    cal = _cal_with_points("Mg", [
        ("s1", 0.4, 0.06), ("s2", 0.6, 0.05), ("s3", 0.8, 0.03), ("s4", 1.0, 0.02),
    ])
    cal.build_curves(model=MODEL_LINEAR, weighted=False)
    curve = cal.curves["Mg"]
    assert curve.slope < 0
    assert curve.fitted is False
    assert "Negative" in curve.message
    assert cal.fitted_curves() == []


def test_negative_r_squared_is_not_a_fitted_curve():
    # Real OREAS La-K intensities: spots agree, but cps does not track wt%.
    # Weighted WLS then yields R² < 0 (worse than the mean).
    cal = StandardsCalibration()
    for std, wt, cps, sem in (
        ("oreas 466", 0.02, 0.3539, 0.876),
        ("oreas 460b", 0.12, 0.2978, 0.888),
        ("oreas 462", 0.37, 0.1871, 0.389),
        ("oreas 464", 1.06, 0.3655, 0.410),
    ):
        path = f"{std}.txt"
        cal.add_standard(StandardRecord(
            name=std, concentrations={"La": wt}, spectrum_paths=[path],
        ))
        cal.intensities.setdefault(std, {})[path] = {
            "La": _spot(f"{std}.txt", path, cps, err=sem * 30.0),
        }
    cal.line_groups["La"] = "K"
    cal.build_curves(model=MODEL_LINEAR, weighted=True)
    curve = cal.curves["La"]
    assert curve.r_squared < 0
    assert curve.fitted is False
    assert "No correlation" in curve.message


def test_suggest_outlier_exclusions_rescues_cu_like_scatter():
    """Two low-cps ores wreck an otherwise linear Cu set (real CRM numbers)."""
    cal = _cal_with_points("Cu", [
        ("OREAS_466", 0.00342, 12.615),
        ("OREAS_460b", 0.0041, 13.136),
        ("LKSD_1", 0.0044, 10.482),
        ("STSD_2", 0.0047, 12.380),
        ("TILL_1", 0.0047, 14.785),
        ("OREAS_462", 0.0061, 0.941),
        ("NIST_SRM_2586", 0.0081, 18.009),
        ("OREAS_464", 0.0092, 2.024),
        ("NIST_SRM_2587", 0.0160, 29.537),
        ("PACS_2", 0.0310, 37.302),
    ])
    cal.build_curves(model=MODEL_LINEAR, weighted=True)
    assert cal.curves["Cu"].r_squared < 0.80
    applied = cal.exclude_outliers(["Cu"])
    assert set(applied["Cu"]) == {"OREAS_462", "OREAS_464"}
    curve = cal.curves["Cu"]
    assert curve.fitted
    assert curve.r_squared > 0.90
    assert curve.slope > 0
    used = {p.standard for p in curve.points if p.included}
    assert "OREAS_462" not in used and "OREAS_464" not in used
    assert "Excluded outlier" in curve.message


def test_suggest_outlier_exclusions_leaves_good_curves_alone():
    cal = _cal_with_points("Ba", [
        ("s1", 0.10, 20.0), ("s2", 0.20, 40.0),
        ("s3", 0.40, 80.0), ("s4", 0.80, 160.0),
        ("s5", 1.20, 240.0),
    ])
    cal.build_curves(model=MODEL_LINEAR, weighted=False)
    assert cal.curves["Ba"].r_squared > 0.99
    assert suggest_outlier_exclusions(cal.curves["Ba"].points) == []
    assert cal.exclude_outliers(["Ba"]) == {}


def test_suggest_outlier_exclusions_does_not_invent_a_fit():
    """V-like: intensity does not track concentration, no useful subset."""
    cal = _cal_with_points("V", [
        ("NIST_SRM_2587", 0.0078, 0.846),
        ("OREAS_466", 0.0128, 1.208),
        ("NIST_SRM_2586", 0.0160, 2.538),
        ("OREAS_464", 0.0207, 0.413),
        ("OREAS_460b", 0.0222, 0.565),
        ("OREAS_462", 0.0353, 0.813),
    ])
    cal.build_curves(model=MODEL_LINEAR, weighted=True)
    assert cal.curves["V"].fitted is False
    assert cal.exclude_outliers(["V"]) == {}
    assert all(p.included for p in cal.curves["V"].points)


def test_exclude_outliers_drops_single_zr_wrecking_point():
    cal = _cal_with_points("Zr", [
        ("OREAS_466", 0.0163, 70.876),
        ("OREAS_464", 0.0210, 5.185),
        ("OREAS_462", 0.0270, 13.233),
        ("OREAS_460b", 0.0415, 64.297),
    ])
    cal.build_curves(model=MODEL_LINEAR, weighted=True)
    assert cal.curves["Zr"].r_squared < 0.1
    applied = cal.exclude_outliers(["Zr"])
    assert applied["Zr"] == ["OREAS_466"]
    assert cal.curves["Zr"].fitted
    assert cal.curves["Zr"].r_squared > 0.90


def test_clear_point_exclusions_restores_curve():
    cal = _cal_with_points("Zr", [
        ("OREAS_466", 0.0163, 70.876),
        ("OREAS_464", 0.0210, 5.185),
        ("OREAS_462", 0.0270, 13.233),
        ("OREAS_460b", 0.0415, 64.297),
    ])
    cal.build_curves(model=MODEL_LINEAR, weighted=True)
    cal.exclude_outliers(["Zr"])
    cal.clear_point_exclusions("Zr")
    cal.build_curves(model=MODEL_LINEAR, weighted=True)
    assert all(p.included for p in cal.curves["Zr"].points)
    assert cal.curves["Zr"].r_squared < 0.1


def _mdc_curve(c0, c1, residual_variance=0.04, fitted=True):
    points = [
        StandardPoint("a", 1.0, 10.0, 0.2, 0.1, 3, 2.0, True),
        StandardPoint("b", 2.0, 20.0, 0.3, 0.15, 3, 1.5, True),
        StandardPoint("c", 4.0, 40.0, 0.4, 0.2, 3, 1.0, True),
    ]
    return ElementCurve(
        element="Cu",
        line_group="K",
        model=MODEL_LINEAR,
        coefficients=[c0, c1, 0.0],
        coefficient_errors=[0.0, 0.0, 0.0],
        r_squared=0.99,
        rmse=math.sqrt(residual_variance),
        n_standards=3,
        n_spectra=9,
        mean_rsd_percent=1.5,
        residual_variance=residual_variance,
        covariance=[],
        points=points,
        enabled=True,
        fitted=fitted,
    )


def test_mdc_is_always_non_negative_even_with_negative_intercept():
    # s = 0.2 wt%, slope = 0.1 → 3σ MDC = 0.6, independent of intercept
    pos = _mdc_curve(0.5, 0.1)
    neg = _mdc_curve(-2.0, 0.1)
    assert pos.minimum_detectable_concentration() == pytest.approx(0.6)
    assert neg.minimum_detectable_concentration() == pytest.approx(0.6)
    assert pos.minimum_detectable_concentration() > 0
    assert neg.minimum_detectable_concentration() > 0


def test_mdc_undefined_when_slope_is_not_positive():
    failed = _mdc_curve(0.01, -0.002, fitted=False)
    assert failed.minimum_detectable_concentration() is None
    zero_slope = _mdc_curve(0.01, 0.0)
    assert zero_slope.minimum_detectable_concentration() is None


def test_quantify_can_filter_to_one_element():
    PeakFitter.activate(PeakFitter())
    fitter = SpectrumFitter()
    energy = np.arange(0.5, 12.0, 0.01)
    cal = StandardsCalibration()
    truth = {"S1": (20.0, 5.0), "S2": (40.0, 10.0), "S3": (60.0, 15.0)}
    spectra = {}
    for i, (name, (fe, ca)) in enumerate(truth.items()):
        cal.add_standard(StandardRecord(
            name=name, concentrations={"Fe": fe, "Ca": ca},
            spectrum_paths=[f"{name}.txt"],
        ))
        spectra[name] = [(
            f"{name}.txt",
            _synth_std(energy, 100.0 * fe, 60.0 * ca, fitter, i),
        )]
    cal.fit_spectra(
        spectra, fitter, elements=["Fe", "Ca"],
        fit_kwargs={"background_method": "linear", "peak_shape": "gaussian",
                    "include_tube_lines": False, "grouped_lines": True},
        line_selection=LINE_SERIES, normalise="live_time",
    )
    cal.build_curves(model=MODEL_THROUGH_ORIGIN, weighted=True)
    unknown = _synth_std(energy, 100.0 * 30.0, 60.0 * 8.0, fitter, 99)
    res = fitter.fit_spectrum(
        energy, unknown.counts,
        elements=[{"symbol": "Fe", "z": 26}, {"symbol": "Ca", "z": 20}],
        background_method="linear", peak_shape="gaussian", auto_find_peaks=False,
        include_tube_lines=False, grouped_lines=True,
    )
    q = cal.quantify(
        res.peaks, unknown.live_time, fit_result=res, energy=energy,
        elements=["Fe"],
    )
    assert list(q) == ["Fe"]
    assert q["Fe"]["concentration"] == pytest.approx(30.0, rel=0.05)


def test_negative_prediction_is_detected_not_quantified():
    curve = _mdc_curve(-2.0, 0.1)
    # C = -2 + 0.1 * I; I=5 → C=-1.5, below MDC 0.6
    flags = annotate_curve_prediction(curve, -1.5, 5.0)
    assert flags["not_quantified"] is True
    assert flags["quant_flag"] in {"negative", "below_mdc"}
    row = {"concentration": -1.5, "not_quantified": True, "quant_flag": "negative"}
    assert crm_display_concentration(row) == "n.q."


def test_recipe_mismatches_and_condition_warnings():
    settings = {
        "background_method": "snip",
        "peak_shape": "tail_gaussian",
        "grouped_lines": True,
        "excitation_kv": 50.0,
        "tube_current": 1.0,
    }
    ok = {
        "background_method": "SNIP",
        "peak_shape": "tail_gaussian",
        "grouped_lines": True,
        "excitation_kv": 50.0,
        "tube_current": 1.0,
    }
    assert recipe_mismatches(settings, ok) == []
    bad = dict(ok, background_method="linear", grouped_lines=False)
    mismatches = recipe_mismatches(settings, bad)
    assert any("background" in m for m in mismatches)
    assert any("line ratios" in m for m in mismatches)
    assert condition_warnings(settings, ok) == []
    assert condition_warnings(settings, {**ok, "excitation_kv": 40.0})
    assert condition_warnings(settings, {**ok, "tube_current": 2.0})
