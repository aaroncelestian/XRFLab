"""Tests for the HTML analysis report engine (no Qt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.composition import VALUE_RELATIVE, VALUE_WT, CompositionRow
from core.fitting import FitResult
from core.fwhm_calibration import FWHMCalibration
from core.peak_fitting import Peak
from core.report import (
    SECTION_COMPOSITION,
    SECTION_ERRORS,
    SECTION_FIT,
    SECTION_FWHM,
    SECTION_METHODS,
    SECTION_STANDARDS,
    SECTION_TUBE,
    ReportContext,
    ReportOptions,
    app_version,
    build_report_html,
    curve_recovery_stats,
    flagged_residuals,
    lod_loq,
    options_from_composition_state,
    point_recovery,
    precision_stats,
    section_availability,
    write_report,
)
from core.spectrum import Spectrum
from core.standards_calibration import (
    ElementCurve,
    StandardPoint,
    StandardsCalibration,
)
from core.tube_profile import TubeProfile, TubeProfileLibrary


def _curve() -> ElementCurve:
    """Linear C = 0.1·I with residual variance 0.04 (s = 0.2 wt%)."""
    points = [
        StandardPoint("CRM1", 10.0, 100.0, 1.0, 0.5, 2, 1.0, True, 10.1, 0.1),
        StandardPoint("CRM2", 20.0, 200.0, 1.5, 0.8, 3, 0.8, True, 19.8, -0.2),
        StandardPoint("CRM3", 5.0, 50.0, 0.4, 0.2, 2, 0.8, True, 5.0, 0.0),
    ]
    return ElementCurve(
        element="Fe",
        line_group="Kα",
        model="linear",
        coefficients=[0.0, 0.1, 0.0],
        coefficient_errors=[0.0, 0.01, 0.0],
        r_squared=0.99,
        rmse=0.2,
        n_standards=3,
        n_spectra=7,
        mean_rsd_percent=0.9,
        residual_variance=0.04,
        covariance=[[0.0, 0.0], [0.0, 0.0001]],
        points=points,
        enabled=True,
        fitted=True,
    )


def _fwhm() -> FWHMCalibration:
    return FWHMCalibration(
        model_type="linear",
        parameters={"intercept": 0.08, "slope": 0.01},
        parameter_errors={"intercept": 0.002, "slope": 0.0005},
        r_squared=0.97,
        rmse=0.004,
        aic=10.0,
        bic=12.0,
        n_peaks=8,
        energy_range=(1.0, 16.0),
        calibration_date="2026-01-01T00:00:00",
    )


def _fit() -> FitResult:
    energy = np.linspace(0.5, 10.0, 200)
    counts = np.exp(-((energy - 6.4) ** 2) / 0.08) * 800 + 20
    fitted = counts + 1.0
    return FitResult(
        background=np.full_like(counts, 20.0),
        fitted_spectrum=fitted,
        residuals=counts - fitted,
        peaks=[
            Peak(energy=6.404, amplitude=800, fwhm=0.15, area=1200, element="Fe", line="Kα1"),
        ],
        statistics={
            "chi_squared": 210.0,
            "reduced_chi_squared": 1.1,
            "r_squared": 0.992,
            "dof": 190,
            "fit_mode": "grouped",
        },
    )


def _context(**overrides) -> ReportContext:
    energy = np.linspace(0.5, 10.0, 200)
    counts = np.exp(-((energy - 6.4) ** 2) / 0.08) * 800 + 20
    spec = Spectrum(energy=energy, counts=counts, live_time=30.0)
    std = StandardsCalibration(curves={"Fe": _curve()})
    tube = TubeProfileLibrary()
    tube.set_profile(
        TubeProfile(
            tube_element="Rh",
            tube_kv=50.0,
            line_ratios={"Kα1": 1.0, "Kβ1": 0.18},
            source="measured",
        )
    )
    row = CompositionRow(
        name="Basalt_1",
        source_id="basalt_1.txt",
        sample="Basalt",
        values={"Si": 50.0, "Fe": 10.0},
        relative={"Si": 50.0, "Fe": 10.0},
        wt={"Si": 24.0, "Fe": 8.0},
        formula_wt={"SiO2": 51.3, "FeO": 10.3},
    )
    row2 = CompositionRow(
        name="Basalt_2",
        source_id="basalt_2.txt",
        sample="Basalt",
        values={"Si": 52.0, "Fe": 11.0},
        relative={"Si": 52.0, "Fe": 11.0},
        wt={"Si": 25.0, "Fe": 8.5},
        formula_wt={"SiO2": 53.5, "FeO": 10.9},
    )
    data = dict(
        spectrum=spec,
        spectrum_path="basalt.mca",
        fit_result=_fit(),
        concentrations={"Fe": {"concentration": 8.0, "method": "standards_curve"}},
        quantification_method="standards_curve",
        fwhm_calibration=_fwhm(),
        fwhm_measurements=[
            type("M", (), {"element": "Fe", "line": "Kα1", "energy": 6.4, "fwhm": 0.145, "fit_quality": 0.99})()
        ],
        tube_library=tube,
        standards_calibration=std,
        composition_rows=[row, row2],
        excitation_kv=50.0,
        tube_element="Rh",
        fit_settings={"peak_shape": "tail_gaussian", "grouped_lines": True},
    )
    data.update(overrides)
    return ReportContext(**data)


def test_lod_loq_and_recovery():
    curve = _curve()
    lod, loq = lod_loq(curve)
    assert lod == pytest.approx(0.6)
    assert loq == pytest.approx(2.0)
    assert point_recovery(curve.points[0]) == pytest.approx(101.0)
    stats = curve_recovery_stats(curve)
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(100.0, abs=1.0)
    assert flagged_residuals(curve) == []


def test_lod_undefined_when_no_dof():
    curve = _curve()
    curve.points = curve.points[:2]
    curve.residual_variance = 0.04
    lod, loq = lod_loq(curve)
    assert lod is None and loq is None


def test_precision_stats():
    prec = precision_stats(10.0, 1.0, 4)
    assert prec["sem"] == pytest.approx(0.5)
    assert prec["rsd"] == pytest.approx(10.0)
    assert precision_stats(10.0, 1.0, 1)["sem"] is None


def test_html_includes_and_omits_sections():
    ctx = _context()
    full = build_report_html(
        ctx,
        ReportOptions(sections=set(_context_sections()), include_plots=False),
    )
    for key in (
        SECTION_METHODS,
        SECTION_FWHM,
        SECTION_TUBE,
        SECTION_STANDARDS,
        SECTION_FIT,
        SECTION_COMPOSITION,
        SECTION_ERRORS,
    ):
        assert f'id="section-{key}"' in full

    slim = build_report_html(
        ctx,
        ReportOptions(sections={SECTION_METHODS, SECTION_FIT}, include_plots=False),
    )
    assert 'id="section-methods"' in slim
    assert 'id="section-fit"' in slim
    assert 'id="section-fwhm"' not in slim
    assert 'id="section-standards"' not in slim
    assert 'id="section-composition"' not in slim


def _context_sections():
    return {
        SECTION_METHODS,
        SECTION_FWHM,
        SECTION_TUBE,
        SECTION_STANDARDS,
        SECTION_FIT,
        SECTION_COMPOSITION,
        SECTION_ERRORS,
    }


def test_composition_table_oxides_and_close():
    ctx = _context()
    html = build_report_html(
        ctx,
        ReportOptions(
            sections={SECTION_COMPOSITION},
            value_source=VALUE_RELATIVE,
            as_oxides=True,
            fe_as="FeO",
            close=False,
            include_plots=False,
        ),
    )
    assert "FeO" in html
    assert "SiO2" in html
    # Fe 10.5 mean * 1.2865 ≈ 13.508
    assert "13.508" in html or "13.50" in html

    closed = build_report_html(
        ctx,
        ReportOptions(
            sections={SECTION_COMPOSITION},
            value_source=VALUE_RELATIVE,
            as_oxides=True,
            fe_as="FeO",
            close=True,
            include_plots=False,
        ),
    )
    assert "% (closed)" in closed


def test_composition_fp_formula_mode():
    ctx = _context()
    html = build_report_html(
        ctx,
        ReportOptions(
            sections={SECTION_COMPOSITION},
            value_source=VALUE_WT,
            as_oxides=True,
            include_plots=False,
        ),
    )
    assert "wt% (formula)" in html
    assert "51.3" in html or "52.4" in html


def test_section_availability_reasons():
    empty = ReportContext()
    reasons = section_availability(empty)
    assert reasons[SECTION_FWHM]
    assert reasons[SECTION_STANDARDS]
    assert reasons[SECTION_FIT]
    assert reasons[SECTION_COMPOSITION]
    assert reasons[SECTION_METHODS] is None


def test_options_from_composition_state():
    opts = options_from_composition_state(
        {"value_source": "wt", "oxides": True, "fe_as": "Fe2O3", "close": True},
        sections=[SECTION_COMPOSITION],
    )
    assert opts.value_source == VALUE_WT
    assert opts.as_oxides is True
    assert opts.fe_as == "Fe2O3"
    assert opts.close is True
    assert opts.sections == {SECTION_COMPOSITION}


def test_write_report_embeds_png(tmp_path: Path):
    ctx = _context()
    dest = tmp_path / "report.html"
    write_report(
        dest,
        ctx,
        ReportOptions(sections=_context_sections(), include_plots=True),
    )
    text = dest.read_text(encoding="utf-8")
    assert "<img" in text
    assert "data:image/png;base64," in text
    assert "XRFLab" in text
    assert app_version()


def test_plot_helpers_return_png_bytes():
    from core.report_plots import (
        plot_calibration_curves,
        plot_composition_bars,
        plot_fit_residuals,
        plot_fwhm_calibration,
        plot_predicted_vs_certified,
        plot_spectrum_fit,
    )

    energy = np.linspace(1, 10, 50)
    counts = np.ones(50)
    png = plot_spectrum_fit(energy, counts, counts * 0.9, counts * 0.1)
    assert png and png.startswith(b"\x89PNG")
    png = plot_fit_residuals(energy, np.linspace(-1, 1, 50))
    assert png and png.startswith(b"\x89PNG")
    png = plot_fwhm_calibration(_fwhm())
    assert png and png.startswith(b"\x89PNG")
    curves = {"Fe": _curve()}
    png = plot_predicted_vs_certified(curves)
    assert png and png.startswith(b"\x89PNG")
    png = plot_calibration_curves(curves)
    assert png and png.startswith(b"\x89PNG")
    png = plot_composition_bars(
        [{"name": "A", "values": {"SiO2": 50.0, "FeO": 10.0}}],
        ["SiO2", "FeO"],
    )
    assert png and png.startswith(b"\x89PNG")
