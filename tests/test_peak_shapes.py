"""Peak profile functions: Hypermet, Tail-Gaussian, and aliases."""

import numpy as np
import pytest

from core.peak_fitting import (
    Peak,
    PeakFitter,
    normalize_peak_shape,
    peak_shape_ui_label,
)


def test_normalize_peak_shape_aliases():
    assert normalize_peak_shape("Tail-Gaussian") == "tail_gaussian"
    assert normalize_peak_shape("tail-gaussian") == "tail_gaussian"
    assert normalize_peak_shape("Hypermet") == "hypermet"
    assert normalize_peak_shape("Gaussian") == "gaussian"
    assert normalize_peak_shape(None) == "tail_gaussian"
    assert peak_shape_ui_label("Voigt") == "Tail-Gaussian"
    assert peak_shape_ui_label("voigt") == "Tail-Gaussian"
    assert peak_shape_ui_label("tail_gaussian") == "Tail-Gaussian"


def test_hypermet_is_continuous_and_left_tailed():
    x = np.linspace(5.5, 7.5, 2001)
    center, sigma = 6.4, 0.06
    y = PeakFitter.hypermet(
        x, 1000.0, center, sigma, tail_amplitude=0.15, tail_beta=0.18, step_amplitude=0.0
    )
    i0 = int(np.argmin(np.abs(x - center)))
    # Continuous at the centroid (no truncated-exponential kink)
    assert abs(y[i0] - y[i0 - 1]) < 2.0
    assert abs(y[i0 + 1] - y[i0]) < 2.0
    left = y[np.argmin(np.abs(x - (center - 2.0 * sigma)))]
    right = y[np.argmin(np.abs(x - (center + 2.0 * sigma)))]
    assert left > right * 1.15


def test_hypermet_step_shelf():
    x = np.linspace(4.0, 9.0, 2500)
    center, sigma, amp, step = 6.4, 0.06, 1000.0, 0.05
    y = PeakFitter.hypermet(
        x, amp, center, sigma, 0.1, 0.18, step_amplitude=step
    )
    far_left = y[x < center - 15.0 * sigma].mean()
    far_right = y[x > center + 15.0 * sigma].mean()
    assert far_left == pytest.approx(amp * step, rel=0.15)
    assert far_right < amp * 0.01


def test_hypermet_area_excludes_step():
    amp, sigma, tail_amp, beta = 800.0, 0.05, 0.12, 0.15
    x = np.linspace(-4.0, 4.0, 40001) + 6.4
    y = PeakFitter.hypermet(
        x, amp, 6.4, sigma, tail_amp, beta, step_amplitude=0.0
    )
    numeric = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    analytic = PeakFitter.compute_peak_area(
        amp, sigma, "hypermet",
        {"tail_amplitude": tail_amp, "tail_beta": beta, "step_amplitude": 0.0},
    )
    assert abs(numeric - analytic) / analytic < 0.02


def test_legacy_hypermet_tail_slope_still_evaluates():
    x = np.linspace(6.0, 7.0, 200)
    peak = Peak(
        energy=6.4,
        amplitude=100.0,
        fwhm=0.14,
        area=1.0,
        shape="hypermet",
        shape_params={"sigma": 0.06, "tail_amplitude": 0.1, "tail_slope": 2.0},
    )
    y = PeakFitter.evaluate_peak(peak, x)
    assert np.all(np.isfinite(y))
    assert y.max() > 50.0


def test_tail_gaussian_area_includes_wide_component():
    amp, sigma, frac, tsig = 500.0, 0.06, 0.2, 0.18
    x = np.linspace(5.5, 7.5, 20001)
    y = PeakFitter.tail_gaussian(x, amp, 6.4, sigma, frac, tsig)
    numeric = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    analytic = PeakFitter.compute_peak_area(
        amp, sigma, "tail_gaussian",
        {"tail_fraction": frac, "tail_sigma": tsig},
    )
    assert abs(numeric - analytic) / analytic < 0.01
    gauss_only = amp * sigma * np.sqrt(2.0 * np.pi)
    assert analytic > gauss_only * 1.2


def test_fwhm_calibration_locks_gaussian_only():
    from core.fwhm_calibration import FWHMCalibration

    energy = np.linspace(5.8, 7.0, 500)
    true_sigma = 0.040
    counts = PeakFitter.gaussian(energy, 1200.0, 6.40, true_sigma)

    cal = FWHMCalibration(
        model_type="linear",
        parameters={"intercept": 0.30, "slope": 0.0},
        parameter_errors={"intercept": 0.0, "slope": 0.0},
        r_squared=0.99,
        rmse=0.001,
        aic=0.0,
        bic=0.0,
        n_peaks=4,
        energy_range=(1.0, 20.0),
        calibration_date="2026-01-01T00:00:00",
    )

    prev_active = PeakFitter._active
    prev_use = PeakFitter.USE_CALIBRATED_SHAPES
    prev_cal = PeakFitter._fwhm_calibration
    try:
        PeakFitter._active = None
        PeakFitter.USE_CALIBRATED_SHAPES = True
        PeakFitter._fwhm_calibration = cal

        assert PeakFitter.fwhm_cal_locks_width("gaussian")
        assert not PeakFitter.fwhm_cal_locks_width("tail_gaussian")
        assert not PeakFitter.fwhm_cal_locks_width("hypermet")

        locked = PeakFitter.fit_single_peak(
            energy, counts, 6.40, shape="gaussian", known_line=True
        )
        assert locked is not None
        assert abs(locked.fwhm - 0.30) < 0.01

        for shape in ("tail_gaussian", "hypermet"):
            peak = PeakFitter.fit_single_peak(
                energy, counts, 6.40, shape=shape, known_line=True
            )
            assert peak is not None
            assert abs(peak.fwhm - 0.30) > 0.05
            assert abs(peak.fwhm - 2.355 * true_sigma) < 0.05
    finally:
        PeakFitter._active = prev_active
        PeakFitter.USE_CALIBRATED_SHAPES = prev_use
        PeakFitter._fwhm_calibration = prev_cal


def test_fit_tail_gaussian_and_hypermet_on_synthetic():
    energy = np.linspace(5.8, 7.0, 400)
    true = PeakFitter.tail_gaussian(energy, 900.0, 6.40, 0.055, 0.15, 0.16)
    counts = true + 20.0
    for shape in ("gaussian", "tail_gaussian", "hypermet"):
        peak = PeakFitter.fit_single_peak(
            energy, counts, 6.40, shape=shape, known_line=True
        )
        assert peak is not None
        assert abs(peak.energy - 6.40) < 0.03
        assert peak.amplitude > 100.0
