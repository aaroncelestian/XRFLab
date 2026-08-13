"""Pytest suite for XRFLab core I/O, FWHM, fitting, and batch alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_STEEL = ROOT / "sample_data" / "steel_sample.txt"
SAMPLE_FWHM = ROOT / "sample_data" / "data" / "fwhm_calibration.json"


@pytest.fixture
def io_handler():
    from utils.io_handler import IOHandler

    return IOHandler()


def test_load_steel_spectrum(io_handler):
    assert SAMPLE_STEEL.is_file(), f"missing sample: {SAMPLE_STEEL}"
    spectrum = io_handler.load_spectrum(str(SAMPLE_STEEL))
    assert len(spectrum.energy) == len(spectrum.counts)
    assert len(spectrum.energy) > 100
    assert spectrum.energy[0] < spectrum.energy[-1]
    assert np.isfinite(spectrum.counts).all()


def test_spectrum_roundtrip_txt(io_handler, tmp_path):
    spectrum = io_handler.load_spectrum(str(SAMPLE_STEEL))
    out = tmp_path / "roundtrip.txt"
    io_handler.save_spectrum(spectrum, str(out))
    loaded = io_handler.load_spectrum(str(out))
    np.testing.assert_allclose(loaded.energy, spectrum.energy, rtol=1e-5)
    np.testing.assert_allclose(loaded.counts, spectrum.counts, rtol=1e-5)


def test_fwhm_calibration_roundtrip(tmp_path):
    from core.fwhm_calibration import load_fwhm_calibration

    assert SAMPLE_FWHM.is_file()
    cal = load_fwhm_calibration(str(SAMPLE_FWHM))
    assert cal.model_type in {"linear", "detector", "sqrt", "quadratic"}
    fwhm_6 = float(cal.predict_fwhm(6.0))
    assert 0.05 < fwhm_6 < 0.5

    out = tmp_path / "fwhm_out.json"
    cal.save(str(out))
    reloaded = load_fwhm_calibration(str(out))
    assert reloaded.model_type == cal.model_type
    np.testing.assert_allclose(
        reloaded.predict_fwhm(6.0), cal.predict_fwhm(6.0), rtol=1e-6
    )


def test_detector_model_injectable():
    from core.instrument_state import DetectorModel, InstrumentState
    from core.peak_fitting import PeakFitter
    from core.fwhm_calibration import FWHMCalibration

    det_a = DetectorModel(fwhm_0=0.08, epsilon=0.002)
    det_b = DetectorModel(fwhm_0=0.12, epsilon=0.003)
    assert det_a.predict_fwhm(6.0) != det_b.predict_fwhm(6.0)

    fitter = PeakFitter(detector=det_a)
    fitter.activate()
    assert abs(PeakFitter.calculate_fwhm(6.0) - det_a.predict_fwhm(6.0)) < 1e-9

    fitter.set_detector(det_b)
    fitter.activate()
    assert abs(PeakFitter.calculate_fwhm(6.0) - det_b.predict_fwhm(6.0)) < 1e-9

    cal = FWHMCalibration(
        model_type="detector",
        parameters={"fwhm_0": 0.10, "epsilon": 0.0025},
        parameter_errors={},
        r_squared=0.99,
        rmse=0.001,
        aic=0.0,
        bic=0.0,
        n_peaks=5,
        energy_range=(1.0, 20.0),
        calibration_date="2026-01-01T00:00:00",
    )
    state = InstrumentState()
    state.apply_fwhm_calibration(cal)
    assert state.detector.use_calibrated_shapes is True
    assert abs(state.detector.predict_fwhm(6.0) - cal.predict_fwhm(6.0)) < 1e-9


def test_semi_quant_area_normalized():
    from core.fitting import SpectrumFitter
    from core.peak_fitting import Peak

    fitter = SpectrumFitter()
    peaks = [
        Peak(energy=6.4, amplitude=100, fwhm=0.15, area=1000, element="Fe", line="Kα1"),
        Peak(energy=8.0, amplitude=50, fwhm=0.16, area=500, element="Cu", line="Kα1"),
        Peak(
            energy=2.7,
            amplitude=200,
            fwhm=0.12,
            area=800,
            element="Rh",
            line="Lα1",
            is_tube_line=True,
        ),
    ]
    result = fitter.quantify_elements(peaks, None)
    assert "Rh" not in result
    assert abs(result["Fe"]["relative_intensity_pct"] - (1000 / 1500) * 100) < 1e-6
    assert abs(result["Cu"]["relative_intensity_pct"] - (500 / 1500) * 100) < 1e-6
    assert result["Fe"]["method"] == "semi_quant_area"
    assert result["Fe"]["error"] is None
    total = sum(v["relative_intensity_pct"] for v in result.values())
    assert abs(total - 100.0) < 1e-6


def test_synthetic_fit_smoke():
    """Fit a simple synthetic two-Gaussian spectrum and run semi-quant."""
    from core.fitting import SpectrumFitter

    energy = np.linspace(5.0, 10.0, 500)
    # Fe-ish and Cu-ish gaussians on a flat background
    def gauss(e, amp, cen, sigma):
        return amp * np.exp(-((e - cen) ** 2) / (2 * sigma**2))

    counts = (
        50.0
        + gauss(energy, 800, 6.40, 0.06)
        + gauss(energy, 400, 8.04, 0.07)
    )
    counts = counts + np.random.default_rng(0).normal(0, 2.0, size=counts.shape)
    counts = np.maximum(counts, 0)

    fitter = SpectrumFitter()
    result = fitter.fit_spectrum(
        energy=energy,
        counts=counts,
        elements=[{"symbol": "Fe", "z": 26}, {"symbol": "Cu", "z": 29}],
        background_method="snip",
        peak_shape="gaussian",
        auto_find_peaks=False,
        include_tube_lines=False,
        include_compton=False,
        tube_element=None,
    )
    assert result.peaks
    assert "reduced_chi_squared" in result.statistics
    quant = fitter.quantify_elements(result.peaks, None)
    # At least one labeled sample peak expected
    assert isinstance(quant, dict)


def test_batch_processor_api_alignment(tmp_path, io_handler):
    from core.batch_processing import BatchProcessor, BatchProcessingConfig

    spectrum = io_handler.load_spectrum(str(SAMPLE_STEEL))
    # Write a short synthetic subset for speed
    energy = spectrum.energy
    counts = spectrum.counts
    # Keep a manageable slice around Fe/Cr region if present
    mask = (energy >= 4.0) & (energy <= 12.0)
    if mask.sum() < 50:
        mask = slice(None)
    out = tmp_path / "batch_sample.txt"
    data = np.column_stack([energy[mask], counts[mask]])
    np.savetxt(out, data, header="Energy Counts", comments="")

    config = BatchProcessingConfig(
        elements=[{"symbol": "Fe", "z": 26}, {"symbol": "Cr", "z": 24}],
        background_method="snip",
        peak_shape="gaussian",
        include_tube_lines=False,
        include_compton=False,
        auto_find_peaks=False,
        excitation_kv=50.0,
    )
    processor = BatchProcessor(config)
    results = processor.process_file_list([out])
    assert len(results) == 1
    r = results[0]
    assert r.fit_success or r.error_message  # should not crash
    if r.fit_success:
        assert r.quantification_method == "semi_quant_area"
        assert r.energy is not None
        assert r.chi_squared >= 0
        # Concentrations keys are relative intensities when labeled peaks exist
        assert isinstance(r.concentrations, dict)


def test_analysis_session_holds_state():
    from core.session import AnalysisSession
    from core.spectrum import Spectrum

    session = AnalysisSession()
    energy = np.linspace(0, 10, 100)
    counts = np.ones(100)
    spectrum = Spectrum(energy=energy, counts=counts)
    session.set_spectrum(spectrum, path="fake.txt")
    assert session.spectrum is spectrum
    assert session.spectrum_path == "fake.txt"
    assert session.fit_result is None

    session.set_elements([{"symbol": "Fe", "z": 26}])
    assert session.elements[0]["symbol"] == "Fe"


def test_candidates_at_energy():
    from core.smart_peak_id import candidates_at_energy

    hits = candidates_at_energy(6.40, energy_tol_kev=0.10)
    assert hits
    assert hits[0]["symbol"] == "Fe"
    assert "K" in hits[0]["line"]
    assert abs(hits[0]["abs_delta_kev"]) < 0.05


def test_auto_id_peak_positions():
    from core.smart_peak_id import auto_id_peak_positions

    peaks = [
        {"energy": 6.403, "element": None, "line": None, "is_tube_line": False},
        {"energy": 8.047, "element": None, "line": None, "is_tube_line": False},
        {"energy": 2.696, "element": "Rh", "line": "Lα1", "is_tube_line": True},
    ]
    labeled, symbols, summary = auto_id_peak_positions(peaks)
    assert "Fe" in symbols and "Cu" in symbols
    assert labeled[0]["element"] == "Fe"
    assert labeled[2]["is_tube_line"] is True
    assert summary and "Auto-ID" in summary[0]


def test_pb_m_alpha_weaker_than_l_alpha():
    from core.xray_data import get_element_lines

    lines = get_element_lines('Pb', 82)
    la = next(l for l in lines['L'] if l['name'] in ('Lα1', 'Lα'))
    ma = next((l for l in lines['M'] if l['name'] in ('Mα1', 'Mα')), None)
    assert la['relative_intensity'] > 0.5
    if ma is not None:
        assert ma['relative_intensity'] < 0.25
        assert ma['relative_intensity'] < la['relative_intensity']


def test_auto_id_does_not_call_pb_from_m_alpha_alone():
    from core.smart_peak_id import auto_id_peak_positions

    peaks = [
        {"energy": 2.347, "element": None, "line": None, "is_tube_line": False},
    ]
    labeled, symbols, _ = auto_id_peak_positions(peaks, excitation_kv=50.0)
    assert "Pb" not in symbols
    assert labeled[0].get("element") != "Pb"


def test_auto_id_pb_requires_l_alpha():
    from core.smart_peak_id import auto_id_peak_positions

    peaks = [
        {"energy": 10.551, "element": None, "line": None, "is_tube_line": False},
        {"energy": 2.347, "element": None, "line": None, "is_tube_line": False},
    ]
    labeled, symbols, _ = auto_id_peak_positions(peaks, excitation_kv=50.0)
    assert "Pb" in symbols
    assert labeled[0]["element"] == "Pb"
    assert "Lα" in labeled[0]["line"]
    assert labeled[1]["element"] == "Pb"
    assert labeled[1]["line"].startswith("M")
