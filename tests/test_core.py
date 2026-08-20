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


def test_fit_spectrum_accepts_bare_element_symbols():
    from core.fitting import SpectrumFitter

    energy = np.linspace(5.0, 10.0, 400)
    counts = np.full_like(energy, 40.0)
    counts += 600.0 * np.exp(-((energy - 6.40) ** 2) / (2 * 0.06**2))
    fitter = SpectrumFitter()
    # Line-scan tab passes symbols, not {'symbol','z'} dicts
    positions = fitter.build_peak_positions(
        energy,
        counts_bg_subtracted=counts,
        elements=["Fe", "As"],
        auto_find_peaks=False,
        include_tube_lines=False,
        include_compton=False,
        tube_element=None,
    )
    labeled = {p["element"] for p in positions if p.get("element")}
    assert "Fe" in labeled
    result = fitter.fit_spectrum(
        energy=energy,
        counts=counts,
        elements=["Fe", "As"],
        background_method="snip",
        peak_shape="gaussian",
        auto_find_peaks=False,
        include_tube_lines=False,
        include_compton=False,
        tube_element=None,
    )
    assert result.peaks
    quant = fitter.quantify_elements(result.peaks)
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
    fe_kb = [
        p for p in labeled
        if p.get("element") == "Fe" and p.get("line") and "β" in p["line"]
    ]
    assert fe_kb, "Fe Kβ should be added even without a detected peak"
    assert fe_kb[0].get("inferred") is True


def test_auto_id_adds_companion_lines_below_threshold():
    from core.smart_peak_id import auto_id_peak_positions

    peaks = [
        {"energy": 6.403, "element": None, "line": None, "is_tube_line": False},
    ]
    labeled, symbols, summary = auto_id_peak_positions(peaks)
    assert "Fe" in symbols
    lines = {(p.get("element"), p.get("line")) for p in labeled}
    assert ("Fe", "Kα1") in lines or ("Fe", "Kα") in lines
    assert any(el == "Fe" and line and "β" in line for el, line in lines)
    inferred = [p for p in labeled if p.get("inferred")]
    assert inferred
    assert any("expected line" in s for s in summary)
    assert not any(
        p.get("element") == "Fe" and p.get("line", "").startswith("L")
        for p in labeled
    )


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
    pb_lb = [
        p for p in labeled
        if p.get("element") == "Pb" and p.get("line") and "Lβ" in p["line"]
    ]
    assert pb_lb, "Pb Lβ should be shown even if it was below the peak-find threshold"


def test_auto_id_finds_hg_l_lines():
    from core.smart_peak_id import COMMON_XRF_SYMBOLS, auto_id_peak_positions

    assert "Hg" in COMMON_XRF_SYMBOLS
    # Measured peaks matching the user's spectrum (Hg Lα ~9.989 keV, Lβ ~11.823 keV)
    peaks = [
        {"energy": 9.970, "element": None, "line": None, "is_tube_line": False},
        {"energy": 11.830, "element": None, "line": None, "is_tube_line": False},
    ]
    labeled, symbols, _ = auto_id_peak_positions(peaks, excitation_kv=50.0)
    assert "Hg" in symbols
    assert labeled[0]["element"] == "Hg"
    assert "Lα" in labeled[0]["line"]
    assert labeled[1]["element"] == "Hg"
    assert "Lβ" in labeled[1]["line"]


def test_batch_processor_in_memory_spectrum(io_handler):
    from core.batch_processing import BatchProcessor, BatchProcessingConfig

    spectrum = io_handler.load_spectrum(str(SAMPLE_STEEL))
    spectrum.metadata["name"] = "steel_in_memory"
    config = BatchProcessingConfig(
        elements=[{"symbol": "Fe", "z": 26}],
        background_method="snip",
        peak_shape="gaussian",
        include_tube_lines=False,
        include_compton=False,
        auto_find_peaks=False,
        excitation_kv=50.0,
    )
    processor = BatchProcessor(config)
    results = processor.process_spectrum_list([spectrum])
    assert len(results) == 1
    assert results[0].spectrum_name == "steel_in_memory"
    assert results[0].fit_success or results[0].error_message


def test_sanitize_sample_name():
    from core.batch_processing import sanitize_sample_name

    assert sanitize_sample_name("  Basalt 01  ") == "Basalt 01"
    assert sanitize_sample_name('a/b:c*d') == "a_b_c_d"
    assert sanitize_sample_name("...") == ""


def test_rename_files_in_place(tmp_path):
    from core.batch_processing import rename_files_in_place

    a = tmp_path / "scan001.txt"
    b = tmp_path / "scan002.txt"
    a.write_text("one")
    b.write_text("two")
    rename_files_in_place(
        [(a, tmp_path / "Basalt01_1.txt"), (b, tmp_path / "Basalt01_2.txt")]
    )
    assert (tmp_path / "Basalt01_1.txt").read_text() == "one"
    assert (tmp_path / "Basalt01_2.txt").read_text() == "two"
    assert not a.exists()
    assert not b.exists()


def test_rename_files_in_place_swap(tmp_path):
    from core.batch_processing import rename_files_in_place

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A")
    b.write_text("B")
    rename_files_in_place([(a, tmp_path / "b.txt"), (b, tmp_path / "a.txt")])
    assert (tmp_path / "a.txt").read_text() == "B"
    assert (tmp_path / "b.txt").read_text() == "A"


def test_rename_files_in_place_collision(tmp_path):
    from core.batch_processing import rename_files_in_place

    src = tmp_path / "scan.txt"
    other = tmp_path / "taken.txt"
    src.write_text("data")
    other.write_text("nope")
    try:
        rename_files_in_place([(src, other)])
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
    assert src.read_text() == "data"


def test_mapping_project_spectra_only_helpers():
    from core.mapping.models import MappingFOV, MappingProject, MapSpectrum
    from core.spectrum import Spectrum

    energy = np.linspace(0.0, 10.0, 64)
    counts = np.ones(64)
    spec = Spectrum(energy=energy, counts=counts, metadata={"name": "Spectrum 1"})
    spot = MapSpectrum(spectrum=spec, name="Spectrum 1", kind="spot")
    summed = MapSpectrum(
        spectrum=Spectrum(energy=energy, counts=counts * 2, metadata={"name": "Sum Spectrum"}),
        name="Sum Spectrum",
        kind="sum",
    )
    site = MappingFOV(id="s1", name="Site 1", spectra=[spot, summed])
    project = MappingProject(path="points.ipj", fovs=[site])
    assert project.is_spectra_only()
    points = project.point_spectra()
    assert [p.name for p in points] == ["Spectrum 1"]
    assert not project.has_spatial_data()

    mapped = MappingFOV(
        id="s2",
        name="Site 2",
        spectra=[spot],
        element_maps=[],
    )
    mapped.element_maps = []
    # A site with only spectra is still spectra-only; add a dummy map via spatial flag
    from core.mapping.models import ElementMap

    mapped.element_maps.append(
        ElementMap(name="Fe Ka1", data=np.zeros((4, 4)))
    )
    mapped_project = MappingProject(path="maps.ipj", fovs=[mapped])
    assert mapped_project.has_spatial_data()
    assert not mapped_project.is_spectra_only()


def test_find_fov_for_multipoint_spectrum():
    from core.mapping.models import LineScan, MappingFOV, MappingProject, MapSpectrum
    from core.spectrum import Spectrum

    energy = np.linspace(0.0, 10.0, 64)
    points = [
        MapSpectrum(
            spectrum=Spectrum(energy=energy, counts=np.ones(64) * (i + 1)),
            name=f"Pt {i}",
            x=float(i),
            y=0.0,
            index=i,
            kind="line_point",
        )
        for i in range(3)
    ]
    series = LineScan(name="Multipoint 1", points=points, kind="multipoint")
    site = MappingFOV(id="s1", name="Site 1", line_scans=[series])
    project = MappingProject(path="multi.ipj", fovs=[site])

    found = project.find_fov_for_spectrum(points[1])
    assert found is site
    assert project.find_fov_for_spectrum(points[0]) is site


def test_site_contents_tags():
    from core.mapping.models import (
        ElementMap,
        LineScan,
        MappingFOV,
        MapSpectrum,
    )
    from core.spectrum import Spectrum

    energy = np.linspace(0.0, 10.0, 16)
    spec = Spectrum(energy=energy, counts=np.ones(16))
    spot = MapSpectrum(spectrum=spec, name="Spectrum 1", kind="spot")
    summed = MapSpectrum(
        spectrum=Spectrum(energy=energy, counts=np.ones(16) * 2),
        name="Sum Spectrum",
        kind="sum",
    )
    empty = MappingFOV(id="e", name="Empty", spectra=[summed])
    assert empty.contents_tags() == []

    mapped = MappingFOV(
        id="m",
        name="Mapped",
        element_maps=[ElementMap(name="Fe Ka1", data=np.zeros((4, 4)))],
        spectra=[spot, summed],
    )
    assert mapped.contents_label() == "1 map · 1 spot"

    line_pts = [
        MapSpectrum(
            spectrum=Spectrum(energy=energy, counts=np.ones(16) * (i + 1)),
            name=f"Pt {i}",
            kind="line_point",
        )
        for i in range(5)
    ]
    multi_pts = [
        MapSpectrum(
            spectrum=Spectrum(energy=energy, counts=np.ones(16) * (i + 1)),
            name=f"M {i}",
            kind="line_point",
        )
        for i in range(3)
    ]
    mixed = MappingFOV(
        id="x",
        name="Mixed",
        element_maps=[
            ElementMap(name="Ca Ka1", data=np.zeros((2, 2))),
            ElementMap(name="Fe Ka1", data=np.zeros((2, 2))),
        ],
        line_scans=[
            LineScan(name="Line 1", points=line_pts, kind="line_scan"),
            LineScan(name="Multi 1", points=multi_pts, kind="multipoint"),
        ],
        metadata={"has_smartmap": True},
    )
    assert mixed.contents_tags() == [
        "SmartMap",
        "2 maps",
        "5-pt line",
        "3-pt multi",
    ]


def test_find_peaks_rejects_below_min_energy():
    from core.peak_fitting import PeakFitter

    energy = np.linspace(-0.40, 2.0, 241)
    counts = np.zeros_like(energy)
    # Electronic zero/noise near 0 keV, C Kα, Na Kα
    for e0, amp, sig in ((0.00, 800.0, 0.04), (0.277, 200.0, 0.04), (1.041, 300.0, 0.04)):
        counts += amp * np.exp(-((energy - e0) ** 2) / (2 * sig**2))

    peaks = PeakFitter.find_peaks(energy, counts, prominence=20)
    peak_e = [e for e, _ in peaks]
    assert all(e >= PeakFitter.MIN_PEAK_ENERGY_KEV for e in peak_e)
    assert not any(e < 0.17 for e in peak_e)
    assert any(abs(e - 0.277) < 0.05 for e in peak_e)
    assert any(abs(e - 1.041) < 0.05 for e in peak_e)


def test_fit_single_peak_rejects_below_min_energy():
    from core.peak_fitting import PeakFitter

    energy = np.linspace(-0.2, 0.4, 120)
    counts = 500.0 * np.exp(-((energy - 0.05) ** 2) / (2 * 0.03**2))
    assert PeakFitter.fit_single_peak(energy, counts, 0.05) is None


def test_build_peak_positions_keeps_c_drops_be():
    from core.fitting import SpectrumFitter

    energy = np.linspace(-0.40, 5.0, 540)
    counts = np.ones_like(energy)
    fitter = SpectrumFitter()
    positions = fitter.build_peak_positions(
        energy,
        counts_bg_subtracted=counts,
        elements=["Be", "C", "Na"],
        auto_find_peaks=False,
        include_tube_lines=False,
        include_compton=False,
        tube_element=None,
    )
    symbols = {p["element"] for p in positions if p.get("element")}
    assert "Be" not in symbols
    assert "C" in symbols
    assert "Na" in symbols
    assert all(p["energy"] >= 0.17 for p in positions)
