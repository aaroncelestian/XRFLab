"""Round-trip tests for XRFLab .xrfp project files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.fitting import FitResult
from core.fp_quantification import FPQuantResult
from core.matrix_model import MatrixAssumptions, MatrixKind
from core.peak_fitting import Peak
from core.project_file import (
    FORMAT_VERSION,
    ProjectDocument,
    ProjectFileError,
    load_project,
    save_project,
)
from core.spectrum import Spectrum


def _spectrum(n=64):
    energy = np.linspace(0.0, 20.0, n)
    counts = np.exp(-((energy - 6.4) ** 2) / 0.1) * 1000.0 + 10.0
    return Spectrum(
        energy=energy,
        counts=counts,
        live_time=30.0,
        real_time=32.0,
        metadata={"name": "test", "excitation_energy": 50.0},
    )


def test_empty_project_roundtrip(tmp_path):
    path = tmp_path / "empty.xrfp"
    save_project(str(path), ProjectDocument())
    loaded = load_project(str(path))
    assert loaded.analysis["spectrum"] is None
    assert loaded.mapping_project is None
    assert loaded.batch["results"] == []
    assert loaded.composition == {} or loaded.composition.get("rows") in (None, [])


def test_analysis_fit_and_fp_roundtrip(tmp_path):
    spec = _spectrum()
    n = spec.num_channels
    peak = Peak(
        energy=6.403,
        amplitude=800.0,
        fwhm=0.14,
        area=1200.0,
        element="Fe",
        line="Kα1",
        shape="voigt",
        is_tube_line=False,
    )
    fit = FitResult(
        background=np.full(n, 10.0),
        fitted_spectrum=spec.counts.copy(),
        residuals=np.zeros(n),
        peaks=[peak],
        statistics={"chi_squared": 1.2, "r_squared": 0.99, "reduced_chi_squared": 1.1},
        tube_overlap_flags=[{"message": "ok"}],
    )
    matrix = MatrixAssumptions(kind=MatrixKind.OXIDE, fe_as="FeO", h2o_wt=2.5)
    fp = FPQuantResult(
        success=True,
        element_wt={"Fe": 40.0, "O": 60.0},
        formula_wt={"FeO": 100.0},
        concentrations={"Fe": {"concentration": 40.0, "role": "measured", "line": "Kα1"}},
        iterations=4,
        residual=0.01,
        assumptions=matrix,
        measured_cation_pct=40.0,
    )
    doc = ProjectDocument(
        analysis={
            "spectrum": spec,
            "spectrum_path": "/tmp/steel.txt",
            "elements": [{"symbol": "Fe", "z": 26}],
            "fit_result": fit,
            "concentrations": fp.concentrations,
            "quantification_method": "fp_matrix",
            "matrix": matrix,
            "fp_result": fp,
            "ui": {"left_tab": 2, "element_panel": {"sample_name": "basalt"}},
        }
    )
    path = tmp_path / "analysis.xrfp"
    save_project(str(path), doc)
    loaded = load_project(str(path))
    a = loaded.analysis
    np.testing.assert_allclose(a["spectrum"].energy, spec.energy)
    np.testing.assert_allclose(a["spectrum"].counts, spec.counts)
    assert a["spectrum_path"] == "/tmp/steel.txt"
    assert a["elements"][0]["symbol"] == "Fe"
    assert len(a["fit_result"].peaks) == 1
    assert a["fit_result"].peaks[0].element == "Fe"
    np.testing.assert_allclose(a["fit_result"].background, fit.background)
    assert a["matrix"].kind == MatrixKind.OXIDE
    assert a["matrix"].h2o_wt == pytest.approx(2.5)
    assert a["fp_result"].success
    assert a["fp_result"].element_wt["Fe"] == pytest.approx(40.0)
    assert a["ui"]["element_panel"]["sample_name"] == "basalt"


def test_mapping_project_roundtrip(tmp_path):
    from core.mapping.cube import SpectrumCube
    from core.mapping.models import (
        ElementMap,
        MappingFOV,
        MappingProject,
        MappingSample,
        MapSpectrum,
        OverviewImage,
    )

    em = ElementMap(name="Fe Ka1", data=np.arange(12, dtype=np.float32).reshape(3, 4))
    overview = OverviewImage(name="photo", data=np.zeros((3, 4, 3), dtype=np.uint8))
    ms = MapSpectrum(
        spectrum=_spectrum(32),
        name="sum",
        kind="sum",
        x=1.0,
        y=2.0,
    )
    cube = SpectrumCube(
        data=np.arange(8 * 3 * 4, dtype=np.uint16).reshape(8, 3, 4),
        ev_per_channel=10.0,
    )
    fov = MappingFOV(
        id="site1",
        name="Site 1",
        width=4,
        height=3,
        element_maps=[em],
        overview=overview,
        spectra=[ms],
        cube=cube,
        pixel_size_mm=0.02,
        stage_center_mm=(10.0, 20.0),
        metadata={"comment": "hello"},
    )
    sample = MappingSample(id="s1", name="Rock", sites=[fov])
    project = MappingProject(
        path="/tmp/demo.ipj",
        name="demo",
        samples=[sample],
        metadata={"n_cubes": 1},
    )
    doc = ProjectDocument(
        mapping_project=project,
        mapping_ui={"checked_map_names": ["Fe Ka1"], "rgb": True},
    )
    path = tmp_path / "map.xrfp"
    save_project(str(path), doc)
    loaded = load_project(str(path))
    proj = loaded.mapping_project
    assert proj.name == "demo"
    site = proj.fovs[0]
    assert site.id == "site1"
    np.testing.assert_array_equal(site.element_maps[0].data, em.data)
    assert site.spectra[0].kind == "sum"
    assert site.has_cube
    assert site.cube.data.shape == (8, 3, 4)
    assert site.stage_center_mm == (10.0, 20.0)
    assert loaded.mapping_ui["checked_map_names"] == ["Fe Ka1"]


def test_batch_results_roundtrip(tmp_path):
    from core.batch_processing import BatchFitResult

    result = BatchFitResult(
        spectrum_name="spot1",
        spectrum_path="ipj::spot1",
        fit_success=True,
        chi_squared=1.5,
        r_squared=0.98,
        elements_found=["Fe", "Ca"],
        concentrations={"Fe": 55.0, "Ca": 45.0},
        concentration_errors={},
        peak_areas={"Fe": {"Kα1": 1000.0}},
        fitted_spectrum=np.ones(16),
        residuals=np.zeros(16),
        energy=np.linspace(0, 10, 16),
        measured_counts=np.ones(16) * 3,
        element_contributions={"Fe": np.ones(16)},
        fit_time=0.4,
    )
    spec = _spectrum(16)
    doc = ProjectDocument(
        batch={
            "file_paths": ["spot1"],
            "memory_spectra": {"spot1": spec},
            "results": [result],
            "config": {"peak_shape": "voigt", "excitation_energy": 50.0},
            "ui": {"save_fits": True, "trend_elements": ["Fe"]},
        }
    )
    path = tmp_path / "batch.xrfp"
    save_project(str(path), doc)
    loaded = load_project(str(path))
    assert loaded.batch["file_paths"] == ["spot1"]
    np.testing.assert_allclose(
        loaded.batch["memory_spectra"]["spot1"].counts, spec.counts
    )
    br = loaded.batch["results"][0]
    assert br.spectrum_name == "spot1"
    assert br.concentrations["Fe"] == pytest.approx(55.0)
    np.testing.assert_allclose(br.fitted_spectrum, result.fitted_spectrum)
    assert "Fe" in br.element_contributions


def test_calibrations_and_composition_roundtrip(tmp_path):
    from core.composition import CompositionRow
    from core.fwhm_calibration import FWHMCalibration
    from core.tube_profile import TubeProfileLibrary, default_tube_profile

    fwhm = FWHMCalibration(
        model_type="detector",
        parameters={"fwhm_0": 0.09, "epsilon": 0.002},
        parameter_errors={"fwhm_0": 0.001, "epsilon": 0.0001},
        r_squared=0.99,
        rmse=0.005,
        aic=1.0,
        bic=2.0,
        n_peaks=12,
        energy_range=(1.0, 15.0),
        calibration_date="2026-08-13",
    )
    lib = TubeProfileLibrary()
    lib.set_profile(default_tube_profile("Rh", 50.0))
    row = CompositionRow(
        name="a_1",
        source_id="/tmp/a_1.txt",
        sample="a",
        values={"Fe": 10.0, "Si": 20.0},
    )
    doc = ProjectDocument(
        calibrations={"fwhm": fwhm.to_dict(), "tube": lib.to_dict()},
        composition={
            "rows": [row.to_dict()],
            "group_mode": "auto",
            "oxides": True,
        },
        window={"tab": 3, "log_y": True, "grid": False},
    )
    path = tmp_path / "misc.xrfp"
    save_project(str(path), doc)
    loaded = load_project(str(path))
    fwhm2 = FWHMCalibration.from_dict(loaded.calibrations["fwhm"])
    assert fwhm2.model_type == "detector"
    assert tuple(fwhm2.energy_range) == (1.0, 15.0)
    lib2 = TubeProfileLibrary.from_dict(loaded.calibrations["tube"])
    assert lib2.get_profile(50.0) is not None
    assert loaded.composition["rows"][0]["sample"] == "a"
    assert loaded.window["tab"] == 3


def test_rejects_wrong_format(tmp_path):
    import h5py

    path = tmp_path / "bogus.xrfp"
    with h5py.File(path, "w") as f:
        f.attrs["format_id"] = "NOPE"
        f.attrs["format_version"] = FORMAT_VERSION
    with pytest.raises(ProjectFileError):
        load_project(str(path))


def test_peak_dict_roundtrip():
    peak = Peak(
        energy=8.04,
        amplitude=10.0,
        fwhm=0.15,
        area=50.0,
        element="Cu",
        line="Kα1",
        shape_params={"sigma": np.float64(0.06)},
        fixed_fwhm=0.2,
    )
    restored = Peak.from_dict(peak.to_dict())
    assert restored.element == "Cu"
    assert restored.fixed_fwhm == pytest.approx(0.2)
    assert restored.shape_params["sigma"] == pytest.approx(0.06)
