"""Tests for merging multipoint / line-scan IPJ projects."""

from pathlib import Path

import numpy as np
import pytest

from core.mapping.merge import (
    composite_label,
    is_default_sample_name,
    is_default_site_name,
    merge_ipj_line_scans,
    merge_line_scan_projects,
    sanitize_name_token,
)
from core.mapping.models import LineScan, MappingFOV, MappingProject, MappingSample, MapSpectrum
from core.spectrum import Spectrum

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data" / "data"
IPJ_FILES = {
    "barstow": SAMPLE_DIR / "barstow1.ipj",
    "dylan": SAMPLE_DIR / "dylan_corsetti_slide_1_STROM.ipj",
}


def _spec(name: str = "Spectrum 1") -> MapSpectrum:
    energy = np.linspace(0.0, 20.0, 64)
    counts = np.ones(64, dtype=np.float64)
    return MapSpectrum(
        spectrum=Spectrum(energy=energy, counts=counts, live_time=1.0, real_time=1.0),
        name=name,
        x=1.0,
        y=2.0,
        index=1,
        kind="line_point",
    )


def _project_with_multipoint(
    path: str,
    *,
    sample_name: str = "Sample 1",
    site_name: str = "Site of Interest 1",
    n_points: int = 3,
) -> MappingProject:
    points = [_spec(f"Spectrum {i}") for i in range(1, n_points + 1)]
    for i, p in enumerate(points, start=1):
        p.index = i
        p.x = float(i)
        p.y = 0.0
    ls = LineScan(
        name=f"Multipoint ({n_points} points)",
        points=points,
        source="ipj",
        kind="multipoint",
    )
    site = MappingFOV(
        id="fov1",
        name=site_name,
        spectra=list(points),
        line_scans=[ls],
    )
    sample = MappingSample(id="s1", name=sample_name, sites=[site])
    return MappingProject(path=path, name=Path(path).stem, samples=[sample])


def test_sanitize_and_defaults():
    assert sanitize_name_token("Marble Canyon north") == "Marble_Canyon_north"
    assert is_default_sample_name("Sample 1")
    assert is_default_sample_name("Sample 12")
    assert not is_default_sample_name("Barstow tufa")
    assert is_default_site_name("Site 1")
    assert is_default_site_name("Site of Interest 2")
    assert not is_default_site_name("Vein east")


def test_composite_label_four_parts():
    label = composite_label(
        "Marble Canyon",
        "Sample 1",
        "Site of Interest 1",
        "Spectrum 3",
    )
    assert label == "Marble_Canyon_Sample_1_Site_of_Interest_1_Spectrum_3"


def test_merge_flattens_to_one_sample():
    a = _project_with_multipoint("/data/siteA.ipj", sample_name="Sample 1")
    b = _project_with_multipoint(
        "/data/siteB.ipj",
        sample_name="Outcrop B",
        site_name="Vein",
        n_points=4,
    )
    # Site without series must be skipped
    empty = MappingProject(
        path="/data/maps_only.ipj",
        samples=[
            MappingSample(
                id="s",
                name="Sample 1",
                sites=[MappingFOV(id="empty", name="Site of Interest 1")],
            )
        ],
    )
    merged = merge_line_scan_projects([a, b, empty], name="Campaign")
    assert len(merged.samples) == 1
    assert merged.samples[0].name == "Merged"
    assert len(merged.fovs) == 2
    assert merged.metadata["format"] == "merged_ipj_line_scans"
    assert merged.metadata["n_source_files"] == 3
    assert merged.metadata["skipped_sites_without_series"] == 1

    names = {s.name for s in merged.fovs}
    assert "siteA_Sample_1_Site_of_Interest_1" in names
    assert "siteB_Outcrop_B_Vein" in names

    point_names = [ms.name for ms in merged.all_spectra()]
    assert any(n.startswith("siteA_Sample_1_Site_of_Interest_1_Spectrum_") for n in point_names)
    assert any(n.startswith("siteB_Outcrop_B_Vein_Spectrum_") for n in point_names)
    assert len(set(point_names)) == len(point_names)
    assert not any(s.element_maps or s.cube is not None for s in merged.fovs)
    assert merged.has_line_scans()


def test_merge_requires_series():
    empty = MappingProject(
        path="/data/empty.ipj",
        samples=[
            MappingSample(
                id="s",
                name="Sample 1",
                sites=[MappingFOV(id="e", name="Site of Interest 1")],
            )
        ],
    )
    with pytest.raises(ValueError, match="No line scans"):
        merge_line_scan_projects([empty])


def _have_olefile():
    try:
        import olefile  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_olefile(), reason="olefile not installed")
def test_merge_real_ipj_multipoint_files():
    paths = [IPJ_FILES["barstow"], IPJ_FILES["dylan"]]
    if not all(p.exists() for p in paths):
        pytest.skip("sample .ipj files not present")
    merged = merge_ipj_line_scans(paths, name="barstow_dylan_merged")
    assert len(merged.samples) == 1
    assert len(merged.fovs) >= 2
    assert all(site.line_scans for site in merged.fovs)
    assert all("_" in ms.name for ms in merged.all_spectra())
    # File stem appears in every spectrum name
    assert all(
        ms.name.startswith("barstow1_") or ms.name.startswith("dylan_corsetti_slide_1_STROM_")
        for ms in merged.all_spectra()
    )
