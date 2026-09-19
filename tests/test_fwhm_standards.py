"""Tests for FWHM foil discovery and certified-composition CSV matching."""

from pathlib import Path

from core.fwhm_standards import (
    assignments_to_file_peaks,
    example_standards_dir,
    fwhm_lines_for_element,
    guess_element_from_stem,
    is_likely_mixed_standard,
    scan_fwhm_folder,
)
from core.reference_composition import (
    discover_standards_from_folders,
    find_composition_csv,
    load_composition_csv,
)


ROOT = Path(__file__).resolve().parents[1]
STANDARDS = ROOT / "sample_data" / "STANDARDS"
NIST_2586_DIR = STANDARDS / "NIST_SRM_2586"


def test_guess_element_from_filename():
    assert guess_element_from_stem("Fe") == "Fe"
    assert guess_element_from_stem("copper") == "Cu"
    assert guess_element_from_stem("cubic zirconia") == "Zr"
    assert guess_element_from_stem("Al") == "Al"
    assert guess_element_from_stem("NIST 2586") is None
    assert guess_element_from_stem("stainless steel") is None


def test_mixed_standard_names():
    assert is_likely_mixed_standard("NIST 2586")
    assert is_likely_mixed_standard("NIST_SRM_2586_01")
    assert is_likely_mixed_standard("LKSD standard")
    assert is_likely_mixed_standard("LKSD_1_01")
    assert is_likely_mixed_standard("PACS standard")
    assert is_likely_mixed_standard("OREAS_460b_01")
    assert is_likely_mixed_standard("STSD_2_01")
    assert is_likely_mixed_standard("TILL_1_01")
    assert not is_likely_mixed_standard("Fe")
    assert not is_likely_mixed_standard("cubic zirconia")
    assert not is_likely_mixed_standard("cubic_zirconia")


def test_fwhm_lines_use_k_not_l_for_zirconium():
    lines = fwhm_lines_for_element("Zr", include_holder_al=False)
    names = [name for name, _e in lines]
    assert any("Kα" in n for n in names)
    assert not any(" L" in n for n in names)


def test_scan_example_folder_includes_foils_skips_nist():
    folder = example_standards_dir()
    assert folder is not None
    files = scan_fwhm_folder(folder, include_holder_al=True)
    by_name = {item.filename: item for item in files}
    assert "Fe.txt" in by_name
    assert by_name["Fe.txt"].included
    assert by_name["Fe.txt"].element == "Fe"
    assert not any("NIST" in name for name in by_name)
    peaks = assignments_to_file_peaks(files)
    assert "Fe.txt" in peaks
    assert not any("NIST" in name for name in peaks)

    crm_files = scan_fwhm_folder(STANDARDS, include_holder_al=True)
    crm_names = {item.filename for item in crm_files}
    assert "foils/Fe.txt" in crm_names
    assert not any("NIST_SRM_2586" in name for name in crm_names)


def test_load_nist_composition_csv_wt_percent():
    path = NIST_2586_DIR / "NIST_SRM_2586_elements.csv"
    conc = load_composition_csv(path)
    assert "Fe" in conc
    # 51610 mg/kg → 5.161 wt%
    assert conc["Fe"] == 5.161
    assert conc["Si"] > 20  # major


def test_find_composition_csv_does_not_require_same_filename():
    spectrum = NIST_2586_DIR / "NIST_SRM_2586_01.txt"
    found = find_composition_csv([spectrum], standard_name="NIST_SRM_2586")
    assert found is not None
    assert found.name == "NIST_SRM_2586_elements.csv"


def test_find_composition_csv_matches_replicate_stem():
    spectrum = NIST_2586_DIR / "NIST_SRM_2586_03.txt"
    found = find_composition_csv([spectrum])
    assert found is not None
    assert found.name == "NIST_SRM_2586_elements.csv"


def test_discover_standards_from_parent_and_children():
    imported, skipped = discover_standards_from_folders([STANDARDS])
    names = {item.name for item in imported}
    assert "NIST_SRM_2586" in names
    assert "OREAS_460b" in names
    assert "MBH_REE_LO_22_P" in names
    assert "TILL_1" in names
    assert "foils" not in names
    assert "alloys" not in names
    nist = next(item for item in imported if item.name == "NIST_SRM_2586")
    assert len(nist.spectrum_paths) == 6
    assert nist.concentrations["Fe"] == 5.161
    skipped_names = {path.name for path, _reason in skipped}
    assert "foils" in skipped_names
    assert "alloys" in skipped_names

    two, _skipped = discover_standards_from_folders([
        STANDARDS / "OREAS_460b",
        STANDARDS / "NIST_SRM_2587",
    ])
    assert [item.name for item in two] == ["NIST_SRM_2587", "OREAS_460b"]

    mixed, _skipped = discover_standards_from_folders([
        STANDARDS,
        STANDARDS / "OREAS_460b",
    ])
    assert len([item for item in mixed if item.name == "OREAS_460b"]) == 1


def test_simple_element_concentration_csv(tmp_path):
    csv_path = tmp_path / "Fe.csv"
    csv_path.write_text("Element,Concentration\nFe,99.5\nSi,0.2\n", encoding="utf-8")
    conc = load_composition_csv(csv_path)
    assert conc["Fe"] == 99.5
    spec = tmp_path / "Fe.txt"
    spec.write_text("placeholder", encoding="utf-8")
    found = find_composition_csv([spec])
    assert found == csv_path
