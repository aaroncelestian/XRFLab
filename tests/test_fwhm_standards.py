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
from core.reference_composition import find_composition_csv, load_composition_csv


SAMPLE_DATA = Path(__file__).resolve().parents[1] / "sample_data" / "data"


def test_guess_element_from_filename():
    assert guess_element_from_stem("Fe") == "Fe"
    assert guess_element_from_stem("copper") == "Cu"
    assert guess_element_from_stem("cubic zirconia") == "Zr"
    assert guess_element_from_stem("Al") == "Al"
    assert guess_element_from_stem("NIST 2586") is None
    assert guess_element_from_stem("stainless steel") is None


def test_mixed_standard_names():
    assert is_likely_mixed_standard("NIST 2586")
    assert is_likely_mixed_standard("LKSD standard")
    assert is_likely_mixed_standard("PACS standard")
    assert not is_likely_mixed_standard("Fe")
    assert not is_likely_mixed_standard("cubic zirconia")


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
    assert "NIST 2586.txt" in by_name
    assert not by_name["NIST 2586.txt"].included
    peaks = assignments_to_file_peaks(files)
    assert "Fe.txt" in peaks
    assert "NIST 2586.txt" not in peaks


def test_load_nist_composition_csv_wt_percent():
    path = SAMPLE_DATA / "NIST_SRM_2586_elements.csv"
    conc = load_composition_csv(path)
    assert "Fe" in conc
    # 51610 mg/kg → 5.161 wt%
    assert conc["Fe"] == 5.161
    assert conc["Si"] > 20  # major


def test_find_composition_csv_does_not_require_same_filename():
    spectrum = SAMPLE_DATA / "NIST 2586.txt"
    found = find_composition_csv([spectrum], standard_name="NIST 2586")
    assert found is not None
    assert found.name == "NIST_SRM_2586_elements.csv"


def test_simple_element_concentration_csv(tmp_path):
    csv_path = tmp_path / "Fe.csv"
    csv_path.write_text("Element,Concentration\nFe,99.5\nSi,0.2\n", encoding="utf-8")
    conc = load_composition_csv(csv_path)
    assert conc["Fe"] == 99.5
    spec = tmp_path / "Fe.txt"
    spec.write_text("placeholder", encoding="utf-8")
    found = find_composition_csv([spec])
    assert found == csv_path
