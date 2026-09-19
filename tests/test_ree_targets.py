"""REE suite + spectroscopic overlap partners for targeted quantification."""

from core.peak_fitting import PeakFitter
from core.ree_targets import (
    REE_LANTHANIDES,
    apply_ree_roles,
    build_ree_fit_set,
    find_ree_overlaps,
    is_ree,
    ree_elements,
)


def _fwhm():
    PeakFitter.activate(PeakFitter())
    return PeakFitter.calculate_fwhm


def test_ree_elements_are_y_and_lanthanides_without_pm():
    symbols = ree_elements()
    assert symbols[0] == "Y"
    assert "Sc" not in symbols
    assert list(REE_LANTHANIDES) == [s for s in symbols if s in REE_LANTHANIDES]
    assert "Pm" not in symbols
    assert is_ree("La")
    assert is_ree("Y")
    assert not is_ree("Sc")
    assert not is_ree("Fe")
    assert not is_ree("Ba")
    assert "Sc" in ree_elements(include_sc=True)


def test_50kv_ree_overlaps_include_classic_interferents():
    hits = find_ree_overlaps(ree_elements(), excitation_kv=50.0, fwhm_fn=_fwhm())
    partners = {h.element for h in hits}
    # Ba L / Ti K sit on La–Ce L; Fe Kα on Dy L; Rb Kβ on Y K
    assert "Ba" in partners
    assert "Ti" in partners
    assert "Fe" in partners
    assert "Rb" in partners
    # Matrix majors stay out unless they sit on a REE line
    assert "Si" not in partners
    assert "Al" not in partners
    assert "Ca" not in partners
    assert "Pb" not in partners


def test_overlap_hits_name_the_conflicting_lines():
    hits = find_ree_overlaps(["La", "Dy", "Y"], excitation_kv=50.0, fwhm_fn=_fwhm())
    ba_la = [h for h in hits if h.element == "Ba" and h.target == "La"]
    fe_dy = [h for h in hits if h.element == "Fe" and h.target == "Dy"]
    rb_y = [h for h in hits if h.element == "Rb" and h.target == "Y"]
    assert ba_la
    assert fe_dy
    assert rb_y
    assert any("L" in h.element_line for h in ba_la)
    assert any(h.element_line.startswith("K") for h in fe_dy)


def test_calibration_cluster_adds_missing_partner():
    hits = find_ree_overlaps(
        ["La"],
        excitation_kv=50.0,
        fwhm_fn=_fwhm(),
        candidates=[],  # no spectroscopic search
        extra_clusters=[["Ba", "La"]],
    )
    assert any(h.element == "Ba" and h.source == "calibration" for h in hits)


def test_build_ree_fit_set_orders_rees_then_overlaps():
    fit_set = build_ree_fit_set(excitation_kv=50.0, fwhm_fn=_fwhm())
    assert fit_set.rees == ree_elements()
    assert fit_set.overlaps
    assert fit_set.symbols[: len(fit_set.rees)] == fit_set.rees
    assert "Ba" in fit_set.overlaps
    assert "La" not in fit_set.overlaps
    assert "overlap" in fit_set.summary()


def test_apply_ree_roles_sorts_and_tags():
    fit_set = build_ree_fit_set(excitation_kv=50.0, fwhm_fn=_fwhm())
    raw = {
        "Fe": {"concentration": 2.0, "error": 0.1},
        "La": {"concentration": 0.10, "error": 0.01},
        "Ba": {"concentration": 0.07, "error": 0.01},
        "Ce": {"concentration": 0.14, "error": 0.02},
    }
    tagged = apply_ree_roles(raw, fit_set)
    assert list(tagged) == ["La", "Ce", "Fe", "Ba"]
    assert tagged["La"]["role"] == "ree"
    assert tagged["Ce"]["role"] == "ree"
    assert tagged["Ba"]["role"] == "overlap"
    assert tagged["Fe"]["role"] == "overlap"
