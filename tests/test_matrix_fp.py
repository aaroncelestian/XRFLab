"""Tests for matrix expansion and iterative FP quantification."""

from __future__ import annotations

from core.fp_quantification import (
    FPQuantResult,
    normalize_line,
    quantify_from_peaks,
)
from core.fundamental_parameters import FundamentalParameters
from core.matrix_model import (
    MatrixAssumptions,
    MatrixKind,
    atomic_weight,
    expand_composition,
    extras_per_cation_mass,
    formula_for,
    parse_formula,
)
from core.peak_fitting import Peak


def test_parse_formula_parentheses():
    assert parse_formula("SiO2") == {"Si": 1, "O": 2}
    assert parse_formula("Al2O3") == {"Al": 2, "O": 3}
    assert parse_formula("Al(OH)3") == {"Al": 1, "O": 3, "H": 3}
    assert parse_formula("FeOOH") == {"Fe": 1, "O": 2, "H": 1}
    assert parse_formula("CaCO3") == {"Ca": 1, "C": 1, "O": 3}


def test_galena_measured_only_no_oxygen():
    assumptions = MatrixAssumptions(kind=MatrixKind.MEASURED)
    element_wt, formula_wt = expand_composition({"Pb": 86.6, "S": 13.4}, assumptions)
    assert "O" not in element_wt
    assert "H" not in element_wt
    assert abs(sum(element_wt.values()) - 100.0) < 1e-6
    assert abs(element_wt["Pb"] - 86.6) < 0.05
    assert abs(element_wt["S"] - 13.4) < 0.05
    assert "Pb" in formula_wt and "S" in formula_wt


def test_quartz_oxide_stoichiometry():
    assumptions = MatrixAssumptions(kind=MatrixKind.OXIDE)
    element_wt, formula_wt = expand_composition({"Si": 1.0}, assumptions)
    aw_si = atomic_weight("Si")
    aw_o = atomic_weight("O")
    si_frac = aw_si / (aw_si + 2.0 * aw_o) * 100.0
    assert abs(element_wt["Si"] - si_frac) < 0.05
    assert abs(element_wt["O"] - (100.0 - si_frac)) < 0.05
    assert abs(formula_wt["SiO2"] - 100.0) < 1e-6


def test_h2o_knob_dilutes_cations():
    dry = MatrixAssumptions(kind=MatrixKind.OXIDE, h2o_wt=0.0)
    wet = MatrixAssumptions(kind=MatrixKind.OXIDE, h2o_wt=10.0)
    dry_el, dry_f = expand_composition({"Si": 1.0}, dry)
    wet_el, wet_f = expand_composition({"Si": 1.0}, wet)
    assert abs(wet_f["H2O"] - 10.0) < 1e-6
    assert abs(wet_f["SiO2"] - 90.0) < 1e-6
    assert wet_el["Si"] < dry_el["Si"]
    assert wet_el["H"] > 0
    assert abs(sum(wet_el.values()) - 100.0) < 1e-6


def test_oh_and_co2_knobs():
    assumptions = MatrixAssumptions(
        kind=MatrixKind.MEASURED, oh_wt=5.0, co2_wt=8.0
    )
    element_wt, formula_wt = expand_composition({"Pb": 1.0, "S": 1.0}, assumptions)
    assert abs(formula_wt["OH"] - 5.0) < 1e-6
    assert abs(formula_wt["CO2"] - 8.0) < 1e-6
    assert element_wt["H"] > 0 and element_wt["C"] > 0 and element_wt["O"] > 0
    assert abs(sum(element_wt.values()) - 100.0) < 1e-6


def test_carbonate_adds_carbon():
    assumptions = MatrixAssumptions(kind=MatrixKind.CARBONATE)
    element_wt, formula_wt = expand_composition({"Ca": 1.0}, assumptions)
    assert element_wt["C"] > 0
    assert abs(formula_wt["CaCO3"] - 100.0) < 1e-6
    extras = extras_per_cation_mass("Ca", "CaCO3")
    assert "C" in extras and "O" in extras


def test_hydroxide_adds_hydrogen():
    assumptions = MatrixAssumptions(kind=MatrixKind.HYDROXIDE)
    element_wt, formula_wt = expand_composition({"Fe": 1.0}, assumptions)
    assert formula_for("Fe", assumptions) == "FeOOH"
    assert element_wt["H"] > 0
    assert "FeOOH" in formula_wt


def test_light_sum_too_high_raises():
    assumptions = MatrixAssumptions(kind=MatrixKind.OXIDE, h2o_wt=100.0)
    try:
        expand_composition({"Si": 1.0}, assumptions)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_normalize_line():
    assert normalize_line("Kα1") == "Kα1"
    assert normalize_line("Ka") == "Kα1"
    assert normalize_line("KA1") == "Kα1"
    assert normalize_line("L") == "Lα1"
    assert normalize_line("Lb1") == "Lβ1"


def _peak(element, line, area, energy, tube=False):
    return Peak(
        energy=energy,
        amplitude=area,
        fwhm=0.15,
        area=area,
        element=element,
        line=line,
        is_tube_line=tube,
    )


def test_fp_ignores_tube_lines():
    peaks = [
        _peak("Fe", "Kα1", 1000, 6.4),
        _peak("Rh", "Lα1", 800, 2.7, tube=True),
    ]
    result = quantify_from_peaks(
        peaks,
        MatrixAssumptions(kind=MatrixKind.MEASURED),
        {"excitation_energy": 50.0, "incident_angle": 45.0},
        tube_element="Rh",
        sample_contains_tube_element=False,
    )
    assert result.success
    assert "Fe" in result.element_wt
    assert "Rh" not in result.element_wt


def test_fp_excludes_sample_rh_without_opt_in():
    peaks = [
        _peak("Fe", "Kα1", 1000, 6.4),
        _peak("Rh", "Kα1", 500, 20.2, tube=False),
    ]
    excluded = quantify_from_peaks(
        peaks,
        MatrixAssumptions(kind=MatrixKind.MEASURED),
        {"excitation_energy": 50.0, "incident_angle": 45.0},
        tube_element="Rh",
        sample_contains_tube_element=False,
    )
    assert excluded.success
    assert "Rh" not in excluded.element_wt

    included = quantify_from_peaks(
        peaks,
        MatrixAssumptions(kind=MatrixKind.MEASURED),
        {"excitation_energy": 50.0, "incident_angle": 45.0},
        tube_element="Rh",
        sample_contains_tube_element=True,
    )
    assert included.success
    assert "Rh" in included.element_wt


def test_fp_roundtrip_quartz():
    """Predicted Si Kα from SiO2 inverts back to SiO2."""
    assumptions = MatrixAssumptions(kind=MatrixKind.OXIDE)
    true_el, true_f = expand_composition({"Si": 1.0}, assumptions)
    frac = {k: v / 100.0 for k, v in true_el.items()}
    fp = FundamentalParameters(excitation_energy=50.0)
    import xraylib as xrl

    z = xrl.SymbolToAtomicNumber("Si")
    intensity = fp.calculate_intensity("Si", z, "Kα1", frac["Si"], frac)
    assert intensity > 0
    peaks = [_peak("Si", "Kα1", intensity, 1.74)]
    result = quantify_from_peaks(
        peaks, assumptions, {"excitation_energy": 50.0, "incident_angle": 45.0}
    )
    assert result.success
    assert abs(result.element_wt["Si"] - true_el["Si"]) < 0.5
    assert abs(result.formula_wt["SiO2"] - 100.0) < 0.5


def test_fp_roundtrip_galena():
    assumptions = MatrixAssumptions(kind=MatrixKind.MEASURED)
    true_el, _ = expand_composition({"Pb": 86.6, "S": 13.4}, assumptions)
    frac = {k: v / 100.0 for k, v in true_el.items()}
    fp = FundamentalParameters(excitation_energy=50.0)
    import xraylib as xrl

    i_pb = fp.calculate_intensity(
        "Pb", xrl.SymbolToAtomicNumber("Pb"), "Lα1", frac["Pb"], frac
    )
    i_s = fp.calculate_intensity(
        "S", xrl.SymbolToAtomicNumber("S"), "Kα1", frac["S"], frac
    )
    assert i_pb > 0 and i_s > 0
    peaks = [
        _peak("Pb", "Lα1", i_pb, 10.55),
        _peak("S", "Kα1", i_s, 2.31),
    ]
    result = quantify_from_peaks(
        peaks, assumptions, {"excitation_energy": 50.0, "incident_angle": 45.0}
    )
    assert result.success
    assert "O" not in result.element_wt
    assert abs(result.element_wt["Pb"] - true_el["Pb"]) < 1.5
    assert abs(result.element_wt["S"] - true_el["S"]) < 1.5


def test_fp_h2o_changes_composition():
    dry = MatrixAssumptions(kind=MatrixKind.OXIDE, h2o_wt=0.0)
    wet = MatrixAssumptions(kind=MatrixKind.OXIDE, h2o_wt=20.0)
    peaks = [_peak("Si", "Kα1", 1000.0, 1.74)]
    params = {"excitation_energy": 50.0, "incident_angle": 45.0}
    dry_r = quantify_from_peaks(peaks, dry, params)
    wet_r = quantify_from_peaks(peaks, wet, params)
    assert dry_r.success and wet_r.success
    assert wet_r.formula_wt["H2O"] > 19.0
    assert wet_r.element_wt["Si"] < dry_r.element_wt["Si"]


def test_fp_result_concentrations_roles():
    result = quantify_from_peaks(
        [_peak("Si", "Kα1", 500.0, 1.74)],
        MatrixAssumptions(kind=MatrixKind.OXIDE, h2o_wt=5.0),
        {"excitation_energy": 50.0},
    )
    assert result.success
    assert result.concentrations["Si"]["role"] == "measured"
    assert result.concentrations["O"]["role"] == "assumed"
    assert result.concentrations["H"]["role"] == "assumed"
    assert isinstance(result, FPQuantResult)
    assert "SiO2" in result.formula_summary()


def test_empirical_formula_calcite_quartz_magnetite():
    from core.matrix_model import empirical_formula

    calcite_el, _ = expand_composition(
        {"Ca": 1.0}, MatrixAssumptions(kind=MatrixKind.CARBONATE)
    )
    assert empirical_formula(calcite_el) == "CaCO₃"

    quartz_el, _ = expand_composition(
        {"Si": 1.0}, MatrixAssumptions(kind=MatrixKind.OXIDE)
    )
    assert empirical_formula(quartz_el) == "SiO₂"

    mag_el, mag_f = expand_composition(
        {"Fe": 1.0},
        MatrixAssumptions(kind=MatrixKind.OXIDE, fe_as="Fe3O4"),
    )
    assert formula_for("Fe", MatrixAssumptions(kind=MatrixKind.OXIDE, fe_as="Fe3O4")) == "Fe3O4"
    assert abs(mag_f["Fe3O4"] - 100.0) < 1e-6
    assert empirical_formula(mag_el) == "Fe₃O₄"


def test_empirical_formula_h2o_as_hydrate_and_subscripts():
    from core.matrix_model import empirical_formula

    el, fw = expand_composition(
        {"Si": 1.0},
        MatrixAssumptions(kind=MatrixKind.OXIDE, h2o_wt=10.0),
    )
    formula = empirical_formula(el, formula_wt=fw)
    assert "H₂O" in formula
    assert "·" in formula
    # Free H must not appear in the anhydrous part (before the hydrate dot).
    anhydrous = formula.split("·", 1)[0]
    assert "H" not in anhydrous
    # Stoichiometry in the anhydrous formula uses Unicode subscripts only.
    assert not any(ch in "0123456789" for ch in anhydrous), formula


def test_empirical_formula_decimal_subscripts():
    from core.matrix_model import empirical_formula

    # Mixed cations → non-integer Al/Na coefficients, all subscripted.
    el, fw = expand_composition(
        {"Si": 24.5, "Al": 16.3, "Na": 10.7, "Ca": 0.3},
        MatrixAssumptions(kind=MatrixKind.OXIDE, h2o_wt=2.0),
    )
    formula = empirical_formula(el, formula_wt=fw)
    assert "H₂O" in formula
    assert formula.startswith("Si")
    assert "₂" in formula or "₄" in formula or "₆" in formula
    # No free HO… mashed into the main formula.
    assert "HO" not in formula.split("·", 1)[0]


def test_coerce_matrix_kind_from_combo_strings():
    from core.matrix_model import coerce_matrix_kind

    assert coerce_matrix_kind("carbonate") == MatrixKind.CARBONATE
    assert coerce_matrix_kind("Carbonate") == MatrixKind.CARBONATE
    assert coerce_matrix_kind("Oxide / silicate") == MatrixKind.OXIDE
    assert coerce_matrix_kind(MatrixKind.HYDROXIDE) == MatrixKind.HYDROXIDE
