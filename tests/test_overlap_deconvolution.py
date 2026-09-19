"""Certified-concentration overlap priors and mixing-matrix unmix."""

from __future__ import annotations

import numpy as np
import pytest

from core.line_groups import GroupLine, LineGroup, fit_grouped
from core.overlap_deconvolution import (
    OverlapMixingModel,
    OverlapPair,
    composition_ratio_priors,
    find_overlap_pairs,
    learn_mixing_models,
    ratio_prior_rows,
)
from core.peak_fitting import PeakFitter


def test_ba_la_l_lines_are_an_overlap_pair():
    PeakFitter.activate(PeakFitter())
    ba = LineGroup("Ba", "L3", [GroupLine("Lα1", 4.466, 1.0)])
    la = LineGroup("La", "L3", [GroupLine("Lα1", 4.651, 1.0)])
    fe = LineGroup("Fe", "K", [GroupLine("Kα1", 6.404, 1.0)])
    pairs = find_overlap_pairs([ba, la, fe], PeakFitter.calculate_fwhm)
    els = {(p.element_a, p.element_b) for p in pairs}
    assert ("Ba", "La") in els or ("La", "Ba") in els
    assert all("Fe" not in (p.element_a, p.element_b) for p in pairs)


def test_composition_prior_splits_unresolved_blend():
    """Two identical design columns: only the certified ratio can unmix them."""
    PeakFitter.activate(PeakFitter())
    e0 = 4.50
    a = LineGroup("Ba", "L3", [GroupLine("Lα1", e0, 1.0)])
    b = LineGroup("La", "L3", [GroupLine("Lα1", e0, 1.0)])
    energy = np.arange(3.5, 5.5, 0.01)
    sigma = PeakFitter.calculate_fwhm(e0) / 2.355
    y = 5000.0 * np.exp(-(energy - e0) ** 2 / (2 * sigma ** 2))
    counts = y + 20.0

    free = fit_grouped(
        energy, y, counts, [a, b], [], shape="gaussian",
        refine_energy=False, refine_shape=False,
    )
    # Unconstrained NNLS dumps the blend into one column
    assert min(free.group_amplitudes.values()) < 0.15 * max(free.group_amplitudes.values())

    priors = composition_ratio_priors(
        [OverlapPair("Ba L3", "La L3", "Ba", "La", e0, e0, 0.0)],
        {"Ba": 0.16, "La": 1.06},
        excitation_kv=50.0,
    )
    assert priors
    split = fit_grouped(
        energy, y, counts, [a, b], [], shape="gaussian",
        refine_energy=False, refine_shape=False, ratio_priors=priors,
    )
    ba_a = split.group_amplitudes["Ba L3"]
    la_a = split.group_amplitudes["La L3"]
    assert ba_a > 0 and la_a > 0
    # Certified La/Ba ≈ 6.6; allow the theoretical ε to move that some
    assert 2.0 < la_a / ba_a < 15.0
    assert abs((ba_a + la_a) - max(free.group_amplitudes.values())) < 0.15 * (ba_a + la_a)


def test_mixing_matrix_recovers_concentrations():
    # I_Ba = 2 C_Ba + 0.4 C_La ; I_La = 0.3 C_Ba + 1.5 C_La
    s_true = np.array([[2.0, 0.4], [0.3, 1.5]])
    concs = {
        "s1": {"Ba": 0.07, "La": 0.02},
        "s2": {"Ba": 0.08, "La": 0.12},
        "s3": {"Ba": 0.10, "La": 0.37},
        "s4": {"Ba": 0.16, "La": 1.06},
    }
    ints = {}
    for name, c in concs.items():
        vec = s_true @ np.array([c["Ba"], c["La"]])
        ints[name] = {"Ba": float(vec[0]), "La": float(vec[1])}
    models = learn_mixing_models(
        [["Ba", "La"]],
        concentrations=concs, intensities=ints,
        energies={"Ba": 4.47, "La": 4.65},
    )
    assert len(models) == 1
    model = models[0]
    assert model.usable
    pred = model.predict(ints["s4"])
    assert pred["Ba"][0] == pytest.approx(0.16, rel=0.05)
    assert pred["La"][0] == pytest.approx(1.06, rel=0.05)


def test_ratio_prior_rows_encode_a_minus_rho_b():
    p, b, notes = ratio_prior_rows(
        ["Ba L3", "tube", "La L3"],
        [("Ba L3", "La L3", 0.5, 0.4)],
        3,
        weight=2.0,
    )
    assert p.shape == (1, 3)
    assert b.shape == (1,)
    assert p[0, 0] == pytest.approx(5.0)
    assert p[0, 2] == pytest.approx(-2.5)
    assert p[0, 1] == 0.0
    assert notes


def test_overlap_model_roundtrip():
    m = OverlapMixingModel(
        elements=["Ba", "La"], energies=[4.47, 4.65],
        sensitivities=[[2.0, 0.1], [0.1, 1.5]], intercepts=[0.0, 0.0],
        condition=2.0, n_standards=4, r_squared=0.99, usable=True,
        message="ok",
    )
    back = OverlapMixingModel.from_dict(m.to_dict())
    assert back.elements == ["Ba", "La"]
    assert back.usable
    pred = back.predict({"Ba": 0.32, "La": 1.60})
    assert pred["Ba"][0] == pytest.approx(0.107, rel=0.05)
    assert pred["La"][0] == pytest.approx(1.06, rel=0.05)
