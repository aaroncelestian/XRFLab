"""Compton scattering-angle estimation from the anode Compton hump."""

import numpy as np
import pytest

from core.xray_data import (
    DEFAULT_SCATTER_ANGLE_DEG,
    SCATTER_ANGLE_MAX_DEG,
    compton_energy,
    estimate_scatter_angle,
    get_tube_compton_lines,
    scatter_angle_from_compton_energy,
    tube_compton_incident_energy,
)
from core.tube_profile import TubeProfile, default_tube_profile


def _synthetic_scatter_spectrum(theta_deg, compton_fwhm_kev=0.6, compton_amp=800.0,
                                seed=0):
    """Rh tube blank: bremsstrahlung + elastic Kα1/Kα2/Kβ1 + Compton Kα/Kβ."""
    rng = np.random.default_rng(seed)
    energy = np.arange(0.0, 30.0, 0.01)
    brems = 120.0 * np.exp(-((energy - 12.0) / 9.0) ** 2) + 20.0

    def g(c, fwhm, amp):
        s = fwhm / 2.3548
        return amp * np.exp(-0.5 * ((energy - c) / s) ** 2)

    det_fwhm = 0.27
    spec = brems.copy()
    for e0, amp in ((20.216, 500.0), (20.074, 250.0), (22.724, 100.0)):
        spec += g(e0, det_fwhm, amp)
        # Compton partner of each elastic line
        spec += g(compton_energy(e0, theta_deg), compton_fwhm_kev, compton_amp * amp / 500.0)
    counts = rng.poisson(spec).astype(float)
    return energy, counts


def test_compton_energy_inversion_round_trip():
    for theta in (45.0, 90.0, 135.0, 155.0, 175.0):
        ec = compton_energy(20.216, theta)
        back = scatter_angle_from_compton_energy(20.216, ec)
        assert back is not None
        assert abs(back - theta) < 1e-6


def test_compton_energy_inversion_rejects_unphysical():
    # Above incident energy → not Compton
    assert scatter_angle_from_compton_energy(20.216, 20.5) is None
    # Below the 180° limit → unreachable
    e180 = compton_energy(20.216, 180.0)
    assert scatter_angle_from_compton_energy(20.216, e180 - 0.2) is None


def test_incident_energy_is_weighted_kalpha_mean():
    e_in = tube_compton_incident_energy("Rh", 50.0)
    assert e_in is not None
    # Between Kα2 and Kα1, closer to Kα1
    assert 20.074 < e_in < 20.216
    assert e_in > 0.5 * (20.074 + 20.216)


@pytest.mark.parametrize("theta_true", [110.0, 135.0, 150.0, 165.0])
def test_estimate_scatter_angle_recovers_synthetic_geometry(theta_true):
    energy, counts = _synthetic_scatter_spectrum(theta_true)
    res = estimate_scatter_angle(energy, counts, "Rh", 50.0)
    assert res is not None, f"no hump found at θ={theta_true}"
    # Compton energy is a shallow function of θ near backscatter, so a few
    # degrees of tolerance corresponds to ~10–20 eV in centroid.
    tol = 4.0 if theta_true < 150 else 10.0
    assert abs(res["angle_deg"] - theta_true) < tol
    assert 0.45 < res["fwhm_kev"] < 0.8
    assert res["snr"] > 5


def test_estimate_scatter_angle_rejects_narrow_fluorescence_line():
    """A detector-width line near 19 keV (e.g. Mo Kβ) must not be taken as Compton."""
    rng = np.random.default_rng(1)
    energy = np.arange(0.0, 30.0, 0.01)
    spec = 30.0 + 2000.0 * np.exp(-0.5 * ((energy - 19.6) / (0.27 / 2.3548)) ** 2)
    counts = rng.poisson(spec).astype(float)
    assert estimate_scatter_angle(energy, counts, "Rh", 50.0) is None


def test_estimate_scatter_angle_none_when_no_hump():
    rng = np.random.default_rng(2)
    energy = np.arange(0.0, 30.0, 0.01)
    counts = rng.poisson(np.full_like(energy, 40.0)).astype(float)
    assert estimate_scatter_angle(energy, counts, "Rh", 50.0) is None


def test_estimate_scatter_angle_requires_k_lines():
    energy, counts = _synthetic_scatter_spectrum(150.0)
    # 15 kV Rh: no K lines → no Compton Kα hump can exist
    assert estimate_scatter_angle(energy, counts, "Rh", 15.0) is None


def test_default_angle_is_near_backscatter_and_seeds_follow_it():
    assert 140.0 <= DEFAULT_SCATTER_ANGLE_DEG <= SCATTER_ANGLE_MAX_DEG
    seeds = get_tube_compton_lines("Rh", 50.0)
    ka = [s for s in seeds if s["line"] == "Compton Kα"][0]
    assert abs(
        ka["energy"] - compton_energy(ka["parent_energy"], DEFAULT_SCATTER_ANGLE_DEG)
    ) < 1e-9
    # Far from the 90° position (19.45 keV)
    assert ka["energy"] < 19.0


def test_tube_profile_default_angle_and_round_trip():
    p = default_tube_profile("Rh", 50.0)
    assert p.scatter_angle_deg == DEFAULT_SCATTER_ANGLE_DEG
    assert p.compton_fitted is False
    p.scatter_angle_deg = 152.0
    p.compton_fitted = True
    q = TubeProfile.from_dict(p.to_dict())
    assert q.scatter_angle_deg == 152.0
    assert q.compton_fitted is True
    # Legacy files without the key fall back to the default
    d = p.to_dict()
    del d["scatter_angle_deg"]
    del d["compton_fitted"]
    r = TubeProfile.from_dict(d)
    assert r.scatter_angle_deg == DEFAULT_SCATTER_ANGLE_DEG
    assert r.compton_fitted is False
