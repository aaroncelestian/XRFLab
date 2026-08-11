"""
Tube-profile soft constraints and known sample/tube overlap doublets.

Uses measured (or default) TubeProfile relative intensities to:
1. Soft-prior tube-line amplitudes toward the blank profile
2. Joint-fit unresolved tube+sample pairs (e.g. Mo Lα / S Kα)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from core.peak_fitting import PeakFitter, Peak


# Soft prior: residual weight on (A - A_prior) / A_prior
DEFAULT_AMPLITUDE_PRIOR_WEIGHT = 0.25

# Known near-degenerate tube L vs sample K overlaps (keV separation ~ detector FWHM)
# tube_element, tube_line, sample_element, sample_line, max_separation_kev
KNOWN_TUBE_SAMPLE_OVERLAPS = [
    ('Mo', 'Lα1', 'S', 'Kα1', 0.08),
    ('Mo', 'Lα1', 'S', 'Kα2', 0.08),
    ('Mo', 'Lβ1', 'S', 'Kβ1', 0.12),
    ('Rh', 'Lα1', 'Cl', 'Kα1', 0.12),
    ('Rh', 'Lα1', 'Cl', 'Kα2', 0.12),
    ('Rh', 'Lβ1', 'Cl', 'Kβ1', 0.12),
    ('Ag', 'Lα1', 'Ar', 'Kα1', 0.12),  # air Ar sometimes near Ag L
]


@dataclass
class OverlapPair:
    tube_pos: dict
    sample_pos: dict
    separation_kev: float


def find_overlap_pairs(peak_positions: List[dict]) -> Tuple[List[OverlapPair], List[dict]]:
    """
    Split peak seeds into overlap pairs + remaining positions.

    A position used in a pair is removed from the remaining list.
    """
    positions = [dict(p) for p in (peak_positions or [])]
    used = set()
    pairs: List[OverlapPair] = []

    tube_idxs = [
        i for i, p in enumerate(positions)
        if p.get('is_tube_line') and p.get('element') and p.get('line')
        and not str(p.get('line', '')).startswith('Compton')
    ]
    sample_idxs = [
        i for i, p in enumerate(positions)
        if (not p.get('is_tube_line')) and p.get('element') and p.get('line')
    ]

    for tube_el, tube_line, samp_el, samp_line, max_sep in KNOWN_TUBE_SAMPLE_OVERLAPS:
        t_idx = None
        for i in tube_idxs:
            if i in used:
                continue
            p = positions[i]
            if p.get('element') == tube_el and p.get('line') == tube_line:
                t_idx = i
                break
        if t_idx is None:
            continue

        s_idx = None
        for i in sample_idxs:
            if i in used:
                continue
            p = positions[i]
            if p.get('element') == samp_el and p.get('line') == samp_line:
                sep = abs(float(p['energy']) - float(positions[t_idx]['energy']))
                if sep <= max_sep:
                    s_idx = i
                    break
        if s_idx is None:
            continue

        sep = abs(
            float(positions[t_idx]['energy']) - float(positions[s_idx]['energy'])
        )
        pairs.append(OverlapPair(
            tube_pos=positions[t_idx],
            sample_pos=positions[s_idx],
            separation_kev=sep,
        ))
        used.add(t_idx)
        used.add(s_idx)

    remaining = [p for i, p in enumerate(positions) if i not in used]
    return pairs, remaining


def expected_tube_amplitude(
    profile,
    line: str,
    reference_amplitude: float,
    reference_line: Optional[str] = None,
) -> Optional[float]:
    """Scale reference tube amplitude by profile line ratio."""
    if profile is None or reference_amplitude is None or reference_amplitude <= 0:
        return None
    ref = reference_line or profile.reference_line
    ratio = profile.expected_relative_to(line, reference=ref)
    if ratio != ratio or ratio < 0:  # NaN
        return None
    # profile ratios are relative to ref line's ratio value (usually 1.0)
    ref_ratio = profile.line_ratios.get(ref, 1.0)
    line_ratio = profile.line_ratios.get(line)
    if line_ratio is None or ref_ratio <= 0:
        return None
    return float(reference_amplitude) * (float(line_ratio) / float(ref_ratio))


def fit_peak_with_amplitude_prior(
    energy,
    counts,
    initial_center,
    shape='gaussian',
    fixed_fwhm=None,
    fix_center=True,
    amplitude_prior=None,
    prior_weight: float = DEFAULT_AMPLITUDE_PRIOR_WEIGHT,
    known_line: bool = True,
) -> Optional[Peak]:
    """
    Fit a peak with an optional soft amplitude prior.

    Augments the least-squares residual with:
        prior_weight * (A - A_prior) / max(A_prior, eps)
    """
    if amplitude_prior is None or amplitude_prior <= 0 or prior_weight <= 0:
        return PeakFitter.fit_single_peak(
            energy, counts, initial_center,
            shape=shape,
            known_line=known_line,
            fix_center=fix_center,
            fixed_fwhm=fixed_fwhm,
        )

    if fixed_fwhm is not None:
        fwhm_estimate = float(fixed_fwhm)
    else:
        fwhm_estimate = float(PeakFitter.calculate_fwhm(initial_center))

    window_width = 3.0 * fwhm_estimate
    if initial_center < 3.0 and fixed_fwhm is None:
        window_width = 5.0 * fwhm_estimate

    mask = np.abs(np.asarray(energy) - initial_center) < window_width
    if int(np.sum(mask)) < 5:
        return None

    x = np.asarray(energy, dtype=float)[mask]
    y = np.asarray(counts, dtype=float)[mask]
    sigma_data = np.sqrt(np.maximum(np.abs(y), 1.0))

    center = float(initial_center)
    sigma = fwhm_estimate / 2.355
    amp0 = max(float(np.max(y)), 1.0)
    prior = float(amplitude_prior)
    w = float(prior_weight)

    # Prefer locked width + fixed center for tube secondary lines
    def residual(params):
        amp = params[0]
        if shape == 'voigt':
            gamma = sigma * PeakFitter.VOIGT_GAMMA_RATIO
            model = PeakFitter.voigt(x, amp, center, sigma, gamma)
        else:
            model = PeakFitter.gaussian(x, amp, center, sigma)
        data_resid = (model - y) / sigma_data
        prior_resid = np.array([w * (amp - prior) / max(prior, 1.0)])
        return np.concatenate([data_resid, prior_resid])

    try:
        result = optimize.least_squares(
            residual,
            x0=np.array([amp0]),
            bounds=([0.0], [np.inf]),
            max_nfev=5000,
        )
        amp = float(result.x[0])
    except Exception as e:
        print(f"Prior fit failed at {initial_center:.3f} keV: {e}")
        return PeakFitter.fit_single_peak(
            energy, counts, initial_center,
            shape=shape, known_line=known_line,
            fix_center=fix_center, fixed_fwhm=fixed_fwhm,
        )

    fwhm = 2.355 * sigma
    area = amp * sigma * np.sqrt(2 * np.pi)
    shape_params = {'sigma': sigma}
    if shape == 'voigt':
        gamma = sigma * PeakFitter.VOIGT_GAMMA_RATIO
        shape_params['gamma'] = gamma
        fwhm_g = fwhm
        fwhm_l = 2.0 * gamma
        fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)

    return Peak(
        energy=center,
        amplitude=amp,
        fwhm=fwhm,
        area=area,
        shape='voigt' if shape == 'voigt' else 'gaussian',
        shape_params=shape_params,
        fixed_fwhm=float(fixed_fwhm) if fixed_fwhm is not None else None,
    )


def fit_overlap_doublet(
    energy,
    counts,
    tube_energy: float,
    sample_energy: float,
    tube_amplitude_prior: Optional[float] = None,
    shape: str = 'gaussian',
    prior_weight: float = DEFAULT_AMPLITUDE_PRIOR_WEIGHT,
) -> Tuple[Optional[Peak], Optional[Peak]]:
    """
    Joint fit of unresolved tube + sample peaks (shared detector FWHM).

    Centers fixed at tabulated energies. Tube amplitude softly pulled to prior
    when provided (from profile × reference tube line).
    """
    e_t = float(tube_energy)
    e_s = float(sample_energy)
    fwhm = float(PeakFitter.calculate_fwhm(0.5 * (e_t + e_s)))
    sigma = fwhm / 2.355

    # Window covering both
    mid = 0.5 * (e_t + e_s)
    half = max(3.0 * fwhm, abs(e_t - e_s) + 2.0 * fwhm)
    mask = np.abs(np.asarray(energy) - mid) < half
    if int(np.sum(mask)) < 8:
        return None, None

    x = np.asarray(energy, dtype=float)[mask]
    y = np.asarray(counts, dtype=float)[mask]
    sigma_data = np.sqrt(np.maximum(np.abs(y), 1.0))

    # Initial amplitudes from local heights
    i_t = int(np.argmin(np.abs(x - e_t)))
    i_s = int(np.argmin(np.abs(x - e_s)))
    a_t0 = max(float(y[i_t]), 1.0)
    a_s0 = max(float(y[i_s]), 1.0)
    if tube_amplitude_prior is not None and tube_amplitude_prior > 0:
        a_t0 = float(tube_amplitude_prior)

    use_voigt = (shape == 'voigt')
    gamma = sigma * PeakFitter.VOIGT_GAMMA_RATIO

    def model(a_t, a_s):
        if use_voigt:
            return (
                PeakFitter.voigt(x, a_t, e_t, sigma, gamma)
                + PeakFitter.voigt(x, a_s, e_s, sigma, gamma)
            )
        return (
            PeakFitter.gaussian(x, a_t, e_t, sigma)
            + PeakFitter.gaussian(x, a_s, e_s, sigma)
        )

    def residual(params):
        a_t, a_s = params
        data_resid = (model(a_t, a_s) - y) / sigma_data
        if tube_amplitude_prior is not None and tube_amplitude_prior > 0 and prior_weight > 0:
            prior_resid = np.array([
                prior_weight * (a_t - tube_amplitude_prior) / max(tube_amplitude_prior, 1.0)
            ])
            return np.concatenate([data_resid, prior_resid])
        return data_resid

    try:
        result = optimize.least_squares(
            residual,
            x0=np.array([a_t0, a_s0]),
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            max_nfev=8000,
        )
        a_t, a_s = [float(v) for v in result.x]
    except Exception as e:
        print(f"Doublet fit failed near {mid:.3f} keV: {e}")
        return None, None

    area_t = a_t * sigma * np.sqrt(2 * np.pi)
    area_s = a_s * sigma * np.sqrt(2 * np.pi)
    shape_name = 'voigt' if use_voigt else 'gaussian'
    sp = {'sigma': sigma}
    if use_voigt:
        sp['gamma'] = gamma

    tube_peak = Peak(
        energy=e_t, amplitude=a_t, fwhm=fwhm, area=area_t,
        shape=shape_name, shape_params=dict(sp),
    )
    sample_peak = Peak(
        energy=e_s, amplitude=a_s, fwhm=fwhm, area=area_s,
        shape=shape_name, shape_params=dict(sp),
    )
    return tube_peak, sample_peak


def subtract_peak_from_residual(peak_fitter, energy, residual, peak: Peak):
    """Subtract a fitted peak model from the residual spectrum (in-place copy)."""
    energy = np.asarray(energy, dtype=float)
    out = np.asarray(residual, dtype=float).copy()
    sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
    if peak.shape == 'voigt':
        gamma = peak.shape_params.get('gamma', sigma * PeakFitter.VOIGT_GAMMA_RATIO)
        out -= peak_fitter.voigt(energy, peak.amplitude, peak.energy, sigma, gamma)
    elif peak.shape == 'pseudo_voigt':
        eta = peak.shape_params.get('eta', 0.5)
        out -= peak_fitter.pseudo_voigt(energy, peak.amplitude, peak.energy, sigma, eta)
    elif peak.shape == 'hypermet':
        out -= peak_fitter.hypermet(
            energy, peak.amplitude, peak.energy, sigma,
            peak.shape_params.get('tail_amplitude', 0.1),
            peak.shape_params.get('tail_slope', 2.0),
        )
    elif peak.shape == 'tail_gaussian':
        out -= peak_fitter.tail_gaussian(
            energy, peak.amplitude, peak.energy, sigma,
            peak.shape_params.get('tail_fraction', 0.15),
            peak.shape_params.get('tail_sigma', sigma * 3),
        )
    else:
        out -= peak_fitter.gaussian(energy, peak.amplitude, peak.energy, sigma)
    return out


def pick_tube_reference_peak(fitted_tube_peaks: List[Peak], profile) -> Optional[Peak]:
    """Choose the profile reference line among fitted tube peaks."""
    if not fitted_tube_peaks:
        return None
    ref_name = getattr(profile, 'reference_line', None) if profile else None
    if ref_name:
        for p in fitted_tube_peaks:
            if p.line == ref_name:
                return p
    # Prefer Kα1 then Lα1 then strongest
    for name in ('Kα1', 'Lα1', 'Lβ1'):
        for p in fitted_tube_peaks:
            if p.line == name:
                return p
    return max(fitted_tube_peaks, key=lambda p: p.amplitude)
