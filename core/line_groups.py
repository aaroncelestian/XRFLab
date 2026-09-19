"""
Grouped (shared-amplitude) line fitting.

Each element sub-shell — K; L1, L2, L3; M — is fitted as ONE free amplitude
with a fixed intra-group line pattern. Within a sub-shell the branching
ratios are pure radiative rates (known to a few %); between sub-shells and
between series the populations depend on the excitation spectrum, so those
stay free. Tube lines, Compton humps and unlabeled peaks are single
columns. Because widths come from the detector model and centres from the
tables, every component is linear in amplitude and the whole spectrum is
solved in one bounded linear least squares (AXIL / PyMca style). Overlaps
such as As Kα / Pb Lα are then resolved by each element's clean lines.

An optional global zero/gain refinement absorbs small energy-calibration
errors so fixed-centre fitting is not penalised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from core.peak_fitting import Peak, PeakFitter, normalize_peak_shape
from core.smart_peak_id import line_relative_intensities

# Lines below this fraction of the group's strongest line are dropped
MIN_GROUP_RATIO = 0.01
# Soft tube-profile ratio prior: 1σ width of (A_line − ρ·A_ref) relative to
# the expected amplitude. A measured profile is trusted more than the
# built-in approximate ratios.
TUBE_PRIOR_REL_SIGMA_MEASURED = 0.30
TUBE_PRIOR_REL_SIGMA_DEFAULT = 0.60
# Energy-calibration refinement bounds
MAX_OFFSET_KEV = 0.040
MAX_GAIN_DEV = 0.004

# L-line → originating sub-shell (Siegbahn)
_L_SUBSHELL = {
    'Lα1': 'L3', 'Lα2': 'L3', 'Lα': 'L3', 'Lβ2': 'L3', 'Lβ5': 'L3',
    'Lβ6': 'L3', 'Lβ15': 'L3', 'Ll': 'L3', 'Ls': 'L3', 'Lt': 'L3', 'Lu': 'L3',
    'Lβ1': 'L2', 'Lγ1': 'L2', 'Lγ5': 'L2', 'Lγ6': 'L2', 'Lγ8': 'L2',
    'Lη': 'L2', 'Lν': 'L2',
    'Lβ3': 'L1', 'Lβ4': 'L1', 'Lβ9': 'L1', 'Lβ10': 'L1', 'Lβ': 'L1',
    'Lγ2': 'L1', 'Lγ3': 'L1', 'Lγ4': 'L1', 'Lγ11': 'L1', 'Lγ': 'L1',
}


def subshell_of(line_name: Optional[str]) -> str:
    """'Kβ1' → 'K', 'Lβ1' → 'L2', 'Mα1' → 'M'."""
    s = str(line_name or '')
    if s.startswith('K'):
        return 'K'
    if s.startswith('L'):
        return _L_SUBSHELL.get(s, 'L3')
    if s.startswith('M'):
        return 'M'
    return ''


@dataclass
class GroupLine:
    name: str
    energy: float          # tabulated (keV)
    ratio: float           # amplitude ratio to the group's strongest line
    seed: dict = field(default_factory=dict)


@dataclass
class LineGroup:
    element: str
    subshell: str          # 'K', 'L1', 'L2', 'L3', 'M'
    lines: List[GroupLine]

    @property
    def key(self) -> str:
        return f"{self.element} {self.subshell}"

    @property
    def strongest(self) -> GroupLine:
        return max(self.lines, key=lambda l: l.ratio)


def make_ratio_corrector(
    fp=None, matrix_frac: Optional[Dict[str, float]] = None
) -> Optional[Callable[[str, str, float], float]]:
    """
    Multiplicative correction turning emission ratios into *observed* ratios:
    detector efficiency × (optionally) matrix absorption at the line energy.

    Only the variation *within* a group matters, so any common factor cancels.
    """
    if fp is None:
        return None

    def corr(_element: str, _line: str, energy: float) -> float:
        f = float(fp._detector_efficiency(float(energy)))
        if matrix_frac:
            try:
                f *= float(fp._calculate_absorption(float(energy), matrix_frac))
            except Exception:
                pass
        return max(f, 1e-9)

    return corr


def build_line_groups(
    peak_positions: List[dict],
    energy_min: float,
    energy_max: float,
    ratio_corrector: Optional[Callable[[str, str, float], float]] = None,
) -> Tuple[List[LineGroup], List[dict]]:
    """
    Partition seeds into element sub-shell groups and single columns.

    Sample lines with element + line labels form groups; tube lines,
    Compton humps and unlabeled peaks are returned as `singles`.
    """
    buckets: Dict[Tuple[str, str], List[dict]] = {}
    singles: List[dict] = []
    for pos in peak_positions:
        e = float(pos.get('energy', 0.0))
        if not (energy_min <= e <= energy_max):
            continue
        el = pos.get('element')
        line = pos.get('line')
        if pos.get('is_tube_line') or not el or not line:
            singles.append(pos)
            continue
        sub = subshell_of(line)
        if not sub:
            singles.append(pos)
            continue
        buckets.setdefault((el, sub), []).append(pos)

    groups: List[LineGroup] = []
    rel_cache: Dict[str, Dict[str, float]] = {}
    for (el, sub), members in buckets.items():
        # Emission relative intensities (element-wide scale is fine: within
        # one sub-shell every line shares the same yield factor).
        raw = []
        for pos in members:
            rel = pos.get('relative_intensity')
            if rel is None:
                if el not in rel_cache:
                    from core.advanced_peak_fitting import get_element_z
                    z = int(get_element_z(el) or 0)
                    rel_cache[el] = line_relative_intensities(el, z) if z else {}
                rel = rel_cache[el].get(str(pos['line']), 1.0)
            rel = float(rel or 0.0)
            if ratio_corrector is not None:
                rel *= ratio_corrector(el, str(pos['line']), float(pos['energy']))
            raw.append((pos, rel))
        top = max(r for _p, r in raw) if raw else 0.0
        if top <= 0:
            singles.extend(members)
            continue
        lines = []
        for pos, rel in raw:
            ratio = rel / top
            if ratio < MIN_GROUP_RATIO:
                continue
            lines.append(GroupLine(
                name=str(pos['line']), energy=float(pos['energy']),
                ratio=ratio, seed=pos,
            ))
        if lines:
            groups.append(LineGroup(element=el, subshell=sub, lines=lines))
    groups.sort(key=lambda g: (g.element, g.subshell))
    return groups, singles


# --- design matrix -----------------------------------------------------------

def _sigma_for(energy_kev: float, seed: Optional[dict] = None) -> float:
    if seed is not None and seed.get('fixed_fwhm') is not None:
        return float(seed['fixed_fwhm']) / 2.355
    return float(PeakFitter.calculate_fwhm(float(energy_kev))) / 2.355


def global_shape_bounds(shape: str):
    """
    Spectrum-wide shape extras, expressed relative to σ so one set serves
    every peak: (names, p0, lower, upper). Empty for a pure Gaussian.
    """
    shape = normalize_peak_shape(shape)
    if shape == 'tail_gaussian':
        return (
            ['tail_fraction', 'tail_sigma_mult'],
            [PeakFitter.TAIL_GAUSSIAN_FRAC, PeakFitter.TAIL_GAUSSIAN_SIGMA_MULT],
            [0.0, 1.2], [0.5, 8.0],
        )
    if shape == 'hypermet':
        return (
            ['tail_amplitude', 'tail_beta_mult', 'step_amplitude'],
            [PeakFitter.HYPERMET_TAIL_AMP, PeakFitter.HYPERMET_TAIL_BETA_SIGMA,
             PeakFitter.HYPERMET_STEP_AMP],
            [0.0, 0.5, 0.0], [0.5, 15.0, 0.1],
        )
    if shape == 'voigt':
        return (['gamma_ratio'], [PeakFitter.VOIGT_GAMMA_RATIO], [0.01], [2.0])
    if shape == 'pseudo_voigt':
        return (['eta'], [0.3], [0.0], [1.0])
    return ([], [], [], [])


def shape_params_from_global(shape: str, sigma: float, g) -> dict:
    """Per-peak shape_params from σ and the spectrum-wide extras `g`."""
    shape = normalize_peak_shape(shape)
    sigma = float(sigma)
    g = list(g or [])
    if shape == 'tail_gaussian' and len(g) >= 2:
        return {'sigma': sigma, 'tail_fraction': float(g[0]),
                'tail_sigma': sigma * float(g[1])}
    if shape == 'hypermet' and len(g) >= 3:
        return {'sigma': sigma, 'tail_amplitude': float(g[0]),
                'tail_beta': sigma * float(g[1]), 'step_amplitude': float(g[2])}
    if shape == 'voigt' and len(g) >= 1:
        return {'sigma': sigma, 'gamma': sigma * float(g[0])}
    if shape == 'pseudo_voigt' and len(g) >= 1:
        return {'sigma': sigma, 'eta': float(g[0])}
    return PeakFitter.default_shape_params(shape, sigma)


def _unit_profile(x, center, sigma, shape, g=None) -> np.ndarray:
    peak = Peak(
        energy=float(center), amplitude=1.0, fwhm=2.355 * float(sigma), area=0.0,
        shape=normalize_peak_shape(shape),
        shape_params=shape_params_from_global(shape, sigma, g),
    )
    return PeakFitter.evaluate_peak(peak, x)


@dataclass
class _Column:
    kind: str                 # 'group' | 'tube' | 'compton' | 'unknown' | 'sample'
    label: str
    group: Optional[LineGroup] = None
    seed: Optional[dict] = None


def _build_design(
    x: np.ndarray,
    groups: List[LineGroup],
    singles: List[dict],
    shape: str,
    offset: float = 0.0,
    gain: float = 1.0,
    gshape=None,
) -> Tuple[np.ndarray, List[_Column]]:
    cols: List[np.ndarray] = []
    meta: List[_Column] = []

    def obs_e(e_tab: float) -> float:
        return float(e_tab) * gain + offset

    for g in groups:
        col = np.zeros_like(x)
        for l in g.lines:
            e = obs_e(l.energy)
            col += l.ratio * _unit_profile(x, e, _sigma_for(l.energy), shape, gshape)
        if np.any(col > 0):
            cols.append(col)
            meta.append(_Column('group', g.key, group=g))

    for pos in singles:
        e_tab = float(pos['energy'])
        line = str(pos.get('line') or '')
        if pos.get('is_tube_line'):
            kind = 'compton' if line.startswith('Compton') else 'tube'
            e = obs_e(e_tab)
        elif pos.get('element'):
            kind = 'sample'
            e = obs_e(e_tab)
        else:
            kind = 'unknown'
            e = e_tab  # detected in the observed scale already
        col = _unit_profile(x, e, _sigma_for(e_tab, pos), shape, gshape)
        if np.any(col > 0):
            cols.append(col)
            label = f"{pos.get('element') or '?'} {line}".strip() if kind != 'unknown' else f"unk {e_tab:.3f}"
            meta.append(_Column(kind, label, seed=pos))

    if not cols:
        return np.zeros((x.size, 0)), meta
    return np.column_stack(cols), meta


def _tube_prior_rows(
    meta: List[_Column],
    amps: np.ndarray,
    profile,
    n_cols: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Soft rows  w·(A_t − ρ_t·A_ref) = 0  from the tube profile ratios."""
    notes: List[str] = []
    if profile is None:
        return np.zeros((0, n_cols)), np.zeros(0), notes
    ref_name = getattr(profile, 'reference_line', None)
    ref_idx = None
    for i, m in enumerate(meta):
        if m.kind == 'tube' and m.seed is not None and m.seed.get('line') == ref_name:
            ref_idx = i
            break
    if ref_idx is None:
        return np.zeros((0, n_cols)), np.zeros(0), notes
    a_ref = float(amps[ref_idx])
    if a_ref <= 0:
        return np.zeros((0, n_cols)), np.zeros(0), notes
    rel_sigma = (
        TUBE_PRIOR_REL_SIGMA_MEASURED
        if getattr(profile, 'source', 'default') == 'measured'
        else TUBE_PRIOR_REL_SIGMA_DEFAULT
    )

    rows = []
    for i, m in enumerate(meta):
        if i == ref_idx or m.kind != 'tube' or m.seed is None:
            continue
        line = m.seed.get('line')
        rho = profile.expected_relative_to(line, ref_name) if line else float('nan')
        if not np.isfinite(rho) or rho <= 0:
            continue
        expected = rho * a_ref
        w = 1.0 / (rel_sigma * max(expected, 1.0))
        row = np.zeros(n_cols)
        row[i] = w
        row[ref_idx] = -w * rho
        rows.append(row)
        notes.append(
            f"{m.label}: profile prior A≈{expected:.1f} ±{rel_sigma:.0%} "
            f"(ρ={rho:.3f} × {ref_name})"
        )
    if not rows:
        return np.zeros((0, n_cols)), np.zeros(0), notes
    return np.vstack(rows), np.zeros(len(rows)), notes


def _solve(A_w: np.ndarray, b_w: np.ndarray) -> np.ndarray:
    """Non-negative least squares (Lawson–Hanson, compiled)."""
    if A_w.shape[1] == 0:
        return np.zeros(0)
    try:
        x, _rnorm = optimize.nnls(A_w, b_w, maxiter=max(200, 30 * A_w.shape[1]))
    except Exception:
        res = optimize.lsq_linear(A_w, b_w, bounds=(0.0, np.inf), method='trf')
        x = res.x
    return np.asarray(x, dtype=float)


@dataclass
class GroupedFitResult:
    peaks: List[Peak]
    n_params: int
    notes: List[str]
    energy_offset_kev: float = 0.0
    energy_gain: float = 1.0
    group_amplitudes: Dict[str, float] = field(default_factory=dict)
    shape_globals: Dict[str, float] = field(default_factory=dict)
    chi2: float = float('inf')


def fit_grouped(
    energy,
    counts_bg_subtracted,
    counts_raw,
    groups: List[LineGroup],
    singles: List[dict],
    shape: str = 'tail_gaussian',
    profile=None,
    refine_energy: bool = True,
    refine_shape: bool = True,
    max_outer_evals: int = 250,
    ratio_priors=None,
) -> GroupedFitResult:
    """
    Linear least squares for every component amplitude, wrapped in a small
    nonlinear search over spectrum-wide nuisance parameters: energy zero /
    gain and the peak-shape extras (tail fraction/width etc., shared by all
    peaks in σ units). Core widths stay on the detector model.

    Args:
        energy, counts_bg_subtracted: spectrum with background removed
        counts_raw: original counts (Poisson weights)
        groups, singles: from build_line_groups()
        shape: peak shape; core widths locked to the detector model
        profile: TubeProfile for soft tube-ratio priors (optional)
        refine_energy: fit a global zero/gain correction
        refine_shape: fit the global shape extras (else use defaults)
        ratio_priors: optional (key_a, key_b, rho, rel_sigma) rows that
            softly enforce A_a = ρ A_b (certified-concentration split)
    """
    shape = normalize_peak_shape(shape)
    x = np.asarray(energy, dtype=float)
    y = np.asarray(counts_bg_subtracted, dtype=float)
    w = 1.0 / np.sqrt(np.maximum(np.asarray(counts_raw, dtype=float), 1.0))

    g_names, g0, g_lo, g_hi = global_shape_bounds(shape)
    g0 = [float(v) for v in g0]

    def solve_at(offset: float, gain: float, gshape):
        A, meta = _build_design(x, groups, singles, shape, offset, gain, gshape)
        if A.shape[1] == 0:
            return A, meta, np.zeros(0), float('inf')
        A_w = A * w[:, None]
        b_w = y * w
        extra_a = []
        extra_b = []
        if ratio_priors:
            from core.overlap_deconvolution import ratio_prior_rows
            labels = [m.label for m in meta]
            P, pb, _ = ratio_prior_rows(labels, ratio_priors, A.shape[1])
            if P.shape[0]:
                extra_a.append(P)
                extra_b.append(pb)
        if extra_a:
            A_w = np.vstack([A_w, *extra_a])
            b_w = np.concatenate([b_w, *extra_b])
        amps = _solve(A_w, b_w)
        if profile is not None:
            P, pb, _ = _tube_prior_rows(meta, amps, profile, A.shape[1])
            if P.shape[0]:
                stacked_a = [A * w[:, None], *extra_a, P]
                stacked_b = [y * w, *extra_b, pb]
                amps = _solve(np.vstack(stacked_a), np.concatenate(stacked_b))
        resid = (A @ amps - y) * w
        return A, meta, amps, float(np.sum(resid ** 2))

    offset, gain, gshape = 0.0, 1.0, list(g0)
    A, meta, amps, chi2 = solve_at(offset, gain, gshape)

    # --- outer search over nuisance parameters -----------------------------
    free: List[str] = []
    p0: List[float] = []
    steps: List[float] = []
    if refine_energy:
        free += ['offset', 'gain']
        p0 += [0.0, 1.0]
        steps += [0.010, 0.0010]
    if refine_shape and g_names:
        free += list(g_names)
        p0 += list(g0)
        steps += [0.25 * (hi - lo) for lo, hi in zip(g_lo, g_hi)]

    if free and A.shape[1] and np.isfinite(chi2):
        def unpack(p):
            off, gn = 0.0, 1.0
            gs = list(g0)
            i = 0
            if refine_energy:
                off, gn = float(p[0]), float(p[1])
                i = 2
            if refine_shape and g_names:
                gs = [float(v) for v in p[i:i + len(g_names)]]
            return off, gn, gs

        def in_bounds(off, gn, gs):
            if abs(off) > MAX_OFFSET_KEV or abs(gn - 1.0) > MAX_GAIN_DEV:
                return False
            return all(lo <= v <= hi for v, lo, hi in zip(gs, g_lo, g_hi))

        def cost(p):
            off, gn, gs = unpack(p)
            if not in_bounds(off, gn, gs):
                return chi2 * 10.0
            return solve_at(off, gn, gs)[3]

        p0a = np.asarray(p0, dtype=float)
        simplex = [p0a.copy()]
        for k, st in enumerate(steps):
            v = p0a.copy()
            v[k] += st
            # keep the initial simplex inside the box
            off, gn, gs = unpack(v)
            if not in_bounds(off, gn, gs):
                v[k] -= 2 * st
            simplex.append(v)
        try:
            res = optimize.minimize(
                cost, x0=p0a, method='Nelder-Mead',
                options={'xatol': 1e-4, 'fatol': max(chi2 * 1e-5, 1e-6),
                         'maxfev': int(max_outer_evals),
                         'initial_simplex': np.array(simplex)},
            )
            if np.isfinite(res.fun) and res.fun < chi2:
                offset, gain, gshape = unpack(res.x)
                A, meta, amps, chi2 = solve_at(offset, gain, gshape)
        except Exception:
            pass

    notes: List[str] = []
    if ratio_priors and A.shape[1]:
        from core.overlap_deconvolution import ratio_prior_rows
        _, _, prior_notes = ratio_prior_rows(
            [m.label for m in meta], ratio_priors, A.shape[1]
        )
        notes.extend(prior_notes)
    if profile is not None and A.shape[1]:
        _, _, prior_notes = _tube_prior_rows(meta, amps, profile, A.shape[1])
        notes.extend(prior_notes)
    if refine_energy and (abs(offset) > 1e-4 or abs(gain - 1.0) > 1e-5):
        notes.append(
            f"Energy refinement: offset {offset*1000:+.1f} eV, gain {gain:.5f}"
        )
    shape_globals = {n: float(v) for n, v in zip(g_names, gshape)}
    if shape_globals:
        notes.append(
            "Shape (all peaks, σ units): "
            + ", ".join(f"{k}={v:.3f}" for k, v in shape_globals.items())
        )

    peaks: List[Peak] = []
    group_amps: Dict[str, float] = {}
    for m, a in zip(meta, amps):
        a = float(a)
        if m.kind == 'group' and m.group is not None:
            g = m.group
            group_amps[g.key] = a
            pattern = ", ".join(f"{l.name} {l.ratio:.3f}" for l in sorted(g.lines, key=lambda l: -l.ratio))
            notes.append(f"{g.key}: A={a:.1f}  ratios fixed [{pattern}]")
            for l in g.lines:
                e_obs = l.energy * gain + offset
                sigma = _sigma_for(l.energy)
                sp = shape_params_from_global(shape, sigma, gshape)
                amp = a * l.ratio
                peaks.append(Peak(
                    energy=e_obs, amplitude=amp,
                    fwhm=PeakFitter.fwhm_for_shape(sigma, shape, sp),
                    area=PeakFitter.compute_peak_area(amp, sigma, shape, sp),
                    element=g.element, line=l.name, shape=shape,
                    shape_params=dict(sp), is_tube_line=False, group=g.key,
                ))
        else:
            pos = m.seed or {}
            e_tab = float(pos.get('energy', 0.0))
            e_obs = e_tab if m.kind == 'unknown' else e_tab * gain + offset
            sigma = _sigma_for(e_tab, pos)
            sp = shape_params_from_global(shape, sigma, gshape)
            peaks.append(Peak(
                energy=e_obs, amplitude=a,
                fwhm=PeakFitter.fwhm_for_shape(sigma, shape, sp),
                area=PeakFitter.compute_peak_area(a, sigma, shape, sp),
                element=pos.get('element'), line=pos.get('line'), shape=shape,
                shape_params=dict(sp), is_tube_line=bool(pos.get('is_tube_line', False)),
                fixed_fwhm=(
                    float(pos['fixed_fwhm']) if pos.get('fixed_fwhm') is not None else None
                ),
            ))

    n_params = int(A.shape[1]) + len(free)
    return GroupedFitResult(
        peaks=peaks, n_params=n_params, notes=notes,
        energy_offset_kev=offset, energy_gain=gain, group_amplitudes=group_amps,
        shape_globals=shape_globals, chi2=chi2,
    )
