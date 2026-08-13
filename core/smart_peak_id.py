"""
Post-fit smart peak identification helpers.

Uses:
- FWHM excess vs the detector model (suspected unresolved overlap)
- Peak-shape hints (high Lorentzian mix / strong low-energy tail)
- Multi-line confirmation (Kα + Kβ, and L lines when useful)

This is heuristic — meant as a user-optional review aid after fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.peak_fitting import Peak, PeakFitter
from core.xray_data import get_element_lines

try:
    import xraylib as xrl
    _HAS_XRAYLIB = True
except ImportError:
    _HAS_XRAYLIB = False


# Soft expected Kβ / Kα area ratio range (matrix + detector dependent)
_KB_KA_RATIO_LO = 0.04
_KB_KA_RATIO_HI = 0.40


@dataclass
class SmartIDConfig:
    """User-tunable thresholds for post-fit smart ID."""
    fwhm_excess_kev: float = 0.030  # flag if measured FWHM > expected + this
    energy_match_tol_kev: float = 0.080
    min_relabel_score: float = 2.5  # require this score to suggest a new label
    score_margin: float = 0.75  # new label must beat current by this much
    apply_suggestions: bool = False  # relabel + add overlap seeds when True


@dataclass
class PeakAssessment:
    """Smart-ID assessment for one fitted peak."""
    peak_index: int
    energy: float
    measured_fwhm: float
    expected_fwhm: float
    fwhm_excess: float
    overlap_suspect: bool
    shape_flags: List[str] = field(default_factory=list)
    current_element: Optional[str] = None
    current_line: Optional[str] = None
    suggested_element: Optional[str] = None
    suggested_line: Optional[str] = None
    suggestion_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    secondary_seed_energy: Optional[float] = None
    alternate_candidates: List[Tuple[str, str, float]] = field(default_factory=list)


@dataclass
class SmartIDReport:
    """Full post-fit smart ID report."""
    assessments: List[PeakAssessment]
    n_overlap_suspects: int = 0
    n_relabel_suggestions: int = 0
    n_applied: int = 0
    summary_lines: List[str] = field(default_factory=list)


def _line_lookup(symbol: str, z: int) -> Dict[str, float]:
    """Map major line name -> energy (keV)."""
    out = {}
    lines = get_element_lines(symbol, z)
    for series in ('K', 'L', 'M'):
        for line in lines.get(series, []):
            out[line['name']] = float(line['energy'])
    return out


def _kb_ka_ratio(z: int) -> float:
    """Approximate radiative Kβ/Kα intensity ratio."""
    if not _HAS_XRAYLIB:
        return 0.12
    try:
        ka = float(xrl.RadRate(z, xrl.KA1_LINE)) + float(xrl.RadRate(z, xrl.KA2_LINE))
        kb = float(xrl.RadRate(z, xrl.KB1_LINE))
        if ka > 0:
            return kb / ka
    except Exception:
        pass
    return 0.12


def _diagnostic_line(
    line_map: Dict[str, float],
    *,
    e_max_kev: float = 40.0,
    e_min_kev: float = 0.70,
) -> Optional[Tuple[str, float]]:
    """The line that must be present to identify this element.

    Prefer Kα when it can be excited and detected; otherwise Lα; else Mα.
    Pb Kα is ~75 keV, so at 50 kV the diagnostic line is Lα (~10.55 keV).
    """
    def _pick(names):
        for name in names:
            if name in line_map:
                e = float(line_map[name])
                if e_min_kev <= e <= e_max_kev:
                    return name, e
        return None

    return (
        _pick(('Kα1', 'Kα'))
        or _pick(('Lα1', 'Lα'))
        or _pick(('Mα1', 'Mα'))
    )


def _is_minor_line(line_name: str, diagnostic_name: Optional[str]) -> bool:
    """True if this line must not ID the element without the diagnostic line."""
    if not diagnostic_name or not line_name:
        return False
    ln = line_name.replace('α', 'a').replace('β', 'b')
    dn = diagnostic_name.replace('α', 'a').replace('β', 'b')
    # Same family as diagnostic (Kα2 with Kα1, Lα2 with Lα1) is not minor
    if ln[:2] == dn[:2] and 'a' in ln[:3] and 'a' in dn[:3]:
        return False
    if line_name == diagnostic_name:
        return False
    return True


def _primary_lines(line_map: Dict[str, float]) -> List[Tuple[str, float]]:
    """Preferred ID lines in priority order."""
    preferred = [
        'Kα1', 'Kα', 'Kα2',
        'Lα1', 'Lα',
        'Mα1', 'Mα',
        'Kβ1', 'Kβ',
        'Lβ1', 'Lβ',
    ]
    found = []
    for name in preferred:
        if name in line_map:
            found.append((name, line_map[name]))
    return found


def _confirming_partners(line_name: str, line_map: Dict[str, float]) -> List[Tuple[str, float]]:
    """Other lines that help confirm an ID for this primary line."""
    partners = []
    if line_name.startswith('Kα'):
        for name in ('Kβ1', 'Kβ', 'Kβ3', 'Kβ2'):
            if name in line_map:
                partners.append((name, line_map[name]))
    elif line_name.startswith('Kβ'):
        for name in ('Kα1', 'Kα', 'Kα2'):
            if name in line_map:
                partners.append((name, line_map[name]))
    elif line_name.startswith('Lα'):
        for name in ('Lβ1', 'Lβ', 'Lβ2', 'Lγ1'):
            if name in line_map:
                partners.append((name, line_map[name]))
    elif line_name.startswith('Lβ'):
        for name in ('Lα1', 'Lα', 'Lγ1'):
            if name in line_map:
                partners.append((name, line_map[name]))
    elif line_name.startswith('M'):
        for name in ('Lα1', 'Lα', 'Lβ1', 'Lβ'):
            if name in line_map:
                partners.append((name, line_map[name]))
    return partners


def _shape_flags(peak: Peak) -> List[str]:
    flags = []
    params = peak.shape_params or {}
    eta = params.get('eta')
    if eta is not None and float(eta) >= 0.55:
        flags.append(f"high Lorentzian mix (η={float(eta):.2f})")
    tail_amp = params.get('tail_amplitude', params.get('tail_fraction'))
    if tail_amp is not None and float(tail_amp) >= 0.20:
        flags.append(f"strong low-energy tail ({float(tail_amp):.2f})")
    return flags


def _nearest_peak(
    peaks: Sequence[Peak],
    energy: float,
    tol: float,
    exclude_idx: Optional[int] = None,
) -> Optional[Tuple[int, Peak, float]]:
    best = None
    for i, p in enumerate(peaks):
        if exclude_idx is not None and i == exclude_idx:
            continue
        if getattr(p, 'is_tube_line', False):
            continue
        dist = abs(float(p.energy) - energy)
        if dist <= tol and (best is None or dist < best[2]):
            best = (i, p, dist)
    return best


def _score_element_for_peak(
    peak: Peak,
    peak_index: int,
    symbol: str,
    z: int,
    all_peaks: Sequence[Peak],
    tol: float,
    e_max_kev: float = 40.0,
) -> Tuple[float, Optional[str], List[str]]:
    """
    Score how well an element explains this peak via primary + confirming lines.

    Returns (score, best_line_name, reasons).
    """
    line_map = _line_lookup(symbol, z)
    if not line_map:
        return 0.0, None, []

    diagnostic = _diagnostic_line(line_map, e_max_kev=e_max_kev)

    best_score = 0.0
    best_line = None
    best_reasons: List[str] = []

    for line_name, line_e in _primary_lines(line_map):
        d_primary = abs(float(peak.energy) - line_e)
        if d_primary > tol:
            continue

        if diagnostic and _is_minor_line(line_name, diagnostic[0]):
            hit = _nearest_peak(all_peaks, diagnostic[1], tol, exclude_idx=peak_index)
            if hit is None:
                # e.g. Pb Mα without Pb Lα — do not ID the element
                continue

        score = 1.0 + max(0.0, 1.0 - d_primary / tol)  # 1–2 for energy match
        reasons = [f"{symbol} {line_name} within {d_primary*1000:.0f} eV"]

        for partner_name, partner_e in _confirming_partners(line_name, line_map):
            hit = _nearest_peak(all_peaks, partner_e, tol, exclude_idx=peak_index)
            if hit is None:
                continue
            _, partner_peak, d_partner = hit
            score += 1.5 + max(0.0, 0.5 - d_partner / tol)
            reasons.append(
                f"confirming {symbol} {partner_name} at {partner_peak.energy:.3f} keV"
            )

            # Soft Kα/Kβ ratio check when both are K lines
            if line_name.startswith('Kα') and partner_name.startswith('Kβ'):
                ka_area = max(float(peak.area), 1.0)
                kb_area = max(float(partner_peak.area), 0.0)
                ratio = kb_area / ka_area
                expected = _kb_ka_ratio(z)
                if _KB_KA_RATIO_LO <= ratio <= _KB_KA_RATIO_HI:
                    score += 0.75
                    reasons.append(
                        f"Kβ/Kα area ratio {ratio:.2f} looks plausible "
                        f"(~{expected:.2f} expected)"
                    )
                elif ratio > 0:
                    # Present but odd — still useful, smaller bonus
                    score += 0.25
                    reasons.append(
                        f"Kβ/Kα area ratio {ratio:.2f} unusual "
                        f"(~{expected:.2f} expected)"
                    )

        if score > best_score:
            best_score = score
            best_line = line_name
            best_reasons = reasons

    return best_score, best_line, best_reasons


def analyze_fitted_peaks(
    peaks: Sequence[Peak],
    candidate_elements: Sequence[Dict],
    config: Optional[SmartIDConfig] = None,
) -> SmartIDReport:
    """
    Post-fit smart ID / overlap analysis.

    Args:
        peaks: Fitted Peak objects
        candidate_elements: Element dicts with 'symbol' and 'z'
            (Elements-tab selection, or identified set)
        config: Thresholds / apply flag
    """
    config = config or SmartIDConfig()
    assessments: List[PeakAssessment] = []

    elements = [
        e for e in (candidate_elements or [])
        if e.get('symbol') and e.get('z')
    ]

    for i, peak in enumerate(peaks):
        if getattr(peak, 'is_tube_line', False):
            continue

        expected = float(PeakFitter.calculate_fwhm(peak.energy))
        measured = float(peak.fwhm)
        excess = measured - expected
        overlap = excess >= float(config.fwhm_excess_kev)
        flags = _shape_flags(peak)
        if overlap:
            flags = flags + [f"FWHM +{excess*1000:.0f} eV vs model"]

        assessment = PeakAssessment(
            peak_index=i,
            energy=float(peak.energy),
            measured_fwhm=measured,
            expected_fwhm=expected,
            fwhm_excess=excess,
            overlap_suspect=overlap or bool(flags),
            shape_flags=flags,
            current_element=peak.element,
            current_line=peak.line,
        )

        # Score candidate elements
        scored: List[Tuple[str, str, float, List[str]]] = []
        for elem in elements:
            score, line_name, reasons = _score_element_for_peak(
                peak, i, elem['symbol'], int(elem['z']),
                peaks, config.energy_match_tol_kev,
            )
            if score > 0 and line_name:
                scored.append((elem['symbol'], line_name, score, reasons))

        scored.sort(key=lambda t: t[2], reverse=True)
        assessment.alternate_candidates = [
            (s, ln, sc) for s, ln, sc, _ in scored[:4]
        ]

        current_score = 0.0
        if peak.element and peak.line:
            for s, ln, sc, reasons in scored:
                if s == peak.element and (
                    ln == peak.line
                    or (ln.startswith('Kα') and str(peak.line).startswith('Kα'))
                    or (ln.startswith('Kβ') and str(peak.line).startswith('Kβ'))
                ):
                    current_score = sc
                    break
            # If labeled but not in scored list, give a weak current score
            if current_score == 0.0 and peak.element:
                current_score = 1.0

        if scored:
            best_sym, best_line, best_score, best_reasons = scored[0]
            assessment.reasons.extend(best_reasons)

            should_suggest = False
            if not peak.element and best_score >= config.min_relabel_score:
                should_suggest = True
            elif (
                peak.element
                and best_sym != peak.element
                and best_score >= config.min_relabel_score
                and best_score >= current_score + config.score_margin
            ):
                should_suggest = True
            elif (
                peak.element
                and best_sym == peak.element
                and best_line
                and peak.line
                and best_line != peak.line
                and best_score >= config.min_relabel_score
            ):
                # Same element, better line name (e.g. Kα1 vs Kβ mis-tag)
                should_suggest = best_score >= current_score + 0.25

            if should_suggest:
                assessment.suggested_element = best_sym
                assessment.suggested_line = best_line
                assessment.suggestion_score = best_score
                if peak.element and best_sym != peak.element:
                    assessment.reasons.append(
                        f"suggest relabel {peak.element} → {best_sym} {best_line} "
                        f"(score {best_score:.1f} vs {current_score:.1f})"
                    )
                elif not peak.element:
                    assessment.reasons.append(
                        f"suggest label {best_sym} {best_line} (score {best_score:.1f})"
                    )

        # Overlap: propose a second seed from next-best distinct-element candidate
        if overlap and len(scored) >= 2:
            top_sym = scored[0][0]
            for alt_sym, alt_line, alt_score, _ in scored[1:]:
                if alt_sym == top_sym:
                    continue
                z_alt = next(int(e['z']) for e in elements if e['symbol'] == alt_sym)
                line_map = _line_lookup(alt_sym, z_alt)
                if alt_line not in line_map:
                    continue
                seed_e = line_map[alt_line]
                seed_name = alt_line
                # If alternate primary sits inside this envelope, seed its
                # confirming line (Kβ / Lβ) instead — that's the useful check.
                if abs(seed_e - float(peak.energy)) < max(0.05, 0.5 * measured):
                    partners = _confirming_partners(alt_line, line_map)
                    if partners:
                        seed_name, seed_e = partners[0]
                assessment.secondary_seed_energy = seed_e
                assessment.reasons.append(
                    f"overlap seed: try {alt_sym} {seed_name} at {seed_e:.3f} keV"
                )
                break
            if assessment.secondary_seed_energy is None:
                assessment.secondary_seed_energy = float(peak.energy) + 0.5 * max(
                    excess, config.fwhm_excess_kev
                )
                assessment.reasons.append(
                    f"overlap seed: +{0.5 * max(excess, config.fwhm_excess_kev)*1000:.0f} eV "
                    f"from center (generic split)"
                )
        elif overlap and not scored:
            assessment.secondary_seed_energy = float(peak.energy) + 0.5 * max(
                excess, config.fwhm_excess_kev
            )
            assessment.reasons.append(
                f"overlap seed: +{0.5 * max(excess, config.fwhm_excess_kev)*1000:.0f} eV "
                f"from center (generic split)"
            )

        assessments.append(assessment)

    n_overlap = sum(1 for a in assessments if a.overlap_suspect)
    n_suggest = sum(
        1 for a in assessments
        if a.suggested_element and (
            a.suggested_element != a.current_element
            or a.suggested_line != a.current_line
            or not a.current_element
        )
    )

    summary = [
        f"Smart ID: {len(assessments)} sample peaks reviewed",
        f"  Overlap / shape suspects: {n_overlap}",
        f"  Label suggestions: {n_suggest}",
        f"  FWHM excess threshold: {config.fwhm_excess_kev*1000:.0f} eV",
    ]
    for a in assessments:
        if not (a.overlap_suspect or a.suggested_element):
            continue
        tag = a.current_element or 'unknown'
        line = f" {a.current_line}" if a.current_line else ''
        bits = [f"{a.energy:.3f} keV ({tag}{line})"]
        if a.overlap_suspect:
            bits.append(
                f"FWHM {a.measured_fwhm*1000:.0f} eV "
                f"(expected {a.expected_fwhm*1000:.0f}, "
                f"+{a.fwhm_excess*1000:.0f})"
            )
        if a.suggested_element:
            bits.append(f"→ {a.suggested_element} {a.suggested_line}")
        if a.shape_flags:
            bits.append('; '.join(a.shape_flags))
        summary.append("  • " + " | ".join(bits))

    return SmartIDReport(
        assessments=assessments,
        n_overlap_suspects=n_overlap,
        n_relabel_suggestions=n_suggest,
        summary_lines=summary,
    )


def apply_smart_id_suggestions(
    peaks: List[Peak],
    report: SmartIDReport,
) -> Tuple[List[Peak], List[Dict], int]:
    """
    Apply high-confidence label suggestions and collect overlap seed dicts.

    Returns:
        (peaks, new_seed_dicts, n_applied_labels)
    """
    n_applied = 0
    seeds: List[Dict] = []
    existing_energies = [float(p.energy) for p in peaks]

    for a in report.assessments:
        if a.peak_index < 0 or a.peak_index >= len(peaks):
            continue
        peak = peaks[a.peak_index]
        if getattr(peak, 'is_tube_line', False):
            continue

        if a.suggested_element and a.suggested_line:
            if (
                peak.element != a.suggested_element
                or peak.line != a.suggested_line
            ):
                peak.element = a.suggested_element
                peak.line = a.suggested_line
                n_applied += 1

        if a.secondary_seed_energy is not None:
            e_seed = float(a.secondary_seed_energy)
            if all(abs(e_seed - e0) > 0.04 for e0 in existing_energies):
                seeds.append({
                    'energy': e_seed,
                    'element': a.suggested_element if (
                        a.suggested_element and a.suggested_element != a.current_element
                    ) else None,
                    'line': a.suggested_line if (
                        a.suggested_element and a.suggested_element != a.current_element
                    ) else None,
                    'is_tube_line': False,
                })
                existing_energies.append(e_seed)

    report.n_applied = n_applied
    if n_applied or seeds:
        report.summary_lines.append(
            f"  Applied: {n_applied} relabel(s), {len(seeds)} overlap seed(s) added"
        )
    return peaks, seeds, n_applied


# Common XRF survey set used for pre-fit auto-ID after peak find
COMMON_XRF_SYMBOLS = (
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca',
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Ag', 'Cd', 'Sn', 'Sb', 'Ba', 'W', 'Pb', 'Bi',
)


def common_xrf_elements() -> List[Dict]:
    """Return [{symbol, z}, ...] for the common XRF survey set."""
    from core.advanced_peak_fitting import get_element_z

    out = []
    for symbol in COMMON_XRF_SYMBOLS:
        z = get_element_z(symbol)
        if z:
            out.append({'symbol': symbol, 'z': int(z)})
    return out


def candidates_at_energy(
    energy_kev: float,
    candidate_elements: Optional[Sequence[Dict]] = None,
    energy_tol_kev: float = 0.150,
    max_results: int = 12,
) -> List[Dict]:
    """
    Rank emission-line candidates near a clicked energy.

    Returns list of dicts sorted by |ΔE|:
      symbol, z, line, line_energy, delta_kev, delta_ev
    """
    e0 = float(energy_kev)
    candidates = list(candidate_elements) if candidate_elements else common_xrf_elements()
    hits: List[Dict] = []

    for elem in candidates:
        symbol = elem.get('symbol')
        z = elem.get('z')
        if not symbol or not z:
            continue
        z = int(z)
        line_map = _line_lookup(symbol, z)
        for line_name, line_e in _primary_lines(line_map):
            dist = abs(e0 - float(line_e))
            if dist > energy_tol_kev:
                continue
            hits.append({
                'symbol': symbol,
                'z': z,
                'line': line_name,
                'line_energy': float(line_e),
                'delta_kev': float(e0 - line_e),
                'delta_ev': float((e0 - line_e) * 1000.0),
                'abs_delta_kev': dist,
            })

    hits.sort(key=lambda h: (h['abs_delta_kev'], h['symbol'], h['line']))
    return hits[: max(1, int(max_results))]


def auto_id_peak_positions(
    peak_positions: Sequence[dict],
    candidate_elements: Optional[Sequence[Dict]] = None,
    energy_tol_kev: float = 0.080,
    require_confirming_line: bool = False,
    excitation_kv: float = 50.0,
) -> Tuple[List[dict], List[str], List[str]]:
    """
    Label unknown peak-find seeds by matching energies to element emission lines.

    Intended for the Peak Find → Elements workflow: detect peaks first, then
    auto-ID against a survey library (default: COMMON_XRF_SYMBOLS), then let
    the user confirm on the Elements tab before Fitting.

    An element is only identified if its diagnostic line is present (Kα when
    it can be excited, otherwise Lα, else Mα). Minor lines such as Pb Mα
    are labeled only after that diagnostic match — never used to introduce Pb.

    Args:
        peak_positions: Peak seed dicts (energy, element, line, is_tube_line, ...)
        candidate_elements: Optional [{symbol, z}, ...]; defaults to common XRF
        energy_tol_kev: Max |E_peak - E_line| for a match
        require_confirming_line: If True, only accept IDs with a second line nearby
        excitation_kv: Tube kV; lines above ~0.95× this cannot be diagnostic

    Returns:
        (updated_positions, identified_symbols, summary_lines)
    """
    candidates = list(candidate_elements) if candidate_elements else common_xrf_elements()
    candidates = [
        e for e in candidates
        if e.get('symbol') and e.get('z')
    ]

    e_max_kev = min(40.0, float(excitation_kv) * 0.95)

    # Diagnostic line per element, plus full primary-line catalog
    diag_catalog: List[Tuple[str, int, str, float]] = []
    line_catalog: Dict[str, List[Tuple[int, str, float]]] = {}
    line_maps: Dict[str, Dict[str, float]] = {}
    for elem in candidates:
        symbol = elem['symbol']
        z = int(elem['z'])
        line_map = _line_lookup(symbol, z)
        line_maps[symbol] = line_map
        diagnostic = _diagnostic_line(line_map, e_max_kev=e_max_kev)
        if diagnostic:
            diag_catalog.append((symbol, z, diagnostic[0], float(diagnostic[1])))
        line_catalog[symbol] = [
            (z, name, float(energy))
            for name, energy in _primary_lines(line_map)
        ]

    positions = [dict(p) for p in (peak_positions or [])]
    all_energies = [float(p.get('energy', 0.0)) for p in positions]

    def _partners_ok(symbol: str, line_name: str) -> bool:
        if not require_confirming_line:
            return True
        partners = _confirming_partners(line_name, line_maps.get(symbol, {}))
        return any(
            any(abs(ae - pe) <= energy_tol_kev for ae in all_energies)
            for _, pe in partners
        )

    # Pass 1: which elements have their diagnostic line on a measured peak?
    present: set = set()
    for pos in positions:
        if pos.get('is_tube_line'):
            continue
        if pos.get('element') and pos.get('line'):
            present.add(pos['element'])
            continue
        e_peak = float(pos.get('energy', 0.0))
        best = None
        for symbol, z, line_name, line_e in diag_catalog:
            dist = abs(e_peak - line_e)
            if dist > energy_tol_kev:
                continue
            if best is None or dist < best[0]:
                best = (dist, symbol, line_name)
        if best is None:
            continue
        _, symbol, line_name = best
        if _partners_ok(symbol, line_name):
            present.add(symbol)

    # Pass 2: assign each unlabeled peak to the nearest line of a present element
    identified: List[str] = []
    summary: List[str] = []
    n_labeled = 0

    for pos in positions:
        if pos.get('is_tube_line'):
            continue
        if pos.get('element') and pos.get('line'):
            if pos['element'] not in identified:
                identified.append(pos['element'])
            continue

        e_peak = float(pos.get('energy', 0.0))
        best = None  # (dist, symbol, z, line_name)
        for symbol in present:
            for z, line_name, line_e in line_catalog.get(symbol, []):
                dist = abs(e_peak - line_e)
                if dist > energy_tol_kev:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, symbol, z, line_name)

        if best is None:
            continue

        dist, symbol, z, line_name = best
        pos['element'] = symbol
        pos['line'] = line_name
        n_labeled += 1
        if symbol not in identified:
            identified.append(symbol)
        summary.append(
            f"  {e_peak:.3f} keV → {symbol} {line_name} "
            f"(Δ={dist*1000:.0f} eV)"
        )

    header = [
        f"Auto-ID: labeled {n_labeled} peak(s); "
        f"{len(identified)} element(s): {', '.join(identified) if identified else 'none'}"
    ]
    return positions, identified, header + summary
