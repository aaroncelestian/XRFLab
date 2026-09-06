"""
Per-voltage X-ray tube scatter profiles.

Detector FWHM is one FWHM(E) curve. Tube *shape* (relative intensities of
elastic Rh K/L lines and Compton) changes with tube kV and must be known
well — ratio deviations flag sample peaks overlapping tube lines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Instrument modes the user can run
DEFAULT_TUBE_KVS = (15.0, 30.0, 50.0)

# Reference line for relative intensities within a profile
REFERENCE_LINE_PREFERENCE = ('Kα1', 'Lα1', 'Lβ1')


@dataclass
class TubeProfile:
    """Measured or default tube scatter profile at one tube voltage."""

    tube_element: str
    tube_kv: float
    # Relative areas/intensities keyed by line name (Kα1, Lα1, Compton Kα, ...)
    line_ratios: Dict[str, float]
    compton_scale: float = 0.5  # Compton Kα / elastic Kα (or vs reference)
    scatter_angle_deg: float = 90.0
    compton_fwhm_kev: float = 0.250
    # Absolute reference peak area from the blank (optional, for scaling)
    reference_line: str = 'Kα1'
    reference_area: float = 0.0
    source: str = 'default'  # 'default' | 'measured'
    spectrum_path: Optional[str] = None
    measured_date: Optional[str] = None
    notes: str = ''
    scatterer: str = ''  # e.g. 'air', 'Fe', 'polymer'
    skipped_tube_lines: List[str] = field(default_factory=list)

    def ratio(self, line: str, default: float = 1.0) -> float:
        return float(self.line_ratios.get(line, default))

    def expected_relative_to(self, line: str, reference: Optional[str] = None) -> float:
        """Return line_ratios[line] / line_ratios[reference]."""
        ref = reference or self.reference_line
        ref_val = self.line_ratios.get(ref)
        line_val = self.line_ratios.get(line)
        if ref_val is None or ref_val <= 0 or line_val is None:
            return float('nan')
        return float(line_val) / float(ref_val)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'TubeProfile':
        return cls(
            tube_element=str(data.get('tube_element', 'Rh')),
            tube_kv=float(data['tube_kv']),
            line_ratios={str(k): float(v) for k, v in (data.get('line_ratios') or {}).items()},
            compton_scale=float(data.get('compton_scale', 0.5)),
            scatter_angle_deg=float(data.get('scatter_angle_deg', 90.0)),
            compton_fwhm_kev=float(data.get('compton_fwhm_kev', 0.250)),
            reference_line=str(data.get('reference_line', 'Kα1')),
            reference_area=float(data.get('reference_area', 0.0)),
            source=str(data.get('source', 'default')),
            spectrum_path=data.get('spectrum_path'),
            measured_date=data.get('measured_date'),
            notes=str(data.get('notes', '')),
            scatterer=str(data.get('scatterer', '')),
            skipped_tube_lines=list(data.get('skipped_tube_lines') or []),
        )


def default_tube_profile(tube_element: str = 'Rh', tube_kv: float = 50.0) -> TubeProfile:
    """
    Approximate relative intensities when no blank has been measured.

    These are order-of-magnitude defaults (not instrument-specific).
    At 15 kV Rh K is off; L lines dominate scatter.
    """
    kv = float(tube_kv)
    if kv < 20.5:
        # Below Rh K-edge (~23.2 keV binding; K emission needs >~23 keV practically
        # but characteristic Kα needs tube above ~20.2 keV). Treat <20.5 as L-only.
        ratios = {
            'Lα1': 1.0,
            'Lβ1': 0.55,
        }
        ref = 'Lα1'
        compton = 0.0
    elif kv < 40.0:
        ratios = {
            'Kα1': 1.0,
            'Kα2': 0.50,
            'Kβ1': 0.18,
            'Lα1': 0.35,
            'Lβ1': 0.20,
            'Compton Kα': 0.45,
            'Compton Kβ': 0.08,
        }
        ref = 'Kα1'
        compton = 0.45
    else:
        ratios = {
            'Kα1': 1.0,
            'Kα2': 0.50,
            'Kβ1': 0.20,
            'Lα1': 0.25,
            'Lβ1': 0.14,
            'Compton Kα': 0.50,
            'Compton Kβ': 0.10,
        }
        ref = 'Kα1'
        compton = 0.50

    return TubeProfile(
        tube_element=tube_element,
        tube_kv=kv,
        line_ratios=ratios,
        compton_scale=compton,
        reference_line=ref,
        source='default',
        notes='Built-in approximate ratios; replace with a scatter / known-material measurement',
    )


# Known scatterers for tube-profile measurement when a true empty-beam
# blank is not possible. Air is valid; a polymer or foil is often cleaner.
SCATTERER_PRESETS: Dict[str, dict] = {
    "air": {
        "label": "Air scatter (Ar)",
        "elements": ["Ar"],
        "hint": "Ar Kα (~2.96 keV) can sit near Rh L; those L lines are skipped.",
    },
    "polymer": {
        "label": "Polymer / carbon (no XRF lines)",
        "elements": [],
        "hint": "Best practical blank: C/O/H have no lines in the typical XRF range.",
    },
    "Al": {
        "label": "Aluminum",
        "elements": ["Al"],
        "hint": "Al K is far from Rh L and Rh K — a good foil scatterer.",
    },
    "Si": {
        "label": "Silicon / quartz / glass",
        "elements": ["Si"],
        "hint": "Si Kα ~1.74 keV; keep away from materials with lines near the tube anode.",
    },
    "Mg": {
        "label": "Magnesium",
        "elements": ["Mg"],
        "hint": "Mg K is well below Rh L.",
    },
    "Ti": {
        "label": "Titanium",
        "elements": ["Ti"],
        "hint": "Sample lines are fitted as Ti; overlapping tube lines are skipped.",
    },
    "Fe": {
        "label": "Iron / steel",
        "elements": ["Fe"],
        "hint": "Use a foil you already have for FWHM. Fe K does not overlap Rh K/L.",
    },
    "Cu": {
        "label": "Copper",
        "elements": ["Cu"],
        "hint": "Cu K is between Rh L and Rh K — usually a clean scatterer.",
    },
    "Zn": {
        "label": "Zinc",
        "elements": ["Zn"],
        "hint": "Zn K is between Rh L and Rh K.",
    },
    "other": {
        "label": "Other element…",
        "elements": None,
        "hint": "Pick the main element in the scatterer. Overlapping tube lines are skipped.",
    },
}

# Two typical SDD widths — skip tube lines this close to a sample line
TUBE_SAMPLE_OVERLAP_KEV = 0.30


def _element_dicts(symbols: Optional[List[str]]) -> List[dict]:
    from core.advanced_peak_fitting import get_element_z

    out = []
    for symbol in symbols or []:
        sym = str(symbol).strip()
        if not sym:
            continue
        z = get_element_z(sym)
        if z:
            out.append({"symbol": sym, "z": int(z)})
    return out


def sample_emission_energies(symbols: Optional[List[str]]) -> List[Tuple[str, str, float]]:
    """Return (symbol, line, energy_keV) for major lines of the scatterer."""
    from core.xray_data import get_element_lines

    lines_out: List[Tuple[str, str, float]] = []
    for elem in _element_dicts(symbols):
        symbol = elem["symbol"]
        series = get_element_lines(symbol, elem["z"])
        for entries in series.values():
            for item in entries:
                energy = float(item.get("energy") or 0.0)
                name = str(item.get("name") or "")
                if energy >= 0.5 and name:
                    lines_out.append((symbol, name, energy))
    return lines_out


def tube_line_overlaps_sample(
    energy_kev: float,
    sample_lines: List[Tuple[str, str, float]],
    overlap_tol_kev: float = TUBE_SAMPLE_OVERLAP_KEV,
) -> Optional[Tuple[str, str, float]]:
    """If *energy_kev* is within *overlap_tol_kev* of a sample line, return that line."""
    e0 = float(energy_kev)
    best = None
    best_d = overlap_tol_kev
    for symbol, name, energy in sample_lines:
        delta = abs(e0 - float(energy))
        if delta <= best_d:
            best_d = delta
            best = (symbol, name, energy)
    return best


def tube_areas_from_peaks(
    peaks,
    sample_elements: Optional[List[str]] = None,
    overlap_tol_kev: float = TUBE_SAMPLE_OVERLAP_KEV,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Sum tube/Compton peak areas, dropping lines that overlap scatterer emission.

    Returns (areas_by_line, skipped_line_names).
    """
    sample_lines = sample_emission_energies(sample_elements)
    areas: Dict[str, float] = {}
    skipped: List[str] = []
    for peak in peaks or []:
        if not getattr(peak, "is_tube_line", False):
            continue
        line = getattr(peak, "line", None)
        if not line:
            continue
        energy = float(getattr(peak, "energy", 0.0) or 0.0)
        hit = tube_line_overlaps_sample(energy, sample_lines, overlap_tol_kev)
        if hit is not None:
            if line not in skipped:
                skipped.append(str(line))
            continue
        areas[str(line)] = areas.get(str(line), 0.0) + float(getattr(peak, "area", 0.0) or 0.0)
    return areas, skipped


@dataclass
class TubeProfileLibrary:
    """Collection of tube profiles keyed by nominal kV."""

    tube_element: str = 'Rh'
    available_kvs: Tuple[float, ...] = DEFAULT_TUBE_KVS
    profiles: Dict[str, TubeProfile] = field(default_factory=dict)
    # Optional: FWHM measurement tags from multi-kV campaigns
    fwhm_kv_tags: List[dict] = field(default_factory=list)

    @staticmethod
    def _key(kv: float) -> str:
        return f"{float(kv):g}"

    def set_profile(self, profile: TubeProfile):
        self.profiles[self._key(profile.tube_kv)] = profile
        self.tube_element = profile.tube_element

    def get_profile(self, tube_kv: float, fallback_default: bool = True) -> Optional[TubeProfile]:
        """Exact key match, else nearest available kV, else default profile."""
        key = self._key(tube_kv)
        if key in self.profiles:
            return self.profiles[key]

        if self.profiles:
            nearest_kv = min(
                (float(k) for k in self.profiles.keys()),
                key=lambda k: abs(k - float(tube_kv)),
            )
            return self.profiles[self._key(nearest_kv)]

        if fallback_default:
            return default_tube_profile(self.tube_element, tube_kv)
        return None

    def select_for_excitation(self, excitation_kev: float) -> TubeProfile:
        """
        Map spectrum excitation energy to the nearest instrument mode
        (15 / 30 / 50) then return that profile (measured or default).
        """
        modes = self.available_kvs or DEFAULT_TUBE_KVS
        nearest_mode = min(modes, key=lambda k: abs(k - float(excitation_kev)))
        profile = self.get_profile(nearest_mode, fallback_default=True)
        return profile

    def to_dict(self) -> dict:
        return {
            'tube_element': self.tube_element,
            'available_kvs': list(self.available_kvs),
            'profiles': {k: p.to_dict() for k, p in self.profiles.items()},
            'fwhm_kv_tags': list(self.fwhm_kv_tags),
            'saved_date': datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TubeProfileLibrary':
        lib = cls(
            tube_element=str(data.get('tube_element', 'Rh')),
            available_kvs=tuple(
                float(x) for x in (data.get('available_kvs') or DEFAULT_TUBE_KVS)
            ),
            fwhm_kv_tags=list(data.get('fwhm_kv_tags') or []),
        )
        for key, pdata in (data.get('profiles') or {}).items():
            lib.profiles[str(key)] = TubeProfile.from_dict(pdata)
        return lib

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'TubeProfileLibrary':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def measure_tube_profile_from_spectrum(
    energy,
    counts,
    tube_element: str = 'Rh',
    tube_kv: float = 50.0,
    scatter_angle_deg: float = 90.0,
    compton_fwhm_kev: float = 0.250,
    background_method: str = 'snip',
    peak_shape: str = 'gaussian',
    spectrum_path: Optional[str] = None,
) -> TubeProfile:
    """
    Fit tube + Compton lines on a blank / scatter spectrum and build ratios.

    Relative intensities are peak areas normalized to the reference line
    (Kα1 if present, else Lα1).
    """
    from core.fitting import SpectrumFitter

    fitter = SpectrumFitter()
    result = fitter.fit_spectrum(
        energy=np.asarray(energy, dtype=float),
        counts=np.asarray(counts, dtype=float),
        elements=[],
        background_method=background_method,
        peak_shape=peak_shape,
        auto_find_peaks=False,
        tube_element=tube_element,
        excitation_kv=tube_kv,
        include_tube_lines=True,
        include_compton=True,
        scatter_angle_deg=scatter_angle_deg,
        compton_fwhm_kev=compton_fwhm_kev,
    )

    areas: Dict[str, float] = {}
    for peak in result.peaks:
        if not getattr(peak, 'is_tube_line', False):
            continue
        if not peak.line:
            continue
        areas[peak.line] = areas.get(peak.line, 0.0) + float(peak.area)

    if not areas:
        profile = default_tube_profile(tube_element, tube_kv)
        profile.notes = 'Measurement found no tube peaks; kept defaults'
        profile.spectrum_path = spectrum_path
        return profile

    ref = None
    for cand in REFERENCE_LINE_PREFERENCE:
        if cand in areas and areas[cand] > 0:
            ref = cand
            break
    if ref is None:
        ref = max(areas.keys(), key=lambda k: areas[k])

    ref_area = float(areas[ref])
    ratios = {name: float(a) / ref_area for name, a in areas.items()}

    compton_scale = float(ratios.get('Compton Kα', 0.0))
    if 'Kα1' in ratios and ratios['Kα1'] > 0 and 'Compton Kα' in areas:
        # Prefer Compton / elastic Kα1 even if reference is L
        k_area = areas.get('Kα1', 0.0)
        if k_area > 0:
            compton_scale = float(areas['Compton Kα']) / float(k_area)

    return TubeProfile(
        tube_element=tube_element,
        tube_kv=float(tube_kv),
        line_ratios=ratios,
        compton_scale=compton_scale,
        scatter_angle_deg=float(scatter_angle_deg),
        compton_fwhm_kev=float(compton_fwhm_kev),
        reference_line=ref,
        reference_area=ref_area,
        source='measured',
        spectrum_path=spectrum_path,
        measured_date=datetime.now().isoformat(),
        notes=f'Measured from blank; {len(ratios)} tube/Compton lines',
    )


def compare_fitted_tube_ratios(
    fitted_peaks,
    profile: TubeProfile,
    tolerance: float = 0.35,
) -> List[dict]:
    """
    Compare fitted tube-line area ratios to the profile.

    Returns flags when a ratio is *elevated* beyond ``tolerance``
    (fractional). High ratios often mean a sample peak under that tube line.
    Missing/weak tube lines are not flagged as overlaps.
    """
    tube_areas: Dict[str, float] = {}
    for peak in fitted_peaks or []:
        if not getattr(peak, 'is_tube_line', False):
            continue
        line = getattr(peak, 'line', None)
        if not line:
            continue
        tube_areas[line] = tube_areas.get(line, 0.0) + float(peak.area)

    if not tube_areas or not profile.line_ratios:
        return []

    ref = profile.reference_line
    if ref not in tube_areas or tube_areas[ref] <= 0:
        # Fall back to strongest fitted tube line that exists in profile
        candidates = [k for k in tube_areas if k in profile.line_ratios]
        if not candidates:
            return []
        ref = max(candidates, key=lambda k: tube_areas[k])

    ref_fit = tube_areas[ref]
    flags = []
    for line, fit_area in tube_areas.items():
        if line == ref:
            continue
        expected = profile.expected_relative_to(line, reference=ref)
        if expected != expected or expected <= 0:  # NaN or non-positive
            continue
        observed = float(fit_area) / float(ref_fit)
        # Ignore absent/weak lines — not an overlap signature
        if observed < 0.05 * expected:
            continue
        frac_err = (observed - expected) / expected
        # Only elevated ratios suggest sample intensity under the tube line
        if frac_err > tolerance:
            flags.append({
                'line': line,
                'reference': ref,
                'observed_ratio': observed,
                'expected_ratio': expected,
                'fractional_error': frac_err,
                'message': (
                    f"{profile.tube_element} {line}/{ref} = {observed:.2f} "
                    f"(profile {expected:.2f}, {frac_err:+.0%}) — "
                    f"possible overlap under {line}"
                ),
            })
    return flags


def attach_profile_to_peak_seeds(peak_positions: List[dict], profile: TubeProfile) -> List[dict]:
    """Copy expected relative intensity from profile onto tube/Compton seeds."""
    out = []
    for pos in peak_positions or []:
        p = dict(pos)
        if p.get('is_tube_line') and p.get('line'):
            line = p['line']
            if line in profile.line_ratios:
                p['expected_relative_intensity'] = float(profile.line_ratios[line])
                p['tube_profile_kv'] = float(profile.tube_kv)
                p['tube_profile_source'] = profile.source
            if str(line).startswith('Compton') and profile.compton_fwhm_kev:
                p.setdefault('fixed_fwhm', float(profile.compton_fwhm_kev))
        out.append(p)
    return out
