"""
Main fitting engine for XRF spectra
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from core.background import BackgroundModeler
from core.peak_fitting import PeakFitter, Peak
from core.xray_data import get_element_lines, get_tube_lines, get_tube_compton_lines


@dataclass
class FitResult:
    """Results from spectrum fitting"""
    background: np.ndarray
    fitted_spectrum: np.ndarray
    residuals: np.ndarray
    peaks: List[Peak]
    statistics: Dict
    tube_overlap_flags: List[Dict] = None

    def __post_init__(self):
        if self.tube_overlap_flags is None:
            self.tube_overlap_flags = []

    def to_dict(self, include_arrays: bool = False) -> dict:
        """Serialize fit metadata; arrays are optional (HDF5 stores them separately)."""
        peaks = []
        for p in self.peaks or []:
            if hasattr(p, "to_dict"):
                peaks.append(p.to_dict())
            elif isinstance(p, dict):
                peaks.append(p)
        data = {
            "peaks": peaks,
            "statistics": dict(self.statistics or {}),
            "tube_overlap_flags": list(self.tube_overlap_flags or []),
        }
        if include_arrays:
            data["background"] = np.asarray(self.background).tolist()
            data["fitted_spectrum"] = np.asarray(self.fitted_spectrum).tolist()
            data["residuals"] = np.asarray(self.residuals).tolist()
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        background=None,
        fitted_spectrum=None,
        residuals=None,
    ) -> "FitResult":
        from core.peak_fitting import Peak as PeakType

        peaks = []
        for p in data.get("peaks") or []:
            if isinstance(p, PeakType):
                peaks.append(p)
            elif isinstance(p, dict):
                peaks.append(PeakType.from_dict(p))
        if background is None and data.get("background") is not None:
            background = np.asarray(data["background"], dtype=np.float64)
        if fitted_spectrum is None and data.get("fitted_spectrum") is not None:
            fitted_spectrum = np.asarray(data["fitted_spectrum"], dtype=np.float64)
        if residuals is None and data.get("residuals") is not None:
            residuals = np.asarray(data["residuals"], dtype=np.float64)
        return cls(
            background=np.asarray(background if background is not None else []),
            fitted_spectrum=np.asarray(
                fitted_spectrum if fitted_spectrum is not None else []
            ),
            residuals=np.asarray(residuals if residuals is not None else []),
            peaks=peaks,
            statistics=dict(data.get("statistics") or {}),
            tube_overlap_flags=list(data.get("tube_overlap_flags") or []),
        )
    
    def __str__(self):
        return (f"Fit Result: {len(self.peaks)} peaks, "
                f"χ²ᵣ = {self.statistics['reduced_chi_squared']:.2f}, "
                f"R² = {self.statistics['r_squared']:.4f}")


class SpectrumFitter:
    """Main fitting engine for XRF spectra"""
    
    def __init__(self, detector=None):
        self.background_modeler = BackgroundModeler()
        self.peak_fitter = PeakFitter(detector=detector)
        self.tube_profile_library = None  # Optional TubeProfileLibrary
        self.peak_fitter.activate()

    def set_tube_profile_library(self, library):
        """Attach per-kV tube profile library for ratio constraints / flags."""
        self.tube_profile_library = library

    def apply_instrument_state(self, instrument_state):
        """Apply an InstrumentState (detector + tube profiles) to this fitter."""
        if instrument_state is None:
            return
        instrument_state.apply_to_fitter(self)
        self.peak_fitter.activate()

    @staticmethod
    def _element_dicts(elements) -> List[Dict]:
        """Accept Analysis dicts or bare symbols; return [{'symbol', 'z'}, ...]."""
        from core.advanced_peak_fitting import get_element_z

        out: List[Dict] = []
        seen = set()
        for item in elements or []:
            if isinstance(item, dict):
                symbol = str(item.get("symbol") or "").strip()
                z = item.get("z") or 0
            else:
                symbol = str(item or "").strip()
                z = 0
            if not symbol:
                continue
            try:
                z = int(z or 0)
            except (TypeError, ValueError):
                z = 0
            if z <= 0:
                z = int(get_element_z(symbol) or 0)
            if z <= 0 or symbol in seen:
                continue
            seen.add(symbol)
            out.append({"symbol": symbol, "z": z})
        return out
    
    def build_peak_positions(self, energy, counts_bg_subtracted=None, elements=None,
                             auto_find_peaks=True, tube_element='Rh',
                             excitation_kv=50.0, include_tube_lines=True,
                             include_compton=True, scatter_angle_deg=90.0,
                             compton_fwhm_kev=0.250, **kwargs):
        """
        Build the list of peak seed positions (element lines, tube lines, auto-find).

        Returns:
            List of dicts with keys: energy, element, line, is_tube_line
            (optional fixed_fwhm / exclusion_half_width_kev for Compton)
        """
        peak_positions = []
        elements = self._element_dicts(elements)
        e_lo = max(float(energy[0]), PeakFitter.MIN_PEAK_ENERGY_KEV)
        e_hi = float(energy[-1])

        if elements:
            print(f"Using emission lines from {len(elements)} elements...")
            for elem in elements:
                symbol = elem.get('symbol', '')
                z = elem.get('z', 0)

                if symbol and z:
                    lines = get_element_lines(symbol, z)

                    major_lines = {
                        'K': ['Kα1', 'Kα2', 'Kβ1'],
                        'L': ['Lα1', 'Lα2', 'Lβ1', 'Lβ2'],
                        'M': ['Mα1', 'Mα2']
                    }

                    for series in ['K', 'L', 'M']:
                        for line in lines.get(series, []):
                            line_name = line['name']
                            line_energy = line['energy']

                            if line_name in major_lines.get(series, []):
                                if e_lo <= line_energy <= e_hi:
                                    peak_positions.append({
                                        'energy': line_energy,
                                        'element': symbol,
                                        'line': line_name,
                                        'is_tube_line': False
                                    })

        if include_tube_lines and tube_element:
            print(f"Including {tube_element} tube lines at {excitation_kv} keV...")
            tube_lines = get_tube_lines(tube_element, excitation_kv)

            for series in ['K', 'L']:
                for line in tube_lines.get(series, []):
                    line_name = line['name']
                    line_energy = line['energy']

                    if series == 'K' and line_name in ['Kα1', 'Kα2', 'Kβ1']:
                        if e_lo <= line_energy <= e_hi:
                            peak_positions.append({
                                'energy': line_energy,
                                'element': tube_element,
                                'line': line_name,
                                'is_tube_line': True
                            })
                    elif series == 'L' and line_name in ['Lα1', 'Lβ1']:
                        if e_lo <= line_energy <= e_hi:
                            peak_positions.append({
                                'energy': line_energy,
                                'element': tube_element,
                                'line': line_name,
                                'is_tube_line': True
                            })

            # Broad inelastic Compton tube scatter (~19 keV for Rh Kα)
            if include_compton:
                # Prefer measured profile geometry / width when available
                if self.tube_profile_library is not None:
                    profile = self.tube_profile_library.select_for_excitation(
                        excitation_kv
                    )
                    if profile is not None and profile.source == 'measured':
                        scatter_angle_deg = profile.scatter_angle_deg
                        compton_fwhm_kev = profile.compton_fwhm_kev

                compton_lines = get_tube_compton_lines(
                    tube_element=tube_element,
                    excitation_kv=excitation_kv,
                    scatter_angle_deg=scatter_angle_deg,
                    fwhm_kev=compton_fwhm_kev,
                )
                n_c = 0
                for c in compton_lines:
                    if e_lo <= c['energy'] <= e_hi:
                        peak_positions.append(c)
                        n_c += 1
                if n_c:
                    print(
                        f"Including {n_c} {tube_element} Compton line(s) "
                        f"(θ={scatter_angle_deg:.0f}°, FWHM={compton_fwhm_kev*1000:.0f} eV)"
                    )

        # Attach expected relative intensities from tube profile
        if self.tube_profile_library is not None and include_tube_lines:
            from core.tube_profile import attach_profile_to_peak_seeds
            profile = self.tube_profile_library.select_for_excitation(excitation_kv)
            peak_positions = attach_profile_to_peak_seeds(peak_positions, profile)
            print(
                f"Tube profile @ {profile.tube_kv:g} kV "
                f"({profile.source}, ref={profile.reference_line})"
            )

        if auto_find_peaks and counts_bg_subtracted is not None:
            print("Auto-detecting peaks...")
            auto_peaks = self.peak_fitter.find_peaks(
                energy, counts_bg_subtracted,
                prominence=kwargs.get('prominence', None),
                distance=kwargs.get('distance', None),
                height=kwargs.get('min_height', kwargs.get('height', None)),
                prominence_percent=kwargs.get('prominence_percent', None),
                min_separation_ev=kwargs.get('min_separation_ev', None),
            )

            match_tol = 0.1
            if kwargs.get('min_separation_ev') is not None:
                match_tol = max(0.05, float(kwargs['min_separation_ev']) / 1000.0)

            n_skipped_compton = 0
            for peak_energy, peak_height in auto_peaks:
                near_existing = False
                for pos in peak_positions:
                    # Wide exclusion under Compton / other fixed-width tube features
                    excl = pos.get('exclusion_half_width_kev')
                    if excl is not None:
                        tol = max(match_tol, float(excl))
                    elif pos.get('fixed_fwhm') is not None:
                        tol = max(match_tol, 1.5 * float(pos['fixed_fwhm']))
                    else:
                        tol = match_tol
                    if abs(peak_energy - pos['energy']) < tol:
                        near_existing = True
                        if pos.get('line') and str(pos['line']).startswith('Compton'):
                            n_skipped_compton += 1
                        break

                if not near_existing:
                    peak_positions.append({
                        'energy': peak_energy,
                        'element': None,
                        'line': None,
                        'is_tube_line': False,
                    })

            n_unknown = sum(1 for p in peak_positions if p.get('element') is None)
            msg = (
                f"Auto-detected {len(auto_peaks)} peaks "
                f"({n_unknown} unknown added)"
            )
            if n_skipped_compton:
                msg += f"; skipped {n_skipped_compton} under Compton"
            print(msg)

        return peak_positions

    def apply_selected_elements_to_positions(
        self, peak_positions, energy, elements, match_tol_kev=0.1
    ):
        """
        Re-label peaks from the current Elements-tab selection.

        Clears previous sample labels (tube lines kept), then:
        - labels unknowns near selected-element lines
        - adds missing theoretical line seeds

        This lets the user uncheck false IDs / check missing ones and refit.
        """
        positions = [
            {
                'energy': float(p['energy']),
                'element': p.get('element'),
                'line': p.get('line'),
                'is_tube_line': bool(p.get('is_tube_line', False)),
                **({'fixed_fwhm': float(p['fixed_fwhm'])}
                   if p.get('fixed_fwhm') is not None else {}),
                **({'exclusion_half_width_kev': float(p['exclusion_half_width_kev'])}
                   if p.get('exclusion_half_width_kev') is not None else {}),
            }
            for p in (peak_positions or [])
            if float(p.get('energy', 0.0)) >= PeakFitter.MIN_PEAK_ENERGY_KEV
        ]

        # Drop prior sample labels so unchecked elements cannot stick around
        n_cleared = 0
        for pos in positions:
            if pos.get('is_tube_line'):
                continue
            if pos.get('element') or pos.get('line'):
                n_cleared += 1
            pos['element'] = None
            pos['line'] = None

        if not elements:
            if n_cleared:
                print(f"Cleared {n_cleared} sample peak label(s) (no elements selected)")
            return positions

        elements = self._element_dicts(elements)
        if not elements:
            return positions

        major_lines = {
            'K': {'Kα1', 'Kα2', 'Kα', 'Kβ1', 'Kβ3', 'Kβ'},
            'L': {'Lα1', 'Lα2', 'Lα', 'Lβ1', 'Lβ2', 'Lβ'},
            'M': {'Mα1', 'Mα2', 'Mα'},
        }

        n_labeled = 0
        n_added = 0
        e_min = max(float(energy[0]), PeakFitter.MIN_PEAK_ENERGY_KEV)
        e_max = float(energy[-1])

        for elem in elements:
            symbol = elem.get('symbol', '')
            z = elem.get('z', 0)
            if not symbol or not z:
                continue

            lines = get_element_lines(symbol, z)
            for series in ['K', 'L', 'M']:
                for line in lines.get(series, []):
                    line_name = line['name']
                    line_energy = float(line['energy'])
                    if line_name not in major_lines.get(series, set()):
                        continue
                    if not (e_min <= line_energy <= e_max):
                        continue

                    # Closest existing peak within tolerance
                    best_idx = None
                    best_dist = match_tol_kev
                    for i, pos in enumerate(positions):
                        if pos.get('is_tube_line'):
                            continue
                        dist = abs(pos['energy'] - line_energy)
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i

                    if best_idx is not None:
                        pos = positions[best_idx]
                        # Prefer leaving an already-assigned closer match alone
                        if not pos.get('element'):
                            pos['element'] = symbol
                            pos['line'] = line_name
                            n_labeled += 1
                        continue

                    already = any(
                        p.get('element') == symbol
                        and p.get('line') == line_name
                        for p in positions
                    )
                    if not already:
                        positions.append({
                            'energy': line_energy,
                            'element': symbol,
                            'line': line_name,
                            'is_tube_line': False,
                        })
                        n_added += 1

        print(
            f"Applied {len(elements)} selected element(s) to peak list: "
            f"cleared {n_cleared}, labeled {n_labeled}, added {n_added} seed(s)"
        )
        return positions

    def _ensure_compton_positions(
        self,
        peak_positions,
        energy,
        tube_element='Rh',
        excitation_kv=50.0,
        scatter_angle_deg=90.0,
        compton_fwhm_kev=0.250,
    ):
        """Add missing Compton tube seeds to an existing peak list."""
        positions = list(peak_positions or [])
        compton_lines = get_tube_compton_lines(
            tube_element=tube_element,
            excitation_kv=excitation_kv,
            scatter_angle_deg=scatter_angle_deg,
            fwhm_kev=compton_fwhm_kev,
        )
        e_min = max(float(energy[0]), PeakFitter.MIN_PEAK_ENERGY_KEV)
        e_max = float(energy[-1])
        n_added = 0
        for c in compton_lines:
            if not (e_min <= c['energy'] <= e_max):
                continue
            already = any(
                p.get('is_tube_line')
                and p.get('line') == c['line']
                and abs(float(p['energy']) - float(c['energy'])) < 0.15
                for p in positions
            )
            if not already:
                positions.append(dict(c))
                n_added += 1
        if n_added:
            print(f"Added {n_added} Compton seed(s) to peak list")
        return positions

    def fit_spectrum(self, energy, counts, elements=None, 
                    background_method='snip', peak_shape='gaussian',
                    auto_find_peaks=True, tube_element='Rh', 
                    excitation_kv=50.0, include_tube_lines=True,
                    peak_positions=None, **kwargs):
        """
        Fit XRF spectrum with background and peaks
        
        Args:
            energy: Energy array (keV)
            counts: Counts array
            elements: List of element dicts with 'symbol' and 'z' keys
            background_method: 'snip', 'polynomial', 'linear', 'adaptive', 'none'
            peak_shape: 'gaussian', 'voigt', 'pseudo_voigt'
            auto_find_peaks: If True, automatically find peaks
            peak_positions: Optional pre-built peak seeds (skips rebuild when provided)
            **kwargs: Additional parameters for background/peak fitting
            
        Returns:
            FitResult object
        """
        # Ensure class-level PeakFitter helpers use this instance's detector
        self.peak_fitter.activate()

        # Step 1: Estimate background
        print(f"Estimating background using {background_method} method...")
        background = self.background_modeler.estimate_background(
            energy, counts, method=background_method, **kwargs
        )
        
        # Step 2: Subtract background
        counts_bg_subtracted = self.background_modeler.subtract_background(
            counts, background
        )
        
        # Step 3: Identify peak positions (or use caller-provided list)
        if peak_positions is not None:
            peak_positions = [
                {
                    'energy': float(p['energy']),
                    'element': p.get('element'),
                    'line': p.get('line'),
                    'is_tube_line': p.get('is_tube_line', False),
                    **({'fixed_fwhm': float(p['fixed_fwhm'])}
                       if p.get('fixed_fwhm') is not None else {}),
                    **({'exclusion_half_width_kev': float(p['exclusion_half_width_kev'])}
                       if p.get('exclusion_half_width_kev') is not None else {}),
                }
                for p in peak_positions
                if float(p['energy']) >= PeakFitter.MIN_PEAK_ENERGY_KEV
            ]
            print(f"Using {len(peak_positions)} caller-provided peak positions...")
            # Peak-find lists are mostly unlabeled; fold in Elements-tab selection
            match_tol = 0.1
            if kwargs.get('min_separation_ev') is not None:
                match_tol = max(0.05, float(kwargs['min_separation_ev']) / 1000.0)
            peak_positions = self.apply_selected_elements_to_positions(
                peak_positions, energy, elements, match_tol_kev=match_tol
            )
            # Keep Compton in the editable list path when enabled
            if include_tube_lines and kwargs.get('include_compton', True):
                peak_positions = self._ensure_compton_positions(
                    peak_positions,
                    energy,
                    tube_element=tube_element,
                    excitation_kv=excitation_kv,
                    scatter_angle_deg=kwargs.get('scatter_angle_deg', 90.0),
                    compton_fwhm_kev=kwargs.get('compton_fwhm_kev', 0.250),
                )
        else:
            peak_positions = self.build_peak_positions(
                energy,
                counts_bg_subtracted=counts_bg_subtracted,
                elements=elements,
                auto_find_peaks=auto_find_peaks,
                tube_element=tube_element,
                excitation_kv=excitation_kv,
                include_tube_lines=include_tube_lines,
                **kwargs,
            )
        
        # Step 4: Fit peaks with tube-profile priors + known overlap doublets
        print(f"Fitting {len(peak_positions)} peaks using {peak_shape} shape...")
        fitted_peaks = []
        residual_counts = np.asarray(counts_bg_subtracted, dtype=float).copy()
        tube_constraint_notes = []

        profile = None
        if self.tube_profile_library is not None:
            profile = self.tube_profile_library.select_for_excitation(excitation_kv)

        from core.tube_constraints import (
            find_overlap_pairs,
            fit_overlap_doublet,
            fit_peak_with_amplitude_prior,
            expected_tube_amplitude,
            subtract_peak_from_residual,
            pick_tube_reference_peak,
            DEFAULT_AMPLITUDE_PRIOR_WEIGHT,
        )

        overlap_pairs, remaining_positions = find_overlap_pairs(peak_positions)
        if overlap_pairs:
            print(f"Known tube/sample overlap pairs: {len(overlap_pairs)}")
            for pair in overlap_pairs:
                print(
                    f"  {pair.tube_pos.get('element')} {pair.tube_pos.get('line')} + "
                    f"{pair.sample_pos.get('element')} {pair.sample_pos.get('line')} "
                    f"(ΔE={pair.separation_kev*1000:.0f} eV)"
                )

        def _local_height(pos):
            e0 = pos['energy']
            idx = int(np.argmin(np.abs(energy - e0)))
            return float(residual_counts[idx])

        def _label_peak(peak, pos, is_tube=None):
            peak.element = pos.get('element')
            peak.line = pos.get('line')
            peak.is_tube_line = (
                bool(pos.get('is_tube_line', False))
                if is_tube is None else bool(is_tube)
            )
            if pos.get('fixed_fwhm') is not None:
                peak.fixed_fwhm = float(pos['fixed_fwhm'])
            return peak

        # --- 4a: Fit non-overlap peaks (tube reference early when possible) ---
        tube_remaining = [
            p for p in remaining_positions
            if p.get('is_tube_line') and not str(p.get('line', '')).startswith('Compton')
        ]
        compton_remaining = [
            p for p in remaining_positions
            if p.get('is_tube_line') and str(p.get('line', '')).startswith('Compton')
        ]
        sample_remaining = [
            p for p in remaining_positions if not p.get('is_tube_line')
        ]

        # Prefer fitting profile reference tube line first (for amplitude scale)
        ref_name = getattr(profile, 'reference_line', None) if profile else None
        tube_ordered = sorted(tube_remaining, key=_local_height, reverse=True)
        if ref_name:
            tube_ordered = sorted(
                tube_ordered,
                key=lambda p: (0 if p.get('line') == ref_name else 1, -_local_height(p)),
            )

        tube_ref_peak = None

        for pos in tube_ordered:
            known_line = bool(pos.get('element') and pos.get('line'))
            amp_prior = None
            if tube_ref_peak is not None and profile is not None and pos.get('line'):
                amp_prior = expected_tube_amplitude(
                    profile, pos['line'], tube_ref_peak.amplitude,
                    reference_line=tube_ref_peak.line,
                )

            if amp_prior is not None:
                peak = fit_peak_with_amplitude_prior(
                    energy, residual_counts,
                    initial_center=pos['energy'],
                    shape=peak_shape,
                    fixed_fwhm=pos.get('fixed_fwhm'),
                    fix_center=True,
                    amplitude_prior=amp_prior,
                    prior_weight=DEFAULT_AMPLITUDE_PRIOR_WEIGHT,
                    known_line=known_line,
                )
                if peak is not None:
                    tube_constraint_notes.append(
                        f"{pos.get('element')} {pos.get('line')}: "
                        f"soft prior A={amp_prior:.1f} → {peak.amplitude:.1f}"
                    )
            else:
                peak = self.peak_fitter.fit_single_peak(
                    energy, residual_counts,
                    initial_center=pos['energy'],
                    shape=peak_shape,
                    known_line=known_line,
                    fix_center=True,
                    fixed_fwhm=pos.get('fixed_fwhm'),
                )

            if peak is not None:
                _label_peak(peak, pos, is_tube=True)
                fitted_peaks.append(peak)
                residual_counts = subtract_peak_from_residual(
                    self.peak_fitter, energy, residual_counts, peak
                )
                if tube_ref_peak is None:
                    tube_ref_peak = peak
                elif profile and peak.line == getattr(profile, 'reference_line', None):
                    tube_ref_peak = peak

        # If reference wasn't first, re-pick from fitted tube peaks
        if profile is not None:
            tube_so_far = [p for p in fitted_peaks if p.is_tube_line]
            picked = pick_tube_reference_peak(tube_so_far, profile)
            if picked is not None:
                tube_ref_peak = picked

        # --- 4b: Overlap doublets (tube amp from profile × reference) ---
        for pair in overlap_pairs:
            amp_prior = None
            if tube_ref_peak is not None and profile is not None:
                amp_prior = expected_tube_amplitude(
                    profile,
                    pair.tube_pos.get('line'),
                    tube_ref_peak.amplitude,
                    reference_line=tube_ref_peak.line,
                )
            tube_peak, sample_peak = fit_overlap_doublet(
                energy, residual_counts,
                tube_energy=pair.tube_pos['energy'],
                sample_energy=pair.sample_pos['energy'],
                tube_amplitude_prior=amp_prior,
                shape=peak_shape,
                prior_weight=DEFAULT_AMPLITUDE_PRIOR_WEIGHT,
            )
            if tube_peak is None or sample_peak is None:
                # Fallback: sequential independent fits
                for pos, is_tube in (
                    (pair.tube_pos, True),
                    (pair.sample_pos, False),
                ):
                    peak = self.peak_fitter.fit_single_peak(
                        energy, residual_counts,
                        initial_center=pos['energy'],
                        shape=peak_shape,
                        known_line=True,
                        fix_center=True,
                        fixed_fwhm=pos.get('fixed_fwhm'),
                    )
                    if peak is not None:
                        _label_peak(peak, pos, is_tube=is_tube)
                        fitted_peaks.append(peak)
                        residual_counts = subtract_peak_from_residual(
                            self.peak_fitter, energy, residual_counts, peak
                        )
                continue

            _label_peak(tube_peak, pair.tube_pos, is_tube=True)
            _label_peak(sample_peak, pair.sample_pos, is_tube=False)
            fitted_peaks.extend([tube_peak, sample_peak])
            residual_counts = subtract_peak_from_residual(
                self.peak_fitter, energy, residual_counts, tube_peak
            )
            residual_counts = subtract_peak_from_residual(
                self.peak_fitter, energy, residual_counts, sample_peak
            )
            note = (
                f"Doublet {pair.tube_pos.get('element')} {pair.tube_pos.get('line')} + "
                f"{pair.sample_pos.get('element')} {pair.sample_pos.get('line')}: "
                f"A_tube={tube_peak.amplitude:.1f}, A_sample={sample_peak.amplitude:.1f}"
            )
            if amp_prior is not None:
                note += f" (tube prior {amp_prior:.1f})"
            tube_constraint_notes.append(note)
            print(f"  {note}")

        # --- 4c: Compton (wide, fixed) then remaining sample / unknown ---
        for pos in sorted(compton_remaining, key=_local_height, reverse=True):
            amp_prior = None
            if tube_ref_peak is not None and profile is not None and pos.get('line'):
                amp_prior = expected_tube_amplitude(
                    profile, pos['line'], tube_ref_peak.amplitude,
                    reference_line=tube_ref_peak.line,
                )
            peak = fit_peak_with_amplitude_prior(
                energy, residual_counts,
                initial_center=pos['energy'],
                shape=peak_shape,
                fixed_fwhm=pos.get('fixed_fwhm'),
                fix_center=True,
                amplitude_prior=amp_prior,
                prior_weight=DEFAULT_AMPLITUDE_PRIOR_WEIGHT,
                known_line=True,
            ) if amp_prior is not None else self.peak_fitter.fit_single_peak(
                energy, residual_counts,
                initial_center=pos['energy'],
                shape=peak_shape,
                known_line=True,
                fix_center=True,
                fixed_fwhm=pos.get('fixed_fwhm'),
            )
            if peak is not None:
                _label_peak(peak, pos, is_tube=True)
                fitted_peaks.append(peak)
                residual_counts = subtract_peak_from_residual(
                    self.peak_fitter, energy, residual_counts, peak
                )

        for pos in sorted(sample_remaining, key=_local_height, reverse=True):
            known_line = bool(pos.get('element') and pos.get('line'))
            local_h = _local_height(pos)
            residual_max = float(np.nanmax(residual_counts)) if residual_counts.size else 0.0
            fix_center = known_line or (
                residual_max > 0 and local_h < 0.05 * residual_max
            )
            peak = self.peak_fitter.fit_single_peak(
                energy, residual_counts,
                initial_center=pos['energy'],
                shape=peak_shape,
                known_line=known_line,
                fix_center=fix_center,
                fixed_fwhm=pos.get('fixed_fwhm'),
            )
            if peak is not None:
                _label_peak(peak, pos)
                fitted_peaks.append(peak)
                residual_counts = subtract_peak_from_residual(
                    self.peak_fitter, energy, residual_counts, peak
                )
        
        print(f"Successfully fitted {len(fitted_peaks)} peaks")
        if tube_constraint_notes:
            print(f"Tube constraints applied: {len(tube_constraint_notes)}")
        
        # Step 5: Reconstruct fitted spectrum
        fitted_spectrum = np.copy(background)
        
        for peak in fitted_peaks:
            # Use the correct peak shape for reconstruction
            if peak.shape == 'gaussian':
                sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                fitted_spectrum += self.peak_fitter.gaussian(
                    energy, peak.amplitude, peak.energy, sigma
                )
            elif peak.shape == 'voigt':
                sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                gamma = peak.shape_params.get('gamma', 0.05)
                fitted_spectrum += self.peak_fitter.voigt(
                    energy, peak.amplitude, peak.energy, sigma, gamma
                )
            elif peak.shape == 'pseudo_voigt':
                sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                eta = peak.shape_params.get('eta', 0.5)
                fitted_spectrum += self.peak_fitter.pseudo_voigt(
                    energy, peak.amplitude, peak.energy, sigma, eta
                )
            elif peak.shape == 'hypermet':
                sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                tail_amp = peak.shape_params.get('tail_amplitude', 0.1)
                tail_slope = peak.shape_params.get('tail_slope', 2.0)
                fitted_spectrum += self.peak_fitter.hypermet(
                    energy, peak.amplitude, peak.energy, sigma, tail_amp, tail_slope
                )
            elif peak.shape == 'tail_gaussian':
                sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                tail_frac = peak.shape_params.get('tail_fraction', 0.15)
                tail_sigma = peak.shape_params.get('tail_sigma', sigma * 3)
                fitted_spectrum += self.peak_fitter.tail_gaussian(
                    energy, peak.amplitude, peak.energy, sigma, tail_frac, tail_sigma
                )
            else:
                # Default to Gaussian if shape not recognized
                sigma = peak.fwhm / 2.355
                fitted_spectrum += self.peak_fitter.gaussian(
                    energy, peak.amplitude, peak.energy, sigma
                )
        
        # Step 6: Calculate residuals
        residuals = counts - fitted_spectrum
        
        # Step 7: Calculate fit statistics
        n_params = len(fitted_peaks) * 3 + 1  # 3 params per peak + background
        statistics = self.peak_fitter.calculate_fit_statistics(
            counts, fitted_spectrum, n_params
        )
        
        # Add iteration count (placeholder for now)
        statistics['iterations'] = 1
        if tube_constraint_notes:
            statistics['tube_constraint_notes'] = list(tube_constraint_notes)

        # Compare fitted tube line ratios to the instrument tube profile
        tube_overlap_flags = []
        if self.tube_profile_library is not None:
            from core.tube_profile import compare_fitted_tube_ratios
            profile = self.tube_profile_library.select_for_excitation(excitation_kv)
            tube_overlap_flags = compare_fitted_tube_ratios(fitted_peaks, profile)
            if tube_overlap_flags:
                statistics['tube_overlap_warnings'] = [
                    f['message'] for f in tube_overlap_flags
                ]
                print(f"Tube profile overlap flags: {len(tube_overlap_flags)}")
                for flag in tube_overlap_flags:
                    print(f"  ⚠ {flag['message']}")
        
        return FitResult(
            background=background,
            fitted_spectrum=fitted_spectrum,
            residuals=residuals,
            peaks=fitted_peaks,
            statistics=statistics,
            tube_overlap_flags=tube_overlap_flags,
        )
    
    def identify_peaks(self, peaks, tolerance=0.05):
        """
        Identify unknown peaks by matching to element emission lines
        
        Args:
            peaks: List of Peak objects
            tolerance: Energy tolerance for matching (keV)
            
        Returns:
            List of peaks with updated element/line information
        """
        # This is a placeholder for future implementation
        # Would search through xraylib database to identify peaks
        return peaks
    
    def quantify_elements(self, peaks, experimental_params=None):
        """
        Semi-quantitative relative intensities from fitted peak areas.

        This is NOT fundamental-parameters quantification. Peak areas for each
        labeled sample element are summed and normalized to 100% relative
        intensity. Tube lines are excluded. Use Calibration → Standards for
        instrument-calibrated / FP-style concentrations.

        Args:
            peaks: List of fitted Peak objects
            experimental_params: Unused; kept for API compatibility
            
        Returns:
            Dict keyed by element with relative_intensity_pct, total_area, lines,
            and method='semi_quant_area'. 'concentration' mirrors relative % for
            existing UI callers; 'error' is None (no uncertainty model).
        """
        _ = experimental_params  # reserved for future FP wiring
        element_totals = {}
        element_lines = {}

        for peak in peaks:
            if peak.is_tube_line:
                continue

            if peak.element:
                if peak.element not in element_totals:
                    element_totals[peak.element] = 0.0
                    element_lines[peak.element] = []

                element_totals[peak.element] += peak.area
                element_lines[peak.element].append(peak.line)

        total_intensity = sum(element_totals.values())

        if total_intensity == 0:
            return {}

        concentrations = {}
        for element, total_area in element_totals.items():
            relative_pct = (total_area / total_intensity) * 100.0

            concentrations[element] = {
                'concentration': relative_pct,  # UI compat: relative intensity %
                'relative_intensity_pct': relative_pct,
                'error': None,  # no uncertainty for area-normalized semi-quant
                'lines': element_lines[element],
                'total_area': total_area,
                'method': 'semi_quant_area',
            }

        return concentrations
