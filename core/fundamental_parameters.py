"""
Fundamental Parameters (FP) method for XRF quantification

Primary intensities use a polychromatic tube spectrum (Kramers continuum plus
anode characteristic lines) in the Sherman equation. A monochromatic fallback
remains for debugging. Relative FP inversion only needs intensity ratios.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import xraylib as xrl


TUBE_Z = {
    "Rh": 45,
    "W": 74,
    "Mo": 42,
    "Ag": 47,
    "Cu": 29,
    "Cr": 24,
    "Fe": 26,
    "Co": 27,
    "Au": 79,
}

_ANODE_DENSITY = {
    "Rh": 12.41,
    "W": 19.25,
    "Mo": 10.28,
    "Ag": 10.49,
    "Cu": 8.96,
    "Cr": 7.19,
    "Fe": 7.87,
    "Co": 8.86,
    "Au": 19.32,
}

# Characteristic K photons / continuum photons above the K-edge (or L-edge
# when K is not excited). Rh at 50 kV is characteristic-rich; this mix is what
# brings NiAs Kα ratios near measured μXRF values instead of a 50 keV line.
K_LINE_TO_CONTINUUM = 8.0
L_TO_K_WHEN_K_ON = 0.25
BE_WINDOW_CM = 0.005  # 50 µm
BE_DENSITY = 1.848


def _trapz(y, x) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size == 0:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def tube_atomic_number(symbol: str) -> int:
    if symbol in TUBE_Z:
        return TUBE_Z[symbol]
    try:
        return int(xrl.SymbolToAtomicNumber(symbol))
    except Exception:
        return 45


def _electron_range_cm(z: int, atomic_weight: float, density: float, kv: float) -> float:
    """Kanaya–Okayama electron range (cm) in the anode."""
    if density <= 0 or kv <= 0 or z <= 0:
        return 0.0
    r_um = 0.0276 * atomic_weight * (kv ** 1.67) / ((z ** 0.89) * density)
    return float(r_um) * 1e-4


def _continuum_grid(kv: float) -> np.ndarray:
    if kv <= 2.0:
        return np.array([max(kv * 0.5, 0.6)], dtype=float)
    low_top = min(15.0, kv)
    parts = [np.arange(1.0, low_top, 0.2)]
    if kv > 15.0:
        parts.append(np.arange(15.0, kv, 0.5))
    parts.append(np.array([max(kv - 0.05, 1.01)]))
    energy = np.unique(np.concatenate(parts))
    return energy[(energy > 0.5) & (energy < kv)]


def _be_transmission(energy: np.ndarray) -> np.ndarray:
    if BE_WINDOW_CM <= 0:
        return np.ones_like(energy, dtype=float)
    out = np.empty_like(energy, dtype=float)
    for i, e in enumerate(energy):
        try:
            mu = float(xrl.CS_Total(4, float(e)))
            out[i] = float(np.exp(-mu * BE_DENSITY * BE_WINDOW_CM))
        except Exception:
            out[i] = 1.0
    return np.clip(out, 0.0, 1.0)


def _anode_transmission(z: int, density: float, thickness_cm: float, energy) -> np.ndarray:
    e_arr = np.atleast_1d(np.asarray(energy, dtype=float))
    out = np.empty(e_arr.shape, dtype=float)
    if thickness_cm <= 0:
        return np.ones_like(out)
    for i, e in enumerate(e_arr.ravel()):
        try:
            mu = float(xrl.CS_Total(z, float(e)))
            out.ravel()[i] = float(np.exp(-mu * density * thickness_cm))
        except Exception:
            out.ravel()[i] = 1.0
    return np.clip(out, 0.0, 1.0)


def _char_lines(z: int, kv: float) -> List[Tuple[str, float, float]]:
    """(name, energy, radiative rate) for K then L lines below kV."""
    from core.xray_data import get_element_lines

    try:
        symbol = xrl.AtomicNumberToSymbol(z)
    except Exception:
        symbol = "Rh"
    try:
        lines = get_element_lines(symbol, z)
    except Exception:
        return []
    out: List[Tuple[str, float, float]] = []
    for series in ("K", "L"):
        for info in lines.get(series, []) or []:
            energy = float(info.get("energy") or 0.0)
            if energy <= 0.0 or energy >= kv:
                continue
            rate = float(info.get("intensity") or info.get("relative_intensity") or 0.0)
            out.append((str(info.get("name") or series), energy, max(rate, 0.0)))
    return out


def build_tube_spectrum(
    kv: float,
    tube_element: str = "Rh",
    *,
    char_to_continuum: float = K_LINE_TO_CONTINUUM,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """
    Kramers continuum + characteristic anode lines.

    Returns:
        energy_grid (keV), continuum flux density (photons/keV),
        characteristic lines as (energy_keV, photon_weight).
    """
    kv = float(kv)
    z = tube_atomic_number(tube_element)
    energy = _continuum_grid(kv)
    if energy.size == 0:
        return energy, energy, []

    continuum = z * (kv - energy) / np.maximum(energy, 1e-6)
    try:
        aw = float(xrl.AtomicWeight(z))
    except Exception:
        aw = 100.0
    density = float(_ANODE_DENSITY.get(tube_element, 12.0))
    depth = _electron_range_cm(z, aw, density, kv) / 3.0
    continuum = continuum * _anode_transmission(z, density, depth, energy)
    continuum = continuum * _be_transmission(energy)

    char_lines: List[Tuple[float, float]] = []
    try:
        k_edge = float(xrl.EdgeEnergy(z, xrl.K_SHELL))
    except Exception:
        k_edge = 1e9
    try:
        l_edge = float(xrl.EdgeEnergy(z, xrl.L3_SHELL))
    except Exception:
        l_edge = 1e9

    scale = max(float(char_to_continuum), 0.0)
    k_on = kv > k_edge + 0.2
    int_above_k = _trapz(continuum[energy > k_edge], energy[energy > k_edge]) if k_on else 0.0
    int_above_l = _trapz(continuum[energy > l_edge], energy[energy > l_edge]) if kv > l_edge else 0.0
    k_photons = scale * int_above_k if k_on else 0.0
    if k_photons > 0:
        l_photons = L_TO_K_WHEN_K_ON * k_photons
    else:
        l_photons = scale * int_above_l

    catalog = _char_lines(z, kv)
    k_items = [c for c in catalog if c[0].startswith("K")]
    l_items = [c for c in catalog if c[0].startswith("L")]

    def _split(items, total):
        if total <= 0 or not items:
            return
        weights = np.array([max(rate, 0.0) for _n, _e, rate in items], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(items))
        weights = weights / weights.sum()
        window = _be_transmission(np.array([e for _n, e, _r in items], dtype=float))
        anode = _anode_transmission(
            z, density, depth, np.array([e for _n, e, _r in items], dtype=float)
        )
        for (_name, e_line, _rate), w, tw, aw_t in zip(items, weights, window, anode):
            char_lines.append((float(e_line), float(total * w * tw * aw_t)))

    _split(k_items, k_photons)
    _split(l_items, l_photons)
    return energy, continuum, char_lines


class FundamentalParameters:
    """
    Fundamental Parameters calculator for XRF.

    Default excitation is a polychromatic tube spectrum (continuum + anode
    K/L lines). Intensity ratios follow the infinite-thickness Sherman equation.
    """

    def __init__(
        self,
        excitation_energy: float = 50.0,
        takeoff_angle: float = 45.0,
        incident_angle: float = 45.0,
        tube_element: str = "Rh",
        polychromatic: bool = True,
    ):
        self.excitation_energy = float(excitation_energy)
        self.takeoff_angle = np.radians(takeoff_angle)
        self.incident_angle = np.radians(incident_angle)
        self.tube_element = tube_element or "Rh"
        self.polychromatic = bool(polychromatic)
        self.geometric_factor = 1.0 / (
            np.sin(self.incident_angle) + np.sin(self.takeoff_angle)
        )
        self.tube_energy = np.array([], dtype=float)
        self.tube_continuum = np.array([], dtype=float)
        self.tube_lines: List[Tuple[float, float]] = []
        if self.polychromatic:
            self.tube_energy, self.tube_continuum, self.tube_lines = build_tube_spectrum(
                self.excitation_energy, self.tube_element
            )
        self._tau_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._mu_comp_key = None
        self._mu_in = None
        self._mu_out_cache: Dict[float, float] = {}

    def calculate_intensity(
        self,
        element: str,
        z: int,
        line: str,
        concentration: float,
        matrix_composition: Dict[str, float],
    ) -> float:
        """
        Expected X-ray intensity for one emission line.

        Args:
            element: Element symbol
            z: Atomic number
            line: Line name (e.g., 'Kα1', 'Lα1')
            concentration: Weight fraction (0-1)
            matrix_composition: {element: weight_fraction} for the sample
        """
        try:
            if (
                self.polychromatic
                and self.tube_energy.size > 1
                and (self.tube_continuum.size or self.tube_lines)
            ):
                return self._intensity_poly(
                    element, z, line, concentration, matrix_composition
                )
            return self._intensity_mono(
                element, z, line, concentration, matrix_composition
            )
        except Exception as e:
            print(f"Error calculating intensity for {element} {line}: {e}")
            return 0.0

    def _intensity_poly(
        self,
        element: str,
        z: int,
        line: str,
        concentration: float,
        matrix_composition: Dict[str, float],
    ) -> float:
        line_energy = self._get_line_energy(z, line)
        if line_energy is None or line_energy >= self.excitation_energy:
            return 0.0
        yield_line = self._get_fluorescence_yield(z, line)
        if yield_line <= 0:
            return 0.0
        shell = self._shell_for_line(line)
        if shell is None:
            return 0.0
        try:
            edge = float(xrl.EdgeEnergy(z, shell))
        except Exception:
            return 0.0
        if concentration <= 0:
            return 0.0

        mu_in = self._mu_on_tube_grid(matrix_composition)
        mu_out = self._mu_at(matrix_composition, line_energy)
        sin_in = max(float(np.sin(self.incident_angle)), 1e-6)
        sin_out = max(float(np.sin(self.takeoff_angle)), 1e-6)

        energy = self.tube_energy
        tau = self._photo_shell_on_grid(z, shell)
        absorb = 1.0 / (mu_in / sin_in + mu_out / sin_out)
        mask = energy > edge
        integral = 0.0
        if np.any(mask):
            integral += _trapz(
                self.tube_continuum[mask] * tau[mask] * absorb[mask],
                energy[mask],
            )
        for e_line, photons in self.tube_lines:
            if photons <= 0 or e_line <= edge:
                continue
            tau_e = self._photo_shell_at(z, shell, e_line)
            if tau_e <= 0:
                continue
            mu_e = self._mu_at(matrix_composition, e_line)
            a_e = 1.0 / (mu_e / sin_in + mu_out / sin_out)
            integral += photons * tau_e * a_e

        if integral <= 0:
            return 0.0
        det = self._detector_efficiency(line_energy)
        primary = (
            float(concentration)
            * yield_line
            * integral
            * self.geometric_factor
            * det
        )
        secondary = self._calculate_secondary_fluorescence(
            element, z, line, line_energy, concentration, matrix_composition
        )
        return float(primary * (1.0 + secondary))

    def _intensity_mono(
        self,
        element: str,
        z: int,
        line: str,
        concentration: float,
        matrix_composition: Dict[str, float],
    ) -> float:
        line_energy = self._get_line_energy(z, line)
        if line_energy is None or line_energy >= self.excitation_energy:
            return 0.0
        fluorescence_yield = self._get_fluorescence_yield(z, line)
        if fluorescence_yield == 0:
            return 0.0
        cross_section = self._get_cross_section(z, line)
        absorption_factor = self._calculate_absorption(
            line_energy, matrix_composition
        )
        detector_efficiency = self._detector_efficiency(line_energy)
        primary_intensity = (
            concentration
            * cross_section
            * fluorescence_yield
            * absorption_factor
            * self.geometric_factor
            * detector_efficiency
        )
        secondary_enhancement = self._calculate_secondary_fluorescence(
            element, z, line, line_energy, concentration, matrix_composition
        )
        return float(primary_intensity * (1.0 + secondary_enhancement))

    def _get_line_energy(self, z: int, line: str) -> Optional[float]:
        try:
            line_map = {
                "Kα1": xrl.KA1_LINE,
                "Kα2": xrl.KA2_LINE,
                "Kβ1": xrl.KB1_LINE,
                "Kβ2": xrl.KB2_LINE,
                "Kβ3": xrl.KB3_LINE,
                "Lα1": xrl.LA1_LINE,
                "Lα2": xrl.LA2_LINE,
                "Lβ1": xrl.LB1_LINE,
                "Lβ2": xrl.LB2_LINE,
                "Lγ1": xrl.LG1_LINE,
                "Mα1": xrl.MA1_LINE,
                "Mα2": xrl.MA2_LINE,
            }
            if line in line_map:
                return float(xrl.LineEnergy(z, line_map[line]))
            return None
        except Exception:
            return None

    def _shell_for_line(self, line: str):
        if line.startswith("K"):
            return xrl.K_SHELL
        if line.startswith("L"):
            return xrl.L3_SHELL
        if line.startswith("M"):
            return xrl.M5_SHELL
        return None

    def _get_fluorescence_yield(self, z: int, line: str) -> float:
        try:
            if line.startswith("K"):
                omega_k = xrl.FluorYield(z, xrl.K_SHELL)
                line_map = {
                    "Kα1": xrl.KA1_LINE,
                    "Kα2": xrl.KA2_LINE,
                    "Kβ1": xrl.KB1_LINE,
                    "Kβ2": xrl.KB2_LINE,
                    "Kβ3": xrl.KB3_LINE,
                }
                if line in line_map:
                    return float(omega_k * xrl.RadRate(z, line_map[line]))

            elif line.startswith("L"):
                try:
                    omega_l1 = xrl.FluorYield(z, xrl.L1_SHELL)
                    omega_l2 = xrl.FluorYield(z, xrl.L2_SHELL)
                    omega_l3 = xrl.FluorYield(z, xrl.L3_SHELL)
                    omega_l = (omega_l1 + omega_l2 + omega_l3) / 3.0
                except Exception:
                    omega_l = 0.1
                line_map = {
                    "Lα1": xrl.LA1_LINE,
                    "Lα2": xrl.LA2_LINE,
                    "Lβ1": xrl.LB1_LINE,
                    "Lβ2": xrl.LB2_LINE,
                    "Lγ1": xrl.LG1_LINE,
                }
                if line in line_map:
                    return float(omega_l * xrl.RadRate(z, line_map[line]))

            elif line.startswith("M"):
                return 0.05 * 0.5
            return 0.0
        except Exception:
            return 0.0

    def _photo_shell_at(self, z: int, shell: int, energy: float) -> float:
        try:
            edge = float(xrl.EdgeEnergy(z, shell))
        except Exception:
            return 0.0
        if energy <= edge:
            return 0.0
        try:
            return float(xrl.CS_Photo_Partial(z, shell, float(energy)))
        except Exception:
            try:
                cs = float(xrl.CS_Photo(z, float(energy)))
                jump = float(xrl.JumpFactor(z, shell))
                return cs * (jump - 1.0) / jump if jump > 1 else cs
            except Exception:
                return 0.0

    def _photo_shell_on_grid(self, z: int, shell: int) -> np.ndarray:
        key = (int(z), int(shell))
        cached = self._tau_cache.get(key)
        if cached is not None and cached.shape == self.tube_energy.shape:
            return cached
        tau = np.array(
            [self._photo_shell_at(z, shell, float(e)) for e in self.tube_energy],
            dtype=float,
        )
        self._tau_cache[key] = tau
        return tau

    def _mu_at(self, composition: Dict[str, float], energy: float) -> float:
        total = 0.0
        for elem, weight in composition.items():
            w = float(weight)
            if w <= 0:
                continue
            z = _element_z(elem)
            if z is None:
                continue
            try:
                total += w * float(xrl.CS_Total(z, float(energy)))
            except Exception:
                continue
        return total if total > 0 else 1e-12

    def _mu_on_tube_grid(self, composition: Dict[str, float]) -> np.ndarray:
        key = tuple(
            sorted(
                (str(k), round(float(v), 7))
                for k, v in composition.items()
                if float(v) > 0
            )
        )
        if key == self._mu_comp_key and self._mu_in is not None:
            return self._mu_in
        mu = np.array(
            [self._mu_at(composition, float(e)) for e in self.tube_energy],
            dtype=float,
        )
        self._mu_comp_key = key
        self._mu_in = mu
        self._mu_out_cache = {}
        return mu

    def _get_cross_section(self, z: int, line: str) -> float:
        """Photoionization cross-section at the tube kV (monochromatic fallback)."""
        try:
            shell = self._shell_for_line(line)
            if shell is None:
                return 0.0
            cross_section = xrl.CS_Photo(z, self.excitation_energy)
            try:
                jump_ratio = xrl.JumpFactor(z, shell)
                shell_fraction = (jump_ratio - 1.0) / jump_ratio
            except Exception:
                shell_fraction = 0.8
            return float(cross_section * shell_fraction)
        except Exception:
            return 1.0

    def _calculate_secondary_fluorescence(
        self,
        element: str,
        z: int,
        line: str,
        line_energy: float,
        concentration: float,
        matrix_composition: Dict[str, float],
    ) -> float:
        """Simplified secondary-fluorescence enhancement (typically 0–0.3)."""
        _ = line_energy, concentration
        try:
            if line.startswith("K"):
                edge_energy = xrl.EdgeEnergy(z, xrl.K_SHELL)
            elif line.startswith("L"):
                edge_energy = xrl.EdgeEnergy(z, xrl.L3_SHELL)
            else:
                return 0.0

            enhancement = 0.0
            for other_elem, other_conc in matrix_composition.items():
                if other_elem == element or other_conc < 0.001:
                    continue
                other_z = _element_z(other_elem)
                if other_z is None:
                    continue
                try:
                    other_ka_energy = xrl.LineEnergy(other_z, xrl.KA1_LINE)
                    if other_ka_energy > edge_energy:
                        energy_factor = min(
                            (other_ka_energy - edge_energy) / edge_energy, 1.0
                        )
                        enhancement += other_conc * energy_factor * 0.3
                except Exception:
                    continue
            return float(min(enhancement, 0.5))
        except Exception:
            return 0.0

    def _detector_efficiency(self, energy: float) -> float:
        """Simplified Si SDD efficiency (relative, 0–1)."""
        if energy < 1.0:
            return 0.3 + 0.7 * (energy / 1.0)
        if energy < 10.0:
            return 1.0
        if energy < 20.0:
            return 1.0 - 0.5 * ((energy - 10.0) / 10.0)
        return 0.5 * np.exp(-(energy - 20.0) / 10.0)

    def _calculate_absorption(
        self,
        line_energy: float,
        matrix_composition: Dict[str, float],
    ) -> float:
        """Monochromatic Sherman absorption factor."""
        try:
            mu_in = 0.0
            mu_out = 0.0
            for elem, weight_frac in matrix_composition.items():
                if weight_frac <= 0:
                    continue
                z_elem = _element_z(elem)
                if z_elem is None:
                    continue
                mu_in += weight_frac * xrl.CS_Total(z_elem, self.excitation_energy)
                mu_out += weight_frac * xrl.CS_Total(z_elem, line_energy)
            mu_total = mu_in / np.sin(self.incident_angle) + mu_out / np.sin(
                self.takeoff_angle
            )
            if mu_total > 0:
                absorption_factor = 1.0 / mu_total
            else:
                absorption_factor = 1.0
            return float(np.clip(absorption_factor, 0.01, 10.0))
        except Exception as e:
            print(f"Error calculating absorption: {e}")
            return 1.0

    def calculate_spectrum_intensities(
        self, composition: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Expected intensities for major lines in a composition."""
        from core.xray_data import get_element_lines

        total = sum(composition.values())
        if total > 0:
            composition = {k: v / total for k, v in composition.items()}

        results = {}
        for element, conc in composition.items():
            z = _element_z(element)
            if z is None or conc <= 0:
                continue
            lines_data = get_element_lines(element, z)
            element_intensities = {}
            for series in ["K", "L", "M"]:
                for line_info in lines_data.get(series, []):
                    line_name = line_info["name"]
                    if series == "K" and line_name not in ["Kα1", "Kα2", "Kβ1"]:
                        continue
                    if series == "L" and line_name not in ["Lα1", "Lα2", "Lβ1"]:
                        continue
                    if series == "M" and line_name not in ["Mα1"]:
                        continue
                    intensity = self.calculate_intensity(
                        element, z, line_name, conc, composition
                    )
                    if intensity > 0:
                        element_intensities[line_name] = intensity
            if element_intensities:
                results[element] = element_intensities
        return results


def _element_z(symbol: str) -> Optional[int]:
    try:
        return int(xrl.SymbolToAtomicNumber(symbol))
    except Exception:
        return None
