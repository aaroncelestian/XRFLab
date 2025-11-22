"""
Fundamental Parameters (FP) method for XRF quantification

This module implements the physics-based approach to calculate X-ray intensities
from element concentrations, accounting for:
- X-ray production (excitation cross-sections)
- Fluorescence yields
- Absorption (matrix effects)
- Detector efficiency
- Geometric factors
"""

import numpy as np
import xraylib as xrl
from typing import Dict, List, Tuple


class FundamentalParameters:
    """
    Fundamental Parameters calculator for XRF
    
    Implements the Sherman equation and related physics to convert
    element concentrations to expected X-ray intensities.
    """
    
    def __init__(self, 
                 excitation_energy: float = 50.0,
                 takeoff_angle: float = 45.0,
                 incident_angle: float = 45.0):
        """
        Initialize FP calculator
        
        Args:
            excitation_energy: X-ray tube voltage (keV)
            takeoff_angle: Detector takeoff angle (degrees)
            incident_angle: X-ray incident angle (degrees)
        """
        self.excitation_energy = excitation_energy
        self.takeoff_angle = np.radians(takeoff_angle)
        self.incident_angle = np.radians(incident_angle)
        
        # Geometric factor
        self.geometric_factor = 1.0 / (np.sin(self.incident_angle) + np.sin(self.takeoff_angle))
    
    def calculate_intensity(self,
                           element: str,
                           z: int,
                           line: str,
                           concentration: float,
                           matrix_composition: Dict[str, float]) -> float:
        """
        Calculate expected X-ray intensity for an element line
        
        Args:
            element: Element symbol
            z: Atomic number
            line: Line name (e.g., 'Kα1', 'Lα1')
            concentration: Element concentration (weight fraction, 0-1)
            matrix_composition: Dict of {element: weight_fraction} for entire sample
            
        Returns:
            Relative intensity (arbitrary units)
        """
        try:
            # Get line energy
            line_energy = self._get_line_energy(z, line)
            if line_energy is None or line_energy >= self.excitation_energy:
                return 0.0
            
            # Get fluorescence yield
            fluorescence_yield = self._get_fluorescence_yield(z, line)
            if fluorescence_yield == 0:
                return 0.0
            
            # Get photoionization cross-section
            cross_section = self._get_cross_section(z, line)
            
            # Calculate absorption correction (matrix effect)
            absorption_factor = self._calculate_absorption(
                line_energy, matrix_composition
            )
            
            # Get detector efficiency
            detector_efficiency = self._detector_efficiency(line_energy)
            
            # Calculate primary intensity
            primary_intensity = (concentration * 
                               cross_section * 
                               fluorescence_yield * 
                               absorption_factor * 
                               self.geometric_factor *
                               detector_efficiency)
            
            # Add secondary fluorescence enhancement (simplified)
            secondary_enhancement = self._calculate_secondary_fluorescence(
                element, z, line, line_energy, concentration, matrix_composition
            )
            
            # Total intensity = primary + secondary
            intensity = primary_intensity * (1.0 + secondary_enhancement)
            
            return intensity
            
        except Exception as e:
            print(f"Error calculating intensity for {element} {line}: {e}")
            return 0.0
    
    def _get_line_energy(self, z: int, line: str) -> float:
        """Get X-ray line energy"""
        try:
            line_map = {
                'Kα1': xrl.KA1_LINE,
                'Kα2': xrl.KA2_LINE,
                'Kβ1': xrl.KB1_LINE,
                'Kβ2': xrl.KB2_LINE,
                'Kβ3': xrl.KB3_LINE,
                'Lα1': xrl.LA1_LINE,
                'Lα2': xrl.LA2_LINE,
                'Lβ1': xrl.LB1_LINE,
                'Lβ2': xrl.LB2_LINE,
                'Lγ1': xrl.LG1_LINE,
                'Mα1': xrl.MA1_LINE,
                'Mα2': xrl.MA2_LINE,
            }
            
            if line in line_map:
                return xrl.LineEnergy(z, line_map[line])
            return None
        except:
            return None
    
    def _get_fluorescence_yield(self, z: int, line: str) -> float:
        """Get fluorescence yield for a line"""
        try:
            # Determine shell
            if line.startswith('K'):
                # K-shell fluorescence yield
                omega_k = xrl.FluorYield(z, xrl.K_SHELL)
                
                # Get relative intensity of this line within K series
                line_map = {
                    'Kα1': xrl.KA1_LINE,
                    'Kα2': xrl.KA2_LINE,
                    'Kβ1': xrl.KB1_LINE,
                    'Kβ2': xrl.KB2_LINE,
                    'Kβ3': xrl.KB3_LINE,
                }
                
                if line in line_map:
                    # Radiative rate for this line
                    rad_rate = xrl.RadRate(z, line_map[line])
                    return omega_k * rad_rate
                    
            elif line.startswith('L'):
                # L-shell fluorescence yield (average of L subshells)
                try:
                    omega_l1 = xrl.FluorYield(z, xrl.L1_SHELL)
                    omega_l2 = xrl.FluorYield(z, xrl.L2_SHELL)
                    omega_l3 = xrl.FluorYield(z, xrl.L3_SHELL)
                    omega_l = (omega_l1 + omega_l2 + omega_l3) / 3.0
                except:
                    omega_l = 0.1  # Fallback
                
                line_map = {
                    'Lα1': xrl.LA1_LINE,
                    'Lα2': xrl.LA2_LINE,
                    'Lβ1': xrl.LB1_LINE,
                    'Lβ2': xrl.LB2_LINE,
                    'Lγ1': xrl.LG1_LINE,
                }
                
                if line in line_map:
                    rad_rate = xrl.RadRate(z, line_map[line])
                    return omega_l * rad_rate
            
            elif line.startswith('M'):
                # M-shell (simplified)
                omega_m = 0.05  # Typical M-shell yield
                return omega_m * 0.5  # Approximate
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def _get_cross_section(self, z: int, line: str) -> float:
        """Get photoionization cross-section"""
        try:
            # Determine shell
            if line.startswith('K'):
                shell = xrl.K_SHELL
            elif line.startswith('L'):
                shell = xrl.L3_SHELL  # Use L3 as representative
            elif line.startswith('M'):
                shell = xrl.M5_SHELL  # Use M5 as representative
            else:
                return 0.0
            
            # Photoionization cross-section at excitation energy
            cross_section = xrl.CS_Photo(z, self.excitation_energy)
            
            # Get jump ratio to account for shell-specific excitation
            try:
                jump_ratio = xrl.JumpFactor(z, shell)
                # Fraction of photoionization in this shell
                shell_fraction = (jump_ratio - 1.0) / jump_ratio
            except:
                shell_fraction = 0.8  # Typical value
            
            return cross_section * shell_fraction
            
        except Exception as e:
            return 1.0  # Fallback
    
    def _calculate_secondary_fluorescence(self,
                                         element: str,
                                         z: int,
                                         line: str,
                                         line_energy: float,
                                         concentration: float,
                                         matrix_composition: Dict[str, float]) -> float:
        """
        Calculate secondary fluorescence enhancement (simplified)
        
        Secondary fluorescence occurs when X-rays from other elements
        in the sample excite the element of interest.
        
        Args:
            element: Element symbol
            z: Atomic number
            line: Line name
            line_energy: Energy of the line (keV)
            concentration: Element concentration
            matrix_composition: Full sample composition
            
        Returns:
            Enhancement factor (0-1, typically 0-0.3)
        """
        try:
            enhancement = 0.0
            
            # Get excitation edge energy for this element
            if line.startswith('K'):
                try:
                    edge_energy = xrl.EdgeEnergy(z, xrl.K_SHELL)
                except:
                    return 0.0
            elif line.startswith('L'):
                try:
                    edge_energy = xrl.EdgeEnergy(z, xrl.L3_SHELL)
                except:
                    return 0.0
            else:
                return 0.0
            
            z_map = {
                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
                'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
                'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
                'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                'Zn': 30, 'Ga': 31, 'As': 33, 'Pb': 82, 'Ba': 56
            }
            
            # Check each other element in the matrix
            for other_elem, other_conc in matrix_composition.items():
                if other_elem == element or other_elem not in z_map:
                    continue
                
                if other_conc < 0.001:  # Skip trace amounts
                    continue
                
                other_z = z_map[other_elem]
                
                # Get characteristic line energies of the other element
                # Check if they can excite our element
                try:
                    # Check K-alpha of other element
                    other_ka_energy = xrl.LineEnergy(other_z, xrl.KA1_LINE)
                    
                    # Can this line excite our element?
                    if other_ka_energy > edge_energy:
                        # Calculate enhancement (simplified)
                        # Enhancement depends on:
                        # - Concentration of exciting element
                        # - Fluorescence yield of exciting element
                        # - Absorption of exciting radiation
                        
                        # Simplified: proportional to concentration and energy difference
                        energy_factor = (other_ka_energy - edge_energy) / edge_energy
                        energy_factor = min(energy_factor, 1.0)  # Cap at 1
                        
                        # Enhancement is typically 5-30% for major elements
                        element_enhancement = other_conc * energy_factor * 0.3
                        enhancement += element_enhancement
                        
                except:
                    pass
            
            # Cap total enhancement at 50%
            enhancement = min(enhancement, 0.5)
            
            return enhancement
            
        except Exception as e:
            return 0.0
    
    def _detector_efficiency(self, energy: float) -> float:
        """
        Simplified detector efficiency model
        
        Typical Si detector efficiency:
        - Low at very low energies (absorption in Be window, dead layer)
        - High in mid-range (1-10 keV)
        - Decreasing at high energies (X-rays pass through)
        
        Args:
            energy: X-ray energy (keV)
            
        Returns:
            Relative efficiency (0-1)
        """
        # Simplified model based on typical Si(Li) or SDD detector
        if energy < 1.0:
            # Low energy: absorbed by window
            return 0.3 + 0.7 * (energy / 1.0)
        elif energy < 10.0:
            # Mid range: high efficiency
            return 1.0
        elif energy < 20.0:
            # High energy: decreasing
            return 1.0 - 0.5 * ((energy - 10.0) / 10.0)
        else:
            # Very high energy: low efficiency
            return 0.5 * np.exp(-(energy - 20.0) / 10.0)
    
    def _calculate_absorption(self,
                              line_energy: float,
                              matrix_composition: Dict[str, float]) -> float:
        """
        Calculate absorption correction factor
        
        Args:
            line_energy: Energy of emitted X-ray (keV)
            matrix_composition: Dict of {element: weight_fraction}
            
        Returns:
            Absorption correction factor (0-1)
        """
        try:
            # Calculate mass attenuation coefficients
            # μ_in = attenuation at incident energy
            # μ_out = attenuation at emitted line energy
            
            mu_in = 0.0
            mu_out = 0.0
            
            z_map = {
                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
                'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
                'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
                'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
                'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
                'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
                'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
                'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
                'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
                'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
                'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92
            }
            
            for elem, weight_frac in matrix_composition.items():
                if elem not in z_map or weight_frac <= 0:
                    continue
                
                z_elem = z_map[elem]
                
                # Mass attenuation coefficient (cm²/g)
                mu_in += weight_frac * xrl.CS_Total(z_elem, self.excitation_energy)
                mu_out += weight_frac * xrl.CS_Total(z_elem, line_energy)
            
            # Absorption factor using simplified geometry
            # A = 1 / (μ_in/sin(θ_in) + μ_out/sin(θ_out))
            # For thin samples or infinite thickness approximation
            
            mu_total = mu_in / np.sin(self.incident_angle) + mu_out / np.sin(self.takeoff_angle)
            
            if mu_total > 0:
                # Infinite thickness approximation
                absorption_factor = 1.0 / mu_total
            else:
                absorption_factor = 1.0
            
            # Normalize to reasonable range
            absorption_factor = np.clip(absorption_factor, 0.01, 10.0)
            
            return absorption_factor
            
        except Exception as e:
            print(f"Error calculating absorption: {e}")
            return 1.0  # No correction if error
    
    def calculate_spectrum_intensities(self,
                                       composition: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate expected intensities for all major lines in a composition
        
        Args:
            composition: Dict of {element: weight_fraction} (must sum to 1.0)
            
        Returns:
            Dict of {element: {line: intensity}}
        """
        from core.xray_data import get_element_lines
        
        # Normalize composition
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}
        
        z_map = {
            'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'K': 19, 'Ca': 20, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'As': 33, 'Se': 34, 'Rb': 37, 'Sr': 38, 'Y': 39,
            'Zr': 40, 'Nb': 41, 'Mo': 42, 'Ag': 47, 'Cd': 48, 'Sn': 50, 'Sb': 51,
            'Ba': 56, 'La': 57, 'Ce': 58, 'Nd': 60, 'Sm': 62, 'Eu': 63, 'Gd': 64,
            'Dy': 66, 'Er': 68, 'Yb': 70, 'Hf': 72, 'Ta': 73, 'W': 74, 'Au': 79,
            'Hg': 80, 'Pb': 82, 'Th': 90, 'U': 92
        }
        
        results = {}
        
        for element, conc in composition.items():
            if element not in z_map or conc <= 0:
                continue
            
            z = z_map[element]
            lines_data = get_element_lines(element, z)
            
            element_intensities = {}
            
            # Calculate intensities for major lines
            for series in ['K', 'L', 'M']:
                for line_info in lines_data.get(series, []):
                    line_name = line_info['name']
                    
                    # Only major lines
                    if series == 'K' and line_name not in ['Kα1', 'Kα2', 'Kβ1']:
                        continue
                    if series == 'L' and line_name not in ['Lα1', 'Lα2', 'Lβ1']:
                        continue
                    if series == 'M' and line_name not in ['Mα1']:
                        continue
                    
                    intensity = self.calculate_intensity(
                        element, z, line_name, conc, composition
                    )
                    
                    if intensity > 0:
                        element_intensities[line_name] = intensity
            
            if element_intensities:
                results[element] = element_intensities
        
        return results
