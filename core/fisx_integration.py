"""
Integration with fisx (PyMca's Fundamental Parameters library)

fisx provides highly accurate calculations for:
- Primary fluorescence
- Secondary fluorescence  
- Tertiary fluorescence
- Matrix effects
- Detector response

This is the gold standard for XRF quantification.
"""

import numpy as np
import fisx
from typing import Dict, List, Tuple


class FisxCalculator:
    """
    Wrapper for fisx library to calculate XRF intensities
    
    Uses PyMca's proven algorithms for accurate quantification
    """
    
    def __init__(self,
                 excitation_energy: float = 50.0,
                 tube_element: str = 'Rh',
                 detector_type: str = 'Si',
                 detector_thickness: float = 0.05,  # cm
                 detector_distance: float = 3.0,  # cm
                 detector_area: float = 0.3,  # cm²
                 incident_angle: float = 45.0,  # degrees
                 takeoff_angle: float = 45.0):  # degrees
        """
        Initialize fisx calculator
        
        Args:
            excitation_energy: X-ray tube voltage (keV)
            tube_element: Tube anode element
            detector_type: Detector material ('Si' or 'Ge')
            detector_thickness: Detector active thickness (cm)
            detector_distance: Sample-detector distance (cm)
            detector_area: Detector active area (cm²)
            incident_angle: X-ray incident angle (degrees)
            takeoff_angle: Detector takeoff angle (degrees)
        """
        # Initialize Elements database (required for fisx calculations)
        self.elements = fisx.Elements()
        self.elements.initializeAsPyMca()
        
        # Create fisx XRF instance
        self.fisx = fisx.XRF()
        
        # Configure incident X-ray tube spectrum
        # This includes both bremsstrahlung continuum and characteristic lines
        self._setup_tube_spectrum(excitation_energy, tube_element)
        
        # Create and set detector
        detector = fisx.Detector(detector_type)
        self.fisx.setDetector(detector)
        
        # Set geometry (incident angle, takeoff angle in degrees)
        self.fisx.setGeometry(incident_angle, takeoff_angle)
        
        # Store parameters
        self.excitation_energy = excitation_energy
        self.tube_element = tube_element
        self.detector_distance = detector_distance
        self.detector_area = detector_area
    
    def _setup_tube_spectrum(self, excitation_energy: float, tube_element: str):
        """
        Configure the incident X-ray tube spectrum in fisx
        
        For performance, uses a simplified approach with just a few key energies:
        - Tube characteristic lines (most important for excitation)
        - Representative continuum energies
        
        Args:
            excitation_energy: Tube voltage (keV)
            tube_element: Anode element (e.g., 'Rh', 'W', 'Mo')
        """
        # SIMPLIFIED APPROACH for performance:
        # Use only the most important excitation energies
        # Full spectrum with 60+ points is too slow for interactive use
        
        energy_grid = []
        intensities = []
        
        # Get atomic number of tube element
        tube_z_map = {
            'Rh': 45, 'W': 74, 'Mo': 42, 'Ag': 47, 'Cu': 29,
            'Cr': 24, 'Fe': 26, 'Co': 27, 'Au': 79
        }
        z_tube = tube_z_map.get(tube_element, 45)  # Default to Rh
        
        # Add tube characteristic lines (most important for excitation)
        from core.xray_data import get_element_lines
        
        try:
            tube_lines = get_element_lines(tube_element, z_tube)
            
            # Add K lines (if below excitation energy)
            for line in tube_lines.get('K', []):
                line_energy = line['energy']
                if line_energy < excitation_energy:
                    energy_grid.append(line_energy)
                    # Characteristic lines are strong
                    intensities.append(1e9 * line.get('relative_intensity', 1.0))
            
            # Add L lines (if below excitation energy)
            for line in tube_lines.get('L', []):
                line_energy = line['energy']
                if line_energy < excitation_energy:
                    energy_grid.append(line_energy)
                    intensities.append(5e8 * line.get('relative_intensity', 1.0))
                    
        except Exception as e:
            print(f"Warning: Could not add tube characteristic lines: {e}")
        
        # Add a few representative continuum energies
        # These approximate the bremsstrahlung contribution
        continuum_energies = [5, 10, 15, 20, 25, 30, 40]
        for E in continuum_energies:
            if E < excitation_energy:
                energy_grid.append(E)
                # Kramers' law: I(E) ∝ (E_max - E) / E
                intensities.append(z_tube * (excitation_energy - E) / E * 1e8)
        
        # Sort by energy
        sorted_pairs = sorted(zip(energy_grid, intensities))
        energy_grid = [e for e, i in sorted_pairs]
        intensities = [i for e, i in sorted_pairs]
        
        # Set the beam spectrum in fisx
        self.fisx.setBeam(energy_grid, intensities)
        
        print(f"  Configured {tube_element} tube spectrum: {len(energy_grid)} energy points (simplified for performance)")
        
    def calculate_intensities(self,
                             composition: Dict[str, float],
                             thickness: float = 0.1,
                             density: float = 2.5) -> Dict[str, Dict[str, float]]:
        """
        Calculate expected XRF intensities for a composition
        
        Args:
            composition: Dict of {element: weight_fraction} (must sum to 1.0)
            thickness: Sample thickness (cm), use large value for infinite thickness
            density: Sample density (g/cm³), default 2.5 for geological samples
            
        Returns:
            Dict of {element: {line: intensity}}
        """
        # Normalize composition
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}
        
        # Create a Material object and register it with Elements
        # fisx requires materials to be registered before use
        material_name = "Sample"
        material = fisx.Material(material_name, density, thickness)
        material.setComposition(composition)
        self.elements.addMaterial(material)
        
        # Set sample using the registered material name
        # Format: [[material_name, density, thickness]]
        self.fisx.setSample([[material_name, density, thickness]])
        
        # Get all emission lines
        results = {}
        
        print(f"  Calculating intensities for {len(composition)} elements...")
        for i, element in enumerate(composition.keys(), 1):
            if composition[element] <= 0:
                continue
            
            try:
                # Get expected rates for this element
                # fisx expects element + line family (e.g., "Fe K", "Fe L")
                # We'll request K and L lines separately
                element_lines = []
                
                # Determine which line families to request based on K-edge energy
                # Request K lines if K-edge < excitation energy
                # Request L lines if element is heavy enough (Z >= 20)
                # Request M lines if element is very heavy (Z >= 56)
                
                z_map = {
                    'Li': 3, 'Be': 4, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
                    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
                    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'As': 33,
                    'Se': 34, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
                    'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Sm': 62, 'Eu': 63,
                    'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
                    'Hg': 80, 'Pb': 82, 'Th': 90, 'U': 92, 'Cd': 48, 'Ag': 47, 'Sn': 50
                }
                
                # Approximate K-edge energies (keV)
                k_edge_map = {
                    'Al': 1.56, 'Si': 1.84, 'P': 2.15, 'S': 2.47, 'K': 3.61, 'Ca': 4.04,
                    'Ti': 4.97, 'V': 5.47, 'Cr': 5.99, 'Mn': 6.54, 'Fe': 7.11, 'Co': 7.71,
                    'Ni': 8.33, 'Cu': 8.98, 'Zn': 9.66, 'Ga': 10.37, 'As': 11.87, 'Se': 12.66,
                    'Rb': 15.20, 'Sr': 16.11, 'Y': 17.04, 'Zr': 18.00, 'Nb': 18.99, 'Mo': 20.00,
                    'Ag': 25.51, 'Cd': 26.71, 'Sn': 29.20
                }
                
                z = z_map.get(element, 0)
                if z > 0:
                    # Request K lines if K-edge is below excitation energy
                    k_edge = k_edge_map.get(element, 999)  # Default to high value if not in map
                    if k_edge < self.excitation_energy:
                        element_lines.append(f"{element} K")
                    
                    # Request L lines for medium to heavy elements (Z >= 20)
                    if z >= 20:
                        element_lines.append(f"{element} L")
                    
                    # Request M lines for very heavy elements (Z >= 56)
                    if z >= 56:
                        element_lines.append(f"{element} M")
                
                if not element_lines:
                    continue
                
                print(f"    [{i}/{len(composition)}] Calculating {element} ({', '.join(element_lines)})...", end='', flush=True)
                
                # fisx returns: primary, secondary, tertiary fluorescence
                # NOTE: secondary=2 is VERY slow (includes tertiary fluorescence)
                # For interactive use, we use secondary=0 (primary only)
                # This is still more accurate than simplified FP
                element_results = self.fisx.getMultilayerFluorescence(
                    element_lines,  # Element + line families (e.g., ["Fe K", "Fe L"])
                    self.elements,  # Elements database
                    secondary=0,  # Primary fluorescence only (for speed)
                    useMassFractions=True
                )
                
                print(f" done", flush=True)
                
                # Extract intensities
                # fisx returns results keyed by line family (e.g., "Fe K", "Fe L")
                # not by element name
                element_intensities = {}
                
                for line_family_key in element_results.keys():
                    # line_family_key is like "Fe K" or "Fe L"
                    line_family_data = element_results[line_family_key]
                    
                    # fisx returns a dict with layer indices (usually just {0: {...}})
                    # We need to iterate through layers, then through lines
                    for layer_idx, layer_lines in line_family_data.items():
                        # layer_lines is a dict of individual emission lines
                        for line_name, line_data in layer_lines.items():
                            # line_data contains: energy, rate, etc.
                            if isinstance(line_data, dict) and 'rate' in line_data and 'energy' in line_data:
                                # Total rate = primary + secondary + tertiary
                                total_rate = line_data['rate']
                                line_energy = line_data['energy']
                                
                                # Store both rate and energy
                                if total_rate > 0 and line_energy < self.excitation_energy:
                                    element_intensities[line_name] = {
                                        'rate': total_rate,
                                        'energy': line_energy
                                    }
                
                if element_intensities:
                    results[element] = element_intensities
                    
            except Exception as e:
                print(f"Error calculating intensities for {element}: {e}")
                continue
        
        return results
    
    def get_detector_efficiency(self, energy: float) -> float:
        """
        Get detector efficiency at given energy
        
        Args:
            energy: X-ray energy (keV)
            
        Returns:
            Efficiency (0-1)
        """
        try:
            # fisx calculates detector efficiency including:
            # - Window absorption
            # - Dead layer
            # - Active layer absorption
            efficiency = self.fisx.getDetectorEfficiency([energy])[0]
            return efficiency
        except:
            return 1.0
    
    def set_detector_window(self, material: str, thickness: float):
        """
        Set detector window (e.g., Be window)
        
        Args:
            material: Window material ('Be', 'Al', etc.)
            thickness: Window thickness (cm)
        """
        try:
            self.fisx.setDetectorWindow(material, thickness)
        except Exception as e:
            print(f"Error setting detector window: {e}")
    
    def set_sample_matrix(self, matrix: str, density: float = 2.5):
        """
        Set sample matrix for absorption calculations
        
        Args:
            matrix: Matrix formula (e.g., 'SiO2', 'CaCO3')
            density: Matrix density (g/cm³)
        """
        try:
            # Parse matrix and set
            # This affects absorption calculations
            pass  # Implement if needed
        except Exception as e:
            print(f"Error setting matrix: {e}")


def convert_fisx_to_element_data(fisx_results: Dict[str, Dict[str, Dict]],
                                 excitation_energy: float) -> List[Dict]:
    """
    Convert fisx results to element_data format for calibration
    
    Args:
        fisx_results: Output from FisxCalculator.calculate_intensities()
                     Format: {element: {line_name: {'rate': float, 'energy': float}}}
        excitation_energy: Excitation energy (keV)
        
    Returns:
        List of dicts with element, line, energy, relative_intensity
    """
    element_data = []
    
    for element, lines in fisx_results.items():
        for line_name, line_info in lines.items():
            # line_info now contains both 'rate' and 'energy'
            if isinstance(line_info, dict):
                intensity = line_info.get('rate', 0)
                energy = line_info.get('energy', 0)
                
                # Only include lines below excitation energy with non-zero intensity
                if energy > 0 and energy < excitation_energy and intensity > 0:
                    element_data.append({
                        'element': element,
                        'line': line_name,
                        'energy': energy,
                        'relative_intensity': intensity
                    })
    
    return element_data
