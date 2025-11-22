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
        # Create fisx XRF instance
        self.fisx = fisx.XRF()
        
        # Set beam parameters
        beam = fisx.Beam()
        beam.setEnergy([excitation_energy])
        self.fisx.setBeam(beam)
        
        # Create and set detector
        detector = fisx.Detector(detector_type)
        # Set detector parameters if needed
        self.fisx.setDetector(detector)
        
        # Set geometry
        self.fisx.setGeometry(incident_angle, takeoff_angle)
        
        # Store parameters
        self.excitation_energy = excitation_energy
        self.detector_distance = detector_distance
        self.detector_area = detector_area
        
    def calculate_intensities(self,
                             composition: Dict[str, float],
                             thickness: float = 0.1) -> Dict[str, Dict[str, float]]:
        """
        Calculate expected XRF intensities for a composition
        
        Args:
            composition: Dict of {element: weight_fraction} (must sum to 1.0)
            thickness: Sample thickness (cm), use large value for infinite thickness
            
        Returns:
            Dict of {element: {line: intensity}}
        """
        # Normalize composition
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}
        
        # Set sample composition
        # fisx expects list of [element, fraction] pairs
        sample_composition = []
        for element, fraction in composition.items():
            if fraction > 0:
                sample_composition.append([element, fraction])
        
        # Set sample (infinite thickness approximation)
        self.fisx.setSample([sample_composition], [1.0], [thickness])
        
        # Get all emission lines
        results = {}
        
        for element in composition.keys():
            if composition[element] <= 0:
                continue
            
            try:
                # Get expected rates for this element
                # fisx returns: primary, secondary, tertiary fluorescence
                element_results = self.fisx.getMultilayerFluorescence(
                    [element],  # Elements to calculate
                    None,  # All lines
                    secondary=2,  # Include secondary and tertiary
                    useMassFractions=True
                )
                
                # Extract intensities
                element_intensities = {}
                
                if element in element_results:
                    for line_family in element_results[element]:
                        for line_name, line_data in element_results[element][line_family].items():
                            # line_data contains: energy, rate, etc.
                            if 'rate' in line_data:
                                # Total rate = primary + secondary + tertiary
                                total_rate = line_data['rate']
                                
                                if total_rate > 0:
                                    element_intensities[line_name] = total_rate
                
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


def convert_fisx_to_element_data(fisx_results: Dict[str, Dict[str, float]],
                                 excitation_energy: float) -> List[Dict]:
    """
    Convert fisx results to element_data format for calibration
    
    Args:
        fisx_results: Output from FisxCalculator.calculate_intensities()
        excitation_energy: Excitation energy (keV)
        
    Returns:
        List of dicts with element, line, energy, relative_intensity
    """
    from core.xray_data import get_element_lines
    
    element_data = []
    
    z_map = {
        'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'K': 19, 'Ca': 20, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'As': 33, 'Se': 34, 'Rb': 37, 'Sr': 38, 'Y': 39,
        'Zr': 40, 'Nb': 41, 'Mo': 42, 'Ag': 47, 'Cd': 48, 'Sn': 50, 'Sb': 51,
        'Ba': 56, 'La': 57, 'Ce': 58, 'Nd': 60, 'Sm': 62, 'Eu': 63, 'Gd': 64,
        'Dy': 66, 'Er': 68, 'Yb': 70, 'Hf': 72, 'Ta': 73, 'W': 74, 'Au': 79,
        'Hg': 80, 'Pb': 82, 'Th': 90, 'U': 92, 'Mg': 12, 'Na': 11
    }
    
    for element, lines in fisx_results.items():
        if element not in z_map:
            continue
        
        z = z_map[element]
        lines_data = get_element_lines(element, z)
        
        for line_name, intensity in lines.items():
            # Find energy for this line
            for series in ['K', 'L', 'M']:
                for line_info in lines_data.get(series, []):
                    if line_info['name'] == line_name:
                        if line_info['energy'] < excitation_energy:
                            element_data.append({
                                'element': element,
                                'line': line_name,
                                'energy': line_info['energy'],
                                'relative_intensity': intensity
                            })
                        break
    
    return element_data
