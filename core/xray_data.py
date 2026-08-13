"""
X-ray emission line data using xraylib
"""

import math

try:
    import xraylib as xrl
    XRAYLIB_AVAILABLE = True
except ImportError:
    XRAYLIB_AVAILABLE = False
    print("Warning: xraylib not available. Using fallback data.")


def get_element_lines(symbol, z):
    """
    Get X-ray emission lines for an element
    
    Args:
        symbol: Element symbol
        z: Atomic number
        
    Returns:
        dict: Dictionary with line series (K, L, M, N). Each entry is
        {'name', 'energy', 'intensity', 'relative_intensity'} where
        intensity is the xraylib radiative rate (or fallback approx) and
        relative_intensity is normalized to the strongest line in that series (0–1).
    """
    if not XRAYLIB_AVAILABLE:
        return _get_fallback_lines(symbol, z)
    
    lines = {
        'K': [],
        'L': [],
        'M': [],
        'N': []
    }
    
    try:
        # K lines
        k_lines = [
            ('Kα1', xrl.KA1_LINE),
            ('Kα2', xrl.KA2_LINE),
            ('Kβ1', xrl.KB1_LINE),
            ('Kβ2', xrl.KB2_LINE),
            ('Kβ3', xrl.KB3_LINE),
        ]
        
        for name, line_code in k_lines:
            try:
                energy = xrl.LineEnergy(z, line_code)
                if energy > 0:
                    intensity = _line_radiative_rate(z, line_code)
                    lines['K'].append({
                        'name': name,
                        'energy': energy,
                        'intensity': intensity,
                    })
            except Exception:
                pass
        
        # L lines
        l_lines = [
            ('Lα1', xrl.LA1_LINE),
            ('Lα2', xrl.LA2_LINE),
            ('Lβ1', xrl.LB1_LINE),
            ('Lβ2', xrl.LB2_LINE),
            ('Lβ3', xrl.LB3_LINE),
            ('Lβ4', xrl.LB4_LINE),
            ('Lγ1', xrl.LG1_LINE),
            ('Lγ2', xrl.LG2_LINE),
            ('Lγ3', xrl.LG3_LINE),
        ]
        
        for name, line_code in l_lines:
            try:
                energy = xrl.LineEnergy(z, line_code)
                if energy > 0:
                    intensity = _line_radiative_rate(z, line_code)
                    lines['L'].append({
                        'name': name,
                        'energy': energy,
                        'intensity': intensity,
                    })
            except Exception:
                pass
        
        # M lines
        m_lines = [
            ('Mα1', xrl.MA1_LINE),
            ('Mα2', xrl.MA2_LINE),
            ('Mβ', xrl.MB_LINE),
            ('Mγ', xrl.MG_LINE),
        ]
        
        for name, line_code in m_lines:
            try:
                energy = xrl.LineEnergy(z, line_code)
                if energy > 0:
                    intensity = _line_radiative_rate(z, line_code)
                    lines['M'].append({
                        'name': name,
                        'energy': energy,
                        'intensity': intensity,
                    })
            except Exception:
                pass
        
    except Exception as e:
        print(f"Error getting lines for {symbol}: {e}")

    _normalize_series_intensities(lines)
    return lines


def _line_radiative_rate(z, line_code):
    """Return xraylib radiative rate, or 0 if unavailable."""
    try:
        rate = float(xrl.RadRate(z, line_code))
        return rate if rate > 0 else 0.0
    except Exception:
        return 0.0


def _normalize_series_intensities(lines):
    """
    Attach relative_intensity (0–1) within each series.

    If all radiative rates are missing, fall back to approximate branching ratios.
    """
    approx = {
        'Kα1': 1.00, 'Kα': 1.00, 'Kα2': 0.50,
        'Kβ1': 0.17, 'Kβ': 0.17, 'Kβ2': 0.05, 'Kβ3': 0.09,
        'Lα1': 1.00, 'Lα': 1.00, 'Lα2': 0.11,
        'Lβ1': 0.60, 'Lβ': 0.60, 'Lβ2': 0.25, 'Lβ3': 0.10, 'Lβ4': 0.08,
        'Lγ1': 0.12, 'Lγ2': 0.04, 'Lγ3': 0.03,
        'Mα1': 1.00, 'Mα': 1.00, 'Mα2': 0.50, 'Mβ': 0.60, 'Mγ': 0.15,
    }

    for series, entries in lines.items():
        if not entries:
            continue
        for entry in entries:
            if entry.get('intensity', 0) <= 0:
                entry['intensity'] = float(approx.get(entry.get('name'), 0.1))
        max_i = max(float(e.get('intensity', 0.0)) for e in entries) or 1.0
        for entry in entries:
            entry['relative_intensity'] = float(entry.get('intensity', 0.0)) / max_i


def get_tube_lines(tube_element='Rh', excitation_kv=50.0):
    """
    Get X-ray tube characteristic lines
    
    Args:
        tube_element: Tube anode element (e.g., 'Rh', 'W', 'Mo', 'Ag')
        excitation_kv: Tube voltage in keV
        
    Returns:
        dict: Dictionary with line series and their energies
    """
    # Map tube elements to atomic numbers
    tube_z_map = {
        'Rh': 45,  # Rhodium
        'W': 74,   # Tungsten
        'Mo': 42,  # Molybdenum
        'Ag': 47,  # Silver
        'Cr': 24,  # Chromium
        'Cu': 29,  # Copper
    }
    
    if tube_element not in tube_z_map:
        return {'K': [], 'L': [], 'M': []}
    
    z = tube_z_map[tube_element]
    
    # Get all emission lines for tube element
    lines = get_element_lines(tube_element, z)
    
    # Filter lines below excitation voltage
    filtered_lines = {'K': [], 'L': [], 'M': []}
    for series in ['K', 'L', 'M']:
        for line in lines.get(series, []):
            if line['energy'] < excitation_kv:
                filtered_lines[series].append(line)
    
    return filtered_lines


def compton_energy(incident_energy_kev, scatter_angle_deg=90.0):
    """
    Compton-scattered photon energy (keV).

    E' = E / (1 + (E/511) * (1 - cos θ))
    """
    e0 = float(incident_energy_kev)
    cos_theta = math.cos(math.radians(float(scatter_angle_deg)))
    return e0 / (1.0 + (e0 / 511.0) * (1.0 - cos_theta))


def get_tube_compton_lines(
    tube_element='Rh',
    excitation_kv=50.0,
    scatter_angle_deg=90.0,
    fwhm_kev=0.250,
):
    """
    Inelastic (Compton) tube scatter lines for analysis fitting.

    These are much broader than elastic fluorescence / Rayleigh scatter
    because of scattering-angle spread and Doppler broadening.

    Returns:
        List of dicts: energy, element, line, is_tube_line, fixed_fwhm,
        exclusion_half_width_kev
    """
    tube_lines = get_tube_lines(tube_element, excitation_kv)
    results = []
    fwhm = float(fwhm_kev)
    # Exclude auto-find under the broad Compton hump (~±1.5 FWHM)
    half_width = max(0.30, 1.5 * fwhm)

    # Major K lines dominate the Compton continuum feature (~19 keV for Rh).
    # Use one Compton Kα (from Kα1) + Compton Kβ — avoid stacking Kα1/Kα2.
    preferred = {
        'Kα1': 'Compton Kα',
        'Kβ1': 'Compton Kβ',
    }
    for line in tube_lines.get('K', []):
        name = line['name']
        if name not in preferred:
            continue
        e_in = float(line['energy'])
        if e_in >= float(excitation_kv):
            continue
        e_c = compton_energy(e_in, scatter_angle_deg)
        results.append({
            'energy': e_c,
            'element': tube_element,
            'line': preferred[name],
            'is_tube_line': True,
            'fixed_fwhm': fwhm,
            'exclusion_half_width_kev': half_width,
            'parent_energy': e_in,
        })

    return results


def get_element_info(symbol, z):
    """
    Get detailed information about an element
    
    Args:
        symbol: Element symbol
        z: Atomic number
        
    Returns:
        dict: Element information including atomic weight, density, etc.
    """
    info = {
        'symbol': symbol,
        'z': z,
        'name': _get_element_name(z),
        'atomic_weight': 0.0,
        'density': 0.0,
    }
    
    if XRAYLIB_AVAILABLE:
        try:
            info['atomic_weight'] = xrl.AtomicWeight(z)
        except:
            pass
        
        try:
            info['density'] = xrl.ElementDensity(z)
        except:
            pass
    
    return info


def _get_element_name(z):
    """Get element name from atomic number"""
    names = {
        1: 'Hydrogen', 2: 'Helium', 3: 'Lithium', 4: 'Beryllium', 5: 'Boron',
        6: 'Carbon', 7: 'Nitrogen', 8: 'Oxygen', 9: 'Fluorine', 10: 'Neon',
        11: 'Sodium', 12: 'Magnesium', 13: 'Aluminum', 14: 'Silicon', 15: 'Phosphorus',
        16: 'Sulfur', 17: 'Chlorine', 18: 'Argon', 19: 'Potassium', 20: 'Calcium',
        21: 'Scandium', 22: 'Titanium', 23: 'Vanadium', 24: 'Chromium', 25: 'Manganese',
        26: 'Iron', 27: 'Cobalt', 28: 'Nickel', 29: 'Copper', 30: 'Zinc',
        31: 'Gallium', 32: 'Germanium', 33: 'Arsenic', 34: 'Selenium', 35: 'Bromine',
        36: 'Krypton', 37: 'Rubidium', 38: 'Strontium', 39: 'Yttrium', 40: 'Zirconium',
        41: 'Niobium', 42: 'Molybdenum', 43: 'Technetium', 44: 'Ruthenium', 45: 'Rhodium',
        46: 'Palladium', 47: 'Silver', 48: 'Cadmium', 49: 'Indium', 50: 'Tin',
        51: 'Antimony', 52: 'Tellurium', 53: 'Iodine', 54: 'Xenon', 55: 'Cesium',
        56: 'Barium', 57: 'Lanthanum', 58: 'Cerium', 59: 'Praseodymium', 60: 'Neodymium',
        61: 'Promethium', 62: 'Samarium', 63: 'Europium', 64: 'Gadolinium', 65: 'Terbium',
        66: 'Dysprosium', 67: 'Holmium', 68: 'Erbium', 69: 'Thulium', 70: 'Ytterbium',
        71: 'Lutetium', 72: 'Hafnium', 73: 'Tantalum', 74: 'Tungsten', 75: 'Rhenium',
        76: 'Osmium', 77: 'Iridium', 78: 'Platinum', 79: 'Gold', 80: 'Mercury',
        81: 'Thallium', 82: 'Lead', 83: 'Bismuth', 84: 'Polonium', 85: 'Astatine',
        86: 'Radon', 87: 'Francium', 88: 'Radium', 89: 'Actinium', 90: 'Thorium',
        91: 'Protactinium', 92: 'Uranium', 93: 'Neptunium', 94: 'Plutonium', 95: 'Americium',
        96: 'Curium', 97: 'Berkelium', 98: 'Californium',
    }
    return names.get(z, f'Element {z}')


def _get_fallback_lines(symbol, z):
    """
    Fallback emission line data when xraylib is not available
    Approximate K and L alpha/beta lines
    """
    # Simplified Moseley's law approximation: E ≈ 10.2 * (Z - σ)^2 eV for K-alpha
    # This is very approximate but better than nothing
    
    lines = {'K': [], 'L': [], 'M': [], 'N': []}
    
    if z >= 11:  # Na and above have measurable K lines
        k_alpha = 10.2 * (z - 1.5)**2 / 1000  # Convert to keV
        k_beta = 10.2 * (z - 1.3)**2 / 1000
        
        lines['K'].append({'name': 'Kα1', 'energy': k_alpha, 'intensity': 1.0})
        lines['K'].append({'name': 'Kα2', 'energy': k_alpha * 0.998, 'intensity': 0.5})
        lines['K'].append({'name': 'Kβ1', 'energy': k_beta, 'intensity': 0.17})
    
    if z >= 21:  # Sc and above have measurable L lines
        l_alpha = 10.2 * (z - 7.4)**2 / 1000 * 0.15  # Rough approximation
        l_beta = 10.2 * (z - 7.2)**2 / 1000 * 0.15
        
        lines['L'].append({'name': 'Lα1', 'energy': l_alpha, 'intensity': 1.0})
        lines['L'].append({'name': 'Lβ1', 'energy': l_beta, 'intensity': 0.6})

    _normalize_series_intensities(lines)
    return lines
