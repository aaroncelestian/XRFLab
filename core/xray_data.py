"""
X-ray emission line data using xraylib
"""

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
        dict: Dictionary with line series (K, L, M, N) and their energies
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
                    lines['K'].append({'name': name, 'energy': energy})
            except:
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
                    lines['L'].append({'name': name, 'energy': energy})
            except:
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
                    lines['M'].append({'name': name, 'energy': energy})
            except:
                pass
        
    except Exception as e:
        print(f"Error getting lines for {symbol}: {e}")
    
    return lines


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
        
        lines['K'].append({'name': 'Kα', 'energy': k_alpha})
        lines['K'].append({'name': 'Kβ', 'energy': k_beta})
    
    if z >= 21:  # Sc and above have measurable L lines
        l_alpha = 10.2 * (z - 7.4)**2 / 1000 * 0.15  # Rough approximation
        l_beta = 10.2 * (z - 7.2)**2 / 1000 * 0.15
        
        lines['L'].append({'name': 'Lα', 'energy': l_alpha})
        lines['L'].append({'name': 'Lβ', 'energy': l_beta})
    
    return lines
