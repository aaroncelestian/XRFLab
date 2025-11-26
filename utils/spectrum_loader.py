"""
Utility functions for loading XRF spectrum files
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict


def load_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load XRF spectrum from EMSA/MAS format file
    
    Args:
        filepath: Path to spectrum file (.txt)
    
    Returns:
        Tuple of (energy, counts) arrays
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Spectrum file not found: {filepath}")
    
    # Read file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find where spectrum data starts
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('#SPECTRUM'):
            data_start = i + 1
            break
    
    if data_start is None:
        raise ValueError("Could not find #SPECTRUM marker in file")
    
    # Parse spectrum data
    energies = []
    counts = []
    
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        try:
            parts = line.split(',')
            if len(parts) >= 2:
                energy = float(parts[0].strip())
                count = float(parts[1].strip())
                energies.append(energy)
                counts.append(count)
        except ValueError:
            continue
    
    if len(energies) == 0:
        raise ValueError("No spectrum data found in file")
    
    return np.array(energies), np.array(counts)


def load_spectrum_with_metadata(filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load XRF spectrum with metadata from EMSA/MAS format file
    
    Args:
        filepath: Path to spectrum file (.txt)
    
    Returns:
        Tuple of (energy, counts, metadata_dict)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Spectrum file not found: {filepath}")
    
    # Read file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse metadata
    metadata = {}
    data_start = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line.startswith('#SPECTRUM'):
            data_start = i + 1
            break
        
        if line.startswith('#'):
            # Parse metadata line
            parts = line[1:].split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                # Try to convert to appropriate type
                try:
                    # Try float first
                    if '.' in value or 'E' in value.upper():
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                
                metadata[key] = value
    
    if data_start is None:
        raise ValueError("Could not find #SPECTRUM marker in file")
    
    # Parse spectrum data
    energies = []
    counts = []
    
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        try:
            parts = line.split(',')
            if len(parts) >= 2:
                energy = float(parts[0].strip())
                count = float(parts[1].strip())
                energies.append(energy)
                counts.append(count)
        except ValueError:
            continue
    
    if len(energies) == 0:
        raise ValueError("No spectrum data found in file")
    
    return np.array(energies), np.array(counts), metadata
