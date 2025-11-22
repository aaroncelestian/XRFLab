"""
File I/O handler for various XRF spectrum formats
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from core.spectrum import Spectrum


class IOHandler:
    """Handler for loading and saving XRF spectrum files"""
    
    def load_spectrum(self, file_path: str) -> Spectrum:
        """
        Load spectrum from file
        
        Args:
            file_path: Path to spectrum file
            
        Returns:
            Spectrum object
            
        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix in ['.txt', '.dat']:
            return self._load_text_spectrum(file_path)
        elif suffix == '.csv':
            return self._load_csv_spectrum(file_path)
        elif suffix == '.mca':
            return self._load_mca_spectrum(file_path)
        elif suffix in ['.h5', '.hdf5']:
            return self._load_hdf5_spectrum(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_text_spectrum(self, file_path: Path) -> Spectrum:
        """Load spectrum from text file (two-column: energy, counts or EMSA format)"""
        try:
            # Check if it's EMSA format
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if first_line.startswith('#FORMAT') and 'EMSA' in first_line:
                    return self._load_emsa_spectrum(file_path)
            
            # Try to load as two-column data
            data = np.loadtxt(file_path)
            
            if data.ndim == 1:
                # Single column - assume channel numbers, create energy axis
                counts = data
                channels = np.arange(len(counts))
                energy = channels * 0.01  # Default 10 eV/channel
            elif data.shape[1] == 2:
                # Two columns - energy and counts
                energy = data[:, 0]
                counts = data[:, 1]
            else:
                raise ValueError("Text file must have 1 or 2 columns")
            
            return Spectrum(
                energy=energy,
                counts=counts,
                metadata={'file_path': str(file_path)}
            )
        except Exception as e:
            raise ValueError(f"Failed to load text spectrum: {str(e)}")
    
    def _load_csv_spectrum(self, file_path: Path) -> Spectrum:
        """Load spectrum from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Try to find energy and counts columns
            energy_col = None
            counts_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'energy' in col_lower or 'kev' in col_lower:
                    energy_col = col
                elif 'count' in col_lower or 'intensity' in col_lower:
                    counts_col = col
            
            if energy_col is None or counts_col is None:
                # Assume first two columns are energy and counts
                energy = df.iloc[:, 0].values
                counts = df.iloc[:, 1].values
            else:
                energy = df[energy_col].values
                counts = df[counts_col].values
            
            return Spectrum(
                energy=energy,
                counts=counts,
                metadata={'file_path': str(file_path)}
            )
        except Exception as e:
            raise ValueError(f"Failed to load CSV spectrum: {str(e)}")
    
    def _load_emsa_spectrum(self, file_path: Path) -> Spectrum:
        """Load spectrum from EMSA/MAS format file"""
        try:
            metadata = {}
            energy_data = []
            counts_data = []
            
            # Parse EMSA header and data
            with open(file_path, 'r') as f:
                in_data_section = False
                xperchan = 0.01  # Default
                offset = 0.0
                
                for line in f:
                    line = line.strip()
                    
                    # Parse header
                    if line.startswith('#'):
                        if ':' in line:
                            key, value = line[1:].split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            metadata[key] = value
                            
                            # Extract key parameters
                            if key == 'XPERCHAN':
                                xperchan = float(value)
                            elif key == 'OFFSET':
                                offset = float(value)
                            elif key == 'LIVETIME':
                                metadata['live_time'] = float(value)
                            elif key == 'REALTIME':
                                metadata['real_time'] = float(value)
                            elif key == 'BEAMKV':
                                metadata['excitation_energy'] = float(value)
                            elif key == 'PROBECUR':
                                metadata['tube_current'] = float(value)
                            elif key == 'ELEVANGLE':
                                metadata['takeoff_angle'] = float(value)
                            elif key == 'XTILTSTGE':
                                metadata['incident_angle'] = float(value) if float(value) != 0 else 45.0
                            elif key == 'AZIMANGLE':
                                metadata['azimuth_angle'] = float(value)
                            elif key == 'MAGCAM':
                                metadata['magnification'] = float(value)
                            elif key == 'XPOSITION mm':
                                metadata['x_position'] = float(value)
                            elif key == 'YPOSITION mm':
                                metadata['y_position'] = float(value)
                            elif key == 'ZPOSITION mm':
                                metadata['z_position'] = float(value)
                        
                        if 'SPECTRUM' in line or 'Spectral Data Starts Here' in line:
                            in_data_section = True
                        continue
                    
                    # Parse data section
                    if in_data_section and line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            try:
                                energy_val = float(parts[0].strip())
                                counts_val = float(parts[1].strip())
                                energy_data.append(energy_val)
                                counts_data.append(counts_val)
                            except ValueError:
                                continue
            
            # Convert to numpy arrays
            energy = np.array(energy_data)
            counts = np.array(counts_data)
            
            # If energy is not explicitly in file, calculate from channels
            if len(energy) == 0 or np.allclose(energy, 0):
                channels = np.arange(len(counts))
                energy = offset + channels * xperchan
            
            return Spectrum(
                energy=energy,
                counts=counts,
                live_time=float(metadata.get('live_time', 100.0)),
                real_time=float(metadata.get('real_time', 100.0)),
                metadata=metadata
            )
        except Exception as e:
            raise ValueError(f"Failed to load EMSA spectrum: {str(e)}")
    
    def _load_mca_spectrum(self, file_path: Path) -> Spectrum:
        """Load spectrum from MCA file format"""
        try:
            # MCA format parser (simplified)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse header for metadata
            metadata = {}
            data_start = 0
            live_time = 100.0
            real_time = 100.0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('LIVE_TIME'):
                    live_time = float(line.split('-')[1].strip())
                elif line.startswith('REAL_TIME'):
                    real_time = float(line.split('-')[1].strip())
                elif line.startswith('<<DATA>>'):
                    data_start = i + 1
                    break
            
            # Read counts data
            counts = []
            for line in lines[data_start:]:
                line = line.strip()
                if line.startswith('<<') or not line:
                    break
                try:
                    counts.append(float(line))
                except ValueError:
                    continue
            
            counts = np.array(counts)
            
            # Create energy axis (default calibration)
            channels = np.arange(len(counts))
            energy = channels * 0.01  # Default 10 eV/channel
            
            return Spectrum(
                energy=energy,
                counts=counts,
                live_time=live_time,
                real_time=real_time,
                metadata={'file_path': str(file_path)}
            )
        except Exception as e:
            raise ValueError(f"Failed to load MCA spectrum: {str(e)}")
    
    def _load_hdf5_spectrum(self, file_path: Path) -> Spectrum:
        """Load spectrum from HDF5 file"""
        try:
            import h5py
            
            with h5py.File(file_path, 'r') as f:
                # Try common dataset names
                energy = None
                counts = None
                
                for key in ['energy', 'Energy', 'ENERGY']:
                    if key in f:
                        energy = f[key][:]
                        break
                
                for key in ['counts', 'Counts', 'COUNTS', 'intensity', 'Intensity']:
                    if key in f:
                        counts = f[key][:]
                        break
                
                if energy is None or counts is None:
                    # Try first two datasets
                    keys = list(f.keys())
                    if len(keys) >= 2:
                        energy = f[keys[0]][:]
                        counts = f[keys[1]][:]
                    else:
                        raise ValueError("Could not find energy and counts datasets")
                
                # Load metadata
                metadata = {'file_path': str(file_path)}
                if 'metadata' in f:
                    for key in f['metadata'].attrs:
                        metadata[key] = f['metadata'].attrs[key]
                
                live_time = metadata.get('live_time', 100.0)
                real_time = metadata.get('real_time', 100.0)
            
            return Spectrum(
                energy=energy,
                counts=counts,
                live_time=live_time,
                real_time=real_time,
                metadata=metadata
            )
        except ImportError:
            raise ValueError("h5py not installed. Install with: pip install h5py")
        except Exception as e:
            raise ValueError(f"Failed to load HDF5 spectrum: {str(e)}")
    
    def save_spectrum(self, spectrum: Spectrum, file_path: str, format: str = 'auto'):
        """
        Save spectrum to file
        
        Args:
            spectrum: Spectrum object to save
            file_path: Output file path
            format: Output format ('txt', 'csv', 'hdf5', or 'auto' to detect from extension)
        """
        file_path = Path(file_path)
        
        if format == 'auto':
            suffix = file_path.suffix.lower()
            if suffix == '.csv':
                format = 'csv'
            elif suffix in ['.h5', '.hdf5']:
                format = 'hdf5'
            else:
                format = 'txt'
        
        if format == 'txt':
            self._save_text_spectrum(spectrum, file_path)
        elif format == 'csv':
            self._save_csv_spectrum(spectrum, file_path)
        elif format == 'hdf5':
            self._save_hdf5_spectrum(spectrum, file_path)
        else:
            raise ValueError(f"Unsupported save format: {format}")
    
    def _save_text_spectrum(self, spectrum: Spectrum, file_path: Path):
        """Save spectrum as text file"""
        data = np.column_stack([spectrum.energy, spectrum.counts])
        np.savetxt(file_path, data, fmt='%.6f', header='Energy(keV)\tCounts')
    
    def _save_csv_spectrum(self, spectrum: Spectrum, file_path: Path):
        """Save spectrum as CSV file"""
        df = pd.DataFrame({
            'Energy (keV)': spectrum.energy,
            'Counts': spectrum.counts
        })
        df.to_csv(file_path, index=False)
    
    def _save_hdf5_spectrum(self, spectrum: Spectrum, file_path: Path):
        """Save spectrum as HDF5 file"""
        try:
            import h5py
            
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('energy', data=spectrum.energy)
                f.create_dataset('counts', data=spectrum.counts)
                
                # Save metadata
                metadata_group = f.create_group('metadata')
                metadata_group.attrs['live_time'] = spectrum.live_time
                metadata_group.attrs['real_time'] = spectrum.real_time
                
                for key, value in spectrum.metadata.items():
                    if isinstance(value, (int, float, str, bool)):
                        metadata_group.attrs[key] = value
        except ImportError:
            raise ValueError("h5py not installed. Install with: pip install h5py")
    
    def export_results(self, results: list, file_path: str):
        """
        Export quantification results to file
        
        Args:
            results: List of result dictionaries
            file_path: Output file path
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        df = pd.DataFrame(results)
        
        if suffix == '.xlsx':
            df.to_excel(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)
