"""
Spectrum data class for XRF analysis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Spectrum:
    """
    Container for XRF spectrum data
    
    Attributes:
        energy: Energy axis in keV
        counts: Intensity/counts at each energy
        live_time: Acquisition live time in seconds
        real_time: Acquisition real time in seconds
        metadata: Additional metadata dictionary
    """
    energy: np.ndarray
    counts: np.ndarray
    live_time: float = 100.0
    real_time: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate spectrum data after initialization"""
        if len(self.energy) != len(self.counts):
            raise ValueError("Energy and counts arrays must have same length")
        
        if len(self.energy) == 0:
            raise ValueError("Spectrum cannot be empty")
        
        # Ensure arrays are numpy arrays
        self.energy = np.asarray(self.energy, dtype=np.float64)
        self.counts = np.asarray(self.counts, dtype=np.float64)
    
    @property
    def num_channels(self):
        """Return number of channels in spectrum"""
        return len(self.energy)
    
    @property
    def energy_range(self):
        """Return (min, max) energy range"""
        return (self.energy[0], self.energy[-1])
    
    @property
    def total_counts(self):
        """Return total counts in spectrum"""
        return np.sum(self.counts)
    
    @property
    def max_counts(self):
        """Return maximum counts value"""
        return np.max(self.counts)
    
    def get_energy_calibration(self):
        """
        Get energy calibration parameters assuming linear calibration
        
        Returns:
            tuple: (offset, gain) where E = offset + gain * channel
        """
        if self.num_channels < 2:
            return (0.0, 1.0)
        
        # Assume uniform spacing
        gain = (self.energy[-1] - self.energy[0]) / (self.num_channels - 1)
        offset = self.energy[0]
        
        return (offset, gain)
    
    def set_energy_calibration(self, offset, gain):
        """
        Set energy calibration parameters
        
        Args:
            offset: Energy offset in keV
            gain: Energy gain in keV/channel
        """
        channels = np.arange(self.num_channels)
        self.energy = offset + gain * channels
    
    def get_roi(self, energy_min, energy_max):
        """
        Get region of interest
        
        Args:
            energy_min: Minimum energy in keV
            energy_max: Maximum energy in keV
            
        Returns:
            tuple: (energy_roi, counts_roi) arrays for the ROI
        """
        mask = (self.energy >= energy_min) & (self.energy <= energy_max)
        return self.energy[mask], self.counts[mask]
    
    def get_roi_sum(self, energy_min, energy_max):
        """
        Get sum of counts in region of interest
        
        Args:
            energy_min: Minimum energy in keV
            energy_max: Maximum energy in keV
            
        Returns:
            float: Sum of counts in ROI
        """
        _, counts_roi = self.get_roi(energy_min, energy_max)
        return np.sum(counts_roi)
    
    def normalize(self, method='live_time'):
        """
        Normalize spectrum
        
        Args:
            method: Normalization method ('live_time', 'total_counts', 'max')
            
        Returns:
            Spectrum: New normalized spectrum
        """
        if method == 'live_time':
            norm_factor = self.live_time
        elif method == 'total_counts':
            norm_factor = self.total_counts
        elif method == 'max':
            norm_factor = self.max_counts
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if norm_factor == 0:
            raise ValueError("Cannot normalize by zero")
        
        normalized_counts = self.counts / norm_factor
        
        return Spectrum(
            energy=self.energy.copy(),
            counts=normalized_counts,
            live_time=self.live_time,
            real_time=self.real_time,
            metadata=self.metadata.copy()
        )
    
    def rebin(self, factor):
        """
        Rebin spectrum by averaging adjacent channels
        
        Args:
            factor: Rebinning factor (must divide evenly into num_channels)
            
        Returns:
            Spectrum: New rebinned spectrum
        """
        if self.num_channels % factor != 0:
            raise ValueError(f"Rebinning factor {factor} does not divide evenly into {self.num_channels} channels")
        
        new_size = self.num_channels // factor
        
        # Reshape and average
        energy_rebinned = self.energy[:new_size * factor].reshape(new_size, factor).mean(axis=1)
        counts_rebinned = self.counts[:new_size * factor].reshape(new_size, factor).sum(axis=1)
        
        return Spectrum(
            energy=energy_rebinned,
            counts=counts_rebinned,
            live_time=self.live_time,
            real_time=self.real_time,
            metadata=self.metadata.copy()
        )
    
    def smooth(self, window_size=5):
        """
        Smooth spectrum using moving average
        
        Args:
            window_size: Size of smoothing window (must be odd)
            
        Returns:
            Spectrum: New smoothed spectrum
        """
        if window_size % 2 == 0:
            window_size += 1
        
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        counts_smoothed = np.convolve(self.counts, kernel, mode='same')
        
        return Spectrum(
            energy=self.energy.copy(),
            counts=counts_smoothed,
            live_time=self.live_time,
            real_time=self.real_time,
            metadata=self.metadata.copy()
        )
    
    def copy(self):
        """Create a deep copy of the spectrum"""
        return Spectrum(
            energy=self.energy.copy(),
            counts=self.counts.copy(),
            live_time=self.live_time,
            real_time=self.real_time,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self):
        """Convert spectrum to dictionary for serialization"""
        return {
            'energy': self.energy.tolist(),
            'counts': self.counts.tolist(),
            'live_time': self.live_time,
            'real_time': self.real_time,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create spectrum from dictionary"""
        return cls(
            energy=np.array(data['energy']),
            counts=np.array(data['counts']),
            live_time=data.get('live_time', 100.0),
            real_time=data.get('real_time', 100.0),
            metadata=data.get('metadata', {})
        )
    
    def __repr__(self):
        return (f"Spectrum(channels={self.num_channels}, "
                f"energy_range={self.energy_range}, "
                f"total_counts={self.total_counts:.0f})")
