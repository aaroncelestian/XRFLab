"""
Background modeling for XRF spectra
"""

import numpy as np
from scipy import signal, ndimage
from scipy.interpolate import UnivariateSpline
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve


class BackgroundModeler:
    """Background estimation and removal for XRF spectra"""
    
    @staticmethod
    def snip_background(counts, iterations=20, decreasing=True):
        """
        SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) algorithm
        
        Args:
            counts: Array of spectrum counts
            iterations: Number of iterations (controls smoothness)
            decreasing: If True, use decreasing window (recommended)
            
        Returns:
            Array of background values
        """
        # Work with log-transformed data for better results
        spectrum = np.copy(counts).astype(float)
        spectrum[spectrum <= 0] = 1  # Avoid log(0)
        log_spectrum = np.log(np.log(np.sqrt(spectrum + 1) + 1) + 1)
        
        # Apply SNIP algorithm
        background = np.copy(log_spectrum)
        
        if decreasing:
            # Decreasing window size (more common)
            window_sizes = range(iterations, 0, -1)
        else:
            # Increasing window size
            window_sizes = range(1, iterations + 1)
        
        for window in window_sizes:
            for i in range(window, len(background) - window):
                # Compare point with average of neighbors at distance 'window'
                left = background[i - window]
                right = background[i + window]
                avg = (left + right) / 2.0
                
                # Keep minimum (peak clipping)
                background[i] = min(background[i], avg)
        
        # Transform back to linear scale
        background = (np.exp(np.exp(background) - 1) - 1) ** 2 - 1
        background[background < 0] = 0
        
        return background
    
    @staticmethod
    def polynomial_background(energy, counts, degree=3, roi_mask=None):
        """
        Polynomial background fitting
        
        Args:
            energy: Energy array
            counts: Counts array
            degree: Polynomial degree (1=linear, 2=quadratic, 3=cubic, etc.)
            roi_mask: Boolean mask of regions to exclude from fit (peaks)
            
        Returns:
            Array of background values
        """
        if roi_mask is None:
            # Use all points
            fit_energy = energy
            fit_counts = counts
        else:
            # Exclude masked regions
            fit_energy = energy[~roi_mask]
            fit_counts = counts[~roi_mask]
        
        # Fit polynomial
        coeffs = np.polyfit(fit_energy, fit_counts, degree)
        background = np.polyval(coeffs, energy)
        
        # Ensure non-negative
        background[background < 0] = 0
        
        return background
    
    @staticmethod
    def linear_background(energy, counts, start_idx=None, end_idx=None):
        """
        Simple linear background between two points
        
        Args:
            energy: Energy array
            counts: Counts array
            start_idx: Start index (default: first 5% average)
            end_idx: End index (default: last 5% average)
            
        Returns:
            Array of background values
        """
        n = len(counts)
        
        if start_idx is None:
            # Average first 5%
            start_idx = int(n * 0.05)
            start_value = np.mean(counts[:start_idx])
            start_energy = np.mean(energy[:start_idx])
        else:
            start_value = counts[start_idx]
            start_energy = energy[start_idx]
        
        if end_idx is None:
            # Average last 5%
            end_idx = int(n * 0.95)
            end_value = np.mean(counts[end_idx:])
            end_energy = np.mean(energy[end_idx:])
        else:
            end_value = counts[end_idx]
            end_energy = energy[end_idx]
        
        # Linear interpolation
        slope = (end_value - start_value) / (end_energy - start_energy)
        background = start_value + slope * (energy - start_energy)
        
        return background
    
    @staticmethod
    def adaptive_background(counts, window_size=50, percentile=5):
        """
        Adaptive background using moving percentile filter
        
        Args:
            counts: Counts array
            window_size: Size of moving window
            percentile: Percentile to use (lower = more aggressive)
            
        Returns:
            Array of background values
        """
        # Use moving percentile filter
        background = ndimage.percentile_filter(counts, percentile, size=window_size)
        
        # Smooth the result
        background = ndimage.gaussian_filter1d(background, sigma=window_size/4)
        
        return background
    
    @staticmethod
    def als_background(counts, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares (AsLS) baseline correction
        
        This algorithm fits a baseline by iteratively solving a weighted least squares 
        problem that penalizes asymmetry. It follows the lower envelope of the spectrum 
        (baseline) while avoiding peaks.
        
        The key insight is asymmetric weighting: points above the current baseline (peaks) 
        get low weight p, while points below get high weight (1-p). This forces the 
        baseline to hug the bottom of the spectrum.
        
        Reference:
        Eilers, P.H.C., Boelens, H.F.M. (2005). Baseline Correction with Asymmetric 
        Least Squares Smoothing. Leiden University Medical Centre Report.
        
        Args:
            counts: Counts array (spectrum intensities)
            lam: Smoothness parameter (10³ to 10⁷, default 10⁵)
                 Higher values = smoother baseline
            p: Asymmetry parameter (0.001 to 0.05, default 0.01)
               Lower values = baseline follows minimum more closely
               Points above baseline get weight p, below get (1-p)
            niter: Number of iterations (default 10, usually sufficient)
            
        Returns:
            Array of background values
            
        Example:
            >>> background = BackgroundModeler.als_background(counts, lam=1e6, p=0.001)
            >>> corrected = counts - background
        """
        y = np.asarray(counts, dtype=float)
        L = len(y)
        
        # Second-order difference matrix (penalizes roughness)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = csc_matrix(D)
        
        # Initialize weights
        w = np.ones(L)
        
        # Iterative refinement
        for i in range(niter):
            # Weighted matrix
            W = diags(w, 0, shape=(L, L))
            W = csc_matrix(W)
            
            # Solve: (W + λ * D * D^T) * z = W * y
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            
            # Update weights asymmetrically
            # Points above baseline (peaks) get low weight p
            # Points below baseline get high weight (1-p)
            w = p * (y > z) + (1 - p) * (y <= z)
        
        return z
    
    @staticmethod
    def estimate_background(energy, counts, method='snip', **kwargs):
        """
        Estimate background using specified method
        
        Args:
            energy: Energy array
            counts: Counts array
            method: 'snip', 'polynomial', 'linear', 'adaptive', 'als', or 'none'
            **kwargs: Method-specific parameters
            
        Returns:
            Array of background values
        """
        if method.lower() == 'snip':
            iterations = kwargs.get('iterations', 20)
            return BackgroundModeler.snip_background(counts, iterations=iterations)
        
        elif method.lower() == 'polynomial':
            degree = kwargs.get('degree', 3)
            roi_mask = kwargs.get('roi_mask', None)
            return BackgroundModeler.polynomial_background(
                energy, counts, degree=degree, roi_mask=roi_mask
            )
        
        elif method.lower() == 'linear':
            start_idx = kwargs.get('start_idx', None)
            end_idx = kwargs.get('end_idx', None)
            return BackgroundModeler.linear_background(
                energy, counts, start_idx=start_idx, end_idx=end_idx
            )
        
        elif method.lower() == 'adaptive':
            window_size = kwargs.get('window_size', 50)
            percentile = kwargs.get('percentile', 5)
            return BackgroundModeler.adaptive_background(
                counts, window_size=window_size, percentile=percentile
            )
        
        elif method.lower() == 'als' or method.lower() == 'asls':
            lam = kwargs.get('lam', 1e5)
            p = kwargs.get('p', 0.01)
            niter = kwargs.get('niter', 10)
            return BackgroundModeler.als_background(
                counts, lam=lam, p=p, niter=niter
            )
        
        elif method.lower() == 'none':
            return np.zeros_like(counts)
        
        else:
            raise ValueError(f"Unknown background method: {method}")
    
    @staticmethod
    def subtract_background(counts, background):
        """
        Subtract background from spectrum
        
        Args:
            counts: Original counts
            background: Background to subtract
            
        Returns:
            Background-subtracted counts (non-negative)
        """
        subtracted = counts - background
        subtracted[subtracted < 0] = 0
        return subtracted
