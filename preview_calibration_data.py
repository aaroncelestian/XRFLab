#!/usr/bin/env python3
"""
Preview XRF calibration data

Quick visualization of all standard spectra to verify data quality
before running full calibration.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.spectrum_loader import load_spectrum
from core.background import BackgroundModeler


def preview_all_standards():
    """Preview all standard spectra"""
    
    data_dir = Path("sample_data/data")
    standards = ['Fe', 'Cu', 'Ti', 'Zn', 'Mg', 'cubic zirconia']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    bg_modeler = BackgroundModeler()
    
    for idx, standard in enumerate(standards):
        ax = axes[idx]
        filepath = data_dir / f"{standard}.txt"
        
        try:
            # Load spectrum
            energy, counts = load_spectrum(str(filepath))
            
            # Estimate background
            background = bg_modeler.estimate_background(
                energy, counts, method='snip', window_length=50
            )
            counts_bg_sub = counts - background
            
            # Plot
            ax.plot(energy, counts, 'b-', linewidth=0.5, alpha=0.7, label='Raw')
            ax.plot(energy, background, 'r-', linewidth=1, label='Background')
            ax.plot(energy, counts_bg_sub, 'g-', linewidth=0.5, alpha=0.8, label='BG-subtracted')
            
            ax.set_xlabel('Energy (keV)', fontsize=10)
            ax.set_ylabel('Counts', fontsize=10)
            ax.set_title(f'{standard}', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 20)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
            
            # Add peak annotations for major lines
            peak_annotations = {
                'Fe': [(1.49, 'Al Kα'), (6.40, 'Fe Kα'), (7.06, 'Fe Kβ')],
                'Cu': [(1.49, 'Al Kα'), (8.05, 'Cu Kα'), (8.91, 'Cu Kβ')],
                'Ti': [(1.49, 'Al Kα'), (4.51, 'Ti Kα'), (4.93, 'Ti Kβ')],
                'Zn': [(1.49, 'Al Kα'), (8.64, 'Zn Kα'), (9.57, 'Zn Kβ')],
                'Mg': [(1.25, 'Mg Kα'), (1.49, 'Al Kα')],
                'cubic zirconia': [(2.04, 'Zr Lα'), (15.75, 'Zr Kα')]
            }
            
            if standard in peak_annotations:
                for peak_e, peak_label in peak_annotations[standard]:
                    if peak_e < 20:  # Only annotate if in visible range
                        ax.axvline(peak_e, color='orange', linestyle='--', 
                                  linewidth=0.8, alpha=0.5)
                        # Find peak height for annotation
                        mask = np.abs(energy - peak_e) < 0.1
                        if np.any(mask):
                            peak_height = np.max(counts_bg_sub[mask])
                            ax.text(peak_e, peak_height * 1.05, peak_label,
                                   rotation=90, fontsize=7, alpha=0.7,
                                   verticalalignment='bottom')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading:\n{e}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color='red')
            ax.set_title(f'{standard} (ERROR)', fontsize=12, fontweight='bold', color='red')
    
    plt.suptitle('XRF Calibration Standards - Data Preview', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = Path("sample_data/calibration_data_preview.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Preview saved to: {output_path}")
    
    plt.show()


def print_data_summary():
    """Print summary statistics for all standards"""
    
    data_dir = Path("sample_data/data")
    standards = ['Fe', 'Cu', 'Ti', 'Zn', 'Mg', 'cubic zirconia']
    
    print("\n" + "=" * 70)
    print("XRF Calibration Data Summary")
    print("=" * 70)
    
    for standard in standards:
        filepath = data_dir / f"{standard}.txt"
        
        try:
            energy, counts = load_spectrum(str(filepath))
            
            print(f"\n{standard}:")
            print(f"  File: {filepath.name}")
            print(f"  Channels: {len(energy)}")
            print(f"  Energy range: {energy[0]:.3f} - {energy[-1]:.3f} keV")
            print(f"  Energy step: {np.mean(np.diff(energy)):.5f} keV/channel")
            print(f"  Total counts: {np.sum(counts):.0f}")
            print(f"  Max counts: {np.max(counts):.0f}")
            print(f"  Peak count rate: {np.max(counts):.0f} counts/channel")
            
        except Exception as e:
            print(f"\n{standard}: ❌ Error - {e}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("XRF Calibration Data Preview")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Load all standard spectra")
    print("  2. Estimate and subtract backgrounds")
    print("  3. Display all spectra with peak annotations")
    print("  4. Print data quality summary")
    print()
    
    # Print summary
    print_data_summary()
    
    # Show plots
    print("\nGenerating preview plots...")
    preview_all_standards()
    
    print("\n✓ Preview complete!")
    print("\nNext step: Run full calibration with:")
    print("  python run_peak_shape_calibration.py")
