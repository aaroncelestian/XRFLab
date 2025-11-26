#!/usr/bin/env python3
"""
Complete Calibration Workflow for XRFLab

This script demonstrates the recommended two-step calibration process:
1. FWHM calibration using pure element standards
2. Instrument calibration using reference standards with pre-calibrated FWHM

Usage:
    python calibration_workflow.py --step fwhm     # Step 1: FWHM calibration
    python calibration_workflow.py --step instrument  # Step 2: Instrument calibration
    python calibration_workflow.py --step both     # Both steps
"""

import argparse
from pathlib import Path
import numpy as np

def step1_fwhm_calibration():
    """
    Step 1: FWHM Calibration using pure element standards
    
    Uses Fe, Cu, Ti, Zn, Mg, cubic zirconia to calibrate detector resolution
    """
    print("=" * 70)
    print("STEP 1: FWHM CALIBRATION")
    print("=" * 70)
    print()
    print("Calibrating detector resolution using pure element standards...")
    print()
    
    from calibrate_peak_shape import PeakShapeCalibrator
    from core.fwhm_calibration import convert_peak_shape_calibration
    
    # Set paths
    data_dir = Path("sample_data/data")
    output_dir = Path("calibrations")
    output_dir.mkdir(exist_ok=True)
    
    # Create calibrator
    calibrator = PeakShapeCalibrator(data_dir)
    
    # Process all files
    calibrator.process_all_files()
    
    if len(calibrator.measurements) < 3:
        print("\n❌ Not enough measurements for calibration!")
        print("   Check that data files exist in sample_data/data/")
        return None
    
    # Fit detector model (standard physics-based model)
    print("\nFitting detector resolution model...")
    print("-" * 70)
    
    try:
        results = calibrator.fit_resolution_model(
            remove_outliers=True,
            model='detector'
        )
        
        print(f"\n✓ FWHM Calibration successful!")
        print(f"  FWHM₀ = {results['fwhm_0']*1000:.1f} ± {results['fwhm_0_err']*1000:.1f} eV")
        print(f"  ε = {results['epsilon']*1000:.2f} ± {results['epsilon_err']*1000:.2f} eV/keV")
        print(f"  R² = {results['r_squared']:.4f}")
        print(f"  RMSE = {results['rmse']*1000:.1f} eV")
        print(f"  Peaks used: {len(calibrator.measurements)}")
        
        # Convert to FWHMCalibration object
        from core.fwhm_calibration import FWHMCalibration
        from datetime import datetime
        
        fwhm_cal = FWHMCalibration(
            model_type='detector',
            parameters={
                'fwhm_0': results['fwhm_0'],
                'epsilon': results['epsilon']
            },
            parameter_errors={
                'fwhm_0': results['fwhm_0_err'],
                'epsilon': results['epsilon_err']
            },
            r_squared=results['r_squared'],
            rmse=results['rmse'],
            aic=results['aic'],
            bic=results['bic'],
            n_peaks=len(calibrator.measurements),
            energy_range=(
                min(m.energy for m in calibrator.measurements),
                max(m.energy for m in calibrator.measurements)
            ),
            calibration_date=datetime.now().isoformat()
        )
        
        # Save
        output_path = output_dir / "fwhm_calibration.json"
        fwhm_cal.save(str(output_path))
        print(f"\n✓ FWHM calibration saved to: {output_path}")
        
        # Generate plot
        calibrator.plot_calibration(results, output_dir / "fwhm_calibration.png")
        
        return fwhm_cal
        
    except Exception as e:
        print(f"\n❌ FWHM calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def step2_instrument_calibration(fwhm_calibration_path=None):
    """
    Step 2: Instrument Calibration using reference standards
    
    Uses FWHM calibration from Step 1 to constrain peak shapes
    """
    print("=" * 70)
    print("STEP 2: INSTRUMENT CALIBRATION")
    print("=" * 70)
    print()
    
    from core.calibration import InstrumentCalibrator
    from core.fwhm_calibration import load_fwhm_calibration
    from utils.spectrum_loader import load_spectrum
    
    # Load FWHM calibration if available
    fwhm_cal = None
    if fwhm_calibration_path and Path(fwhm_calibration_path).exists():
        print(f"Loading FWHM calibration from: {fwhm_calibration_path}")
        fwhm_cal = load_fwhm_calibration(fwhm_calibration_path)
        print(f"  FWHM₀ = {fwhm_cal.parameters['fwhm_0']*1000:.1f} eV")
        print(f"  ε = {fwhm_cal.parameters['epsilon']*1000:.2f} eV/keV")
        print(f"  R² = {fwhm_cal.r_squared:.4f}")
        print()
    else:
        print("⚠ No FWHM calibration provided - will optimize FWHM parameters")
        print()
    
    # Example: Load a reference standard spectrum
    # YOU NEED TO REPLACE THIS with your actual reference standard
    print("NOTE: This is a placeholder - you need to provide:")
    print("  1. Reference standard spectrum (energy, counts)")
    print("  2. Known concentrations for the reference")
    print("  3. Excitation energy (tube voltage)")
    print()
    print("Example usage:")
    print("```python")
    print("from utils.spectrum_loader import load_spectrum")
    print()
    print("# Load reference spectrum")
    print("energy, counts = load_spectrum('path/to/reference.txt')")
    print()
    print("# Define known concentrations (ppm)")
    print("concentrations = {")
    print("    'Fe': 50000,  # 5% Fe")
    print("    'Cu': 10000,  # 1% Cu")
    print("    'Ti': 5000,   # 0.5% Ti")
    print("    # ... etc")
    print("}")
    print()
    print("# Create calibrator with FWHM")
    print("calibrator = InstrumentCalibrator(fwhm_calibration=fwhm_cal)")
    print()
    print("# Run calibration")
    print("result = calibrator.calibrate(")
    print("    energy=energy,")
    print("    counts=counts,")
    print("    reference_concentrations=concentrations,")
    print("    excitation_energy=30.0  # kV")
    print(")")
    print()
    print("# Save")
    print("calibrator.save_calibration(result, 'calibrations/instrument_calibration.json')")
    print("```")
    print()
    print("=" * 70)
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="XRFLab Calibration Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FWHM calibration only
  python calibration_workflow.py --step fwhm
  
  # Run instrument calibration (requires FWHM calibration first)
  python calibration_workflow.py --step instrument
  
  # Run both steps
  python calibration_workflow.py --step both
        """
    )
    
    parser.add_argument(
        '--step',
        choices=['fwhm', 'instrument', 'both'],
        default='fwhm',
        help='Which calibration step to run'
    )
    
    parser.add_argument(
        '--fwhm-file',
        default='calibrations/fwhm_calibration.json',
        help='Path to FWHM calibration file (for instrument step)'
    )
    
    args = parser.parse_args()
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "XRFLab Calibration Workflow" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    if args.step in ['fwhm', 'both']:
        fwhm_cal = step1_fwhm_calibration()
        
        if fwhm_cal is None:
            print("\n❌ FWHM calibration failed. Cannot proceed to instrument calibration.")
            return
        
        print("\n" + "=" * 70)
        print("✓ Step 1 complete!")
        print("=" * 70)
        
        if args.step == 'both':
            print("\n")
            step2_instrument_calibration(args.fwhm_file)
    
    elif args.step == 'instrument':
        step2_instrument_calibration(args.fwhm_file)
    
    print("\n" + "=" * 70)
    print("Calibration workflow complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Review calibration results in calibrations/")
    print("  2. Check plots in calibrations/fwhm_calibration.png")
    print("  3. Use calibrated parameters for sample analysis")
    print()


if __name__ == "__main__":
    main()
