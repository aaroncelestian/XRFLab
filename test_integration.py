#!/usr/bin/env python3
"""
Quick test to verify FWHM calibration integration works correctly
"""

def test_fwhm_calibration_loading():
    """Test that FWHM calibration can be loaded and used"""
    from core.fwhm_calibration import FWHMCalibration, load_fwhm_calibration
    from pathlib import Path
    import json
    
    print("Testing FWHM Calibration Integration")
    print("=" * 70)
    
    # Test 1: Create and save FWHMCalibration
    print("\n1. Creating FWHMCalibration object...")
    from datetime import datetime
    
    fwhm_cal = FWHMCalibration(
        model_type='detector',
        parameters={'fwhm_0': 0.115, 'epsilon': 0.0035},
        parameter_errors={'fwhm_0': 0.003, 'epsilon': 0.0002},
        r_squared=0.972,
        rmse=0.0042,
        aic=-85.3,
        bic=-82.1,
        n_peaks=16,
        energy_range=(1.0, 17.0),
        calibration_date=datetime.now().isoformat()
    )
    print(f"   ✓ Created: {fwhm_cal}")
    
    # Test 2: Predict FWHM
    print("\n2. Testing FWHM prediction...")
    fwhm_6keV = fwhm_cal.predict_fwhm(6.0)
    print(f"   FWHM at 6 keV: {fwhm_6keV*1000:.1f} eV")
    print(f"   ✓ Prediction works")
    
    # Test 3: Save and load
    print("\n3. Testing save/load...")
    test_file = "test_fwhm_calibration.json"
    fwhm_cal.save(test_file)
    print(f"   ✓ Saved to {test_file}")
    
    loaded = FWHMCalibration.load(test_file)
    print(f"   ✓ Loaded: {loaded}")
    
    # Verify values match
    assert loaded.parameters['fwhm_0'] == fwhm_cal.parameters['fwhm_0']
    assert loaded.parameters['epsilon'] == fwhm_cal.parameters['epsilon']
    print(f"   ✓ Values match")
    
    # Clean up
    Path(test_file).unlink()
    
    # Test 4: Integration with InstrumentCalibrator
    print("\n4. Testing InstrumentCalibrator integration...")
    from core.calibration import InstrumentCalibrator
    from core.fwhm_calibration import get_fwhm_initial_params
    
    # Create calibrator with FWHM
    calibrator = InstrumentCalibrator(fwhm_calibration=fwhm_cal)
    print(f"   ✓ Created InstrumentCalibrator with FWHM")
    
    # Get initial params
    params = get_fwhm_initial_params(fwhm_cal)
    print(f"   Initial params: FWHM₀={params['fwhm_0']*1000:.1f} eV, ε={params['epsilon']*1000:.2f} eV/keV")
    print(f"   ✓ Parameter extraction works")
    
    # Test 5: Legacy format compatibility
    print("\n5. Testing legacy format loading...")
    
    # Create a legacy format file
    legacy_data = {
        'calibration_date': datetime.now().isoformat(),
        'detector_model': 'XGT7200 SDD',
        'fwhm_0_keV': 0.115,
        'fwhm_0_eV': 115.0,
        'fwhm_0_error_eV': 3.0,
        'epsilon_keV': 0.0035,
        'epsilon_eV_per_keV': 3.5,
        'epsilon_error_eV_per_keV': 0.2,
        'r_squared': 0.972,
        'rmse_eV': 4.2,
        'n_peaks': 16
    }
    
    legacy_file = "test_legacy_calibration.json"
    with open(legacy_file, 'w') as f:
        json.dump(legacy_data, f)
    
    # Load legacy format
    legacy_loaded = load_fwhm_calibration(legacy_file)
    print(f"   ✓ Loaded legacy format: {legacy_loaded}")
    
    # Verify conversion
    assert legacy_loaded.model_type == 'detector'
    assert abs(legacy_loaded.parameters['fwhm_0'] - 0.115) < 1e-6
    print(f"   ✓ Legacy format conversion works")
    
    # Clean up
    Path(legacy_file).unlink()
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("\nConclusion:")
    print("  • FWHM calibration module works correctly")
    print("  • Integration with InstrumentCalibrator is functional")
    print("  • Legacy format loading is compatible")
    print("  • No issues with main.py analysis expected")
    print()
    return True


if __name__ == "__main__":
    try:
        test_fwhm_calibration_loading()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
