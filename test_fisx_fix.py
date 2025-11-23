#!/usr/bin/env python3
"""
Test script to verify fisx integration fixes
"""

try:
    from core.fisx_integration import FisxCalculator
    
    print("Testing fisx integration...")
    
    # Create calculator
    calc = FisxCalculator(
        excitation_energy=50.0,
        incident_angle=45.0,
        takeoff_angle=45.0
    )
    print("✓ FisxCalculator initialized successfully")
    
    # Test composition (simple example)
    composition = {
        'Fe': 0.5,
        'Si': 0.3,
        'Al': 0.2
    }
    
    print("\nTesting calculate_intensities with composition:", composition)
    
    # This should now work without the setSample() error
    intensities = calc.calculate_intensities(composition, thickness=0.1, density=2.5)
    
    print("✓ calculate_intensities completed successfully")
    print(f"\nCalculated intensities for {len(intensities)} elements:")
    for element, lines in intensities.items():
        print(f"  {element}: {len(lines)} emission lines")
        for line_name, intensity in list(lines.items())[:3]:  # Show first 3 lines
            print(f"    {line_name}: {intensity:.3e}")
    
    print("\n✓ All tests passed!")
    
except ImportError as e:
    print(f"⚠ fisx not installed: {e}")
    print("Install with: pip install fisx")
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
