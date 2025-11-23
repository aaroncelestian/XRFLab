#!/usr/bin/env python3
"""
Test script for tube spectrum configuration
"""

from core.fisx_integration import FisxCalculator

print("Testing tube spectrum configuration...")
print()

# Create calculator with Rh tube at 50 keV
calc = FisxCalculator(
    excitation_energy=50.0,
    tube_element='Rh',
    incident_angle=45.0,
    takeoff_angle=45.0
)

print()
print("Tube spectrum configured successfully!")
print()

# Test intensity calculation
composition = {
    'Fe': 0.5,
    'Ca': 0.3,
    'Si': 0.2
}

print("Calculating intensities for test composition...")
intensities = calc.calculate_intensities(composition, thickness=0.1, density=2.5)

print(f"Calculated intensities for {len(intensities)} elements")
for element, lines in intensities.items():
    print(f"  {element}: {len(lines)} lines")

print()
print("Test complete!")
