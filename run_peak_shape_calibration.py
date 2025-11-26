#!/usr/bin/env python3
"""
Quick script to run peak shape calibration on new XRF data

This will:
1. Load all element standard spectra (Fe, Cu, Ti, Zn, Mg, cubic zirconia)
2. Measure FWHM of characteristic peaks at different energies
3. Fit the detector resolution model: FWHM(E) = sqrt(FWHM_0^2 + 2.355^2 * epsilon * E)
4. Save calibration parameters and plots
"""

from calibrate_peak_shape import main

if __name__ == "__main__":
    print("=" * 70)
    print("XRF Detector Peak Shape Calibration")
    print("=" * 70)
    print()
    print("This calibration uses pure element standards to determine how")
    print("detector resolution (FWHM) varies with photon energy.")
    print()
    print("Standards:")
    print("  - Fe.txt: Fe Kα (6.40 keV), Fe Kβ (7.06 keV)")
    print("  - Cu.txt: Cu Kα (8.05 keV), Cu Kβ (8.91 keV)")
    print("  - Ti.txt: Ti Kα (4.51 keV), Ti Kβ (4.93 keV)")
    print("  - Zn.txt: Zn Kα (8.64 keV), Zn Kβ (9.57 keV)")
    print("  - Mg.txt: Mg Kα (1.25 keV)")
    print("  - cubic zirconia.txt: Zr Lα (2.04 keV), Zr Kα (15.75 keV)")
    print()
    print("All except cubic zirconia also have Al Kα (1.49 keV) from holder")
    print()
    print("=" * 70)
    print()
    
    main()
