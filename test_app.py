#!/usr/bin/env python3
"""
Quick test script to verify the XRF application components
Run this before launching the full GUI to check for import errors
"""

import sys

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    tests = [
        ("PySide6", "PySide6.QtWidgets"),
        ("PyQtGraph", "pyqtgraph"),
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("Pandas", "pandas"),
        ("H5py", "h5py"),
    ]
    
    optional_tests = [
        ("xraylib", "xraylib"),
        ("openpyxl", "openpyxl"),
    ]
    
    failed = []
    
    # Test required imports
    for name, module in tests:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)
    
    # Test optional imports
    print("\nOptional dependencies:")
    for name, module in optional_tests:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} (optional, not required for Phase 1)")
    
    return len(failed) == 0


def test_project_structure():
    """Test that all project files exist"""
    print("\nTesting project structure...")
    
    from pathlib import Path
    
    required_files = [
        "main.py",
        "ui/__init__.py",
        "ui/main_window.py",
        "ui/spectrum_widget.py",
        "ui/element_panel.py",
        "ui/results_panel.py",
        "core/__init__.py",
        "core/spectrum.py",
        "utils/__init__.py",
        "utils/io_handler.py",
        "utils/sample_data.py",
        "resources/styles.qss",
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            missing.append(file)
    
    return len(missing) == 0


def test_spectrum_class():
    """Test the Spectrum class"""
    print("\nTesting Spectrum class...")
    
    try:
        import numpy as np
        from core.spectrum import Spectrum
        
        # Create test spectrum
        energy = np.linspace(0, 20, 100)
        counts = np.random.poisson(1000, 100)
        
        spectrum = Spectrum(energy=energy, counts=counts)
        
        print(f"  ✓ Spectrum created: {spectrum}")
        print(f"  ✓ Energy range: {spectrum.energy_range}")
        print(f"  ✓ Total counts: {spectrum.total_counts:.0f}")
        
        # Test methods
        spectrum.normalize('max')
        print(f"  ✓ Normalization works")
        
        roi_energy, roi_counts = spectrum.get_roi(5, 10)
        print(f"  ✓ ROI extraction works: {len(roi_energy)} channels")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_io_handler():
    """Test the IO handler"""
    print("\nTesting IO handler...")
    
    try:
        import numpy as np
        from core.spectrum import Spectrum
        from utils.io_handler import IOHandler
        import tempfile
        from pathlib import Path
        
        # Create test spectrum
        energy = np.linspace(0, 20, 100)
        counts = np.random.poisson(1000, 100)
        spectrum = Spectrum(energy=energy, counts=counts)
        
        io_handler = IOHandler()
        
        # Test saving and loading
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test TXT format
            txt_path = Path(tmpdir) / "test.txt"
            io_handler.save_spectrum(spectrum, str(txt_path))
            loaded = io_handler.load_spectrum(str(txt_path))
            print(f"  ✓ TXT format: saved and loaded")
            
            # Test CSV format
            csv_path = Path(tmpdir) / "test.csv"
            io_handler.save_spectrum(spectrum, str(csv_path))
            loaded = io_handler.load_spectrum(str(csv_path))
            print(f"  ✓ CSV format: saved and loaded")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_sample_data_generator():
    """Test the sample data generator"""
    print("\nTesting sample data generator...")
    
    try:
        from utils.sample_data import generate_sample_spectrum
        
        spectrum = generate_sample_spectrum(
            num_channels=1024,
            elements=['Fe', 'Cu', 'Zn']
        )
        
        print(f"  ✓ Generated spectrum: {spectrum}")
        print(f"  ✓ Elements: {spectrum.metadata.get('elements')}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("XRF Application Component Tests")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Spectrum Class", test_spectrum_class()))
    results.append(("IO Handler", test_io_handler()))
    results.append(("Sample Data Generator", test_sample_data_generator()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to run the application.")
        print("\nTo launch the GUI, run:")
        print("  python main.py")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
