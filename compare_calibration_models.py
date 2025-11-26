#!/usr/bin/env python3
"""
Compare different FWHM vs Energy models

This script fits multiple models to your calibration data and compares them
using statistical criteria (R², RMSE, AIC, BIC) to determine which best
describes your detector's resolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from calibrate_peak_shape import PeakShapeCalibrator


def compare_all_models():
    """Compare all available models"""
    
    # Set paths
    data_dir = Path("sample_data/data")
    output_dir = Path("sample_data")
    
    # Create calibrator and process data
    print("=" * 70)
    print("Model Comparison for Peak Shape Calibration")
    print("=" * 70)
    
    calibrator = PeakShapeCalibrator(data_dir)
    calibrator.process_all_files()
    
    if len(calibrator.measurements) < 3:
        print("\n❌ Not enough measurements for calibration!")
        return
    
    # Available models
    models = ['detector', 'linear', 'quadratic', 'exponential', 'power']
    results_dict = {}
    
    print("\n" + "=" * 70)
    print("Fitting all models...")
    print("=" * 70)
    
    # Fit each model
    for model in models:
        print(f"\n{model.upper()} MODEL:")
        print("-" * 40)
        
        try:
            # Create a fresh copy of measurements for each fit
            calibrator_copy = PeakShapeCalibrator(data_dir)
            calibrator_copy.measurements = calibrator.measurements.copy()
            
            results = calibrator_copy.fit_resolution_model(
                remove_outliers=True, 
                model=model
            )
            results_dict[model] = results
            
            # Print results
            print(f"  R² = {results['r_squared']:.4f}")
            print(f"  RMSE = {results['rmse']*1000:.2f} eV")
            print(f"  AIC = {results['aic']:.2f}")
            print(f"  BIC = {results['bic']:.2f}")
            
            # Print model-specific parameters
            if model == 'detector':
                print(f"  FWHM₀ = {results['fwhm_0']*1000:.1f} ± {results['fwhm_0_err']*1000:.1f} eV")
                print(f"  ε = {results['epsilon']*1000:.2f} ± {results['epsilon_err']*1000:.2f} eV/keV")
            elif model == 'linear':
                print(f"  Intercept = {results['intercept']*1000:.1f} ± {results['intercept_err']*1000:.1f} eV")
                print(f"  Slope = {results['slope']*1000:.2f} ± {results['slope_err']*1000:.2f} eV/keV")
            elif model == 'quadratic':
                print(f"  a = {results['intercept']*1000:.1f} eV")
                print(f"  b = {results['linear_coef']*1000:.2f} eV/keV")
                print(f"  c = {results['quadratic_coef']*1000:.3f} eV/keV²")
            elif model == 'exponential':
                print(f"  Amplitude = {results['amplitude']*1000:.1f} eV")
                print(f"  Exponent = {results['exponent']:.4f} keV⁻¹")
            elif model == 'power':
                print(f"  Amplitude = {results['amplitude']*1000:.1f} eV")
                print(f"  Power = {results['power']:.3f}")
            
            print(f"  ✓ Fit successful")
            
        except Exception as e:
            print(f"  ✗ Fit failed: {e}")
            results_dict[model] = None
    
    # Model comparison summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<15} {'R²':<10} {'RMSE (eV)':<12} {'AIC':<10} {'BIC':<10} {'Rank'}")
    print("-" * 70)
    
    # Sort by AIC (lower is better)
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['aic'])
    
    for rank, (model, results) in enumerate(sorted_models, 1):
        r2 = results['r_squared']
        rmse = results['rmse'] * 1000
        aic = results['aic']
        bic = results['bic']
        
        # Add indicator for best model
        indicator = "⭐" if rank == 1 else "  "
        
        print(f"{indicator} {model:<13} {r2:<10.4f} {rmse:<12.2f} {aic:<10.2f} {bic:<10.2f} #{rank}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    best_model = sorted_models[0][0]
    best_results = sorted_models[0][1]
    
    print(f"\n⭐ Best model: {best_model.upper()}")
    print(f"   R² = {best_results['r_squared']:.4f}")
    print(f"   RMSE = {best_results['rmse']*1000:.2f} eV")
    
    # Model selection guidance
    print("\nModel Selection Criteria:")
    print("  • AIC (Akaike Information Criterion): Lower is better")
    print("    - Balances fit quality with model complexity")
    print("    - Difference < 2: Models are equivalent")
    print("    - Difference 2-10: Substantial support for lower AIC")
    print("    - Difference > 10: Strong support for lower AIC")
    print()
    print("  • BIC (Bayesian Information Criterion): Lower is better")
    print("    - More strongly penalizes complex models than AIC")
    print("    - Prefer when you want simpler models")
    print()
    print("  • R²: Higher is better (but doesn't penalize complexity)")
    print("  • RMSE: Lower is better (absolute fit quality)")
    
    # Physical interpretation
    print("\nPhysical Interpretation:")
    if best_model == 'detector':
        print("  ✓ Standard detector model is best")
        print("    This is expected for Si detectors with Fano statistics")
        print("    FWHM² = FWHM₀² + 2.355² · ε · E")
    elif best_model == 'linear':
        print("  ⚠ Linear model fits better than detector model")
        print("    This suggests simplified behavior or limited energy range")
        print("    FWHM(E) = a + b·E")
    elif best_model == 'quadratic':
        print("  ⚠ Quadratic model fits better")
        print("    May indicate additional energy-dependent effects")
        print("    Could be detector artifacts or incomplete charge collection")
    elif best_model == 'exponential':
        print("  ⚠ Exponential model fits better")
        print("    Unusual for detectors - check for systematic issues")
    elif best_model == 'power':
        print("  ⚠ Power law fits better")
        print("    May work empirically but lacks physical basis")
    
    # Check if detector model is close to best
    if best_model != 'detector' and 'detector' in valid_results:
        detector_aic = valid_results['detector']['aic']
        best_aic = best_results['aic']
        delta_aic = detector_aic - best_aic
        
        print(f"\nDetector model comparison:")
        print(f"  ΔAIC = {delta_aic:.2f}")
        if delta_aic < 2:
            print("  → Detector model is essentially equivalent (ΔAIC < 2)")
            print("  → Recommend using detector model for physical interpretation")
        elif delta_aic < 10:
            print("  → Detector model has some support (2 < ΔAIC < 10)")
            print("  → Consider using detector model if physical basis is important")
        else:
            print("  → Detector model is substantially worse (ΔAIC > 10)")
            print("  → May indicate detector issues or non-standard behavior")
    
    # Create comparison plot
    print("\n" + "=" * 70)
    print("Creating comparison plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    energies = np.array([m.energy for m in calibrator.measurements])
    fwhms = np.array([m.fwhm * 1000 for m in calibrator.measurements])
    e_model = np.linspace(min(energies), max(energies), 200)
    
    for idx, (model, results) in enumerate(sorted_models):
        if idx >= 6:  # Only plot first 6
            break
        
        ax = axes[idx]
        
        # Plot data
        ax.scatter(energies, fwhms, s=80, alpha=0.6, c='blue', edgecolors='black', label='Data')
        
        # Plot model
        if model == 'detector':
            fwhm_0, eps = results['fwhm_0'], results['epsilon']
            fwhm_model = np.sqrt(fwhm_0**2 + 2.355**2 * eps * e_model) * 1000
            fwhm_pred = np.sqrt(fwhm_0**2 + 2.355**2 * eps * energies) * 1000
        elif model == 'linear':
            a, b = results['intercept'], results['slope']
            fwhm_model = (a + b * e_model) * 1000
            fwhm_pred = (a + b * energies) * 1000
        elif model == 'quadratic':
            a, b, c = results['intercept'], results['linear_coef'], results['quadratic_coef']
            fwhm_model = (a + b * e_model + c * e_model**2) * 1000
            fwhm_pred = (a + b * energies + c * energies**2) * 1000
        elif model == 'exponential':
            a, b = results['amplitude'], results['exponent']
            fwhm_model = a * np.exp(b * e_model) * 1000
            fwhm_pred = a * np.exp(b * energies) * 1000
        elif model == 'power':
            a, b = results['amplitude'], results['power']
            fwhm_model = a * e_model**b * 1000
            fwhm_pred = a * energies**b * 1000
        
        ax.plot(e_model, fwhm_model, 'r-', linewidth=2, label='Fit')
        
        # Add residuals as error bars
        residuals = fwhms - fwhm_pred
        for e, f, r in zip(energies, fwhms, residuals):
            ax.plot([e, e], [f, f-r], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Energy (keV)', fontsize=10)
        ax.set_ylabel('FWHM (eV)', fontsize=10)
        
        # Title with rank indicator
        rank_str = "⭐ BEST" if idx == 0 else f"#{idx+1}"
        ax.set_title(f'{model.upper()} ({rank_str})\nR²={results["r_squared"]:.4f}, RMSE={results["rmse"]*1000:.1f} eV',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(sorted_models), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Peak Shape Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    
    plt.show()
    
    # Save best model
    best_calibrator = PeakShapeCalibrator(data_dir)
    best_calibrator.measurements = calibrator.measurements.copy()
    best_calibrator.save_calibration(best_results, output_dir / f"peak_shape_calibration_{best_model}.json")
    
    print("\n" + "=" * 70)
    print("✓ Model comparison complete!")
    print(f"✓ Best model: {best_model.upper()}")
    print(f"✓ Results saved to: peak_shape_calibration_{best_model}.json")
    print("=" * 70)


if __name__ == "__main__":
    compare_all_models()
