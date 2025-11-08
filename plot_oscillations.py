#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Plot oscillatory structure of correlation function.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from cmb_anomaly.correlation_function import (
    compute_cmb_correlation_function,
    fit_correlation_oscillations,
)
from cmb_anomaly.array_backend import array_load
from cmb_anomaly.utils import ensure_dir_for_file

# Load data (NO MASK)
temperature_path = "cache/cmb_temperature.npy"
print("Loading temperature map...")
temperature = array_load(temperature_path)

# Compute correlation function
print("Computing correlation function...")
theta_deg, xi = compute_cmb_correlation_function(
    temperature=temperature,
    mask=None,  # No mask
    theta_min_deg=0.1,
    theta_max_deg=180.0,
    n_bins=1000,
    use_gpu=False,
)

# Fit oscillatory template
print("Fitting oscillatory template...")
fit_result = fit_correlation_oscillations(
    theta_deg=theta_deg,
    xi=xi,
    kappa_min_per_deg=0.001,
    kappa_max_per_deg=100.0,
    beta_range=(0.5, 1.5),
    use_envelope_weights=True,
)

# Extract parameters
kappa_star = fit_result["best_kappa_per_deg"]
beta = fit_result["best_beta"]
spacing_delta_theta = fit_result["spacing_deg"]
correlation = fit_result["correlation"]
p_value = fit_result["p_value"]
amplitude = fit_result["best_amplitude"]
phase = fit_result["best_phase"]
offset = fit_result["best_offset"]

# Compute fitted curve
theta_rad = np.deg2rad(theta_deg)
envelope = theta_rad ** (-(3 - 2 * beta))
kappa_rad = np.deg2rad(kappa_star)
xi_fit = amplitude * envelope * np.cos(kappa_rad * theta_rad + phase) + offset

# Create focused visualization of oscillations
output_file = "results/correlation_function/oscillations_plot.png"
ensure_dir_for_file(output_file)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Oscillatory Structure in CMB Correlation Function', 
             fontsize=16, fontweight='bold')

# Plot 1: Full correlation function with fit
ax1 = axes[0]
ax1.plot(theta_deg, xi, 'b-', alpha=0.7, linewidth=1.5, label='Data ξ(θ)')
ax1.plot(theta_deg, xi_fit, 'r-', linewidth=2.5, 
         label=f'Fit: κ_*={kappa_star:.2f} deg⁻¹, β={beta:.2f}')
ax1.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax1.set_xlabel('Angular Separation θ (degrees)', fontsize=12)
ax1.set_ylabel('Correlation Function ξ(θ)', fontsize=12)
ax1.set_title('Full Correlation Function with Oscillatory Fit', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('symlog', linthresh=1e-10)

# Plot 2: Small angles - oscillations visible
ax2 = axes[1]
small_angle_mask = theta_deg <= 5.0
ax2.plot(theta_deg[small_angle_mask], xi[small_angle_mask], 
         'b-', alpha=0.8, linewidth=2, label='Data ξ(θ)', marker='o', markersize=3)
ax2.plot(theta_deg[small_angle_mask], xi_fit[small_angle_mask], 
         'r-', linewidth=2.5, label='Oscillatory Fit')
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
# Mark spacing
for n in range(1, 6):
    spacing_theta = n * spacing_delta_theta
    if spacing_theta <= 5.0:
        ax2.axvline(spacing_theta, color='g', linestyle=':', alpha=0.5, linewidth=1)
        ax2.text(spacing_theta, ax2.get_ylim()[1]*0.9, f'{n}×Δθ', 
                ha='center', fontsize=9, color='g')
ax2.set_xlabel('Angular Separation θ (degrees)', fontsize=12)
ax2.set_ylabel('Correlation Function ξ(θ)', fontsize=12)
ax2.set_title(f'Oscillations at Small Angles (θ ≤ 5°) - Spacing Δθ = {spacing_delta_theta:.4f}° = {spacing_delta_theta*60:.2f} arcmin', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Detrended oscillations (pure cosine)
ax3 = axes[2]
# Remove envelope to see pure oscillations
xi_detrended = (xi - offset) / (amplitude * envelope + 1e-15)
xi_fit_detrended = np.cos(kappa_rad * theta_rad + phase)
very_small_mask = theta_deg <= 2.0
ax3.plot(theta_deg[very_small_mask], xi_detrended[very_small_mask], 
         'b-', alpha=0.8, linewidth=2, label='Data (detrended)', marker='o', markersize=4)
ax3.plot(theta_deg[very_small_mask], xi_fit_detrended[very_small_mask], 
         'r-', linewidth=2.5, label=f'cos(κ_*θ + φ), κ_*={kappa_star:.2f} deg⁻¹')
ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
# Mark zeros of cosine
for n in range(0, 10):
    zero_theta = (n * np.pi - phase) / kappa_rad
    if zero_theta <= 2.0 and zero_theta > 0:
        ax3.axvline(np.rad2deg(zero_theta), color='g', linestyle=':', alpha=0.4, linewidth=1)
ax3.set_xlabel('Angular Separation θ (degrees)', fontsize=12)
ax3.set_ylabel('Detrended Correlation (normalized)', fontsize=12)
ax3.set_title(f'Pure Oscillatory Pattern: Δθ = {spacing_delta_theta:.4f}° = {spacing_delta_theta*60:.2f} arcmin', 
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Graph saved to: {output_file}")

# Print summary
print("\n" + "="*70)
print("OSCILLATORY STRUCTURE SUMMARY")
print("="*70)
print(f"κ_* (preferred wavenumber) = {kappa_star:.4f} per deg")
print(f"β (envelope exponent) = {beta:.4f}")
print(f"Δθ (spacing) = {spacing_delta_theta:.4f}° = {spacing_delta_theta*60:.2f} arcmin")
print(f"Correlation (fit quality) = {correlation:.4f}")
print(f"p-value = {p_value:.6e}")
print(f"Amplitude A = {amplitude:.6e}")
print(f"Phase φ = {phase:.4f} rad = {np.degrees(phase):.2f}°")
print("="*70)

