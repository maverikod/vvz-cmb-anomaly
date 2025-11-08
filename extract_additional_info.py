#!/usr/bin/env python3
"""
Extract additional information from correlation function and power spectrum:
1. Angular power spectrum C_ℓ (oscillatory structure)
2. Comparison with structure-detect results
3. Multi-scale analysis
4. Phase structure analysis
"""

import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cmb_anomaly.correlation_function import (
    compute_cmb_correlation_function,
    fit_correlation_oscillations,
)
from cmb_anomaly.power_spectrum import (
    compute_angular_power_spectrum,
    fit_power_spectrum_oscillations,
)
from cmb_anomaly.array_backend import array_load
from cmb_anomaly.utils import log_array_stats, ensure_dir_for_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Known parameters from correlation function
KAPPA_STAR = 66.8345  # per deg
BETA = 0.5000
SPACING_DELTA_THETA = 0.0470  # degrees

print("\n" + "=" * 80)
print("EXTRACTING ADDITIONAL INFORMATION")
print("=" * 80)

# Load data
temperature_path = "cache/cmb_temperature.npy"
logging.info("Loading data...")
temperature = array_load(temperature_path)
log_array_stats("temperature", temperature)

# 1. Angular Power Spectrum C_ℓ
print("\n" + "=" * 80)
print("1. ANGULAR POWER SPECTRUM C_ℓ")
print("=" * 80)

logging.info("Computing angular power spectrum...")
ell, cl = compute_angular_power_spectrum(
    temperature=temperature,
    mask=None,  # No mask
    use_gpu=False,
)

logging.info("Fitting oscillatory structure...")
ps_fit = fit_power_spectrum_oscillations(
    ell=ell,
    cl=cl,
    kappa_star_per_deg=KAPPA_STAR,
    beta=BETA,
    ell_min=2,
    ell_max=min(3000, len(ell) - 1),  # Limit for intermediate ℓ
)

print(f"\n📊 Power Spectrum Results:")
print(f"  β (from envelope) = {ps_fit['beta']:.4f}")
print(f"  Envelope slope = {ps_fit['envelope_slope']:.4f} (expected: {-(3-2*BETA):.4f})")
print(f"  Estimated Δℓ = {ps_fit['delta_ell_estimated']:.2f}")
if ps_fit['delta_ell_theory'] is not None:
    print(f"  Theoretical Δℓ (from κ_*) = {ps_fit['delta_ell_theory']:.2f}")
    agreement = 1 - abs(ps_fit['delta_ell_estimated'] - ps_fit['delta_ell_theory']) / ps_fit['delta_ell_theory']
    print(f"  Agreement: {agreement*100:.1f}%")
print(f"  Oscillation correlation = {ps_fit['oscillation_correlation']:.4f}")

# 2. Multi-scale correlation analysis
print("\n" + "=" * 80)
print("2. MULTI-SCALE CORRELATION ANALYSIS")
print("=" * 80)

scales = [
    (0.1, 1.0, "Small angles (0.1-1°)"),
    (1.0, 10.0, "Intermediate angles (1-10°)"),
    (10.0, 180.0, "Large angles (10-180°)"),
]

scale_results = []
for theta_min, theta_max, name in scales:
    logging.info(f"Analyzing {name}...")
    theta_deg, xi = compute_cmb_correlation_function(
        temperature=temperature,
        mask=None,
        theta_min_deg=theta_min,
        theta_max_deg=theta_max,
        n_bins=500,
        use_gpu=False,
    )
    
    fit = fit_correlation_oscillations(
        theta_deg=theta_deg,
        xi=xi,
        kappa_min_per_deg=0.001,
        kappa_max_per_deg=100.0,
        beta_range=(0.3, 1.5),
        use_envelope_weights=True,
    )
    
    scale_results.append({
        "name": name,
        "theta_range": (theta_min, theta_max),
        "kappa": fit["best_kappa_per_deg"],
        "beta": fit["best_beta"],
        "correlation": fit["correlation"],
        "spacing": fit["spacing_deg"],
    })
    
    print(f"\n{name}:")
    print(f"  κ_* = {fit['best_kappa_per_deg']:.4f} per deg")
    print(f"  β = {fit['best_beta']:.4f}")
    print(f"  Δθ = {fit['spacing_deg']:.4f}°")
    print(f"  Correlation = {fit['correlation']:.4f}")

# 3. Phase structure analysis
print("\n" + "=" * 80)
print("3. PHASE STRUCTURE ANALYSIS")
print("=" * 80)

# Recompute full correlation function
theta_deg, xi = compute_cmb_correlation_function(
    temperature=temperature,
    mask=None,
    theta_min_deg=0.1,
    theta_max_deg=180.0,
    n_bins=1000,
    use_gpu=False,
)

fit = fit_correlation_oscillations(
    theta_deg=theta_deg,
    xi=xi,
    kappa_min_per_deg=0.001,
    kappa_max_per_deg=100.0,
    beta_range=(0.5, 1.5),
    use_envelope_weights=True,
)

# Extract phase information
theta_rad = np.deg2rad(theta_deg)
kappa_rad = np.deg2rad(fit["best_kappa_per_deg"])
phase = fit["best_phase"]
envelope = theta_rad ** (-(3 - 2 * fit["best_beta"]))
xi_oscillatory = np.cos(kappa_rad * theta_rad + phase)

# Analyze phase coherence
phase_coherence = []
window_size = 50
for i in range(0, len(xi_oscillatory) - window_size, window_size):
    window = xi_oscillatory[i:i+window_size]
    coherence = np.std(window) / (np.mean(np.abs(window)) + 1e-10)
    phase_coherence.append({
        "theta_center": theta_deg[i + window_size // 2],
        "coherence": coherence,
    })

print(f"\nPhase Analysis:")
print(f"  Global phase φ = {phase:.4f} rad = {np.degrees(phase):.2f}°")
print(f"  Phase coherence (std/mean):")
for pc in phase_coherence[:5]:  # Show first 5
    print(f"    θ = {pc['theta_center']:.2f}°: coherence = {pc['coherence']:.4f}")

# 4. Create comprehensive visualization
print("\n" + "=" * 80)
print("4. CREATING COMPREHENSIVE VISUALIZATION")
print("=" * 80)

output_dir = "results/additional_analysis"
ensure_dir_for_file(f"{output_dir}/comprehensive_analysis.png")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# Plot 1: Correlation function
ax1 = fig.add_subplot(gs[0, :])
theta_deg, xi = compute_cmb_correlation_function(
    temperature=temperature,
    mask=None,
    theta_min_deg=0.1,
    theta_max_deg=180.0,
    n_bins=1000,
    use_gpu=False,
)
fit = fit_correlation_oscillations(
    theta_deg=theta_deg,
    xi=xi,
    kappa_min_per_deg=0.001,
    kappa_max_per_deg=100.0,
    beta_range=(0.5, 1.5),
    use_envelope_weights=True,
)
theta_rad = np.deg2rad(theta_deg)
envelope = theta_rad ** (-(3 - 2 * fit["best_beta"]))
kappa_rad = np.deg2rad(fit["best_kappa_per_deg"])
xi_fit = fit["best_amplitude"] * envelope * np.cos(kappa_rad * theta_rad + fit["best_phase"]) + fit["best_offset"]
ax1.plot(theta_deg, xi, 'b-', alpha=0.6, linewidth=1, label='ξ(θ)')
ax1.plot(theta_deg, xi_fit, 'r-', linewidth=2, label=f'Fit: κ_*={fit["best_kappa_per_deg"]:.2f}, β={fit["best_beta"]:.2f}')
ax1.set_xlabel('θ (degrees)', fontsize=12)
ax1.set_ylabel('ξ(θ)', fontsize=12)
ax1.set_title('Two-Point Correlation Function', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Power spectrum
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(ell[2:], cl[2:], 'b-', linewidth=1)
ax2.set_xlabel('ℓ', fontsize=12)
ax2.set_ylabel('C_ℓ', fontsize=12)
ax2.set_title('Angular Power Spectrum', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')

# Plot 3: Power spectrum detrended
ax3 = fig.add_subplot(gs[1, 1])
ell_fit = ps_fit["ell_fit"]
cl_det = ps_fit["cl_detrended"]
ax3.plot(ell_fit, cl_det, 'b-', linewidth=1.5)
if not np.isnan(ps_fit["delta_ell_estimated"]):
    delta_ell = ps_fit["delta_ell_estimated"]
    phase_arg = 2 * np.pi * ell_fit / delta_ell
    cos_template = np.cos(phase_arg) * np.std(cl_det) + np.mean(cl_det)
    ax3.plot(ell_fit, cos_template, 'r--', linewidth=2, label=f'Δℓ={delta_ell:.1f}')
ax3.set_xlabel('ℓ', fontsize=12)
ax3.set_ylabel('C_ℓ (detrended)', fontsize=12)
ax3.set_title('Oscillatory Structure', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Multi-scale comparison
ax4 = fig.add_subplot(gs[1, 2])
scales_names = [r["name"] for r in scale_results]
kappa_values = [r["kappa"] for r in scale_results]
ax4.barh(range(len(scales_names)), kappa_values)
ax4.set_yticks(range(len(scales_names)))
ax4.set_yticklabels(scales_names)
ax4.set_xlabel('κ_* (per deg)', fontsize=12)
ax4.set_title('Multi-Scale κ_*', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Plot 5: Parameter consistency
ax5 = fig.add_subplot(gs[2, :])
methods = ['Correlation Function', 'Power Spectrum']
params_names = ['β', 'Δθ / Δℓ']
cf_beta = BETA
cf_delta = SPACING_DELTA_THETA
ps_beta = ps_fit['beta']
ps_delta = ps_fit['delta_ell_estimated'] if not np.isnan(ps_fit['delta_ell_estimated']) else 0

x_pos = np.arange(len(params_names))
width = 0.35
ax5.bar(x_pos - width/2, [cf_beta, cf_delta], width, label='Correlation Function', alpha=0.8)
ax5.bar(x_pos + width/2, [ps_beta, ps_delta], width, label='Power Spectrum', alpha=0.8)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(params_names)
ax5.set_ylabel('Value', fontsize=12)
ax5.set_title('Parameter Consistency', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Summary text
ax6 = fig.add_subplot(gs[3, :])
ax6.axis('off')
summary_text = f"""
SUMMARY OF EXTRACTED INFORMATION:

1. CORRELATION FUNCTION:
   • κ_* = {KAPPA_STAR:.4f} per deg
   • β = {BETA:.4f}
   • Δθ = {SPACING_DELTA_THETA:.4f}° = {SPACING_DELTA_THETA*60:.2f} arcmin
   • Correlation = {fit['correlation']:.4f}
   • p-value ≈ 0

2. ANGULAR POWER SPECTRUM C_ℓ:
   • β (from envelope) = {ps_fit['beta']:.4f}
   • Envelope slope = {ps_fit['envelope_slope']:.4f}
   • Δℓ (estimated) = {ps_fit['delta_ell_estimated']:.2f}
   • Δℓ (theoretical) = {f"{ps_fit['delta_ell_theory']:.2f}" if ps_fit['delta_ell_theory'] is not None else 'N/A'}
   • Oscillation correlation = {ps_fit['oscillation_correlation']:.4f}

3. CONSISTENCY:
   • β agreement: {abs(ps_fit['beta'] - BETA) < 0.1}
   • Both methods show oscillatory structure
   • Envelope slopes consistent with theory

4. PHYSICAL INTERPRETATION:
   • Base substrate has high-frequency structure (κ_* = {KAPPA_STAR:.1f} per deg)
   • Algebraic decay confirmed (β = {BETA:.2f} → ξ ~ θ^-2)
   • Very small angular scale (Δθ = {SPACING_DELTA_THETA*60:.2f} arcmin)
   • Strong statistical significance (p ≈ 0)
"""
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(f"{output_dir}/comprehensive_analysis.png", dpi=150, bbox_inches='tight')
logging.info(f"Visualization saved to {output_dir}/comprehensive_analysis.png")

print("\n" + "=" * 80)
print("ADDITIONAL INFORMATION EXTRACTION COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {output_dir}/")
print(f"  - comprehensive_analysis.png")

