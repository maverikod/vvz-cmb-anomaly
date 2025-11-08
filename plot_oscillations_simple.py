#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple script to plot oscillations from CSV data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os

# Load data from CSV
csv_file = "results/correlation_function/xi_theta.csv"
print(f"Loading data from {csv_file}...")

theta_deg = []
xi = []
xi_fit = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        theta_deg.append(float(row['theta_deg']))
        xi.append(float(row['xi']))
        xi_fit.append(float(row['xi_fit']))

theta_deg = np.array(theta_deg)
xi = np.array(xi)
xi_fit = np.array(xi_fit)

# Extract parameters from fit (approximate from data)
# Find spacing by looking at zeros of fitted function
zero_crossings = []
for i in range(len(xi_fit) - 1):
    if (xi_fit[i] > 0 and xi_fit[i+1] < 0) or (xi_fit[i] < 0 and xi_fit[i+1] > 0):
        zero_crossings.append(theta_deg[i])

if len(zero_crossings) > 1:
    spacing = np.mean(np.diff(zero_crossings[:5])) if len(zero_crossings) >= 5 else zero_crossings[1] - zero_crossings[0]
else:
    spacing = 0.047  # Default from known results

# Create visualization
output_file = "results/correlation_function/oscillations_plot.png"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Oscillatory Structure in CMB Correlation Function\n' + 
             f'Spacing Δθ ≈ {spacing:.4f}° = {spacing*60:.2f} arcmin', 
             fontsize=16, fontweight='bold')

# Plot 1: Full correlation function
ax1 = axes[0]
ax1.plot(theta_deg, xi, 'b-', alpha=0.7, linewidth=1.5, label='Data ξ(θ)')
ax1.plot(theta_deg, xi_fit, 'r-', linewidth=2.5, label='Oscillatory Fit')
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
    spacing_theta = n * spacing
    if spacing_theta <= 5.0:
        ax2.axvline(spacing_theta, color='g', linestyle=':', alpha=0.5, linewidth=1)
        ax2.text(spacing_theta, ax2.get_ylim()[1]*0.9, f'{n}×Δθ', 
                ha='center', fontsize=9, color='g')
ax2.set_xlabel('Angular Separation θ (degrees)', fontsize=12)
ax2.set_ylabel('Correlation Function ξ(θ)', fontsize=12)
ax2.set_title(f'Oscillations at Small Angles (θ ≤ 5°)', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Very small angles - detailed oscillations
ax3 = axes[2]
very_small_mask = theta_deg <= 2.0
ax3.plot(theta_deg[very_small_mask], xi[very_small_mask], 
         'b-', alpha=0.8, linewidth=2, label='Data ξ(θ)', marker='o', markersize=4)
ax3.plot(theta_deg[very_small_mask], xi_fit[very_small_mask], 
         'r-', linewidth=2.5, label='Oscillatory Fit')
ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
# Mark zeros
zero_indices = []
for i in range(len(xi_fit[very_small_mask]) - 1):
    if (xi_fit[very_small_mask][i] > 0 and xi_fit[very_small_mask][i+1] < 0) or \
       (xi_fit[very_small_mask][i] < 0 and xi_fit[very_small_mask][i+1] > 0):
        zero_theta = theta_deg[very_small_mask][i]
        ax3.axvline(zero_theta, color='g', linestyle=':', alpha=0.4, linewidth=1)
ax3.set_xlabel('Angular Separation θ (degrees)', fontsize=12)
ax3.set_ylabel('Correlation Function ξ(θ)', fontsize=12)
ax3.set_title(f'Detailed Oscillations (θ ≤ 2°)', 
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Graph saved to: {output_file}")

# Print summary
print("\n" + "="*70)
print("OSCILLATORY STRUCTURE")
print("="*70)
print(f"Spacing Δθ ≈ {spacing:.4f}° = {spacing*60:.2f} arcmin")
print(f"Number of oscillations in 5°: {5.0/spacing:.1f}")
print("="*70)

