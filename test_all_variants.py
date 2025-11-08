#!/usr/bin/env python3
"""
Test all variants of correlation function fitting:
1. With envelope weights (current method)
2. With uniform weights
3. Different beta ranges
4. Different kappa ranges
"""

import sys
import logging
from cmb_anomaly.correlation_function import (
    compute_cmb_correlation_function,
    fit_correlation_oscillations,
)
from cmb_anomaly.array_backend import array_load
from cmb_anomaly.utils import log_array_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load data (NO MASK - use all pixels)
temperature_path = "cache/cmb_temperature.npy"

logging.info("Loading data (NO MASK - using all pixels)...")
temperature = array_load(temperature_path)
log_array_stats("temperature", temperature)

# No mask - use all pixels
mask = None

# Compute correlation function once (without mask)
logging.info("\nComputing correlation function (all pixels, no mask)...")
theta_deg, xi = compute_cmb_correlation_function(
    temperature=temperature,
    mask=None,  # No mask - use all pixels
    theta_min_deg=0.1,
    theta_max_deg=180.0,
    n_bins=1000,
    use_gpu=False,
)

# Test variants
variants = [
    {
        "name": "Envelope weights, β=[0.5,1.5], κ=[0.001,10.0]",
        "use_envelope_weights": True,
        "beta_range": (0.5, 1.5),
        "kappa_min": 0.001,
        "kappa_max": 10.0,
    },
    {
        "name": "Uniform weights, β=[0.5,1.5], κ=[0.001,10.0]",
        "use_envelope_weights": False,
        "beta_range": (0.5, 1.5),
        "kappa_min": 0.001,
        "kappa_max": 10.0,
    },
    {
        "name": "Envelope weights, β=[0.0,2.0], κ=[0.001,10.0]",
        "use_envelope_weights": True,
        "beta_range": (0.0, 2.0),
        "kappa_min": 0.001,
        "kappa_max": 10.0,
    },
    {
        "name": "Uniform weights, β=[0.0,2.0], κ=[0.001,10.0]",
        "use_envelope_weights": False,
        "beta_range": (0.0, 2.0),
        "kappa_min": 0.001,
        "kappa_max": 10.0,
    },
    {
        "name": "Envelope weights, β=[0.5,1.5], κ=[0.01,1.0]",
        "use_envelope_weights": True,
        "beta_range": (0.5, 1.5),
        "kappa_min": 0.01,
        "kappa_max": 1.0,
    },
    {
        "name": "Uniform weights, β=[0.5,1.5], κ=[0.01,1.0]",
        "use_envelope_weights": False,
        "beta_range": (0.5, 1.5),
        "kappa_min": 0.01,
        "kappa_max": 1.0,
    },
    {
        "name": "Envelope weights, β=[0.5,1.5], κ=[0.001,50.0]",
        "use_envelope_weights": True,
        "beta_range": (0.5, 1.5),
        "kappa_min": 0.001,
        "kappa_max": 50.0,
    },
    {
        "name": "Envelope weights, β=[0.5,1.5], κ=[0.001,100.0]",
        "use_envelope_weights": True,
        "beta_range": (0.5, 1.5),
        "kappa_min": 0.001,
        "kappa_max": 100.0,
    },
]

print("\n" + "=" * 80)
print("TESTING ALL VARIANTS")
print("=" * 80 + "\n")

results = []
for i, variant in enumerate(variants, 1):
    logging.info(f"\n{'=' * 80}")
    logging.info(f"Variant {i}/{len(variants)}: {variant['name']}")
    logging.info(f"{'=' * 80}\n")
    
    try:
        fit_result = fit_correlation_oscillations(
            theta_deg=theta_deg,
            xi=xi,
            kappa_min_per_deg=variant["kappa_min"],
            kappa_max_per_deg=variant["kappa_max"],
            beta_range=variant["beta_range"],
            use_envelope_weights=variant["use_envelope_weights"],
        )
        
        result = {
            "variant": variant["name"],
            "kappa": fit_result["best_kappa_per_deg"],
            "beta": fit_result["best_beta"],
            "correlation": fit_result["correlation"],
            "p_value": fit_result["p_value"],
            "spacing": fit_result["spacing_deg"],
            "amplitude": fit_result["best_amplitude"],
        }
        results.append(result)
        
        logging.info(f"\n✓ Success:")
        logging.info(f"  κ_* = {result['kappa']:.4f} per deg")
        logging.info(f"  β = {result['beta']:.4f}")
        logging.info(f"  Correlation = {result['correlation']:.4f}")
        logging.info(f"  Spacing Δθ = {result['spacing']:.2f}°")
        
    except Exception as e:
        logging.error(f"✗ Failed: {e}")
        results.append({
            "variant": variant["name"],
            "error": str(e),
        })

# Summary
print("\n" + "=" * 80)
print("SUMMARY OF ALL VARIANTS")
print("=" * 80 + "\n")

print(f"{'Variant':<50} {'κ_*':>10} {'β':>8} {'Corr':>8} {'Δθ':>10}")
print("-" * 80)

for r in results:
    if "error" in r:
        print(f"{r['variant']:<50} {'ERROR':>10}")
    else:
        print(f"{r['variant']:<50} {r['kappa']:>10.4f} {r['beta']:>8.2f} {r['correlation']:>8.4f} {r['spacing']:>10.2f}")

print("\n" + "=" * 80)

