"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Angular power spectrum C_ℓ analysis for CMB maps.
Theory predicts oscillatory structure with spacing Δℓ ~ π r_* κ_*
and envelope C_ℓ ~ ℓ^-(3-2β).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import healpy as hp
import numpy as np
from scipy.stats import t as student_t

from .array_backend import is_gpu_array, cp
from .utils import ensure_dir_for_file, log_array_stats


def compute_angular_power_spectrum(
    temperature: np.ndarray,
    mask: Optional[np.ndarray] = None,
    lmax: Optional[int] = None,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute angular power spectrum C_ℓ from CMB temperature map.
    
    Theory predicts (Section E):
    C_ℓ = 4π∫ dk k² P_Θ(k) |Δ_ℓ(k)|²
    
    With oscillatory structure:
    - Spacing: Δℓ ~ π r_* κ_*
    - Envelope: C_ℓ ~ ℓ^-(3-2β) (intermediate ℓ)
    
    Args:
        temperature: Full-sky CMB temperature map (HEALPix ring ordering)
        mask: Optional boolean mask (True = valid, False = masked)
        lmax: Maximum multipole (None = 2*nside - 1)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        ell: Multipole values
        cl: Angular power spectrum C_ℓ
    """
    # Convert to CPU arrays (healpy works on CPU)
    if is_gpu_array(temperature):
        temperature = cp.asnumpy(temperature)
    temperature = np.asarray(temperature, dtype=np.float64)
    
    nside = hp.npix2nside(temperature.size)
    npix = temperature.size
    
    if lmax is None:
        lmax = 2 * nside - 1
    
    logging.info(f"Computing angular power spectrum:")
    logging.info(f"  NSIDE: {nside}, total pixels: {npix}")
    logging.info(f"  lmax: {lmax}")
    
    # Apply mask if provided
    if mask is not None:
        if is_gpu_array(mask):
            mask = cp.asnumpy(mask)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != temperature.shape:
            raise ValueError("Mask shape must match temperature shape")
        
        # Set masked pixels to UNSEEN
        temperature_masked = temperature.copy()
        temperature_masked[~mask] = hp.UNSEEN
        valid_fraction = np.sum(mask) / npix
        logging.info(f"  Mask applied: {valid_fraction*100:.2f}% of sky")
    else:
        temperature_masked = temperature
        logging.info(f"  No mask: using full sky")
    
    # Remove mean (work with fluctuations)
    valid_pixels = np.isfinite(temperature_masked) & (temperature_masked != hp.UNSEEN)
    if np.sum(valid_pixels) == 0:
        raise ValueError("No valid pixels in temperature map")
    
    mean_temp = np.nanmean(temperature_masked[valid_pixels])
    delta_t = temperature_masked - mean_temp
    delta_t[~valid_pixels] = hp.UNSEEN
    
    # Compute spherical harmonic coefficients
    logging.info("  Computing spherical harmonic coefficients...")
    alm = hp.map2alm(delta_t, lmax=lmax)
    
    # Compute power spectrum
    logging.info("  Computing power spectrum...")
    cl = hp.alm2cl(alm, lmax=lmax)
    
    # Create multipole array
    ell = np.arange(len(cl))
    
    logging.info(f"  Power spectrum computed: {len(cl)} multipoles")
    logging.info(f"  C_ℓ range: [{np.min(cl[2:]):.6e}, {np.max(cl[2:]):.6e}]")
    
    return ell, cl


def fit_power_spectrum_oscillations(
    ell: np.ndarray,
    cl: np.ndarray,
    ell_min: int = 2,
    ell_max: Optional[int] = None,
    kappa_star_per_deg: Optional[float] = None,
    beta: Optional[float] = None,
    r_star_mpc: float = 1.0,
) -> dict:
    """
    Fit oscillatory template to angular power spectrum C_ℓ.
    
    Theory predicts:
    - Oscillatory structure: spacing Δℓ ~ π r_* κ_*
    - Envelope: C_ℓ ~ ℓ^-(3-2β) (intermediate ℓ)
    
    Args:
        ell: Multipole values
        cl: Power spectrum values
        ell_min: Minimum multipole to fit (skip monopole/dipole)
        ell_max: Maximum multipole to fit (None = all)
        kappa_star_per_deg: Known κ_* from correlation function (for consistency check)
        beta: Known β from correlation function (for consistency check)
        r_star_mpc: Characteristic distance r_* in Mpc (for Δℓ calculation)
    
    Returns:
        Dictionary with fit parameters and analysis
    """
    # Filter multipoles
    if ell_max is None:
        ell_max = len(ell) - 1
    
    valid = (ell >= ell_min) & (ell <= ell_max) & np.isfinite(cl) & (cl > 0)
    ell_fit = ell[valid]
    cl_fit = cl[valid]
    
    if len(ell_fit) < 10:
        raise ValueError("Not enough valid multipoles for fitting")
    
    logging.info(f"Fitting power spectrum oscillations:")
    logging.info(f"  Multipole range: {ell_min} ≤ ℓ ≤ {ell_max}")
    logging.info(f"  Valid multipoles: {len(ell_fit)}")
    
    # Convert to log space for envelope fitting
    log_ell = np.log(ell_fit)
    log_cl = np.log(cl_fit)
    
    # Fit envelope: C_ℓ ~ ℓ^-(3-2β)
    # log(C_ℓ) = -(3-2β) * log(ℓ) + const
    if beta is None:
        # Fit β from envelope
        coeffs = np.polyfit(log_ell, log_cl, 1)
        beta_fit = (3 + coeffs[0]) / 2.0
        envelope_slope = coeffs[0]
    else:
        # Use known β
        beta_fit = beta
        envelope_slope = -(3 - 2 * beta)
        # Fit constant
        const = np.mean(log_cl - envelope_slope * log_ell)
    
    # Remove envelope to see oscillations
    cl_detrended = cl_fit / (ell_fit ** envelope_slope)
    
    # Estimate spacing Δℓ from oscillations
    # Use FFT to find dominant frequency
    cl_detrended_norm = (cl_detrended - np.mean(cl_detrended)) / np.std(cl_detrended)
    fft = np.fft.rfft(cl_detrended_norm)
    freqs = np.fft.rfftfreq(len(cl_detrended_norm), d=1.0)
    power = np.abs(fft) ** 2
    
    # Find peak frequency (skip DC component)
    peak_idx = np.argmax(power[1:]) + 1
    peak_freq = freqs[peak_idx]
    
    if peak_freq > 0:
        delta_ell_estimated = 1.0 / peak_freq
    else:
        delta_ell_estimated = np.nan
    
    # Compare with theoretical prediction if κ_* is known
    if kappa_star_per_deg is not None:
        # Convert κ_* from per deg to per rad
        kappa_star_per_rad = kappa_star_per_deg * np.pi / 180.0
        # Theoretical spacing: Δℓ ~ π r_* κ_*
        # Need to convert κ_* to appropriate units
        # For now, use approximate: Δℓ ~ π * r_* * κ_* (in appropriate units)
        # This is approximate - need proper unit conversion
        delta_ell_theory = np.pi * r_star_mpc * kappa_star_per_rad
    else:
        delta_ell_theory = None
    
    # Compute correlation of detrended spectrum with cosine
    if not np.isnan(delta_ell_estimated) and delta_ell_estimated > 0:
        # Try to fit cosine with estimated spacing
        phase_arg = 2 * np.pi * ell_fit / delta_ell_estimated
        cos_template = np.cos(phase_arg)
        corr = np.corrcoef(cl_detrended_norm, cos_template)[0, 1]
    else:
        corr = 0.0
    
    logging.info(f"Best fit:")
    logging.info(f"  β (from envelope) = {beta_fit:.4f}")
    logging.info(f"  Envelope slope = {envelope_slope:.4f} (expected: {-(3-2*beta_fit):.4f})")
    logging.info(f"  Estimated Δℓ = {delta_ell_estimated:.2f}")
    if delta_ell_theory is not None:
        logging.info(f"  Theoretical Δℓ (from κ_*) = {delta_ell_theory:.2f}")
    logging.info(f"  Oscillation correlation = {corr:.4f}")
    
    return {
        "beta": beta_fit,
        "envelope_slope": envelope_slope,
        "delta_ell_estimated": delta_ell_estimated,
        "delta_ell_theory": delta_ell_theory,
        "oscillation_correlation": corr,
        "ell_fit": ell_fit,
        "cl_fit": cl_fit,
        "cl_detrended": cl_detrended,
    }


def analyze_power_spectrum(
    temperature_path: str,
    mask_path: Optional[str] = None,
    output_dir: str = "results/power_spectrum",
    kappa_star_per_deg: Optional[float] = None,
    beta: Optional[float] = None,
    use_gpu: bool = False,
) -> None:
    """
    Complete angular power spectrum analysis pipeline.
    
    Args:
        temperature_path: Path to temperature map .npy file
        mask_path: Optional path to mask .npy file
        output_dir: Output directory for results
        kappa_star_per_deg: Known κ_* from correlation function
        beta: Known β from correlation function
        use_gpu: Whether to use GPU acceleration
    """
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .array_backend import array_load
    
    ensure_dir_for_file(f"{output_dir}/cl.csv")
    
    logging.info("=" * 60)
    logging.info("Angular Power Spectrum Analysis")
    logging.info("=" * 60)
    
    # Load data
    logging.info("Loading temperature map...")
    temperature = array_load(temperature_path)
    log_array_stats("temperature", temperature)
    
    mask = None
    if mask_path:
        logging.info("Loading mask...")
        mask = array_load(mask_path)
        if is_gpu_array(mask):
            mask = cp.asnumpy(mask)
        mask = np.asarray(mask, dtype=bool)
        log_array_stats("mask", mask)
    
    # Compute power spectrum
    logging.info("\nComputing angular power spectrum...")
    ell, cl = compute_angular_power_spectrum(
        temperature=temperature,
        mask=mask,
        use_gpu=use_gpu,
    )
    
    # Fit oscillations
    logging.info("\nFitting oscillatory structure...")
    fit_result = fit_power_spectrum_oscillations(
        ell=ell,
        cl=cl,
        kappa_star_per_deg=kappa_star_per_deg,
        beta=beta,
    )
    
    # Save results
    with open(f"{output_dir}/cl.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ell", "cl", "cl_detrended"])
        for i in range(len(ell)):
            cl_det = fit_result["cl_detrended"][i] if i < len(fit_result["cl_detrended"]) else np.nan
            writer.writerow([ell[i], cl[i], cl_det])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Angular Power Spectrum C_ℓ Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Full power spectrum
    ax1 = axes[0, 0]
    ax1.plot(ell[2:], cl[2:], 'b-', linewidth=1, label='C_ℓ')
    if fit_result["beta"] is not None:
        envelope = ell[2:] ** fit_result["envelope_slope"]
        envelope_norm = envelope * np.max(cl[2:]) / np.max(envelope)
        ax1.plot(ell[2:], envelope_norm, 'r--', linewidth=2, 
                label=f'Envelope ~ ℓ^({fit_result["envelope_slope"]:.2f})')
    ax1.set_xlabel('Multipole ℓ', fontsize=12)
    ax1.set_ylabel('C_ℓ', fontsize=12)
    ax1.set_title('Angular Power Spectrum', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Detrended (oscillations)
    ax2 = axes[0, 1]
    ell_fit = fit_result["ell_fit"]
    cl_det = fit_result["cl_detrended"]
    ax2.plot(ell_fit, cl_det, 'b-', linewidth=1.5, label='C_ℓ (detrended)')
    if not np.isnan(fit_result["delta_ell_estimated"]):
        delta_ell = fit_result["delta_ell_estimated"]
        phase_arg = 2 * np.pi * ell_fit / delta_ell
        cos_template = np.cos(phase_arg) * np.std(cl_det) + np.mean(cl_det)
        ax2.plot(ell_fit, cos_template, 'r--', linewidth=2, 
                label=f'cos(2πℓ/{delta_ell:.1f})')
    ax2.set_xlabel('Multipole ℓ', fontsize=12)
    ax2.set_ylabel('C_ℓ (detrended)', fontsize=12)
    ax2.set_title(f'Oscillatory Structure (Δℓ ≈ {fit_result["delta_ell_estimated"]:.1f})', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Envelope comparison
    ax3 = axes[1, 0]
    ax3.plot(ell[2:], cl[2:], 'b-', linewidth=1, alpha=0.6, label='C_ℓ')
    if fit_result["beta"] is not None:
        beta_val = fit_result["beta"]
        envelope = ell[2:] ** (-(3 - 2 * beta_val))
        envelope_norm = envelope * np.max(cl[2:]) / np.max(envelope)
        ax3.plot(ell[2:], envelope_norm, 'g--', linewidth=2, 
                label=f'β = {beta_val:.2f} → ℓ^({-(3-2*beta_val):.2f})')
    ax3.set_xlabel('Multipole ℓ', fontsize=12)
    ax3.set_ylabel('C_ℓ', fontsize=12)
    ax3.set_title(f'Envelope: β = {fit_result["beta"]:.2f}', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # Plot 4: Comparison with correlation function results
    ax4 = axes[1, 1]
    if kappa_star_per_deg is not None and beta is not None:
        ax4.text(0.1, 0.9, f'From Correlation Function:', transform=ax4.transAxes, 
                fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.8, f'  κ_* = {kappa_star_per_deg:.2f} per deg', 
                transform=ax4.transAxes, fontsize=11)
        ax4.text(0.1, 0.7, f'  β = {beta:.2f}', transform=ax4.transAxes, fontsize=11)
        ax4.text(0.1, 0.6, f'From Power Spectrum:', transform=ax4.transAxes, 
                fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.5, f'  β = {fit_result["beta"]:.2f}', 
                transform=ax4.transAxes, fontsize=11)
        if fit_result["delta_ell_theory"] is not None:
            ax4.text(0.1, 0.4, f'  Δℓ (theory) = {fit_result["delta_ell_theory"]:.1f}', 
                    transform=ax4.transAxes, fontsize=11)
        ax4.text(0.1, 0.3, f'  Δℓ (estimated) = {fit_result["delta_ell_estimated"]:.1f}', 
                transform=ax4.transAxes, fontsize=11)
        if fit_result["delta_ell_theory"] is not None:
            agreement = abs(fit_result["delta_ell_estimated"] - fit_result["delta_ell_theory"]) / fit_result["delta_ell_theory"]
            ax4.text(0.1, 0.2, f'  Agreement: {100*(1-agreement):.1f}%', 
                    transform=ax4.transAxes, fontsize=11, 
                    color='green' if agreement < 0.2 else 'orange')
    else:
        ax4.text(0.5, 0.5, 'No correlation function\nresults for comparison', 
                transform=ax4.transAxes, fontsize=12, ha='center', va='center')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Consistency Check', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/power_spectrum_analysis.png", dpi=150, bbox_inches='tight')
    logging.info(f"\nVisualization saved to {output_dir}/power_spectrum_analysis.png")
    
    logging.info("=" * 60)
    logging.info("Power spectrum analysis complete!")
    logging.info("=" * 60)

