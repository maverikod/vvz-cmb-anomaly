"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Two-point correlation function analysis for CMB maps.
This directly probes the base substrate correlation structure
ξ_Θ(r) ~ r^-(3-2β) cos(κ_*r + φ_*) through the CMB projection.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import healpy as hp
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t

from .array_backend import is_gpu_array, cp, np as np_backend
from .utils import ensure_dir_for_file, log_array_stats


def compute_cmb_correlation_function(
    temperature: np.ndarray,
    mask: Optional[np.ndarray] = None,
    theta_min_deg: float = 0.1,
    theta_max_deg: float = 180.0,
    n_bins: int = 1000,
    n_jobs: Optional[int] = None,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two-point correlation function ξ(θ) for CMB map.
    
    This directly probes the base substrate correlation structure
    ξ_Θ(r) ~ r^-(3-2β) cos(κ_*r + φ_*)
    
    For CMB projection, this becomes:
    ξ_CMB(θ) ~ θ^-(3-2β) cos(κ_*θ + φ_*)
    
    Args:
        temperature: Full-sky CMB temperature map (HEALPix ring ordering)
        mask: Optional boolean mask (True = valid, False = masked)
        theta_min_deg: Minimum angular separation (degrees)
        theta_max_deg: Maximum angular separation (degrees)
        n_bins: Number of angular separation bins
        n_jobs: Number of parallel workers (None = auto)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        theta_centers: Angular separation bin centers (degrees)
        xi_theta: Correlation function values
    """
    # Convert to CPU arrays for correlation computation
    if is_gpu_array(temperature):
        temperature = cp.asnumpy(temperature)
    temperature = np.asarray(temperature, dtype=np.float64)
    
    if mask is not None:
        if is_gpu_array(mask):
            mask = cp.asnumpy(mask)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != temperature.shape:
            raise ValueError("Mask shape must match temperature shape")
        # Apply mask: set masked pixels to NaN
        temperature = np.where(mask, temperature, np.nan)
    
    nside = hp.npix2nside(temperature.size)
    npix = temperature.size
    
    # Remove mean (work with fluctuations)
    valid_pixels = np.isfinite(temperature)
    if np.sum(valid_pixels) == 0:
        raise ValueError("No valid pixels in temperature map")
    
    mean_temp = np.nanmean(temperature[valid_pixels])
    delta_t = temperature - mean_temp
    
    # Angular separation bins
    theta_edges = np.linspace(
        np.deg2rad(theta_min_deg),
        np.deg2rad(theta_max_deg),
        n_bins + 1
    )
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    theta_centers_deg = np.rad2deg(theta_centers)
    
    # Initialize correlation function
    xi_theta = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    
    logging.info(f"Computing correlation function:")
    logging.info(f"  NSIDE: {nside}, total pixels: {npix}")
    logging.info(f"  Valid pixels: {np.sum(valid_pixels)}")
    logging.info(f"  Angular range: {theta_min_deg}° to {theta_max_deg}°")
    logging.info(f"  Number of bins: {n_bins}")
    
    # Get pixel vectors for all valid pixels
    valid_pix_indices = np.where(valid_pixels)[0]
    n_valid = len(valid_pix_indices)
    
    if n_valid == 0:
        raise ValueError("No valid pixels after masking")
    
    logging.info(f"  Processing {n_valid} valid pixels...")
    
    # Compute correlation function
    # For efficiency, sample pairs if too many
    max_pairs = 10_000_000  # Limit to avoid memory issues
    if n_valid * (n_valid - 1) // 2 > max_pairs:
        # Sample pairs randomly
        n_samples = max_pairs
        logging.info(f"  Too many pairs ({n_valid * (n_valid - 1) // 2}), sampling {n_samples} pairs")
        np.random.seed(42)
        i_indices = np.random.choice(n_valid, size=n_samples, replace=True)
        j_indices = np.random.choice(n_valid, size=n_samples, replace=True)
        # Remove self-pairs
        mask_pairs = i_indices != j_indices
        i_indices = i_indices[mask_pairs]
        j_indices = j_indices[mask_pairs]
    else:
        # Use all pairs
        logging.info(f"  Using all {n_valid * (n_valid - 1) // 2} pixel pairs")
        i_indices, j_indices = np.triu_indices(n_valid, k=1)
    
    # Get pixel vectors
    pix_vecs_i = hp.pix2vec(nside, valid_pix_indices[i_indices])
    pix_vecs_j = hp.pix2vec(nside, valid_pix_indices[j_indices])
    
    # Compute angular separations
    # Angular distance: arccos(dot product)
    dot_products = (
        pix_vecs_i[0] * pix_vecs_j[0] +
        pix_vecs_i[1] * pix_vecs_j[1] +
        pix_vecs_i[2] * pix_vecs_j[2]
    )
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_separations = np.arccos(dot_products)
    
    # Get temperature values
    delta_t_i = delta_t[valid_pix_indices[i_indices]]
    delta_t_j = delta_t[valid_pix_indices[j_indices]]
    
    # Compute products
    products = delta_t_i * delta_t_j
    
    # Bin by angular separation
    bin_indices = np.digitize(angular_separations, theta_edges) - 1
    
    # Clip to valid bin range
    valid_bins = (bin_indices >= 0) & (bin_indices < n_bins)
    bin_indices = bin_indices[valid_bins]
    products = products[valid_bins]
    
    # Accumulate in bins
    for bi in range(n_bins):
        in_bin = bin_indices == bi
        if np.any(in_bin):
            xi_theta[bi] = np.mean(products[in_bin])
            counts[bi] = np.sum(in_bin)
    
    # Normalize by variance (if needed)
    # For correlation function, we want covariance, not correlation coefficient
    # So we keep it as is (mean product of fluctuations)
    
    logging.info(f"Correlation function computed:")
    logging.info(f"  Valid bins: {np.sum(np.isfinite(xi_theta))} / {n_bins}")
    logging.info(f"  Min pairs per bin: {np.min(counts[counts > 0]) if np.any(counts > 0) else 0}")
    logging.info(f"  Max pairs per bin: {np.max(counts)}")
    
    return theta_centers_deg, xi_theta


def fit_correlation_oscillations(
    theta_deg: np.ndarray,
    xi: np.ndarray,
    kappa_min_per_deg: float = 0.1,
    kappa_max_per_deg: float = 3.0,
    kappa_num: int = 200,
    beta_range: Tuple[float, float] = (0.5, 1.5),
    use_envelope_weights: bool = True,
    use_gpu: bool = False,
) -> dict:
    """
    Fit oscillatory template to correlation function.
    
    Fits: ξ(θ) = A θ^-(3-2β) cos(κθ + φ) + offset
    
    Args:
        theta_deg: Angular separations (degrees)
        xi: Correlation function values
        kappa_min_per_deg: Minimum κ to search (1/deg)
        kappa_max_per_deg: Maximum κ to search (1/deg)
        kappa_num: Number of κ values to test
        beta_range: Range of β values to test (min, max)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Dictionary with fit parameters:
            best_kappa: Preferred wavenumber κ_* (1/deg)
            best_beta: Envelope exponent β
            best_amplitude: A
            best_phase: φ (radians)
            best_offset: Constant offset
            correlation: Quality of fit (correlation coefficient)
            p_value: Statistical significance
    """
    # Filter valid data
    valid = np.isfinite(xi) & (theta_deg > 0)
    theta = theta_deg[valid]
    y = xi[valid]
    
    if len(theta) < 10:
        raise ValueError("Not enough valid data points for fitting")
    
    # Convert to radians for computation
    theta_rad = np.deg2rad(theta)
    
    # Detrend: remove mean
    y_mean = np.mean(y)
    y_detrended = y - y_mean
    
    # Normalize
    y_std = np.std(y_detrended)
    if y_std == 0:
        raise ValueError("Zero variance in correlation function")
    y_norm = y_detrended / y_std
    
    # Kappa grid
    kappa_grid = np.linspace(kappa_min_per_deg, kappa_max_per_deg, kappa_num)
    kappa_grid_rad = np.deg2rad(kappa_grid)  # Convert to rad^-1
    
    # Beta grid
    beta_grid = np.linspace(beta_range[0], beta_range[1], 50)
    
    best_corr = -np.inf
    best_kappa = np.nan
    best_beta = np.nan
    best_phase = np.nan
    best_amp = np.nan
    
    logging.info(f"Fitting oscillatory template:")
    logging.info(f"  κ range: [{kappa_min_per_deg:.3f}, {kappa_max_per_deg:.3f}] per deg")
    logging.info(f"  β range: [{beta_range[0]:.2f}, {beta_range[1]:.2f}]")
    logging.info(f"  Valid data points: {len(theta)}")
    
    # Grid search over kappa and beta
    for beta in beta_grid:
        # Algebraic envelope: θ^-(3-2β)
        envelope = theta_rad ** (-(3 - 2 * beta))
        
        if use_envelope_weights:
            # Method 1: Use envelope as weights (gives more weight to small angles)
            w = envelope
            # Normalize weights to avoid numerical issues
            w_sum = np.sum(w)
            if w_sum <= 0 or not np.isfinite(w_sum):
                continue
            w = w / w_sum  # Normalize so sum(w) = 1
        else:
            # Method 2: Uniform weights (equal weight to all angles)
            w = np.ones_like(theta_rad)
            w = w / np.sum(w)  # Normalize
        
        # For each kappa, compute correlation with cos(κθ)
        for kappa_rad in kappa_grid_rad:
            # Oscillatory part: cos(κθ + φ)
            phase_arg = kappa_rad * theta_rad
            
            # Project onto cos and sin
            cos_part = np.cos(phase_arg)
            sin_part = np.sin(phase_arg)
            
            # Weighted projections (following structure_detector method)
            # Compute weighted averages
            c = np.sum(w * y_norm * cos_part)
            s = np.sum(w * y_norm * sin_part)
            
            # Normalize by weighted norms to get proper correlation in [-1, 1]
            # Weighted correlation: corr = sum(w * y * x) / sqrt(sum(w * y^2) * sum(w * x^2))
            # For oscillatory template, we need to combine cos and sin properly
            y_w_norm = np.sqrt(np.sum(w * y_norm**2))
            
            # For oscillatory template cos(κθ + φ), we need to normalize the combined template
            # Template = cos_part * cos(φ) + sin_part * sin(φ)
            # The norm of this template is: sqrt(sum(w * (cos^2 + sin^2))) = sqrt(sum(w))
            # Since cos^2 + sin^2 = 1, and sum(w) = 1 (normalized), template_norm = 1
            template_norm = 1.0  # Since w is normalized and cos^2 + sin^2 = 1
            
            if y_w_norm > 0 and template_norm > 0:
                # Normalized projections
                # c and s are already weighted averages, so normalize by y_norm
                c_norm = c / y_w_norm
                s_norm = s / y_w_norm
                # Correlation is the amplitude, but clip to [0, 1] to avoid numerical errors
                corr = np.hypot(c_norm, s_norm)
                corr = min(corr, 1.0)  # Clip to [0, 1]
            else:
                corr = 0.0
            
            # Compute amplitude for best fit
            # Normalize cos_part and sin_part by their weighted norms
            cos_norm_sq = np.sum(w * cos_part**2)
            sin_norm_sq = np.sum(w * sin_part**2)
            if cos_norm_sq > 0 and sin_norm_sq > 0:
                # Optimal amplitude (weighted least squares)
                a_c = np.sum(w * y * cos_part) / cos_norm_sq
                a_s = np.sum(w * y * sin_part) / sin_norm_sq
                amp = np.hypot(a_c, a_s)
                phase = np.arctan2(a_s, a_c)
            else:
                amp = 0.0
                phase = 0.0
            
            if corr > best_corr:
                best_corr = corr
                best_kappa = np.rad2deg(kappa_rad)
                best_beta = beta
                best_phase = phase
                best_amp = amp
    
    # Compute p-value
    n = len(theta)
    r_clip = min(max(best_corr, -0.999999), 0.999999)
    t_stat = r_clip * np.sqrt((n - 2) / (1.0 - r_clip**2)) if n > 2 else 0.0
    p_val = 2.0 * (1.0 - student_t.cdf(abs(t_stat), df=max(n - 2, 1))) if n > 2 else 1.0
    
    logging.info(f"Best fit:")
    logging.info(f"  κ_* = {best_kappa:.4f} per deg")
    logging.info(f"  β = {best_beta:.4f}")
    logging.info(f"  Spacing Δθ = {np.pi / best_kappa:.2f}°")
    logging.info(f"  Amplitude A = {best_amp:.6e}")
    logging.info(f"  Phase φ = {best_phase:.4f} rad")
    logging.info(f"  Correlation = {best_corr:.4f}")
    logging.info(f"  p-value = {p_val:.6e}")
    
    return {
        "best_kappa_per_deg": best_kappa,
        "best_beta": best_beta,
        "best_amplitude": best_amp,
        "best_phase": best_phase,
        "best_offset": y_mean,
        "correlation": best_corr,
        "p_value": p_val,
        "spacing_deg": np.pi / best_kappa if best_kappa > 0 else np.inf,
        "num_points": n,
    }


def run_correlation_analysis(
    temperature_path: str,
    mask_path: Optional[str] = None,
    output_csv: str = "results/correlation_function/xi_theta.csv",
    theta_min_deg: float = 0.1,
    theta_max_deg: float = 180.0,
    n_bins: int = 1000,
    kappa_min_per_deg: float = 0.1,
    kappa_max_per_deg: float = 3.0,
    use_envelope_weights: bool = True,
    use_gpu: bool = False,
) -> None:
    """
    Complete correlation function analysis pipeline.
    
    Args:
        temperature_path: Path to temperature map .npy file
        mask_path: Optional path to mask .npy file
        output_csv: Path to output CSV file
        theta_min_deg: Minimum angular separation
        theta_max_deg: Maximum angular separation
        n_bins: Number of bins
        kappa_min_per_deg: Minimum κ for fitting
        kappa_max_per_deg: Maximum κ for fitting
        use_gpu: Whether to use GPU
    """
    import csv
    from .array_backend import array_load
    
    logging.info("=" * 60)
    logging.info("Correlation Function Analysis")
    logging.info("=" * 60)
    
    # Load data
    logging.info("Loading temperature map...")
    temperature = array_load(temperature_path)
    log_array_stats("temperature", temperature)
    
    mask = None
    if mask_path:
        logging.info("Loading mask...")
        mask = array_load(mask_path)
        # Convert GPU array to CPU if needed
        if is_gpu_array(mask):
            mask = cp.asnumpy(mask)
        mask = np.asarray(mask, dtype=bool)
        log_array_stats("mask", mask)
    
    # Compute correlation function
    logging.info("\nComputing correlation function...")
    theta_deg, xi = compute_cmb_correlation_function(
        temperature=temperature,
        mask=mask,
        theta_min_deg=theta_min_deg,
        theta_max_deg=theta_max_deg,
        n_bins=n_bins,
        use_gpu=use_gpu,
    )
    
    # Fit oscillatory template
    logging.info("\nFitting oscillatory template...")
    fit_result = fit_correlation_oscillations(
        theta_deg=theta_deg,
        xi=xi,
        kappa_min_per_deg=kappa_min_per_deg,
        kappa_max_per_deg=kappa_max_per_deg,
        use_envelope_weights=use_envelope_weights,
        use_gpu=use_gpu,
    )
    
    # Save results
    ensure_dir_for_file(output_csv)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_deg", "xi", "xi_fit"])
        
        # Compute fitted curve
        theta_rad = np.deg2rad(theta_deg)
        envelope = theta_rad ** (-(3 - 2 * fit_result["best_beta"]))
        phase_arg = np.deg2rad(fit_result["best_kappa_per_deg"]) * theta_rad
        xi_fit = (
            fit_result["best_offset"] +
            fit_result["best_amplitude"] * envelope *
            np.cos(phase_arg + fit_result["best_phase"])
        )
        
        for th, x, xf in zip(theta_deg, xi, xi_fit):
            writer.writerow([th, x, xf])
    
    logging.info(f"\nResults saved to {output_csv}")
    logging.info("=" * 60)
    logging.info("Correlation function analysis complete!")
    logging.info("=" * 60)

