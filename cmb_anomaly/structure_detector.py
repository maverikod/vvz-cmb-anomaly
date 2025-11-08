"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Structure detector for CMB maps based on oscillatory radial ripples expected
from the quench → tails model. The detector looks for ring-like oscillations
around candidate centers by matching the detrended radial mean profile to
cos(κ r + φ) templates with algebraic weighting. No exponential templates or
mass-like terms are used.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import healpy as hp
import numpy as np
from scipy.stats import t as student_t

from .array_backend import is_gpu_array, cp
from .utils import get_centers, ensure_dir_for_file, log_array_stats


@dataclass
class DetectionResult:
    """Single-center structure detection outcome."""

    center_pix: int
    l_deg: float
    b_deg: float
    best_kappa_per_deg: float
    best_spacing_deg: float
    amplitude: float
    correlation: float
    p_value: float
    num_bins: int
    radius_min_deg: float
    radius_max_deg: float


def compute_radial_profile(
    values: np.ndarray,
    nside: int,
    center_pix: int,
    max_radius_deg: float = 15.0,
    bin_size_deg: float = 0.5,
    valid_mask: Optional[np.ndarray] = None,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute radial mean profile T̄(r) around a center using concentric annuli.

    Args:
        values: Full-sky scalar map (HEALPix ring ordering).
        nside: HEALPix NSIDE.
        center_pix: Center pixel (int).
        max_radius_deg: Max radius in degrees.
        bin_size_deg: Angular bin size in degrees.
        valid_mask: Optional boolean mask of valid pixels; if provided, masked
            pixels are excluded from annuli statistics.

    Returns:
        r_centers_deg: 1D array of radial bin centers (deg).
        mean_profile: 1D array of mean values in annuli.
        counts: 1D array of pixel counts per annulus.
    """
    num_bins = max(1, int(np.floor(max_radius_deg / bin_size_deg)))
    r_edges = np.linspace(0.0, max_radius_deg, num_bins + 1)
    r_centers_deg = 0.5 * (r_edges[:-1] + r_edges[1:])
    center_vec = hp.pix2vec(nside, int(center_pix))
    # Preselect pixels within max_radius to reduce cost
    region_pix = hp.query_disc(
        nside, center_vec, np.deg2rad(max_radius_deg), inclusive=True, fact=4
    )
    region_pix = np.asarray(region_pix, dtype=int)
    if valid_mask is not None:
        # Handle GPU mask: convert region_pix to GPU array if mask is on GPU
        if use_gpu and is_gpu_array(valid_mask):
            region_pix_gpu = cp.asarray(region_pix)
            mask_vals = valid_mask[region_pix_gpu]
            region_pix = region_pix[cp.asnumpy(mask_vals)]
        else:
            region_pix = region_pix[valid_mask[region_pix]]
    if region_pix.size == 0:
        return (
            r_centers_deg,
            np.full_like(r_centers_deg, np.nan),
            np.zeros_like(r_centers_deg, dtype=int),
        )
    # Angular distance of region pixels from center (deg)
    ang_rad = hp.rotator.angdist(center_vec, hp.pix2vec(nside, region_pix))
    ang_deg = np.rad2deg(ang_rad)

    # Use GPU arrays if available and requested
    if use_gpu:
        if not is_gpu_array(values):
            raise RuntimeError(
                f"compute_radial_profile: use_gpu=True but values is not on GPU"
            )
        xp = cp
        # Convert to GPU arrays
        ang_deg = cp.asarray(ang_deg)
        r_edges = cp.asarray(r_edges)
        logging.debug(
            f"compute_radial_profile: Using GPU for center_pix={center_pix}, region_pix.size={region_pix.size}"
        )
    else:
        xp = np

    # Get values - handle GPU indexing
    if use_gpu and is_gpu_array(values):
        # region_pix is CPU array, need to convert for GPU indexing
        region_pix_gpu = cp.asarray(region_pix, dtype=cp.int32)
        vals = values[region_pix_gpu]
    else:
        vals = values[region_pix]

    mean_profile = xp.empty_like(r_centers_deg)
    counts = xp.zeros_like(r_centers_deg, dtype=int)
    # Bin by radius
    bin_indices = xp.digitize(ang_deg, r_edges) - 1
    for bi in range(num_bins):
        in_bin = bin_indices == bi
        if not xp.any(in_bin):
            mean_profile[bi] = xp.nan
            counts[bi] = 0
        else:
            v = vals[in_bin]
            # Robust cleanup of extreme/NaN values inside bin
            v = xp.asarray(v, dtype=xp.float64)
            v = xp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            mean_profile[bi] = float(xp.mean(v))
            counts[bi] = int(v.size)

    # Convert back to CPU if needed
    if use_gpu and is_gpu_array(values):
        mean_profile = cp.asnumpy(mean_profile)
        counts = cp.asnumpy(counts)
        r_centers_deg = cp.asnumpy(r_centers_deg)
    return r_centers_deg, mean_profile, counts


def match_oscillatory_template(
    r_deg: np.ndarray,
    m_profile: np.ndarray,
    counts: np.ndarray,
    kappa_grid_per_deg: np.ndarray,
    min_bins: int = 10,
    power_weight_p: Optional[float] = None,
    use_gpu: bool = True,
) -> Tuple[float, float, float, float, int]:
    """
    Match detrended mean profile to cos(κ r + φ) over a grid of κ and return
    the best-fit κ, amplitude, correlation, and p-value.

    Vectorized version: evaluates all kappa values simultaneously.

    Args:
        r_deg: Radii (deg).
        m_profile: Mean profile values per radius.
        counts: Pixel counts per radius bin.
        kappa_grid_per_deg: Grid of κ in 1/deg.
        min_bins: Minimal number of valid bins required.
        power_weight_p: If provided, weight profile by 1 / r^p (algebraic).

    Returns:
        best_kappa, best_amplitude, best_correlation, p_value, used_bins
    """
    # Use GPU if requested - CUDA is REQUIRED if use_gpu=True
    if use_gpu:
        xp = cp
        # Verify inputs are on GPU
        if not is_gpu_array(r_deg):
            r_deg = cp.asarray(r_deg)
        if not is_gpu_array(m_profile):
            m_profile = cp.asarray(m_profile)
        if not is_gpu_array(counts):
            counts = cp.asarray(counts)
        if not is_gpu_array(kappa_grid_per_deg):
            kappa_grid_per_deg = cp.asarray(kappa_grid_per_deg)

        # Verify all arrays are on GPU
        if (
            not is_gpu_array(r_deg)
            or not is_gpu_array(m_profile)
            or not is_gpu_array(counts)
            or not is_gpu_array(kappa_grid_per_deg)
        ):
            raise RuntimeError(
                "match_oscillatory_template: use_gpu=True but some arrays are not on GPU"
            )

        logging.debug(
            f"match_oscillatory_template: Using GPU, r.size={r_deg.size}, kappa_grid.size={kappa_grid_per_deg.size}"
        )
    else:
        xp = np

    # Valid bins: non-NaN and non-zero counts
    valid = xp.isfinite(m_profile) & (counts > 0)
    r = r_deg[valid]
    y = m_profile[valid]
    if power_weight_p is not None:
        w = 1.0 / xp.maximum(r, 1e-6) ** float(power_weight_p)
    else:
        w = xp.ones_like(r)
    # Detrend: remove weighted mean
    y = y - xp.average(y, weights=w)
    # Normalize by weighted std to obtain correlation-like metric
    y_std = xp.sqrt(xp.average((y**2), weights=w))
    if y.size < min_bins or y_std == 0.0:
        return np.nan, np.nan, np.nan, np.nan, int(y.size)
    y_norm = y / y_std

    # Vectorized evaluation: r.shape = (n_bins,), kappa_grid.shape = (n_kappa,)
    # Create meshgrid: r_mesh.shape = (n_bins, n_kappa), kappa_mesh.shape = (n_bins, n_kappa)
    r_mesh = r[:, xp.newaxis]  # (n_bins, 1)
    kappa_mesh = kappa_grid_per_deg[xp.newaxis, :]  # (1, n_kappa)
    phase_arg = xp.deg2rad(kappa_mesh * r_mesh)  # (n_bins, n_kappa)

    phase_cos = xp.cos(phase_arg)  # (n_bins, n_kappa)
    phase_sin = xp.sin(phase_arg)  # (n_bins, n_kappa)

    # Weighted projections: broadcast y_norm and w
    # y_norm: (n_bins,), w: (n_bins,)
    # phase_cos: (n_bins, n_kappa)
    # Result: (n_kappa,)
    # Manual weighted average: sum(w * x) / sum(w) for each kappa
    y_norm_expanded = y_norm[:, xp.newaxis]  # (n_bins, 1)
    w_expanded = w[:, xp.newaxis]  # (n_bins, 1)
    c_all = xp.sum(w_expanded * y_norm_expanded * phase_cos, axis=0) / xp.sum(
        w_expanded, axis=0
    )
    s_all = xp.sum(w_expanded * y_norm_expanded * phase_sin, axis=0) / xp.sum(
        w_expanded, axis=0
    )
    corr_all = xp.hypot(c_all, s_all)  # (n_kappa,)

    # Find best kappa
    best_idx = int(xp.argmax(corr_all))
    best_corr = float(corr_all[best_idx])
    best_kappa = float(kappa_grid_per_deg[best_idx])

    # Reconstruct amplitude for best kappa
    phase_cos_best = phase_cos[:, best_idx]  # (n_bins,)
    phase_sin_best = phase_sin[:, best_idx]  # (n_bins,)
    denom = xp.average(phase_cos_best**2 + phase_sin_best**2, weights=w)
    if denom <= 0:
        best_amp = 0.0
    else:
        a_c = xp.average(y * phase_cos_best, weights=w) / denom
        a_s = xp.average(y * phase_sin_best, weights=w) / denom
        best_amp = float(xp.hypot(a_c, a_s))

    # Convert best_corr to p-value
    n = int(y.size)
    r_clip = min(max(best_corr, -0.999999), 0.999999)
    t_stat = r_clip * np.sqrt((n - 2) / (1.0 - r_clip**2)) if n > 2 else 0.0
    p_val = 2.0 * (1.0 - student_t.cdf(abs(t_stat), df=max(n - 2, 1))) if n > 2 else 1.0
    return best_kappa, best_amp, best_corr, float(p_val), n


def _get_gpu_window_size(
    use_gpu: bool, memory_fraction: float = 0.8, max_radius_deg: float = 15.0
) -> int:
    """
    Calculate window size based on available GPU memory.

    Args:
        use_gpu: Whether GPU is available.
        memory_fraction: Fraction of GPU memory to use (default: 0.8 = 80%).
        max_radius_deg: Maximum radius in degrees (affects memory per center).

    Returns:
        Maximum number of centers that can fit in GPU memory window.
    """
    import logging

    if not use_gpu:
        return 1000  # Default for CPU

    try:
        from .array_backend import cp

        mempool = cp.get_default_memory_pool()
        meminfo = cp.cuda.runtime.memGetInfo()
        free_mem = meminfo[0]  # Free memory in bytes
        total_mem = meminfo[1]  # Total memory in bytes

        # For small GPUs (< 4 GB), use more conservative memory fraction
        if total_mem < 4 * 1024**3:  # Less than 4 GB
            memory_fraction = min(
                memory_fraction, 0.5
            )  # Use at most 50% for small GPUs
            logging.info(
                f"  Small GPU detected ({total_mem / (1024**3):.2f} GB), using conservative memory fraction: {memory_fraction:.1%}"
            )

        # Use memory_fraction of free memory
        available_mem = int(free_mem * memory_fraction)

        # Estimate memory per center based on radius:
        # For large radius (60deg), region_pix can be very large (millions of pixels)
        # - Region pixels: ~(max_radius_deg * 60)^2 pixels * 4 bytes = ~(max_radius_deg^2 * 14400) bytes
        # - Radial profile bins: ~(max_radius_deg / bin_size) * 8 bytes
        # - Kappa grid operations: ~200 kappa * bins * 8 bytes
        # - Intermediate arrays: ~500 KB per center (conservative)
        # For 60deg: region can be ~3M pixels = 12 MB just for indices
        region_size_estimate = int((max_radius_deg * 60) ** 2 * 4)  # bytes
        bins_estimate = int(max_radius_deg / 0.5)  # conservative bin estimate
        kappa_ops = 200 * bins_estimate * 8  # bytes
        intermediate = 500 * 1024  # bytes

        # Total: region + operations + intermediate
        bytes_per_center = region_size_estimate + kappa_ops + intermediate

        # Add 100% safety margin for small GPUs (more conservative)
        safety_margin = 2.0 if total_mem < 4 * 1024**3 else 1.5
        bytes_per_center = int(bytes_per_center * safety_margin)

        window_size = max(1, available_mem // bytes_per_center)

        # Cap window size to prevent OOM (especially for large radius and small GPUs)
        if total_mem < 4 * 1024**3:
            # Very conservative for small GPUs
            # For large radius, process one center at a time to avoid OOM
            max_window_size = 1 if max_radius_deg > 30 else 5
        elif max_radius_deg > 30:
            max_window_size = 50
        else:
            max_window_size = 2000
        window_size = min(window_size, max_window_size)

        logging.info(
            f"  GPU memory: {free_mem / 1024**3:.2f} GB free / {total_mem / 1024**3:.2f} GB total, "
            f"estimated {bytes_per_center / 1024**2:.1f} MB per center, "
            f"window size: {window_size} centers"
        )
        return window_size
    except Exception as e:
        import logging

        logging.warning(
            f"  Could not determine GPU memory, using default window size: {e}"
        )
        return 100 if max_radius_deg > 30 else 500


def _process_center_window_vectorized(
    center_window: List[int],
    temperature: np.ndarray,
    mask: Optional[np.ndarray],
    nside: int,
    max_radius_deg: float,
    bin_size_deg: float,
    kappa_grid: np.ndarray,
    min_bins: int,
    power_weight_p: Optional[float],
    p_value_threshold: float,
    use_gpu: bool = True,
) -> List[DetectionResult]:
    """
    Process a window of centers with vectorized GPU operations.
    All centers in the window are processed simultaneously on GPU.

    Args:
        center_window: List of center pixel indices to process.
        temperature: Full-sky temperature map (can be GPU or CPU array).
        mask: Optional mask (can be GPU or CPU array).
        nside: HEALPix NSIDE.
        max_radius_deg: Max radius for radial profile.
        bin_size_deg: Bin size for radial profile.
        kappa_grid: Grid of κ values to test.
        min_bins: Minimum number of bins required.
        power_weight_p: Power-law weighting exponent.
        p_value_threshold: P-value threshold for reporting.
        use_gpu: Whether to use GPU for computations.

    Returns:
        List of DetectionResult for significant detections in this window.
    """
    # For very small windows or if GPU is not available, use sequential processing
    # Vectorized processing requires enough memory for all centers at once
    if not use_gpu or len(center_window) == 0 or len(center_window) <= 1:
        # Fallback to sequential processing
        return _process_center_block(
            center_window,
            temperature,
            mask,
            nside,
            max_radius_deg,
            bin_size_deg,
            kappa_grid,
            min_bins,
            power_weight_p,
            p_value_threshold,
            use_gpu,
        )

    # CUDA is REQUIRED for vectorized processing
    import logging
    import os

    # Set CUDA device for this process (important for multiprocessing with spawn)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from .array_backend import cp

    logging.debug(
        f"_process_center_window_vectorized: Initializing CUDA for {len(center_window)} centers"
    )
    try:
        # Initialize CUDA runtime in this process
        # For spawn method, each process needs to initialize CUDA separately
        cp.cuda.runtime.setDevice(0)
        device = cp.cuda.Device(0)
        device.use()
        logging.debug("  ✓ CUDA device 0 initialized in worker process")
    except Exception as e:
        error_msg = f"CUDA is REQUIRED for vectorized processing but not available: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Convert to GPU if needed - CUDA is REQUIRED
    try:
        if not is_gpu_array(temperature):
            temperature = cp.asarray(temperature)
            logging.debug(f"  ✓ Temperature loaded to GPU: {temperature.shape}")
        if mask is not None and not is_gpu_array(mask):
            mask = cp.asarray(mask)
            logging.debug(f"  ✓ Mask loaded to GPU: {mask.shape}")
        if not is_gpu_array(kappa_grid):
            kappa_grid = cp.asarray(kappa_grid)
            logging.debug(f"  ✓ Kappa grid loaded to GPU: {kappa_grid.shape}")

        # Verify all arrays are on GPU
        if not is_gpu_array(temperature):
            raise RuntimeError("Temperature not on GPU after conversion")
        if mask is not None and not is_gpu_array(mask):
            raise RuntimeError("Mask not on GPU after conversion")
        if not is_gpu_array(kappa_grid):
            raise RuntimeError("Kappa grid not on GPU after conversion")
    except (RuntimeError, cp.cuda.memory.OutOfMemoryError) as e:
        # If OOM or other GPU error, fall back to sequential processing
        error_msg = f"GPU memory error in vectorized processing, falling back to sequential: {e}"
        logging.warning(error_msg)
        return _process_center_block(
            center_window,
            temperature,
            mask,
            nside,
            max_radius_deg,
            bin_size_deg,
            kappa_grid,
            min_bins,
            power_weight_p,
            p_value_threshold,
            use_gpu,
        )
    except Exception as e:
        error_msg = f"Failed to load arrays to GPU for vectorized processing: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    results: List[DetectionResult] = []

    # Process centers sequentially with GPU acceleration
    # For large radius, processing all centers at once would cause OOM
    # Process one at a time and free memory between centers
    import logging

    n_detections = 0
    n_processed = 0
    for i, center_pix in enumerate(center_window):
        # Free GPU memory periodically to prevent accumulation
        if use_gpu and i > 0 and i % 10 == 0:
            try:
                from .array_backend import cp

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            except Exception:
                pass

        # Log progress every 50 centers
        if (i + 1) % 50 == 0:
            logging.info(
                f"  Window progress: {i + 1}/{len(center_window)} centers processed, "
                f"{n_detections} detections found"
            )

        center_pix = int(center_pix)
        if mask is not None:
            mask_val = mask[center_pix]
            if is_gpu_array(mask_val):
                mask_val = bool(cp.asnumpy(mask_val))
            if not mask_val:
                continue

        r_deg, mean_prof, counts = compute_radial_profile(
            temperature,
            nside,
            center_pix,
            max_radius_deg,
            bin_size_deg,
            mask,
            use_gpu=True,
        )

        if r_deg.size < min_bins:
            continue

        best_kappa, best_amp, best_corr, p_val, used_bins = match_oscillatory_template(
            r_deg,
            mean_prof,
            counts,
            kappa_grid,
            min_bins=min_bins,
            power_weight_p=power_weight_p,
            use_gpu=True,
        )

        if not np.isfinite(p_val) or p_val > p_value_threshold:
            continue

        theta, phi = hp.pix2ang(nside, center_pix)
        l_deg = float(np.rad2deg(phi))
        b_deg = float(90.0 - np.rad2deg(theta))
        detection = DetectionResult(
            center_pix=center_pix,
            l_deg=l_deg,
            b_deg=b_deg,
            best_kappa_per_deg=best_kappa,
            best_spacing_deg=(
                float(np.pi / best_kappa) if best_kappa > 0 else np.inf
            ),
            amplitude=best_amp,
            correlation=best_corr,
            p_value=p_val,
            num_bins=used_bins,
            radius_min_deg=float(r_deg[0]) if r_deg.size > 0 else 0.0,
            radius_max_deg=float(r_deg[-1]) if r_deg.size > 0 else 0.0,
        )
        results.append(detection)
        
        # Write immediately to file if callback provided
        if write_callback is not None:
            try:
                write_callback(detection)
            except Exception as e:
                logging.warning(f"Failed to write detection to file: {e}")

    return results


def _process_center_block(
    center_block: List[int],
    temperature: np.ndarray,
    mask: Optional[np.ndarray],
    nside: int,
    max_radius_deg: float,
    bin_size_deg: float,
    kappa_grid: np.ndarray,
    min_bins: int,
    power_weight_p: Optional[float],
    p_value_threshold: float,
    use_gpu: bool = True,
    write_callback: Optional[callable] = None,
) -> List[DetectionResult]:
    """
    Process a block of centers in parallel. This function is called by worker processes.
    Can use GPU if available and use_gpu=True.

    Args:
        center_block: List of center pixel indices to process.
        temperature: Full-sky temperature map (can be GPU or CPU array).
        mask: Optional mask (can be GPU or CPU array).
        nside: HEALPix NSIDE.
        max_radius_deg: Max radius for radial profile.
        bin_size_deg: Bin size for radial profile.
        kappa_grid: Grid of κ values to test.
        min_bins: Minimum number of bins required.
        power_weight_p: Power-law weighting exponent.
        p_value_threshold: P-value threshold for reporting.
        use_gpu: Whether to use GPU for computations.

    Returns:
        List of DetectionResult for significant detections in this block.
    """
    # Initialize CUDA in this worker process if using GPU - CUDA is REQUIRED
    if use_gpu:
        import logging
        import os

        # Set CUDA device for this process (important for multiprocessing with spawn)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

        from .array_backend import cp

        logging.debug(
            f"Worker process: Initializing CUDA (window size: {len(center_block)} centers)"
        )
        try:
            # Initialize CUDA runtime in this process
            # For spawn method, each process needs to initialize CUDA separately
            cp.cuda.runtime.setDevice(0)
            device = cp.cuda.Device(0)
            device.use()
            logging.debug("Worker process: ✓ CUDA device 0 initialized")
        except Exception as e:
            error_msg = f"Worker process: CUDA is REQUIRED but not available: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Check GPU memory and decide if we can use GPU
        # For small GPUs with large radius, use CPU to avoid OOM
        try:
            meminfo = cp.cuda.runtime.memGetInfo()
            total_mem = meminfo[1]  # Total memory in bytes
            free_mem = meminfo[0]  # Free memory in bytes
            
            # For small GPUs (< 4 GB) with large radius (> 30 deg), use CPU
            # Large radius creates huge region_pix arrays that cause OOM
            if total_mem < 4 * 1024**3 and max_radius_deg > 30:
                logging.warning(
                    f"Worker process: Small GPU ({total_mem / (1024**3):.2f} GB) with large radius ({max_radius_deg}°), "
                    f"switching to CPU to avoid OOM"
                )
                use_gpu = False
                # Convert GPU arrays to CPU if needed
                if is_gpu_array(temperature):
                    temperature = cp.asnumpy(temperature)
                if mask is not None and is_gpu_array(mask):
                    mask = cp.asnumpy(mask)
                if is_gpu_array(kappa_grid):
                    kappa_grid = cp.asnumpy(kappa_grid)
            else:
                # Convert to GPU if not already - CUDA is REQUIRED
                if not is_gpu_array(temperature):
                    temperature = cp.asarray(temperature)
                    logging.debug(
                        f"Worker process: ✓ Temperature loaded to GPU: {temperature.shape}"
                    )
                if mask is not None and not is_gpu_array(mask):
                    mask = cp.asarray(mask)
                    logging.debug(f"Worker process: ✓ Mask loaded to GPU: {mask.shape}")
                if not is_gpu_array(kappa_grid):
                    kappa_grid = cp.asarray(kappa_grid)
                    logging.debug(
                        f"Worker process: ✓ Kappa grid loaded to GPU: {kappa_grid.shape}"
                    )

                # Verify all arrays are on GPU
                if not is_gpu_array(temperature):
                    raise RuntimeError("Temperature not on GPU after conversion")
                if mask is not None and not is_gpu_array(mask):
                    raise RuntimeError("Mask not on GPU after conversion")
                if not is_gpu_array(kappa_grid):
                    raise RuntimeError("Kappa grid not on GPU after conversion")
        except cp.cuda.memory.OutOfMemoryError as e:
            # If OOM during setup, fall back to CPU
            logging.warning(
                f"Worker process: GPU OOM during setup, switching to CPU: {e}"
            )
            use_gpu = False
            # Convert GPU arrays to CPU if needed
            if is_gpu_array(temperature):
                temperature = cp.asnumpy(temperature)
            if mask is not None and is_gpu_array(mask):
                mask = cp.asnumpy(mask)
            if is_gpu_array(kappa_grid):
                kappa_grid = cp.asnumpy(kappa_grid)
        except Exception as e:
            error_msg = f"Worker process: Failed to load arrays to GPU: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    else:
        # Ensure CPU arrays
        if is_gpu_array(temperature):
            temperature = cp.asnumpy(temperature)
        if mask is not None and is_gpu_array(mask):
            mask = cp.asnumpy(mask)
        if is_gpu_array(kappa_grid):
            kappa_grid = cp.asnumpy(kappa_grid)

    results: List[DetectionResult] = []
    n_processed = 0
    n_detections = 0
    for center_pix in center_block:
        center_pix = int(center_pix)
        if mask is not None:
            mask_val = mask[center_pix]
            if is_gpu_array(mask_val):
                mask_val = bool(cp.asnumpy(mask_val))
            if not mask_val:
                continue

        # Try GPU first, fall back to CPU on OOM
        try:
            r_deg, mean_prof, counts = compute_radial_profile(
                temperature,
                nside,
                center_pix,
                max_radius_deg,
                bin_size_deg,
                mask,
                use_gpu=use_gpu,
            )
        except Exception as e:
            # Check if it's an OOM error (CuPy or other)
            is_oom = False
            if use_gpu:
                try:
                    from .array_backend import cp
                    if isinstance(e, cp.cuda.memory.OutOfMemoryError):
                        is_oom = True
                except (ImportError, AttributeError):
                    pass
                # Also check error message for OOM
                if "out of memory" in str(e).lower() or "OutOfMemoryError" in str(type(e)):
                    is_oom = True
            
            if is_oom:
                # OOM on GPU, switch to CPU for this center
                logging.warning(
                    f"Worker process: GPU OOM for center {center_pix}, switching to CPU"
                )
                # Convert to CPU arrays
                try:
                    from .array_backend import cp
                    temp_cpu = (
                        cp.asnumpy(temperature) if is_gpu_array(temperature) else temperature
                    )
                    mask_cpu = (
                        cp.asnumpy(mask) if mask is not None and is_gpu_array(mask) else mask
                    )
                    # Free GPU memory
                    try:
                        mempool = cp.get_default_memory_pool()
                        mempool.free_all_blocks()
                    except Exception:
                        pass
                except (ImportError, AttributeError):
                    # If cp not available, arrays are already CPU
                    temp_cpu = temperature
                    mask_cpu = mask
                # Retry with CPU
                r_deg, mean_prof, counts = compute_radial_profile(
                    temp_cpu,
                    nside,
                    center_pix,
                    max_radius_deg,
                    bin_size_deg,
                    mask_cpu,
                    use_gpu=False,
                )
            else:
                raise

        n_processed += 1
        
        if r_deg.size < min_bins:
            continue

        best_kappa, best_amp, best_corr, p_val, used_bins = match_oscillatory_template(
            r_deg,
            mean_prof,
            counts,
            kappa_grid,
            min_bins=min_bins,
            power_weight_p=power_weight_p,
            use_gpu=use_gpu,
        )

        if not np.isfinite(p_val) or p_val > p_value_threshold:
            continue

        n_detections += 1
        # Log significant detection with detailed info
        theta, phi = hp.pix2ang(nside, center_pix)
        l_deg = float(np.rad2deg(phi))
        b_deg = float(90.0 - np.rad2deg(theta))
        spacing_deg = float(np.pi / best_kappa) if best_kappa > 0 else np.inf
        logging.info(
            f"  ✓ [{n_detections}] Detection found at (l={l_deg:.2f}°, b={b_deg:.2f}°): "
            f"κ={best_kappa:.4f}/deg, spacing={spacing_deg:.2f}°, "
            f"amplitude={best_amp:.4f}, corr={best_corr:.4f}, p={p_val:.2e}, bins={used_bins}"
        )
        detection = DetectionResult(
            center_pix=center_pix,
            l_deg=l_deg,
            b_deg=b_deg,
            best_kappa_per_deg=best_kappa,
            best_spacing_deg=(
                float(np.pi / best_kappa) if best_kappa > 0 else np.inf
            ),
            amplitude=best_amp,
            correlation=best_corr,
            p_value=p_val,
            num_bins=used_bins,
            radius_min_deg=float(r_deg[0]) if r_deg.size > 0 else 0.0,
            radius_max_deg=float(r_deg[-1]) if r_deg.size > 0 else 0.0,
        )
        results.append(detection)
        
        # Write immediately to file if callback provided
        if write_callback is not None:
            try:
                write_callback(detection)
            except Exception as e:
                logging.warning(f"Failed to write detection to file: {e}")

    logging.info(
        f"  Window complete: {n_processed} centers processed, {n_detections} detections found"
    )
    return results


def run_structure_detection(
    temperature: np.ndarray,
    mask: Optional[np.ndarray] = None,
    step_deg: int = 5,
    max_radius_deg: float = 15.0,
    bin_size_deg: float = 0.5,
    min_bins: int = 10,
    kappa_min_per_deg: float = 0.1,
    kappa_max_per_deg: float = 3.0,
    kappa_num: int = 200,
    power_weight_p: Optional[float] = 1.0,
    p_value_threshold: float = 1e-3,
    n_jobs: Optional[int] = None,
    block_size: int = 500,
    use_gpu: bool = True,
    centers_block: Optional[List[int]] = None,
    write_callback: Optional[callable] = None,
) -> List[DetectionResult]:
    """
    Detect ring-like oscillatory structures over a sky grid.

    Note: For GPU processing with multiprocessing, the start method is automatically
    set to 'spawn' to ensure CUDA compatibility in worker processes.

    Args:
        temperature: Full-sky CMB temperature map (HEALPix ring).
        mask: Optional boolean mask of valid pixels (same shape as map).
        step_deg: Approximate step for center grid generation.
        max_radius_deg: Max radius for radial profile (deg).
        bin_size_deg: Angular bin width (deg).
        min_bins: Minimal number of valid annuli for detection.
        kappa_min_per_deg: Minimal κ (1/deg) to scan.
        kappa_max_per_deg: Max κ (1/deg) to scan.
        kappa_num: Number of κ grid points.
        power_weight_p: Optional algebraic weighting 1/r^p; None to disable.
        p_value_threshold: Significance threshold for reporting.

    Returns:
        List of DetectionResult for significant detections.
    """
    # Don't convert GPU arrays with np.asarray - it will fail
    if not is_gpu_array(temperature):
        temperature = np.asarray(temperature, dtype=np.float64)
    if mask is not None:
        if not is_gpu_array(mask):
            mask = np.asarray(mask, dtype=bool)
        if mask.shape != temperature.shape:
            raise ValueError("mask shape must match temperature shape")
    nside = hp.npix2nside(temperature.size)
    
    # Use provided centers_block if specified, otherwise generate all centers
    if centers_block is not None:
        centers = centers_block
        logging.info(f"  Using provided centers block: {len(centers)} centers")
    else:
        centers = get_centers(nside, step_deg)
        # Filter centers by mask upfront
        if mask is not None:
            if is_gpu_array(mask):
                # Use global cp import - convert GPU array to CPU for filtering
                mask_cpu = cp.asnumpy(mask)
            else:
                mask_cpu = mask
            centers = [c for c in centers if mask_cpu[int(c)]]
    
    kappa_grid = np.linspace(kappa_min_per_deg, kappa_max_per_deg, int(kappa_num))

    logging.info("Structure detection parameters:")
    logging.info(f"  NSIDE: {nside}, total pixels: {temperature.size}")
    logging.info(f"  Centers to scan: {len(centers)}")
    logging.info(f"  Max radius: {max_radius_deg} deg, bin size: {bin_size_deg} deg")
    logging.info(
        f"  κ range: [{kappa_min_per_deg:.3f}, {kappa_max_per_deg:.3f}] per deg ({kappa_num} points)"
    )
    logging.info(f"  P-value threshold: {p_value_threshold}")
    logging.info(f"  Min bins: {min_bins}")
    logging.info(
        f"  Parallel processing: {n_jobs if n_jobs else 'auto'} workers, block size: {block_size}"
    )
    logging.info(f"  GPU acceleration: {use_gpu}")

    # Load to GPU if requested - CUDA is REQUIRED if use_gpu=True
    # Note: cp is already imported at module level (line 24)
    if use_gpu:
        # Step 1: Check CUDA availability
        logging.info("  Checking CUDA availability...")
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                raise RuntimeError("No CUDA devices found")
            device_props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = (
                device_props.get("name", "Unknown").decode()
                if isinstance(device_props.get("name"), bytes)
                else device_props.get("name", "Unknown")
            )
            logging.info(
                f"  ✓ CUDA available: {device_count} GPU(s), device 0: {gpu_name}"
            )
        except Exception as e:
            error_msg = f"CUDA is REQUIRED but not available: {e}"
            logging.error(f"  ✗ {error_msg}")
            raise RuntimeError(error_msg) from e

        # Step 2: Check GPU memory
        try:
            meminfo = cp.cuda.runtime.memGetInfo()
            free_mem_gb = meminfo[0] / (1024**3)
            total_mem_gb = meminfo[1] / (1024**3)
            logging.info(
                f"  GPU memory: {free_mem_gb:.2f} GB free / {total_mem_gb:.2f} GB total"
            )
        except Exception as e:
            logging.warning(f"  Could not query GPU memory: {e}")

        # Step 3: Load temperature to GPU
        logging.info("  Loading temperature to GPU...")
        try:
            if not is_gpu_array(temperature):
                temperature = cp.asarray(temperature)
                logging.info(
                    f"  ✓ Temperature loaded to GPU: shape={temperature.shape}, dtype={temperature.dtype}"
                )
            else:
                logging.info(
                    f"  ✓ Temperature already on GPU: shape={temperature.shape}, dtype={temperature.dtype}"
                )
            # Verify it's actually on GPU
            if not is_gpu_array(temperature):
                raise RuntimeError("Temperature array is not on GPU after loading")
        except Exception as e:
            error_msg = f"Failed to load temperature to GPU: {e}"
            logging.error(f"  ✗ {error_msg}")
            raise RuntimeError(error_msg) from e

        # Step 4: Load mask to GPU
        if mask is not None:
            logging.info("  Loading mask to GPU...")
            try:
                if not is_gpu_array(mask):
                    mask = cp.asarray(mask)
                    logging.info(
                        f"  ✓ Mask loaded to GPU: shape={mask.shape}, dtype={mask.dtype}"
                    )
                else:
                    logging.info(
                        f"  ✓ Mask already on GPU: shape={mask.shape}, dtype={mask.dtype}"
                    )
                # Verify it's actually on GPU
                if not is_gpu_array(mask):
                    raise RuntimeError("Mask array is not on GPU after loading")
            except Exception as e:
                error_msg = f"Failed to load mask to GPU: {e}"
                logging.error(f"  ✗ {error_msg}")
                raise RuntimeError(error_msg) from e

        # Step 5: Load kappa_grid to GPU
        logging.info("  Loading kappa_grid to GPU...")
        try:
            if not is_gpu_array(kappa_grid):
                kappa_grid = cp.asarray(kappa_grid)
                logging.info(
                    f"  ✓ Kappa grid loaded to GPU: shape={kappa_grid.shape}, dtype={kappa_grid.dtype}"
                )
            else:
                logging.info(
                    f"  ✓ Kappa grid already on GPU: shape={kappa_grid.shape}, dtype={kappa_grid.dtype}"
                )
            # Verify it's actually on GPU
            if not is_gpu_array(kappa_grid):
                raise RuntimeError("Kappa grid array is not on GPU after loading")
        except Exception as e:
            error_msg = f"Failed to load kappa_grid to GPU: {e}"
            logging.error(f"  ✗ {error_msg}")
            raise RuntimeError(error_msg) from e

        logging.info("  ✓ All arrays successfully loaded to GPU")
    else:
        # Ensure CPU arrays
        if is_gpu_array(temperature):
            temperature = cp.asnumpy(temperature)
        if mask is not None and is_gpu_array(mask):
            mask = cp.asnumpy(mask)
        if is_gpu_array(kappa_grid):
            kappa_grid = cp.asnumpy(kappa_grid)

    # Determine number of workers first (needed to decide if we use GPU arrays directly)
    if n_jobs is None:
        if use_gpu:
            # For GPU, use 2-3 workers (can work with proper memory management)
            # Each worker process needs its own CUDA context and GPU memory
            # Limit to 3 to avoid memory conflicts on small GPUs
            n_jobs = min(3, max(2, mp.cpu_count() // 2))
            logging.info(f"  GPU mode: using {n_jobs} workers (limited to avoid memory conflicts)")
        else:
            n_jobs = max(1, mp.cpu_count() - 1)  # Leave one core free

    # Calculate window size based on GPU memory if using GPU
    if use_gpu:
        window_size = _get_gpu_window_size(
            use_gpu, memory_fraction=0.8, max_radius_deg=max_radius_deg
        )
        window_step = max(1, window_size // 2)  # Step is half of window size
        logging.info(f"  GPU window size: {window_size} centers, step: {window_step}")
    else:
        window_size = block_size
        window_step = block_size

    # Create sliding windows with overlap
    center_windows = []
    for i in range(0, len(centers), window_step):
        window = centers[i : i + window_size]
        if len(window) > 0:
            center_windows.append(window)

    logging.info(
        f"  Split into {len(center_windows)} windows (size: {window_size}, step: {window_step})"
    )

    n_jobs = min(n_jobs, len(center_windows))  # Don't use more workers than windows
    
    # For multiprocessing, we need CPU arrays (CuPy arrays can't be pickled)
    # But each worker will convert to GPU if use_gpu=True
    # For sequential processing with GPU, use original GPU arrays directly
    if n_jobs == 1 and use_gpu:
        # Sequential GPU processing: use original GPU arrays
        temp_for_workers = temperature
        mask_for_workers = mask
        kappa_grid_for_workers = kappa_grid
    else:
        # Multiprocessing: convert to CPU for pickling
        if is_gpu_array(temperature):
            temp_for_workers = cp.asnumpy(temperature)
        else:
            temp_for_workers = temperature
        if mask is not None:
            if is_gpu_array(mask):
                mask_for_workers = cp.asnumpy(mask)
            else:
                mask_for_workers = mask
        else:
            mask_for_workers = None
        if is_gpu_array(kappa_grid):
            kappa_grid_for_workers = cp.asnumpy(kappa_grid)
        else:
            kappa_grid_for_workers = kappa_grid
    
    # Additional safety: if GPU and too many workers requested, warn and limit
    if use_gpu and n_jobs > 3:
        logging.warning(
            f"  GPU mode with {n_jobs} workers may cause memory issues. "
            f"Limiting to 3 workers. Use --n-jobs to override if needed."
        )
        n_jobs = 3

    results: List[DetectionResult] = []

    if n_jobs > 1 and len(center_windows) > 1:
        # Parallel processing
        logging.info(f"  Using {n_jobs} parallel workers")
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            tqdm = None
            use_tqdm = False

        # Create partial function with fixed arguments
        # Use vectorized window processing if GPU is available
        if use_gpu:
            process_func = partial(
                _process_center_window_vectorized,
                temperature=temp_for_workers,
                mask=mask_for_workers,
                nside=nside,
                max_radius_deg=max_radius_deg,
                bin_size_deg=bin_size_deg,
                kappa_grid=kappa_grid_for_workers,
                min_bins=min_bins,
                power_weight_p=power_weight_p,
                p_value_threshold=p_value_threshold,
                use_gpu=use_gpu,
            )
        else:
            process_func = partial(
                _process_center_block,
                temperature=temp_for_workers,
                mask=mask_for_workers,
                nside=nside,
                max_radius_deg=max_radius_deg,
                bin_size_deg=bin_size_deg,
                kappa_grid=kappa_grid_for_workers,
                min_bins=min_bins,
                power_weight_p=power_weight_p,
                p_value_threshold=p_value_threshold,
                use_gpu=use_gpu,
                write_callback=write_callback,
            )

        # Process windows in parallel
        # Use 'spawn' method for CUDA compatibility (each process gets its own CUDA context)
        if use_gpu:
            try:
                mp_start_method = mp.get_start_method(allow_none=True)
                if mp_start_method != "spawn":
                    logging.info(
                        f"  Setting multiprocessing start method to 'spawn' for CUDA compatibility"
                    )
                    mp.set_start_method("spawn", force=True)
            except RuntimeError:
                # Already set, ignore
                pass

        with mp.Pool(processes=n_jobs) as pool:
            if use_tqdm:
                window_results = list(
                    tqdm(
                        pool.imap(process_func, center_windows),
                        total=len(center_windows),
                        desc="Processing windows",
                    )
                )
            else:
                window_results = pool.map(process_func, center_windows)
                # Log progress periodically
                for i, _ in enumerate(window_results):
                    if (i + 1) % max(1, len(center_windows) // 10) == 0:
                        logging.info(
                            f"  Progress: {i + 1}/{len(center_windows)} windows processed"
                        )

        # Flatten results (remove duplicates from overlapping windows)
        seen_centers = set()
        total_detections = 0
        for window_idx, window_res in enumerate(window_results):
            window_detections = 0
            for res in window_res:
                if res.center_pix not in seen_centers:
                    results.append(res)
                    seen_centers.add(res.center_pix)
                    window_detections += 1
                    total_detections += 1

            # Log progress every 10 windows
            if (window_idx + 1) % 10 == 0 or window_idx == len(window_results) - 1:
                logging.info(
                    f"  Progress: {window_idx + 1}/{len(window_results)} windows processed, "
                    f"{total_detections} unique detections found so far"
                )
    else:
        # Sequential processing (fallback or single worker)
        logging.info("  Using sequential processing")
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            tqdm = None
            use_tqdm = False

        center_iter = (
            tqdm(centers, desc="Scanning centers", disable=not use_tqdm)
            if use_tqdm
            else centers
        )
        n_processed = 0

        for center_pix in center_iter:
            center_pix = int(center_pix)
            r_deg, mean_prof, counts = compute_radial_profile(
                temp_for_workers,
                nside,
                center_pix,
                max_radius_deg,
                bin_size_deg,
                mask_for_workers,
                use_gpu=use_gpu,
            )

            if r_deg.size < min_bins:
                continue

            best_kappa, best_amp, best_corr, p_val, used_bins = (
                match_oscillatory_template(
                    r_deg,
                    mean_prof,
                    counts,
                    kappa_grid_for_workers,
                    min_bins=min_bins,
                    power_weight_p=power_weight_p,
                    use_gpu=use_gpu,
                )
            )

            n_processed += 1

            if not np.isfinite(p_val) or p_val > p_value_threshold:
                continue

            # Log significant detection with detailed info
            theta, phi = hp.pix2ang(nside, center_pix)
            l_deg = float(np.rad2deg(phi))
            b_deg = float(90.0 - np.rad2deg(theta))
            spacing_deg = float(np.pi / best_kappa) if best_kappa > 0 else np.inf
            detection_num = len(results) + 1
            logging.info(
                f"  ✓ [{detection_num}] Detection found at (l={l_deg:.2f}°, b={b_deg:.2f}°): "
                f"κ={best_kappa:.4f}/deg, spacing={spacing_deg:.2f}°, "
                f"amplitude={best_amp:.4f}, corr={best_corr:.4f}, p={p_val:.2e}, bins={used_bins}"
            )
            results.append(
                DetectionResult(
                    center_pix=center_pix,
                    l_deg=l_deg,
                    b_deg=b_deg,
                    best_kappa_per_deg=best_kappa,
                    best_spacing_deg=(
                        float(np.pi / best_kappa) if best_kappa > 0 else np.inf
                    ),
                    amplitude=best_amp,
                    correlation=best_corr,
                    p_value=p_val,
                    num_bins=used_bins,
                    radius_min_deg=float(r_deg[0]) if r_deg.size > 0 else 0.0,
                    radius_max_deg=float(r_deg[-1]) if r_deg.size > 0 else 0.0,
                )
            )

            # Periodic status update
            if n_processed % 100 == 0:
                logging.info(
                    f"  Progress: {n_processed} centers processed, {len(results)} detections found so far"
                )
                if len(results) > 0:
                    recent_kappa = [r.best_kappa_per_deg for r in results[-10:]]
                    logging.info(
                        f"    Recent κ range: {min(recent_kappa):.4f} - {max(recent_kappa):.4f} per deg"
                    )

    logging.info("\n" + "=" * 60)
    logging.info("Structure detection complete:")
    logging.info(f"  Total centers processed: {len(centers)}")
    logging.info(f"  Total detections found: {len(results)}")
    logging.info("=" * 60)

    if len(results) > 0:
        kappa_vals = [r.best_kappa_per_deg for r in results]
        spacing_vals = [r.best_spacing_deg for r in results]
        logging.info("\nDetection statistics:")
        logging.info(
            f"  κ (per deg): min={min(kappa_vals):.3f}, max={max(kappa_vals):.3f}, "
            f"mean={np.mean(kappa_vals):.3f}"
        )
        logging.info(
            f"  Spacing (deg): min={min(spacing_vals):.3f}, max={max(spacing_vals):.3f}, "
            f"mean={np.mean(spacing_vals):.3f}"
        )
        logging.info(
            f"  Correlation: min={min(r.correlation for r in results):.3f}, "
            f"max={max(r.correlation for r in results):.3f}, "
            f"mean={np.mean([r.correlation for r in results]):.3f}"
        )

    return results


def count_blocks(
    temperature_path: str,
    mask_path: Optional[str],
    step_deg: int = 5,
    block_size: int = 500,
) -> int:
    """
    Count the number of blocks for block-based processing.
    
    Args:
        temperature_path: Path to .npy file with temperature map.
        mask_path: Optional path to .npy file with mask.
        step_deg: Step in degrees for center grid generation.
        block_size: Number of centers per block.
    
    Returns:
        Total number of blocks.
    """
    from .array_backend import array_load
    import healpy as hp
    
    temperature = array_load(temperature_path)
    nside = hp.npix2nside(temperature.size)
    centers = get_centers(nside, step_deg)
    
    if mask_path:
        mask = array_load(mask_path)
        centers = [c for c in centers if mask[int(c)]]
    
    total_blocks = (len(centers) + block_size - 1) // block_size
    return total_blocks


def run_structure_detection_from_paths(
    temperature_path: str,
    mask_path: Optional[str],
    results_csv_path: str,
    step_deg: int = 5,
    max_radius_deg: float = 15.0,
    bin_size_deg: float = 0.5,
    p_value_threshold: float = 1e-3,
    n_jobs: Optional[int] = None,
    block_size: int = 500,
    use_gpu: bool = True,
    kappa_min_per_deg: float = 0.1,
    kappa_max_per_deg: float = 3.0,
    block_number: Optional[int] = None,
    count_blocks_only: bool = False,
) -> None:
    """
    Convenience wrapper: load arrays and write detections to CSV.
    
    Args:
        temperature_path: Path to .npy file with temperature map.
        mask_path: Optional path to .npy file with mask.
        results_csv_path: Path to output CSV file (will append if exists).
        step_deg: Step in degrees for center grid.
        max_radius_deg: Max radius for radial profile.
        bin_size_deg: Radial bin size.
        p_value_threshold: P-value threshold for reporting.
        n_jobs: Number of parallel workers.
        block_size: Number of centers per block for block-based processing.
        use_gpu: Whether to use GPU acceleration.
        kappa_min_per_deg: Minimum κ in 1/deg.
        kappa_max_per_deg: Maximum κ in 1/deg.
        block_number: If specified, process only this block (0-indexed).
        count_blocks_only: If True, only count and print number of blocks, then exit.
    """
    from .array_backend import array_load
    import csv
    import logging
    import healpy as hp
    import os
    
    # Count blocks if requested
    if count_blocks_only:
        total_blocks = count_blocks(temperature_path, mask_path, step_deg, block_size)
        logging.info(f"Total number of blocks: {total_blocks} (block_size={block_size})")
        return

    # Check CUDA availability
    from .array_backend import print_backend_info

    print_backend_info()

    logging.info("Loading temperature...")
    temperature = array_load(temperature_path)
    # Load to GPU if requested - CUDA is REQUIRED if use_gpu=True
    if use_gpu:
        from .array_backend import cp

        # Check CUDA availability
        logging.info("  Checking CUDA availability...")
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                raise RuntimeError("No CUDA devices found")
            device_props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = (
                device_props.get("name", "Unknown").decode()
                if isinstance(device_props.get("name"), bytes)
                else device_props.get("name", "Unknown")
            )
            logging.info(
                f"  ✓ CUDA available: {device_count} GPU(s), device 0: {gpu_name}"
            )
        except Exception as e:
            error_msg = f"CUDA is REQUIRED but not available: {e}"
            logging.error(f"  ✗ {error_msg}")
            raise RuntimeError(error_msg) from e

        # Load temperature to GPU
        logging.info("  Loading temperature to GPU...")
        try:
            if not is_gpu_array(temperature):
                temperature = cp.asarray(temperature)
                logging.info(
                    f"  ✓ Temperature loaded to GPU: shape={temperature.shape}, dtype={temperature.dtype}"
                )
            else:
                logging.info(
                    f"  ✓ Temperature already on GPU: shape={temperature.shape}, dtype={temperature.dtype}"
                )
            if not is_gpu_array(temperature):
                raise RuntimeError("Temperature array is not on GPU after loading")
            log_array_stats("temperature", temperature)
        except Exception as e:
            error_msg = f"Failed to load temperature to GPU: {e}"
            logging.error(f"  ✗ {error_msg}")
            raise RuntimeError(error_msg) from e
    else:
        if is_gpu_array(temperature):
            temperature = cp.asnumpy(temperature)
        temperature = np.asarray(temperature, dtype=np.float64)
        log_array_stats("temperature", temperature)

    if mask_path:
        logging.info("Loading mask...")
        mask = array_load(mask_path)
        if use_gpu:
            # CUDA is REQUIRED - no fallback
            logging.info("  Loading mask to GPU...")
            try:
                if not is_gpu_array(mask):
                    mask = cp.asarray(mask)
                    logging.info(
                        f"  ✓ Mask loaded to GPU: shape={mask.shape}, dtype={mask.dtype}"
                    )
                else:
                    logging.info(
                        f"  ✓ Mask already on GPU: shape={mask.shape}, dtype={mask.dtype}"
                    )
                if not is_gpu_array(mask):
                    raise RuntimeError("Mask array is not on GPU after loading")
                log_array_stats("mask", mask)
            except Exception as e:
                error_msg = f"Failed to load mask to GPU: {e}"
                logging.error(f"  ✗ {error_msg}")
                raise RuntimeError(error_msg) from e
        else:
            if is_gpu_array(mask):
                mask = cp.asnumpy(mask)
            mask = np.asarray(mask, dtype=bool)
            log_array_stats("mask", mask)
    else:
        mask = None
        logging.info("No mask provided; using all pixels.")
    # Get centers and split into blocks if block_number is specified
    nside = hp.npix2nside(temperature.size)
    centers = get_centers(nside, step_deg)
    
    if mask is not None:
        if is_gpu_array(mask):
            mask_cpu = cp.asnumpy(mask)
        else:
            mask_cpu = mask
        centers = [c for c in centers if mask_cpu[int(c)]]
    
    total_blocks = (len(centers) + block_size - 1) // block_size
    
    if block_number is not None:
        if block_number < 0 or block_number >= total_blocks:
            raise ValueError(
                f"block_number must be in range [0, {total_blocks-1}], got {block_number}"
            )
        start_idx = block_number * block_size
        end_idx = min(start_idx + block_size, len(centers))
        centers_block = centers[start_idx:end_idx]
        logging.info(
            f"Processing block {block_number + 1}/{total_blocks} "
            f"(centers {start_idx} to {end_idx-1}, total {len(centers_block)} centers)"
        )
    else:
        centers_block = None
        logging.info(f"Processing all {len(centers)} centers in {total_blocks} blocks")
    
    logging.info("\n" + "=" * 60)
    logging.info("Starting structure detection scan...")
    logging.info("=" * 60)

    # Prepare CSV file for incremental writing
    ensure_dir_for_file(results_csv_path)
    file_exists = os.path.exists(results_csv_path)
    csv_columns = [
        "center_pix",
        "l_deg",
        "b_deg",
        "best_kappa_per_deg",
        "best_spacing_deg",
        "amplitude",
        "correlation",
        "p_value",
        "num_bins",
        "radius_min_deg",
        "radius_max_deg",
    ]
    
    # Write header if file doesn't exist
    if not file_exists:
        with open(results_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_columns)
        logging.info(f"Created results file: {results_csv_path}")
    else:
        logging.info(f"Appending to existing results file: {results_csv_path}")

    # Global counter for detections (shared across all workers via file or manager)
    # Use a simple file-based counter for multiprocessing compatibility
    detection_count_file = results_csv_path.replace(".csv", "_count.txt")
    detection_count = 0
    
    # Callback function to write detections immediately to CSV
    def write_detection_to_csv(detection: DetectionResult):
        """Write a single detection to CSV file immediately and update counter."""
        nonlocal detection_count
        detection_count += 1
        
        # Write to CSV
        with open(results_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    detection.center_pix,
                    detection.l_deg,
                    detection.b_deg,
                    detection.best_kappa_per_deg,
                    detection.best_spacing_deg,
                    detection.amplitude,
                    detection.correlation,
                    detection.p_value,
                    detection.num_bins,
                    detection.radius_min_deg,
                    detection.radius_max_deg,
                ]
            )
        
        # Update counter file
        try:
            with open(detection_count_file, "w") as f:
                f.write(str(detection_count))
        except Exception:
            pass
        
        # Print detection info to console
        logging.info(
            f"  [{detection_count}] ✓ Detection at (l={detection.l_deg:.2f}°, b={detection.b_deg:.2f}°): "
            f"κ={detection.best_kappa_per_deg:.3f}/deg, spacing={detection.best_spacing_deg:.2f}°, "
            f"corr={detection.correlation:.3f}, p={detection.p_value:.2e}"
        )

    detections = run_structure_detection(
        temperature=temperature,
        mask=mask,
        step_deg=step_deg,
        max_radius_deg=max_radius_deg,
        bin_size_deg=bin_size_deg,
        p_value_threshold=p_value_threshold,
        n_jobs=n_jobs,
        block_size=block_size,
        use_gpu=use_gpu,
        kappa_min_per_deg=kappa_min_per_deg,
        kappa_max_per_deg=kappa_max_per_deg,
        centers_block=centers_block,
        write_callback=write_detection_to_csv,
    )

    logging.info(f"\nBlock processing complete: {len(detections)} detections found")
    logging.info(f"Results appended to {results_csv_path}")
    logging.info("=" * 60)
    if block_number is not None:
        logging.info(f"Block {block_number + 1}/{total_blocks} complete!")
    else:
        logging.info("Structure detection complete!")
    logging.info("=" * 60)
