import logging
import numpy as np
from .array_backend import cp, np, array_load, array_save
from astropy.io import fits
import os


def log_array_stats(name, arr):
    """
    Log shape, dtype, percent of valid, NaN, and Inf values for the array.
    Args:
        name (str): Name for logging.
        arr (np.ndarray): Array to analyze.
    """
    total = arr.size
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    n_valid = np.isfinite(arr).sum()
    logging.info(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"valid={n_valid/total:.2%}, NaN={n_nan/total:.2%}, Inf={n_inf/total:.2%}"
    )


def check_shape_dtype(arr, expected_shape=None, expected_dtype=None, name=None):
    """
    Check array shape and dtype, raise ValueError if mismatch.
    Args:
        arr (np.ndarray): Array to check.
        expected_shape (tuple): Expected shape.
        expected_dtype (type): Expected dtype.
        name (str): Name for logging.
    """
    if expected_shape and arr.shape != expected_shape:
        logging.error(f"Shape mismatch for {name}: expected {expected_shape}, got {arr.shape}")
        raise ValueError(f"Shape mismatch for {name}")
    if expected_dtype and arr.dtype != expected_dtype:
        logging.error(f"Dtype mismatch for {name}: expected {expected_dtype}, got {arr.dtype}")
        raise ValueError(f"Dtype mismatch for {name}")


def filter_invalid(arr, name=None):
    """
    Replace NaN/Inf with zero, log percent of invalid values.
    Args:
        arr (np.ndarray): Input array.
        name (str): Name for logging.
    Returns:
        np.ndarray: Array with NaN/Inf replaced by zero.
    """
    n_total = arr.size
    n_invalid = (~np.isfinite(arr)).sum()
    if n_invalid > 0:
        logging.warning(f"{name}: {n_invalid/n_total:.2%} pixels are NaN/Inf; replaced with zero.")
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def extract_temperature_component(fits_path: str, expected_shape=None) -> np.ndarray:
    """
    Extract the temperature (I) component from a Planck SMICA FITS file,
    or the dust intensity (I) from a Planck dust map FITS file.
    Tries 'I_STOKES' (CMB), then 'I' (dust), then first row if 2D.
    Checks shape, dtype, logs stats, and filters invalid values (keeps shape).
    Args:
        fits_path (str): Path to the input FITS file.
        expected_shape (tuple, optional): Expected shape for validation.
    Returns:
        np.ndarray: 1D array of temperature or dust values (float32).
    Raises:
        ValueError: If no valid temperature/dust column is found or array is empty.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        # Try CMB temperature
        if 'I_STOKES' in data.names:
            temperature = data['I_STOKES']
        # Try dust map
        elif 'I' in data.names:
            temperature = data['I']
        elif hasattr(data, 'shape') and len(data.shape) == 2:
            temperature = data[0]
        else:
            logging.error(f"Cannot find 'I_STOKES' or 'I' column in FITS data: {fits_path}")
            raise ValueError("Cannot find 'I_STOKES' or 'I' column in FITS data.")
    arr = np.array(temperature, dtype=np.float32)
    log_array_stats(f"temperature:{fits_path}", arr)
    check_shape_dtype(arr, expected_shape, np.float32, name="temperature")
    arr = filter_invalid(arr, name="temperature")
    if arr.size == 0:
        logging.error(f"All pixels are invalid in {fits_path}")
        raise ValueError("All pixels are invalid (NaN/Inf) in temperature map.")
    return arr


def extract_mask_component(fits_path: str, expected_shape=None) -> np.ndarray:
    """
    Extract binary mask from a Planck FITS mask file.
    Checks shape, dtype, logs stats, and filters invalid values (keeps shape).
    Args:
        fits_path (str): Path to the input FITS file.
        expected_shape (tuple, optional): Expected shape for validation.
    Returns:
        np.ndarray: 1D boolean array (True=use, False=masked)
    Raises:
        ValueError: If mask shape is empty or contains only invalid values.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        # Try to extract the first column/field as mask
        if hasattr(data, 'shape') and len(data.shape) == 2:
            mask = data[0]
        elif hasattr(data, 'field'):
            mask = data.field(0)
        else:
            mask = data
    mask_bin = np.array(mask > 0, dtype=bool)
    log_array_stats(f"mask:{fits_path}", mask_bin)
    check_shape_dtype(mask_bin, expected_shape, np.bool_, name="mask")
    if mask_bin.size == 0:
        logging.error(f"Mask is empty in {fits_path}")
        raise ValueError("Mask is empty.")
    percent_unmasked = 100.0 * np.sum(mask_bin) / mask_bin.size
    logging.info(f"Unmasked pixels: {percent_unmasked:.2f}%")
    if percent_unmasked == 0.0:
        logging.warning(f"All pixels are masked in {fits_path}")
    return mask_bin


def ensure_dir_for_file(filepath):
    """Create directory for file if it does not exist."""
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def get_centers(nside, step_deg):
    """
    Generate HEALPix pixel indices for centers with given step in degrees.
    Args:
        nside (int): HEALPix NSIDE
        step_deg (int): step in degrees
    Returns:
        np.ndarray: array of pixel indices
    """
    import healpy as hp
    npix = hp.nside2npix(nside)
    # Uniform sampling over the sky with step step_deg
    n_centers = int(npix / (4 * 180 // step_deg))
    centers = np.arange(0, npix, max(1, npix // n_centers), dtype=int)
    return centers 


def check_maps_compatibility(arr1, arr2, name1="map1", name2="map2"):
    """
    Check that two arrays have the same shape and compatible dtype.
    Args:
        arr1 (np.ndarray): First array.
        arr2 (np.ndarray): Second array.
        name1 (str): Name for logging.
        name2 (str): Name for logging.
    Raises:
        ValueError: If shapes or dtypes are not compatible.
    """
    if arr1.shape != arr2.shape:
        logging.error(f"Shape mismatch: {name1} {arr1.shape} vs {name2} {arr2.shape}")
        raise ValueError(f"Shape mismatch: {name1} {arr1.shape} vs {name2} {arr2.shape}")
    if arr1.dtype != arr2.dtype:
        logging.warning(f"Dtype mismatch: {name1} {arr1.dtype} vs {name2} {arr2.dtype}") 