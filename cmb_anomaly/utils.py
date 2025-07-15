from .array_backend import cp, np, array_load, array_save
from astropy.io import fits

def extract_temperature_component(fits_path: str) -> np.ndarray:
    """
    Extract the temperature (I) component from a Planck SMICA FITS file,
    or the dust intensity (I) from a Planck dust map FITS file.
    Tries 'I_STOKES' (CMB), then 'I' (dust), then first row if 2D.
    Args:
        fits_path (str): Path to the input FITS file.
    Returns:
        np.ndarray: 1D array of temperature or dust values (float32).
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
            raise ValueError("Cannot find 'I_STOKES' or 'I' column in FITS data.")
    return np.array(temperature, dtype=np.float32)

def extract_mask_component(fits_path: str) -> np.ndarray:
    """
    Extract binary mask from a Planck FITS mask file.
    Args:
        fits_path (str): Path to the input FITS file.
    Returns:
        np.ndarray: 1D boolean array (True=use, False=masked)
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
    # Convert to bool: True (use), False (masked)
    mask_bin = np.array(mask > 0, dtype=bool)
    return mask_bin

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
    # Пример: равномерная выборка по небу с шагом step_deg
    # Можно использовать grid по theta/phi, но для простоты — равномерно по пикселям
    n_centers = int(npix / (4 * 180 // step_deg))
    centers = np.arange(0, npix, max(1, npix // n_centers), dtype=int)
    return centers 