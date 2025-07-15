import os
# import numpy as np
from .array_backend import cp, np, array_load, array_save
from .utils import extract_temperature_component, extract_mask_component
import astropy.io.fits as fits

def run_convert_command(input_path: str, output_path: str) -> None:
    """
    Универсальная конвертация FITS в NPY для температуры, маски, пыли, HI, CO и др.
    Args:
        input_path (str): Path to input FITS file.
        output_path (str): Path to output .npy file.
    """
    import astropy.io.fits as fits
    fname = os.path.basename(input_path).lower()
    with fits.open(input_path) as hdul:
        data = hdul[1].data
        columns = list(data.names) if hasattr(data, 'names') else []
    # Универсальная логика выбора компонента
    if 'mask' in fname:
        print(f"[CONVERT] Извлекаю маску из {input_path}")
        arr = extract_mask_component(input_path)
    elif any(col in columns for col in ['NHI', 'HI', 'CO']):
        # HI/CO карты
        col = 'NHI' if 'NHI' in columns else ('HI' if 'HI' in columns else 'CO')
        print(f"[CONVERT] Извлекаю компонент '{col}' из {input_path}")
        arr = extract_gas_component(input_path, column=col)
    elif any(col in columns for col in ['I_STOKES', 'I']):
        # CMB SMICA или dust
        print(f"[CONVERT] Извлекаю компонент '{'I_STOKES' if 'I_STOKES' in columns else 'I'}' из {input_path}")
        arr = extract_temperature_component(input_path)
    else:
        raise ValueError(f"Не удалось определить нужный компонент для {input_path}. Доступные колонки: {columns}")
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    arr_size = arr.size
    chunk_size = 1024 * 1024  # 1M elements per chunk
    if tqdm:
        with open(output_path, 'wb') as f, tqdm(total=arr_size, unit='el', unit_scale=True, desc='Saving .npy') as pbar:
            np.lib.format.write_array_header_1_0(f, {'descr': np.lib.format.dtype_to_descr(arr.dtype), 'fortran_order': False, 'shape': arr.shape})
            for i in range(0, arr_size, chunk_size):
                chunk = arr[i:i+chunk_size]
                f.write(chunk.tobytes())
                pbar.update(len(chunk))
        tqdm.write(f"Saved array to {output_path} (shape: {arr.shape}, dtype: {arr.dtype})")
    else:
        np.save(output_path, arr)
        print(f"Saved array to {output_path} (shape: {arr.shape}, dtype: {arr.dtype})")

def extract_gas_component(fits_path: str, column: str = 'NHI') -> np.ndarray:
    """
    Извлекает компонент газа (HI, CO и др.) из FITS-файла по имени колонки.
    Args:
        fits_path (str): Path to FITS file.
        column (str): Column name (e.g. 'NHI', 'CO').
    Returns:
        np.ndarray: 1D array of gas values (float32).
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        if column in data.names:
            arr = data[column]
        else:
            raise ValueError(f"Column '{column}' not found in {fits_path}")
    return np.array(arr, dtype=np.float32)

def convert_dust_fits_to_npy(fits_path: str, output_path: str) -> None:
    """
    Convert Planck dust map FITS (COMP-MAP, column 'I') to npy array.
    Args:
        fits_path (str): Path to dust FITS file.
        output_path (str): Path to output .npy file.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data['I']
        arr = np.array(data, dtype=np.float32)
    np.save(output_path, arr)
    print(f"Saved dust map to {output_path} (shape: {arr.shape}, dtype: {arr.dtype})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert FITS to NPY for temperature, mask, dust, HI, CO, etc.')
    parser.add_argument('--input', required=True, help='Path to input FITS file')
    parser.add_argument('--output', required=True, help='Path to output .npy file')
    parser.add_argument('--column', default=None, help='Column name for gas/dust (e.g. NHI, CO, I). Optional.')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    column = args.column

    fname = os.path.basename(input_path).lower()
    if 'mask' in fname:
        arr = extract_mask_component(input_path)
    elif 'nhi' in fname or 'hi' in fname or (column is not None):
        arr = extract_gas_component(input_path, column=column or 'NHI')
    elif 'co' in fname:
        arr = extract_gas_component(input_path, column=column or 'CO')
    elif column is not None:
        # Для пыли и других карт с явной колонкой
        with fits.open(input_path) as hdul:
            data = hdul[1].data[column]
            arr = np.array(data, dtype=np.float32)
    else:
        arr = extract_temperature_component(input_path)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    arr_size = arr.size
    chunk_size = 1024 * 1024  # 1M elements per chunk
    if tqdm:
        with open(output_path, 'wb') as f, tqdm(total=arr_size, unit='el', unit_scale=True, desc='Saving .npy') as pbar:
            np.lib.format.write_array_header_1_0(f, {'descr': np.lib.format.dtype_to_descr(arr.dtype), 'fortran_order': False, 'shape': arr.shape})
            for i in range(0, arr_size, chunk_size):
                chunk = arr[i:i+chunk_size]
                f.write(chunk.tobytes())
                pbar.update(len(chunk))
        tqdm.write(f"Saved array to {output_path} (shape: {arr.shape}, dtype: {arr.dtype})")
    else:
        np.save(output_path, arr)
        print(f"Saved array to {output_path} (shape: {arr.shape}, dtype: {arr.dtype})") 