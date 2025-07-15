"""
array_backend.py

Unified array backend for NumPy/CuPy with automatic fallback.

- Tries to import CuPy as cp, otherwise uses NumPy as cp.
- Use 'cp' for all array operations in project code.
- Provides is_gpu_array, is_cpu_array for backend checks.
- Use 'cp.load' and 'cp.save' for .npy files; fallback to NumPy if needed.
"""

try:
    import cupy as cp
    _IS_CUPY = True
except ImportError:
    import numpy as cp
    _IS_CUPY = False

import numpy as np

def is_gpu_array(arr):
    """Return True if array is a CuPy array (on GPU)."""
    if _IS_CUPY:
        import cupy
        return isinstance(arr, cupy.ndarray)
    return False

def is_cpu_array(arr):
    """Return True if array is a NumPy array (on CPU)."""
    return isinstance(arr, np.ndarray)

def print_backend_info():
    """Prints info about current array backend and CUDA availability."""
    if _IS_CUPY:
        try:
            n_gpus = cp.cuda.runtime.getDeviceCount()
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'] if n_gpus > 0 else 'N/A'
            print(f"[array_backend] CUDA detected: CuPy backend, {n_gpus} GPU(s), first GPU: {gpu_name}")
        except Exception as e:
            print(f"[array_backend] CuPy backend, but CUDA not available: {e}")
    else:
        print("[array_backend] CPU only: NumPy backend in use")

# Safe load/save wrappers for .npy files

def array_load(path, allow_pickle=False):
    """Load .npy file to backend array. Fallback to NumPy if CuPy fails."""
    try:
        return cp.load(path, allow_pickle=allow_pickle)
    except Exception:
        arr = np.load(path, allow_pickle=allow_pickle)
        if _IS_CUPY:
            return cp.asarray(arr)
        return arr

def array_save(path, arr):
    """Save array to .npy file. Fallback to NumPy if CuPy fails."""
    try:
        cp.save(path, arr)
    except Exception:
        np.save(path, cp.asnumpy(arr) if _IS_CUPY else arr) 