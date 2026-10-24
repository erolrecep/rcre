"""
Python FFI Bindings for RCRE (Recep's Code Run Everywhere)
Unified C23 Computer Vision & Machine Learning Engine
"""

import ctypes
from ctypes import c_void_p, c_size_t, c_float, c_uint8, c_uint32, c_int, c_char_p, POINTER, Structure
import os
import sys
from pathlib import Path
import numpy as np

# Find shared library
def _find_library():
    # Check parent directories for lib/librcre.dylib or librcre.so
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        for dylib_name in ["librcre.dylib", "librcre.so", "rcre.dll"]:
            candidate = parent / "lib" / dylib_name
            if candidate.exists():
                return str(candidate)
    return "librcre.dylib"

_lib_path = _find_library()
try:
    _rcre = ctypes.CDLL(_lib_path)
except OSError:
    _rcre = None

# Structs
class RCRE_Arena(Structure):
    _fields_ = [
        ("buffer", c_void_p),
        ("capacity", c_size_t),
        ("offset", c_size_t)
    ]

class RCRE_Tensor(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("shape", c_size_t * 4),
        ("strides", ctypes.c_ssize_t * 4),
        ("ndim", c_uint8),
        ("dtype", c_uint8),
        ("layout", c_uint8)
    ]

# Setup C Function Signatures if library is loaded
if _rcre is not None:
    _rcre.rcre_arena_create.argtypes = [c_size_t]
    _rcre.rcre_arena_create.restype = POINTER(RCRE_Arena)

    _rcre.rcre_arena_destroy.argtypes = [POINTER(RCRE_Arena)]
    _rcre.rcre_arena_destroy.restype = None

    _rcre.rcre_arena_reset.argtypes = [POINTER(RCRE_Arena)]
    _rcre.rcre_arena_reset.restype = None

    _rcre.rcre_tensor_create.argtypes = [POINTER(c_size_t), c_uint8, c_uint8, c_uint8, POINTER(RCRE_Arena)]
    _rcre.rcre_tensor_create.restype = RCRE_Tensor

    _rcre.rcre_cv_rgb_to_gray.argtypes = [POINTER(RCRE_Tensor), POINTER(RCRE_Tensor)]
    _rcre.rcre_cv_rgb_to_gray.restype = c_int

    _rcre.rcre_hal_init.argtypes = []
    _rcre.rcre_hal_init.restype = None

    _rcre.rcre_cpu_get_brand.argtypes = []
    _rcre.rcre_cpu_get_brand.restype = c_char_p


class Arena:
    """Safe Python context manager for RCRE_Arena linear memory allocator."""
    def __init__(self, capacity_bytes: int = 16 * 1024 * 1024):
        if _rcre is None:
            raise RuntimeError(f"Could not load RCRE shared library from {_lib_path}")
        self._ptr = _rcre.rcre_arena_create(capacity_bytes)
        if not self._ptr:
            raise MemoryError("Failed to allocate RCRE_Arena")

    def reset(self):
        _rcre.rcre_arena_reset(self._ptr)

    def destroy(self):
        if self._ptr:
            _rcre.rcre_arena_destroy(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    @property
    def capacity(self) -> int:
        return self._ptr.contents.capacity

    @property
    def offset(self) -> int:
        return self._ptr.contents.offset


def rgb_to_gray(rgb_array: np.ndarray, arena: Arena) -> np.ndarray:
    """Converts HWC RGB image array to Grayscale using Rec.601 15-bit fixed point."""
    assert rgb_array.ndim == 3 and rgb_array.shape[2] == 3
    assert rgb_array.dtype == np.uint8

    H, W, _ = rgb_array.shape
    c_shape_in = (c_size_t * 3)(H, W, 3)
    c_shape_out = (c_size_t * 2)(H, W)

    t_in = _rcre.rcre_tensor_create(c_shape_in, 3, 0, 1, arena._ptr)
    t_out = _rcre.rcre_tensor_create(c_shape_out, 2, 0, 1, arena._ptr)

    # Copy input bytes
    ctypes.memmove(t_in.data, rgb_array.ctypes.data, H * W * 3)

    # Execute C23 SIMD kernel
    status = _rcre.rcre_cv_rgb_to_gray(ctypes.byref(t_in), ctypes.byref(t_out))
    if status != 0:
        raise RuntimeError(f"rcre_cv_rgb_to_gray failed with code {status}")

    # Read output bytes into numpy array
    out_buf = np.empty((H, W), dtype=np.uint8)
    ctypes.memmove(out_buf.ctypes.data, t_out.data, H * W)
    return out_buf


def cpu_brand() -> str:
    """Returns the host CPU brand string as detected by RCRE HAL."""
    if _rcre is None:
        return "Unknown"
    return _rcre.rcre_cpu_get_brand().decode('utf-8')
