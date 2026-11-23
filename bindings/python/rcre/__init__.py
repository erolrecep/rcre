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
        ("grad", POINTER(c_float)),
        ("shape", c_size_t * 8),
        ("strides", ctypes.c_ssize_t * 8),
        ("numel", c_size_t),
        ("ndim", c_uint8),
        ("dtype", c_int),
        ("layout", c_int),
        ("is_view", ctypes.c_bool),
        ("label", c_char_p)
    ]

class RCRE_StatSummary(Structure):
    _fields_ = [
        ("count", c_size_t),
        ("null_count", c_size_t),
        ("mean", c_float),
        ("variance", c_float),
        ("stddev", c_float),
        ("skewness", c_float),
        ("kurtosis", c_float),
        ("min", c_float),
        ("max", c_float),
        ("q25", c_float),
        ("median", c_float),
        ("q75", c_float),
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

    # Metrics
    _rcre.rcre_metric_roc_auc.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, POINTER(RCRE_Arena)]
    _rcre.rcre_metric_roc_auc.restype = c_float

    _rcre.rcre_metric_pr_auc.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, POINTER(RCRE_Arena)]
    _rcre.rcre_metric_pr_auc.restype = c_float

    _rcre.rcre_metric_mse.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t]
    _rcre.rcre_metric_mse.restype = c_float

    _rcre.rcre_metric_rmse.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t]
    _rcre.rcre_metric_rmse.restype = c_float

    _rcre.rcre_metric_mae.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t]
    _rcre.rcre_metric_mae.restype = c_float

    _rcre.rcre_metric_r2_score.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t]
    _rcre.rcre_metric_r2_score.restype = c_float

    _rcre.rcre_stat_describe.argtypes = [POINTER(c_float), c_size_t, POINTER(RCRE_Arena)]
    _rcre.rcre_stat_describe.restype = RCRE_StatSummary

    _rcre.rcre_abi_version.argtypes = []
    _rcre.rcre_abi_version.restype = c_uint32

    _rcre.rcre_io_save_npy.argtypes = [POINTER(RCRE_Tensor), c_char_p]
    _rcre.rcre_io_save_npy.restype = c_int

    _rcre.rcre_io_load_npy.argtypes = [c_char_p, POINTER(RCRE_Arena), POINTER(RCRE_Tensor)]
    _rcre.rcre_io_load_npy.restype = c_int


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


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray, arena: Arena) -> float:
    """Computes Area Under ROC Curve (ROC-AUC) in O(N log N) using C23 kernel."""
    y_true_c = np.ascontiguousarray(y_true, dtype=np.float32)
    y_score_c = np.ascontiguousarray(y_score, dtype=np.float32)
    assert len(y_true_c) == len(y_score_c)
    n = len(y_true_c)
    p_true = y_true_c.ctypes.data_as(POINTER(c_float))
    p_score = y_score_c.ctypes.data_as(POINTER(c_float))
    return float(_rcre.rcre_metric_roc_auc(p_true, p_score, n, arena._ptr))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes R-squared coefficient of determination."""
    y_true_c = np.ascontiguousarray(y_true, dtype=np.float32)
    y_pred_c = np.ascontiguousarray(y_pred, dtype=np.float32)
    assert len(y_true_c) == len(y_pred_c)
    n = len(y_true_c)
    p_true = y_true_c.ctypes.data_as(POINTER(c_float))
    p_pred = y_pred_c.ctypes.data_as(POINTER(c_float))
    return float(_rcre.rcre_metric_r2_score(p_true, p_pred, n))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error."""
    y_true_c = np.ascontiguousarray(y_true, dtype=np.float32)
    y_pred_c = np.ascontiguousarray(y_pred, dtype=np.float32)
    assert len(y_true_c) == len(y_pred_c)
    n = len(y_true_c)
    p_true = y_true_c.ctypes.data_as(POINTER(c_float))
    p_pred = y_pred_c.ctypes.data_as(POINTER(c_float))
    return float(_rcre.rcre_metric_rmse(p_true, p_pred, n))


def stat_describe(data: np.ndarray, arena: Arena) -> dict:
    """Computes exploratory summary statistics via single-pass Welford algorithm."""
    arr = np.ascontiguousarray(data, dtype=np.float32)
    n = len(arr)
    p_data = arr.ctypes.data_as(POINTER(c_float))
    res = _rcre.rcre_stat_describe(p_data, n, arena._ptr)
    return {
        "count": res.count,
        "null_count": res.null_count,
        "mean": res.mean,
        "variance": res.variance,
        "stddev": res.stddev,
        "skewness": res.skewness,
        "kurtosis": res.kurtosis,
        "min": res.min,
        "max": res.max,
        "q25": res.q25,
        "median": res.median,
        "q75": res.q75,
    }


def abi_version() -> int:
    """Returns packed RCRE C ABI version hex (e.g. 0x020000)."""
    return int(_rcre.rcre_abi_version())


def load_npy(filepath: str, arena: Arena) -> np.ndarray:
    """Loads an .npy tensor via C23 mmap engine and returns a NumPy F32 view/copy."""
    t = RCRE_Tensor()
    rc = _rcre.rcre_io_load_npy(filepath.encode('utf-8'), arena._ptr, ctypes.byref(t))
    if rc != 0:
        raise RuntimeError(f"rcre_io_load_npy failed with code {rc}")
    shape = tuple(t.shape[i] for i in range(t.ndim))
    return np.ctypeslib.as_array(ctypes.cast(t.data, POINTER(c_float)), shape=shape).copy()


def save_npy(arr: np.ndarray, filepath: str) -> None:
    """Saves a contiguous float32 NumPy array as a 64B-aligned .npy via RCRE."""
    c_arr = np.ascontiguousarray(arr, dtype=np.float32)
    shape = (c_size_t * 8)(*([c_arr.shape[i] if i < c_arr.ndim else 0 for i in range(8)]))
    t = RCRE_Tensor()
    t.data = c_arr.ctypes.data_as(c_void_p)
    t.ndim = c_arr.ndim
    for i in range(c_arr.ndim):
        t.shape[i] = c_arr.shape[i]
        t.strides[i] = c_arr.strides[i] // 4
    t.dtype = 3  # RCRE_DTYPE_F32
    t.layout = 3  # RCRE_LAYOUT_TABULAR
    t.is_view = True
    _ = shape
    rc = _rcre.rcre_io_save_npy(ctypes.byref(t), filepath.encode('utf-8'))
    if rc != 0:
        raise RuntimeError(f"rcre_io_save_npy failed with code {rc}")
