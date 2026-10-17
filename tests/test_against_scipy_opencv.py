#!/usr/bin/env python3
"""
test_against_scipy_opencv.py
Layer 3 Verification Gatekeeper for RCRE (QA-REQ-03)

Asserts that RCRE C23 outputs match OpenCV and Scikit-Learn within numerical tolerance.
"""

import sys
import ctypes
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression, LogisticRegression

# ---------------------------------------------------------------------------
# C23 RCRE Types & FFI Definitions
# ---------------------------------------------------------------------------

class RCRE_Tensor(ctypes.Structure):
    _fields_ = [
        ("storage", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
        ("shape", ctypes.c_size_t * 4),
        ("strides", ctypes.c_ssize_t * 4),
        ("ndim", ctypes.c_uint8),
        ("dtype", ctypes.c_int),
        ("layout", ctypes.c_int),
    ]

# DType enum
RCRE_DTYPE_U8  = 0
RCRE_DTYPE_I32 = 1
RCRE_DTYPE_I64 = 2
RCRE_DTYPE_F32 = 3

# Layout enum
RCRE_LAYOUT_UNKNOWN = 0
RCRE_LAYOUT_HWC     = 1
RCRE_LAYOUT_CHW     = 2
RCRE_LAYOUT_TABULAR = 3

def load_rcre():
    import os
    lib_path = os.path.join(os.path.dirname(__file__), "..", "lib", "librcre.dylib")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(os.path.dirname(__file__), "..", "lib", "librcre.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Could not find librcre at {lib_path}")

    rcre = ctypes.CDLL(lib_path)

    # Arena lifecycle
    rcre.rcre_arena_create.argtypes = [ctypes.c_size_t]
    rcre.rcre_arena_create.restype = ctypes.c_void_p

    rcre.rcre_arena_destroy.argtypes = [ctypes.c_void_p]
    rcre.rcre_arena_destroy.restype = None

    # Tensor creation
    rcre.rcre_tensor_create.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_uint8,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p
    ]
    rcre.rcre_tensor_create.restype = RCRE_Tensor

    # CV primitives
    rcre.rcre_cv_rgb_to_gray.argtypes = [ctypes.POINTER(RCRE_Tensor), ctypes.POINTER(RCRE_Tensor)]
    rcre.rcre_cv_rgb_to_gray.restype = ctypes.c_int

    rcre.rcre_cv_gaussian_blur.argtypes = [
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor),
        ctypes.c_float,
        ctypes.c_void_p
    ]
    rcre.rcre_cv_gaussian_blur.restype = ctypes.c_int

    rcre.rcre_cv_sobel.argtypes = [
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor),
        ctypes.c_void_p
    ]
    rcre.rcre_cv_sobel.restype = ctypes.c_int

    # ML primitives
    rcre.rcre_ml_ols_fit.argtypes = [
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor),
        ctypes.c_void_p
    ]
    rcre.rcre_ml_ols_fit.restype = ctypes.c_int

    rcre.rcre_ml_logistic_fit.argtypes = [
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor),
        ctypes.c_float,
        ctypes.c_size_t,
        ctypes.c_void_p
    ]
    rcre.rcre_ml_logistic_fit.restype = ctypes.c_int

    rcre.rcre_ml_predict_linear.argtypes = [
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor)
    ]
    rcre.rcre_ml_predict_linear.restype = ctypes.c_int

    return rcre

def wrap_numpy_tensor(np_arr, dtype, layout, rcre, arena):
    """Wraps a NumPy array into an RCRE_Tensor allocated inside arena."""
    shape_c = (ctypes.c_size_t * len(np_arr.shape))(*np_arr.shape)
    t = rcre.rcre_tensor_create(shape_c, len(np_arr.shape), dtype, layout, arena)
    ctypes.memmove(t.data, np_arr.ctypes.data, np_arr.nbytes)
    return t

# ---------------------------------------------------------------------------
# Parity Tests
# ---------------------------------------------------------------------------

def test_color_parity(rcre):
    print("🔬 [Parity 1/4] Testing RGB -> Grayscale against OpenCV...")
    arena = rcre.rcre_arena_create(16 * 1024 * 1024)

    # Test on a randomized 64x64 RGB image
    np.random.seed(42)
    rgb_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    # OpenCV reference (cv2.cvtColor COLOR_RGB2GRAY uses Rec.601)
    cv_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # RCRE implementation
    t_src = wrap_numpy_tensor(rgb_img, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, rcre, arena)
    dst_shape = (ctypes.c_size_t * 2)(64, 64)
    t_dst = rcre.rcre_tensor_create(dst_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_TABULAR, arena)

    status = rcre.rcre_cv_rgb_to_gray(ctypes.byref(t_src), ctypes.byref(t_dst))
    assert status == 0, f"rcre_cv_rgb_to_gray returned {status}"

    rcre_gray = np.ctypeslib.as_array(ctypes.cast(t_dst.data, ctypes.POINTER(ctypes.c_uint8)), shape=(64, 64))

    # Allow at most 1 LSB difference due to fixed-point integer rounding differences
    diff = np.abs(rcre_gray.astype(np.int32) - cv_gray.astype(np.int32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"     Max LSB diff: {max_diff}, Mean diff: {mean_diff:.4f}")
    assert max_diff <= 1, f"Max difference {max_diff} exceeded 1 LSB!"
    rcre.rcre_arena_destroy(arena)
    print("     ✓ RGB -> Grayscale parity verified (matches OpenCV within 1 LSB)")

def test_gaussian_blur_parity(rcre):
    print("🔬 [Parity 2/4] Testing 2D Gaussian Blur against OpenCV...")
    arena = rcre.rcre_arena_create(16 * 1024 * 1024)

    sigma = 1.2
    # Radius = ceil(3 * sigma) = 4, ksize = 2 * 4 + 1 = 9
    ksize = 2 * int(np.ceil(3.0 * sigma)) + 1

    np.random.seed(1337)
    img = np.random.uniform(0.0, 1.0, (48, 48)).astype(np.float32)

    # OpenCV reference with BORDER_REFLECT_101
    cv_blur = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)

    # RCRE implementation
    t_src = wrap_numpy_tensor(img, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, rcre, arena)
    shape_c = (ctypes.c_size_t * 2)(48, 48)
    t_dst = rcre.rcre_tensor_create(shape_c, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena)

    status = rcre.rcre_cv_gaussian_blur(ctypes.byref(t_src), ctypes.byref(t_dst), ctypes.c_float(sigma), arena)
    assert status == 0, f"rcre_cv_gaussian_blur returned {status}"

    rcre_blur = np.ctypeslib.as_array(ctypes.cast(t_dst.data, ctypes.POINTER(ctypes.c_float)), shape=(48, 48))

    diff = np.abs(rcre_blur - cv_blur)
    max_diff = np.max(diff)
    print(f"     Max abs diff vs OpenCV GaussianBlur: {max_diff:.6e}")
    assert max_diff <= 1e-4, f"Max difference {max_diff} exceeded tolerance 1e-4!"
    rcre.rcre_arena_destroy(arena)
    print("     ✓ Gaussian blur parity verified (epsilon <= 1e-4)")

def test_sobel_parity(rcre):
    print("🔬 [Parity 3/4] Testing 3x3 Sobel spatial derivatives against OpenCV...")
    arena = rcre.rcre_arena_create(16 * 1024 * 1024)

    np.random.seed(999)
    img = np.random.uniform(0.0, 100.0, (32, 32)).astype(np.float32)

    # OpenCV references
    cv_dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    cv_dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)

    # RCRE implementation
    t_src = wrap_numpy_tensor(img, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, rcre, arena)
    shape_c = (ctypes.c_size_t * 2)(32, 32)
    t_dx = rcre.rcre_tensor_create(shape_c, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena)
    t_dy = rcre.rcre_tensor_create(shape_c, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena)

    status = rcre.rcre_cv_sobel(ctypes.byref(t_src), ctypes.byref(t_dx), ctypes.byref(t_dy), arena)
    assert status == 0, f"rcre_cv_sobel returned {status}"

    rcre_dx = np.ctypeslib.as_array(ctypes.cast(t_dx.data, ctypes.POINTER(ctypes.c_float)), shape=(32, 32))
    rcre_dy = np.ctypeslib.as_array(ctypes.cast(t_dy.data, ctypes.POINTER(ctypes.c_float)), shape=(32, 32))

    diff_dx = np.max(np.abs(rcre_dx - cv_dx))
    diff_dy = np.max(np.abs(rcre_dy - cv_dy))

    print(f"     Max abs diff Sobel dx: {diff_dx:.6e}, dy: {diff_dy:.6e}")
    assert diff_dx <= 1e-4, f"Sobel dx diff {diff_dx} exceeded tolerance 1e-4!"
    assert diff_dy <= 1e-4, f"Sobel dy diff {diff_dy} exceeded tolerance 1e-4!"

    rcre.rcre_arena_destroy(arena)
    print("     ✓ Sobel filter parity verified (dx, dy match OpenCV within epsilon <= 1e-4)")

def test_ols_parity(rcre):
    print("🔬 [Parity 4/4] Testing OLS Linear Regression against Scikit-Learn...")
    arena = rcre.rcre_arena_create(16 * 1024 * 1024)

    np.random.seed(123)
    N = 100
    D = 5
    X = np.random.randn(N, D).astype(np.float32)
    true_weights = np.array([2.5, -1.8, 0.4, 3.2, -0.9], dtype=np.float32)
    y = (X @ true_weights).astype(np.float32)

    # Scikit-learn reference
    sk_model = LinearRegression(fit_intercept=False)
    sk_model.fit(X, y)
    sk_weights = sk_model.coef_.astype(np.float32)

    # RCRE implementation
    t_X = wrap_numpy_tensor(X, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, rcre, arena)
    t_y = wrap_numpy_tensor(y.reshape(N, 1), RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, rcre, arena)

    w_shape = (ctypes.c_size_t * 2)(D, 1)
    t_w = rcre.rcre_tensor_create(w_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena)

    status = rcre.rcre_ml_ols_fit(ctypes.byref(t_X), ctypes.byref(t_y), ctypes.byref(t_w), arena)
    assert status == 0, f"rcre_ml_ols_fit returned {status}"

    rcre_weights = np.ctypeslib.as_array(ctypes.cast(t_w.data, ctypes.POINTER(ctypes.c_float)), shape=(D,))

    diff = np.max(np.abs(rcre_weights - sk_weights))
    print(f"     Learned weights: {rcre_weights}")
    print(f"     Sklearn weights: {sk_weights}")
    print(f"     Max abs weight diff vs Scikit-Learn: {diff:.6e}")
    assert diff <= 1e-3, f"OLS weights diff {diff} exceeded tolerance 1e-3!"

    rcre.rcre_arena_destroy(arena)
    print("     ✓ OLS regression parity verified against Scikit-Learn")

# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("🛡️  RCRE Phase 2 Numerical Parity Harness against OpenCV & Scikit-Learn")
    print("=" * 70)

    rcre = load_rcre()
    test_color_parity(rcre)
    test_gaussian_blur_parity(rcre)
    test_sobel_parity(rcre)
    test_ols_parity(rcre)

    print("=" * 70)
    print("🎉 ALL LAYER 3 NUMERICAL PARITY GATES PASSED (OpenCV & Scikit-Learn)!")
    print("=" * 70)

if __name__ == "__main__":
    main()
