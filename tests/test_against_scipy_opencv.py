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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

    # Clustering & PCA
    rcre.rcre_ml_kmeans_fit.argtypes = [
        ctypes.POINTER(RCRE_Tensor),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor),
        ctypes.c_void_p
    ]
    rcre.rcre_ml_kmeans_fit.restype = ctypes.c_int

    rcre.rcre_ml_pca_fit.argtypes = [
        ctypes.POINTER(RCRE_Tensor),
        ctypes.c_size_t,
        ctypes.POINTER(RCRE_Tensor),
        ctypes.POINTER(RCRE_Tensor),
        ctypes.c_void_p
    ]
    rcre.rcre_ml_pca_fit.restype = ctypes.c_int

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
    print("🔬 [Parity 1/6] Testing RGB -> Grayscale against OpenCV...")
    arena = rcre.rcre_arena_create(16 * 1024 * 1024)

    np.random.seed(42)
    rgb_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    cv_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    t_src = wrap_numpy_tensor(rgb_img, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, rcre, arena)
    dst_shape = (ctypes.c_size_t * 2)(64, 64)
    t_dst = rcre.rcre_tensor_create(dst_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_TABULAR, arena)

    status = rcre.rcre_cv_rgb_to_gray(ctypes.byref(t_src), ctypes.byref(t_dst))
    assert status == 0, f"rcre_cv_rgb_to_gray returned {status}"

    rcre_gray = np.ctypeslib.as_array(ctypes.cast(t_dst.data, ctypes.POINTER(ctypes.c_uint8)), shape=(64, 64))

    diff = np.abs(rcre_gray.astype(np.int32) - cv_gray.astype(np.int32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"     Max LSB diff: {max_diff}, Mean diff: {mean_diff:.4f}")
    assert max_diff <= 1, f"Max difference {max_diff} exceeded 1 LSB!"
    rcre.rcre_arena_destroy(arena)
    print("     ✓ RGB -> Grayscale parity verified (matches OpenCV within 1 LSB)")

def test_gaussian_blur_parity(rcre):
    print("🔬 [Parity 2/6] Testing 2D Gaussian Blur against OpenCV...")
    arena = rcre.rcre_arena_create(16 * 1024 * 1024)

    sigma = 1.2
    ksize = 2 * int(np.ceil(3.0 * sigma)) + 1

    np.random.seed(1337)
    img = np.random.uniform(0.0, 1.0, (48, 48)).astype(np.float32)

    cv_blur = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)

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
    print("🔬 [Parity 3/6] Testing 3x3 Sobel spatial derivatives against OpenCV...")
    arena = rcre.rcre_arena_create(16 * 1024 * 1024)

    np.random.seed(999)
    img = np.random.uniform(0.0, 100.0, (32, 32)).astype(np.float32)

    cv_dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    cv_dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)

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
    print("🔬 [Parity 4/6] Testing OLS Linear Regression against Scikit-Learn...")
    arena = rcre.rcre_arena_create(16 * 1024 * 1024)

    np.random.seed(123)
    N = 100
    D = 5
    X = np.random.randn(N, D).astype(np.float32)
    true_weights = np.array([2.5, -1.8, 0.4, 3.2, -0.9], dtype=np.float32)
    y = (X @ true_weights).astype(np.float32)

    sk_model = LinearRegression(fit_intercept=False)
    sk_model.fit(X, y)
    sk_weights = sk_model.coef_.astype(np.float32)

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

def test_kmeans_parity(rcre):
    print("🔬 [Parity 5/6] Testing k-Means Clustering against Scikit-Learn...")
    arena = rcre.rcre_arena_create(16 * 1024 * 1024)

    np.random.seed(42)
    # Generate 3 well-separated 2D clusters
    c0 = np.random.randn(30, 2).astype(np.float32) + np.array([-5.0, -5.0], dtype=np.float32)
    c1 = np.random.randn(30, 2).astype(np.float32) + np.array([5.0, -5.0], dtype=np.float32)
    c2 = np.random.randn(30, 2).astype(np.float32) + np.array([0.0, 5.0], dtype=np.float32)
    X = np.vstack([c0, c1, c2])
    N, D = X.shape
    K = 3

    sk_kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    sk_kmeans.fit(X)
    sk_centers = np.sort(sk_kmeans.cluster_centers_, axis=0)

    t_X = wrap_numpy_tensor(X, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, rcre, arena)
    c_shape = (ctypes.c_size_t * 2)(K, D)
    t_C = rcre.rcre_tensor_create(c_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena)

    status = rcre.rcre_ml_kmeans_fit(ctypes.byref(t_X), K, 100, ctypes.byref(t_C), None, arena)
    assert status == 0, f"rcre_ml_kmeans_fit returned {status}"

    rcre_centers = np.ctypeslib.as_array(ctypes.cast(t_C.data, ctypes.POINTER(ctypes.c_float)), shape=(K, D))
    rcre_centers_sorted = np.sort(rcre_centers, axis=0)

    diff = np.max(np.abs(rcre_centers_sorted - sk_centers))
    print(f"     Max centroid diff vs Scikit-Learn: {diff:.6e}")
    assert diff < 0.25, f"Centroid difference {diff} too large!"

    rcre.rcre_arena_destroy(arena)
    print("     ✓ k-Means clustering parity verified against Scikit-Learn")

def test_pca_parity(rcre):
    print("🔬 [Parity 6/6] Testing PCA Dimensionality Reduction against Scikit-Learn...")
    arena = rcre.rcre_arena_create(16 * 1024 * 1024)

    np.random.seed(777)
    N, D, K = 50, 4, 2
    # Create correlated data
    A = np.random.randn(N, D).astype(np.float32)
    X = (A @ np.diag([5.0, 2.0, 0.5, 0.1])).astype(np.float32)

    sk_pca = PCA(n_components=K)
    sk_pca.fit(X)

    t_X = wrap_numpy_tensor(X, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, rcre, arena)
    comp_shape = (ctypes.c_size_t * 2)(K, D)
    t_W = rcre.rcre_tensor_create(comp_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena)

    mean_shape = (ctypes.c_size_t * 2)(D, 1)
    t_mean = rcre.rcre_tensor_create(mean_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena)

    status = rcre.rcre_ml_pca_fit(ctypes.byref(t_X), K, ctypes.byref(t_W), ctypes.byref(t_mean), arena)
    assert status == 0, f"rcre_ml_pca_fit returned {status}"

    rcre_mean = np.ctypeslib.as_array(ctypes.cast(t_mean.data, ctypes.POINTER(ctypes.c_float)), shape=(D,))
    mean_diff = np.max(np.abs(rcre_mean - sk_pca.mean_))
    assert mean_diff < 1e-4, f"PCA mean diff {mean_diff} too large!"

    rcre_W = np.ctypeslib.as_array(ctypes.cast(t_W.data, ctypes.POINTER(ctypes.c_float)), shape=(K, D))

    # Eigenvector sign check: dot product absolute value should be ~ 1.0
    for k in range(K):
        dot_sim = np.abs(np.dot(rcre_W[k], sk_pca.components_[k]))
        print(f"     Component {k} cosine similarity with Scikit-Learn: {dot_sim:.6f}")
        assert dot_sim > 0.99, f"Component {k} direction divergence!"

    rcre.rcre_arena_destroy(arena)
    print("     ✓ PCA principal components parity verified against Scikit-Learn")

# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("🛡️  RCRE Numerical Parity Harness against OpenCV & Scikit-Learn")
    print("=" * 70)

    rcre = load_rcre()
    test_color_parity(rcre)
    test_gaussian_blur_parity(rcre)
    test_sobel_parity(rcre)
    test_ols_parity(rcre)
    test_kmeans_parity(rcre)
    test_pca_parity(rcre)

    print("=" * 70)
    print("🎉 ALL LAYER 3 NUMERICAL PARITY GATES PASSED (OpenCV & Scikit-Learn)!")
    print("=" * 70)

if __name__ == "__main__":
    main()
