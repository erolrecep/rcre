# RCRE Development Roadmap & Test Verification Plan
## From Inception to Production & Academic Release

---

## 1. Phased Development Roadmap

```
+---------------------------------------------------------------------------------------------------+
| MVP 1: Foundation (Weeks 1-5)    | Memory Arenas, ShapeTracker Tensors, Input Adaptor Spec        |
+----------------------------------+----------------------------------------------------------------+
| MVP 2: Concurrency & HAL (W 6-11)| Threadpool, Scalar CV/ML Reference, Python Parity Test Harness  |
+----------------------------------+----------------------------------------------------------------+
| MVP 3: SIMD Acceleration (W12-16)| ARM NEON (RPi/Apple Silicon), AVX-2 / AVX-512, WASM SIMD       |
+----------------------------------+----------------------------------------------------------------+
| MVP 4: Modern AI & I/O (W 17-20) | GBDT Inference, HNSW Search, rcre_cam.h (V4L2/AVFoundation)    |
+----------------------------------+----------------------------------------------------------------+
| MVP 5: Web & Education (W 21-24) | WASM Playground, Three.js 3D Visualizer, Benchmarks, Packaging |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Detailed Milestone & Deliverables Breakdown

### Phase 1: Core Foundation & Memory (MVP 1 — Weeks 1 to 5)
* **Goal**: Establish the zero-allocation memory foundation and strided tensor engine.
* **Deliverables**:
  * `src/core/arena.c`: Linear arena allocator with 64-byte alignment verification.
  * `src/core/tensor.c`: Strided N-D tensor implementation with zero-copy $\mathcal{O}(1)$ slicing, transposing, and reshaping.
  * `src/core/adapt.c`: Input preprocessing adaptor (`rcre_tensor_adapt`) supporting letterboxing, layout conversion ($HWC \leftrightarrow CHW$), and normalization.
  * `tests/test_tensor_strides.c`: Automated verification of 1D through 4D stride math.
* **Exit Gate / Definition of Done**: Zero memory leaks verified under Valgrind; 100% test pass on arbitrary slicing and transpositions.

### Phase 2: Concurrency & Pure C23 Reference Algorithms (MVP 2 — Weeks 6 to 11)
* **Goal**: Enable lock-free multicore processing and complete the pure C23 reference mathematical suite.
* **Deliverables**:
  * `src/core/threadpool.c`: C23 `<threads.h>` lock-free work-stealing threadpool.
  * `src/cv/*.c`: Scalar implementations of Grayscale, Bilinear Resize, Filter2D, Gaussian Blur, Sobel, CLAHE.
  * `src/ml/*.c`: Scalar implementations of OLS, Logistic Regression, Linear SVM (Pegasos subgradient descent), k-Means.
  * `tests/test_against_scipy_opencv.py`: Ground-truth numerical parity test suite.
* **Exit Gate / Definition of Done**: Every scalar algorithm produces numerical outputs matching OpenCV and Scikit-Learn within $\epsilon \le 10^{-5}$.

### Phase 3: Hardware Abstraction Layer & SIMD Optimization (MVP 3 — Weeks 12 to 16)
* **Goal**: Accelerate compute primitives to reach native hardware speeds on Edge and HPC.
* **Deliverables**:
  * `src/hal/cpu_probe.c`: Runtime feature detection for x86 and ARM.
  * `src/hal/kernels_neon.c`: ARM NEON 128-bit vectorization (Raspberry Pi 4/5, Apple Silicon).
  * `src/hal/kernels_avx2.c`: x86 256-bit AVX2 + FMA vectorization.
  * `src/hal/kernels_avx512.c`: x86 512-bit AVX-512 vectorization (Xeon, Threadripper).
  * `src/hal/kernels_wasm.c`: WebAssembly 128-bit SIMD via Emscripten.
* **Exit Gate / Definition of Done**: At least $4\times$ speedup over pure scalar baseline on target architectures without introducing instruction faults (`SIGILL`).

### Phase 4: Modern Algorithms & Camera Hardware I/O (MVP 4 — Weeks 17 to 20)
* **Goal**: Deliver modern production algorithms and real-time camera ingestion.
* **Deliverables**:
  * `src/ml/gbdt_infer.c`: Flat-array decision tree inference engine supporting $U8$ histogram bins.
  * `src/ml/hnsw.c`: Hierarchical Navigable Small World vector search index.
  * `include/extras/rcre_cam.h` & `src/extras/cam_*.c`: V4L2 (Linux) and AVFoundation (macOS) zero-copy camera drivers.
  * `examples/webcam_edge_filter.c`: Standalone demo capturing live frames and applying real-time filters at $>60$ FPS.
* **Exit Gate / Definition of Done**: Live 1080p webcam streaming running real-time edge detection and GBDT inference on a Raspberry Pi 4 without frame drops.

### Phase 5: WebAssembly, Educational Visualizers & Packaging (MVP 5 — Weeks 21 to 24)
* **Goal**: Publish the web playground, interactive animations, and final library distribution.
* **Deliverables**:
  * `web/playground.html`: In-browser WASM playground with live WebRTC camera capture.
  * `web/learn/*.html`: Interactive Three.js 3D tensor stride visualizer and 2D canvas sliding stencil visualizers.
  * `rcre.h`: Single-header amalgamated distribution file.
  * FFI bindings: Python (`py_rcre`) and Rust (`rcre-rs`).
* **Exit Gate / Definition of Done**: Live deployment on GitHub Pages; 100% passing cross-platform CI on GitHub Actions.

---

## 3. Comprehensive Verification & Testing Strategy

```
+---------------------------------------------------------------------------------+
|                                 TEST HARNESS LAYERS                             |
+---------------------------------------------------------------------------------+
| Layer 4: System Benchmarks (GFLOPS, Throughput FPS, Latency p99)                |
| Layer 3: Python Numerical Parity (Assert vs. Scikit-Learn / OpenCV)             |
| Layer 2: Sanitizers & Memory Checks (ASan, UBSan, TSan, Valgrind)               |
| Layer 1: Unit & Boundary Tests (Strides, Alignment, Edge-cases)                 |
+---------------------------------------------------------------------------------+
```

### 3.1 Layer 1: Unit & Boundary Testing (`tests/test_tensor_strides.c`)
* **Striding Invariants**: Verify that slicing a $4\text{D}$ tensor does not alter the storage pointer of unselected elements.
* **Alignment Enforcement**: Assert that all pointers returned by `rcre_arena_alloc()` satisfy:
  $$\text{ptr} \pmod{64} == 0$$
* **Boundary Conditions**: Test zero-sized dimensions, $1 \times 1$ tensors, negative strides, and maximum dimension limits.

### 3.2 Layer 2: Memory & Concurrency Sanitizers
All C23 tests must pass without warnings or errors under three distinct compiler sanitizer profiles:
1. **AddressSanitizer (ASan) & LeakSanitizer (LSan)**:
   ```bash
   clang -std=c2x -fsanitize=address,leak -O1 -g tests/test_tensor.c src/*.c
   ```
2. **UndefinedBehaviorSanitizer (UBSan)**:
   ```bash
   clang -std=c2x -fsanitize=undefined -O1 -g tests/test_tensor.c src/*.c
   ```
3. **ThreadSanitizer (TSan)**:
   ```bash
   clang -std=c2x -fsanitize=thread -O1 -g tests/test_threadpool.c src/core/threadpool.c
   ```
4. **Valgrind Memcheck**:
   ```bash
   valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./test_runner
   ```

### 3.3 Layer 3: Numerical Parity with Scikit-Learn & OpenCV
An automated Python test harness (`tests/test_against_scipy_opencv.py`) generates random inputs, runs both the reference Python library and the compiled C23 library via ctypes, and compares results:

```python
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression

def test_bilinear_resize_parity():
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    expected = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    actual = rcre_cv_resize_bilinear(img, (224, 224))
    np.testing.assert_allclose(actual, expected, atol=1.0) # Within rounding tolerance

def test_logistic_regression_parity():
    X = np.random.randn(1000, 20).astype(np.float32)
    y = np.random.randint(0, 2, (1000,)).astype(np.int32)
    sk_model = LogisticRegression(solver='lbfgs', max_iter=200).fit(X, y)
    rcre_weights = rcre_ml_logistic_train(X, y, max_iter=200)
    np.testing.assert_allclose(rcre_weights, sk_model.coef_, atol=1e-4)
```

### 3.4 Layer 4: Performance Benchmarks (`benchmarks/`)
* **GEMM FLOPS**: Measure achieved GFLOPS vs. theoretical hardware peak across matrix sizes $N \in [64, 128, 256, 512, 1024, 2048]$.
* **CV Filter Throughput**: Measure MegaPixels/sec for $3 \times 3$ Gaussian and Sobel filters across $1080\text{p}$ and $4\text{K}$ video frames.
* **GBDT Inference Latency**: Measure single-sample latency ($p50, p99$) and batch throughput (samples/sec).
* **Multi-Core Scaling Efficiency**: Evaluate speedup curve from 1 core to $N$ cores on AMD Threadripper ($64\text{ cores}$) to verify absence of thread contention.
