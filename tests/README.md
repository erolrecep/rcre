# RCRE Testing & Verification Suite

<p align="center">
  <strong>A 4-layer verification matrix guaranteeing mathematical correctness, bit-level parity, zero memory leaks, and memory safety across all platforms.</strong>
</p>

---

## 🛡️ The RCRE Verification Philosophy

Writing high-performance systems code in pure **ISO C23** requires an uncompromising approach to correctness. Because RCRE operates directly on raw memory buffers, linear memory arenas, hardware SIMD vector registers, and dynamic execution tapes, standard compiler warnings are insufficient.

Every test in this directory exists to enforce **concrete engineering invariants**:
1. **Zero Heap Allocations in Hot Paths**: Loops must run entirely on linear scratchpad memory arenas with deterministic $\mathcal{O}(1)$ bulk rewinding.
2. **Sanitizer Cleanliness**: Every unit test must compile and execute under Clang **AddressSanitizer (ASan)** and **UndefinedBehaviorSanitizer (UBSan)** with **0 leaks, 0 heap-use-after-free, and 0 undefined behavior**.
3. **Bit-Exact & Numerical Parity**: Low-level algorithms must match established industry standards (OpenCV, Scikit-Learn, NumPy) within strict tolerances ($\epsilon \le 10^{-4}$ for floats, **0 LSB discrepancy** for 15-bit fixed-point color conversions).
4. **Graceful Failure & Boundary Protection**: Edge cases (undersized destination buffers, malformed/ragged CSV inputs, arena exhaustion) must fail safely with explicit `nullptr`/`false` rather than crashing or causing silent cross-arena memory corruption.
5. **Contract Pinning & Roadmaps**: Unresolved architectural boundaries (such as recursion depth or dual-write semantics) are pinned by dedicated tests documenting current behavior and signposted with `CONTINUE HERE` for future development.

---

## 📁 Test Suite Catalog & Purpose

The test directory contains **18 test suites** spanning over **77,500 assertions**, organized into distinct subsystems:

```
tests/
├── Core Systems & Memory Architecture
│   ├── test_arena.c                 # Linear memory arena, 64-byte alignment, temp checkpoints, dual-arena context
│   ├── test_tensor_strides.c        # N-D strided tensor layout, ShapeTracker views, zero-copy transpose/slicing
│   ├── test_adapt_spec.c            # Input adaptation, letterboxing, normalization, dtype validation
│   └── test_threadpool.c            # Multicore work-stealing pool, 1D/2D tile partitioning, teardown synchronization
│
├── Computer Vision, Signal Processing & Hardware Acceleration
│   ├── test_cv_filters.c            # Rec.601 RGB-to-gray (0 LSB), separable Gaussian blur, Sobel derivatives
│   ├── test_fft.c                   # 1D & 2D Fast Fourier Transform, FFTShift, Gaussian spectral filtering
│   ├── test_hal.c                   # Dynamic CPU feature probing, dispatch table, ARM NEON / AVX2 / WASM SIMD
│   └── test_cam.c                   # Hardware camera companion (AVFoundation macOS, V4L2 Linux, synthetic dummy)
│
├── Classical Machine Learning Primitives
│   ├── test_ml_linear.c             # OLS regression (Cholesky/QR), Logistic regression, Pegasos primal SVM
│   ├── test_cluster.c               # k-Means, k-Means++, DBSCAN density clustering, PCA dimensionality reduction
│   ├── test_gbdt.c                  # Flat-array tabular Gradient Boosted Decision Tree (GBDT) inference
│   ├── test_hnsw.c                  # Hierarchical Navigable Small World (HNSW) vector search (L2 & Cosine)
│   └── test_nn.c                    # Inference CNN layers (Conv2D, Depthwise, Pointwise, Fused BN, ReLU6, Softmax)
│
├── Differentiable Computing & Autograd
│   ├── test_autograd_scalar.c       # Micrograd-style reverse-mode scalar DAG autograd, Graphviz DOT export
│   ├── test_autograd_tensor.c       # Dynamic tensor tape autograd, VJPs, tape growth across rewinds, shape guards
│   └── test_nn_train.c              # Loss functions, AdamW optimization, 1D target split, long CSV line parsing
│
├── Merger Hardening & Gap-Pinning
│   └── test_merger_report.c         # 11 adversarial regression locks & contract-pinning tests from merger review
│
└── Integration, Packaging & Python Parity
    ├── test_amalgamated.c           # Single-header dist/rcre.h standalone compilation self-test
    ├── test_against_scipy_opencv.py # 9 Automated numerical parity gates against OpenCV, Scikit-Learn & NumPy
    └── test_python_bindings.py      # Python ctypes FFI bindings validation
```

---

## 🔬 Subsystem Deep Dive & Rationale

### 1. Core Systems & Memory Architecture

* **[`test_arena.c`](test_arena.c)**:
  * *Purpose*: Validates the bedrock of RCRE's memory management: monotonic linear allocation, 64-byte hardware cache-line alignment, and scoped scratchpad checkpoints.
  * *Key Invariants*: Asserts that `rcre_arena_alloc()` aligns all pointers to 64 bytes for SIMD operations, tests `rcre_arena_temp_begin()` / `rcre_arena_temp_end()` for $\mathcal{O}(1)$ scoped rewinds, and validates the `RCRE_GraphCtx` dual-arena system separating persistent weights from ephemeral activation memory.
* **[`test_tensor_strides.c`](test_tensor_strides.c)**:
  * *Purpose*: Validates the `tinygrad`-style ShapeTracker architecture.
  * *Key Invariants*: Proves that matrix transpositions, axis permutations, contiguous sub-slicing, and reshaping operate in $\mathcal{O}(1)$ time with **zero data copying** solely by manipulating stride vectors and pointer offsets.
* **[`test_adapt_spec.c`](test_adapt_spec.c)**:
  * *Purpose*: Validates real-world input adaptation pipelines.
  * *Key Invariants*: Tests letterboxing (aspect-ratio preserving padding with constant borders), pixel normalization ($[0, 255] \to [0.0, 1.0]$ or $[-1.0, 1.0]$), and channel-order transpositions ($HWC \leftrightarrow CHW$).
* **[`test_threadpool.c`](test_threadpool.c)**:
  * *Purpose*: Validates lightweight multicore concurrency without runtime bloat.
  * *Key Invariants*: Tests 1D and 2D parallel spatial domain decomposition across hardware threads. Guarantees that worker threads join and terminate cleanly before deallocation, eliminating thread-destruction race conditions.

---

### 2. Computer Vision, Signal Processing & HAL

* **[`test_cv_filters.c`](test_cv_filters.c)**:
  * *Purpose*: Tests low-level spatial filtering and pixel math.
  * *Key Invariants*: Verifies 15-bit fixed-point Rec.601 RGB-to-grayscale conversion against OpenCV (`0 LSB` discrepancy), verifies 2D separable Gaussian convolution mathematical equivalence ($O(K^2) \to O(2K)$), and asserts Sobel gradient magnitude accuracy.
* **[`test_fft.c`](test_fft.c)**:
  * *Purpose*: Validates 1D and 2D Cooley-Tukey Fast Fourier Transforms.
  * *Key Invariants*: Tests forward FFT and inverse IFFT reconstruction identity ($\text{IFFT}(\text{FFT}(X)) \approx X$ with $\epsilon < 10^{-6}$), 2D quadrant `fftshift`, and frequency-domain Gaussian lowpass/highpass filtering.
* **[`test_hal.c`](test_hal.c)**:
  * *Purpose*: Validates the Hardware Abstraction Layer (HAL) and dynamic dispatch table.
  * *Key Invariants*: Tests runtime CPU feature detection (ARM NEON, x86 AVX2/FMA, WebAssembly SIMD128), asserts single-cycle indirect function pointer dispatch, and validates SIMD dot products, AXPY vector scaling, and cache-blocked GEMM against scalar reference routines.
* **[`test_cam.c`](test_cam.c)**:
  * *Purpose*: Validates real-time video ingestion.
  * *Key Invariants*: Tests macOS AVFoundation (`CVPixelBuffer` zero-copy wrapping), Linux V4L2 (`mmap`), and deterministic synthetic test generators for headless CI.

---

### 3. Classical Machine Learning Primitives

* **[`test_ml_linear.c`](test_ml_linear.c)**:
  * *Purpose*: Tests continuous optimization and differentiable linear models.
  * *Key Invariants*: Validates Ordinary Least Squares (OLS) closed-form normal equations, Logistic Regression with cross-entropy loss, and the Pegasos primal subgradient Support Vector Machine (SVM).
* **[`test_cluster.c`](test_cluster.c)**:
  * *Purpose*: Tests clustering and dimensionality reduction algorithms.
  * *Key Invariants*: Validates standard $k$-Means, $k$-Means++ initialization determinism, DBSCAN density-based noise rejection, and Principal Component Analysis (PCA) via power-iteration singular value decomposition.
* **[`test_gbdt.c`](test_gbdt.c)**:
  * *Purpose*: Tests high-throughput tabular inference.
  * *Key Invariants*: Verifies flat-array decision tree ensemble traversal using cache-friendly contiguous node arrays and quantized split thresholds.
* **[`test_hnsw.c`](test_hnsw.c)**:
  * *Purpose*: Tests high-dimensional vector search.
  * *Key Invariants*: Validates Hierarchical Navigable Small World (HNSW) proximity graph indexing, greedy layer traversal, and SIMD-accelerated Euclidean L2 and Cosine distance metrics.
* **[`test_nn.c`](test_nn.c)**:
  * *Purpose*: Tests classical feedforward neural network layers.
  * *Key Invariants*: Validates standard Conv2D, depthwise separable convolutions, fused convolution-batchnorm folding, ReLU6 clipping, and numerically stable Softmax with log-sum-exp stabilization.

---

### 4. Differentiable Computing & Autograd

* **[`test_autograd_scalar.c`](test_autograd_scalar.c)**:
  * *Purpose*: Tests scalar reverse-mode automatic differentiation on `RCRE_ScalarVal`.
  * *Key Invariants*: Validates forward evaluation and backward chain-rule gradient accumulation across multi-branch DAGs ($+$, $-$, $*$, $/$, $\text{pow}$), transcendental operations ($\tanh, \text{exp}, \text{log}, \text{ReLU}, \text{Sigmoid}$), and Graphviz `.dot` visual export.
* **[`test_autograd_tensor.c`](test_autograd_tensor.c)**:
  * *Purpose*: Tests high-performance dynamic tensor tape autograd on the unified `RCRE_Tensor` (v0.2.2: the separate `RCRE_AGTensor` type and bridge were removed).
  * *Key Invariants*:
    - **`test_tensor_tape_matmul_and_backward`**: Validates matrix multiplication Vector-Jacobian Products ($dA = dY \cdot B^T$ and $dB = A^T \cdot dY$) against analytical calculus.
    - **`test_tensor_tape_add_bias_and_activations`**: Validates row-broadcast bias addition and column-wise gradient reduction ($db = \sum_i dY_i$).
    - **`test_tape_grow_then_rewind`**: Critical regression test ensuring the tape's heap-owned node array survives dynamic doubling across `rcre_arena_temp_end()` activation scratchpad rewinds.
    - **`test_autograd_shape_validation`**: Validates dimension mismatch rejection, non-F32 dtype rejection, and that plain core F32 tensors (value type) feed autograd operators directly.
* **[`test_nn_train.c`](test_nn_train.c)**:
  * *Purpose*: Tests loss functions, data ingestion, and optimization loops.
  * *Key Invariants*:
    - **`test_loss_mse_and_grad`**: Verifies analytical MSE loss and gradient generation.
    - **`test_adamw_convergence_loop`**: Validates 50-epoch linear model optimization with Decoupled Weight Decay AdamW, converging from loss $1.29 \to < 0.005$ ($> 99\%$ reduction).
    - **`test_split_1d_y`**: Enforces correct handling of 1-D regression targets in `train_test_split` without uninitialized dimension reads.
    - **`test_optimizer_hyperparam_validation`**: Rejects invalid parameters (e.g., $\beta_1 = 1.0$, $\beta_2 = 1.0$, $\text{lr} \le 0$, momentum $\notin [0, 1)$) preventing division-by-zero during bias correction.
    - **`test_batch_iter_undersized_output`**: Protects against silent cross-arena memory corruptions when batch buffers are undersized.
    - **`test_csv_long_line_and_ragged_reject`**: Tests streaming CSV parser with 1,000-column rows (~7 KB lines) and rejects ragged/malformed rows.

---

### 5. Merger Review Findings — Regressions & v0.2.2 Contracts ([`test_merger_report.c`](test_merger_report.c))

This dedicated test suite maps directly to the architectural merger review findings, split into two distinct parts:

#### Part 1: Granular Regressions & Bug Fixes
* **`test_tensor_alloc_shape_zeroed_beyond_ndim`**: Verifies that `rcre_tensor_alloc()` zero-initializes the struct so dimensions $\ge \text{ndim}$ always read 0, preventing stack/arena garbage reads.
* **`test_csv_old_boundary_and_dynamic_growth`**: Stresses the CSV loader past the previous 4,096-byte limit:
  - Line length of 4,102 chars (just past old cap).
  - A 20,000-column, ~120 KB line proving unbounded dynamic line growth.
  - Final lines without a trailing newline.
* **`test_autograd_ops_reject_non_f32`**: Pins the v0.2.2 dtype gate: every autograd/optimizer entry point rejects `U8`/`I32` tensors with `NULL` (the gate formerly in `rcre_ag_from_tensor()` now lives in the ops), while core F32 tensors are accepted.
* **`test_tape_arena_exhaustion_graceful_null`**: Guarantees that when an arena runs out of memory, autograd operations fail gracefully with `nullptr` rather than crashing or causing undefined behavior, and the arena recovers normally after a reset.
* **`test_batch_iter_y_only_undersized`**: Independently isolates and tests destination buffer bounds checking on regression targets.
* **`test_unified_rng_determinism`**: Verifies that the unified XorShift64* PRNG ([`src/core/rcre_rng.h`](../src/core/rcre_rng.h)) produces bit-identical sequences across seeds and guarantees reproducible Xavier/He weight initialization.

#### Part 2: v0.2.2 Unified-Engine Contracts (previously "CONTINUE HERE" items)
* **`test_unified_core_view_feeds_autograd`**: Pins the type-unification payoff — core value-style F32 tensors feed autograd operators directly (no bridge), backward silently skips inputs without a `grad` buffer, and core view ops on training tensors (`rcre_tensor_slice` / `rcre_tensor_reshape_view`) never share the parent's grad buffer.
* **`test_scalar_deep_chain_iterative_topo`**: Pins the iterative (explicit heap frame stack) topological sort: a 100,000-deep differentiable chain differentiates correctly, so scalar AD is no longer bounded by the C call stack (safe for WASM 64 KB stacks / embedded).
* **`test_loss_grad_only_contract`**: Pins the retired dual-write contract: loss gradients land in `grad_out->grad` ONLY; the forward `.data` buffer is never modified (sentinel-pinned), whether `grad_out` is separate or `grad_out == preds`; borrowed views receive the loss value without a gradient.
* **`test_cross_entropy_clamp_and_label_validation`**: Pins the cross-entropy failure policy — out-of-range class indices clamp to the boundary class (behaviour-identical), and the new `rcre_data_validate_labels()` reports the first invalid index (or −1) for dataset hygiene.
* **`test_scalar_epoch_reuse_single_thread`**: Verifies that the now-`thread_local` visited-epoch re-initializes cleanly across multiple backward passes on the same computational graph.

---

### 6. Integration, Packaging & Python Parity

* **[`test_amalgamated.c`](test_amalgamated.c)**:
  * Compiles against the standalone single-header distribution [`dist/rcre.h`](../dist/rcre.h) in a completely clean translation unit (`#define RCRE_IMPLEMENTATION`), verifying that all internal headers, symbols, and static functions are properly scoped with zero symbol collisions.
* **[`test_against_scipy_opencv.py`](test_against_scipy_opencv.py)**:
  * Runs 9 automated numerical parity gates comparing RCRE C23 outputs directly against Python reference packages:
    1. **RGB $\to$ Grayscale**: Matches OpenCV `cv2.cvtColor` within **0 LSB**.
    2. **2D Gaussian Blur**: Matches OpenCV `cv2.GaussianBlur` within $\epsilon \le 1.8 \times 10^{-7}$.
    3. **3x3 Sobel Derivatives**: Matches OpenCV `cv2.Sobel` within $\epsilon \le 4.6 \times 10^{-5}$.
    4. **OLS Linear Regression**: Matches Scikit-Learn `LinearRegression` weights within $\epsilon \le 1.5 \times 10^{-6}$.
    5. **$k$-Means Clustering**: Matches Scikit-Learn `KMeans` centroids within $\epsilon \le 4.8 \times 10^{-7}$.
    6. **PCA Dimensionality Reduction**: Matches Scikit-Learn `PCA` components with **cosine similarity = 1.000000**.
    7. **DBSCAN Density Clustering**: Matches Scikit-Learn `DBSCAN` with **Adjusted Rand Index = 1.000000**.
    8. **2D Fast Fourier Transform**: Matches NumPy `fft2` and `ifft2` within $\epsilon \le 3.9 \times 10^{-6}$.
    9. **CNN Inference Operators**: Matches NumPy reference Conv2D and Softmax within $\epsilon \le 1.5 \times 10^{-6}$.
* **[`test_python_bindings.py`](test_python_bindings.py)**:
  * Validates the Python ctypes FFI package ([`bindings/python/`](../bindings/python/)), verifying runtime CPU detection, safe context-managed arena allocation, zero-copy NumPy array ingestion, and memory reclamation.

---

## ⚙️ The Pragmatic Unit Testing Framework ([`test_framework.h`](test_framework.h))

All C23 unit tests run on an internal, zero-external-dependency testing header:
* **Zero Allocations**: Test runner state (`RCRE_TestSuiteState`) is statically allocated, preventing interference with RCRE linear memory arenas.
* **Stack Safe**: Free of `setjmp` / `longjmp` constructs, guaranteeing clean C23 stack unwinding, threadpool safety, and zero false positives under sanitizers.
* **Sub-Millisecond Telemetry**: High-resolution monotonic timers (`clock_gettime(CLOCK_MONOTONIC)`) measure individual test execution latencies.
* **Rich Diagnostic Assertions**: Formatted macros (`ASSERT_FLOAT_NEAR`, `ASSERT_FLOAT_ARRAY_NEAR`, `ASSERT_UINT_EQ`, `ASSERT_MEM_EQ`, `ASSERT_NOT_NULL`) print clear expected-vs-actual failure diagnostics.

---

## 🚀 Execution & Command Center

Run the test matrix from the project root (`apps/rcre`):

```bash
# 🧪 Run all 18 C23 unit test suites (~77,500 assertions)
make test

# 🛡️ Run the full test matrix under Clang AddressSanitizer & UndefinedBehaviorSanitizer
make test-sanitizers

# 🐍 Run the 9 Python numerical parity gates against OpenCV, Scikit-Learn & NumPy
make test-parity

# 🎉 Run all verification layers at once (Unit Tests + Sanitizers + Parity + Amalgamation)
make test-all
```
