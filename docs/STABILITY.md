# RCRE Module Stability Tiers & Platform Support Matrix

This document defines the official platform targets, module stability tiers, security boundaries, and verification contracts for **RCRE (Recep's Code Run Everywhere)**.

---

## 1. Target Platforms

RCRE is engineered specifically for high-performance scientific computing, AI/ML workloads, and edge deployments:

| Tier | Platforms | SIMD / Compute Path | CI / Verification Status |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Primary)** | **Apple Silicon (macOS ARM64)** | ARM NEON 128-bit + optional Apple Metal (`rcre_gpu.h`) | Full unit tests, ASan+UBSan, fuzz suite, Python parity, benchmarks |
| **Tier 1 (Primary)** | **Linux x86_64 (Scientific / HPC)** | AVX-512F / AVX2 + FMA (`gcc` & `clang`) | Full unit tests, ASan+UBSan, fuzz suite, Python parity |
| **Tier 1 (Primary)** | **Linux ARM64 Edge (Raspberry Pi, NVIDIA Jetson)** | ARM NEON 128-bit (`gcc` / `clang`, `epoll` extras) | Standard POSIX C23 + NEON HAL |
| **Tier 2 (Supported)** | **WebAssembly (Emscripten)** | WASM SIMD128 + optional WebGPU bridge | Browser playground & educational visualizers |
| **Tier 2 (Supported)** | **Freestanding Embedded (`-DRCRE_FREESTANDING`)** | Scalar / NEON (`rcre_arena_from_buffer`, zero `malloc`) | Verified freestanding compile profile |

> **Scope Note**: Native Windows/MSVC is intentionally out of primary scope so engineering focus remains on Apple Silicon, Linux HPC clusters, and ARM64 edge hardware.

---

## 2. Module Stability Tiers

### Tier 1 — Production-Stable Core

Battle-tested, zero-`malloc` in hot paths, ASan+UBSan verified, and (where applicable) numerically verified against OpenCV / NumPy / SciPy / Scikit-Learn reference oracles:

| Subsystem | Header(s) | Guarantees | Verification |
| :--- | :--- | :--- | :--- |
| **Memory & Views** | `rcre.h` (`RCRE_Arena`, `RCRE_Tensor`, `RCRE_Slice`) | 64-byte alignment, O(1) arena rewind, 0 heap allocs in loops, 8D strided broadcasting | `test_arena`, `test_slice`, `test_tensor_strides`, `test_adapt_spec` |
| **Safe Strings & UTF-8** | `rcre_str.h` | Zero `malloc`, strict RFC 3629 NEON/AVX2 UTF-8, `RCRE_StrBuf`, `RCRE_StrPool`, `RCRE_BAN_UNSAFE_CSTR` | `test_str` (78k assertions), `fuzz_str` |
| **Hardware HAL & ThreadPool** | `rcre.h` (`rcre_hal`, `RCRE_ThreadPool`) | Race-free atomic `rcre_hal_init()`, NEON/AVX2/AVX-512/WASM dispatch, QoS hints | `test_hal`, `test_threadpool`, TSan-clean |
| **Scientific I/O** | `rcre_io.h` | Zero-copy `mmap`, `.npy`, `.safetensors`, SIMD CSV, `RCRZ`, DLPack & Arrow bridges | `test_io`, `fuzz_io`, `bench_io` |
| **HPC Math (H0–H4)** | `rcre.h`, `rcre_sparse.h` | F64 BLAS/GEMM, Kahan-Neumaier, PCG64/Sobol', F64 FFT/Bluestein, LU/Chol/QR/SYEV, PCG/BiCGStab/GMRES, Multigrid | `test_linalg`, `test_rng`, `test_fft_f64`, `test_direct`, `test_h4`, `test_sparse` |
| **Classical ML & Search** | `rcre.h`, `rcre_stat.h`, `rcre_prep.h`, `rcre_metric.h` | Flat GBDT, HNSW, e-Means (0-ULP parity), k-Means, PCA, OLS/L1/SVM, Random Forest | `test_gbdt`, `test_hnsw`, `test_emeans`, `test_cluster`, `test_ml_linear`, `test_stat`, `test_metric`, `test_against_scipy_opencv.py` |
| **HPC NLP Engine** | `rcre_nlp.h` | Zero-malloc BM25 (incl. reusable `RCRE_BM25Scratch`), TF-IDF->CSR, Aho–Corasick DFA, Myers64, NMF, LDA, Sparse NB/Softmax, TextRank | `test_nlp`, `fuzz_nlp`, `bench_nlp` |
| **DataFrames & Profiler** | `rcre_df.h`, `rcre_prof.h` | Columnar DataFrames (Arrow bitmaps, Radix Sort), `<7 ns` hardware cycle profiler | `test_df`, `test_profiler` |
| **Core Computer Vision** | `rcre.h` (`src/cv/color,filter,geom,features,homography,optical_flow`) | 0-LSB OpenCV grayscale parity, bilinear resize, Sobel/Blur, ORB/Harris, RANSAC homography | `test_cv_*`, `test_against_scipy_opencv.py` |

### Tier 2 — Preview / Active Hardening

Functional and covered by unit tests and fuzz regression suites, but either evolving in API or parsing complex external binary formats where continuous fuzzing is recommended before exposing to untrusted public internet inputs:

| Subsystem | Header(s) | Status & Guidance |
| :--- | :--- | :--- |
| **Zero-Dep Image Codecs** | `rcre_codec.h` (`codec.c`: PNG, JPEG, BMP, PPM) | Covered by `fuzz_codec` (ASan+UBSan). Suitable for trusted/internal pipelines; run continuous `libFuzzer` if ingesting untrusted public uploads. |
| **ONNX Subset Runner** | `rcre_codec.h` (`onnx_subset.c`) | Covered by `fuzz_onnx` (ASan+UBSan). Supports linear/Conv2D/ReLU/Pool/Softmax subset; validate model provenance. |
| **GGUF & LLM Engine** | `rcre_llm.h`, `rcre_transformer.h` | Covered by `fuzz_gguf` (ASan+UBSan) & `test_transformer_*`. Active development as quantization & attention kernels expand. |
| **Autograd & NN Training** | `rcre_autograd.h` | Tape VJP autograd + AdamW/SGD/Lion/Adafactor. Verified via `test_autograd_*` and `test_nn_train`. |
| **OS Extras** | `extras/rcre_http.h`, `extras/rcre_cam.h`, `extras/rcre_gpu.h`, `extras/rcre_window.h` | Isolated OS-dependent companions (`kqueue`/`epoll`, AVFoundation/V4L2, Metal). Intentionally excluded from `include/rcre.h` and freestanding builds. |
| **Non-C Bindings** | `bindings/{python,rust,go,zig,julia}` | Python `ctypes` bindings are verified in CI (`test_python_bindings.py`); Rust/Go/Zig/Julia bindings are starter FFI wrappers. |

---

## 3. Verification & ABI Policy

* **Static Linking & Single-Header (`librcre.a`, `dist/rcre.h`)**: Recommended production distribution mode. Public structs (`RCRE_Arena`, `RCRE_Tensor`, `RCRE_BM25Index`) are exposed by value for zero-overhead stack/inline allocation.
* **Shared Library (`librcre.dylib` / `librcre.so`)**: Versioned via `RCRE_ABI_VERSION` (`rcre_abi_version()`). Callers linking dynamically should assert `rcre_abi_version() == RCRE_ABI_VERSION` at startup.
* **Security Boundary**: Run `make test-sanitizers` and `make test-fuzz` (covering `fuzz_io`, `fuzz_str`, `fuzz_nlp`, `fuzz_codec`, `fuzz_gguf`, and `fuzz_onnx`) before tagging releases.
