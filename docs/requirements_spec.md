# RCRE System Requirements Specification (SRS)
## Version 1.0.0 — Recep's Code Run Everywhere

---

## 1. Introduction & Scope

This document defines the formal software requirements for **RCRE (Recep's Code Run Everywhere)**, a unified, zero-dependency C23 engine combining Computer Vision and Machine Learning. The specifications here govern the core library, optional hardware companions, SIMD backends, and platform abstractions.

---

## 2. Requirement Numbering Taxonomy

| Prefix | Subsystem Description |
| :--- | :--- |
| `MEM-REQ` | Memory Management & Arena Allocators |
| `TNS-REQ` | Tensor Engine & ShapeTracker |
| `ADP-REQ` | Input Data Recognition & Adaptation Pipeline |
| `THR-REQ` | Concurrency & Work-Stealing Threadpool |
| `HAL-REQ` | Hardware Abstraction Layer & SIMD Dispatch |
| `CV-REQ`  | Computer Vision Algorithmic Kernels |
| `ML-REQ`  | Machine Learning & Vector Search Kernels |
| `CAM-REQ` | Companion Hardware Video Capture (`rcre_cam.h`) |
| `WSM-REQ` | WebAssembly Compilation & Web Runtime |
| `TEL-REQ` | Educational Telemetry & Step-by-Step Tracing |

---

## 3. Subsystem Requirements

### 3.1 Memory Management (`MEM-REQ`)
* **MEM-REQ-01 (Arena Allocation)**: The memory subsystem must provide a contiguous linear allocator (`RCRE_Arena`) capable of allocating arbitrary byte sizes with deterministic alignment (8, 16, 32, and 64 bytes).
* **MEM-REQ-02 (Zero Hot-Path Heap)**: Compute and inference functions must never call `malloc()`, `calloc()`, or `realloc()`. Temporary buffers must be supplied via a scratchpad `RCRE_Arena*`.
* **MEM-REQ-03 (Bulk Free)**: The arena must support an $\mathcal{O}(1)$ reset operation (`rcre_arena_reset()`) by resetting its internal offset pointer back to zero without freeing the underlying OS page allocation.
* **MEM-REQ-04 (Storage Reference Counting)**: For heap-backed tensors that outlive local scopes, `RCRE_Storage` must provide thread-safe atomic reference counting (`ref_count`) using C23 `<stdatomic.h>`.

### 3.2 Tensor Engine & ShapeTracker (`TNS-REQ`)
* **TNS-REQ-01 (Dimensionality)**: Support tensors with rank $1 \le \text{ndim} \le 4$.
* **TNS-REQ-02 (Data Types)**: Support four fundamental data types:
  * `DTYPE_U8`: 8-bit unsigned integer (image pixels, quantized bins).
  * `DTYPE_I32`: 32-bit signed integer (class labels, indices).
  * `DTYPE_F16`: 16-bit half-precision IEEE 754-2008 float (edge AI weights).
  * `DTYPE_F32`: 32-bit single-precision float (standard compute).
* **TNS-REQ-03 (Strided Addressing)**: Element access at index $(i_0, i_1, i_2, i_3)$ must resolve to offset:
  $$\text{byte\_offset} = \sum_{k=0}^{\text{ndim}-1} (i_k \times \text{strides}[k]) \times \text{sizeof(dtype)}$$
* **TNS-REQ-04 (Zero-Copy Views)**: `rcre_tensor_slice()`, `rcre_tensor_reshape()`, and `rcre_tensor_permute()` must complete in $\mathcal{O}(1)$ time by updating `shape`, `strides`, and `data` pointer offset without copying buffer payloads.
* **TNS-REQ-05 (Contiguity Verification)**: Must provide `rcre_tensor_is_contiguous(const RCRE_Tensor* t)` to verify if row-major memory continuity holds before dispatching to BLAS or DMA.

### 3.3 Input Recognition & Adaptation Pipeline (`ADP-REQ`)
* **ADP-REQ-01 (Explicit Specification)**: Provide `RCRE_PreprocessSpec` defining target shape, target layout ($HWC \leftrightarrow CHW$), normalization mean/std vectors, and aspect-ratio preservation.
* **ADP-REQ-02 (Aspect-Ratio Letterboxing)**: When `preserve_aspect = true`, automatically compute optimal scaling factor, center the source image within target canvas, and fill border margins with `pad_value`.
* **ADP-REQ-03 (Fused Normalization & Quantization)**: Combine color-space convert, resize, float-cast, and mean/std normalization into a single loop pass over pixels to avoid intermediate RAM roundtrips.

### 3.4 Concurrency & Threading (`THR-REQ`)
* **THR-REQ-01 (Zero External Dependencies)**: Implement threading via standard C23 `<threads.h>` (with fallback to POSIX `pthreads` on platforms lacking C23 threads). OpenMP is strictly prohibited in the core engine.
* **THR-REQ-02 (Work-Stealing Architecture)**: Worker threads must maintain local lock-free ring-buffers. Idle workers must steal tasks from victim threads using atomic compare-and-swap (CAS).
* **THR-REQ-03 (2D Image Tile Partitioning)**: Provide `rcre_parallel_for_2d()` dividing image heights and widths into cache-friendly rectangular tiles ($64 \times 64$ or $128 \times 128$).
* **THR-REQ-04 (NUMA Awareness)**: Support thread affinity binding on multi-socket systems (Intel Xeon, AMD Threadripper) to prevent cross-NUMA interconnect bus saturation.

### 3.5 Hardware Abstraction Layer & SIMD Dispatch (`HAL-REQ`)
* **HAL-REQ-01 (Dynamic Probing)**: At startup (`rcre_init()`), probe the host CPU features via `cpuid` (x86) or `sysctl`/HWCAP (ARM) and populate `RCRE_HALTable`.
* **HAL-REQ-02 (Zero Dynamic Branching in Kernels)**: Kernel dispatches must execute through pre-resolved function pointers (`rcre_hal.gemm_f32(...)`) without conditional `if (has_avx512)` checks inside loops.
* **HAL-REQ-03 (Micro-Kernel Primitives)**:
  * `gemm_f32`: General matrix multiply ($C = \alpha A B + \beta C$).
  * `conv2d_3x3_f32`: 2D direct convolution for $3 \times 3$ kernels.
  * `pointwise_binary_f32`: Vector Add, Sub, Mul, Div, Max, Min.
  * `reduce_sum_f32` / `reduce_max_f32`: Horizontal vector reductions.
* **HAL-REQ-04 (Architecture Isolation)**: SIMD code must reside in strictly isolated source files compiled with distinct compiler flags:
  * `kernels_scalar.c`: Generic `-O3`.
  * `kernels_neon.c`: `-march=armv8-a+simd`.
  * `kernels_avx2.c`: `-mavx2 -mfma`.
  * `kernels_avx512.c`: `-mavx512f -mavx512bw -mavx512vl`.
  * `kernels_wasm.c`: `-msimd128`.

### 3.6 Computer Vision Algorithmic Kernels (`CV-REQ`)
* **CV-REQ-01 (Color Transformations)**: RGB $\leftrightarrow$ BGR, RGB $\leftrightarrow$ Grayscale, RGB $\leftrightarrow$ HSV.
* **CV-REQ-02 (Spatial Interpolation)**: Bilinear and Nearest-Neighbor resizing.
* **CV-REQ-03 (Linear Stencils)**: Separable Gaussian Blur, Box Blur, Sobel horizontal/vertical gradients.
* **CV-REQ-04 (Non-Linear Filters)**: CLAHE (with configurable grid tiles and clip limits), Fast Guided Filter, Fast Bilateral Filter.
* **CV-REQ-05 (Morphology & Segmentation)**: Otsu automatic global thresholding, Adaptive local thresholding, Dilation, Erosion ($3 \times 3$ rectangular structuring elements).

### 3.7 Machine Learning & Vector Search (`ML-REQ`)
* **ML-REQ-01 (Continuous Linear Solvers)**:
  * Ordinary Least Squares via Cholesky decomposition or QR solver.
  * Logistic Regression with Cross-Entropy Loss via L-BFGS or SGD.
  * Soft-Margin Linear Support Vector Machine (SVM) via Pegasos subgradient descent.
* **ML-REQ-02 (Clustering & Dimensionality)**:
  * k-Means with k-Means++ initialization.
  * Principal Component Analysis (PCA) via power iteration.
* **ML-REQ-03 (Modern Tabular GBDT Inference)**:
  * Fast inference runtime traversing flat array-serialized decision trees.
  * Support for 8-bit quantized feature histograms ($U8$ feature bins).
* **ML-REQ-04 (Vector Similarity Index)**:
  * Cosine and Euclidean squared distance matrix computation.
  * Hierarchical Navigable Small World (HNSW) graph index for high-dimensional approximate nearest neighbor search.

### 3.8 Companion Video Capture Module (`CAM-REQ`)
* **CAM-REQ-01 (Physical File Isolation)**: Camera capturing must reside exclusively in `include/extras/rcre_cam.h` and `src/extras/cam_*.c`. Core `rcre.h` must compile without referencing any camera header.
* **CAM-REQ-02 (Driver Implementations)**:
  * Linux: V4L2 using `VIDIOC_REQBUFS` and `mmap()`.
  * macOS: AVFoundation capturing frames into `CVPixelBufferRef`.
  * Web: WebRTC / HTML5 Canvas capture via Emscripten JavaScript bridge.
* **CAM-REQ-03 (Zero-Copy Frame Egress)**: Video frames must be wrapped directly into an `RCRE_Tensor` pointing to driver-mapped memory without intermediate copying.

### 3.9 WebAssembly Compilation & Web Runtime (`WSM-REQ`)
* **WSM-REQ-01 (Emscripten Parity)**: The complete core library must compile to `rcre.wasm` using Emscripten with `-s STANDALONE_WASM=1` and `-msimd128`.
* **WSM-REQ-02 (JavaScript FFI)**: Provide `rcre_bridge.js` exposing memory allocation, tensor loading, and model execution to browser frontends.
* **WSM-REQ-03 (Payload Constraint)**: Uncompressed `rcre.wasm` binary must not exceed 1.2 MB (target < 600 KB gzip).

### 3.10 Educational Telemetry & Step-by-Step Hooks (`TEL-REQ`)
* **TEL-REQ-01 (Compile-Time Toggle)**: Telemetry instrumentation must be gated behind `#ifdef RCRE_ENABLE_TELEMETRY`. Production builds with telemetry disabled must incur 0% runtime penalty.
* **TEL-REQ-02 (Event Payload)**: When enabled, `rcre_step_callback` must pass an `RCRE_StepInfo` struct containing the operation name, monotonic step counter, active tensor view, active cache tile indices, and decision split metadata.
* **TEL-REQ-03 (Frontend Visualizer Serialization)**: Provide a utility to serialize step events into JSON streams for replay in the Three.js and Canvas web visualizers.
