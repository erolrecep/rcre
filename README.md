# RCRE — Recep's Code Run Everywhere

<p align="center">
  <strong>A single-header, zero-dependency C23 engine unifying modern Computer Vision and Classical Machine Learning across Edge, HPC, and WebAssembly.</strong>
</p>

<p align="center">
  <a href="https://erolrecep.github.io/rcre/"><img src="https://img.shields.io/badge/Demo-Live_Playground-00d2ff?logo=github&style=flat" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/Language-ISO_C23-00599C?logo=c&style=flat" alt="C23 Standard">
  <img src="https://img.shields.io/badge/Sanitizers-ASan_%7C_UBSan-success?style=flat" alt="Sanitizers Clean">
  <img src="https://img.shields.io/badge/OpenCV_Parity-0_LSB-brightgreen?logo=opencv&logoColor=white&style=flat" alt="OpenCV Parity">
  <img src="https://img.shields.io/badge/Loop_Allocs-0-orange?style=flat" alt="Zero Heap Allocations">
  <a href="https://github.com/erolrecep/rcre-book"><img src="https://img.shields.io/badge/Textbook-rcre--book-red?logo=gitbook&style=flat" alt="Official Textbook"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL_3.0_%2F_Commercial-blue?style=flat" alt="AGPL-3.0 / Commercial License"></a>
</p>

---

## 🚀 Key Highlights

* **Pure C23 Zero-Dependency Core**: Compiles cleanly with standard ISO C23 (`clang -std=c2x` / `gcc -std=c23`) using modern attributes `[[nodiscard]]`, `constexpr`, `nullptr`, and `<stdatomic.h>`.
* **Accompanying Academic Textbook**: Complete 16-chapter monograph and open-access book explaining the physical silicon math, cache hierarchies, and bare-metal algorithms from first principles at [**github.com/erolrecep/rcre-book**](https://github.com/erolrecep/rcre-book).
* **Zero Heap Allocations in Inference Loops**: All memory is managed strictly through 64-byte aligned linear scratchpad memory arenas (`RCRE_Arena`) with $\mathcal{O}(1)$ bulk rewinding. `malloc` is strictly forbidden in hot paths.
* **OpenCV & Scikit-Learn Bit-Exact Parity**: 15-bit fixed-point integer color kernels deliver **0 LSB discrepancy** against OpenCV `cv2.cvtColor(RGB2GRAY)`. Linear SVM, $k$-Means, and PCA match Scikit-Learn within float numerical tolerance ($\epsilon \le 10^{-4}$).
* **Dynamic Hardware Abstraction Layer (HAL)**: Runtime CPU feature detection dispatching single-cycle function pointers to ARM NEON (Apple Silicon), x86 AVX2/FMA, WebAssembly SIMD128, or optimized scalar reference fallbacks.
* **Dual Reverse-Mode Autograd Engine**: Micrograd-style scalar computation DAGs (`RCRE_ScalarVal`) with topological backprop and Graphviz export, alongside high-performance Tensor Tape autograd (`RCRE_AGTensor`) with automatic Vector-Jacobian Product (VJP) propagation and cache-blocked GEMM.
* **Zero-Allocation Deep Learning Suite**: Production loss functions (MSE, BCE, Softmax Cross-Entropy, KL Divergence), optimizers (Decoupled AdamW, Momentum SGD), streaming CSV dataset loaders, and live terminal ANSI/SVG/WebGL visualizers.
* **Isolated Camera Companion (`rcre_cam.h`)**: Native hardware ingestion using macOS AVFoundation (`CVPixelBuffer` zero-copy), Linux Video4Linux2 (`mmap`), and deterministic synthetic test generators for headless CI.
* **Drop-in Single-Header Release**: Amalgamated into a single portable header file [`dist/rcre.h`](dist/rcre.h) (~250 KB).
* **Interactive Educational Visualizers**: Three.js 3D tensor stride visualizers, cache hierarchy simulators, and live in-browser webcam playgrounds in [`web/`](web/).

---

## 📊 Verified Hardware Benchmarks

*Benchmarks recorded on host Apple M5 Max with Clang 17 (`-std=c2x -O3`):*

| Workload / Kernel | Hardware Path | Throughput | Latency | Memory Allocation |
| :--- | :--- | :--- | :--- | :--- |
| **Cache-Tiled GEMM** (37 × 43 × 29) | ARM NEON 128-bit | **72.96 GFLOPS** | — | Linear Arena (0 in loop) |
| **RGB → Grayscale (Rec.601)** | ARM NEON 15-bit | **7,922 MP/s** | **~0.26 ms** @ 1080p | Linear Arena (0 in loop) |
| **Tabular GBDT Prediction** (100 Trees) | Flat-Array Nodes | **7.86M preds / sec** | **127 ns** / pred | Linear Arena (0 in loop) |
| **HNSW Vector Search** (64-D, M=16) | SIMD Euclidean L2 | **44,434 QPS** | **22.5 µs** / query | Linear Arena (0 in loop) |
| **Live Camera Edge Filter** (640 × 480) | Camera → Gray → Blur → Sobel | **239.39 FPS** | **4.18 ms** / frame | Single scratchpad rewind |

---

## 📁 Repository Architecture

```
apps/rcre/
├── dist/
│   └── rcre.h                  # Amalgamated single-header C23 release (250 KB)
├── include/
│   ├── rcre.h                  # Core public C23 header
│   ├── rcre_autograd.h         # Autograd, Losses, Optimizers, Ingestion & Vis C23 header
│   └── extras/
│       └── rcre_cam.h          # Hardware camera companion header
├── src/
│   ├── core/                   # Arena allocator, temp checkpoints, strided tensor, threadpool
│   ├── cv/                     # Rec.601 color conversion, separable Gaussian, Sobel
│   ├── ml/                     # OLS regression, Logistic, Linear SVM, KMeans, PCA, GBDT, HNSW
│   ├── autograd/               # Scalar reverse-mode DAG autograd & Tensor tape VJP autograd
│   ├── nn/                     # Losses, AdamW/SGD optimizers, CSV streaming, ANSI/SVG visualizers
│   ├── hal/                    # CPU probing, dynamic dispatch, NEON, AVX2, WASM SIMD
│   └── extras/                 # AVFoundation (macOS), V4L2 (Linux), Dummy camera backends
├── examples/                   # Standalone real-world applications & benchmarks
│   ├── mlp_autograd_train.c    # 2-layer MLP training with AdamW, MSE, and terminal curve vis
│   ├── webcam_edge_filter.c    # Real-time camera streaming with Gaussian + Sobel at 239 FPS
│   ├── tabular_gbdt_predict.c  # High-throughput tabular GBDT inference benchmark
│   ├── hnsw_vector_search.c    # Approximate Nearest Neighbor vector search on 64-D embeddings
│   └── README.md               # Examples usage and command-line guide
├── bindings/
│   ├── python/                 # Python FFI bindings (ctypes + NumPy interop)
│   └── rust/                   # Safe Rust FFI bindings crate
├── tests/                      # 17 Unit tests, sanitizer tests, Python parity suite
├── benchmarks/                 # Micro-benchmarks for HAL kernels and GEMM
├── web/                        # Interactive Three.js 3D visualizers and webcam playground
└── Makefile                    # Unified, self-contained, zero-dependency build system
```

---

## ⚡ Quickstart

### Option 1: Drop-In Single-Header C23 (`dist/rcre.h`)

Copy [`dist/rcre.h`](dist/rcre.h) into your C or C++ project:

```c
#include <stdio.h>

#define RCRE_IMPLEMENTATION
#include "rcre.h"

int main(void) {
    // 1. Initialize hardware acceleration table
    rcre_hal_init();

    // 2. Create a 16 MB 64-byte aligned linear memory arena
    RCRE_Arena* arena = rcre_arena_create(16 * 1024 * 1024);

    // 3. Allocate a contiguous 3D RGB image tensor [480, 640, 3] U8
    size_t shape[3] = {480, 640, 3};
    RCRE_Tensor rgb = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);

    // 4. Execute 15-bit SIMD Rec.601 Grayscale conversion
    size_t gray_shape[2] = {480, 640};
    RCRE_Tensor gray = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    rcre_cv_rgb_to_gray(&rgb, &gray);

    // 5. Clean teardown in O(1)
    rcre_arena_destroy(arena);
    return 0;
}
```

Compile with your standard C23 compiler:
```bash
clang -std=c2x -O3 -lm -lpthread main.c -o main
./main
```

---

### Option 2: Building & Running from the `Makefile` Command Center

`Makefile` is the central command hub for the entire project:

```bash
# Display all available commands & targets
make help

# 🌐 Launch educational visualizers and in-browser playground at http://localhost:8080
make up_web_app

# 📷 Launch interactive webcam test (prompts for Terminal or Native GUI Window)
make webcam_test

# 🖥️ Launch live webcam edge detection directly with Native GUI Window (Side-by-side)
make webcam_test_gui

# 📟 Launch live webcam edge detection directly in Terminal (ANSI ASCII art)
make webcam_test_terminal

# 📷 Launch synthetic camera stream (ideal for headless SSH or CI)
make webcam_test_dummy

# 🧠 Train a 2-layer MLP using C23 Tensor Tape Autograd & AdamW with live terminal ANSI loss curve
make mlp_train

# ⚡ Run high-throughput GBDT inference benchmark (100k samples)
make gbdt_test

# 🔍 Run HNSW vector similarity search benchmark (64-D embeddings)
make hnsw_test

# 🚀 Build librcre.a, librcre.dylib, dist/rcre.h, and all example binaries
make all

# 🧪 Run full 12-test unit test suite
make test

# 🛡️ Run AddressSanitizer & UndefinedBehaviorSanitizer memory safety checks
make test-sanitizers

# 🐍 Run Python numerical parity tests against OpenCV and Scikit-Learn
make test-parity

# 🎉 Run all verification layers at once
make test-all

# 📊 Run the hardware acceleration GEMM benchmark
make bench
```

---

## 🐍 Python FFI Bindings

Install or use the Python FFI package directly from [`bindings/python/`](bindings/python/):

```python
import numpy as np
import rcre

# Auto-detect host CPU and hardware SIMD capabilities
print("Detected CPU:", rcre.cpu_brand())

# Safe memory arena context manager
with rcre.Arena(16 * 1024 * 1024) as arena:
    # Zero-copy numpy array input
    rgb_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Fast C23 SIMD Rec.601 conversion (0 LSB discrepancy vs cv2.cvtColor)
    gray_img = rcre.rgb_to_gray(rgb_img, arena)
    
    # Bulk memory reclamation in O(1) time
    arena.reset()
```

Run tests:
```bash
uv run --with numpy python tests/test_python_bindings.py
```

---

## 🦀 Rust FFI Bindings

Add RCRE to your Rust project from [`bindings/rust/`](bindings/rust/):

```rust
use rcre::Arena;

fn main() {
    println!("Host CPU: {}", rcre::cpu_brand());

    let mut arena = Arena::new(16 * 1024 * 1024).expect("Failed to allocate arena");
    println!("Arena allocated: {} bytes", arena.capacity());

    arena.reset();
}
```

---

## 🎮 Interactive Educational Suite (`web/`)

RCRE includes four dedicated interactive educational visualizers to master low-level systems programming and real-time vision algorithms:

🌐 **Live Web Portal**: [https://erolrecep.github.io/rcre/](https://erolrecep.github.io/rcre/)  
🔬 **Live Playground & PCA Lab**: [https://erolrecep.github.io/rcre/playground.html](https://erolrecep.github.io/rcre/playground.html)

* **[web/index.html](web/index.html)**: Main interactive showcase portal featuring real-time hardware diagnostics, architecture principles, and visualizer directory.
* **[web/playground.html](web/playground.html)**: **Live In-Browser Webcam Playground & PCA Image Compression Lab**:
  * **5-Stage Live Vision Pipeline**: Ordered sequentially from Stage 1 (Raw RGB) $\rightarrow$ Stage 2 (Rec.601 Grayscale) $\rightarrow$ Stage 3 (Separable Gaussian Blur) $\rightarrow$ Stage 4 (3&times;3 Sobel Edge Magnitude) $\rightarrow$ Stage 5 (Real-Time PCA Compression & Quantization).
  * **📸 Take Photo Button with Direct Filesystem Storage**: Shutter flash effect, synthesized camera sound, format options (PNG, JPEG, BMP), and storage to disk using the modern **File System Access API** (`window.showSaveFilePicker`) with automatic download fallback.
  * **📁 Captured Snapshots Gallery Tray**: Instant thumbnail previews, filesystem re-saving, and 1-click loading into the PCA Lab.
  * **🔬 Interactive PCA Image Compression Test Lab**:
    * Zero-dependency client-side implementation of RCRE's C23 PCA spatial block codec ([`projects/pca_compression/`](projects/pca_compression/)).
    * Interactive parameter tuning: Block size $B$ ($4\times 4$, $8\times 8$, $16\times 16$), component count $K$ (slider 1 to 64 with presets $K=2, 8, 12, 24, 48$), RGB vs Grayscale, and 8-bit uniform quantization.
    * 4 display modes: Draggable interactive **Split Comparison Slider**, **Side-by-Side View**, **Error Heatmap ($\times 5$)**, and **Learned Eigen-Basis Filter Gallery**.
    * Real-time metrics: PSNR (dB), Compression Ratio, MSE, Bitstream size, and Latency.
    * Filesystem export: Reconstructed image PNG, Comparison PNG, and standalone `.rpca` binary bitstreams matching RCRE's C23 codec format.
    * **Automated Rate-Distortion Benchmark Sweep**: One-click sweep across $K \in [2, 4, 8, 12, 16, 24, 32, 48, 64]$ generating rate-distortion curves identical to `rcre-pca --benchmark`.
* **[web/learn/strides.html](web/learn/strides.html)**: Three.js 3D voxel grid visualizing tensor strides, row-major ($HWC$) vs planar ($CHW$) memory layouts, and zero-copy $\mathcal{O}(1)$ slicing.
* **[web/learn/gemm_cache.html](web/learn/gemm_cache.html)**: Animated memory hierarchy showing DRAM, L1 cache tiles, and CPU vector registers during matrix multiplication.
* **[web/learn/stencils.html](web/learn/stencils.html)**: Sliding $3 \times 3$ convolution stencil showing SIMD vector lanes processing 4 or 8 pixels in parallel.

To run locally with the operations hub:
```bash
make up_web_app
# Automatically launches local HTTP server and opens http://localhost:8080 in your browser
```

---

## 📜 Licensing & Commercial Use

RCRE is licensed under a **Dual-Licensing Model**:

### 1. Open-Source Community License (GNU AGPLv3)
For open-source projects, students, academic researchers, and hobbyists, RCRE is freely available under the **[GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)**.
* Under the AGPL-3.0 copyleft terms, any software, cloud service, or derivative product that incorporates or links RCRE **must also make its complete source code publicly available under AGPL-3.0**.

### 2. Commercial Proprietary License (Enterprise & Closed-Source)
If you represent a company, startup, or OEM hardware vendor that wishes to embed RCRE in proprietary, closed-source commercial software, hardware appliances, cameras, or commercial SaaS products **without open-sourcing your proprietary intellectual property**, a commercial license is required.

**Commercial License Benefits:**
* ✅ **Exemption from AGPL-3.0 copyleft obligations**: Keep your application source code closed and proprietary.
* ✅ **Royalty-free commercial deployment**: Distribute RCRE in commercial hardware devices or cloud endpoints.
* ✅ **Direct Author Support**: Architecture consultation, custom SIMD kernel optimization, and prioritized feature requests.

📧 **Commercial & Enterprise Inquiries**: Contact **Recep Erol** via GitHub ([@erolrecep](https://github.com/erolrecep)) or open an inquiry issue.

