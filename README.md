<h1 align="center">RCRE — Recep's Code Run Everywhere</h1>

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
* **OpenCV & Scikit-Learn Bit-Exact Parity**: 15-bit fixed-point integer color kernels deliver **0 LSB discrepancy** against OpenCV `cv2.cvtColor(RGB2GRAY)`. Bilinear resizing matches `cv2.resize(..., INTER_LINEAR)` within machine epsilon ($\le 1.2 \times 10^{-7}$). Linear SVM, $k$-Means, and PCA match Scikit-Learn within float numerical tolerance ($\epsilon \le 10^{-4}$).
* **Dynamic Hardware Abstraction Layer (HAL) & GPU Offload**: Runtime CPU feature detection dispatching to ARM NEON (`ARM_SVE2`/`ARM_DOTPROD` probes), x86 AVX2/FMA, **AVX-512F** (`kernels_avx512.c`), WebAssembly SIMD128, or scalar fallbacks across F32 (`gemm_f32`, `gemv_f32`, `rcre_hal_gemm_f32_mt`), INT8 (`dot_i8`, `gemm_i8`), and BF16 (`gemm_bf16`), plus optional **Apple Metal (`rcre_gpu.h`)** and browser **WebGPU WGSL (`rcre_webgpu.js`)** compute offload and `-DRCRE_FREESTANDING` zero-`malloc` embedded builds (`rcre_arena_from_buffer`).
* **Unified Geometric, Feature, Codec & Motion Computer Vision**: Subpixel nearest/bilinear resizing, HSV $\leftrightarrow$ RGB, Harris & **FAST-9 + rotated 256-bit BRIEF (`ORB`)** binary feature matching, DLT/RANSAC homography & panorama stitching, optical flow, **zero-dependency PNG (DEFLATE/zlib), baseline JPEG (8x8 DCT), BMP, and PPM codecs (`rcre_codec.h`)**, plus a minimal **zero-dep ONNX Protobuf subset runner (`onnx_subset.c`)**.
* **Dual Reverse-Mode Autograd & Deep Learning Suite**: Scalar DAGs + Tensor Tape VJP autograd supporting `matmul`, `conv2d` (`rcre_autograd_conv2d`), and scaled dot-product `attention` (`rcre_autograd_attention`), losses (MSE, BCE, Softmax Cross-Entropy, KL Divergence), and first-order optimizers (**Decoupled AdamW**, Momentum **SGD**, **Lion**, and factored **Adafactor**).
* **Full-Stack Data Science, Statistics & Classical ML (`rcre_df.h`, `rcre_stat.h`, `rcre_prep.h`, `rcre_metric.h`, `rcre_vis_ds.h`)**: Columnar DataFrames with Arrow validity bitmaps, $\mathcal{O}(N)$ IEEE-754 float LSD radix sort & multi-threaded GroupBy; single-pass Welford higher moments, quickselect quantiles & PCA (`svd_flip` deflation); `StandardScaler`/`MinMaxScaler`/`RobustScaler`/Imputer/One-Hot; ROC-AUC/PR-AUC/MCC/Silhouette/K-Fold metrics; ANSI correlation heatmaps & boxplots; OLS, **L1 Sparse Linear**, Pegasos SVM, **Random Forest**, GBDT, HNSW, and **e-Means**.
* **e-Means Mini-Batch Clustering Operator (`rcre_ml_emeans_*`)**: Faithful C23 port of the 2026 e-means algorithm (Bessa et al., arXiv 2611.09501) — exponential-moving-average centroid updates with a cosine-annealed step schedule, scaling-law batch/step auto-tuning (shipped-reference constants), and Gumbel-max dead-cluster revival with deterministic per-dimension splits. Dual-kernel exact assignment (per-row FP32 for $k \\cdot d \\le 16384$, HAL GEMM above), FP64 objective & finalization, and a pool-parallel slab partition that is **bit-identical across pool sizes and machines** (the partition depends only on the batch size). Verified: 205,790-assertion 0-ULP strict-parity replay against an independent Python oracle, plus statistical quality parity (C run at 0.589× the oracle's MSE). On an 8.3M-pixel palette workload (d=3, k=256) the pooled operator trains in ~1.1 s vs 34–78 s for 25-iteration Lloyd k-means; on 100K × 128-d embeddings with k=4096 it is 9× faster **and** 0.566× the final MSE.
* **Isolated Camera Companion (`rcre_cam.h`)**: Native hardware ingestion using macOS AVFoundation (`CVPixelBuffer` zero-copy), Linux Video4Linux2 (`mmap`), and deterministic synthetic test generators for headless CI.
* **High-Performance Scientific Data I/O (`rcre_io.h`)**: Zero-copy virtual memory mapping (`mmap` + `madvise` + `F_NOCACHE` + Win32 `MapViewOfFile`), 64-byte SIMD-aligned 8D **NumPy (`.npy` v1/v2)** and **Hugging Face (`.safetensors`)** containers, **2.61+ GB/s** multi-threaded SIMD CSV ingestion (with RFC-4180 quoted field support) and Clinger/Eisel–Lemire 0-ULP float32 parsing, **50.81 GB/s** ARM NEON byte-unshuffling with `"RCRZ"` LZ4-class lossless block compression, async `pthread` ping-pong `.npy` streaming, and zero-copy **DLPack v1.0** & **Apache Arrow C Data Interface** bridges. See [**io_benchmark.md**](io_benchmark.md) and [**docs/benchmarks_sota.md**](docs/benchmarks_sota.md).
* **Modern LLM Inference & LoRA Training (`rcre_llm.h`)**: LLaMA-style decoder (RMSNorm, RoPE, SwiGLU, Grouped-Query Attention, tiled online-softmax Flash-Attention), `Q8_0`/`Q4_0` block quantization with NEON/SIMD dot kernels, zero-copy **GGUF v2/v3** reader/writer, hash-indexed `RCRE_ByteBPE`, thread-safe sampler (top-k/top-p/min-p/rep-penalty), LoRA fine-tuning, BF16 gradient-checkpointed training, sharded binary token streaming, plus local interactive CLI & OpenAI-compatible HTTP server (`bin/rcre_chat --serve`).
* **Zero-Dependency Image Codecs, ORB Features, Minimal ONNX & Optional Metal/WebGPU (`rcre_codec.h`, `rcre_gpu.h`)**: Built-in PNG (DEFLATE/zlib), baseline JPEG, BMP, and PPM codecs; FAST-9 + rotated 256-bit BRIEF (`RCRE_ORBKeypoint`) feature matching; zero-dependency ONNX Protobuf subset graph runner; Random Forest, L1 Sparse Linear, Lion, Adafactor; optional macOS Metal compute offload, WebGPU bridge, freestanding no-malloc profile (`-DRCRE_FREESTANDING`), CMake/Homebrew/vcpkg packaging, and Python/Rust/Go/Zig/Julia bindings.
* **Drop-in Single-Header Release**: Amalgamated into a single portable header file [`dist/rcre.h`](dist/rcre.h) (**1031.3 KB**, **29,072 lines**).
* **Native Hardware Profiling & Telemetry (`rcre_prof`)**: Microsecond-accurate cycle tracing (< 7 ns overhead per scope) on ARM64 (`cntvct_el0`) and x86 (`rdtsc`), tracking dual-layer RAM (Arena watermarks vs OS RSS), CPU time, engine storage I/O, and zero-dependency Chrome Tracing/Perfetto JSON and standalone HTML exports. See [**rcre_profiling.md**](rcre_profiling.md).
* **Interactive Educational Visualizers**: Three.js 3D tensor stride visualizers, cache hierarchy simulators, and live in-browser webcam playgrounds in [`web/`](web/).

---

## 📊 Verified Hardware Benchmarks

*Benchmarks recorded on host Apple M5 Max with Clang 17 (`-std=c2x -O3`):*

| Workload / Kernel | Hardware Path | Throughput | Latency | Memory Allocation |
| :--- | :--- | :--- | :--- | :--- |
| **Profiler Scope Overhead** | ARM64 `mrs cntvct_el0` | **~232M scopes / sec** | **4.30 ns** / scope | Zero in hot path (Ring buffer) |
| **Cache-Tiled GEMM** (37 × 43 × 29) | ARM NEON 128-bit | **72.96 GFLOPS** | — | Linear Arena (0 in loop) |
| **RGB → Grayscale (Rec.601)** | ARM NEON 15-bit | **7,922 MP/s** | **~0.26 ms** @ 1080p | Linear Arena (0 in loop) |
| **Bilinear Continuous Resize** (640 × 480 → 1280 × 960) | Subpixel continuous | **812 MP/s** | **~1.51 ms** | Linear Arena (0 in loop) |
| **Integral Image Prefix Sum** (1080p) | 2D SAT Cumulative | **1,450 MP/s** | **~1.43 ms** | Linear Arena (0 in loop) |
| **Tabular GBDT Prediction** (100 Trees) | Flat-Array Nodes | **7.86M preds / sec** | **127 ns** / pred | Linear Arena (0 in loop) |
| **XGBoost Inference** (82 Trees, 100k Rows) | Flat 24-byte Cache-Aligned | **1.18B rows / sec** | **0.8 µs** / row | Linear Arena (0 in loop) |
| **HNSW Vector Search** (64-D, M=16) | SIMD Euclidean L2 | **44,434 QPS** | **22.5 µs** / query | Linear Arena (0 in loop) |
| **E-Means Clustering (8.3M × 3, k=256, T=5)** | Pool Slabs + Exact-Kernel Argmin | **12.3x vs serial** | **1.11 s** (Lloyd-25: 35.9 s) | Linear Arena (0 in loop) |
| **E-Means Clustering (100K × 128, k=4096, T=5)** | Pool Slabs + HAL GEMM Argmin | **8.8x wall vs Lloyd-25** | **1.16 s** (0.566× Lloyd MSE) | Linear Arena (0 in loop) |
| **LSH Multimodal Vector Search (Top-20)** | Multi-Probe Hash + SIMD Re-rank | **113,392 QPS** | **8.82 µs** / query | Linear Arena (0 in loop) |
| **Live Camera Edge Filter** (640 × 480) | Camera → Gray → Blur → Sobel | **239.39 FPS** | **4.18 ms** / frame | Single scratchpad rewind |
| **BCSR4 SpMV (MobileNet Head 75% Sparse)** | ARM NEON 4x4 Register Blocks | **28.78x speedup** | **1.56 µs** / pass | Linear Arena (0 in loop) |
| **Wavelet LeGall 5/3 (512 × 512, 31x Lossy)** | 2D Integer Lifting DWT | **54.63 dB PSNR** | **4.55 ms** enc / **4.16 ms** dec | Linear Arena (0 in loop) |
| **Cache-Tiled SGEMM (512 × 512, 1T)** | ARM NEON 128-bit | **69.13 GFLOP/s** | **3.88 ms** | Linear Arena (0 in loop) |
| **Parallel SGEMM (512 × 512, MT)** | ARM NEON + ThreadPool Slabs | **315.32 GFLOP/s** (**4.56x**) | **0.85 ms** | Linear Arena (0 in loop) |
| **Memory-Bound GEMV (4096 × 4096 F32)** | ARM NEON Streaming Dot | **31.38 GB/s** (**15.68 GFLOP/s**) | **2.14 ms** | Linear Arena (0 in loop) |
| **Quantized IGEMM (512 × 512 INT8)** | `rcre_hal.gemm_i8` | **39.62 GIOP/s** | **6.77 ms** | Linear Arena (0 in loop) |
| **Brain Float16 BGEMM (512 × 512 BF16)** | `rcre_hal.gemm_bf16` | **41.29 GFLOP/s** | **6.50 ms** | Linear Arena (0 in loop) |
| **FP16 / BF16 SIMD Dequant (16M elems)** | NEON `vcvt` / `vshll` | **71.93 GB/s** / **71.31 GB/s** | **1.40 ms** / **1.41 ms** | Scoped Arena |
| **Cache Streaming vs Thrashing** (4096 × 4096) | Sequential Row-Major | **7.33 GB/s** | **8.96 ms** | 6.5x faster than column-major |
| **ShapeTracker Zero-Copy Transpose** | O(1) Stride Swap | **Instantaneous** | **0 ns** | 0 bytes memory allocated |
| **Parallel SIMD CSV Ingestion** (16,384 × 64 F32) | `mmap` + SIMD Slabs + Eisel–Lemire | **2.61 GB/s (2,670.9 MB/s)** | **2.81 ms** | **29.4x vs C89** |
| **Zero-Copy `.npy` / `.safetensors` View** (64 MB) | 64-Byte Padded `mmap` (`is_view=true`) | **9.26 GB/s sustained** | **49.07 µs** (`.npy`) / **9.5 ns** (lookup) | **0 bytes allocated (107x vs `fread`)** |
| **SIMD Byte-Unshuffle & `RCRZ` Block Codec** (16 MB) | ARM NEON `vst4q_u8` + LZ4-Class Match | **50.81 GB/s** (unshuffle) / **3.50 GB/s** (codec) | **0.31 ms** / **4.47 ms** | **0-ULP Lossless** (254.56x ratio) |
| **DLPack v1.0 & Arrow C Data Zero-Copy Handoff** | Plain C ABI Pointer & Stride Cast | **0 Bytes Copied** | **12.5 ns** / **12.5 ns** | Zero-Copy Stack Views |

📊 *Full Roofline & SOTA hardware results: [**docs/benchmarks_sota.md**](docs/benchmarks_sota.md) and [`web/benchmarks.html`](web/benchmarks.html).*

---

## 🛠️ Turnkey Production Projects (`projects/`)

In addition to modular core kernels and unit benchmarks, RCRE includes twelve self-contained, turnkey production applications built 100% on RCRE and ISO C23:

| Project | Directory | Purpose | Core RCRE Technologies | Live Launch |
| :--- | :--- | :--- | :--- | :--- |
| **`rcre-lens`** | [`projects/reverse_image_search/`](projects/reverse_image_search/) | High-speed reverse image search engine & visual database indexing | Spatial Pyramid Descriptors, Sobel Contours, HNSW Graph Index | `make project_lens_demo` |
| **`rcre-facex`** | [`projects/facial_expression_analysis/`](projects/facial_expression_analysis/) | Real-time face detection, 13-point landmarks, FACS Action Units & emotion classifier | Skin SAT Integral Image, Valley Tracking, FACS Biomechanics, Softmax | `make project_facex_demo` |
| **`rcre-mobilenet`** | [`projects/cifar100_mobilenet/`](projects/cifar100_mobilenet/) | Sub-4ms MobileNet CNN training and zero-heap inference on CIFAR-100 | Depthwise Separable Conv, Edge BN Folding, SIMD axpy, Softmax | `make project_mobilenet_demo` |
| **`rcre-pca`** | [`projects/pca_compression/`](projects/pca_compression/) | Zero-dependency spatial PCA image compression & 8-bit quantization codec | Power Iteration, Deflation, 8-bit Uniform Quant, Web Studio | `make -C projects/pca_compression demo` |
| **`rcre-perf`** | [`projects/c_performance/`](projects/c_performance/) | Systems microarchitecture, cache hierarchy, strided view & SIMD benchmark suite | Memory Arenas, ShapeTracker Views, HAL SIMD Dispatch, Perfetto Telemetry | `make project_perf_demo` |
| **`rcre-sr`** | [`projects/document_super_resolution/`](projects/document_super_resolution/) | Document image super-resolution, Wiener deconvolution, Lanczos-3 & adaptive OCR pipeline | Separable Lanczos-3 / Bicubic, Wiener Deconvolution (2D FFT), Guided Denoise, CLAHE, Otsu & Adaptive Threshold | `make project_sr_demo` |
| **`rcre-wavelet`** | [`projects/wavelet_compression/`](projects/wavelet_compression/) | Reversible multi-resolution integer wavelet image codec (lossless to 32x lossy) | LeGall 5/3 Integer Lifting, 2D DWT Pyramid, Dead-Zone Quant, Run-Length Codec | `make project_wavelet_demo` |
| **`rcre-xgb`** | [`projects/rcre_xgb/`](projects/rcre_xgb/) | XGBoost-style regularized second-order gradient boosted trees | Quantile Histograms, Newton Updates, Missing NaN Routing, Flat-Array Predictor | `make project_xgb_demo` |
| **`rcre-omnisearch`** | [`projects/omnisearch_hash/`](projects/omnisearch_hash/) | 30M multimodal vector search & LSH hash engine for mixed text & images | L2 Spherical Normalization, Random Hyperplanes, Arena Hash Table, SIMD ADC | `make project_omnisearch_demo` |
| **`rcre-galton`** | [`projects/rcre_galtonboard/`](projects/rcre_galtonboard/) | High-speed Galton Board (Quincunx) flow simulator, Bernoulli trials & CLT | SIMD Bit-Level Popcount, Dynamic Histograms, Chi-Square Goodness-of-Fit, WebAssembly | `make project_galton_demo` |
| **`rcre-emeans`** | [`projects/rcre_emeans/`](projects/rcre_emeans/) | Ultra-fast mini-batch EMA k-means clustering with scaling-law recipes & dead-cluster revival | EMA Updates, Gumbel-Max Revival, Dual-Kernel Argmin, Deterministic Slabs | `make project_emeans_demo` |
| **`rcre-manifold`** | [`projects/rcre_manifold/`](projects/rcre_manifold/) | Multi-dimensional manifold learning, UMAP, Barnes-Hut t-SNE & PaCMAP untangling | Fuzzy Simplicial Sets, 2D Quadtree Repulsion, 3-Tier Pair Annealing, WebAssembly | `make project_manifold_demo` |

📖 *For complete CLI documentation and architectural guides, see [`projects/README.md`](projects/README.md).*

---

## 📁 Repository Architecture

```
apps/rcre/
├── dist/
│   └── rcre.h                  # Amalgamated single-header C23 release (1,031.3 KB, 29,072 lines)
├── project-template/           # Turnkey starter project with rcre.h, main.c, tests, Makefile & docs
│   ├── include/rcre.h          # Single-header engine distribution
│   ├── src/main.c              # Pedagogical starter app with 1080p SIMD pipeline
│   ├── tests/test_main.c       # ASan/UBSan unit test runner
│   ├── Makefile                # Turnkey build targets (all, run, test, clean, help)
│   ├── logs/sample_run.log     # Sample telemetry execution log
│   ├── README.md               # Complete newcomer onboarding manual
│   ├── MINI_EBOOK.md           # 10-page visual gentle guide for non-CS practitioners
│   └── GUIDELINES.md           # Developer mini guideline & silicon best practices cheat sheet
├── include/
│   ├── rcre.h                  # Core public C23 header (ABI v2.0.0, 8D broadcasting tensors, CV, ML)
│   ├── rcre_io.h               # Zero-copy mmap (POSIX/Win32), async POSIX prefetch, .npy/.safetensors, CSV, DLPack & Arrow
│   ├── rcre_llm.h              # LLaMA (RMSNorm/RoPE/SwiGLU/GQA/Flash), Q8_0/Q4_0, GGUF v2/v3, ByteBPE, Sampler & LoRA
│   ├── rcre_codec.h            # Zero-dep PNG (DEFLATE), JPEG, BMP, PPM codecs & ONNX Protobuf subset runner
│   ├── rcre_df.h               # Columnar DataFrame engine (Arrow validity bitmaps, Radix Sort, GroupBy)
│   ├── rcre_stat.h             # Welford moments, Quickselect quantiles, Spearman/Pearson & PCA (svd_flip)
│   ├── rcre_prep.h             # Standard/MinMax/Robust scalers, SimpleImputer & One-Hot encoding
│   ├── rcre_metric.h           # ROC-AUC, PR-AUC, F1, MCC, Silhouette, Davies-Bouldin & K-Fold CV
│   ├── rcre_vis_ds.h           # Terminal ANSI correlation heatmaps, histograms & IQR boxplots
│   ├── rcre_sparse.h           # CSR & 4x4 Block-CSR (BCSR4) sparse matrix & pruning engine
│   ├── rcre_prof.h             # High-frequency hardware profiler & telemetry header
│   ├── rcre_autograd.h         # Scalar/Tensor Tape VJP autograd (incl. Conv2D/Attention) & AdamW/SGD/Lion/Adafactor
│   ├── rcre_transformer.h      # NanoGPT, Attention, KV-Cache & Tokenizers C23 header
│   └── extras/
│       ├── rcre_cam.h          # Hardware camera companion header
│       ├── rcre_gpu.h          # Optional GPU compute offload header (Apple Metal & CPU HAL fallback)
│       └── rcre_window.h       # Native macOS GUI preview helper
├── projects/                   # 12 turnkey production applications built on RCRE
│   ├── reverse_image_search/   # rcre-lens: Spatial pyramid + HNSW visual search engine
│   ├── facial_expression_analysis/ # rcre-facex: Real-time FACS landmarks & emotion classifier
│   ├── cifar100_mobilenet/     # rcre-mobilenet: Sub-4ms MobileNet inference on CIFAR-100
│   ├── pca_compression/        # rcre-pca: Spatial PCA image compression & 8-bit codec
│   ├── c_performance/          # rcre-perf: Hardware microarchitecture & SIMD benchmark suite
│   ├── document_super_resolution/ # rcre-sr: Adaptive document super-resolution & OCR restoration
│   ├── wavelet_compression/    # rcre-wavelet: Multi-resolution integer wavelet image codec
│   ├── rcre_xgb/               # rcre-xgb: XGBoost-style gradient boosted decision trees
│   ├── omnisearch_hash/        # rcre-omnisearch: 30M multimodal vector search & LSH hash engine
│   ├── rcre_galtonboard/       # rcre-galton: High-speed Galton Board Bernoulli & CLT simulator
│   ├── rcre_emeans/            # rcre-emeans: Fast mini-batch EMA k-means with Gumbel revival
│   ├── rcre_manifold/          # rcre-manifold: UMAP, Barnes-Hut t-SNE & PaCMAP manifold learning
│   └── README.md               # Production Projects Hub guide
├── src/
│   ├── core/                   # Arena allocator (incl. freestanding), Slice, 8D Tensor, DataFrame, Prof, Sparse, I/O
│   ├── cv/                     # Color, Geom, Filter, FFT, Harris/FAST+ORB, Homography, Flow, CNN, Codecs, ONNX subset
│   ├── ml/                     # OLS/Sparse-L1, Logistic, SVM, KMeans, E-Means, PCA, GBDT, Random Forest, HNSW, Stat/Prep/Metric
│   ├── autograd/               # Scalar DAG autograd & Tensor tape VJP autograd (GEMM/Conv2D/Attention)
│   ├── transformer/            # GPT-2 + LLaMA, GGUF v2/v3, Q8_0/Q4_0, Flash-GQA, ByteBPE, Sampler, LoRA
│   ├── nn/                     # Losses, AdamW/SGD/Lion/Adafactor optimizers, CSV streaming, ANSI/DS visualizers
│   ├── hal/                    # CPU probing, dynamic dispatch, NEON, AVX2, AVX-512F, WASM SIMD128
│   └── extras/                 # AVFoundation (macOS), V4L2 (Linux), Metal compute (gpu_metal.m) & stubs
├── examples/                   # Standalone real-world applications & CLI tools
│   ├── rcre_chat.c             # Local LLM CLI (--cli/--demo) & OpenAI-compatible HTTP server (--serve)
│   ├── gpt_generate.c          # Local GPT trainer & .safetensors zero-copy KV-cache text generator
│   ├── mlp_autograd_train.c    # 2-layer MLP training with AdamW, MSE, and terminal curve vis
│   ├── webcam_edge_filter.c    # Real-time camera streaming with Gaussian + Sobel at 239 FPS
│   ├── tabular_gbdt_predict.c  # High-throughput tabular GBDT inference benchmark
│   ├── hnsw_vector_search.c    # Approximate Nearest Neighbor vector search on 64-D embeddings
│   ├── sparse_prune_bench.c    # BCSR4 sparse pruning benchmark
│   └── README.md               # Examples usage and command-line guide
├── bindings/
│   ├── python/                 # Python FFI bindings (ctypes + NumPy / .npy / metrics / stats)
│   ├── rust/                   # Safe Rust FFI bindings crate (ABI v2 / 8D tensor layout)
│   ├── go/                     # Verified cgo bindings & unit tests (bindings/go/rcre.go)
│   ├── zig/                    # Zig @cImport bindings (bindings/zig/rcre.zig)
│   └── julia/                  # Julia ccall module (bindings/julia/RCRE.jl)
├── packaging/                  # pkg-config template (rcre.pc.in), Homebrew formula & vcpkg portfile
├── tests/                      # 35 Unit test suites (incl. test_sota, test_df, test_stat, test_metric) + tests/fuzz/fuzz_io.c
├── benchmarks/                 # Micro-benchmarks: HAL (bench_hal), Roofline (bench_roofline), I/O (bench_io), Prof (bench_prof)
├── docs/                       # SOTA benchmarks report (docs/benchmarks_sota.md) & technical notes
├── web/                        # Visualizers, API docs (api_data.json), benchmarks UI & WebGPU bridge (wasm/rcre_webgpu.js)
├── CMakeLists.txt              # CMake build, shared/static targets & pkg-config installation
└── Makefile                    # Unified, self-contained, zero-dependency build system
```

---

## ✅ Required Steps (Prerequisites & First Run)

Follow these steps once to build, verify, and run everything in this project (including the new `rcre_io` engine and the local small-LLM pipeline):

1. **Install a C23-capable toolchain** (required; the core is strict ISO C23, `-std=c2x`):
   * macOS: Xcode Command Line Tools (`xcode-select --install`), Apple Clang 15+ (Clang 17+ recommended).
   * Linux: `clang` 17+ or `gcc` 13+ plus `make` (`sudo apt install clang make`). `-D_GNU_SOURCE` is added automatically.
   * Windows: use WSL2 (Ubuntu) and follow the Linux steps.
2. **Install `python3`** (3.9+): required by `make dist` (`scripts/amalgamate.py`) and `make docs`.
3. **Install [`uv`](https://docs.astral.sh/uv/)** *(optional)*: only needed for `make test-parity` (pulls `numpy`, `opencv-python`, `scikit-learn`, `scipy` on demand, verifying `.npy` / `.safetensors` 0-ULP parity).
4. **Clone and enter the project**:
   ```bash
   git clone https://github.com/erolrecep/rcre.git && cd rcre   # or: cd apps/rcre inside the monorepo
   ```
5. **Prepare the build environment** (detects OS / architecture / SIMD): `make prep`
6. **Build the library, single header, and examples**: `make all` → produces `lib/librcre.a`, `lib/librcre.dylib`, `dist/rcre.h`, and `bin/*`.
7. **Verify the install**: `make test` (all unit suites, including `test_io`), then optionally `make test-sanitizers` (ASan + UBSan) and `make test-parity`.
8. **Run the new capabilities**:
   ```bash
   make bench_io     # Scientific data I/O bandwidth benchmark (see io_benchmark.md)
   make gpt_train    # Train a tiny LLM locally, save .safetensors, zero-copy reload + KV-cache inference
   ```
   * `make gpt_train` needs the bundled corpora in [`data/`](data/) (`tinystories.txt`, `tinyshakespeare.txt`, `arithmetic.txt`). Use another one with `make gpt_train ARGS="--corpus data/tinyshakespeare.txt --steps 600"`.
   * It needs ~64 MB of free RAM and about 5 seconds on Apple M-series; it writes a temporary checkpoint to `/tmp` and removes it afterwards.
9. **Optional hardware add-ons**: macOS camera demos use AVFoundation (permission prompt on first run); Linux uses V4L2 (`/dev/video0`); use `make webcam_test_dummy` for headless/CI.

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

# 🛠️ Run Production Projects Demos
make project_lens_demo       # Reverse Image Search (rcre-lens)
make project_facex_demo      # Facial Expression & Landmarks (rcre-facex)
make project_mobilenet_demo  # CIFAR-100 MobileNet Inference (rcre-mobilenet)
make project_perf_demo       # Systems Performance & Cache Benchmarks (rcre-perf)
make project_perf_test       # Systems Performance Verification Suite
make project_perf_bench      # Full Benchmark with Perfetto & HTML Export
make project_sr_demo          # Document Super-Resolution & OCR (rcre-sr)
make project_sr_test          # Document Super-Resolution Verification Suite
make project_wavelet_demo     # Wavelet Image Codec Demo (rcre-wavelet)
make project_wavelet_test     # Wavelet Codec Verification Suite
make project_xgb_demo         # XGBoost Gradient Boosted Trees Demo (rcre-xgb)
make project_xgb_test         # XGBoost Trees Verification Suite (55 checks)
make project_xgb_bench        # XGBoost Scaling & Parity Benchmark Suite
make project_omnisearch_demo  # Multimodal Vector Search & LSH Hash Demo (rcre-omnisearch)
make project_omnisearch_test  # Vector Search Invariant & Recall Parity Verification
make project_omnisearch_bench # SIMD & Hash Table Micro-Benchmarks

# 🚀 Build librcre.a, librcre.dylib, dist/rcre.h, and all example binaries
make all

# 🧪 Run full 35-test unit test suite (Core, DataFrames, Stats, Metrics, I/O, SOTA)
make test

# 🛡️ Run AddressSanitizer & UndefinedBehaviorSanitizer checks across all 35 suites
make test-sanitizers

# 🐛 Run deterministic ASan+UBSan fuzz regression suite over I/O & codec parsers
make test-fuzz

# 🐍 Run Python numerical parity tests against OpenCV, SciPy & Scikit-Learn + FFI tests
make test-parity

# 🎉 Run unit tests, sanitizers & Python parity layers together
make test-all

# 📊 Run HAL GEMM, Roofline (SGEMM/GEMV/I8/BF16), Scientific I/O & Profiler workloads
make bench
make bench_roofline
make bench_io
make profile

# 🤖 Launch local quantized LLM demo or OpenAI-compatible HTTP server (/v1/chat/completions)
./bin/rcre_chat --demo
./bin/rcre_chat --serve 8080 --threads 4

# 🐹 Verify Go cgo FFI bindings
(cd bindings/go && GOWORK=off go test -v .)

# 🔒 Build freestanding no-malloc core arena profile & standalone header check
make test-portable
clang -std=c2x -O2 -DRCRE_FREESTANDING -Iinclude -c src/core/arena.c -o /tmp/arena_fs.o

# 🧩 Alternative CMake build & install (static, shared, rcre_chat & pkg-config)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

---

### Option 3: Turnkey Starter Template & Mini Guideline (`project-template/`)

For newcomers starting a new application without tackling build-system complexities or external dependencies, RCRE provides a turnkey starter kit in [`project-template/`](project-template/):

```bash
# 1. Enter the starter template
cd project-template

# 2. Build & run optimized release binary (1080p SIMD Rec.601 pipeline > 7,500 MP/s)
make run

# 3. Verify under AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan)
make test
```

#### The 5-Step RCRE Architecture
```text
[1. rcre_hal_init()]  -->  [2. rcre_arena_create(16MB)]  -->  [3. rcre_tensor_create(HWC)]  -->  [4. SIMD CV/ML Kernel]  -->  [5. rcre_arena_reset()]
 (Probe Host CPU)           (64-byte aligned pool)             (Zero syscall allocations)          (High-throughput vector math)    (O(1) memory rewind)
```

#### Core Silicon Rules for Newcomers
1. **Zero `malloc()` in Inner Loops**: Allocate all temporary buffers from `RCRE_Arena` and call `rcre_arena_reset()` between frames or iterations.
2. **64-Byte Alignment**: Always preserve 64-byte alignment boundaries for maximum vector memory bandwidth (enforced automatically by arenas).
3. **Single Translation Unit Implementation**: Define `#define RCRE_IMPLEMENTATION` in `main.c` only; include `rcre.h` cleanly everywhere else.

📖 *Non-CS background? Read our visual [10-Page Mini E-Book](project-template/MINI_EBOOK.md) (or on the [Book Website](https://erolrecep.github.io/rcre-book/chapters/starter_guide_ebook/))! For the full technical reference, see [`project-template/README.md`](project-template/README.md) and the developer cheat sheet [`project-template/GUIDELINES.md`](project-template/GUIDELINES.md).*

---

## 🔬 Native Hardware Profiling & Telemetry (`rcre_prof`)

RCRE includes a zero-dependency, sub-7ns C23 profiling engine built directly into the kernel:
* **Hardware Cycle Counter**: Sub-nanosecond reads via ARM64 `mrs cntvct_el0` (Apple Silicon) and x86 `rdtsc`.
* **Dual-Layer Memory Telemetry**: Captures exact byte watermarks inside `RCRE_Arena` across rewind cycles without kernel syscalls, alongside OS-level physical RSS.
* **Storage I/O & CPU Accounting**: Tracks exact checkpoint byte rates and user+sys CPU time.
* **Zero-Dependency Exporters**: Produces ANSI terminal summaries with colored bar charts, Chrome Tracing / Perfetto JSON timelines (`prof/rcre_profile.json`), and standalone HTML dashboards (`prof/rcre_profile.html`).

```bash
# Run the complete profiling workload
make profile

# View the generated HTML dashboard
open prof/rcre_profile.html  # On macOS (or xdg-open on Linux)

# Drag prof/rcre_profile.json into https://ui.perfetto.dev for interactive timeline exploration!
```
📖 For detailed architectural derivations and C23 API integration instructions, see [**rcre_profiling.md**](rcre_profiling.md).

---

## ⚡ High-Performance Scientific Data I/O (`rcre_io`) & Bandwidth Benchmarks

RCRE includes a zero-dependency scientific data I/O engine ([`include/rcre_io.h`](include/rcre_io.h), [`src/core/io.c`](src/core/io.c)) engineered from first principles to prevent storage bottlenecks from starving SIMD compute kernels:

* **Zero-Copy 64-Byte Aligned Containers (`.npy` & `.safetensors`)**: Pads binary headers to exact multiples of 64 bytes ($\Delta_{\text{file}} \equiv 0 \pmod{64}$) so OS page-aligned `mmap` views (`is_view = true`) land directly on 64-byte SIMD boundaries with **0 bytes copied and 0 arena allocations**.
* **Multi-Threaded SIMD Tabular CSV Ingestion**: Combines `mmap` + `rcre_hal.find_byte` SIMD scanning + `RCRE_ThreadPool` row-slab partitioning + a Clinger/Eisel–Lemire **0-ULP fast float32 parser** (`rcre_slice_parse_f32`), reaching **2.77 GB/s** on Apple M5 Max (**32.7x faster** than C89 `stdio`, **11.3x faster** than Python `numpy.loadtxt`).
* **SIMD Byte-Shuffle & Lossless `"RCRZ"` Block Codec**: Uses hardware de-interleaving (`vld4q_u8` / `vst4q_u8` on NEON, `_mm_shuffle_epi8` on AVX2, `wasm_i8x16_shuffle` on WASM SIMD128) at **45.45 GB/s** to group correlated IEEE-754 exponent bytes prior to zero-dependency LZ4-class match compression.
* **Zero-Copy Cross-Framework C ABI Bridges**: Built-in **DLPack v1.0** (`RCRE_DLTensor`, **10.5 ns**) and **Apache Arrow C Data Interface** (`RCRE_ArrowSchema` / `RCRE_ArrowArray`, **11.0 ns**) structs for instant zero-copy pointer exchange with PyTorch, NumPy, JAX, Polars, and DuckDB.

| I/O Workload | Legacy C89 `stdio` | Python `NumPy 2.x` | **RCRE (`rcre_io` C23)** | **RCRE Speedup** |
| :--- | :---: | :---: | :---: | :---: |
| **CSV Ingestion (1M Floats, 7.5 MB)** | 86.53 ms (86.7 MB/s) | 29.96 ms (250.3 MB/s) | **2.64 ms (2,836.6 MB/s / 2.77 GB/s)** | **32.7x vs C89 / 11.3x vs NumPy** |
| **64 MB `.npy` Cold Map / View Init** | 6.55 ms (`fread` copy) | 144.16 µs (`mmap_mode='r'`) | **55.03 µs (`rcre_io_view_npy`, 0 B alloc)** | **119x vs `fread` / 2.6x vs NumPy** |
| **`.safetensors` Tensor View Lookup** | — | — | **8.2 ns (`is_view = true`, 64B aligned)** | **Zero-Copy $\mathcal{O}(1)$ Slice Cast** |
| **SIMD Byte-Unshuffle + `RCRZ` Codec** | — | — | **45.45 GB/s unshuffle / 2.61 GB/s compress** | **0-ULP Bit-Exact Lossless** |
| **DLPack & Arrow C ABI Handoff** | `memcpy` required | `__dlpack__` capsule | **10.5 ns (DLPack) / 11.0 ns (Arrow)** | **0 Bytes Copied** |

```bash
# Run the full C23 Scientific Data I/O Bandwidth Benchmark Suite
make bench_io
```
📖 For the complete multi-hardware bandwidth matrix (Apple Silicon, x86_64 AVX2 HPC, ARM64 Edge, WebAssembly SIMD128) and architectural comparison against HDF5, NumPy, Safetensors-rs, Blosc2, and Apache Arrow, see [**io_benchmark.md**](io_benchmark.md).

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

## 🐹 Zig, Go & Julia Bindings (`bindings/`)

* **Go (`bindings/go/rcre.go`)**: cgo wrappers exposing `ABIVersion()`, `InitHAL()`, `CPUBrand()`, and HAL-dispatched `DotF32()`. Verify with `(cd bindings/go && GOWORK=off go test -v .)`.
* **Zig (`bindings/zig/rcre.zig`)**: Direct `@cImport("rcre.h")` wrappers for `arenaCreate`, `tensorCreate2D`, and `abiVersion`.
* **Julia (`bindings/julia/RCRE.jl`)**: Zero-overhead `ccall` wrappers over `librcre.dylib` / `librcre.so`.
* **Packaging (`packaging/`, `CMakeLists.txt`)**: Ready-to-use `rcre.pc.in`, Homebrew formula (`packaging/homebrew/rcre.rb`), and `vcpkg` portfile (`packaging/vcpkg/portfile.cmake`).

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

