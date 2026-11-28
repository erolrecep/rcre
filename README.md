<h1 align="center">RCRE — Recep's Code Run Everywhere</h1>

<p align="center">
  <strong>A single-header, zero-dependency C23 engine unifying modern Computer Vision and Classical Machine Learning across Edge, HPC, and WebAssembly.</strong>
</p>

<p align="center">
  <a href="https://erolrecep.github.io/rcre/"><img src="https://img.shields.io/badge/Demo-Live_Playground-00d2ff?logo=github&style=flat" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/Language-ISO_C23-00599C?logo=c&style=flat" alt="C23 Standard">
  <img src="https://img.shields.io/badge/Sanitizers-ASan_%7C_UBSan_%7C_LSan_Clean-success?style=flat" alt="Sanitizers Clean">
  <img src="https://img.shields.io/badge/OpenCV_Parity-0_LSB-brightgreen?logo=opencv&logoColor=white&style=flat" alt="OpenCV Parity">
  <img src="https://img.shields.io/badge/Single_Header-34k_lines-blue?style=flat" alt="Single Header Amalgamation">
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
* **Dynamic Hardware Abstraction Layer (HAL) & GPU Offload**: Runtime CPU feature detection dispatching to ARM NEON (`ARM_SVE2`/`ARM_DOTPROD` probes), x86 AVX2/FMA, **AVX-512F** (`kernels_avx512.c`), WebAssembly SIMD128, or scalar fallbacks across F64 (`gemm_f64`, `gemv_f64`, `dot_f64`, `axpy_f64`, `scale_f64`, `sum_f64` — the HPC scientific substrate), F32 (`gemm_f32`, `gemv_f32`, `rcre_hal_gemm_f32_mt`), INT8 (`dot_i8`, `gemm_i8`), and BF16 (`gemm_bf16`), plus optional **Apple Metal (`rcre_gpu.h`)** and browser **WebGPU WGSL (`rcre_webgpu.js`)** compute offload and `-DRCRE_FREESTANDING` zero-`malloc` embedded builds (`rcre_arena_from_buffer`).
* **Unified Geometric, Feature, Codec & Motion Computer Vision**: Subpixel nearest/bilinear resizing, **GraphicsMagick-exact continuous resamplers with 16 windowed filter kernels (`rcre_cv_resize_gm`)**, HSV $\leftrightarrow$ RGB, Harris & **FAST-9 + rotated 256-bit BRIEF (`ORB`)** binary feature matching, DLT/RANSAC homography & panorama stitching, optical flow, **zero-dependency PNG (DEFLATE/zlib), baseline JPEG (8x8 DCT), BMP, and PPM codecs (`rcre_codec.h`)**, plus a minimal **zero-dep ONNX Protobuf subset runner (`onnx_subset.c`)**.
* **Dual Reverse-Mode Autograd & Deep Learning Suite**: Scalar DAGs + Tensor Tape VJP autograd supporting `matmul`, `conv2d` (`rcre_autograd_conv2d`), and scaled dot-product `attention` (`rcre_autograd_attention`), losses (MSE, BCE, Softmax Cross-Entropy, KL Divergence), and first-order optimizers (**Decoupled AdamW**, Momentum **SGD**, **Lion**, and factored **Adafactor**).
* **Generative & Vision Deep Learning Suite (`rcre_gen.h`)**: Four unified zero-dependency architectures on the tensor tape — **Convolutional / Denoising Autoencoder** (`RCRE_ConvAE`), **Convolutional β-VAE** (`RCRE_ConvVAE`, reparameterized sampling + accumulating KL), **DDPM / DDIM diffusion** with linear & cosine schedules and a GroupNorm/FiLM **Tiny U-Net** (`RCRE_TinyUNet`, ancestral + η-DDIM samplers), and **ViT / DiT with AdaLN-Zero** (`RCRE_ViT`, `RCRE_DiT`, unconditional & class-conditioned); plus 25 new tape VJP ops (layer/group norm, MHSA, patchify/unpatchify, FiLM/AdaLN/gate, upsample/avgpool, embed lookup, reparameterize…), global grad-norm clipping, sinusoidal timestep embeddings, a deterministic 4-class synthetic shapes dataset, and `.safetensors` checkpoint I/O. Verified by an 8-suite / 17,120-assertion gate: central-difference gradient checks on every new op, independent-reference schedule values, ViT ≥ 87.5% 4-class accuracy in 50 steps, and **same-seed bit-identity for DDIM (η = 0) / DDPM / VAE sampling**.
* **Full-Stack Data Science, Statistics & Classical ML (`rcre_df.h`, `rcre_stat.h`, `rcre_prep.h`, `rcre_metric.h`, `rcre_vis_ds.h`)**: Columnar DataFrames with Arrow validity bitmaps, $\mathcal{O}(N)$ IEEE-754 float LSD radix sort & multi-threaded GroupBy; single-pass Welford higher moments, quickselect quantiles & PCA (`svd_flip` deflation); `StandardScaler`/`MinMaxScaler`/`RobustScaler`/Imputer/One-Hot; ROC-AUC/PR-AUC/MCC/Silhouette/K-Fold metrics; ANSI correlation heatmaps & boxplots; OLS, **L1 Sparse Linear**, Pegasos SVM, **Random Forest**, GBDT, HNSW, and **e-Means**.
* **Zero-Malloc, SIMD-Accelerated C23 Safe String Library (`rcre_str.h`)**: Length-prefixed zero-copy views (`RCRE_Slice` / `RCRE_StrView`) with `RCRE_STR(lit)` zero-cost compile-time literals (0 `strlen` overhead); stack/struct bounded buffers (`RCRE_StrBuf` and `rcre_cstr_copy_bounded`) replacing `strcpy`/`snprintf` with guaranteed NUL-termination and sticky truncation tracking; pull-based re-entrant token splitting (`RCRE_SplitIter`) replacing `strtok`; strict RFC 3629 UTF-8 validation (**44.5 GB/s** ASCII, **13.3 GB/s** multi-byte via NEON/AVX2), codepoint-boundary truncation, and zero-allocation parsers (`parse_f64`, `parse_hex_u64`, `parse_bool`); open-addressed arena string interning pool (`RCRE_StrPool`) delivering **68.9M lookups/s** (14.5 ns/op, 0 mallocs); safe path utilities (`rcre_path_*`); and optional compile-time legacy API poisoning via `#define RCRE_BAN_UNSAFE_CSTR` (`#pragma GCC poison`).
* **HPC Zero-Malloc Natural Language Processing Engine (`rcre_nlp.h`)**: Built directly on `rcre_str.h` and `rcre_sparse.h` with **0 `malloc`s** — zero-copy pull tokenizers & UTF-8/word $n$-grams (`663.9 MB/s`, `92.8M tokens/s`); two-pass $\mathcal{O}(\text{tokens})$ flat CSR-layout **BM25 inverted index** (`11.3 ms` build / `253 µs` top-10 query on 10k docs) + deterministic **Reciprocal Rank Fusion (`rcre_nlp_rrf_fuse`)** for hybrid lexical + HNSW vector search; **Scikit-Learn-parity `RCRE_TfidfVectorizer`** emitting 64-byte-aligned `RCRE_CSRMatrix` (`561.7k docs/s`) + lock-free signed Feature Hashing; flat-table **Aho–Corasick DFA** multi-pattern matcher & single-pass redactor (`550.8 MB/s`, 1 load/byte); **64-bit word-parallel Myers Levenshtein** (**22.27×** over scalar DP) + Jaro–Winkler & Jaccard; **SimHash64** (`3.00M docs/s`), **MinHash**, and UTF-8-safe zero-copy **RAG chunking**; **Topic Modeling** via **Sparse HALS NMF** (`22.9 ms` for 10k docs, $K=8$) and deterministic PCG64 **Collapsed Gibbs LDA** (`11.7 ms` for 2k docs, 20 sweeps); **Native Sparse Text Classification** directly on `RCRE_CSRMatrix` via **Multinomial / Complement Naive Bayes** (`0.09 ms` fit / **`49.5M docs/s`** predict) and **Multi-Class Softmax SGD** (`6.76 ms` for 10 epochs); plus **Zero-Copy Sentence Splitting**, **LexRank/TextRank + MMR Extractive Summarization** (`0.13 ms/doc`), and **Graph Keyword Extraction**.
* **HPC Numerical Layer (Phases H0–H9 Complete, `docs/hpc.md`)**:
  - **H0 Scientific Substrate (2.1.0)**: Dispatched F64 `gemm/gemv/dot/axpy/scale/sum` across scalar / NEON / AVX2 / AVX-512, Kahan-Neumaier compensated reductions (bit-exact on adversarial layouts), and special function library (`erf/erfc` via DLMF 8.9.8 CF, `lgamma`, `digamma`, `logsumexp`, `expit`, tail-accurate `norm_cdf`).
  - **H1 Reproducible Streams (2.2.0)**: **PCG64** bit-exact against NumPy's engine with $\mathcal{O}(\text{popcount}(n))$ skip-ahead and deterministic worker substreams; **Sobol'** ($d=1..20$) bit-exact against SciPy `scipy.stats.qmc.Sobol` (87,040 assertions) and **Halton** sequences.
  - **H2 General-Size F64 FFT & Blocked BLAS (2.2.0)**: Mixed-radix Cooley-Tukey + **Bluestein chirp-z** arbitrary-length F64 FFT (verified against direct $\mathcal{O}(N^2)$ DFT oracle across ~700 sizes), cache-blocked F64 GEMM, and triangular solve (`trsm`/`trsv`).
  - **H3 Direct Factorizations & Symmetric Eigensystem (2.2.0)**: Dense F64 **LU** (partial pivoting), **Cholesky** (`dpotrf`), **Householder QR**, and **`rcre_linalg_syev_f64`** symmetric eigensystem via Householder tridiagonalization + implicit QL with exact Givens similarity update (4096² in 53 ms).
  - **H4 Iterative Krylov Solvers & 2D Multigrid (2.2.0)**: Sparse F64/F32 **PCG/FCG**, **BiCGStab** (verified on 40-step trajectory parity against independent Python fixture), restarted **GMRES** with **ILU(0)** and **Jacobi** preconditioning; 2D-Poisson multigrid (V-cycle & FAS cycles) with documented $h$-dependent smoothest-mode floor and **PCG + ILU(0)** workhorse.
* **Out-of-core Zarr v3 Chunked Arrays (2.2.0, `docs/hpc.md` §8)**: `rcre_zarr_*` — chunked 1–8D arrays on the POSIX file store in the canonical **Zarr v3** format (12 numpy-compatible dtypes incl. f16/bool, fill-value semantics for missing chunks — NaN/±inf as JSON strings, identity `bytes` codec + **opt-in zstd** via a separate bridge TU (hook-dispatched registration, link-order immune; the single-header amalgamation excludes it and runs identity-only)). Slab-parallel whole-array I/O, `write_block` (strided sub-blocks, read-modify-write edge chunks), `for_each_chunk`, and a deterministic `reduce_sum` (per-chunk Neumaier/integral partials combined in fixed chunk order → **bit-identical across pool sizes**). Conformance: `make conformance-zarr` round-trips **13 fixed arrays bit-identical C ↔ zarr-python in both directions** (FNV-1a 64 gates + golden metadata + NaN fill survival; skips clean without a zarr venv). Measured (M5 Max): pooled 8-thread write ~550 MB/s (2.1–2.7x serial), read ~4.6 GB/s (page-cache-warm) at 16 MB; zstd L3 ~720 MB/s pooled write with 292x ratio on structured data. 9-test / 135-assertion unit gate + the cross-tooling conformance bar.
* **GPU Bridge: Context ABI + CUDA Backend (2.3.0, `docs/hpc.md` §9)**: `rcre_gpu_*` — one C ABI over self-registering backend TUs (the arrow/dlpack/zstd bridge pattern). The **`cpu` backend is always linked** and is the conformance oracle: its ops delegate to the core/HAL, so CPU-backend results are bit-identical to the CPU reference by construction. The **CUDA backend** is an opt-in TU pair (`gpu_cuda.c` + nvcc `gpu_kernels.cu`, Makefile auto-detect, infra-immune) with fixed launch configurations (no auto-tuning) and a no-atomics reduction design. **Philox-4x32-10** (core + device kernels) is **bit-identical across backends by construction** — the H6 RNG determinism bar (65,536-word gate). Device ops: GEMM f32/f64 (row-major, `alpha·A·B + beta·C`), SpMV f32/f64 (CSR), AXPY, DOT, Philox fill, host-pointer convenience. Measured on an **RTX 5060 Ti (sm_120, CUDA 13.3)**: SpMV up to 19x, GEMM f32 547 GF/s (9–19x) / f64 359 GF/s (19–26x), Philox fill **495x** (106 Gwords/s); the honest end-to-end CG+ILU(0) bar at n=4096 is 0.65x (per-iteration transfer floor — published, not hidden). 5-test / 363-assertion gate (device gates skip clean without CUDA).
* **Domain Layer: PDE, N-body, MCMC & Streaming Statistics (2.4.0, `docs/hpc.md` §10)**: F64 structured-grid **2-D Poisson** (separable DST-I spectral solver via the H2 FFT + the H4 V-cycle multigrid behind one AUTO driver, both paths on the H4 layout), **2-D heat FTCS** (stability-gated), **1-D Burgers MUSCL-upwind** (minmod + Roe-split face flux, `dt·max|u|/dx` CFL gate) with **Shu-Osher + Hopf-Lax** reference, and **1-D wave DDK leapfrog**; **2-D Barnes-Hut N-body** (single-global-permutation tree, point-mass `r^-3` law, slab-parallel force, velocity-Verlet, energy) — pool(8) bit-identical to serial on both; **MH / HMC / NUTS** samplers (exact slice-HMC; NUTS = H-G tree growth + exact uniform slice sampling over the nodes under the slice, acceptance-based dual averaging with a divergence guard after the measured retirement of the Stan-form DA on easy targets) with a Geyer initial-monotone-sequence **ESS** and plain/antithetic/stratified/Sobol'/Halton **quasi-MC integrators** — every chain bit-identical under a fixed PCG64 seed; and **streaming statistics** (Chan-merge Welford moments with bias-corrected Fisher skew/kurtosis, **P²** online quantile — rank error 0.0005 measured on 1e6 uniforms vs a 0.02 bar — and PCG64-substream **bootstrap** (ordinary + moving block), slab-parallel by construction). 37-test gate (`test_pde`, `test_nbody`, `test_mcmc`, `test_stream`); measured bars: Poisson AUTO 61.3 ms @ 1023² (residual 1e-13), Barnes-Hut 100k bodies 11.1 ms pooled (5.46x), NUTS 8-D anisotropic Gaussian per-dim sd within 4% (0 divergences).
* **Optimization Polish: L-BFGS, FISTA, CD-LASSO, SPSA (2.5.0, `docs/hpc.md` §11)**: F64 **two-loop L-BFGS** (Armijo backtracking, scipy-matching ftol/gtol tests, optional box bounds) measured against scipy 1.18.1 L-BFGS-B on an 8-problem fixture — final f within 1e-10 relative on every converged run (measured 1e-13..1e-16) and accepted iterations within 1.25x; **FISTA** (Beck-Teboulle, backtracking L) and **cyclic coordinate descent** for LASSO agree to 1e-16 relative objective on a 2000x500 instance (KKT 3e-9); **SPSA** zeroth-order (2-point and 4-point sign estimates, PCG64 perturbations — fixed-seed bit-identical runs). 98-check gate (`test_h8`); measured bars: L-BFGS rosenbrock-100 0.04 ms, CD-LASSO 2000x500 259.5 ms, FISTA-L1 318.4 ms (objective == CD to printed digits), SPSA-4 0.42 M evals/s (95.9% of the drop on quad-100 in 10k steps).
* **HPC Wave: MPI shim, distributed ops, 3-D Poisson 256³, pow2 FFT (2.6.0, `docs/hpc.md` §12)**: a **12-function MPI shim** over self-registering backends (`loopback` always present as the conformance oracle; real MPI optional behind a make probe — the H6 CUDA pattern), with the **first distributed operations**: a row-block distributed SpMV (+ all-to-all gather) and a **halo-exchange 5-point stencil** — bit-identical to the core serial operators (the 2-process gate runs under `mpirun -np 2` where MPI exists; skips clean without). A **3-D PDE layer** (7-point stencil + the separable DST-I 3-D Poisson solver) with the game-plan **256³ full run measured bit-reproducible** across pool sizes {NULL,1,2,4,8,18} (6/6 FNV-1a hashes identical; residual 1.8e-14 vs bar 1e-13) — 46 s serial / 4.1 s pooled (M5 Max, 18 cores); an irregular 33×31×29 cross-check against an independent numpy DST-I reference matches to 1.1e-15 relative. The H2 bar-gated FFT optimization ships as `rcre_fft_pow2_f64` (in-place pow2 F64 FFT, per-call twiddle table — 512-pt 8.6 µs vs 40.1 µs recursion, 4.7×) and `-ffp-contract=off` is now library-wide (measured: FMA contraction across code shapes broke the 256³ bit-identity). SVE2/AMX/RVV silicon validation and the cross-machine half of the reproducibility bar are deferred with the server silicon / RPi5 registration (documented, §12.4).

* **e-Means Mini-Batch Clustering Operator (`rcre_ml_emeans_*`)**: Faithful C23 port of the 2026 e-means algorithm (Bessa et al., arXiv 2611.09501) — exponential-moving-average centroid updates with a cosine-annealed step schedule, scaling-law batch/step auto-tuning (shipped-reference constants), and Gumbel-max dead-cluster revival with deterministic per-dimension splits. Dual-kernel exact assignment (per-row FP32 for $k \\cdot d \\le 16384$, HAL GEMM above), FP64 objective & finalization, and a pool-parallel slab partition that is **bit-identical across pool sizes and machines** (the partition depends only on the batch size). Verified: 205,790-assertion 0-ULP strict-parity replay against an independent Python oracle, plus statistical quality parity (C run at 0.589× the oracle's MSE). On an 8.3M-pixel palette workload (d=3, k=256) the pooled operator trains in ~1.1 s vs 34–78 s for 25-iteration Lloyd k-means; on 100K × 128-d embeddings with k=4096 it is 9× faster **and** 0.566× the final MSE.
* **Isolated Camera Companion (`rcre_cam.h`)**: Native hardware ingestion using macOS AVFoundation (`CVPixelBuffer` zero-copy), Linux Video4Linux2 (`mmap`), and deterministic synthetic test generators for headless CI.
* **High-Performance Scientific Data I/O (`rcre_io.h`)**: Zero-copy virtual memory mapping (`mmap` + `madvise` + `F_NOCACHE` + Win32 `MapViewOfFile`), 64-byte SIMD-aligned 8D **NumPy (`.npy` v1/v2)** and **Hugging Face (`.safetensors`)** containers, **2.61+ GB/s** multi-threaded SIMD CSV ingestion (with RFC-4180 quoted field support) and Clinger/Eisel–Lemire 0-ULP float32 parsing, **50.81 GB/s** ARM NEON byte-unshuffling with `"RCRZ"` LZ4-class lossless block compression, async `pthread` ping-pong `.npy` streaming, and zero-copy **DLPack v1.0** & **Apache Arrow C Data Interface** bridges. See [**io_benchmark.md**](io_benchmark.md) and [**docs/benchmarks_sota.md**](docs/benchmarks_sota.md).
* **Modern LLM Inference & LoRA Training (`rcre_llm.h`)**: LLaMA-style decoder (RMSNorm, RoPE, SwiGLU, Grouped-Query Attention, tiled online-softmax Flash-Attention), `Q8_0`/`Q4_0` block quantization with NEON/SIMD dot kernels, zero-copy **GGUF v2/v3** reader/writer, hash-indexed `RCRE_ByteBPE`, thread-safe sampler (top-k/top-p/min-p/rep-penalty), LoRA fine-tuning, BF16 gradient-checkpointed training, sharded binary token streaming, plus local interactive CLI & OpenAI-compatible HTTP server (`bin/rcre_chat --serve`).
* **Zero-Dependency Image Codecs, ORB Features, Minimal ONNX & Optional Metal/WebGPU (`rcre_codec.h`, `rcre_gpu.h`)**: Built-in PNG (DEFLATE/zlib), baseline JPEG, BMP, and PPM codecs; FAST-9 + rotated 256-bit BRIEF (`RCRE_ORBKeypoint`) feature matching; zero-dependency ONNX Protobuf subset graph runner; Random Forest, L1 Sparse Linear, Lion, Adafactor; optional macOS Metal compute offload, WebGPU bridge, freestanding no-malloc profile (`-DRCRE_FREESTANDING`), CMake/Homebrew/vcpkg packaging, and Python/Rust/Go/Zig/Julia bindings.
* **Drop-in Single-Header Release**: Amalgamated into a single portable header file [`dist/rcre.h`](dist/rcre.h) (**1,964.9 KB**, **52,666 lines**).
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
| **UTF-8 SIMD Validation (16 MB ASCII)** | `rcre_hal.utf8_validate` (NEON) | **44.50 GB/s** (**10.68x vs scalar**) | **0.36 ms** | Zero mallocs |
| **UTF-8 SIMD Validation (16 MB Multi-Byte)** | `rcre_hal.utf8_validate` (NEON) | **13.26 GB/s** (**3.45x vs scalar**) | **1.21 ms** | Zero mallocs |
| **Substring Search (16 MB `find_substr`)** | `rcre_hal.find_substr` (NEON) | **25.06 GB/s** (**9.40x vs scalar**) | **0.64 ms** | Zero mallocs |
| **ASCII Lowercase Conversion (16 MB)** | `rcre_hal.ascii_to_lower` (NEON) | **34.18 GB/s** | **0.47 ms** | Zero mallocs |
| **Arena StrPool Interning & Lookup** | `rcre_strpool_intern` (4096 keys) | **68.93M lookups / sec** | **14.5 ns** / op | Linear Arena (0 mallocs) |
| **Zero-Copy Word Tokenization (16 MB)** | `RCRE_WordIter` (UTF-8 pull iter) | **663.94 MB/s** (**92.76M tok/s**) | **24.1 ms** | Zero allocations |
| **Aho–Corasick 16-Pattern Flat DFA (16 MB)** | `rcre_ac_scan` (1 load/byte) | **550.78 MB/s** (1.17M hits) | **29.1 ms** | Flat Arena DFA (0 mallocs) |
| **Myers64 Bit-Parallel Levenshtein (61 B)** | `rcre_nlp_levenshtein` (64-bit) | **7.69M ops / sec** (**22.27x vs DP**) | **130 ns** / op | Zero allocations (registers) |
| **BM25 Inverted Index (10k docs)** | `rcre_bm25_build` / `search` (CSR) | **3,957 QPS** (top-10) | **11.33 ms** build / **253 µs** q | Linear Arena (0 mallocs) |
| **TF-IDF → `RCRE_CSRMatrix` (10k docs)** | `rcre_tfidf_fit` + `transform` | **561.7k docs / sec** (`nnz=109k`) | **17.80 ms** | 64B-Aligned CSR Arena |
| **Sparse HALS NMF Topic Modeling (10k, K=8)** | `rcre_nmf_fit` (20 HALS iters) | **436k doc-sweeps / sec** | **22.94 ms** | Linear Arena (0 mallocs) |
| **Collapsed Gibbs LDA (2k docs, K=8)** | `rcre_lda_fit` (PCG64, 20 sweeps) | **3.42M doc-sweeps / sec** | **11.69 ms** | Linear Arena (0 mallocs) |
| **Sparse Multinomial Naive Bayes (10k)** | `rcre_nlp_nb_fit` / `predict` | **49.5M docs / sec** predict | **0.09 ms** fit / **0.20 ms** pred | Zero dense expansion |
| **Sparse Multi-Class Softmax SGD (10k)** | `rcre_nlp_sparse_clf_fit` (10 ep) | **14.78M docs / sec / ep** | **6.76 ms** | Zero dense expansion |
| **LexRank + MMR Summarization (64 snt)** | `rcre_nlp_summarize_extractive` | **7,692 docs / sec** | **0.13 ms** / doc | Zero-copy sentence slices |
| **SimHash64 Dataset Fingerprinting (10k)** | `rcre_nlp_simhash64` + `popcnt` | **3.00M docs / sec** | **3.34 ms** | Zero allocations |
| **General-Size F64 FFT ($2^{20}$ points)** | `rcre_fft_1d_f64` (Mixed-Radix) | **7.4 Ms/s** (**735.7 Mflop/s**) | **141.7 ms** | Scoped Arena |
| **Bluestein Prime F64 FFT ($10^6+3$ points)** | `rcre_fft_1d_f64` (Chirp-Z) | **1.1 Ms/s** (**110.9 Mflop/s**) | **899.8 ms** | Scoped Arena |
| **Direct Dense LU Solve ($2048 \times 2048$ F64)** | `rcre_direct_lu_factor` (Partial Pivoting) | **13.9 GFLOP/s** | **413 ms** | Linear Arena |
| **Symmetric Eigensystem ($4096 \times 4096$ F64)** | `rcre_linalg_syev_f64` (Tridiag + QL) | **865.7 GFLOP/s eq** (Values+Vec) | **53 ms** | Linear Arena |
| **Poisson $512^2$ PCG + ILU(0) Solve** | `rcre_krylov_pcg` + `rcre_sparse_ilu0` | **$\mathcal{O}(N)$ 556 iters** (rel err 9.7e-11) | **1.79 s** solve | Scoped Arena |

📊 *Full Roofline & SOTA hardware results: [**docs/benchmarks_sota.md**](docs/benchmarks_sota.md) and [`web/benchmarks.html`](web/benchmarks.html).*

---

## 🛠️ Turnkey Production Projects (`projects/`)

In addition to modular core kernels and unit benchmarks, RCRE includes fourteen self-contained, turnkey production applications (plus planned CLI toolings) built 100% on RCRE and ISO C23:

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
| **`rcre-gen-studio`** | [`projects/rcre_gen_studio/`](projects/rcre_gen_studio/) | End-to-end generative & vision AI studio (ConvAE, ConvVAE, Diffusion Tiny U-Net, ViT, DiT) | Tensor Tape Autograd, Reparameterization, DDPM/DDIM, Batched MHSA, AdaLN-Zero, PNG/SafeTensors | `make project_gen_demo` |
| **`rcre-server-bench`** | [`projects/rcre_server_vs_go_benchmark/`](projects/rcre_server_vs_go_benchmark/) | High-performance C23 HTTP/1.1 kqueue reactor vs Go 1.27 web server benchmark | Zero-malloc SPSC kqueue reactor, BM25 Index, GBDT Inference, Arena Allocator | `make project_server_bench_verify` |
| **`rcre-cv-ops`** | [`projects/rcre_cv_ops/`](projects/rcre_cv_ops/) | **[TODO / Planned]** Turnkey computer vision & image processing CLI (absorbing retired `rcvc` ops) | Continuous GM Resizing (16 Filters), Convolutions, Colorimetry, e-Means Palette, Sixel/Kitty Display | *Planned (Roadmap)* |

📖 *For complete CLI documentation and architectural guides, see [`projects/README.md`](projects/README.md).*

---

## 📁 Repository Architecture

```
apps/rcre/
├── dist/
│   └── rcre.h                  # Amalgamated single-header C23 release (1,363.6 KB, 37,667 lines)
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
│   ├── rcre.h                  # Core public C23 header (ABI v2.0.0, 8D broadcasting tensors, CV, ML, HPC)
│   ├── rcre_str.h              # Zero-malloc, SIMD-accelerated C23 safe string library (STR-REQ-01 to 06)
│   ├── rcre_nlp.h              # HPC zero-malloc C23 NLP engine: BM25, TF-IDF->CSR, NMF, LDA, Sparse NB/Softmax, TextRank, Aho-Corasick, Myers64 (NLP-REQ-01 to 10)
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
│   ├── core/                   # Arena, Slice, Str, 8D Tensor, Linalg, Reduce, SF, RNG, FFT, Direct, Krylov, Multigrid, I/O, DF
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
├── tests/                      # 37 Unit test suites (Core, Strings, Factorizations, Krylov, Multigrid, SOTA) + fuzz_io/fuzz_str
├── benchmarks/                 # Micro-benchmarks: HAL, Roofline, I/O, Strings (bench_str), HPC (bench_h2/h3/h4), Prof
├── docs/                       # SOTA benchmarks (docs/benchmarks_sota.md) & HPC Program manual (docs/hpc.md)
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

#### Safe, Zero-Malloc C23 String Processing (`include/rcre_str.h`)

```c
#define RCRE_BAN_UNSAFE_CSTR  // Poison strcpy, sprintf, strtok at compile time
#include "rcre.h"
#include "rcre_str.h"
#include <stdio.h>

int main(void) {
    // 1. Zero-cost compile-time string literal view (0 strlen overhead)
    RCRE_Slice greeting = RCRE_STR("Hello, High-Performance C23!");

    // 2. Stack-allocated bounded buffer (guaranteed NUL-terminated, sticky truncation)
    RCRE_STRBUF_STACK(buf, 64);
    rcre_strbuf_copy(&buf, greeting);
    rcre_strbuf_fmt(&buf, " [Sensor #%d]", 7);

    // 3. Strict RFC 3629 UTF-8 validation (SIMD accelerated up to 44.5 GB/s)
    size_t err_offset = 0;
    bool valid = rcre_utf8_validate(rcre_strbuf_as_slice(&buf), &err_offset);

    // 4. Safe printing using length-delimited macro pair (no heap allocation)
    printf("Message: " RCRE_STR_FMT " (Valid UTF-8: %s)\n",
           RCRE_STR_ARG(rcre_strbuf_as_slice(&buf)), valid ? "true" : "false");
    return 0;
}
```

#### HPC Zero-Malloc NLP & Hybrid Lexical Search (`include/rcre_nlp.h`)

```c
#define RCRE_BAN_UNSAFE_CSTR
#include "rcre_nlp.h"
#include <stdio.h>

int main(void) {
    rcre_hal_init();
    RCRE_Arena* arena   = rcre_arena_create(1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(256 * 1024);

    const RCRE_Slice docs[3] = {
        RCRE_STR("zero malloc c23 arena allocator simd neon"),
        RCRE_STR("bm25 inverted index lexical search with csr postings"),
        RCRE_STR("hybrid rag fuses bm25 lexical and hnsw vector search")
    };

    // 1. Flat CSR-layout BM25 inverted index (2 linear O(tokens) passes, 0 mallocs)
    RCRE_BM25Index idx;
    (void)rcre_bm25_build(&idx, docs, 3, nullptr, arena, scratch);

    RCRE_SearchHit hits[3];
    size_t n_hits = 0;
    (void)rcre_bm25_search(&idx, RCRE_STR("bm25 lexical search"), 3, hits, &n_hits, scratch);

    // 2. Scikit-Learn-parity TF-IDF directly into 64B-aligned RCRE_CSRMatrix
    RCRE_TfidfVectorizer vec;
    RCRE_CSRMatrix csr;
    (void)rcre_tfidf_fit(&vec, docs, 3, nullptr, arena, scratch);
    (void)rcre_tfidf_transform(&vec, docs, 3, arena, scratch, &csr);

    printf("Top BM25 doc=%u (score=%.3f) | TF-IDF CSR: %zux%zu (nnz=%zu)\n",
           hits[0].doc_id, hits[0].score, csr.rows, csr.cols, csr.nnz);

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
    return 0;
}
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

# 🧪 Run full 42-test unit test suite (Core, Strings, NLP, Factorizations, Krylov, Multigrid, I/O, SOTA)
make test

# 🛡️ Run AddressSanitizer & UndefinedBehaviorSanitizer checks across all 42 suites (0 leaks, 0 UB)
make test-sanitizers

# 🐛 Run deterministic ASan+UBSan fuzz regression suite over I/O, codecs, strings & NLP differential oracles
make test-fuzz

# 🐍 Run Python numerical parity tests against OpenCV, SciPy & Scikit-Learn + FFI tests
make test-parity

# 🎉 Run unit tests, sanitizers & Python parity layers together
make test-all

# 📊 Run HAL GEMM, Roofline, Scientific I/O, Strings, NLP & HPC micro-benchmarks
make bench
make bench_str
make bench_nlp
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

## 🛡️ Target Platforms & Module Stability Tiers

RCRE focuses on **Apple Silicon (macOS ARM64)**, **Linux x86_64 scientific/HPC clusters**, and **Linux ARM64 edge devices (Raspberry Pi, NVIDIA Jetson)** (plus WebAssembly and `-DRCRE_FREESTANDING` embedded targets).

- **Tier 1 (Production-Stable Core)**: `RCRE_Arena`, `RCRE_Tensor`/`RCRE_Slice`, `rcre_str.h`, race-free `rcre_hal` + `RCRE_ThreadPool`, `rcre_io.h`, `rcre_nlp.h` (incl. `RCRE_BM25Scratch`), `rcre_sparse.h`, `rcre_df.h`, `rcre_stat.h`/`prep`/`metric`, HPC H0–H4 math, GBDT/HNSW/e-Means/classical ML, core CV, and `rcre_prof.h`.
- **Tier 2 (Preview / Hardening)**: `rcre_codec.h` (PNG/JPEG/BMP/PPM & ONNX subset — fuzz-tested via `fuzz_codec` & `fuzz_onnx`), `rcre_llm.h` (GGUF v2/v3 — fuzz-tested via `fuzz_gguf`), `rcre_autograd.h`, OS extras (`include/extras/rcre_http.h`, `rcre_cam.h`, `rcre_gpu.h`), and starter polyglot bindings.

See the full platform matrix, fuzzing coverage, and ABI policy in [**docs/STABILITY.md**](docs/STABILITY.md).

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

