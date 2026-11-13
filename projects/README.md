# RCRE Production Projects Hub

This directory contains complete, turnkey, plug-and-play production applications built 100% on **RCRE** and C23. 

Unlike the unit benchmarks in `examples/`, each project here is a fully functional, self-contained end-to-end tool designed for immediate real-world deployment.

---

## 📁 Available Projects

| Project | Directory | Purpose | Core RCRE Technologies |
| :--- | :--- | :--- | :--- |
| **`rcre-lens`** | [`reverse_image_search/`](reverse_image_search/) | High-speed reverse image search engine & visual database indexing | Spatial Pyramid Features, Sobel Gradients, HNSW Vector Index |
| **`rcre-facex`** | [`facial_expression_analysis/`](facial_expression_analysis/) | Real-time face detection, 13-point landmarks, FACS Action Units & emotion classification | Skin SAT Integral Image, Valley Tracking, FACS Biomechanics, Softmax |
| **`rcre-mobilenet`** | [`cifar100_mobilenet/`](cifar100_mobilenet/) | Sub-4ms MobileNet CNN training and zero-heap inference on CIFAR-100 | Depthwise Separable Conv, Edge BN Folding, SIMD axpy, Softmax |
| **`rcre-pca`** | [`pca_compression/`](pca_compression/) | Zero-dependency spatial PCA image compression & 8-bit quantization codec | Power Iteration, Deflation, 8-bit Uniform Quant, Web Playground Studio |
| **`rcre-perf`** | [`c_performance/`](c_performance/) | Systems microarchitecture, cache hierarchy, strided view & SIMD benchmark suite | Memory Arenas, ShapeTracker Views, HAL SIMD Dispatch, Perfetto Telemetry |
| **`rcre-sr`** | [`document_super_resolution/`](document_super_resolution/) | Document image super-resolution, Wiener deconvolution, Lanczos-3 & adaptive OCR pipeline | Separable Lanczos-3 / Bicubic, Wiener Deconvolution (2D FFT), Guided Denoise, CLAHE, Otsu & Adaptive Threshold |
| **`rcre-wavelet`** | [`wavelet_compression/`](wavelet_compression/) | Reversible multi-resolution integer wavelet image codec (lossless to 32x lossy) | LeGall 5/3 Integer Lifting, 2D DWT Pyramid, Dead-Zone Quant, Run-Length Codec |
| **`rcre-xgb`** | [`rcre_xgb/`](rcre_xgb/) | XGBoost-style regularized second-order gradient boosted decision trees | Quantile Histograms, Newton Updates, Missing NaN Routing, Flat-Array Predictor |
| **`rcre-omnisearch`** | [`omnisearch_hash/`](omnisearch_hash/) | 30M multimodal vector search & LSH hash engine for mixed text and images | L2 Spherical Normalization, Random Hyperplanes, Arena Hash Table, SIMD ADC |
| **`rcre-galton`** | [`rcre_galtonboard/`](rcre_galtonboard/) | High-speed Galton Board (Quincunx) flow simulator, Bernoulli trials & Central Limit Theorem | xoshiro256** PRNG, SIMD Bit-Level Popcount, Dynamic Histograms, Chi-Square Goodness-of-Fit, WebAssembly Canvas2D |

---

## 1. `rcre-lens` (Reverse Image Search Engine)

A zero-dependency CLI application to index folders of photos and search visually similar images in sub-millisecond query time.

### Key Capabilities
* **Zero Deep-Learning Dependencies**: No PyTorch, no ONNX, no Python runtimes.
* **Deterministic 64-D Spatial Pyramid Descriptors**: Extracts regional color moments and Sobel contour energies across $4 \times 4$ spatial grid cells.
* **Logarithmic Proximity Graph Retrieval**: Powered by RCRE's in-memory HNSW index.
* **Zero Heap Allocations in Hot Paths**: Feature extraction and vector searching run entirely inside 64-byte aligned linear scratchpad memory arenas (`RCRE_Arena`).

### Quick Start
```bash
# Run the complete test suite (extractor + serializer + HNSW invariants):
make project_lens_test

# Run the automated end-to-end gallery demo:
make project_lens_demo
```

For complete CLI documentation, architecture diagrams, and binary file specifications, see [`projects/reverse_image_search/README.md`](reverse_image_search/README.md).

---

## 2. `rcre-facex` (Facial Landmark & Expression Analysis Engine)

A real-time, zero-dependency computer vision application to detect human faces, localize 13 anthropometric landmarks, compute biomechanical FACS Action Units, and classify 6 affective states (Happy, Sad, Anxious, Panic, Angry, Neutral) at 400+ FPS.

### Key Capabilities
* **Integral Image Skin SAT Detection**: $\mathcal{O}(1)$ rectangular queries via YCbCr chrominance locus gating and multi-scale sliding windows.
* **13-Point Anthropometric Localization**: Sub-pixel dark-valley tracking for pupils, eyebrow ridges, nose apex, and lip contours with Inter-Ocular Distance ($\text{IOD}$) scale normalization.
* **Biomechanical FACS Action Unit Engine**: Scale-invariant Smile Curvature (AU12), Mouth Aspect Ratio (MAR AU26/27), Brow Elevation (AU1+AU2), and Sobel Glabella (AU4) wrinkle energy.
* **Numerically Stable Softmax Emotion Classifier**: High-precision categorical probability distributions with real-time ASCII HUD feedback.
* **Zero Heap Allocations**: 100% linear `RCRE_Arena` memory management with zero-cost frame rewinds.

### Quick Start
```bash
# Run the full 9-invariant verification suite:
make project_facex_test

# Run the automated expression archetype demonstration:
make project_facex_demo
```

---

## 3. `rcre-mobilenet` (MobileNet CIFAR-100 Inference & Training Pipeline)

A zero-heap, sub-4ms MobileNet inference engine and PyTorch training pipeline on CIFAR-100 ($32 \times 32$ image classification) implemented in ISO C23.

### Key Capabilities
* **Zero-Heap Allocations**: Entire inference graph runs inside linear memory arenas (`RCRE_Arena`) with zero runtime heap allocations.
* **Zero-Cost Edge BatchNorm Folding**: Fuses BatchNorm scale and shift directly into Conv2D and Depthwise Conv weights and biases.
* **SIMD Hardware Acceleration**: Pointwise convolutions accelerated via ARM NEON / AVX2 `axpy_f32`, and linear head accelerated via `dot_f32`.
* **High-Throughput Sub-4ms Latency**: Executes forward pass in ~3.24 ms (>300 FPS) on Apple Silicon CPU.
* **End-to-End Pipeline**: Includes full PyTorch training script with learning rate scheduling, model checkpointing, weight export, and test sample generation.

### Quick Start
```bash
# Run the automated 10-sample CIFAR-100 demo:
make project_mobilenet_demo

# Run the 50-iteration latency benchmark:
make -C projects/cifar100_mobilenet benchmark
```

For complete CLI documentation, architecture diagrams, and weight formats, see [`projects/cifar100_mobilenet/README.md`](cifar100_mobilenet/README.md).

For complete CLI documentation, mathematical formulas, and rendered pipeline diagrams, see [`projects/facial_expression_analysis/README.md`](facial_expression_analysis/README.md).

---

## 4. `rcre-pca` (PCA Image Compression & Quantization Engine)

A pure C23, zero-dependency dimensionality reduction codec that projects spatial image patches onto principal orthogonal eigenvectors and applies 8-bit uniform scalar quantization.

### Key Capabilities
* **High Compression with Linear Arenas**: Yields $8.8\times$ to $25\times$ compression with $0$ dynamic heap reallocations in loops.
* **Fidelity & Rate-Distortion**: Delivers $33.5\text{ dB}$ to $58.0\text{ dB}$ PSNR with sub-millisecond SIMD inverse reconstruction.
* **Standalone Binary Format (`.rpca`)**: Portable binary bitstream serialization with Little-Endian headers.
* **🌐 Interactive Web Studio**: Test live in your browser with webcam feeds, filesystem storage, error heatmaps, and rate-distortion sweeps at [`web/playground.html`](../web/playground.html) or [live on GitHub Pages](https://erolrecep.github.io/rcre/playground.html#pca-lab-section).

### Quick Start
```bash
# Compile standalone CLI tool:
make -C projects/pca_compression all

# Run automated end-to-end compression & reconstruction demo:
./projects/pca_compression/bin/rcre-pca --demo

# Run rate-distortion benchmark on an image:
./projects/pca_compression/bin/rcre-pca --benchmark [image.bmp] -b 8
```

For complete CLI documentation, mathematical formulas, and benchmark tables, see [`projects/pca_compression/README.md`](pca_compression/README.md).

---

## 5. `rcre-perf` (Hardware Architecture & Systems Performance Suite)

A turnkey, zero-dependency systems benchmarking suite exploring how CPU microarchitectures, cache hierarchies, tensor strides, and SIMD vector execution directly dictate real-world latency.

### Key Capabilities
* **Cache Locality & Streaming**: Quantifies spatial prefetching on Apple Silicon (128-byte cache lines) and x86 (64-byte), demonstrating **6.5x speedups** for row-major vs column-major thrashing.
* **ShapeTracker Strided Views**: Evaluates $\mathcal{O}(1)$ zero-copy transposition view creation (0 ns, 0 bytes memory allocated) vs contiguous memory repacking.
* **SIMD Hardware Acceleration**: Evaluates dynamic HAL micro-kernels (NEON / AVX2 axpy, dot, gemm) delivering **25.6x speedups** and **66+ GFLOPS**.
* **Integrated Profiling Telemetry**: Emits full hardware cycle counts, memory RSS delta, and Chrome Tracing JSON traces for [Perfetto UI](https://ui.perfetto.dev).

### Quick Start
```bash
# Run the live demonstration from the RCRE root:
make project_perf_demo

# Run the test suite:
make project_perf_test

# Run full benchmark with Perfetto trace and interactive HTML export:
make project_perf_bench
```

For complete CLI documentation, architectural deep-dive, and hardware tables, see [`projects/c_performance/README.md`](c_performance/README.md).

---

## 6. `rcre-sr` (Document Image Super-Resolution & OCR Restoration Pipeline)

A zero-dependency ISO C23 document enhancement and arbitrary-scale super-resolution pipeline designed to transform degraded, blurry, or low-resolution document captures into high-contrast, OCR-ready images at 70+ MPix/s.

### Key Capabilities
* **Arbitrary-Scale Separable Resampling**: Implements document-imaging standard Lanczos-3 and Catmull-Rom Bicubic interpolation with precomputed weight tables and zero per-pixel transcendental calls.
* **FFT-Accelerated Wiener Deblurring**: Frequency-domain deconvolution using RCRE's 2D FFT, zero-padded to power-of-two with configurable PSF models (Gaussian camera blur or motion blur).
* **Linear-Time Guided Filtering**: $\mathcal{O}(N)$ edge-preserving denoising via 4 integral images per channel with zero runtime heap allocation.
* **Gated Document Enhancement**: Multi-scale CLAHE, Otsu thresholding, Bradley-Roth adaptive binarization, binary morphology (open/close), and projection-profile deskewing ($\pm 15^\circ$).
* **Gated Execution Policy**: Stages only fire when automated input quality assessment detects blur ($\text{Laplacian variance} < 60$), noise ($\sigma > 3.0$), or low contrast ($\text{range} < 0.55$), ensuring clean documents are never over-processed.
* **High Quality & Test Coverage**: >91% line coverage across all modules, 100% clean under AddressSanitizer and UndefinedBehaviorSanitizer, and verified +3.1% OCR CER improvement on Tesseract.

### Quick Start
```bash
# Run the complete test suite (84 unit & integration checks):
make project_sr_test

# Run the automated end-to-end demo (generate degraded document -> restore -> OCR report):
make project_sr_demo

# Run the arbitrary-scale CLI tool:
./projects/document_super_resolution/bin/sr_cli enhance input.pgm output.pgm --scale 2 --style auto
```

For complete CLI documentation, architecture diagrams, and benchmark tables, see [`projects/document_super_resolution/README.md`](document_super_resolution/README.md).

---

## 7. `rcre-wavelet` (Multi-Resolution Integer Wavelet Image Codec)

A pure C23, zero-dependency reversible image codec implementing the Biorthogonal LeGall 5/3 lifting scheme across multi-scale 2D DWT subband pyramids.

### Key Capabilities
* **100% Bit-Exact Reversible Lossless Mode**: When quantization step $\Delta = 1$, delivers mathematical bit-exact reconstruction ($\text{MSE} = 0.0000, \text{PSNR} = 99.0\text{ dB}$).
* **High Rate-Distortion Lossy Compression**: Up to $32\times$ compression with $> 54\text{ dB}$ PSNR on natural textures without blocking artifacts.
* **Zero Runtime Heap Allocation**: Entire forward/inverse pyramid transforms execute in-place using scratchpad memory arenas (`RCRE_Arena`).
* **Microcontroller-Ready Pure Integer Math**: Eliminates floating-point drift and runs identically on ARM Cortex-M microcontrollers.
* **Standalone Binary Bitstream (`.rdwt`)**: Self-contained 32-byte header with variable-length run-length entropy stream.

### Quick Start
```bash
# Compile standalone CLI tool:
make -C projects/wavelet_compression all

# Run automated end-to-end demo (Lossless vs Lossy verification):
make -C projects/wavelet_compression demo

# Run rate-distortion benchmark on an image:
./projects/wavelet_compression/bin/rcre-wavelet --benchmark [image.bmp]
```

For complete CLI documentation, mathematical formulas, and benchmark tables, see [`projects/wavelet_compression/README.md`](wavelet_compression/README.md).

---

## 8. `rcre-xgb` (XGBoost-Style Gradient Boosted Decision Trees)

A pure C23, zero-dependency second-order gradient boosted decision tree engine implementing regularized Newton updates, histogram-based split finding, missing value default direction learning, and flat-array zero-allocation inference.

### Key Capabilities
* **Second-Order Newton Updates**: Computes analytical gradients and Hessians for squared-error regression, binary logistic classification, and multi-class softmax.
* **Exact $L_1$ and $L_2$ Leaf Regularization**: Analytical soft-thresholding via $\alpha$ and shrinkage via $\lambda$, with $\gamma$ split penalization.
* **Optimal Missing-Value Default Routing**: Evaluates left vs. right gain candidates to learn the optimal default direction for missing values (`NaN`).
* **Subsampling & Early Stopping**: Row and column subsampling with deterministic SplitMix64 PRNG, and automated model truncation at the validation loss minimum.
* **Blistering Flat-Array Inference**: Achieves **1.18 Billion rows/s** ($0.8\,\mu\text{s}$ per row across 82 trees) with $1.79 \times 10^{-7}$ numerical parity against RCRE's native `rcre_ml_gbdt_predict`.
* **Zero Runtime Heap Allocation**: Dataset binning, tree growth, and boosting loop execute within a single linear memory arena (`RCRE_Arena`).
* **Bit-Exact Binary Model Persistence**: 32-byte header `"RCXG"` format storing 24-byte cache-aligned node structs.

### Quick Start
```bash
# Compile library, CLI, tests, and benchmarks:
make -C projects/rcre_xgb all

# Run complete 55-check verification suite:
make project_xgb_test

# Run synthetic regression + binary classification demo:
make project_xgb_demo

# Run scaling, per-stage timing, and RCRE parity benchmarks:
make project_xgb_bench

# Train on a custom CSV dataset:
./projects/rcre_xgb/bin/xgb_cli train data.csv --task binary --label 3 --rounds 50 --depth 4 --eta 0.3 --out model.rcxg
```

For complete design documents, benchmark tables, and CLI guides, see [`projects/rcre_xgb/xgboost_report.md`](rcre_xgb/xgboost_report.md).

---

## 9. `rcre-omnisearch` (Multimodal Vector Search & LSH Hash Engine)

A high-performance C23 vector search engine combining Locality-Sensitive Hashing (LSH) with arena-backed open-addressing hash tables to query 30 Million mixed text sentences and images for Top-20 results with zero heap allocations.

### Key Capabilities
* **Unit Hypersphere Normalization**: In-place $L_2$ Spherical Normalization guaranteeing exact equivalence between Cosine Similarity and Euclidean Distance.
* **Locality-Sensitive Hashing (LSH)**: 64 random Gaussian hyperplanes partitioning continuous embedding space into 64-bit signatures.
* **Open-Addressing Hash Table**: Power-of-two table sizing with linear probing and cache-aligned contiguous postings arrays.
* **Hardware SIMD & Popcount**: $> 71\text{ Million SIMD dot products / sec}$ and single-cycle bitwise Hamming distance filtering.
* **Zero Runtime Heap Allocation**: All query structures, visited bitsets, and bounded min-heap priority queues operate inside an `RCRE_Arena`.
* **Sub-Millisecond Retrieval**: $100,000+\text{ QPS}$ query throughput and $< 0.02\text{ ms}$ query latency with $97.25\%+$ Recall@20.

### Quick Start
```bash
# Run unit tests and recall parity verification:
make project_omnisearch_test

# Run multimodal synthesis, index build, and Top-20 query demo:
make project_omnisearch_demo

# Run SIMD and hash table micro-benchmarks:
make project_omnisearch_bench
```

For complete CLI documentation and binary file specifications, see [`projects/omnisearch_hash/README.md`](omnisearch_hash/README.md).

---

## 10. `rcre-galton` (Galton Board Quincunx Flow & CLT Simulator)

A high-throughput ISO C23 simulation engine and interactive visualizer modeling the emergence of the Gaussian Normal Distribution from discrete Bernoulli trials (Central Limit Theorem).

### Key Capabilities
* **Bit-Level SIMD Popcount Saturation**: Hardware popcount over 64-bit `xoshiro256**` words evaluating 64-row lattices at **750+ Million drops/sec** (1.33 ns/drop).
* **Mathematical Parity**: Exact Binomial PMF computed in log-space (`lgamma`) without floating-point overflow, and Pearson Chi-Square Goodness-of-Fit testing with incomplete gamma p-value calculation.
* **Live ANSI Terminal Visualizer**: Interactive terminal animation with colored marble deflection trails, pin lattice, and Unicode vertical block histograms.
* **Portable Web & WebAssembly Visualizer**: Zero-dependency HTML5 Canvas2D / Web Audio visualizer with continuous particle kinematics, restitution, gravity, and live pentatonic audio synthesis.
* **Python 3.15 FastAPI Streaming Service**: Modern asynchronous service with REST APIs, WebSocket real-time drop streaming, and strict Pydantic v2 validation.

### Quick Start
```bash
# Run the C23 AddressSanitizer & UBSan test suite:
make project_galton_test

# Run the 1,000,000-ball Monte-Carlo demo:
make project_galton_demo

# Run the 20,000,000-ball throughput benchmark:
make project_galton_bench
```

For complete documentation, see [`projects/rcre_galtonboard/README.md`](rcre_galtonboard/README.md).




