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


