# RCRE Production Projects Hub

This directory contains complete, turnkey, plug-and-play production applications built 100% on **RCRE** and C23. 

Unlike the unit benchmarks in `examples/`, each project here is a fully functional, self-contained end-to-end tool designed for immediate real-world deployment.

---

## 📁 Available Projects

| Project | Directory | Purpose | Core RCRE Technologies |
| :--- | :--- | :--- | :--- |
| **`rcre-lens`** | [`reverse_image_search/`](reverse_image_search/) | High-speed reverse image search engine & visual database indexing | Spatial Pyramid Features, Sobel Gradients, HNSW Vector Index |
| **`rcre-facex`** | [`facial_expression_analysis/`](facial_expression_analysis/) | Real-time face detection, 13-point landmarks, FACS Action Units & emotion classification | Skin SAT Integral Image, Valley Tracking, FACS Biomechanics, Softmax |

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

For complete CLI documentation, mathematical formulas, and rendered pipeline diagrams, see [`projects/facial_expression_analysis/README.md`](facial_expression_analysis/README.md).

