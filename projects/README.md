# RCRE Production Projects Hub

This directory contains complete, turnkey, plug-and-play production applications built 100% on **RCRE** and C23. 

Unlike the unit benchmarks in `examples/`, each project here is a fully functional, self-contained end-to-end tool designed for immediate real-world deployment.

---

## 📁 Available Projects

| Project | Directory | Purpose | Core RCRE Technologies |
| :--- | :--- | :--- | :--- |
| **`rcre-lens`** | [`reverse_image_search/`](reverse_image_search/) | High-speed reverse image search engine & visual database indexing | Spatial Pyramid Features, Sobel Gradients, HNSW Vector Index |

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
# Run the Phase 1 feature extractor verification suite:
make project_lens_test
```
