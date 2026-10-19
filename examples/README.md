# RCRE Standalone Production Examples & Benchmarks

This directory contains standalone, end-to-end production example applications demonstrating how to build high-performance Computer Vision and Classical Machine Learning pipelines using **RCRE** and the hardware camera companion module (`rcre_cam.h`).

All examples strictly adhere to RCRE's core architectural tenets:
* **Zero dynamic memory allocations (`malloc`) inside execution loops.**
* **Pure C23 standard compliance.**
* **Deterministic real-time latency and bounded memory footprint.**

---

## 📋 Available Examples

| Example Binary | Source File | Category | Key Highlights |
| :--- | :--- | :--- | :--- |
| `bin/webcam_edge_filter` | [`webcam_edge_filter.c`](webcam_edge_filter.c) | Real-Time Video / CV | Camera stream → Grayscale → Gaussian → Sobel at **239.39 FPS** |
| `bin/tabular_gbdt_predict` | [`tabular_gbdt_predict.c`](tabular_gbdt_predict.c) | Classical Tabular ML | 100-Tree ensemble inference at **7.86M predictions/sec** (127 ns latency) |
| `bin/hnsw_vector_search` | [`hnsw_vector_search.c`](hnsw_vector_search.c) | Vector Search / Embeddings | 64-D vector ANN similarity search at **44,434 QPS** (22.5 µs latency) |

---

## 🛠️ Quick Execution via Makefile

Run any example directly with a single command:

```bash
# Launch interactive camera test (select Terminal or GUI Window from menu)
make webcam_test

# 🖥️ Launch directly in Native GUI Window (Side-by-side: Camera RGB | Sobel Filter)
make webcam_test_gui

# 📟 Launch directly in Terminal (Real-time ANSI ASCII visualization)
make webcam_test_terminal

# 📷 Launch synthetic test pattern camera (ideal for headless CI/SSH)
make webcam_test_dummy

# Run high-throughput GBDT inference benchmark (100k samples)
make gbdt_test

# Run 64-D HNSW vector similarity search benchmark
make hnsw_test

# Compile all example binaries into bin/
make examples
```

---

## 1. Real-Time Camera Edge Detection (`webcam_edge_filter`)

Demonstrates zero-copy camera frame acquisition coupled with high-speed spatial image filtering.

### Pipeline Stages
1. **Frame Ingestion**: `rcre_cam_grab()` fetches a zero-copy $[H, W, 3]$ $U8$ frame via macOS AVFoundation, Linux V4L2, or Synthetic Dummy generator.
2. **Color Conversion**: `rcre_cv_rgb_to_gray()` computes 15-bit fixed-point Rec.601 luma (0 LSB difference from OpenCV).
3. **Spatial Smoothing**: `rcre_cv_gaussian_blur()` applies separable 2D Gaussian blur with reflection padding ($\sigma = 1.2$).
4. **Spatial Gradients**: `rcre_cv_sobel()` calculates horizontal ($dx$) and vertical ($dy$) gradient derivatives.
5. **Display Rendering**: Outputs to either a native hardware-accelerated GUI window or an ANSI ASCII console stream.
6. **Linear Scratchpad Rewind**: The memory arena offset is reset to baseline in $\mathcal{O}(1)$ time, reusing scratch buffers indefinitely with **zero heap allocations in the loop**.

---

### 🖥️ Display Modes Explained

When running the camera test, you can choose between two main display formats:

#### Mode 1: Terminal (ANSI ASCII Art)
* **What it does**: Downsamples edge intensities into an ANSI character ramp (` .:-=+*#%@`) directly inside your console.
* **Why use it**: Zero graphical dependencies; perfect for remote SSH sessions, headless servers, Docker containers, or tmux panes.
* **Command**: `make webcam_test_terminal` or `./bin/webcam_edge_filter --terminal`

#### Mode 2: Native GUI Window (Side-by-Side: Camera RGB | Sobel Filter)
* **What it does**: Opens a hardware-accelerated desktop window displaying **two screens side-by-side**:
  * **Left Screen (Input)**: The raw live camera feed (RGB 640×480) ingested directly from the hardware.
  * **Right Screen (Output)**: The real-time filtered output (Sobel edge derivative highlighted in cyber-cyan).
* **Why two screens?**
  * It provides an **instant input vs. output visual parity check**, making it immediately obvious how contours, motion, and edges are tracked in real time.
* **Controls**: Press <kbd>q</kbd> or <kbd>ESC</kbd> in the window (or click the red close button) to finish capture and print the performance summary.
* **Command**: `make webcam_test_gui` or `./bin/webcam_edge_filter --gui`

#### Mode 3: Native GUI Window (Single Screen: Sobel Output Only)
* **What it does**: Displays only the filtered 640×480 edge output in a single window.
* **Command**: `./bin/webcam_edge_filter --gui-single`

#### Mode 4: Headless Benchmark
* **What it does**: Ingests and processes frames at maximum CPU pipeline throughput without rendering.
* **Command**: `./bin/webcam_edge_filter --headless --frames 500`

---

### 🎮 Testing Options & Back-to-Back Workflow

You can test GUI and Terminal modes back-to-back:

```bash
# 1. Test in Terminal (ANSI ASCII):
make webcam_test_terminal

# 2. Test in Native GUI (Side-by-Side view):
make webcam_test_gui

# 3. Test with Single-Screen GUI:
./bin/webcam_edge_filter --gui-single --frames 200

# 4. Interactive Mode (prompts you each time):
make webcam_test
```

### Command-Line Arguments Reference

```text
Usage: bin/webcam_edge_filter [OPTIONS]

Display Options:
  --terminal, --view        Display live video in terminal via ANSI ASCII art
  --gui                     Display live video in Native GUI window (Side-by-side RGB | Sobel)
  --gui-single              Display live video in Native GUI window (Sobel edge output only)
  --headless, --bench       Run without display to measure maximum pipeline throughput

Configuration Options:
  --dummy                   Force synthetic moving color bars (ideal for headless CI/SSH)
  --frames <N>              Number of frames to process (default: 200, 0 = run until stopped)
  --width <W>               Frame capture width (default: 640)
  --height <H>              Frame capture height (default: 480)
  -h, --help                Show command-line help message
```

### Sample Output (Apple M5 Max)
```
===================================================================
  RCRE Example: Real-time Camera Edge Detection Pipeline           
===================================================================
Configuration:
  Resolution: 640 x 480
  Frames:     100
  Backend:    Synthetic Dummy (Forced)
Active Camera Backend: Synthetic Test Pattern (Dummy)

Starting pipeline loop (100 frames)...

===================================================================
  Performance & Benchmark Results                                  
===================================================================
Processed Frames:     100 / 100
Total Elapsed Time:   0.4177 seconds
Throughput:           239.39 FPS
Average Latency:      4.177 ms / frame
Pixel Throughput:     73.54 MP/s
Arena Memory Used:    3379328 / 33554432 bytes (10.1%)
Heap Allocations in Loop: 0 (100% deterministic linear arena)
===================================================================
```

---

## 2. High-Throughput Tabular GBDT Inference (`tabular_gbdt_predict`)

Demonstrates flat-array decision tree ensemble prediction with sub-microsecond latency.

### Key Architecture
* **Flat-Array Nodes**: All trees and branch nodes reside in a single contiguous, 64-byte cache-aligned array (`RCRE_TreeNode`), eliminating pointer chasing and cache misses.
* **Batch Evaluation**: Evaluates 100,000 tabular feature vectors against a 100-tree ensemble with branch prediction optimization and numerically stable sigmoid margin computation.

### Running the Example
```bash
bin/tabular_gbdt_predict
```

### Sample Output (Apple M5 Max)
```
===================================================================
  RCRE Example: High-Throughput GBDT Tabular Inference            
===================================================================
Model Architecture:
  Trees:            100
  Total Nodes:      700 (flat array, 64-byte cache aligned)
  Feature Vector:   16 continuous features
  Evaluation Load:  100000 samples

Executing GBDT batch prediction...

Sample Predictions:
  Sample #0: Predicted Prob = 0.0067 | Class = Negative [0]
  Sample #1: Predicted Prob = 0.0067 | Class = Negative [0]
  Sample #2: Predicted Prob = 0.0067 | Class = Negative [0]
  Sample #3: Predicted Prob = 0.9478 | Class = Positive [1]
  Sample #4: Predicted Prob = 0.4013 | Class = Negative [0]

===================================================================
  GBDT Benchmark Results                                           
===================================================================
Total Inferences:     100000
Elapsed Time:         0.0127 seconds
Throughput:           7865964.01 predictions / second
Average Latency:      0.127 microseconds / prediction
Arena Memory Used:    6817664 / 33554432 bytes (20.3%)
Heap Allocations:     0 (100% deterministic linear arena)
===================================================================
```

---

## 3. HNSW Vector Similarity Search (`hnsw_vector_search`)

Demonstrates Hierarchical Navigable Small World (HNSW) proximity graph indexing and sub-millisecond approximate nearest neighbor search for high-dimensional embeddings.

### Key Architecture
* **Graph Structure**: Multi-layer proximity graph with logarithmic search complexity $\mathcal{O}(\log N)$.
* **SIMD Distance Kernels**: Fast Euclidean $L_2$ and Cosine distance computations using hardware vector lanes (ARM NEON / AVX2).
* **Zero Allocation Queries**: Search priority queues and visited sets utilize caller-supplied scratchpad memory.

### Running the Example
```bash
bin/hnsw_vector_search
```

### Sample Output (Apple M5 Max)
```
===================================================================
  RCRE Example: HNSW Vector Similarity Search Benchmark            
===================================================================
Database Configuration:
  Vectors:          2000
  Dimensions:       64-D
  Graph M:          16
  ef_construction:  64

Building HNSW proximity graph...
  Index Built in 0.006 seconds (327600.3 vectors/sec)

Executing 200 ANN top-5 queries...

Top-5 Nearest Neighbors for Query #199:
  Rank 1: Vector ID 52    | L2 Distance = 32.35015
  Rank 2: Vector ID 3     | L2 Distance = 33.40972
  Rank 3: Vector ID 1     | L2 Distance = 33.54123
  Rank 4: Vector ID 5     | L2 Distance = 34.19794
  Rank 5: Vector ID 30    | L2 Distance = 34.47865

===================================================================
  HNSW Benchmark Results                                           
===================================================================
Queries Evaluated:    200
Query Throughput:     44434.57 QPS (Queries / sec)
Mean Query Latency:   22.51 microseconds (0.0225 ms)
Arena Memory Used:    5987328 / 67108864 bytes (8.9%)
===================================================================
```

---

## 🛡️ Memory Safety & Sanitizers

All examples are tested against Clang's AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
make test-sanitizers
```

Every example executes with **zero memory leaks** and **zero undefined behavior**.
