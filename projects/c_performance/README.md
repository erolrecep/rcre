# RCRE-PERF: Hardware Microarchitecture & Performance Benchmark Suite

A turnkey, zero-dependency systems benchmarking suite built directly on **RCRE** and ISO C23. 

`rcre-perf` demonstrates how hardware cache hierarchies, memory controllers, tensor strides, and SIMD vector execution units dictate real-world performance on modern processors (such as Apple Silicon M-series and modern x86-64 chips).

---

## 🚀 Key Architectural Capabilities

| Benchmark | Microarchitectural Focus | Core RCRE Primitive | Real-World Hardware Impact |
| :--- | :--- | :--- | :--- |
| **01. Cache Locality** | Cache lines (128B / 64B), spatial prefetching, cache thrashing, L1/L2 tiling | `RCRE_Arena` (64-byte aligned linear allocation) | **6.5x faster** throughput via sequential cache line streaming vs column thrashing |
| **02. Strided Views** | `ShapeTracker` zero-copy permutations vs physical memory materialization | `RCRE_Tensor` (`rcre_tensor_transpose`, `rcre_tensor_is_contiguous`) | **0 ns / 0 byte** $\mathcal{O}(1)$ view creation; break-even analysis for tensor repacking |
| **03. SIMD Vectorization** | Vector registers, fused multiply-accumulate (FMA), arithmetic intensity | `RCRE_HAL` (ARM NEON 128-bit / AVX2 256-bit dynamic dispatch) | **25.6x speedup** in GEMM compute throughput (**66+ GFLOPS**) |
| **04. System Telemetry** | Monotonic cycle counting, RSS memory tracking, Perfetto trace visualization | `RCRE_Prof` (Pillars 1–4 hardware telemetry engine) | Export Chrome Tracing JSON (`ui.perfetto.dev`) & interactive HTML dashboard |

---

## 📁 Project Structure

```text
c_performance/
├── Makefile                 # Independent and root-invocable build targets
├── README.md                # Architectural documentation and benchmark analysis
├── include/
│   └── rcre_perf.h          # Hardware probes (sysctl cacheline), configuration structs
├── src/
│   ├── main.c               # CLI interface and telemetry reporting
│   ├── bench_cache.c        # Benchmark 01: Cache locality & blocked tiling
│   ├── bench_strides.c      # Benchmark 02: Strided tensor views vs contiguous layout
│   └── bench_simd.c         # Benchmark 03: SIMD vectorization (AXPY, Dot, GEMM)
└── bin/                     # Compiled executables, Perfetto traces & HTML dashboards
```

---

## ⚡ Quick Start

### 1. Build and Run the Complete Suite
```bash
# Build the standalone binary
make

# Run the live demo (2048x2048 matrix)
make demo

# Run verification test (1024x1024 matrix)
make test

# Run full benchmark with Perfetto and HTML dashboard export (4096x4096)
make bench
```

### 2. Custom Command-Line Invocations
```bash
# Run cache locality benchmark on a massive 8192 x 8192 matrix (256 MB footprint):
./bin/rcre-perf --cache --matrix-size 8192

# Test custom L1/L2 sub-block tile sizes (e.g. 128x128):
./bin/rcre-perf --cache --tile-size 128

# Run only SIMD benchmarks:
./bin/rcre-perf --simd

# Run all benchmarks and export a visual timeline for Perfetto:
./bin/rcre-perf --all --perfetto trace.json --html report.html
```

---

## 🔬 Microarchitectural Deep-Dive

### 1. Cache Locality & Hardware Prefetchers (`bench_cache.c`)
Modern CPUs do not fetch individual floats or integers from physical DRAM. Instead, the memory bus moves contiguous blocks called **Cache Lines** (typically 64 bytes on x86-64, and **128 bytes on Apple Silicon M-series**):

* **Row-Major Traversal (`mat[i * N + j]`):** Elements sit sequentially in physical memory. When element $(i, j)$ is fetched, hardware prefetchers stream the remaining elements of the 128-byte line into L1 cache. The subsequent 31 accesses are zero-latency L1 cache hits.
* **Column-Major Traversal (`mat[j * N + i]`):** Jumps $N \times 4$ bytes on every iteration (e.g. 16,384 bytes for $N=4096$). Each read falls on a distinct cache line, evicting active cache sets and stalling execution on DRAM latency.
* **Tiled / Blocked Traversal:** Partitions the $N \times N$ domain into $64 \times 64$ sub-blocks (16 KB working set) that fit entirely within L1 cache capacity, restoring peak throughput.

### 2. Zero-Copy Strides vs Physical Repacking (`bench_strides.c`)
In high-performance tensor frameworks like RCRE, transposing or slicing a tensor modifies strides without copying data:
$$\text{stride}_{\text{transposed}} = [\text{stride}_1, \text{stride}_0]$$

* **Zero-Copy View Creation:** Takes **0 ns** and **0 bytes of memory allocation**.
* **Access Cost Trade-Off:** Traversing a non-contiguous transposed view incurs non-linear strides ($6.89\times$ slower due to missing prefetch streams).
* **The Break-Even Frontier:** Repacking the tensor into contiguous memory costs a one-time copy, but pays for itself if the tensor is evaluated more than $\approx 1.3$ times in downstream neural network layers or matrix operations.

### 3. SIMD Hardware Acceleration (`bench_simd.c`)
Compares unvectorized scalar C loops with RCRE's dynamic Hardware Abstraction Layer (`RCRE_HAL`):
* **ARM NEON (128-bit registers):** Processes 4 single-precision floats simultaneously per vector register with Fused Multiply-Accumulate (`fmla`).
* **Compute Throughput:** Single-precision general matrix multiplication (GEMM) achieves **$25.6\times$ higher throughput** (**66.13 GFLOPS**) over scalar loops.

---

## 📊 Benchmark Telemetry Results (Apple M5 Max)

Tested with $4096 \times 4096$ single-precision floating point tensors (64.00 MB footprint):

```text
===================================================================
   ⚡ RCRE-PERF: Hardware Architecture & Performance Suite         
===================================================================
 Host Processor         : Apple M5 Max
 Active HAL Backend     : ARM NEON 128-bit (SIMD Features: 0x0003)
 Hardware Cacheline     : 128 bytes
 Matrix Benchmark Size  : 4096 x 4096 (64.00 MB)
 Tile Blocking Size     : 64 x 64 (16.00 KB working set)
===================================================================

[1] Cache Locality Benchmark:
  • Row-Major Traversal    :   8.96 ms |  6.98 GB/s (High Locality)
  • Column-Major Traversal :  58.28 ms |  1.07 GB/s (Cache Thrashing)
  • Tiled Block Traversal  :   9.38 ms |  6.66 GB/s (Cache-Friendly)
  ➜ Speedup: Row-Major is 6.50x faster than Column-Major.

[2] Strided Tensor Views vs Contiguous Memory:
  • Contiguous Traversal   :   8.96 ms |  6.97 GB/s
  • Zero-Copy Transposed   :  61.74 ms |  1.01 GB/s (Created in 0 ns, 0 bytes allocated)
  • Repacked Contiguity    :  10.35 ms |  6.04 GB/s (One-time pack cost: 65.89 ms)
  ➜ Amortization Break-Even: Repacking pays off after 1.3 iterations.

[3] SIMD Vectorization & Arithmetic Intensity:
  • Vector AXPY (8M floats): Scalar 65.88 GB/s ➜ Active HAL 93.63 GB/s (1.42x)
  • Dot Product (8M floats): Scalar 13.68 GB/s ➜ Active HAL 52.19 GB/s (3.81x)
  • GEMM (512x512 Single)  : Scalar  2.58 GFLOPS ➜ Active HAL 66.13 GFLOPS (25.60x)

======================================================================
  TOTAL TELEMETRY CONSUMPTION (RCRE_Prof)
  Wall time                : 0.678 s
  CPU (user+sys)           : 0.676 s (100% of wall, 1.0 core-equivalent)
  RAM RSS Peak             : 142.2 MB (Delta: +134.2 MB)
  Arena HWM                : 134.2 MB (9 allocations, 3 rewinds)
  Timeline Export          : bin/perf_trace.json (Perfetto Chrome Trace)
  Dashboard Export         : bin/perf_dashboard.html (Interactive HTML)
======================================================================
```

---

## 📈 Visualizing with Perfetto

When compiled or run with `--perfetto <filename>.json`:
1. Open [https://ui.perfetto.dev](https://ui.perfetto.dev) in any web browser.
2. Drag-and-drop the generated `bin/perf_trace.json` into the window.
3. Inspect micro-second scope durations, CPU cycle events, and memory watermarks visually on the interactive timeline.
