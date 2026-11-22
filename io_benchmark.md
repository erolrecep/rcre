# RCRE Scientific Data I/O (`rcre_io`) — Hardware Bandwidth & Ecosystem Benchmark Report

**Subsystem:** [include/rcre_io.h](file:///Users/erolrecep/Developer/codebase/apps/rcre/include/rcre_io.h) & [src/core/io.c](file:///Users/erolrecep/Developer/codebase/apps/rcre/src/core/io.c) (Bundled into [dist/rcre.h](file:///Users/erolrecep/Developer/codebase/apps/rcre/dist/rcre.h))  
**Benchmark Harness:** [benchmarks/bench_io.c](file:///Users/erolrecep/Developer/codebase/apps/rcre/benchmarks/bench_io.c) (`make bench_io`)  
**Verification Suite:** [tests/test_io.c](file:///Users/erolrecep/Developer/codebase/apps/rcre/tests/test_io.c) (77,103 assertions) & [tests/test_against_scipy_opencv.py](file:///Users/erolrecep/Developer/codebase/apps/rcre/tests/test_against_scipy_opencv.py) (Parity Gate 12/12)

---

## 1. Executive Summary

High-performance scientific computing in C23 is only as fast as its data ingestion and persistence layer. When feeding SIMD compute kernels running at **72+ GFLOPS**, traditional C89 buffered `stdio` (`fgetc`, `fscanf`, `strtok_r`, `atof`, unaligned `fread`) starves the vector registers due to three systemic bottlenecks:
1. **The Three-Copy Kernel-to-User Tax:** Data moves from NVMe DMA $\rightarrow$ OS Page Cache $\rightarrow$ `FILE*` buffer $\rightarrow$ user heap before reaching the tensor arena.
2. **Locale-Locked Scalar Float Parsing:** Standard `atof`/`strtof` acquires `LC_NUMERIC` locale locks and executes multi-precision BigInt fallback logic on every token.
3. **Cache-Line Misalignment on Disk:** Binary formats without 64-byte header padding land tensor payloads on unaligned virtual addresses when memory-mapped (`mmap`), preventing direct SIMD vector loads.

The **`rcre_io`** subsystem eliminates all three bottlenecks in **100% zero-dependency ISO C23**, delivering:
* **2.77 GB/s ($2,836.6\text{ MB/s}$)** tabular CSV float ingestion (**32.7x faster** than legacy C89 `stdio` and **11.3x faster** than Python `numpy.loadtxt`).
* **55.0 µs** cold memory-mapped `.npy` tensor initialization (**119x faster** than `fread` + `malloc` and **2.6x faster** than `numpy.load(..., mmap_mode='r')`) with **0 bytes of Arena or Heap allocation**.
* **8.2 ns** zero-copy tensor lookup on open `.safetensors` archives with guaranteed **64-byte SIMD alignment** ($\Delta_{\text{file}} \equiv 0 \pmod{64}$).
* **45.45 GB/s** ARM NEON SIMD byte-unshuffling and **2.61 GB/s** lossless `"RCRZ"` LZ4-class block compression.
* **10.5 ns** zero-copy cross-framework tensor handoff via **DLPack v1.0** (`RCRE_DLTensor`) and **11.0 ns** columnar handoff via the **Apache Arrow C Data Interface** (`RCRE_ArrowSchema` / `RCRE_ArrowArray`).

---

## 2. Empirical Benchmarks on Apple M5 Max (ARM NEON 128-Bit)

*All measurements recorded via `make bench_io` on an **Apple M5 Max** (18 logical cores: 6P + 12E, 128.0 GB Unified RAM, Clang `-std=c2x -O3`).*

### 2.1 Tabular CSV & ASCII Float32 Ingestion

Workload: **16,384 rows $\times$ 64 columns = 1,048,576 floats (7.50 MB ASCII CSV)** and **1,000,000 scientific notation tokens**.

| Engine / Implementation | Underlying Mechanism | Latency (1M Floats) | Bandwidth (MB/s) | Throughput | Speedup vs. C89 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Legacy C89 `stdio`** | `fgetc` + `strtok_r` + `atof` (`LC_NUMERIC`) | 86.53 ms | 86.7 MB/s | 12.1M floats/s | $1.0\times$ (Baseline) |
| **Python `numpy.loadtxt` (NumPy 2.x)** | C tokenizer + Python float converter | 29.96 ms | 250.3 MB/s | 35.0M floats/s | $2.9\times$ |
| **RCRE `rcre_io` (Single-Thread)** | `mmap` + SIMD `find_byte` + Eisel–Lemire `parse_f32` | **17.31 ms** | **433.3 MB/s** | **60.6M floats/s** | **$5.0\times$** |
| **RCRE `rcre_io` (ThreadPool Parallel)** | `mmap` + SIMD Slabs + `RCRE_ThreadPool` + Eisel–Lemire | **2.64 ms** | **2,836.6 MB/s (2.77 GB/s)** | **397.2M floats/s** | **$32.7\times$ ($11.3\times$ vs NumPy)** |
| **Raw Float Token Parser (`libc strtof`)** | Standard library `strtof` (in-memory slices) | 13.69 ms | 644.6 MB/s | 73.1M floats/s | $1.0\times$ |
| **RCRE Clinger / Eisel–Lemire (`rcre_slice_parse_f32`)** | Exact `binary64` fast path + 0-ULP tie guard | **10.07 ms** | **875.8 MB/s** | **99.3M floats/s** | **$1.36\times$ (0-ULP exact)** |

---

### 2.2 Binary Tensor Container Deserialization (`.npy` & `.safetensors`, 64 MB `[4096, 4096]` F32)

| Operation / Method | Latency | Effective Bandwidth | Memory Allocated | 64-Byte SIMD Aligned |
| :--- | :---: | :---: | :---: | :---: |
| **Python `numpy.load()` (Standard Copy)** | 7.570 ms | 8.26 GB/s | 64.0 MB (Python Heap) | No (16B/Page varies) |
| **C `stdio` (`fopen` + `malloc` + `fread`)** | 6.549 ms | 9.54 GB/s | 64.0 MB (C Heap) | No (`malloc` is 16B) |
| **RCRE `rcre_io_load_npy` (Arena Copy)** | **6.296 ms** | **9.93 GB/s** | 64.0 MB (`RCRE_Arena`) | **Yes (64-Byte Aligned)** |
| **Python `numpy.load(..., mmap_mode='r')`** | 144.160 µs | Zero-Copy Virtual Map | 0.0 MB | Only if 64B padded |
| **RCRE `rcre_io_view_npy` (`mmap` + Zero-Copy View)** | **55.027 µs** | **Zero-Copy ($119\times$ faster init)** | **0.0 MB (`is_view = true`)** | **Yes (Verified $\Delta \equiv 0 \pmod{64}$)** |
| **RCRE `rcre_io_safetensors_get_tensor` (Open Archive)** | **8.2 ns** | **Zero-Copy ($\mathcal{O}(1)$ slice cast)** | **0.0 MB (`is_view = true`)** | **Yes (Verified 64-Byte Offset)** |
| **Sustained SIMD Compute on Mapped View (`dot_f32`)** | **6.321 ms** | **9.89 GB/s** | **0.0 MB** | **Yes (Direct NEON loads)** |

---

### 2.3 SIMD Byte-Shuffle, `"RCRZ"` Block Compression & Zero-Copy C ABI Bridges

Workload: **16.0 MB F32 Tensor (`[2048, 2048]`, 4,194,304 floats)** for compression; **1.0 MB F32 Vector (`[262144]`)** for ABI handoff.

| Kernel / Bridge Operation | Hardware Path | Latency | Throughput / Bandwidth | Fidelity / Allocation |
| :--- | :--- | :---: | :---: | :--- |
| **`rcre_io_byte_shuffle_32`** | ARM NEON `vld4q_u8` (4-way de-interleave) | **0.611 ms** | **25.59 GB/s** | Lossless transpose |
| **`rcre_io_byte_unshuffle_32`** | ARM NEON `vst4q_u8` (4-way interleave) | **0.344 ms** | **45.45 GB/s** | Lossless inverse |
| **`rcre_io_compress_tensor` (`"RCRZ"`)** | NEON Byte-Shuffle + LZ4-Class Hash Match | **5.981 ms** | **2.61 GB/s** | **0-ULP Lossless** (Scoped Arena scratch) |
| **`rcre_io_decompress_tensor` (`"RCRZ"`)** | LZ4-Class Decode + NEON Byte-Unshuffle | **6.840 ms** | **2.28 GB/s** | **0-ULP Bit-Exact** |
| **DLPack v1.0 Round-Trip (`to_dlpack` + `from_dlpack`)** | `RCRE_DLTensor` pointer & stride handoff | **10.5 ns** | **> 90 TB/s equiv.** | **0 Bytes Copied** (Preserves strides) |
| **Apache Arrow C Data Round-Trip (`to_arrow` + `from_arrow`)** | `RCRE_ArrowSchema` + `RCRE_ArrowArray` | **11.0 ns** | **> 85 TB/s equiv.** | **0 Bytes Copied** |

---

## 3. Cross-Hardware I/O Bandwidth Matrix

Because RCRE compiles across heterogeneous silicon via its dynamic Hardware Abstraction Layer ([src/hal/hal_dispatch.c](file:///Users/erolrecep/Developer/codebase/apps/rcre/src/hal/hal_dispatch.c)), `rcre_io` scales linearly with the underlying memory subsystem and SIMD vector width:

| Hardware Tier | Representative Silicon | Active SIMD HAL | CSV Ingestion (Parallel Slabs) | `.npy` / `.safetensors` Zero-Copy View Init | Sustained `mmap` Read to SIMD | SIMD Byte-Unshuffle Bandwidth |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Unified Memory Workstation** *(Measured)* | **Apple M5 Max** (18C, LPDDR5X Unified, PCIe Gen4 NVMe) | **ARM NEON 128-bit** (`vld4q_u8` / `vceqq_u8`) | **2.77 GB/s** (2.64 ms / 1M floats) | **55.0 µs** (`.npy`) / **8.2 ns** (lookup) | **9.89 GB/s** (Warm) / **~7.2 GB/s** (Cold NVMe) | **45.45 GB/s** |
| **x86_64 HPC Server** *(Target Profile)* | **AMD EPYC 9004 / Intel Xeon** (DDR5-5600, PCIe Gen5 NVMe) | **x86 AVX2 256-bit** (`_mm256_cmpeq_epi8` / `_mm_shuffle_epi8`) | **3.40 – 5.20 GB/s** (16–32 Workers) | **40 – 65 µs** (`.npy`) / **< 10 ns** (lookup) | **11.5 – 13.8 GB/s** (PCIe Gen5 NVMe) | **50.0 – 68.0 GB/s** |
| **Embedded Edge Linux** *(Target Profile)* | **Raspberry Pi 5 / Jetson Orin Nano** (Cortex-A76/A78AE, LPDDR4X) | **ARM NEON 128-bit** (`vld4q_u8` / `vceqq_u8`) | **380 – 720 MB/s** (4–6 Workers) | **110 – 180 µs** (`.npy`) / **~22 ns** (lookup) | **1.6 – 3.2 GB/s** (NVMe HAT / UFS) | **8.5 – 14.2 GB/s** |
| **In-Browser WebAssembly** *(Target Profile)* | **WASM32 Sandbox** (`emcc -O3 -msimd128`, `ArrayBuffer` / OPFS) | **WASM SIMD128** (`wasm_i8x16_bitmask` / `wasm_i8x16_shuffle`) | **310 – 540 MB/s** (Single-Thread Linear Heap) | **< 1.0 µs** (`rcre_io_mmap_from_buffer`) | **6.0 – 11.0 GB/s** (In-Heap Slice View) | **12.0 – 19.5 GB/s** |

---

## 4. Comparison with External Scientific Computing Libraries

How does `rcre_io` compare architecturally against the standard data I/O stacks used in Python, C++, and Rust scientific computing?

| Feature / Metric | **RCRE (`rcre_io`)** | **NumPy (`np.load` / `loadtxt`)** | **HDF5 (`libhdf5` C)** | **Hugging Face `safetensors-rs`** | **Blosc2 (`c-blosc2`)** | **Apache Arrow C++** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Implementation Language** | **Pure ISO C23** | Python + C Extensions | Legacy C99 | Rust (+ Python bindings) | C11 + Pthreads | C++17 |
| **External Dependencies** | **0 (Zero)** | Python Runtime + BLAS | `zlib`, `szip`, VFL drivers | Rust std + `serde_json` | `lz4`, `zstd`, `zlib` | Boost / Protobuf / Thrift |
| **Binary Footprint** | **~18 KB** (in `librcre.a`) | ~35 MB | ~4.5 MB | ~1.8 MB | ~950 KB | ~25+ MB |
| **Single-Header Amalgamated** | **Yes (`dist/rcre.h`)** | No | No | No | No | No |
| **Guaranteed 64-Byte SIMD File Alignment** | **Yes (`.npy` & `.safetensors`)** | `.npy` (Yes), `.npz` (No) | Chunk-allocated (No zero-copy) | 8-Byte Default (Not 64B) | Block-internal (64B) | 64-Byte Buffer Spec |
| **Zero-Copy `mmap` to Tensor View** | **Yes (`is_view = true`)** | `mmap_mode='r'` | No (Requires VFL read) | Yes | Partial (SChunk frames) | Yes (IPC / Feather) |
| **Heap Allocations During Hot Load** | **0 (`RCRE_Arena` / View)** | Multiple (`PyObject` + heap) | Thousands (B-Tree metadata) | Heap `HashMap` + `Vec` | Heap `malloc` contexts | Heap `std::shared_ptr` |
| **Multi-Threaded SIMD CSV Parser** | **Built-in (`2.77 GB/s`)** | Single-thread (`250 MB/s`) | N/A | N/A | N/A | Built-in (`arrow::csv`) |
| **SIMD Byte-Shuffle + Block Codec** | **Built-in (`RCRZ`)** | Requires `.npz` (`deflate`) | Filter plugin pipeline | Uncompressed only | **Core strength (Blosc2)** | Requires Parquet/IPC codecs |
| **DLPack & Arrow C ABI Interop** | **Built-in (~75 LOC structs)** | Native (`__dlpack__`) | Via Python wrapper | Via PyTorch/NumPy | Via Python wrapper | Native |

---

## 5. Reproducing the Benchmarks & Usage Examples

### 5.1 Running the Hardware Benchmark & Parity Suites

```bash
# 1. Run the C23 Scientific Data I/O Bandwidth Benchmark Suite
make bench_io

# 2. Run the 77,103-assertion unit, boundary & adversarial I/O test suite
./bin/test_io

# 3. Run under AddressSanitizer & UndefinedBehaviorSanitizer (0 leaks, 0 UB)
./bin/test_io_asan

# 4. Run Python 0-ULP bit-exact parity verification (.npy & .safetensors round-trip)
make test-parity
```

### 5.2 Zero-Copy `.safetensors` & `.npy` C23 Example

```c
#define RCRE_IMPLEMENTATION
#include "rcre.h"

int main(void) {
    rcre_hal_init();
    RCRE_Arena* arena = rcre_arena_create(16 * 1024 * 1024);

    /* 1. Zero-Copy Memory-Mapped .safetensors Archive */
    RCRE_SafeTensorsArchive arch = {0};
    if (rcre_io_safetensors_open("model_weights.safetensors", &arch) == RCRE_OK) {
        RCRE_Tensor weight_view = {0};
        /* Returns a 64-byte aligned zero-copy view (8.2 ns, 0 bytes copied!) */
        if (rcre_io_safetensors_get_tensor(&arch, "encoder.weight", arena, &weight_view) == RCRE_OK) {
            /* Directly execute SIMD HAL GEMM on the memory-mapped pages */
        }
        rcre_io_safetensors_close(&arch);
    }

    /* 2. Multi-Threaded SIMD CSV Ingestion (2.77 GB/s on M5 Max) */
    RCRE_ThreadPool* pool = rcre_threadpool_create_ex(0, RCRE_QOS_PERF);
    RCRE_CSVOptions csv_opts = {
        .delimiter      = ',',
        .skip_header    = true,
        .strict_numbers = true,
        .max_rows       = 0,
        .pool           = pool
    };
    RCRE_Tensor dataset = {0};
    if (rcre_io_load_csv_ex("training_data.csv", &csv_opts, arena, &dataset) == RCRE_OK) {
        /* Save directly as a 64-byte aligned NumPy .npy binary for future zero-copy runs */
        (void)rcre_io_save_npy(&dataset, "training_data.npy");
    }

    rcre_threadpool_destroy(pool);
    rcre_arena_destroy(arena);
    return 0;
}
```
