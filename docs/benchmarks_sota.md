# RCRE v2.0.0 Empirical Roofline & I/O Benchmarks

**Host Hardware Verified On:** Apple M5 Max (`18` Logical Cores: `6P + 12E`, `128.0 GB` Unified RAM)  
**Active SIMD Dispatch Backend:** `ARM NEON 128-bit` (with x86 `AVX2`/`AVX-512F`, `WASM SIMD128`, and Apple `Metal` backends)  
**Compiler:** Apple `clang` (`-std=c2x -O3`)

---

## 1. Roofline Kernel Benchmarks (`make bench_roofline`)

| Workload | Execution Mode | Latency | Throughput | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **SGEMM (`512x512`, F32)** | Single-Thread (`rcre_hal.gemm_f32`) | `3.88 ms` | `69.13 GFLOP/s` | `1.00x` |
| **SGEMM (`512x512`, F32)** | Multi-Thread (`rcre_hal_gemm_f32_mt`) | `0.85 ms` | `315.32 GFLOP/s` | `4.56x` |
| **GEMV (`4096x4096`, F32)** | Memory-Bound Vector Matrix Product | `2.14 ms` | `31.38 GB/s` (`15.68 GFLOP/s`) | — |
| **IGEMM (`512x512`, I8)** | Quantized `int8 -> int32` MatMul | `6.77 ms` | `39.62 GIOP/s` | — |
| **BGEMM (`512x512`, BF16)** | Brain Float16 -> F32 Accumulate | `6.50 ms` | `41.29 GFLOP/s` | — |

---

## 2. Scientific Data I/O Bandwidth (`make bench_io`)

| Stage / Benchmark | Metric | Measured Result |
| :--- | :--- | :--- |
| **Float32 ASCII Parsing (1M floats)** | Eisel-Lemire `rcre_slice_parse_f32` vs `strtof` | `9.41 ms` (`937.3 MB/s`, `1.30x` over `libc strtof`) |
| **CSV Ingestion (`7.50 MB`, 1M floats)** | Legacy C89 vs RCRE Serial vs RCRE ThreadPool | `2.81 ms` (`2670.9 MB/s` / `2.61 GB/s`, `29.4x` over C89) |
| **Binary Zero-Copy View (`64 MB` `.npy`)** | `rcre_io_view_npy` (`mmap`) vs `fread` | `49.067 µs` init (`107x` faster, `0 MB` heap alloc) |
| **Safetensors Tensor Lookup** | `rcre_io_safetensors_get_tensor` | `9.5 ns` zero-copy lookup |
| **SIMD Byte-Shuffle (`16 MB` F32)** | Shuffle / Unshuffle (`ARM NEON 128-bit`) | `26.64 GB/s` shuffle / `50.81 GB/s` unshuffle |
| **RCRZ Block Codec (`16 MB` F32)** | Lossless 0-ULP compress / decompress | `3.50 GB/s` comp / `2.76 GB/s` decomp (`254.56x` ratio) |
| **Zero-Copy C ABI Interop** | DLPack v1.0 / Arrow C Data round-trip | `12.5 ns` / `12.5 ns` (`0` bytes copied) |
| **Half-Precision SIMD Dequant (16M elems)** | `FP16 -> F32` / `BF16 -> F32` | `71.93 GB/s` / `71.31 GB/s` |
