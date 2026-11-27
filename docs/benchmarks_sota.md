# RCRE v2.2.0 Empirical Roofline, String, NLP & HPC Benchmarks

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

---

## 3. Zero-Malloc String Engine Throughput (`make bench_str`)

| Benchmark Kernel | Workload Size | Scalar Baseline | Active HAL (NEON) | Hardware Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **UTF-8 Validation (ASCII)** | `16 MB` contiguous buffer | `4.17 GB/s` | **`44.50 GB/s`** | **`10.68x`** |
| **UTF-8 Validation (Multi-byte)** | `16 MB` (`ü`, `世`, `🚀` pattern) | `3.84 GB/s` | **`13.26 GB/s`** | **`3.45x`** |
| **Substring Search (`find_substr`)** | `16 MB` needle scan | `2.67 GB/s` | **`25.06 GB/s`** | **`9.40x`** |
| **ASCII Lowercase (`ascii_to_lower`)** | `16 MB` ASCII text | `29.37 GB/s` | **`34.18 GB/s`** | **`1.16x`** |
| **StrPool Interning & Lookup** | `4,096` unique interned keys | — | **`68.93M ops/s`** (`14.5 ns`/op) | **Zero mallocs** |

---

## 4. HPC Natural Language Processing Engine (`make bench_nlp`)

| Benchmark Kernel | Workload Configuration | Measured Throughput / Latency | Notes |
| :--- | :--- | :---: | :--- |
| **Zero-Copy Word Tokenization** | `16 MB` corpus (`RCRE_WordIter`) | **`663.94 MB/s`** (`92.76M tokens/s`) | Zero copies, 0 `malloc`s |
| **Aho–Corasick Flat DFA Scan** | `16 MB` corpus, 16 case-insensitive patterns | **`550.78 MB/s`** (`1,172,843` hits) | 1 array load/byte, compile-time folding |
| **Levenshtein Edit Distance** | `61`-byte strings (`Myers64` vs scalar DP) | Scalar `0.35M/s` → **`7.69M ops/s`** | **`22.27x` speedup** (64-bit bit-parallel) |
| **BM25 Inverted Index** | `10,000` docs (flat CSR arena postings) | Build **`11.33 ms`** / Top-10 **`252.73 µs/q`** | Two-pass $\mathcal{O}(\text{tokens})$, 0 `malloc`s |
| **TF-IDF → `RCRE_CSRMatrix`** | `10,000` docs (`nnz = 109,912`) | **`17.80 ms`** (**`561.7k docs/s`**) | 64B-aligned CSR, `sklearn`-matching IDF |
| **Sparse HALS NMF** | `10,000` docs, $K = 8$, `20` HALS iters | **`22.94 ms`** | PCG64 deterministic, 0 `malloc`s |
| **Collapsed Gibbs LDA** | `2,000` docs, $K = 8$, `20` sweeps | **`11.69 ms`** (`logPPL = 3.46`) | PCG64 deterministic, flat arena counts |
| **Sparse Multinomial Naive Bayes** | `10,000` docs, `4` classes (CSR direct) | Fit **`0.09 ms`** / Predict **`0.20 ms`** | **`49.5M docs/s`** predict, 0 dense alloc |
| **Sparse Multi-Class Softmax SGD** | `10,000` docs, `4` classes, `10` epochs | **`6.76 ms`** (**`14.78M docs/s/ep`**) | PCG64 shuffled sparse SGD |
| **LexRank + MMR Summarization** | `64` sentences → top-`3` summary | **`0.13 ms / doc`** | Zero-copy sentence slices, doc order |
| **SimHash64 Fingerprinting** | `10,000` docs (`rcre_nlp_simhash64`) | **`3.34 ms`** (**`3.00M docs/s`**) | Hardware `popcnt`, 0 allocations |

---

## 5. HPC Factorizations & Solvers (`make bench_h2`, `make bench_h3`, `make bench_h4`)

| Kernel / Routine | Problem Configuration | Execution Throughput / Metric | Measured Wall Time |
| :--- | :--- | :--- | :--- |
| **General-Size F64 FFT** | $N = 2^{20} = 1,048,576$ (mixed-radix) | **`7.4 Ms/s`** (`735.69 Mflop/s`) | `141.7 ms` |
| **Bluestein Prime F64 FFT** | $N = 1,000,003$ (prime chirp-z) | **`1.1 Ms/s`** (`110.92 Mflop/s`) | `899.8 ms` |
| **Triangular Solve (`trsm_f64`)** | $N = 8192$, `nrhs = 8` | **`1,295.6 Mflop/s`** | — |
| **Dense LU Factorization** | $N = 2048$ (F64, partial pivoting) | **`13.9 GFLOP/s`** | `413 ms` |
| **Dense Cholesky Factorization** | $N = 2048$ (F64, in-place `dpotrf`) | **`4.4 GFLOP/s`** | `656 ms` |
| **Symmetric Eigensystem (`syev`)**| $4096 \times 4096$ (tridiag + implicit QL) | **`865.7 GFLOP/s`** equivalent | **`53 ms`** (values + eigenvectors) |
| **PCG + ILU(0) Poisson Solve** | $512 \times 512$ grid ($262,144$ unknowns) | **`556 iterations`**, rel err $9.7 \times 10^{-11}$ | `1.79 s` solve (`15.5 s` ILU0 fac) |
| **GMRES(32) + ILU(0) Poisson** | $256 \times 256$ grid ($65,536$ unknowns) | **`660 matvecs`** | `1.41 s` solve (`0.91 s` ILU0 fac) |
| **2D-Poisson Multigrid Cycle** | $256 \times 256$ grid ($N=65,536$) | V-cycle: **`1.239 ms`** / FAS: **`1.258 ms`** | $\mathcal{O}(N)$ per-cycle linear scaling |
