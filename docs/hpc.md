# RCRE HPC — Scientific Computing Program

Status: **Phase H0 complete** (RCRE 2.1.0). Phases H1–H9 per the game plan.

This document is the contract for RCRE's high-performance scientific
computing direction: what exists, how it is measured, what it guarantees,
and what comes next. Every number in this file was measured on the
machine that produced it — no marketing fractions.

---

## 1. What H0 ships (2.1.0)

### 1.1 The F64 substrate

Double precision is the scientific substrate. Every dispatched op below
exists for **f32 and f64** and resolves at runtime through the existing
`rcre_hal` probe/dispatch table:

| op        | scalar | NEON (128-bit) | AVX2 (256-bit) | AVX-512 (512-bit) |
|-----------|:------:|:--------------:|:--------------:|:-----------------:|
| `gemm`    |  ✓ (8×8 blocked) | ✓ (4×2 micro-kernel) | ✓ (4×2) | ✓ (8×8) |
| `gemv`    |  ✓     | ✓              | ✓              | ✓                 |
| `dot`     |  ✓     | ✓              | ✓              | ✓                 |
| `axpy`    |  ✓     | ✓              | ✓              | ✓                 |
| `scale`   |  ✓     | ✓              | ✓              | ✓                 |
| `sum`     |  ✓     | ✓              | ✓              | ✓                 |

Scalar cells are the correctness baseline: every vector cell is gated
against them in `tests/test_linalg.c`. A cell a backend does not provide
falls back to scalar, so the API surface is complete on every platform —
algorithms never branch on ISA.

Public entry points (`src/core/linalg.c`):

```c
rcre_linalg_gemm_f32 / rcre_linalg_gemm_f64
rcre_linalg_gemv_f32 / rcre_linalg_gemv_f64
rcre_linalg_dot_f32  / rcre_linalg_dot_f64
rcre_linalg_axpy_f32 / rcre_linalg_axpy_f64
rcre_linalg_scale_f32/ rcre_linalg_scale_f64
rcre_linalg_sum_f32  / rcre_linalg_sum_f64
```

NEON F64 note: aarch64's 128-bit double vector holds **two** doubles
(`float64x2_t`) — the F64 NEON kernels are 2-wide, not 4-wide.

### 1.2 Robust deterministic reductions (`src/core/reduce.c`)

| routine | error model (Higham) | measured |
|---|---|---|
| `rcre_sum_neumaier_f64` | `(2·eps + O(eps²))·|sum|`, independent of n | **0** on `1e20 + 2²⁰·1` (bit-exact), `0` on `1e16, 1, -1e16` recovery |
| `rcre_dot_neumaier_f64` | same, for dot products | gated at the double floor |
| `rcre_sum_pairwise_f64` | `~log2(n)·eps·|sum|` | bit-exact on `10⁶·(1e-6)`; **exactly −1 ulp** (`−16384`) on `1e20 + 2²⁰·1` |
| dispatched `rcre_linalg_sum_*` | vectorized, order-fixed | within a few ulps of the scalar reference |

Measured drift table (n = 2²⁰, layout `x[0]=1e20`, rest `1.0`; exact
`1.0000000000000105e+20`, `ulp(1e20) = 16384`):

```
naive    |err| = 1.049e+06   (~64 ulps — every 1.0 below half-ulp is lost)
neumaier |err| = 0.000e+00   (bit-exact)
pairwise |err| = 1.638e+04   (exactly 1 ulp)
```

These are the yardsticks: H4 solver residual checks and convergence gates
must sit at the Neumaier floor, not the naive drift.

### 1.3 Special functions (`src/core/sf.c`)

| function | method | measured accuracy |
|---|---|---|
| `rcre_sf_erf_f64` / `erfc_f64` | power series (|x| ≤ 1) + incomplete-gamma CF (DLMF 8.9.8, Lentz, a = ½, z = x²); prefactor `x·e^(−x²)/√π` as three correctly-rounded factors | ≤ 3e-15 rel (Lentz product floor at z < 2), ~1e-16 series side; tail-accurate (no `1−erf` cancellation); erfc(±10) exact to ~1e-37 abs |
| `rcre_sf_lgamma_f64` | Lanczos (g = 7, 9 coeffs) + reflection | ≤ 1 ulp (x > 0.5); exact at nonneg integers |
| `rcre_sf_digamma_f64` | asymptotic (6 Bernoulli terms) + recurrence lift to 16 | ~2e-15; gated at `psi(10) = H₉ − γ` (50-digit Decimal) |
| `rcre_sf_logsumexp_f64/f32` | max-shift | no overflow for large |x| |
| `rcre_sf_expit_f64/f32` | branch-free, tail-accurate | `1−expit(20)` = 2.061e-9 to 2.2e-9 gate |
| `rcre_sf_norm_cdf_f64` | `0.5·erfc(∓x/√2)` — the tail-accurate form, **not** `1−Φ` | Φ(−6) within 1e-23 of the 55-digit reference at the actual double input `6·fl(1/√2)`; Φ(−100) = 0 exactly |

The erf/erfc implementation history matters for future maintainers: the
DLMF 7.1.14 CF (aₙ = n/2, bₙ = 1) **does not converge to erfc** for x ≥ 1.5
(measured: stuck at 3.6–5e-2 rel regardless of depth). The working form
is the upper-incomplete-gamma CF (8.9.8) with the Lentz algorithm, where
`gcf(½, x²)` composes with `x·e^(−x²)/√π` to erfc(x) with no extra
constant. And the prefactor must not be computed as
`exp(−z + 0.5·ln z − 0.5·ln π)` — the O(17) argument sum loses ~1 ulp
(1.8e-15 measured); the three-factor product is ~10× tighter.

### 1.4 Measured performance bars (Apple M5 Max, 18 cores)

512×512×512 GEMM, single-threaded, dispatched vs scalar:

```
F32: NEON  54.2 GFLOP/s (2.4× scalar 22.6)
F64: NEON  21.9 GFLOP/s (0.64× scalar 34.1)  ← honest: 2-wide F64 NEON
                                               loses to M5's scalar OOO engine;
                                               the 8-wide AVX-512 path is the
                                               expected F64 win (x86)
```

BLAS-1 at 2²⁰ doubles: dot 37.7 GB/s (vector) vs 28.4 (scalar);
sum 30.3 vs 13.8 GB/s. (F64 memory traffic dominates — these are
bandwidth-bound, not flop-bound.)

These are the published bars for H0. x86/AVX-512 numbers will be added
when validated on an 8-wide machine (see §4, H9).

---

## 2. Standing policies (all phases)

### 2.1 Determinism is a first-class mode

- Dispatched linalg and SF ops are single-threaded → **bitwise
  reproducible** run-to-run, on any machine.
- Parallel reductions follow the **e-means slab contract**: partition
  depends only on the problem size (never on core count); each slab
  reduces in fixed order; partials combine in fixed slab order. Pooled
  runs are bit-identical across pool sizes and machines.
- Serial-vs-parallel comparisons are gated on **quality equivalence**
  (MSE within tolerance, labels ≥ 99% cluster-aligned), never on 1 ulp —
  different accumulation orders are legitimate.
- A fast mode (opt-in, `RCRE_*_fast` variants where they matter) may
  relax combine order for throughput; it is never the default.

### 2.2 Exactness classes

- **Order-preserving** ops (transpose, copy, permute): byte-identical.
- **Integer-quantum** ops: |v − ref| ≤ 1 LSB per sample.
- **Floating-point** ops: relative epsilon `1e-5` (1 ulp-class where the
  reference is a measured stream, as in e-means strict parity with
  injected streams).
- **HPC reductions**: Neumaier is the reference floor; vectorized paths
  gate within a few ulps of it.
- Cross-ISA parity is **tolerance-gated, not bit-gated**: FMA and
  micro-kernel shapes legitimately change rounding; the scalar cell is
  the named reference in every gate.

### 2.3 Zero-dependency core, optional backends

The core remains zero-dependency C23 (SoA tensors, arena allocation,
single-header amalgamation). Anything that needs an external device or
fabric (GPU, MPI) enters as an **optional backend TU behind a C ABI**,
exactly the existing `rcre_io_from_arrow` / DLPack / `extras/rcre_gpu.h`
bridge pattern. The amalgamation stays a buildable single header on
every platform (verified each release: `tests/test_amalgamated`).

### 2.4 Precision ladder

FP64 is the scientific substrate; FP32 the workhorse; BF16/FP16/i8 the
accelerator paths (the i8/BF16 GEMM cells landed in v2.0 for the DS
engine and share the same dispatch table). New numeric families ship
with checked-in reference streams / conformance fixtures where practical.

### 2.5 Benchmark honesty

Bars are **fractions of reference libraries** (OpenBLAS, LAPACK, Ginkgo,
Hypre, pocketfft, …) measured on the same machine in the same build
log. "0.87× OpenBLAS" is an acceptable, publishable result; "100× faster
than everything" is not a sentence this repository writes.

---

## 3. Phase roadmap (abridged — full plan: `rcre_hpc_game_plan.md`)

| phase | content | status |
|---|---|---|
| **H0** | F64 HAL substrate, dispatched linalg, Neumaier/pairwise reductions, SF library, determinism policy, benchmark infra | **done (2.1.0)** |
| H1 | Counter-based RNG (Philox) + quasi-MC (Sobol/FAure) with checked-in streams | next |
| H2 | BLIS-style blocked GEMM (f64 first), general-size FFT (pocketfft-class), triangular solve | |
| H3 | Direct factorizations: Cholesky/LU/QR/SVD + eigen for symmetric | |
| H4 | Krylov solvers (CG/GMRES/BiCGStab) + preconditioners + multigrid — the HPC credibility core; residual gates on the Neumaier floor | |
| H5 | Out-of-core chunked ND arrays (Zarr v3) on the io v1.3 substrate | |
| H6 | GPU backend (Metal → CUDA/HIP via the bridge pattern), CPU references first | |
| H7 | MPI/distributed layer behind the C ABI | |
| H8 | Statistics layer: MCMC, streaming sketches, robust stats — built on H0 SF + H1 RNG | |
| H9 | Continuous silicon validation wave (Apple Silicon, x86 AVX-512/AMX, ARM SBC NEON/SVE2) | ongoing |

Sequencing: H0 → H1 → H2 before any solver work (H3/H4 depend on the
F64 GEMM and RNG foundations); H5 is decoupled (io substrate already
exists); H6 only after CPU references are the named conformance targets.

---

## 4. Validation gates (per release)

- `make all` — zero warnings (`-Wall -Wextra -Wpedantic`).
- `make test` — full suite incl. `test_linalg` (12 suites, 11,665
  assertions) and the amalgamated single-header build.
- `make test-sanitizers` — ASan + UBSan + leak check, zero findings.
- `make bench` — numbers appended to this file / `docs/benchmarks_sota.md`
  with machine + toolchain + commit.
- x86 path: AVX2/AVX-512 cells compile under their pragma-guarded TUs on
  any x86-64 compiler and are only dispatched when CPUID reports the
  feature; validated on the next x86 CI runner (H9).
