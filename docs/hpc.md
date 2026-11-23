# RCRE HPC — Scientific Computing Program

Status: **Phase H2 complete** (RCRE 2.2.0). Phases H3–H9 per the game plan.

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
| **H1** | Counter-based RNG (PCG64, numpy-bit-exact) + Sobol'/Halton quasi-MC with checked-in oracle streams | **done (2.2.0)** |
| **H2** | General-size F64 FFT (mixed-radix + Bluestein), BLIS-style blocked GEMM, triangular solve; **GEMM alpha/beta semantics fixed to BLAS standard** | **done (2.2.0)** |
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

## 4. Phase H1 — Reproducible Streams (2.2.0)

### 4.1 What shipped

`src/core/rng.c` + public API in `rcre.h`:

- **PCG64** — PCG-XSL-RR-128/64 (O'Neill), 128-bit LCG, the exact
  algorithm numpy ships (its `numpy/random/src/pcg64`, MIT-relicensed).
  `rcre_rng_pcg_init_parts` takes numpy's `SeedSequence(seed)` 4-word
  parts and reproduces the numpy engine **bit-exactly** (raw stream,
  verified against 4 checked-in seeds × 512 draws). The single-seed
  `rcre_rng_pcg_init` is an RCRE-native splitmix64 derivation — streams
  are locked by fixture, not by cross-library conformance.
- **Skip-ahead** — `rcre_rng_pcg_skip` fast-forwards in
  O(popcount(n)) using a precomputed power-of-two jump table
  (`rng_pcg_table.h`, 64 × (A_b, S_b)). The closed-form jump
  `s' = M^n s + inc (M^n − 1)/(M − 1) mod 2^128` is **not** computable
  inside 128 bits (M − 1 has v2 = 2; the quotient needs 2 extra bits of
  M^n − 1) — the table sidesteps that entirely. Gated by homomorphism
  (`skip(skip(s,a),b) == skip(s,a+b)`) and stream continuity.
- **Substreams** — `rcre_rng_pcg_substream(base, id)` mixes the id into
  the increment (splitmix64), so worker i gets the same stream for ANY
  pool size on ANY machine: the e-means slab determinism contract,
  promoted to the library-wide policy. Chi-square gated over 16
  substreams × 65,536 draws.
- **Sobol'** — new Joe-Kuo direction numbers, d = 1..20, stored as
  64-bit fixed-point words (`rng_sobol_table.h`); point n = XOR of the
  words for the set bits of n, × 2^-64. **Bit-exact** against
  `scipy.stats.qmc.Sobol(scramble=False)` from two checked-in chunks
  (points 0..4095 and 1000000..1000255, all 20 dims — 87,040
  assertions). Shipped depth is 20 direction numbers per dimension;
  points n < 2^20 are conformance-gated, beyond that the sequence
  continues in natural Sobol fashion (documented, not gated).
- **Halton** — radical inverse in bases 2, 3, 5; self-derivable
  reference (bit-exact fixture + spot identities).
- All streams are pure functions of their seed. No wall clock, no
  address dependence, no platform dependence. Stateless `*_at(n, d, out)`
  accessors are the slab-parallel / out-of-core workhorses.

### 4.2 Measured bars (Apple M5 Max, 18 cores, Apple Clang 21, -O3)

| Operation | RCRE (this machine) | Reference (same machine) |
|---|---|---|
| PCG64 raw u64 draws | **330.8 Ms/s** | numpy PCG64 `random_raw`: 300.4 Ms/s → **1.10× numpy** |
| PCG64 f64 stream | 2.70 GB/s (8 B/value) | — |
| PCG64 skip (avg ~27 bits set) | 8.36 Mskips/s | — |
| PCG64 substream derivation | 547.5 M/s | — |
| Sobol' at (d = 20) | 4.08 Mpts/s (652.85 MB/s of points) | — |
| Halton at (k = 3) | 67.7 Mpts/s | — |

Honest scope: the 1.10× bar is same-algorithm C-vs-C (numpy's engine is
C); it validates the port, not superiority over numpy's API. Sobol' at
4.08 Mpts/s is 20-word XOR per point (one word per set bit) — the
quasi-MC path is generation-bound, and that is the number to optimize
when H4/H8 need it.

### 4.3 Conformance fixtures

`tests/rng_fixtures/rng_fixture_data.h` (generated, checked in, never
regenerated by the build):

- PCG64: 4 seeds × 512 raw u64, oracle numpy (SeedSequence parts
  recorded alongside).
- Sobol': two stream chunks × 20 dims, oracle scipy 1.18.1
  (scramble=False).
- Halton: 100 points × 3 bases.

Test suite: `tests/test_rng.c` — 7 tests, 395,456 assertions, all
bit-exact where an oracle exists.

## 5. Phase H2 — General FFT, Blocked GEMM, Triangular Solve (2.2.0)

### 5.1 General-size F64 FFT (`src/core/fft.c`)

`rcre_fft_1d_f64` — any length n ≥ 1:

- n with a small prime factor (2,3,5,7): the generalized even/odd
  (radix-p) Cooley-Tukey split — `X_k = Σ_d W_n^{dk} F_d(k mod q)` where
  F_d is the q-point FFT of residue class d. Trivially correct by
  construction (no in-place bit-reversal, no layout games); recurrence
  T(n) = p·T(n/p) + O(np) = O(n log n).
- n with no small factor (primes, products of medium primes): **
  Bluestein chirp-z** to a power-of-two FFT of length N' ≥ 2n−1.
  The chirp phase is computed with **exact integer modular reduction**
  (j² mod 2n in 128 bits, angle < 2π) — without it, cos/sin
  argument-reduction on raw angles of ~πj²/n loses ~3e-10 rad per
  twiddle at n ≈ 1e6 and the round-trip floor degrades from 1.5e-12
  to 4.9e-7 (measured).

Conformance: `tests/test_fft_f64.c` — 4,632 assertions. Forward vs the
direct O(n²) DFT oracle for ~700 sizes (n = 2 … 1e6), primes/Bluestein
sizes in both directions vs the direct DFT, round-trips
(`fft → ifft = identity`, gate 1e-10; measured floor 1.33e-11 at n ≈ 1e8),
Parseval for large n, and known transforms (delta, cosine spikes, n=1,
n=0 error). Runtime ~23 s (dominated by the DFT oracle, which is only
run for n ≤ 4096).

### 5.2 Blocked GEMM (`rcre_linalg_gemm_blocked_f64`)

One-sided (A-panel) 64×64×64 blocking over the HAL micro-kernel:
loop order (i-block, kk-slab, j-block) keeps the 32 KB A micropanel
resident while B/C tiles stream. Deterministic (fixed loop order);
the K accumulation order differs from the unblocked path, so the two
agree in the float determinism class (a few ulps), not bit-exactly.

**Measured (Apple M5 Max, Apple Clang 21, -O3):** the blocked path is
**1.00×** the unblocked path (0.98–1.00× across 512³…2048³). Honest
reason: the H0 unblocked kernel already carries 64³ internal blocking,
so the H2 wrapper re-blocks an already-blocked kernel. The blocked API
is kept because it is the stable composition point for staged
(transposed-microkernel) BLIS-style designs in later phases and for
future panel-level caching; it is published at parity, not above it.

### 5.3 Triangular solve (`rcre_linalg_trsm_f64` / `_trsv_f64`)

A·X = alpha·X in place, all four (uplo, trans) combinations, unit/general
diagonal, multi-RHS, **row-major X** (ldx = row stride — the RCRE
convention; BLAS trsm is column-major and that difference is documented
in the header). Zero diagonal is detected and reported (−1). Plain
sequential F64: deterministic, fixed order. This is the H3 LU/QR
primitive.

### 5.4 GEMM alpha/beta semantics fix (H0 bug caught by H2)

The H0 F64 gemm cells (scalar/NEON/AVX2/AVX-512) applied `alpha` to the
whole of `beta*C0 + A*B` (a final `C *= alpha` pass) instead of the
BLAS-standard `alpha*A*B + beta*C0`. The H0 test suite could not catch
it: it compared the dispatched path against the scalar reference —
both shared the bug. H2's blocked-GEMM test compares against an
independent per-element reference and caught it. Fixed in all four cells
by scaling the A load by alpha (the F32 driver's convention, which was
already correct). All H0 gemm tests remain green; the semantics are now
locked by `test_gemm_blocked_vs_reference` (13 size cases, alpha = 1.5,
beta = −0.5, K = 0, and the single-K-block unblocked-vs-blocked check).

### 5.5 Measured H2 bars (Apple M5 Max, 18 cores, Apple Clang 21, -O3)

| Operation | Result |
|---|---|
| GEMM 512³ / 1024³ / 2048³ (unblocked) | 21.7 / 20.4 / 16.2 GF/s |
| GEMM blocked vs unblocked | 0.98–1.00× (parity — see §5.2) |
| FFT 2^20 (power of two) | 8.0 Mpts/s (796.8 Mflop/s at ~5n·log2 n) |
| FFT 1e6 (2^6·5^6, mixed-radix) | 7.8 Mpts/s (779.6 Mflop/s) |
| FFT 1,048,573 (2^20+1, Bluestein) | 1.3 Mpts/s (125.6 Mflop/s) |
| FFT 1,000,003 (prime, Bluestein) | 1.2 Mpts/s (119.8 Mflop/s) |
| FFT 831,400 (smooth × 11, Bluestein tail) | 0.8 Mpts/s (82.4 Mflop/s) |
| TRSM n = 512/2048/8192, nrhs = 8 | 1.33 / 1.32 / 1.30 GF/s |

The Bluestein path is 6.5× slower than the smooth path at 1M scale
(3× the power-of-two work plus the chirp passes) — that is the honest
price of arbitrary prime lengths. The next optimization (H3 work, only
if the bars require it) is a cached-twiddle Bluestein; the current
correctness-first design recomputes cos/sin per call.

## 6. Validation gates (per release)

- `make all` — zero warnings (`-Wall -Wextra -Wpedantic`).
- `make test` — full suite incl. `test_linalg` (14 suites, 827,421
  assertions), `test_rng` (7 tests, 395,456 assertions), `test_fft_f64`
  (3 tests, 4,632 assertions) and the amalgamated single-header build.
- `make test-sanitizers` — ASan + UBSan + leak check, zero findings.
- `make bench` — numbers appended to this file / `docs/benchmarks_sota.md`
  with machine + toolchain + commit.
- x86 path: AVX2/AVX-512 cells compile under their pragma-guarded TUs on
  any x86-64 compiler and are only dispatched when CPUID reports the
  feature; validated on the next x86 CI runner (H9).
