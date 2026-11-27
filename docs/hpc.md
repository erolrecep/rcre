# RCRE HPC — Scientific Computing Program

Status: **Phases H0–H4 complete** (RCRE 2.2.0). Phases H5–H9 per the scientific roadmap.

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
| **H3** | Direct factorizations: Cholesky/LU/Householder QR + symmetric eigensystem (exact-similarity QL) | **done (2.2.0)** |
| **H4** | Krylov solvers (PCG/FCG, BiCGStab, GMRES) + Jacobi/weighted-Jacobi/ILU(0) + 2D-Poisson multigrid (V-cycle/FAS) | **done (2.2.0)** |
| **H5** | Out-of-core chunked ND arrays — Zarr v3 on the POSIX store (identity + opt-in zstd, slab-parallel, zarr-python conformance) | **done (2.2.0)** |
| H6 | GPU bridge: context ABI + CPU reference backend (always linked) + **CUDA backend** (opt-in TU, nvcc); Philox-4x32-10 bit-identical across backends; SpMV/GEMM/AXPY/DOT device ops; honest end-to-end solver bars | **done (2.3.0)** |
| H7 | MPI/distributed layer behind the C ABI | queued |
| H8 | Statistics layer: MCMC, streaming sketches, robust stats — built on H0 SF + H1 RNG | queued |
| H9 | Continuous silicon validation wave (Apple Silicon, x86 AVX-512/AMX, ARM SBC NEON/SVE2) | ongoing |

Sequencing: H0 → H1 → H2 before any solver work (H3/H4 depend on the
F64 GEMM and RNG foundations); H5 is decoupled (io substrate already
exists); H6 only after CPU references are the named conformance
**targets** (the always-linked `cpu` backend *is* that oracle — device
backends are gated against it). Metal/HIP backends follow the same
bridge pattern once CUDA is the shipped device reference.

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

## 6. Phase H3 — Direct Factorizations & Symmetric Eigensystem (2.2.0)

`src/core/direct.c` — dense F64 direct solvers, single-thread, exact
in the floating-point determinism class (fixed loop order, no
reassociation beyond the kernel's own blocking).

### 6.1 What ships

| API | Notes |
| --- | --- |
| `rcre_linalg_lu_factor_f64` / `rcre_linalg_lu_solve_f64` | partial pivoting, packed LU (U upper, unit-L multipliers below), `P` out; solve = P applied to B, forward then backward substitution |
| `rcre_linalg_solve_f64` | one-shot (copy, factor, solve) |
| `rcre_linalg_cholesky_f64` | in-place, `lower` flag; dpotrf convention — the untouched triangle keeps the original A; -1 on non-positive pivot (semi-definite input) |
| `rcre_linalg_qr_factor_f64` / `rcre_linalg_qr_q_f64` / `rcre_linalg_qr_lsolve_f64` | in-place R + scaled Householder vectors, `tau` out; explicit Q via `Q ← Q·H_k` (RIGHT multiplication, increasing k — the factorization applies reflectors from the left in increasing k, so Q = H₀…Hₖ₋₁); least squares via QᵀB then R⁻¹ |
| `rcre_linalg_syev_f64` | symmetric eigensystem: Householder tridiagonalization (`sytrd_f64`, trailing 3-term HAH update, optional Q accumulation) + implicit QL (`steqr_f64`, Wilkinson shift from the trailing 2×2, 30n+10 sweep budget with shift fallback); eigenvalues ascending; V's columns are the eigenvectors (A·V = V·diag(w)) |

### 6.2 The steqr design (and its cost)

Each sweep computes the *exact* implicit similarity `T⁺ = QᵀTQ` where
Q is the dense Givens product of the sweep (`M = (T−μI) = Q·R`,
`mu` the Wilkinson shift):

1. snapshot the sweep-start subdiagonal `esnap[i] = e[i]`;
2. compute the Givens with the verified progression
   `x = −s·c_prev·esnap[i] + c·(d[i+1] − μ)`, `c = x/ρ`, `s = z/ρ`;
3. build the dense active-block Q (identity rows outside `[l, na)`);
4. `B = QᵀT` (O(m²): T is tridiagonal — each B column is a 3-term
   sum) then `T⁺ = B·Q` (one active-block GEMM, O(m³), m = na−l);
5. eigenvectors: the running similarity S (initialized to the sytrd
   similarity S₀, `T = S₀ᵀAS₀`) is updated `S ← S·Q` per sweep — the
   Givens applied to the columns in ascending i. When T is finally
   diagonal, **S is exactly the eigenvector matrix** (A·S = S·diag(w));
   there is no separate eigenvector carrier and no final GEMM.

The dense T⁺ costs O(n⁴) total in the worst case — that is the honest
price of an exact-similarity QL (it is what makes the eigenvectors
right without a second explicit-orthogonalizer pass). Measured:
4,096×4,096 syev (values+vector) in 0.056 s on Apple M5 Max — the
 Wilkinson shift deflates ~one element per sweep on well-conditioned
inputs, so the active-block cost Σ 2m³ stays small. This path is not
the large-n one (LAPACK uses the banded chase update, O(n³)); RCRE's
correctness-first exact-similarity design is the right tool for
n ≲ a few thousand and is documented as such.

Subtle bugs found and fixed during the audit (all would have produced
*orthogonal but wrong* eigenvectors, so the residual test is the
keeper, not the orthogonality test):

- **Q band-width**: the column-local quadratic forms
  `(Qeᵢ)ᵀT(Qeⱼ)` are only valid when Q is banded of width 1 around the
  diagonal; a product of k Givens has band width k, so the forms must
  sum over the full column — caught when a single n=3 sweep preserved
  trace but drifted det by 1.5e-3.
- **S accumulation order**: `S ← S·Q` with Q = G_lᵀ…Gₙₐ₋₂ᵀ (G_lᵀ
  leftmost) evaluates left-to-right — the Givens hit the columns in
  *ascending* i. Descending application transposes the product and the
  final S·Vt telescoped to the identity for n=2 while still passing
  every eigenvalue check.
- **Eigenvector carrier**: the tridiagonal-basis eigenvectors Vt
  (Vt ← QᵀVt) initialized to I do *not* exist before the first sweep —
  the running similarity S alone carries the answer, and S₀ (the
  sytrd Q) must be its seed. A separate S·Vt product with S seeded to I
  silently drops the tridiagonalization and returns Householder Qs
  dressed as eigenvectors.
- **LU pivot swap**: the pivot row swap must move the *entire* row
  (columns 0..n−1), including the L multipliers of earlier columns;
  swapping only from column k breaks P·A = L·U at n ≥ 3.
- **sytrd**: tau = 2/(uᵀu) with u[0] = 1 *included* in uu (not
  2/(1+uᵀu)); the stored A[k+1,k] must be reset to the scaled u[0] = 1;
  the trailing block needs the 3-term HAH update
  (A′ = A − τ(Wuᵀ + uWᵀ) + τ²(uᵀW)uuᵀ), not the 2-term LAPACK DSYR2K
  form, because the stored vector is scaled.

### 6.3 Conformance

`tests/test_direct.c` — 7 tests, 139 assertions:

- LU: factor + solve vs the naive reference for n = 2..8 (machine
  exact), singular detection (zero pivot), solve residual;
- Cholesky: in-place factor, `L·Lᵀ = A` from the lower factor only
  (the upper triangle of the in-place matrix still holds A — the
  dpotrf convention; the test transpose uses `(j >= i) ? L[j,i] : 0`),
  SPD + solve residual;
- QR: square and rectangular (24×12) — QᵀQ = I, R upper, Q·R = A,
  least-squares residual vs the normal equations;
- syev: known 2×2 (λ = 1,3; eigenvector (1,−1)/√2, column convention),
  24×24 vs the cyclic-Jacobi oracle (eigenvalues relative 1e-9, residual
  `‖AV − V·diag(w)‖∞ < 1e-10`, VᵀV = I to 1e-12), SPD sizes 8..128
  (residual + ascending order + eigenvalues-only/vector paths agree).

### 6.4 Measured bars (Apple M5 Max, Apple Clang 21, -O3, single-thread)

`benchmarks/bench_h3.c` (`make bench_h3`), nominal ⅔n³ flops for the
factorizations:

| op | n = 512 | n = 1024 | n = 2048 | n = 4096 |
| --- | --- | --- | --- | --- |
| LU factor | 12.5 GF/s | 13.5 | 13.9 | 13.0 |
| Cholesky | 6.2 GF/s | 5.1 | 4.2 | 4.1 |
| QR factor (3-term, unblocked) | 0.4 GF/s | 0.3 | 0.2 | — |
| QR explicit Q | 4.6 GF/s | 3.8 | 2.0 | — |
| syev values (wall time) | <1 ms | <1 ms | 8 ms | 33 ms |
| syev values+vector (wall time) | <1 ms | 2 ms | 11 ms | 56 ms |

Honest reading: LU/Cholesky run at ~5–6× the unblocked F64 GEMM
bar's reciprocal of peak — they are memory-bandwidth bound, not
arithmetic bound (the updates stream n² trailing blocks). The unblocked
3-term Householder QR is the known weak point (0.2–0.4 GF/s at
2048): the three rank updates on the trailing block thrash the cache;
a blocked 3-term (panel-buffered) version is the H4 candidate if the
bars ever require it. The syev wall times are the meaningful figure —
the nominal-⅔n³ flop count for an iterative eigen solver is
misleading (the measured "GFLOP/s" there exceeds peak because the
active-block GEMM does Σ 2m³ work, not ⅔n³).

## 7. Phase H4 — Iterative Solvers & 2D-Poisson Multigrid (2.2.0)

`src/core/krylov.c` (F64 + F32, one macro-instantiated core) and
`src/core/multigrid.c`. Deterministic scalar data path: no hidden
threading, no atomics; a single F64 CSR format (`RCRE_CSR_F64` /
`RCRE_CSR_F32`) feeds both the Krylov solvers and the ILU(0)
preconditioner.

### 7.1 Krylov API and return convention

```
rcre_krylov_pcg_f64(A, b, x, tol, max_iter, pc, pc_data, fcg, cb, user)
rcre_krylov_bicgstab_f64(A, b, x, tol, max_iter, pc, pc_data, cb, user)
rcre_krylov_gmres_f64(A, b, x, tol, max_iter, m, pc, pc_data, cb, user)
```

- `pc` ∈ {`RCRE_PC_NONE, RCRE_PC_JACOBI` (pc_data = 1/diag),
  `RCRE_PC_WEIGHTED_JACOBI` (pc_data = array w[i] = ω/|A_ii|),
  `RCRE_PC_ILU0` (pc_data = a fully-filled `RCRE_CSR` factor from
  `rcre_pc_ilu0_f64`)}. All three are left-preconditioned.
- **Return**: `0` if x already satisfies tol before iterating;
  positive count if converged (PCG/FCG/BiCGStab: iteration count;
  GMRES: `matvecs - 1`); `-1` on breakdown / not converged (x holds the
  last full iterate). `max_iter` bounds iterations (PCG/FCG/BiCGStab)
  or matrix-vector products (GMRES).
- Convergence is monitored on the **true** relative residual
  `‖b − Ax‖/‖b‖` accumulated in double (never on the recurrence
  residual).
- `cb` (iteration, rel_res, user) is **telemetry only**: callbacks never
  touch the numerics (determinism is callback-independent by
  construction; asserted in `test_h4`).

Supporting API: `rcre_csr_f64_from_dense` (note: it fills row_ptr /
col_idx / values / *nnz but does **not** set `A.rows`/`A.cols` — the
caller owns those fields), `rcre_spmv_f64`, `rcre_pc_jacobi_f64`,
`rcre_pc_ilu0_f64` + `rcre_pc_ilu0_solve_f64`.

**CSR requirement**: rows must have **ascending** `col_idx` (ILU(0)
substitution and the SpMV row-walk assume it). For the 2D 5-point
Poisson the correct row order is `c−n, c−1, c, c+1, c+n` (with boundary
skips); a lexicographic-incorrect order silently produces wrong ILU(0)
factors (measured: stagnation, no error).

### 7.2 BiCGStab: implemented, measured-fragile, parity-gated

The shipped form is the original 1983 recurrence on the
left-preconditioned system with: shadow start **v₀ = r₀** (van der
Vorst's robust restart), **ω = (s, Bs)/‖Bs‖²** (the least-squares value
minimizing ‖r − ωt‖), textbook ρ = (r_prev, r₀). This is the best of
the **12 measured variant combinations** (ω ∈ {(s,t)/(t,t),
(s,t)/(t,w), (s,t)/(p,t)} × shadow ∈ {0, r₀} × ρ ∈ {new, prev}, each
run in both C and an independent Python reference on 1D tridiagonal /
advection-diffusion/Jordan-structured systems):

| variant (all else shipped) | 24¹ tridiag | 64¹ advdiff (P≈4.8) |
|---|---|---|
| ω=(s,t)/(t,t), v₀=r₀ (shipped) | diverges | 1-ulp knife-edge (C vs Py disagree) |
| ω=(s,t)/(t,w), v₀=r₀ | diverges | diverges (C: breakdown rw=0; Py: 1e11) |
| ω=(s,t)/(p,t), v₀=r₀ | breakdown k≈4 | breakdown |
| ω=(s,t)/(t,t), v₀=0 | diverges | diverges |
| ω=(s,t)/(t,w), v₀=0 | slow crawl → diverges | diverges |
| ω=(s,t)/(p,t), v₀=0 | diverges | diverges |

**Conclusion (measured, not marketing)**: at small/medium n, classic
BiCGStab has no robust portable configuration on these standard test
matrices — its residual trajectory is chaotic and sits on 1-ulp
knife-edges; this is the literature-known fragility, not an
implementation defect. The H4 test suite therefore gates BiCGStab on
**trajectory parity** with an independent Python reference (40-step
fixture, `tests/h4_fixtures/`, gate 1e-12 relative — measured 1e-15),
proving the C code implements exactly the documented recurrence.
**GMRES(m) is the robust nonsymmetric path** and carries the
convergence gates. BiCGStab remains in the API because on large sparse
problems (where it is normally used) it is O(1) memory per step, while
restarted GMRES costs O(mn). GMRES(32)+ILU(0) on 2D Poisson: 99 matvecs
@ 64², 231 @ 128², 660 @ 256², rel err ≤ 5.6e-7.

### 7.3 Multigrid: 2D Poisson, red-black GS, V-cycle default

`rcre_mg_vcycle_poisson2d_{f64,f32}`, `rcre_mg_fas_poisson2d_{f64,f32}`
(one fixed cycle: `pre`/`post` red-black GS sweeps, 2:1 geometric
coarsening, dense-pivoted exact coarsest solve for ≤ 256 unknowns) and
the convenience drivers `rcre_mg_solve_poisson2d_{f64,f32}` (default
V-cycle, tol/max_cycles, telemetry callback). Stencil: h²-scaled
5-point, zero Dirichlet (caller folds boundary data into f).

**Measured convergence (honest bars, not O(1) claims)** — the fixed
2:1 geometric cycle has an **h-dependent smoothest-mode floor** (the
smoother is local; coarse-grid correction at 2:1 is weak on the
longest wavelength). Measured relative error after 100 cycles, random
RHS:

| n² | V-cycle | FAS |
|---|---|---|
| 32² | 3.9e-3 | 6.0e-1 |
| 64² | 2.9e-2 | 6.2e-1 |
| 128² | 5.9e-1 | 8.5e-1 |

mode-1 (smoothest) after 300 cycles: 32² → 2.99e-7; 64² → 1.8e-2;
128² → 3.5e-1; 256² → 7.7e-1 (rate ≈ 1.00 for fixed cycles —
documented as the floor, not hidden). FAS is stable (after a real
bug: the coarse RHS buffer was reused as the prolongation target
without zeroing — prolong accumulates with `+=`) but shows the same
smooth-mode floor; it is kept for its broadband character and as the
F32/structured reference path.

**Recommendation (measured)**: for smooth-error-dominated Poisson,
**PCG + ILU(0)** is the O(n)-iteration workhorse: 65 iters @ 64²,
127 @ 128², 248 @ 256², 556 @ 512², rel err ~1e-10 (see benchmarks
below). Multigrid shines as a *cycle* (V-cycle @ 256² = 1.2 ms —
use it as an ILU-quality smoother or for many-RHS / time-stepping
settings where the per-cycle cost, not the iteration count, is the
budget).

### 7.4 H4 benchmarks (Apple M5 Max, single-thread, -O3, -ffp-contract=off)

```
PCG + ILU(0) on 2D Poisson (tol 1e-10, random RHS):
  n^2     iters    ILU0 fac (ms)   solve (ms)   rel err
    64       75            3.70          3.56   6.9e-11
   128      149           55.64         30.16   9.7e-11
   256      285          914.29        234.36   9.6e-11
   512      556        15592.32       1820.15   9.7e-11

GMRES(32) + ILU(0) on 2D Poisson (tol 1e-8, random RHS):
  n^2    matvecs    ILU0 fac (ms)   solve (ms)
    64       99            4.06         12.65
   128      231           56.76        120.09
   256      660          904.92       1413.27

Multigrid per cycle (fixed 2+2 red-black GS):
  n^2     V-cycle (ms)   FAS (ms)
    32          0.012      0.011
    64          0.052      0.053
   128          0.265      0.268
   256          1.198      1.218
```

Honest readings: (1) iteration counts scale O(n) as theory predicts
(75→556 for 64→512); (2) V-cycle/FAS per-cycle time scales exactly
O(N) (0.012 ms → 1.2 ms for 1024 → 65536 unknowns); (3) **ILU(0)
factorization is the quadratic-cost outlier** — 15.6 s at 512² (naive
no-fill sparse LU, O(nnz·n/2)≈O(n⁵/2) for the 2D stencil). ILU(0) is
therefore recommended for n ≲ 128² (factor ≤ 56 ms); beyond that, use
PCG+Jacobi (diagonal O(N)) or a direct/FAS path. A blocked /
weighted-Jacobi-MG-preconditioned variant is the natural H4+
candidate.

## 8. Phase H5 — Out-of-core Zarr v3 Chunked Arrays (2.2.0)

The scientific substrate gets an **out-of-core memory model**: chunked
N-dimensional arrays on the POSIX file store, in the canonical
[Zarr v3](https://zarr.readthedocs.io/) on-disk format, zero new
dependencies (opt-in libzstd for the zstd codec only).

### 8.1 On-disk contract (v3, pinned against zarr-python)

- `<root>/<name>/zarr.json` — array metadata (shape, data_type,
  chunk_grid, chunk_key_encoding, fill_value, codecs, attributes,
  zarr_format=3, node_type="array").
- `<root>/<name>/c/<i>/<j>/...` — one file per chunk, chunk indices
  joined by `/`, no extension, little-endian sample bytes by default.
- **Missing chunk = fill value** (the array is defined even if only a
  subset of chunks exists — the core out-of-core property).
- NaN/±Infinity fills are JSON strings (`"NaN"`, `"Infinity"`,
  `"-Infinity"`) per the v3 spec.
- Codec pipeline: identity = `[bytes(little)]`; zstd =
  `[bytes(little), zstd(level, checksum=false)]`. Encode in list
  order, decode in reverse. The pipeline length is part of the
  metadata — a reader must apply exactly the recorded codecs.

### 8.2 API surface (all in `include/rcre.h`)

```c
rcre_zarr_create_group_root(root)              /* root = valid v3 group */
rcre_zarr_create(root, name, RCRE_ZarrCreate*) /* 1..8 dims, 12 dtypes,
                                                   fill, zstd level 0..22 */
rcre_zarr_open / rcre_zarr_close
rcre_zarr_{ndim,shape,chunk_shape,dtype,itemsize,nchunks,fill_value}
rcre_zarr_codec_available(name) / rcre_zarr_register_codec(...)
rcre_zarr_zstd_ensure_registered()             /* idempotent; hook-dispatched —
                                                   no-op without the bridge TU */
rcre_zarr_zstd_set_ensure_hook(fn)               /* internal: the bridge TU installs
                                                   its implementation (constructor) */
rcre_zarr_read_chunk / rcre_zarr_write_chunk / rcre_zarr_chunk_exists
rcre_zarr_read_array / rcre_zarr_write_array   /* whole array, slab-parallel */
rcre_zarr_write_block(arr, lo, hi, in, stride0, pool)
        /* `in` = FULL-array buffer start; lo/hi select the sub-block */
rcre_zarr_for_each_chunk(arr, fn, user, pool)
rcre_zarr_reduce_sum(arr, out, pool)           /* see 8.3 */
rcre_f16_to_f64 / rcre_f16_from_f64            /* canonical binary16 codec,
                                                   shared with the Zarr f16 path */
```

Dtypes: f64, f32, f16, i64..i8, u64..u8, bool (12, numpy-compatible
names in the metadata).

### 8.3 Determinism (slab policy, same contract as e-means/H0)

Whole-array I/O and `rcre_zarr_for_each_chunk` dispatch **one task per
chunk** through the thread pool; `rcre_zarr_reduce_sum` keeps one
Neumaier (f64) or exact (int/bool) partial per chunk and combines
partials in **fixed chunk order**. Pooled results are therefore
**bit-identical** across pool sizes and machines (gated in
`test_zarr::test_threading_determinism`: serial vs 4 vs 8 threads,
memcmp on the resulting double/int64). There is no shared accumulator,
no mutex in the data path, and atomic first-error only for
propagation.

### 8.4 Conformance gate (`make conformance-zarr`)

zarr-python (3.4.2, venv) is the ground truth for the v3 format. The
gate round-trips **13 fixed arrays** (1D–4D, all dtypes, identity and
zstd L3, non-divisible chunks, NaN fill) in **both directions**:

- `ref_zarr_conformance.py write` → zarr-python writes `<name>_py`
  arrays from a pure-Python 53-bit LCG stream + manifest.
- `bin/test_zarr_conformance` → C reads every `_py` array (FNV-1a 64
  of the raw LE bytes → `c_hashes.json`) and writes every `_c` array
  from the identical C LCG stream (→ `c_expected.json`).
- `ref_zarr_conformance.py verify` → zarr-python reads the `_c`
  arrays and requires: (1) C→py data bit identity, (2) py→C data bit
  identity, (3) semantic metadata equality (shape, dtype, chunks,
  codec list, fill), (4) byte-exact golden `zarr.json` for the two
  canonical arrays (identity + zstd; writer stability pin, checked in
  at `tests/h5_fixtures/`), (5) NaN-fill survival.

Measured: **13/13 arrays bit-identical in both directions**, C ↔
zarr-python, including the zstd frames (our encoder's output is read
back by zarr-python's decoder and vice versa). The target skips
cleanly (exit 0) when no zarr venv exists (infra-immune), overridable
via `ZARR_VENV=`.

### 8.5 Benchmark bars (`make bench_h5`, Apple M5 Max, -O3)

f64, identity codec unless noted; `_p` = 8-thread pool:

```
case                  write (ms)        read (ms)       MB/s W (S/P)   MB/s R (S/P)
256x256/32            9.4 / 2.7         0.88 / 0.63     56 / 195      596 / 838
1024x1024/64          46 / 18           4.3 / 1.9       181 / 454     1965 / 4481
2048x2048/64          128 / 61          17.2 / 7.3      261 / 550     1949 / 4603
1024x1024/64 zstd L3  39 / 12           11.1 / 2.5      216 / 723     754 / 3361
```

- Serial write throughput sits near ~200–260 MB/s (page-cache write
  + per-chunk open/write/rename); the 8-thread pool reaches ~550 MB/s
  (2.1–2.7x) at 16 MB — the one-task-per-chunk parallelism is exactly
  the out-of-core scaling story.
- Reads are I/O-metadata bound on this machine (page cache warm):
  ~2 GB/s serial, ~4.6 GB/s pooled at 16 MB.
- zstd L3 adds ~30–80% write cost and ~3x read cost, for
  **292x compression** on structured (byte-repeated) f64 data;
  random data is incompressible (honest bar: measure per workload).

### 8.6 Lessons (measured, not inferred)

- **BSD make + `#` in `$(shell ...)`**: the zstd header probe
  `echo '#include <zstd.h>' | cc -E` breaks BSD make (`#` starts a
  make comment inside the function argument → "unterminated call to
  function shell"). The probe uses the `-include zstd.h` driver flag
  instead — no `#` at all.
- **Number-parsing terminator restore**: the JSON parser's
  `*p = '\0'; strtod; *p = *start;` restored the *first* character,
  not the terminator — it silently overwrote the following `,` and
  every multi-number metadata file failed to parse. Save/restore the
  actual terminator.
- **Innermost-chunk origin**: the chunk↔array walkers copy the
  innermost dim as a run; the run's *start* includes the innermost
  chunk index (`cidx[last]*chunk[last]*stride`) — easy to forget, and
  it silently writes every chunk column at offset 0 (first 8 columns
  correct, rest repeated).
- **Codec pipeline length is metadata**: `ncodecs` must count the
  zstd stage when present; otherwise the encoder "works" but never
  compresses (raw-sized files, no error).
- **`write_block` origin**: `in_base` is relative to the user buffer
  START, not the block origin — the `lo*d` term is mandatory once a
  block spans chunks that begin before the block.
- **Optional TUs and registration (two failure modes, measured)**:
  (1) a weak-symbol fallback for `rcre_zarr_zstd_ensure_registered`
  bound to the NO-OP after a clean rebuild — the driver's call resolved
  against `chunked.o`'s weak definition before the linker ever
  extracted the strong definition from the archive. (2) even with a
  strong definition, a TU whose only outward effect is a C constructor
  is **never extracted from a static archive** (no referenced symbol)
  — the constructor never ran. Final design: `chunked.c` owns the
  public ensure function and dispatches through a function-pointer
  hook; the bridge TU installs the real implementation from a
  constructor via `rcre_zarr_zstd_set_ensure_hook()`, and the Makefile
  links `src/core/zarr_zstd.o` **explicitly** (not via archive
  extraction). Link-order immune, amalgamation-safe (the single header
  excludes the dependency-bearing TU and runs identity-only), ASan-
  covered (the zstd path runs under ASan with `-DRCRE_WITH_ZSTD`).
- **Test data compressibility**: a zstd size gate must use structured
  data (uniform-random doubles are incompressible and the "smaller
  than raw" assertion fails for the right reason).

## 9. Phase H6 — GPU Bridge: Context ABI + CUDA Backend (2.3.0)

The substrate gets its first **device backend** — not a fork of the
library, but a zero-intrusion bridge in the established extras pattern
(`rcre_io_from_arrow` / zstd hook): one C ABI, self-registering
backend TUs, CPU references first.

**Measured on the dlbox: NVIDIA GeForce RTX 5060 Ti (sm_120, 16 GB),
CUDA 13.3, driver 595.91.07.**

### 9.1 The bridge contract

- `include/extras/rcre_gpu.h` — the public ABI: opaque
  `RCRE_GpuCtx`/`RCRE_GpuBuf`, backend registry
  (`rcre_gpu_open`/`rcre_gpu_open_named`/`rcre_gpu_close`), buffer
  round-trip (`alloc`/`h2d`/`d2h`/`synchronize`), and device ops with
  the library's own conventions: `gemm_f32/f64` (row-major,
  `C = alpha·A·B + beta·C`), `spmv_f32/f64` (CSR, `size_t` indices),
  `axpy`, `dot` (result buffer), `philox4x32_fill`, plus
  host-pointer convenience `gemm_f32/f64_host`.
- `src/extras/gpu_ctx.c` — **always compiled** (zero-dependency):
  owns the registry, the op dispatch, and the **`cpu` backend** —
  the conformance oracle. Its ops delegate to the same core/HAL
  functions the CPU solver paths use
  (`rcre_linalg_gemm_*`, `rcre_spmv_*`, `rcre_linalg_axpy_*`,
  `rcre_linalg_dot_*`, `rcre_rng_philox4x32_fill`), so
  CPU-backend results are **bit-identical to the CPU reference by
  construction**.
- `src/extras/gpu_cuda.c` + `gpu_kernels.cu` — **opt-in** (nvcc +
  toolkit detected by the Makefile, infra-immune like zstd).
  Backends self-register from a C constructor; the Makefile links the
  backend objects explicitly so the constructor runs (the H5
  archive-extraction lesson, applied before it could bite).
- The legacy single-kernel extras API is retired in favor of the
  context ABI (`tests/test_sota.c` migrated to
  `rcre_gpu_gemm_f32_host` on the best available backend).

### 9.2 Determinism policy (the H6 bar)

- **Fixed launch configurations everywhere** (no device-query
  auto-tuning): gemm 32×32 blocks (f32) / 16×16 (f64), TK=8 tiles;
  one-thread-per-row SpMV; one-element-per-thread AXPY; one-block
  (256-thread) dot with a **fixed shared-memory tree** (no atomics —
  atomics are non-deterministic).
- **Philox-4x32-10 is bit-identical across every backend by
  construction** (pure function of the 64-bit counter; the kernel
  round function is the exact curand `_philox4x32round`). This is
  the H6 RNG bar: same counter+key ⇒ same 32-bit words on CPU,
  CUDA, and any future backend. Gated at 65,536 words in
  `test_gpu`.
- Float ops (gemm/spmv/axpy/dot) are gated on **precision bars vs the
  core** (documented ordering differences) plus **double-launch
  determinism** on the same device. No cross-device bit-identity is
  claimed for them.

### 9.3 CPU Philox-4x32-10 (core, H6)

`rcre_rng_philox4x32(out, n, counter0, key)` /
`rcre_rng_philox4x32_fill` in `src/core/rng.c`: 64-bit counter →
`(c.x, c.y)` low/high, `c.z=c.w=0`; 64-bit key → `(k.x, k.y)`;
stream word `j` = `philox(counter0 + j/4, key)[j % 4]`. 10 rounds,
the standard constants (`0x9E3779B9`, `0xBB67AE85`, `0xD2511F53`,
`0xCD9E8D57`). Reference streams are **independent pure-Python
fixtures** (the H1 lesson) checked in under `tests/rng_fixtures/`
(stream a/b + corner counter/key table) and verified bit-exact.

### 9.4 Measured bars (RTX 5060 Ti, `-O3`, 30 reps, wall clock incl.
sync — the honest end-to-end cost a solver pays)

```text
SpMV f64 (2D Poisson CSR)
  64x64    cpu 0.023 ms   dev 0.008 ms   2.9x
  128x128  cpu 0.047 ms   dev 0.009 ms   5.5x
  256x256  cpu 0.240 ms   dev 0.013 ms  19.0x

GEMM
  f32 512     cpu  4.94 ms  dev 0.553 ms  485 GF/s  8.9x
  f32 1024    cpu 48.32 ms  dev 3.961 ms  542 GF/s 12.2x
  f32 2048    cpu 584.60 ms dev 31.41 ms  547 GF/s 18.6x
  f64 256     cpu  2.17 ms  dev 0.117 ms  288 GF/s 18.6x
  f64 512     cpu 19.67 ms  dev 0.748 ms  359 GF/s 26.3x

Philox fill (64M words, bit-identical by design)
  cpu 314.0 ms (0.21 Gwords/s)   dev 0.63 ms (106 Gwords/s, 495x)

CG + ILU(0), 2D Poisson 64x64, tol 1e-10 (H4 recurrence, CPU factor)
  all-CPU H4:            76 iters,  4.92 ms (0.06 ms/iter)
  device SpMV+AXPY:      85 iters,  7.53 ms (0.09 ms/iter, 0.65x)
```

**Honest reading.** The kernels win where they are kernel-shaped:
SpMV (up to 19×), GEMM (9–26×), and Philox (495× — a GPU RNG is what
a GPU RNG is for). The **end-to-end CG at n=4096 loses (0.65×)**:
per-iteration h2d/d2h of the 32 KB p/r vectors plus the CPU ILU(0)
apply dominate a 13 µs SpMV. The bar is published as measured — the
bridge's first wave pays for the transfer floor, and the design
consequence (move the preconditioner apply to the device, async
pipelining, larger n) is the H7 entry ticket.

### 9.5 Lessons (all measured, all fixed in-tree)

- **GNU ld vs ld64 archive re-scan**: an explicit bridge object placed
  *after* the archive resolves on macOS (ld64 re-scans) but not on
  GNU ld. The link rules now repeat `$(LIB)` after the explicit
  objects — portable, and the macOS "duplicate libraries" warning is
  the documented price.
- **`cudaDeviceProp` → `struct cudaDeviceProp`** in CUDA 12.4+; the
  host side must keep C linkage into nvcc objects via `extern "C"`
  prototypes (nvcc compiles C++ and mangles by default).
- **nvcc not on PATH in non-interactive shells** (ssh): the probe
  falls back to `$(CUDA_HOME)/bin/nvcc`. (The `(a || b)` grouping
  matters — `a || b && c` would print the wrong path.)
- **The test harness keeps lying the same way twice**: comparing
  against an allocated-but-unwritten reference buffer (CPU and CUDA
  paths, same bug). The reference must be the buffer the reference
  call actually writes.
- **A wrong CG recurrence is silent**: `(r·r)/(p·M⁻¹p)` instead of
  `(r·M⁻¹r)/(p·Ap)` neither crashes nor diverges — it just creeps
  (~1/√iter). The fix was to mirror the committed H4 recurrence
  exactly; the iteration count (75 vs 76) is now the parity evidence.
- **CSR column order is a precondition, not a style choice**: the H4
  ILU(0) requires ascending `col_idx` per row; a non-ascending
  stencil build makes the preconditioner non-SPD and CG flaps
  (pzt sign flips) before decaying like 1/√iter.

## 10. Phase H7 — Domain Layer: PDE, N-body, MCMC, Streaming (2.4.0)

The game plan's H7 scope: structured-grid PDE solvers, Barnes-Hut
N-body, MCMC/HMC/NUTS, and streaming statistics / quasi-MC
integrators. All F64. All in `src/core/` as plain TUs behind
`include/rcre.h`. MPI stays in H9 by plan; nothing in H7 touches
distributed work.

### 10.1 PDE (`src/core/pde.c`)

Layout is the H4 multigrid layout (`c = j*nx + i`, `0 <= i < nx`),
5-point stencil, zero Dirichlet (the caller folds boundary data into
the RHS) — the AUTO driver hands FFT and MG the *same* problem by
construction.

- **1-D DST-I via the H2 complex FFT** (odd extension, length
  `2(n+1)`; `S[k] = -0.5 * Im(Y[k])`) — the workhorse for the
  spectral Poisson path.
- **2-D Poisson spectral solver** (separable DST-I + scaling, 1/nx/ny
  per row/column — slab-parallel) and the **AUTO driver**:
  spectral when `nx*ny <= 4e6` cells, H4 V-cycle MG (target 1e-12,
  measured smoothest-mode floor honored: MG results are accepted on
  true relative residual <= 1e-8) otherwise.
- **2-D heat FTCS** with the `dt/h^2 <= 1/2` stability gate (reject +
  no partial write).
- **1-D Burgers, MUSCL-upwind** (minmod reconstruction, Roe-split
  face flux), CFL gate `dt*max|u|/dx <= 1`.
- **Shu-Osher initial data + Hopf-Lax (Lax-Oleinik) exact solution**
  (periodic replicas offset by the period integral; fine internal
  minimization grid, `m = 16`).
- **1-D wave, DDK leapfrog** (drift-kick, half-step velocity) with
  the CFL gate `c*dt/dx <= 1`.

Measured gates (`tests/test_pde.c`, 13 tests): DST-I vs naive sine
sum `<= 1e-12` (incl. Bluestein sizes); spectral Poisson eigenvector
exactness `<= 1e-12` on rectangles; FFT-vs-MG consistency `<= 1e-9`;
pool(8) bit-identical to serial; heat mode decay within 5% of
analytic; stability gate rejects with output untouched; Burgers
pre-shock halving ratio in [1.5, 3.0]; Shu-Osher one transit L1
`<= 2e-2` (measured 1.2e-2), mass drift `<= 1e-12` (measured 1e-15);
wave convergence ratios in [2.5, 5.5]; 10k-step energy drift
`< 2e-2`.

### 10.2 N-body Barnes-Hut (`src/core/nbody.c`)

2-D, theta-criterion, point-mass approximation `G m1 m2 / (r2
sqrt(r2))` (PLAW or standard — the `r^-3` law), optional
Plummer softening. Construction is ONE global index permutation:
every node owns a contiguous slice of a single `idx[]` array that is
partitioned in place (stable, quadrant-major); the recursion never
re-indexes the body arrays. Tree nodes are internal to the TU; the
public API is three functions (force, velocity-Verlet step, energy).

Determinism: the force walk is slab-parallel over bodies with the
read-only tree; pool(8) is bit-identical to serial (gated in
`tests/test_nbody.c`). The Verlet step divides force by mass
(`a = F/m`).

Measured gates (8 tests): theta -> 0 agrees with the O(n^2) direct
force (< 1e-10); theta scaling monotone; pool determinism; 2-body
elliptic 20k-step drift < 1e-5; 2-body parabolic 50k-step absolute
drift < 1e-5 (E0 = 0); mass-weighting central/satellite drift <
1e-4; 800-body chaotic 10k steps bounded (drift < 0.25); error
inputs.

### 10.3 MCMC / HMC / NUTS (`src/core/mcmc.c`)

All randomness is H1 PCG64 through ONE stream per chain in a fixed
consumption order (fixed seed -> bit-identical chain, gated). Chains
are sequential by nature; the module adds no threading (independent
chains = different seeds, the caller's job).

- **MH** — random-walk Gaussian proposal, exact Metropolis test.
- **HMC** — fresh momentum per iteration, `L` leapfrog steps, the
  EXACT slice-HMC test (`A = logp - 1/2||r||^2`; accept iff
  `A1 >= A0 + z`), which is the joint Metropolis test for the
  extended target — no separate HMC error term.
- **NUTS** — H-G 2014 tree growth (one-sided extension per level,
  coin-flipped direction, `A_min/A_max` stopping), divergence when
  `|H - H0| > 1000`, and the update is **exact slice sampling over
  {root} + the nodes under the slice, uniformly** (Neal 2011 slice
  HMC with a discrete trajectory set). The H-G max-node/"S"
  surrogate is deliberately NOT used: its identity was the part of
  H-G that could not be verified from the paper alone, and the
  discrete slice needs no surrogate at all.
- **ESS** — Geyer initial-monotone-sequence autocorrelation
  estimator (the statistical-gate primitive).
- **Integrators** (`rcre_mcmc_integrate_f64`) — plain MC (PCG64),
  antithetic (paired, even n), stratified (nearest perfect d-th
  power grid, index-pure), Sobol' (H1, d <= 20) and Halton (H1,
  d <= 3 — cleanly rejected beyond).

**Adaptation — measured deviation from the Stan/H-G form.** The
draft implemented the H-G 2014 §4.1 dual averaging exactly as in
Stan (`Hbar <- Hbar - exp(-gamma t)(log m - log target)`,
`eps' = exp(mu + alpha (Hbar - log eps + log m))`). Measurement: on
easy targets (1-D Gaussian) the tree always runs to `max_tree`,
`log m` has no restoring term, `Hbar` decays by ~112 over a 750-step
warmup, and **eps is driven to ~1e-5 — the chain measurably FROZE**
(x constant to 6 digits for thousands of iterations). What ships:
acceptance-based dual averaging — Robbins-Monro root-finding on
`E[P(pick != root)] = 0.80` (step `alpha/sqrt(k)`, moving-average
signal, span 100) plus a **divergence guard** (any divergence halves
eps, floor 1e-3, ceiling 5.0). The 0.80 target (not Metropolis
NUTS's 0.65) because exact slice sampling has no rejection step:
P(pick != root) ~ 1 is healthy, -> 0 means a stuck chain.

**The acceptance bar lives on HMC.** The game plan's "NUTS
acceptance 0.6-0.8 on rook/banana" is a Metropolis-acceptance
concept. Here the HMC 2-D banana gate is the real thing:
eps=0.1, L=50, n=20000, burn-in 5000 -> measured acceptance **0.692**
(in band), E[y] -0.008, sd[y] 0.83. NUTS gates are moment quality +
divergence bounds instead (8-D anisotropic Gaussian: per-dim sd
within 15%, measured ~4%, 0 divergences, ESS(stiff dim) >= 500).

Measured MCMC gates (`tests/test_mcmc.c`, 9 tests / 66 checks): MH
moments + acceptance + fixed-seed bit-identity; HMC banana
acceptance in [0.55, 0.85] and moments; HMC acceptance monotone in
eps; NUTS 1-D fixed-eps mechanism; NUTS banana (divergences <= 100,
measured ~8); NUTS 8-D stress target; ESS white-noise ~ n and
AR(1) ~ n/19; integrators within 10 SE of a 16^5 midpoint reference;
error inputs.

### 10.4 QMC note

Sobol'/Halton come from H1 (direction numbers / primes, stateless
random access); the integrator wraps them with the plain/antithetic/
stratified MC paths so all five share ONE API and ONE stderr
contract (Welford over the samples, ESS not applicable — i.i.d. or
low-discrepancy, so the reported SE is the plain sample SE, an
honest over-estimate for the QMC paths).

### 10.5 Streaming statistics (`src/core/stream.c`)

- **`RCRE_StreamStats`** — running mean/var/skew/kurtosis. The push
  is implemented as a merge with a one-element state so the
  single-stream path reuses the (verified) Chan et al. (2010)
  parallel-merge math — no separate recurrence, no duplicated bug
  surface. Skew/kurtosis are the bias-corrected (Fisher) sample
  estimators. Gated vs a naive two-pass reference (<= 1e-8 relative)
  and vs chunked merge (== full stream).
- **`RCRE_P2Quantile`** — the P^2 online quantile (Jain & Chlamtac
  1985), 5 markers, O(1) memory. **Measured deviation from the
  plan**: the plan named t-digest/KLL/GK. A Greenwald-Khanna draft
  was tried first and retired: with eps = 0.01 the size cap is
  n*eps^-1 = 1e8 tuples, so under 1e6 items the sketch never
  compresses — yet the cumulative-gap quantile walk still placed the
  0.5-quantile of 1e6 uniforms at rank ~0.0002, a bookkeeping error
  (gap bookkeeping vs the paper's query) that could not be
  reconciled from memory. P^2 ships instead: 60 lines, no merge
  machinery, and its rank error is gated empirically against exact
  quantiles: p in {0.1, 0.5, 0.9} on a 1e6 stream, worst |err|
  **0.0005** (bar 0.02).
- **Bootstrap** — ordinary and moving-block resampling of a
  statistic, PCG64 substreams keyed by (seed, resample_id) so
  resamples are slab-parallel by construction (the same substream
  contract as the e-means/H0 determinism policy). Gated: mean-SE
  ratio 0.85-1.15 (measured 1.01), fixed-seed bit-identity,
  distinct resample ids draw differently.

### 10.6 Measured bugs, H7 (all fixed in-tree, all documented where
they live)

1. **Slab task scratch shared across rows** (PDE DST-I): the serial
   loop passed `&tk[i]` while the task indexed `[row]` from the
   base — out-of-bounds access, SEGV under ASan. Fixed to the
   documented task contract (`arg` base + `task_id`).
2. **Burgers CFL gate** `dt/dx <= 1` silently rejected legitimate
   large-dt cases and accepted unstable `max|u| > 1` ones; the gate
   is now `dt*max|u|/dx <= 1`.
3. **N-body point-mass power**: the first draft used `r^-4`; exact
   2-body at separation 1.0 passed (r^-3 == r^-4 at r=1) and larger
   separations gave ratios 1/3, 1/9. Fixed to `r^-3`.
4. **N-body tree construction** went through two bad designs (per-node
   scratch written with absolute positions -> ASan
   heap-buffer-overflow; relative positions treated as absolute ->
   vanished bodies) before the single-global-permutation design.
5. **Dangling `RCRE_NbNode*` across `realloc`** (heap-use-after-free
   under ASan): node fields are now re-derived through `t->nodes[]`
   after recursive calls.
6. **Verlet force-vs-acceleration**: raw force applied as
   acceleration; mass-1 tests hid it, a 100:0.01 central/satellite
   system exposed it. Fixed to `a = F/m`.
7. **NUTS selection**: sampling all nodes with weights `exp(A_m)` is
   a generic MH kernel, NOT slice sampling — nodes below the slice
   must be EXCLUDED. The chain ran but banana moments were badly
   biased (E[y] +1.04 vs 0). Fixed to uniform-over-nodes-under-slice.
8. **NUTS dual averaging (Stan form)**: the log-m drift has no
   restoring term; easy targets drive eps -> ~1e-5 and the chain
   freezes. Replaced with acceptance-based Robbins-Monro DA +
   divergence guard (docs §10.3). An earlier transcription of the
   same block also double-logged eps (mu -> NaN at iteration 2,
   guard silently pinned eps).
9. **out_accept contract**: the draft wrote an n_samples array
   (`out_accept[k]`) while callers (and the header's intent) passed
   one double — a measured stack-buffer-overflow under ASan. The API
   is now a single double = mean acceptance.
10. **P^2**: three separate bugs — the init-buffer insertion scan
    broke at the first match from the top (mis-sorted buffer); the
    new-max branch incremented only markers 1..3, leaving marker 4
    at n - (record count) until the markers collapsed and the
    parabolic update divided by zero; and a non-paper "slope
    ordering" linear-update condition that crossed adjacent markers
    at item 560 of a uniform stream and then produced 1e308 values.
    The paper-exact piecewise-linear/parabolic update with the OLD
    position passed to P^2 (its formula adds d internally) is what
    ships.
11. **Kurtosis double -3**: the bias-correction formula was fed the
    EXCESS (m4/m2^2 - 3) and subtracted 3 again — a constant -3
    offset (measured -4.1989 vs -1.1989 on uniforms). The formula
    takes the population kurtosis m4/m2^2.
12. **A test that read past its own buffer**: the 1-D NUTS "blowup"
    to 1e308 was the test indexing out[5749] of a 5000-element
    buffer — stack garbage, not a library bug. The library was
    always fine; the DA freeze (item 8) was the real find behind it.

### 10.7 Measured H7 bars (Apple M5 Max, 18 cores, Apple Clang 21,
-O3, `make bench_h7`)

Poisson 2D F64 (spectral / MG@1e-10 / AUTO, residual):

```
  n x n      cells     spectral   MG@1e-10   AUTO      resid
   127 x  127     16129     6.602 ms   85.783 ms    1.524 ms  1.35e-14
   255 x  255     65025     4.232 ms  392.077 ms    3.898 ms  1.52e-14
   511 x  511    261121    15.443 ms 1698.760 ms   14.953 ms  3.25e-14
  1023 x 1023   1046529    64.798 ms 6956.132 ms   61.302 ms  1.01e-13
```

Heat 2D FTCS 500 steps 512x512: serial 69.9 ms / pool 48.4 ms
(1.44x — stencil I/O-bound at this size). Burgers 1-D Shu-Osher
n=4096 one transit (16,384 steps): 214.6 ms (0.0131 ms/step). Wave
1-D 10k steps: 6.4 ms (0.0006 ms/step).

```
N-body Barnes-Hut force (F64, theta=0.5):
  n        serial    pool      speedup   (ms)
     1000       0.369      0.258       1.43x
    10000       4.820      1.108       4.35x
   100000      60.432     11.072       5.46x
N-body Verlet n=10k, 1k steps:  pool 2166.0 ms / serial 11098.8 ms (5.12x)
MCMC (sequential by design):
  NUTS banana     3300 iters     34.886 ms   (accept 0.88, div 19)
  NUTS 8-D gauss  5500 iters    285.049 ms   (accept 0.91, div 0)
  HMC banana      20000 iters (L=50)     14.079 ms   (accept 0.69)
Streaming:
  P^2 add x 10M   101.607 ms   (0.10 M adds/s, est 0.5001)
  bootstrap mean x10k (n=10k)  352.034 ms
  ESS (n=10k)     0.020 ms
```

The PDE and N-body slabs follow the library-wide determinism policy
(bit-identical across pool sizes); MCMC and streaming are sequential
by design and deterministic by seed.

### 10.8 Documented deferrals (H7.1)

The plan's H7 item list also named 3-D Barnes-Hut, 3-D / 7- / 9-point
stencils, advection-diffusion, NUTS window (Wand) adaptation, and
t-digest/KLL sketches. H7 closes at 2.4.0 with the coherent subset
above; the rest is an explicit follow-on, not silent scope drop:

- **3-D Barnes-Hut + 3-D stencils**: the 2-D BH machinery (single
  global permutation, slab-parallel walk, bit-identical pool)
  transfers directly to 3-D octants; queued as the first H7.1 item.
- **Advection-diffusion** (upwind + centered diffusion, von Neumann
  stability-gated): straightforward on the existing stencil
  substrate; queued.
- **NUTS window adaptation**: the acceptance-based DA + divergence
  guard already keeps every measured target in a healthy regime
  (0 divergences on the 8-D Gaussian, ~8/3300 on the banana);
  window (Wand et al.) tuning is a refinement, not a correctness
  gap. Queued with the 3-D BH work.
- **t-digest/KLL**: the P^2 replacement is measured (rank error
  0.0005 vs the 0.02 bar) and gates as well as the GK draft did
  not; switching is only for API-completeness.
- **ESS "within 20% of a Stan run"**: no Stan reference run exists
  on this machine; the shipped ESS gate is the white-noise ~ n and
  AR(1) ~ n/19 bars (Geyer IMS estimator vs theory). The statistical
  gate is what it can be without the reference; the 20%-of-Stan bar
  is queued for H9's conformance wave (where cross-tooling fixtures
  already live).
- **Burgers "±1 cell vs published shock positions"**: the shipped
  bar is L1 <= 2e-2 vs the Hopf-Lax exact solution over one full
  Shu-Osher transit (measured 1.2e-2) — the self-contained version
  of the same property (L-inf stalls at ~half a shock jump by
  design of 2nd-order TVD schemes, so cell-counting against a
  specific published run was the less honest bar anyway).

## 11. Phase H8 — Optimization Polish: L-BFGS, FISTA, CD-LASSO, SPSA (2.5.0)

### 11.1 Scope

Four first-class F64 optimizers in `src/core/opt.c`, all sequential and
deterministic (SPSA under a fixed PCG64 seed), all with report-struct
telemetry (no callbacks in the numerics path — the H4 lesson):

- `rcre_lbfgs_f64` — two-loop L-BFGS (Nocedal-Wright, m = 1..64,
  default 10; strong-positivity pair filter; non-descent reset to −g).
  Armijo backtracking. Stopping tests are the **scipy L-BFGS-B
  definitions** (ftol on max(|f|,|fp|,1), gtol on the infinity-norm
  gradient, optional xtol measure max|x−y|/max(1+|x|) — disabled by
  default here because scipy L-BFGS-B has no xtol option, so the bar
  runs with matched test sets). Optional box bounds via **projected
  backtracking trials** (documented heuristic — not the
  Byrd-Lu-Petryakov bounded algorithm; an explicit deferral, §11.6).
- `rcre_fista_f64` — Beck-Teboulle for min f(x) + g(x) with a user
  `prox_g`; backtracking local-Lipschitz check on the smooth part.
  out_f is the smooth value at the result (f + g is the caller's sum).
- `rcre_cd_lasso_f64` — cyclic coordinate descent for
  min 1/2||Ax−b||² + λ||x||₁ with the exact soft-threshold coordinate
  update and classical AᵀA Gram bookkeeping (O(d²n) upfront,
  documented). KKT-verifiable.
- `rcre_spsa_f64` — Spall zeroth-order: 2-point estimate
  (f(x+cΔ)−f(x))/(cΔ) and 4-point sign estimate (f(x+cΔ)−f(x−cΔ))/(2cΔ),
  a_k = a0/(k+1)^γ, fresh ±1 perturbations from PCG64 (fixed
  consumption order → bit-identical run per seed).

### 11.2 Bars (measured — `tests/test_h8.c`, 98 checks)

**L-BFGS vs scipy 1.18.1 L-BFGS-B** (fixture
`tests/fixtures/h8_scipy_reference.txt`, generated by
`tests/ref_h8_scipy.py`; both sides ftol=gtol=1e-10, xtol disabled on
the C side, maxiter=1000·d):

| problem      | d  | f (ours)            | f (scipy)           | rel err | iters (ours/scipy) |
|--------------|----|---------------------|---------------------|---------|--------------------|
| rosen2       | 2  | 4.5731e-16          | 2.7863e-14          | 2.7e-14 | 9 / 14             |
| rosen10      | 10 | 1.7048e-16          | 6.3934e-17          | 1.1e-16 | 14 / 12            |
| beale        | 2  | 3.2240e-17          | 1.9484e-15          | 1.9e-15 | 12 / 15            |
| wood         | 4  | 2.8480e-15          | 3.8922e-13          | 3.9e-13 | 47 / 58            |
| styblinski5  | 5  | −3.641601197e+01    | −3.641601197e+01    | 0.0     | 5 / 5              |
| exponlin10   | 10 | −5.8065e+01         | −5.7866e+01         | —       | 3 / 1              |
| quad50 (SPD) | 50 | −1.533598751e-01    | −1.533598751e-01    | 1.9e-13 | 11 / 12            |
| boundquad30  | 30 | −1.629004516e-01    | −1.629004516e-01    | 5.0e-13 | 10 / 10 (f only)   |

Bar classes (both documented in the test):

- **status-0 class** (scipy converged by ftol/gtol): |f_ours − f_scipy|
  ≤ 1e-10 relative AND iterations ≤ 1.25× scipy nit (+1 accepted-step
  off-by-one guard). All of the above pass; worst measured relative
  error 5.0e-13, worst iteration ratio 1.17 (rosen10, 14 vs 12 —
  scipy's stronger Wolfe line search finds the valley bottom 2 steps
  sooner; our Armijo takes slightly more, always cheaper steps).
- **early-stop class** (exponlin10): scipy's status=2 means the
  **Fortran L-BFGS-B driver's internal relative-step test** stopped it
  — measured: scipy stops at nit=1 for *any* ftol in 1e-8..1e-15, so
  the step test is internal, not an option. Bar: f_ours ≤ f_scipy
  (we must reach at least as low; −5.8065 < −5.7866 ✓) and bounded
  extra work (3 iters ≤ 4·1+2 ✓).

**CD-LASSO** (400×80 random instance, λ ∈ {0.05, 0.25, 1.0}): KKT
violation ≤ 1e-8·(1+λ) (measured 2.7e-9..3.6e-9), objective improves
from x0=0, bit-identical across repeated runs.

**FISTA vs CD-LASSO** (independent algorithms, same instance):
composite objectives within 1e-4 relative — **measured 1.2e-16..4.9e-16**
(machine precision: they land on the same minimizer).

**SPSA** (SPD quad-20, equal eval budgets: 4-point 500 steps = 1501
evals, 2-point 750 steps = 1501 evals): both reach ≥95% of the total
possible decrease of an L-BFGS reference (measured: 4-point f=0.379,
2-point f=0.328 vs need ≤ 4.23; on this single seed the 2-point edge
is the measured trend — reported, not gated). Fixed-seed runs are
bit-identical (memcmp on x_out + exact f equality).

### 11.3 Bugs measured during H8 (fixed in-tree)

1. **FISTA Lipschitz anchor** — the backtracking check used f(x_k)
   (the previous iterate) instead of f(y_k) (the gradient point): on a
   random 400×80 LASSO the check failed until L doubled **53 times
   (L ≈ 9e15)**, the prox step collapsed to ~2.5e-15, and the chain
   stalled at x0 (identical output for all three λ — the symptom).
   Fix: anchor at f(y), one extra value evaluation per iterate.
2. **L-BFGS y-storage order** — the draft stored y = g_new − g_old
   *after* copying g_new over g_old (y ≡ 0 → every H_k ≡ I, a silent
   degradation to steepest descent). Fix: capture y before the
   overwrite.
3. **SPSA sign-estimator scale** — the 4-point (sign) estimator has
   O(‖∇f‖²) per-component variance (each component is
   Σ_j ∇f_j Δ_jΔ_i): a0 = 0.5 and 0.1 both diverge on quad-20
   (measured f → 1e33..1e108), and a0 = 5e-3 diverges on quad-100.
   The initial step must be O(1): a0 ≈ c/‖∇f(x0)‖₂.
4. Test-side (documented per the H4 lesson, kept out of the library
   blame list): (a) a quad objective draft computed ½xᵀ(Qx−p) — the ½
   applied to the linear term too — giving f = −1.5e-3 at the true
   minimizer of a problem whose f* is −0.153; (b) the KKT check used
   |aⱼᵀr + λ| instead of |aⱼᵀr − λ·sign(xⱼ)| — violations read exactly
   2λ at every λ; (c) a FISTA smoke prox ignored λ (threshold = step
   instead of λ·step) — all three λ runs returned the identical
   solution, which looked like FISTA "ignoring λ".

### 11.4 Benchmark bars (`make bench_h8`, M5 Max 18 cores, -O3)

```text
L-BFGS rosenbrock-100:            0.038 ms  (20 iters, 51 fE, f=3.1e-13)
L-BFGS SPD-quad-500:              3.42 ms   (12 iters, 22 fE, f=-0.131)
CD-LASSO 2000x500 lam=0.5:        259.5 ms  (31 passes, obj=2.637e+02)
FISTA-L1 2000x500 lam=0.5:        318.4 ms  (226 iters, obj=2.637e+02)
SPSA-4 quad-100, 10k steps:       72.1 ms   (30k fE, 95.9% of the drop,
                                              0.42 M evals/s)
```

FISTA and CD-LASSO agree on the 2000×500 objective to all printed
digits (2.637e+02). The CD Gram build (O(d²n) = 5e8 flops) is a
visible part of the CD cost at this size; FISTA's backtracking L adds
~30% over CD's 31 passes (226 prox-grad iters — sublinear O(1/k²)
convergence, as theory predicts; CD is the recommended solver for the
plain LASSO form).

### 11.5 Explicit deferrals (H8.x)

- Byrd-Lu-Petryakov bounded L-BFGS (we ship the projection heuristic;
  box KKT is not exact at the active set).
- Trust-region / conjugate-gradient / Newton (need Hessian or
  Hessian-vector products — a separate API).
- Bundled prox operators (LAD/denoising): the FISTA framework takes a
  user prox; nothing is bundled.
- SPSA iterate averaging / best-point tracking (we return the final
  iterate).
- Parallelism: the optimizers are sequential by design (stateful
  recurrences); parallelizing objective evaluation is the caller's job.
- CD for generalized LASSO / logistic regression (plain LASSO form
  only).

## 12. Phase H9 — HPC Wave: MPI Shim, Distributed Ops, 3-D Poisson, Pow2 FFT (2.6.0)

### 12.1 Scope (game-plan H9, honest local subset)

The game plan's H9 is the *silicon wave*: SVE2/AMX/RVV validation on
real Neoverse/SPR/emission silicon plus a first distributed
operation. This machine is Apple M5 Max with no MPI installed and no
RISC-V userspace target, so the silicon-validation half is **deferred
with the deferral documented** (the RPi5/RPi4 devices are the
registered cross-machine pair; SVE2/AMX require the server silicon
when it is attached). What H9 ships here, in the same closure order
(H0–H8 pattern):

- **A 12-function MPI shim** over self-registering backends (the D1
  bridge pattern; `loopback` always present as the conformance
  oracle, `mpi` optional behind `RCRE_HAVE_MPI`, compiled empty
  without MPI — the H6 CUDA pattern: a missing backend is a skipped
  gate, not a build failure).
- **The first distributed operations** on that shim: a row-block
  distributed SpMV (+ all-to-all gather) and a **halo-exchange
  5-point stencil** (one halo row per neighbor via isend/irecv).
- **A 3-D PDE domain layer**: 7-point Dirichlet stencil (h=1, F64) and
  the **3-D Poisson spectral solver** (separable 1-D DST-I over three
  axes, strided-line driver, no transposes).
- **The H2 bar-gated FFT optimization**: an in-place power-of-two F64
  FFT with a per-call twiddle table (`rcre_fft_pow2_f64`), wired as
  the fast path for all pow2 lengths (recursion and Bluestein
  included).
- **`-ffp-contract=off` library-wide** (measured: FMA contraction
  across code shapes broke the 256³ serial-vs-pool bit-identity;
  identical with the flag off — the rcvc palette precedent, promoted
  to policy).
- **The game-plan full PDE run**: one 256³ Poisson solve, measured
  bit-reproducible across pool sizes {NULL,1,2,4,8,18} (FNV-1a hashes
  identical — `make bench_h9`), residual ≤ 1e-13, and the
  cross-machine half queued for the RPi5 registration.

### 12.2 Determinism (across processes)

The library-wide determinism policy now spans process boundaries:

- The shim's collectives are **ordered**: bcast copies from the root;
  `allreduce_sum_f64` is an ascending-rank left fold (the real-MPI
  backend implements the fold explicitly — gather to rank 0, fold in
  rank order, broadcast — because a vendor's `MPI_Allreduce(MPI_SUM)`
  pairwise tree is NOT bit-stable against the serial path).
- The distributed ops use the **core row expression verbatim**
  (same values, same operation order — the halo rows ARE the true
  neighbor rows), so a 2-process run is bit-identical to the serial
  oracle. Verified on the data path against the loopback backend
  (`test_h9`); the real 2-process run is `make test-mpi2` (skips
  clean without MPI).
- Slab partitions are deterministic functions of (total, size, rank);
  an empty slab (total < size) neither sends nor computes.

### 12.3 3-D Poisson solver

Layout: `c = j*nz*nx + k*nx + i` (the H4/H7 2-D layout extended; x is
fastest). Operator `A = T_x⊗I⊗I + I⊗T_y⊗I + I⊗I⊗T_z`, `T =
tridiag(-1,2,-1)`, zero Dirichlet (boundary folded into b). Solve:
three forward + three inverse strided-line DST-I stages (line lengths
per axis: x lines over i (stride 1), y lines over j (stride nz·nx,
base i+k·nx), z lines over k (stride nx, base j·nz·nx+i)), the
eigenvalue divide `1/(λx[i]+λy[j]+λz[k])` in between, one final
scale. `O(n³ log n)`.

Two measured implementation notes:

- **n = 256 is the pathological DST size**: the odd-extension length
  is 2(n+1) = 514 = 2·257 (257 prime) — every line runs the
  Bluestein half (three 512-FFTs + chirps). The H9 pow2 table path
  makes that 3·(8.6 µs) + chirp ≈ 120 µs/line instead of the H2
  recursion's ~450 µs — the 40× in the 256³ wall times (193 s → 4.8 s
  serial; 193 s → 4.1 s pooled is NOT the speedup — pooled is 4.1 s,
  serial 46 s: ~11× on 18 cores).
- **The no-half writeback**: the 3-D stages write `-Im(FFT)` (2× the
  DST per stage; the 2-D solver keeps its `-0.5·Im` — independent
  code, unchanged) and divide the final scale by 2⁶. Residual
  7e-15 (64³) / 1.8e-14 (256³) vs bar 1e-13.
- **Pow2 fast path + slab scratch**: 65,536 lines × per-line malloc
  cost minutes; the stage now partitions lines into ≤ 4·cores slabs
  with per-slab scratch (the H7 slab contract at 3-D scale — slab
  count never changes the per-line arithmetic, so bit-identity holds
  across pool sizes AND slab counts).
- **NaN gate (measured bug)**: the pow2 dispatch reads `im` as input
  (zero imaginary part) — the line scratch must zero BOTH halves per
  line; the generic path hid the bug with its calloc'd temp. Caught
  by the irregular 33×31×29 fixture (n=31 → 2(n+1)=64 is pow2).

### 12.4 Measured bars (Apple M5 Max, 18 cores, -O3, -ffp-contract=off)

`make bench_h9`:

```text
fft-512 f64:          pow2-inplace 8.6 us  (H2 recursion was 40.1 us — 4.7x)
fft-514 f64 (Bluestein, the 256-DST line): 119.8 us
SpMV 4096^2 CSR:      core 38.2 ms | loopback-dist 36.1 ms | bit-identical
stencil5 256^2:       core 0.1 ms  | loopback-dist 0.1 ms   | bit-identical
Poisson 64^3:         pool-18 ~26 ms | serial 305.7 ms
Poisson 256^3 (full run):
  pool=0 : 46357.9 ms   hash=1e84709827a34f24
  pool=1 : 46098.4 ms   hash=1e84709827a34f24
  pool=2 : 16340.4 ms   hash=1e84709827a34f24
  pool=4 : 10900.5 ms   hash=1e84709827a34f24
  pool=8 :  6790.8 ms   hash=1e84709827a34f24
  pool=18:  4064.7 ms   hash=1e84709827a34f24
  bit-reproducible across pool sizes: YES (6/6 hashes identical)
  residual (test_h9): 7.0e-15 (64^3), 1.8e-14 (256^3) — bar 1e-13
```

Honest bars, per policy:

- The 256³ serial time (46 s) is what a zero-dependency spectral 3-D
  solver costs when the axis size makes every DST line Bluestein.
  Reference libraries (MAGMA/Ginkgo) on the same run are the
  silicon-wave comparison — deferred with the silicon.
- pool=1 ≈ serial (46.1 s vs 46.4 s): the threadpool's main-thread
  participation with a 1-worker pool is measured parity, not 2×.
- The loopback dist-ops cost ≈ the core (same row arithmetic, zero
  communication) — the 2-process numbers (real halo latency) are the
  `test-mpi2`/server measurement.
- Cross-machine bit-reproducibility (the game-plan bar's second half)
  is queued: same binary + pool set on the RPi5 once the runner is
  registered; the FNV-1a hash above is the comparison tag.

### 12.5 Public API (additions)

```c
/* FFT (the H2 header's bar-gated table optimization) */
RCRE_NODISCARD int rcre_fft_pow2_f64(double* re, double* im, size_t n,
                                     int invert); /* in-place, pow2 only */

/* 3-D domain layer (layout c = j*nz*nx + k*nx + i) */
RCRE_NODISCARD int rcre_pde_stencil7_f64(size_t nx, size_t ny, size_t nz,
                                         const double* u, double* v,
                                         RCRE_ThreadPool* pool);
RCRE_NODISCARD int rcre_pde_poisson3d_fft_f64(size_t nx, size_t ny, size_t nz,
                                              const double* b, double* u,
                                              RCRE_ThreadPool* pool);

/* distributed ops on the 12-fn MPI shim (extras/rcre_mpi.h) */
RCRE_NODISCARD int rcre_dist_spmv_rows_f64(RCRE_MpiCtx* ctx,
                                           const size_t* row_ptr,
                                           const size_t* col_idx,
                                           const double* values, int rows,
                                           int r0, int r1, const double* x,
                                           double* y_local);
RCRE_NODISCARD int rcre_dist_gather_rows_f64(RCRE_MpiCtx* ctx, int n, int r0,
                                             int r1, const double* y_local,
                                             double* y_full);
RCRE_NODISCARD int rcre_dist_stencil5_f64(RCRE_MpiCtx* ctx, size_t nx,
                                          size_t ny, const double* u_local,
                                          double* v_local);
/* ctx == NULL is the serial oracle path for every dist op. */

/* the shim (include/extras/rcre_mpi.h, self-contained header) */
rcre_mpi_open / _close / _size / _rank / _barrier / _bcast_f64 /
rcre_mpi_send_f64 / _recv_f64 / _isend_f64 / _irecv_f64 / _wait /
rcre_mpi_allreduce_sum_f64   /* ordered ascending-rank fold */
```

### 12.6 Test gates (tests/test_h9.c — 56 checks)

- **Shim (loopback)**: registration, size/rank, bcast/allreduce
  identity, point-to-point contract, error inputs; the `mpi` backend
  absent → open fails clean (the H6 pattern).
- **Dist SpMV oracle**: serial path, loopback path, and a manual
  2048/2048 row-block split all bit-identical to `rcre_spmv_f64`
  (64×64 Poisson CSR, ascending col order — the H4/H6 convention).
- **Halo stencil oracle**: serial + loopback bit-identical to
  `rcre_pde_stencil5_f64`; the boundary-row arithmetic with a real
  halo value verified against the core row expression.
- **Poisson 64³**: residual ≤ 1e-13 and bit-identity across pool
  sizes {NULL,1,2,4,8,18}.
- **Poisson 256³ (the game-plan run)**: serial vs pool-18 bit-identical
  + residual ≤ 1e-13.
- **Irregular 33×31×29 vs the external numpy DST-I fixture**
  (`tests/fixtures/h9_poisson3d_reference.txt`, regenerated by
  `tests/ref_h9_vectors.py` — infra-immune): max rel err 1.1e-15,
  NaN-aware. 33/31/29 are all prime — the fixture exercises the
  Bluestein half of every DST line.
- `make test-mpi2`: the real 2-process gates (halo stencil + dist
  SpMV/gather + ordered allreduce vs the serial oracle) — `mpirun
  -np 2`, skips clean without MPI.
- `make test-sanitizers` covers `test_h9` (ASan+UBSan, zero findings —
  it caught the 2048/2048 test-side overflow pre-gate).

## 13. Validation gates (per release)

- `make all` — zero warnings (`-Wall -Wextra -Wpedantic`).
- `make test` — full suite incl. `test_linalg` (14 suites, 827,421
  assertions), `test_rng` (9 tests, 395,474 assertions incl. the
  Philox-4x32-10 fixture bits),
  `test_fft_f64` (3 tests, 4,632 assertions), `test_direct` (7 tests,
  139 assertions), `test_h4` (10 tests, 1,097 assertions incl. the
  BiCGStab 40-step parity fixture), `test_zarr` (9 tests, 141
  assertions: metadata golden, 12-dtype round-trips, fill semantics
  incl. NaN/bool, edge chunks, 1D–8D, zstd, slab determinism,
  strided blocks, errors), `test_pde` (13 tests: DST-I, spectral
  Poisson eigenvector + residual, FFT-vs-MG, pool determinism, heat
  mode + stability gate, Burgers Shu-Osher L1/mass/range, Hopf-Lax,
  wave convergence + energy + CFL gate), `test_nbody` (8 tests: BH
  vs direct, theta scaling, pool determinism, 2-body elliptic/
  parabolic, mass weighting, 800-body chaos, errors), `test_mcmc`
  (9 tests / 66 checks: MH, HMC banana acceptance 0.692, NUTS 1-D/
  banana/8-D, ESS, integrators, errors), `test_stream` (7 tests / 32
  checks: Welford vs naive, merge, P^2 rank <= 0.0005, bootstrap, 
  errors), `test_h8` (4 groups / 98 checks: L-BFGS vs the scipy
  fixture — final-f 1e-10 rel + 1.25x iterations on the status-0
  class, f_ours <= f_scipy on the early-stop class — CD-LASSO KKT,
  FISTA-vs-CD objective 1e-4 rel (measured 1e-16), SPSA 95% decrease
  + fixed-seed bit-identity, error inputs), `test_h9` (5 groups / 56
  checks: MPI-shim loopback oracle, dist SpMV/halo-stencil bit-
  identity vs the core ops, Poisson 64^3 residual + pool
  bit-identity, the 256^3 game-plan run serial-vs-pool-18 bit-
  identical + residual, the 33x31x29 numpy DST-I fixture 1e-15 rel,
  error inputs), `test_gpu` (5 tests; 363 assertions on the
  RTX 5060 Ti: buffer round-trips, CPU-backend bit-identity, device
  precision bars + double-launch determinism, 65,536-word Philox
  bit-identity, CG+ILU(0) parity at 75 vs 76 iters; the device gates
  skip clean where no CUDA backend is linked) and the amalgamated
  single-header build.
- `make test-mpi2` — the real 2-process distributed gates (halo
  stencil + dist SpMV/gather + ordered allreduce vs the serial
  oracle) under `mpirun -np 2`; skips clean where mpicc/mpirun are
  absent (the loopback backend covers the data paths).
- `make conformance-zarr` — 13 fixed arrays bit-identical C ↔
  zarr-python in both directions + golden metadata + NaN fill
  (skips clean without a zarr venv).
- `make test-sanitizers` — ASan + UBSan + leak check, zero findings.
- `make bench` — numbers appended to this file / `docs/benchmarks_sota.md`
  with machine + toolchain + commit (`bench_h5` covers the Zarr I/O
  bars of §8.5; `bench_h6` covers the GPU bridge bars of §9.4;
  `bench_h7` covers the domain-layer bars of §10.7;
  `bench_h9` covers the H9 bars of §12.4 — the 256^3 bit-reproducible
  run (6 pool sizes, FNV-1a hashes), dist ops, pow2 FFT;
  `bench_h8` covers the optimization bars of §11.4).
- `make test-portable` — the amalgamated header compiles + runs in a
  truly clean directory (no repo includes on the line — the probe
  deliberately drops `$(CFLAGS)` so a leaked local include cannot be
  rescued by the worktree).
- x86 path: AVX2/AVX-512 cells compile under their pragma-guarded TUs on
  any x86-64 compiler and are only dispatched when CPUID reports the
  feature; validated on the next x86 CI runner (H9).
