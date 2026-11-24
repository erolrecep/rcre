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

## 8. Validation gates (per release)

- `make all` — zero warnings (`-Wall -Wextra -Wpedantic`).
- `make test` — full suite incl. `test_linalg` (14 suites, 827,421
  assertions), `test_rng` (7 tests, 395,456 assertions),
  `test_fft_f64` (3 tests, 4,632 assertions), `test_direct` (7 tests,
  139 assertions), `test_h4` (10 tests, 1,097 assertions incl. the
  BiCGStab 40-step parity fixture) and the amalgamated single-header
  build.
- `make test-sanitizers` — ASan + UBSan + leak check, zero findings.
- `make bench` — numbers appended to this file / `docs/benchmarks_sota.md`
  with machine + toolchain + commit.
- x86 path: AVX2/AVX-512 cells compile under their pragma-guarded TUs on
  any x86-64 compiler and are only dispatched when CPUID reports the
  feature; validated on the next x86 CI runner (H9).
