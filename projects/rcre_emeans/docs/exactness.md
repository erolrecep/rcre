# rcre_e-means — Exactness and Threading Contract

This document is the numerical contract for the e-means operator
(`src/e_means.c`). It is what the test suites in `tests/` verify. Read it
together with `README.md` and the implementation report
(`~/Desktop/emeans_rcre_implementation_2026-11-15.md`).

The reference is the faiss `contrib/e_means` code (Bessa et al., 2026,
arXiv:2611.09501). Where the paper wording and the shipped reference code
differ, the reference code wins.

---

## 1. Three-tier test contract

Because the C port and the Python oracle use different PRNGs
(splitmix64 vs MT19937), their random streams (init row subset, epoch
shuffles, Gumbel revivals) differ even at the same seed. Bit-parity of a
full run is therefore impossible by design. The contract is tiered:

### Tier 1 — Strict parity (0 ULP): `tests/test_parity.c`

The oracle's own streams are serialized (`streams/strict_*.bin`) and
**injected** into the C port: the C run consumes the oracle's labels,
batch-sum byte patterns (F32), counts, and Gumbel-U vectors, so every
algorithmic decision (EMA update, revival set, split targets, alpha
schedule) is identical on both sides.

- **Replay with revival disabled** (`strict_norev`): every centroid
  coordinate and every mass after every update must be **0 ULP** from the
  oracle's post-update state. ~32k assertions.
- **Replay with revival enabled** (`strict_revive`): same, with the
  revival mechanism active (Gumbel keys, donor selection, split writes,
  nextafter collision handling, U-stream alignment). ~174k assertions.

This tier proves the **algorithmic** port is a 1:1 transliteration:
operation order, F32 roundings, the alpha==1 corner case (mass = n_j,
c_j = s_j/n_j), per-dimension ±1e-4 split deltas, first-index-on-tie
assignment, and the consume-full-k-vector-per-check U alignment.

What this tier does NOT prove: that the C port's *own* RNG streams,
assignment kernel, and batching produce a solution of the reference
quality. That is Tier 3.

### Tier 2 — Statistical parity (quality gate): `make quality-par`

A full run on the shared 20000×24 blob dataset (T=5, auto recipe,
revival on) is executed by both sides with their **native** PRNGs.
Gates (`ref/compare_parity.py`):

- final training MSE per dimension (exact nearest assignment of the
  finalized centroids) of the C run within **+0.5%** of the oracle's;
- no unvisited rows (dead assignments) in the C run (T is integral);
- no dead clusters at the end.

Raw label-index agreement is NOT gated — two independent runs may label
an equivalent partition with different cluster indices. A
greedy-cluster-aligned agreement is reported for the record only.

On this machine the C run typically *beats* the oracle on MSE (different
initialization draws); the +0.5% gate is a ceiling, not a target.

### Tier 3 — Quality floors (algorithm sanity): `tests/test_quality.c`

- **Two-blob trap** (the paper's local-minima scenario, n=3200, k=8):
  e-means must put 7/8 centroids on the large blob (EMA + revival escape
  the small-blob trap) and beat or tie Lloyd-25 (k-means++ init) within
  5%.
- **Full-run non-degeneracy** (n=20000, k=256): e-means(5) must beat the
  single-cluster (global-mean) baseline by two orders of magnitude, and
  its numbers vs Lloyd-25 are reported (at this small scale, n=20K,
  k-means++ + 25 Lloyd iterations keeps an init advantage of ~2.4x —
  the paper's ">= at equal epochs" claim is a large-scale one, see
  `docs/benchmarks`).

## 2. F32/F64 split

| Quantity | Type | Notes |
|---|---|---|
| centroids `c_j`, masses `N_j` | F32 | model state (reference: F32) |
| batch sums `s_j`, counts `n_j` | F32 | unbuffered per-row accumulation, processing order (reference: F32) |
| scheduled alpha `alpha(t)` | F64 → F32 | Eq. 5 in double, cast once at use (reference) |
| assignment argmin (GEMM path) | F32 | `cc - 2 x·c` (row-constant removed); first index on ties |
| assignment argmin (exact path) | F32 | per-dimension `(x-c)·(x-c)` sequential, no FMA |
| objective (inertia) | **F64** | true squared distance to the winner per row, F64 sum (reference: F32; **documented improvement** — avoids negative MSE on large-offset data and removes ~1e-3 relative noise from the reported number) |
| finalization pass sums | **F64** | per-cluster sums/counts over the last epoch (reference: F32 pass sums; **documented improvement**) |
| finalization division | F32 | mean = pass_sum / pass_count, cast to F32 (reference) |
| revival keys `float32_log(N) - log(-log(U))` | F64 | the reference draws U in F64 (MT19937) and computes the key in double |

The two F64 improvements do not affect the Tier 1 replay (the oracle
mirrors the C port there, so 0 ULP is preserved) and they make the
reported inertia monotone non-negative on large-offset data
(`cc - 2 x·c` goes negative when the data centroid is far from the
origin — see the sift100k case, where the raw GEMM objective would be
~1e10 while the true MSE is ~1e4).

## 3. Assignment kernels

`rcre_ml_emeans_assign` (and the batch assignment inside `fit_ex`) is
dual-kernel:

- **Exact kernel** (`k*d <= 16384`): per-row, per-dimension
  `(x-c)·(x-c)` in F32, sequential over d, no FMA (`-ffp-contract=off`),
  argmin with first-index-on-tie. No scratch beyond O(1) per row.
  This is the code path for every image-palette use case (k ≤ 256,
  d ≤ 12 → k*d ≤ 3072).
- **GEMM kernel** (`k*d > 16384`): `D = x @ c^T` in F32 via the RCRE HAL
  (dispatched: NEON/AVX-2/scalar), argmin on `cc - 2D` (row-constant
  removed), then the **true** distance to the winner in F64 for the
  objective.

The split threshold 16384 makes the exact kernel's extra scratch (none)
vs the GEMM path's Dmat budget (16 MB) a wash at ~k=4096, d=4, and the
GEMM path wins by a large margin in the high-d regime (sift100k: the
GEMM assignment is the pool-parallelizable part).

## 4. Threading contract

- **Slab partition**: the batch rows are split into contiguous slabs of
  `>= 64` rows, at most 16 (one partial-stat buffer per slab), all-or-
  nothing: `nslab = (B < 64) ? 1 : min(16, ceil(B/64))`.
  The partition depends **only on the batch size B** — never on the pool
  size or the core count.
- **Bit-identity across pool sizes AND machines**: the threadpool only
  decides *which worker* runs which slab; the slab boundaries, the
  within-slab row order, and the slab-merge order are all fixed by B.
  45k-assertion bit-identity between 1-worker and 18-worker pools is
  verified (`tests/test_determinism.c`), and the same bits are produced
  on any other machine.
- **Serial (no pool) = single slab**: with `pool == NULL` the whole
  batch is one slab, i.e. a pure sequential F32 accumulation. A pooled
  run uses the two-level grouping (within-slab sums, then slab merge),
  which can differ from the serial grouping at the ULP level. That ULP
  difference feeds back chaotically through the iterative EMA system
  (one flipped early argmin cascades). The serial-vs-pooled contract is
  therefore **quality equivalence** (final MSE within 0.5%, training
  label agreement ≥ 99%), NOT 1 ULP — enforced and stated by
  `test_serial_vs_parallel_ulp`.
- **No shared mutable state in the data path**: each slab writes its own
  partial buffer; the single-writer (main thread) does the EMA update,
  revival, and finalization. The only synchronization is the
  threadpool's task join.

## 5. Determinism

- Same seed + same (n, k, B) → **bit-identical** centroids, masses,
  labels, split count — on any pool size and on any machine (slab
  partition is a function of B alone, section 4). Only the
  serial-no-pool run (single-slab grouping) is a distinct, separately
  quality-guaranteed configuration.

## 6. Known divergences from the reference (all intentional)

1. **RNG**: splitmix64 streams replace MT19937 (zero-dependency RCRE
   policy; the paper's quality claims are RNG-agnostic at the
   statistical-parity level).
2. **F64 objective + F64 finalization pass sums** (section 2).
3. **Exact-kernel argmin for k*d ≤ 16384**: the reference always uses
   `cc - 2 x·c` in (effectively) F32 matrix ops; for small k*d the C
   port computes the true distance directly. Argmin is identical
   (row-constant removal does not change the argmin; first-index-on-tie
   is preserved because the exact kernel compares the same monotone
   function). Quality-neutral; the objective is more honest.
4. **`nepochs` is validated (>0, ≤16, finite) in the convenience
   `fit`**, while `_ex` treats ≤0 as "use the default 5" (matching the
   reference's `n_epochs=None` semantics on the full path).
5. **Init without replacement via partial Fisher–Yates** on a scratch
   index array (the reference draws `permutation(n)[:k]`; same
   distribution, cheaper).

## 7. What the tests do NOT cover

- GPU (out of v1 scope; see the report's roadmap).
- `max_points_per_centroid` (reserved API field; the reference's
  balanced-assignment extension is not in the shipped `e_means.py`).
- Non-Euclidean metrics (v1: Euclidean only).
- Multi-process / GPU-stream convergence semantics.
