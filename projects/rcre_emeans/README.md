# rcre_e-means

**e-means** (Bessa et al., 2026, arXiv:2611.09501) as a C23 operator in
the [RCRE](https://github.com/erolrecep/rcre) style: zero-dependency,
arena-backed, F32 model state, pool-parallel, deterministic.

e-means is a mini-batch k-means with exponential-moving-average centroid
updates, a cosine-annealed step schedule, scaling-law batch/step
selection, and dead-cluster revival (Gumbel-max donor selection +
per-dimension ±1e-4 splits). It trains in `T` full data passes (default
`T = 5`) and beats full-data k-means at equal or smaller epoch budgets
in the large-scale regime (n ≥ 100K, k ≥ 1024) — see
[`docs/benchmarks.md`](docs/benchmarks.md).

This project is the **development harness** for the e-means operator,
which lives in the core RCRE library as [`src/ml/e_means.c`](../../src/ml/e_means.c)
(API: `rcre_ml_emeans_*` in [`include/rcre.h`](../../include/rcre.h)).
Everything here builds against the core `lib/librcre.a`: the Python
oracle (`ref/`), the stream fixtures (`streams/`), the oracle-dependent
test suites (`tests/`), the benchmark (`bench/`) and the demo
(`examples/`). The core repo carries its own numpy-free test suite with
checked-in fixtures (`../../tests/emeans_fixtures/` +
`../../tests/test_emeans.c`).

## Build and run

```sh
make            # help (the Makefile is the control center)
make all        # library + tests + bench + demo
make demo       # visual 2D cluster demo with real-time ASCII scatter map
make test       # all 6 C suites (incl. strict-parity replay)
make san        # ASan + UBSan over all suites
make refs       # regenerate the Python oracle streams (needs .venv)
make parity     # strict 0-ULP parity vs the injected oracle streams
make quality-par# statistical quality parity vs the oracle (MSE gate)
make bench      # 4krgb + sift100k vs Lloyd-25 (add --big for embed1m)
make venv       # numpy venv for the oracle
```

Toolchain: C23 compiler (Apple Clang 15+, llvm-clang 16+, gcc 13+).
Flags: `-std=c23 -Wall -Wextra -Wpedantic -O3 -ffp-contract=off`.

## API

```c
RCRE_EMMeansParams p = rcre_ml_emeans_params_default();  // T=5, auto laws
p.nepochs = 5.0f;
float inertia; int n_splits;
rcre_ml_emeans_fit_ex(&X, /*k=*/256, &p, &C, &L,
                      &inertia, &n_splits, arena, pool);
/* or: rcre_ml_emeans_fit(&X, k, &p, &C, arena, pool) */
/* exact nearest-centroid assignment + exact MSE: */
rcre_ml_emeans_assign(&X, &C, &L, &inertia, arena);
```

- `X`: `(n, d)` F32 tensor (tabular layout). `C`: `(k, d)` F32.
  `L`: `(n, 1)` I32 (labels; `-1` = unvisited, only possible for
  fractional `nepochs`).
- `rcre_ml_emeans_recipe(&X, k, &p, &rec)` resolves the scaling-law
  batch size, alpha0, epochs, and step schedule for reuse/inspection.
- `arena`: all scratch is arena-backed (labels `4n`, order `4n`,
  per-slab stats ≤ `16·k·d` F32, GEMM Dmat ≤ 16 MB when `k·d > 16384`).
- `pool`: optional `RCRE_ThreadPool`. The slab partition depends only on
  the batch size — results are bit-identical across pool sizes and
  machines (see the threading contract).

## The numerical contract

Three tiers (full text: [`docs/exactness.md`](docs/exactness.md)):

1. **Strict parity, 0 ULP** — the Python oracle's own streams (labels,
   batch sums, counts, Gumbel U's) are injected into the C port; every
   post-update centroid and mass must match the oracle to the last bit
   (~205k assertions, `tests/test_parity.c`).
2. **Statistical parity** — full runs on both sides with native PRNGs;
   C's final MSE within +0.5% of the oracle's (`make quality-par`).
3. **Quality floors** — the paper's two-blob local-minimum trap and a
   non-degeneracy floor vs the single-cluster baseline
   (`tests/test_quality.c`).

Deliberate, documented divergences from the shipped faiss reference:
splitmix64 instead of MT19937 (zero-dependency policy); F64 objective
and F64 finalization pass sums (reference: F32 — the F64 pass removes
the negative-MSE artifact on large-offset data); exact per-row distance
argmin for `k·d ≤ 16384` (reference: always `cc − 2x·c` GEMM form).

## Layout

```
ref/ref_emeans.py       Python oracle (stream recorder + full-run)
ref/compare_parity.py   statistical-parity gate
streams/                generated oracle artifacts (make refs)
tests/                  6 oracle-dependent suites + em_util.h
bench/bench_e_means.c   4krgb / sift100k / embed1m vs Lloyd-25
examples/emeans_demo.c  visual ASCII scatter demo
docs/                   exactness.md, benchmarks.md
# operator + internal header:  ../../src/ml/e_means{,_internal}.h
# core test suite + fixtures:  ../../tests/test_emeans.c,
#                              ../../tests/emeans_fixtures/
```

## Product tie-in

rcvc `-palette:k` (quantize an image to a k-color palette) is the
first consumer: an 8.3M-pixel image → 256-color palette in ~1.1 s on an
18-core pool (exact-kernel regime: k·d = 768 ≤ 16384, no GEMM scratch).
