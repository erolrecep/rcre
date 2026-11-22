# rcre_e-means — Benchmarks

Measured on the development machine (Apple Silicon, 18-core pool via
`rcre_threadpool_create(0)`). One process, single-arena, F32 data.
`make bench` reproduces the e-means rows; the Lloyd rows use the vendored
RCRE k-means (k-means++ init, 25 iterations, serial).

Conventions: final-MSE = exact nearest-assignment MSE of the finalized
centroids, **per row** (sum of squared distances over dimensions, divided
by n). Both sides (e-means and Lloyd) are scored with the same
`rcre_ml_emeans_assign` pass, so the ratios are apples-to-apples.
moving-MSE = in-flight mean of the per-row true distance to the winner,
computed against the pre-update centroid snapshot (the in-training
objective).

## 4krgb — n = 8,294,400, d = 3, k = 256 (palette regime)

| run | wall | final-MSE (per row) | notes |
|---|---|---|---|
| e-means (serial, T=5) | 13.5 s | 5.766 | recipe B=32768, alpha0=0.4826 (auto, d≤16 → quality law) |
| e-means (pool, T=5) | **1.11 s** | 5.766 | 12.2x pool speedup, bit-identical result |
| Lloyd-25 (k-means++ init) | 35.9 s | 5.658 | |

- Wall: e-means serial is 2.65x faster than Lloyd-25; pooled is **32x**.
- Quality: 1.019x (e-means 1.9% higher MSE) — a tie at this scale;
  with k=256/d=3 every centroid is live (splits=0), so the EMA's
  advantage shows up in wall, not quality, here.
- This is the rcvc `-palette:k` workload class: 8.3M pixels → 256-color
  palette in ~1.1 s on the pool.

## sift100k — n = 100,000, d = 128, k = 4,096 (embedding regime)

| run | wall | final-MSE (per row) | notes |
|---|---|---|---|
| e-means (serial, T=5) | 8.9 s | 18292.7 | recipe B=8192, alpha0=0.2810, splits=10947 |
| e-means (pool, T=5) | **1.16 s** | 18292.7 | 7.6x pool speedup |
| Lloyd-25 (k-means++ init) | 78.1 s | 32301.6 | |

- Wall: e-means serial 8.8x faster; pooled **67x** faster.
- Quality: e-means MSE is **0.566x** Lloyd's (43% lower) — the paper's
  ">= at equal epochs" regime: k=4096 on 100K rows leaves most
  Lloyd/k-means++ centroids dead or on wrong modes, and the EMA +
  revival (10,947 splits) keeps the model dense.
- The GEMM assignment path is active here (k*d = 524,288 > 16,384):
  HAL-dispatched matrix ops, 16 MB Dmat budget, F64 true-distance
  objective.

## Small-scale caveat (documented, measured)

On the 20000×24 quality-parity dataset (k=256, T=5), e-means lands at
~2.4x the MSE of k-means++ + 25 Lloyd iterations (6.16 vs 2.61 per row,
seed 42). At n=20K the scaling-law batch is 2048 (10 updates per pass)
and the k-means++ initialization advantage dominates. The paper's
quality claims are a large-scale statement (n ≥ 100K); `make bench`
above is the regime where both wall and quality hold.

The statistical parity gate (`make quality-par`) compares the C port
against the Python **oracle of the same algorithm**, where the C run
measured 0.589x the oracle's MSE (better, seed 42) — well inside the
+0.5% ceiling.
