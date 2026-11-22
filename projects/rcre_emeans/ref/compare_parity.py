#!/usr/bin/env python3
"""Statistical quality parity: C full run (crun.json) vs oracle (fullrun.json).

The two runs use different RNGs (C: splitmix64 streams; oracle: MT19937),
so trajectories are not comparable — this is a QUALITY gate:

  * final training MSE (per dimension, exact nearest assignment) of the C
    centroids within +0.5% of the oracle's;
  * no dead clusters at the end of either run;
  * cluster-structure sanity: the greedy label agreement after optimally
    (greedy) matching cluster identities between the two solutions is
    reported for the record (two equal-quality solutions may label the
    same partition with different indices — raw label agreement is NOT a
    valid gate).

Exits non-zero on a quality violation.
"""
import json
import sys

import numpy as np


def mse_per_dim(X: np.ndarray, C: np.ndarray, L: np.ndarray) -> float:
    d = X - C[L]
    return float((d.astype(np.float64) ** 2).sum() / (X.shape[0] * X.shape[1]))


def greedy_label_agreement(Lo: np.ndarray, Lc: np.ndarray, k: int) -> float:
    """Map C's cluster ids onto the oracle's via the largest-off-diagonal-
    free greedy matching of the joint cluster matrix, then measure
    agreement. Reference information about structural parity."""
    M = np.zeros((k, k), dtype=np.int64)
    np.add.at(M, (Lo, Lc), 1)
    n = Lo.shape[0]
    used_o = np.zeros(k, dtype=bool)
    used_c = np.zeros(k, dtype=bool)
    agree = 0
    for _ in range(k):
        if used_o.all() or used_c.all():
            break
        Mm = M.copy()
        Mm[np.ix_(used_o, used_c)] = -1
        i, j = divmod(int(np.argmax(Mm)), k)
        if Mm[i, j] <= 0:
            break
        agree += int(Mm[i, j])
        used_o[i] = True
        used_c[j] = True
    return agree / n


def main(outdir="streams"):
    with open(f"{outdir}/fullrun.json") as f:
        orc = json.load(f)
    with open(f"{outdir}/crun.json") as f:
        crc = json.load(f)
    X = np.fromfile(f"{outdir}/X_fullrun.f32", dtype="<f4").reshape(orc["n"], orc["d"])

    Co = np.array(orc["centroids"], dtype=np.float32)
    Cc = np.array(crc["centroids"], dtype=np.float32)
    Lo = np.array(orc["labels"], dtype=np.int64)
    Lc = np.array(crc["labels"], dtype=np.int64)
    n, k = orc["n"], orc["k"]

    mse_o = mse_per_dim(X, Co, Lo)
    mse_c = mse_per_dim(X, Cc, Lc)
    dead_o = int((Lo < 0).sum())
    dead_c = int((Lc < 0).sum())
    agree = greedy_label_agreement(Lo, Lc, k)
    raw = float((Lo == Lc).mean())

    print(f"  oracle : MSE/dim={mse_o:.6f} splits={orc['n_splits']} dead={dead_o}")
    print(f"  C port : MSE/dim={mse_c:.6f} splits={crc['n_splits']} dead={dead_c}")
    print(f"  greedy cluster-aligned agreement: {100 * agree:.3f}% "
          f"(raw index agreement {100 * raw:.3f}%, meaningless without "
          f"cluster matching)")
    print(f"  MSE ratio C/oracle: {mse_c / mse_o:.5f}  target <= 1.005")

    ok = True
    if mse_c > 1.005 * mse_o:
        print("  FAIL: C MSE above +0.5% of oracle")
        ok = False
    if dead_c > 0:
        print(f"  FAIL: C run has {dead_c} unvisited rows")
        ok = False
    print("  PARITY_OK" if ok else "  PARITY_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "streams"))
