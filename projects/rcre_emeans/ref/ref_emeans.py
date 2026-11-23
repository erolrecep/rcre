#!/usr/bin/env python3
"""e-means NumPy oracle for the rcre_e-means project.

Ground truth for the C implementation, with two roles:

1. STRICT state machine. Given X, initial centroids, a MANUAL recipe
   (batch B, peak step alpha0, epochs T, revival threshold tau), runs the
   documented FP32 operation order (identical to src/e_means.c) and emits,
   per update: (labels, sums, counts, objective) — the injected streams —
   plus the post-update model state (c, N) and the revival U draws. The C
   test (tests/test_parity.c) replays these streams through the C state
   machine and must reproduce the model states bit-for-bit (0 ULP).

2. FULL RUN. The complete algorithm (MT19937 shuffles and random-row init
   like the real NumpyEmeans, auto recipe, auto tau) used for the
   STATISTICAL quality parity gate (tests/test_quality.c + compare_parity.py)
   against the C full run (splitmix64 streams).

Operation-order contract (mirrors src/e_means.c — the parity contract):
  * batch sums s_j: FP32, unbuffered, rows in processing order;
  * assignment: argmin of (cc - 2 * x @ c.T) in FP32, first index on ties,
    cc = per-row FP32 dot of c (numpy pairwise; the C side uses the HAL
    dot — a few-ULP difference that cannot affect the strict gate, which
    injects the oracle's labels);
  * EMA: N *= decay (all k, FP32), then per touched j: t = alpha*n,
    mass = N + t, mean = s/n, w = t/mass, c += w*(mean - c), c = mean when
    w == 1 — all FP32 scalar math, no FMA;
  * alpha(t) in double (Eq. 5), cast to FP32 once at use;
  * revival: total = sequential FP32 sum, thr = (total*tau)/k in FP32;
    Gumbel U vectors drawn for ALL k per check (MT19937 like the real
    reference); key = float32_log(N) - log(-log(max(U, tiny))); top-nsplit
    by (key desc, index asc); child written first, then donor; delta
    pattern [1,-1,...]*1e-4; nextafter on coord 0 on bit-collision;
  * finalization: FP64 pass sums/counts (the C port's documented
    improvement over the reference's FP32 pass sums — the oracle mirrors
    the C port here so the final gate is 0 ULP too), final division FP32.

Outputs (per case, in --out dir):
  <case>.json   human-readable index + recipe + per-update row counts
  <case>.h      C metadata header (consumed by tests/test_parity.c)
  <case>.bin    payload: [init c f32 k*d] then per update:
                [m u32][labels i32 m][sums f32 k*d][counts f32 k]
                then post-update states: [c f32 k*d][N f32 k]
                then revival U checks in order: [U f64 k]
                then final: [c f32 k*d][N f32 k][pass_sums f64 k*d]
                [pass_counts f64 k]
  X_<case>.f32  the input matrix (n*d f32), shared by oracle and C tests

Plus:
  recipes.json / recipes.h   scaling-law expectations for the recipe grid
  fullrun.json               full-run results (statistical parity, oracle)
"""
import argparse
import json
import math
import os
import struct

import numpy as np

# --- shipped reference constants (faiss/contrib/e_means/base.py) -----------
BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}
ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)
ALPHA_MIN_FACTOR = 0.1
CHECK_EVERY = 5
INIT_USAGE = 2.0 ** -16
SPLIT_EPSILON = 1e-4
UNUSED_TAU = 0.1
NO_REVIVAL_MAX_D = 2
AUTO_QUALITY_MAX_D = 16
DBL_TINY = 2.2250738585072014e-308


def round_pow2(v: float) -> int:
    if v <= 1:
        return 1
    lo = 2 ** int(math.floor(math.log2(v)))
    hi = 2 * lo
    return lo if v / lo < hi / v else hi


def scheduled_alpha(alpha0: float, t: int, total: int) -> float:
    f = min(t, total) / max(1, total)
    cosine = 0.5 * (1.0 + math.cos(math.pi * f))
    return alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)


def epoch_remainder(nepochs: float, n: int) -> int:
    whole = math.floor(nepochs)
    v = (nepochs - whole) * n
    f = math.floor(v)
    diff = v - f
    if diff > 0.5:
        return int(f + 1)
    if diff < 0.5:
        return int(f)
    return int(f) if int(f) % 2 == 0 else int(f) + 1


def epoch_rows(n: int, nepochs: float):
    whole = math.floor(nepochs)
    rem = epoch_remainder(nepochs, n)
    if whole == 0:
        return [rem]
    rows = [n] * (whole - 1)
    if rem:
        rows.append(rem)
    rows.append(n)
    return rows


def resolve_recipe(n: int, d: int, k: int, nepochs: float, law: str,
                   batch=None, alpha0=None):
    """Mirror of em_resolve_recipe (shipped-code constants)."""
    if batch is None:
        if law == "auto":
            law = "quality" if d <= AUTO_QUALITY_MAX_D else "throughput"
        cb, bk, bn, bt = BATCH_LAWS[law]
        raw = cb * k ** bk * n ** bn * nepochs ** bt
        B = round_pow2(raw)
        if law == "throughput":
            B = min(B, round_pow2(n / 2))
        else:
            B = min(B, n // 2)
        resolved = law
    else:
        B = min(int(batch), n)
        resolved = "manual"
    B = max(1, min(B, n))
    if alpha0 is None:
        ca, ak, aB, an, aT = ALPHA_LAW
        a0 = ca * k ** ak * B ** aB * n ** an * nepochs ** aT
        a0 = min(max(a0, 1e-4), 1.0)
    else:
        a0 = float(alpha0)
    rows = epoch_rows(n, nepochs)
    steps = [ (r + B - 1) // B for r in rows ]
    tau = 0.0 if d <= NO_REVIVAL_MAX_D else UNUSED_TAU
    return {
        "batch_size": B, "alpha0": float(np.float32(a0)), "alpha0_full": a0,
        "nepochs": float(nepochs), "epoch_rows": rows,
        "n_epochs": len(rows), "steps_per_pass": (n + B - 1) // B,
        "total_steps": sum(steps), "unused_threshold": tau,
        "batch_law": resolved,
    }


def assign(Xb: np.ndarray, c: np.ndarray):
    """First-argmin over the TRUE squared distances, EXACT sequential FP32
    per row (index order, diff*diff, mul+add, no FMA) — bit-identical to
    the C exact kernel and semantically the reference's faiss.knn L2.
    Vectorized over rows: each op is elementwise across i, so the per-row
    accumulation order is exactly dd = 0..d-1. Returns (labels, dists)."""
    m = Xb.shape[0]
    k = c.shape[0]
    d = c.shape[1]
    labels = np.empty(m, dtype=np.int64)
    dists = np.empty(m, dtype=np.float32)
    for j in range(k):
        acc = None
        for dd in range(d):
            df = Xb[:, dd] - c[j, dd]          # exact elementwise F32
            term = df * df                     # exact elementwise F32
            acc = term if acc is None else acc + term
        if j == 0:
            labels[:] = 0
            dists[:] = acc
        else:
            better = acc < dists               # first-argmin on ties
            labels[better] = j
            dists[better] = acc[better]
    return labels, dists


def ema_update(c: np.ndarray, N: np.ndarray, sums: np.ndarray,
               counts: np.ndarray, alpha_d: float):
    """Reference FP32 operation order (see module docstring)."""
    alpha = np.float32(alpha_d)
    decay = np.float32(1.0 - alpha_d)
    replace = (alpha_d == 1.0)
    if not replace:
        N *= decay
    touched = counts > 0
    j = np.where(touched)[0]
    for jj in j:
        n_j = counts[jj]
        t = alpha * n_j
        mass = N[jj] + t
        weight = t / mass
        mean = sums[jj] / n_j
        pos = c[jj] + weight * (mean - c[jj])
        if weight == 1.0:
            pos = mean
        c[jj] = pos
        N[jj] = mass


def revive(c, N, k, tau, rng_u, U):
    """Appendix H.1. Returns (n_splits, key_margin). U is the k-vector of
    this check (already drawn for ALL k)."""
    total = np.float32(0.0)
    for jj in range(k):
        total = total + N[jj]
    thr = (total * np.float32(tau)) / np.float32(k)
    targets = [jj for jj in range(k) if N[jj] < thr]
    donors = [jj for jj in range(k)
              if not (N[jj] < thr) and N[jj] > 0.0]
    nsplit = min(len(targets), len(donors))
    if nsplit == 0:
        return 0, 0.0
    keys = {}
    for dj in donors:
        u = max(U[dj], DBL_TINY)
        ln_n = float(np.log(np.float32(N[dj])))  # FP32 log, promote
        keys[dj] = ln_n - math.log(-math.log(u))
    order = sorted(donors, key=lambda j: (-keys[j], j))
    src = order[:nsplit]
    # Margin to the first not-picked donor (only meaningful when there is
    # one); guards the parity test against near-tie donor selection.
    margin = float("inf")
    if len(order) > nsplit:
        margin = min(abs(keys[src[i]] - keys[order[nsplit]])
                     for i in range(nsplit))
    d = c.shape[1]
    # Reference _apply_splits: sign = [1,-1,1,-1,...] over DIMENSIONS.
    delta = np.where(np.arange(d) % 2 == 0, SPLIT_EPSILON,
                     -SPLIT_EPSILON).astype(np.float32)
    for i in range(nsplit):
        dj, tj = src[i], targets[i]
        cd = c[dj].copy()
        c[tj] = (cd * (1.0 + delta)).astype(np.float32)
        c[dj] = (cd * (1.0 - delta)).astype(np.float32)
        if np.array_equal(c[dj], c[tj]):
            v = c[tj][0]
            nb = np.nextafter(v, np.float32(np.inf))
            if not np.isfinite(nb):
                nb = np.nextafter(v, np.float32(-np.inf))
            c[tj][0] = nb
        half = N[dj] / 2.0
        N[dj] = half
        N[tj] = half
    return nsplit, margin


def run_state_machine(X, init_c, n, d, k, B, alpha0_d, nepochs, tau,
                      seed, shuffle, record_U):
    """Full trajectory with the documented operation order.

    Returns dict with per-update streams, states, U checks and the final
    (post-finalization) model.
    """
    c = np.array(init_c, dtype=np.float32, copy=True)
    N = np.full(k, np.float32(INIT_USAGE), dtype=np.float32)
    rows_all = epoch_rows(n, nepochs)
    M = sum((r + B - 1) // B for r in rows_all)

    updates = []      # per update: (m, labels, sums, counts, objective)
    states = []       # post-update: (c, N)
    U_checks = []     # per revival check: U vector
    pass_sums = None
    pass_counts = None
    u_pos = 0
    t = 0
    n_splits = 0
    min_margin = float("inf")
    rng = np.random.RandomState(seed)  # MT19937 for U (like the reference)
    for e, rows in enumerate(rows_all):
        last = (e == len(rows_all) - 1)
        order = (np.random.RandomState((seed * 1000003 + e) % 2**32)
                 .permutation(n)) if shuffle else np.arange(n)
        if last:
            pass_sums = np.zeros((k, d), dtype=np.float64)
            pass_counts = np.zeros(k, dtype=np.float64)
        cursor = 0
        while cursor < rows:
            m = min(B, rows - cursor)
            idx = order[cursor:cursor + m]
            Xb = X[idx]
            labels, dists = assign(Xb, c)
            sums = np.zeros((k, d), dtype=np.float32)
            counts = np.zeros(k, dtype=np.float32)
            # unbuffered row-order FP32 accumulation (np.add.at semantics)
            for r in range(m):
                jj = int(labels[r])
                sums[jj] = sums[jj] + Xb[r].astype(np.float32)
                counts[jj] = counts[jj] + 1.0
            objective = 0.0
            for r in range(m):
                objective += float(dists[r])  # sequential FP64, row order
            if last:
                pass_counts += counts.astype(np.float64)
                pass_sums += sums.astype(np.float64)
            alpha_d = scheduled_alpha(alpha0_d, t, M)
            ema_update(c, N, sums, counts, alpha_d)
            if (t + 1) % CHECK_EVERY == 0 and tau > 0:
                U = rng.random_sample(k).astype(np.float64)
                ns, margin = revive(c, N, k, tau, rng, U)
                n_splits += ns
                if ns:
                    min_margin = min(min_margin, margin)
                if record_U:
                    U_checks.append(U)
            updates.append((m, labels.copy(), sums.copy(), counts.copy(),
                            objective))
            states.append((c.copy(), N.copy()))
            t += 1
            cursor += m
    # finalization (FP64 pass sums — the C port's contract)
    final_c = c.copy()
    for jj in range(k):
        if pass_counts[jj] > 0.0:
            cnt = np.float32(pass_counts[jj])
            final_c[jj] = (pass_sums[jj].astype(np.float32)) / cnt
    return {
        "updates": updates, "states": states, "U_checks": U_checks,
        "final_c": final_c, "final_N": N.copy(),
        "pass_sums": pass_sums if pass_sums is not None
                     else np.zeros((k, d), dtype=np.float64),
        "pass_counts": pass_counts if pass_counts is not None
                       else np.zeros(k, dtype=np.float64),
        "n_splits": n_splits, "min_key_margin": min_margin,
        "M": M,
    }


# --- data generators -------------------------------------------------------

def make_blobs(n: int, d: int, n_blobs: int, seed: int):
    """n FP32 points around n_blobs Gaussian centers (unequal weights)."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_blobs, d).astype(np.float32) * 6.0
    w = rng.dirichlet(np.ones(n_blobs) * 0.35)
    assign_blob = rng.choice(n_blobs, size=n, p=w)
    X = (centers[assign_blob]
         + rng.randn(n, d).astype(np.float32) * 0.35).astype(np.float32)
    return X


# --- emitters ---------------------------------------------------------------

def write_case(outdir, name, X, init_c, n, d, k, B, alpha0_d, nepochs, tau,
               seed, shuffle, recipe):
    res = run_state_machine(X, init_c, n, d, k, B, alpha0_d, nepochs, tau,
                            seed, shuffle, record_U=(tau > 0))
    xpath = os.path.join(outdir, f"X_{name}.f32")
    X.tofile(xpath)

    # payload
    with open(os.path.join(outdir, f"{name}.bin"), "wb") as f:
        f.write(init_c.astype(np.float32).tobytes())
        for (m, labels, sums, counts, _obj) in res["updates"]:
            f.write(struct.pack("<I", m))
            f.write(labels.astype("<i4").tobytes())
            f.write(sums.astype("<f4").tobytes())
            f.write(counts.astype("<f4").tobytes())
        for (cst, Nst) in res["states"]:
            f.write(cst.astype("<f4").tobytes())
            f.write(Nst.astype("<f4").tobytes())
        for U in res["U_checks"]:
            f.write(U.astype("<f8").tobytes())
        f.write(res["final_c"].astype("<f4").tobytes())
        f.write(res["final_N"].astype("<f4").tobytes())
        f.write(res["pass_sums"].astype("<f8").tobytes())
        f.write(res["pass_counts"].astype("<f8").tobytes())

    # human-readable index
    meta = {
        "name": name, "n": n, "d": d, "k": k, "B": B,
        "alpha0": float(np.float32(alpha0_d)), "alpha0_full": alpha0_d,
        "T": nepochs, "tau": tau, "seed": seed, "shuffle": shuffle,
        "M": res["M"], "n_updates": len(res["updates"]),
        "update_rows": [m for (m, *_r) in res["updates"]],
        "n_U_checks": len(res["U_checks"]),
        "n_splits": res["n_splits"],
        "min_key_margin": res["min_key_margin"],
        "recipe": {kk: (vv.tolist() if isinstance(vv, list)
                        else (float(vv) if isinstance(vv, np.floating)
                              else (int(vv) if isinstance(vv, np.integer) else vv)))
                   for kk, vv in recipe.items()
                   if kk != "epoch_rows"} | {"epoch_rows": [int(x) for x in recipe["epoch_rows"]]},
    }
    with open(os.path.join(outdir, f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # C metadata header
    with open(os.path.join(outdir, f"{name}.h"), "w") as f:
        f.write(f"/* generated by ref/ref_emeans.py — do not edit */\n")
        f.write(f"#ifndef EM_STREAM_{name.upper()}_H\n")
        f.write(f"#define EM_STREAM_{name.upper()}_H\n")
        f.write("#ifndef EM_STREAM_META_DEFINED\n")
        f.write("#define EM_STREAM_META_DEFINED\n")
        f.write("struct em_stream_meta {\n")
        f.write(f"    size_t n, d, k, B;\n")
        f.write(f"    float T;\n")
        f.write(f"    double alpha0_full;\n")
        f.write(f"    float tau;\n")
        f.write(f"    uint64_t seed;\n")
        f.write(f"    int shuffle;\n")
        f.write(f"    size_t M, n_updates;\n")
        f.write(f"    size_t n_U_checks;\n")
        f.write(f"    size_t update_rows[4096];\n")
        f.write(f"}};\n")
        f.write("#endif\n")
        f.write(f"static const struct em_stream_meta EM_STREAM_{name.upper()} = {{\n")
        f.write(f"    {n}, {d}, {k}, {B},\n")
        f.write(f"    {nepochs!r}f,\n")
        f.write(f"    {alpha0_d!r},\n")
        f.write(f"    {tau!r}f,\n")
        f.write(f"    {seed}ULL,\n")
        f.write(f"    {1 if shuffle else 0},\n")
        f.write(f"    {res['M']}, {len(res['updates'])},\n")
        f.write(f"    {len(res['U_checks'])},\n")
        rows_c = ", ".join(str(m) for m in meta["update_rows"])
        f.write(f"    {{ {rows_c} }}\n")
        f.write("};\n")
        f.write("#endif\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="streams")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------
    # STRICT cases (manual recipe, provided init, shuffle off).
    # ------------------------------------------------------------------
    # Manual alpha0 is an FP32 API parameter: the schedule runs on the
    # promoted value F64(F32(alpha0)), exactly like the C port.
    X1 = make_blobs(2000, 8, 12, seed=11)
    write_case(args.out, "strict_norev", X1, X1[:32], 2000, 8, 32,
               B=256, alpha0_d=float(np.float32(0.5)), nepochs=3.0, tau=0.0,
               seed=77, shuffle=False,
               recipe=resolve_recipe(2000, 8, 32, 3.0, "auto",
                                     batch=256, alpha0=float(np.float32(0.5))))

    X2 = make_blobs(4000, 16, 30, seed=22)
    meta2 = write_case(args.out, "strict_revive", X2, X2[:128], 4000, 16, 128,
                       B=512, alpha0_d=float(np.float32(0.4)), nepochs=2.0,
                       tau=0.1, seed=123, shuffle=False,
                       recipe=resolve_recipe(4000, 16, 128, 2.0, "auto",
                                             batch=512, alpha0=float(np.float32(0.4))))
    if meta2["n_splits"] == 0:
        raise SystemExit("strict_revive fixture produced no splits — "
                         "adjust the blob count/k ratio")
    if meta2["min_key_margin"] <= 1e-9:
        print(f"WARNING: strict_revive key margin {meta2['min_key_margin']:.3g}")

    # ------------------------------------------------------------------
    # FULL RUN (statistical parity vs the C full run; MT19937 like the
    # real NumpyEmeans, auto recipe, auto tau).
    # ------------------------------------------------------------------
    n, d, k, T, seed = 20000, 24, 256, 5.0, 42
    Xf = make_blobs(n, d, 96, seed=seed)
    Xf.tofile(os.path.join(args.out, "X_fullrun.f32"))
    rec = resolve_recipe(n, d, k, T, "auto")
    B, a0 = rec["batch_size"], rec["alpha0_full"]
    tau = rec["unused_threshold"]
    # MT19937 random-row init (reference's RandomState(seed).permutation(n)[:k])
    init_idx = np.random.RandomState(seed).permutation(n)[:k]
    res = run_state_machine(Xf, Xf[init_idx], n, d, k, B, a0, T, tau,
                            seed, shuffle=True, record_U=False)
    labels = np.full(n, -1, dtype=np.int64)
    order_last = (np.random.RandomState((seed * 1000003 + rec["n_epochs"] - 1)
                                        % 2**32).permutation(n))
    # Full-assignment labels of the final centroids (the comparison
    # contract: both sides report nearest-final-centroid labels, which
    # costs each side one extra pass — see compare_parity.py).
    for b0 in range(0, n, 4096):
        lab, _ = assign(Xf[b0:b0 + 4096], res["final_c"])
        labels[b0:b0 + 4096] = lab
    with open(os.path.join(args.out, "fullrun.json"), "w") as f:
        json.dump({
            "n": n, "d": d, "k": k, "T": T, "seed": seed,
            "B": B, "alpha0_full": a0, "tau": tau,
            "recipe": {kk: (float(vv) if isinstance(vv, np.floating)
                            else (int(vv) if isinstance(vv, np.integer) else vv))
                       for kk, vv in rec.items()},
            "centroids": res["final_c"].astype(np.float32).tolist(),
            "labels": labels.tolist(),
            "n_splits": res["n_splits"],
        }, f)
    print("fullrun: n=%d d=%d k=%d T=%s B=%d alpha0=%.4f tau=%s splits=%d"
          % (n, d, k, T, B, a0, tau, res["n_splits"]))

    # ------------------------------------------------------------------
    # Recipe grid (scaling-law expectations).
    # ------------------------------------------------------------------
    grid = [
        (1_000_000, 96, 4096, 5.0),
        (8_294_400, 3, 256, 5.0),
        (100_000, 128, 16384, 5.0),
        (20_000, 8, 50, 2.5),
        (1_000, 4, 8, 0.5),
        (20_000, 24, 256, 5.0),
        (1_000_000, 128, 1024, 3.0),
        (50_000, 16, 128, 5.0),
    ]
    rows = []
    for (gn, gd, gk, gT) in grid:
        r = resolve_recipe(gn, gd, gk, gT, "auto")
        rows.append({"n": gn, "d": gd, "k": gk, "T": gT, **r})
    with open(os.path.join(args.out, "recipes.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(args.out, "recipes.h"), "w") as f:
        f.write("/* generated by ref/ref_emeans.py — do not edit */\n")
        f.write("#ifndef EM_RECIPES_H\n#define EM_RECIPES_H\n")
        f.write("struct em_expect_recipe {\n")
        f.write("    size_t n, d, k;\n    float T;\n")
        f.write("    size_t B; float alpha0_f32; double alpha0_f64;\n")
        f.write("    size_t M; size_t n_epochs; size_t rows[16];\n")
        f.write("    float tau; int batch_law; /* -1 manual,0 auto,1 thr,2 qual */\n")
        f.write("};\n")
        f.write("static const struct em_expect_recipe EM_RECIPES[] = {\n")
        for r in rows:
            law = {"manual": -1, "throughput": 1, "quality": 2}[r["batch_law"]]
            rows_c = ", ".join(str(x) for x in r["epoch_rows"])
            f.write("    { %d, %d, %d, %.17g, %d, %.7g, %.17g, %d, %d, "
                    "{ %s }, %.7g, %d },\n"
                    % (r["n"], r["d"], r["k"], r["T"], r["batch_size"],
                       float(r["alpha0"]), r["alpha0_full"], r["total_steps"],
                       r["n_epochs"], rows_c, r["unused_threshold"], law))
        f.write("};\n")
        f.write("#define EM_N_RECIPES %d\n" % len(rows))
        f.write("#endif\n")

    print("oracle artifacts written to", args.out)


if __name__ == "__main__":
    main()
