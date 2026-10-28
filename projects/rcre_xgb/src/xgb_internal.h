/**
 * @file xgb_internal.h
 * @brief Private shared state for RCRE-XGB (not part of the public API).
 */
#ifndef XGB_INTERNAL_H
#define XGB_INTERNAL_H

#include "rcre.h"   /* RCRE public types (RCRE_TreeNode, RCRE_Arena, ...) */
#include "xgb.h"    /* public API (uses the RCRE types above) */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define XGB_MISSING_BIN 255
#define XGB_MAX_CLASSES 16
#define XGB_MAX_BINS 256
#define XGB_BIN_SAMPLE 32768          /* quantile subsample size */
#define XGB_HESS_FLOOR 1e-10f         /* guard for H+lambda denominators */
#define XGB_PROB_EPS 1e-15f           /* sigmoid clamping */

typedef struct XGBDataset {
    size_t n, d;
    float* X;        /* n x d, copied; NaN = missing */
    float* y;        /* n labels */
    uint8_t* bins;   /* n x d; 0..n_bins[f]-1 or XGB_MISSING_BIN */
    uint8_t* n_bins; /* d: effective bin count per feature (1..XGB_MAX_BINS) */
    float* edges;    /* d x (XGB_MAX_BINS-1) quantile edges */
} XGBDataset;

typedef struct {
    const XGBDataset* ds;
    const float* g;              /* per-sample gradient   (n) */
    const float* h;              /* per-sample Hessian     (n) */
    const int32_t* feats; int nfeats;   /* selected columns this tree */
    const XGBConfig* cfg;
    RCRE_Arena* arena;           /* per-tree temp region for row buffers */
    float* hist_g;               /* nfeats * XGB_MAX_BINS scratch */
    float* hist_h;
    int32_t* hist_stamp;         /* lazy-zeroing stamps (same layout) */
    int32_t stamp_cur;
    uint64_t rng;
} FitCtx;

/* Histogram-based greedy split finding + tree construction.
 * rows: int32 row index array, nrows rows. Nodes are written into
 * out (capacity out_cap, index *out_idx advances). Returns 0 on
 * success. */
int xgb_fit_tree(FitCtx* fc, const int32_t* rows, int nrows,
                 XGBNode* out, size_t out_cap, size_t* out_idx);

/* gain of leaf mass (G,H) under L1+L2 regularization (negative leaf
 * objective contribution); used for split gain computation */
static inline float xgb_leaf_obj(float G, float H, float lambda, float alpha) {
    float q = fabsf(G) - alpha;
    if (q <= 0.0f) return 0.0f;
    return -0.5f * q * q / (H + lambda);
}

static inline float xgb_sigmoidf(float x) {
    if (x > 35.0f)  return 1.0f - XGB_PROB_EPS;
    if (x < -35.0f) return XGB_PROB_EPS;
    return 1.0f / (1.0f + expf(-x));
}

static inline float xgb_logf_eps(float x) {
    if (x <= XGB_PROB_EPS) x = XGB_PROB_EPS;
    if (x >= 1.0f - XGB_PROB_EPS) x = 1.0f - XGB_PROB_EPS;
    return logf(x);
}

/* Per-task validation metric (lower is better) */
float xgb_eval_metric(XGBTask task, const float* y, const float* raw,
                      size_t n, int K);

/* Fisher-Yates prefix shuffle (deterministic via SplitMix64 — same
 * algorithm as RCRE's rcre_splitmix64; local copy so every TU sees it) */
static inline uint64_t xgb_rand64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void xgb_perm_prefix(int32_t* perm, int len, int take, uint64_t* rng);

#endif /* XGB_INTERNAL_H */
