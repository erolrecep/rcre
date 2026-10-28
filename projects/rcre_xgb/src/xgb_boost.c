/**
 * @file xgb_boost.c
 * @brief Second-order boosting loop: gradients, subsampling, early stopping.
 *
 * Each round grows one tree per class by functional Newton steps:
 *   w* = -G / (H + lambda)   (soft-thresholded by alpha for the L1 term)
 * and adds eta * tree to the current function. With a validation set,
 * the model is truncated to the best round (XGBoost early-stopping
 * semantics: stop after `early_stop_rounds` without metric improvement).
 */
#include "xgb_internal.h"

/* ========================================================================= */
/* Config                                                                    */
/* ========================================================================= */

void xgb_config_default(XGBConfig* cfg) {
    cfg->task = XGB_TASK_BINARY;
    cfg->num_classes = 1;
    cfg->rounds = 100;
    cfg->eta = 0.3f;
    cfg->max_depth = 6;
    cfg->min_child_weight = 1.0f;
    cfg->gamma = 0.0f;
    cfg->lambda = 1.0f;
    cfg->alpha = 0.0f;
    cfg->subsample = 1.0f;
    cfg->colsample = 1.0f;
    cfg->max_bins = 256;
    cfg->early_stop_rounds = 0;
    cfg->seed = 42;
    cfg->base_score = 0.0f;
}

/* ========================================================================= */
/* Gradients                                                                 */
/* ========================================================================= */

/* raw: n logits/scores. g,h: n outputs. */
static void grads_reg(const float* F, const float* y, size_t n,
                      float* g, float* h) {
    for (size_t i = 0; i < n; i++) {
        g[i] = F[i] - y[i];
        h[i] = 1.0f;
    }
}

static void grads_bin(const float* F, const float* y, size_t n,
                      float* g, float* h) {
    for (size_t i = 0; i < n; i++) {
        float p = xgb_sigmoidf(F[i]);
        g[i] = p - y[i];
        h[i] = p * (1.0f - p);
        if (h[i] < 0.0f) h[i] = 0.0f;
    }
}

/* Multi-class class k: softmax diagonal gradient/Hessian */
static void grads_multi_k(const float* F, const float* y, size_t n, int K, int k,
                          float* g, float* h) {
    for (size_t i = 0; i < n; i++) {
        const float* r = &F[i * K];
        float m = r[0];
        for (int c = 1; c < K; c++) m = m > r[c] ? m : r[c];
        float s = 0.0f;
        for (int c = 0; c < K; c++) s += expf(r[c] - m);
        float p = expf(r[k] - m) / s;
        g[i] = p - (y[i] == (float)k ? 1.0f : 0.0f);
        h[i] = p * (1.0f - p);
        if (h[i] < 0.0f) h[i] = 0.0f;
    }
}

/* ========================================================================= */
/* Validation metric (lower is better)                                       */
/* ========================================================================= */

float xgb_eval_metric(XGBTask task, const float* y, const float* raw,
                      size_t n, int K) {
    if (task == XGB_TASK_REGRESSION) {
        double s = 0.0;
        for (size_t i = 0; i < n; i++) {
            float e = raw[i] - y[i];
            s += (double)e * e;
        }
        return sqrtf((float)(s / (double)n));
    }
    if (task == XGB_TASK_BINARY) {
        double s = 0.0;
        for (size_t i = 0; i < n; i++) {
            float p = xgb_sigmoidf(raw[i]);
            s -= y[i] * xgb_logf_eps(p) + (1.0f - y[i]) * xgb_logf_eps(1.0f - p);
        }
        return (float)(s / (double)n);
    }
    /* MULTI: mean -log p_{y_i} over the softmax */
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        const float* r = &raw[i * K];
        float m = r[0];
        for (int c = 1; c < K; c++) m = m > r[c] ? m : r[c];
        float best = expf(r[(int)y[i]] - m);
        float ssum = 0.0f;
        for (int c = 0; c < K; c++) ssum += expf(r[c] - m);
        s -= xgb_logf_eps(best / ssum);
    }
    return (float)(s / (double)n);
}

/* ========================================================================= */
/* Tree evaluation (row -> leaf value)                                       */
/* ========================================================================= */

static inline float tree_eval(const XGBNode* nodes, uint32_t root,
                              const float* row, size_t max_steps) {
    const XGBNode* nd = &nodes[root];
    size_t steps = 0;
    while (!nd->is_leaf) {
        if (++steps > max_steps) return 0.0f;  /* cycle guard */
        float x = row[nd->feature];
        bool go_left = (x != x) ? nd->default_left : (x <= nd->threshold);
        nd = &nodes[go_left ? nd->left : nd->right];
    }
    return nd->leaf_value;
}

/* ========================================================================= */
/* Training                                                                  */
/* ========================================================================= */

int xgb_train(const XGBDataset* train, const XGBDataset* val,
              const XGBConfig* cfg0, XGBModel* model,
              RCRE_Arena* arena) {
    if (!train || !model || !arena || !cfg0) return -1;
    XGBConfig cfg = *cfg0;

    size_t n = train->n, d = train->d;
    if (n < 2 || d < 1) return -1;
    if (cfg.rounds < 1) cfg.rounds = 1;
    if (cfg.max_depth < 1) cfg.max_depth = 1;
    if (cfg.max_bins < 2) cfg.max_bins = 2;
    if (cfg.max_bins > XGB_MAX_BINS) cfg.max_bins = XGB_MAX_BINS;
    if (cfg.eta <= 0.0f) cfg.eta = 0.3f;
    if (cfg.subsample <= 0.0f) cfg.subsample = 1.0f;
    if (cfg.subsample > 1.0f) cfg.subsample = 1.0f;
    if (cfg.colsample <= 0.0f) cfg.colsample = 1.0f;
    if (cfg.colsample > 1.0f) cfg.colsample = 1.0f;
    int K = (cfg.task == XGB_TASK_MULTI) ? cfg.num_classes : 1;
    if (K < 1) K = 1;
    if (K > XGB_MAX_CLASSES) K = XGB_MAX_CLASSES;
    if (cfg.task == XGB_TASK_MULTI && K < 2) return -1;

    /* ---------------- model skeleton ---------------- */
    memset(model, 0, sizeof(*model));
    model->task = cfg.task;
    model->num_classes = K;
    model->dim = (int)d;
    model->eta = cfg.eta;
    model->best_iteration = -1;

    int total_trees = cfg.rounds * K;
    model->tree_roots = (uint32_t*)rcre_arena_alloc(arena, total_trees * sizeof(uint32_t), 4);
    if (!model->tree_roots) return -1;
    /* tree_end[t] = node offset just past tree t (grows with the node buf) */
    uint32_t* tree_end = (uint32_t*)rcre_arena_alloc(arena, total_trees * sizeof(uint32_t), 4);
    if (!tree_end) return -1;

    /* base scores */
    if (cfg.task == XGB_TASK_REGRESSION) {
        double m = 0.0;
        for (size_t i = 0; i < n; i++) m += train->y[i];
        model->base_score = (cfg.base_score != 0.0f)
            ? cfg.base_score : (float)(m / (double)n);
    } else if (cfg.task == XGB_TASK_BINARY) {
        double m = 0.0;
        for (size_t i = 0; i < n; i++) m += train->y[i];
        float p0 = (float)(m / (double)n);
        if (p0 < 1e-6f) p0 = 1e-6f;
        if (p0 > 1.0f - 1e-6f) p0 = 1.0f - 1e-6f;
        model->base_score = (cfg.base_score != 0.0f)
            ? cfg.base_score : logf(p0 / (1.0f - p0));
    } else {
        double cnt[XGB_MAX_CLASSES] = {0};
        for (size_t i = 0; i < n; i++)
            if ((int)train->y[i] >= 0 && (int)train->y[i] < XGB_MAX_CLASSES)
                cnt[(int)train->y[i]] += 1.0;
        for (int k = 0; k < K; k++)
            model->base_scores[k] = logf((float)(cnt[k] / (double)n) + 1e-12f);
        model->base_score = 0.0f;
    }

    /* current function values F (n x K) */
    float* F = (float*)rcre_arena_alloc(arena, n * K * sizeof(float), 16);
    float* g = (float*)rcre_arena_alloc(arena, n * sizeof(float), 16);
    float* h = (float*)rcre_arena_alloc(arena, n * sizeof(float), 16);
    if (!F || !g || !h) return -1;
    for (size_t i = 0; i < n; i++)
        for (int k = 0; k < K; k++)
            F[i * K + k] = (cfg.task == XGB_TASK_MULTI) ? model->base_scores[k] : model->base_score;

    float* Fv = NULL;
    if (val) {
        Fv = (float*)rcre_arena_alloc(arena, val->n * K * sizeof(float), 16);
        if (!Fv) return -1;
        for (size_t i = 0; i < val->n; i++)
            for (int k = 0; k < K; k++)
                Fv[i * K + k] = (cfg.task == XGB_TASK_MULTI) ? model->base_scores[k] : model->base_score;
    }
    if (val) {
        model->val_curve = (float*)rcre_arena_alloc(arena, cfg.rounds * sizeof(float), 4);
        if (!model->val_curve) return -1;
    }

    /* persistent row / column permutations */
    int32_t* all_rows = (int32_t*)rcre_arena_alloc(arena, n * sizeof(int32_t), 16);
    int32_t* row_perm = (int32_t*)rcre_arena_alloc(arena, n * sizeof(int32_t), 16);
    int32_t* col_perm = (int32_t*)rcre_arena_alloc(arena, d * sizeof(int32_t), 16);
    if (!all_rows || !row_perm || !col_perm) return -1;
    for (size_t i = 0; i < n; i++) all_rows[i] = (int32_t)i;
    for (size_t i = 0; i < n; i++) row_perm[i] = (int32_t)i;
    for (size_t f = 0; f < d; f++) col_perm[f] = (int32_t)f;

    /* histogram scratch (max width = d features) */
    float* hist_g = (float*)rcre_arena_alloc(arena, d * XGB_MAX_BINS * sizeof(float), 16);
    float* hist_h = (float*)rcre_arena_alloc(arena, d * XGB_MAX_BINS * sizeof(float), 16);
    int32_t* hist_stamp = (int32_t*)rcre_arena_alloc(arena, d * XGB_MAX_BINS * sizeof(int32_t), 16);
    if (!hist_g || !hist_h || !hist_stamp) return -1;
    memset(hist_stamp, 0, d * XGB_MAX_BINS * sizeof(int32_t));

    uint64_t rng = cfg.seed ? cfg.seed : 1;
    int c = (int)lroundf(cfg.colsample * (float)d);
    if (c < 1) c = 1;
    if (c > (int)d) c = (int)d;
    int n_sub = (int)lroundf(cfg.subsample * (float)n);
    if (n_sub < 1) n_sub = 1;
    if (n_sub > (int)n) n_sub = (int)n;

    /* growing contiguous node storage (arena copy-grow) */
    XGBNode* nodes = NULL;
    size_t node_cap = 0, node_off = 0;

    int best_round = -1;
    float best_metric = 0.0f;
    int rounds_done = 0;

    /* per-tree node scratch: PERSISTENT (allocated once, reused by every
     * tree). Only the fit row-buffers live in temp scopes, so the scratch
     * survives the per-fit rewinds. */
    size_t tree_node_cap = (size_t)n_sub * 2 + 1;
    XGBNode* tnodes = (XGBNode*)rcre_arena_alloc(arena, tree_node_cap * sizeof(XGBNode), 16);
    if (!tnodes) return -1;

    for (int r = 0; r < cfg.rounds; r++) {
        rounds_done = r + 1;

        /* column subset (per round) */
        xgb_perm_prefix(col_perm, (int)d, c, &rng);

        /* row subset: drawn once per round, shared by all class trees */
        const int32_t* rows;
        int nrows;
        if (n_sub == (int)n) {
            rows = all_rows;
            nrows = (int)n;
        } else {
            xgb_perm_prefix(row_perm, (int)n, n_sub, &rng);
            rows = row_perm;
            nrows = n_sub;
        }

        for (int k = 0; k < K; k++) {
            /* gradients for this class */
            if (cfg.task == XGB_TASK_REGRESSION)      grads_reg(&F[k], train->y, n, g, h);
            else if (cfg.task == XGB_TASK_BINARY)     grads_bin(&F[k], train->y, n, g, h);
            else                                      grads_multi_k(F, train->y, n, K, k, g, h);

            FitCtx fc;
            fc.ds = train;
            fc.g = g; fc.h = h;
            fc.feats = col_perm; fc.nfeats = c;
            fc.cfg = &cfg;
            fc.arena = arena;
            fc.hist_g = hist_g; fc.hist_h = hist_h;
            fc.hist_stamp = hist_stamp; fc.stamp_cur = 0;
            fc.rng = rng;

            RCRE_ArenaTemp tt = rcre_arena_temp_begin(arena);
            size_t idx = 0;
            if (xgb_fit_tree(&fc, rows, nrows, tnodes, tree_node_cap, &idx) != 0) {
                rcre_arena_temp_end(tt);
                return -1;
            }

            /* grow the function BEFORE rebasing: tnodes still holds the
             * tree-local child indices (root = 0). */
            for (size_t i = 0; i < n; i++)
                F[i * K + k] += cfg.eta * tree_eval(tnodes, 0, &train->X[i * d], idx + 2);
            if (val)
                for (size_t i = 0; i < val->n; i++)
                    Fv[i * K + k] += cfg.eta * tree_eval(tnodes, 0, &val->X[i * d], idx + 2);

            /* close the fit's temp scope (row buffers); tnodes and the
             * node buffer are persistent and survive the rewind */
            rcre_arena_temp_end(tt);

            if (node_off + idx > node_cap) {
                size_t ncap = node_cap ? node_cap : 256;
                while (node_off + idx > ncap) ncap *= 2;
                XGBNode* nb = (XGBNode*)rcre_arena_alloc(arena, ncap * sizeof(XGBNode), 16);
                if (!nb) return -1;
                if (node_off) memcpy(nb, nodes, node_off * sizeof(XGBNode));
                nodes = nb;
                node_cap = ncap;
            }
            /* rebase tree-local child indices to the global buffer
             * (trees are concatenated; the RCRE predictor resolves
             * global indices through per-tree roots) */
            for (size_t i2 = 0; i2 < idx; i2++) {
                if (!tnodes[i2].is_leaf) {
                    tnodes[i2].left  += (uint32_t)node_off;
                    tnodes[i2].right += (uint32_t)node_off;
                }
            }
            memcpy(&nodes[node_off], tnodes, idx * sizeof(XGBNode));
            model->tree_roots[r * K + k] = (uint32_t)node_off;
            node_off += idx;
            tree_end[r * K + k] = (uint32_t)node_off;
        }

        /* validation bookkeeping */
        if (val) {
            float m = xgb_eval_metric(cfg.task, val->y, Fv, val->n, K);
            model->val_curve[r] = m;
            model->val_curve_len = r + 1;
            if (best_round < 0 || m < best_metric - 1e-12f) {
                best_metric = m;
                best_round = r;
            }
            if (cfg.early_stop_rounds > 0 && r - best_round >= cfg.early_stop_rounds)
                break;
        }
    }

    /* finalize: keep trees up to the best round (or all) */
    int kept_rounds = (best_round >= 0) ? best_round + 1 : rounds_done;
    if (kept_rounds < 1) kept_rounds = 1;
    model->nodes = nodes;
    model->num_nodes = tree_end[(kept_rounds * K) - 1];
    model->num_trees = kept_rounds * K;
    model->best_iteration = (best_round >= 0) ? best_round : -1;
    return 0;
}
