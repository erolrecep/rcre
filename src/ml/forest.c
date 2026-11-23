/**
 * @file forest.c
 * @brief Random Forest Binary Classifier (Bagging + Random Feature Subset Gini Split, ISO C23)
 */

#include "rcre.h"
#include <math.h>
#include <string.h>

static uint64_t rf_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static size_t rf_build_tree(
    RCRE_RandomForest* rf,
    size_t max_nodes,
    const float* X,
    const float* y,
    size_t D,
    size_t* sample_idx,
    size_t n_samples,
    size_t depth,
    size_t max_depth,
    size_t mtry,
    uint64_t* rng
) {
    size_t node_id = rf->num_nodes++;
    RCRE_RFNode* nd = &rf->nodes[node_id];
    memset(nd, 0, sizeof(*nd));

    float pos = 0.0f;
    for (size_t i = 0; i < n_samples; ++i) pos += (y[sample_idx[i]] > 0.5f) ? 1.0f : 0.0f;
    float p = (n_samples > 0) ? (pos / (float)n_samples) : 0.5f;
    nd->leaf_prob = p;
    nd->feature_idx = -1;
    nd->left = -1;
    nd->right = -1;

    if (depth >= max_depth || n_samples < 4 || p <= 0.0f || p >= 1.0f || rf->num_nodes + 2 >= max_nodes) {
        return node_id;
    }

    float best_gini = 1e30f;
    int32_t best_feat = -1;
    float best_thresh = 0.0f;

    for (size_t f_try = 0; f_try < mtry; ++f_try) {
        size_t feat = (size_t)(rf_splitmix64(rng) % D);
        /* Evaluate up to 8 threshold quantile probes from sample subset */
        for (int probe = 0; probe < 8; ++probe) {
            size_t ref = sample_idx[(size_t)(rf_splitmix64(rng) % n_samples)];
            float thr = X[ref * D + feat];
            size_t nl = 0, nr = 0;
            float pl = 0.0f, pr = 0.0f;
            for (size_t i = 0; i < n_samples; ++i) {
                size_t si = sample_idx[i];
                float label = (y[si] > 0.5f) ? 1.0f : 0.0f;
                if (X[si * D + feat] <= thr) { nl++; pl += label; }
                else { nr++; pr += label; }
            }
            if (nl == 0 || nr == 0) continue;
            float ql = pl / (float)nl;
            float qr = pr / (float)nr;
            float gl = 2.0f * ql * (1.0f - ql);
            float gr = 2.0f * qr * (1.0f - qr);
            float gini = ((float)nl * gl + (float)nr * gr) / (float)n_samples;
            if (gini < best_gini) {
                best_gini = gini;
                best_feat = (int32_t)feat;
                best_thresh = thr;
            }
        }
    }

    if (best_feat < 0) return node_id;

    /* Partition sample_idx in-place */
    size_t l = 0, r = n_samples;
    while (l < r) {
        if (X[sample_idx[l] * D + (size_t)best_feat] <= best_thresh) {
            l++;
        } else {
            r--;
            size_t tmp = sample_idx[l];
            sample_idx[l] = sample_idx[r];
            sample_idx[r] = tmp;
        }
    }
    if (l == 0 || l == n_samples) return node_id;

    nd->feature_idx = best_feat;
    nd->threshold   = best_thresh;
    size_t left_id  = rf_build_tree(rf, max_nodes, X, y, D, sample_idx, l, depth + 1U, max_depth, mtry, rng);
    size_t right_id = rf_build_tree(rf, max_nodes, X, y, D, sample_idx + l, n_samples - l, depth + 1U, max_depth, mtry, rng);
    rf->nodes[node_id].left  = (int32_t)left_id;
    rf->nodes[node_id].right = (int32_t)right_id;
    return node_id;
}

RCRE_NODISCARD int rcre_ml_rf_fit(const RCRE_Tensor* X, const RCRE_Tensor* y,
                                  size_t num_trees, size_t max_depth, uint64_t seed,
                                  RCRE_Arena* arena, RCRE_RandomForest* out_rf) {
    if (!X || !X->data || !y || !y->data || !arena || !out_rf || num_trees == 0 || max_depth == 0) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (X->ndim != 2 || y->ndim < 1 || y->shape[0] != X->shape[0]) return RCRE_ERR_SHAPE_MISMATCH;

    size_t N = X->shape[0], D = X->shape[1];
    size_t mtry = (size_t)ceil(sqrt((double)D));
    if (mtry == 0) mtry = 1;
    if (max_depth > 10) max_depth = 10;

    size_t max_per_tree = (1ULL << (max_depth + 1U));
    size_t total_cap = num_trees * max_per_tree;

    memset(out_rf, 0, sizeof(*out_rf));
    out_rf->nodes        = (RCRE_RFNode*)rcre_arena_alloc(arena, total_cap * sizeof(RCRE_RFNode), 16);
    out_rf->tree_roots   = (size_t*)rcre_arena_alloc(arena, num_trees * sizeof(size_t), 16);
    out_rf->num_trees    = num_trees;
    out_rf->num_features = D;
    out_rf->max_depth    = max_depth;
    if (!out_rf->nodes || !out_rf->tree_roots) return RCRE_ERR_OUT_OF_MEMORY;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
    size_t* bag = (size_t*)rcre_arena_alloc(arena, N * sizeof(size_t), 16);
    if (!bag) { rcre_arena_temp_end(tmp); return RCRE_ERR_OUT_OF_MEMORY; }

    const float* Xd = (const float*)X->data;
    const float* yd = (const float*)y->data;
    if (seed == 0) seed = 0x123456789ABCDEF0ULL;

    for (size_t t = 0; t < num_trees; ++t) {
        for (size_t i = 0; i < N; ++i) bag[i] = (size_t)(rf_splitmix64(&seed) % N);
        out_rf->tree_roots[t] = rf_build_tree(out_rf, total_cap, Xd, yd, D, bag, N, 0, max_depth, mtry, &seed);
    }

    rcre_arena_temp_end(tmp);
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_ml_rf_predict(const RCRE_RandomForest* rf, const RCRE_Tensor* X, RCRE_Tensor* out_probs) {
    if (!rf || !rf->nodes || !X || !X->data || !out_probs || !out_probs->data) return RCRE_ERR_INVALID_ARG;
    if (X->ndim != 2 || X->shape[1] != rf->num_features || out_probs->shape[0] != X->shape[0]) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }
    const float* Xd = (const float*)X->data;
    float* pd = (float*)out_probs->data;
    size_t N = X->shape[0], D = rf->num_features;

    for (size_t i = 0; i < N; ++i) {
        const float* row = Xd + i * D;
        float sum = 0.0f;
        for (size_t t = 0; t < rf->num_trees; ++t) {
            int32_t cur = (int32_t)rf->tree_roots[t];
            while (cur >= 0) {
                const RCRE_RFNode* nd = &rf->nodes[cur];
                if (nd->feature_idx < 0) {
                    sum += nd->leaf_prob;
                    break;
                }
                cur = (row[nd->feature_idx] <= nd->threshold) ? nd->left : nd->right;
            }
        }
        pd[i] = sum / (float)rf->num_trees;
    }
    return RCRE_OK;
}
