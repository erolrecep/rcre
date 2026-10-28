/**
 * @file xgb_predict.c
 * @brief Inference: unified predictor (NaN-aware, multi-class) + RCRE
 *        flat-inference adapter.
 */
#include "xgb_internal.h"

static inline float tree_eval(const XGBNode* nodes, uint32_t root,
                              const float* row, size_t max_steps) {
    const XGBNode* nd = &nodes[root];
    size_t steps = 0;
    while (!nd->is_leaf) {
        if (++steps > max_steps) return 0.0f;  /* cycle guard (corrupt model) */
        float x = row[nd->feature];
        bool go_left = (x != x) ? nd->default_left : (x <= nd->threshold);
        nd = &nodes[go_left ? nd->left : nd->right];
    }
    return nd->leaf_value;
}

int xgb_predict_raw(const XGBModel* m, const float* X, size_t n, size_t d, float* out) {
    if (!m || !X || !out || m->num_trees == 0) return -1;
    int K = m->num_classes;
    if ((size_t)m->dim > d) return -1;
    float raw[XGB_MAX_CLASSES];
    for (size_t i = 0; i < n; i++) {
        const float* row = &X[i * d];
        for (int k = 0; k < K; k++)
            raw[k] = (m->task == XGB_TASK_MULTI) ? m->base_scores[k] : m->base_score;
        for (int t = 0; t < m->num_trees; t++) {
            int k = (m->task == XGB_TASK_MULTI) ? (t % K) : 0;
            raw[k] += m->eta * tree_eval(m->nodes, m->tree_roots[t], row,
                                         m->num_nodes + 2);
        }
        for (int k = 0; k < K; k++) out[i * K + k] = raw[k];
    }
    return 0;
}

int xgb_predict(const XGBModel* m, const float* X, size_t n, size_t d, float* out) {
    if (xgb_predict_raw(m, X, n, d, out) != 0) return -1;
    int K = m->num_classes;
    if (m->task == XGB_TASK_BINARY) {
        for (size_t i = 0; i < n; i++) out[i] = xgb_sigmoidf(out[i]);
    } else if (m->task == XGB_TASK_MULTI) {
        for (size_t i = 0; i < n; i++) {
            float* o = &out[i * K];
            float mx = o[0];
            for (int c = 1; c < K; c++) mx = mx > o[c] ? mx : o[c];
            float s = 0.0f;
            for (int c = 0; c < K; c++) s += expf(o[c] - mx);
            for (int c = 0; c < K; c++) o[c] = expf(o[c] - mx) / s;
        }
    }
    return 0;
}

/* ========================================================================= */
/* RCRE interop                                                              */
/* ========================================================================= */

int xgb_to_rcre_gbdt(const XGBModel* m, RCRE_GBDTModel* out,
                     RCRE_TreeNode* out_nodes, size_t* out_roots) {
    if (!m || !out || !out_nodes || !out_roots) return -1;
    if (m->task == XGB_TASK_MULTI) return -1;   /* RCRE predictor: single output */
    if (m->num_trees == 0) return -1;

    for (size_t i = 0; i < m->num_nodes; i++) {
        const XGBNode* nd = &m->nodes[i];
        out_nodes[i].feature_idx = nd->feature;
        out_nodes[i].threshold = nd->threshold;
        out_nodes[i].left_child = nd->left;
        out_nodes[i].right_child = nd->right;
        out_nodes[i].leaf_value = nd->leaf_value * m->eta;  /* eta-baked */
        out_nodes[i].is_leaf = nd->is_leaf;
    }
    for (size_t t = 0; t < (size_t)m->num_trees; t++)
        out_roots[t] = (size_t)m->tree_roots[t];

    out->nodes = out_nodes;
    out->tree_roots = out_roots;
    out->num_trees = (size_t)m->num_trees;
    out->num_nodes = m->num_nodes;
    out->base_score = m->base_score;
    out->is_binary_classification = (m->task == XGB_TASK_BINARY);
    return 0;
}

/* ========================================================================= */
/* Diagnostics                                                               */
/* ========================================================================= */

void xgb_model_counts(const XGBModel* m, int* internal, int* leaves) {
    int ni = 0, nl = 0;
    if (!m) { *internal = *leaves = 0; return; }
    for (size_t i = 0; i < m->num_nodes; i++) {
        if (m->nodes[i].is_leaf) nl++;
        else ni++;
    }
    *internal = ni;
    *leaves = nl;
}

void xgb_model_print(const XGBModel* m, const char* const* feature_names) {
    if (!m) return;
    printf("model: task=%s classes=%d trees=%d nodes=%zu eta=%.3f base=%g best_iter=%d\n",
           m->task == XGB_TASK_REGRESSION ? "regression"
           : m->task == XGB_TASK_BINARY ? "binary" : "multi",
           m->num_classes, m->num_trees, m->num_nodes, m->eta,
           (double)m->base_score, m->best_iteration);
    if (m->task == XGB_TASK_MULTI)
        for (int k = 0; k < m->num_classes; k++)
            printf("  base_score[%d] = %g\n", k, (double)m->base_scores[k]);

    int ni = 0, nl = 0;
    xgb_model_counts(m, &ni, &nl);
    printf("  internal=%d leaves=%d\n", ni, nl);

    for (int t = 0; t < m->num_trees; t++) {
        printf("tree %d (root node %u):\n", t, m->tree_roots[t]);
        /* simple recursive indent print (depth-limited stack via recursion) */
        struct P { uint32_t idx; int depth; };
        struct P st[512];
        int sp = 0;
        st[sp].idx = m->tree_roots[t];
        st[sp].depth = 0;
        sp++;
        while (sp > 0) {
            sp--;
            struct P* p = &st[sp];
            const XGBNode* nd = &m->nodes[p->idx];
                /* print in pre-order: current then children (right first so
                 * left is popped first) */
                if (nd->is_leaf) {
                    for (int i = 0; i < p->depth; i++) printf("  ");
                    printf("[leaf] value=%.6g\n", (double)nd->leaf_value);
                } else {
                    for (int i = 0; i < p->depth; i++) printf("  ");
                    char fbuf[64];
                    if (feature_names && feature_names[nd->feature])
                        snprintf(fbuf, sizeof fbuf, "%s", feature_names[nd->feature]);
                    else
                        snprintf(fbuf, sizeof fbuf, "f%u", nd->feature);
                    printf("[split] %s <= %g (default %s)\n", fbuf,
                           (double)nd->threshold,
                           nd->default_left ? "left" : "right");
                    if (p->depth + 2 < 512) {
                        st[sp].idx = nd->right; st[sp].depth = p->depth + 1; sp++;
                        st[sp].idx = nd->left;  st[sp].depth = p->depth + 1; sp++;
                    }
                }
        }
    }
}
