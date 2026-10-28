/**
 * @file xgb_metrics.c
 * @brief Evaluation metrics (logloss, AUC, RMSE, MAE, accuracy).
 */
#include "xgb_internal.h"

float xgb_logloss(const float* y, const float* p, size_t n) {
    if (!y || !p || n == 0) return 0.0f;
    double s = 0.0;
    for (size_t i = 0; i < n; i++)
        s -= y[i] * xgb_logf_eps(p[i]) + (1.0f - y[i]) * xgb_logf_eps(1.0f - p[i]);
    return (float)(s / (double)n);
}

/* AUC via rank-sum with tie-averaged ranks (O(n log n)) */
typedef struct { float p; float y; } ScorePair;
static int cmp_score(const void* a, const void* b) {
    float x = ((const ScorePair*)a)->p, yv = ((const ScorePair*)b)->p;
    return (x > yv) - (x < yv);
}

float xgb_auc(const float* y, const float* p, size_t n) {
    if (!y || !p || n < 2) return 0.5f;
    ScorePair* a = (ScorePair*)malloc(n * sizeof(ScorePair));
    if (!a) return 0.5f;
    size_t npos = 0, nneg = 0;
    for (size_t i = 0; i < n; i++) {
        a[i].p = p[i];
        a[i].y = y[i];
        if (y[i] > 0.5f) npos++;
        else nneg++;
    }
    if (npos == 0 || nneg == 0) { free(a); return 0.5f; }
    qsort(a, n, sizeof(ScorePair), cmp_score);

    /* assign average ranks to ties, then sum ranks of positives */
    double rank_sum = 0.0;
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j + 1 < n && a[j + 1].p == a[i].p) j++;
        double avg_rank = 0.5 * ((double)(i + 1) + (double)(j + 1)); /* 1-based */
        for (size_t k = i; k <= j; k++)
            if (a[k].y > 0.5f) rank_sum += avg_rank;
        i = j + 1;
    }
    double auc = (rank_sum - (double)npos * ((double)npos + 1.0) / 2.0)
                 / ((double)npos * (double)nneg);
    free(a);
    return (float)auc;
}

float xgb_rmse(const float* y, const float* p, size_t n) {
    if (!y || !p || n == 0) return 0.0f;
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        float e = p[i] - y[i];
        s += (double)e * e;
    }
    return sqrtf((float)(s / (double)n));
}

float xgb_mae(const float* y, const float* p, size_t n) {
    if (!y || !p || n == 0) return 0.0f;
    double s = 0.0;
    for (size_t i = 0; i < n; i++)
        s += (double)fabsf(p[i] - y[i]);
    return (float)(s / (double)n);
}

/* p: n x K probabilities (row-major); y: class ids */
float xgb_accuracy(const float* y, const float* p, size_t n, int K) {
    if (!y || !p || n == 0 || K < 1) return 0.0f;
    size_t ok = 0;
    for (size_t i = 0; i < n; i++) {
        int best = 0;
        float bv = p[i * K];
        for (int c = 1; c < K; c++)
            if (p[i * K + c] > bv) { bv = p[i * K + c]; best = c; }
        if (best == (int)y[i]) ok++;
    }
    return (float)((double)ok / (double)n);
}
