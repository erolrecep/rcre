/**
 * @file stat.c
 * @brief RCRE Exploratory Data Analysis & Scientific Statistics Implementation (ISO C23)
 */

#include "rcre_stat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================= */
/* 1. Descriptive Summary Statistics (Welford Single-Pass)                   */
/* ========================================================================= */

static float quickselect_kth(float* a, size_t n, size_t k) {
    if (n == 0) return 0.0f;
    if (n == 1) return a[0];
    if (k >= n) k = n - 1;

    size_t left = 0;
    size_t right = n - 1;

    while (left < right) {
        size_t mid = left + (right - left) / 2;
        float pivot = a[mid];
        /* Swap pivot with right */
        a[mid] = a[right];
        a[right] = pivot;

        size_t store_idx = left;
        for (size_t i = left; i < right; ++i) {
            if (a[i] < pivot) {
                float tmp = a[store_idx];
                a[store_idx] = a[i];
                a[i] = tmp;
                store_idx++;
            }
        }
        a[right] = a[store_idx];
        a[store_idx] = pivot;

        if (store_idx == k) {
            return a[k];
        } else if (k < store_idx) {
            right = (store_idx > 0) ? (store_idx - 1) : 0;
        } else {
            left = store_idx + 1;
        }
    }
    return a[left];
}

float rcre_stat_quantile(
    const float* x,
    size_t n,
    float q,
    RCRE_Arena* scratch
) {
    if (!x || n == 0 || q < 0.0f || q > 1.0f || !scratch) {
        return 0.0f;
    }
    if (n == 1) {
        return x[0];
    }

    RCRE_ArenaTemp tmp_scope = rcre_arena_temp_begin(scratch);
    float* a = (float*)rcre_arena_alloc(scratch, n * sizeof(float), 64);
    if (!a) {
        rcre_arena_temp_end(tmp_scope);
        return 0.0f;
    }

    float idx_f = q * (float)(n - 1);
    size_t lo = (size_t)idx_f;
    size_t hi = (lo + 1 < n) ? (lo + 1) : lo;
    float frac = idx_f - (float)lo;

    memcpy(a, x, n * sizeof(float));
    float val_lo = quickselect_kth(a, n, lo);
    float val_hi = val_lo;

    if (hi != lo) {
        memcpy(a, x, n * sizeof(float));
        val_hi = quickselect_kth(a, n, hi);
    }

    rcre_arena_temp_end(tmp_scope);
    return (1.0f - frac) * val_lo + frac * val_hi;
}

RCRE_StatSummary rcre_stat_describe(
    const float* x,
    size_t n,
    RCRE_Arena* scratch
) {
    RCRE_StatSummary out = {0};
    if (!x || n == 0 || !scratch) {
        return out;
    }

    double mean = 0.0;
    double m2 = 0.0;
    double m3 = 0.0;
    double m4 = 0.0;
    float min_val = x[0];
    float max_val = x[0];
    size_t valid_count = 0;
    size_t null_count = 0;

    for (size_t i = 0; i < n; ++i) {
        float val = x[i];
        if (isnan(val)) {
            null_count++;
            continue;
        }
        if (valid_count == 0 || val < min_val) min_val = val;
        if (valid_count == 0 || val > max_val) max_val = val;

        valid_count++;
        double n1 = (double)valid_count;
        double delta = (double)val - mean;
        double delta_n = delta / n1;
        double delta_n2 = delta_n * delta_n;
        double term1 = delta * delta_n * (double)(valid_count - 1);

        m4 += term1 * delta_n2 * (n1 * n1 - 3.0 * n1 + 3.0) + 6.0 * delta_n2 * m2 - 4.0 * delta_n * m3;
        m3 += term1 * delta_n * (n1 - 2.0) - 3.0 * delta_n * m2;
        m2 += term1;
        mean += delta_n;
    }

    out.count = valid_count;
    out.null_count = null_count;
    out.mean = (float)mean;
    out.min = min_val;
    out.max = max_val;

    if (valid_count > 1) {
        double variance = m2 / (double)(valid_count - 1);
        out.variance = (float)variance;
        out.stddev = (float)sqrt(variance);

        if (m2 > 1e-12) {
            double s = sqrt(m2 / (double)valid_count);
            out.skewness = (float)((m3 / (double)valid_count) / (s * s * s));
            out.kurtosis = (float)(((m4 / (double)valid_count) / (s * s * s * s)) - 3.0);
        }
    }

    /* Compute quantiles */
    if (valid_count > 0) {
        float q25 = rcre_stat_quantile(x, n, 0.25f, scratch);
        float median = rcre_stat_quantile(x, n, 0.50f, scratch);
        float q75 = rcre_stat_quantile(x, n, 0.75f, scratch);

        out.q25 = q25;
        out.median = median;
        out.q75 = q75;
        out.iqr = q75 - q25;
    }

    return out;
}

/* ========================================================================= */
/* 2. Covariance & Correlation Matrices                                      */
/* ========================================================================= */

RCRE_Tensor rcre_stat_cov_matrix(
    const RCRE_Tensor* X,
    RCRE_Arena* arena
) {
    RCRE_Tensor out_cov = {0};
    if (!X || !X->data || X->ndim != 2 || X->dtype != RCRE_DTYPE_F32 || !arena) {
        return out_cov;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (N < 2 || P == 0) return out_cov;

    size_t cov_shape[2] = { P, P };
    out_cov = rcre_tensor_create(cov_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!out_cov.data) return out_cov;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
    float* mean = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    float* X_c = (float*)rcre_arena_alloc(arena, N * P * sizeof(float), 64);
    if (!mean || !X_c) {
        rcre_arena_temp_end(tmp);
        return out_cov;
    }

    const float* src = (const float*)X->data;

    /* 1. Compute column means */
    for (size_t j = 0; j < P; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += (double)src[i * P + j];
        }
        mean[j] = (float)(sum / (double)N);
    }

    /* 2. Center dataset: X_c = X - mean */
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < P; ++j) {
            X_c[i * P + j] = src[i * P + j] - mean[j];
        }
    }

    /* 3. Covariance: C = (X_c^T * X_c) / (N - 1) */
    float inv_n1 = 1.0f / (float)(N - 1);
    float* dst = (float*)out_cov.data;

    for (size_t j = 0; j < P; ++j) {
        for (size_t k = j; k < P; ++k) {
            double dot = 0.0;
            for (size_t i = 0; i < N; ++i) {
                dot += (double)X_c[i * P + j] * (double)X_c[i * P + k];
            }
            float cov_jk = (float)(dot * (double)inv_n1);
            dst[j * P + k] = cov_jk;
            dst[k * P + j] = cov_jk;
        }
    }

    rcre_arena_temp_end(tmp);
    return out_cov;
}

RCRE_Tensor rcre_stat_corr_matrix(
    const RCRE_Tensor* X,
    RCRE_Arena* arena
) {
    RCRE_Tensor out_corr = {0};
    if (!X || !arena) {
        return out_corr;
    }

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
    RCRE_Tensor cov = rcre_stat_cov_matrix(X, arena);
    if (!cov.data) {
        rcre_arena_temp_end(tmp);
        return out_corr;
    }

    size_t P = X->shape[1];
    size_t corr_shape[2] = { P, P };
    out_corr = rcre_tensor_create(corr_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!out_corr.data) {
        rcre_arena_temp_end(tmp);
        return out_corr;
    }

    const float* cov_data = (const float*)cov.data;
    float* corr_data = (float*)out_corr.data;

    float* stds = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    if (!stds) {
        rcre_arena_temp_end(tmp);
        return out_corr;
    }

    for (size_t j = 0; j < P; ++j) {
        float var_j = cov_data[j * P + j];
        stds[j] = (var_j > 1e-12f) ? sqrtf(var_j) : 1e-6f;
    }

    for (size_t j = 0; j < P; ++j) {
        for (size_t k = j; k < P; ++k) {
            if (j == k) {
                corr_data[j * P + k] = 1.0f;
            } else {
                float r = cov_data[j * P + k] / (stds[j] * stds[k]);
                if (r > 1.0f) r = 1.0f;
                if (r < -1.0f) r = -1.0f;
                corr_data[j * P + k] = r;
                corr_data[k * P + j] = r;
            }
        }
    }

    rcre_arena_temp_end(tmp);
    return out_corr;
}

RCRE_Tensor rcre_stat_spearman_matrix(
    const RCRE_Tensor* X,
    RCRE_Arena* arena
) {
    RCRE_Tensor out_spearman = {0};
    if (!X || !X->data || X->ndim != 2 || !arena) {
        return out_spearman;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (N < 2 || P == 0) return out_spearman;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);

    size_t ranked_shape[2] = { N, P };
    RCRE_Tensor ranked = rcre_tensor_create(ranked_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!ranked.data) {
        rcre_arena_temp_end(tmp);
        return out_spearman;
    }

    const float* src = (const float*)X->data;
    float* dst = (float*)ranked.data;

    /* Rank each column */
    for (size_t j = 0; j < P; ++j) {
        for (size_t i = 0; i < N; ++i) {
            float val_i = src[i * P + j];
            size_t rank = 1;
            for (size_t k = 0; k < N; ++k) {
                if (src[k * P + j] < val_i) {
                    rank++;
                }
            }
            dst[i * P + j] = (float)rank;
        }
    }

    out_spearman = rcre_stat_corr_matrix(&ranked, arena);
    rcre_arena_temp_end(tmp);
    return out_spearman;
}

/* ========================================================================= */
/* 3. Principal Component Analysis (PCA)                                     */
/* ========================================================================= */

RCRE_PCA rcre_pca_fit(
    const RCRE_Tensor* X,
    size_t n_components,
    RCRE_Arena* arena
) {
    RCRE_PCA out_pca = {0};
    if (!X || !X->data || X->ndim != 2 || n_components == 0 || !arena) {
        return out_pca;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (n_components > P || N < 2) {
        return out_pca;
    }

    size_t comp_shape[2] = { n_components, P };
    out_pca.components = rcre_tensor_create(comp_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    out_pca.explained_variance = (float*)rcre_arena_alloc(arena, n_components * sizeof(float), 64);
    out_pca.explained_variance_ratio = (float*)rcre_arena_alloc(arena, n_components * sizeof(float), 64);
    out_pca.mean = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    out_pca.n_components = n_components;
    out_pca.n_features = P;

    if (!out_pca.components.data || !out_pca.explained_variance || !out_pca.explained_variance_ratio || !out_pca.mean) {
        return out_pca;
    }

    /* 1. Compute means */
    const float* src = (const float*)X->data;
    for (size_t j = 0; j < P; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += (double)src[i * P + j];
        }
        out_pca.mean[j] = (float)(sum / (double)N);
    }

    /* 2. Covariance matrix C */
    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
    RCRE_Tensor cov = rcre_stat_cov_matrix(X, arena);
    if (!cov.data) {
        rcre_arena_temp_end(tmp);
        return out_pca;
    }

    float* C = (float*)rcre_arena_alloc(arena, P * P * sizeof(float), 64);
    if (!C) {
        rcre_arena_temp_end(tmp);
        return out_pca;
    }
    memcpy(C, cov.data, P * P * sizeof(float));

    /* Compute total variance (trace of covariance) */
    double total_variance = 0.0;
    for (size_t j = 0; j < P; ++j) {
        total_variance += (double)C[j * P + j];
    }
    if (total_variance <= 1e-12) total_variance = 1.0;

    /* 3. Power Iteration with Hotelling Deflation for top K components */
    float* v = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    float* v_next = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    if (!v || !v_next) {
        rcre_arena_temp_end(tmp);
        return out_pca;
    }

    float* comp_data = (float*)out_pca.components.data;

    for (size_t k = 0; k < n_components; ++k) {
        /* Initialize deterministic random vector */
        uint64_t seed = (uint64_t)(k * 1337 + 7);
        for (size_t j = 0; j < P; ++j) {
            seed = seed * 6364136223846793005ULL + 1ULL;
            v[j] = (float)((double)(seed >> 33) / 2147483648.0) - 0.5f;
        }

        /* Power iterations */
        for (int iter = 0; iter < 100; ++iter) {
            double norm = 0.0;
            for (size_t j = 0; j < P; ++j) {
                double dot = 0.0;
                for (size_t m = 0; m < P; ++m) {
                    dot += (double)C[j * P + m] * (double)v[m];
                }
                v_next[j] = (float)dot;
                norm += dot * dot;
            }
            norm = sqrt(norm);
            if (norm <= 1e-12) break;
            for (size_t j = 0; j < P; ++j) {
                v[j] = (float)(v_next[j] / norm);
            }
        }

        /* Compute eigenvalue lambda_k = v^T * C * v */
        double lambda = 0.0;
        for (size_t j = 0; j < P; ++j) {
            double dot = 0.0;
            for (size_t m = 0; m < P; ++m) {
                dot += (double)C[j * P + m] * (double)v[m];
            }
            lambda += (double)v[j] * dot;
        }

        out_pca.explained_variance[k] = (float)lambda;
        out_pca.explained_variance_ratio[k] = (float)(lambda / total_variance);

        /* Canonicalize eigenvector sign (svd_flip: largest-magnitude element positive) */
        size_t max_j = 0;
        for (size_t j = 1; j < P; ++j) {
            if (fabsf(v[j]) > fabsf(v[max_j])) max_j = j;
        }
        if (v[max_j] < 0.0f) {
            for (size_t j = 0; j < P; ++j) v[j] = -v[j];
        }

        /* Save component vector */
        memcpy(comp_data + k * P, v, P * sizeof(float));

        /* Deflate: C = C - lambda * v * v^T */
        for (size_t j = 0; j < P; ++j) {
            for (size_t m = 0; m < P; ++m) {
                C[j * P + m] -= (float)(lambda * (double)v[j] * (double)v[m]);
            }
        }
    }

    rcre_arena_temp_end(tmp);
    return out_pca;
}

RCRE_Tensor rcre_pca_transform(
    const RCRE_PCA* pca,
    const RCRE_Tensor* X,
    RCRE_Arena* arena
) {
    RCRE_Tensor out_projected = {0};
    if (!pca || !X || !X->data || X->ndim != 2 || !arena) {
        return out_projected;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    size_t K = pca->n_components;
    if (P != pca->n_features) return out_projected;

    size_t out_shape[2] = { N, K };
    out_projected = rcre_tensor_create(out_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!out_projected.data) return out_projected;

    const float* src = (const float*)X->data;
    const float* comp = (const float*)pca->components.data;
    float* dst = (float*)out_projected.data;

    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < K; ++k) {
            double dot = 0.0;
            for (size_t j = 0; j < P; ++j) {
                double centered = (double)src[i * P + j] - (double)pca->mean[j];
                dot += centered * (double)comp[k * P + j];
            }
            dst[i * K + k] = (float)dot;
        }
    }

    return out_projected;
}
