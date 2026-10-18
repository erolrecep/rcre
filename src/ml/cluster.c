/**
 * @file cluster.c
 * @brief Clustering & Dimensionality Reduction (k-Means, PCA)
 * 
 * Satisfies ML-REQ-02.
 */

#include "rcre.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Compute squared Euclidean distance between two D-dimensional vectors */
static inline float squared_l2_distance(const float* a, const float* b, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

RCRE_NODISCARD int rcre_ml_kmeans_fit(const RCRE_Tensor* X, size_t k, size_t max_iter,
                                      RCRE_Tensor* centroids, RCRE_Tensor* labels,
                                      RCRE_Arena* arena) {
    if (!X || !X->data || !centroids || !centroids->data || !arena || k == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (X->ndim != 2 || centroids->ndim != 2) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t N = X->shape[0];
    size_t D = X->shape[1];

    if (k > N || centroids->shape[0] != k || centroids->shape[1] != D) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* X_data = (const float*)X->data;
    float* C_data = (float*)centroids->data;

    /* Allocate transient working memory from arena */
    float* new_C = (float*)rcre_arena_alloc(arena, k * D * sizeof(float), 64);
    size_t* counts = (size_t*)rcre_arena_alloc(arena, k * sizeof(size_t), 64);
    size_t* cluster_assign = (size_t*)rcre_arena_alloc(arena, N * sizeof(size_t), 64);
    float* min_sq_dist = (float*)rcre_arena_alloc(arena, N * sizeof(float), 64);

    if (!new_C || !counts || !cluster_assign || !min_sq_dist) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* k-Means++ initialization */
    /* Centroid 0: first point */
    memcpy(&C_data[0], &X_data[0], D * sizeof(float));

    for (size_t c = 1; c < k; ++c) {
        float max_d2 = -1.0f;
        size_t best_idx = 0;

        for (size_t i = 0; i < N; ++i) {
            float d2 = squared_l2_distance(&X_data[i * D], &C_data[(c - 1) * D], D);
            if (c == 1 || d2 < min_sq_dist[i]) {
                min_sq_dist[i] = d2;
            }
            if (min_sq_dist[i] > max_d2) {
                max_d2 = min_sq_dist[i];
                best_idx = i;
            }
        }
        memcpy(&C_data[c * D], &X_data[best_idx * D], D * sizeof(float));
    }

    /* Lloyd's iteration loop */
    for (size_t iter = 0; iter < max_iter; ++iter) {
        bool changed = false;

        /* 1. Assignment step */
        for (size_t i = 0; i < N; ++i) {
            const float* x_i = &X_data[i * D];
            float best_dist = INFINITY;
            size_t best_c = 0;

            for (size_t c = 0; c < k; ++c) {
                float dist = squared_l2_distance(x_i, &C_data[c * D], D);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }

            if (cluster_assign[i] != best_c) {
                cluster_assign[i] = best_c;
                changed = true;
            }
        }

        if (!changed && iter > 0) {
            break; /* Converged */
        }

        /* 2. Update step */
        memset(new_C, 0, k * D * sizeof(float));
        memset(counts, 0, k * sizeof(size_t));

        for (size_t i = 0; i < N; ++i) {
            size_t c = cluster_assign[i];
            counts[c]++;
            const float* x_i = &X_data[i * D];
            float* nc_i = &new_C[c * D];
            for (size_t d = 0; d < D; ++d) {
                nc_i[d] += x_i[d];
            }
        }

        for (size_t c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                float inv_count = 1.0f / (float)counts[c];
                float* nc_i = &new_C[c * D];
                for (size_t d = 0; d < D; ++d) {
                    C_data[c * D + d] = nc_i[d] * inv_count;
                }
            }
        }
    }

    /* Copy labels if requested */
    if (labels && labels->data) {
        if (labels->dtype == RCRE_DTYPE_F32) {
            float* l_data = (float*)labels->data;
            for (size_t i = 0; i < N; ++i) {
                l_data[i] = (float)cluster_assign[i];
            }
        } else if (labels->dtype == RCRE_DTYPE_I32) {
            int32_t* l_data = (int32_t*)labels->data;
            for (size_t i = 0; i < N; ++i) {
                l_data[i] = (int32_t)cluster_assign[i];
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_ml_kmeans_predict(const RCRE_Tensor* X, const RCRE_Tensor* centroids,
                                          RCRE_Tensor* labels) {
    if (!X || !X->data || !centroids || !centroids->data || !labels || !labels->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t N = X->shape[0];
    size_t D = X->shape[1];
    size_t K = centroids->shape[0];

    if (centroids->shape[1] != D || labels->shape[0] != N) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* X_data = (const float*)X->data;
    const float* C_data = (const float*)centroids->data;

    for (size_t i = 0; i < N; ++i) {
        const float* x_i = &X_data[i * D];
        float best_dist = INFINITY;
        size_t best_c = 0;

        for (size_t c = 0; c < K; ++c) {
            float dist = squared_l2_distance(x_i, &C_data[c * D], D);
            if (dist < best_dist) {
                best_dist = dist;
                best_c = c;
            }
        }

        if (labels->dtype == RCRE_DTYPE_F32) {
            ((float*)labels->data)[i] = (float)best_c;
        } else if (labels->dtype == RCRE_DTYPE_I32) {
            ((int32_t*)labels->data)[i] = (int32_t)best_c;
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_ml_pca_fit(const RCRE_Tensor* X, size_t k, 
                                   RCRE_Tensor* components, RCRE_Tensor* mean,
                                   RCRE_Arena* arena) {
    if (!X || !X->data || !components || !components->data || !mean || !mean->data || !arena || k == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t N = X->shape[0];
    size_t D = X->shape[1];

    if (k > D || components->shape[0] != k || components->shape[1] != D || mean->shape[0] != D) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* X_data = (const float*)X->data;
    float* W_data = (float*)components->data;
    float* mean_data = (float*)mean->data;

    /* 1. Compute column feature means */
    memset(mean_data, 0, D * sizeof(float));
    for (size_t i = 0; i < N; ++i) {
        const float* row = &X_data[i * D];
        for (size_t d = 0; d < D; ++d) {
            mean_data[d] += row[d];
        }
    }
    float inv_N = 1.0f / (float)N;
    for (size_t d = 0; d < D; ++d) {
        mean_data[d] *= inv_N;
    }

    /* 2. Compute covariance matrix C (D x D) */
    float* C = (float*)rcre_arena_alloc(arena, D * D * sizeof(float), 64);
    float* v = (float*)rcre_arena_alloc(arena, D * sizeof(float), 64);
    float* w = (float*)rcre_arena_alloc(arena, D * sizeof(float), 64);
    if (!C || !v || !w) return RCRE_ERR_OUT_OF_MEMORY;

    memset(C, 0, D * D * sizeof(float));

    float inv_N_minus_1 = (N > 1) ? (1.0f / (float)(N - 1)) : 1.0f;
    for (size_t i = 0; i < N; ++i) {
        const float* row = &X_data[i * D];
        for (size_t r = 0; r < D; ++r) {
            float x_r = row[r] - mean_data[r];
            for (size_t c = r; c < D; ++c) {
                float x_c = row[c] - mean_data[c];
                C[r * D + c] += x_r * x_c;
            }
        }
    }

    for (size_t r = 0; r < D; ++r) {
        for (size_t c = r; c < D; ++c) {
            C[r * D + c] *= inv_N_minus_1;
            C[c * D + r] = C[r * D + c];
        }
    }

    /* 3. Power iteration with deflation for top K eigenvectors */
    for (size_t comp = 0; comp < k; ++comp) {
        /* Deterministic initial guess */
        for (size_t d = 0; d < D; ++d) {
            v[d] = (float)((int)((d + comp) % 5) + 1) * 0.2f;
        }

        /* Power iteration */
        for (size_t iter = 0; iter < 100; ++iter) {
            /* w = C * v */
            for (size_t r = 0; r < D; ++r) {
                w[r] = rcre_hal.dot_f32(&C[r * D], v, D);
            }

            /* Gram-Schmidt orthogonalization against previous components */
            for (size_t prev = 0; prev < comp; ++prev) {
                float proj = rcre_hal.dot_f32(w, &W_data[prev * D], D);
                rcre_hal.axpy_f32(D, -proj, &W_data[prev * D], w);
            }

            /* Normalize */
            float norm_sq = rcre_hal.dot_f32(w, w, D);
            if (norm_sq < 1e-12f) break;
            float inv_norm = 1.0f / sqrtf(norm_sq);

            for (size_t d = 0; d < D; ++d) {
                v[d] = w[d] * inv_norm;
            }
        }

        /* Store component */
        memcpy(&W_data[comp * D], v, D * sizeof(float));

        /* Deflate C: C = C - lambda * v * v^T */
        /* lambda = v^T * C * v */
        for (size_t r = 0; r < D; ++r) {
            w[r] = rcre_hal.dot_f32(&C[r * D], v, D);
        }
        float lambda = rcre_hal.dot_f32(v, w, D);

        for (size_t r = 0; r < D; ++r) {
            for (size_t c = 0; c < D; ++c) {
                C[r * D + c] -= lambda * v[r] * v[c];
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_ml_pca_transform(const RCRE_Tensor* X, const RCRE_Tensor* components,
                                         const RCRE_Tensor* mean, RCRE_Tensor* X_proj) {
    if (!X || !X->data || !components || !components->data || !mean || !mean->data || !X_proj || !X_proj->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t N = X->shape[0];
    size_t D = X->shape[1];
    size_t K = components->shape[0];

    if (components->shape[1] != D || mean->shape[0] != D || X_proj->shape[0] != N || X_proj->shape[1] != K) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* X_data = (const float*)X->data;
    const float* W_data = (const float*)components->data;
    const float* mean_data = (const float*)mean->data;
    float* out_data = (float*)X_proj->data;

    for (size_t i = 0; i < N; ++i) {
        const float* row = &X_data[i * D];

        for (size_t k_idx = 0; k_idx < K; ++k_idx) {
            const float* comp_vec = &W_data[k_idx * D];
            float dot_val = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                dot_val += (row[d] - mean_data[d]) * comp_vec[d];
            }
            out_data[i * K + k_idx] = dot_val;
        }
    }

    return RCRE_OK;
}
