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

/* SplitMix64 pseudo-random number generator for reproducible sampling */
static inline uint64_t rcre_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline float rcre_rand_f32(uint64_t* state) {
    return (float)((rcre_splitmix64(state) >> 40) * (1.0 / 16777216.0));
}

RCRE_NODISCARD int rcre_ml_kmeans_fit_ex(const RCRE_Tensor* X, size_t k,
                                         const RCRE_KMeansParams* params,
                                         RCRE_Tensor* centroids, RCRE_Tensor* labels,
                                         float* out_inertia, RCRE_Arena* arena) {
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

    size_t max_iter = (params && params->max_iter > 0) ? params->max_iter : 100;
    size_t n_init = (params && params->n_init > 0) ? params->n_init : 10;
    float tol = (params && params->tol > 0.0f) ? params->tol : 1e-4f;
    uint64_t rng_state = (params && params->seed != 0) ? params->seed : 42ULL;

    const float* X_data = (const float*)X->data;

    /* Allocate transient working memory from arena */
    float* cand_C = (float*)rcre_arena_alloc(arena, k * D * sizeof(float), 64);
    float* new_C = (float*)rcre_arena_alloc(arena, k * D * sizeof(float), 64);
    size_t* counts = (size_t*)rcre_arena_alloc(arena, k * sizeof(size_t), 64);
    size_t* cand_assign = (size_t*)rcre_arena_alloc(arena, N * sizeof(size_t), 64);
    float* min_sq_dist = (float*)rcre_arena_alloc(arena, N * sizeof(float), 64);

    float* best_C = (float*)centroids->data;
    size_t* best_assign = (size_t*)rcre_arena_alloc(arena, N * sizeof(size_t), 64);

    if (!cand_C || !new_C || !counts || !cand_assign || !min_sq_dist || !best_assign) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    float best_total_inertia = INFINITY;

    for (size_t init_run = 0; init_run < n_init; ++init_run) {
        /* 1. K-Means++ Probabilistic Initialization */
        /* Pick first centroid uniformly at random */
        size_t first_idx = (size_t)(rcre_splitmix64(&rng_state) % N);
        memcpy(&cand_C[0], &X_data[first_idx * D], D * sizeof(float));

        /* Initialize squared distances to first centroid */
        for (size_t i = 0; i < N; ++i) {
            min_sq_dist[i] = squared_l2_distance(&X_data[i * D], &cand_C[0], D);
        }

        /* Choose remaining k - 1 centroids via D^2 probabilistic sampling */
        for (size_t c = 1; c < k; ++c) {
            float total_weight = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                total_weight += min_sq_dist[i];
            }

            size_t chosen_idx = 0;
            if (total_weight > 1e-12f) {
                float threshold = rcre_rand_f32(&rng_state) * total_weight;
                float running_sum = 0.0f;
                for (size_t i = 0; i < N; ++i) {
                    running_sum += min_sq_dist[i];
                    if (running_sum >= threshold) {
                        chosen_idx = i;
                        break;
                    }
                }
            } else {
                chosen_idx = (size_t)(rcre_splitmix64(&rng_state) % N);
            }

            memcpy(&cand_C[c * D], &X_data[chosen_idx * D], D * sizeof(float));

            /* Update min_sq_dist for newly selected centroid */
            for (size_t i = 0; i < N; ++i) {
                float d2 = squared_l2_distance(&X_data[i * D], &cand_C[c * D], D);
                if (d2 < min_sq_dist[i]) {
                    min_sq_dist[i] = d2;
                }
            }
        }

        /* 2. Lloyd Iteration Loop with Early Stopping */
        for (size_t iter = 0; iter < max_iter; ++iter) {
            /* Assignment step */
            for (size_t i = 0; i < N; ++i) {
                const float* x_i = &X_data[i * D];
                float best_dist = INFINITY;
                size_t best_c = 0;

                for (size_t c = 0; c < k; ++c) {
                    float dist = squared_l2_distance(x_i, &cand_C[c * D], D);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                cand_assign[i] = best_c;
            }

            /* Update step */
            memset(new_C, 0, k * D * sizeof(float));
            memset(counts, 0, k * sizeof(size_t));

            for (size_t i = 0; i < N; ++i) {
                size_t c = cand_assign[i];
                counts[c]++;
                const float* x_i = &X_data[i * D];
                float* nc_i = &new_C[c * D];
                for (size_t d = 0; d < D; ++d) {
                    nc_i[d] += x_i[d];
                }
            }

            float max_shift = 0.0f;
            for (size_t c = 0; c < k; ++c) {
                if (counts[c] > 0) {
                    float inv_count = 1.0f / (float)counts[c];
                    float shift = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        float old_val = cand_C[c * D + d];
                        float new_val = new_C[c * D + d] * inv_count;
                        float diff = new_val - old_val;
                        shift += diff * diff;
                        cand_C[c * D + d] = new_val;
                    }
                    if (shift > max_shift) {
                        max_shift = shift;
                    }
                }
            }

            if (max_shift < tol * tol) {
                break; /* Centroids converged within tolerance */
            }
        }

        /* 3. Compute Within-Cluster Sum of Squares (Inertia) */
        float run_inertia = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            size_t c = cand_assign[i];
            run_inertia += squared_l2_distance(&X_data[i * D], &cand_C[c * D], D);
        }

        /* Retain best run across n_init trials */
        if (run_inertia < best_total_inertia) {
            best_total_inertia = run_inertia;
            memcpy(best_C, cand_C, k * D * sizeof(float));
            memcpy(best_assign, cand_assign, N * sizeof(size_t));
        }
    }

    if (out_inertia) {
        *out_inertia = best_total_inertia;
    }

    /* Copy labels if requested */
    if (labels && labels->data) {
        if (labels->dtype == RCRE_DTYPE_F32) {
            float* l_data = (float*)labels->data;
            for (size_t i = 0; i < N; ++i) {
                l_data[i] = (float)best_assign[i];
            }
        } else if (labels->dtype == RCRE_DTYPE_I32) {
            int32_t* l_data = (int32_t*)labels->data;
            for (size_t i = 0; i < N; ++i) {
                l_data[i] = (int32_t)best_assign[i];
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_ml_kmeans_fit(const RCRE_Tensor* X, size_t k, size_t max_iter,
                                      RCRE_Tensor* centroids, RCRE_Tensor* labels,
                                      RCRE_Arena* arena) {
    RCRE_KMeansParams params = {
        .max_iter = max_iter ? max_iter : 100,
        .n_init = 10,
        .tol = 1e-4f,
        .seed = 42ULL
    };
    return rcre_ml_kmeans_fit_ex(X, k, &params, centroids, labels, nullptr, arena);
}

RCRE_NODISCARD int rcre_ml_dbscan_fit(const RCRE_Tensor* X,
                                      const RCRE_DBSCANParams* params,
                                      RCRE_Tensor* labels,
                                      size_t* out_num_clusters,
                                      RCRE_Arena* arena) {
    if (!X || !X->data || !params || !labels || !labels->data || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (X->ndim != 2 || labels->shape[0] != X->shape[0]) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t N = X->shape[0];
    size_t D = X->shape[1];
    float eps = params->eps;
    size_t min_samples = params->min_samples;

    if (eps <= 0.0f || min_samples == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    const float* X_data = (const float*)X->data;
    float eps_sq = eps * eps;

    /* Working memory: status/labels, neighbor index buffer, FIFO queue */
    int32_t* tag = (int32_t*)rcre_arena_alloc(arena, N * sizeof(int32_t), 64);
    size_t* neighbor_buf = (size_t*)rcre_arena_alloc(arena, N * sizeof(size_t), 64);
    size_t* queue = (size_t*)rcre_arena_alloc(arena, N * sizeof(size_t), 64);

    if (!tag || !neighbor_buf || !queue) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Initialize all points as UNVISITED (-2) */
    for (size_t i = 0; i < N; ++i) {
        tag[i] = -2;
    }

    int32_t cluster_id = 0;

    for (size_t i = 0; i < N; ++i) {
        if (tag[i] != -2) {
            continue; /* Already processed */
        }

        /* Query neighbors within epsilon */
        const float* p_i = &X_data[i * D];
        size_t neighbor_count = 0;
        for (size_t j = 0; j < N; ++j) {
            if (squared_l2_distance(p_i, &X_data[j * D], D) <= eps_sq) {
                neighbor_buf[neighbor_count++] = j;
            }
        }

        if (neighbor_count < min_samples) {
            tag[i] = -1; /* Mark as noise (may become border point later) */
        } else {
            /* Start a new cluster and expand */
            tag[i] = cluster_id;

            size_t head = 0;
            size_t tail = 0;

            for (size_t n = 0; n < neighbor_count; ++n) {
                size_t pt = neighbor_buf[n];
                if (pt != i) {
                    queue[tail++] = pt;
                }
            }

            while (head < tail) {
                size_t q = queue[head++];
                int32_t q_status = tag[q];

                if (q_status == -1) {
                    /* Noise point reached from core point becomes border point */
                    tag[q] = cluster_id;
                }

                if (q_status != -2) {
                    continue; /* Already visited */
                }

                tag[q] = cluster_id;

                /* Query neighbors of q */
                const float* p_q = &X_data[q * D];
                size_t q_neighbor_count = 0;
                for (size_t j = 0; j < N; ++j) {
                    if (squared_l2_distance(p_q, &X_data[j * D], D) <= eps_sq) {
                        q_neighbor_count++;
                    }
                }

                /* If q is also a core point, expand neighbors */
                if (q_neighbor_count >= min_samples) {
                    for (size_t j = 0; j < N; ++j) {
                        if (squared_l2_distance(p_q, &X_data[j * D], D) <= eps_sq) {
                            if (tag[j] == -2 || tag[j] == -1) {
                                queue[tail++] = j;
                            }
                        }
                    }
                }
            }

            cluster_id++;
        }
    }

    if (out_num_clusters) {
        *out_num_clusters = (size_t)cluster_id;
    }

    /* Output labels */
    if (labels->dtype == RCRE_DTYPE_F32) {
        float* l_data = (float*)labels->data;
        for (size_t i = 0; i < N; ++i) {
            l_data[i] = (float)tag[i];
        }
    } else if (labels->dtype == RCRE_DTYPE_I32) {
        int32_t* l_data = (int32_t*)labels->data;
        for (size_t i = 0; i < N; ++i) {
            l_data[i] = tag[i];
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

RCRE_NODISCARD int rcre_ml_pca_inverse_transform(const RCRE_Tensor* X_proj, const RCRE_Tensor* components,
                                                 const RCRE_Tensor* mean, RCRE_Tensor* X_recon) {
    if (!X_proj || !X_proj->data || !components || !components->data ||
        !mean || !mean->data || !X_recon || !X_recon->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (X_proj->ndim != 2 || components->ndim != 2 || X_recon->ndim != 2) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t N = X_proj->shape[0];
    size_t K = X_proj->shape[1];
    size_t D = components->shape[1];

    if (components->shape[0] != K || X_recon->shape[0] != N || X_recon->shape[1] != D) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t mean_len = 1;
    for (size_t d = 0; d < mean->ndim; ++d) {
        mean_len *= mean->shape[d];
    }
    if (mean_len != D) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* P_data = (const float*)X_proj->data;
    const float* W_data = (const float*)components->data;
    const float* mean_data = (const float*)mean->data;
    float* out_data = (float*)X_recon->data;

    for (size_t i = 0; i < N; ++i) {
        float* row_out = &out_data[i * D];
        /* Initialize row with mean */
        memcpy(row_out, mean_data, D * sizeof(float));

        /* Add linear combination of principal components */
        const float* p_row = &P_data[i * K];
        for (size_t k = 0; k < K; ++k) {
            float alpha = p_row[k];
            if (fabsf(alpha) > 1e-15f) {
                rcre_hal.axpy_f32(D, alpha, &W_data[k * D], row_out);
            }
        }
    }

    return RCRE_OK;
}

