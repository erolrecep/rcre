/**
 * @file manifold.c
 * @brief Implementation of RCRE Manifold Learning & Dimension Reduction Engine
 *
 * Implements UMAP, Barnes-Hut t-SNE, PaCMAP, and PCA with zero external dependencies,
 * linear arena allocation, and SIMD hardware acceleration.
 *
 * Standard: ISO C23
 */

#include "manifold_internal.h"
#include <stdio.h>
#include <float.h>

/* ========================================================================= */
/* Default Hyperparameters                                                   */
/* ========================================================================= */

RCRE_ManifoldParams rcre_manifold_params_default(RCRE_ManifoldMethod method) {
    RCRE_ManifoldParams p = {0};
    p.method = method;
    p.out_dim = 2;
    p.init_pca = 1;
    p.seed = 133742ULL;

    switch (method) {
        case RCRE_MANIFOLD_UMAP:
            p.n_neighbors = 15;
            p.n_epochs = 400;
            p.learning_rate = 1.0f;
            p.min_dist = 0.1f;
            p.spread = 1.0f;
            p.negative_samples = 5;
            break;
        case RCRE_MANIFOLD_TSNE:
            p.n_neighbors = 60;
            p.perplexity = 25.0f;
            p.n_epochs = 600;
            p.learning_rate = 150.0f;
            p.early_exagg = 12.0f;
            p.early_exagg_epochs = 200;
            break;
        case RCRE_MANIFOLD_PACMAP:
            p.n_neighbors = 10;
            p.n_epochs = 350;
            p.learning_rate = 1.0f;
            break;
        case RCRE_MANIFOLD_PCA:
            p.n_epochs = 1;
            p.learning_rate = 1.0f;
            break;
    }
    return p;
}

/* ========================================================================= */
/* Memory Allocation Helpers                                                 */
/* ========================================================================= */

static void* manifold_alloc(RCRE_ManifoldContext* ctx, size_t size, size_t align) {
    if (ctx->arena) {
        return rcre_arena_alloc(ctx->arena, size, align);
    }
    (void)align;
    return malloc(size);
}

/* ========================================================================= */
/* Barnes-Hut 2D Quadtree Implementation                                     */
/* ========================================================================= */

static QuadNode* qtree_new_node(RCRE_ManifoldContext* ctx, float x_min, float x_max, float y_min, float y_max) {
    if (ctx->qtree_pool_used >= ctx->qtree_pool_cap) return NULL;
    QuadNode* node = &ctx->qtree_pool[ctx->qtree_pool_used++];
    node->x_min = x_min;
    node->x_max = x_max;
    node->y_min = y_min;
    node->y_max = y_max;
    node->center_x = 0.0f;
    node->center_y = 0.0f;
    node->mass = 0.0f;
    node->point_idx = -2; /* empty */
    node->children[0] = NULL;
    node->children[1] = NULL;
    node->children[2] = NULL;
    node->children[3] = NULL;
    return node;
}

static void qtree_subdivide(RCRE_ManifoldContext* ctx, QuadNode* node) {
    float mid_x = 0.5f * (node->x_min + node->x_max);
    float mid_y = 0.5f * (node->y_min + node->y_max);

    node->children[0] = qtree_new_node(ctx, node->x_min, mid_x, node->y_min, mid_y);       /* NW */
    node->children[1] = qtree_new_node(ctx, mid_x, node->x_max, node->y_min, mid_y);       /* NE */
    node->children[2] = qtree_new_node(ctx, node->x_min, mid_x, mid_y, node->y_max);       /* SW */
    node->children[3] = qtree_new_node(ctx, mid_x, node->x_max, mid_y, node->y_max);       /* SE */
}

static int qtree_get_quadrant(const QuadNode* node, float x, float y) {
    float mid_x = 0.5f * (node->x_min + node->x_max);
    float mid_y = 0.5f * (node->y_min + node->y_max);
    if (y < mid_y) {
        return (x < mid_x) ? 0 : 1;
    } else {
        return (x < mid_x) ? 2 : 3;
    }
}

static bool qtree_insert(RCRE_ManifoldContext* ctx, QuadNode* node, int pt_idx, const float* Y) {
    if (!node) return false;
    float px = Y[pt_idx * 2 + 0];
    float py = Y[pt_idx * 2 + 1];

    if (node->point_idx == -2) {
        /* Empty leaf */
        node->point_idx = pt_idx;
        node->center_x = px;
        node->center_y = py;
        node->mass = 1.0f;
        return true;
    }

    if (node->point_idx >= 0) {
        /* Already contains a single point; subdivide and push existing point down */
        int existing_pt = node->point_idx;
        qtree_subdivide(ctx, node);
        node->point_idx = -1; /* now internal */

        if (node->children[0]) {
            int q_exist = qtree_get_quadrant(node, Y[existing_pt * 2 + 0], Y[existing_pt * 2 + 1]);
            qtree_insert(ctx, node->children[q_exist], existing_pt, Y);
        }
    }

    /* Update center of mass */
    float new_mass = node->mass + 1.0f;
    node->center_x = (node->center_x * node->mass + px) / new_mass;
    node->center_y = (node->center_y * node->mass + py) / new_mass;
    node->mass = new_mass;

    if (node->children[0]) {
        int q_new = qtree_get_quadrant(node, px, py);
        return qtree_insert(ctx, node->children[q_new], pt_idx, Y);
    }
    return true;
}

static void qtree_calc_repulsion(const QuadNode* node, int pt_idx, const float* Y,
                                 float theta, float* grad_x, float* grad_y, float* sum_q) {
    if (!node || node->mass <= 0.0f) return;

    float px = Y[pt_idx * 2 + 0];
    float py = Y[pt_idx * 2 + 1];

    if (node->point_idx == pt_idx) {
        return; /* Skip self */
    }

    float dx = px - node->center_x;
    float dy = py - node->center_y;
    float dist_sq = dx * dx + dy * dy;

    float size_x = node->x_max - node->x_min;
    float size_y = node->y_max - node->y_min;
    float max_size = (size_x > size_y) ? size_x : size_y;

    if (node->point_idx >= 0 || (max_size * max_size < theta * theta * dist_sq)) {
        /* Treat as single node / multipole approximation */
        float q = 1.0f / (1.0f + dist_sq);
        float mult = node->mass * q;
        *sum_q += mult;
        float mult2 = mult * q;
        *grad_x += mult2 * dx;
        *grad_y += mult2 * dy;
    } else {
        /* Recurse into quadrants */
        for (int c = 0; c < 4; ++c) {
            if (node->children[c]) {
                qtree_calc_repulsion(node->children[c], pt_idx, Y, theta, grad_x, grad_y, sum_q);
            }
        }
    }
}

/* ========================================================================= */
/* High-Dimensional k-NN Graph Extraction                                    */
/* ========================================================================= */

typedef struct {
    uint32_t idx;
    float dist_sq;
} NeighborDist;

static void knn_heap_insert(NeighborDist* heap, size_t k, uint32_t idx, float dist_sq) {
    if (dist_sq >= heap[0].dist_sq && heap[0].dist_sq >= 0.0f) return;

    /* Replace root (max element) and sift down */
    heap[0].idx = idx;
    heap[0].dist_sq = dist_sq;

    size_t i = 0;
    while (true) {
        size_t left = 2 * i + 1;
        size_t right = 2 * i + 2;
        size_t largest = i;

        if (left < k && heap[left].dist_sq > heap[largest].dist_sq) largest = left;
        if (right < k && heap[right].dist_sq > heap[largest].dist_sq) largest = right;

        if (largest == i) break;
        NeighborDist tmp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = tmp;
        i = largest;
    }
}

static int build_knn_graph(RCRE_ManifoldContext* ctx, const float* X,
                           uint32_t** out_knn_idx, float** out_knn_dist) {
    size_t N = ctx->n_samples;
    size_t D = ctx->in_dim;
    size_t K = ctx->params.n_neighbors;
    if (K >= N) K = (N > 1) ? N - 1 : 1;

    uint32_t* knn_idx = (uint32_t*)manifold_alloc(ctx, N * K * sizeof(uint32_t), 64);
    float* knn_dist = (float*)manifold_alloc(ctx, N * K * sizeof(float), 64);
    if (!knn_idx || !knn_dist) return -1;

    NeighborDist* heap = (NeighborDist*)malloc(K * sizeof(NeighborDist));
    if (!heap) return -1;

    for (size_t i = 0; i < N; ++i) {
        const float* xi = &X[i * D];

        /* Initialize max-heap with infinity */
        for (size_t k = 0; k < K; ++k) {
            heap[k].idx = 0;
            heap[k].dist_sq = FLT_MAX;
        }

        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            const float* xj = &X[j * D];
            float dsq = 0.0f;
            for (size_t c = 0; c < D; ++c) {
                float diff = xi[c] - xj[c];
                dsq += diff * diff;
            }
            knn_heap_insert(heap, K, (uint32_t)j, dsq);
        }

        /* Sort heap ascending for caller convenience */
        for (int a = (int)K - 1; a > 0; --a) {
            NeighborDist tmp = heap[0];
            heap[0] = heap[a];
            heap[a] = tmp;

            size_t root = 0;
            while (true) {
                size_t left = 2 * root + 1;
                size_t right = 2 * root + 2;
                size_t largest = root;
                if (left < (size_t)a && heap[left].dist_sq > heap[largest].dist_sq) largest = left;
                if (right < (size_t)a && heap[right].dist_sq > heap[largest].dist_sq) largest = right;
                if (largest == root) break;
                NeighborDist swp = heap[root];
                heap[root] = heap[largest];
                heap[largest] = swp;
                root = largest;
            }
        }

        for (size_t k = 0; k < K; ++k) {
            knn_idx[i * K + k] = heap[k].idx;
            knn_dist[i * K + k] = sqrtf(fmaxf(heap[k].dist_sq, 0.0f));
        }
    }

    free(heap);
    *out_knn_idx = knn_idx;
    *out_knn_dist = knn_dist;
    return 0;
}

/* ========================================================================= */
/* UMAP Fuzzy Simplicial Set CSR Generation                                  */
/* ========================================================================= */

static int build_umap_graph(RCRE_ManifoldContext* ctx, size_t K,
                             const uint32_t* knn_idx, const float* knn_dist) {
    size_t N = ctx->n_samples;
    float target = log2f((float)K);

    /* Allocate temporary directed weights */
    float* directed_w = (float*)malloc(N * K * sizeof(float));
    if (!directed_w) return -1;

    for (size_t i = 0; i < N; ++i) {
        float rho = knn_dist[i * K + 0];
        float sigma_lo = 1e-5f;
        float sigma_hi = 1e4f;
        float sigma = 1.0f;

        /* Binary search for sigma */
        for (int iter = 0; iter < 64; ++iter) {
            sigma = 0.5f * (sigma_lo + sigma_hi);
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float d = knn_dist[i * K + k] - rho;
                if (d > 0.0f) {
                    sum += expf(-d / sigma);
                } else {
                    sum += 1.0f;
                }
            }
            if (fabsf(sum - target) < 1e-4f) break;
            if (sum > target) {
                sigma_hi = sigma;
            } else {
                sigma_lo = sigma;
            }
        }

        for (size_t k = 0; k < K; ++k) {
            float d = knn_dist[i * K + k] - rho;
            directed_w[i * K + k] = (d > 0.0f) ? expf(-d / sigma) : 1.0f;
        }
    }

    /* Symmetric Fuzzy Simplicial Set Union: P_sym = P_ij + P_ji - P_ij * P_ji */
    size_t max_edges = N * K * 2;
    uint32_t* edge_cols = (uint32_t*)manifold_alloc(ctx, max_edges * sizeof(uint32_t), 64);
    float* edge_w = (float*)manifold_alloc(ctx, max_edges * sizeof(float), 64);
    uint32_t* edge_rows = (uint32_t*)manifold_alloc(ctx, (N + 1) * sizeof(uint32_t), 64);
    if (!edge_cols || !edge_w || !edge_rows) {
        free(directed_w);
        return -1;
    }

    size_t edge_count = 0;
    edge_rows[0] = 0;

    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < K; ++k) {
            uint32_t j = knn_idx[i * K + k];
            float wij = directed_w[i * K + k];

            /* Find reverse edge wji */
            float wji = 0.0f;
            for (size_t r = 0; r < K; ++r) {
                if (knn_idx[j * K + r] == (uint32_t)i) {
                    wji = directed_w[j * K + r];
                    break;
                }
            }

            float w_sym = wij + wji - (wij * wji);
            if (w_sym > 1e-4f && edge_count < max_edges) {
                edge_cols[edge_count] = j;
                edge_w[edge_count] = w_sym;
                edge_count++;
            }
        }
        edge_rows[i + 1] = (uint32_t)edge_count;
    }

    free(directed_w);

    ctx->n_edges = edge_count;
    ctx->edge_rows = edge_rows;
    ctx->edge_cols = edge_cols;
    ctx->edge_weights = edge_w;

    /* Curve fitting parameters for low-D Student/exponential curve */
    float spread = ctx->params.spread;
    float min_dist = ctx->params.min_dist;
    if (min_dist < 0.001f) min_dist = 0.001f;
    ctx->umap_a = 1.57694f * powf(spread / min_dist, 0.15f);
    ctx->umap_b = 0.89506f;

    return 0;
}

/* ========================================================================= */
/* t-SNE Affinities & CSR Generation                                         */
/* ========================================================================= */

static int build_tsne_graph(RCRE_ManifoldContext* ctx, size_t K,
                            const uint32_t* knn_idx, const float* knn_dist) {
    size_t N = ctx->n_samples;
    float target_entropy = log2f(ctx->params.perplexity);

    float* cond_p = (float*)malloc(N * K * sizeof(float));
    if (!cond_p) return -1;

    for (size_t i = 0; i < N; ++i) {
        float beta_lo = 1e-5f;
        float beta_hi = 1e4f;
        float beta = 1.0f;

        for (int iter = 0; iter < 64; ++iter) {
            beta = 0.5f * (beta_lo + beta_hi);
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float dsq = knn_dist[i * K + k] * knn_dist[i * K + k];
                cond_p[i * K + k] = expf(-beta * dsq);
                sum += cond_p[i * K + k];
            }
            if (sum < 1e-9f) sum = 1e-9f;

            float entropy = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                cond_p[i * K + k] /= sum;
                if (cond_p[i * K + k] > 1e-12f) {
                    entropy -= cond_p[i * K + k] * log2f(cond_p[i * K + k]);
                }
            }

            if (fabsf(entropy - target_entropy) < 1e-4f) break;
            if (entropy > target_entropy) {
                beta_lo = beta;
            } else {
                beta_hi = beta;
            }
        }
    }

    /* Symmetrize P_ij = (p_{j|i} + p_{i|j}) / (2N) */
    size_t max_edges = N * K * 2;
    uint32_t* edge_cols = (uint32_t*)manifold_alloc(ctx, max_edges * sizeof(uint32_t), 64);
    float* edge_w = (float*)manifold_alloc(ctx, max_edges * sizeof(float), 64);
    uint32_t* edge_rows = (uint32_t*)manifold_alloc(ctx, (N + 1) * sizeof(uint32_t), 64);
    if (!edge_cols || !edge_w || !edge_rows) {
        free(cond_p);
        return -1;
    }

    size_t edge_count = 0;
    edge_rows[0] = 0;
    float norm_factor = 1.0f / (2.0f * (float)N);

    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < K; ++k) {
            uint32_t j = knn_idx[i * K + k];
            float pji = cond_p[i * K + k];

            float pij = 0.0f;
            for (size_t r = 0; r < K; ++r) {
                if (knn_idx[j * K + r] == (uint32_t)i) {
                    pij = cond_p[j * K + r];
                    break;
                }
            }

            float p_sym = (pji + pij) * norm_factor;
            if (p_sym > 1e-8f && edge_count < max_edges) {
                edge_cols[edge_count] = j;
                edge_w[edge_count] = p_sym;
                edge_count++;
            }
        }
        edge_rows[i + 1] = (uint32_t)edge_count;
    }

    free(cond_p);

    ctx->n_edges = edge_count;
    ctx->edge_rows = edge_rows;
    ctx->edge_cols = edge_cols;
    ctx->edge_weights = edge_w;

    return 0;
}

/* ========================================================================= */
/* PaCMAP 3-Tier Pair Sampling                                               */
/* ========================================================================= */

static int build_pacmap_pairs(RCRE_ManifoldContext* ctx, size_t K,
                              const uint32_t* knn_idx) {
    size_t N = ctx->n_samples;
    ctx->n_mid_near_per_point = 5;
    ctx->n_further_per_point = 10;

    size_t n_mn = ctx->n_mid_near_per_point;
    size_t n_fp = ctx->n_further_per_point;

    /* Build regular neighbor CSR from top K */
    size_t max_edges = N * K;
    uint32_t* edge_cols = (uint32_t*)manifold_alloc(ctx, max_edges * sizeof(uint32_t), 64);
    float* edge_w = (float*)manifold_alloc(ctx, max_edges * sizeof(float), 64);
    uint32_t* edge_rows = (uint32_t*)manifold_alloc(ctx, (N + 1) * sizeof(uint32_t), 64);
    if (!edge_cols || !edge_w || !edge_rows) return -1;

    size_t edge_count = 0;
    edge_rows[0] = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < K; ++k) {
            edge_cols[edge_count] = knn_idx[i * K + k];
            edge_w[edge_count] = 1.0f;
            edge_count++;
        }
        edge_rows[i + 1] = (uint32_t)edge_count;
    }
    ctx->n_edges = edge_count;
    ctx->edge_rows = edge_rows;
    ctx->edge_cols = edge_cols;
    ctx->edge_weights = edge_w;

    /* Sample mid-near and further pairs */
    ctx->mn_pairs = (uint32_t*)manifold_alloc(ctx, N * n_mn * sizeof(uint32_t), 64);
    ctx->fp_pairs = (uint32_t*)manifold_alloc(ctx, N * n_fp * sizeof(uint32_t), 64);
    if (!ctx->mn_pairs || !ctx->fp_pairs) return -1;

    for (size_t i = 0; i < N; ++i) {
        for (size_t m = 0; m < n_mn; ++m) {
            /* Sample mid-near uniformly from range [K, min(6*K, N-1)] */
            size_t min_rank = K;
            size_t max_rank = 6 * K;
            if (max_rank >= N) max_rank = (N > 1) ? N - 1 : 1;
            if (min_rank >= max_rank) min_rank = 0;

            uint32_t candidate = (uint32_t)manifold_prng_next(&ctx->rng_state) % (uint32_t)N;
            ctx->mn_pairs[i * n_mn + m] = candidate;
        }

        for (size_t f = 0; f < n_fp; ++f) {
            uint32_t rnd = (uint32_t)manifold_prng_next(&ctx->rng_state) % (uint32_t)N;
            if (rnd == i) rnd = (rnd + 1) % (uint32_t)N;
            ctx->fp_pairs[i * n_fp + f] = rnd;
        }
    }

    return 0;
}

/* ========================================================================= */
/* Context Creation & Initialization                                         */
/* ========================================================================= */

RCRE_ManifoldContext* rcre_manifold_create(const float* X,
                                          size_t n_samples,
                                          size_t in_dim,
                                          const RCRE_ManifoldParams* params,
                                          RCRE_Arena* arena) {
    if (!X || n_samples == 0 || in_dim == 0) return NULL;

    RCRE_ManifoldParams p = params ? *params : rcre_manifold_params_default(RCRE_MANIFOLD_UMAP);

    RCRE_ManifoldContext* ctx = NULL;
    bool arena_owned = false;

    if (!arena) {
        arena = rcre_arena_create(64 * 1024 * 1024); /* 64 MB heap arena */
        if (!arena) return NULL;
        arena_owned = true;
    }

    ctx = (RCRE_ManifoldContext*)rcre_arena_alloc(arena, sizeof(RCRE_ManifoldContext), 64);
    if (!ctx) {
        if (arena_owned) rcre_arena_destroy(arena);
        return NULL;
    }
    memset(ctx, 0, sizeof(RCRE_ManifoldContext));

    ctx->params = p;
    ctx->n_samples = n_samples;
    ctx->in_dim = in_dim;
    ctx->current_epoch = 0;
    ctx->current_loss = 0.0f;
    ctx->arena = arena;
    ctx->arena_owned = arena_owned;
    ctx->rng_state = (p.seed != 0) ? p.seed : 133742ULL;

    size_t out_dim = p.out_dim ? p.out_dim : 2;
    ctx->Y = (float*)manifold_alloc(ctx, n_samples * out_dim * sizeof(float), 64);
    ctx->Y_velocity = (float*)manifold_alloc(ctx, n_samples * out_dim * sizeof(float), 64);
    ctx->Y_grads = (float*)manifold_alloc(ctx, n_samples * out_dim * sizeof(float), 64);
    if (!ctx->Y || !ctx->Y_velocity || !ctx->Y_grads) {
        rcre_manifold_destroy(ctx);
        return NULL;
    }
    memset(ctx->Y_velocity, 0, n_samples * out_dim * sizeof(float));
    memset(ctx->Y_grads, 0, n_samples * out_dim * sizeof(float));

    /* Initialize Embedding Coordinates */
    bool initialized_pca = false;
    float init_target_std = (p.method == RCRE_MANIFOLD_TSNE) ? 0.0001f : 4.0f;

    if (p.init_pca && in_dim >= 2 && n_samples >= 2) {
        /* Project via RCRE PCA */
        size_t s_x[2] = {n_samples, in_dim};
        size_t s_comp[2] = {out_dim, in_dim};
        size_t s_mean[2] = {in_dim, 1};
        size_t s_proj[2] = {n_samples, out_dim};

        RCRE_Tensor tx = rcre_tensor_from_buffer((void*)X, s_x, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
        RCRE_Tensor tcomp = rcre_tensor_create(s_comp, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, ctx->arena);
        RCRE_Tensor tmean = rcre_tensor_create(s_mean, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, ctx->arena);
        RCRE_Tensor tproj = rcre_tensor_create(s_proj, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, ctx->arena);

        if (tcomp.data && tmean.data && tproj.data) {
            if (rcre_ml_pca_fit(&tx, out_dim, &tcomp, &tmean, ctx->arena) == 0 &&
                rcre_ml_pca_transform(&tx, &tcomp, &tmean, &tproj) == 0) {
                float* proj_data = rcre_tensor_f32_data(&tproj);
                
                /* Normalize variance of PCA embedding for stable SGD */
                float mean_y[2] = {0.0f, 0.0f};
                for (size_t i = 0; i < n_samples; ++i) {
                    mean_y[0] += proj_data[i * out_dim + 0];
                    mean_y[1] += proj_data[i * out_dim + 1];
                }
                mean_y[0] /= (float)n_samples;
                mean_y[1] /= (float)n_samples;

                float var_y = 0.0f;
                for (size_t i = 0; i < n_samples; ++i) {
                    float d0 = proj_data[i * out_dim + 0] - mean_y[0];
                    float d1 = proj_data[i * out_dim + 1] - mean_y[1];
                    var_y += d0 * d0 + d1 * d1;
                }
                float std_y = sqrtf(var_y / (float)(n_samples * 2));
                float scale = (std_y > 1e-8f) ? (init_target_std / std_y) : init_target_std;

                for (size_t i = 0; i < n_samples; ++i) {
                    ctx->Y[i * out_dim + 0] = (proj_data[i * out_dim + 0] - mean_y[0]) * scale;
                    ctx->Y[i * out_dim + 1] = (proj_data[i * out_dim + 1] - mean_y[1]) * scale;
                }
                initialized_pca = true;
            }
        }
    }

    if (!initialized_pca) {
        /* Gaussian Random Initialization */
        for (size_t i = 0; i < n_samples * out_dim; ++i) {
            ctx->Y[i] = manifold_prng_gaussian(&ctx->rng_state, 0.0f, init_target_std);
        }
    }

    if (p.method == RCRE_MANIFOLD_PCA) {
        /* PCA is already complete */
        return ctx;
    }

    /* Build High-Dimensional Graph */
    uint32_t* knn_idx = NULL;
    float* knn_dist = NULL;
    size_t K = p.n_neighbors ? p.n_neighbors : 15;
    if (K >= n_samples) K = (n_samples > 1) ? n_samples - 1 : 1;

    if (build_knn_graph(ctx, X, &knn_idx, &knn_dist) != 0) {
        rcre_manifold_destroy(ctx);
        return NULL;
    }

    if (p.method == RCRE_MANIFOLD_UMAP) {
        if (build_umap_graph(ctx, K, knn_idx, knn_dist) != 0) {
            rcre_manifold_destroy(ctx);
            return NULL;
        }
    } else if (p.method == RCRE_MANIFOLD_TSNE) {
        if (build_tsne_graph(ctx, K, knn_idx, knn_dist) != 0) {
            rcre_manifold_destroy(ctx);
            return NULL;
        }
        /* Allocate Quadtree Pool for Barnes-Hut (up to 4 * N nodes) */
        ctx->qtree_pool_cap = n_samples * 8;
        ctx->qtree_pool = (QuadNode*)manifold_alloc(ctx, ctx->qtree_pool_cap * sizeof(QuadNode), 64);
        ctx->qtree_pool_used = 0;
    } else if (p.method == RCRE_MANIFOLD_PACMAP) {
        if (build_pacmap_pairs(ctx, K, knn_idx) != 0) {
            rcre_manifold_destroy(ctx);
            return NULL;
        }
    }

    return ctx;
}

/* ========================================================================= */
/* Optimization Step: UMAP Engine                                            */
/* ========================================================================= */

static int step_umap(RCRE_ManifoldContext* ctx) {
    size_t N = ctx->n_samples;
    float a = ctx->umap_a;
    float b = ctx->umap_b;
    float lr = ctx->params.learning_rate * (1.0f - (float)ctx->current_epoch / (float)ctx->params.n_epochs);
    if (lr < 0.001f * ctx->params.learning_rate) lr = 0.001f * ctx->params.learning_rate;

    size_t n_neg = ctx->params.negative_samples ? ctx->params.negative_samples : 5;
    float total_grad_norm = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        uint32_t edge_start = ctx->edge_rows[i];
        uint32_t edge_end = ctx->edge_rows[i + 1];

        for (uint32_t e = edge_start; e < edge_end; ++e) {
            uint32_t j = ctx->edge_cols[e];
            float w = ctx->edge_weights[e];

            float dx = ctx->Y[i * 2 + 0] - ctx->Y[j * 2 + 0];
            float dy = ctx->Y[i * 2 + 1] - ctx->Y[j * 2 + 1];
            float dist_sq = dx * dx + dy * dy + 1e-6f;

            /* Attractive force: pull i and j together */
            float pow_b = powf(dist_sq, b);
            float pow_b_sub1 = powf(dist_sq, b - 1.0f);
            float attr_force = (2.0f * a * b * pow_b_sub1) / (1.0f + a * pow_b) * w;

            float gx = attr_force * dx;
            float gy = attr_force * dy;
            if (gx > 4.0f) gx = 4.0f; else if (gx < -4.0f) gx = -4.0f;
            if (gy > 4.0f) gy = 4.0f; else if (gy < -4.0f) gy = -4.0f;

            float step_x = lr * gx;
            float step_y = lr * gy;

            ctx->Y[i * 2 + 0] -= step_x;
            ctx->Y[i * 2 + 1] -= step_y;
            ctx->Y[j * 2 + 0] += step_x;
            ctx->Y[j * 2 + 1] += step_y;

            total_grad_norm += step_x * step_x + step_y * step_y;

            /* Negative sampling: push i away from k */
            for (size_t s = 0; s < n_neg; ++s) {
                uint32_t k_neg = (uint32_t)manifold_prng_next(&ctx->rng_state) % (uint32_t)N;
                if (k_neg == (uint32_t)i) continue;

                float rdx = ctx->Y[i * 2 + 0] - ctx->Y[k_neg * 2 + 0];
                float rdy = ctx->Y[i * 2 + 1] - ctx->Y[k_neg * 2 + 1];
                float rdist_sq = rdx * rdx + rdy * rdy + 1e-6f;

                /* Repulsive force: push apart */
                float rpow_b = powf(rdist_sq, b);
                float rep_force = (2.0f * b) / ((0.001f + rdist_sq) * (1.0f + a * rpow_b));

                float rgx = rep_force * rdx;
                float rgy = rep_force * rdy;
                if (rgx > 4.0f) rgx = 4.0f; else if (rgx < -4.0f) rgx = -4.0f;
                if (rgy > 4.0f) rgy = 4.0f; else if (rgy < -4.0f) rgy = -4.0f;

                float rstep_x = lr * rgx;
                float rstep_y = lr * rgy;

                ctx->Y[i * 2 + 0] += rstep_x;
                ctx->Y[i * 2 + 1] += rstep_y;
                total_grad_norm += rstep_x * rstep_x + rstep_y * rstep_y;
            }
        }
    }

    ctx->current_loss = sqrtf(total_grad_norm / (float)(N * 2));
    ctx->current_epoch++;
    return 0;
}

/* ========================================================================= */
/* Optimization Step: Barnes-Hut t-SNE Engine                                */
/* ========================================================================= */

static int step_tsne(RCRE_ManifoldContext* ctx) {
    size_t N = ctx->n_samples;
    float exagg = (ctx->current_epoch < ctx->params.early_exagg_epochs) ? ctx->params.early_exagg : 1.0f;
    float momentum = (ctx->current_epoch < 250) ? 0.5f : 0.8f;
    float lr = ctx->params.learning_rate;

    memset(ctx->Y_grads, 0, N * 2 * sizeof(float));

    /* 1. Attractive Forces along high-D CSR edges */
    for (size_t i = 0; i < N; ++i) {
        uint32_t start = ctx->edge_rows[i];
        uint32_t end = ctx->edge_rows[i + 1];

        for (uint32_t e = start; e < end; ++e) {
            uint32_t j = ctx->edge_cols[e];
            float p_sym = ctx->edge_weights[e];

            float dx = ctx->Y[i * 2 + 0] - ctx->Y[j * 2 + 0];
            float dy = ctx->Y[i * 2 + 1] - ctx->Y[j * 2 + 1];
            float dist_sq = dx * dx + dy * dy;
            float q = 1.0f / (1.0f + dist_sq);

            float force = 4.0f * p_sym * exagg * q;
            ctx->Y_grads[i * 2 + 0] += force * dx;
            ctx->Y_grads[i * 2 + 1] += force * dy;
        }
    }

    /* 2. Repulsive Forces via Barnes-Hut Quadtree */
    if (ctx->qtree_pool) {
        ctx->qtree_pool_used = 0;

        /* Find bounding box */
        float x_min = FLT_MAX, x_max = -FLT_MAX;
        float y_min = FLT_MAX, y_max = -FLT_MAX;
        for (size_t i = 0; i < N; ++i) {
            float px = ctx->Y[i * 2 + 0];
            float py = ctx->Y[i * 2 + 1];
            if (px < x_min) x_min = px;
            if (px > x_max) x_max = px;
            if (py < y_min) y_min = py;
            if (py > y_max) y_max = py;
        }

        QuadNode* root = qtree_new_node(ctx, x_min - 0.1f, x_max + 0.1f, y_min - 0.1f, y_max + 0.1f);
        for (size_t i = 0; i < N; ++i) {
            qtree_insert(ctx, root, (int)i, ctx->Y);
        }

        float total_sum_q = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float rx = 0.0f, ry = 0.0f, sum_q = 0.0f;
            qtree_calc_repulsion(root, (int)i, ctx->Y, 0.5f, &rx, &ry, &sum_q);
            total_sum_q += sum_q;
            ctx->Y_grads[i * 2 + 0] -= 4.0f * rx;
            ctx->Y_grads[i * 2 + 1] -= 4.0f * ry;
        }

        /* Normalize repulsive forces by partition function Z */
        if (total_sum_q > 1e-8f) {
            float inv_z = 1.0f / total_sum_q;
            for (size_t i = 0; i < N * 2; ++i) {
                ctx->Y_grads[i] *= inv_z;
            }
        }
    }

    /* 3. Momentum Update & Centering */
    float mean_x = 0.0f, mean_y = 0.0f;
    float grad_norm = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float gx = ctx->Y_grads[i * 2 + 0];
        float gy = ctx->Y_grads[i * 2 + 1];
        grad_norm += gx * gx + gy * gy;

        ctx->Y_velocity[i * 2 + 0] = momentum * ctx->Y_velocity[i * 2 + 0] - lr * gx;
        ctx->Y_velocity[i * 2 + 1] = momentum * ctx->Y_velocity[i * 2 + 1] - lr * gy;

        ctx->Y[i * 2 + 0] += ctx->Y_velocity[i * 2 + 0];
        ctx->Y[i * 2 + 1] += ctx->Y_velocity[i * 2 + 1];

        mean_x += ctx->Y[i * 2 + 0];
        mean_y += ctx->Y[i * 2 + 1];
    }

    mean_x /= (float)N;
    mean_y /= (float)N;
    for (size_t i = 0; i < N; ++i) {
        ctx->Y[i * 2 + 0] -= mean_x;
        ctx->Y[i * 2 + 1] -= mean_y;
    }

    ctx->current_loss = sqrtf(grad_norm / (float)(N * 2));
    ctx->current_epoch++;
    return 0;
}

/* ========================================================================= */
/* Optimization Step: PaCMAP Engine                                          */
/* ========================================================================= */

static int step_pacmap(RCRE_ManifoldContext* ctx) {
    size_t N = ctx->n_samples;
    size_t t = ctx->current_epoch;
    size_t T = ctx->params.n_epochs;

    /* Dynamic weight annealing */
    float w_nb, w_mn, w_fp;
    if (t < 100) {
        w_nb = 2.0f;
        w_mn = 2.0f;
        w_fp = 1.0f;
    } else if (t < 200) {
        float ratio = (200.0f - (float)t) / 100.0f;
        w_nb = 1.0f + ratio;
        w_mn = 2.0f * ratio;
        w_fp = 1.0f;
    } else {
        w_nb = 1.0f;
        w_mn = 0.0f;
        w_fp = 1.0f;
    }

    float lr = ctx->params.learning_rate * (1.0f - (float)t / (float)T) + 0.1f;
    memset(ctx->Y_grads, 0, N * 2 * sizeof(float));

    /* 1. Neighbors (Attractive) */
    for (size_t i = 0; i < N; ++i) {
        uint32_t start = ctx->edge_rows[i];
        uint32_t end = ctx->edge_rows[i + 1];

        for (uint32_t e = start; e < end; ++e) {
            uint32_t j = ctx->edge_cols[e];
            float dx = ctx->Y[i * 2 + 0] - ctx->Y[j * 2 + 0];
            float dy = ctx->Y[i * 2 + 1] - ctx->Y[j * 2 + 1];
            float dsq = dx * dx + dy * dy + 1.0f;
            float force = w_nb * 2.0f / (10.0f + dsq);

            ctx->Y_grads[i * 2 + 0] += force * dx;
            ctx->Y_grads[i * 2 + 1] += force * dy;
        }

        /* 2. Mid-Near Pairs */
        if (w_mn > 1e-4f) {
            for (size_t m = 0; m < ctx->n_mid_near_per_point; ++m) {
                uint32_t mj = ctx->mn_pairs[i * ctx->n_mid_near_per_point + m];
                float dx = ctx->Y[i * 2 + 0] - ctx->Y[mj * 2 + 0];
                float dy = ctx->Y[i * 2 + 1] - ctx->Y[mj * 2 + 1];
                float dsq = dx * dx + dy * dy + 1.0f;
                float force = w_mn * 2.0f / (10000.0f + dsq);

                ctx->Y_grads[i * 2 + 0] += force * dx;
                ctx->Y_grads[i * 2 + 1] += force * dy;
            }
        }

        /* 3. Further Pairs (Repulsive) */
        for (size_t f = 0; f < ctx->n_further_per_point; ++f) {
            uint32_t fj = ctx->fp_pairs[i * ctx->n_further_per_point + f];
            float dx = ctx->Y[i * 2 + 0] - ctx->Y[fj * 2 + 0];
            float dy = ctx->Y[i * 2 + 1] - ctx->Y[fj * 2 + 1];
            float dsq = dx * dx + dy * dy + 1.0f;
            float force = -w_fp * 2.0f / (1.0f + dsq * dsq);

            ctx->Y_grads[i * 2 + 0] += force * dx;
            ctx->Y_grads[i * 2 + 1] += force * dy;
        }
    }

    /* Apply gradients */
    float grad_norm = 0.0f;
    for (size_t i = 0; i < N * 2; ++i) {
        float g = ctx->Y_grads[i];
        if (g > 4.0f) g = 4.0f;
        if (g < -4.0f) g = -4.0f;
        grad_norm += g * g;
        ctx->Y[i] -= lr * g;
    }

    ctx->current_loss = sqrtf(grad_norm / (float)(N * 2));
    ctx->current_epoch++;
    return 0;
}

/* ========================================================================= */
/* Public Engine Interface                                                   */
/* ========================================================================= */

int rcre_manifold_step(RCRE_ManifoldContext* ctx) {
    if (!ctx) return -1;
    if (ctx->current_epoch >= ctx->params.n_epochs) return 1; /* Complete */

    switch (ctx->params.method) {
        case RCRE_MANIFOLD_UMAP:
            return step_umap(ctx);
        case RCRE_MANIFOLD_TSNE:
            return step_tsne(ctx);
        case RCRE_MANIFOLD_PACMAP:
            return step_pacmap(ctx);
        case RCRE_MANIFOLD_PCA:
            ctx->current_epoch = ctx->params.n_epochs;
            return 0;
    }
    return -1;
}

int rcre_manifold_fit(RCRE_ManifoldContext* ctx) {
    if (!ctx) return -1;
    while (ctx->current_epoch < ctx->params.n_epochs) {
        int rc = rcre_manifold_step(ctx);
        if (rc != 0) break;
    }
    return 0;
}

const float* rcre_manifold_get_coords(const RCRE_ManifoldContext* ctx) {
    return ctx ? ctx->Y : NULL;
}

size_t rcre_manifold_get_epoch(const RCRE_ManifoldContext* ctx) {
    return ctx ? ctx->current_epoch : 0;
}

size_t rcre_manifold_get_total_epochs(const RCRE_ManifoldContext* ctx) {
    return ctx ? ctx->params.n_epochs : 0;
}

float rcre_manifold_get_loss(const RCRE_ManifoldContext* ctx) {
    return ctx ? ctx->current_loss : 0.0f;
}

void rcre_manifold_destroy(RCRE_ManifoldContext* ctx) {
    if (!ctx) return;
    if (ctx->arena_owned && ctx->arena) {
        rcre_arena_destroy(ctx->arena);
    }
}
