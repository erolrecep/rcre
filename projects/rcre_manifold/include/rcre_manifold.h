/**
 * @file rcre_manifold.h
 * @brief RCRE Manifold Learning & Multi-Dimensional Visualization Engine
 *
 * Implements high-performance, zero-dependency dimensionality reduction algorithms:
 *   - UMAP (Uniform Manifold Approximation and Projection)
 *   - Barnes-Hut t-SNE (t-Distributed Stochastic Neighbor Embedding)
 *   - PaCMAP (Pairwise Controlled Manifold Approximation and Projection)
 *   - PCA (Principal Component Analysis linear projection baseline)
 *
 * Designed for real-time interactive untangling, web-based WebAssembly visualizers,
 * and large-scale embedding projections using RCRE Arena allocators and HAL SIMD dispatch.
 *
 * Standard: ISO C23
 */

#ifndef RCRE_MANIFOLD_H
#define RCRE_MANIFOLD_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Manifold learning algorithm selection.
 */
typedef enum {
    RCRE_MANIFOLD_UMAP   = 0, /**< Uniform Manifold Approximation & Projection */
    RCRE_MANIFOLD_TSNE   = 1, /**< Barnes-Hut t-SNE with Early Exaggeration */
    RCRE_MANIFOLD_PACMAP = 2, /**< Pairwise Controlled Manifold Approximation (Wang et al. 2021) */
    RCRE_MANIFOLD_PCA    = 3  /**< Orthogonal Eigenvector PCA Projection */
} RCRE_ManifoldMethod;

/**
 * @brief Hyperparameters for manifold learning optimization.
 */
typedef struct {
    RCRE_ManifoldMethod method;
    size_t   n_neighbors;        /**< Number of nearest neighbors in high-D (default: 15) */
    size_t   out_dim;            /**< Target embedding dimensionality (default: 2) */
    size_t   n_epochs;           /**< Total optimization epochs (default: 500 UMAP/PaCMAP, 1000 t-SNE) */
    float    learning_rate;      /**< Initial optimization learning rate */
    float    min_dist;           /**< Effective minimum distance between points in low-D (UMAP) */
    float    spread;             /**< Effective scale of embedded points (UMAP) */
    float    perplexity;         /**< Target Shannon entropy perplexity (t-SNE, default: 30.0) */
    float    early_exagg;        /**< Early exaggeration factor (t-SNE, default: 12.0) */
    size_t   early_exagg_epochs; /**< Duration of early exaggeration phase (t-SNE, default: 250) */
    size_t   negative_samples;   /**< Negative sampling ratio per positive edge (UMAP, default: 5) */
    int      init_pca;           /**< 1: Initialize with 2D PCA, 0: random Gaussian (default: 1) */
    uint64_t seed;               /**< PRNG seed for stochastic reproducibility */
} RCRE_ManifoldParams;

/**
 * @brief Opaque handle to the manifold projection context.
 */
typedef struct RCRE_ManifoldContext RCRE_ManifoldContext;

/**
 * @brief Returns sensible default hyperparameters for the chosen algorithm.
 */
RCRE_NODISCARD RCRE_ManifoldParams rcre_manifold_params_default(RCRE_ManifoldMethod method);

/**
 * @brief Creates a manifold learning context and prepares high-dimensional graphs/pairs.
 *
 * @param X High-dimensional input data matrix in row-major layout [n_samples * in_dim].
 * @param n_samples Number of data points.
 * @param in_dim Dimensionality of input vectors.
 * @param params Algorithm parameters (or NULL for defaults).
 * @param arena Memory arena for scratch and state allocations (or NULL to use malloc/free).
 * @return Allocated manifold context, or NULL on error.
 */
RCRE_NODISCARD RCRE_ManifoldContext* rcre_manifold_create(const float* X,
                                                          size_t n_samples,
                                                          size_t in_dim,
                                                          const RCRE_ManifoldParams* params,
                                                          RCRE_Arena* arena);

/**
 * @brief Performs a single optimization step (epoch).
 *
 * Ideal for interactive animation frames in WebAssembly Canvas/WebGL and terminal loops.
 *
 * @param ctx Manifold context.
 * @return 0 on success, non-zero if optimization has reached completion or errored.
 */
int rcre_manifold_step(RCRE_ManifoldContext* ctx);

/**
 * @brief Runs all remaining optimization steps until completion.
 *
 * @param ctx Manifold context.
 * @return 0 on success, non-zero on error.
 */
int rcre_manifold_fit(RCRE_ManifoldContext* ctx);

/**
 * @brief Retrieves the low-dimensional embedding coordinates [n_samples * out_dim].
 */
RCRE_NODISCARD const float* rcre_manifold_get_coords(const RCRE_ManifoldContext* ctx);

/**
 * @brief Retrieves the current optimization epoch count.
 */
size_t rcre_manifold_get_epoch(const RCRE_ManifoldContext* ctx);

/**
 * @brief Retrieves the total target optimization epochs.
 */
size_t rcre_manifold_get_total_epochs(const RCRE_ManifoldContext* ctx);

/**
 * @brief Retrieves the current epoch loss or gradient norm approximation.
 */
float rcre_manifold_get_loss(const RCRE_ManifoldContext* ctx);

/**
 * @brief Destroys the manifold context and releases resources.
 */
void rcre_manifold_destroy(RCRE_ManifoldContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_MANIFOLD_H */
