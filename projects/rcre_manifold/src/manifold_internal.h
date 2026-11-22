/**
 * @file manifold_internal.h
 * @brief Internal data structures and algorithms for RCRE Manifold Engine
 */

#ifndef MANIFOLD_INTERNAL_H
#define MANIFOLD_INTERNAL_H

#include "rcre_manifold.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Quadtree node for Barnes-Hut 2D spatial partitioning.
 */
typedef struct QuadNode {
    float x_min, x_max;
    float y_min, y_max;
    float center_x, center_y;
    float mass;
    int point_idx; /* -1 if internal, >= 0 if leaf containing single point, -2 if empty */
    struct QuadNode* children[4]; /* NW, NE, SW, SE */
} QuadNode;

/**
 * @brief Internal Manifold context representation.
 */
struct RCRE_ManifoldContext {
    RCRE_ManifoldParams params;
    size_t n_samples;
    size_t in_dim;
    size_t current_epoch;
    float  current_loss;

    float* Y;          /**< Low-D coordinates [n_samples * out_dim] */
    float* Y_velocity; /**< Momentum buffer [n_samples * out_dim] */
    float* Y_grads;    /**< Gradient buffer [n_samples * out_dim] */

    RCRE_Arena* arena; /**< Allocation arena */
    bool arena_owned;  /**< Context allocated the arena */
    uint64_t rng_state;/**< Marsaglia xorshift64* PRNG state */

    /* High-D CSR Proximity Graph */
    size_t n_edges;
    uint32_t* edge_rows;    /**< [n_samples + 1] row offsets */
    uint32_t* edge_cols;    /**< [n_edges] column targets */
    float*    edge_weights; /**< [n_edges] edge affinities / probabilities */

    /* PaCMAP sampled pairs */
    size_t n_mid_near_per_point;
    size_t n_further_per_point;
    uint32_t* mn_pairs;     /**< [n_samples * n_mid_near_per_point] */
    uint32_t* fp_pairs;     /**< [n_samples * n_further_per_point] */

    /* Quadtree node pool for Barnes-Hut */
    QuadNode* qtree_pool;
    size_t qtree_pool_cap;
    size_t qtree_pool_used;

    /* UMAP low-D curve parameters */
    float umap_a;
    float umap_b;
};

/* Fast PRNG inline helper */
static inline uint64_t manifold_prng_next(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float manifold_prng_uniform(uint64_t* state, float lo, float hi) {
    uint64_t r = manifold_prng_next(state);
    double u = (double)(r >> 11) * (1.0 / 9007199254740992.0);
    return lo + (float)(u * (hi - lo));
}

static inline float manifold_prng_gaussian(uint64_t* state, float mean, float std) {
    float u1 = manifold_prng_uniform(state, 1e-7f, 1.0f);
    float u2 = manifold_prng_uniform(state, 0.0f, 1.0f);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return mean + z * std;
}

#ifdef __cplusplus
}
#endif

#endif /* MANIFOLD_INTERNAL_H */
