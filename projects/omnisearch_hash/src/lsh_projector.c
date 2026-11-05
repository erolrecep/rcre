/**
 * @file lsh_projector.c
 * @brief Locality-Sensitive Hashing (LSH) Random Hyperplane Projections
 */

#include "omnisearch.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 64-bit SplitMix PRNG */
static inline uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* Uniform float in (0, 1] */
static inline float rand_uniform(uint64_t* state) {
    uint64_t r = splitmix64(state);
    return ((float)(r >> 11) + 1.0f) * (1.0f / 9007199254740992.0f);
}

/* Box-Muller Gaussian standard normal N(0, 1) */
static inline float rand_normal(uint64_t* state) {
    float u1 = rand_uniform(state);
    float u2 = rand_uniform(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

RCRE_NODISCARD int omni_lsh_init_hyperplanes(float* hyperplanes, size_t num_hyperplanes, 
                                             size_t dim, uint64_t seed) {
    if (!hyperplanes || num_hyperplanes == 0 || dim == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    uint64_t state = (seed == 0) ? 0x1234567890abcdefULL : seed;

    for (size_t b = 0; b < num_hyperplanes; ++b) {
        float* row = &hyperplanes[b * dim];
        for (size_t d = 0; d < dim; ++d) {
            row[d] = rand_normal(&state);
        }
        /* Normalize each hyperplane to unit length */
        (void)omni_normalize_l2(row, dim);
    }

    return RCRE_OK;
}

RCRE_NODISCARD uint64_t omni_lsh_compute_hash(const float* vec, const float* hyperplanes, 
                                              size_t num_hyperplanes, size_t dim) {
    if (!vec || !hyperplanes || dim == 0) return 0;

    uint64_t hash = 0;
    size_t bits = (num_hyperplanes > 64) ? 64 : num_hyperplanes;

    for (size_t b = 0; b < bits; ++b) {
        const float* plane = &hyperplanes[b * dim];
        float proj = omni_cosine_similarity(vec, plane, dim);
        if (proj >= 0.0f) {
            hash |= (1ULL << b);
        }
    }

    return hash;
}
