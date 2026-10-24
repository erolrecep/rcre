/**
 * @file ml_internal.h
 * @brief Internal Mathematical & Distance Primitives for RCRE ML Subsystem
 */

#ifndef RCRE_ML_INTERNAL_H
#define RCRE_ML_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

/**
 * @brief Canonical direct squared Euclidean L2 distance between two vectors.
 */
static inline float rcre_squared_l2_distance(const float* a, const float* b, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * @brief Numerically clamped logistic sigmoid activation.
 */
static inline float rcre_fast_sigmoidf(float z) {
    if (z > 16.0f) return 1.0f;
    if (z < -16.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-z));
}

/**
 * @brief SplitMix64 pseudo-random number generator for reproducible sampling.
 */
static inline uint64_t rcre_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/**
 * @brief Uniform floating-point random number generator in range [0.0, 1.0).
 */
static inline float rcre_rand_f32(uint64_t* state) {
    return (float)((rcre_splitmix64(state) >> 40) * (1.0 / 16777216.0));
}

#endif /* RCRE_ML_INTERNAL_H */
