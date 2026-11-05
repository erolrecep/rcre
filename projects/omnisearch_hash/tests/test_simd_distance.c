/**
 * @file test_simd_distance.c
 * @brief Numerical Parity & Mathematical Invariant Tests for Normalization and Distance
 */

#include "omnisearch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-5f

int main(void) {
    printf("🧪 Running Distance & Normalization Parity Tests...\n");

    /* 1. Test Hamming Distance */
    uint64_t h1 = 0b10101010ULL;
    uint64_t h2 = 0b11101000ULL;
    /* Differences at bit 6 and bit 1 -> 2 bits */
    assert(omni_hamming_distance64(h1, h2) == 2);
    assert(omni_hamming_distance64(0ULL, 0xFFFFFFFFFFFFFFFFULL) == 64);
    assert(omni_hamming_distance64(0x1234567890ABCDEFULL, 0x1234567890ABCDEFULL) == 0);

    /* 2. Test L2 Normalization */
    const size_t D = 256;
    float vec[256];
    for (size_t i = 0; i < D; ++i) {
        vec[i] = (float)(i + 1);
    }
    float orig_norm = omni_vector_norm_l2(vec, D);
    assert(orig_norm > 1.0f);

    int err = omni_normalize_l2(vec, D);
    assert(err == RCRE_OK);

    float unit_norm = omni_vector_norm_l2(vec, D);
    assert(fabsf(unit_norm - 1.0f) < EPSILON);

    /* Self-similarity of unit vector must be identically 1.0 */
    float self_sim = omni_cosine_similarity(vec, vec, D);
    assert(fabsf(self_sim - 1.0f) < EPSILON);

    /* 3. Test Mathematical Invariant: ||u - v||^2 == 2 * (1 - <u, v>) */
    float u[256], v[256];
    for (size_t i = 0; i < D; ++i) {
        u[i] = sinf((float)i * 0.1f);
        v[i] = cosf((float)i * 0.15f);
    }
    (void)omni_normalize_l2(u, D);
    (void)omni_normalize_l2(v, D);

    float dot = omni_cosine_similarity(u, v, D);
    float l2_sq = omni_squared_l2_distance(u, v, D);
    float expected_l2_sq = 2.0f * (1.0f - dot);

    printf("  Dot Product: %f, L2 Squared: %f, Expected 2*(1-cos): %f\n", dot, l2_sq, expected_l2_sq);
    assert(fabsf(l2_sq - expected_l2_sq) < 1e-4f && "Mathematical identity violated!");

    /* 4. Test Zero Vector Fallback Normalization */
    float zero_vec[16] = {0};
    err = omni_normalize_l2(zero_vec, 16);
    assert(err == RCRE_OK);
    float zero_norm = omni_vector_norm_l2(zero_vec, 16);
    assert(fabsf(zero_norm - 1.0f) < EPSILON);

    printf("✅ test_simd_distance passed all checks!\n");
    return 0;
}
