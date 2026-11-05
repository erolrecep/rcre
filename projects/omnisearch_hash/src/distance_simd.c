/**
 * @file distance_simd.c
 * @brief Vector Normalization, Distance Calculations, and SIMD Kernels
 */

#include "omnisearch.h"
#include <math.h>

static void ensure_hal_init(void) {
    static bool initialized = false;
    if (!initialized) {
        rcre_hal_init();
        initialized = true;
    }
}

RCRE_NODISCARD int omni_normalize_l2(float* vec, size_t dim) {
    if (!vec || dim == 0) return RCRE_ERR_INVALID_ARG;
    ensure_hal_init();

    float sum_sq = 0.0f;
    if (rcre_hal.dot_f32) {
        sum_sq = rcre_hal.dot_f32(vec, vec, dim);
    } else {
        for (size_t i = 0; i < dim; ++i) {
            sum_sq += vec[i] * vec[i];
        }
    }

    if (sum_sq <= 1e-12f) {
        /* Vector is zero or near-zero; set uniform unit vector */
        float uniform_val = 1.0f / sqrtf((float)dim);
        for (size_t i = 0; i < dim; ++i) {
            vec[i] = uniform_val;
        }
        return RCRE_OK;
    }

    float inv_norm = 1.0f / sqrtf(sum_sq);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] *= inv_norm;
    }

    return RCRE_OK;
}

RCRE_NODISCARD float omni_vector_norm_l2(const float* vec, size_t dim) {
    if (!vec || dim == 0) return 0.0f;
    ensure_hal_init();

    float sum_sq = 0.0f;
    if (rcre_hal.dot_f32) {
        sum_sq = rcre_hal.dot_f32(vec, vec, dim);
    } else {
        for (size_t i = 0; i < dim; ++i) {
            sum_sq += vec[i] * vec[i];
        }
    }
    return sqrtf(sum_sq);
}

RCRE_NODISCARD float omni_cosine_similarity(const float* a, const float* b, size_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    ensure_hal_init();

    if (rcre_hal.dot_f32) {
        return rcre_hal.dot_f32(a, b, dim);
    }

    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

RCRE_NODISCARD float omni_squared_l2_distance(const float* a, const float* b, size_t dim) {
    if (!a || !b || dim == 0) return 0.0f;

    float sum = 0.0f;
    size_t i = 0;
    for (; i + 3 < dim; i += 4) {
        float d0 = a[i] - b[i];
        float d1 = a[i+1] - b[i+1];
        float d2 = a[i+2] - b[i+2];
        float d3 = a[i+3] - b[i+3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

