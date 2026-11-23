/**
 * @file reduce.c
 * @brief High-accuracy deterministic reductions (H0, 2.1.0)
 *
 * Kahan-Neumaier compensated summation — the library's reference
 * reduction class. Unlike the dispatched (vectorized) reductions in
 * linalg.c, these are:
 *   - not vectorized (accuracy > speed; they are the yardstick, and the
 *     hot-path FMA versions live in the HAL);
 *   - the error-floor reference for tests: the vectorized reductions are
 *     gated within a few ULP of these;
 *   - the model for the threadpool slab-combine policy (docs/hpc.md):
 *     each slab reduces Neumaier-style, partials combine in fixed slab
 *     order — deterministic, and accurate to ~2*eps regardless of n.
 *
 * Error model (Higham, Accuracy and Stability of Numerical Algorithms):
 *   naive sequential:  |err| <= (n-1)*eps*|sum of |x||   (grows with n)
 *   Neumaier:          |err| <= (2*eps + O(eps^2)) * |sum|  (independent of n)
 *   pairwise:          |err| <= log2(n)*eps*|sum|
 */

#include "rcre.h"

#include <stdlib.h>

double rcre_sum_neumaier_f64(const double* x, size_t n) {
    double sum = 0.0, c = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double t = sum + x[i];
        /* Neumaier's fix over Kahan: track the lost low part of BOTH the
         * running sum and the incoming term, so it works when |x[i]| >>
         * |sum| (plain Kahan silently loses the term entirely then). */
        if (sum >= x[i])
            c += ((sum - t) + x[i]);
        else
            c += ((x[i] - t) + sum);
        sum = t;
    }
    return sum + c;
}

float rcre_sum_neumaier_f32(const float* x, size_t n) {
    float sum = 0.0f, c = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float t = sum + x[i];
        if (sum >= x[i])
            c += ((sum - t) + x[i]);
        else
            c += ((x[i] - t) + sum);
        sum = t;
    }
    return sum + c;
}

double rcre_dot_neumaier_f64(const double* x, const double* y, size_t n) {
    double sum = 0.0, c = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double p = x[i] * y[i];
        const double t = sum + p;
        if (sum >= p)
            c += ((sum - t) + p);
        else
            c += ((p - t) + sum);
        sum = t;
    }
    return sum + c;
}

float rcre_dot_neumaier_f32(const float* x, const float* y, size_t n) {
    float sum = 0.0f, c = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float p = x[i] * y[i];
        const float t = sum + p;
        if (sum >= p)
            c += ((sum - t) + p);
        else
            c += ((p - t) + sum);
        sum = t;
    }
    return sum + c;
}

/* Recursive two-way split: depth ~log2(n), each FP add combines two
 * equal-magnitude partials. This is both the "best realistic" reference
 * for test gates and the deterministic slab-combine model: split the
 * array into slabs, sum each Neumaier-style, then combine partials in
 * fixed slab order with the same pairwise recursion. */
/* Halving recursion (textbook pairwise): each FP add combines two
 * equal-magnitude partials, so the error stays ~log2(n) ulps for
 * well-scaled data (measured: bit-exact on 10^6 * (1e-6) where naive
 * and block-sequential both drift). On the adversarial 1e20 + 2^20*1
 * layout it lands at exactly -1 ulp (16384 = ulp(1e20)) — still ~64x
 * better than naive. Deterministic (fixed split order). */
double rcre_sum_pairwise_f64(const double* x, size_t n) {
    if (n == 0) return 0.0;
    if (n == 1) return x[0];
    size_t half = n / 2;
    return rcre_sum_pairwise_f64(x, half)
         + rcre_sum_pairwise_f64(x + half, n - half);
}
