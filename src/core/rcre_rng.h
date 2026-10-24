/**
 * @file rcre_rng.h
 * @brief RCRE Unified Random Number Generation (internal header)
 *
 * Single source of truth for PRNG state and distributions across RCRE
 * (tensor init, data ingestion, sampling). All code that needs randomness
 * MUST use these routines instead of local static copies.
 *
 * Algorithm: XorShift64* (Marsaglia, 2003) — 64-bit state, fast,
 * reproducible across platforms. Seed with a non-zero constant.
 */

#ifndef RCRE_RNG_H
#define RCRE_RNG_H

#include <stdint.h>
#include <math.h>

static inline uint64_t rcre_rng_next(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/** Uniform float in [lo, hi). */
static inline float rcre_rng_uniform_f32(uint64_t* state, float lo, float hi) {
    uint64_t r = rcre_rng_next(state);
    double u = (double)(r >> 11) * (1.0 / 9007199254740992.0); /* [0,1) */
    return lo + (float)(u * (hi - lo));
}

/** Standard-normal float (Box–Muller), scaled by `std` and shifted by `mean`. */
static inline float rcre_rng_gaussian_f32(uint64_t* state, float mean, float std) {
    float u1 = rcre_rng_uniform_f32(state, 1e-7f, 1.0f);
    float u2 = rcre_rng_uniform_f32(state, 0.0f, 1.0f);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
    return mean + z * std;
}

#endif /* RCRE_RNG_H */
