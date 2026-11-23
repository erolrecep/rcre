/**
 * @file rng.c
 * @brief Reproducible random & quasi-random streams (H1, 2.2.0)
 *
 * The library's single source of randomness for all stochastic work
 * (Monte Carlo, sampling, shuffles, stochastic optimizers). Design per
 * docs/hpc.md:
 *
 *  PCG64 — PCG-XSL-RR-128/64 (Melissa O'Neill, pcg-random.org; the
 *  implementation vendored here is numerically identical to
 *  numpy/random/src/pcg64/pcg64.h, MIT). 128-bit LCG state with the
 *  128-bit multiplier 0x2360ed051fc65da4_4385df649fccf645 and output
 *  rotr64(hi ^ lo, hi >> 58). Counter-based in the PCG sense: the state
 *  is a pure function of (parts, steps drawn), so a stream is fully
 *  reproducible from its seed — and `rcre_rng_pcg_skip` fast-forwards
 *  via a precomputed power-of-two jump table (the LCG jump sum
 *  (M^n - 1)/(M - 1) mod 2^128 is NOT computable by dividing inside
 *  128 bits: M - 1 has v2 = 2, and the quotient needs 2 extra bits of
 *  M^n - 1; the jump table avoids that entirely).
 *
 *  Substreams — the slab determinism contract (promoted from e-means):
 *  worker i of ANY pool size derives its stream from (base, i) via a
 *  deterministic splitmix-style mix into the increment, so pooled runs
 *  are bit-identical across pool sizes and machines.
 *
 *  Sobol' — new Joe-Kuo direction numbers for d = 1..20, stored as
 *  64-bit fixed-point words (word_j = v_{k,j} * 2^64, exact). Points:
 *  x_n = XOR of the words for the set bits of n, scaled by 2^-64.
 *  Bit-exact conformance against scipy 1.18.1 qmc.Sobol(scramble=False)
 *  is checked in from two stream chunks (points 0..4095 and
 *  1000000..1000255, all 20 dims).
 *
 *  Halton — radical inverse in bases 2, 3, 5; self-derivable reference
 *  (no external oracle needed).
 *
 *  Every stream is a pure function of its seed — no wall-clock, no
 *  address-dependence, no platform dependence.
 */

#include "rcre.h"

#include <stdint.h>
#include <string.h>

#include "rng_pcg_table.h"
#include "rng_sobol_table.h"

/* PCG64 128-bit multiplier (numpy/random/src/pcg64: PCG_DEFAULT_MULTIPLIER_128). */
static const uint64_t RCRE_RNG_PCG_MULT_H = 2549297995355413924ull; /* 0x2360ed051fc65da4 */
static const uint64_t RCRE_RNG_PCG_MULT_L = 4865540595714422341ull; /* 0x4385df649fccf645 */

/* ========================================================================= */
/* PCG64 — PCG-XSL-RR-128/64                                                 */
/* ========================================================================= */

/* 128-bit multiply/add on the platform's __uint128_t (toolchain floor:
 * Apple Clang 15+, gcc 13+ — all provide it).
 *   out = s * M + inc  (mod 2^128), s/M/inc split hi/lo.
 * Cross terms (hi*lo, lo*hi) contribute at bit offset 64; hi*hi is
 * >= 2^128 and drops. */
static inline void rng128_mul_add(uint64_t* sh, uint64_t* sl,
                                  const uint64_t* mh, const uint64_t* ml,
                                  const uint64_t* ih, const uint64_t* il) {
    __uint128_t prod = (__uint128_t)(*sl) * (*ml);
    prod += ((__uint128_t)(*sl) * (*mh)) << 64;
    prod += ((__uint128_t)(*sh) * (*ml)) << 64;
    prod += (((__uint128_t)(*ih)) << 64) | (*il);
    *sl = (uint64_t)prod;
    *sh = (uint64_t)(prod >> 64);
}

void rcre_rng_pcg_init_parts(RCRE_RNG* r,
                             uint64_t s0, uint64_t s1,
                             uint64_t s2, uint64_t s3) {
    /* numpy-compatible seeding: inc = (initseq << 1) | 1 with
     * initseq = (s2, s3); state = 0 -> step -> + initstate -> step,
     * initstate = (s0, s1). */
    r->i_hi = (s2 << 1) | (s3 >> 63);
    r->i_lo = (s3 << 1) | 1;
    uint64_t sh = 0, sl = 0;
    rng128_mul_add(&sh, &sl,
                   &RCRE_RNG_PCG_MULT_H, &RCRE_RNG_PCG_MULT_L,
                   &r->i_hi, &r->i_lo);
    uint64_t t = sl + s1;
    sh += (t < sl) + s0;
    sl = t;
    rng128_mul_add(&sh, &sl,
                   &RCRE_RNG_PCG_MULT_H, &RCRE_RNG_PCG_MULT_L,
                   &r->i_hi, &r->i_lo);
    r->s_hi = sh;
    r->s_lo = sl;
}

/* SplitMix64 finalizer (Vigna) — the deterministic mixer for
 * RCRE-native seeding and substream derivation. */
static uint64_t rng_splitmix64(uint64_t* z) {
    *z += 0x9E3779B97F4A7C15ull;
    uint64_t w = *z;
    w = (w ^ (w >> 30)) * 0xBF58476D1CE4E5B9ull;
    w = (w ^ (w >> 27)) * 0x94D049BB133111EBull;
    return w ^ (w >> 31);
}

void rcre_rng_pcg_init(RCRE_RNG* r, uint64_t seed) {
    /* RCRE-native single-seed derivation (deterministic, documented):
     * two splitmix rounds give the 4 numpy-style parts. Streams are
     * locked by tests/rng_fixtures (the numpy-compatible entry is
     * rcre_rng_pcg_init_parts). */
    uint64_t z = seed;
    uint64_t a = rng_splitmix64(&z);
    uint64_t b = rng_splitmix64(&z);
    uint64_t c = rng_splitmix64(&z);
    uint64_t d = rng_splitmix64(&z);
    rcre_rng_pcg_init_parts(r, a, b, c, d);
}

uint64_t rcre_rng_pcg_next_u64(RCRE_RNG* r) {
    /* step THEN output (numpy ordering — verified bit-exact). */
    rng128_mul_add(&r->s_hi, &r->s_lo,
                   &RCRE_RNG_PCG_MULT_H, &RCRE_RNG_PCG_MULT_L,
                   &r->i_hi, &r->i_lo);
    const uint64_t x = r->s_hi ^ r->s_lo;
    const uint32_t rot = (uint32_t)(r->s_hi >> 58);
    return (x >> rot) | (x << ((- (int64_t)rot) & 63));
}

uint32_t rcre_rng_pcg_next_u32(RCRE_RNG* r) {
    return (uint32_t)(rcre_rng_pcg_next_u64(r) >> 32);
}

double rcre_rng_pcg_next_f64(RCRE_RNG* r) {
    /* numpy's uint64_to_double: (u >> 11) * 2^-53 in [0,1), 53 bits. */
    const uint64_t u = rcre_rng_pcg_next_u64(r);
    return (double)(u >> 11) * 0x1.0p-53;
}

float rcre_rng_pcg_next_f32(RCRE_RNG* r) {
    /* top 24 bits — 24-bit resolution in [0,1). */
    const uint64_t u = rcre_rng_pcg_next_u64(r);
    return (float)(u >> 40) * 0x1.0p-24;
}

void rcre_rng_pcg_substream(const RCRE_RNG* base, uint64_t id, RCRE_RNG* out) {
    /* Slab determinism contract: the substream is a pure function of
     * (base increment, id). The id is mixed into the INCREMENT (PCG
     * substreams differ by increment, sharing the state lineage), so
     * worker i gets the same stream regardless of how many workers the
     * pool has — pooled reductions stay bit-identical across machines. */
    uint64_t z = base->i_hi ^ (id * 0x9E3779B97F4A7C15ull);
    uint64_t lo = rng_splitmix64(&z);
    uint64_t hi = rng_splitmix64(&z);
    out->i_hi = (hi << 1) | (lo >> 63);
    out->i_lo = (lo << 1) | 1;
    out->s_hi = base->s_hi;
    out->s_lo = base->s_lo;
    /* one step to decorrelate the initial state from the base state */
    rng128_mul_add(&out->s_hi, &out->s_lo,
                   &RCRE_RNG_PCG_MULT_H, &RCRE_RNG_PCG_MULT_L,
                   &out->i_hi, &out->i_lo);
}

void rcre_rng_pcg_skip(RCRE_RNG* r, uint64_t n) {
    /* Affine LCG fast-forward: for each set bit b of n,
     * s <- A_b * s + inc * S_b  (A_b = M^(2^b), S_b = sum_{k<2^b} M^k,
     * both mod 2^128, precomputed and verified in rng_pcg_table.h). */
    for (int b = 0; b < 64; ++b) {
        if (!((n >> b) & 1)) continue;
        const RCRE_RNG_PCG_Jump* j = &RCRE_RNG_PCG_JUMP[b];
        /* A_b * s (128x128 -> low 128; cross terms at bit offset 64) */
        __uint128_t acc = (__uint128_t)j->al * r->s_lo;
        acc += ((__uint128_t)j->al * r->s_hi) << 64;
        acc += ((__uint128_t)j->ah * r->s_lo) << 64;
        /* + inc * S_b */
        acc += (__uint128_t)j->sl * r->i_lo;
        acc += ((__uint128_t)j->sl * r->i_hi) << 64;
        acc += ((__uint128_t)j->sh * r->i_lo) << 64;
        r->s_lo = (uint64_t)acc;
        r->s_hi = (uint64_t)(acc >> 64);
    }
}

/* ========================================================================= */
/* Sobol' (d = 1..20)                                                        */
/* ========================================================================= */

void rcre_qrng_sobol_at(uint64_t n, int d, double* out) {
    if (d < 1) d = 1;
    if (d > RCRE_QRNG_SOBOL_DIM) d = RCRE_QRNG_SOBOL_DIM;
    for (int k = 0; k < d; ++k) {
        uint64_t acc = 0;
        uint64_t m = n;
        int j = 0;
        while (m) {
            if (m & 1) acc ^= RCRE_SOBOL_V[k][j];
            m >>= 1;
            ++j;
        }
        /* acc * 2^-64: exact (acc is an integer < 2^64) */
        out[k] = (double)acc * 0x1.0p-64;
    }
}

void rcre_qrng_sobol_init(RCRE_QRNG* q, int d) {
    q->n = 0;
    q->d = (d >= 1 && d <= RCRE_QRNG_SOBOL_DIM) ? d : 1;
}

void rcre_qrng_sobol_point(RCRE_QRNG* q, double* out) {
    rcre_qrng_sobol_at(q->n, q->d, out);
    q->n++;
}

/* ========================================================================= */
/* Halton (bases 2, 3, 5)                                                    */
/* ========================================================================= */

/* radical inverse of n in base b, in [0,1). n >= 1. */
static double halton_at(uint64_t n, uint32_t b) {
    double f = 1.0, r = 0.0;
    while (n) {
        f /= (double)b;
        r += f * (double)(n % b);
        n /= b;
    }
    return r;
}

void rcre_qrng_halton_at(uint64_t n, int k, double* out) {
    static const uint32_t bases[RCRE_QRNG_HALTON_DIM] = { 2, 3, 5 };
    if (k < 1) k = 1;
    if (k > RCRE_QRNG_HALTON_DIM) k = RCRE_QRNG_HALTON_DIM;
    for (int i = 0; i < k; ++i)
        out[i] = halton_at(n + 1, bases[i]);   /* n is 0-based: point n+1 */
}

void rcre_qrng_halton_init(RCRE_QRNG* q, int k) {
    q->n = 0;
    q->d = (k >= 1 && k <= RCRE_QRNG_HALTON_DIM) ? k : 1;
}

void rcre_qrng_halton_point(RCRE_QRNG* q, double* out) {
    rcre_qrng_halton_at(q->n, q->d, out);
    q->n++;
}
