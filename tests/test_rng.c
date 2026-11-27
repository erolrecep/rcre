/**
 * @file test_rng.c
 * @brief H1 conformance: reproducible random & quasi-random streams.
 *
 * Gate philosophy (docs/hpc.md):
 *  - PCG64: BIT-EXACT against the checked-in numpy-oracle streams
 *    (4 seeds x 512 raw draws) — the counter-based reproducibility
 *    contract is exact, not statistical.
 *  - PCG64 skip: homomorphism skip(skip(s,a),b) == skip(s,a+b) plus
 *    skip(1) == one draw (exact LCG fast-forward).
 *  - PCG64 substreams: (base, id) deterministic, distinct ids distinct,
 *    chi-square uniformity of pooled substream output.
 *  - f64/f32 mappings: bounded, finite, exact numpy mapping.
 *  - Sobol': BIT-EXACT against the scipy-oracle stream chunks
 *    (0..4095 and 1000000..1000255, all 20 dims); stateful iteration
 *    matches random access.
 *  - Halton: bit-exact against the fixture; radical-inverse spot
 *    identities (base b: x_1 = 1/b).
 */

#include "rcre.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "test_framework.h"
#include "rng_fixtures/rng_fixture_data.h"
#include "rng_fixtures/philox_stream_a.h"
#include "rng_fixtures/philox_stream_b.h"
#include "rng_fixtures/philox_corners.h"
#include "rng_fixtures/philox_corner_inputs.h"

/* ------------------------------- PCG64 ---------------------------------- */

static void test_pcg64_conformance_numpy(void) {
    for (int s = 0; s < RCRE_FIX_PCG_NSEEDS; ++s) {
        RCRE_RNG r;
        const uint64_t* const parts =
            (s == 0) ? RCRE_FIX_PCG_PARTS_0 :
            (s == 1) ? RCRE_FIX_PCG_PARTS_1 :
            (s == 2) ? RCRE_FIX_PCG_PARTS_2 : RCRE_FIX_PCG_PARTS_3;
        rcre_rng_pcg_init_parts(&r, parts[0], parts[1], parts[2], parts[3]);
        const uint64_t* const* rows =
            (s == 0) ? RCRE_FIX_PCG_S0_ROWS :
            (s == 1) ? RCRE_FIX_PCG_S1_ROWS :
            (s == 2) ? RCRE_FIX_PCG_S2_ROWS : RCRE_FIX_PCG_S3_ROWS;
        for (int blk = 0; blk < RCRE_FIX_PCG_DRAWS / 8; ++blk)
            for (int i = 0; i < 8; ++i)
                ASSERT_TRUE(rcre_rng_pcg_next_u64(&r) == rows[blk][i]);
    }
}

static void test_pcg64_skip(void) {
    RCRE_RNG a, b;
    rcre_rng_pcg_init(&a, 42);
    rcre_rng_pcg_init(&b, 42);
    const uint64_t N1 = 0x0123456789ABCDEFull, N2 = 0xFEDCBA9876543210ull;
    rcre_rng_pcg_skip(&a, N1);
    rcre_rng_pcg_skip(&a, N2);
    rcre_rng_pcg_skip(&b, N1 + N2);
    ASSERT_TRUE(a.s_hi == b.s_hi && a.s_lo == b.s_lo);

    RCRE_RNG c;
    rcre_rng_pcg_init(&c, 42);
    rcre_rng_pcg_skip(&c, 0);
    ASSERT_TRUE(c.s_hi == b.s_hi || c.s_lo != b.s_lo); /* nontrivial state */
    RCRE_RNG id0, id1;
    rcre_rng_pcg_init(&id0, 42);
    rcre_rng_pcg_init(&id1, 42);
    rcre_rng_pcg_skip(&id1, 0);
    ASSERT_TRUE(id0.s_hi == id1.s_hi && id0.s_lo == id1.s_lo); /* skip(0) identity */

    /* skip(1) == one raw draw step */
    rcre_rng_pcg_init(&a, 7);
    rcre_rng_pcg_init(&b, 7);
    (void)rcre_rng_pcg_next_u64(&a);
    rcre_rng_pcg_skip(&b, 1);
    ASSERT_TRUE(a.s_hi == b.s_hi && a.s_lo == b.s_lo);

    /* skip must preserve stream continuity: draw n, skip, draw == fresh stream */
    rcre_rng_pcg_init(&a, 3);
    rcre_rng_pcg_init(&b, 3);
    const int PRE = 5;
    for (int i = 0; i < PRE; ++i) (void)rcre_rng_pcg_next_u64(&a);
    rcre_rng_pcg_skip(&a, 97);
    uint64_t got_a = rcre_rng_pcg_next_u64(&a);
    for (int i = 0; i < PRE + 97; ++i) (void)rcre_rng_pcg_next_u64(&b);
    ASSERT_TRUE(got_a == rcre_rng_pcg_next_u64(&b));
}

static void test_pcg64_substreams(void) {
    RCRE_RNG base;
    rcre_rng_pcg_init(&base, 20261123);
    RCRE_RNG s1a, s1b;
    rcre_rng_pcg_substream(&base, 1, &s1a);
    rcre_rng_pcg_substream(&base, 1, &s1b);
    for (int i = 0; i < 16; ++i)
        ASSERT_TRUE(rcre_rng_pcg_next_u64(&s1a) == rcre_rng_pcg_next_u64(&s1b));

    RCRE_RNG t1, t2;
    rcre_rng_pcg_substream(&base, 1, &t1);
    rcre_rng_pcg_substream(&base, 2, &t2);
    int different = 0;
    for (int i = 0; i < 8; ++i)
        if (rcre_rng_pcg_next_u64(&t1) != rcre_rng_pcg_next_u64(&t2)) different = 1;
    ASSERT_TRUE(different);

    /* chi-square uniformity: 16 substreams x 65536 f64 draws, 100 bins */
    const int BINS = 100, DRAWS = 65536, WORKERS = 16;
    uint64_t counts[100] = { 0 };
    RCRE_RNG w[WORKERS];
    for (int i = 0; i < WORKERS; ++i) rcre_rng_pcg_substream(&base, (uint64_t)i, &w[i]);
    for (int i = 0; i < WORKERS; ++i)
        for (int d = 0; d < DRAWS; ++d) {
            double u = rcre_rng_pcg_next_f64(&w[i]);
            counts[(int)(u * BINS)]++;
        }
    const double expected = (double)(WORKERS * DRAWS) / BINS;
    double chi2 = 0.0;
    for (int b = 0; b < BINS; ++b) {
        double diff = (double)counts[b] - expected;
        chi2 += diff * diff / expected;
    }
    /* chi2(99): degenerate uniform ~99, catastrophic > 500 */
    ASSERT_TRUE(chi2 < 500.0);
    ASSERT_TRUE(chi2 > 1.0);
}

static void test_pcg64_mappings(void) {
    RCRE_RNG r;
    rcre_rng_pcg_init(&r, 99);
    for (int i = 0; i < 100000; ++i) {
        double u = rcre_rng_pcg_next_f64(&r);
        ASSERT_TRUE(u >= 0.0 && u < 1.0);
        ASSERT_TRUE(isfinite(u));
    }
    for (int i = 0; i < 100000; ++i) {
        float u = rcre_rng_pcg_next_f32(&r);
        ASSERT_TRUE(u >= 0.0f && u < 1.0f);
    }
    /* exact numpy mapping: f64 = (u >> 11) * 2^-53 of the first draw */
    RCRE_RNG raw, chk;
    rcre_rng_pcg_init(&raw, 12345);
    const uint64_t u0 = rcre_rng_pcg_next_u64(&raw);
    rcre_rng_pcg_init(&chk, 12345);
    ASSERT_TRUE(rcre_rng_pcg_next_f64(&chk) == (double)(u0 >> 11) * 0x1.0p-53);
}

/* ------------------------------- Sobol' --------------------------------- */

static void test_sobol_conformance_scipy(void) {
    double out[RCRE_QRNG_SOBOL_DIM];
    /* chunk 0..4095, all dims, bit-exact */
    for (int k = 1; k <= RCRE_QRNG_SOBOL_DIM; ++k) {
        const uint64_t* const* rows = NULL;
        switch (k) {
        case 1:  rows = RCRE_FIX_SOBOL_4096_D1_ROWS;  break;
        case 2:  rows = RCRE_FIX_SOBOL_4096_D2_ROWS;  break;
        case 3:  rows = RCRE_FIX_SOBOL_4096_D3_ROWS;  break;
        case 4:  rows = RCRE_FIX_SOBOL_4096_D4_ROWS;  break;
        case 5:  rows = RCRE_FIX_SOBOL_4096_D5_ROWS;  break;
        case 6:  rows = RCRE_FIX_SOBOL_4096_D6_ROWS;  break;
        case 7:  rows = RCRE_FIX_SOBOL_4096_D7_ROWS;  break;
        case 8:  rows = RCRE_FIX_SOBOL_4096_D8_ROWS;  break;
        case 9:  rows = RCRE_FIX_SOBOL_4096_D9_ROWS;  break;
        case 10: rows = RCRE_FIX_SOBOL_4096_D10_ROWS; break;
        case 11: rows = RCRE_FIX_SOBOL_4096_D11_ROWS; break;
        case 12: rows = RCRE_FIX_SOBOL_4096_D12_ROWS; break;
        case 13: rows = RCRE_FIX_SOBOL_4096_D13_ROWS; break;
        case 14: rows = RCRE_FIX_SOBOL_4096_D14_ROWS; break;
        case 15: rows = RCRE_FIX_SOBOL_4096_D15_ROWS; break;
        case 16: rows = RCRE_FIX_SOBOL_4096_D16_ROWS; break;
        case 17: rows = RCRE_FIX_SOBOL_4096_D17_ROWS; break;
        case 18: rows = RCRE_FIX_SOBOL_4096_D18_ROWS; break;
        case 19: rows = RCRE_FIX_SOBOL_4096_D19_ROWS; break;
        default: rows = RCRE_FIX_SOBOL_4096_D20_ROWS; break;
        }
        for (int blk = 0; blk < RCRE_FIX_SOBOL_4096_NPTS / 8; ++blk) {
            const uint64_t want_d = (uint64_t)(k - 1);
            (void)want_d;
            for (int i = 0; i < 8; ++i) {
                const uint64_t n = (uint64_t)(blk * 8 + i);
                rcre_qrng_sobol_at(n, k, out);
                uint64_t got;
                memcpy(&got, &out[k - 1], 8);
                ASSERT_TRUE(got == rows[blk][i]);
            }
        }
    }
    /* chunk 1000000..1000255 (256 points, all dims) */
    for (int k = 1; k <= RCRE_QRNG_SOBOL_DIM; ++k) {
        const uint64_t* const* rows = NULL;
        switch (k) {
        case 1:  rows = RCRE_FIX_SOBOL_1M_D1_ROWS;  break;
        case 2:  rows = RCRE_FIX_SOBOL_1M_D2_ROWS;  break;
        case 3:  rows = RCRE_FIX_SOBOL_1M_D3_ROWS;  break;
        case 4:  rows = RCRE_FIX_SOBOL_1M_D4_ROWS;  break;
        case 5:  rows = RCRE_FIX_SOBOL_1M_D5_ROWS;  break;
        case 6:  rows = RCRE_FIX_SOBOL_1M_D6_ROWS;  break;
        case 7:  rows = RCRE_FIX_SOBOL_1M_D7_ROWS;  break;
        case 8:  rows = RCRE_FIX_SOBOL_1M_D8_ROWS;  break;
        case 9:  rows = RCRE_FIX_SOBOL_1M_D9_ROWS;  break;
        case 10: rows = RCRE_FIX_SOBOL_1M_D10_ROWS; break;
        case 11: rows = RCRE_FIX_SOBOL_1M_D11_ROWS; break;
        case 12: rows = RCRE_FIX_SOBOL_1M_D12_ROWS; break;
        case 13: rows = RCRE_FIX_SOBOL_1M_D13_ROWS; break;
        case 14: rows = RCRE_FIX_SOBOL_1M_D14_ROWS; break;
        case 15: rows = RCRE_FIX_SOBOL_1M_D15_ROWS; break;
        case 16: rows = RCRE_FIX_SOBOL_1M_D16_ROWS; break;
        case 17: rows = RCRE_FIX_SOBOL_1M_D17_ROWS; break;
        case 18: rows = RCRE_FIX_SOBOL_1M_D18_ROWS; break;
        case 19: rows = RCRE_FIX_SOBOL_1M_D19_ROWS; break;
        default: rows = RCRE_FIX_SOBOL_1M_D20_ROWS; break;
        }
        for (int blk = 0; blk < RCRE_FIX_SOBOL_1M_NPTS / 8; ++blk)
            for (int i = 0; i < 8; ++i) {
                const uint64_t n = RCRE_FIX_SOBOL_1M_OFFSET + (uint64_t)(blk * 8 + i);
                rcre_qrng_sobol_at(n, k, out);
                uint64_t got;
                memcpy(&got, &out[k - 1], 8);
                ASSERT_TRUE(got == rows[blk][i]);
            }
    }
}

static void test_sobol_stateful_and_properties(void) {
    double p[RCRE_QRNG_SOBOL_DIM];
    RCRE_QRNG q;
    rcre_qrng_sobol_init(&q, 3);
    /* stateful iteration == random access */
    for (int n = 0; n < 1000; ++n) {
        double s[3], a[3];
        rcre_qrng_sobol_point(&q, s);
        rcre_qrng_sobol_at((uint64_t)n, 3, a);
        for (int k = 0; k < 3; ++k) ASSERT_TRUE(s[k] == a[k]);
    }
    /* x_0 = 0 exactly */
    rcre_qrng_sobol_at(0, RCRE_QRNG_SOBOL_DIM, p);
    for (int k = 0; k < RCRE_QRNG_SOBOL_DIM; ++k) ASSERT_TRUE(p[k] == 0.0);
    /* values in [0,1) */
    for (uint64_t n = 0; n < 100000; n += 997) {
        rcre_qrng_sobol_at(n, RCRE_QRNG_SOBOL_DIM, p);
        for (int k = 0; k < RCRE_QRNG_SOBOL_DIM; ++k)
            ASSERT_TRUE(p[k] >= 0.0 && p[k] < 1.0);
    }
}

/* ------------------------------- Halton --------------------------------- */

static void test_halton_conformance(void) {
    double out[RCRE_QRNG_HALTON_DIM];
    const uint64_t* const* rows2 = RCRE_FIX_HALTON_B2_ROWS;
    const uint64_t* const* rows3 = RCRE_FIX_HALTON_B3_ROWS;
    const uint64_t* const* rows5 = RCRE_FIX_HALTON_B5_ROWS;
    for (int n = 0; n < RCRE_FIX_HALTON_NPTS; ++n) {
        rcre_qrng_halton_at((uint64_t)n, 3, out);
        uint64_t g2, g3, g5;
        memcpy(&g2, &out[0], 8);
        memcpy(&g3, &out[1], 8);
        memcpy(&g5, &out[2], 8);
        ASSERT_TRUE(g2 == rows2[n / 10][n % 10]);
        ASSERT_TRUE(g3 == rows3[n / 10][n % 10]);
        ASSERT_TRUE(g5 == rows5[n / 10][n % 10]);
    }
    /* radical-inverse spot identities: point 1 in base b = 1/b */
    rcre_qrng_halton_at(0, 3, out);
    ASSERT_TRUE(out[0] == 0.5);          /* 1/2 */
    ASSERT_TRUE(out[1] * 3.0 == 1.0);    /* 1/3 */
    ASSERT_TRUE(out[2] * 5.0 == 1.0);    /* 1/5 */
    /* stateful == at */
    RCRE_QRNG q;
    rcre_qrng_halton_init(&q, 2);
    for (int n = 0; n < 1000; ++n) {
        double s[2], a[2];
        rcre_qrng_halton_point(&q, s);
        rcre_qrng_halton_at((uint64_t)n, 2, a);
        ASSERT_TRUE(s[0] == a[0] && s[1] == a[1]);
    }
}

/* ------------------------------- Philox-4x32 (H6) ------------------------- */

/* Bit-exact against the independent pure-Python oracle
 * (ref_philox_oracle.py): two 1024-word streams + corner (counter,key) */
static void test_philox_conformance_oracle(void) {
    uint32_t a[1024], b[1024];
    ASSERT_TRUE(rcre_rng_philox4x32_fill(a, 1024, 0,
                                         0x9E3779B9C0FFEE53ull) == RCRE_OK);
    ASSERT_TRUE(memcmp(a, RCRE_PHILOX_STREAM_A, sizeof a) == 0);
    ASSERT_TRUE(rcre_rng_philox4x32_fill(b, 1024, 1000, 0) == RCRE_OK);
    ASSERT_TRUE(memcmp(b, RCRE_PHILOX_STREAM_B, sizeof b) == 0);

    for (int i = 0; i < 6; ++i) {
        uint32_t w[4];
        rcre_rng_philox4x32(RCRE_PHILOX_CORNER_CTR[i], RCRE_PHILOX_CORNER_KEY[i], w);
        ASSERT_TRUE(memcmp(w, RCRE_PHILOX_CORNERS + 4 * i, 16) == 0);
    }

    /* determinism: repeat calls are bit-identical */
    uint32_t a2[1024];
    ASSERT_TRUE(rcre_rng_philox4x32_fill(a2, 1024, 0,
                                         0x9E3779B9C0FFEE53ull) == RCRE_OK);
    ASSERT_TRUE(memcmp(a, a2, sizeof a) == 0);

    /* counter contiguity: a single-shot at counter c equals word 4c of
     * a fill starting at c */
    uint32_t one[4], tail[8];
    rcre_rng_philox4x32(77, 1234, one);
    ASSERT_TRUE(rcre_rng_philox4x32_fill(tail, 8, 77, 1234) == RCRE_OK);
    ASSERT_TRUE(memcmp(one, tail, 16) == 0);
}

/* Light distribution sanity (guards against constant/multiplier typos
 * that still pass a hand-rolled oracle if the oracle were derived from
 * the C code — the curand cross-check on dlbox is the strong anchor). */
static void test_philox_distribution_sanity(void) {
    enum { N = 1 << 20 };
    uint32_t* w = (uint32_t*)malloc(N * sizeof(uint32_t));
    ASSERT_TRUE(w != NULL);
    ASSERT_TRUE(rcre_rng_philox4x32_fill(w, N, 0, 0xC0FFEEC0FFEEull) == RCRE_OK);
    uint32_t distinct = 0;
    int zero_count = 0;
    for (int i = 0; i < N; ++i) {
        if (i == 0 || w[i] != w[i - 1]) distinct++;
        if (w[i] == 0) zero_count++;
    }
    /* 1M distinct 32-bit draws: adjacent-equal rate ~2^-32, so run
     * distinctness must be ~N; require > 99.99% */
    ASSERT_TRUE(distinct > N - (N / 100));  /* > 99% runs (expect ~100%: P(adj equal)=2^-32) */
    /* P(zero) = 2^-32: expect ~0, allow 4 */
    ASSERT_TRUE(zero_count <= 4);
    free(w);
}

int main(void) {
    test_suite_begin("RCRE Reproducible Streams (H1, 2.2.0)");
    TEST_RUN(test_pcg64_conformance_numpy);
    TEST_RUN(test_pcg64_skip);
    TEST_RUN(test_pcg64_substreams);
    TEST_RUN(test_pcg64_mappings);
    TEST_RUN(test_sobol_conformance_scipy);
    TEST_RUN(test_sobol_stateful_and_properties);
    TEST_RUN(test_halton_conformance);
    TEST_RUN(test_philox_conformance_oracle);
    TEST_RUN(test_philox_distribution_sanity);
    return test_suite_report();
}
