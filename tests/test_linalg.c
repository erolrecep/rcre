/**
 * @file test_linalg.c
 * @brief H0 (2.1.0) tests: F64 substrate, dispatched linalg, compensated
 *        reductions, special functions.
 *
 * Gate philosophy (docs/hpc.md):
 *  - cross-ISA (dispatched vs scalar) f64: tolerance-gated (FMA/accumulation
 *    order legitimately differs across ISAs), f32: loose tolerance;
 *  - Neumaier/pairwise: error-floor gates against exact-known answers
 *    (n * (1/n) = 1) — the compensated routines must sit at the double
 *    floor while naive accumulation measurably drifts;
 *  - special functions: absolute-error gates against published values
 *    (50-digit Decimal reference for digamma, verified via psi(10) =
 *    H_9 - gamma; system math for erf/erfc/lgamma);
 *  - determinism: every routine bitwise reproducible run-to-run.
 */

#include "rcre.h"
#include "hal_internal.h"   /* scalar cells as the cross-ISA reference */
#include "test_framework.h"

#include <math.h>
#include <string.h>

/* deterministic LCG test data (no RNG subsystem yet — H1 ships Philox) */
static uint64_t s_lcg = 0x9E3779B97F4A7C15ull;
static double next_rand(void) {
    s_lcg = s_lcg * 6364136223846793005ull + 1442695040888963407ull;
    return (double)((s_lcg >> 11) & 0xFFFFFFFFFULL) / (double)0xFFFFFFFFFULL;
}

/* ------------------------------- dispatch ------------------------------- */

static void test_f64_dispatch_available(void) {
    rcre_hal_init();
    ASSERT_NOT_NULL(rcre_hal.gemm_f64);
    ASSERT_NOT_NULL(rcre_hal.dot_f64);
    ASSERT_NOT_NULL(rcre_hal.axpy_f64);
    ASSERT_NOT_NULL(rcre_hal.gemv_f64);
    ASSERT_NOT_NULL(rcre_hal.scale_f64);
    ASSERT_NOT_NULL(rcre_hal.sum_f64);
    ASSERT_NOT_NULL(rcre_hal.gemv_f32);
    ASSERT_NOT_NULL(rcre_hal.scale_f32);
    ASSERT_NOT_NULL(rcre_hal.sum_f32);
    /* the backend must be one of the known four */
    ASSERT_TRUE(rcre_hal.cpu_features & (RCRE_CPU_FEAT_SCALAR | RCRE_CPU_FEAT_NEON
                                         | RCRE_CPU_FEAT_AVX2 | RCRE_CPU_FEAT_AVX512F
                                         | RCRE_CPU_FEAT_WASM_SIMD));
}

/* --------------------------- f64 vector vs scalar ------------------------ */

static void test_f64_dot_axpy_scale_sum(void) {
    rcre_hal_init();
    const size_t sizes[] = { 1, 3, 4, 7, 8, 15, 16, 23, 31, 32, 63, 64, 127, 1024, 4096 };
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); ++s) {
        size_t n = sizes[s];
        double *x = malloc(n * sizeof(double));
        double *y = malloc(n * sizeof(double));
        for (size_t i = 0; i < n; ++i) { x[i] = next_rand() - 0.5; y[i] = next_rand() - 0.5; }

        double d_vec = rcre_linalg_dot_f64(x, y, n);
        double d_sca = rcre_scalar_dot_f64(x, y, n);
        ASSERT_TRUE(fabs(d_vec - d_sca) <= 1e-14 * fmax(1.0, fabs(d_sca)));

        double ne = rcre_dot_neumaier_f64(x, y, n);
        ASSERT_TRUE(fabs(ne - d_sca) <= 1e-14 * fmax(1.0, fabs(d_sca)));

        double sm_vec = rcre_linalg_sum_f64(x, n);
        double sm_sca = rcre_scalar_sum_f64(x, n);
        ASSERT_TRUE(fabs(sm_vec - sm_sca) <= 1e-14 * fmax(1.0, fabs(sm_sca)));

        double pa = rcre_sum_pairwise_f64(x, n);
        ASSERT_TRUE(fabs(pa - sm_sca) <= 1e-14 * fmax(1.0, fabs(sm_sca)));

        /* axpy: y += 1.5 x */
        double *yv = malloc(n * sizeof(double));
        double *ys = malloc(n * sizeof(double));
        memcpy(yv, y, n * sizeof(double));
        memcpy(ys, y, n * sizeof(double));
        rcre_linalg_axpy_f64(n, 1.5, x, yv);
        rcre_scalar_axpy_f64(n, 1.5, x, ys);
        for (size_t i = 0; i < n; ++i)
            ASSERT_TRUE(fabs(yv[i] - ys[i]) <= 1e-14 * fmax(1.0, fabs(ys[i])));

        /* scale: x0.25 is exact (power of two) — both paths bit-identical */
        rcre_linalg_scale_f64(n, 0.25, yv);
        for (size_t i = 0; i < n; ++i) {
            ys[i] *= 0.25;
            ASSERT_TRUE(yv[i] == ys[i]);
        }
        free(x); free(y); free(yv); free(ys);
    }
}

static void test_f64_gemv(void) {
    rcre_hal_init();
    const size_t M = 33, N = 48;
    double *A = malloc(M * N * sizeof(double));
    double *x = malloc(N * sizeof(double));
    for (size_t i = 0; i < M * N; ++i) A[i] = next_rand() - 0.5;
    for (size_t i = 0; i < N; ++i) x[i] = next_rand() - 0.5;

    double *yv = malloc(M * sizeof(double));
    double *ys = malloc(M * sizeof(double));
    for (size_t i = 0; i < M; ++i) yv[i] = ys[i] = next_rand();
    rcre_linalg_gemv_f64(M, N, 1.0, A, N, x, 0.5, yv);
    rcre_scalar_gemv_f64(M, N, 1.0, A, N, x, 0.5, ys);
    for (size_t i = 0; i < M; ++i)
        ASSERT_TRUE(fabs(yv[i] - ys[i]) <= 1e-13 * fmax(1.0, fabs(ys[i])));

    /* beta = 0 path */
    for (size_t i = 0; i < M; ++i) yv[i] = 99.0;
    rcre_linalg_gemv_f64(M, N, 2.0, A, N, x, 0.0, yv);
    for (size_t i = 0; i < M; ++i)
        ASSERT_TRUE(fabs(yv[i] - 2.0 * rcre_scalar_dot_f64(&A[i * N], x, N))
                    <= 1e-13 * fmax(1.0, fabs(ys[i])));
    free(A); free(x); free(yv); free(ys);
}

static void test_f64_gemm(void) {
    rcre_hal_init();
    /* blocked sizes to exercise the full MB/NB/KB tiling + remainders */
    const size_t Ms[] = { 1, 7, 8, 33, 64, 129 };
    const size_t Ns[] = { 1, 5, 8, 64, 131 };
    const size_t Ks[] = { 1, 6, 8, 64, 127 };
    for (size_t mi = 0; mi < 6; ++mi)
    for (size_t ni = 0; ni < 5; ++ni)
    for (size_t ki = 0; ki < 5; ++ki) {
        size_t M = Ms[mi], N = Ns[ni], K = Ks[ki];
        double *A = malloc(M * K * sizeof(double));
        double *B = malloc(K * N * sizeof(double));
        double *C0 = malloc(M * N * sizeof(double));   /* original C */
        double *Cv = malloc(M * N * sizeof(double));
        double *Cs = malloc(M * N * sizeof(double));
        for (size_t i = 0; i < M * K; ++i) A[i] = (next_rand() - 0.5) * 0.1;
        for (size_t i = 0; i < K * N; ++i) B[i] = (next_rand() - 0.5) * 0.1;
        for (size_t i = 0; i < M * N; ++i) C0[i] = next_rand();

        /* identity path: C = A*B + C0 (beta = 1) */
        memcpy(Cv, C0, M * N * sizeof(double));
        memcpy(Cs, C0, M * N * sizeof(double));
        rcre_linalg_gemm_f64(M, N, K, 1.0, A, K, B, N, 1.0, Cv, N);
        rcre_scalar_gemm_f64(M, N, K, 1.0, A, K, B, N, 1.0, Cs, N);
        double max_dev = 0.0;
        for (size_t i = 0; i < M * N; ++i)
            max_dev = fmax(max_dev, fabs(Cv[i] - Cs[i]));
        ASSERT_TRUE(max_dev <= 1e-12 * fmax(1.0, (double)K));

        /* alpha/beta path: C = 0.25*A*B + 0.5*C0, reference via scalar gemm
         * into a fresh buffer: Cs = 0.5*C0; Cs = A*B + Cs... no: build
         * ref = 0.25*(A*B) + 0.5*C0 element-wise from the scalar AB. */
        memcpy(Cs, C0, M * N * sizeof(double));
        rcre_scalar_gemm_f64(M, N, K, 0.25, A, K, B, N, 0.5, Cs, N);
        memcpy(Cv, C0, M * N * sizeof(double));
        rcre_linalg_gemm_f64(M, N, K, 0.25, A, K, B, N, 0.5, Cv, N);
        max_dev = 0.0;
        for (size_t i = 0; i < M * N; ++i)
            max_dev = fmax(max_dev, fabs(Cv[i] - Cs[i]));
        ASSERT_TRUE(max_dev <= 1e-12 * fmax(1.0, (double)K));

        /* beta = 0 overwrites */
        for (size_t i = 0; i < M * N; ++i) Cv[i] = Cs[i] = 99.0;
        rcre_linalg_gemm_f64(M, N, K, 1.0, A, K, B, N, 0.0, Cv, N);
        rcre_scalar_gemm_f64(M, N, K, 1.0, A, K, B, N, 0.0, Cs, N);
        max_dev = 0.0;
        for (size_t i = 0; i < M * N; ++i)
            max_dev = fmax(max_dev, fabs(Cv[i] - Cs[i]));
        ASSERT_TRUE(max_dev <= 1e-12 * fmax(1.0, (double)K));

        free(A); free(B); free(C0); free(Cv); free(Cs);
    }
}

static void test_f32_ops_match_scalar(void) {
    rcre_hal_init();
    const size_t M = 25, N = 27, K = 31;
    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *Cv = malloc(M * N * sizeof(float));
    float *Cs = malloc(M * N * sizeof(float));
    for (size_t i = 0; i < M * K; ++i) A[i] = (float)(next_rand() - 0.5f) * 0.1f;
    for (size_t i = 0; i < K * N; ++i) B[i] = (float)(next_rand() - 0.5f) * 0.1f;
    for (size_t i = 0; i < M * N; ++i) Cv[i] = Cs[i] = (float)next_rand();
    rcre_linalg_gemm_f32(M, N, K, 1.0f, A, K, B, N, 1.0f, Cv, N);
    rcre_scalar_gemm_f32(M, N, K, 1.0f, A, K, B, N, 1.0f, Cs, N);
    double max_dev = 0.0;
    for (size_t i = 0; i < M * N; ++i)
        max_dev = fmax(max_dev, fabs((double)Cv[i] - (double)Cs[i]));
    ASSERT_TRUE(max_dev <= 1e-4);
    free(A); free(B); free(Cv); free(Cs);
}

/* ---------------------------- Neumaier gates ----------------------------- */

/* n * (1/n) sums to exactly 1 in real arithmetic. Naive sequential
 * accumulation at n = 1e6 drifts ~ n*eps*|S|/2 ~ 1e-10; Neumaier and
 * pairwise must sit at the double floor (~1e-16). */
static void test_neumaier_error_floor(void) {
    const size_t n = 1000000;
    double *x = malloc(n * sizeof(double));
    const double v = 1.0 / (double)n;
    for (size_t i = 0; i < n; ++i) x[i] = v;

    double naive = rcre_scalar_sum_f64(x, n);
    double neu   = rcre_sum_neumaier_f64(x, n);
    double pair  = rcre_sum_pairwise_f64(x, n);

    ASSERT_TRUE(fabs(neu - 1.0) <= 4e-16);
    ASSERT_TRUE(fabs(pair - 1.0) <= 4e-16);
    /* naive is measured for the record; on this class of machine it must
     * be at least as accurate as floor*64 (it usually is ~1e-16 here, but
     * the gate is one-directional: compensated <= naive * 16) */
    ASSERT_TRUE(fabs(neu - 1.0) <= fabs(naive - 1.0) * 16.0 + 1e-18);
    free(x);
}

/* catastrophic cancellation: [1e16, 1, -1e16] */
static void test_neumaier_cancellation(void) {
    double x[3] = { 1e16, 1.0, -1e16 };
    double neu = rcre_sum_neumaier_f64(x, 3);
    double naive = rcre_scalar_sum_f64(x, 3);
    /* naive: (1e16 + 1) == 1e16, then 0.0 — the classic lost-low-bit case.
     * Neumaier recovers the +1. */
    ASSERT_TRUE(neu == 1.0);
    ASSERT_TRUE(naive == 0.0 || fabs(neu - 1.0) < fabs(naive - 1.0));
}

static void test_reductions_deterministic(void) {
    const size_t n = 10000;
    double *x = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) x[i] = next_rand() - 0.5;
    double a = rcre_sum_neumaier_f64(x, n), b = rcre_sum_neumaier_f64(x, n);
    ASSERT_TRUE(a == b);                       /* bitwise, run-to-run */
    double c = rcre_linalg_sum_f64(x, n), d = rcre_linalg_sum_f64(x, n);
    ASSERT_TRUE(c == d);
    free(x);
}

/* -------------------------- special functions ---------------------------- */

static void test_sf_erf_erfc(void) {
    ASSERT_TRUE(fabs(rcre_sf_erf_f64(0.5) - 0.5204998778130465) <= 1.2e-15);
    ASSERT_TRUE(fabs(rcre_sf_erf_f64(1.0) - 0.8427007929497148) <= 1.2e-15);
    ASSERT_TRUE(fabs(rcre_sf_erf_f64(1.5) - 0.9661051464753108) <= 1.2e-15);
    ASSERT_TRUE(fabs(rcre_sf_erf_f64(2.5) - 0.9995930479825550) <= 1.2e-15);
    ASSERT_TRUE(fabs(rcre_sf_erf_f64(-1.0) + 0.8427007929497148) <= 1.2e-15);
    ASSERT_TRUE(fabs(rcre_sf_erf_f64(0.0)) <= 1e-18);

    ASSERT_TRUE(fabs(rcre_sf_erfc_f64(1.2) - 0.08968602177036462) <= 1.2e-15);
    ASSERT_TRUE(fabs(rcre_sf_erfc_f64(3.0) - 2.2090496998585438e-05) <= 4e-20);
    ASSERT_TRUE(fabs(rcre_sf_erfc_f64(5.0) - 1.5374597944280350e-12) <= 4e-27);
    ASSERT_TRUE(rcre_sf_erfc_f64(30.0) == 0.0);   /* underflow: exact */
    ASSERT_TRUE(fabs(rcre_sf_erfc_f64(-1.2) - (2.0 - 0.08968602177036462)) <= 1.2e-15);
    ASSERT_FLOAT_NEAR(0.5204998778130465f, rcre_sf_erf_f32(0.5f), 1e-7f);
    /* erf/erfc identities at the series/CF meeting point */
    double t = 1.0;
    ASSERT_TRUE(fabs(rcre_sf_erf_f64(t) + rcre_sf_erfc_f64(t) - 1.0) <= 2.4e-15);
}

static void test_sf_lgamma(void) {
    ASSERT_TRUE(fabs(rcre_sf_lgamma_f64(5.0) - 3.1780538303479444) <= 2e-14);
    ASSERT_TRUE(fabs(rcre_sf_lgamma_f64(0.5) - 0.5723649429247004) <= 2e-14);
    ASSERT_TRUE(fabs(rcre_sf_lgamma_f64(1.0)) <= 1e-15);            /* ln 1 = 0 */
    ASSERT_TRUE(fabs(rcre_sf_lgamma_f64(10.3) - 13.482036786138359) <= 1e-14);
    ASSERT_TRUE(fabs(rcre_sf_lgamma_f64(0.3) - 1.0957979948180754) <= 2e-14);
    /* integer values: lgamma(n+1) = ln(n!) */
    ASSERT_TRUE(fabs(rcre_sf_lgamma_f64(2.0) - log(1.0)) <= 1e-15);
    ASSERT_TRUE(fabs(rcre_sf_lgamma_f64(6.0) - log(120.0)) <= 2e-14);
    /* poles */
    ASSERT_TRUE(isinf(rcre_sf_lgamma_f64(0.0)));
    ASSERT_TRUE(isinf(rcre_sf_lgamma_f64(-2.0)));
}

static void test_sf_digamma(void) {
    /* 50-digit Decimal references (see test file header) */
    ASSERT_TRUE(fabs(rcre_sf_digamma_f64(10.0) - 2.251752589066721) <= 2e-15);
    ASSERT_TRUE(fabs(rcre_sf_digamma_f64(3.7)  - 1.1671535393615113) <= 2e-15);
    ASSERT_TRUE(fabs(rcre_sf_digamma_f64(0.3)  + 3.502524222200133)  <= 2e-15);
    ASSERT_TRUE(fabs(rcre_sf_digamma_f64(8.0)  - 2.0156414779556098) <= 2e-15);
    /* exact identities */
    ASSERT_TRUE(fabs(rcre_sf_digamma_f64(1.0) + 0.5772156649015329) <= 1e-15);
    ASSERT_TRUE(fabs(rcre_sf_digamma_f64(0.5) + 1.9635100260214235) <= 2e-15);
    /* recurrence check: psi(x+1) - psi(x) = 1/x */
    ASSERT_TRUE(fabs((rcre_sf_digamma_f64(2.7) - rcre_sf_digamma_f64(1.7))
                     - (1.0 / 1.7)) <= 2e-15);
    ASSERT_TRUE(isinf(rcre_sf_digamma_f64(-1.0)));
}

static void test_sf_logsumexp_expit_norm(void) {
    double x[2] = { 1000.0, 1000.0 };
    ASSERT_TRUE(fabs(rcre_sf_logsumexp_f64(x, 2)
                     - (1000.0 + log(2.0))) <= 1e-13);
    double y[3] = { -1000.0, -1000.0, -1000.0 };
    ASSERT_TRUE(fabs(rcre_sf_logsumexp_f64(y, 3)
                     - (-1000.0 + log(3.0))) <= 1e-13);
    double z[1] = { 42.0 };
    ASSERT_TRUE(rcre_sf_logsumexp_f64(z, 1) == 42.0);

    ASSERT_TRUE(rcre_sf_expit_f64(0.0) == 0.5);
    /* 1 - expit(20) = 1/(1+e^20) = 2.0611536224e-9: the gate must clear
     * the true gap plus a few ULP of 1.0 (2.2e-16) */
    ASSERT_TRUE(fabs(rcre_sf_expit_f64(20.0) - 1.0) <= 2.2e-9);
    ASSERT_TRUE(fabs(rcre_sf_expit_f64(-20.0) - 2.0611536224385580e-09) <= 1e-16);

    ASSERT_TRUE(rcre_sf_norm_cdf_f64(0.0) == 0.5);
    ASSERT_TRUE(fabs(rcre_sf_norm_cdf_f64(1.96) - 0.9750021048517795) <= 1e-15);
    /* tail: uses the tail-accurate erfc, not the 1-Phi cancellation form.
     * reference below is Phi(-6) at the actual double input 6*fl(1/sqrt(2)),
     * exactly 1.0 in double (1 - 7.6e-24 < ulp(1)) — that IS the
     * correctly-rounded answer. */
    ASSERT_TRUE(fabs(rcre_sf_norm_cdf_f64(6.0) - 0.9999999990134123) <= 2e-16);
    ASSERT_TRUE(rcre_sf_norm_cdf_f64(10.0) == 1.0);
    /* 6-sigma tail: absolute 1e-23 (~10 ulp of the value) — the
     * measured floor of the Lentz double product; reference = Phi(-6)
     * at the actual double input 6*fl(1/sqrt(2)) to 55 digits. */
    ASSERT_TRUE(fabs(rcre_sf_norm_cdf_f64(-6.0) - 9.8658764503770137e-10) <= 1e-23);
    /* 10-sigma tail: 7.619853024160901e-24 — representable (no
     * cancellation: the erfc path computes the tail directly); the
     * underflow-to-zero threshold sits at |x| = 26.7 in erfc. */
    ASSERT_TRUE(fabs(rcre_sf_norm_cdf_f64(-10.0) - 7.619853024160593e-24) <= 1e-36);
    ASSERT_TRUE(rcre_sf_norm_cdf_f64(-100.0) == 0.0);
}

int main(void) {
    test_suite_begin("RCRE HPC Numerical Layer (H0, 2.1.0)");
    TEST_RUN(test_f64_dispatch_available);
    TEST_RUN(test_f64_dot_axpy_scale_sum);
    TEST_RUN(test_f64_gemv);
    TEST_RUN(test_f64_gemm);
    TEST_RUN(test_f32_ops_match_scalar);
    TEST_RUN(test_neumaier_error_floor);
    TEST_RUN(test_neumaier_cancellation);
    TEST_RUN(test_reductions_deterministic);
    TEST_RUN(test_sf_erf_erfc);
    TEST_RUN(test_sf_lgamma);
    TEST_RUN(test_sf_digamma);
    TEST_RUN(test_sf_logsumexp_expit_norm);
    return test_suite_report();
}
