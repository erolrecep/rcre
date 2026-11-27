/**
 * @file test_stream.c
 * @brief H7 streaming statistics tests (2.4.0).
 *
 * Measured gates (see docs/hpc.md 10.5):
 *
 *  - Welford/merge stream stats vs a naive two-pass reference
 *    (mean, var, skew, kurtosis): relative difference <= 1e-8 on
 *    a 1e5 fixed-seed stream; merge of two halves == full stream.
 *  - P^2 online quantile (Jain & Chlamtac): rank error <= 0.02 on
 *    1e6 uniform stream at p in {0.1, 0.5, 0.9} (measured <= 0.001).
 *    (The plan named t-digest/KLL; the Greenwald-Khanna draft was
 *    retired after its compression/query bookkeeping could not be
 *    reconciled with the paper's bound in measurement — P^2's
 *    rank error is gated empirically instead. docs/hpc.md 10.5.)
 *  - Bootstrap: ordinary resampling SE for the mean of n=200
 *    within [0.85, 1.15] of the analytic SE (measured 1.01);
 *    moving-block resampling runs cleanly; error inputs rejected.
 *  - Determinism: fixed seed -> bit-identical bootstrap draws.
 */
#include "rcre.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0;
static int g_fails = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        ++g_checks;                                                 \
        if (!(cond)) {                                              \
            ++g_fails;                                              \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);  \
        }                                                           \
    } while (0)

/* fixed-seed deterministic stream (LCG — reproducible everywhere) */
static double lcg64(uint64_t* st) {
    *st = *st * 6364136223846793005ull + 1442695040888963407ull;
    return ((*st >> 11) & ((1ull << 53) - 1)) * 0x1.0p-53;
}

/* qsort comparator */
static int cmp_double(const void* a, const void* b) {
    const double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

/* statistic for bootstrap: the sample mean */
static double mean_stat(const double* y, int n, void* user) {
    (void)user;
    double m = 0.0;
    for (int i = 0; i < n; ++i) m += y[i];
    return m / (double)n;
}

int main(void) {
    rcre_hal_init();
    printf("H7 stream stats tests:\n");

    const int n = 100000;
    double* data = malloc((size_t)n * 8);
    uint64_t st = 0xDEADBEEFCAFEBABEull;
    for (int k = 0; k < n; ++k) data[k] = lcg64(&st) * 2.0 - 1.0;

    /* [Welford vs naive] */
    {
        printf("  [welford vs naive]");
        RCRE_StreamStats ss;
        rcre_stream_stats_init(&ss);
        for (int k = 0; k < n; ++k) rcre_stream_stats_push(&ss, data[k]);
        /* naive two-pass */
        double m = 0.0;
        for (int k = 0; k < n; ++k) m += data[k];
        m /= n;
        double m2 = 0.0, m3 = 0.0, m4 = 0.0;
        for (int k = 0; k < n; ++k) {
            double dv = data[k] - m;
            m2 += dv * dv; m3 += dv * dv * dv; m4 += dv * dv * dv * dv;
        }
        m2 /= n; m3 /= n; m4 /= n;
        /* the library reports BIAS-CORRECTED (Fisher) sample skewness
         * and excess kurtosis — the reference must match exactly */
        double svar = (double)n / (double)(n - 1) * m2; /* sample var */
        double sstd = sqrt(m2);
        double g1 = m3 / (m2 * sstd);
        double sskew = g1 * sqrt((double)n * (double)(n - 1)) / (double)(n - 2);
        double g2 = m4 / (m2 * m2); /* population KURTOSIS (not excess) */
        double skurt = ((double)(n - 1) /
                        ((double)(n - 2) * (double)(n - 3))) *
                           ((double)(n + 1) * g2 + 6.0) - 3.0;
        CHECK(fabs(rcre_stream_stats_mean(&ss) - m) <= 1e-12 * (1.0 + fabs(m)),
              "mean");
        CHECK(fabs(rcre_stream_stats_var(&ss) - svar) <= 1e-8 * (1.0 + fabs(svar)),
              "var");
        CHECK(fabs(rcre_stream_stats_skew(&ss) - sskew) <= 1e-8, "skew");
        CHECK(fabs(rcre_stream_stats_kurtosis(&ss) - skurt) <= 1e-7,
              "kurtosis");
        printf(" OK\n");
    }

    /* [merge] two halves == full stream */
    {
        printf("  [merge]");
        RCRE_StreamStats a, b, ab;
        rcre_stream_stats_init(&a);
        rcre_stream_stats_init(&b);
        rcre_stream_stats_init(&ab);
        for (int k = 0; k < n / 2; ++k) rcre_stream_stats_push(&a, data[k]);
        for (int k = n / 2; k < n; ++k)
            rcre_stream_stats_push(&b, data[k]);
        rcre_stream_stats_merge(&ab, &a);
        rcre_stream_stats_merge(&ab, &b);
        RCRE_StreamStats full;
        rcre_stream_stats_init(&full);
        for (int k = 0; k < n; ++k) rcre_stream_stats_push(&full, data[k]);
        CHECK(fabs(rcre_stream_stats_mean(&ab) - rcre_stream_stats_mean(&full)) <= 1e-12,
              "merge mean");
        CHECK(fabs(rcre_stream_stats_var(&ab) - rcre_stream_stats_var(&full)) <= 1e-9,
              "merge var");
        CHECK(fabs(rcre_stream_stats_kurtosis(&ab) -
                   rcre_stream_stats_kurtosis(&full)) <= 1e-7,
              "merge kurtosis");
        printf(" OK\n");
    }

    /* [P^2 quantiles] vs exact quantiles of the same 1e6 stream */
    {
        printf("  [P^2 quantiles]");
        const int m = 1000000;
        double* big = malloc((size_t)m * 8);
        uint64_t s2 = 0x1234567890ABCDEFull;
        for (int k = 0; k < m; ++k) big[k] = lcg64(&s2);
        /* exact quantiles via in-place sort of a copy */
        double* sorted = malloc((size_t)m * 8);
        memcpy(sorted, big, (size_t)m * 8);
        qsort(sorted, (size_t)m, 8, cmp_double);
        const double ps[3] = {0.1, 0.5, 0.9};
        for (int pi = 0; pi < 3; ++pi) {
            RCRE_P2Quantile* q = rcre_p2_create(ps[pi]);
            CHECK(q != NULL, "p2 create");
            for (int k = 0; k < m; ++k) rcre_p2_add(q, big[k]);
            double est = rcre_p2_value(q);
            double exact = sorted[(int)(ps[pi] * (double)(m - 1))];
            double err = fabs(est - exact);
            CHECK(err <= 0.02, "p2 rank error <= 0.02");
            printf("    p=%.1f est=%.4f exact=%.4f err=%.4f\n",
                   ps[pi], est, exact, err);
            rcre_p2_destroy(q);
        }
        free(big); free(sorted);
        printf(" OK\n");
    }

    /* [P^2 error inputs] */
    {
        printf("  [P^2 errors]");
        CHECK(rcre_p2_create(0.0) == NULL, "p=0 rejected");
        CHECK(rcre_p2_create(1.0) == NULL, "p=1 rejected");
        CHECK(rcre_p2_create(-0.5) == NULL, "p<0 rejected");
        rcre_p2_add(NULL, 1.0); /* must not crash */
        RCRE_P2Quantile* q = rcre_p2_create(0.5);
        CHECK(rcre_p2_value(q) == 0.0, "value at n=0 is 0");
        rcre_p2_add(q, 3.0);
        CHECK(rcre_p2_value(q) == 3.0, "value at n=1 is the sample");
        rcre_p2_destroy(q);
        printf(" OK\n");
    }

    /* [bootstrap ordinary] SE of the mean */
    {
        printf("  [bootstrap ordinary]");
        const int bn = 200;
        double* x = malloc((size_t)bn * 8);
        uint64_t s3 = 0x7777777777777777ull;
        for (int k = 0; k < bn; ++k) x[k] = lcg64(&s3) * 2.0 - 1.0;
        /* analytic SE for a symmetric uniform-ish stream:
         * SE = sd/sqrt(n); use the sample sd of x as the truth */
        double mx = 0.0, vx = 0.0;
        for (int k = 0; k < bn; ++k) mx += x[k];
        mx /= bn;
        for (int k = 0; k < bn; ++k) vx += (x[k] - mx) * (x[k] - mx);
        double truth_se = sqrt(vx / (bn - 1)) / sqrt((double)bn);
        const int nboot = 4000;
        double* boots = malloc((size_t)nboot * 8);
        int rc = rcre_bootstrap_f64(x, bn, 999, 0, nboot, 1, mean_stat,
                                    NULL, boots);
        CHECK(rc == 0, "bootstrap rc");
        double bm = 0.0;
        for (int k = 0; k < nboot; ++k) bm += boots[k];
        bm /= nboot;
        double bv = 0.0;
        for (int k = 0; k < nboot; ++k) bv += (boots[k] - bm) * (boots[k] - bm);
        double boot_se = sqrt(bv / (nboot - 1));
        double ratio = boot_se / truth_se;
        CHECK(ratio > 0.85 && ratio < 1.15,
              "bootstrap SE ratio in [0.85,1.15]");
        /* determinism: same seed+id -> bit-identical */
        double* boots2 = malloc((size_t)nboot * 8);
        CHECK(rcre_bootstrap_f64(x, bn, 999, 0, nboot, 1, mean_stat, NULL,
                                 boots2) == 0, "bootstrap rc2");
        int same = 1;
        for (int k = 0; k < nboot; ++k) if (boots[k] != boots2[k]) same = 0;
        CHECK(same, "bootstrap fixed-seed bit-identical");
        /* different resample id -> different draws (substreams differ) */
        double* boots3 = malloc((size_t)nboot * 8);
        CHECK(rcre_bootstrap_f64(x, bn, 999, 1, nboot, 1, mean_stat, NULL,
                                 boots3) == 0, "bootstrap rc3");
        int diff = 0;
        for (int k = 0; k < nboot; ++k)
            if (boots[k] != boots3[k]) { diff = 1; break; }
        CHECK(diff, "different resample id -> different draws");
        free(x); free(boots); free(boots2); free(boots3);
        printf(" OK\n");
    }

    /* [bootstrap moving block] */
    {
        printf("  [bootstrap block]");
        const int bn = 400;
        double* x = malloc((size_t)bn * 8);
        uint64_t s4 = 0x5555555555555555ull;
        double e = 0.0;
        for (int k = 0; k < bn; ++k) {
            e = 0.9 * e + (lcg64(&s4) * 2.0 - 1.0) * 0.6;
            x[k] = e;
        }
        const int nboot = 2000;
        double* boots = malloc((size_t)nboot * 8);
        int rc = rcre_bootstrap_f64(x, bn, 42, 0, nboot, 7, mean_stat, NULL,
                                    boots);
        CHECK(rc == 0, "block bootstrap rc");
        int finite = 1;
        for (int k = 0; k < nboot; ++k)
            if (!isfinite(boots[k])) finite = 0;
        CHECK(finite, "block bootstrap finite");
        free(x); free(boots);
        printf(" OK\n");
    }

    /* [error inputs] */
    {
        printf("  [error inputs]");
        double one = 0.0;
        CHECK(rcre_bootstrap_f64(NULL, 10, 1, 0, 100, 1, mean_stat, NULL,
                                 &one) == -1, "NULL x");
        CHECK(rcre_bootstrap_f64(data, 0, 1, 0, 100, 1, mean_stat, NULL,
                                 &one) == -1, "n<=0");
        CHECK(rcre_bootstrap_f64(data, 10, 1, 0, 0, 1, mean_stat, NULL,
                                 &one) == -1, "n_boot<=0");
        CHECK(rcre_bootstrap_f64(data, 10, 1, 0, 100, 1, NULL, NULL,
                                 &one) == -1, "NULL stat");
        CHECK(rcre_bootstrap_f64(data, 10, 1, 0, 100, 1, mean_stat, NULL,
                                 NULL) == -1, "NULL out");
        CHECK(rcre_bootstrap_f64(data, 10, 1, 0, 100, -3, mean_stat, NULL,
                                 &one) == -1, "block<0");
        printf(" OK\n");
    }

    free(data);
    printf("\n%s (%d checks, %d failures)\n",
           g_fails == 0 ? "ALL OK" : "FAILURES", g_checks, g_fails);
    return g_fails;
}
