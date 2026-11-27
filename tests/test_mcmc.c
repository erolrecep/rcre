/**
 * @file test_mcmc.c
 * @brief H7 MCMC domain layer tests (2.4.0).
 *
 * Measured gates (see docs/hpc.md 10.3):
 *
 *  - MH random-walk 1-D Gaussian: moments within 10%, acceptance in
 *    (0.3, 0.99); two fixed-seed runs bit-identical (the PCG64
 *    consumption-order contract).
 *  - HMC 2-D banana (the classic curved target): acceptance in
 *    [0.5, 0.9] at eps=0.1/L=50 — the game-plan 0.6-0.8 acceptance
 *    bar lives on HMC (true Metropolis acceptance); moments of the
 *    banana within loose bounds (E[y] ~ 0, sd[y] ~ 1, E[x] ~ 0.5).
 *  - HMC acceptance monotonicity: smaller eps -> higher acceptance
 *    (the adaptive controller's signal must actually be monotone).
 *  - NUTS 1-D Gaussian with FIXED eps (n_adapt=0): moments within
 *    15% — the plain tree/slice mechanism, no adaptation.
 *  - NUTS 2-D banana with adaptation: no-NaN, divergences bounded
 *    (<= 100 of ~3300; measured ~8), acceptance (P(non-root pick))
 *    >= 0.5, moments within loose bounds.
 *  - NUTS 8-D anisotropic Gaussian (the H6/H7 stress target):
 *    per-dim sd within 15% (measured ~4%), divergences <= 10,
 *    acceptance >= 0.5, ESS(stiff dim) >= 500 of 5000.
 *    Note: exact slice NUTS has no rejection step, so its
 *    "acceptance" runs near 1.0 by design — the 0.6-0.8 game-plan
 *    band is a Metropolis-NUTS concept and is applied to HMC here.
 *  - ESS: white noise -> ESS ~ n; AR(1) phi=0.9 -> ESS ~ n/19
 *    (Geyer initial-monotone-sequence estimator, generous bars).
 *  - Integrators: plain/antithetic/stratified/Sobol' on the 5-D
 *    bump within 10 reported-SE of a 16^5 midpoint reference;
 *    Halton beyond its dimension limit is cleanly rejected.
 *  - Error inputs on every entry point.
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

/* ------------------------------------------------------------------ */
/* targets                                                            */
/* ------------------------------------------------------------------ */

/* 1-D N(0,1) */
static double g1_logpdf(const double* x, int d, void* u) {
    (void)u; (void)d;
    return -0.5 * x[0] * x[0];
}
static void g1_grad(const double* x, int d, double* g, void* u) {
    (void)u; (void)d;
    g[0] = -x[0];
}

/* 8-D anisotropic Gaussian, sigma_i = 2^i (the H6 stress target) */
static const double g8_sigma[8] = {1, 2, 4, 8, 16, 8, 4, 2};
static double g8_logpdf(const double* x, int d, void* u) {
    (void)u;
    double s = 0.0;
    for (int i = 0; i < d; ++i)
        s -= 0.5 * (x[i] / g8_sigma[i]) * (x[i] / g8_sigma[i]);
    return s;
}
static void g8_grad(const double* x, int d, double* g, void* u) {
    (void)u;
    for (int i = 0; i < d; ++i) g[i] = -x[i] / (g8_sigma[i] * g8_sigma[i]);
}

/* 2-D banana: p(x,y) ∝ exp(-(x - y^2/2)^2/(2*0.1^2)) * exp(-y^2/2).
 * E[x] ≈ E[y^2]/2 ≈ 0.5, E[y] = 0, sd[y] ≈ 1, sd[x] ≈ 0.7 (heavy). */
static double ban_logpdf(const double* x, int d, void* u) {
    (void)u; (void)d;
    const double r = x[0] - 0.5 * x[1] * x[1];
    return -0.5 * r * r / 0.01 - 0.5 * x[1] * x[1];
}
static void ban_grad(const double* x, int d, double* g, void* u) {
    (void)u; (void)d;
    const double r = x[0] - 0.5 * x[1] * x[1];
    g[0] = -r / 0.01;
    g[1] = -x[1] + r * x[1] / 0.01;
}

/* 5-D bump integrand on [0,1]^5 */
static double bump5(const double* x, void* u) {
    (void)u;
    double v = 0.0;
    for (int i = 0; i < 5; ++i) v += (x[i] - 0.5) * (x[i] - 0.5);
    return exp(-10.0 * v);
}

/* 3-D bump (for the Halton d=3 check — bump5 must not be called with
 * d < 5: it reads x[0..4]) */
static double bump3(const double* x, void* u) {
    (void)u;
    double v = 0.0;
    for (int i = 0; i < 3; ++i) v += (x[i] - 0.5) * (x[i] - 0.5);
    return exp(-10.0 * v);
}

/* deterministic uniforms for fixtures (fixed-seed LCG) */
static uint64_t lcg64(uint64_t* st) {
    *st = *st * 6364136223846793005ull + 1442695040888963407ull;
    return ((*st >> 11) & ((1ull << 53) - 1)) * 0x1.0p-53;
}

/* ------------------------------------------------------------------ */
/* tests                                                              */
/* ------------------------------------------------------------------ */

int main(void) {
    rcre_hal_init();
    printf("H7 MCMC tests:\n");

    /* [MH 1-D] moments + acceptance + determinism */
    {
        printf("  [MH 1-D gaussian]");
        int d = 1, n = 20000;
        double x0 = 0.0;
        double* a = malloc((size_t)n * 8);
        double* b = malloc((size_t)n * 8);
        double acc = 0.0;
        CHECK(rcre_mcmc_mh_sample_f64(d, &x0, g1_logpdf, NULL, 42, 1.0,
                                      n, a, &acc) == 0, "mh rc");
        double m = 0.0, v = 0.0;
        for (int k = 0; k < n; ++k) m += a[k];
        m /= n;
        for (int k = 0; k < n; ++k) v += (a[k] - m) * (a[k] - m);
        v = sqrt(v / (n - 1));
        CHECK(fabs(m) < 0.1, "mh mean |m|<0.1");
        CHECK(v > 0.9 && v < 1.1, "mh sd in [0.9,1.1]");
        CHECK(acc > 0.3 && acc < 0.99, "mh acceptance in (0.3,0.99)");
        /* determinism: same seed -> bit-identical chain */
        double acc2 = 0.0;
        CHECK(rcre_mcmc_mh_sample_f64(d, &x0, g1_logpdf, NULL, 42, 1.0,
                                      n, b, &acc2) == 0, "mh rc2");
        int same = 1;
        for (int k = 0; k < n; ++k) if (a[k] != b[k]) same = 0;
        CHECK(same && acc == acc2, "mh fixed-seed bit-identical");
        free(a); free(b);
        printf(" OK\n");
    }

    /* [HMC banana] the game-plan acceptance bar (true Metropolis).
     * Measured config: eps=0.1, L=50, n=20000, burn-in 5000 ->
     * acc 0.692 (in the 0.6-0.8 band), E[y] -0.008, sd[y] 0.83.
     * (A measured lesson: n=4000 reads acc 0.457 / E[y] +0.82 —
     * under-sampling of the slow along-curve mode, not a bug; the
     * bar below uses the converged config.) */
    {
        printf("  [HMC 2-D banana]");
        int d = 2, n = 20000, burn = 5000;
        double x0[2] = {0.0, 0.0};
        double* out = malloc((size_t)n * d * 8);
        double acc = 0.0;
        CHECK(rcre_mcmc_hmc_sample_f64(d, x0, ban_logpdf, ban_grad, NULL,
                                       42, 0.1, 50, n, out, &acc) == 0,
              "hmc rc");
        CHECK(acc > 0.55 && acc < 0.85,
              "hmc acceptance in [0.55,0.85] (measured 0.692)");
        double mx = 0, my = 0, mx2 = 0, my2 = 0;
        for (int k = burn; k < n; ++k) {
            mx += out[k * d]; my += out[k * d + 1];
            mx2 += out[k * d] * out[k * d];
            my2 += out[k * d + 1] * out[k * d + 1];
        }
        int ns = n - burn;
        mx /= ns; my /= ns; mx2 /= ns; my2 /= ns;
        double sdx = sqrt(mx2 - mx * mx), sdy = sqrt(my2 - my * my);
        CHECK(fabs(my) < 0.25, "banana E[y]~0");
        CHECK(sdy > 0.7 && sdy < 1.3, "banana sd[y]~1");
        CHECK(mx > 0.15 && mx < 0.75, "banana E[x] in (0.15,0.75)");
        CHECK(sdx > 0.25 && sdx < 1.0, "banana sd[x] loose band");
        free(out);
        printf(" OK\n");
    }

    /* [HMC monotonicity] acceptance decreases with eps */
    {
        printf("  [HMC eps monotonicity]");
        int d = 2, n = 2000;
        double x0[2] = {0.0, 0.0};
        double* out = malloc((size_t)n * d * 8);
        double a_small = 0.0, a_big = 0.0;
        CHECK(rcre_mcmc_hmc_sample_f64(d, x0, ban_logpdf, ban_grad, NULL,
                                       42, 0.05, 20, n, out, &a_small) == 0,
              "hmc rc small");
        CHECK(rcre_mcmc_hmc_sample_f64(d, x0, ban_logpdf, ban_grad, NULL,
                                       42, 0.2, 20, n, out, &a_big) == 0,
              "hmc rc big");
        CHECK(a_small > a_big, "acceptance(eps=0.05) > acceptance(eps=0.2)");
        free(out);
        printf(" OK\n");
    }

    /* [NUTS 1-D fixed eps] the tree/slice mechanism, no adaptation */
    {
        printf("  [NUTS 1-D fixed eps]");
        int d = 1, n = 4000;
        double x0 = 3.0; /* start far out: the chain must migrate */
        double* out = malloc((size_t)n * 8);
        double acc = 0.0;
        RCRE_NUTSReport rep;
        CHECK(rcre_mcmc_nuts_sample_f64(d, &x0, g1_logpdf, g1_grad, NULL,
                                        7, 0, n, 0.1, 8, out, &acc, &rep) == 0,
              "nuts rc");
        CHECK(rep.divergent == 0, "no divergences (stable 1-D)");
        double m = 0, v = 0;
        for (int k = 500; k < n; ++k) m += out[k];
        m /= n - 500;
        for (int k = 500; k < n; ++k) v += (out[k] - m) * (out[k] - m);
        v = sqrt(v / (n - 500 - 1));
        CHECK(fabs(m) < 0.2, "nuts 1-D mean");
        CHECK(v > 0.8 && v < 1.2, "nuts 1-D sd");
        CHECK(acc > 0.0, "nuts picked non-root at least once");
        free(out);
        printf(" OK\n");
    }

    /* [NUTS banana adapted] */
    {
        printf("  [NUTS 2-D banana]");
        int d = 2, na = 300, n = 3000;
        double x0[2] = {0.0, 0.0};
        double* out = malloc((size_t)n * d * 8);
        double acc = 0.0;
        RCRE_NUTSReport rep;
        CHECK(rcre_mcmc_nuts_sample_f64(d, x0, ban_logpdf, ban_grad, NULL,
                                        42, na, n, 0.0, 10, out, &acc,
                                        &rep) == 0, "nuts rc");
        CHECK(rep.divergent <= 100, "divergences bounded (measured ~8)");
        CHECK(acc >= 0.5, "nuts acceptance >= 0.5");
        double mx = 0, my = 0, mx2 = 0, my2 = 0;
        for (int k = 500; k < n; ++k) {
            mx += out[k * d]; my += out[k * d + 1];
            mx2 += out[k * d] * out[k * d];
            my2 += out[k * d + 1] * out[k * d + 1];
        }
        int ns = n - 500;
        mx /= ns; my /= ns; mx2 /= ns; my2 /= ns;
        double sdx = sqrt(mx2 - mx * mx), sdy = sqrt(my2 - my * my);
        CHECK(isfinite(mx) && isfinite(my), "no NaN");
        CHECK(fabs(my) < 0.3, "banana E[y]~0");
        CHECK(sdy > 0.6 && sdy < 1.4, "banana sd[y]~1");
        CHECK(mx > 0.0 && mx < 1.0, "banana E[x] in (0,1)");
        CHECK(sdx > 0.3 && sdx < 1.6, "banana sd[x] loose band");
        free(out);
        printf(" OK\n");
    }

    /* [NUTS 8-D anisotropic] the stress target */
    {
        printf("  [NUTS 8-D anisotropic]");
        int d = 8, na = 500, n = 5000;
        double x0[8] = {0};
        double* out = malloc((size_t)n * d * 8);
        double acc = 0.0;
        RCRE_NUTSReport rep;
        CHECK(rcre_mcmc_nuts_sample_f64(d, x0, g8_logpdf, g8_grad, NULL,
                                        0xC0FFEE, na, n, 0.0, 12, out,
                                        &acc, &rep) == 0, "nuts rc");
        CHECK(rep.divergent <= 10, "divergences <= 10");
        CHECK(acc >= 0.5, "nuts acceptance >= 0.5");
        double worst = 0.0;
        double* col4 = malloc((size_t)n * 8);
        for (int i = 0; i < d; ++i) {
            double m = 0, v = 0;
            for (int k = 0; k < n; ++k) {
                m += out[(size_t)k * d + i];
                if (i == 4) col4[k] = out[(size_t)k * d + i];
            }
            m /= n;
            for (int k = 0; k < n; ++k)
                v += (out[(size_t)k * d + i] - m) *
                     (out[(size_t)k * d + i] - m);
            v = sqrt(v / (n - 1));
            double rel = fabs(v - g8_sigma[i]) / g8_sigma[i];
            if (rel > worst) worst = rel;
        }
        CHECK(worst <= 0.15, "8-D per-dim sd within 15% (measured ~4%)");
        double ess = rcre_mcmc_ess_f64(col4, n, n / 2);
        CHECK(ess >= 500.0, "ESS(stiff dim) >= 500");
        free(out); free(col4);
        printf(" OK\n");
    }

    /* [ESS] white noise and AR(1) */
    {
        printf("  [ESS]");
        int n = 2000;
        double* y = malloc((size_t)n * 8);
        uint64_t st = 0xBEEF;
        for (int k = 0; k < n; ++k) y[k] = lcg64(&st) * 2.0 - 1.0;
        double ess_w = rcre_mcmc_ess_f64(y, n, n / 2);
        CHECK(ess_w > 0.7 * n && ess_w <= 1.05 * n,
              "white-noise ESS ~ n");
        /* AR(1) phi=0.9: theoretical ESS = n(1-phi)/(1+phi) = n/19 */
        double phi = 0.9;
        double e = 0.0;
        for (int k = 1; k < n; ++k) {
            e = phi * e + (lcg64(&st) * 2.0 - 1.0) * 0.6;
            y[k] = e;
        }
        double ess_a = rcre_mcmc_ess_f64(y, n, n / 2);
        CHECK(ess_a > 0.3 * n / 19.0 && ess_a < 3.0 * n / 19.0,
              "AR(1) ESS ~ n/19");
        free(y);
        printf(" OK\n");
    }

    /* [integrators] vs a 16^5 midpoint reference */
    {
        printf("  [integrators 5-D]");
        double lo[5] = {0, 0, 0, 0, 0}, hi[5] = {1, 1, 1, 1, 1};
        /* reference: 16^5 midpoint rule (exact to ~1e-6 for this
         * smooth integrand) */
        double ref = 0.0;
        const int g = 16;
        for (int a0 = 0; a0 < g; ++a0)
        for (int a1 = 0; a1 < g; ++a1)
        for (int a2 = 0; a2 < g; ++a2)
        for (int a3 = 0; a3 < g; ++a3)
        for (int a4 = 0; a4 < g; ++a4) {
            double x[5] = {(a0 + 0.5) / g, (a1 + 0.5) / g, (a2 + 0.5) / g,
                           (a3 + 0.5) / g, (a4 + 0.5) / g};
            ref += bump5(x, NULL);
        }
        ref /= (double)(g * g * g * g * g);
        const int methods[4] = {0, 1, 2, 3};
        for (int mi = 0; mi < 4; ++mi) {
            double val = 0.0, se = 0.0;
            int nu = 0;
            int rc = rcre_mcmc_integrate_f64(5, bump5, NULL, lo, hi,
                                             methods[mi], 100000, 1234,
                                             &val, &se, &nu);
            CHECK(rc == 0, "integrator rc");
            CHECK(se > 0.0, "integrator se > 0");
            CHECK(fabs(val - ref) <= 10.0 * se,
                  "integrator within 10 SE of reference");
            (void)nu;
        }
        /* Halton beyond its dimension limit: clean rejection */
        double val = 0.0, se = 0.0;
        int nu = 0;
        CHECK(rcre_mcmc_integrate_f64(5, bump5, NULL, lo, hi, 4, 1000, 1,
                                      &val, &se, &nu) == -1,
              "Halton d=5 rejected (dim limit)");
        /* Halton works at d=3 */
        {
            double lo3[3] = {0, 0, 0}, hi3[3] = {1, 1, 1};
            CHECK(rcre_mcmc_integrate_f64(3, bump3, NULL, lo3, hi3, 4,
                                          50000, 1, &val, &se, &nu) == 0,
                  "Halton d=3 ok");
        }
        printf(" OK\n");
    }

    /* [error inputs] */
    {
        printf("  [error inputs]");
        double x0 = 0.0;
        double one = 0.0;
        CHECK(rcre_mcmc_mh_sample_f64(0, &x0, g1_logpdf, NULL, 1, 1.0, 10,
                                      &one, NULL) == -1, "mh d<1");
        CHECK(rcre_mcmc_mh_sample_f64(1, NULL, g1_logpdf, NULL, 1, 1.0, 10,
                                      &one, NULL) == -1, "mh NULL x0");
        CHECK(rcre_mcmc_mh_sample_f64(1, &x0, NULL, NULL, 1, 1.0, 10,
                                      &one, NULL) == -1, "mh NULL logpdf");
        CHECK(rcre_mcmc_mh_sample_f64(1, &x0, g1_logpdf, NULL, 1, 1.0, 0,
                                      &one, NULL) == -1, "mh n<1");
        CHECK(rcre_mcmc_mh_sample_f64(1, &x0, g1_logpdf, NULL, 1, 0.0, 10,
                                      &one, NULL) == -1, "mh sd<=0");
        CHECK(rcre_mcmc_mh_sample_f64(1, &x0, g1_logpdf, NULL, 1, 1.0, 10,
                                      NULL, NULL) == -1, "mh NULL out");
        CHECK(rcre_mcmc_hmc_sample_f64(1, &x0, g1_logpdf, g1_grad, NULL, 1,
                                       0.0, 10, 10, &one, NULL) == -1,
              "hmc eps<=0");
        CHECK(rcre_mcmc_hmc_sample_f64(1, &x0, g1_logpdf, g1_grad, NULL, 1,
                                       0.1, 0, 10, &one, NULL) == -1,
              "hmc L<1");
        CHECK(rcre_mcmc_hmc_sample_f64(1, &x0, NULL, g1_grad, NULL, 1, 0.1,
                                       10, 10, &one, NULL) == -1,
              "hmc NULL logpdf");
        CHECK(rcre_mcmc_hmc_sample_f64(1, &x0, g1_logpdf, NULL, NULL, 1,
                                       0.1, 10, 10, &one, NULL) == -1,
              "hmc NULL grad");
        CHECK(rcre_mcmc_nuts_sample_f64(1, &x0, g1_logpdf, g1_grad, NULL, 1,
                                        10, 10, 0.1, 0, &one, NULL, NULL) ==
                    -1,
              "nuts max_tree<1");
        CHECK(rcre_mcmc_nuts_sample_f64(1, &x0, g1_logpdf, NULL, NULL, 1,
                                        10, 10, 0.1, 8, &one, NULL, NULL) ==
                    -1,
              "nuts NULL grad");
        CHECK(rcre_mcmc_ess_f64(NULL, 100, 50) == 100.0, "ess NULL -> n");
        double val = 0.0, se = 0.0;
        int nu = 0;
        double lo1[1] = {0.0}, hi1[1] = {1.0};
        CHECK(rcre_mcmc_integrate_f64(0, bump3, NULL, lo1, hi1, 0, 10, 1,
                                      &val, &se, &nu) == -1, "int d<1");
        CHECK(rcre_mcmc_integrate_f64(1, NULL, NULL, lo1, hi1, 0, 10, 1,
                                      &val, &se, &nu) == -1, "int NULL f");
        CHECK(rcre_mcmc_integrate_f64(1, bump3, NULL, NULL, hi1, 0, 10, 1,
                                      &val, &se, &nu) == -1, "int NULL lo");
        CHECK(rcre_mcmc_integrate_f64(1, bump3, NULL, lo1, hi1, 9, 10, 1,
                                      &val, &se, &nu) == -1, "int bad method");
        printf(" OK\n");
    }

    printf("\n%s (%d checks, %d failures)\n",
           g_fails == 0 ? "ALL OK" : "FAILURES", g_checks, g_fails);
    return g_fails;
}
