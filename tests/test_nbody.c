/*
 * H7 N-body (Barnes-Hut) formal test suite.
 *
 * Gates:
 *   [bh vs direct]  theta -> 0 agrees with the O(n^2) direct force
 *   [theta scaling] approximation error decreases as theta shrinks
 *   [pool determinism] pool(8) bit-identical to serial
 *   [2-body elliptic] symplectic energy conservation (E0 < 0)
 *   [2-body parabolic] energy bounded on an escape orbit (E0 = 0)
 *   [mass weighting] central + satellite (regression: force-vs-accel)
 *   [chaotic]       800-body 10k-step run: no NaN, bounded drift
 *   [error inputs]  bad arguments rejected
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

static uint64_t lcg(uint64_t* st) {
    *st = *st * 6364136223846793005ull + 1442695040888963407ull;
    return (*st >> 11) & ((1ull << 53) - 1);
}

static double direct_force_max_rel(const double* x, const double* y,
                                   const double* m, int n, double G,
                                   double soft, const double* fx,
                                   const double* fy) {
    double mx = 0.0, sc = 0.0;
    for (int i = 0; i < n; ++i) {
        double ax = 0.0, ay = 0.0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const double dx = x[j] - x[i], dy = y[j] - y[i];
            const double r2 = dx * dx + dy * dy + soft * soft;
            const double f = G * m[i] * m[j] / (r2 * sqrt(r2));
            ax += f * dx;
            ay += f * dy;
        }
        const double d = fabs(fx[i] - ax) + fabs(fy[i] - ay);
        const double v = fabs(ax) + fabs(ay);
        if (d > mx) mx = d;
        if (v > sc) sc = v;
    }
    return sc > 0.0 ? mx / sc : mx;
}

int main(void) {
    rcre_hal_init();
    printf("H7 N-body tests:\n");

    /* ---------------------------------------------------------- */
    {
        printf("  [bh vs direct]");
        int n = 64;
        double* x = malloc(n * 8), *y = malloc(n * 8), *m = malloc(n * 8);
        double* fx = malloc(n * 8), *fy = malloc(n * 8);
        uint64_t st = 0xABCD;
        for (int i = 0; i < n; ++i) {
            x[i] = (double)lcg(&st) * 0x1.0p-53 * 2.0 - 1.0;
            y[i] = (double)lcg(&st) * 0x1.0p-53 * 2.0 - 1.0;
            m[i] = 0.1 + (double)lcg(&st) * 0x1.0p-53;
        }
        const double G = 1.0, soft = 0.05;
        CHECK(rcre_nb_bh_force_f64(x, y, m, n, G, 1e-8, soft, NULL, fx, fy) == 0,
              "force rc");
        const double rel = direct_force_max_rel(x, y, m, n, G, soft, fx, fy);
        printf(" rel=%.3e ", rel);
        CHECK(rel < 1e-10, "theta->0 agrees with direct O(n^2)");

        /* theta scaling: error must decrease as theta shrinks */
        double e50 = 0.0, e20 = 0.0, e10 = 0.0;
        for (double th = 0.5; th > 0.05; th -= 0.1) {
            (void)rcre_nb_bh_force_f64(x, y, m, n, G, th, soft, NULL, fx, fy);
            const double e = direct_force_max_rel(x, y, m, n, G, soft, fx, fy);
            if (th > 0.45) e50 = e;
            if (th > 0.15 && th <= 0.25) e20 = e;
            if (th <= 0.15) e10 = e;
        }
        printf("e50=%.2e e20=%.2e e10=%.2e", e50, e20, e10);
        CHECK(e20 < e50 && e10 < e20, "error decreases as theta shrinks");
        CHECK(e10 < 5e-3, "theta=0.1 error < 5e-3");
        printf(" OK\n");
        free(x); free(y); free(m); free(fx); free(fy);
    }

    /* ---------------------------------------------------------- */
    {
        printf("  [pool determinism]");
        int n = 300;
        double* x = malloc(n * 8), *y = malloc(n * 8), *m = malloc(n * 8);
        double* fs = malloc(2 * n * 8), *fp = malloc(2 * n * 8);
        uint64_t st = 0xEF01;
        for (int i = 0; i < n; ++i) {
            x[i] = (double)lcg(&st) * 0x1.0p-53 * 2.0 - 1.0;
            y[i] = (double)lcg(&st) * 0x1.0p-53 * 2.0 - 1.0;
            m[i] = 0.1 + (double)lcg(&st) * 0x1.0p-53;
        }
        CHECK(rcre_nb_bh_force_f64(x, y, m, n, 1.0, 0.3, 0.1, NULL, fs, fs + n) == 0,
              "serial rc");
        RCRE_ThreadPool* pool = rcre_threadpool_create(8);
        CHECK(pool != NULL, "pool create");
        CHECK(rcre_nb_bh_force_f64(x, y, m, n, 1.0, 0.3, 0.1, pool, fp, fp + n) == 0,
              "pool rc");
        rcre_threadpool_destroy(pool);
        CHECK(memcmp(fs, fp, 2 * n * 8) == 0, "pool(8) bit-identical to serial");
        printf(" OK\n");
        free(x); free(y); free(m); free(fs); free(fp);
    }

    /* ---------------------------------------------------------- */
    {
        printf("  [2-body elliptic]");
        double r = 1.0, G = 1.0;
        const double vc = sqrt(G / r);
        double x[2] = {-0.5, 0.5}, y[2] = {0.0, 0.0};
        double vx[2] = {0.0, 0.0}, vy[2] = {0.8 * vc, -0.8 * vc};
        double m[2] = {1.0, 1.0};
        const double E0 = rcre_nb_energy_f64(x, y, vx, vy, m, 2, G, 0.0);
        CHECK(E0 < 0.0, "bound orbit E0 < 0");
        double worst = 0.0;
        for (int k = 0; k < 20000; ++k) {
            CHECK(rcre_nb_verlet_step_f64(x, y, vx, vy, m, 2, G, 0.5, 0.0,
                                          0.001, NULL) == 0,
                  "step rc");
            if (k % 100 == 99) {
                const double E = rcre_nb_energy_f64(x, y, vx, vy, m, 2, G, 0.0);
                const double d = fabs(E - E0) / fabs(E0);
                if (d > worst) worst = d;
            }
        }
        printf(" worst=%.3e ", worst);
        CHECK(worst < 1e-5, "20k-step drift < 1e-5");
        printf(" OK\n");
    }

    /* ---------------------------------------------------------- */
    {
        printf("  [2-body parabolic]");
        double x[2] = {-0.5, 0.5}, y[2] = {0.0, 0.0};
        double vx[2] = {0.0, 0.0}, vy[2] = {1.0, -1.0};
        double m[2] = {1.0, 1.0};
        const double G = 1.0;
        const double E0 = rcre_nb_energy_f64(x, y, vx, vy, m, 2, G, 0.0);
        CHECK(fabs(E0) < 1e-15, "parabolic E0 = 0");
        double worst = 0.0;
        for (int k = 0; k < 50000; ++k) {
            (void)rcre_nb_verlet_step_f64(x, y, vx, vy, m, 2, G, 0.5, 0.0,
                                          0.001, NULL);
            if (k % 1000 == 999) {
                const double E = rcre_nb_energy_f64(x, y, vx, vy, m, 2, G, 0.0);
                const double d = fabs(E - E0); /* absolute: E0 = 0 */
                if (d > worst) worst = d;
            }
        }
        printf(" worst=%.3e ", worst);
        CHECK(worst < 1e-5, "50k-step absolute drift < 1e-5");
        printf(" OK\n");
    }

    /* ---------------------------------------------------------- */
    {
        printf("  [mass weighting]");
        /* central M=100 + 1 satellite m=0.01 on a circular orbit:
         * regression gate for the force-vs-acceleration draft bug
         * (400% "drift" that mass-1 tests could not see). */
        const double G = 1.0, M = 100.0, r = 0.5, soft = 0.02;
        const double v = sqrt(G * M / r);
        double x[2] = {0.0, r}, y[2] = {0.0, 0.0};
        double vx[2] = {0.0, 0.0}, vy[2] = {0.0, v};
        double m[2] = {M, 0.01};
        const double E0 = rcre_nb_energy_f64(x, y, vx, vy, m, 2, G, soft);
        CHECK(E0 < 0.0, "bound E0 < 0");
        double worst = 0.0;
        for (int k = 0; k < 20000; ++k) {
            (void)rcre_nb_verlet_step_f64(x, y, vx, vy, m, 2, G, 0.5, soft,
                                          0.0005, NULL);
            if (k % 200 == 199) {
                const double E = rcre_nb_energy_f64(x, y, vx, vy, m, 2, G, soft);
                const double d = fabs(E - E0) / fabs(E0);
                if (d > worst) worst = d;
            }
        }
        printf(" worst=%.3e ", worst);
        CHECK(worst < 1e-4, "20k-step drift < 1e-4 (mass-100:0.01)");
        printf(" OK\n");
    }

    /* ---------------------------------------------------------- */
    {
        printf("  [chaotic]");
        int n = 800;
        double G = 1.0, soft = 0.3;
        double* x = malloc(n * 8), *y = malloc(n * 8);
        double* vx = malloc(n * 8), *vy = malloc(n * 8), *m = malloc(n * 8);
        uint64_t st = 0xE17E;
        for (int i = 0; i < n; ++i) {
            x[i] = (double)lcg(&st) * 0x1.0p-53 * 4.0 - 2.0;
            y[i] = (double)lcg(&st) * 0x1.0p-53 * 4.0 - 2.0;
            vx[i] = (double)lcg(&st) * 0x1.0p-53 * 0.2 - 0.1;
            vy[i] = (double)lcg(&st) * 0x1.0p-53 * 0.2 - 0.1;
            m[i] = 0.1 + (double)lcg(&st) * 0x1.0p-53 * 0.9;
        }
        const double E0 = rcre_nb_energy_f64(x, y, vx, vy, m, n, G, soft);
        int bad = 0;
        double worst = 0.0;
        for (int k = 0; k < 10000; ++k) {
            if (rcre_nb_verlet_step_f64(x, y, vx, vy, m, n, G, 0.5, soft,
                                        0.01, NULL) != 0)
                ++bad;
            if (k % 1000 == 999) {
                const double E = rcre_nb_energy_f64(x, y, vx, vy, m, n, G, soft);
                double d = 0.0;
                for (int i = 0; i < n; ++i)
                    if (!(isfinite(x[i]) && isfinite(y[i]) &&
                          isfinite(vx[i]) && isfinite(vy[i])))
                        d = 1.0;
                if (d == 0.0)
                    d = fabs(E - E0) / fabs(E0);
                if (d > worst) worst = d;
            }
        }
        printf(" worst=%.4e ", worst);
        CHECK(bad == 0, "no step failures");
        CHECK(worst < 0.25, "bounded drift (chaotic, not a 1e-6 case)");
        printf(" OK\n");
        free(x); free(y); free(vx); free(vy); free(m);
    }

    /* ---------------------------------------------------------- */
    {
        printf("  [error inputs]");
        double x[1] = {0.0}, y[1] = {0.0}, m[1] = {1.0}, f[1] = {0.0};
        CHECK(rcre_nb_bh_force_f64(NULL, y, m, 1, 1.0, 0.5, 0.0, NULL, f, f) == -1,
              "null x");
        CHECK(rcre_nb_bh_force_f64(x, y, m, 0, 1.0, 0.5, 0.0, NULL, f, f) == -1,
              "n=0");
        CHECK(rcre_nb_bh_force_f64(x, y, m, 1, -1.0, 0.5, 0.0, NULL, f, f) == -1,
              "G<=0");
        CHECK(rcre_nb_bh_force_f64(x, y, m, 1, 1.0, 0.0, 0.0, NULL, f, f) == -1,
              "theta<=0");
        CHECK(rcre_nb_bh_force_f64(x, y, m, 1, 1.0, 1.5, 0.0, NULL, f, f) == -1,
              "theta>=1");
        double vx[1] = {0.0}, vy[1] = {0.0};
        CHECK(rcre_nb_verlet_step_f64(x, y, vx, vy, m, 1, 1.0, 0.5, 0.0,
                                      0.0, NULL) == -1,
              "h<=0");
        CHECK(rcre_nb_energy_f64(x, y, vx, vy, m, 1, 1.0, 0.0) == 0.0,
              "single-body energy = 0");
        printf(" OK\n");
    }

    printf("\n%s (%d checks, %d failures)\n",
           g_fails == 0 ? "ALL OK" : "FAILURES", g_checks, g_fails);
    return g_fails;
}
