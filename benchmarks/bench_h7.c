/* RCRE H7 benchmark — domain layer (2.4.0)
 *
 * Measures (Apple Silicon; F64):
 *
 *   Poisson 2D     spectral (FFT) vs MG vs AUTO, 128^2 .. 2048^2
 *   Heat 2D        FTCS 500 steps, 256x256 (serial vs pool)
 *   Burgers 1D     Shu-Osher one shock transit, n=4096
 *   Wave 1D        DDK leapfrog 10k steps, n=1024
 *   N-body BH      force build+walk, n = 1k/10k/100k (theta=0.5, serial vs pool)
 *   N-body Verlet  1k steps, n=10k
 *   MCMC           NUTS/HMC wall time (banana + 8-D Gaussian)
 *   streaming      P^2 add rate, bootstrap, ESS
 *
 * All bars are measured, not claimed.
 */
#include "rcre.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static double* fill_f64(size_t n, uint64_t seed) {
    double* p = (double*)malloc(n * sizeof(double));
    if (!p) return NULL;
    uint64_t st = seed;
    for (size_t i = 0; i < n; ++i) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        p[i] = ((double)((st >> 11) & ((1ull << 53) - 1)) * 0x1.0p-53) * 2.0 - 1.0;
    }
    return p;
}

/* MCMC benchmark targets (file scope: C has no nested functions) */
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

static double mean_stat(const double* y, int n, void* u) {
    (void)u;
    double m = 0.0;
    for (int i = 0; i < n; ++i) m += y[i];
    return m / (double)n;
}

int main(void) {
    rcre_hal_init();
    RCRE_ThreadPool* pool = rcre_threadpool_create(0);
    const int nt = (int)sysconf(_SC_NPROCESSORS_ONLN);

    printf("== RCRE H7 domain-layer benchmark (pool=%d threads) ==\n\n", nt);

    /* ---------------------------------------------------------- */
    /* Spectral sizes are n-1 grids (127, 255, ...): the DST-I needs
     * an FFT of length 2(n+1), and n+1 a power of two keeps it off
     * the 7x-slower Bluestein path (measured H2 bar: 7.5 vs 1.1
     * Ms/s). */
    printf("Poisson 2D (F64, zero Dirichlet, random RHS; n-1 grids:\n"
           "DST length n+1 is a power of two -> no Bluestein):\n");
    printf("  n x n      cells     spectral   MG@1e-10   AUTO      resid\n");
    for (int nx = 127; nx <= 1023; nx = 2 * nx + 1) {
        const size_t nn = (size_t)nx * nx;
        double* b = fill_f64(nn, 0x51u + (uint64_t)nx);
        double* us = (double*)malloc(nn * 8);
        double* um = (double*)malloc(nn * 8);
        double* ua = (double*)malloc(nn * 8);
        double tb = now_ms();
        (void)rcre_pde_poisson2d_fft_f64(nx, nx, b, us, pool);
        double ts = now_ms() - tb;
        tb = now_ms();
        (void)rcre_mg_solve_poisson2d_f64(nx, nx, b, um, 1e-10, 300, NULL,
                                          NULL);
        double tm = now_ms() - tb;
        tb = now_ms();
        (void)rcre_pde_poisson2d_solve_f64(nx, nx, b, ua, pool, 1);
        double ta = now_ms() - tb;
        /* true relative residual of the AUTO solution */
        double rf = 0.0, ra = 0.0;
        for (size_t i = 0; i < nn; ++i) {
            rf += b[i] * b[i];
            /* A u in MG layout */
            int i0 = (int)(i % (size_t)nx), j0 = (int)(i / (size_t)nx);
            double s = 0.0;
            if (i0 > 0) s -= ua[i - 1];
            if (i0 < nx - 1) s -= ua[i + 1];
            if (j0 > 0) s -= ua[i - nx];
            if (j0 < nx - 1) s -= ua[i + nx];
            s += 4.0 * ua[i];
            ra += (b[i] - s) * (b[i] - s);
        }
        printf("  %4d x %4d %9zu  %8.3f ms %8.3f ms %8.3f ms %9.2e\n",
               nx, nx, nn, ts, tm, ta, sqrt(ra / rf));
        free(b); free(us); free(um); free(ua);
    }

    /* ---------------------------------------------------------- */
    printf("\nHeat 2D FTCS 500 steps, 512x512 (h=1/512, dt=0.5 h^2):\n");
    {
        const int n = 512, nn = n * n;
        double* u = fill_f64(nn, 0x47);
        double* u1 = (double*)malloc(nn * 8);
        const double h = 1.0 / n, dt = 0.5 * h * h;
        double t = now_ms();
        for (int k = 0; k < 500; ++k)
            (void)rcre_pde_heat2d_step_f64(n, n, u, u1, dt, h, NULL);
        double ts = now_ms() - t;
        t = now_ms();
        for (int k = 0; k < 500; ++k)
            (void)rcre_pde_heat2d_step_f64(n, n, u, u1, dt, h, pool);
        double tp = now_ms() - t;
        printf("  serial %8.3f ms   pool %8.3f ms   speedup %6.2fx\n",
               ts, tp, ts / tp);
        free(u); free(u1);
    }

    /* ---------------------------------------------------------- */
    printf("\nBurgers 1D Shu-Osher, n=4096, one transit (MUSCL-upwind):\n");
    {
        const int n = 4096;
        double L = 2.0 * M_PI;
        double* u = (double*)malloc(n * 8);
        double* u1 = (double*)malloc(n * 8);
        (void)rcre_pde_burgers_shu_oshers_init(u, n, L);
        /* one transit ~ L/1 = 2*pi; dt = 0.25*dx (Burgers CFL =
         * 0.25*max|u| ~ 0.4, safely inside) */
        const double dt = 0.25 * L / n;
        const int steps = (int)(L / dt);
        double t = now_ms();
        for (int k = 0; k < steps; ++k) {
            (void)rcre_pde_burgers1d_step_f64(u, u1, n, dt, L / n);
            double* s = u; u = u1; u1 = s;
        }
        double tb = now_ms() - t;
        printf("  %d steps (1 transit) %8.3f ms   (%.4f ms/step)\n",
               steps, tb, tb / steps);
        free(u); free(u1);
    }

    /* ---------------------------------------------------------- */
    printf("\nWave 1D DDK leapfrog, n=1024, 10k steps (CFL 0.5):\n");
    {
        const int n = 1024;
        double* u = fill_f64(n, 0x15);
        double* v = (double*)calloc(n, 8);
        double* u1 = (double*)malloc(n * 8);
        double* v1 = (double*)malloc(n * 8);
        const double dx = 1.0 / n, c = 1.0, dt = 0.5 * dx / c;
        (void)rcre_pde_wave1d_start_f64(u, v, n, c, dt, dx);
        double t = now_ms();
        for (int k = 0; k < 10000; ++k)
            (void)rcre_pde_wave1d_step_f64(u, v, u1, v1, n, c, dt, dx);
        double tb = now_ms() - t;
        printf("  10k steps %8.3f ms   (%.3f ms/step)\n", tb, tb / 10000.0);
        free(u); free(v); free(u1); free(v1);
    }

    /* ---------------------------------------------------------- */
    printf("\nN-body Barnes-Hut force (F64, theta=0.5, soft=0.01):\n");
    printf("  n        serial    pool      speedup   (ms)\n");
    for (int n = 1000; n <= 100000; n *= 10) {
        double* x = fill_f64(n, 0x6A), *y = fill_f64(n, 0x6B);
        double* m = fill_f64(n, 0x6C);
        for (int i = 0; i < n; ++i) m[i] = fabs(m[i]) + 0.5;
        double* fx = (double*)malloc(n * 8), *fy = (double*)malloc(n * 8);
        double t = now_ms();
        (void)rcre_nb_bh_force_f64(x, y, m, n, 1.0, 0.5, 0.01, NULL, fx, fy);
        double ts = now_ms() - t;
        t = now_ms();
        (void)rcre_nb_bh_force_f64(x, y, m, n, 1.0, 0.5, 0.01, pool, fx, fy);
        double tp = now_ms() - t;
        printf("  %7d   %9.3f  %9.3f     %6.2fx\n", n, ts, tp, ts / tp);
        free(x); free(y); free(m); free(fx); free(fy);
    }

    printf("\nN-body Verlet (n=10k, theta=0.5, h=0.005, 1k steps):\n");
    {
        const int n = 10000;
        double* x = fill_f64(n, 0x7A), *y = fill_f64(n, 0x7B);
        double* vx = fill_f64(n, 0x7C), *vy = fill_f64(n, 0x7D);
        double* m = fill_f64(n, 0x7E);
        for (int i = 0; i < n; ++i) m[i] = fabs(m[i]) + 0.5;
        double t = now_ms();
        for (int k = 0; k < 1000; ++k)
            (void)rcre_nb_verlet_step_f64(x, y, vx, vy, m, n, 1.0, 0.5, 0.01,
                                                  0.005, pool);
        double tp = now_ms() - t;
        t = now_ms();
        for (int k = 0; k < 1000; ++k)
            (void)rcre_nb_verlet_step_f64(x, y, vx, vy, m, n, 1.0, 0.5, 0.01,
                                                  0.005, NULL);
        double ts = now_ms() - t;
        printf("  pool %10.3f ms   serial %10.3f ms   speedup %6.2fx\n",
               tp, ts, ts / tp);
        free(x); free(y); free(vx); free(vy); free(m);
    }

    /* ---------------------------------------------------------- */
    /* MCMC: NUTS/HMC wall time, banana + 8-D Gaussian             */
    /* ---------------------------------------------------------- */
    printf("MCMC (F64, sequential by design — bars are wall clock):\n");
    {
        double x0[8] = {0};
        double* out = malloc(6000 * 8 * 8);
        double acc;
        RCRE_NUTSReport rep;
        double t = now_ms();
        (void)rcre_mcmc_nuts_sample_f64(2, x0, ban_logpdf, ban_grad, NULL,
                                        42, 300, 3000, 0.0, 10, out, &acc,
                                        &rep);
        double tb = now_ms() - t;
        printf("  NUTS banana     3300 iters %10.3f ms   (accept %.2f, div %d)\n",
               tb, acc, rep.divergent);
        t = now_ms();
        (void)rcre_mcmc_nuts_sample_f64(8, x0, g8_logpdf, g8_grad, NULL,
                                        0xC0FFEE, 500, 5000, 0.0, 12, out,
                                        &acc, &rep);
        double tg = now_ms() - t;
        printf("  NUTS 8-D gauss  5500 iters %10.3f ms   (accept %.2f, div %d)\n",
               tg, acc, rep.divergent);
        t = now_ms();
        (void)rcre_mcmc_hmc_sample_f64(2, x0, ban_logpdf, ban_grad, NULL, 42,
                                       0.1, 50, 20000, out, &acc);
        double th = now_ms() - t;
        printf("  HMC banana      20000 iters (L=50) %10.3f ms   (accept %.2f)\n",
               th, acc);
        free(out);
    }

    /* ---------------------------------------------------------- */
    /* streaming: P^2 add rate, bootstrap, ESS                     */
    /* ---------------------------------------------------------- */
    printf("Streaming stats (F64, sequential):\n");
    {
        RCRE_P2Quantile* q = rcre_p2_create(0.5);
        uint64_t st = 0x5EED;
        double t = now_ms();
        for (int k = 0; k < 10000000; ++k) {
            st = st * 6364136223846793005ull + 1442695040888963407ull;
            rcre_p2_add(q, ((double)((st >> 11) & ((1ull << 53) - 1)) *
                           0x1.0p-53));
        }
        double tp2 = now_ms() - t;
        printf("  P^2 add x 10M   %10.3f ms   (%.2f M adds/s, est %.4f)\n",
               tp2, 10.0 / tp2, rcre_p2_value(q));
        rcre_p2_destroy(q);
        const int n = 10000;
        double* x = fill_f64(n, 0xAB);
        double* boots = malloc(10000 * 8);
        t = now_ms();
        (void)rcre_bootstrap_f64(x, n, 42, 0, 10000, 1, mean_stat, NULL,
                                 boots);
        double tb = now_ms() - t;
        printf("  bootstrap mean x10k (n=10k) %10.3f ms\n", tb);
        t = now_ms();
        (void)rcre_mcmc_ess_f64(x, n, n / 2);
        double te = now_ms() - t;
        printf("  ESS (n=10k)         %10.3f ms\n", te);
        free(x); free(boots);
    }

    rcre_threadpool_destroy(pool);
    printf("\n== done ==\n");
    return 0;
}
