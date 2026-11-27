/* RCRE H8 optimization polish benchmark (2.5.0).
 *
 * Measured bars (docs/hpc.md 11): wall time per problem + rate.
 * -O3, M5 Max. The iteration/accuracy bars live in tests/test_h8.c
 * (vs the scipy fixture); this benchmark is the speed side.
 */
#include "rcre.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static double lcg01(uint64_t* st) {
    *st = *st * 6364136223846793005ull + 1442695040888963407ull;
    return (double)((*st >> 11) & ((1ull << 53) - 1)) / 9007199254740992.0;
}

/* rosenbrock */
static double rosen_f(const double* x, void* u) {
    int n = *(int*)u;
    double s = 0.0;
    for (int i = 0; i + 1 < n; ++i) {
        double a = x[i + 1] - x[i] * x[i];
        double b = 1.0 - x[i];
        s += a * a + 100.0 * b * b;
    }
    return s;
}
static void rosen_g(const double* x, double* g, void* u) {
    int n = *(int*)u;
    for (int i = 0; i < n; ++i) g[i] = 0.0;
    for (int i = 0; i + 1 < n; ++i) {
        g[i] += -2.0 * (x[i + 1] - x[i] * x[i]) - 200.0 * (1.0 - x[i]);
        g[i + 1] += 2.0 * (x[i + 1] - x[i] * x[i]);
    }
}

/* SPD quadratic */
typedef struct {
    int d;
    double* Q;
    double* p;
} QuadT;
static double quad_f(const double* x, void* u) {
    QuadT* q = (QuadT*)u;
    int d = q->d;
    double s = 0.0, sp = 0.0;
    for (int i = 0; i < d; ++i) {
        double r = 0.0;
        for (int j = 0; j < d; ++j) r += q->Q[(size_t)i * d + j] * x[j];
        s += r * x[i];
        sp += q->p[i] * x[i];
    }
    return 0.5 * s - sp;
}
static void quad_g(const double* x, double* g, void* u) {
    QuadT* q = (QuadT*)u;
    int d = q->d;
    for (int i = 0; i < d; ++i) {
        double r = -q->p[i];
        for (int j = 0; j < d; ++j) r += q->Q[(size_t)i * d + j] * x[j];
        g[i] = r;
    }
}

/* LASSO pieces (n=2000, d=500) */
#define BN 2000
#define BD 500
static double gA[BN * BD];
static double gb[BN];
static double las_f(const double* x, void* u) {
    (void)u;
    double s = 0.0;
    for (int i = 0; i < BN; ++i) {
        double r = -gb[i];
        for (int j = 0; j < BD; ++j) r += gA[(size_t)i * BD + j] * x[j];
        s += r * r;
    }
    return 0.5 * s;
}
static void las_g(const double* x, double* g, void* u) {
    (void)u;
    for (int j = 0; j < BD; ++j) g[j] = 0.0;
    for (int i = 0; i < BN; ++i) {
        double r = -gb[i];
        for (int j = 0; j < BD; ++j) r += gA[(size_t)i * BD + j] * x[j];
        for (int j = 0; j < BD; ++j) g[j] += gA[(size_t)i * BD + j] * r;
    }
}
static void las_prox(const double* z, double step, double* out, void* u) {
    double lam = *(double*)u;
    for (int j = 0; j < BD; ++j) {
        double v = z[j], th = step * lam;
        if (v > th) out[j] = v - th;
        else if (v < -th) out[j] = v + th;
        else out[j] = 0.0;
    }
}

int main(void) {
    rcre_hal_init();
    printf("=== RCRE H8 optimization polish (2.5.0) ===\n");

    /* L-BFGS: rosenbrock-100 */
    {
        int n = 100;
        double* x0 = (double*)malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; ++i) x0[i] = (i % 2 == 0) ? -1.2 : 1.0;
        RCRE_LbfgsOptions o;
        rcre_lbfgs_options_default(&o);
        o.ftol = 1e-10;
        o.gtol = 1e-10;
        o.xtol = 0.0;
        double* x = (double*)malloc((size_t)n * sizeof(double));
        double f;
        RCRE_OptReport rep;
        double t0 = now_ms();
        int rc = rcre_lbfgs_f64(n, rosen_f, rosen_g, &n, x0, x, &o, &f, &rep);
        double ms = now_ms() - t0;
        printf("L-BFGS rosenbrock-100: %s %.4f ms  iters=%d fE=%d f=%.3e\n",
               rc == 0 ? "OK" : "FAIL", ms, rep.iter, rep.f_evals, f);
        free(x0);
        free(x);
    }

    /* L-BFGS: SPD quad-500 */
    {
        int d = 500;
        double* Q = (double*)malloc((size_t)d * d * sizeof(double));
        double* p = (double*)malloc((size_t)d * sizeof(double));
        double* x0 = (double*)malloc((size_t)d * sizeof(double));
        uint64_t st = 0x7B;
        double tmp[250000];
        for (int i = 0; i < d * d; ++i) tmp[i] = lcg01(&st) * 2.0 - 1.0;
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) {
                double s = 0.0;
                for (int k = 0; k < d; ++k)
                    s += tmp[(size_t)k * d + i] * tmp[(size_t)k * d + j];
                Q[(size_t)i * d + j] = s;
            }
        for (int i = 0; i < d; ++i) Q[(size_t)i * d + i] += d;
        for (int i = 0; i < d; ++i) p[i] = x0[i] = lcg01(&st) * 2.0 - 1.0;
        QuadT q = {d, Q, p};
        RCRE_LbfgsOptions o;
        rcre_lbfgs_options_default(&o);
        o.ftol = 1e-10;
        o.gtol = 1e-10;
        o.xtol = 0.0;
        double* x = (double*)malloc((size_t)d * sizeof(double));
        double f;
        RCRE_OptReport rep;
        double t0 = now_ms();
        int rc = rcre_lbfgs_f64(d, quad_f, quad_g, &q, x0, x, &o, &f, &rep);
        double ms = now_ms() - t0;
        printf("L-BFGS SPD-quad-500: %s %.4f ms  iters=%d fE=%d f=%.3e\n",
               rc == 0 ? "OK" : "FAIL", ms, rep.iter, rep.f_evals, f);
        free(Q);
        free(p);
        free(x0);
        free(x);
    }

    /* LASSO data */
    {
        uint64_t st = 0x4242;
        for (int i = 0; i < BN * BD; ++i) gA[i] = lcg01(&st) * 2.0 - 1.0;
        for (int i = 0; i < BN; ++i) gb[i] = lcg01(&st) * 2.0 - 1.0;
    }

    /* CD-LASSO n=2000 d=500 lambda=0.5 */
    {
        double* x0 = (double*)calloc(BD, sizeof(double));
        double* x = (double*)malloc((size_t)BD * sizeof(double));
        double obj;
        RCRE_OptReport rep;
        double t0 = now_ms();
        int rc = rcre_cd_lasso_f64(gA, BN, BD, gb, 0.5, x0, x, 100000, 1e-10,
                                   &obj, &rep);
        double ms = now_ms() - t0;
        printf("CD-LASSO 2000x500 lam=0.5: %s %.1f ms  passes=%d obj=%.3e\n",
               rc == 0 ? "OK" : "FAIL", ms, rep.iter, obj);
        free(x0);
        free(x);
    }

    /* FISTA same instance */
    {
        double* x0 = (double*)calloc(BD, sizeof(double));
        double* x = (double*)malloc((size_t)BD * sizeof(double));
        double lam = 0.5;
        double f;
        RCRE_OptReport rep;
        double t0 = now_ms();
        int rc = rcre_fista_f64(BD, las_f, las_g, las_prox, &lam, x0, x, 0.0,
                                3000000, 1e-10, &f, &rep);
        double ms = now_ms() - t0;
        double l1 = 0.0;
        for (int j = 0; j < BD; ++j) l1 += fabs(x[j]);
        printf("FISTA-L1 2000x500 lam=0.5 (backtracked L): %s %.1f ms  iters=%d "
               "fE=%d obj=%.3e (f=%.3e + lam*L1=%.3e)\n",
               rc == 0 ? "OK" : "FAIL", ms, rep.iter, rep.f_evals, f + lam * l1,
               f, lam * l1);
        free(x0);
        free(x);
    }

    /* SPSA: SPD quad-100 (4-point), 2000 steps */
    {
        int d = 100;
        double* Q = (double*)malloc((size_t)d * d * sizeof(double));
        double* p = (double*)malloc((size_t)d * sizeof(double));
        double* x0 = (double*)malloc((size_t)d * sizeof(double));
        uint64_t st = 0x51A5;
        double tmp[10000];
        for (int i = 0; i < d * d; ++i) tmp[i] = lcg01(&st) * 2.0 - 1.0;
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) {
                double s = 0.0;
                for (int k = 0; k < d; ++k)
                    s += tmp[(size_t)k * d + i] * tmp[(size_t)k * d + j];
                Q[(size_t)i * d + j] = s;
            }
        for (int i = 0; i < d; ++i) Q[(size_t)i * d + i] += d;
        for (int i = 0; i < d; ++i) p[i] = x0[i] = lcg01(&st) * 2.0 - 1.0;
        QuadT q = {d, Q, p};
        double* x = (double*)malloc((size_t)d * sizeof(double));
        /* a0 = 0.1 / ||grad f(x0)||_2: O(0.1) initial step regardless of
         * problem scale (the sign estimator has O(||g||^2) per-component
         * variance, so a fixed a0 diverges as the problem grows —
         * measured on this problem for a0 = 5e-3) */
        double g0[100];
        quad_g(x0, g0, &q);
        double gn = 0.0;
        for (int i = 0; i < d; ++i) gn += g0[i] * g0[i];
        const double a0 = 0.1 / sqrt(gn);
        const double f_start = quad_f(x0, &q);
        double f;
        RCRE_OptReport rep;
        double t0 = now_ms();
        int rc = rcre_spsa_f64(d, quad_f, &q, x0, x, 7, 0.1, a0, 0.6022, 10000, 1,
                               &f, &rep);
        double ms = now_ms() - t0;
        printf("SPSA-4 quad-100 (10000 steps, a0=%.3e): %s %.1f ms  fE=%d "
               "f=%.3e (drop %.1f%% of |f_start|) (%.2f M evals/s)\n",
               a0, rc == 0 ? "OK" : "FAIL", ms, rep.f_evals, f,
               100.0 * (f_start - f) / fabs(f_start),
               (double)rep.f_evals / 1e6 / (ms / 1000.0));
        free(Q);
        free(p);
        free(x0);
        free(x);
    }
    return 0;
}
