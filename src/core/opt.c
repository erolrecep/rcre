/* RCRE H8 optimization polish (2.5.0)
 *
 * Design (docs/hpc.md 11):
 *   - L-BFGS (Nocedal-Wright two-loop, m=10 default), f64. Stopping
 *     tests are the scipy L-BFGS-B definitions (xtol measure
 *     max|x-y|/max(1+|x|), gtol on the infinity norm, ftol on
 *     max(|f|,|fp|,1)) so the iteration bars against scipy are
 *     interpretable. Optional box bounds via projected backtracking
 *     (documented heuristic — not the Byrd-Lu-Petryakov bounded
 *     algorithm; an H8.x item).
 *   - FISTA (Beck-Teboulle) for f(x) + g(x) with a user prox_g and a
 *     backtracking local-Lipschitz estimate on the smooth part.
 *   - Cyclic coordinate descent for LASSO (min 1/2||Ax-b||^2 +
 *     lambda||x||_1) with exact soft-threshold coordinate updates and
 *     the classical A^T A Gram bookkeeping; KKT-verifiable.
 *   - SPSA (Spall) zeroth-order: 2-point and 4-point (sign) gradient
 *     estimates, PCG64 perturbations (fixed seed -> bit-identical
 *     run, the library determinism policy).
 *
 * All entry points are sequential and deterministic (L-BFGS/FISTA/CD
 * by construction; SPSA under a fixed seed). Telemetry is via report
 * structs, not callbacks — no callback can perturb the numerics (the
 * H4 lesson).
 */
#include "rcre.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static double opt_infnorm(const double* g, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double a = fabs(g[i]);
        if (a > m) m = a;
    }
    return m;
}

/* scipy xtol measure: max|x-y| / max(1 + |x|) */
static double opt_xtol_measure(const double* x, const double* y, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        double a = fabs(x[i] - y[i]);
        if (a > num) num = a;
        double b = 1.0 + fabs(x[i]);
        if (b > den) den = b;
    }
    return num / den;
}

/* scipy ftol test: |fp - f| <= ftol * max(|f|, |fp|, 1) */
static int opt_ftol_met(double fp, double f, double ftol) {
    double scale = fabs(f);
    if (fabs(fp) > scale) scale = fabs(fp);
    if (scale < 1.0) scale = 1.0;
    return fabs(fp - f) <= ftol * scale;
}

static double opt_proj_box(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* ------------------------------------------------------------------ */
/* L-BFGS (two-loop recursion, backtracking line search)              */
/* ------------------------------------------------------------------ */

void rcre_lbfgs_options_default(RCRE_LbfgsOptions* o) {
    o->max_iter = 0; /* 1000*n */
    o->ftol = 1e-7;  /* scipy default */
    o->gtol = 1e-5;  /* scipy default */
    o->xtol = 1e-9;  /* scipy default */
    o->history = 10;
    o->bounded = 0;
    o->lo = NULL;
    o->hi = NULL;
}

RCRE_NODISCARD int rcre_lbfgs_f64(int n, double (*fun)(const double*, void*),
                                  void (*grad)(const double*, double*, void*),
                                  void* user, const double* x0, double* x_out,
                                  const RCRE_LbfgsOptions* opt,
                                  double* out_f, RCRE_OptReport* report) {
    if (n < 1 || !fun || !grad || !x0 || !x_out || !opt || !out_f ||
        !report)
        return -1;
    if (opt->bounded && (!opt->lo || !opt->hi)) return -1;
    const int max_iter = (opt->max_iter > 0) ? opt->max_iter : 1000 * n;
    const double ftol = (opt->ftol > 0.0) ? opt->ftol : 1e-7;
    const double gtol = (opt->gtol > 0.0) ? opt->gtol : 1e-5;
    const double xtol = opt->xtol; /* 0 = disabled (scipy L-BFGS-B parity) */
    const int m =
        (opt->history >= 1 && opt->history <= 64) ? opt->history : 10;

    double* x = (double*)malloc((size_t)n * sizeof(double));
    double* xprev = (double*)malloc((size_t)n * sizeof(double));
    double* xnext = (double*)malloc((size_t)n * sizeof(double));
    double* d = (double*)malloc((size_t)n * sizeof(double));
    double* g = (double*)malloc((size_t)n * sizeof(double));
    double* gnext = (double*)malloc((size_t)n * sizeof(double));
    double* sx = (double*)malloc((size_t)m * n * sizeof(double));
    double* sy = (double*)malloc((size_t)m * n * sizeof(double));
    double* srho = (double*)malloc((size_t)m * sizeof(double));
    double* q = (double*)malloc((size_t)n * sizeof(double));
    double* rr = (double*)malloc((size_t)n * sizeof(double));
    double* acoef = (double*)malloc((size_t)m * sizeof(double));
    double* bcoef = (double*)malloc((size_t)m * sizeof(double));
    if (!x || !xprev || !xnext || !d || !g || !gnext || !sx || !sy ||
        !srho || !q || !rr || !acoef || !bcoef) {
        free(x); free(xprev); free(xnext); free(d); free(g); free(gnext);
        free(sx); free(sy); free(srho); free(q); free(rr);
        free(acoef); free(bcoef);
        return -1;
    }
    memcpy(x, x0, (size_t)n * sizeof(double));
    if (opt->bounded)
        for (int i = 0; i < n; ++i)
            x[i] = opt_proj_box(x[i], opt->lo[i], opt->hi[i]);

    double f = fun(x, user);
    int f_evals = 1, g_evals = 1, it_done = 0;
    if (!isfinite(f)) {
        free(x); free(xprev); free(xnext); free(d); free(g); free(gnext);
        free(sx); free(sy); free(srho); free(q); free(rr);
        free(acoef); free(bcoef);
        return -1;
    }
    grad(x, g, user);

    int k = 0;   /* stored pairs */
    int head = 0; /* ring index of the oldest pair */
    int status = 1;

    /* scipy order: the gradient-norm test runs on the gradient of the
     * current iterate, before the step */
    if (opt_infnorm(g, n) < gtol) status = 0;

    for (int it = 0; status == 1 && it < max_iter; ++it) {
        /* --- two-loop recursion: d = -H g --- */
        if (k == 0) {
            for (int i = 0; i < n; ++i) d[i] = -g[i];
        } else {
            memcpy(q, g, (size_t)n * sizeof(double));
            for (int ii = k - 1; ii >= 0; --ii) {
                const int idx = (head + ii) % m;
                const double* si = &sx[(size_t)idx * n];
                const double* yi = &sy[(size_t)idx * n];
                double dot = 0.0;
                for (int i = 0; i < n; ++i) dot += si[i] * q[i];
                acoef[ii] = srho[idx] * dot;
                for (int i = 0; i < n; ++i) q[i] -= acoef[ii] * yi[i];
            }
            const int last = (head + k - 1) % m;
            const double* sl = &sx[(size_t)last * n];
            const double* yl = &sy[(size_t)last * n];
            double ss = 0.0, syd = 0.0;
            for (int i = 0; i < n; ++i) {
                ss += sl[i] * sl[i];
                syd += sl[i] * yl[i];
            }
            const double gamma = (syd > 1e-15) ? ss / syd : 1.0;
            for (int i = 0; i < n; ++i) rr[i] = gamma * q[i];
            for (int ii = 0; ii < k; ++ii) {
                const int idx = (head + ii) % m;
                const double* si = &sx[(size_t)idx * n];
                const double* yi = &sy[(size_t)idx * n];
                double dot = 0.0;
                for (int i = 0; i < n; ++i) dot += yi[i] * rr[i];
                bcoef[ii] = srho[idx] * dot;
                for (int i = 0; i < n; ++i)
                    rr[i] += (acoef[ii] - bcoef[ii]) * si[i];
            }
            for (int i = 0; i < n; ++i) d[i] = -rr[i];
            /* stale curvature can make d a non-descent direction:
             * reset the memory and fall back to -g (documented) */
            double dg = 0.0;
            for (int i = 0; i < n; ++i) dg += d[i] * g[i];
            if (dg > 0.0) {
                for (int i = 0; i < n; ++i) d[i] = -g[i];
                k = 0;
            }
        }

        /* --- Armijo backtracking (bounded: projected trials) --- */
        double dg0 = 0.0;
        for (int i = 0; i < n; ++i) dg0 += g[i] * d[i];
        double alpha = 1.0;
        double fnew = 0.0;
        int ok = 0;
        for (int ls = 0; ls < 60 && !ok; ++ls) {
            int moved = 1;
            for (int i = 0; i < n; ++i) {
                double v = x[i] + alpha * d[i];
                xnext[i] =
                    opt->bounded ? opt_proj_box(v, opt->lo[i], opt->hi[i]) : v;
                if (opt->bounded && xnext[i] == x[i]) moved = 0;
            }
            if (opt->bounded && !moved) break; /* bounded stall */
            fnew = fun(xnext, user);
            ++f_evals;
            if (isfinite(fnew) && fnew <= f + 1e-4 * alpha * dg0) {
                ok = 1;
                break;
            }
            alpha *= 0.5;
        }
        if (!ok) {
            status = 1;
            break;
        }

        /* --- accept --- */
        memcpy(xprev, x, (size_t)n * sizeof(double));
        memcpy(x, xnext, (size_t)n * sizeof(double));
        const double fp = f;
        f = fnew;
        grad(x, gnext, user);
        ++g_evals;
        ++it_done;

        /* scipy-order tests on the accepted step */
        if (xtol > 0.0 && opt_xtol_measure(xprev, x, n) < xtol) {
            memcpy(g, gnext, (size_t)n * sizeof(double));
            status = 0;
            break;
        }
        if (opt_infnorm(gnext, n) < gtol) {
            memcpy(g, gnext, (size_t)n * sizeof(double));
            status = 0;
            break;
        }
        if (opt_ftol_met(fp, f, ftol)) {
            memcpy(g, gnext, (size_t)n * sizeof(double));
            status = 0;
            break;
        }

        /* --- store (s, y) under the strong-positivity condition.
         * y = g_new - g_old MUST be captured before g is overwritten.
         * (Measured draft bug: storing after the memcpy made y = 0,
         * which silently turned every H_k into the identity.) --- */
        double sny = 0.0, snn = 0.0;
        for (int i = 0; i < n; ++i) {
            const double si = x[i] - xprev[i];
            const double yi = gnext[i] - g[i];
            sny += si * yi;
            snn += si * si;
        }
        if (sny > 1e-12 && snn > 0.0) {
            const int slot = (head + k) % m;
            for (int i = 0; i < n; ++i)
                sx[(size_t)slot * n + i] = x[i] - xprev[i];
            for (int i = 0; i < n; ++i)
                sy[(size_t)slot * n + i] = gnext[i] - g[i];
            srho[slot] = 1.0 / sny;
            ++k;
            if (k > m) {
                --k;
                head = (head + 1) % m;
            }
        }
        memcpy(g, gnext, (size_t)n * sizeof(double));
    }

    memcpy(x_out, x, (size_t)n * sizeof(double));
    *out_f = f;
    report->iter = it_done;
    report->f_evals = f_evals;
    report->g_evals = g_evals;
    report->status = status;
    free(x); free(xprev); free(xnext); free(d); free(g); free(gnext);
    free(sx); free(sy); free(srho); free(q); free(rr);
    free(acoef); free(bcoef);
    return 0;
}

/* ------------------------------------------------------------------ */
/* FISTA (Beck-Teboulle; composite f + g with user prox)              */
/* ------------------------------------------------------------------ */

RCRE_NODISCARD int rcre_fista_f64(int n, double (*fun)(const double*, void*),
                                  void (*fgrad)(const double*, double*, void*),
                                  void (*prox)(const double*, double, double*,
                                               void*),
                                  void* user, const double* x0, double* x_out,
                                  double L, int max_iter, double rtol,
                                  double* out_f, RCRE_OptReport* report) {
    if (n < 1 || !fun || !fgrad || !prox || !x0 || !x_out || !out_f ||
        !report || max_iter < 1 || rtol <= 0.0)
        return -1;
    double* x = (double*)malloc((size_t)n * sizeof(double));
    double* xold = (double*)malloc((size_t)n * sizeof(double));
    double* y = (double*)malloc((size_t)n * sizeof(double));
    double* xnew = (double*)malloc((size_t)n * sizeof(double));
    double* g = (double*)malloc((size_t)n * sizeof(double));
    if (!x || !xold || !y || !xnew || !g) {
        free(x); free(xold); free(y); free(xnew); free(g);
        return -1;
    }
    memcpy(x, x0, (size_t)n * sizeof(double));
    memcpy(xold, x0, (size_t)n * sizeof(double));
    memcpy(y, x0, (size_t)n * sizeof(double));
    double Lcur = (L > 0.0) ? L : 1.0;
    double f = fun(x, user);
    int f_evals = 1;
    if (!isfinite(f)) {
        free(x); free(xold); free(y); free(xnew); free(g);
        return -1;
    }
    double t = 1.0;
    int it_done = 0;
    int status = 1;

    for (int it = 0; it < max_iter && status == 1; ++it) {
        fgrad(y, g, user);
        /* the Lipschitz check is anchored at y (the gradient point),
         * NOT at x: f(xnew) <= f(y) + g.(xnew-y) + L/2||xnew-y||^2.
         * (Measured bug: anchoring at f(x) made backtracking diverge to
         * L ~ 1e15 on random LASSO instances — the chain stalled at x0.) */
        const double fy = fun(y, user);
        ++f_evals;
        for (int i = 0; i < n; ++i) xnew[i] = y[i] - g[i] / Lcur;
        prox(xnew, 1.0 / Lcur, xnew, user);
        double fn = 0.0, lin = 0.0, qq = 0.0;
        for (int b = 0; b < 64; ++b) {
            fn = fun(xnew, user);
            ++f_evals;
            lin = fy;
            qq = 0.0;
            for (int i = 0; i < n; ++i) {
                const double dd = xnew[i] - y[i];
                lin += g[i] * dd;
                qq += dd * dd;
            }
            if (isfinite(fn) && fn <= lin + 0.5 * Lcur * qq + 1e-12) break;
            Lcur *= 2.0;
            for (int i = 0; i < n; ++i) xnew[i] = y[i] - g[i] / Lcur;
            prox(xnew, 1.0 / Lcur, xnew, user);
        }
        if (!isfinite(fn)) {
            status = 1;
            break;
        }
        /* stopping: relative x-change (same measure as L-BFGS xtol) */
        if (opt_xtol_measure(x, xnew, n) < rtol) {
            memcpy(x, xnew, (size_t)n * sizeof(double));
            f = fn;
            ++it_done;
            status = 0;
            break;
        }
        memcpy(xold, x, (size_t)n * sizeof(double));
        memcpy(x, xnew, (size_t)n * sizeof(double));
        f = fn;
        ++it_done;
        /* FISTA momentum: y = x_{k+1} + ((t-1)/t') (x_{k+1} - x_k) */
        const double tnext = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t));
        const double beta = (t - 1.0) / tnext;
        for (int i = 0; i < n; ++i)
            y[i] = x[i] + beta * (x[i] - xold[i]);
        t = tnext;
    }

    memcpy(x_out, x, (size_t)n * sizeof(double));
    *out_f = f;
    report->iter = it_done;
    report->f_evals = f_evals;
    report->g_evals = 0;
    report->status = status;
    free(x); free(xold); free(y); free(xnew); free(g);
    return 0;
}

/* ------------------------------------------------------------------ */
/* cyclic coordinate descent — LASSO                                  */
/* ------------------------------------------------------------------ */

RCRE_NODISCARD int rcre_cd_lasso_f64(const double* A, int n, int d,
                                     const double* b, double lambda,
                                     const double* x0, double* x_out,
                                     int max_iter, double tol,
                                     double* out_obj, RCRE_OptReport* report) {
    if (n < 1 || d < 1 || !A || !b || !x0 || !x_out || !out_obj || !report ||
        lambda < 0.0 || max_iter < 1 || tol <= 0.0)
        return -1;
    double* x = (double*)malloc((size_t)d * sizeof(double));
    double* r = (double*)malloc((size_t)n * sizeof(double)); /* b - A x */
    double* atb = (double*)malloc((size_t)d * sizeof(double)); /* A^T r */
    double* gram = (double*)malloc((size_t)d * d * sizeof(double));
    if (!x || !r || !atb || !gram) {
        free(x); free(r); free(atb); free(gram);
        return -1;
    }
    memcpy(x, x0, (size_t)d * sizeof(double));
    /* residual + A^T r */
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < d; ++j) s += A[(size_t)i * d + j] * x[j];
        r[i] = b[i] - s;
    }
    for (int j = 0; j < d; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += A[(size_t)i * d + j] * r[i];
        atb[j] = s;
    }
    /* Gram G = A^T A (the classical CD-LASSO bookkeeping; O(d^2 n)
     * upfront — documented cost) */
    for (int j = 0; j < d; ++j) {
        gram[(size_t)j * d + j] = 0.0;
        for (int i = 0; i < n; ++i)
            gram[(size_t)j * d + j] +=
                A[(size_t)i * d + j] * A[(size_t)i * d + j];
        for (int kk = j + 1; kk < d; ++kk) {
            double c = 0.0;
            for (int i = 0; i < n; ++i)
                c += A[(size_t)i * d + j] * A[(size_t)i * d + kk];
            gram[(size_t)j * d + kk] = c;
            gram[(size_t)kk * d + j] = c;
        }
    }

    int it_done = 0;
    int status = 1;
    for (int it = 0; it < max_iter && status == 1; ++it) {
        double maxchange = 0.0;
        for (int j = 0; j < d; ++j) {
            const double wj = gram[(size_t)j * d + j];
            if (wj <= 0.0) continue; /* zero column: coordinate is free */
            const double rho = atb[j] + wj * x[j];
            double xn = 0.0;
            if (rho > lambda) xn = (rho - lambda) / wj;
            else if (rho < -lambda) xn = (rho + lambda) / wj;
            const double delta = xn - x[j];
            if (delta != 0.0) {
                x[j] = xn;
                /* keep atb = A^T r current: atb[k] -= G[k,j] delta */
                for (int k = 0; k < d; ++k)
                    atb[k] -= gram[(size_t)k * d + j] * delta;
                for (int i = 0; i < n; ++i)
                    r[i] -= A[(size_t)i * d + j] * delta;
                const double rel = fabs(delta) / (1.0 + fabs(xn));
                if (rel > maxchange) maxchange = rel;
            }
        }
        ++it_done;
        if (maxchange < tol) status = 0;
    }
    /* objective 1/2||Ax-b||^2 + lambda||x||_1 (from the tracked r) */
    double obj = 0.0;
    for (int i = 0; i < n; ++i) obj += r[i] * r[i];
    obj *= 0.5;
    for (int j = 0; j < d; ++j) obj += lambda * fabs(x[j]);
    memcpy(x_out, x, (size_t)d * sizeof(double));
    *out_obj = obj;
    report->iter = it_done;
    report->f_evals = 0;
    report->g_evals = 0;
    report->status = status;
    free(x); free(r); free(atb); free(gram);
    return 0;
}

/* ------------------------------------------------------------------ */
/* SPSA (Spall) — zeroth-order                                         */
/* ------------------------------------------------------------------ */

RCRE_NODISCARD int rcre_spsa_f64(int n, double (*fun)(const double*, void*),
                                 void* user, const double* x0, double* x_out,
                                 uint64_t seed, double c0, double a0,
                                 double gamma, int max_iter, int four_point,
                                 double* out_f, RCRE_OptReport* report) {
    if (n < 1 || !fun || !x0 || !x_out || !out_f || !report || max_iter < 1 ||
        c0 <= 0.0 || a0 <= 0.0 || gamma <= 0.0 || gamma > 1.0 ||
        four_point < 0 || four_point > 1)
        return -1;
    double* x = (double*)malloc((size_t)n * sizeof(double));
    double* xp = (double*)malloc((size_t)n * sizeof(double));
    double* xm = (double*)malloc((size_t)n * sizeof(double));
    double* del = (double*)malloc((size_t)n * sizeof(double));
    double* gk = (double*)malloc((size_t)n * sizeof(double));
    if (!x || !xp || !xm || !del || !gk) {
        free(x); free(xp); free(xm); free(del); free(gk);
        return -1;
    }
    memcpy(x, x0, (size_t)n * sizeof(double));
    RCRE_RNG rng;
    rcre_rng_pcg_init(&rng, seed);
    double f = fun(x, user);
    int f_evals = 1;
    if (!isfinite(f)) {
        free(x); free(xp); free(xm); free(del); free(gk);
        return -1;
    }
    int it_done = 0;
    int status = 1;
    for (int it = 0; it < max_iter && status == 1; ++it) {
        const double ak = a0 / pow((double)(it + 1), gamma);
        /* fresh +/-1 perturbation (PCG64, fixed order -> the run is a
         * pure function of the seed) */
        for (int i = 0; i < n; ++i)
            del[i] = (rcre_rng_pcg_next_f64(&rng) < 0.5) ? -1.0 : 1.0;
        for (int i = 0; i < n; ++i) {
            xp[i] = x[i] + c0 * del[i];
            xm[i] = x[i] - c0 * del[i];
        }
        const double fp = fun(xp, user);
        ++f_evals;
        if (!isfinite(fp)) break;
        if (four_point) {
            const double fm = fun(xm, user);
            ++f_evals;
            if (!isfinite(fm)) break;
            for (int i = 0; i < n; ++i)
                gk[i] = (fp - fm) / (2.0 * c0 * del[i]);
        } else {
            /* 2-point: reuse the current value as the minus side */
            for (int i = 0; i < n; ++i)
                gk[i] = (fp - f) / (c0 * del[i]);
        }
        for (int i = 0; i < n; ++i) x[i] -= ak * gk[i];
        const double fn = fun(x, user);
        ++f_evals;
        if (!isfinite(fn)) {
            for (int i = 0; i < n; ++i) x[i] += ak * gk[i]; /* revert */
            break;
        }
        f = fn;
        ++it_done;
    }
    memcpy(x_out, x, (size_t)n * sizeof(double));
    *out_f = f;
    report->iter = it_done;
    report->f_evals = f_evals;
    report->g_evals = 0;
    report->status = status;
    free(x); free(xp); free(xm); free(del); free(gk);
    return 0;
}
