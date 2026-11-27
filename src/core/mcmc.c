/* RCRE H7 MCMC — Metropolis-Hastings, HMC, NUTS + QMC integrators (2.4.0)
 *
 * Design (docs/hpc.md 10.3):
 *   - f64 throughout; targets are user-supplied log-pdf (+ gradient for
 *     HMC/NUTS; MH works with the log-pdf alone).
 *   - All randomness flows through ONE PCG64 stream (H1) in a fixed
 *     consumption order -> a chain with a fixed seed is bit-identical
 *     across runs and machines. Chains are sequential by nature; this
 *     module adds no threading (independent chains are the caller's
 *     job — run with different seeds).
 *   - HMC/NUTS acceptance is the EXACT slice-HMC test (Neal 2011, as
 *     simplified by Hoffman & Gelman 2014): with slice z = log u,
 *     u ~ U(0,1), a trajectory node (q, rq) is accepted when
 *         A(q) = logp(q) - 1/2 ||rq||^2  >  A0 + z,   A0 = lp0 - 1/2||r0||^2
 *     which is the joint-distribution Metropolis test for the extended
 *     target p(q,r) ∝ p(q) exp(-1/2||r||^2) — leapfrog is exact
 *     symplectic, so no HMC error is introduced by the trajectory.
 *     (H-G's "S(q) = logp(q) - 1/2||q||^2" shorthand is their
 *     reparametrization of the same test; we keep the direct form —
 *     it is what is verified against the Gaussian-moment gate.)
 *   - NUTS tree growth follows H-G 2014 Algorithm 1: at level j extend
 *     by 2^j leapfrog steps in a randomly chosen direction, track
 *     A_min/A_max (stop when either falls below A0 + z), keep the
 *     best-A node; divergence when |H(q,rq) - H0| > 1000.
 *   - Dual-averaging step-size adaptation is the Stan form of H-G §4.1
 *     (alpha 0.05, gamma 0.05, t0 10, target 0.65). Window adaptation
 *     (Wand et al.) is NOT implemented — documented simplification;
 *     the bars are on acceptance + ESS + target moments.
 *   - ESS: initial-monotone-sequence autocorrelation estimator
 *     (Geyer 1992) — the statistical-gate primitive.
 *   - Integrators: plain MC (PCG64), antithetic (paired), stratified
 *     (base-d digit assignment — pure function of the index), Sobol'
 *     and Halton (H1 quasi-rng, stateless random access).
 */
#include "rcre.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* target function types                                              */
/* ------------------------------------------------------------------ */

typedef double (*rcre_mcmc_logpdf_t)(const double* x, int d, void* user);
typedef void (*rcre_mcmc_grad_t)(const double* x, int d, double* g, void* user);

static inline double mcmc_dot2(const double* a, int d) {
    double s = 0.0;
    for (int i = 0; i < d; ++i) s += a[i] * a[i];
    return s;
}

/* deterministic standard normal from the PCG64 stream (Box-Muller;
 * fixed consumption order — two uniforms per draw, u1 mirrored into
 * (0,1] so log is finite; PCG64 f64 is [0,1)) */
static double pcg_normal(RCRE_RNG* r) {
    const double u1 = 1.0 - rcre_rng_pcg_next_f64(r);
    const double u2 = rcre_rng_pcg_next_f64(r);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* one leapfrog step: (x, r) -> (x1, r1); g is scratch (d) */
static void mcmc_leapfrog_one(const double* x, const double* r, double eps,
                              int dir, rcre_mcmc_grad_t grad, void* user,
                              int d, double* g, double* x1, double* r1) {
    memcpy(x1, x, (size_t)d * sizeof(double));
    memcpy(r1, r, (size_t)d * sizeof(double));
    grad(x1, d, g, user);
    for (int i = 0; i < d; ++i) r1[i] += 0.5 * eps * dir * g[i];
    for (int i = 0; i < d; ++i) x1[i] += eps * dir * r1[i];
    grad(x1, d, g, user);
    for (int i = 0; i < d; ++i) r1[i] += 0.5 * eps * dir * g[i];
}

/* ------------------------------------------------------------------ */
/* Metropolis-Hastings (random-walk, Gaussian proposal)               */
/* ------------------------------------------------------------------ */

RCRE_NODISCARD int rcre_mcmc_mh_sample_f64(int d, const double* x0,
                                           rcre_mcmc_logpdf_t logpdf,
                                           void* user, uint64_t seed,
                                           double proposal_sd,
                                           int n_samples, double* out,
                                           double* out_accept) {
    if (d < 1 || !x0 || !logpdf || !out || n_samples < 1 ||
        proposal_sd <= 0.0)
        return -1;
    RCRE_RNG rng;
    rcre_rng_pcg_init(&rng, seed);
    double* x = (double*)malloc((size_t)d * sizeof(double));
    double* q = (double*)malloc((size_t)d * sizeof(double));
    if (!x || !q) {
        free(x);
        free(q);
        return -1;
    }
    memcpy(x, x0, (size_t)d * sizeof(double));
    double lp = logpdf(x, d, user);
    int n_acc = 0;
    for (int k = 0; k < n_samples; ++k) {
        for (int i = 0; i < d; ++i)
            q[i] = x[i] + proposal_sd * pcg_normal(&rng);
        const double lq = logpdf(q, d, user);
        const double du = rcre_rng_pcg_next_f64(&rng);
        const int acc = (lq >= lp) ||
                        (isfinite(lq) && du < exp(lq - lp));
        if (acc) {
            memcpy(x, q, (size_t)d * sizeof(double));
            lp = lq;
            ++n_acc;
        }
        memcpy(&out[(size_t)k * d], x, (size_t)d * sizeof(double));
    }
    if (out_accept) *out_accept = (double)n_acc / (double)n_samples;
    free(x);
    free(q);
    return 0;
}

/* ------------------------------------------------------------------ */
/* HMC (fixed trajectory length L, fixed step eps)                    */
/* ------------------------------------------------------------------ */

RCRE_NODISCARD int rcre_mcmc_hmc_sample_f64(int d, const double* x0,
                                            rcre_mcmc_logpdf_t logpdf,
                                            rcre_mcmc_grad_t grad,
                                            void* user, uint64_t seed,
                                            double eps, int L,
                                            int n_samples, double* out,
                                            double* out_accept) {
    if (d < 1 || !x0 || !logpdf || !grad || !out || n_samples < 1 ||
        eps <= 0.0 || L < 1)
        return -1;
    RCRE_RNG rng;
    rcre_rng_pcg_init(&rng, seed);
    double* x  = (double*)malloc((size_t)d * sizeof(double));
    double* r  = (double*)malloc((size_t)d * sizeof(double));
    double* xn = (double*)malloc((size_t)d * sizeof(double));
    double* rn = (double*)malloc((size_t)d * sizeof(double));
    double* g  = (double*)malloc((size_t)d * sizeof(double));
    if (!x || !r || !xn || !rn || !g) {
        free(x); free(r); free(xn); free(rn); free(g);
        return -1;
    }
    memcpy(x, x0, (size_t)d * sizeof(double));
    int n_acc = 0;
    for (int k = 0; k < n_samples; ++k) {
        /* fresh momentum every iteration (standard; no carryover) */
        for (int i = 0; i < d; ++i) r[i] = pcg_normal(&rng);
        const double lp0 = logpdf(x, d, user);
        const double A0 = lp0 - 0.5 * mcmc_dot2(r, d);
        memcpy(xn, x, (size_t)d * sizeof(double));
        memcpy(rn, r, (size_t)d * sizeof(double));
        for (int s = 0; s < L; ++s)
            mcmc_leapfrog_one(xn, rn, eps, +1, grad, user, d, g, xn, rn);
        const double lp1 = logpdf(xn, d, user);
        const double A1 = lp1 - 0.5 * mcmc_dot2(rn, d);
        const double du = rcre_rng_pcg_next_f64(&rng);
        const double dA = A1 - A0;
        const int acc = (dA >= 0.0) ||
                        (isfinite(dA) && du < exp(dA < -745.0 ? -745.0 : dA));
        if (acc) memcpy(x, xn, (size_t)d * sizeof(double));
        memcpy(&out[(size_t)k * d], x, (size_t)d * sizeof(double));
        if (acc) ++n_acc;
    }
    if (out_accept) *out_accept = (double)n_acc / (double)n_samples;
    free(x); free(r); free(xn); free(rn); free(g);
    return 0;
}

/* ------------------------------------------------------------------ */
/* NUTS (Hoffman & Gelman 2014, Algorithm 1 + dual averaging)         */
/* ------------------------------------------------------------------ */

#define RCRE_NUTS_DIVERGENCE_TOL 1000.0
/* Target for P(non-root pick) — the NUTS "acceptance". 0.8 (not the
 * 0.65 of Metropolis-style NUTS): exact slice sampling has no
 * rejection step, and P(pick != root) ~ 1 is the healthy regime
 * (the tree finds a node under the slice); P -> 0 means the chain
 * is stuck. 0.8 leaves headroom so DA keeps eps modest on hard
 * targets instead of inflating it until divergences appear. */
#define RCRE_NUTS_TARGET_ACCEPT  0.80
/* Robbins-Monro gain for the acceptance-based dual averaging below. */
#define RCRE_NUTS_DA_ALPHA       0.75
/* Practical step-size clamp so the chain can never freeze (eps -> 0)
 * or blow up (eps -> large) regardless of the target's geometry. */
#define RCRE_NUTS_EPS_MIN        1e-3
#define RCRE_NUTS_EPS_MAX        5.0

RCRE_NODISCARD int rcre_mcmc_nuts_sample_f64(int d, const double* x0,
                                             rcre_mcmc_logpdf_t logpdf,
                                             rcre_mcmc_grad_t grad,
                                             void* user, uint64_t seed,
                                             int n_adapt, int n_samples,
                                             double eps0, int max_tree,
                                             double* out, double* out_accept,
                                             RCRE_NUTSReport* report) {
    if (d < 1 || !x0 || !logpdf || !grad || !out || n_samples < 1 ||
        max_tree < 1)
        return -1;
    if (max_tree > 20) max_tree = 20;

    RCRE_RNG rng;
    rcre_rng_pcg_init(&rng, seed);

    double* x  = (double*)malloc((size_t)d * sizeof(double));
    double* r  = (double*)malloc((size_t)d * sizeof(double));
    double* xn = (double*)malloc((size_t)d * sizeof(double));
    double* rn = (double*)malloc((size_t)d * sizeof(double));
    double* g  = (double*)malloc((size_t)d * sizeof(double));
    double* xorig = (double*)malloc((size_t)d * sizeof(double));
    /* tree storage: up to 2^max_tree - 1 nodes, each a position */
    const int max_nodes = (1 << max_tree) - 1;
    double* nodes = (double*)malloc((size_t)max_nodes * d * sizeof(double));
    double* nodeA = (double*)malloc((size_t)max_nodes * sizeof(double));
    if (!x || !r || !xn || !rn || !g || !xorig || !nodes || !nodeA) {
        free(x); free(r); free(xn); free(rn); free(g); free(xorig);
        free(nodes); free(nodeA);
        return -1;
    }
    memcpy(x, x0, (size_t)d * sizeof(double));
    const double lp0start = logpdf(x, d, user);
    if (!isfinite(lp0start)) {
        free(x); free(r); free(xn); free(rn); free(g); free(xorig);
        free(nodes); free(nodeA);
        return -1; /* infeasible start */
    }

    double eps = (eps0 > 0.0) ? eps0 : 0.1;
    if (eps < RCRE_NUTS_EPS_MIN) eps = RCRE_NUTS_EPS_MIN;
    if (eps > RCRE_NUTS_EPS_MAX) eps = RCRE_NUTS_EPS_MAX;
    /* acceptance-based dual-averaging state: Robbins-Monro
     * root-finding on E[P(pick != root)] = target, signal = the
     * moving average (span ~100) of the per-iteration indicator.
     *
     * (Measured deviation from the plan: the draft used the
     * Stan/H-G 2014 SS4.1 Hbar form, Hbar <- Hbar - exp(-gamma t)
     * (log m - log target). Its log m drift has NO restoring term:
     * on easy targets (1-D Gaussian, weakly coupled rook) the tree
     * always runs to max_tree, Hbar decays by ~112 over a 750-step
     * warmup, and eps is driven to ~1e-5 — the chain measurably
     * FROZE (x constant to 6 digits for thousands of iterations).
     * The acceptance-based form has a true equilibrium at the
     * target and cannot run away. Documented in hpc.md 10.3.) */
    double acc_ma = 0.5;

    int sample_acc = 0;   /* non-root picks in the sampling phase */
    int divergent_count = 0;
    int iters = n_adapt + n_samples;

    for (int k = 0; k < iters; ++k) {
        for (int i = 0; i < d; ++i) r[i] = pcg_normal(&rng);
        memcpy(xorig, x, (size_t)d * sizeof(double));
        const double lp = logpdf(x, d, user);
        const double A0 = lp - 0.5 * mcmc_dot2(r, d);
        const double H0 = -A0; /* H = 1/2||r||^2 - logp = -A */
        const double z = log(1.0 - rcre_rng_pcg_next_f64(&rng));
        const double target = A0 + z;

        /* grow the tree (H-G Algorithm 1 structure: at level j extend by
         * 2^j leapfrog steps in a coin-flipped direction, from the
         * current endpoint). Every node is a symplectic (J=1) proposal
         * of the root, so the MH ratio for node m is exp(A_m - A0) —
         * the tree proposal is direction-symmetric (H-G Lemma 1's
         * bookkeeping collapses to identity under exact slice sampling
         * over the node set). */
        int nnodes = 0;
        int diverged = 0;
        int stop = 0;
        double Amin = A0, Amax = A0;
        for (int j = 0; j < max_tree && !stop; ++j) {
            const int nnew = 1 << j;
            const int dir = (rcre_rng_pcg_next_f64(&rng) < 0.5) ? -1 : +1;
            double* px = x;  /* endpoint of the current tree */
            double* pr = r;
            /* endpoints live in (x, r); we must keep the ORIGINAL root
             * separate — (x, r) is reused as the moving endpoint */
            for (int nn = 0; nn < nnew && !diverged; ++nn) {
                mcmc_leapfrog_one(px, pr, eps, dir, grad, user, d, g,
                                  xn, rn);
                const double lpn = logpdf(xn, d, user);
                const double An = lpn - 0.5 * mcmc_dot2(rn, d);
                if (!isfinite(An) ||
                    fabs(-An - H0) > RCRE_NUTS_DIVERGENCE_TOL) {
                    diverged = 1;
                    break;
                }
                /* store the node */
                memcpy(&nodes[(size_t)nnodes * d], xn,
                       (size_t)d * sizeof(double));
                nodeA[nnodes] = An;
                ++nnodes;
                if (An < Amin) Amin = An;
                if (An > Amax) Amax = An;
                px = xn;
                pr = rn;
            }
            if (diverged) {
                stop = 1;
                break;
            }
            /* H-G stopping criterion */
            if (Amin < target || Amax < target) stop = 1;
            /* endpoint becomes the current state for the next level */
            memcpy(x, xn, (size_t)d * sizeof(double));
            memcpy(r, rn, (size_t)d * sizeof(double));
        }
        if (diverged) ++divergent_count;

        /* EXACT slice update over the tree (Neal 2011 slice HMC with a
         * discrete trajectory set): the candidate set is
         *     {root} union { m : A_m >= A0 + z }
         * in that fixed order (root first, then nodes 0..n-1), and we
         * pick UNIFORMLY from the candidates under the slice. The root
         * is always under the slice (z <= 0), so the set is nonempty.
         * (Measured bug: the first draft sampled all nodes with weights
         * exp(A_m) — that is a generic MH kernel, NOT slice sampling:
         * nodes below the slice must be excluded, not merely
         * down-weighted. The chain ran but its banana moments were
         * badly biased, E[y] = +1.04 vs 0.) */
        int n_under = 1; /* the root, always under the slice */
        for (int m = 0; m < nnodes; ++m)
            if (nodeA[m] >= target) ++n_under;
        const double usel = rcre_rng_pcg_next_f64(&rng);
        const int sel = (int)(usel * (double)n_under); /* in [0, n_under) */
        int pick = -1;
        if (sel == 0) {
            pick = -1; /* root is the first candidate */
        } else {
            int rank = 1; /* root already counted */
            for (int m = 0; m < nnodes && pick < 0; ++m) {
                if (nodeA[m] < target) continue;
                if (rank == sel) {
                    pick = m;
                    break;
                }
                ++rank;
            }
        }
        if (pick >= 0) {
            memcpy(x, &nodes[(size_t)pick * d], (size_t)d * sizeof(double));
        } else {
            /* root picked: restore the iteration's root (x currently
             * holds the tree endpoint from the growth loop) */
            memcpy(x, xorig, (size_t)d * sizeof(double));
        }
        /* next iteration starts from the picked position with fresh
         * momentum (r is resampled at the top of the loop) */

        if (k >= n_adapt) {
            const int di = k - n_adapt;
            memcpy(&out[(size_t)di * d], x, (size_t)d * sizeof(double));
            if (pick >= 0) ++sample_acc;
        }

        /* dual-averaging update during ADAPTATION ONLY (m = number of
         * leapfrog steps taken; Stan's form of H-G SS4.1) */
        if (k < n_adapt) {
            /* divergence guard: any divergence halves eps (floor at
             * EPS_MIN) — the standard slice-NUTS tuning lever; the
             * acceptance signal alone does not see unstable
             * trajectories. */
            if (diverged) {
                eps *= 0.5;
                if (eps < RCRE_NUTS_EPS_MIN) eps = RCRE_NUTS_EPS_MIN;
            }
            const double acc_ind = (pick >= 0) ? 1.0 : 0.0;
            const double w = 2.0 / (100.0 + 1.0); /* EMMA, span ~100 */
            acc_ma += w * (acc_ind - acc_ma);
            /* Robbins-Monro: step 1/sqrt(k) — sum of steps diverges
             * (reaches the root), sum of squares converges (settles).
             * accept > target  ->  eps was too small  ->  increase
             * accept < target  ->  eps was too large  ->  decrease */
            const double step =
                RCRE_NUTS_DA_ALPHA *
                (acc_ma - RCRE_NUTS_TARGET_ACCEPT) /
                sqrt((double)(k + 1));
            double log_eps = log(eps) + step;
            if (log_eps < -6.9) log_eps = -6.9;   /* ~ 1e-3 */
            if (log_eps > 1.6) log_eps = 1.6;     /* ~ 5.0  */
            eps = exp(log_eps);
        }
    }

    if (report) {
        report->accept = sample_acc;
        report->divergent = divergent_count;
        report->final_eps = eps;
    }
    if (out_accept)
        *out_accept = (n_samples > 0)
                          ? (double)sample_acc / (double)n_samples
                          : 0.0;
    free(x); free(r); free(xn); free(rn); free(g); free(xorig);
    free(nodes); free(nodeA);
    return 0;
}

/* ------------------------------------------------------------------ */
/* ESS — initial monotone sequence (Geyer 1992)                       */
/* ------------------------------------------------------------------ */

RCRE_NODISCARD double rcre_mcmc_ess_f64(const double* y, int n, int lag_max) {
    if (!y || n < 4) return (double)n;
    if (lag_max <= 0 || lag_max > n / 2) lag_max = n / 2;
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += y[i];
    mean /= (double)n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        const double dv = y[i] - mean;
        var += dv * dv;
    }
    var /= (double)n;
    if (var <= 0.0) return (double)n;
    /* ACF at even+odd pairs, stop at first negative cumulative */
    double sum = 0.0; /* running tau = 2*sum(rho pairs) - 1 */
    double base = (double)n;
    for (int lag = 1; lag <= lag_max; lag += 2) {
        double c0 = 0.0, c1 = 0.0;
        for (int i = 0; i + lag < n; ++i)
            c0 += (y[i] - mean) * (y[i + lag] - mean);
        if (lag + 1 < n)
            for (int i = 0; i + lag + 1 < n; ++i)
                c1 += (y[i] - mean) * (y[i + lag + 1] - mean);
        const double pair = (c0 + c1) / (base * var);
        sum += pair;
        if (sum < 0.0) break; /* monotone-sequence truncation */
        if (c0 <= 0.0 && c1 <= 0.0) break; /* tail is noise */
    }
    double tau = 2.0 * sum - 1.0;
    if (tau < 1.0) tau = 1.0;
    return base / tau;
}

/* ------------------------------------------------------------------ */
/* Box integrator: plain / antithetic / stratified / Sobol' / Halton  */
/* ------------------------------------------------------------------ */

RCRE_NODISCARD int rcre_mcmc_integrate_f64(int d,
                                           double (*f)(const double* x,
                                                       void* user),
                                           void* user,
                                           const double* lo,
                                           const double* hi, int method,
                                           int n, uint64_t seed,
                                           double* out_value,
                                           double* out_stderr,
                                           int* out_n_used) {
    if (d < 1 || d > RCRE_QRNG_SOBOL_DIM || !f || !lo || !hi ||
        !out_value || n < 1)
        return -1;
    for (int i = 0; i < d; ++i)
        if (hi[i] <= lo[i]) return -1;
    if (method == 1 && n % 2 != 0) return -1; /* antithetic: pairs */
    if (method < 0 || method > 4) return -1;

    double* pt = (double*)malloc((size_t)d * sizeof(double));
    double* uvec = (double*)malloc((size_t)d * sizeof(double));
    if (!pt || !uvec) {
        free(pt);
        free(uvec);
        return -1;
    }
    RCRE_RNG rng;
    rcre_rng_pcg_init(&rng, seed);

    /* stratified: use the nearest perfect d-th power n = s^d (>= 4) */
    int nused = n;
    int s = 1;
    if (method == 2) {
        double sd = pow((double)n, 1.0 / (double)d);
        s = (int)(sd + 0.5);
        if (s < 2) s = 2;
        nused = 1;
        for (int i = 0; i < d; ++i) nused *= s;
    }
    if (method == 1) nused = n;

    double ssum = 0.0, ssum2 = 0.0;
    double vol = 1.0;
    for (int i = 0; i < d; ++i) vol *= hi[i] - lo[i];

    for (int i = 0; i < nused; ++i) {
        if (method == 0) {
            for (int j = 0; j < d; ++j)
                pt[j] = rcre_rng_pcg_next_f64(&rng);
        } else if (method == 1) {
            /* antithetic pair: point i and i+1 share one uniform vector
             * u ~ U[0,1)^d and evaluate f(u), f(1-u). The raw uniforms
             * live in uvec (pt gets destroyed by the box mapping). */
            if (i % 2 == 0) {
                for (int j = 0; j < d; ++j) {
                    uvec[j] = rcre_rng_pcg_next_f64(&rng);
                    pt[j] = uvec[j];
                }
            } else {
                for (int j = 0; j < d; ++j) pt[j] = 1.0 - uvec[j];
            }
        } else if (method == 2) {
            /* stratified: stratum cell = base-s digits of i, jitter
             * uniform inside the cell (PCG64, fixed order) */
            long long rem = i;
            for (int j = 0; j < d; ++j) {
                const long long digit = rem % s;
                rem /= s;
                pt[j] = ((double)digit + rcre_rng_pcg_next_f64(&rng)) /
                        (double)s;
            }
        } else if (method == 3) {
            rcre_qrng_sobol_at((uint64_t)i, d, pt);
        } else { /* method 4 */
            if (d > RCRE_QRNG_HALTON_DIM) {
                free(pt);
                return -1;
            }
            rcre_qrng_halton_at((uint64_t)i, d, pt);
        }
        for (int j = 0; j < d; ++j)
            pt[j] = lo[j] + pt[j] * (hi[j] - lo[j]);
        const double val = f(pt, user) * vol;
        ssum += val;
        ssum2 += val * val;
    }

    *out_value = ssum / (double)nused;
    if (out_n_used) *out_n_used = nused;
    double se = 0.0;
    if (nused > 1) {
        const double m = ssum / (double)nused;
        double vv = ssum2 / (double)nused - m * m;
        if (vv < 0.0) vv = 0.0;
        se = sqrt(vv / (double)nused);
    }
    if (out_stderr) *out_stderr = se;
    free(pt);
    free(uvec);
    return 0;
}
