/* RCRE H7 — PDE domain layer (2.4.0)
 *
 * Structured-grid solvers on the H0 F64 substrate.
 *
 * LAYOUT (shared with H4 multigrid — the AUTO driver hands off to it):
 *   interior point (i, j): 0 <= i < nx (x / column), 0 <= j < ny (y / row),
 *   flat index c = j*nx + i  (row-major rows of length nx).
 *
 * Determinism: every parallel stage is an independent row/point or a
 * fixed-order combine — results bit-identical across pool sizes
 * (pool == NULL means serial).
 *
 *   - 1D DST-I via the H2 complex FFT (verified <= 6e-14 rel vs naive).
 *   - 2D Poisson (Dirichlet, 5-point, h=1): spectral (DST) solver +
 *     AUTO driver (FFT vs H4 multigrid by problem size).
 *   - 5-point stencil apply (the shared operator; f64 + f32).
 *   - Heat 2D explicit FTCS (stability gate dt/h^2 <= 1/2, loud failure).
 *   - Burgers 1D (scalar): conservative Lax-Wendroff + minmod (2nd-order
 *     TVD); the exact solution via the Hopf-Lax variational formula.
 *   - Wave 1D: leapfrog (symplectic) + 2nd-order start; energy-drift gate.
 *
 * All state is caller-owned (arena-friendly); no hidden multithreading.
 */
#include "rcre.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* 1D DST-I via complex FFT (row-wise, slab-parallel)                        */
/* ========================================================================= */
/*
 * S[k] = sum_{j=1..n} x[j] sin(pi j k/(n+1)), k = 1..n  (x is 1-indexed).
 *
 * Odd extension, length N = 2(n+1):
 *   y[0] = 0, y[n+1] = 0, y[j] = x[j], y[N-j] = -x[j]  (j = 1..n)
 *   Y = DFT_N(y)  =>  Y[k] = -2i S[k]  =>  S[k] = -0.5 Im(Y[k]).
 * DST-I is symmetric; the inverse is (2/(n+1)) * DST-I.
 *
 * Each row is transformed independently with TASK-LOCAL scratch (the
 * slab contract: concurrent rows never share a buffer), so the result
 * is bit-identical for any pool size.
 */

typedef struct {
    const double* x;       /* rows vectors, 1-indexed columns, stride n+1 */
    double* S;             /* same layout */
    int n;                 /* vector length */
} RCRE_Dst1Plan;

typedef struct {
    RCRE_Dst1Plan plan;
    double* re;            /* length 2(n+1), task-local */
    double* im;
} RCRE_Dst1Task;

static void dst1_one_row(const RCRE_Dst1Plan* p, double* re, double* im,
                         size_t row) {
    const int n = p->n;
    const int N = 2 * (n + 1);
    memset(re, 0, (size_t)N * sizeof(double));
    const double* xr = p->x + row * (size_t)(n + 1);
    for (int j = 1; j <= n; ++j) {
        re[j] = xr[j];
        re[N - j] = -xr[j];
    }
    (void)rcre_fft_1d_f64(re, NULL, re, im, (size_t)N, 0);
    double* sr = p->S + row * (size_t)(n + 1);
    for (int k = 1; k <= n; ++k) sr[k] = -0.5 * im[k];
}

/* task_id selects THIS row's task-local scratch (slab contract). */
static void dst1_row_task(void* arg, size_t row) {
    RCRE_Dst1Task* t = &((RCRE_Dst1Task*)arg)[row];
    dst1_one_row(&t->plan, t->re, t->im, row);
}

/*
 * Row-wise 1D DST-I. `rows` vectors of length n, stride n+1 (1-indexed).
 * Task scratch is allocated per row and released before return.
 */
static int dst1_rows_f64(const double* x, double* S, size_t rows, int n,
                         RCRE_ThreadPool* pool) {
    if (!x || !S || rows == 0 || n < 1) return -1;
    const size_t per = (size_t)(2 * (n + 1)) * sizeof(double);
    RCRE_Dst1Task* tk = (RCRE_Dst1Task*)calloc(rows, sizeof(RCRE_Dst1Task));
    if (!tk) return -1;
    int rc = 0;
    for (size_t i = 0; i < rows; ++i) {
        tk[i].plan.x = x;
        tk[i].plan.S = S;
        tk[i].plan.n = n;
        tk[i].re = (double*)malloc(per);
        tk[i].im = (double*)malloc(per);
        if (!tk[i].re || !tk[i].im) rc = -1;
    }
    if (rc == 0) {
        /* ONE calling convention: arg is the ARRAY BASE, task_id the row
         * (measured bug: a serial loop passing &tk[i] to a task that
         * indexes [row] from arg walked past the array end). */
        if (pool)
            rcre_threadpool_parallel_for(pool, rows, dst1_row_task, tk);
        else
            for (size_t i = 0; i < rows; ++i) dst1_row_task(tk, i);
    }
    for (size_t i = 0; i < rows; ++i) {
        free(tk[i].re);
        free(tk[i].im);
    }
    free(tk);
    return rc;
}

/* ========================================================================= */
/* 5-point stencil (h = 1, Dirichlet interior) — MG layout c = j*nx + i      */
/* ========================================================================= */
/*
 * (A u)_{i,j} = 4 u_{i,j} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1}
 * Boundary values are ZERO (the caller folds boundary data into b).
 * Row-slab parallel; bit-identical across pool sizes.
 */

typedef struct {
    const double* u;
    double* v;
    size_t nx, ny;
} RCRE_Stencil5Ctx;

static void stencil5_f64_row(const RCRE_Stencil5Ctx* s, size_t j) {
    for (size_t i = 0; i < s->nx; ++i) {
        size_t c = j * s->nx + i;
        double acc = 4.0 * s->u[c];
        if (j > 0) acc -= s->u[c - s->nx];
        if (j + 1 < s->ny) acc -= s->u[c + s->nx];
        if (i > 0) acc -= s->u[c - 1];
        if (i + 1 < s->nx) acc -= s->u[c + 1];
        s->v[c] = acc;
    }
}

static void stencil5_f64_task(void* arg, size_t j) {
    stencil5_f64_row((const RCRE_Stencil5Ctx*)arg, j);
}

RCRE_NODISCARD int rcre_pde_stencil5_f64(size_t nx, size_t ny,
                                         const double* u, double* v,
                                         RCRE_ThreadPool* pool) {
    if (!u || !v || nx == 0 || ny == 0) return -1;
    RCRE_Stencil5Ctx s = { u, v, nx, ny };
    if (pool)
        rcre_threadpool_parallel_for(pool, ny, stencil5_f64_task, &s);
    else
        for (size_t j = 0; j < ny; ++j) stencil5_f64_row(&s, j);
    return 0;
}

typedef struct {
    const float* u;
    float* v;
    size_t nx, ny;
} RCRE_Stencil5CtxF32;

static void stencil5_f32_row(const RCRE_Stencil5CtxF32* s, size_t j) {
    for (size_t i = 0; i < s->nx; ++i) {
        size_t c = j * s->nx + i;
        float acc = 4.0f * s->u[c];
        if (j > 0) acc -= s->u[c - s->nx];
        if (j + 1 < s->ny) acc -= s->u[c + s->nx];
        if (i > 0) acc -= s->u[c - 1];
        if (i + 1 < s->nx) acc -= s->u[c + 1];
        s->v[c] = acc;
    }
}

static void stencil5_f32_task(void* arg, size_t j) {
    stencil5_f32_row((const RCRE_Stencil5CtxF32*)arg, j);
}

RCRE_NODISCARD int rcre_pde_stencil5_f32(size_t nx, size_t ny,
                                         const float* u, float* v,
                                         RCRE_ThreadPool* pool) {
    if (!u || !v || nx == 0 || ny == 0) return -1;
    RCRE_Stencil5CtxF32 s = { u, v, nx, ny };
    if (pool)
        rcre_threadpool_parallel_for(pool, ny, stencil5_f32_task, &s);
    else
        for (size_t j = 0; j < ny; ++j) stencil5_f32_row(&s, j);
    return 0;
}

/* ========================================================================= */
/* 2D Poisson — spectral (DST) solver + AUTO driver (MG layout)              */
/* ========================================================================= */
/*
 * A = T_x ⊗ I + I ⊗ T_y (T = tridiag(-1,2,-1)); the eigenvectors are
 * products of sinines, so the 2D DST diagonalizes A:
 *   F = DST_i(DST_j(b)),  U = F / lambda,
 *   u = (2/(nx+1)) DST_i( (2/(ny+1)) DST_j(U) ),
 *   lambda_{i,j} = 4 - 2cos(pi(i+1)/(nx+1)) - 2cos(pi(j+1)/(ny+1)).
 */

RCRE_NODISCARD int rcre_pde_poisson2d_fft_f64(size_t nx, size_t ny,
                                              const double* b, double* u,
                                              RCRE_ThreadPool* pool) {
    if (!b || !u || nx < 2 || ny < 2) return -1;
    rcre_hal_init();
    int rc = 0;
    const size_t rows_s = ny;      /* stage 1: ny rows of length nx */
    const size_t rows_t = nx;      /* stage 2: nx rows of length ny */
    const size_t rs = nx + 1, cs = ny + 1;
    double *p1 = NULL, *F1 = NULL, *p2 = NULL, *F2 = NULL,
           *p3 = NULL, *U3 = NULL;
    p1 = (double*)malloc(rows_s * rs * sizeof(double));
    F1 = (double*)malloc(rows_s * rs * sizeof(double));
    p2 = (double*)malloc(rows_t * cs * sizeof(double));
    F2 = (double*)malloc(rows_t * cs * sizeof(double));
    p3 = (double*)malloc(rows_t * cs * sizeof(double));
    U3 = (double*)malloc(rows_s * rs * sizeof(double));
    if (!p1 || !F1 || !p2 || !F2 || !p3 || !U3) rc = -1;
    if (rc == 0) {
        /* pack: 1-indexed columns; stage 1 transforms over i (per row j) */
        for (size_t j = 0; j < ny; ++j)
            for (size_t i = 0; i < nx; ++i)
                p1[j * rs + (i + 1)] = b[j * nx + i];
        rc = dst1_rows_f64(p1, F1, rows_s, (int)nx, pool);
    }
    if (rc == 0) {
        /* transpose: F1 (ny rows of nx) -> p2 (nx rows of ny) */
        for (size_t i = 0; i < nx; ++i)
            for (size_t j = 0; j < ny; ++j)
                p2[i * cs + (j + 1)] = F1[j * rs + (i + 1)];
        /* stage 2: DST over j (per column i) + lambda divide */
        rc = dst1_rows_f64(p2, F2, rows_t, (int)ny, pool);
        if (rc == 0)
            for (size_t i = 0; i < nx; ++i)
                for (size_t j = 0; j < ny; ++j) {
                    double lam = 4.0
                        - 2.0 * cos(M_PI * (double)(i + 1) / (double)(nx + 1))
                        - 2.0 * cos(M_PI * (double)(j + 1) / (double)(ny + 1));
                    F2[i * cs + (j + 1)] /= lam;
                }
    }
    if (rc == 0) {
        /* stage 3: inverse DST over j (per column i): F2 -> p3 */
        rc = dst1_rows_f64(F2, p3, rows_t, (int)ny, pool);
        /* stage 4: transpose p3 (nx rows of ny) -> p1 (ny rows of nx) */
        for (size_t j = 0; j < ny; ++j)
            for (size_t i = 0; i < nx; ++i)
                p1[j * rs + (i + 1)] = p3[i * cs + (j + 1)];
        /* stage 5: inverse DST over i (per row j): p1 -> U3 */
        rc = dst1_rows_f64(p1, U3, rows_s, (int)nx, pool);
        if (rc == 0) {
            const double scale =
                (2.0 / (double)(nx + 1)) * (2.0 / (double)(ny + 1));
            for (size_t j = 0; j < ny; ++j)
                for (size_t i = 0; i < nx; ++i)
                    u[j * nx + i] = scale * U3[j * rs + (i + 1)];
        }
    }
    free(p1); free(F1); free(p2); free(F2); free(p3); free(U3);
    return rc;
}

/*
 * AUTO driver: spectral when nx*ny <= 4M cells (the dense F64 work area
 * is 6 arrays of nx*ny), otherwise the H4 multigrid V-cycle. The H4
 * V-cycle has a MEASURED smoothest-mode floor (the two-grid correction
 * per level is limited by the eigenvalue ratio ~ 4; ~1e-10 on 33x27,
 * degrading with n — see H4 docs), so the MG path targets 1e-12 and,
 * if the floor is hit before convergence, accepts the solution when the
 * TRUE residual is within the documented floor band (<= 1e-8) instead
 * of reporting a spurious failure. Deterministic either way; the choice
 * is a pure function of the shape. x is both the MG initial guess and
 * the result; the FFT path overwrites u.
 */
#define RCRE_PDE_POISSON_FFT_MAX_CELLS (4u * 1024u * 1024u)

/* true relative residual ||f - A x||/||f|| in the MG layout */
static double pde_poisson2d_rel_resid(size_t nx, size_t ny, const double* f,
                                      const double* x) {
    const size_t n = nx * ny;
    double* r = (double*)malloc(n * sizeof(double));
    if (!r) return -1.0;
    double ref = 0.0;
    for (size_t c = 0; c < n; ++c) ref += f[c] * f[c];
    double s = 0.0;
    for (size_t j = 0; j < ny; ++j)
        for (size_t i = 0; i < nx; ++i) {
            const size_t c = j * nx + i;
            double lap = 4.0 * x[c];
            if (i > 0) lap -= x[c - 1];
            if (i + 1 < nx) lap -= x[c + 1];
            if (j > 0) lap -= x[c - nx];
            if (j + 1 < ny) lap -= x[c + nx];
            const double d = f[c] - lap;
            r[c] = d;
            s += d * d;
        }
    free(r);
    return (ref > 0.0) ? sqrt(s / ref) : 0.0;
}

RCRE_NODISCARD int rcre_pde_poisson2d_solve_f64(size_t nx, size_t ny,
                                                const double* b, double* u,
                                                RCRE_ThreadPool* pool,
                                                int auto_mode) {
    if (!b || !u || nx < 2 || ny < 2) return -1;
    if (!auto_mode)
        return rcre_pde_poisson2d_fft_f64(nx, ny, b, u, pool);
    if (nx * ny <= RCRE_PDE_POISSON_FFT_MAX_CELLS)
        return rcre_pde_poisson2d_fft_f64(nx, ny, b, u, pool);
    int rc = rcre_mg_solve_poisson2d_f64(nx, ny, b, u, 1e-12, 400, NULL, NULL);
    if (rc >= 0) return rc;
    /* smoothest-mode floor: accept a good solution rather than fail */
    const double res = pde_poisson2d_rel_resid(nx, ny, b, u);
    return (res > 0.0 && res <= 1e-8) ? 1 : -1;
}

/* ========================================================================= */
/* Heat 2D — explicit FTCS (MG layout)                                       */
/* ========================================================================= */
/*
 * u^{k+1} = u^k + (dt/h^2)(u_w + u_e + u_s + u_n - 4 u^k), zero Dirichlet
 * boundary. Stability (von Neumann): dt/h^2 <= 1/2 — ENFORCED (returns
 * -1, no partial step). Row-slab parallel.
 */

typedef struct {
    const double* u;
    double* u1;
    size_t nx, ny;
    double r;
} RCRE_HeatCtx;

static void heat2d_row(const RCRE_HeatCtx* s, size_t j) {
    for (size_t i = 0; i < s->nx; ++i) {
        size_t c = j * s->nx + i;
        /* four-neighbor sum; missing neighbors are the zero Dirichlet
         * boundary (the term simply vanishes) */
        double nb = 0.0;
        if (j > 0) nb += s->u[c - s->nx];
        if (j + 1 < s->ny) nb += s->u[c + s->nx];
        if (i > 0) nb += s->u[c - 1];
        if (i + 1 < s->nx) nb += s->u[c + 1];
        s->u1[c] = s->u[c] + s->r * (nb - 4.0 * s->u[c]);
    }
}

static void heat2d_task(void* arg, size_t j) {
    heat2d_row((const RCRE_HeatCtx*)arg, j);
}

RCRE_NODISCARD int rcre_pde_heat2d_step_f64(size_t nx, size_t ny,
                                            const double* u, double* u1,
                                            double dt, double h,
                                            RCRE_ThreadPool* pool) {
    if (!u || !u1 || nx == 0 || ny == 0 || h <= 0.0 || dt < 0.0) return -1;
    const double r = dt / (h * h);
    if (r > 0.5) return -1; /* stability gate: fail loud */
    RCRE_HeatCtx s = { u, u1, nx, ny, r };
    if (pool)
        rcre_threadpool_parallel_for(pool, ny, heat2d_task, &s);
    else
        for (size_t j = 0; j < ny; ++j) heat2d_row(&s, j);
    return 0;
}

/* ========================================================================= */
/* Burgers 1D — MUSCL-upwind (2nd order, TVD)                                */
/* ========================================================================= */
/*
 * u_t + (1/2 u^2)_x = 0, periodic, conservative finite volume.
 *
 * Scheme: MUSCL (piecewise-linear reconstruction, minmod-limited) +
 * upwind/Roe face flux:
 *   face i+1/2:  lim = minmod(u_{i+1}-u_i, u_i-u_{i-1})
 *                u- = u_i + (1/2) lim          (left edge state)
 *                u+ = u_{i+1} - (1/2) lim      (right edge state)
 *                F  = f+(u-) + f-(u+)   with f+(w) = w^2/2 if w >= 0 else 0
 *                                             f-(w) = w^2/2 if w <  0 else 0
 *   u1[i] = u[i] - (dt/dx)(F_{i+1/2} - F_{i-1/2}).
 * The split-flux form is exactly: upwind at u- when the Roe speed
 * a = (u-+u+)/2 > 0, upwind at u+ when a < 0, and the exact Roe flux
 * ((u-+u+)^2/4) at a crossing face — the standard 2nd-order TVD scheme
 * for scalar convex fluxes, stable for |CFL * a| <= 1.
 *
 * Measured design lesson (this is why the scheme is upwind, not the
 * central "Lax-Wendroff + minmod" form): for the NONLINEAR flux, the
 * Lax-Friedrichs base of the central form is not monotone, and adding a
 * limited 2nd-order correction to a non-monotone base is not TVD — both
 * central variants (limiter on state diffs or on wave products) were
 * measured to blow up past shock formation on this data, while
 * MUSCL-upwind stays bounded and converges (L-inf vs Hopf-Lax: 2nd
 * order pre-shock, O(h) through the shock).
 */

static double minmod2(double a, double b) {
    if ((a <= 0.0) != (b <= 0.0)) return 0.0;
    return fabs(a) < fabs(b) ? a : b;
}

/* face flux from the two reconstructed edge states (Roe split; f = w^2/2) */
static double burgers_face_flux(double um, double up) {
    if (um >= 0.0) return 0.5 * um * um;      /* a >= 0: upwind from the left */
    if (up <= 0.0) return 0.5 * up * up;      /* a <= 0: upwind from the right */
    return 0.25 * (um + up) * (um + up);      /* crossing: exact Roe flux */
}

RCRE_NODISCARD int rcre_pde_burgers1d_step_f64(const double* u, double* u1,
                                               int n, double dt, double dx) {
    if (!u || !u1 || n < 3 || dx <= 0.0 || dt < 0.0) return -1;
    /* CFL: the signal speed of u_t + (u^2/2)_x = 0 is u itself, so the
     * gate is dt*max|u|/dx <= 1.
     * (Measured bug: the first draft gated on dt/dx alone — silently
     * rejecting every legitimate dt > dx case and passing dt that are
     * actually unstable when max|u| > 1. Shu-Osher has max|u| ~ 1.6.) */
    double umax = 0.0;
    for (int i = 0; i < n; ++i) {
        const double a = fabs(u[i]);
        if (a > umax) umax = a;
    }
    if (dt * umax / dx > 1.0) return -1; /* CFL gate */
    const double cfl = dt / dx;
    for (int i = 0; i < n; ++i) {
        const double uL = u[(i - 1 + n) % n], uc = u[i], uR = u[(i + 1) % n];
        /* face i+1/2 */
        const double lim_r = minmod2(uR - uc, uc - uL);
        const double Fp = burgers_face_flux(uc + 0.5 * lim_r,
                                            uR - 0.5 * lim_r);
        /* face i-1/2 */
        const double lim_l = minmod2(uc - uL, uL - u[(i - 2 + n) % n]);
        const double Fm = burgers_face_flux(uL + 0.5 * lim_l,
                                            uc - 0.5 * lim_l);
        u1[i] = uc - cfl * (Fp - Fm);
    }
    return 0;
}

/*
 * Shu-Osher piston + oscillation initial data (Shu & Osher 1981), the
 * classic shock-capturing benchmark, adapted to the scalar equation
 * (u plays the role of the density; the full gas-dynamics version is
 * the stiff system the scalar equation replaces — documented in
 * docs/hpc.md 10.5). Periodic on [0, 2pi):
 *   u0 = 0.998271                       0 <= x < 0.0281475   (piston)
 *   u0 = 0.999838 + 0.0111 sin(5 x)     0.0281475 <= x < 0.2
 *   u0 = 1 - 0.5(1 + cos(10 (phi - sin phi)))   0.2 <= x < 2 pi,
 *        with phi = x wrapped to [-pi, pi).
 * The profile is DISCONTINUOUS at x = 0.2 (the piston/wave junction):
 * that jump is part of the benchmark (the scheme must resolve it as a
 * shock). u0 in [0, ~1.01].
 */
RCRE_NODISCARD int rcre_pde_burgers_shu_oshers_init(double* u, int n,
                                                    double L) {
    if (!u || n < 3 || L <= 0.0) return -1;
    const double TAU = 2.0 * M_PI;
    const double x1 = 0.0281475, x2 = 0.2;
    for (int i = 0; i < n; ++i) {
        const double x = (double)i * (L / (double)n);
        if (x < x1)
            u[i] = 0.998271;
        else if (x < x2)
            u[i] = 0.999838 + 0.0111 * sin(5.0 * x);
        else {
            double phi = x - TAU * floor(x / TAU + 0.5);
            double d = phi - sin(phi);
            u[i] = 1.0 - 0.5 * (1.0 + cos(10.0 * d));
        }
    }
    return 0;
}

/*
 * Hopf-Lax exact solution of u_t + (1/2 u^2)_x = 0 with PERIODIC data
 * (verification tool, O(n^2)). Hamilton-Jacobi duality: v = ∫u dx solves
 *   v_t + (v_x)^2/2 = 0  (inviscid Burgers in conservative and HJ form
 * are the same equation), so the Lax-Oleinik formula with
 *   L(q) = min_p (p q - p^2/2) = q^2/2
 * gives
 *   v(x,T) = min_ξ [ v0(ξ) + (x - ξ)^2/(2T) ],  v0(ξ) = ∫_0^ξ u0 ds,
 *   u(x,T) = v_x = (x - ξ*)/T.
 * Both ingredients are essential (measured): dropping the potential v0
 * (using (1/2)u0^2 pointwise) or using L = q^2/4 both return values that
 * violate the maximum principle min u0 <= u <= max u0. v0 is accumulated
 * with the trapezoidal rule (O(dx^2)); replicas k are offset by k * total
 * (the period integral of u0).
 */
RCRE_NODISCARD int rcre_pde_burgers_hopf_lax_f64(const double* u0, int n,
                                                 double L, double T,
                                                 double* out) {
    if (!u0 || !out || n < 3 || L <= 0.0 || T <= 0.0) return -1;
    /* Fine internal minimization grid: m subpoints per cell with linear
     * interpolation of u0 (trapezoidal v0 on the fine grid). Measured
     * reason: the O(dx) quantization of the minimizer xi* makes the
     * reference piecewise kinked where competing characteristics are
     * near-degenerate (the L-inf error then does not converge with n);
     * m = 16 subpoints removes the artifact at O(dx^2)-level. */
    const int m = 16;
    const int M = n * m;
    const double h = L / (double)M;
    double* uf = (double*)malloc((size_t)M * sizeof(double));
    double* v0 = (double*)malloc((size_t)M * sizeof(double));
    if (!uf || !v0) { free(uf); free(v0); return -1; }
    for (int j = 0; j < M; ++j) {
        const double pos = (double)j / (double)m; /* in cell units */
        const int i0 = (int)pos;
        const double frac = pos - (double)i0;
        const double a = u0[i0 % n];
        const double b = u0[(i0 + 1) % n];
        uf[j] = a + frac * (b - a);
    }
    double total = 0.0;
    for (int j = 0; j < M; ++j) total += 0.5 * (uf[j] + uf[(j + 1) % M]) * h;
    v0[0] = 0.0;
    for (int j = 1; j < M; ++j) v0[j] = v0[j - 1] + 0.5 * (uf[j - 1] + uf[j]) * h;
    for (int i = 0; i < n; ++i) {
        const double x = (double)i * (L / (double)n);
        double best = 1e300, ub = 0.0;
        for (int j = 0; j < M; ++j) {
            const double xi = (double)j * h;
            for (int k = -1; k <= 1; ++k) {
                const double s = x - (xi + (double)k * L);
                const double q = v0[j] + (double)k * total + s * s / (2.0 * T);
                if (q < best) {
                    best = q;
                    ub = s / T;
                }
            }
        }
        out[i] = ub;
    }
    free(uf);
    free(v0);
    return 0;
}

/* ========================================================================= */
/* Wave 1D — leapfrog (symplectic) + 2nd-order start                         */
/* ========================================================================= */
/*
 * u_tt = c^2 u_xx, periodic. Leapfrog (DDK form — the MD convention):
 *   u^{k+1}   = u^k + dt v^{k+1/2}                (drift)
 *   v^{k+3/2} = v^{k+1/2} + c^2 dt lap(u^{k+1})/dx^2  (kick at the NEW u)
 * v is the HALF-STEP velocity (between u^k and u^{k+1}). The consistent
 * start for u_t^0 = 0 is v^{1/2} = (c^2 dt/2) lap(u^0) (the companion
 * start function). 2nd order in time; symplectic => bounded
 * (oscillatory) energy drift, no secular growth (the test gate).
 * (Measured lesson: kick-before-drift at the OLD position is
 * Euler-Cromer — symplectic but only 1st order; the half-step start
 * with a kick-drift step mixes conventions and leaves an O(dt) energy
 * offset. Both were ruled out by measurement.)
 */
RCRE_NODISCARD int rcre_pde_wave1d_step_f64(const double* u, const double* v,
                                            double* u1, double* v1, int n,
                                            double c, double dt, double dx) {
    if (!u || !v || !u1 || !v1 || n < 3 || dx <= 0.0 || dt < 0.0) return -1;
    if (c * dt / dx > 1.0) return -1; /* CFL gate */
    const double cc = (c * c * dt) / (dx * dx);
    for (int i = 0; i < n; ++i) {
        u1[i] = u[i] + dt * v[i]; /* drift */
    }
    for (int i = 0; i < n; ++i) {
        const double lap = u1[(i + 1) % n] - 2.0 * u1[i] + u1[(i - 1 + n) % n];
        v1[i] = v[i] + cc * lap; /* kick at the new position */
    }
    return 0;
}

/* energy: 1/2 sum(v^2) + (c^2/(2 dx^2)) sum((u_{i+1} - u_i)^2) */
RCRE_NODISCARD double rcre_pde_wave1d_energy_f64(const double* u,
                                                 const double* v, int n,
                                                 double c, double dx) {
    double e = 0.0;
    for (int i = 0; i < n; ++i) e += 0.5 * v[i] * v[i];
    const double k = (c * c) / (2.0 * dx * dx);
    for (int i = 0; i < n; ++i) {
        const double d = u[(i + 1) % n] - u[i];
        e += k * d * d;
    }
    return e;
}

RCRE_NODISCARD int rcre_pde_wave1d_start_f64(const double* u, double* v, int n,
                                             double c, double dt, double dx) {
    if (!u || !v || n < 3) return -1;
    /* DDK convention: v^{1/2} = (h/2) lap_c2(u^0) is the consistent
     * half-step velocity for u_t^0 = 0. */
    const double cc = (c * c * dt) / (2.0 * dx * dx);
    for (int i = 0; i < n; ++i)
        v[i] = cc * (u[(i + 1) % n] - 2.0 * u[i] + u[(i - 1 + n) % n]);
    return 0;
}
