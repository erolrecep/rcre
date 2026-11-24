/* ==========================================================================
 * RCRE — Geometric Multigrid, 2D Poisson (H4, 2.2.0)
 *
 * Two classic geometric multigrid cycle types for the 2D Poisson equation
 * with Dirichlet boundary conditions on a structured tensor-product grid
 * (the canonical HPC model problem: the 5-point Laplacian, h^2-scaled:
 *
 *     (A u)_{i,j} = 4 u_{i,j} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1}
 *
 * with interior indices 0..nx-1 / 0..ny-1 and zero boundary values (the
 * caller folds the boundary data into the RHS f)). Both share the same
 * operators: red-black Gauss-Seidel smoothing (exact substitution; the two
 * colors are independent and the fixed traversal is deterministic),
 * full-weighting restriction R (1/16 of [4 1 1; 1 4 1; 1 1 4]), bilinear
 * prolongation P, and an exact dense Gaussian solve at the coarsest level.
 *
 *   V-cycle (rcre_mg_vcycle_poisson2d): residual correction — the classic
 *   pattern. The defect is restricted, a coarse V-cycle (started from zero)
 *   computes its correction, which is prolonged and added.
 *
 *   FAS (rcre_mg_fas_poisson2d): Full Approximation Scheme — the coarse
 *   level solves against the full restricted RHS R f starting from the
 *   full-weighting of u, and the fine update is u += P (u_c_after -
 *   u_c_before). FAS corrects against the full RHS at every level instead
 *   of only the residual; it is provided as the alternative cycle for
 *   users to pick per problem.
 *
 * Measured behavior (Apple M5 Max, -O3, see docs/hpc.md): both cycles
 * reduce broadband residuals to 1e-10 in a modest number of cycles on
 * 32^2..512^2. A PURE smoothest-mode error decays slowly and h-dependently
 * (per-cycle rate -> 1 as n grows; the two-grid correction of the smoothest
 * mode is limited by the eigenvalue ratio lambda_1,coarse/lambda_1,fine ~ 4
 * per level). That is a property of fixed-count 2:1 geometric MG for the
 * model problem, not an implementation defect: for smooth-error-dominated
 * solves use rcre_krylov_pcg + RCRE_PC_ILU0 instead.
 *
 * The recursion is depth-first and deterministic: same inputs produce the
 * same outputs on every machine and core count. OOM at any level degrades
 * gracefully (no correction at that level; the cycle still applies
 * smoothing).
 * ========================================================================== */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "rcre.h"

/* --- F64 core ------------------------------------------------------------- */

/* Red-black Gauss-Seidel sweep: exact substitution in color order. The
 * sequential in-place update is what gives GS its strong low-frequency
 * smoothing — with weighted Jacobi the V-cycle's mode-1 factor drifts to 1
 * faster as the grid grows, with red-black GS it stays the classical
 * model-problem behavior. The two colors are independent (parallelizable
 * later); the fixed traversal keeps the result deterministic. */
static void mg2d_gs_sweep_f64(size_t nx, size_t ny, const double* f, double* u,
                              int color) {
    for (size_t j = (size_t)color; j < ny; j += 2) {
        for (size_t i = 0; i < nx; ++i) {
            const size_t c = j * nx + i;
            double lap = 4.0 * u[c];
            if (i > 0) lap -= u[c - 1];
            if (i + 1 < nx) lap -= u[c + 1];
            if (j > 0) lap -= u[c - nx];
            if (j + 1 < ny) lap -= u[c + nx];
            u[c] += (f[c] - lap) / 4.0;
        }
    }
}

static void mg2d_res_op_f64(size_t nx, size_t ny, const double* f, const double* u,
                            double* out) {
    /* out = f - A u (the 5-point stencil; A has 4 on the diagonal and -1
     * off-diagonal — the neighbors enter with a MINUS sign). */
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            const size_t c = j * nx + i;
            double lap = 4.0 * u[c];
            if (i > 0) lap -= u[c - 1];
            if (i + 1 < nx) lap -= u[c + 1];
            if (j > 0) lap -= u[c - nx];
            if (j + 1 < ny) lap -= u[c + nx];
            out[c] = f[c] - lap;
        }
    }
}

static void mg2d_restrict_f64(size_t nx, size_t ny, const double* f,
                              size_t nxc, size_t nyc, double* g) {
    /* Full weighting: g(i,j) = (4 f(2i+1,2j+1) + f(2i,2j+1) + f(2i+2,2j+1)
     *                    + f(2i+1,2j) + f(2i+1,2j+2)) / 16.
     * Points outside the fine interior are zero (Dirichlet ghosts). */
    for (size_t j = 0; j < nyc; ++j) {
        for (size_t i = 0; i < nxc; ++i) {
            const size_t ci = 2 * i + 1;
            const size_t cj = 2 * j + 1;
            double s = 0.0;
            if (ci < nx && cj < ny) s += 4.0 * f[cj * nx + ci];
            if (2 * i < nx && cj < ny) s += f[cj * nx + (2 * i)];
            if (2 * i + 2 < nx && cj < ny) s += f[cj * nx + (2 * i + 2)];
            if (ci < nx && 2 * j < ny) s += f[(2 * j) * nx + ci];
            if (ci < nx && 2 * j + 2 < ny) s += f[(2 * j + 2) * nx + ci];
            g[j * nxc + i] = s / 16.0;
        }
    }
}

static void mg2d_prolong_f64(size_t nx, size_t ny, const double* g,
                             size_t nxc, size_t nyc, double* h) {
    /* Bilinear 4-2-1 interpolation into h (h must be zeroed first). */
    for (size_t j = 0; j < nyc; ++j) {
        for (size_t i = 0; i < nxc; ++i) {
            const double v = g[j * nxc + i];
            const size_t ci = 2 * i + 1;
            const size_t cj = 2 * j + 1;
            if (ci < nx && cj < ny) h[cj * nx + ci] += v;
            if (2 * i < nx && cj < ny) h[cj * nx + (2 * i)] += 0.5 * v;
            if (2 * i + 2 < nx && cj < ny) h[cj * nx + (2 * i + 2)] += 0.5 * v;
            if (ci < nx && 2 * j < ny) h[(2 * j) * nx + ci] += 0.5 * v;
            if (ci < nx && 2 * j + 2 < ny) h[(2 * j + 2) * nx + ci] += 0.5 * v;
        }
    }
}

/* Exact dense solver for the coarsest level (tiny: nx*ny <= 256). FAS needs
 * an exact coarsest-level correction: the coarsest level carries the whole
 * smoothest mode of the RHS, so an inexact solve sets a stagnation floor. */
static int mg2d_dense_solve_f64(size_t nx, size_t ny, const double* rhs, double* x) {
    const size_t n = nx * ny;
    if (n > 256) return -1;
    double* M = (double*)malloc((n + 1) * n * sizeof(double)); /* augmented */
    if (!M) return -1;
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            const size_t c = j * nx + i;
            for (size_t q = 0; q < n; ++q) M[c * (n + 1) + q] = 0.0;
            M[c * (n + 1) + c] = 4.0;
            if (i > 0) M[c * (n + 1) + (j * nx + (i - 1))] = -1.0;
            if (i + 1 < nx) M[c * (n + 1) + (j * nx + (i + 1))] = -1.0;
            if (j > 0) M[c * (n + 1) + ((j - 1) * nx + i)] = -1.0;
            if (j + 1 < ny) M[c * (n + 1) + ((j + 1) * nx + i)] = -1.0;
            M[c * (n + 1) + n] = rhs[c];
        }
    }
    for (size_t c = 0; c < n; ++c) {
        size_t p = c;
        for (size_t r = c + 1; r < n; ++r)
            if (fabs(M[r * (n + 1) + c]) > fabs(M[p * (n + 1) + c])) p = r;
        for (size_t q = c; q <= n; ++q) {
            double t = M[c * (n + 1) + q];
            M[c * (n + 1) + q] = M[p * (n + 1) + q];
            M[p * (n + 1) + q] = t;
        }
        const double piv = M[c * (n + 1) + c];
        if (piv == 0.0) {
            free(M);
            return -1;
        }
        for (size_t r = c + 1; r < n; ++r) {
            const double g = M[r * (n + 1) + c] / piv;
            if (g == 0.0) continue;
            for (size_t q = c; q <= n; ++q) M[r * (n + 1) + q] -= g * M[c * (n + 1) + q];
        }
    }
    for (size_t i = n; i-- > 0;) {
        double sv = M[i * (n + 1) + n];
        for (size_t j = i + 1; j < n; ++j) sv -= M[i * (n + 1) + j] * x[j];
        x[i] = sv / M[i * (n + 1) + i];
    }
    free(M);
    return 0;
}

/* One FAS cycle at this level. f: RHS at this level; u: approximation
 * (in/out). Each level allocates its own scratch (sizes shrink geometrically). */
static void mg2d_fas_rec_f64(size_t nx, size_t ny, const double* f, double* u,
                             int pre, int post) {
    double* r = (double*)malloc((size_t)nx * ny * sizeof(double));
    double* z = (double*)malloc((size_t)nx * ny * sizeof(double));
    if (!r || !z) {
        free(r);
        free(z);
        return; /* OOM: no correction at this level */
    }
    /* 1. pre-smoothing */
    for (int s = 0; s < pre; ++s) {
        mg2d_gs_sweep_f64(nx, ny, f, u, 0);
        mg2d_gs_sweep_f64(nx, ny, f, u, 1);
    }
    /* 2. defect */
    mg2d_res_op_f64(nx, ny, f, u, r);
    const size_t nxc = (nx + 1) / 2;
    const size_t nyc = (ny + 1) / 2;
    if (nxc < 2 || nyc < 2) {
        /* 3. coarsest level: exact correction u += A^-1 r */
        double* e = (double*)malloc((size_t)nx * ny * sizeof(double));
        if (e) {
            memset(e, 0, (size_t)nx * ny * sizeof(double));
            if (mg2d_dense_solve_f64(nx, ny, r, e) == 0)
                for (size_t i = 0; i < (size_t)nx * ny; ++i) u[i] += e[i];
        }
        free(e);
        free(r);
        free(z);
        return;
    }
    /* 4. coarse FAS: RHS = R f, start from P u */
    mg2d_restrict_f64(nx, ny, f, nxc, nyc, z); /* z = coarse RHS */
    double* cu = (double*)malloc(nxc * nyc * sizeof(double));
    double* c0 = (double*)malloc(nxc * nyc * sizeof(double));
    if (cu && c0) {
        /* coarse start = prolongation of the current fine approximation */
        memset(cu, 0, nxc * nyc * sizeof(double));
        /* P u: the coarse-grid approximation of u */
        mg2d_restrict_f64(nx, ny, u, nxc, nyc, c0); /* full-weighting of u */
        memcpy(cu, c0, nxc * nyc * sizeof(double));
        mg2d_fas_rec_f64(nxc, nyc, z, cu, pre, post);
        /* 5. add the coarse-grid correction */
        double* d = c0; /* reuse: d = cu - c0 */
        for (size_t i = 0; i < nxc * nyc; ++i) d[i] = cu[i] - c0[i];
        /* z held the coarse RHS; prolong accumulates with +=, so zero it first */
        memset(z, 0, (size_t)nx * ny * sizeof(double));
        mg2d_prolong_f64(nx, ny, d, nxc, nyc, z); /* z = P d */
        for (size_t i = 0; i < (size_t)nx * ny; ++i) u[i] += z[i];
    }
    free(cu);
    free(c0);
    /* 6. post-smoothing */
    for (int s = 0; s < post; ++s) {
        mg2d_gs_sweep_f64(nx, ny, f, u, 0);
        mg2d_gs_sweep_f64(nx, ny, f, u, 1);
    }
    free(r);
    free(z);
}

RCRE_NODISCARD int rcre_mg_fas_poisson2d_f64(size_t nx, size_t ny,
                                             const double* f, double* x,
                                             int pre, int post) {
    if (!f || !x || nx < 2 || ny < 2 || pre < 0 || post < 0 || pre + post == 0)
        return -1;
    mg2d_fas_rec_f64(nx, ny, f, x, pre, post);
    return 1;
}

/* Classic residual-correction V-cycle (F64). u is the running approximation
 * (in/out); the coarse-level approximation always starts from zero (the
 * standard V-cycle coarse start). Anatomy: pre-smooth, defect, restrict the
 * DEFECT (not the RHS), recurse, prolong + add the coarse correction,
 * post-smooth; exact dense solve at the coarsest level. */
static void mg2d_vcycle_rec_f64(size_t nx, size_t ny, const double* f, double* u,
                                int pre, int post) {
    double* r = (double*)malloc((size_t)nx * ny * sizeof(double));
    if (!r) return; /* OOM: no correction at this level */
    /* 1. pre-smoothing */
    for (int s = 0; s < pre; ++s) {
        mg2d_gs_sweep_f64(nx, ny, f, u, 0);
        mg2d_gs_sweep_f64(nx, ny, f, u, 1);
    }
    /* 2. defect */
    mg2d_res_op_f64(nx, ny, f, u, r);
    const size_t nxc = (nx + 1) / 2;
    const size_t nyc = (ny + 1) / 2;
    if (nxc < 2 || nyc < 2) {
        /* 3. coarsest level: exact correction u += A^-1 r */
        double* e = (double*)malloc((size_t)nx * ny * sizeof(double));
        if (e) {
            memset(e, 0, (size_t)nx * ny * sizeof(double));
            if (mg2d_dense_solve_f64(nx, ny, r, e) == 0)
                for (size_t i = 0; i < (size_t)nx * ny; ++i) u[i] += e[i];
        }
        free(e);
        free(r);
        return;
    }
    /* 4. coarse V-cycle on the restricted defect, started from zero */
    double* rc = (double*)malloc(nxc * nyc * sizeof(double));
    double* uc = (double*)calloc(nxc * nyc, sizeof(double));
    if (rc && uc) {
        mg2d_restrict_f64(nx, ny, r, nxc, nyc, rc); /* rc = R r */
        mg2d_vcycle_rec_f64(nxc, nyc, rc, uc, pre, post);
        /* 5. prolong and add the coarse correction */
        double* pd = (double*)calloc((size_t)nx * ny, sizeof(double));
        if (pd) {
            mg2d_prolong_f64(nx, ny, uc, nxc, nyc, pd); /* pd = P uc */
            for (size_t i = 0; i < (size_t)nx * ny; ++i) u[i] += pd[i];
        }
        free(pd);
    }
    free(rc);
    free(uc);
    /* 6. post-smoothing */
    for (int s = 0; s < post; ++s) {
        mg2d_gs_sweep_f64(nx, ny, f, u, 0);
        mg2d_gs_sweep_f64(nx, ny, f, u, 1);
    }
    free(r);
}

RCRE_NODISCARD int rcre_mg_vcycle_poisson2d_f64(size_t nx, size_t ny,
                                                const double* f, double* x,
                                                int pre, int post) {
    if (!f || !x || nx < 2 || ny < 2 || pre < 0 || post < 0 || pre + post == 0)
        return -1;
    mg2d_vcycle_rec_f64(nx, ny, f, x, pre, post);
    return 1;
}

RCRE_NODISCARD int rcre_mg_solve_poisson2d_f64(size_t nx, size_t ny,
                                               const double* f, double* x,
                                               double tol, int max_cycles,
                                               RCRE_KrylovCallback cb, void* user) {
    if (!f || !x || nx < 2 || ny < 2) return -1;
    const size_t n = nx * ny;
    double* r = (double*)malloc(n * sizeof(double));
    if (!r) return -1;
    double nf = 0.0;
    for (size_t i = 0; i < n; ++i) nf += f[i] * f[i];
    const double ref = nf > 0.0 ? sqrt(nf) : 1.0;
    mg2d_res_op_f64(nx, ny, f, x, r);
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += r[i] * r[i];
    double rel = sqrt(s) / ref;
    if (cb) cb(0, rel, user);
    if (rel <= tol) {
        free(r);
        return 0;
    }
    int cycles = 0;
    int ok = 0;
    while (cycles < max_cycles) {
        ++cycles;
        if (rcre_mg_vcycle_poisson2d_f64(nx, ny, f, x, 2, 2) < 0) break;
        mg2d_res_op_f64(nx, ny, f, x, r);
        s = 0.0;
        for (size_t i = 0; i < n; ++i) s += r[i] * r[i];
        rel = sqrt(s) / ref;
        if (cb) cb((size_t)cycles, rel, user);
        if (rel <= tol) {
            ok = 1;
            break;
        }
    }
    free(r);
    return ok ? cycles : -1;
}

/* --- F32 twin -------------------------------------------------------------- */

/* Red-black Gauss-Seidel sweep, F32 twin. */
static void mg2d_gs_sweep_f32(size_t nx, size_t ny, const float* f, float* u,
                              int color) {
    for (size_t j = (size_t)color; j < ny; j += 2) {
        for (size_t i = 0; i < nx; ++i) {
            const size_t c = j * nx + i;
            float lap = (float)4 * u[c];
            if (i > 0) lap -= u[c - 1];
            if (i + 1 < nx) lap -= u[c + 1];
            if (j > 0) lap -= u[c - nx];
            if (j + 1 < ny) lap -= u[c + nx];
            u[c] += (f[c] - lap) * 0.25f;
        }
    }
}

static void mg2d_res_op_f32(size_t nx, size_t ny, const float* f, const float* u,
                            float* out) {
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            const size_t c = j * nx + i;
            float lap = (float)4 * u[c];
            if (i > 0) lap -= u[c - 1];
            if (i + 1 < nx) lap -= u[c + 1];
            if (j > 0) lap -= u[c - nx];
            if (j + 1 < ny) lap -= u[c + nx];
            out[c] = f[c] - lap;
        }
    }
}

static void mg2d_restrict_f32(size_t nx, size_t ny, const float* f,
                              size_t nxc, size_t nyc, float* g) {
    for (size_t j = 0; j < nyc; ++j) {
        for (size_t i = 0; i < nxc; ++i) {
            const size_t ci = 2 * i + 1;
            const size_t cj = 2 * j + 1;
            float s = (float)0;
            if (ci < nx && cj < ny) s += (float)4 * f[cj * nx + ci];
            if (2 * i < nx && cj < ny) s += f[cj * nx + (2 * i)];
            if (2 * i + 2 < nx && cj < ny) s += f[cj * nx + (2 * i + 2)];
            if (ci < nx && 2 * j < ny) s += f[(2 * j) * nx + ci];
            if (ci < nx && 2 * j + 2 < ny) s += f[(2 * j + 2) * nx + ci];
            g[j * nxc + i] = s * (float)(1.0 / 16.0);
        }
    }
}

static void mg2d_prolong_f32(size_t nx, size_t ny, const float* g,
                             size_t nxc, size_t nyc, float* h) {
    for (size_t j = 0; j < nyc; ++j) {
        for (size_t i = 0; i < nxc; ++i) {
            const float v = g[j * nxc + i];
            const size_t ci = 2 * i + 1;
            const size_t cj = 2 * j + 1;
            if (ci < nx && cj < ny) h[cj * nx + ci] += v;
            if (2 * i < nx && cj < ny) h[cj * nx + (2 * i)] += (float)0.5 * v;
            if (2 * i + 2 < nx && cj < ny) h[cj * nx + (2 * i + 2)] += (float)0.5 * v;
            if (ci < nx && 2 * j < ny) h[(2 * j) * nx + ci] += (float)0.5 * v;
            if (ci < nx && 2 * j + 2 < ny) h[(2 * j + 2) * nx + ci] += (float)0.5 * v;
        }
    }
}

static int mg2d_dense_solve_f32(size_t nx, size_t ny, const float* rhs, float* x) {
    const size_t n = nx * ny;
    if (n > 256) return -1;
    float* M = (float*)malloc((n + 1) * n * sizeof(float));
    if (!M) return -1;
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            const size_t c = j * nx + i;
            for (size_t q = 0; q < n; ++q) M[c * (n + 1) + q] = (float)0;
            M[c * (n + 1) + c] = (float)4;
            if (i > 0) M[c * (n + 1) + (j * nx + (i - 1))] = (float)-1;
            if (i + 1 < nx) M[c * (n + 1) + (j * nx + (i + 1))] = (float)-1;
            if (j > 0) M[c * (n + 1) + ((j - 1) * nx + i)] = (float)-1;
            if (j + 1 < ny) M[c * (n + 1) + ((j + 1) * nx + i)] = (float)-1;
            M[c * (n + 1) + n] = rhs[c];
        }
    }
    for (size_t c = 0; c < n; ++c) {
        size_t p = c;
        for (size_t r = c + 1; r < n; ++r)
            if (fabsf(M[r * (n + 1) + c]) > fabsf(M[p * (n + 1) + c])) p = r;
        for (size_t q = c; q <= n; ++q) {
            float t = M[c * (n + 1) + q];
            M[c * (n + 1) + q] = M[p * (n + 1) + q];
            M[p * (n + 1) + q] = t;
        }
        const float piv = M[c * (n + 1) + c];
        if (piv == (float)0) {
            free(M);
            return -1;
        }
        for (size_t r = c + 1; r < n; ++r) {
            const float g = M[r * (n + 1) + c] / piv;
            if (g == (float)0) continue;
            for (size_t q = c; q <= n; ++q) M[r * (n + 1) + q] -= g * M[c * (n + 1) + q];
        }
    }
    for (size_t i = n; i-- > 0;) {
        float sv = M[i * (n + 1) + n];
        for (size_t j = i + 1; j < n; ++j) sv -= M[i * (n + 1) + j] * x[j];
        x[i] = sv / M[i * (n + 1) + i];
    }
    free(M);
    return 0;
}

static void mg2d_fas_rec_f32(size_t nx, size_t ny, const float* f, float* u,
                             int pre, int post) {
    float* r = (float*)malloc((size_t)nx * ny * sizeof(float));
    float* z = (float*)malloc((size_t)nx * ny * sizeof(float));
    if (!r || !z) {
        free(r);
        free(z);
        return;
    }
    for (int s = 0; s < pre; ++s) {
        mg2d_gs_sweep_f32(nx, ny, f, u, 0);
        mg2d_gs_sweep_f32(nx, ny, f, u, 1);
    }
    mg2d_res_op_f32(nx, ny, f, u, r);
    const size_t nxc = (nx + 1) / 2;
    const size_t nyc = (ny + 1) / 2;
    if (nxc < 2 || nyc < 2) {
        float* e = (float*)malloc((size_t)nx * ny * sizeof(float));
        if (e) {
            memset(e, 0, (size_t)nx * ny * sizeof(float));
            if (mg2d_dense_solve_f32(nx, ny, r, e) == 0)
                for (size_t i = 0; i < (size_t)nx * ny; ++i) u[i] += e[i];
        }
        free(e);
        free(r);
        free(z);
        return;
    }
    mg2d_restrict_f32(nx, ny, f, nxc, nyc, z); /* z = coarse RHS */
    float* cu = (float*)malloc(nxc * nyc * sizeof(float));
    float* c0 = (float*)malloc(nxc * nyc * sizeof(float));
    if (cu && c0) {
        mg2d_restrict_f32(nx, ny, u, nxc, nyc, c0); /* full-weighting of u */
        memcpy(cu, c0, nxc * nyc * sizeof(float));
        mg2d_fas_rec_f32(nxc, nyc, z, cu, pre, post);
        float* d = c0;
        for (size_t i = 0; i < nxc * nyc; ++i) d[i] = cu[i] - c0[i];
        mg2d_prolong_f32(nx, ny, d, nxc, nyc, z);
        for (size_t i = 0; i < (size_t)nx * ny; ++i) u[i] += z[i];
    }
    free(cu);
    free(c0);
    for (int s = 0; s < post; ++s) {
        mg2d_gs_sweep_f32(nx, ny, f, u, 0);
        mg2d_gs_sweep_f32(nx, ny, f, u, 1);
    }
    free(r);
    free(z);
}

RCRE_NODISCARD int rcre_mg_fas_poisson2d_f32(size_t nx, size_t ny,
                                             const float* f, float* x,
                                             int pre, int post) {
    if (!f || !x || nx < 2 || ny < 2 || pre < 0 || post < 0 || pre + post == 0)
        return -1;
    mg2d_fas_rec_f32(nx, ny, f, x, pre, post);
    return 1;
}

/* Classic residual-correction V-cycle, F32 twin (same structure as the
 * F64 version; residual monitoring happens in double in the public
 * wrapper). */
static void mg2d_vcycle_rec_f32(size_t nx, size_t ny, const float* f, float* u,
                                int pre, int post) {
    float* r = (float*)malloc((size_t)nx * ny * sizeof(float));
    if (!r) return; /* OOM: no correction at this level */
    for (int s = 0; s < pre; ++s) {
        mg2d_gs_sweep_f32(nx, ny, f, u, 0);
        mg2d_gs_sweep_f32(nx, ny, f, u, 1);
    }
    mg2d_res_op_f32(nx, ny, f, u, r);
    const size_t nxc = (nx + 1) / 2;
    const size_t nyc = (ny + 1) / 2;
    if (nxc < 2 || nyc < 2) {
        float* e = (float*)malloc((size_t)nx * ny * sizeof(float));
        if (e) {
            memset(e, 0, (size_t)nx * ny * sizeof(float));
            if (mg2d_dense_solve_f32(nx, ny, r, e) == 0)
                for (size_t i = 0; i < (size_t)nx * ny; ++i) u[i] += e[i];
        }
        free(e);
        free(r);
        return;
    }
    float* rc = (float*)malloc(nxc * nyc * sizeof(float));
    float* uc = (float*)calloc(nxc * nyc, sizeof(float));
    if (rc && uc) {
        mg2d_restrict_f32(nx, ny, r, nxc, nyc, rc); /* rc = R r */
        mg2d_vcycle_rec_f32(nxc, nyc, rc, uc, pre, post);
        float* pd = (float*)calloc((size_t)nx * ny, sizeof(float));
        if (pd) {
            mg2d_prolong_f32(nx, ny, uc, nxc, nyc, pd); /* pd = P uc */
            for (size_t i = 0; i < (size_t)nx * ny; ++i) u[i] += pd[i];
        }
        free(pd);
    }
    free(rc);
    free(uc);
    for (int s = 0; s < post; ++s) {
        mg2d_gs_sweep_f32(nx, ny, f, u, 0);
        mg2d_gs_sweep_f32(nx, ny, f, u, 1);
    }
    free(r);
}

RCRE_NODISCARD int rcre_mg_vcycle_poisson2d_f32(size_t nx, size_t ny,
                                                const float* f, float* x,
                                                int pre, int post) {
    if (!f || !x || nx < 2 || ny < 2 || pre < 0 || post < 0 || pre + post == 0)
        return -1;
    mg2d_vcycle_rec_f32(nx, ny, f, x, pre, post);
    return 1;
}

RCRE_NODISCARD int rcre_mg_solve_poisson2d_f32(size_t nx, size_t ny,
                                               const float* f, float* x,
                                               double tol, int max_cycles,
                                               RCRE_KrylovCallback cb, void* user) {
    if (!f || !x || nx < 2 || ny < 2) return -1;
    const size_t n = nx * ny;
    float* r = (float*)malloc(n * sizeof(float));
    if (!r) return -1;
    double nf = 0.0;
    for (size_t i = 0; i < n; ++i) nf += (double)f[i] * (double)f[i];
    const double ref = nf > 0.0 ? sqrt(nf) : 1.0;
    mg2d_res_op_f32(nx, ny, f, x, r);
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += (double)r[i] * (double)r[i];
    double rel = sqrt(s) / ref;
    if (cb) cb(0, rel, user);
    if (rel <= tol) {
        free(r);
        return 0;
    }
    int cycles = 0;
    int ok = 0;
    while (cycles < max_cycles) {
        ++cycles;
        if (rcre_mg_vcycle_poisson2d_f32(nx, ny, f, x, 2, 2) < 0) break;
        mg2d_res_op_f32(nx, ny, f, x, r);
        s = 0.0;
        for (size_t i = 0; i < n; ++i) s += (double)r[i] * (double)r[i];
        rel = sqrt(s) / ref;
        if (cb) cb((size_t)cycles, rel, user);
        if (rel <= tol) {
            ok = 1;
            break;
        }
    }
    free(r);
    return ok ? cycles : -1;
}
