/* RCRE H7 — N-body Barnes-Hut gravity (2.4.0)
 *
 * 2D Newtonian gravity with the Barnes-Hut tree approximation:
 *   - array-based quadtree (arena-friendly; nodes are appended in a
 *     fixed order — the construction is a pure function of the body
 *     ordering, hence deterministic);
 *   - node mass + center of mass (COM) aggregated bottom-up;
 *   - opening criterion s/d > theta (plus a d ~ 0 guard that always
 *     opens internal nodes);
 *   - softening: r2 = dx^2 + dy^2 + soft^2 (avoids the close-encounter
 *     singularities that make the 10k-step energy-drift bar meaningless
 *     for unsoftened random data);
 *   - force on each body = an independent tree walk => the body loop is
 *     slab-parallel (shared read-only tree) and bit-identical across
 *     pool sizes (the determinism policy);
 *   - velocity Verlet (symplectic) step for the energy-drift gate:
 *     v += (h/2) a^k;  x += h v;  a^{k+1} = F(x^{k+1});  v += (h/2) a^{k+1}.
 *
 * Reference energy (the drift diagnostic) is the EXACT O(n^2) pair sum
 * (deterministic, fixed order).
 */
#include "rcre.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================== */
/* Quadtree                                                                    */
/* ========================================================================== */

typedef struct {
    double cx, cy;   /* cell center */
    double half;     /* cell half-size */
    double m;        /* total mass */
    double mx, my;   /* COM position */
    int body;        /* leaf body index, or -1 */
    int child[4];    /* child node indices, or -1 (SW, NW, SE, NE) */
} RCRE_NbNode;

typedef struct {
    RCRE_NbNode* nodes;
    size_t cap;
    size_t n;
    double G;
    double theta;
    double soft2;
    const double* x;
    const double* y;
    const double* m;
    int nbody;
    int max_depth;
} RCRE_NbTree;

static int node_alloc(RCRE_NbTree* t) {
    if (t->n >= t->cap) {
        size_t ncap = t->cap ? t->cap * 2 : 1024;
        RCRE_NbNode* nn =
            (RCRE_NbNode*)realloc(t->nodes, ncap * sizeof(RCRE_NbNode));
        if (!nn) return -1;
        t->nodes = nn;
        t->cap = ncap;
    }
    RCRE_NbNode* nd = &t->nodes[t->n];
    nd->m = 0.0;
    nd->mx = 0.0;
    nd->my = 0.0;
    nd->body = -1;
    nd->child[0] = nd->child[1] = nd->child[2] = nd->child[3] = -1;
    return (int)t->n++;
}

/* which quadrant does (px,py) fall in for the cell centered at (cx,cy)?
 * 0 = SW, 1 = NW, 2 = SE, 3 = NE */
static int quadrant(double px, double py, double cx, double cy) {
    return ((py >= cy) ? 2 : 0) | ((px >= cx) ? 1 : 0);
    /* note: bit0 = x-side (1 if east), bit1 = y-side (2 if north) */
}

static double quadrant_dx(int q) { return (q & 1) ? 0.5 : -0.5; }
static double quadrant_dy(int q) { return (q & 2) ? 0.5 : -0.5; }

/* Simpler, fully deterministic construction: sort-independent quadtree by
 * recursive cell subdivision over the index range.
 *
 * build(range [lo,hi) inside cell (cx,cy,half), depth):
 *   - if empty: -1
 *   - if 1 body: leaf node
 *   - else: create an internal node, recurse into the 4 quadrants with
 *     the index partition of the range, aggregate.
 * The partition is a pure function of the indices and the cell => the
 * tree is a pure function of the body ordering. No insertion order
 * ambiguity at all (this is what makes the tree trivially
 * reproducible — and it is the construction the force walk assumes:
 * a node's children partition the node's bodies exactly).
 */
/* ------------------------------------------------------------------
 * Construction operates on ONE global index permutation `idx` of
 * 0..n-1: every node's bodies are the CONTIGUOUS SLICE idx[lo,hi), and
 * each split partitions that slice IN PLACE (stable, quadrant-major).
 * Recursion never re-indexes the body arrays — it only ever looks at
 * idx[lo..hi).
 *
 * (Measured design bugs in the two earlier drafts: (1) partitioning
 * into a per-node scratch array sized (hi-lo) but scattering with
 * ABSOLUTE positions -> out-of-bounds heap writes for lo>0; (2) passing
 * RELATIVE slice positions into the recursion, which then used them as
 * ABSOLUTE body indices -> 56 of 64 bodies silently vanished. Both are
 * why the tree invariants (bijection bodies<->leaves, mass
 * conservation) are formal tests, not smoke checks.)
 *
 * Determinism: pure function of (bodies, order) — quadrant membership
 * from positions only, ties (px==cx / py==cy) go EAST/NORTH by >=,
 * stable within-quadrant order, fixed aggregation order.
 * ------------------------------------------------------------------ */
static int bh_build_slice(RCRE_NbTree* t, int* idx, int lo, int hi,
                          double cx, double cy, double half, int depth) {
    if (lo >= hi) return -1;
    int node = node_alloc(t);
    if (node < 0) return -1;
    RCRE_NbNode* nd = &t->nodes[node];
    nd->cx = cx;
    nd->cy = cy;
    nd->half = half;
    if (hi - lo == 1) {
        const int b = idx[lo];
        nd->body = b;
        nd->m = t->m[b];
        nd->mx = t->x[b];
        nd->my = t->y[b];
        return node;
    }
    if (depth >= t->max_depth) {
        /* aggregation leaf: many bodies, one COM (the walk uses the
         * point-mass approximation for this node — exact for external
         * bodies; the contained bodies lose pair accuracy at this
         * scale. max_depth is generous, so this is a safety valve, not
         * a normal path). */
        double m = 0.0, mx = 0.0, my = 0.0;
        for (int i = lo; i < hi; ++i) {
            const int b = idx[i];
            m += t->m[b];
            mx += t->m[b] * t->x[b];
            my += t->m[b] * t->y[b];
        }
        nd->m = m;
        nd->mx = m > 0.0 ? mx / m : 0.0;
        nd->my = m > 0.0 ? my / m : 0.0;
        return node;
    }
    /* In-place stable partition of idx[lo,hi) by quadrant. */
    int cnt[4] = { 0, 0, 0, 0 };
    for (int i = lo; i < hi; ++i) {
        const int b = idx[i];
        ++cnt[quadrant(t->x[b], t->y[b], cx, cy)];
    }
    int start[4];
    start[0] = lo;
    for (int q = 1; q < 4; ++q) start[q] = start[q - 1] + cnt[q - 1];
    int* tmp = (int*)malloc((size_t)(hi - lo) * sizeof(int));
    if (!tmp) return -1;
    int pos[4] = { start[0], start[1], start[2], start[3] };
    for (int i = lo; i < hi; ++i) {
        const int b = idx[i];
        /* `pos` holds ABSOLUTE slice positions (they become the child
         * ranges), but `tmp` is sized (hi-lo) — index it RELATIVE.
         * (Third and final member of the absolute/relative bug family
         * measured in this file's history.) */
        const int p = pos[quadrant(t->x[b], t->y[b], cx, cy)]++;
        tmp[p - lo] = b;
    }
    memcpy(&idx[lo], tmp, (size_t)(hi - lo) * sizeof(int));
    free(tmp);

    const double hh = half * 0.5;
    /* NOTE: every write to this node AFTER a recursive call must go
     * through t->nodes[node] (re-derived), never through `nd` — the
     * recursion's node_alloc may realloc t->nodes and dangle `nd`
     * (measured use-after-free at n=100k: the small tests only
     * survived because their reallocs happened not to move). */
    for (int q = 0; q < 4; ++q) t->nodes[node].child[q] = -1;
    int oom = 0;
    for (int q = 0; q < 4; ++q) {
        if (cnt[q] == 0) continue;
        const double ccx = cx + quadrant_dx(q) * half;
        const double ccy = cy + quadrant_dy(q) * half;
        const int c = bh_build_slice(t, idx, start[q], start[q] + cnt[q],
                                     ccx, ccy, hh, depth + 1);
        t->nodes[node].child[q] = c; /* re-derive after recursion */
        if (c < 0) oom = 1;
    }
    if (oom) return -1;
    /* aggregate bottom-up (children were just built) */
    double m = 0.0, mx = 0.0, my = 0.0;
    for (int q = 0; q < 4; ++q) {
        const int cidx = t->nodes[node].child[q];
        if (cidx < 0) continue;
        RCRE_NbNode* c = &t->nodes[cidx];
        m += c->m;
        mx += c->m * c->mx;
        my += c->m * c->my;
    }
    RCRE_NbNode* self = &t->nodes[node]; /* re-derive (realloc may have
                                            moved the array) */
    self->m = m;
    self->mx = m > 0.0 ? mx / m : 0.0;
    self->my = m > 0.0 ? my / m : 0.0;
    return node;
}

/* build the whole tree over bodies [0,n) in a bounding cell */
static int bh_build(RCRE_NbTree* t) {
    if (t->nbody < 1) return 0;
    double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
    for (int i = 0; i < t->nbody; ++i) {
        if (t->x[i] < xmin) xmin = t->x[i];
        if (t->x[i] > xmax) xmax = t->x[i];
        if (t->y[i] < ymin) ymin = t->y[i];
        if (t->y[i] > ymax) ymax = t->y[i];
    }
    double cx = 0.5 * (xmin + xmax), cy = 0.5 * (ymin + ymax);
    double half = 0.5 * fmax(xmax - xmin, ymax - ymin);
    if (half <= 0.0) half = 1.0; /* degenerate (all bodies co-located) */
    int* idx = (int*)malloc((size_t)t->nbody * sizeof(int));
    if (!idx) return -1;
    for (int i = 0; i < t->nbody; ++i) idx[i] = i;
    const int root = bh_build_slice(t, idx, 0, t->nbody, cx, cy, half, 0);
    free(idx);
    return root;
}

/* ========================================================================== */
/* Force walk                                                                  */
/* ========================================================================== */

typedef struct {
    const RCRE_NbTree* t;
    int i;          /* the body being forced */
    double px, py;
    double m;
    double fx, fy;
} RCRE_NbWalk;

static void nb_walk(RCRE_NbWalk* w, int node) {
    if (node < 0) return;
    const RCRE_NbTree* t = w->t;
    const RCRE_NbNode* nd = &t->nodes[node];
    if (nd->m <= 0.0) return;
    const double dx = nd->mx - w->px;
    const double dy = nd->my - w->py;
    const double d2 = dx * dx + dy * dy;
    const double s = nd->half;
    /* opening: internal node AND (s/d > theta OR the body is ~at the
     * COM) */
    int internal = nd->child[0] >= 0 || nd->child[1] >= 0 ||
                   nd->child[2] >= 0 || nd->child[3] >= 0;
    if (internal) {
        const double d = sqrt(d2);
        /* open on the theta criterion — and ALWAYS when the forced body
         * lies inside this cell: otherwise a large-theta walk would
         * apply the node's point-mass approximation to itself (a
         * self-interaction, which is wrong). */
        const int inside = (fabs(w->px - nd->cx) <= s) &&
                           (fabs(w->py - nd->cy) <= s);
        if (inside || (d > 0.0 ? s / d > t->theta : 1)) {
            for (int q = 0; q < 4; ++q) nb_walk(w, nd->child[q]);
            return;
        }
    } else if (nd->body == w->i) {
        return; /* self: skip */
    }
    /* point-mass (multipole-0) approximation:
     * F = G m1 m2 (dx, dy) / r^3   (measured bug: an r^-4 coefficient
     * was EXACT at unit separation, which is how it survived the first
     * 2-body check) */
    const double r2 = d2 + t->soft2;
    const double f = t->G * w->m * nd->m / (r2 * sqrt(r2));
    w->fx += f * dx;
    w->fy += f * dy;
}

/* one body's force: a single deterministic walk over the shared tree */
static void nb_force_one(const RCRE_NbTree* t, int i, double* fx, double* fy) {
    RCRE_NbWalk w = { t, i, t->x[i], t->y[i], t->m[i], 0.0, 0.0 };
    nb_walk(&w, 0); /* the root is always node 0 */
    *fx = w.fx;
    *fy = w.fy;
}

/* slab task: arg = tree base (READ-ONLY), task_id = body index */
typedef struct {
    const RCRE_NbTree* tree;
    double* fx;
    double* fy;
} RCRE_NbTask;

static void nb_force_task(void* arg, size_t i) {
    const RCRE_NbTask* tk = &((const RCRE_NbTask*)arg)[i];
    nb_force_one(tk->tree, (int)i, &tk->fx[i], &tk->fy[i]);
}

/* ========================================================================== */
/* Public API                                                                  */
/* ========================================================================== */

RCRE_NODISCARD int rcre_nb_bh_force_f64(const double* x, const double* y,
                                        const double* m, int n, double G,
                                        double theta, double soft,
                                        RCRE_ThreadPool* pool, double* fx,
                                        double* fy) {
    if (!x || !y || !m || !fx || !fy || n < 1 || G <= 0.0 || theta <= 0.0 ||
        theta >= 1.0)
        return -1;
    RCRE_NbTree t;
    memset(&t, 0, sizeof(t));
    t.G = G;
    t.theta = theta;
    t.soft2 = soft * soft;
    t.x = x;
    t.y = y;
    t.m = m;
    t.nbody = n;
    t.max_depth = 24;
    if (bh_build(&t) < 0) {
        free(t.nodes);
        return -1;
    }
    if (n > 64) {
        RCRE_NbTask* tk = (RCRE_NbTask*)malloc((size_t)n * sizeof(RCRE_NbTask));
        if (!tk) {
            free(t.nodes);
            return -1;
        }
        for (int i = 0; i < n; ++i) {
            tk[i].tree = &t;
            tk[i].fx = fx;
            tk[i].fy = fy;
        }
        if (pool)
            rcre_threadpool_parallel_for(pool, (size_t)n, nb_force_task, tk);
        else
            for (size_t i = 0; i < (size_t)n; ++i) nb_force_task(tk, i);
        free(tk);
    } else {
        for (int i = 0; i < n; ++i) nb_force_one(&t, i, &fx[i], &fy[i]);
    }
    free(t.nodes);
    return 0;
}

RCRE_NODISCARD int rcre_nb_verlet_step_f64(double* x, double* y,
                                           double* vx, double* vy,
                                           const double* m, int n, double G,
                                           double theta, double soft,
                                           double h, RCRE_ThreadPool* pool) {
    if (!x || !y || !vx || !vy || !m || n < 1 || G <= 0.0 || h <= 0.0)
        return -1;
    double* ax = (double*)malloc((size_t)n * sizeof(double));
    double* ay = (double*)malloc((size_t)n * sizeof(double));
    double* ax1 = (double*)malloc((size_t)n * sizeof(double));
    double* ay1 = (double*)malloc((size_t)n * sizeof(double));
    if (!ax || !ay || !ax1 || !ay1) {
        free(ax); free(ay); free(ax1); free(ay1);
        return -1;
    }
    /* a^k */
    int rc = rcre_nb_bh_force_f64(x, y, m, n, G, theta, soft, pool, ax, ay);
    if (rc == 0)
        for (int i = 0; i < n; ++i) {
            /* F -> a: divide by the body's own mass.
             * (Measured bug: the first draft applied the raw force as an
             * acceleration. Every mass-1 test hid it — a 100:0.01 two-body
             * system then showed a 400% "energy drift" that was actually a
             * mass-weighting error, h-independent because it is not an
             * integrator error at all.) */
            vx[i] += 0.5 * h * ax[i] / m[i];
            vy[i] += 0.5 * h * ay[i] / m[i];
            x[i] += h * vx[i];
            y[i] += h * vy[i];
        }
    if (rc == 0)
        rc = rcre_nb_bh_force_f64(x, y, m, n, G, theta, soft, pool, ax1, ay1);
    if (rc == 0)
        for (int i = 0; i < n; ++i) {
            vx[i] += 0.5 * h * ax1[i] / m[i];
            vy[i] += 0.5 * h * ay1[i] / m[i];
        }
    free(ax); free(ay); free(ax1); free(ay1);
    return rc;
}

/* Exact O(n^2) total energy (the drift diagnostic; fixed order):
 * E = 1/2 sum m v^2 - sum_{i<j} G m_i m_j / sqrt(r^2 + soft^2) */
RCRE_NODISCARD double rcre_nb_energy_f64(const double* x, const double* y,
                                         const double* vx, const double* vy,
                                         const double* m, int n, double G,
                                         double soft) {
    double e = 0.0;
    for (int i = 0; i < n; ++i)
        e += 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i]);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const double dx = x[i] - x[j], dy = y[i] - y[j];
            const double r = sqrt(dx * dx + dy * dy + soft * soft);
            e -= G * m[i] * m[j] / r;
        }
    return e;
}
