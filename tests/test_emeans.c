/**
 * @file test_emeans.c
 * @brief Unit tests for the e-means operator (src/ml/e_means.c).
 *
 * Coverage (the full dev harness with the Python oracle lives in
 * projects/rcre_emeans — this core suite is self-contained on checked-in
 * fixtures in tests/emeans_fixtures/, generated once by that project's
 * `make refs`):
 *   T1/T2  scaling laws, recipe resolution, alpha schedule
 *          (oracle grid: expected values from the independent Python
 *          implementation of the shipped reference formulas)
 *   T3/T4  strict state-machine bit parity vs the oracle (0 ULP on every
 *          post-update centroid/mass, finalization and FP64 pass sums)
 *   T5     assignment: true argmin + first-index tie break
 *   T6     determinism: run-to-run and pool-size bit identity; serial vs
 *          pooled quality equivalence (MSE <= 0.5%, labels >= 99%)
 *   T7/T8  quality: two-blob local-minimum trap; full-run non-degeneracy
 *   T10    error paths and input validation
 */
#include "rcre.h"
#include "e_means_internal.h"   /* EMState, em_* recipe helpers (test-visible) */
#include "test_framework.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Oracle-generated fixtures (checked in; see projects/rcre_emeans/ref/) */
#include "emeans_fixtures/recipes.h"
#include "emeans_fixtures/strict_norev.h"
#include "emeans_fixtures/strict_revive.h"

#define EM_FIX_DIR "tests/emeans_fixtures"

/* ========================================================================= */
/* Helpers (mirrored from projects/rcre_emeans/tests/em_util.h)              */
/* ========================================================================= */

static void *em_read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    void* buf = malloc((size_t)sz);
    if (buf && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    if (out_size) *out_size = (size_t)sz;
    return buf;
}

static RCRE_Tensor em_tensor2(RCRE_Arena* arena, size_t n, size_t d,
                              RCRE_DType dtype) {
    size_t shape[2] = { n, d };
    return rcre_tensor_create(shape, 2, dtype, RCRE_LAYOUT_TABULAR, arena);
}

/* Deterministic synthetic blobs (splitmix64 centers, no distance ties). */
static void em_make_blobs(float* X, size_t n, size_t d, size_t n_blobs,
                          uint64_t seed) {
    uint64_t rs = seed;
    float* centers = malloc(n_blobs * d * sizeof(float));
    for (size_t i = 0; i < n_blobs * d; i++) {
        uint64_t r = rcre_splitmix64(&rs);
        float u = (float)((r >> 11) * (1.0 / 9007199254740992.0));
        uint64_t r2 = rcre_splitmix64(&rs);
        float v = (float)((r2 >> 11) * (1.0 / 9007199254740992.0));
        centers[i] = (float)sqrt(-2.0 * log((double)u + 1e-12))
                     * (float)cos(2.0 * M_PI * v) * 6.0f;
    }
    for (size_t i = 0; i < n; i++) {
        uint64_t r = rcre_splitmix64(&rs);
        size_t b = (size_t)(r % n_blobs);
        for (size_t dd = 0; dd < d; dd++) {
            uint64_t r1 = rcre_splitmix64(&rs);
            float u = (float)((r1 >> 11) * (1.0 / 9007199254740992.0));
            uint64_t r2 = rcre_splitmix64(&rs);
            float v = (float)((r2 >> 11) * (1.0 / 9007199254740992.0));
            float g = (float)sqrt(-2.0 * log((double)u + 1e-12))
                      * (float)cos(2.0 * M_PI * v) * 0.35f;
            X[i * d + dd] = centers[b * d + dd] + g;
        }
    }
    free(centers);
}

/* ========================================================================= */
/* T1/T2 — scaling laws, recipe resolution, alpha schedule                    */
/* ========================================================================= */

static void test_recipe_round_pow2(void) {
    ASSERT_UINT_EQ(1, em_round_pow2(0.4));
    ASSERT_UINT_EQ(1, em_round_pow2(1.0));
    ASSERT_UINT_EQ(2, em_round_pow2(1.5));
    ASSERT_UINT_EQ(2, em_round_pow2(2.0));
    ASSERT_UINT_EQ(2, em_round_pow2(2.8));       /* below sqrt(8) ~ 2.8284 */
    ASSERT_UINT_EQ(4, em_round_pow2(2.83));      /* above sqrt(8)          */
    ASSERT_UINT_EQ(4, em_round_pow2(5.0));
    ASSERT_UINT_EQ(8, em_round_pow2(5.7));       /* above sqrt(32)         */
    ASSERT_UINT_EQ(4, em_round_pow2(5.6));       /* below sqrt(32)         */
    ASSERT_UINT_EQ(1024, em_round_pow2(1024.0));
    ASSERT_UINT_EQ(2048, em_round_pow2(1500.0));
}

static void test_recipe_epoch_remainder(void) {
    ASSERT_UINT_EQ(0, em_epoch_remainder(5.0f, 1000));   /* integer T  */
    ASSERT_UINT_EQ(500, em_epoch_remainder(1.5f, 1000));
    ASSERT_UINT_EQ(250, em_epoch_remainder(1.25f, 1000));
    ASSERT_UINT_EQ(0, em_epoch_remainder(1.125f, 4));    /* 0.5 -> even 0 */
    ASSERT_UINT_EQ(2, em_epoch_remainder(1.375f, 4));    /* 1.5 -> even 2 */
    ASSERT_UINT_EQ(2, em_epoch_remainder(1.625f, 4));    /* 2.5 -> even 2 */
    ASSERT_UINT_EQ(4, em_epoch_remainder(1.875f, 4));    /* 3.5 -> even 4 */
}

static void test_recipe_scheduled_alpha(void) {
    /* t=0 -> peak; t=mid -> midpoint; t=M-1 -> ~0.1*peak (not exact) */
    double a0 = 0.648;
    ASSERT_DOUBLE_NEAR(0.648, em_scheduled_alpha(a0, 0, 80), 1e-15);
    double mid = em_scheduled_alpha(a0, 40, 80);
    ASSERT_DOUBLE_NEAR(a0 * (0.1 + 0.9 * 0.5 * (1.0 + cos(M_PI * 0.5))),
                       mid, 1e-15);
    double last = em_scheduled_alpha(a0, 79, 80);
    ASSERT_TRUE(last > a0 * 0.1 && last < a0 * 0.11);
    ASSERT_DOUBLE_NEAR(a0, em_scheduled_alpha(a0, 0, 0), 1e-15); /* total 0 */
}

static void test_recipe_defaults(void) {
    /* n=1000, d=128 (> 16 -> throughput), k=16, T=5: hand-anchored cells */
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);
    RCRE_Tensor Xt = em_tensor2(arena, 1000, 128, RCRE_DTYPE_F32);
    ASSERT_NOT_NULL(Xt.data);

    RCRE_EMMeansRecipe r;
    int rc = rcre_ml_emeans_recipe(&Xt, 16, NULL, &r);
    ASSERT_INT_EQ(RCRE_OK, rc);
    /* raw ~ 80.5 -> pow2 64 (hand anchor); formula recompute below */
    ASSERT_UINT_EQ(64, r.batch_size);
    double raw = 0.035134 * pow(16.0, 0.134749) * pow(1000.0, 0.876473)
               * pow(5.0, 0.813344);
    ASSERT_UINT_EQ(em_round_pow2(raw), r.batch_size);
    double a0 = 0.8244 * pow(16.0, -0.4617) * pow((double)r.batch_size, 0.9003)
              * pow(1000.0, -0.4498) * pow(5.0, -0.1057);
    ASSERT_DOUBLE_NEAR(a0, (double)r.alpha0_full, 1e-12);
    ASSERT_UINT_EQ(5, r.n_epochs);
    ASSERT_UINT_EQ(5 * ((1000 + r.batch_size - 1) / r.batch_size),
                   r.total_steps);
    ASSERT_FLOAT_NEAR(0.1f, r.unused_threshold, 1e-7f);  /* d=128 > 2 */
    ASSERT_UINT_EQ(1234, r.seed);                        /* NULL params */

    /* d = 2: quality law auto + revival auto-off */
    RCRE_Tensor X2 = em_tensor2(arena, 1000, 2, RCRE_DTYPE_F32);
    RCRE_EMMeansRecipe r2;
    ASSERT_INT_EQ(RCRE_OK, rcre_ml_emeans_recipe(&X2, 8, NULL, &r2));
    double raw2 = 0.00439175 * pow(8.0, 0.134749) * pow(1000.0, 0.876473)
                 * pow(5.0, 0.813344);
    size_t B2 = em_round_pow2(raw2);
    if (B2 > 1000 / 2) B2 = 1000 / 2;
    ASSERT_UINT_EQ(B2, r2.batch_size);
    ASSERT_FLOAT_NEAR(0.0f, r2.unused_threshold, 0.0f);  /* d <= 2 */

    /* manual batch keeps the step law at that batch */
    RCRE_EMMeansParams p;
    memset(&p, 0, sizeof(p));
    p.batch_size = 128;
    RCRE_EMMeansRecipe r3;
    ASSERT_INT_EQ(RCRE_OK, rcre_ml_emeans_recipe(&Xt, 16, &p, &r3));
    ASSERT_UINT_EQ(128, r3.batch_size);
    ASSERT_INT_EQ(-1, r3.batch_law);

    rcre_arena_destroy(arena);
}

/* Oracle grid: independent Python implementation of the shipped formulas */
static void test_recipe_oracle_grid(void) {
    for (int i = 0; i < EM_N_RECIPES; i++) {
        const struct em_expect_recipe* e = &EM_RECIPES[i];
        /* shape-only tensor: the recipe needs n and d, not the data */
        RCRE_Tensor X;
        memset(&X, 0, sizeof(X));
        X.shape[0] = e->n;
        X.shape[1] = e->d;
        X.ndim = 2;
        X.dtype = RCRE_DTYPE_F32;
        RCRE_EMMeansParams p = rcre_ml_emeans_params_default();
        p.nepochs = e->T;
        RCRE_EMMeansRecipe r;
        int rc = rcre_ml_emeans_recipe(&X, e->k, &p, &r);
        ASSERT_INT_EQ(RCRE_OK, rc);
        ASSERT_UINT_EQ(e->B, r.batch_size);
        ASSERT_UINT_EQ(e->M, r.total_steps);
        ASSERT_UINT_EQ(e->n_epochs, r.n_epochs);
        ASSERT_INT_EQ(e->batch_law, r.batch_law);
        for (size_t j = 0; j < e->n_epochs; j++)
            ASSERT_UINT_EQ(e->rows[j], r.epoch_rows[j]);
        ASSERT_FLOAT_NEAR((float)e->alpha0_f64, r.alpha0, 0.0f);
        ASSERT_DOUBLE_NEAR(e->alpha0_f64, r.alpha0_full, 1e-12);
        ASSERT_FLOAT_NEAR(e->tau, r.unused_threshold, 0.0f);
    }
}

/* ========================================================================= */
/* T3/T4 — strict state-machine bit parity vs the oracle                     */
/* ========================================================================= */

/* Replay one strict case: drive the C state machine with the oracle's
 * injected (sums, counts, U) streams; every post-update model state and
 * the finalization output must be bit-identical (0 ULP). */
static void run_strict_case(const struct em_stream_meta* m,
                            const char* bin_path, const char* x_path) {
    size_t xsz = 0;
    float* X = (float*)em_read_file(x_path, &xsz);
    ASSERT_NOT_NULL(X);
    ASSERT_UINT_EQ(m->n * m->d, xsz / sizeof(float));

    size_t bs = 0;
    uint8_t* buf = (uint8_t*)em_read_file(bin_path, &bs);
    ASSERT_NOT_NULL(buf);

    /* Layout: init c | updates | states | U checks | final */
    size_t kd = m->k * m->d;
    size_t off = 0;
    float* init_c = (float*)(buf + off);
    off += kd * sizeof(float);

    size_t* u_off = malloc(m->n_updates * sizeof(size_t));
    for (size_t i = 0; i < m->n_updates; i++) {
        u_off[i] = off;
        uint32_t mm;
        memcpy(&mm, buf + off, 4); off += 4;
        ASSERT_UINT_EQ(m->update_rows[i], mm);
        off += (size_t)mm * sizeof(int32_t)
             + kd * sizeof(float)
             + m->k * sizeof(float);
    }
    size_t state_off = off;
    off += m->n_updates * (kd * sizeof(float) + m->k * sizeof(float));
    size_t U_off = off;
    off += m->n_U_checks * m->k * sizeof(double);
    size_t final_off = off;
    size_t expect_total = final_off
        + kd * sizeof(float) + m->k * sizeof(float)
        + kd * sizeof(double) + m->k * sizeof(double);
    ASSERT_UINT_EQ(expect_total, bs);

    /* Build the state (serial, no pool). */
    RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);
    RCRE_Tensor Xt = em_tensor2(arena, m->n, m->d, RCRE_DTYPE_F32);
    memcpy(Xt.data, X, m->n * m->d * sizeof(float));
    RCRE_Tensor Ct = em_tensor2(arena, m->k, m->d, RCRE_DTYPE_F32);
    memcpy(Ct.data, init_c, kd * sizeof(float));

    RCRE_EMMeansParams p;
    memset(&p, 0, sizeof(p));
    p.nepochs = m->T;
    p.batch_size = m->B;
    p.alpha0 = (float)m->alpha0_full; /* fixtures use F32-exact values */
    p.unused_threshold = m->tau;
    p.shuffle = false;
    p.init_centroids = &Ct;
    p.seed = m->seed;
    p.capture_labels = false;

    EMState st;
    int rc = em_state_init(&st, &Xt, m->k, &p, arena, NULL);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_MEM_EQ(init_c, st.c, kd * sizeof(float));   /* model start exact */

    /* Inject the oracle's revival U stream (no-op when tau == 0). */
    if (m->n_U_checks > 0) {
        st.inject_U = (const double*)(buf + U_off);
        st.inject_U_len = m->n_U_checks * m->k;
        st.inject_U_pos = 0;
    }

    /* Final-epoch update range (same derivation as the oracle). */
    size_t rows_arr[64];
    size_t ne = 0;
    {
        double T = (double)m->T;
        size_t w = (size_t)T;
        size_t rem = em_epoch_remainder(m->T, m->n);
        if (w == 0) { rows_arr[0] = rem; ne = 1; }
        else {
            for (size_t i = 0; i + 1 < w; i++) rows_arr[ne++] = m->n;
            if (rem > 0) rows_arr[ne++] = rem;
            rows_arr[ne++] = m->n;
        }
    }
    size_t final_start_updates = 0;
    for (size_t e = 0; e + 1 < ne; e++)
        final_start_updates += (rows_arr[e] + m->B - 1) / m->B;

    /* Replay, comparing every post-update state. */
    int state_checks = 0;
    for (size_t t = 0; t < m->n_updates; t++) {
        st.last_epoch = (t >= final_start_updates);
        if (t == final_start_updates && ne > 0) {
            memset(st.pass_sums, 0, kd * sizeof(double));
            memset(st.pass_counts, 0, m->k * sizeof(double));
            st.obj_last = 0.0;
            st.obj_last_rows = 0;
        }
        uint8_t* up = buf + u_off[t];
        uint32_t mm;
        memcpy(&mm, up, 4);
        float* sums = (float*)(up + 4 + (size_t)mm * sizeof(int32_t));
        float* counts = (float*)(sums + kd);

        rc = em_state_update(&st, sums, counts, 0.0);
        ASSERT_INT_EQ(RCRE_OK, rc);

        const uint8_t* exp = buf + state_off
            + t * (kd * sizeof(float) + m->k * sizeof(float));
        ASSERT_MEM_EQ(exp, st.c, kd * sizeof(float));      /* 0 ULP, c */
        ASSERT_MEM_EQ(exp + kd * sizeof(float), st.N,
                      m->k * sizeof(float));               /* 0 ULP, N */
        state_checks += 2;
    }

    /* Finalization (T4): FP64 pass sums + FP32 division. */
    rc = em_state_finalize(&st);
    ASSERT_INT_EQ(RCRE_OK, rc);
    const uint8_t* fin = buf + final_off;
    ASSERT_MEM_EQ(fin, st.c, kd * sizeof(float));
    ASSERT_MEM_EQ(fin + kd * sizeof(float), st.N, m->k * sizeof(float));
    ASSERT_MEM_EQ(fin + kd * sizeof(float) + m->k * sizeof(float),
                  st.pass_sums, kd * sizeof(double));              /* FP64 */
    ASSERT_MEM_EQ(fin + kd * sizeof(float) + m->k * sizeof(float)
                  + kd * sizeof(double),
                  st.pass_counts, m->k * sizeof(double));          /* FP64 */

    printf("  [%s] %zu updates, %d state checks, 4 final checks, "
           "%zu U-checks, splits=%d\n",
           bin_path, m->n_updates, state_checks, m->n_U_checks, st.n_splits);

    free(u_off);
    free(buf);
    free(X);
    rcre_arena_destroy(arena);
}

static void test_parity_strict_norev(void) {
    char path[256];
    snprintf(path, sizeof(path), EM_FIX_DIR "/strict_norev.bin");
    char xpath[256];
    snprintf(xpath, sizeof(xpath), EM_FIX_DIR "/X_strict_norev.f32");
    run_strict_case(&EM_STREAM_STRICT_NOREV, path, xpath);
}

static void test_parity_strict_revive(void) {
    /* The fixture guarantees splits occurred and the Gumbel key margin is
     * away from the libm-log tie zone (the oracle prints the margin). */
    char path[256];
    snprintf(path, sizeof(path), EM_FIX_DIR "/strict_revive.bin");
    char xpath[256];
    snprintf(xpath, sizeof(xpath), EM_FIX_DIR "/X_strict_revive.f32");
    run_strict_case(&EM_STREAM_STRICT_REVIVE, path, xpath);
}

/* ========================================================================= */
/* T5 — assignment correctness                                                */
/* ========================================================================= */

static void test_assign_true_argmin(void) {
    const size_t n = 4096, d = 12, k = 96;
    RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_Tensor X = em_tensor2(arena, n, d, RCRE_DTYPE_F32);
    em_make_blobs((float*)X.data, n, d, 40, 991);
    RCRE_Tensor C = em_tensor2(arena, k, d, RCRE_DTYPE_F32);
    em_make_blobs((float*)C.data, k, d, 16, 41); /* arbitrary centroids */
    size_t lshape[2] = { n, 1 };
    RCRE_Tensor Lt = rcre_tensor_create(lshape, 2, RCRE_DTYPE_I32,
                                        RCRE_LAYOUT_TABULAR, arena);

    float mse = -1.0f;
    int rc = rcre_ml_emeans_assign(&X, &C, &Lt, &mse, arena);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_TRUE(mse >= 0.0f);

    const float* Xd = (const float*)X.data;
    const float* Cd = (const float*)C.data;
    const int32_t* Ld = (const int32_t*)Lt.data;

    size_t margin_ok = 0, tie_zone = 0;
    double max_rel_dev = 0.0;
    for (size_t i = 0; i < n; i++) {
        /* F64 recomputation of all k squared distances */
        double best64 = INFINITY, second = INFINITY;
        size_t best64_j = 0, second_j = 0;
        for (size_t j = 0; j < k; j++) {
            double s = 0.0;
            for (size_t dd = 0; dd < d; dd++) {
                double diff = (double)Xd[i * d + dd] - (double)Cd[j * d + dd];
                s += diff * diff;
            }
            if (s < best64) { second = best64; second_j = best64_j;
                              best64 = s; best64_j = j; }
            else if (s < second) { second = s; second_j = j; }
        }
        if (second - best64 > 0.05) {
            /* clear F64 winner (above the F32 accumulation noise floor) */
            ASSERT_UINT_EQ(best64_j, (size_t)Ld[i]);
            margin_ok++;
        } else {
            /* F32-tie zone: the label may be either of the two nearest */
            int32_t l = Ld[i];
            ASSERT_TRUE(l == (int32_t)best64_j || l == (int32_t)second_j);
            tie_zone++;
        }
        /* identity: ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x.c (includes the
         * row-constant term so it matches the F64 direct distance) */
        double xxj = 0.0, ccj = 0.0, dot = 0.0;
        for (size_t dd = 0; dd < d; dd++) {
            xxj += (double)Xd[i * d + dd] * (double)Xd[i * d + dd];
            ccj += (double)Cd[best64_j * d + dd] * (double)Cd[best64_j * d + dd];
            dot += (double)Xd[i * d + dd] * (double)Cd[best64_j * d + dd];
        }
        double via_identity = xxj + ccj - 2.0 * dot;
        if (best64 > 1e-3) {
            double rel = fabs(via_identity - best64) / best64;
            if (rel > max_rel_dev) max_rel_dev = rel;
        }
    }
    ASSERT_TRUE(margin_ok >= n / 2);          /* mostly tie-free fixture */
    ASSERT_TRUE(max_rel_dev < 1e-4);          /* identity within F32 noise */
    printf("  [assign] margin-cleared rows: %zu/%zu (tie zone %zu), "
           "max identity dev %.2e\n", margin_ok, n, tie_zone, max_rel_dev);
    rcre_arena_destroy(arena);
}

static void test_assign_tie_first_index(void) {
    /* Two identical centroids: every row must take cluster 0. */
    const size_t n = 64, d = 4, k = 2;
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    RCRE_Tensor X = em_tensor2(arena, n, d, RCRE_DTYPE_F32);
    for (size_t i = 0; i < n * d; i++)
        ((float*)X.data)[i] = 0.5f;
    RCRE_Tensor C = em_tensor2(arena, k, d, RCRE_DTYPE_F32);
    for (size_t i = 0; i < k * d; i++)
        ((float*)C.data)[i] = 0.25f;
    size_t lshape[2] = { n, 1 };
    RCRE_Tensor L = rcre_tensor_create(lshape, 2, RCRE_DTYPE_I32,
                                       RCRE_LAYOUT_TABULAR, arena);
    float mse = 0.0f;
    ASSERT_INT_EQ(RCRE_OK, rcre_ml_emeans_assign(&X, &C, &L, &mse, arena));
    const int32_t* Ld = (const int32_t*)L.data;
    size_t firsts = 0;
    for (size_t i = 0; i < n; i++) if (Ld[i] == 0) firsts++;
    ASSERT_UINT_EQ(n, firsts);
    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* T6 — determinism                                                           */
/* ========================================================================= */

typedef struct EMRunResult {
    float* centroids;   /* owned, k*d */
    int32_t* labels;    /* owned, n   */
} EMRunResult;

static int do_run(const float* Xdata, size_t n, size_t d, size_t k, float T,
                  RCRE_Arena* arena, RCRE_ThreadPool* pool,
                  EMRunResult* out) {
    size_t xshape[2] = { n, d };
    RCRE_Tensor Xt = rcre_tensor_create(xshape, 2, RCRE_DTYPE_F32,
                                        RCRE_LAYOUT_TABULAR, arena);
    memcpy(Xt.data, Xdata, n * d * sizeof(float));
    RCRE_Tensor C = em_tensor2(arena, k, d, RCRE_DTYPE_F32);
    size_t lshape[2] = { n, 1 };
    RCRE_Tensor L = rcre_tensor_create(lshape, 2, RCRE_DTYPE_I32,
                                       RCRE_LAYOUT_TABULAR, arena);
    RCRE_EMMeansParams p;
    memset(&p, 0, sizeof(p));
    p.nepochs = T;
    p.shuffle = true;
    p.unused_threshold = -1.0f;
    p.seed = 20261115;
    p.capture_labels = true;

    int rc = rcre_ml_emeans_fit_ex(&Xt, k, &p, &C, &L, NULL, NULL, arena, pool);
    if (rc != RCRE_OK) return rc;

    out->centroids = malloc(k * d * sizeof(float));
    out->labels = malloc(n * sizeof(int32_t));
    memcpy(out->centroids, C.data, k * d * sizeof(float));
    memcpy(out->labels, L.data, n * sizeof(int32_t));
    return RCRE_OK;
}

static void test_determinism_run_to_run(void) {
    const size_t n = 8192, d = 24, k = 128;
    float* X = malloc(n * d * sizeof(float));
    em_make_blobs(X, n, d, 32, 7);

    RCRE_Arena* a1 = rcre_arena_create(32 * 1024 * 1024);
    RCRE_Arena* a2 = rcre_arena_create(32 * 1024 * 1024);
    EMRunResult r1, r2;
    ASSERT_INT_EQ(RCRE_OK, do_run(X, n, d, k, 2.0f, a1, NULL, &r1));
    ASSERT_INT_EQ(RCRE_OK, do_run(X, n, d, k, 2.0f, a2, NULL, &r2));
    ASSERT_MEM_EQ(r1.centroids, r2.centroids, k * d * sizeof(float));
    ASSERT_MEM_EQ(r1.labels, r2.labels, n * sizeof(int32_t));
    free(r1.centroids); free(r1.labels);
    free(r2.centroids); free(r2.labels);
    free(X);
    rcre_arena_destroy(a1);
    rcre_arena_destroy(a2);
}

static void test_determinism_pool_size(void) {
    const size_t n = 8192, d = 24, k = 128;
    float* X = malloc(n * d * sizeof(float));
    em_make_blobs(X, n, d, 32, 7);

    RCRE_Arena* a1 = rcre_arena_create(32 * 1024 * 1024);
    RCRE_Arena* a2 = rcre_arena_create(32 * 1024 * 1024);
    RCRE_Arena* a3 = rcre_arena_create(32 * 1024 * 1024);
    RCRE_ThreadPool* p4 = rcre_threadpool_create(4);
    RCRE_ThreadPool* p16 = rcre_threadpool_create(16);
    EMRunResult r1, r2, r3;
    ASSERT_INT_EQ(RCRE_OK, do_run(X, n, d, k, 2.0f, a1, NULL, &r1));
    ASSERT_INT_EQ(RCRE_OK, do_run(X, n, d, k, 2.0f, a2, p4, &r2));
    ASSERT_INT_EQ(RCRE_OK, do_run(X, n, d, k, 2.0f, a3, p16, &r3));
    /* pool(4) and pool(16) bit-identical: the slab partition depends only
     * on the batch size, never on the pool */
    ASSERT_MEM_EQ(r2.centroids, r3.centroids, k * d * sizeof(float));
    ASSERT_MEM_EQ(r2.labels, r3.labels, n * sizeof(int32_t));
    free(r1.centroids); free(r1.labels);
    free(r2.centroids); free(r2.labels);
    free(r3.centroids); free(r3.labels);
    rcre_threadpool_destroy(p4);
    rcre_threadpool_destroy(p16);
    free(X);
    rcre_arena_destroy(a1);
    rcre_arena_destroy(a2);
    rcre_arena_destroy(a3);
}

static void test_determinism_serial_vs_pool(void) {
    const size_t n = 16384, d = 32, k = 256;
    float* X = malloc(n * d * sizeof(float));
    em_make_blobs(X, n, d, 64, 13);

    RCRE_Arena* a1 = rcre_arena_create(64 * 1024 * 1024);
    RCRE_Arena* a2 = rcre_arena_create(64 * 1024 * 1024);
    RCRE_ThreadPool* p18 = rcre_threadpool_create(18);
    EMRunResult rs, rp;
    ASSERT_INT_EQ(RCRE_OK, do_run(X, n, d, k, 2.0f, a1, NULL, &rs));
    ASSERT_INT_EQ(RCRE_OK, do_run(X, n, d, k, 2.0f, a2, p18, &rp));
    rcre_threadpool_destroy(p18);

    /* Quality equivalence: exact final MSEs within 0.5%, labels >= 99%.
     * (The F32 slab partial-sum regrouping differs from the single-slab
     * serial order at the ULP level, and that difference feeds back
     * chaotically through the EMA system — 1 ULP is NOT the contract.) */
    size_t xs[2] = { n, d };
    RCRE_Tensor Xt = rcre_tensor_create(xs, 2, RCRE_DTYPE_F32,
                                        RCRE_LAYOUT_TABULAR, a1);
    memcpy(Xt.data, X, n * d * sizeof(float));
    RCRE_Tensor Cs = em_tensor2(a1, k, d, RCRE_DTYPE_F32);
    RCRE_Tensor Cp = em_tensor2(a1, k, d, RCRE_DTYPE_F32);
    memcpy(Cs.data, rs.centroids, k * d * sizeof(float));
    memcpy(Cp.data, rp.centroids, k * d * sizeof(float));
    size_t ls[2] = { n, 1 };
    RCRE_Tensor Ls = rcre_tensor_create(ls, 2, RCRE_DTYPE_I32,
                                        RCRE_LAYOUT_TABULAR, a1);
    RCRE_Tensor Lp = rcre_tensor_create(ls, 2, RCRE_DTYPE_I32,
                                        RCRE_LAYOUT_TABULAR, a1);
    float mse_s = 0.0f, mse_p = 0.0f;
    ASSERT_INT_EQ(RCRE_OK, rcre_ml_emeans_assign(&Xt, &Cs, &Ls, &mse_s, a1));
    ASSERT_INT_EQ(RCRE_OK, rcre_ml_emeans_assign(&Xt, &Cp, &Lp, &mse_p, a1));
    double rel = 2.0 * fabs((double)mse_s - (double)mse_p)
                 / ((double)mse_s + (double)mse_p);
    size_t agree = 0;
    for (size_t i = 0; i < n; i++)
        if (rs.labels[i] == rp.labels[i]) agree++;
    printf("  [serial vs pool18] MSE %.6f vs %.6f (rel %.4f%%), "
           "training-label agreement %.4f%%\n",
           (double)mse_s, (double)mse_p, 100.0 * rel,
           100.0 * (double)agree / (double)n);
    ASSERT_TRUE(rel <= 0.005);
    ASSERT_TRUE((double)agree >= 0.99 * (double)n);
    free(rs.centroids); free(rs.labels);
    free(rp.centroids); free(rp.labels);
    free(X);
    rcre_arena_destroy(a1);
    rcre_arena_destroy(a2);
}

/* ========================================================================= */
/* T7/T8 — quality                                                            */
/* ========================================================================= */

/* The paper's two-blob local-minimum trap: one small blob (n=200) vs one
 * large blob (n=3000), k=8. Lloyd can strand centroids on the small blob;
 * e-means' EMA + revival escape. */
static void test_quality_two_blob_trap(void) {
    const size_t n_small = 200, n_big = 3000, n = n_small + n_big;
    const size_t d = 2, k = 8;
    RCRE_Arena* arena = rcre_arena_create(32 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    float* X = malloc(n * d * sizeof(float));
    for (size_t i = 0; i < n_small; i++) {
        X[i * 2 + 0] = 0.0f + 0.3f * (float)sin((double)i * 1.7 + 1.0);
        X[i * 2 + 1] = 0.0f + 0.3f * (float)cos((double)i * 1.3 + 2.0);
    }
    for (size_t i = 0; i < n_big; i++) {
        X[(n_small + i) * 2 + 0] = 20.0f + 1.0f * (float)sin((double)i * 1.1 + 3.0);
        X[(n_small + i) * 2 + 1] = -8.0f + 1.0f * (float)cos((double)i * 1.9 + 4.0);
    }

    size_t xshape[2] = { n, d };
    RCRE_Tensor Xt = rcre_tensor_create(xshape, 2, RCRE_DTYPE_F32,
                                        RCRE_LAYOUT_TABULAR, arena);
    memcpy(Xt.data, X, n * d * sizeof(float));
    RCRE_Tensor C = em_tensor2(arena, k, d, RCRE_DTYPE_F32);
    size_t lshape[2] = { n, 1 };
    RCRE_Tensor L = rcre_tensor_create(lshape, 2, RCRE_DTYPE_I32,
                                       RCRE_LAYOUT_TABULAR, arena);

    float moving_mse = 0.0f;
    int n_splits = 0;
    RCRE_EMMeansParams p;
    memset(&p, 0, sizeof(p));
    p.nepochs = 5.0f;
    p.shuffle = true;
    p.unused_threshold = -1.0f;
    p.capture_labels = true;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_fit_ex(&Xt, k, &p, &C, &L, &moving_mse,
                                        &n_splits, arena, NULL));
    float em_final_mse = 0.0f;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_assign(&Xt, &C, &L, &em_final_mse, arena));

    /* how many centroids landed on the large blob (x ~ 20)? */
    const float* Cd = (const float*)C.data;
    int on_big = 0;
    for (size_t j = 0; j < k; j++)
        if (Cd[j * 2 + 0] > 10.0f) on_big++;
    printf("  [trap] e-means centroids on the large blob: %d/%d, "
           "final MSE %.4f\n", on_big, (int)k, (double)em_final_mse);
    ASSERT_TRUE(on_big >= 5); /* paper: 7 of 8 typical; 5 is the floor */

    /* Lloyd baseline: RCRE k-means (k-means++ init, 25 iters), scored
     * with the same exact assignment pass */
    RCRE_Tensor Cl = em_tensor2(arena, k, d, RCRE_DTYPE_F32);
    size_t llshape[2] = { n, 1 };
    RCRE_Tensor Ll = rcre_tensor_create(llshape, 2, RCRE_DTYPE_I32,
                                        RCRE_LAYOUT_TABULAR, arena);
    float lloyd_mse = 0.0f;
    RCRE_KMeansParams kp;
    memset(&kp, 0, sizeof(kp));
    kp.max_iter = 25;
    kp.n_init = 1;
    kp.tol = 1e-6f;
    kp.seed = 42;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_kmeans_fit_ex(&Xt, k, &kp, &Cl, &Ll, &lloyd_mse,
                                        arena));
    float lloyd_final = 0.0f;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_assign(&Xt, &Cl, &Ll, &lloyd_final, arena));
    printf("  [trap] Lloyd-25 MSE %.4f vs e-means-5 MSE %.4f\n",
           (double)lloyd_final, (double)em_final_mse);
    /* Quality must be competitive with Lloyd at equal budget; tiny F32-
     * scale differences between two good solutions are expected. */
    ASSERT_TRUE(em_final_mse <= lloyd_final * 1.05f);

    free(X);
    rcre_arena_destroy(arena);
}

/* Full run on the oracle's 20000x24 blob dataset (checked-in X). The hard
 * quality gate is the statistical oracle parity in the dev project
 * (projects/rcre_emeans, make quality-par); in the core we gate
 * NON-DEGENERACY: e-means must beat the single-cluster (global-mean)
 * baseline by two orders of magnitude. At this small scale (n=20K)
 * k-means++ + 25 Lloyd iterations keeps an init advantage — reported, not
 * gated (see projects/rcre_emeans/docs/benchmarks.md). */
static void test_quality_fullrun_nondegenerate(void) {
    const size_t n = 20000, d = 24, k = 256;
    char xpath[256];
    snprintf(xpath, sizeof(xpath), EM_FIX_DIR "/X_fullrun.f32");
    size_t xsz = 0;
    float* X = (float*)em_read_file(xpath, &xsz);
    ASSERT_NOT_NULL(X);
    ASSERT_UINT_EQ(n * d, xsz / sizeof(float));

    RCRE_Arena* arena = rcre_arena_create(256 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);
    size_t xshape[2] = { n, d };
    RCRE_Tensor Xt = rcre_tensor_create(xshape, 2, RCRE_DTYPE_F32,
                                        RCRE_LAYOUT_TABULAR, arena);
    memcpy(Xt.data, X, n * d * sizeof(float));
    RCRE_Tensor C = em_tensor2(arena, k, d, RCRE_DTYPE_F32);
    size_t lshape[2] = { n, 1 };
    RCRE_Tensor L = rcre_tensor_create(lshape, 2, RCRE_DTYPE_I32,
                                       RCRE_LAYOUT_TABULAR, arena);

    float moving_mse = 0.0f;
    int n_splits = 0;
    RCRE_EMMeansParams p;
    memset(&p, 0, sizeof(p));
    p.nepochs = 5.0f;
    p.shuffle = true;
    p.unused_threshold = -1.0f;
    p.seed = 42;
    p.capture_labels = true;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_fit_ex(&Xt, k, &p, &C, &L, &moving_mse,
                                        &n_splits, arena, NULL));
    float em_final_mse = 0.0f;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_assign(&Xt, &C, &L, &em_final_mse, arena));

    /* Lloyd-25 for the record (same exact scoring pass) */
    RCRE_Tensor Cl = em_tensor2(arena, k, d, RCRE_DTYPE_F32);
    size_t llshape[2] = { n, 1 };
    RCRE_Tensor Ll = rcre_tensor_create(llshape, 2, RCRE_DTYPE_I32,
                                        RCRE_LAYOUT_TABULAR, arena);
    RCRE_KMeansParams kp;
    memset(&kp, 0, sizeof(kp));
    kp.max_iter = 25;
    kp.n_init = 1;
    kp.tol = 1e-6f;
    kp.seed = 42;
    float lloyd_mse = 0.0f;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_kmeans_fit_ex(&Xt, k, &kp, &Cl, &Ll, &lloyd_mse,
                                        arena));
    float lloyd_final = 0.0f;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_assign(&Xt, &Cl, &Ll, &lloyd_final, arena));
    printf("  [fullrun] e-means(5) MSE %.6f vs Lloyd-25 MSE %.6f "
           "(ratio %.3f, splits=%d)\n", (double)em_final_mse,
           (double)lloyd_final,
           (double)em_final_mse / (double)lloyd_final, n_splits);

    double baseline = 0.0;
    {
        double mean[24] = { 0 };
        const float* Xd = (const float*)Xt.data;
        for (size_t i = 0; i < n; i++)
            for (size_t dd = 0; dd < d; dd++)
                mean[dd] += (double)Xd[i * d + dd];
        for (size_t dd = 0; dd < d; dd++) mean[dd] /= (double)n;
        for (size_t i = 0; i < n; i++)
            for (size_t dd = 0; dd < d; dd++) {
                double df = (double)Xd[i * d + dd] - mean[dd];
                baseline += df * df;
            }
    }
    printf("  [fullrun] single-cluster baseline MSE %.6f (em %.6f = "
           "%.1fx better)\n", (double)(baseline / (double)n),
           (double)em_final_mse, (double)baseline / (double)n /
           (double)em_final_mse);
    ASSERT_TRUE((double)em_final_mse < 0.01 * (baseline / (double)n));

    free(X);
    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* T10 — error paths                                                          */
/* ========================================================================= */

static void test_errors_arg_validation(void) {
    RCRE_Arena* arena = rcre_arena_create(8 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);
    size_t xshape[2] = { 1024, 8 };
    RCRE_Tensor X = rcre_tensor_create(xshape, 2, RCRE_DTYPE_F32,
                                       RCRE_LAYOUT_TABULAR, arena);
    for (size_t i = 0; i < 1024 * 8; i++)
        ((float*)X.data)[i] = (float)(i % 7);
    RCRE_Tensor C = em_tensor2(arena, 16, 8, RCRE_DTYPE_F32);
    size_t lshape[2] = { 1024, 1 };
    RCRE_Tensor L = rcre_tensor_create(lshape, 2, RCRE_DTYPE_I32,
                                       RCRE_LAYOUT_TABULAR, arena);

    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 0, 2.0f, &C, &L, arena));
    RCRE_Tensor Cbig = em_tensor2(arena, 2048, 8, RCRE_DTYPE_F32);
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 2048, 2.0f, &Cbig, &L, arena));
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 16, 0.0f, &C, &L, arena));
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 16, -1.0f, &C, &L, arena));
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 16, 999.0f, &C, &L, arena));
    RCRE_Tensor Cwrong = em_tensor2(arena, 16, 16, RCRE_DTYPE_F32);
    ASSERT_INT_EQ(RCRE_ERR_SHAPE_MISMATCH,
                  rcre_ml_emeans_fit(&X, 16, 2.0f, &Cwrong, &L, arena));
    RCRE_Tensor XI = em_tensor2(arena, 1024, 8, RCRE_DTYPE_I32);
    ASSERT_INT_EQ(RCRE_ERR_SHAPE_MISMATCH,
                  rcre_ml_emeans_fit(&XI, 16, 2.0f, &C, &L, arena));
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(NULL, 16, 2.0f, &C, &L, arena));
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 16, 2.0f, NULL, &L, arena));

    /* params: negative tau means auto (not an error); NaN alpha0 errors */
    RCRE_EMMeansParams p;
    memset(&p, 0, sizeof(p));
    p.nepochs = 2.0f;
    p.shuffle = true;
    p.unused_threshold = -2.0f; /* any negative = auto */
    p.capture_labels = true;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_fit_ex(&X, 16, &p, &C, &L, NULL, NULL,
                                        arena, NULL));
    p.unused_threshold = 0.0f;
    p.alpha0 = NAN;
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit_ex(&X, 16, &p, &C, &L, NULL, NULL,
                                        arena, NULL));
    p.alpha0 = 0.5f;
    p.batch_law = 42;
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit_ex(&X, 16, &p, &C, &L, NULL, NULL,
                                        arena, NULL));
    p.batch_law = RCRE_EM_BATCH_AUTO;
    RCRE_Tensor Ci = em_tensor2(arena, 8, 8, RCRE_DTYPE_F32);
    p.init_centroids = &Ci;
    ASSERT_INT_EQ(RCRE_ERR_SHAPE_MISMATCH,
                  rcre_ml_emeans_fit_ex(&X, 16, &p, &C, &L, NULL, NULL,
                                        arena, NULL));

    /* batch_size = 0 means "law" — not an error */
    memset(&p, 0, sizeof(p));
    p.nepochs = 2.0f;
    p.shuffle = true;
    p.unused_threshold = 0.0f;
    p.capture_labels = true;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_fit_ex(&X, 16, &p, &C, &L, NULL, NULL,
                                        arena, NULL));

    /* recipe API validation */
    RCRE_EMMeansRecipe r;
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_recipe(NULL, 16, NULL, &r));
    ASSERT_INT_EQ(RCRE_OK, rcre_ml_emeans_recipe(&X, 16, NULL, &r));
    ASSERT_UINT_EQ(1234, r.seed);

    rcre_arena_destroy(arena);
}

static void test_errors_assign_validation(void) {
    RCRE_Arena* arena = rcre_arena_create(8 * 1024 * 1024);
    size_t xshape[2] = { 64, 4 };
    RCRE_Tensor X = rcre_tensor_create(xshape, 2, RCRE_DTYPE_F32,
                                       RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor C = em_tensor2(arena, 8, 4, RCRE_DTYPE_F32);
    size_t lshape[2] = { 64, 1 };
    RCRE_Tensor L = rcre_tensor_create(lshape, 2, RCRE_DTYPE_I32,
                                       RCRE_LAYOUT_TABULAR, arena);
    float mse = 0.0f;
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_assign(NULL, &C, &L, &mse, arena));
    RCRE_Tensor Cwrong = em_tensor2(arena, 8, 8, RCRE_DTYPE_F32);
    ASSERT_INT_EQ(RCRE_ERR_SHAPE_MISMATCH,
                  rcre_ml_emeans_assign(&X, &Cwrong, &L, &mse, arena));
    ASSERT_INT_EQ(RCRE_OK, rcre_ml_emeans_assign(&X, &C, &L, &mse, arena));
    rcre_arena_destroy(arena);
}

static void test_errors_tiny_problems(void) {
    /* n == k: init uses all rows in order; T = 1.0 */
    RCRE_Arena* arena = rcre_arena_create(8 * 1024 * 1024);
    size_t xshape[2] = { 4, 2 };
    RCRE_Tensor X = rcre_tensor_create(xshape, 2, RCRE_DTYPE_F32,
                                       RCRE_LAYOUT_TABULAR, arena);
    const float raw[8] = { 0, 0, 10, 10, 0.1f, 0.2f, 10.1f, 9.9f };
    memcpy(X.data, raw, sizeof(raw));
    RCRE_Tensor C = em_tensor2(arena, 4, 2, RCRE_DTYPE_F32);
    size_t lshape[2] = { 4, 1 };
    RCRE_Tensor L = rcre_tensor_create(lshape, 2, RCRE_DTYPE_I32,
                                       RCRE_LAYOUT_TABULAR, arena);
    float inertia = 0.0f;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_fit_ex(&X, 4, NULL, &C, &L, &inertia, NULL,
                                        arena, NULL));
    const float* Cd = (const float*)C.data;
    int near_origin = 0, near_far = 0;
    for (size_t j = 0; j < 4; j++) {
        if (Cd[j * 2 + 0] < 5.0f) near_origin++;
        else near_far++;
    }
    ASSERT_UINT_EQ(2, (size_t)near_origin);
    ASSERT_UINT_EQ(2, (size_t)near_far);

    /* T < 1: single partial pass; unvisited rows keep label -1 */
    RCRE_EMMeansParams p;
    memset(&p, 0, sizeof(p));
    p.nepochs = 0.5f;
    p.shuffle = true;
    p.unused_threshold = 0.0f;
    p.capture_labels = true;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_fit_ex(&X, 4, &p, &C, &L, &inertia, NULL,
                                        arena, NULL));
    const int32_t* Ld = (const int32_t*)L.data;
    int visited = 0, unvisited = 0;
    for (size_t i = 0; i < 4; i++)
        if (Ld[i] >= 0) visited++; else unvisited++;
    ASSERT_TRUE(visited >= 1 && visited <= 2); /* 0.5*4 = 2 rows */
    ASSERT_UINT_EQ(4 - (size_t)visited, (size_t)unvisited);
    rcre_arena_destroy(arena);
}

/* ========================================================================= */

int main(void) {
    test_suite_begin("e-means (RCRE operator): recipe, parity, assign, "
                     "determinism, quality, errors");
    TEST_RUN(test_recipe_round_pow2);
    TEST_RUN(test_recipe_epoch_remainder);
    TEST_RUN(test_recipe_scheduled_alpha);
    TEST_RUN(test_recipe_defaults);
    TEST_RUN(test_recipe_oracle_grid);
    TEST_RUN(test_parity_strict_norev);
    TEST_RUN(test_parity_strict_revive);
    TEST_RUN(test_assign_true_argmin);
    TEST_RUN(test_assign_tie_first_index);
    TEST_RUN(test_determinism_run_to_run);
    TEST_RUN(test_determinism_pool_size);
    TEST_RUN(test_determinism_serial_vs_pool);
    TEST_RUN(test_quality_two_blob_trap);
    TEST_RUN(test_quality_fullrun_nondegenerate);
    TEST_RUN(test_errors_arg_validation);
    TEST_RUN(test_errors_assign_validation);
    TEST_RUN(test_errors_tiny_problems);
    return test_suite_report();
}
