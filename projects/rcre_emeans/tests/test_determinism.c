/**
 * @file test_determinism.c
 * @brief T6 — determinism contract:
 *   * bit-identical run-to-run at a fixed configuration;
 *   * bit-identical across pool sizes (the slab partition depends only on
 *     (m, k, B), never on the pool);
 *   * serial vs parallel: the F32 batch-sum regrouping (slab partials
 *     merged in row order) differs at the ULP level, and that difference
 *     feeds back through the iterative system, so state can diverge
 *     chaotically — the honest contract is QUALITY equivalence: final
 *     exact MSE within 0.5% and >= 99% label agreement (same guarantee
 *     class as GPU k-means with nondeterministic reduction order).
 */
#include "em_util.h"

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

static void test_run_to_run_bit_identical(void) {
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

static void test_pool_size_invariance(void) {
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
    /* pool(4) and pool(16) must be bit-identical (slab partition is
     * pool-size independent) */
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

static void test_serial_vs_parallel_ulp(void) {
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

    /* Quality equivalence: exact final MSEs within 0.5%, labels >= 99%. */
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

int main(void) {
    test_suite_begin("e-means T6: determinism");
    TEST_RUN(test_run_to_run_bit_identical);
    TEST_RUN(test_pool_size_invariance);
    TEST_RUN(test_serial_vs_parallel_ulp);
    return test_suite_report();
}
