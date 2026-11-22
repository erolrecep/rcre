/**
 * @file test_assign.c
 * @brief T5 — assignment correctness: the reported label must be a true
 * argmin (verified by recomputing all k distances in FP64) and ties must
 * break to the first index. Also checks the cc - 2*x.c identity against
 * the direct squared distance on the margin.
 */
#include "em_util.h"

static void test_assign_true_argmin(void) {
    const size_t n = 4096, d = 12, k = 96;
    RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_Tensor X = em_tensor2(arena, n, d, RCRE_DTYPE_F32);
    em_make_blobs((float*)X.data, n, d, 40, 991);
    RCRE_Tensor C = em_tensor2(arena, k, d, RCRE_DTYPE_F32);
    em_make_blobs((float*)C.data, k, d, 16, 41); /* arbitrary centroids */
    RCRE_Tensor Lt;
    size_t lshape[2] = { n, 1 };
    Lt = rcre_tensor_create(lshape, 2, RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);

    float mse = -1.0f;
    int rc = rcre_ml_emeans_assign(&X, &C, &Lt, &mse, arena);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_TRUE(mse >= 0.0f);

    const float* Xd = (const float*)X.data;
    const float* Cd = (const float*)C.data;
    const int32_t* Ld = (const int32_t*)Lt.data;

    size_t verified = 0, margin_ok = 0, tie_zone = 0;
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
            /* clear F64 winner (absolute margin above the F32 accumulation
             * noise floor for this fixture scale): labels must agree */
            ASSERT_UINT_EQ(best64_j, (size_t)Ld[i]);
            margin_ok++;
        } else {
            /* F32-tie zone: the label may be either of the two nearest */
            int32_t l = Ld[i];
            ASSERT_TRUE(l == (int32_t)best64_j || l == (int32_t)second_j);
            tie_zone++;
        }
        verified++;
        /* identity check: ||x-c||² = ||x||² + ||c||² - 2*x.c (the argmin
         * ignores the row-constant ||x||² term; include it here so the
         * values match the F64 direct distance) */
        double xxj = 0.0, ccj = 0.0, dot = 0.0;
        for (size_t dd = 0; dd < d; dd++) {
            xxj += (double)Xd[i * d + dd] * (double)Xd[i * d + dd];
            ccj += (double)Cd[best64_j * d + dd] * (double)Cd[best64_j * d + dd];
            dot += (double)Xd[i * d + dd] * (double)Cd[best64_j * d + dd];
        }
        double via_identity = xxj + ccj - 2.0 * dot;
        /* the identity's F64 error scales with |c|², so only compare
         * relative where the distance is well above that floor */
        if (best64 > 1e-3) {
            double rel = fabs(via_identity - best64) / best64;
            if (rel > max_rel_dev) max_rel_dev = rel;
        }
    }
    ASSERT_UINT_EQ(n, verified);
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

int main(void) {
    test_suite_begin("e-means T5: assignment correctness");
    TEST_RUN(test_assign_true_argmin);
    TEST_RUN(test_assign_tie_first_index);
    return test_suite_report();
}
