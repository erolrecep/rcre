/**
 * @file test_errors.c
 * @brief T10 — error paths and input validation.
 */
#include "em_util.h"

static void test_arg_validation(void) {
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

    /* k = 0 */
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 0, 2.0f, &C, &L, arena));
    /* k > n */
    RCRE_Tensor Cbig = em_tensor2(arena, 2048, 8, RCRE_DTYPE_F32);
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 2048, 2.0f, &Cbig, &L, arena));
    /* T out of range */
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 16, 0.0f, &C, &L, arena));
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 16, -1.0f, &C, &L, arena));
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit(&X, 16, 999.0f, &C, &L, arena));
    /* centroids shape mismatch */
    RCRE_Tensor Cwrong = em_tensor2(arena, 16, 16, RCRE_DTYPE_F32);
    ASSERT_INT_EQ(RCRE_ERR_SHAPE_MISMATCH,
                  rcre_ml_emeans_fit(&X, 16, 2.0f, &Cwrong, &L, arena));
    /* non-F32 input */
    RCRE_Tensor XI = em_tensor2(arena, 1024, 8, RCRE_DTYPE_I32);
    ASSERT_INT_EQ(RCRE_ERR_SHAPE_MISMATCH,
                  rcre_ml_emeans_fit(&XI, 16, 2.0f, &C, &L, arena));
    /* NULLs */
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
                  rcre_ml_emeans_fit_ex(&X, 16, &p, &C, &L, NULL, NULL, arena, NULL));
    p.unused_threshold = 0.0f;
    p.alpha0 = NAN;
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit_ex(&X, 16, &p, &C, &L, NULL, NULL, arena, NULL));
    p.alpha0 = 0.5f;
    p.batch_law = 42;
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_fit_ex(&X, 16, &p, &C, &L, NULL, NULL, arena, NULL));
    /* init centroids wrong shape */
    p.batch_law = RCRE_EM_BATCH_AUTO;
    RCRE_Tensor Ci = em_tensor2(arena, 8, 8, RCRE_DTYPE_F32);
    p.init_centroids = &Ci;
    ASSERT_INT_EQ(RCRE_ERR_SHAPE_MISMATCH,
                  rcre_ml_emeans_fit_ex(&X, 16, &p, &C, &L, NULL, NULL, arena, NULL));

    /* batch_size = 0 means "law" — not an error */
    memset(&p, 0, sizeof(p));
    p.nepochs = 2.0f;
    p.shuffle = true;
    p.unused_threshold = 0.0f;
    p.capture_labels = true;
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_fit_ex(&X, 16, &p, &C, &L, NULL, NULL, arena, NULL));

    /* recipe API validation */
    RCRE_EMMeansRecipe r;
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG,
                  rcre_ml_emeans_recipe(NULL, 16, NULL, &r));
    ASSERT_INT_EQ(RCRE_OK,
                  rcre_ml_emeans_recipe(&X, 16, NULL, &r));
    ASSERT_UINT_EQ(1234, r.seed);

    rcre_arena_destroy(arena);
}

static void test_assign_validation(void) {
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
    /* dimension mismatch */
    RCRE_Tensor Cwrong = em_tensor2(arena, 8, 8, RCRE_DTYPE_F32);
    ASSERT_INT_EQ(RCRE_ERR_SHAPE_MISMATCH,
                  rcre_ml_emeans_assign(&X, &Cwrong, &L, &mse, arena));
    /* happy path */
    ASSERT_INT_EQ(RCRE_OK, rcre_ml_emeans_assign(&X, &C, &L, &mse, arena));
    rcre_arena_destroy(arena);
}

static void test_tiny_problems(void) {
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
    /* with 4 distinct points as centroids and one pass, the exact means
     * are the points themselves (each cluster has its own point plus the
     * near-duplicate of its partner) */
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

int main(void) {
    test_suite_begin("e-means T10: error paths");
    TEST_RUN(test_arg_validation);
    TEST_RUN(test_assign_validation);
    TEST_RUN(test_tiny_problems);
    return test_suite_report();
}
