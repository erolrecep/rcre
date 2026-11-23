/**
 * @file test_stat.c
 * @brief Unit tests for RCRE Statistics, EDA & Preprocessing (Phase 2)
 */

#include "rcre.h"
#include "rcre_stat.h"
#include "rcre_prep.h"
#include "test_framework.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void test_stat_describe(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 1 to 10 */
    float data[10];
    for (int i = 0; i < 10; ++i) {
        data[i] = (float)(i + 1);
    }

    RCRE_StatSummary s = rcre_stat_describe(data, 10, arena);
    ASSERT_UINT_EQ(10, s.count);
    ASSERT_UINT_EQ(0, s.null_count);
    ASSERT_FLOAT_NEAR(5.5f, s.mean, 1e-4f);
    ASSERT_FLOAT_NEAR(1.0f, s.min, 1e-4f);
    ASSERT_FLOAT_NEAR(10.0f, s.max, 1e-4f);
    ASSERT_FLOAT_NEAR(5.5f, s.median, 1e-4f);
    /* Sample variance of 1..10 is 82.5 / 9 = 9.166667 */
    ASSERT_FLOAT_NEAR(9.166667f, s.variance, 1e-4f);
    ASSERT_FLOAT_NEAR(3.027650f, s.stddev, 1e-4f);
    /* Skewness of symmetric sequence should be ~0 */
    ASSERT_FLOAT_NEAR(0.0f, s.skewness, 1e-4f);

    rcre_arena_destroy(arena);
}

static void test_stat_quantiles(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    float data[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    size_t n = 5;

    float q0 = rcre_stat_quantile(data, n, 0.0f, arena);
    float q50 = rcre_stat_quantile(data, n, 0.5f, arena);
    float q100 = rcre_stat_quantile(data, n, 1.0f, arena);
    float q25 = rcre_stat_quantile(data, n, 0.25f, arena);

    ASSERT_FLOAT_NEAR(10.0f, q0, 1e-4f);
    ASSERT_FLOAT_NEAR(30.0f, q50, 1e-4f);
    ASSERT_FLOAT_NEAR(50.0f, q100, 1e-4f);
    ASSERT_FLOAT_NEAR(20.0f, q25, 1e-4f);

    rcre_arena_destroy(arena);
}

static void test_stat_cov_and_corr_matrices(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 4 samples, 2 features: y = 2x */
    size_t shape[] = {4, 2};
    RCRE_Tensor X = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* d = (float*)X.data;
    d[0] = 1.0f; d[1] = 2.0f;
    d[2] = 2.0f; d[3] = 4.0f;
    d[4] = 3.0f; d[5] = 6.0f;
    d[6] = 4.0f; d[7] = 8.0f;

    RCRE_Tensor cov = rcre_stat_cov_matrix(&X, arena);
    ASSERT_UINT_EQ(2, cov.shape[0]);
    ASSERT_UINT_EQ(2, cov.shape[1]);
    const float* cov_d = (const float*)cov.data;
    ASSERT_FLOAT_NEAR(1.666667f, cov_d[0 * 2 + 0], 1e-4f); /* var(X0) */
    ASSERT_FLOAT_NEAR(6.666667f, cov_d[1 * 2 + 1], 1e-4f); /* var(X1) */
    ASSERT_FLOAT_NEAR(3.333333f, cov_d[0 * 2 + 1], 1e-4f); /* cov(X0, X1) */

    RCRE_Tensor corr = rcre_stat_corr_matrix(&X, arena);
    const float* corr_d = (const float*)corr.data;
    ASSERT_FLOAT_NEAR(1.0f, corr_d[0 * 2 + 0], 1e-5f);
    ASSERT_FLOAT_NEAR(1.0f, corr_d[1 * 2 + 1], 1e-5f);
    ASSERT_FLOAT_NEAR(1.0f, corr_d[0 * 2 + 1], 1e-5f);
    ASSERT_FLOAT_NEAR(1.0f, corr_d[1 * 2 + 0], 1e-5f);

    rcre_arena_destroy(arena);
}

static void test_stat_spearman_matrix(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 4 samples, 2 features: monotonic non-linear y = x^3 */
    size_t shape[] = {4, 2};
    RCRE_Tensor X = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* d = (float*)X.data;
    d[0] = 1.0f; d[1] = 1.0f;
    d[2] = 2.0f; d[3] = 8.0f;
    d[4] = 3.0f; d[5] = 27.0f;
    d[6] = 4.0f; d[7] = 64.0f;

    RCRE_Tensor spearman = rcre_stat_spearman_matrix(&X, arena);
    const float* sp_d = (const float*)spearman.data;
    ASSERT_FLOAT_NEAR(1.0f, sp_d[0 * 2 + 0], 1e-4f);
    ASSERT_FLOAT_NEAR(1.0f, sp_d[1 * 2 + 1], 1e-4f);
    ASSERT_FLOAT_NEAR(1.0f, sp_d[0 * 2 + 1], 1e-4f);
    ASSERT_FLOAT_NEAR(1.0f, sp_d[1 * 2 + 0], 1e-4f);

    rcre_arena_destroy(arena);
}

static void test_stat_pca(void) {
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 6 samples, 2 features with strong correlation along y = x */
    size_t shape[] = {6, 2};
    RCRE_Tensor X = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* d = (float*)X.data;
    d[0]  = -3.0f; d[1]  = -3.0f;
    d[2]  = -2.0f; d[3]  = -2.0f;
    d[4]  = -1.0f; d[5]  = -1.0f;
    d[6]  =  1.0f; d[7]  =  1.0f;
    d[8]  =  2.0f; d[9]  =  2.0f;
    d[10] =  3.0f; d[11] =  3.0f;

    RCRE_PCA pca = rcre_pca_fit(&X, 1, arena);
    ASSERT_UINT_EQ(1, pca.n_components);
    ASSERT_NOT_NULL(pca.components.data);
    ASSERT_NOT_NULL(pca.explained_variance_ratio);
    /* 100% variance is along y = x */
    ASSERT_FLOAT_NEAR(1.0f, pca.explained_variance_ratio[0], 1e-3f);

    RCRE_Tensor X_proj = rcre_pca_transform(&pca, &X, arena);
    ASSERT_UINT_EQ(6, X_proj.shape[0]);
    ASSERT_UINT_EQ(1, X_proj.shape[1]);

    const float* p = (const float*)X_proj.data;
    /* First sample and last sample must be well-separated along principal axis */
    ASSERT_TRUE(fabsf(p[5] - p[0]) > 5.0f);

    rcre_arena_destroy(arena);
}

static void test_prep_scalers(void) {
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t shape[] = {5, 2};
    RCRE_Tensor X = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* d = (float*)X.data;
    for (size_t r = 0; r < 5; ++r) {
        d[r * 2 + 0] = (float)((r + 1) * 10);
        d[r * 2 + 1] = (float)((r + 1) * 100);
    }

    /* StandardScaler */
    RCRE_PrepStandardScaler std_scaler = rcre_prep_standard_scaler_fit(&X, arena);
    ASSERT_FLOAT_NEAR(30.0f, std_scaler.mean[0], 1e-4f);
    ASSERT_FLOAT_NEAR(300.0f, std_scaler.mean[1], 1e-4f);

    RCRE_Tensor X_std = rcre_prep_standard_scaler_transform(&std_scaler, &X, arena);
    const float* std_d = (const float*)X_std.data;
    /* Row 2 is the mean (30, 300) -> should scale to 0.0 */
    ASSERT_FLOAT_NEAR(0.0f, std_d[2 * 2 + 0], 1e-4f);
    ASSERT_FLOAT_NEAR(0.0f, std_d[2 * 2 + 1], 1e-4f);

    /* MinMaxScaler */
    RCRE_PrepMinMaxScaler minmax = rcre_prep_minmax_scaler_fit(&X, 0.0f, 1.0f, arena);
    ASSERT_FLOAT_NEAR(10.0f, minmax.min[0], 1e-4f);

    RCRE_Tensor X_mm = rcre_prep_minmax_scaler_transform(&minmax, &X, arena);
    const float* mm_d = (const float*)X_mm.data;
    ASSERT_FLOAT_NEAR(0.0f, mm_d[0 * 2 + 0], 1e-4f);
    ASSERT_FLOAT_NEAR(1.0f, mm_d[4 * 2 + 0], 1e-4f);
    ASSERT_FLOAT_NEAR(0.5f, mm_d[2 * 2 + 0], 1e-4f);

    rcre_arena_destroy(arena);
}

static void test_prep_robust_and_imputer(void) {
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* RobustScaler */
    size_t shape[] = {5, 1};
    RCRE_Tensor X = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* d = (float*)X.data;
    d[0] = 10.0f; d[1] = 20.0f; d[2] = 30.0f; d[3] = 40.0f; d[4] = 1000.0f; /* outlier */

    RCRE_RobustScaler rob = rcre_prep_robust_scaler_fit(&X, arena);
    ASSERT_FLOAT_NEAR(30.0f, rob.center[0], 1e-4f);
    RCRE_Tensor X_rob = rcre_prep_robust_scaler_transform(&rob, &X, arena);
    const float* rob_d = (const float*)X_rob.data;
    ASSERT_FLOAT_NEAR(0.0f, rob_d[2], 1e-4f);

    /* SimpleImputer */
    RCRE_Tensor X_nan = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* nan_d = (float*)X_nan.data;
    nan_d[0] = 10.0f; nan_d[1] = 20.0f; nan_d[2] = NAN; nan_d[3] = 40.0f; nan_d[4] = 50.0f;

    RCRE_SimpleImputer imp = rcre_prep_simple_imputer_fit(&X_nan, RCRE_IMPUTE_MEAN, arena);
    /* Mean of 10+20+40+50 = 120 / 4 = 30.0 */
    ASSERT_FLOAT_NEAR(30.0f, imp.statistics[0], 1e-4f);

    RCRE_Tensor X_imp = rcre_prep_simple_imputer_transform(&imp, &X_nan, arena);
    const float* imp_d = (const float*)X_imp.data;
    ASSERT_FLOAT_NEAR(30.0f, imp_d[2], 1e-4f);

    rcre_arena_destroy(arena);
}

static void test_prep_one_hot_encode(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    int32_t categories[] = {0, 2, 1, 0};
    RCRE_Tensor oh = rcre_one_hot_encode(categories, 4, 3, arena);
    ASSERT_UINT_EQ(4, oh.shape[0]);
    ASSERT_UINT_EQ(3, oh.shape[1]);

    const float* d = (const float*)oh.data;
    /* Row 0: class 0 */
    ASSERT_FLOAT_NEAR(1.0f, d[0 * 3 + 0], 1e-5f);
    ASSERT_FLOAT_NEAR(0.0f, d[0 * 3 + 1], 1e-5f);
    ASSERT_FLOAT_NEAR(0.0f, d[0 * 3 + 2], 1e-5f);
    /* Row 1: class 2 */
    ASSERT_FLOAT_NEAR(0.0f, d[1 * 3 + 0], 1e-5f);
    ASSERT_FLOAT_NEAR(0.0f, d[1 * 3 + 1], 1e-5f);
    ASSERT_FLOAT_NEAR(1.0f, d[1 * 3 + 2], 1e-5f);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Statistics, EDA & Preprocessing (Phase 2)");

    TEST_RUN(test_stat_describe);
    TEST_RUN(test_stat_quantiles);
    TEST_RUN(test_stat_cov_and_corr_matrices);
    TEST_RUN(test_stat_spearman_matrix);
    TEST_RUN(test_stat_pca);
    TEST_RUN(test_prep_scalers);
    TEST_RUN(test_prep_robust_and_imputer);
    TEST_RUN(test_prep_one_hot_encode);

    return test_suite_report();
}
