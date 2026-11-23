/**
 * @file test_metric.c
 * @brief Unit tests for RCRE Model Evaluation Metrics, Validation & Terminal Viz (Phase 3 & 4)
 */

#include "rcre.h"
#include "rcre_metric.h"
#include "rcre_vis_ds.h"
#include "test_framework.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void test_metric_roc_and_pr_auc(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    float y_true[] = {0.0f, 0.0f, 1.0f, 1.0f};
    float y_score_perfect[] = {0.1f, 0.2f, 0.8f, 0.9f};
    float y_score_inv[] = {0.9f, 0.8f, 0.2f, 0.1f};
    float y_score_tied[] = {0.5f, 0.5f, 0.5f, 0.5f};

    float auc_perf = rcre_metric_roc_auc(y_true, y_score_perfect, 4, arena);
    ASSERT_FLOAT_NEAR(1.0f, auc_perf, 1e-5f);

    float auc_inv = rcre_metric_roc_auc(y_true, y_score_inv, 4, arena);
    ASSERT_FLOAT_NEAR(0.0f, auc_inv, 1e-5f);

    float auc_tied = rcre_metric_roc_auc(y_true, y_score_tied, 4, arena);
    ASSERT_FLOAT_NEAR(0.5f, auc_tied, 1e-5f);

    float pr_perf = rcre_metric_pr_auc(y_true, y_score_perfect, 4, arena);
    ASSERT_FLOAT_NEAR(1.0f, pr_perf, 1e-4f);

    rcre_arena_destroy(arena);
}

static void test_metric_confusion_and_classification(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 6 samples: 3 pos, 3 neg */
    int32_t y_true[] = {1, 1, 1, 0, 0, 0};
    int32_t y_pred[] = {1, 1, 0, 0, 0, 1};

    RCRE_Tensor cm = rcre_metric_confusion_matrix(y_true, y_pred, 6, 2, arena);
    ASSERT_UINT_EQ(2, cm.shape[0]);
    ASSERT_UINT_EQ(2, cm.shape[1]);
    const int32_t* cm_d = (const int32_t*)cm.data;
    /* row 0: true negatives (actual 0, pred 0) = 2, false positives (actual 0, pred 1) = 1 */
    ASSERT_INT_EQ(2, cm_d[0 * 2 + 0]);
    ASSERT_INT_EQ(1, cm_d[0 * 2 + 1]);
    /* row 1: false negatives (actual 1, pred 0) = 1, true positives (actual 1, pred 1) = 2 */
    ASSERT_INT_EQ(1, cm_d[1 * 2 + 0]);
    ASSERT_INT_EQ(2, cm_d[1 * 2 + 1]);

    RCRE_ClassificationReport rep = rcre_metric_classification_summary(y_true, y_pred, 6, 0.5f);
    ASSERT_FLOAT_NEAR(4.0f / 6.0f, rep.accuracy, 1e-4f);
    ASSERT_FLOAT_NEAR(2.0f / 3.0f, rep.precision, 1e-4f);
    ASSERT_FLOAT_NEAR(2.0f / 3.0f, rep.recall, 1e-4f);
    ASSERT_FLOAT_NEAR(2.0f / 3.0f, rep.f1_score, 1e-4f);

    rcre_arena_destroy(arena);
}

static void test_metric_log_loss_and_regression(void) {
    float y_true_bin[] = {1.0f, 0.0f};
    float y_prob_bin[] = {0.5f, 0.5f};
    float loss = rcre_metric_log_loss(y_true_bin, y_prob_bin, 2);
    /* -ln(0.5) = 0.693147 */
    ASSERT_FLOAT_NEAR(0.693147f, loss, 1e-4f);

    float y_true[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y_pred[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float r2_perf = rcre_metric_r2_score(y_true, y_pred, 4);
    float mse_perf = rcre_metric_mse(y_true, y_pred, 4);
    ASSERT_FLOAT_NEAR(1.0f, r2_perf, 1e-5f);
    ASSERT_FLOAT_NEAR(0.0f, mse_perf, 1e-5f);

    float y_pred_noisy[] = {1.1f, 1.9f, 3.1f, 3.9f};
    float rmse = rcre_metric_rmse(y_true, y_pred_noisy, 4);
    float mae = rcre_metric_mae(y_true, y_pred_noisy, 4);
    ASSERT_FLOAT_NEAR(0.1f, rmse, 1e-4f);
    ASSERT_FLOAT_NEAR(0.1f, mae, 1e-4f);
}

static void test_metric_clustering_scores(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 6 samples, 2D coordinates into 2 well-separated clusters */
    size_t shape[] = {6, 2};
    RCRE_Tensor X = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* d = (float*)X.data;
    /* Cluster 0 around (0, 0) */
    d[0] = 0.0f;  d[1] = 0.0f;
    d[2] = 0.1f;  d[3] = 0.0f;
    d[4] = 0.0f;  d[5] = 0.1f;
    /* Cluster 1 around (100, 100) */
    d[6] = 100.0f; d[7] = 100.0f;
    d[8] = 100.1f; d[9] = 100.0f;
    d[10] = 100.0f; d[11] = 100.1f;

    int32_t labels[] = {0, 0, 0, 1, 1, 1};

    float sil = rcre_metric_silhouette_score(&X, labels, arena);
    ASSERT_TRUE(sil > 0.95f);

    float db = rcre_metric_davies_bouldin_score(&X, labels, 2, arena);
    ASSERT_TRUE(db < 0.05f);

    rcre_arena_destroy(arena);
}

static void test_metric_cross_validation(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t n = 10;
    size_t k = 5;

    RCRE_FoldSplit splits[5];
    int rc = rcre_cv_kfold(n, k, false, 0, arena, splits);
    ASSERT_INT_EQ(RCRE_OK, rc);
    for (size_t f = 0; f < k; ++f) {
        ASSERT_UINT_EQ(2, splits[f].test_count);
        ASSERT_UINT_EQ(8, splits[f].train_count);
    }

    int32_t y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    RCRE_FoldSplit s_splits[5];
    rc = rcre_cv_stratified_kfold(y, n, k, arena, s_splits);
    ASSERT_INT_EQ(RCRE_OK, rc);
    for (size_t f = 0; f < k; ++f) {
        ASSERT_UINT_EQ(2, s_splits[f].test_count);
        /* Stratified: each fold test set has 1 class 0 and 1 class 1 */
        size_t c0 = 0, c1 = 0;
        for (size_t t = 0; t < s_splits[f].test_count; ++t) {
            int32_t idx = (int32_t)s_splits[f].test_indices[t];
            if (y[idx] == 0) c0++;
            else if (y[idx] == 1) c1++;
        }
        ASSERT_UINT_EQ(1, c0);
        ASSERT_UINT_EQ(1, c1);
    }

    rcre_arena_destroy(arena);
}

static void test_vis_terminal_ds(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 3x3 Correlation matrix */
    size_t shape[] = {3, 3};
    RCRE_Tensor corr = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* c = (float*)corr.data;
    c[0] = 1.0f;  c[1] = 0.85f; c[2] = -0.6f;
    c[3] = 0.85f; c[4] = 1.0f;  c[5] = -0.3f;
    c[6] = -0.6f; c[7] = -0.3f; c[8] = 1.0f;
    const char* names[] = {"FeatureA", "FeatureB", "FeatureC"};

    printf("\n");
    rcre_vis_corr_heatmap(&corr, names);

    float data[] = {1.0f, 2.0f, 2.5f, 3.0f, 3.2f, 3.5f, 4.0f, 5.0f, 8.0f, 9.5f};
    rcre_vis_histogram(data, 10, 6, "Sample Distribution");

    rcre_vis_boxplot(data, 10, "Target Metric Boxplot", arena);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Model Metrics, Validation & Terminal Viz (Phase 3 & 4)");

    TEST_RUN(test_metric_roc_and_pr_auc);
    TEST_RUN(test_metric_confusion_and_classification);
    TEST_RUN(test_metric_log_loss_and_regression);
    TEST_RUN(test_metric_clustering_scores);
    TEST_RUN(test_metric_cross_validation);
    TEST_RUN(test_vis_terminal_ds);

    return test_suite_report();
}
