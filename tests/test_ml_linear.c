/**
 * @file test_ml_linear.c
 * @brief Unit tests for RCRE Classical ML Linear Models (OLS, Logistic, Pegasos SVM)
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-2f

static void test_ols_regression(void) {
    printf("  -> Testing Ordinary Least Squares (OLS) normal equations...\n");
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    /* Synthetic ground truth: y = 2.0 * x0 - 3.0 * x1 */
    const size_t N = 10;
    const size_t D = 2;

    size_t X_shape[2] = {N, D};
    RCRE_Tensor X = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float X_raw[20] = {
        1.0f, 1.0f,
        2.0f, 1.0f,
        3.0f, 2.0f,
        1.0f, 4.0f,
        5.0f, 2.0f,
        4.0f, 3.0f,
        2.0f, 5.0f,
        6.0f, 1.0f,
        7.0f, 3.0f,
        8.0f, 4.0f
    };
    for (size_t i = 0; i < N * D; ++i) {
        ((float*)X.data)[i] = X_raw[i];
    }

    size_t y_shape[2] = {N, 1};
    RCRE_Tensor y = rcre_tensor_create(y_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* y_data = (float*)y.data;
    for (size_t i = 0; i < N; ++i) {
        y_data[i] = 2.0f * X_raw[i * 2 + 0] - 3.0f * X_raw[i * 2 + 1];
    }

    size_t w_shape[2] = {D, 1};
    RCRE_Tensor weights = rcre_tensor_create(w_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    size_t p_shape[2] = {N, 1};
    RCRE_Tensor preds = rcre_tensor_create(p_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    int status = rcre_ml_ols_fit(&X, &y, &weights, arena);
    assert(status == RCRE_OK);

    float* w_data = (float*)weights.data;
    /* Assert learned weights match ground truth [2.0, -3.0] */
    assert(fabsf(w_data[0] - 2.0f) < EPSILON);
    assert(fabsf(w_data[1] - (-3.0f)) < EPSILON);

    status = rcre_ml_predict_linear(&X, &weights, &preds);
    assert(status == RCRE_OK);

    float* p_data = (float*)preds.data;
    for (size_t i = 0; i < N; ++i) {
        assert(fabsf(p_data[i] - y_data[i]) < EPSILON);
    }

    printf("     ✓ OLS fitted weights [%.4f, %.4f] within epsilon 1e-2\n", w_data[0], w_data[1]);
    rcre_arena_destroy(arena);
}

static void test_logistic_regression(void) {
    printf("  -> Testing Logistic Regression gradient descent...\n");
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    /* 6 points: 3 negative (-2, -2 region), 3 positive (+2, +2 region) */
    const size_t N = 6;
    const size_t D = 2;

    size_t X_shape[2] = {N, D};
    RCRE_Tensor X = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float X_raw[12] = {
        -2.0f, -1.5f,
        -1.5f, -2.0f,
        -2.5f, -1.8f,
        2.0f,  1.5f,
        1.5f,  2.0f,
        2.5f,  1.8f
    };
    for (size_t i = 0; i < N * D; ++i) {
        ((float*)X.data)[i] = X_raw[i];
    }

    size_t y_shape[2] = {N, 1};
    RCRE_Tensor y = rcre_tensor_create(y_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float y_raw[6] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < N; ++i) {
        ((float*)y.data)[i] = y_raw[i];
    }

    size_t w_shape[2] = {D, 1};
    RCRE_Tensor weights = rcre_tensor_create(w_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    size_t p_shape[2] = {N, 1};
    RCRE_Tensor preds = rcre_tensor_create(p_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    int status = rcre_ml_logistic_fit(&X, &y, &weights, 0.5f, 200, arena);
    assert(status == RCRE_OK);

    float* w_data = (float*)weights.data;
    /* Weights should be positive in both coordinates */
    assert(w_data[0] > 0.1f);
    assert(w_data[1] > 0.1f);

    status = rcre_ml_predict_linear(&X, &weights, &preds);
    assert(status == RCRE_OK);

    float* p_data = (float*)preds.data;
    /* Classification accuracy check: sign should match class (0 -> negative, 1 -> positive) */
    for (size_t i = 0; i < 3; ++i) {
        assert(p_data[i] < 0.0f);
    }
    for (size_t i = 3; i < 6; ++i) {
        assert(p_data[i] > 0.0f);
    }

    rcre_arena_destroy(arena);
    printf("     ✓ Logistic regression converged to 100%% classification accuracy\n");
}

static void test_pegasos_svm(void) {
    printf("  -> Testing Pegasos Soft-Margin Linear SVM...\n");
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    const size_t N = 6;
    const size_t D = 2;

    size_t X_shape[2] = {N, D};
    RCRE_Tensor X = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float X_raw[12] = {
        -3.0f, -2.5f,
        -2.5f, -3.0f,
        -3.5f, -2.8f,
        3.0f,  2.5f,
        2.5f,  3.0f,
        3.5f,  2.8f
    };
    for (size_t i = 0; i < N * D; ++i) {
        ((float*)X.data)[i] = X_raw[i];
    }

    size_t y_shape[2] = {N, 1};
    RCRE_Tensor y = rcre_tensor_create(y_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    /* Pegasos expects y in {-1, +1} */
    float y_raw[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < N; ++i) {
        ((float*)y.data)[i] = y_raw[i];
    }

    size_t w_shape[2] = {D, 1};
    RCRE_Tensor weights = rcre_tensor_create(w_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    size_t p_shape[2] = {N, 1};
    RCRE_Tensor preds = rcre_tensor_create(p_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    int status = rcre_ml_pegasos_svm_fit(&X, &y, &weights, 0.01f, 100, arena);
    assert(status == RCRE_OK);

    float* w_data = (float*)weights.data;
    assert(w_data[0] > 0.0f);
    assert(w_data[1] > 0.0f);

    status = rcre_ml_predict_linear(&X, &weights, &preds);
    assert(status == RCRE_OK);

    float* p_data = (float*)preds.data;
    for (size_t i = 0; i < 3; ++i) {
        assert(p_data[i] < 0.0f);
    }
    for (size_t i = 3; i < 6; ++i) {
        assert(p_data[i] > 0.0f);
    }

    rcre_arena_destroy(arena);
    printf("     ✓ Pegasos SVM separated classes with correct hyperplanes\n");
}

int main(void) {
    printf("🧪 Running RCRE Classical ML Linear Model Tests (ML-REQ-01)...\n");
    test_ols_regression();
    test_logistic_regression();
    test_pegasos_svm();
    printf("✅ All Classical ML Linear Model Tests Passed Successfully!\n");
    return 0;
}
