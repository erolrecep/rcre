/**
 * @file test_nn_train.c
 * @brief Unit tests for RCRE Loss Functions, Optimizers, and Scalers
 */

#include "rcre.h"
#include "rcre_autograd.h"
#include "test_framework.h"
#include <math.h>

#define EPSILON 1e-3f

static void test_loss_mse_and_grad(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_AGTensor* preds = rcre_ag_tensor_create_1d(arena, 4);
    RCRE_AGTensor* targets = rcre_ag_tensor_create_1d(arena, 4);
    RCRE_AGTensor* grad = rcre_ag_tensor_create_1d(arena, 4);

    preds->data[0] = 1.0f; preds->data[1] = 2.0f; preds->data[2] = 3.0f; preds->data[3] = 4.0f;
    targets->data[0] = 2.0f; targets->data[1] = 2.0f; targets->data[2] = 4.0f; targets->data[3] = 4.0f;

    /* diff = [-1, 0, -1, 0], sum_sq = 2.0, mse = 2.0 / 4 = 0.5 */
    float loss = rcre_loss_mse(preds, targets, grad);
    ASSERT_FLOAT_NEAR(0.5f, loss, EPSILON);

    /* grad = 2 * diff / 4 = diff / 2 = [-0.5, 0.0, -0.5, 0.0] */
    ASSERT_FLOAT_NEAR(-0.5f, grad->data[0], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f,  grad->data[1], EPSILON);
    ASSERT_FLOAT_NEAR(-0.5f, grad->data[2], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f,  grad->data[3], EPSILON);

    rcre_arena_destroy(arena);
}

static void test_standard_scaler_and_batch_iter(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 4 samples, 2 features */
    RCRE_AGTensor* X = rcre_ag_tensor_create_2d(arena, 4, 2);
    RCRE_AGTensor* y = rcre_ag_tensor_create_2d(arena, 4, 1);

    /* Feature 0: [2, 4, 6, 8] -> mean=5, var=5 */
    /* Feature 1: [10, 20, 30, 40] -> mean=25 */
    X->data[0] = 2.0f;  X->data[1] = 10.0f;
    X->data[2] = 4.0f;  X->data[3] = 20.0f;
    X->data[4] = 6.0f;  X->data[5] = 30.0f;
    X->data[6] = 8.0f;  X->data[7] = 40.0f;

    y->data[0] = 1.0f; y->data[1] = 2.0f; y->data[2] = 3.0f; y->data[3] = 4.0f;

    RCRE_StandardScaler* scaler = rcre_standard_scaler_fit(arena, X);
    ASSERT_NOT_NULL(scaler);
    ASSERT_FLOAT_NEAR(5.0f, scaler->mean[0], EPSILON);
    ASSERT_FLOAT_NEAR(25.0f, scaler->mean[1], EPSILON);

    rcre_standard_scaler_transform(scaler, X);
    /* Mean of transformed data must be 0 */
    float mean_0 = (X->data[0] + X->data[2] + X->data[4] + X->data[6]) / 4.0f;
    ASSERT_FLOAT_NEAR(0.0f, mean_0, EPSILON);

    /* Mini-batch iterator */
    RCRE_BatchIter* iter = rcre_batch_iter_create(arena, X, y, 2, false);
    ASSERT_NOT_NULL(iter);

    RCRE_AGTensor* batch_X = rcre_ag_tensor_create_2d(arena, 2, 2);
    RCRE_AGTensor* batch_y = rcre_ag_tensor_create_2d(arena, 2, 1);

    bool has_b1 = rcre_batch_iter_next(iter, batch_X, batch_y);
    ASSERT_TRUE(has_b1);
    ASSERT_FLOAT_NEAR(1.0f, batch_y->data[0], EPSILON);
    ASSERT_FLOAT_NEAR(2.0f, batch_y->data[1], EPSILON);

    bool has_b2 = rcre_batch_iter_next(iter, batch_X, batch_y);
    ASSERT_TRUE(has_b2);
    ASSERT_FLOAT_NEAR(3.0f, batch_y->data[0], EPSILON);
    ASSERT_FLOAT_NEAR(4.0f, batch_y->data[1], EPSILON);

    bool has_b3 = rcre_batch_iter_next(iter, batch_X, batch_y);
    ASSERT_FALSE(has_b3);

    rcre_arena_destroy(arena);
}

static void test_adamw_convergence_loop(void) {
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Target function: y = 3.0 * x1 + 2.0 * x2 */
    const size_t N = 50;
    RCRE_AGTensor* X = rcre_ag_tensor_create_2d(arena, N, 2);
    RCRE_AGTensor* y = rcre_ag_tensor_create_2d(arena, N, 1);

    for (size_t i = 0; i < N; ++i) {
        float x1 = (float)(i % 7) * 0.5f + 0.1f;
        float x2 = (float)(i % 5) * 0.3f - 0.2f;
        X->data[i * 2 + 0] = x1;
        X->data[i * 2 + 1] = x2;
        y->data[i] = 3.0f * x1 + 2.0f * x2;
    }

    /* Model: W: [2, 1] initialized to 0 */
    RCRE_AGTensor* W = rcre_ag_tensor_create_2d(arena, 2, 1);
    W->data[0] = 0.0f;
    W->data[1] = 0.0f;

    RCRE_AGTensor* params[1] = { W };
    RCRE_TensorOptim* opt = rcre_tensor_optim_adamw_create(arena, params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);
    ASSERT_NOT_NULL(opt);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 32);
    ASSERT_NOT_NULL(tape);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (size_t epoch = 0; epoch < 300; ++epoch) {
        rcre_tensor_tape_reset(tape);
        rcre_tensor_optim_zero_grad(opt);

        /* Forward: preds = X * W */
        RCRE_AGTensor* preds = rcre_autograd_matmul(tape, X, W);

        /* Compute loss */
        float loss = rcre_loss_mse(preds, y, preds);
        if (epoch == 0) initial_loss = loss;
        final_loss = loss;

        /* Backprop */
        rcre_tensor_tape_backward(tape);

        /* Optimizer step */
        rcre_tensor_optim_step(opt);
    }

    /* Assert substantial loss reduction */
    ASSERT_TRUE(final_loss < initial_loss * 0.01f);
    /* Learned weights should be near [3.0, 2.0] */
    ASSERT_FLOAT_NEAR(3.0f, W->data[0], 0.2f);
    ASSERT_FLOAT_NEAR(2.0f, W->data[1], 0.2f);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(arena);
}

static void test_split_1d_y(void) {
    /* Regression: a 1-D regression target must not read an uninitialized
     * shape[1]; y is treated as (n, 1) and outputs are 2-D. */
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_AGTensor* X = rcre_ag_tensor_create_2d(arena, 10, 3);
    RCRE_AGTensor* y = rcre_ag_tensor_create_1d(arena, 10);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);
    for (int i = 0; i < 30; ++i) X->data[i] = (float)i;
    for (int i = 0; i < 10; ++i) y->data[i] = (float)(10 - i);

    RCRE_AGTensor *Xtr, *Xte, *ytr, *yte;
    rcre_data_train_test_split(arena, X, y, 0.7f, &Xtr, &ytr, &Xte, &yte);
    ASSERT_NOT_NULL(Xtr);
    ASSERT_NOT_NULL(ytr);
    ASSERT_NOT_NULL(Xte);
    ASSERT_NOT_NULL(yte);

    ASSERT_UINT_EQ(7, Xtr->shape[0]);
    ASSERT_UINT_EQ(3, Xtr->shape[1]);
    ASSERT_UINT_EQ(7, ytr->shape[0]);
    ASSERT_UINT_EQ(1, ytr->shape[1]);
    ASSERT_UINT_EQ(3, Xte->shape[0]);

    /* Content: first 7 rows train, last 3 test, values preserved
       (y = [10, 9, 8, ..., 1]) */
    ASSERT_FLOAT_NEAR(10.0f, ytr->data[0], EPSILON);
    ASSERT_FLOAT_NEAR(4.0f, ytr->data[6], EPSILON);
    ASSERT_FLOAT_NEAR(3.0f, yte->data[0], EPSILON);
    ASSERT_FLOAT_NEAR(1.0f, yte->data[2], EPSILON);

    rcre_arena_destroy(arena);
}

static void test_optimizer_hyperparam_validation(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_AGTensor* w = rcre_ag_tensor_create_2d(arena, 2, 2);
    ASSERT_NOT_NULL(w);
    RCRE_AGTensor* params[1] = { w };

    /* beta1 == 1.0 would divide by zero in the bias correction: reject. */
    ASSERT_NULL(rcre_tensor_optim_adamw_create(arena, params, 1, 0.1f, 1.0f, 0.999f, 1e-8f, 0.0f));
    ASSERT_NULL(rcre_tensor_optim_adamw_create(arena, params, 1, 0.1f, 0.9f, 1.0f, 1e-8f, 0.0f));
    ASSERT_NULL(rcre_tensor_optim_adamw_create(arena, params, 1, 0.0f, 0.9f, 0.999f, 1e-8f, 0.0f));
    /* Valid hyperparameters must still succeed. */
    ASSERT_NOT_NULL(rcre_tensor_optim_adamw_create(arena, params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f));

    /* SGD momentum outside [0, 1) is rejected. */
    ASSERT_NULL(rcre_tensor_optim_sgd_create(arena, params, 1, 0.1f, -0.1f, 0.0f));
    ASSERT_NOT_NULL(rcre_tensor_optim_sgd_create(arena, params, 1, 0.1f, 0.9f, 0.0f));

    rcre_arena_destroy(arena);
}

static void test_batch_iter_undersized_output(void) {
    /* Undersized batch buffers must fail cleanly (false), not corrupt
     * neighbouring arena allocations. */
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_AGTensor* X = rcre_ag_tensor_create_2d(arena, 10, 4);
    RCRE_AGTensor* y = rcre_ag_tensor_create_2d(arena, 10, 2);
    RCRE_BatchIter* iter = rcre_batch_iter_create(arena, X, y, 8, true);
    ASSERT_NOT_NULL(iter);

    RCRE_AGTensor* bx = rcre_ag_tensor_create_2d(arena, 4, 4); /* need 8x4 */
    RCRE_AGTensor* by = rcre_ag_tensor_create_2d(arena, 4, 2);
    ASSERT_FALSE(rcre_batch_iter_next(iter, bx, by));
    ASSERT_UINT_EQ(0, iter->current_idx); /* nothing consumed */

    /* Correctly sized buffers work. */
    RCRE_AGTensor* bx2 = rcre_ag_tensor_create_2d(arena, 8, 4);
    RCRE_AGTensor* by2 = rcre_ag_tensor_create_2d(arena, 8, 2);
    ASSERT_TRUE(rcre_batch_iter_next(iter, bx2, by2));
    ASSERT_UINT_EQ(8, iter->current_idx);

    rcre_arena_destroy(arena);
}

static void test_csv_long_line_and_ragged_reject(void) {
    /* 1000 columns per row => each line ~7KB, well past any fixed buffer. */
    const char* path = "/tmp/rcre_test_longline.csv";
    FILE* f = fopen(path, "w");
    ASSERT_NOT_NULL(f);
    fprintf(f, "header0,header1\n");
    for (int r = 0; r < 3; ++r) {
        fprintf(f, "%d", r);
        for (int c = 1; c < 1000; ++c) fprintf(f, ",%d", r * 1000 + c);
        fprintf(f, "\n");
    }
    fclose(f);

    RCRE_Arena* arena = rcre_arena_create(4 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_AGTensor* t = rcre_data_load_csv(arena, path, true, 0);
    ASSERT_NOT_NULL(t);
    ASSERT_UINT_EQ(3, t->shape[0]);
    ASSERT_UINT_EQ(1000, t->shape[1]);
    ASSERT_FLOAT_NEAR(999.0f, t->data[999], EPSILON);         /* row 0, col 999 */
    ASSERT_FLOAT_NEAR(2 * 1000 + 999, t->data[2000 + 999], EPSILON); /* row 2, col 999 */

    /* max_rows still honoured on long-line files */
    RCRE_AGTensor* t2 = rcre_data_load_csv(arena, path, true, 2);
    ASSERT_NOT_NULL(t2);
    ASSERT_UINT_EQ(2, t2->shape[0]);

    remove(path);

    /* Ragged rows must be rejected outright (no silent zero-fill). */
    const char* rpath = "/tmp/rcre_test_ragged.csv";
    f = fopen(rpath, "w");
    ASSERT_NOT_NULL(f);
    fprintf(f, "a,b\n1,2\n3\n"); /* third row has one field */
    fclose(f);
    ASSERT_NULL(rcre_data_load_csv(arena, rpath, false, 0));
    remove(rpath);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Losses, Scalers & Optimization (AUTOGRAD-OPTIM-01)");
    TEST_RUN(test_loss_mse_and_grad);
    TEST_RUN(test_standard_scaler_and_batch_iter);
    TEST_RUN(test_adamw_convergence_loop);
    TEST_RUN(test_split_1d_y);
    TEST_RUN(test_optimizer_hyperparam_validation);
    TEST_RUN(test_batch_iter_undersized_output);
    TEST_RUN(test_csv_long_line_and_ragged_reject);
    return test_suite_report();
}
