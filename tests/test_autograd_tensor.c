/**
 * @file test_autograd_tensor.c
 * @brief Unit tests for RCRE Dynamic Tensor Tape Autograd Engine
 */

#include "rcre.h"
#include "rcre_autograd.h"
#include "test_framework.h"
#include <math.h>

#define EPSILON 1e-4f

static void test_tensor_tape_matmul_and_backward(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 32);
    ASSERT_NOT_NULL(tape);

    /* A: [2, 3], B: [3, 2] */
    RCRE_Tensor* A = rcre_tensor_alloc_2d(arena, 2, 3);
    RCRE_Tensor* B = rcre_tensor_alloc_2d(arena, 3, 2);
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);

    /* Populate A */
    rcre_tensor_f32_data(A)[0] = 1.0f; rcre_tensor_f32_data(A)[1] = 2.0f; rcre_tensor_f32_data(A)[2] = 3.0f;
    rcre_tensor_f32_data(A)[3] = 4.0f; rcre_tensor_f32_data(A)[4] = 5.0f; rcre_tensor_f32_data(A)[5] = 6.0f;

    /* Populate B */
    rcre_tensor_f32_data(B)[0] = 7.0f;  rcre_tensor_f32_data(B)[1] = 8.0f;
    rcre_tensor_f32_data(B)[2] = 9.0f;  rcre_tensor_f32_data(B)[3] = 1.0f;
    rcre_tensor_f32_data(B)[4] = 2.0f;  rcre_tensor_f32_data(B)[5] = 3.0f;

    /* Forward: C = A * B */
    RCRE_Tensor* C = rcre_autograd_matmul(tape, A, B);
    ASSERT_NOT_NULL(C);
    ASSERT_UINT_EQ(2, C->shape[0]);
    ASSERT_UINT_EQ(2, C->shape[1]);

    /* Expected:
       C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
       C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
       C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
       C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    */
    ASSERT_FLOAT_NEAR(31.0f, rcre_tensor_f32_data(C)[0], EPSILON);
    ASSERT_FLOAT_NEAR(19.0f, rcre_tensor_f32_data(C)[1], EPSILON);
    ASSERT_FLOAT_NEAR(85.0f, rcre_tensor_f32_data(C)[2], EPSILON);
    ASSERT_FLOAT_NEAR(55.0f, rcre_tensor_f32_data(C)[3], EPSILON);

    /* Seed output gradients dC = 1 */
    C->grad[0] = 1.0f; C->grad[1] = 1.0f;
    C->grad[2] = 1.0f; C->grad[3] = 1.0f;

    /* Backprop */
    rcre_tensor_tape_backward(tape);

    /* dA = dC * B^T
       B^T:
       [7, 9, 2]
       [8, 1, 3]
       dA[0, :] = [1, 1] * B^T = [15, 10, 5]
       dA[1, :] = [1, 1] * B^T = [15, 10, 5]
    */
    ASSERT_FLOAT_NEAR(15.0f, A->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(10.0f, A->grad[1], EPSILON);
    ASSERT_FLOAT_NEAR(5.0f,  A->grad[2], EPSILON);
    ASSERT_FLOAT_NEAR(15.0f, A->grad[3], EPSILON);
    ASSERT_FLOAT_NEAR(10.0f, A->grad[4], EPSILON);
    ASSERT_FLOAT_NEAR(5.0f,  A->grad[5], EPSILON);

    /* dB = A^T * dC
       A^T:
       [1, 4]
       [2, 5]
       [3, 6]
       dB[0, :] = [1, 4] * [1, 1; 1, 1] = [5, 5]
       dB[1, :] = [2, 5] * [1, 1; 1, 1] = [7, 7]
       dB[2, :] = [3, 6] * [1, 1; 1, 1] = [9, 9]
    */
    ASSERT_FLOAT_NEAR(5.0f, B->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(5.0f, B->grad[1], EPSILON);
    ASSERT_FLOAT_NEAR(7.0f, B->grad[2], EPSILON);
    ASSERT_FLOAT_NEAR(7.0f, B->grad[3], EPSILON);
    ASSERT_FLOAT_NEAR(9.0f, B->grad[4], EPSILON);
    ASSERT_FLOAT_NEAR(9.0f, B->grad[5], EPSILON);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(arena);
}

static void test_tensor_tape_add_bias_and_activations(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 32);
    ASSERT_NOT_NULL(tape);

    /* X: [2, 2], bias: [2] */
    RCRE_Tensor* X = rcre_tensor_alloc_2d(arena, 2, 2);
    RCRE_Tensor* bias = rcre_tensor_alloc_1d(arena, 2);

    rcre_tensor_f32_data(X)[0] = -2.0f; rcre_tensor_f32_data(X)[1] = 1.0f;
    rcre_tensor_f32_data(X)[2] = 3.0f;  rcre_tensor_f32_data(X)[3] = -4.0f;

    rcre_tensor_f32_data(bias)[0] = 0.5f;
    rcre_tensor_f32_data(bias)[1] = -0.5f;

    /* Add bias */
    RCRE_Tensor* Z = rcre_autograd_add_bias(tape, X, bias);
    /* Z: [-1.5, 0.5; 3.5, -4.5] */
    ASSERT_FLOAT_NEAR(-1.5f, rcre_tensor_f32_data(Z)[0], EPSILON);
    ASSERT_FLOAT_NEAR(0.5f,  rcre_tensor_f32_data(Z)[1], EPSILON);
    ASSERT_FLOAT_NEAR(3.5f,  rcre_tensor_f32_data(Z)[2], EPSILON);
    ASSERT_FLOAT_NEAR(-4.5f, rcre_tensor_f32_data(Z)[3], EPSILON);

    /* ReLU */
    RCRE_Tensor* A = rcre_autograd_relu(tape, Z);
    /* A: [0.0, 0.5; 3.5, 0.0] */
    ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_f32_data(A)[0], EPSILON);
    ASSERT_FLOAT_NEAR(0.5f, rcre_tensor_f32_data(A)[1], EPSILON);
    ASSERT_FLOAT_NEAR(3.5f, rcre_tensor_f32_data(A)[2], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_f32_data(A)[3], EPSILON);

    /* Seed dA = 1 */
    A->grad[0] = 1.0f; A->grad[1] = 1.0f;
    A->grad[2] = 1.0f; A->grad[3] = 1.0f;

    rcre_tensor_tape_backward(tape);

    /* Z grad through ReLU:
       Z_grad: [0.0, 1.0; 1.0, 0.0]
    */
    ASSERT_FLOAT_NEAR(0.0f, X->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(1.0f, X->grad[1], EPSILON);
    ASSERT_FLOAT_NEAR(1.0f, X->grad[2], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, X->grad[3], EPSILON);

    /* bias grad: row sum of Z_grad = [1.0, 1.0] */
    ASSERT_FLOAT_NEAR(1.0f, bias->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(1.0f, bias->grad[1], EPSILON);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(arena);
}

static void test_tape_grow_then_rewind(void) {
    /* P0 regression: the tape node array must survive rcre_arena_temp_end()
     * rewinds even when it grows mid-batch. Node storage is heap-owned; only
     * activations live in the rewound arena. */
    RCRE_Arena* tape_arena = rcre_arena_create(1 << 20);
    RCRE_Arena* perma = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(tape_arena);
    ASSERT_NOT_NULL(perma);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(tape_arena, 4); /* tiny cap -> forces growth */
    ASSERT_NOT_NULL(tape);

    RCRE_Tensor* X = rcre_tensor_alloc_2d(perma, 4, 8);
    RCRE_Tensor* W = rcre_tensor_alloc_2d(perma, 8, 8);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(W);
    for (int i = 0; i < 32; ++i) rcre_tensor_f32_data(X)[i] = 0.25f;
    for (int i = 0; i < 64; ++i) rcre_tensor_f32_data(W)[i] = 0.1f;

    float run_grads[2];
    for (int batch = 0; batch < 2; ++batch) {
        RCRE_ArenaTemp temp = rcre_arena_temp_begin(tape_arena);
        rcre_tensor_tape_reset(tape);
        rcre_tensor_zero_grad(X); /* backward accumulates: start clean */

        RCRE_Tensor* cur = X;
        for (int i = 0; i < 8; ++i) {
            RCRE_Tensor* nxt = rcre_autograd_matmul(tape, cur, W);
            ASSERT_NOT_NULL(nxt);
            rcre_tensor_zero_grad(nxt);
            nxt->grad[0] = 1.0f;
            cur = nxt;
        }
        rcre_tensor_tape_backward(tape);
        run_grads[batch] = X->grad[0];

        rcre_arena_temp_end(temp);
    }

    /* Both batches are identical workloads: gradients must match exactly. */
    ASSERT_FLOAT_NEAR(run_grads[0], run_grads[1], 0.0f);
    ASSERT_TRUE(run_grads[0] != 0.0f);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(tape_arena);
    rcre_arena_destroy(perma);
}

static void test_autograd_shape_validation(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 8);
    ASSERT_NOT_NULL(tape);

    /* add with mismatched numel must be rejected */
    RCRE_Tensor* A = rcre_tensor_alloc_2d(arena, 2, 2); /* numel 4 */
    RCRE_Tensor* B = rcre_tensor_alloc_2d(arena, 3, 1); /* numel 3 */
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NULL(rcre_autograd_add(tape, A, B));

    /* add_bias with wrong bias width must be rejected */
    RCRE_Tensor* bias2 = rcre_tensor_alloc_1d(arena, 2);
    RCRE_Tensor* bias3 = rcre_tensor_alloc_1d(arena, 3);
    ASSERT_NOT_NULL(bias2);
    ASSERT_NOT_NULL(bias3);
    ASSERT_NOT_NULL(rcre_autograd_add_bias(tape, A, bias2)); /* valid: 2 cols */
    ASSERT_NULL(rcre_autograd_add_bias(tape, A, bias3));     /* invalid: 3 cols */

    /* Autograd ops must refuse non-F32 tensors (the dtype gate that used to
     * live in rcre_ag_from_tensor now lives in every op, via is_f32()). */
    uint8_t px[4] = { 200, 100, 50, 250 };
    RCRE_Tensor img = {
        .data = px, .ndim = 1, .dtype = RCRE_DTYPE_U8, .layout = RCRE_LAYOUT_TABULAR,
        .shape = { 4, 1, 1, 1 }, .strides = { 1, 0, 0, 0 }, .numel = 4
    };
    ASSERT_NULL(rcre_autograd_relu(tape, &img));
    ASSERT_NULL(rcre_autograd_add(tape, A, &img)); /* mixed numel too, but dtype fails first */

    /* A plain core F32 tensor (value type, from_buffer) feeds autograd ops
     * directly — the payoff of the v0.2.2 type unification. */
    float fx[4] = { -1.0f, 2.0f, -3.0f, 4.0f };
    size_t cshape[2] = { 2, 2 };
    RCRE_Tensor core = rcre_tensor_from_buffer(fx, cshape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
    RCRE_Tensor* ok = rcre_autograd_relu(tape, &core);
    ASSERT_NOT_NULL(ok);
    ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_f32_data(ok)[0], 1e-6f);
    ASSERT_FLOAT_NEAR(2.0f, rcre_tensor_f32_data(ok)[1], 1e-6f);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Dynamic Tensor Tape Autograd (AUTOGRAD-TENSOR-01)");
    TEST_RUN(test_tensor_tape_matmul_and_backward);
    TEST_RUN(test_tensor_tape_add_bias_and_activations);
    TEST_RUN(test_tape_grow_then_rewind);
    TEST_RUN(test_autograd_shape_validation);
    return test_suite_report();
}
