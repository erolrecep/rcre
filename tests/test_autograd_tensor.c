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
    RCRE_AGTensor* A = rcre_ag_tensor_create_2d(arena, 2, 3);
    RCRE_AGTensor* B = rcre_ag_tensor_create_2d(arena, 3, 2);
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);

    /* Populate A */
    A->data[0] = 1.0f; A->data[1] = 2.0f; A->data[2] = 3.0f;
    A->data[3] = 4.0f; A->data[4] = 5.0f; A->data[5] = 6.0f;

    /* Populate B */
    B->data[0] = 7.0f;  B->data[1] = 8.0f;
    B->data[2] = 9.0f;  B->data[3] = 1.0f;
    B->data[4] = 2.0f;  B->data[5] = 3.0f;

    /* Forward: C = A * B */
    RCRE_AGTensor* C = rcre_autograd_matmul(tape, A, B);
    ASSERT_NOT_NULL(C);
    ASSERT_UINT_EQ(2, C->shape[0]);
    ASSERT_UINT_EQ(2, C->shape[1]);

    /* Expected:
       C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
       C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
       C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
       C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    */
    ASSERT_FLOAT_NEAR(31.0f, C->data[0], EPSILON);
    ASSERT_FLOAT_NEAR(19.0f, C->data[1], EPSILON);
    ASSERT_FLOAT_NEAR(85.0f, C->data[2], EPSILON);
    ASSERT_FLOAT_NEAR(55.0f, C->data[3], EPSILON);

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

    rcre_arena_destroy(arena);
}

static void test_tensor_tape_add_bias_and_activations(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 32);
    ASSERT_NOT_NULL(tape);

    /* X: [2, 2], bias: [2] */
    RCRE_AGTensor* X = rcre_ag_tensor_create_2d(arena, 2, 2);
    RCRE_AGTensor* bias = rcre_ag_tensor_create_1d(arena, 2);

    X->data[0] = -2.0f; X->data[1] = 1.0f;
    X->data[2] = 3.0f;  X->data[3] = -4.0f;

    bias->data[0] = 0.5f;
    bias->data[1] = -0.5f;

    /* Add bias */
    RCRE_AGTensor* Z = rcre_autograd_add_bias(tape, X, bias);
    /* Z: [-1.5, 0.5; 3.5, -4.5] */
    ASSERT_FLOAT_NEAR(-1.5f, Z->data[0], EPSILON);
    ASSERT_FLOAT_NEAR(0.5f,  Z->data[1], EPSILON);
    ASSERT_FLOAT_NEAR(3.5f,  Z->data[2], EPSILON);
    ASSERT_FLOAT_NEAR(-4.5f, Z->data[3], EPSILON);

    /* ReLU */
    RCRE_AGTensor* A = rcre_autograd_relu(tape, Z);
    /* A: [0.0, 0.5; 3.5, 0.0] */
    ASSERT_FLOAT_NEAR(0.0f, A->data[0], EPSILON);
    ASSERT_FLOAT_NEAR(0.5f, A->data[1], EPSILON);
    ASSERT_FLOAT_NEAR(3.5f, A->data[2], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, A->data[3], EPSILON);

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

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Dynamic Tensor Tape Autograd (AUTOGRAD-TENSOR-01)");
    TEST_RUN(test_tensor_tape_matmul_and_backward);
    TEST_RUN(test_tensor_tape_add_bias_and_activations);
    return test_suite_report();
}
