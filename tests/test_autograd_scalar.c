/**
 * @file test_autograd_scalar.c
 * @brief Unit tests for RCRE Scalar Reverse-Mode Autograd Engine (Micrograd-style)
 */

#include "rcre.h"
#include "rcre_autograd.h"
#include "test_framework.h"
#include <math.h>

#define EPSILON 1e-4

static void test_scalar_arithmetic_and_backward(void) {
    RCRE_GraphCtx* ctx = rcre_graph_ctx_create(256 * 1024, 256 * 1024);
    ASSERT_NOT_NULL(ctx);

    /* Construct f(a, b, c) = (a + b) * c */
    RCRE_ScalarVal* a = rcre_scalar_val_param(ctx, 2.0, "a");
    RCRE_ScalarVal* b = rcre_scalar_val_param(ctx, 3.0, "b");
    RCRE_ScalarVal* c = rcre_scalar_val_param(ctx, 4.0, "c");
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(c);

    RCRE_ScalarVal* sum = rcre_scalar_add(ctx, a, b);
    RCRE_ScalarVal* out = rcre_scalar_mul(ctx, sum, c);
    ASSERT_NOT_NULL(out);

    /* (2 + 3) * 4 = 20 */
    ASSERT_DOUBLE_NEAR(20.0, out->data, EPSILON);

    /* Backprop */
    rcre_scalar_backward(out);

    /* dout/da = c = 4.0, dout/db = c = 4.0, dout/dc = a + b = 5.0 */
    ASSERT_DOUBLE_NEAR(4.0, a->grad, EPSILON);
    ASSERT_DOUBLE_NEAR(4.0, b->grad, EPSILON);
    ASSERT_DOUBLE_NEAR(5.0, c->grad, EPSILON);

    /* Zero grad */
    rcre_scalar_zero_grad(out);
    ASSERT_DOUBLE_NEAR(0.0, a->grad, EPSILON);
    ASSERT_DOUBLE_NEAR(0.0, b->grad, EPSILON);
    ASSERT_DOUBLE_NEAR(0.0, c->grad, EPSILON);

    rcre_graph_ctx_destroy(ctx);
}

static void test_scalar_transcendental_ops(void) {
    RCRE_GraphCtx* ctx = rcre_graph_ctx_create(256 * 1024, 256 * 1024);
    ASSERT_NOT_NULL(ctx);

    /* Test Sigmoid: sigma(0.0) = 0.5, d/dx = 0.5 * 0.5 = 0.25 */
    RCRE_ScalarVal* x = rcre_scalar_val_param(ctx, 0.0, "x");
    RCRE_ScalarVal* sig = rcre_scalar_sigmoid(ctx, x);
    ASSERT_DOUBLE_NEAR(0.5, sig->data, EPSILON);

    rcre_scalar_backward(sig);
    ASSERT_DOUBLE_NEAR(0.25, x->grad, EPSILON);

    /* Reset tape */
    rcre_graph_ctx_reset_tape(ctx);

    /* Test ReLU */
    RCRE_ScalarVal* x_pos = rcre_scalar_val_param(ctx, 2.5, "x_pos");
    RCRE_ScalarVal* relu_pos = rcre_scalar_relu(ctx, x_pos);
    ASSERT_DOUBLE_NEAR(2.5, relu_pos->data, EPSILON);
    rcre_scalar_backward(relu_pos);
    ASSERT_DOUBLE_NEAR(1.0, x_pos->grad, EPSILON);

    RCRE_ScalarVal* x_neg = rcre_scalar_val_param(ctx, -1.5, "x_neg");
    RCRE_ScalarVal* relu_neg = rcre_scalar_relu(ctx, x_neg);
    ASSERT_DOUBLE_NEAR(0.0, relu_neg->data, EPSILON);
    rcre_scalar_backward(relu_neg);
    ASSERT_DOUBLE_NEAR(0.0, x_neg->grad, EPSILON);

    rcre_graph_ctx_destroy(ctx);
}

static void test_scalar_graphviz_export(void) {
    RCRE_GraphCtx* ctx = rcre_graph_ctx_create(128 * 1024, 128 * 1024);
    ASSERT_NOT_NULL(ctx);

    RCRE_ScalarVal* a = rcre_scalar_val_param(ctx, 1.5, "x");
    RCRE_ScalarVal* b = rcre_scalar_exp(ctx, a);
    RCRE_ScalarVal* c = rcre_scalar_log(ctx, b);
    rcre_scalar_backward(c);

    bool export_ok = rcre_scalar_export_dot(c, "/tmp/rcre_test_autograd.dot");
    ASSERT_TRUE(export_ok);

    rcre_graph_ctx_destroy(ctx);
}

int main(void) {
    test_suite_begin("RCRE Scalar Reverse-Mode Autograd (AUTOGRAD-SCALAR-01)");
    TEST_RUN(test_scalar_arithmetic_and_backward);
    TEST_RUN(test_scalar_transcendental_ops);
    TEST_RUN(test_scalar_graphviz_export);
    return test_suite_report();
}
