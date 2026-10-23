/**
 * @file test_gbdt.c
 * @brief Unit tests for RCRE Flat-Array GBDT Inference Engine
 */

#include "rcre.h"
#include "test_framework.h"

#define EPSILON 1e-4f

static void test_gbdt_ensemble_inference(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Construct 2-tree ensemble */
    /* Tree 0: root at 0, leaves at 1 and 2 */
    /* Tree 1: root at 3, leaves at 4 and 5 */
    RCRE_TreeNode nodes[6] = {
        /* Tree 0 */
        {.feature_idx = 0, .threshold = 0.5f, .left_child = 1, .right_child = 2, .leaf_value = 0.0f, .is_leaf = false},
        {.feature_idx = 0, .threshold = 0.0f, .left_child = 0, .right_child = 0, .leaf_value = -1.0f, .is_leaf = true},
        {.feature_idx = 0, .threshold = 0.0f, .left_child = 0, .right_child = 0, .leaf_value = 1.0f, .is_leaf = true},

        /* Tree 1 */
        {.feature_idx = 1, .threshold = 0.5f, .left_child = 4, .right_child = 5, .leaf_value = 0.0f, .is_leaf = false},
        {.feature_idx = 0, .threshold = 0.0f, .left_child = 0, .right_child = 0, .leaf_value = -0.5f, .is_leaf = true},
        {.feature_idx = 0, .threshold = 0.0f, .left_child = 0, .right_child = 0, .leaf_value = 0.5f, .is_leaf = true}
    };

    size_t tree_roots[2] = {0, 3};

    RCRE_GBDTModel model = {
        .nodes = nodes,
        .tree_roots = tree_roots,
        .num_trees = 2,
        .num_nodes = 6,
        .base_score = 0.0f,
        .is_binary_classification = true
    };

    const size_t N = 4;
    const size_t D = 2;

    size_t X_shape[2] = {N, D};
    RCRE_Tensor X = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float X_raw[8] = {
        0.2f, 0.2f,  /* expected raw score: -1.0 + (-0.5) = -1.5 */
        0.8f, 0.8f,  /* expected raw score: +1.0 + (+0.5) = +1.5 */
        0.2f, 0.8f,  /* expected raw score: -1.0 + (+0.5) = -0.5 */
        0.8f, 0.2f   /* expected raw score: +1.0 + (-0.5) = +0.5 */
    };
    memcpy(X.data, X_raw, sizeof(X_raw));

    size_t P_shape[2] = {N, 1};
    RCRE_Tensor preds = rcre_tensor_create(P_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    int status = rcre_ml_gbdt_predict(&model, &X, &preds);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* p = (float*)preds.data;

    float exp_0 = 1.0f / (1.0f + expf(-(-1.5f)));
    float exp_1 = 1.0f / (1.0f + expf(-(1.5f)));
    float exp_2 = 1.0f / (1.0f + expf(-(-0.5f)));
    float exp_3 = 1.0f / (1.0f + expf(-(0.5f)));

    ASSERT_FLOAT_NEAR(exp_0, p[0], EPSILON);
    ASSERT_FLOAT_NEAR(exp_1, p[1], EPSILON);
    ASSERT_FLOAT_NEAR(exp_2, p[2], EPSILON);
    ASSERT_FLOAT_NEAR(exp_3, p[3], EPSILON);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Flat-Array GBDT Inference Engine (ML-REQ-03)");
    TEST_RUN(test_gbdt_ensemble_inference);
    return test_suite_report();
}
