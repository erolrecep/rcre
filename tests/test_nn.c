/**
 * @file test_nn.c
 * @brief Unit Tests for RCRE Convolutional Neural Network Operators
 * 
 * Verifies NN-REQ-01 through NN-REQ-05:
 * - Conv2D spatial convolution with padding & stride
 * - Depthwise 2D convolution
 * - Pointwise 1x1 channel projection
 * - BatchNorm2D and Zero-Cost BN Folding equivalence
 * - ReLU6 piecewise activation
 * - Global Average Pooling
 * - Linear (Fully Connected) & Numerically Stable Softmax
 */

#include "rcre.h"
#include "test_framework.h"

#define EPSILON 1e-4f

static void test_conv2d(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Input: 1 channel, 3x3 */
    float in_data[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    size_t in_shape[3] = {1, 3, 3};
    RCRE_Tensor in = rcre_tensor_from_buffer(in_data, in_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

    /* Weights: 1 out_channel, 1 in_channel, 2x2 */
    float w_data[4] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    size_t w_shape[4] = {1, 1, 2, 2};
    RCRE_Tensor weights = rcre_tensor_from_buffer(w_data, w_shape, 4, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

    float b_data[1] = {0.5f};
    size_t b_shape[1] = {1};
    RCRE_Tensor bias = rcre_tensor_from_buffer(b_data, b_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);

    RCRE_Conv2DParams params = {
        .stride_h = 1,
        .stride_w = 1,
        .pad_h = 0,
        .pad_w = 0
    };

    RCRE_Tensor out = {0};
    int status = rcre_nn_conv2d(&in, &weights, &bias, &params, &out, arena);
    ASSERT_INT_EQ(RCRE_OK, status);
    ASSERT_UINT_EQ(1, out.shape[0]);
    ASSERT_UINT_EQ(2, out.shape[1]);
    ASSERT_UINT_EQ(2, out.shape[2]);

    /* Expected:
       (0,0): 1*1 + 5*1 + 0.5 = 6.5
       (0,1): 2*1 + 6*1 + 0.5 = 8.5
       (1,0): 4*1 + 8*1 + 0.5 = 12.5
       (1,1): 5*1 + 9*1 + 0.5 = 14.5
    */
    const float* out_data = (const float*)out.data;
    ASSERT_FLOAT_NEAR(6.5f, out_data[0], EPSILON);
    ASSERT_FLOAT_NEAR(8.5f, out_data[1], EPSILON);
    ASSERT_FLOAT_NEAR(12.5f, out_data[2], EPSILON);
    ASSERT_FLOAT_NEAR(14.5f, out_data[3], EPSILON);

    rcre_arena_destroy(arena);
}

static void test_depthwise_conv2d(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 2 channels, 3x3 */
    float in_data[18] = {
        /* Ch 0: all 1s */
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        /* Ch 1: all 2s */
        2.0f, 2.0f, 2.0f,
        2.0f, 2.0f, 2.0f,
        2.0f, 2.0f, 2.0f
    };
    size_t in_shape[3] = {2, 3, 3};
    RCRE_Tensor in = rcre_tensor_from_buffer(in_data, in_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

    /* Weights: 2 channels, 1x2x2 */
    float w_data[8] = {
        /* Ch 0 filter: all 1s -> sum over 2x2 = 4 */
        1.0f, 1.0f,
        1.0f, 1.0f,
        /* Ch 1 filter: all 0.5s -> 2 * 0.5 * 4 = 4 */
        0.5f, 0.5f,
        0.5f, 0.5f
    };
    size_t w_shape[4] = {2, 1, 2, 2};
    RCRE_Tensor weights = rcre_tensor_from_buffer(w_data, w_shape, 4, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

    RCRE_Conv2DParams params = {
        .stride_h = 1,
        .stride_w = 1,
        .pad_h = 0,
        .pad_w = 0
    };

    RCRE_Tensor out = {0};
    int status = rcre_nn_depthwise_conv2d(&in, &weights, NULL, &params, &out, arena);
    ASSERT_INT_EQ(RCRE_OK, status);
    ASSERT_UINT_EQ(2, out.shape[0]);
    ASSERT_UINT_EQ(2, out.shape[1]);
    ASSERT_UINT_EQ(2, out.shape[2]);

    const float* out_data = (const float*)out.data;
    /* Both channels should evaluate to 4.0 for all elements */
    for (int i = 0; i < 8; ++i) {
        ASSERT_FLOAT_NEAR(4.0f, out_data[i], EPSILON);
    }

    rcre_arena_destroy(arena);
}

static void test_pointwise_conv2d(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 2 channels, 2x2 */
    float in_data[8] = {
        /* Ch 0 */
        1.0f, 2.0f,
        3.0f, 4.0f,
        /* Ch 1 */
        5.0f, 6.0f,
        7.0f, 8.0f
    };
    size_t in_shape[3] = {2, 2, 2};
    RCRE_Tensor in = rcre_tensor_from_buffer(in_data, in_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

    /* Weights: 3 out_channels, 2 in_channels, 1x1 */
    float w_data[6] = {
        /* Out Ch 0: 1*Ch0 + 0*Ch1 = Ch0 */
        1.0f, 0.0f,
        /* Out Ch 1: 0*Ch0 + 1*Ch1 = Ch1 */
        0.0f, 1.0f,
        /* Out Ch 2: 1*Ch0 + 1*Ch1 = Ch0 + Ch1 */
        1.0f, 1.0f
    };
    size_t w_shape[4] = {3, 2, 1, 1};
    RCRE_Tensor weights = rcre_tensor_from_buffer(w_data, w_shape, 4, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

    RCRE_Tensor out = {0};
    int status = rcre_nn_pointwise_conv2d(&in, &weights, NULL, &out, arena);
    ASSERT_INT_EQ(RCRE_OK, status);
    ASSERT_UINT_EQ(3, out.shape[0]);
    ASSERT_UINT_EQ(2, out.shape[1]);
    ASSERT_UINT_EQ(2, out.shape[2]);

    const float* out_data = (const float*)out.data;
    /* Out Ch 0 == in Ch 0 */
    ASSERT_FLOAT_NEAR(1.0f, out_data[0], EPSILON);
    ASSERT_FLOAT_NEAR(4.0f, out_data[3], EPSILON);
    /* Out Ch 1 == in Ch 1 */
    ASSERT_FLOAT_NEAR(5.0f, out_data[4], EPSILON);
    ASSERT_FLOAT_NEAR(8.0f, out_data[7], EPSILON);
    /* Out Ch 2 == in Ch 0 + in Ch 1: 1+5=6, 2+6=8, 3+7=10, 4+8=12 */
    ASSERT_FLOAT_NEAR(6.0f, out_data[8], EPSILON);
    ASSERT_FLOAT_NEAR(8.0f, out_data[9], EPSILON);
    ASSERT_FLOAT_NEAR(10.0f, out_data[10], EPSILON);
    ASSERT_FLOAT_NEAR(12.0f, out_data[11], EPSILON);

    rcre_arena_destroy(arena);
}

static void test_batchnorm_and_fusion(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Create 1 channel 2x2 input */
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t in_shape[3] = {1, 2, 2};
    RCRE_Tensor in = rcre_tensor_from_buffer(in_data, in_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

    /* Conv weights: 2 filters of 1x1x1 */
    float w_data[2] = {2.0f, -1.0f};
    size_t w_shape[4] = {2, 1, 1, 1};
    RCRE_Tensor weights = rcre_tensor_from_buffer(w_data, w_shape, 4, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

    float b_data[2] = {0.5f, 1.0f};
    size_t b_shape[1] = {2};
    RCRE_Tensor bias = rcre_tensor_from_buffer(b_data, b_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);

    /* 1. Run Conv2D */
    RCRE_Conv2DParams params = {.stride_h = 1, .stride_w = 1, .pad_h = 0, .pad_w = 0};
    RCRE_Tensor conv_out = {0};
    int status = rcre_nn_conv2d(&in, &weights, &bias, &params, &conv_out, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* 2. Run BatchNorm2D */
    float gamma_data[2] = {1.2f, 0.8f};
    float beta_data[2]  = {0.1f, -0.2f};
    float mean_data[2]  = {3.0f, -2.0f};
    float var_data[2]   = {1.5f, 2.5f};
    size_t bn_shape[1]  = {2};

    RCRE_Tensor gamma = rcre_tensor_from_buffer(gamma_data, bn_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
    RCRE_Tensor beta  = rcre_tensor_from_buffer(beta_data,  bn_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
    RCRE_Tensor mean  = rcre_tensor_from_buffer(mean_data,  bn_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
    RCRE_Tensor var   = rcre_tensor_from_buffer(var_data,   bn_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);

    size_t bn_out_shape[3] = {2, 2, 2};
    RCRE_Tensor bn_out = rcre_tensor_create(bn_out_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW, arena);
    status = rcre_nn_batchnorm2d(&conv_out, &gamma, &beta, &mean, &var, 1e-5f, &bn_out);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* 3. Fuse Conv + BN */
    float fused_w_data[2] = {2.0f, -1.0f};
    float fused_b_data[2] = {0.5f, 1.0f};
    RCRE_Tensor fused_weights = rcre_tensor_from_buffer(fused_w_data, w_shape, 4, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);
    RCRE_Tensor fused_bias = rcre_tensor_from_buffer(fused_b_data, b_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);

    status = rcre_nn_fuse_conv_batchnorm(&fused_weights, &fused_bias, &gamma, &beta, &mean, &var, 1e-5f);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* 4. Run Fused Conv2D directly */
    RCRE_Tensor fused_out = {0};
    status = rcre_nn_conv2d(&in, &fused_weights, &fused_bias, &params, &fused_out, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* Compare bn_out vs fused_out */
    const float* p_bn = (const float*)bn_out.data;
    const float* p_fused = (const float*)fused_out.data;
    ASSERT_FLOAT_ARRAY_NEAR(p_bn, p_fused, 8, 1e-4f);

    rcre_arena_destroy(arena);
}

static void test_relu6_and_pooling(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* ReLU6 */
    float data[5] = {-2.0f, 0.0f, 3.5f, 6.0f, 8.5f};
    size_t shape[1] = {5};
    RCRE_Tensor t_in = rcre_tensor_from_buffer(data, shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
    RCRE_Tensor t_out = rcre_tensor_create(shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    int status = rcre_nn_relu6(&t_in, &t_out);
    ASSERT_INT_EQ(RCRE_OK, status);
    const float* out_data = (const float*)t_out.data;
    ASSERT_FLOAT_NEAR(0.0f, out_data[0], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, out_data[1], EPSILON);
    ASSERT_FLOAT_NEAR(3.5f, out_data[2], EPSILON);
    ASSERT_FLOAT_NEAR(6.0f, out_data[3], EPSILON);
    ASSERT_FLOAT_NEAR(6.0f, out_data[4], EPSILON);

    /* Global Avg Pool */
    float pool_in_data[8] = {
        /* Ch 0: 1, 2, 3, 4 -> mean = 2.5 */
        1.0f, 2.0f, 3.0f, 4.0f,
        /* Ch 1: 10, 20, 30, 40 -> mean = 25.0 */
        10.0f, 20.0f, 30.0f, 40.0f
    };
    size_t pool_in_shape[3] = {2, 2, 2};
    RCRE_Tensor pool_in = rcre_tensor_from_buffer(pool_in_data, pool_in_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);
    size_t pool_out_shape[3] = {2, 1, 1};
    RCRE_Tensor pool_out = rcre_tensor_create(pool_out_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW, arena);

    status = rcre_nn_global_avgpool(&pool_in, &pool_out);
    ASSERT_INT_EQ(RCRE_OK, status);
    const float* p_avg = (const float*)pool_out.data;
    ASSERT_FLOAT_NEAR(2.5f, p_avg[0], EPSILON);
    ASSERT_FLOAT_NEAR(25.0f, p_avg[1], EPSILON);

    rcre_arena_destroy(arena);
}

static void test_linear_and_softmax(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Linear: input [1.0, 2.0] */
    float in_data[2] = {1.0f, 2.0f};
    size_t in_shape[1] = {2};
    RCRE_Tensor in = rcre_tensor_from_buffer(in_data, in_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);

    /* Weights: 2 out, 2 in */
    float w_data[4] = {
        2.0f, 1.0f,
        0.0f, 3.0f
    };
    size_t w_shape[2] = {2, 2};
    RCRE_Tensor weights = rcre_tensor_from_buffer(w_data, w_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);

    float b_data[2] = {0.5f, -1.0f};
    size_t b_shape[1] = {2};
    RCRE_Tensor bias = rcre_tensor_from_buffer(b_data, b_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);

    RCRE_Tensor out = {0};
    int status = rcre_nn_linear(&in, &weights, &bias, &out, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* Expected:
       out[0] = 1*2 + 2*1 + 0.5 = 4.5
       out[1] = 1*0 + 2*3 - 1.0 = 5.0
    */
    const float* out_data = (const float*)out.data;
    ASSERT_FLOAT_NEAR(4.5f, out_data[0], EPSILON);
    ASSERT_FLOAT_NEAR(5.0f, out_data[1], EPSILON);

    /* Softmax on logits */
    size_t sm_shape[1] = {2};
    RCRE_Tensor sm_out = rcre_tensor_create(sm_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    status = rcre_nn_softmax(&out, &sm_out);
    ASSERT_INT_EQ(RCRE_OK, status);

    const float* probs = (const float*)sm_out.data;
    ASSERT_TRUE(probs[0] > 0.0f);
    ASSERT_TRUE(probs[1] > 0.0f);
    ASSERT_FLOAT_NEAR(1.0f, probs[0] + probs[1], EPSILON);
    ASSERT_TRUE(probs[1] > probs[0]);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Convolutional Neural Network Operators (NN-REQ-01 to 05)");
    TEST_RUN(test_conv2d);
    TEST_RUN(test_depthwise_conv2d);
    TEST_RUN(test_pointwise_conv2d);
    TEST_RUN(test_batchnorm_and_fusion);
    TEST_RUN(test_relu6_and_pooling);
    TEST_RUN(test_linear_and_softmax);
    return test_suite_report();
}
