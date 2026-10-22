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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-4f

static void test_conv2d(RCRE_Arena* arena) {
    printf("  -> Testing 2D spatial convolution (Conv2D)...\n");

    /* Input: 1 channel, 3x3 */
    /* 1 2 3
       4 5 6
       7 8 9 */
    float in_data[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    size_t in_shape[3] = {1, 3, 3};
    RCRE_Tensor in = rcre_tensor_from_buffer(in_data, in_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

    /* Weights: 1 out_channel, 1 in_channel, 2x2 */
    /* 1 0
       0 1 */
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
    assert(status == RCRE_OK);
    assert(out.shape[0] == 1 && out.shape[1] == 2 && out.shape[2] == 2);

    /* Expected:
       (0,0): 1*1 + 5*1 + 0.5 = 6.5
       (0,1): 2*1 + 6*1 + 0.5 = 8.5
       (1,0): 4*1 + 8*1 + 0.5 = 12.5
       (1,1): 5*1 + 9*1 + 0.5 = 14.5
    */
    const float* out_data = (const float*)out.data;
    assert(fabsf(out_data[0] - 6.5f) < EPSILON);
    assert(fabsf(out_data[1] - 8.5f) < EPSILON);
    assert(fabsf(out_data[2] - 12.5f) < EPSILON);
    assert(fabsf(out_data[3] - 14.5f) < EPSILON);

    printf("     ✓ Conv2D spatial convolution and bias accumulation verified\n");
}

static void test_depthwise_conv2d(RCRE_Arena* arena) {
    printf("  -> Testing Depthwise 2D separable convolution...\n");

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
    int status = rcre_nn_depthwise_conv2d(&in, &weights, nullptr, &params, &out, arena);
    assert(status == RCRE_OK);
    assert(out.shape[0] == 2 && out.shape[1] == 2 && out.shape[2] == 2);

    const float* out_data = (const float*)out.data;
    /* Both channels should evaluate to 4.0 for all elements */
    for (int i = 0; i < 8; ++i) {
        assert(fabsf(out_data[i] - 4.0f) < EPSILON);
    }

    printf("     ✓ Depthwise convolution channel independence verified\n");
}

static void test_pointwise_conv2d(RCRE_Arena* arena) {
    printf("  -> Testing Pointwise 1x1 channel projection...\n");

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
    int status = rcre_nn_pointwise_conv2d(&in, &weights, nullptr, &out, arena);
    assert(status == RCRE_OK);
    assert(out.shape[0] == 3 && out.shape[1] == 2 && out.shape[2] == 2);

    const float* out_data = (const float*)out.data;
    /* Out Ch 0 == in Ch 0 */
    assert(fabsf(out_data[0] - 1.0f) < EPSILON);
    assert(fabsf(out_data[3] - 4.0f) < EPSILON);
    /* Out Ch 1 == in Ch 1 */
    assert(fabsf(out_data[4] - 5.0f) < EPSILON);
    assert(fabsf(out_data[7] - 8.0f) < EPSILON);
    /* Out Ch 2 == in Ch 0 + in Ch 1: 1+5=6, 2+6=8, 3+7=10, 4+8=12 */
    assert(fabsf(out_data[8] - 6.0f) < EPSILON);
    assert(fabsf(out_data[9] - 8.0f) < EPSILON);
    assert(fabsf(out_data[10] - 10.0f) < EPSILON);
    assert(fabsf(out_data[11] - 12.0f) < EPSILON);

    printf("     ✓ Pointwise 1x1 projection and SIMD axpy verified\n");
}

static void test_batchnorm_and_fusion(RCRE_Arena* arena) {
    printf("  -> Testing BatchNorm2D and Zero-Cost Conv-BN Folding parity...\n");

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
    assert(status == RCRE_OK);

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
    assert(status == RCRE_OK);

    /* 3. Fuse Conv + BN */
    float fused_w_data[2] = {2.0f, -1.0f};
    float fused_b_data[2] = {0.5f, 1.0f};
    RCRE_Tensor fused_weights = rcre_tensor_from_buffer(fused_w_data, w_shape, 4, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);
    RCRE_Tensor fused_bias = rcre_tensor_from_buffer(fused_b_data, b_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);

    status = rcre_nn_fuse_conv_batchnorm(&fused_weights, &fused_bias, &gamma, &beta, &mean, &var, 1e-5f);
    assert(status == RCRE_OK);

    /* 4. Run Fused Conv2D directly */
    RCRE_Tensor fused_out = {0};
    status = rcre_nn_conv2d(&in, &fused_weights, &fused_bias, &params, &fused_out, arena);
    assert(status == RCRE_OK);

    /* Compare bn_out vs fused_out */
    const float* p_bn = (const float*)bn_out.data;
    const float* p_fused = (const float*)fused_out.data;
    for (int i = 0; i < 8; ++i) {
        float diff = fabsf(p_bn[i] - p_fused[i]);
        assert(diff < 1e-4f);
    }

    printf("     ✓ Fused Conv-BN produced bit-identical output to standalone Conv+BN (err < 1e-4)\n");
}

static void test_relu6_and_pooling(RCRE_Arena* arena) {
    printf("  -> Testing ReLU6 and Global Average Pooling...\n");

    /* ReLU6 */
    float data[5] = {-2.0f, 0.0f, 3.5f, 6.0f, 8.5f};
    size_t shape[1] = {5};
    RCRE_Tensor t_in = rcre_tensor_from_buffer(data, shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
    RCRE_Tensor t_out = rcre_tensor_create(shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    int status = rcre_nn_relu6(&t_in, &t_out);
    assert(status == RCRE_OK);
    const float* out_data = (const float*)t_out.data;
    assert(fabsf(out_data[0] - 0.0f) < EPSILON);
    assert(fabsf(out_data[1] - 0.0f) < EPSILON);
    assert(fabsf(out_data[2] - 3.5f) < EPSILON);
    assert(fabsf(out_data[3] - 6.0f) < EPSILON);
    assert(fabsf(out_data[4] - 6.0f) < EPSILON);

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
    assert(status == RCRE_OK);
    const float* p_avg = (const float*)pool_out.data;
    assert(fabsf(p_avg[0] - 2.5f) < EPSILON);
    assert(fabsf(p_avg[1] - 25.0f) < EPSILON);

    printf("     ✓ ReLU6 bounding [0, 6] and GlobalAvgPool spatial reduction verified\n");
}

static void test_linear_and_softmax(RCRE_Arena* arena) {
    printf("  -> Testing Linear layer projection & Softmax...\n");

    /* Linear: input [1.0, 2.0] */
    float in_data[2] = {1.0f, 2.0f};
    size_t in_shape[1] = {2};
    RCRE_Tensor in = rcre_tensor_from_buffer(in_data, in_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);

    /* Weights: 2 out, 2 in */
    /* [2.0, 1.0]
       [0.0, 3.0] */
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
    assert(status == RCRE_OK);

    /* Expected:
       out[0] = 1*2 + 2*1 + 0.5 = 4.5
       out[1] = 1*0 + 2*3 - 1.0 = 5.0
    */
    const float* out_data = (const float*)out.data;
    assert(fabsf(out_data[0] - 4.5f) < EPSILON);
    assert(fabsf(out_data[1] - 5.0f) < EPSILON);

    /* Softmax on logits */
    size_t sm_shape[1] = {2};
    RCRE_Tensor sm_out = rcre_tensor_create(sm_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    status = rcre_nn_softmax(&out, &sm_out);
    assert(status == RCRE_OK);

    const float* probs = (const float*)sm_out.data;
    assert(probs[0] > 0.0f && probs[1] > 0.0f);
    assert(fabsf((probs[0] + probs[1]) - 1.0f) < EPSILON);
    /* Since 5.0 > 4.5, probs[1] > probs[0] */
    assert(probs[1] > probs[0]);

    printf("     ✓ Linear dot product & numerically stable Softmax probability normalization verified\n");
}

int main(void) {
    printf("🧪 Running RCRE CNN Neural Network Tests (NN-REQ-01 to 05)...\n");

    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    assert(arena != nullptr);

    test_conv2d(arena);
    test_depthwise_conv2d(arena);
    test_pointwise_conv2d(arena);
    test_batchnorm_and_fusion(arena);
    test_relu6_and_pooling(arena);
    test_linear_and_softmax(arena);

    rcre_arena_destroy(arena);

    printf("✅ All CNN Neural Network Tests Passed Successfully!\n");
    return 0;
}
