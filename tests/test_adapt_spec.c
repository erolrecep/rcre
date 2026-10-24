/**
 * @file test_adapt_spec.c
 * @brief Unit tests for RCRE input adaptation pipeline (rcre_tensor_adapt)
 */

#include "rcre.h"
#include "test_framework.h"

static void test_adapt_letterbox_and_normalization(void) {
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Create 100x200 3-channel image (aspect ratio 2:1) */
    size_t in_shape[3] = {100, 200, 3};
    RCRE_Tensor src = rcre_tensor_create(in_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(src.data);

    /* Fill image with constant color (120, 150, 180) */
    uint8_t* raw = (uint8_t*)src.data;
    for (size_t y = 0; y < 100; ++y) {
        for (size_t x = 0; x < 200; ++x) {
            size_t idx = (y * 200 + x) * 3;
            raw[idx + 0] = 120;
            raw[idx + 1] = 150;
            raw[idx + 2] = 180;
        }
    }

    /* Adapt to square [1, 3, 224, 224] with letterboxing and ImageNet normalization */
    RCRE_PreprocessSpec spec = {
        .target_shape = {1, 3, 224, 224},
        .target_layout = RCRE_LAYOUT_CHW,
        .target_dtype = RCRE_DTYPE_F32,
        .mean = {123.675f, 116.28f, 103.53f, 0.0f},
        .std  = {58.395f, 57.12f, 57.375f, 1.0f},
        .preserve_aspect = true,
        .pad_value = 0.0f
    };

    RCRE_Tensor dst = {0};
    int status = rcre_tensor_adapt(&src, &spec, arena, &dst);
    ASSERT_INT_EQ(RCRE_OK, status);
    ASSERT_NOT_NULL(dst.data);
    ASSERT_UINT_EQ(4, dst.ndim);
    ASSERT_UINT_EQ(1, dst.shape[0]);
    ASSERT_UINT_EQ(3, dst.shape[1]);
    ASSERT_UINT_EQ(224, dst.shape[2]);
    ASSERT_UINT_EQ(224, dst.shape[3]);
    ASSERT_INT_EQ(RCRE_LAYOUT_CHW, dst.layout);
    ASSERT_INT_EQ(RCRE_DTYPE_F32, dst.dtype);

    float* out = (float*)dst.data;

    /* Top row y=0 is outside the 2:1 centered content, so it should equal normalized pad_value (0 - mean) / std */
    float expected_pad_c0 = (0.0f - spec.mean[0]) / spec.std[0];
    float top_left_c0 = out[(0 * 224 + 0) * 224 + 0];
    ASSERT_FLOAT_NEAR(expected_pad_c0, top_left_c0, 1e-4f);

    /* Center pixel (112, 112) is inside the content, should match normalized 120, 150, 180 */
    float expected_center_c0 = (120.0f - spec.mean[0]) / spec.std[0];
    float expected_center_c1 = (150.0f - spec.mean[1]) / spec.std[1];
    float expected_center_c2 = (180.0f - spec.mean[2]) / spec.std[2];

    float center_c0 = out[(0 * 224 + 112) * 224 + 112];
    float center_c1 = out[(1 * 224 + 112) * 224 + 112];
    float center_c2 = out[(2 * 224 + 112) * 224 + 112];

    ASSERT_FLOAT_NEAR(expected_center_c0, center_c0, 1e-2f);
    ASSERT_FLOAT_NEAR(expected_center_c1, center_c1, 1e-2f);
    ASSERT_FLOAT_NEAR(expected_center_c2, center_c2, 1e-2f);

    rcre_arena_destroy(arena);
}

static void test_adapt_unsupported_dtype_rejected(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t in_shape[2] = {10, 10};
    /* Create I32 tensor (unsupported for visual adaptation) */
    RCRE_Tensor src = rcre_tensor_create(in_shape, 2, RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);
    ASSERT_NOT_NULL(src.data);

    RCRE_PreprocessSpec spec = {
        .target_shape = {1, 1, 10, 10},
        .target_layout = RCRE_LAYOUT_CHW,
        .target_dtype = RCRE_DTYPE_F32,
        .preserve_aspect = false,
        .pad_value = 0.0f
    };

    RCRE_Tensor dst = {0};
    int status = rcre_tensor_adapt(&src, &spec, arena, &dst);
    ASSERT_INT_EQ(RCRE_ERR_UNSUPPORTED_OP, status);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Input Adaptation Pipeline (CORE-REQ-03)");
    TEST_RUN(test_adapt_letterbox_and_normalization);
    TEST_RUN(test_adapt_unsupported_dtype_rejected);
    return test_suite_report();
}
