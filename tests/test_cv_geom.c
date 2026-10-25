/**
 * @file test_cv_geom.c
 * @brief Unit tests for RCRE Continuous Resizing & Geometric Transforms
 */

#include "rcre.h"
#include "test_framework.h"
#include <math.h>

#define EPSILON 1e-4f

static void test_resize_identity(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t shape[3] = {4, 4, 3};
    RCRE_Tensor src = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor dst = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(src.data);
    ASSERT_NOT_NULL(dst.data);

    float* in_data = (float*)src.data;
    for (size_t i = 0; i < 4 * 4 * 3; ++i) {
        in_data[i] = (float)i * 0.05f;
    }

    /* Test Bilinear identity */
    int status_bl = rcre_cv_resize_bilinear(&src, &dst);
    ASSERT_INT_EQ(RCRE_OK, status_bl);

    float* out_data = (float*)dst.data;
    for (size_t i = 0; i < 4 * 4 * 3; ++i) {
        ASSERT_FLOAT_NEAR(in_data[i], out_data[i], EPSILON);
    }

    /* Test Nearest-Neighbor identity */
    int status_nn = rcre_cv_resize_nn(&src, &dst);
    ASSERT_INT_EQ(RCRE_OK, status_nn);
    for (size_t i = 0; i < 4 * 4 * 3; ++i) {
        ASSERT_FLOAT_NEAR(in_data[i], out_data[i], EPSILON);
    }

    rcre_arena_destroy(arena);
}

static void test_resize_bilinear_2x2_to_4x4(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t in_shape[2] = {2, 2};
    size_t out_shape[2] = {4, 4};
    RCRE_Tensor src = rcre_tensor_create(in_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor dst = rcre_tensor_create(out_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(src.data);
    ASSERT_NOT_NULL(dst.data);

    float* in_data = (float*)src.data;
    /* Top row: 0.0, 1.0; Bottom row: 0.0, 1.0 */
    in_data[0] = 0.0f; in_data[1] = 1.0f;
    in_data[2] = 0.0f; in_data[3] = 1.0f;

    int status = rcre_cv_resize_bilinear(&src, &dst);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* out_data = (float*)dst.data;
    /* For subpixel continuous mapping: x_src = (i + 0.5) * (2 / 4) - 0.5 = 0.5*i - 0.25 */
    /* Check monotonic horizontal gradient */
    for (size_t y = 0; y < 4; ++y) {
        ASSERT_TRUE(out_data[y * 4 + 0] <= out_data[y * 4 + 1]);
        ASSERT_TRUE(out_data[y * 4 + 1] <= out_data[y * 4 + 2]);
        ASSERT_TRUE(out_data[y * 4 + 2] <= out_data[y * 4 + 3]);
    }

    rcre_arena_destroy(arena);
}

static void test_color_hsv_roundtrip(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t shape[3] = {2, 2, 3};
    RCRE_Tensor rgb = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor hsv = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor rgb_back = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(rgb.data);
    ASSERT_NOT_NULL(hsv.data);
    ASSERT_NOT_NULL(rgb_back.data);

    float* in = (float*)rgb.data;
    /* Pure Red */
    in[0] = 1.0f; in[1] = 0.0f; in[2] = 0.0f;
    /* Pure Green */
    in[3] = 0.0f; in[4] = 1.0f; in[5] = 0.0f;
    /* Pure Blue */
    in[6] = 0.0f; in[7] = 0.0f; in[8] = 1.0f;
    /* Custom Color */
    in[9] = 0.8f; in[10] = 0.4f; in[11] = 0.2f;

    int s1 = rcre_cv_rgb_to_hsv(&rgb, &hsv);
    ASSERT_INT_EQ(RCRE_OK, s1);

    int s2 = rcre_cv_hsv_to_rgb(&hsv, &rgb_back);
    ASSERT_INT_EQ(RCRE_OK, s2);

    float* out = (float*)rgb_back.data;
    for (size_t i = 0; i < 12; ++i) {
        ASSERT_FLOAT_NEAR(in[i], out[i], 1e-3f);
    }

    /* Channel scaling and clamp tests */
    (void)rcre_cv_scale_channel(&rgb_back, 0, 2.0f);
    (void)rcre_cv_clamp(&rgb_back, 0.0f, 1.0f);
    ASSERT_TRUE(out[0] <= 1.0f);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Image Resizing & Color Spaces (CV-REQ-02)");
    TEST_RUN(test_resize_identity);
    TEST_RUN(test_resize_bilinear_2x2_to_4x4);
    TEST_RUN(test_color_hsv_roundtrip);
    return test_suite_report();
}
