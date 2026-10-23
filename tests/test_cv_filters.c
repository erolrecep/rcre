/**
 * @file test_cv_filters.c
 * @brief Unit tests for RCRE Computer Vision Filters (Color, Gaussian, Sobel)
 */

#include "rcre.h"
#include "test_framework.h"

#define EPSILON 1e-4f

static void test_color_rgb_to_gray(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t rgb_shape[3] = {2, 2, 3};
    RCRE_Tensor rgb_tensor = rcre_tensor_create(rgb_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(rgb_tensor.data);

    uint8_t* in_data = (uint8_t*)rgb_tensor.data;
    /* (0, 0): Red */
    in_data[0] = 255; in_data[1] = 0;   in_data[2] = 0;
    /* (0, 1): Green */
    in_data[3] = 0;   in_data[4] = 255; in_data[5] = 0;
    /* (1, 0): Blue */
    in_data[6] = 0;   in_data[7] = 0;   in_data[8] = 255;
    /* (1, 1): White */
    in_data[9] = 255; in_data[10] = 255; in_data[11] = 255;

    size_t gray_shape[2] = {2, 2};
    RCRE_Tensor gray_tensor = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_TABULAR, arena);
    ASSERT_NOT_NULL(gray_tensor.data);

    int status = rcre_cv_rgb_to_gray(&rgb_tensor, &gray_tensor);
    ASSERT_INT_EQ(RCRE_OK, status);

    uint8_t* gray_data = (uint8_t*)gray_tensor.data;
    /* Rec.601 integer values: Red->76, Green->150, Blue->29, White->255 */
    uint8_t expected[4] = {76, 150, 29, 255};
    ASSERT_MEM_EQ(expected, gray_data, 4);

    rcre_arena_destroy(arena);
}

static void test_color_bgr_to_rgb(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t shape[3] = {1, 1, 3};
    RCRE_Tensor bgr = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor rgb = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(bgr.data);
    ASSERT_NOT_NULL(rgb.data);

    uint8_t* bgr_data = (uint8_t*)bgr.data;
    bgr_data[0] = 10;
    bgr_data[1] = 20;
    bgr_data[2] = 30;

    int status = rcre_cv_bgr_to_rgb(&bgr, &rgb);
    ASSERT_INT_EQ(RCRE_OK, status);

    uint8_t expected_rgb[3] = {30, 20, 10};
    ASSERT_MEM_EQ(expected_rgb, rgb.data, 3);

    rcre_arena_destroy(arena);
}

static void test_gaussian_blur(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const size_t H = 16;
    const size_t W = 16;
    size_t shape[2] = {H, W};

    RCRE_Tensor src = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor dst = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    ASSERT_NOT_NULL(src.data);
    ASSERT_NOT_NULL(dst.data);

    float* src_data = (float*)src.data;
    for (size_t i = 0; i < H * W; ++i) {
        src_data[i] = 42.0f;
    }

    int status = rcre_cv_gaussian_blur(&src, &dst, 1.5f, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* dst_data = (float*)dst.data;
    for (size_t i = 0; i < H * W; ++i) {
        ASSERT_FLOAT_NEAR(42.0f, dst_data[i], EPSILON);
    }

    rcre_arena_destroy(arena);
}

static void test_sobel_derivatives(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const size_t H = 8;
    const size_t W = 8;
    size_t shape[2] = {H, W};

    RCRE_Tensor src = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor dst_dx = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor dst_dy = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    ASSERT_NOT_NULL(src.data);
    ASSERT_NOT_NULL(dst_dx.data);
    ASSERT_NOT_NULL(dst_dy.data);

    float* src_data = (float*)src.data;
    /* Horizontal linear ramp: I(x, y) = 2.0 * x */
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            src_data[y * W + x] = 2.0f * (float)x;
        }
    }

    int status = rcre_cv_sobel(&src, &dst_dx, &dst_dy, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* dx_data = (float*)dst_dx.data;
    float* dy_data = (float*)dst_dy.data;

    for (size_t y = 1; y < H - 1; ++y) {
        for (size_t x = 1; x < W - 1; ++x) {
            ASSERT_FLOAT_NEAR(16.0f, dx_data[y * W + x], EPSILON);
            ASSERT_FLOAT_NEAR(0.0f, dy_data[y * W + x], EPSILON);
        }
    }

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Computer Vision Filters (CV-REQ-01 to 03)");
    TEST_RUN(test_color_rgb_to_gray);
    TEST_RUN(test_color_bgr_to_rgb);
    TEST_RUN(test_gaussian_blur);
    TEST_RUN(test_sobel_derivatives);
    return test_suite_report();
}
