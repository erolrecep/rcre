/**
 * @file test_cv_filters.c
 * @brief Unit tests for RCRE Computer Vision Filters (Color, Gaussian, Sobel)
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-4f

static void test_color_rgb_to_gray(void) {
    printf("  -> Testing RGB to Grayscale Rec.601 conversion...\n");
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    size_t rgb_shape[3] = {2, 2, 3};
    RCRE_Tensor rgb_tensor = rcre_tensor_create(rgb_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);

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

    int status = rcre_cv_rgb_to_gray(&rgb_tensor, &gray_tensor);
    assert(status == RCRE_OK);

    uint8_t* gray_data = (uint8_t*)gray_tensor.data;
    /* Rec.601: 0.299 * R + 0.587 * G + 0.114 * B */
    /* Red: 0.299 * 255 = 76.245 -> 76 */
    /* Green: 0.587 * 255 = 149.685 -> 150 */
    /* Blue: 0.114 * 255 = 29.07 -> 29 */
    /* White: 255 */
    assert(gray_data[0] == 76);
    assert(gray_data[1] == 150);
    assert(gray_data[2] == 29);
    assert(gray_data[3] == 255);

    rcre_arena_destroy(arena);
    printf("     ✓ Rec.601 exact integer quantization verified\n");
}

static void test_color_bgr_to_rgb(void) {
    printf("  -> Testing BGR to RGB channel swap...\n");
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    size_t shape[3] = {1, 1, 3};
    RCRE_Tensor bgr = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor rgb = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);

    uint8_t* bgr_data = (uint8_t*)bgr.data;
    bgr_data[0] = 10;
    bgr_data[1] = 20;
    bgr_data[2] = 30;

    int status = rcre_cv_bgr_to_rgb(&bgr, &rgb);
    assert(status == RCRE_OK);

    uint8_t* rgb_data = (uint8_t*)rgb.data;
    assert(rgb_data[0] == 30);
    assert(rgb_data[1] == 20);
    assert(rgb_data[2] == 10);

    rcre_arena_destroy(arena);
    printf("     ✓ Channel swap BGR -> RGB verified\n");
}

static void test_gaussian_blur(void) {
    printf("  -> Testing 2D Gaussian blur with reflection padding...\n");
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    const size_t H = 16;
    const size_t W = 16;
    size_t shape[2] = {H, W};

    RCRE_Tensor src = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor dst = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float* src_data = (float*)src.data;
    for (size_t i = 0; i < H * W; ++i) {
        src_data[i] = 42.0f;
    }

    int status = rcre_cv_gaussian_blur(&src, &dst, 1.5f, arena);
    assert(status == RCRE_OK);

    float* dst_data = (float*)dst.data;
    for (size_t i = 0; i < H * W; ++i) {
        float diff = fabsf(dst_data[i] - 42.0f);
        assert(diff < EPSILON);
    }

    rcre_arena_destroy(arena);
    printf("     ✓ Constant image invariant preserved under Gaussian blur\n");
}

static void test_sobel_derivatives(void) {
    printf("  -> Testing 3x3 Sobel gradient operators (dx, dy)...\n");
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    const size_t H = 8;
    const size_t W = 8;
    size_t shape[2] = {H, W};

    RCRE_Tensor src = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor dst_dx = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor dst_dy = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float* src_data = (float*)src.data;
    /* Horizontal linear ramp: I(x, y) = 2.0 * x */
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            src_data[y * W + x] = 2.0f * (float)x;
        }
    }

    int status = rcre_cv_sobel(&src, &dst_dx, &dst_dy, arena);
    assert(status == RCRE_OK);

    float* dx_data = (float*)dst_dx.data;
    float* dy_data = (float*)dst_dy.data;

    /* In the interior (x in [1, W-2]), dx for ramp I(x)=2*x with Sobel [-1 0 1; -2 0 2; -1 0 1] is:
     * (2*(x+1) - 2*(x-1)) + 2*(2*(x+1) - 2*(x-1)) + (2*(x+1) - 2*(x-1))
     * = 4 + 8 + 4 = 16.0f
     * And dy should be 0.0f everywhere because intensity is constant along y.
     */
    for (size_t y = 1; y < H - 1; ++y) {
        for (size_t x = 1; x < W - 1; ++x) {
            float dx_val = dx_data[y * W + x];
            float dy_val = dy_data[y * W + x];
            assert(fabsf(dx_val - 16.0f) < EPSILON);
            assert(fabsf(dy_val - 0.0f) < EPSILON);
        }
    }

    rcre_arena_destroy(arena);
    printf("     ✓ Sobel horizontal ramp dx=16.0, dy=0.0 verified\n");
}

int main(void) {
    printf("🧪 Running RCRE Computer Vision Filter Tests (CV-REQ-01 to CV-REQ-03)...\n");
    test_color_rgb_to_gray();
    test_color_bgr_to_rgb();
    test_gaussian_blur();
    test_sobel_derivatives();
    printf("✅ All Computer Vision Filter Tests Passed Successfully!\n");
    return 0;
}
