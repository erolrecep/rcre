/**
 * @file test_cv_optical_flow.c
 * @brief Unit tests for Integral Images, Box Filtering, and Lucas-Kanade Optical Flow
 */

#include "rcre.h"
#include "test_framework.h"
#include <math.h>

#define EPSILON 1e-4f

static void test_integral_image_prefix_sum(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const size_t H = 8, W = 8;
    size_t shape[3] = {H, W, 1};
    RCRE_Tensor src = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor integ = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(src.data);
    ASSERT_NOT_NULL(integ.data);

    float* src_data = (float*)src.data;
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            src_data[y * W + x] = (float)(y + x + 1);
        }
    }

    int status = rcre_cv_integral_image(&src, &integ);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* int_data = (float*)integ.data;

    /* Verify prefix-sum identity across every pixel */
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            float expected_sum = 0.0f;
            for (size_t j = 0; j <= y; ++j) {
                for (size_t i = 0; i <= x; ++i) {
                    expected_sum += src_data[j * W + i];
                }
            }
            ASSERT_FLOAT_NEAR(expected_sum, int_data[y * W + x], EPSILON);
        }
    }

    /* Test O(1) box filter vs naive sum */
    RCRE_Tensor box_out = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(box_out.data);
    status = rcre_cv_box_filter_integral(&integ, &box_out, 1);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* box_data = (float*)box_out.data;
    /* For pixel (2, 2) with radius 1: 3x3 window from (1,1) to (3,3) */
    float manual_sum = 0.0f;
    for (size_t j = 1; j <= 3; ++j) {
        for (size_t i = 1; i <= 3; ++i) {
            manual_sum += src_data[j * W + i];
        }
    }
    ASSERT_FLOAT_NEAR(manual_sum / 9.0f, box_data[2 * W + 2], EPSILON);

    rcre_arena_destroy(arena);
}

static void test_lucas_kanade_horizontal_motion(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 32x32 frames with smooth Gaussian spot moving from x=14 to x=16 (vx > 0) */
    const size_t H = 32, W = 32;
    size_t shape[3] = {H, W, 1};
    RCRE_Tensor prev = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor curr = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);

    size_t vel_shape[3] = {H / 2, W / 2, 2};
    RCRE_Tensor vel = rcre_tensor_create(vel_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(prev.data);
    ASSERT_NOT_NULL(curr.data);
    ASSERT_NOT_NULL(vel.data);

    float* p_data = (float*)prev.data;
    float* c_data = (float*)curr.data;

    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            float dx0 = (float)x - 14.0f;
            float dy0 = (float)y - 16.0f;
            p_data[y * W + x] = expf(-(dx0 * dx0 + dy0 * dy0) / 18.0f);

            float dx1 = (float)x - 16.0f;
            float dy1 = (float)y - 16.0f;
            c_data[y * W + x] = expf(-(dx1 * dx1 + dy1 * dy1) / 18.0f);
        }
    }

    int status = rcre_cv_optical_flow_lk(&curr, &prev, &vel, 7, 2, 5.0f, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* v_data = (float*)vel.data;
    size_t mid_y = (H / 2) / 2;
    size_t mid_x = (W / 2) / 2;
    float vx = v_data[(mid_y * (W / 2) + mid_x) * 2 + 0];
    float vy = v_data[(mid_y * (W / 2) + mid_x) * 2 + 1];

    /* Horizontal displacement should be detected as positive x velocity */
    ASSERT_TRUE(vx > 0.05f);
    /* Vertical velocity should be negligible */
    ASSERT_TRUE(fabsf(vy) < 0.5f);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Integral Images & Lucas-Kanade Optical Flow");
    TEST_RUN(test_integral_image_prefix_sum);
    TEST_RUN(test_lucas_kanade_horizontal_motion);
    return test_suite_report();
}
