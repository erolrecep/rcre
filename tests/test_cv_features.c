/**
 * @file test_cv_features.c
 * @brief Unit tests for Harris Corner Detection, NMS, and Feature Matching
 */

#include "rcre.h"
#include "test_framework.h"
#include <math.h>

#define EPSILON 1e-4f

static void test_harris_corners_synthetic_box(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 32x32 image with a 12x12 white box in the center [10..21, 10..21] */
    const size_t H = 32, W = 32;
    size_t shape[3] = {H, W, 1};
    RCRE_Tensor img = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor resp = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor nms = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(img.data);
    ASSERT_NOT_NULL(resp.data);
    ASSERT_NOT_NULL(nms.data);

    float* data = (float*)img.data;
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            if (y >= 10 && y <= 21 && x >= 10 && x <= 21) {
                data[y * W + x] = 1.0f;
            } else {
                data[y * W + x] = 0.0f;
            }
        }
    }

    int status = rcre_cv_harris_response(&img, &resp, 1.0f, 0.06f, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* r_data = (float*)resp.data;

    /* Flat interior should have near zero response */
    ASSERT_TRUE(fabsf(r_data[15 * W + 15]) < 0.1f);

    /* Straight edge should not be a corner (trace^2 dominates det) */
    ASSERT_TRUE(r_data[10 * W + 15] <= 0.0f || fabsf(r_data[10 * W + 15]) < 1.0f);

    /* The 4 corners (10,10), (10,21), (21,10), (21,21) should have significant positive responses */
    float c00 = r_data[10 * W + 10];
    float c01 = r_data[10 * W + 21];
    float c10 = r_data[21 * W + 10];
    float c11 = r_data[21 * W + 21];

    ASSERT_TRUE(c00 > 0.0f);
    ASSERT_TRUE(c01 > 0.0f);
    ASSERT_TRUE(c10 > 0.0f);
    ASSERT_TRUE(c11 > 0.0f);

    /* NMS test */
    status = rcre_cv_nms_2d(&resp, &nms, 2, 0.01f);
    ASSERT_INT_EQ(RCRE_OK, status);

    float* nms_data = (float*)nms.data;
    /* Verified that corner locations or their immediate neighbors survive NMS */
    int corner_count = 0;
    for (size_t y = 8; y <= 23; ++y) {
        for (size_t x = 8; x <= 23; ++x) {
            if (nms_data[y * W + x] > 0.05f) {
                corner_count++;
            }
        }
    }
    ASSERT_TRUE(corner_count >= 4);

    rcre_arena_destroy(arena);
}

static void test_descriptor_matching_self(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const size_t H = 32, W = 32;
    size_t shape[3] = {H, W, 1};
    RCRE_Tensor img = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(img.data);

    float* data = (float*)img.data;
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            data[y * W + x] = ((y >= 8 && y <= 16 && x >= 8 && x <= 16) ||
                               (y >= 20 && y <= 26 && x >= 20 && x <= 26)) ? 1.0f : 0.0f;
        }
    }

    RCRE_CV_Descriptor* desc_a = NULL;
    int count_a = 0;
    int status = rcre_cv_harris_detect(&img, 1.0f, 0.05f, 2, &desc_a, &count_a, arena);
    ASSERT_INT_EQ(RCRE_OK, status);
    ASSERT_TRUE(count_a > 0);

    /* Match descriptors against themselves */
    RCRE_CV_Match* matches = NULL;
    int match_count = 0;
    status = rcre_cv_match_descriptors(desc_a, count_a, desc_a, count_a, 1.0f, &matches, &match_count, arena);
    ASSERT_INT_EQ(RCRE_OK, status);
    ASSERT_TRUE(match_count > 0);

    /* Diagonal self-matches must have distance 0.0 */
    for (int i = 0; i < match_count; ++i) {
        ASSERT_FLOAT_NEAR(0.0f, matches[i].distance, 1e-4f);
    }

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Harris Corner Detection & Feature Descriptors");
    TEST_RUN(test_harris_corners_synthetic_box);
    TEST_RUN(test_descriptor_matching_self);
    return test_suite_report();
}
