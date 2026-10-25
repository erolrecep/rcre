/**
 * @file test_cv_homography.c
 * @brief Unit tests for Direct Linear Transformation (DLT), RANSAC, and Perspective Warping
 */

#include "rcre.h"
#include "test_framework.h"
#include <math.h>

#define EPSILON 1e-3f

static void test_dlt_homography_affine(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Construct 4 point correspondences under known affine map:
     * x' = 2.0*x + 10.0
     * y' = 2.0*y + 5.0
     */
    RCRE_CV_Match matches[4];
    matches[0].p = (RCRE_CV_Point){0.0f, 0.0f};
    matches[0].q = (RCRE_CV_Point){10.0f, 5.0f};

    matches[1].p = (RCRE_CV_Point){100.0f, 0.0f};
    matches[1].q = (RCRE_CV_Point){210.0f, 5.0f};

    matches[2].p = (RCRE_CV_Point){0.0f, 100.0f};
    matches[2].q = (RCRE_CV_Point){10.0f, 205.0f};

    matches[3].p = (RCRE_CV_Point){100.0f, 100.0f};
    matches[3].q = (RCRE_CV_Point){210.0f, 205.0f};

    float H[3][3];
    int status = rcre_cv_compute_homography_dlt(matches, 4, H, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* Verify H[0][0] == 2.0, H[0][2] == 10.0, H[1][1] == 2.0, H[1][2] == 5.0, H[2][2] == 1.0 */
    ASSERT_FLOAT_NEAR(2.0f, H[0][0], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, H[0][1], EPSILON);
    ASSERT_FLOAT_NEAR(10.0f, H[0][2], EPSILON);

    ASSERT_FLOAT_NEAR(0.0f, H[1][0], EPSILON);
    ASSERT_FLOAT_NEAR(2.0f, H[1][1], EPSILON);
    ASSERT_FLOAT_NEAR(5.0f, H[1][2], EPSILON);

    ASSERT_FLOAT_NEAR(0.0f, H[2][0], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, H[2][1], EPSILON);
    ASSERT_FLOAT_NEAR(1.0f, H[2][2], EPSILON);

    rcre_arena_destroy(arena);
}

static void test_ransac_outlier_rejection(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Ground truth translation: dx = 20, dy = 30 */
    const int num_inliers = 10;
    const int num_outliers = 4;
    const int total = num_inliers + num_outliers;
    RCRE_CV_Match matches[14];

    for (int i = 0; i < num_inliers; ++i) {
        float x = (float)((i % 4) * 30 + 10);
        float y = (float)((i / 4) * 30 + 10);
        matches[i].p = (RCRE_CV_Point){x, y};
        matches[i].q = (RCRE_CV_Point){x + 20.0f, y + 30.0f};
    }

    /* Outliers with corrupted points */
    matches[10].p = (RCRE_CV_Point){15.0f, 25.0f};
    matches[10].q = (RCRE_CV_Point){500.0f, -200.0f};

    matches[11].p = (RCRE_CV_Point){45.0f, 65.0f};
    matches[11].q = (RCRE_CV_Point){-100.0f, 400.0f};

    matches[12].p = (RCRE_CV_Point){85.0f, 95.0f};
    matches[12].q = (RCRE_CV_Point){300.0f, 300.0f};

    matches[13].p = (RCRE_CV_Point){125.0f, 135.0f};
    matches[13].q = (RCRE_CV_Point){0.0f, 0.0f};

    float H[3][3];
    int recovered_inliers = 0;
    int status = rcre_cv_find_homography_ransac(matches, total, 2.0f, 300, 10, H, &recovered_inliers, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* Verify RANSAC found at least 8 inliers and rejected gross outliers */
    ASSERT_TRUE(recovered_inliers >= 8);

    /* Verify recovered translation */
    ASSERT_FLOAT_NEAR(1.0f, H[0][0], 0.05f);
    ASSERT_FLOAT_NEAR(20.0f, H[0][2], 1.0f);
    ASSERT_FLOAT_NEAR(1.0f, H[1][1], 0.05f);
    ASSERT_FLOAT_NEAR(30.0f, H[1][2], 1.0f);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Homography, DLT & RANSAC Estimators");
    TEST_RUN(test_dlt_homography_affine);
    TEST_RUN(test_ransac_outlier_rejection);
    return test_suite_report();
}
