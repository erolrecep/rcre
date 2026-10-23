/**
 * @file test_cam.c
 * @brief Unit tests for RCRE Camera Companion Module
 * 
 * Verifies camera initialization, frame acquisition, error handling,
 * and seamless pipeline integration with CV filter operators.
 */

#include "rcre.h"
#include "extras/rcre_cam.h"
#include "test_framework.h"

static void test_camera_dummy_pipeline(void) {
    const size_t arena_size = 16 * 1024 * 1024; /* 16 MB */
    RCRE_Arena* arena = rcre_arena_create(arena_size);
    ASSERT_NOT_NULL(arena);

    RCRE_CamConfig cfg = {
        .width = 640,
        .height = 480,
        .fps = 30,
        .device_index = 0,
        .backend = RCRE_CAM_BACKEND_DUMMY
    };

    RCRE_Camera* cam = rcre_cam_open(&cfg, arena);
    ASSERT_NOT_NULL(cam);
    ASSERT_STR_EQ("Synthetic Test Pattern (Dummy)", rcre_cam_backend_name(cam));

    /* Allocate frame tensor: [480, 640, 3] U8 */
    size_t frame_shape[3] = {480, 640, 3};
    RCRE_Tensor frame = rcre_tensor_create(frame_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(frame.data);

    /* Grab 5 consecutive frames and verify animated pixel values */
    for (int i = 0; i < 5; ++i) {
        int status = rcre_cam_grab(cam, &frame);
        ASSERT_INT_EQ(RCRE_OK, status);

        uint8_t* px = (uint8_t*)frame.data;
        /* First pixel at (0, 0) in frame i */
        ASSERT_UINT_EQ((uint8_t)((0 + i * 3) % 256), px[0]);
        ASSERT_UINT_EQ((uint8_t)((0 + i * 2) % 256), px[1]);
        ASSERT_UINT_EQ((uint8_t)((0 + 0 + i) % 256), px[2]);
    }

    /* CV Pipeline Integration */
    size_t gray_shape[2] = {480, 640};
    RCRE_Tensor gray = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int status_gray = rcre_cv_rgb_to_gray(&frame, &gray);
    ASSERT_INT_EQ(RCRE_OK, status_gray);

    RCRE_Tensor blur = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int status_blur = rcre_cv_gaussian_blur(&gray, &blur, 1.0f, arena);
    ASSERT_INT_EQ(RCRE_OK, status_blur);

    RCRE_Tensor sobel_dx = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor sobel_dy = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int status_sobel = rcre_cv_sobel(&blur, &sobel_dx, &sobel_dy, arena);
    ASSERT_INT_EQ(RCRE_OK, status_sobel);

    /* Check edge response exists (non-zero edges from synthetic color transitions) */
    uint8_t* sobel_px = (uint8_t*)sobel_dx.data;
    size_t edge_sum = 0;
    for (size_t j = 0; j < 480 * 640; ++j) {
        edge_sum += sobel_px[j];
    }
    ASSERT_TRUE(edge_sum > 0);

    /* Test error cases */
    size_t bad_shape[3] = {300, 300, 3};
    RCRE_Tensor bad_tensor = rcre_tensor_create(bad_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int bad_status = rcre_cam_grab(cam, &bad_tensor);
    ASSERT_INT_EQ(RCRE_ERR_SHAPE_MISMATCH, bad_status);

    bad_status = rcre_cam_grab(NULL, &frame);
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, bad_status);

    rcre_cam_close(cam);
    rcre_arena_destroy(arena);
}

static void test_camera_auto_backend(void) {
    const size_t arena_size = 4 * 1024 * 1024;
    RCRE_Arena* arena = rcre_arena_create(arena_size);
    ASSERT_NOT_NULL(arena);

    RCRE_CamConfig cfg = rcre_cam_default_config();
    cfg.backend = RCRE_CAM_BACKEND_AUTO;

    RCRE_Camera* cam = rcre_cam_open(&cfg, arena);
    ASSERT_NOT_NULL(cam);
    const char* bname = rcre_cam_backend_name(cam);
    ASSERT_NOT_NULL(bname);
    ASSERT_TRUE(strlen(bname) > 0);

    rcre_cam_close(cam);
    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Camera Companion & CV Pipeline (EXTRAS-CAM-01)");
    TEST_RUN(test_camera_dummy_pipeline);
    TEST_RUN(test_camera_auto_backend);
    return test_suite_report();
}
