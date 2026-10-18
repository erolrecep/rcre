/**
 * @file test_cam.c
 * @brief Unit tests for RCRE Camera Companion Module
 * 
 * Verifies camera initialization, frame acquisition, error handling,
 * and seamless pipeline integration with CV filter operators.
 */

#include "rcre.h"
#include "extras/rcre_cam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static void test_camera_dummy_pipeline(void) {
    printf("  [TEST] Synthetic Camera -> RGB to Gray -> Gaussian -> Sobel Pipeline...\n");

    const size_t arena_size = 16 * 1024 * 1024; /* 16 MB */
    RCRE_Arena* arena = rcre_arena_create(arena_size);
    assert(arena != NULL);

    RCRE_CamConfig cfg = {
        .width = 640,
        .height = 480,
        .fps = 30,
        .device_index = 0,
        .backend = RCRE_CAM_BACKEND_DUMMY
    };

    RCRE_Camera* cam = rcre_cam_open(&cfg, arena);
    assert(cam != NULL);
    assert(strcmp(rcre_cam_backend_name(cam), "Synthetic Test Pattern (Dummy)") == 0);

    /* Allocate frame tensor: [480, 640, 3] U8 */
    size_t frame_shape[3] = {480, 640, 3};
    RCRE_Tensor frame = rcre_tensor_create(frame_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    assert(frame.data != NULL);

    /* Grab 5 consecutive frames and verify animated pixel values */
    for (int i = 0; i < 5; ++i) {
        int status = rcre_cam_grab(cam, &frame);
        assert(status == RCRE_OK);

        uint8_t* px = (uint8_t*)frame.data;
        /* First pixel at (0, 0) in frame i */
        assert(px[0] == (uint8_t)((0 + i * 3) % 256));
        assert(px[1] == (uint8_t)((0 + i * 2) % 256));
        assert(px[2] == (uint8_t)((0 + 0 + i) % 256));
    }

    /* CV Pipeline Integration */
    size_t gray_shape[2] = {480, 640};
    RCRE_Tensor gray = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int status_gray = rcre_cv_rgb_to_gray(&frame, &gray);
    assert(status_gray == RCRE_OK);

    RCRE_Tensor blur = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int status_blur = rcre_cv_gaussian_blur(&gray, &blur, 1.0f, arena);
    assert(status_blur == RCRE_OK);

    RCRE_Tensor sobel_dx = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor sobel_dy = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int status_sobel = rcre_cv_sobel(&blur, &sobel_dx, &sobel_dy, arena);
    assert(status_sobel == RCRE_OK);

    /* Check edge response exists (non-zero edges from synthetic color transitions) */
    uint8_t* sobel_px = (uint8_t*)sobel_dx.data;
    size_t edge_sum = 0;
    for (size_t j = 0; j < 480 * 640; ++j) {
        edge_sum += sobel_px[j];
    }
    assert(edge_sum > 0);

    /* Test error cases */
    size_t bad_shape[3] = {300, 300, 3};
    RCRE_Tensor bad_tensor = rcre_tensor_create(bad_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int bad_status = rcre_cam_grab(cam, &bad_tensor);
    assert(bad_status == RCRE_ERR_SHAPE_MISMATCH);

    bad_status = rcre_cam_grab(NULL, &frame);
    assert(bad_status == RCRE_ERR_INVALID_ARG);

    rcre_cam_close(cam);
    rcre_arena_destroy(arena);
    printf("  [PASS] Synthetic Camera & CV Pipeline Integration Verified!\n");
}

static void test_camera_auto_backend(void) {
    printf("  [TEST] Camera Auto Backend Selection...\n");

    const size_t arena_size = 4 * 1024 * 1024;
    RCRE_Arena* arena = rcre_arena_create(arena_size);
    assert(arena != NULL);

    RCRE_CamConfig cfg = rcre_cam_default_config();
    cfg.backend = RCRE_CAM_BACKEND_AUTO;

    RCRE_Camera* cam = rcre_cam_open(&cfg, arena);
    assert(cam != NULL);
    const char* bname = rcre_cam_backend_name(cam);
    assert(bname != NULL && strlen(bname) > 0);
    printf("    -> Active Backend: %s\n", bname);

    rcre_cam_close(cam);
    rcre_arena_destroy(arena);
    printf("  [PASS] Camera Auto Backend Initialized & Closed Cleanly!\n");
}

int main(void) {
    printf("==================================================\n");
    printf("  RCRE Camera Module Verification Suite           \n");
    printf("==================================================\n");

    test_camera_dummy_pipeline();
    test_camera_auto_backend();

    printf("\nAll Camera tests passed successfully!\n");
    return 0;
}
