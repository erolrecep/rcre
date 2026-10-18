/**
 * @file webcam_edge_filter.c
 * @brief Real-time Camera Stream with Gaussian Blur and Sobel Edge Detection
 * 
 * Demonstrates high-performance zero-copy camera ingestion and CV filter pipelining.
 * Measures end-to-end processing throughput (FPS) and latency.
 */

#include "rcre.h"
#include "extras/rcre_cam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    size_t width = 640;
    size_t height = 480;
    size_t max_frames = 100;
    int force_dummy = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dummy") == 0) {
            force_dummy = 1;
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            max_frames = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = (size_t)atoi(argv[++i]);
        }
    }

    printf("===================================================================\n");
    printf("  RCRE Example: Real-time Camera Edge Detection Pipeline           \n");
    printf("===================================================================\n");
    printf("Configuration:\n");
    printf("  Resolution: %zu x %zu\n", width, height);
    printf("  Frames:     %zu\n", max_frames);
    printf("  Backend:    %s\n", force_dummy ? "Synthetic Dummy (Forced)" : "Auto-Detect");

    /* Allocate 32 MB arena for entire pipeline (Zero heap allocations in loop!) */
    const size_t arena_size = 32 * 1024 * 1024;
    RCRE_Arena* arena = rcre_arena_create(arena_size);
    if (!arena) {
        fprintf(stderr, "Failed to create memory arena.\n");
        return 1;
    }

    RCRE_CamConfig cfg = {
        .width = width,
        .height = height,
        .fps = 30,
        .device_index = 0,
        .backend = force_dummy ? RCRE_CAM_BACKEND_DUMMY : RCRE_CAM_BACKEND_AUTO
    };

    RCRE_Camera* cam = rcre_cam_open(&cfg, arena);
    if (!cam) {
        fprintf(stderr, "Failed to open camera.\n");
        rcre_arena_destroy(arena);
        return 1;
    }

    printf("Active Camera Backend: %s\n", rcre_cam_backend_name(cam));

    /* Pre-allocate all pipeline tensors in arena */
    size_t frame_shape[3] = {height, width, 3};
    size_t gray_shape[2]  = {height, width};

    RCRE_Tensor frame_tensor = rcre_tensor_create(frame_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor gray_tensor  = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor blur_tensor  = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor sobel_dx     = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor sobel_dy     = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);

    if (!frame_tensor.data || !gray_tensor.data || !blur_tensor.data || !sobel_dx.data || !sobel_dy.data) {
        fprintf(stderr, "Failed to allocate pipeline tensors.\n");
        rcre_cam_close(cam);
        rcre_arena_destroy(arena);
        return 1;
    }

    size_t baseline_offset = arena->offset;

    printf("\nStarting pipeline loop (%zu frames)...\n", max_frames);
    double t_start = get_time_sec();
    size_t processed_frames = 0;

    for (size_t f = 0; f < max_frames; ++f) {
        /* Rewind scratchpad to baseline (tensors remain valid) */
        arena->offset = baseline_offset;

        /* Step 1: Zero-copy frame grab */
        int grab_res = rcre_cam_grab(cam, &frame_tensor);
        if (grab_res != RCRE_OK) {
            if (grab_res == RCRE_ERR_TIMEOUT) {
                continue;
            }
            break;
        }

        /* Step 2: RGB -> Grayscale conversion (15-bit fixed point) */
        int status = rcre_cv_rgb_to_gray(&frame_tensor, &gray_tensor);
        if (status != RCRE_OK) break;

        /* Step 3: Separable Gaussian Blur (sigma = 1.2) */
        status = rcre_cv_gaussian_blur(&gray_tensor, &blur_tensor, 1.2f, arena);
        if (status != RCRE_OK) break;

        /* Step 4: Spatial 3x3 Sobel edge derivative */
        status = rcre_cv_sobel(&blur_tensor, &sobel_dx, &sobel_dy, arena);
        if (status != RCRE_OK) break;

        processed_frames++;
    }

    double t_end = get_time_sec();
    double total_time = t_end - t_start;
    double fps = (double)processed_frames / (total_time > 1e-6 ? total_time : 1e-6);
    double avg_latency_ms = (total_time / (double)(processed_frames > 0 ? processed_frames : 1)) * 1000.0;
    double megapixels_per_sec = (fps * (double)(width * height)) / 1e6;

    printf("\n===================================================================\n");
    printf("  Performance & Benchmark Results                                  \n");
    printf("===================================================================\n");
    printf("Processed Frames:     %zu / %zu\n", processed_frames, max_frames);
    printf("Total Elapsed Time:   %.4f seconds\n", total_time);
    printf("Throughput:           %.2f FPS\n", fps);
    printf("Average Latency:      %.3f ms / frame\n", avg_latency_ms);
    printf("Pixel Throughput:     %.2f MP/s\n", megapixels_per_sec);
    printf("Arena Memory Used:    %zu / %zu bytes (%.1f%%)\n", 
           arena->offset, arena->capacity, ((double)arena->offset / (double)arena->capacity) * 100.0);
    printf("Heap Allocations in Loop: 0 (100%% deterministic linear arena)\n");
    printf("===================================================================\n");

    rcre_cam_close(cam);
    rcre_arena_destroy(arena);
    return 0;
}
