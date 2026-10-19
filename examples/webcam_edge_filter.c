/**
 * @file webcam_edge_filter.c
 * @brief Real-time Camera Stream with Gaussian Blur and Sobel Edge Detection
 * 
 * Demonstrates high-performance zero-copy camera ingestion and CV filter pipelining.
 * Supports Terminal (ASCII/ANSI) and Native GUI Window display modes.
 * Measures end-to-end processing throughput (FPS) and latency.
 */

#include "rcre.h"
#include "extras/rcre_cam.h"
#include "extras/rcre_window.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef enum {
    DISPLAY_UNSPECIFIED = 0,
    DISPLAY_TERMINAL,
    DISPLAY_GUI_SIDE_BY_SIDE,
    DISPLAY_GUI_SINGLE,
    DISPLAY_BENCH_ONLY
} DisplayMode;

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  --terminal, --view        Display live video in terminal via ANSI ASCII art\n");
    printf("  --gui                     Display live video in Native GUI window (Side-by-side)\n");
    printf("  --gui-single              Display live video in Native GUI window (Edge filter only)\n");
    printf("  --headless, --bench       Run without display to benchmark maximum throughput\n");
    printf("  --dummy                   Use synthetic moving gradient pattern instead of webcam\n");
    printf("  --frames <N>              Number of frames to process (default: 200, 0 = unlimited)\n");
    printf("  --width <W>               Frame capture width (default: 640)\n");
    printf("  --height <H>              Frame capture height (default: 480)\n");
    printf("  -h, --help                Show this help message\n");
}

int main(int argc, char** argv) {
    size_t width = 640;
    size_t height = 480;
    size_t max_frames = 200;
    int force_dummy = 0;
    DisplayMode mode = DISPLAY_UNSPECIFIED;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dummy") == 0) {
            force_dummy = 1;
        } else if (strcmp(argv[i], "--terminal") == 0 || strcmp(argv[i], "--view") == 0) {
            mode = DISPLAY_TERMINAL;
        } else if (strcmp(argv[i], "--gui") == 0 || strcmp(argv[i], "--side-by-side") == 0) {
            mode = DISPLAY_GUI_SIDE_BY_SIDE;
        } else if (strcmp(argv[i], "--gui-single") == 0) {
            mode = DISPLAY_GUI_SINGLE;
        } else if (strcmp(argv[i], "--headless") == 0 || strcmp(argv[i], "--bench") == 0) {
            mode = DISPLAY_BENCH_ONLY;
        } else if (strncmp(argv[i], "--display=", 10) == 0) {
            const char* val = argv[i] + 10;
            if (strcmp(val, "terminal") == 0 || strcmp(val, "ascii") == 0) {
                mode = DISPLAY_TERMINAL;
            } else if (strcmp(val, "gui") == 0 || strcmp(val, "side-by-side") == 0) {
                mode = DISPLAY_GUI_SIDE_BY_SIDE;
            } else if (strcmp(val, "gui-single") == 0) {
                mode = DISPLAY_GUI_SINGLE;
            } else if (strcmp(val, "none") == 0 || strcmp(val, "bench") == 0) {
                mode = DISPLAY_BENCH_ONLY;
            }
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            max_frames = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    /* Interactive mode prompt if no display flag was provided */
    if (mode == DISPLAY_UNSPECIFIED) {
        if (isatty(STDIN_FILENO)) {
            printf("\n===================================================================\n");
            printf("  🎥 RCRE Live Camera Edge Detection Test                          \n");
            printf("===================================================================\n");
            printf("Please select display mode for this test:\n");
            printf("  [1] Terminal (Real-time ANSI ASCII visualization in console)\n");
            if (rcre_window_is_supported()) {
                printf("  [2] Native GUI Window (Side-by-side: Camera RGB | Sobel Filter) [Recommended]\n");
                printf("  [3] Native GUI Window (Single: Sobel Edge Output)\n");
            } else {
                printf("  [2] Native GUI Window [Unavailable on this platform]\n");
            }
            printf("  [4] Headless Benchmark (Maximum throughput, no rendering)\n");
            printf("-------------------------------------------------------------------\n");
            printf("Choice [1-%d] (default: %d): ", rcre_window_is_supported() ? 4 : 4, rcre_window_is_supported() ? 2 : 1);
            fflush(stdout);

            char line[32];
            if (fgets(line, sizeof(line), stdin)) {
                int choice = atoi(line);
                if (choice == 1) {
                    mode = DISPLAY_TERMINAL;
                } else if (choice == 2) {
                    mode = rcre_window_is_supported() ? DISPLAY_GUI_SIDE_BY_SIDE : DISPLAY_TERMINAL;
                } else if (choice == 3) {
                    mode = rcre_window_is_supported() ? DISPLAY_GUI_SINGLE : DISPLAY_TERMINAL;
                } else if (choice == 4) {
                    mode = DISPLAY_BENCH_ONLY;
                } else {
                    /* Default selection */
                    mode = rcre_window_is_supported() ? DISPLAY_GUI_SIDE_BY_SIDE : DISPLAY_TERMINAL;
                }
            } else {
                mode = rcre_window_is_supported() ? DISPLAY_GUI_SIDE_BY_SIDE : DISPLAY_TERMINAL;
            }
        } else {
            /* Non-interactive pipe / CI */
            mode = DISPLAY_TERMINAL;
        }
    }

    /* Fallback if GUI requested on unsupported platform */
    if ((mode == DISPLAY_GUI_SIDE_BY_SIDE || mode == DISPLAY_GUI_SINGLE) && !rcre_window_is_supported()) {
        printf("⚠️  Native desktop GUI window not supported in current environment.\n");
        printf("    Falling back to Terminal ASCII display.\n");
        mode = DISPLAY_TERMINAL;
    }

    printf("\n===================================================================\n");
    printf("  RCRE Example: Real-time Camera Edge Detection Pipeline           \n");
    printf("===================================================================\n");
    printf("Configuration:\n");
    printf("  Resolution:   %zu x %zu\n", width, height);
    printf("  Frames:       %zu\n", max_frames);
    printf("  Backend:      %s\n", force_dummy ? "Synthetic Dummy (Forced)" : "Auto-Detect");
    const char* mode_str = "Headless Benchmark";
    if (mode == DISPLAY_TERMINAL) mode_str = "Terminal (ANSI ASCII Art)";
    else if (mode == DISPLAY_GUI_SIDE_BY_SIDE) mode_str = "Native GUI Window (Side-by-side Camera | Sobel)";
    else if (mode == DISPLAY_GUI_SINGLE) mode_str = "Native GUI Window (Sobel Output)";
    printf("  Display Mode: %s\n", mode_str);

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

    /* Pre-allocate pipeline tensors in arena */
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

    /* Initialize GUI Window if selected */
    RCRE_Window* win = NULL;
    uint8_t* gui_buffer = NULL;
    size_t win_w = width;
    size_t win_h = height;

    if (mode == DISPLAY_GUI_SIDE_BY_SIDE) {
        win_w = width * 2;
        win = rcre_window_create("RCRE Video Stream [Original RGB | Sobel Filter]", win_w, win_h);
        gui_buffer = (uint8_t*)rcre_arena_alloc(arena, win_w * win_h * 3, 16);
    } else if (mode == DISPLAY_GUI_SINGLE) {
        win = rcre_window_create("RCRE Sobel Edge Detector", win_w, win_h);
        gui_buffer = (uint8_t*)rcre_arena_alloc(arena, win_w * win_h * 3, 16);
    }

    size_t baseline_offset = arena->offset;

    printf("\nStarting pipeline loop (%zu frames)...\n", max_frames);
    if (mode == DISPLAY_GUI_SIDE_BY_SIDE || mode == DISPLAY_GUI_SINGLE) {
        printf("👉 Native GUI window launched! Press 'q' or ESC in window or Ctrl+C in terminal to finish.\n");
    }

    double t_start = get_time_sec();
    size_t processed_frames = 0;

    for (size_t f = 0; max_frames == 0 || f < max_frames; ++f) {
        arena->offset = baseline_offset;

        /* Step 1: Zero-copy frame grab */
        int grab_res = rcre_cam_grab(cam, &frame_tensor);
        if (grab_res != RCRE_OK) {
            if (grab_res == RCRE_ERR_TIMEOUT) {
                continue;
            }
            printf("grab err: %d\n", grab_res); break;
        }

        /* Step 2: RGB -> Grayscale conversion (15-bit fixed point) */
        int status = rcre_cv_rgb_to_gray(&frame_tensor, &gray_tensor);
        if (status != RCRE_OK) { printf("status err at line %d: %d\n", __LINE__, status); break; }

        /* Step 3: Separable Gaussian Blur (sigma = 1.2) */
        status = rcre_cv_gaussian_blur(&gray_tensor, &blur_tensor, 1.2f, arena);
        if (status != RCRE_OK) { printf("status err at line %d: %d\n", __LINE__, status); break; }

        /* Step 4: Spatial 3x3 Sobel edge derivative */
        status = rcre_cv_sobel(&blur_tensor, &sobel_dx, &sobel_dy, arena);
        if (status != RCRE_OK) { printf("status err at line %d: %d\n", __LINE__, status); break; }

        processed_frames++;

        /* --- DISPLAY ROUTINES --- */

        /* Option A: Native GUI Window Display */
        if (win && gui_buffer) {
            if (mode == DISPLAY_GUI_SIDE_BY_SIDE) {
                const uint8_t* raw_rgb = (const uint8_t*)frame_tensor.data;
                const uint8_t* raw_edge = (const uint8_t*)sobel_dx.data;

                for (size_t y = 0; y < height; ++y) {
                    uint8_t* row_out = gui_buffer + y * (win_w * 3);
                    const uint8_t* row_in_rgb = raw_rgb + y * (width * 3);
                    const uint8_t* row_in_edge = raw_edge + y * width;

                    /* Left side: Original Camera RGB */
                    memcpy(row_out, row_in_rgb, width * 3);

                    /* Right side: Edge detection with high-contrast cyan highlight */
                    uint8_t* edge_rgb = row_out + (width * 3);
                    for (size_t x = 0; x < width; ++x) {
                        uint8_t v = row_in_edge[x];
                        edge_rgb[x * 3 + 0] = v / 2; /* Red */
                        edge_rgb[x * 3 + 1] = v;     /* Green */
                        edge_rgb[x * 3 + 2] = v;     /* Blue */
                    }
                }
            } else if (mode == DISPLAY_GUI_SINGLE) {
                const uint8_t* raw_edge = (const uint8_t*)sobel_dx.data;
                for (size_t y = 0; y < height; ++y) {
                    uint8_t* row_out = gui_buffer + y * (win_w * 3);
                    const uint8_t* row_in_edge = raw_edge + y * width;
                    for (size_t x = 0; x < width; ++x) {
                        uint8_t v = row_in_edge[x];
                        row_out[x * 3 + 0] = v / 2;
                        row_out[x * 3 + 1] = v;
                        row_out[x * 3 + 2] = v;
                    }
                }
            }

            int win_active = rcre_window_update_rgb(win, gui_buffer, win_w, win_h);

            /* Update dynamic title bar every 15 frames */
            if (processed_frames % 15 == 0) {
                double elapsed = get_time_sec() - t_start;
                double cur_fps = (double)processed_frames / (elapsed > 1e-4 ? elapsed : 1e-4);
                char title[128];
                snprintf(title, sizeof(title), "RCRE Camera Pipeline [%s] | Frame %zu | %.1f FPS (Press 'q' to stop)",
                         rcre_cam_backend_name(cam), processed_frames, cur_fps);
                rcre_window_set_title(win, title);
            }

            if (!win_active) {
                printf("\n[GUI] Window close requested by user. Finishing benchmark...\n");
                break;
            }
        }

        /* Option B: Terminal ANSI ASCII Art Display */
        else if (mode == DISPLAY_TERMINAL) {
            printf("\033[H\033[2J");
            printf("===================================================================\n");
            printf("  🎥 RCRE Live Camera Stream (Frame %zu / %zu) | Backend: %s\n", 
                   f + 1, max_frames, rcre_cam_backend_name(cam));
            printf("===================================================================\n");
            const char* ramp = " .:-=+*#%@";
            const uint8_t* edge_data = (const uint8_t*)sobel_dx.data;
            const size_t ascii_w = 64;
            const size_t ascii_h = 20;
            const size_t step_x = width / ascii_w;
            const size_t step_y = height / ascii_h;

            for (size_t y = 0; y < ascii_h; ++y) {
                printf("|");
                for (size_t x = 0; x < ascii_w; ++x) {
                    uint8_t px = edge_data[(y * step_y) * width + (x * step_x)];
                    int idx = (px * 9) / 255;
                    if (idx < 0) idx = 0;
                    if (idx > 9) idx = 9;
                    putchar(ramp[idx]);
                }
                printf("|\n");
            }
            printf("===================================================================\n");
            printf("  Live Pipeline: Frame %zu | Press Ctrl+C to Stop\n", f + 1);
            fflush(stdout);
        }
    }

    double t_end = get_time_sec();
    double total_time = t_end - t_start;
    double fps = (double)processed_frames / (total_time > 1e-6 ? total_time : 1e-6);
    double avg_latency_ms = (total_time / (double)(processed_frames > 0 ? processed_frames : 1)) * 1000.0;
    double megapixels_per_sec = (fps * (double)(width * height)) / 1e6;

    if (win) {
        rcre_window_destroy(win);
    }

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
