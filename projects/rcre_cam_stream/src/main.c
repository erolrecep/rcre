/**
 * @file main.c
 * @brief High-Performance Zero-Copy Camera Stream & Live Studio on RCRE
 *
 * Architecture:
 *   - Background Capture Worker (AVFoundation / V4L2 / Synthetic Dummy)
 *   - Lock-Free SPSC Frame Ring Buffer (C23 <stdatomic.h> + RCRE_Arena)
 *   - Hardware Accelerated OpenGL 3.3 Core Renderer with GPU Fragment Shaders
 *   - Interactive Floating Toolbar, Snapshot Capture & Telemetry HUD
 */

#include "rcre.h"
#include "extras/rcre_cam.h"
#include "extras/rcre_ringbuf.h"
#include "render.h"

#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void print_usage(const char* prog) {
    printf("RCRE Camera Stream & Video Studio (Pure C23 + SDL3 + OpenGL)\n\n");
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  --width <W>       Capture width in pixels (default: 1280)\n");
    printf("  --height <H>      Capture height in pixels (default: 720)\n");
    printf("  --fps <FPS>       Target framerate (default: 30)\n");
    printf("  --dummy           Use synthetic animated test pattern (ideal for headless / CI)\n");
    printf("  --headless        Run in benchmark mode without creating an SDL3 window\n");
    printf("  --frames <N>      Exit automatically after N frames (default: 0 = unlimited)\n");
    printf("  --device <N>      Camera device index (default: 0)\n");
    printf("  -h, --help        Show this help message\n\n");
    printf("Interactive Controls:\n");
    printf("  M                 Flip video horizontally (Mirror / Selfie mode)\n");
    printf("  Space / P         Pause or Resume live video preview\n");
    printf("  C / Enter         Capture snapshot to snapshots/*.bmp (with shutter flash)\n");
    printf("  Tab               Cycle GPU filter (Normal, Grayscale, Contrast, Sepia, Invert)\n");
    printf("  S / F3            Toggle on-screen telemetry HUD (FPS, latency, resolution)\n");
    printf("  F                 Toggle Fullscreen\n");
    printf("  ESC / Q           Quit application\n");
}

int main(int argc, char** argv) {
    size_t width = 1280;
    size_t height = 720;
    size_t fps = 30;
    int force_dummy = 0;
    int headless = 0;
    size_t max_frames = 0;
    int device_idx = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dummy") == 0) {
            force_dummy = 1;
        } else if (strcmp(argv[i], "--headless") == 0 || strcmp(argv[i], "--bench") == 0) {
            headless = 1;
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            fps = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            max_frames = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_idx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("======================================================================\n");
    printf("🎥 RCRE High-Performance Camera Stream Studio (v2.7.1)\n");
    printf("======================================================================\n");
    printf("Target Resolution : %zux%zu @ %zu FPS\n", width, height, fps);
    printf("Capture Backend   : %s\n", force_dummy ? "Synthetic Dummy" : "Native HAL (Auto)");
    printf("Display Mode      : %s\n", headless ? "Headless Benchmark" : "SDL3 + OpenGL 3.3 Core");

    /* Allocate scratchpad memory arena for hardware buffers & ring slots */
    const size_t arena_size = 48 * 1024 * 1024; /* 48 MB */
    RCRE_Arena* arena = rcre_arena_create(arena_size);
    if (!arena) {
        fprintf(stderr, "[-] Failed to allocate RCRE_Arena (%zu bytes)\n", arena_size);
        return 1;
    }

    /* Initialize hardware camera */
    RCRE_CamConfig cam_cfg = {
        .width = width,
        .height = height,
        .fps = fps,
        .device_index = device_idx,
        .backend = force_dummy ? RCRE_CAM_BACKEND_DUMMY : RCRE_CAM_BACKEND_AUTO
    };

    RCRE_Camera* cam = rcre_cam_open(&cam_cfg, arena);
    if (!cam) {
        fprintf(stderr, "[-] Failed to open camera device (backend: %s)\n",
                force_dummy ? "Dummy" : "Auto");
        rcre_arena_destroy(arena);
        return 1;
    }

    printf("Active Camera     : %s\n", rcre_cam_backend_name(cam));

    /* Initialize Lock-Free SPSC Frame Ring Buffer (Triple-buffered) */
    size_t frame_bytes = width * height * 3;
    RCRE_RingBuf* ring = rcre_ringbuf_create(3, frame_bytes, arena);
    if (!ring) {
        fprintf(stderr, "[-] Failed to create lock-free ring buffer\n");
        rcre_cam_close(cam);
        rcre_arena_destroy(arena);
        return 1;
    }

    /* Start asynchronous capture thread */
    int start_status = rcre_cam_start_async(cam, ring);
    if (start_status != RCRE_OK) {
        fprintf(stderr, "[-] Failed to start async capture worker: %d\n", start_status);
        rcre_cam_close(cam);
        rcre_arena_destroy(arena);
        return 1;
    }

    printf("Capture Worker    : Running (Lock-Free SPSC Thread Decoupled)\n");
    printf("----------------------------------------------------------------------\n");

    uint8_t* rx_buffer = (uint8_t*)malloc(frame_bytes);
    if (!rx_buffer) {
        fprintf(stderr, "[-] Failed to allocate frame staging buffer\n");
        rcre_cam_close(cam);
        rcre_arena_destroy(arena);
        return 1;
    }

    size_t frames_rendered = 0;
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    if (headless) {
        printf("[*] Running headless benchmark loop...\n");
        while (1) {
            size_t out_sz = 0, out_w = 0, out_h = 0;
            int out_fmt = 0;
            if (rcre_ringbuf_pull_latest(ring, rx_buffer, frame_bytes, &out_sz, &out_w, &out_h, &out_fmt)) {
                frames_rendered++;
                if (max_frames > 0 && frames_rendered >= max_frames) {
                    break;
                }
            } else {
                usleep(1000); /* 1ms backoff */
            }
        }
    } else {
        renderer_t* ren = render_create((int)width, (int)height, "RCRE Live Camera Stream Studio");
        if (!ren) {
            fprintf(stderr, "[-] Failed to create SDL3/OpenGL renderer\n");
            free(rx_buffer);
            rcre_cam_close(cam);
            rcre_arena_destroy(arena);
            return 1;
        }

        frame_t f;
        while (!render_poll_quit()) {
            size_t out_sz = 0, out_w = 0, out_h = 0;
            int out_fmt = 0;
            if (rcre_ringbuf_pull_latest(ring, rx_buffer, frame_bytes, &out_sz, &out_w, &out_h, &out_fmt)) {
                f.data = rx_buffer;
                f.size = out_sz;
                f.fmt = PIXFMT_RGB24;
                f.width = (int)out_w;
                f.height = (int)out_h;
                f.stride = (int)out_w * 3;

                render_frame(ren, &f);
                frames_rendered++;

                if (max_frames > 0 && frames_rendered >= max_frames) {
                    break;
                }
            } else {
                SDL_Delay(1);
            }
        }

        render_destroy(ren);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed = (double)(end_time.tv_sec - start_time.tv_sec) +
                     (double)(end_time.tv_nsec - start_time.tv_nsec) * 1e-9;
    double fps_effective = elapsed > 0 ? (double)frames_rendered / elapsed : 0.0;

    printf("\n======================================================================\n");
    printf("📊 Execution Telemetry Summary\n");
    printf("======================================================================\n");
    printf("Frames Pushed     : %llu\n", (unsigned long long)rcre_ringbuf_pushed_count(ring));
    printf("Frames Pulled     : %llu\n", (unsigned long long)rcre_ringbuf_pulled_count(ring));
    printf("Frames Rendered   : %zu\n", frames_rendered);
    printf("Elapsed Time      : %.3f s\n", elapsed);
    printf("Throughput        : %.2f FPS\n", fps_effective);
    printf("======================================================================\n");

    /* Teardown */
    free(rx_buffer);
    rcre_cam_close(cam);
    rcre_arena_destroy(arena);

    return 0;
}
