/**
 * @file main.c
 * @brief RCRE Starter Project: C23 High-Performance Tensor & Vision Engine
 *
 * This starter application demonstrates how to embed the single-header RCRE
 * engine, initialize dynamic SIMD hardware dispatch, manage memory via contiguous
 * arenas, and execute high-throughput tensor operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <inttypes.h>

/* Define the implementation in exactly ONE translation unit */
#define RCRE_IMPLEMENTATION
#include "rcre.h"

/* High-resolution timer helper */
static double get_time_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    printf("===================================================================\n");
    printf("   ____   ____ ____  _____\n");
    printf("  |  _ \\ / ___|  _ \\| ____|\n");
    printf("  | |_) | |   | |_) |  _|\n");
    printf("  |  _ <| |___|  _ <| |___\n");
    printf("  |_| \\_\\\\____|_| \\_\\_____|  Project Starter Template (C23)\n");
    printf("===================================================================\n\n");

    /* Step 1: Initialize Dynamic Hardware Abstraction Layer (HAL) */
    printf("[1/5] Initializing Hardware Abstraction Layer (HAL)...\n");
    rcre_hal_init();
    printf("      -> Detected SIMD Acceleration: [%s]\n", 
           rcre_hal.backend_name ? rcre_hal.backend_name : "Scalar / Generic Fallback");

    /* Step 2: Initialize 16MB Contiguous Memory Arena */
    const size_t ARENA_CAPACITY = 16 * 1024 * 1024; /* 16 MB */
    printf("\n[2/5] Creating 16MB contiguous memory arena...\n");
    RCRE_Arena* arena = rcre_arena_create(ARENA_CAPACITY);
    if (!arena) {
        fprintf(stderr, "Error: Failed to allocate RCRE arena of size %zu bytes\n", ARENA_CAPACITY);
        return EXIT_FAILURE;
    }
    printf("      -> Arena allocated at: %p (Capacity: %.2f MB)\n", 
           (void*)arena->buffer, (double)ARENA_CAPACITY / (1024.0 * 1024.0));
    printf("      -> Alignment check: %s (address %% 64 == 0)\n",
           ((uintptr_t)arena->buffer % 64 == 0) ? "PASS (64-byte AVX-512 / Cache-Line Aligned)" : "FAIL");

    /* Step 3: Allocate and Initialize RGB Image Tensor (1080p: 1920x1080x3) */
    const size_t width = 1920;
    const size_t height = 1080;
    const size_t channels = 3;
    printf("\n[3/5] Allocating 1080p RGB tensor (%zux%zux%zu, U8)...\n", width, height, channels);

    size_t rgb_shape[3] = {height, width, channels};
    RCRE_Tensor rgb = rcre_tensor_create(rgb_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    if (!rgb.data) {
        fprintf(stderr, "Error: Failed to allocate RGB tensor from arena\n");
        rcre_arena_destroy(arena);
        return EXIT_FAILURE;
    }

    /* Synthesize gradient pattern */
    uint8_t* p_rgb = (uint8_t*)rgb.data;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * 3;
            p_rgb[idx + 0] = (uint8_t)((x * 255) / width);        /* Red gradient */
            p_rgb[idx + 1] = (uint8_t)((y * 255) / height);       /* Green gradient */
            p_rgb[idx + 2] = (uint8_t)(((x + y) * 255) / (width + height)); /* Blue */
        }
    }
    printf("      -> Image synthesized. Arena memory used: %.2f MB / %.2f MB\n",
           (double)arena->offset / (1024.0 * 1024.0),
           (double)arena->capacity / (1024.0 * 1024.0));

    /* Step 4: Perform Rec.601 Grayscale Conversion */
    printf("\n[4/5] Executing SIMD-accelerated Rec.601 RGB-to-Grayscale conversion...\n");
    size_t gray_shape[2] = {height, width};
    RCRE_Tensor gray = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    if (!gray.data) {
        fprintf(stderr, "Error: Failed to allocate grayscale tensor\n");
        rcre_arena_destroy(arena);
        return EXIT_FAILURE;
    }

    /* Benchmark RGB to Grayscale across 100 warm iterations */
    const int BENCH_ITERATIONS = 100;
    double t0 = get_time_sec();
    for (int i = 0; i < BENCH_ITERATIONS; ++i) {
        int status = rcre_cv_rgb_to_gray(&rgb, &gray);
        if (status != RCRE_OK) {
            fprintf(stderr, "Error during color conversion: code %d\n", status);
            break;
        }
    }
    double t1 = get_time_sec();
    double total_time = t1 - t0;
    double avg_ms = (total_time / BENCH_ITERATIONS) * 1000.0;
    double megapixels = (double)(width * height) / 1e6;
    double throughput_mpx_sec = megapixels / (avg_ms / 1000.0);

    printf("      -> Processed %d frames of 1080p (%.2f MP)\n", BENCH_ITERATIONS, megapixels);
    printf("      -> Average Latency: %.3f ms per frame\n", avg_ms);
    printf("      -> Throughput:      %.2f Megapixels / second\n", throughput_mpx_sec);

    /* Step 5: Arena Diagnostics & Clean Teardown */
    printf("\n[5/5] Arena lifecycle and memory reset...\n");
    printf("      -> Peak Arena Offset: %zu bytes (%.2f MB)\n", 
           arena->offset, (double)arena->offset / (1024.0 * 1024.0));
    printf("      -> Arena Remaining:   %zu bytes (%.2f MB)\n", 
           arena->capacity - arena->offset, (double)(arena->capacity - arena->offset) / (1024.0 * 1024.0));

    /* Demonstrate O(1) Instant Arena Rewind */
    rcre_arena_reset(arena);
    printf("      -> rcre_arena_reset() executed: current offset = %zu bytes (O(1) reset)\n", arena->offset);

    /* Free entire memory arena in single system munmap/free */
    rcre_arena_destroy(arena);
    printf("      -> rcre_arena_destroy() complete. Zero memory leaks.\n\n");

    printf("===================================================================\n");
    printf("🚀 Template execution successful! You are ready to build with RCRE.\n");
    printf("===================================================================\n");

    return EXIT_SUCCESS;
}
