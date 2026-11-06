/**
 * @file test_main.c
 * @brief Unit tests for RCRE Starter Project
 *
 * Verifies hardware initialization, arena alignment, tensor creation,
 * and mathematical parity of algorithms under AddressSanitizer and UBSan.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#define RCRE_IMPLEMENTATION
#include "rcre.h"

/* Minimal test framework macros */
static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  ❌ FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        return false; \
    } \
} while(0)

#define RUN_TEST(fn) do { \
    g_tests_run++; \
    printf("  [RUN] %s...\n", #fn); \
    if (fn()) { \
        g_tests_passed++; \
        printf("  [PASS] %s\n", #fn); \
    } \
} while(0)

/* Test 1: Dynamic HAL initialization and probe verification */
static bool test_hal_initialization(void) {
    rcre_hal_init();
    TEST_ASSERT(rcre_hal.backend_name != nullptr, "HAL backend_name must not be NULL");
    TEST_ASSERT(rcre_hal.rgb_to_gray_u8 != nullptr, "HAL rgb_to_gray_u8 dispatch pointer must be set");
    return true;
}

/* Test 2: Arena 64-byte alignment and rewind contract */
static bool test_arena_alignment_and_rewind(void) {
    const size_t ARENA_SIZE = 1024 * 1024; /* 1 MB */
    RCRE_Arena* arena = rcre_arena_create(ARENA_SIZE);
    TEST_ASSERT(arena != nullptr, "Arena creation failed");
    TEST_ASSERT(((uintptr_t)arena->buffer & 63) == 0, "Arena base buffer must be 64-byte aligned");

    /* Sub-allocations with 64-byte alignment */
    void* p1 = rcre_arena_alloc(arena, 13, 64);
    TEST_ASSERT(p1 != nullptr, "Alloc 1 failed");
    TEST_ASSERT(((uintptr_t)p1 & 63) == 0, "Alloc 1 must be 64-byte aligned");

    void* p2 = rcre_arena_alloc(arena, 77, 64);
    TEST_ASSERT(p2 != nullptr, "Alloc 2 failed");
    TEST_ASSERT(((uintptr_t)p2 & 63) == 0, "Alloc 2 must be 64-byte aligned");
    TEST_ASSERT((uintptr_t)p2 > (uintptr_t)p1, "Alloc 2 must proceed linearly after Alloc 1");

    /* Test O(1) reset */
    rcre_arena_reset(arena);
    TEST_ASSERT(arena->offset == 0, "Arena reset must zero the offset");

    /* Test reallocation after reset */
    void* p3 = rcre_arena_alloc(arena, 32, 64);
    TEST_ASSERT(p3 == p1, "After reset, first allocation must reuse base buffer start");

    rcre_arena_destroy(arena);
    return true;
}

/* Test 3: Tensor allocation and shape integrity */
static bool test_tensor_creation(void) {
    RCRE_Arena* arena = rcre_arena_create(256 * 1024);
    TEST_ASSERT(arena != nullptr, "Arena creation failed");

    size_t shape[2] = {64, 128};
    RCRE_Tensor tensor = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    TEST_ASSERT(tensor.data != nullptr, "Tensor data allocation failed");
    TEST_ASSERT(tensor.ndim == 2, "Tensor ndim mismatch");
    TEST_ASSERT(tensor.shape[0] == 64 && tensor.shape[1] == 128, "Tensor shape mismatch");
    TEST_ASSERT(((uintptr_t)tensor.data & 63) == 0, "Tensor data pointer must be 64-byte aligned");

    /* Fill and verify */
    rcre_tensor_fill(&tensor, 3.14159f);
    float* data = (float*)tensor.data;
    for (size_t i = 0; i < 64 * 128; ++i) {
        if (fabsf(data[i] - 3.14159f) > 1e-5f) {
            TEST_ASSERT(false, "Tensor element value mismatch after fill");
        }
    }

    rcre_arena_destroy(arena);
    return true;
}

/* Test 4: Rec.601 RGB-to-grayscale math parity */
static bool test_cv_rgb_to_gray_parity(void) {
    RCRE_Arena* arena = rcre_arena_create(64 * 1024);
    TEST_ASSERT(arena != nullptr, "Arena creation failed");

    const size_t H = 4;
    const size_t W = 4;
    size_t rgb_shape[3] = {H, W, 3};
    RCRE_Tensor rgb = rcre_tensor_create(rgb_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    size_t gray_shape[2] = {H, W};
    RCRE_Tensor gray = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);

    uint8_t* p_rgb = (uint8_t*)rgb.data;
    /* Fill with known pixel values: R=100, G=150, B=200 */
    for (size_t i = 0; i < H * W; ++i) {
        p_rgb[i * 3 + 0] = 100;
        p_rgb[i * 3 + 1] = 150;
        p_rgb[i * 3 + 2] = 200;
    }

    int rc = rcre_cv_rgb_to_gray(&rgb, &gray);
    TEST_ASSERT(rc == RCRE_OK, "rcre_cv_rgb_to_gray failed");

    /* Expected: 0.299 * 100 + 0.587 * 150 + 0.114 * 200 = 29.9 + 88.05 + 22.8 = 140.75 -> 140 or 141 */
    uint8_t* p_gray = (uint8_t*)gray.data;
    for (size_t i = 0; i < H * W; ++i) {
        int diff = abs((int)p_gray[i] - 141);
        if (diff > 1) { /* allow +/- 1 integer rounding variance across SIMD integer fixed-point */
            char err[128];
            snprintf(err, sizeof(err), "Expected pixel near 141, got %u", p_gray[i]);
            TEST_ASSERT(false, err);
        }
    }

    rcre_arena_destroy(arena);
    return true;
}

int main(void) {
    printf("===================================================================\n");
    printf("🧪 Running RCRE Starter Project Verification Tests\n");
    printf("===================================================================\n");

    RUN_TEST(test_hal_initialization);
    RUN_TEST(test_arena_alignment_and_rewind);
    RUN_TEST(test_tensor_creation);
    RUN_TEST(test_cv_rgb_to_gray_parity);

    printf("\n-------------------------------------------------------------------\n");
    printf("Test Results: %d/%d passed (%s)\n", 
           g_tests_passed, g_tests_run, 
           (g_tests_passed == g_tests_run) ? "ALL TESTS PASSED ✅" : "SOME TESTS FAILED ❌");
    printf("-------------------------------------------------------------------\n");

    return (g_tests_passed == g_tests_run) ? EXIT_SUCCESS : EXIT_FAILURE;
}
