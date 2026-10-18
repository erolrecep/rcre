/**
 * @file test_amalgamated.c
 * @brief Verification of the single-header dist/rcre.h distribution
 */

#include <stdio.h>
#include <assert.h>

#define RCRE_IMPLEMENTATION
#include "../dist/rcre.h"

int main(void) {
    printf("==================================================\n");
    printf("  RCRE Single-Header (dist/rcre.h) Self-Test      \n");
    printf("==================================================\n");

    // Initialize hardware dispatch
    rcre_hal_init();
    printf("  HAL Backend: %s\n", rcre_hal.backend_name);
    assert(rcre_hal.backend_name != NULL);

    // Create Arena
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    assert(arena != NULL);

    // Tensor Allocation
    size_t shape[3] = {10, 10, 3};
    RCRE_Tensor t = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    assert(t.data != NULL);

    // Rec.601 Grayscale Filter
    size_t gray_shape[2] = {10, 10};
    RCRE_Tensor gray = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int status = rcre_cv_rgb_to_gray(&t, &gray);
    assert(status == RCRE_OK);

    rcre_arena_destroy(arena);

    printf("  ✓ dist/rcre.h compiled and executed flawlessly!\n");
    printf("==================================================\n");
    return 0;
}
