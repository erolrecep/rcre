/**
 * @file test_amalgamated.c
 * @brief Verification of the single-header dist/rcre.h distribution
 */

#include "test_framework.h"

#define RCRE_IMPLEMENTATION
#include "../dist/rcre.h"

static void test_amalgamated_single_header(void) {
    /* Initialize hardware dispatch */
    rcre_hal_init();
    ASSERT_NOT_NULL(rcre_hal.backend_name);

    /* Create Arena */
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Tensor Allocation */
    size_t shape[3] = {10, 10, 3};
    RCRE_Tensor t = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(t.data);

    /* Rec.601 Grayscale Filter */
    size_t gray_shape[2] = {10, 10};
    RCRE_Tensor gray = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int status = rcre_cv_rgb_to_gray(&t, &gray);
    ASSERT_INT_EQ(RCRE_OK, status);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Single-Header (dist/rcre.h) Amalgamation Self-Test");
    TEST_RUN(test_amalgamated_single_header);
    return test_suite_report();
}
