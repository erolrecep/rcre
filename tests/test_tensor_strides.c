/**
 * @file test_tensor_strides.c
 * @brief Unit tests for RCRE strided tensor engine and ShapeTracker operations
 */

#include "rcre.h"
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

void test_tensor_create_and_contiguous(void) {
    printf("Running test_tensor_create_and_contiguous...\n");

    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    size_t shape[3] = {480, 640, 3}; /* 480x640x3 image */
    RCRE_Tensor t = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    assert(t.data != NULL);
    assert(t.ndim == 3);
    assert(rcre_tensor_numel(&t) == 480 * 640 * 3);
    assert(rcre_tensor_nbytes(&t) == 480 * 640 * 3 * 1);

    /* Check row-major strides */
    assert(t.strides[2] == 1);
    assert(t.strides[1] == 3);
    assert(t.strides[0] == 640 * 3);
    assert(rcre_tensor_is_contiguous(&t) == true);

    rcre_arena_destroy(arena);
    printf("✓ test_tensor_create_and_contiguous passed!\n");
}

void test_tensor_slice_zero_copy(void) {
    printf("Running test_tensor_slice_zero_copy...\n");

    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    size_t shape[2] = {10, 10};
    RCRE_Tensor t = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    assert(t.data != NULL);

    float* raw = (float*)t.data;
    for (int r = 0; r < 10; ++r) {
        for (int c = 0; c < 10; ++c) {
            raw[r * 10 + c] = (float)(r * 10 + c);
        }
    }

    /* Slice row 2..5, col 3..7 */
    size_t starts[2] = {2, 3};
    size_t ends[2] = {5, 7};
    RCRE_Tensor view = rcre_tensor_slice(&t, starts, ends);

    assert(view.ndim == 2);
    assert(view.shape[0] == 3);
    assert(view.shape[1] == 4);
    assert(view.strides[0] == 10);
    assert(view.strides[1] == 1);
    assert(rcre_tensor_is_contiguous(&view) == false); /* Slice across rows is non-contiguous */

    /* Verify top-left element of view is element (2, 3) = 23 */
    float first_val = *(float*)view.data;
    assert(first_val == 23.0f);

    rcre_arena_destroy(arena);
    printf("✓ test_tensor_slice_zero_copy passed!\n");
}

void test_tensor_transpose_and_permute(void) {
    printf("Running test_tensor_transpose_and_permute...\n");

    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    size_t shape[3] = {10, 20, 30};
    RCRE_Tensor t = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_UNKNOWN, arena);
    assert(t.data != NULL);

    /* Transpose dim 0 and 2 */
    RCRE_Tensor trans = rcre_tensor_transpose(&t, 0, 2);
    assert(trans.shape[0] == 30);
    assert(trans.shape[1] == 20);
    assert(trans.shape[2] == 10);
    assert(trans.strides[0] == t.strides[2]);
    assert(trans.strides[1] == t.strides[1]);
    assert(trans.strides[2] == t.strides[0]);
    assert(rcre_tensor_is_contiguous(&trans) == false);

    /* Permute layout [2, 0, 1] (HWC -> CHW equivalent) */
    uint8_t order[3] = {2, 0, 1};
    RCRE_Tensor perm = rcre_tensor_permute(&t, order);
    assert(perm.shape[0] == 30);
    assert(perm.shape[1] == 10);
    assert(perm.shape[2] == 20);
    assert(perm.strides[0] == t.strides[2]);
    assert(perm.strides[1] == t.strides[0]);
    assert(perm.strides[2] == t.strides[1]);

    rcre_arena_destroy(arena);
    printf("✓ test_tensor_transpose_and_permute passed!\n");
}

void test_tensor_reshape(void) {
    printf("Running test_tensor_reshape...\n");

    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    size_t shape[2] = {4, 6};
    RCRE_Tensor t = rcre_tensor_create(shape, 2, RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);
    assert(t.data != NULL);
    assert(rcre_tensor_numel(&t) == 24);

    size_t new_shape[3] = {2, 3, 4};
    RCRE_Tensor r = rcre_tensor_reshape(&t, new_shape, 3);
    assert(r.data == t.data); /* Zero copy */
    assert(r.ndim == 3);
    assert(r.shape[0] == 2);
    assert(r.shape[1] == 3);
    assert(r.shape[2] == 4);
    assert(r.strides[0] == 12);
    assert(r.strides[1] == 4);
    assert(r.strides[2] == 1);
    assert(rcre_tensor_is_contiguous(&r) == true);

    rcre_arena_destroy(arena);
    printf("✓ test_tensor_reshape passed!\n");
}

int main(void) {
    printf("=== RCRE Strided Tensor Test Suite ===\n");
    test_tensor_create_and_contiguous();
    test_tensor_slice_zero_copy();
    test_tensor_transpose_and_permute();
    test_tensor_reshape();
    printf("=== All Tensor Tests Passed ===\n");
    return 0;
}
