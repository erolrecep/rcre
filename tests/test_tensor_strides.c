/**
 * @file test_tensor_strides.c
 * @brief Unit tests for RCRE strided tensor engine and ShapeTracker operations
 */

#include "rcre.h"
#include "test_framework.h"

static void test_tensor_create_and_contiguous(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t shape[3] = {480, 640, 3}; /* 480x640x3 image */
    RCRE_Tensor t = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(t.data);
    ASSERT_UINT_EQ(3, t.ndim);
    ASSERT_UINT_EQ(480 * 640 * 3, rcre_tensor_numel(&t));
    ASSERT_UINT_EQ(480 * 640 * 3 * 1, rcre_tensor_nbytes(&t));

    /* Check row-major strides */
    ASSERT_UINT_EQ(1, t.strides[2]);
    ASSERT_UINT_EQ(3, t.strides[1]);
    ASSERT_UINT_EQ(640 * 3, t.strides[0]);
    ASSERT_TRUE(rcre_tensor_is_contiguous(&t));

    rcre_arena_destroy(arena);
}

static void test_tensor_slice_zero_copy(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t shape[2] = {10, 10};
    RCRE_Tensor t = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    ASSERT_NOT_NULL(t.data);

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

    ASSERT_UINT_EQ(2, view.ndim);
    ASSERT_UINT_EQ(3, view.shape[0]);
    ASSERT_UINT_EQ(4, view.shape[1]);
    ASSERT_UINT_EQ(10, view.strides[0]);
    ASSERT_UINT_EQ(1, view.strides[1]);
    ASSERT_FALSE(rcre_tensor_is_contiguous(&view)); /* Slice across rows is non-contiguous */

    /* Verify top-left element of view is element (2, 3) = 23 */
    float first_val = *(float*)view.data;
    ASSERT_FLOAT_NEAR(23.0f, first_val, 1e-6f);

    rcre_arena_destroy(arena);
}

static void test_tensor_transpose_and_permute(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t shape[3] = {10, 20, 30};
    RCRE_Tensor t = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_UNKNOWN, arena);
    ASSERT_NOT_NULL(t.data);

    /* Transpose dim 0 and 2 */
    RCRE_Tensor trans = rcre_tensor_transpose(&t, 0, 2);
    ASSERT_UINT_EQ(30, trans.shape[0]);
    ASSERT_UINT_EQ(20, trans.shape[1]);
    ASSERT_UINT_EQ(10, trans.shape[2]);
    ASSERT_UINT_EQ(t.strides[2], trans.strides[0]);
    ASSERT_UINT_EQ(t.strides[1], trans.strides[1]);
    ASSERT_UINT_EQ(t.strides[0], trans.strides[2]);
    ASSERT_FALSE(rcre_tensor_is_contiguous(&trans));

    /* Permute layout [2, 0, 1] (HWC -> CHW equivalent) */
    uint8_t order[3] = {2, 0, 1};
    RCRE_Tensor perm = rcre_tensor_permute(&t, order);
    ASSERT_UINT_EQ(30, perm.shape[0]);
    ASSERT_UINT_EQ(10, perm.shape[1]);
    ASSERT_UINT_EQ(20, perm.shape[2]);
    ASSERT_UINT_EQ(t.strides[2], perm.strides[0]);
    ASSERT_UINT_EQ(t.strides[0], perm.strides[1]);
    ASSERT_UINT_EQ(t.strides[1], perm.strides[2]);

    rcre_arena_destroy(arena);
}

static void test_tensor_reshape(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t shape[2] = {4, 6};
    RCRE_Tensor t = rcre_tensor_create(shape, 2, RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);
    ASSERT_NOT_NULL(t.data);
    ASSERT_UINT_EQ(24, rcre_tensor_numel(&t));

    size_t new_shape[3] = {2, 3, 4};
    RCRE_Tensor r = rcre_tensor_reshape(&t, new_shape, 3);
    ASSERT_TRUE(r.data == t.data); /* Zero copy */
    ASSERT_UINT_EQ(3, r.ndim);
    ASSERT_UINT_EQ(2, r.shape[0]);
    ASSERT_UINT_EQ(3, r.shape[1]);
    ASSERT_UINT_EQ(4, r.shape[2]);
    ASSERT_UINT_EQ(12, r.strides[0]);
    ASSERT_UINT_EQ(4, r.strides[1]);
    ASSERT_UINT_EQ(1, r.strides[2]);
    ASSERT_TRUE(rcre_tensor_is_contiguous(&r));

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE N-D Strided Tensor & ShapeTracker (CORE-REQ-02)");
    TEST_RUN(test_tensor_create_and_contiguous);
    TEST_RUN(test_tensor_slice_zero_copy);
    TEST_RUN(test_tensor_transpose_and_permute);
    TEST_RUN(test_tensor_reshape);
    return test_suite_report();
}
