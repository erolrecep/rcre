/**
 * @file test_sparse.c
 * @brief Unit tests for RCRE Sparse Matrix Engine (CSR, BCSR4, SpMV, Pruning)
 */

#include "rcre.h"
#include "rcre_sparse.h"
#include "test_framework.h"
#include <math.h>

static void test_sparse_csr_lifecycle_and_spmv(void) {
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 6 x 8 test matrix with ~80% zeros */
    size_t shape[2] = {6, 8};
    RCRE_Tensor dense = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    ASSERT_NOT_NULL(dense.data);
    float* d = (float*)dense.data;
    memset(d, 0, 6 * 8 * sizeof(float));

    /* Insert specific non-zeros */
    d[0 * 8 + 1] = 2.5f;
    d[0 * 8 + 4] = -1.2f;
    d[1 * 8 + 7] = 3.8f;
    d[3 * 8 + 0] = 4.1f;
    d[3 * 8 + 2] = 0.9f;
    d[4 * 8 + 5] = -5.0f;
    d[5 * 8 + 3] = 1.7f;
    d[5 * 8 + 6] = 2.2f;

    /* Convert to CSR */
    RCRE_CSRMatrix csr = rcre_sparse_csr_from_dense(&dense, 0.0f, arena);
    ASSERT_UINT_EQ(6, csr.rows);
    ASSERT_UINT_EQ(8, csr.cols);
    ASSERT_UINT_EQ(8, csr.nnz);
    ASSERT_UINT_EQ(0, csr.row_pointers[0]);
    ASSERT_UINT_EQ(2, csr.row_pointers[1]); /* row 0 has 2 nonzeros */
    ASSERT_UINT_EQ(3, csr.row_pointers[2]); /* row 1 has 1 nonzero */
    ASSERT_UINT_EQ(3, csr.row_pointers[3]); /* row 2 has 0 nonzeros */
    ASSERT_UINT_EQ(8, csr.row_pointers[6]); /* total 8 nonzeros */

    /* Test vector x of length 8 */
    float x[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float y_csr[6] = {0.0f};
    float y_ref[6] = {0.0f};

    /* Dense reference product */
    for (size_t r = 0; r < 6; ++r) {
        float sum = 0.0f;
        for (size_t c = 0; c < 8; ++c) {
            sum += d[r * 8 + c] * x[c];
        }
        y_ref[r] = sum;
    }

    int rc = rcre_sparse_csr_spmv(&csr, x, y_csr);
    ASSERT_INT_EQ(RCRE_OK, rc);

    for (size_t r = 0; r < 6; ++r) {
        ASSERT_FLOAT_NEAR(y_ref[r], y_csr[r], 1e-5f);
    }

    /* Reconstruct dense and assert bit-exact match */
    RCRE_Tensor recon = {0};
    rc = rcre_sparse_csr_to_dense(&csr, arena, &recon);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_NOT_NULL(recon.data);
    const float* recon_data = (const float*)recon.data;
    for (size_t i = 0; i < 6 * 8; ++i) {
        ASSERT_FLOAT_NEAR(d[i], recon_data[i], 1e-6f);
    }

    rcre_arena_destroy(arena);
}

static void test_sparse_bcsr4_lifecycle_and_simd_spmv(void) {
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 12 x 16 test matrix (multiple of 4) */
    size_t shape[2] = {12, 16};
    RCRE_Tensor dense = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* d = (float*)dense.data;
    memset(d, 0, 12 * 16 * sizeof(float));

    /* Populate only 2 active 4x4 blocks out of 3x4=12 total blocks */
    /* Block (0, 1): rows 0..3, cols 4..7 */
    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 4; c < 8; ++c) {
            d[r * 16 + c] = (float)(r + c + 1) * 0.5f;
        }
    }

    /* Block (2, 3): rows 8..11, cols 12..15 */
    for (size_t r = 8; r < 12; ++r) {
        for (size_t c = 12; c < 16; ++c) {
            d[r * 16 + c] = (float)(r * 2 - c) * 0.25f;
        }
    }

    /* Convert to BCSR4 */
    RCRE_BCSR4Matrix bcsr = rcre_sparse_bcsr4_from_dense(&dense, 0.0f, arena);
    ASSERT_UINT_EQ(12, bcsr.rows);
    ASSERT_UINT_EQ(16, bcsr.cols);
    ASSERT_UINT_EQ(3, bcsr.block_rows);
    ASSERT_UINT_EQ(4, bcsr.block_cols);
    ASSERT_UINT_EQ(2, bcsr.num_blocks);
    ASSERT_UINT_EQ(0, (uintptr_t)bcsr.block_values % 64); /* 64-byte aligned */

    ASSERT_UINT_EQ(0, bcsr.block_row_pointers[0]);
    ASSERT_UINT_EQ(1, bcsr.block_row_pointers[1]); /* Row block 0 has 1 block */
    ASSERT_UINT_EQ(1, bcsr.block_row_pointers[2]); /* Row block 1 has 0 blocks */
    ASSERT_UINT_EQ(2, bcsr.block_row_pointers[3]); /* Row block 2 has 1 block */

    ASSERT_UINT_EQ(1, bcsr.block_col_indices[0]); /* Block (0, 1) col index */
    ASSERT_UINT_EQ(3, bcsr.block_col_indices[1]); /* Block (2, 3) col index */

    /* Test vector x of length 16 */
    float x[16];
    for (size_t i = 0; i < 16; ++i) {
        x[i] = (float)(i + 1) * 0.1f;
    }

    float y_bcsr[12] = {0.0f};
    float y_ref[12] = {0.0f};

    /* Dense reference */
    for (size_t r = 0; r < 12; ++r) {
        float sum = 0.0f;
        for (size_t c = 0; c < 16; ++c) {
            sum += d[r * 16 + c] * x[c];
        }
        y_ref[r] = sum;
    }

    /* SpMV execution (SIMD accelerated) */
    int rc = rcre_sparse_bcsr4_spmv(&bcsr, x, y_bcsr);
    ASSERT_INT_EQ(RCRE_OK, rc);

    for (size_t r = 0; r < 12; ++r) {
        ASSERT_FLOAT_NEAR(y_ref[r], y_bcsr[r], 1e-4f);
    }

    /* Reconstruct dense and assert equivalence */
    RCRE_Tensor recon = {0};
    rc = rcre_sparse_bcsr4_to_dense(&bcsr, arena, &recon);
    ASSERT_INT_EQ(RCRE_OK, rc);
    const float* recon_data = (const float*)recon.data;
    for (size_t i = 0; i < 12 * 16; ++i) {
        ASSERT_FLOAT_NEAR(d[i], recon_data[i], 1e-5f);
    }

    rcre_arena_destroy(arena);
}

static void test_sparse_bcsr4_non_multiple_of_4_dimensions(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 7 x 9 matrix (neither dimension multiple of 4) */
    size_t shape[2] = {7, 9};
    RCRE_Tensor dense = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* d = (float*)dense.data;
    memset(d, 0, 7 * 9 * sizeof(float));

    d[1 * 9 + 2] = 4.5f;
    d[6 * 9 + 8] = -3.2f; /* Edge boundary element */

    RCRE_BCSR4Matrix bcsr = rcre_sparse_bcsr4_from_dense(&dense, 0.0f, arena);
    ASSERT_UINT_EQ(7, bcsr.rows);
    ASSERT_UINT_EQ(9, bcsr.cols);
    ASSERT_UINT_EQ(2, bcsr.block_rows); /* (7 + 3)/4 = 2 */
    ASSERT_UINT_EQ(3, bcsr.block_cols); /* (9 + 3)/4 = 3 */
    ASSERT_UINT_EQ(2, bcsr.num_blocks);

    float x[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float y_bcsr[7] = {0.0f};
    float y_ref[7] = {0.0f};

    for (size_t r = 0; r < 7; ++r) {
        float sum = 0.0f;
        for (size_t c = 0; c < 9; ++c) {
            sum += d[r * 9 + c] * x[c];
        }
        y_ref[r] = sum;
    }

    int rc = rcre_sparse_bcsr4_spmv(&bcsr, x, y_bcsr);
    ASSERT_INT_EQ(RCRE_OK, rc);

    for (size_t r = 0; r < 7; ++r) {
        ASSERT_FLOAT_NEAR(y_ref[r], y_bcsr[r], 1e-4f);
    }

    rcre_arena_destroy(arena);
}

static void test_sparse_pruning_scalar_and_block(void) {
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 16 x 16 tensor (256 elements, 16 4x4 blocks) */
    size_t shape[2] = {16, 16};
    RCRE_Tensor tensor = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* data = (float*)tensor.data;

    /* Fill with deterministic gradients */
    for (size_t i = 0; i < 256; ++i) {
        data[i] = (float)(i + 1);
    }

    /* 1. Scalar pruning: prune bottom 75% */
    float threshold = 0.0f;
    int rc = rcre_sparse_prune_scalar(&tensor, 0.75f, &threshold, arena);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_TRUE(threshold > 0.0f);

    RCRE_SparsityStats stats = rcre_sparse_analyze_tensor(&tensor, 0.0f);
    ASSERT_TRUE(stats.scalar_sparsity >= 0.74f);

    /* 2. Re-fill and test block pruning */
    for (size_t i = 0; i < 256; ++i) {
        data[i] = (float)(i + 1);
    }

    float block_thresh = 0.0f;
    rc = rcre_sparse_prune_block4(&tensor, 0.50f, &block_thresh, arena);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_TRUE(block_thresh > 0.0f);

    RCRE_SparsityStats block_stats = rcre_sparse_analyze_tensor(&tensor, 0.0f);
    ASSERT_TRUE(block_stats.block_sparsity >= 0.49f);

    rcre_arena_destroy(arena);
}

static void test_sparse_bcsr4_serialization(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t shape[2] = {8, 8};
    RCRE_Tensor dense = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* d = (float*)dense.data;
    memset(d, 0, 8 * 8 * sizeof(float));

    /* Only Block (0, 0) and Block (1, 1) active */
    d[0 * 8 + 0] = 5.0f;
    d[5 * 8 + 6] = -7.2f;

    RCRE_BCSR4Matrix original = rcre_sparse_bcsr4_from_dense(&dense, 0.0f, arena);
    ASSERT_UINT_EQ(2, original.num_blocks);

    const char* filepath = "/tmp/rcre_test_matrix.rbcsr";
    int rc = rcre_sparse_bcsr4_save(&original, filepath);
    ASSERT_INT_EQ(RCRE_OK, rc);

    /* Load into new arena */
    RCRE_Arena* load_arena = rcre_arena_create(2 * 1024 * 1024);
    RCRE_BCSR4Matrix loaded = {0};
    rc = rcre_sparse_bcsr4_load(filepath, load_arena, &loaded);
    ASSERT_INT_EQ(RCRE_OK, rc);

    ASSERT_UINT_EQ(original.rows, loaded.rows);
    ASSERT_UINT_EQ(original.cols, loaded.cols);
    ASSERT_UINT_EQ(original.num_blocks, loaded.num_blocks);
    ASSERT_UINT_EQ(original.block_rows, loaded.block_rows);
    ASSERT_UINT_EQ(original.block_cols, loaded.block_cols);

    /* Verify SpMV outputs match identically */
    float x[8] = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float y_orig[8] = {0.0f};
    float y_loaded[8] = {0.0f};

    int rc1 = rcre_sparse_bcsr4_spmv(&original, x, y_orig);
    int rc2 = rcre_sparse_bcsr4_spmv(&loaded, x, y_loaded);
    ASSERT_INT_EQ(RCRE_OK, rc1);
    ASSERT_INT_EQ(RCRE_OK, rc2);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_FLOAT_NEAR(y_orig[i], y_loaded[i], 1e-6f);
    }

    remove(filepath);
    rcre_arena_destroy(load_arena);
    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Sparse Matrix & Pruning Engine (SPARSE-REQ-01)");
    TEST_RUN(test_sparse_csr_lifecycle_and_spmv);
    TEST_RUN(test_sparse_bcsr4_lifecycle_and_simd_spmv);
    TEST_RUN(test_sparse_bcsr4_non_multiple_of_4_dimensions);
    TEST_RUN(test_sparse_pruning_scalar_and_block);
    TEST_RUN(test_sparse_bcsr4_serialization);
    return test_suite_report();
}
