/**
 * @file sparse.c
 * @brief RCRE High-Performance Sparse Matrix Engine & Pruning Suite
 *
 * ISO C23 compliant, zero heap allocation in SpMV kernels.
 */

#include "rcre_sparse.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
#include <immintrin.h>
#endif

/* ========================================================================= */
/* Helper Functions                                                          */
/* ========================================================================= */

static int rcre_sparse_float_compare_asc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

/* ========================================================================= */
/* CSR Matrix Implementation                                                 */
/* ========================================================================= */

RCRE_CSRMatrix rcre_sparse_csr_from_dense(
    const RCRE_Tensor* dense,
    float threshold,
    RCRE_Arena* arena
) {
    RCRE_CSRMatrix mat = {0};
    if (!dense || !dense->data || dense->ndim != 2 || dense->dtype != RCRE_DTYPE_F32 || !arena) {
        return mat;
    }

    size_t rows = dense->shape[0];
    size_t cols = dense->shape[1];
    const float* data = (const float*)dense->data;

    /* Pass 1: Count total non-zeros */
    size_t nnz = 0;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float val = data[r * cols + c];
            if (fabsf(val) > threshold) {
                nnz++;
            }
        }
    }

    /* Allocate CSR storage in arena */
    uint32_t* row_ptrs = (uint32_t*)rcre_arena_alloc(arena, (rows + 1) * sizeof(uint32_t), 64);
    uint32_t* col_idxs = (uint32_t*)rcre_arena_alloc(arena, (nnz > 0 ? nnz : 1) * sizeof(uint32_t), 64);
    float* values = (float*)rcre_arena_alloc(arena, (nnz > 0 ? nnz : 1) * sizeof(float), 64);

    if (!row_ptrs || !col_idxs || !values) {
        return mat;
    }

    /* Pass 2: Populate values and indices */
    size_t cur_idx = 0;
    row_ptrs[0] = 0;

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float val = data[r * cols + c];
            if (fabsf(val) > threshold) {
                values[cur_idx] = val;
                col_idxs[cur_idx] = (uint32_t)c;
                cur_idx++;
            }
        }
        row_ptrs[r + 1] = (uint32_t)cur_idx;
    }

    mat.rows = rows;
    mat.cols = cols;
    mat.nnz = nnz;
    mat.values = values;
    mat.col_indices = col_idxs;
    mat.row_pointers = row_ptrs;
    mat.is_view = false;

    return mat;
}

int rcre_sparse_csr_to_dense(
    const RCRE_CSRMatrix* csr,
    RCRE_Arena* arena,
    RCRE_Tensor* dst
) {
    if (!csr || !dst) {
        return RCRE_ERR_INVALID_ARG;
    }

    /* If destination data is NULL, allocate inside arena */
    if (!dst->data) {
        if (!arena) return RCRE_ERR_INVALID_ARG;
        size_t shape[2] = {csr->rows, csr->cols};
        *dst = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
        if (!dst->data) return RCRE_ERR_OUT_OF_MEMORY;
    } else {
        if (dst->shape[0] != csr->rows || dst->shape[1] != csr->cols || dst->dtype != RCRE_DTYPE_F32) {
            return RCRE_ERR_SHAPE_MISMATCH;
        }
    }

    float* dst_data = (float*)dst->data;
    memset(dst_data, 0, csr->rows * csr->cols * sizeof(float));

    for (size_t r = 0; r < csr->rows; ++r) {
        uint32_t start = csr->row_pointers[r];
        uint32_t end = csr->row_pointers[r + 1];
        for (uint32_t k = start; k < end; ++k) {
            uint32_t c = csr->col_indices[k];
            dst_data[r * csr->cols + c] = csr->values[k];
        }
    }

    return RCRE_OK;
}

int rcre_sparse_csr_spmv(
    const RCRE_CSRMatrix* A,
    const float* x,
    float* y
) {
    if (!A || !x || !y) {
        return RCRE_ERR_INVALID_ARG;
    }

    for (size_t r = 0; r < A->rows; ++r) {
        float sum = 0.0f;
        uint32_t start = A->row_pointers[r];
        uint32_t end = A->row_pointers[r + 1];
        for (uint32_t k = start; k < end; ++k) {
            sum += A->values[k] * x[A->col_indices[k]];
        }
        y[r] = sum;
    }

    return RCRE_OK;
}

/* ========================================================================= */
/* BCSR4 Matrix Implementation                                               */
/* ========================================================================= */

RCRE_BCSR4Matrix rcre_sparse_bcsr4_from_dense(
    const RCRE_Tensor* dense,
    float threshold,
    RCRE_Arena* arena
) {
    RCRE_BCSR4Matrix mat = {0};
    if (!dense || !dense->data || dense->ndim != 2 || dense->dtype != RCRE_DTYPE_F32 || !arena) {
        return mat;
    }

    size_t rows = dense->shape[0];
    size_t cols = dense->shape[1];
    size_t block_rows = (rows + 3) / 4;
    size_t block_cols = (cols + 3) / 4;
    const float* data = (const float*)dense->data;

    /* Pass 1: Count non-zero 4x4 blocks */
    size_t num_blocks = 0;
    for (size_t br = 0; br < block_rows; ++br) {
        for (size_t bc = 0; bc < block_cols; ++bc) {
            bool has_nonzero = false;
            for (size_t r = 0; r < 4; ++r) {
                size_t gr = br * 4 + r;
                if (gr >= rows) continue;
                for (size_t c = 0; c < 4; ++c) {
                    size_t gc = bc * 4 + c;
                    if (gc >= cols) continue;
                    if (fabsf(data[gr * cols + gc]) > threshold) {
                        has_nonzero = true;
                        break;
                    }
                }
                if (has_nonzero) break;
            }
            if (has_nonzero) {
                num_blocks++;
            }
        }
    }

    /* Allocate BCSR storage (block_values must be 64-byte aligned for SIMD) */
    uint32_t* row_ptrs = (uint32_t*)rcre_arena_alloc(arena, (block_rows + 1) * sizeof(uint32_t), 64);
    uint32_t* col_idxs = (uint32_t*)rcre_arena_alloc(arena, (num_blocks > 0 ? num_blocks : 1) * sizeof(uint32_t), 64);
    float* block_vals  = (float*)rcre_arena_alloc(arena, (num_blocks > 0 ? num_blocks : 1) * 16 * sizeof(float), 64);

    if (!row_ptrs || !col_idxs || !block_vals) {
        return mat;
    }

    /* Pass 2: Populate blocks */
    size_t blk_idx = 0;
    row_ptrs[0] = 0;

    for (size_t br = 0; br < block_rows; ++br) {
        for (size_t bc = 0; bc < block_cols; ++bc) {
            bool has_nonzero = false;
            float temp_block[16] = {0.0f};

            for (size_t r = 0; r < 4; ++r) {
                size_t gr = br * 4 + r;
                for (size_t c = 0; c < 4; ++c) {
                    size_t gc = bc * 4 + c;
                    if (gr < rows && gc < cols) {
                        float v = data[gr * cols + gc];
                        temp_block[r * 4 + c] = v;
                        if (fabsf(v) > threshold) {
                            has_nonzero = true;
                        }
                    }
                }
            }

            if (has_nonzero) {
                col_idxs[blk_idx] = (uint32_t)bc;
                memcpy(&block_vals[blk_idx * 16], temp_block, 16 * sizeof(float));
                blk_idx++;
            }
        }
        row_ptrs[br + 1] = (uint32_t)blk_idx;
    }

    mat.rows = rows;
    mat.cols = cols;
    mat.block_rows = block_rows;
    mat.block_cols = block_cols;
    mat.num_blocks = num_blocks;
    mat.block_values = block_vals;
    mat.block_col_indices = col_idxs;
    mat.block_row_pointers = row_ptrs;
    mat.is_view = false;

    return mat;
}

int rcre_sparse_bcsr4_to_dense(
    const RCRE_BCSR4Matrix* bcsr,
    RCRE_Arena* arena,
    RCRE_Tensor* dst
) {
    if (!bcsr || !dst) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (!dst->data) {
        if (!arena) return RCRE_ERR_INVALID_ARG;
        size_t shape[2] = {bcsr->rows, bcsr->cols};
        *dst = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
        if (!dst->data) return RCRE_ERR_OUT_OF_MEMORY;
    } else {
        if (dst->shape[0] != bcsr->rows || dst->shape[1] != bcsr->cols || dst->dtype != RCRE_DTYPE_F32) {
            return RCRE_ERR_SHAPE_MISMATCH;
        }
    }

    float* dst_data = (float*)dst->data;
    memset(dst_data, 0, bcsr->rows * bcsr->cols * sizeof(float));

    for (size_t br = 0; br < bcsr->block_rows; ++br) {
        uint32_t start = bcsr->block_row_pointers[br];
        uint32_t end = bcsr->block_row_pointers[br + 1];

        for (uint32_t b = start; b < end; ++b) {
            uint32_t bc = bcsr->block_col_indices[b];
            const float* blk = &bcsr->block_values[b * 16];

            for (size_t r = 0; r < 4; ++r) {
                size_t gr = br * 4 + r;
                if (gr >= bcsr->rows) continue;

                for (size_t c = 0; c < 4; ++c) {
                    size_t gc = bc * 4 + c;
                    if (gc >= bcsr->cols) continue;
                    dst_data[gr * bcsr->cols + gc] = blk[r * 4 + c];
                }
            }
        }
    }

    return RCRE_OK;
}

int rcre_sparse_bcsr4_spmv(
    const RCRE_BCSR4Matrix* A,
    const float* x,
    float* y
) {
    if (!A || !x || !y) {
        return RCRE_ERR_INVALID_ARG;
    }

    memset(y, 0, A->rows * sizeof(float));

    for (size_t br = 0; br < A->block_rows; ++br) {
        uint32_t start = A->block_row_pointers[br];
        uint32_t end = A->block_row_pointers[br + 1];
        if (start == end) continue;

#if defined(__ARM_NEON) || defined(__aarch64__)
        float32x4_t vy = vdupq_n_f32(0.0f);

        for (uint32_t b = start; b < end; ++b) {
            uint32_t bc = A->block_col_indices[b];
            const float* x_ptr = &x[bc * 4];
            const float* w = &A->block_values[b * 16];
            float32x4_t vx = vld1q_f32(x_ptr);

            float32x4_t r0 = vld1q_f32(w + 0);
            float32x4_t r1 = vld1q_f32(w + 4);
            float32x4_t r2 = vld1q_f32(w + 8);
            float32x4_t r3 = vld1q_f32(w + 12);

            float p0 = vaddvq_f32(vmulq_f32(r0, vx));
            float p1 = vaddvq_f32(vmulq_f32(r1, vx));
            float p2 = vaddvq_f32(vmulq_f32(r2, vx));
            float p3 = vaddvq_f32(vmulq_f32(r3, vx));

            float32x4_t p = {p0, p1, p2, p3};
            vy = vaddq_f32(vy, p);
        }

        if (br * 4 + 4 <= A->rows) {
            vst1q_f32(&y[br * 4], vy);
        } else {
            float temp[4];
            vst1q_f32(temp, vy);
            for (size_t r = 0; r < 4 && br * 4 + r < A->rows; ++r) {
                y[br * 4 + r] = temp[r];
            }
        }
#else
        float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;

        for (uint32_t b = start; b < end; ++b) {
            uint32_t bc = A->block_col_indices[b];
            const float* x_ptr = &x[bc * 4];
            const float* w = &A->block_values[b * 16];

            float x0 = x_ptr[0], x1 = x_ptr[1], x2 = x_ptr[2], x3 = x_ptr[3];

            y0 += w[0]*x0 + w[1]*x1 + w[2]*x2 + w[3]*x3;
            y1 += w[4]*x0 + w[5]*x1 + w[6]*x2 + w[7]*x3;
            y2 += w[8]*x0 + w[9]*x1 + w[10]*x2 + w[11]*x3;
            y3 += w[12]*x0 + w[13]*x1 + w[14]*x2 + w[15]*x3;
        }

        if (br * 4 + 0 < A->rows) y[br * 4 + 0] = y0;
        if (br * 4 + 1 < A->rows) y[br * 4 + 1] = y1;
        if (br * 4 + 2 < A->rows) y[br * 4 + 2] = y2;
        if (br * 4 + 3 < A->rows) y[br * 4 + 3] = y3;
#endif
    }

    return RCRE_OK;
}

/* ========================================================================= */
/* Pruning & Sparsity Analytics                                              */
/* ========================================================================= */

int rcre_sparse_prune_scalar(
    RCRE_Tensor* tensor,
    float target_sparsity,
    float* out_threshold,
    RCRE_Arena* scratch
) {
    if (!tensor || !tensor->data || tensor->dtype != RCRE_DTYPE_F32 || !scratch) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (target_sparsity <= 0.0f) {
        if (out_threshold) *out_threshold = 0.0f;
        return RCRE_OK;
    }
    if (target_sparsity >= 1.0f) {
        memset(tensor->data, 0, tensor->numel * sizeof(float));
        if (out_threshold) *out_threshold = 1e9f;
        return RCRE_OK;
    }

    size_t numel = tensor->numel;
    float* data = (float*)tensor->data;

    /* Copy magnitudes to scratch arena for percentile threshold determination */
    RCRE_ArenaTemp temp = rcre_arena_temp_begin(scratch);
    float* mags = (float*)rcre_arena_alloc(scratch, numel * sizeof(float), 64);
    if (!mags) {
        rcre_arena_temp_end(temp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < numel; ++i) {
        mags[i] = fabsf(data[i]);
    }

    qsort(mags, numel, sizeof(float), rcre_sparse_float_compare_asc);

    size_t cutoff_idx = (size_t)(target_sparsity * (float)numel);
    if (cutoff_idx >= numel) cutoff_idx = numel - 1;
    float threshold = mags[cutoff_idx];

    /* Apply threshold pruning */
    for (size_t i = 0; i < numel; ++i) {
        if (fabsf(data[i]) <= threshold) {
            data[i] = 0.0f;
        }
    }

    rcre_arena_temp_end(temp);
    if (out_threshold) *out_threshold = threshold;
    return RCRE_OK;
}

int rcre_sparse_prune_block4(
    RCRE_Tensor* tensor,
    float target_block_sparsity,
    float* out_threshold,
    RCRE_Arena* scratch
) {
    if (!tensor || !tensor->data || tensor->ndim != 2 || tensor->dtype != RCRE_DTYPE_F32 || !scratch) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (target_block_sparsity <= 0.0f) {
        if (out_threshold) *out_threshold = 0.0f;
        return RCRE_OK;
    }

    size_t rows = tensor->shape[0];
    size_t cols = tensor->shape[1];
    size_t block_rows = (rows + 3) / 4;
    size_t block_cols = (cols + 3) / 4;
    size_t total_blocks = block_rows * block_cols;
    float* data = (float*)tensor->data;

    RCRE_ArenaTemp temp = rcre_arena_temp_begin(scratch);
    float* norms = (float*)rcre_arena_alloc(scratch, total_blocks * sizeof(float), 64);
    if (!norms) {
        rcre_arena_temp_end(temp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Compute Frobenius norm per 4x4 block */
    for (size_t br = 0; br < block_rows; ++br) {
        for (size_t bc = 0; bc < block_cols; ++bc) {
            float sum_sq = 0.0f;
            for (size_t r = 0; r < 4; ++r) {
                size_t gr = br * 4 + r;
                if (gr >= rows) continue;
                for (size_t c = 0; c < 4; ++c) {
                    size_t gc = bc * 4 + c;
                    if (gc >= cols) continue;
                    float v = data[gr * cols + gc];
                    sum_sq += v * v;
                }
            }
            norms[br * block_cols + bc] = sqrtf(sum_sq);
        }
    }

    qsort(norms, total_blocks, sizeof(float), rcre_sparse_float_compare_asc);

    size_t cutoff_idx = (size_t)(target_block_sparsity * (float)total_blocks);
    if (cutoff_idx >= total_blocks) cutoff_idx = total_blocks - 1;
    float threshold = norms[cutoff_idx];

    /* Zero out blocks with Frobenius norm <= threshold */
    for (size_t br = 0; br < block_rows; ++br) {
        for (size_t bc = 0; bc < block_cols; ++bc) {
            float sum_sq = 0.0f;
            for (size_t r = 0; r < 4; ++r) {
                size_t gr = br * 4 + r;
                if (gr >= rows) continue;
                for (size_t c = 0; c < 4; ++c) {
                    size_t gc = bc * 4 + c;
                    if (gc >= cols) continue;
                    float v = data[gr * cols + gc];
                    sum_sq += v * v;
                }
            }
            if (sqrtf(sum_sq) <= threshold) {
                for (size_t r = 0; r < 4; ++r) {
                    size_t gr = br * 4 + r;
                    if (gr >= rows) continue;
                    for (size_t c = 0; c < 4; ++c) {
                        size_t gc = bc * 4 + c;
                        if (gc >= cols) continue;
                        data[gr * cols + gc] = 0.0f;
                    }
                }
            }
        }
    }

    rcre_arena_temp_end(temp);
    if (out_threshold) *out_threshold = threshold;
    return RCRE_OK;
}

RCRE_SparsityStats rcre_sparse_analyze_tensor(
    const RCRE_Tensor* tensor,
    float threshold
) {
    RCRE_SparsityStats stats = {0};
    if (!tensor || !tensor->data || tensor->ndim != 2 || tensor->dtype != RCRE_DTYPE_F32) {
        return stats;
    }

    size_t rows = tensor->shape[0];
    size_t cols = tensor->shape[1];
    size_t total_elements = rows * cols;
    size_t block_rows = (rows + 3) / 4;
    size_t block_cols = (cols + 3) / 4;
    size_t total_blocks = block_rows * block_cols;
    const float* data = (const float*)tensor->data;

    size_t nnz = 0;
    size_t non_zero_blocks = 0;

    for (size_t br = 0; br < block_rows; ++br) {
        for (size_t bc = 0; bc < block_cols; ++bc) {
            bool blk_has_nz = false;
            for (size_t r = 0; r < 4; ++r) {
                size_t gr = br * 4 + r;
                if (gr >= rows) continue;
                for (size_t c = 0; c < 4; ++c) {
                    size_t gc = bc * 4 + c;
                    if (gc >= cols) continue;
                    if (fabsf(data[gr * cols + gc]) > threshold) {
                        nnz++;
                        blk_has_nz = true;
                    }
                }
            }
            if (blk_has_nz) {
                non_zero_blocks++;
            }
        }
    }

    stats.total_elements = total_elements;
    stats.non_zero_elements = nnz;
    stats.non_zero_blocks = non_zero_blocks;
    stats.scalar_sparsity = 1.0f - ((float)nnz / (float)total_elements);
    stats.block_sparsity = 1.0f - ((float)non_zero_blocks / (float)total_blocks);

    stats.dense_memory_bytes = total_elements * sizeof(float);
    stats.sparse_memory_bytes = (non_zero_blocks * 16 * sizeof(float)) +
                                (non_zero_blocks * sizeof(uint32_t)) +
                                ((block_rows + 1) * sizeof(uint32_t));

    stats.compression_ratio = stats.sparse_memory_bytes > 0
        ? (float)stats.dense_memory_bytes / (float)stats.sparse_memory_bytes
        : 1.0f;

    return stats;
}

/* ========================================================================= */
/* Binary Serialization                                                      */
/* ========================================================================= */

typedef struct [[gnu::packed]] {
    char     magic[4];       /* "RBCS" (0x52 0x42 0x43 0x53) */
    uint32_t version;        /* 1                            */
    uint64_t rows;
    uint64_t cols;
    uint64_t block_rows;
    uint64_t block_cols;
    uint64_t num_blocks;
} RCRE_BCSRHeader;

int rcre_sparse_bcsr4_save(
    const RCRE_BCSR4Matrix* bcsr,
    const char* filepath
) {
    if (!bcsr || !filepath) {
        return RCRE_ERR_INVALID_ARG;
    }

    FILE* f = fopen(filepath, "wb");
    if (!f) return RCRE_ERR_IO;

    RCRE_BCSRHeader hdr = {
        .magic = {'R', 'B', 'C', 'S'},
        .version = 1,
        .rows = bcsr->rows,
        .cols = bcsr->cols,
        .block_rows = bcsr->block_rows,
        .block_cols = bcsr->block_cols,
        .num_blocks = bcsr->num_blocks
    };

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return RCRE_ERR_IO;
    }

    size_t ptrs_size = (bcsr->block_rows + 1) * sizeof(uint32_t);
    if (fwrite(bcsr->block_row_pointers, ptrs_size, 1, f) != 1) {
        fclose(f);
        return RCRE_ERR_IO;
    }

    if (bcsr->num_blocks > 0) {
        if (fwrite(bcsr->block_col_indices, bcsr->num_blocks * sizeof(uint32_t), 1, f) != 1 ||
            fwrite(bcsr->block_values, bcsr->num_blocks * 16 * sizeof(float), 1, f) != 1) {
            fclose(f);
            return RCRE_ERR_IO;
        }
    }

    fclose(f);
    return RCRE_OK;
}

int rcre_sparse_bcsr4_load(
    const char* filepath,
    RCRE_Arena* arena,
    RCRE_BCSR4Matrix* out_bcsr
) {
    if (!filepath || !arena || !out_bcsr) {
        return RCRE_ERR_INVALID_ARG;
    }

    FILE* f = fopen(filepath, "rb");
    if (!f) return RCRE_ERR_IO;

    RCRE_BCSRHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 ||
        memcmp(hdr.magic, "RBCS", 4) != 0 ||
        hdr.version != 1) {
        fclose(f);
        return RCRE_ERR_INVALID_ARG;
    }

    uint32_t* row_ptrs = (uint32_t*)rcre_arena_alloc(arena, (hdr.block_rows + 1) * sizeof(uint32_t), 64);
    uint32_t* col_idxs = (uint32_t*)rcre_arena_alloc(arena, (hdr.num_blocks > 0 ? hdr.num_blocks : 1) * sizeof(uint32_t), 64);
    float* block_vals  = (float*)rcre_arena_alloc(arena, (hdr.num_blocks > 0 ? hdr.num_blocks : 1) * 16 * sizeof(float), 64);

    if (!row_ptrs || !col_idxs || !block_vals) {
        fclose(f);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    if (fread(row_ptrs, (hdr.block_rows + 1) * sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return RCRE_ERR_IO;
    }

    if (hdr.num_blocks > 0) {
        if (fread(col_idxs, hdr.num_blocks * sizeof(uint32_t), 1, f) != 1 ||
            fread(block_vals, hdr.num_blocks * 16 * sizeof(float), 1, f) != 1) {
            fclose(f);
            return RCRE_ERR_IO;
        }
    }

    fclose(f);

    out_bcsr->rows = (size_t)hdr.rows;
    out_bcsr->cols = (size_t)hdr.cols;
    out_bcsr->block_rows = (size_t)hdr.block_rows;
    out_bcsr->block_cols = (size_t)hdr.block_cols;
    out_bcsr->num_blocks = (size_t)hdr.num_blocks;
    out_bcsr->block_row_pointers = row_ptrs;
    out_bcsr->block_col_indices = col_idxs;
    out_bcsr->block_values = block_vals;
    out_bcsr->is_view = false;

    return RCRE_OK;
}
