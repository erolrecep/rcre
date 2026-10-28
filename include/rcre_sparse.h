/**
 * @file rcre_sparse.h
 * @brief RCRE High-Performance Sparse Matrix Engine & Pruning Suite (ISO C23)
 *
 * Implements hardware-aligned sparse matrix formats and algorithms:
 *  - Standard Compressed Sparse Row (CSR) for arbitrary sparsity patterns
 *  - 4x4 Block Compressed Sparse Row (BCSR4) with 64-byte cache-line alignment
 *  - SIMD-accelerated Sparse Matrix-Vector multiplication (SpMV: y = A * x)
 *  - Unstructured & 4x4 block-structured neural network pruning
 *  - Linear memory arena integration (zero heap allocations in hot paths)
 *
 * Part of Recep's Code Run Everywhere (RCRE).
 */

#ifndef RCRE_SPARSE_H
#define RCRE_SPARSE_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Sparse Storage Formats & Structures                                       */
/* ========================================================================= */

typedef enum {
    RCRE_SPARSE_CSR   = 0, /* Standard Compressed Sparse Row               */
    RCRE_SPARSE_BCSR4 = 1  /* 4x4 Block Compressed Sparse Row (cache-tiled) */
} RCRE_SparseType;

/**
 * @brief Standard Compressed Sparse Row (CSR) Matrix
 */
typedef struct RCRE_CSRMatrix {
    size_t    rows;           /* Number of logical rows (M)                    */
    size_t    cols;           /* Number of logical columns (N)                 */
    size_t    nnz;            /* Number of non-zero scalar values              */
    float*    values;         /* Array of non-zero values [nnz]                */
    uint32_t* col_indices;    /* Column indices for each value [nnz]           */
    uint32_t* row_pointers;   /* Offsets into values/col_indices [rows + 1]    */
    bool      is_view;        /* true if arrays are owned by external buffer   */
} RCRE_CSRMatrix;

/**
 * @brief 4x4 Block Compressed Sparse Row (BCSR4) Matrix
 *
 * Stores non-zero values in dense 4x4 column-major or row-major blocks (16 floats
 * = exactly 64 bytes = 1 CPU cache line). This enables streaming directly into
 * SIMD vector registers without indirect gathering stalls.
 */
typedef struct RCRE_BCSR4Matrix {
    size_t    rows;               /* Logical rows (M)                                 */
    size_t    cols;               /* Logical columns (N)                              */
    size_t    block_rows;         /* Block grid height = (rows + 3) / 4               */
    size_t    block_cols;         /* Block grid width  = (cols + 3) / 4               */
    size_t    num_blocks;         /* Number of non-zero 4x4 blocks                    */
    float*    block_values;       /* 64-byte aligned blocks [num_blocks * 16] (F32)   */
    uint32_t* block_col_indices;  /* Column block indices [num_blocks]                */
    uint32_t* block_row_pointers; /* Offsets into blocks [block_rows + 1]             */
    bool      is_view;            /* true if arrays are owned by external buffer      */
} RCRE_BCSR4Matrix;

/**
 * @brief Sparsity and Compression Telemetry Statistics
 */
typedef struct RCRE_SparsityStats {
    size_t total_elements;        /* Logical M * N                                    */
    size_t non_zero_elements;     /* Count of scalar non-zeros                        */
    size_t non_zero_blocks;       /* Count of active 4x4 blocks (for BCSR)            */
    float  scalar_sparsity;       /* 1.0 - (non_zero_elements / total_elements)       */
    float  block_sparsity;        /* 1.0 - (active_blocks / total_blocks)             */
    size_t dense_memory_bytes;    /* Dense storage = M * N * sizeof(float)            */
    size_t sparse_memory_bytes;   /* Sparse structure size in bytes                   */
    float  compression_ratio;     /* dense_memory_bytes / sparse_memory_bytes         */
} RCRE_SparsityStats;

/* ========================================================================= */
/* Conversion & Constructor Primitives                                       */
/* ========================================================================= */

/**
 * @brief Converts a dense 2D F32 tensor into a Compressed Sparse Row (CSR) matrix.
 * @param dense Source 2D tensor [M, N] (F32).
 * @param threshold Magnitude threshold: elements with |val| <= threshold are treated as zero.
 * @param arena Memory arena to allocate sparse arrays.
 * @return Newly constructed RCRE_CSRMatrix (arena-allocated).
 */
RCRE_NODISCARD RCRE_CSRMatrix rcre_sparse_csr_from_dense(
    const RCRE_Tensor* dense,
    float threshold,
    RCRE_Arena* arena
);

/**
 * @brief Reconstructs a dense 2D F32 tensor from a CSR matrix.
 * @param csr Source CSR matrix.
 * @param arena Memory arena for destination tensor data.
 * @param dst Output tensor of shape [rows, cols] (F32).
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_sparse_csr_to_dense(
    const RCRE_CSRMatrix* csr,
    RCRE_Arena* arena,
    RCRE_Tensor* dst
);

/**
 * @brief Converts a dense 2D F32 tensor into a 4x4 Block Compressed Sparse Row (BCSR4) matrix.
 * @param dense Source 2D tensor [M, N] (F32).
 * @param threshold Magnitude threshold: blocks with max(|val|) <= threshold are dropped.
 * @param arena Memory arena (values array is 64-byte aligned).
 * @return Newly constructed RCRE_BCSR4Matrix.
 */
RCRE_NODISCARD RCRE_BCSR4Matrix rcre_sparse_bcsr4_from_dense(
    const RCRE_Tensor* dense,
    float threshold,
    RCRE_Arena* arena
);

/**
 * @brief Reconstructs a dense 2D F32 tensor from a BCSR4 matrix.
 * @param bcsr Source BCSR4 matrix.
 * @param arena Memory arena for destination tensor data.
 * @param dst Output tensor of shape [rows, cols] (F32).
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_sparse_bcsr4_to_dense(
    const RCRE_BCSR4Matrix* bcsr,
    RCRE_Arena* arena,
    RCRE_Tensor* dst
);

/* ========================================================================= */
/* Sparse Matrix-Vector Multiplication (SpMV)                                */
/* ========================================================================= */

/**
 * @brief Computes y = A * x using standard CSR representation.
 * @param A Source CSR matrix [M, N].
 * @param x Input dense vector of length N (F32).
 * @param y Output dense vector of length M (F32).
 * @return RCRE_OK on success.
 */
RCRE_NODISCARD int rcre_sparse_csr_spmv(
    const RCRE_CSRMatrix* A,
    const float* x,
    float* y
);

/**
 * @brief Computes y = A * x using hardware-accelerated 4x4 BCSR representation.
 * Automatically dispatches to ARM NEON (128-bit) or x86 AVX2 (256-bit) when available.
 * @param A Source BCSR4 matrix [M, N].
 * @param x Input dense vector of length N (F32).
 * @param y Output dense vector of length M (F32).
 * @return RCRE_OK on success.
 */
RCRE_NODISCARD int rcre_sparse_bcsr4_spmv(
    const RCRE_BCSR4Matrix* A,
    const float* x,
    float* y
);

/* ========================================================================= */
/* Pruning & Telemetry Primitives                                            */
/* ========================================================================= */

/**
 * @brief Prunes a dense 2D tensor in-place to achieve a targeted scalar sparsity ratio.
 * @param tensor 2D F32 tensor to prune.
 * @param target_sparsity Desired fraction of zeros (e.g., 0.75f for 75% sparsity).
 * @param out_threshold Pointer to receive the cutoff threshold applied (optional).
 * @param scratch Transient memory arena for sorting/histogram computation.
 * @return RCRE_OK on success.
 */
RCRE_NODISCARD int rcre_sparse_prune_scalar(
    RCRE_Tensor* tensor,
    float target_sparsity,
    float* out_threshold,
    RCRE_Arena* scratch
);

/**
 * @brief Prunes a dense 2D tensor in-place to achieve a targeted 4x4 block sparsity ratio.
 * Drops entire 4x4 blocks whose Frobenius norm is below the cutoff threshold.
 * @param tensor 2D F32 tensor to prune.
 * @param target_block_sparsity Desired fraction of zero blocks (e.g. 0.70f).
 * @param out_threshold Pointer to receive the cutoff block norm applied (optional).
 * @param scratch Transient memory arena.
 * @return RCRE_OK on success.
 */
RCRE_NODISCARD int rcre_sparse_prune_block4(
    RCRE_Tensor* tensor,
    float target_block_sparsity,
    float* out_threshold,
    RCRE_Arena* scratch
);

/**
 * @brief Computes detailed sparsity and compression statistics for a dense tensor.
 * @param tensor Dense 2D F32 tensor.
 * @param threshold Magnitude threshold for scalar non-zeros.
 * @return Sparsity statistics including dense vs sparse memory footprint.
 */
RCRE_NODISCARD RCRE_SparsityStats rcre_sparse_analyze_tensor(
    const RCRE_Tensor* tensor,
    float threshold
);

/* ========================================================================= */
/* Binary Serialization (.rcs / .rbcsr)                                      */
/* ========================================================================= */

/**
 * @brief Serializes a BCSR4 matrix to disk.
 * @param bcsr Matrix to serialize.
 * @param filepath Destination binary file path.
 * @return RCRE_OK on success, RCRE_ERR_IO on failure.
 */
RCRE_NODISCARD int rcre_sparse_bcsr4_save(
    const RCRE_BCSR4Matrix* bcsr,
    const char* filepath
);

/**
 * @brief Deserializes a BCSR4 matrix from disk into a memory arena.
 * @param filepath Source binary file path.
 * @param arena Destination memory arena.
 * @param out_bcsr Pointer to receive loaded matrix.
 * @return RCRE_OK on success, RCRE_ERR_IO or RCRE_ERR_OUT_OF_MEMORY on failure.
 */
RCRE_NODISCARD int rcre_sparse_bcsr4_load(
    const char* filepath,
    RCRE_Arena* arena,
    RCRE_BCSR4Matrix* out_bcsr
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_SPARSE_H */

