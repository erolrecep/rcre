/**
 * @file rcre_df.h
 * @brief RCRE High-Performance Columnar DataFrame Engine (ISO C23)
 *
 * Implements Apache Arrow-compatible columnar layout, null validity bitmasks,
 * zero-copy tensor/CSV bridges, SIMD row filtering, O(N) LSD radix sort,
 * and multi-threaded grouped aggregations.
 */

#ifndef RCRE_DF_H
#define RCRE_DF_H

#include "rcre.h"
#include "rcre_io.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RCRE_DF_MAX_COLS 256
#define RCRE_DF_MAX_NAME 64

/* ========================================================================= */
/* 1. Columnar Memory Model & Apache Arrow Validity Bitmaps                  */
/* ========================================================================= */

typedef struct RCRE_Column {
    char          name[RCRE_DF_MAX_NAME]; /* Column header identifier        */
    RCRE_DType    dtype;                  /* F32, F64, F16, BF16, I32, I64, U8*/
    void*         data;                   /* 64-byte aligned data elements   */
    uint8_t*      validity;               /* Arrow null bitmap (1=valid, 0=null) */
    size_t        length;                 /* Row count                        */
    size_t        null_count;             /* Count of null/missing entries   */
    bool          is_view;                /* true if data points elsewhere   */
} RCRE_Column;

typedef struct RCRE_DataFrame {
    RCRE_Column*  columns;                /* Array of columns                */
    size_t        num_cols;               /* Number of columns               */
    size_t        num_rows;               /* Row count across all columns    */
    RCRE_Arena*   arena;                  /* Arena owning DataFrame buffers  */
} RCRE_DataFrame;

/* ========================================================================= */
/* 2. Arrow Null Validity Bitmap Primitives                                  */
/* ========================================================================= */

static inline bool rcre_col_is_null(const RCRE_Column* col, size_t row_idx) {
    if (!col || !col->validity || col->null_count == 0) {
        return false;
    }
    return (col->validity[row_idx >> 3] & (uint8_t)(1U << (row_idx & 7U))) == 0;
}

static inline void rcre_col_set_valid(RCRE_Column* col, size_t row_idx, bool is_valid) {
    if (!col || !col->validity) {
        return;
    }
    uint8_t mask = (uint8_t)(1U << (row_idx & 7U));
    size_t byte_idx = row_idx >> 3;
    if (is_valid) {
        if ((col->validity[byte_idx] & mask) == 0) {
            col->validity[byte_idx] |= mask;
            if (col->null_count > 0) col->null_count--;
        }
    } else {
        if ((col->validity[byte_idx] & mask) != 0) {
            col->validity[byte_idx] &= ~mask;
            col->null_count++;
        }
    }
}

/* ========================================================================= */
/* 3. DataFrame Lifecycle & Construction                                     */
/* ========================================================================= */

/**
 * @brief Creates an empty DataFrame with specified column schemas and length.
 * @param col_names Array of column names [num_cols].
 * @param col_dtypes Array of column dtypes [num_cols].
 * @param num_cols Number of columns.
 * @param num_rows Number of rows per column.
 * @param arena Arena to allocate column metadata and 64B-aligned data buffers.
 * @param out_df Output initialized DataFrame.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_df_create(
    const char* const* col_names,
    const RCRE_DType* col_dtypes,
    size_t num_cols,
    size_t num_rows,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
);

/**
 * @brief Constructs a zero-copy DataFrame view from a 2D numerical tensor.
 *        Each column references a strided column slice of the tensor.
 * @param tensor Source 2D contiguous or strided tensor [num_rows, num_cols].
 * @param col_names Optional array of column names (if NULL, defaults to "col_0", "col_1", ...).
 * @param arena Arena for column headers.
 * @param out_df Output DataFrame view.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_df_from_tensor(
    const RCRE_Tensor* tensor,
    const char* const* col_names,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
);

/**
 * @brief Extracts selected numerical columns from a DataFrame into a 2D contiguous RCRE_Tensor.
 * @param df Source DataFrame.
 * @param col_indices Indices of columns to extract (must share target dtype).
 * @param num_cols Number of columns to extract.
 * @param target_dtype Desired output tensor dtype (F32 recommended).
 * @param arena Destination arena.
 * @param out_tensor Output contiguous 2D tensor [num_rows, num_cols].
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_df_to_tensor(
    const RCRE_DataFrame* df,
    const size_t* col_indices,
    size_t num_cols,
    RCRE_DType target_dtype,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
);

/**
 * @brief High-speed ingest of CSV/TSV data directly into columnar DataFrame layout.
 * @param filepath Path to CSV file.
 * @param has_header Whether first row contains column headers.
 * @param delimiter Delimiter character (',' or '\t').
 * @param arena Arena for allocations.
 * @param out_df Output parsed DataFrame.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_df_from_csv(
    const char* filepath,
    bool has_header,
    char delimiter,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
);

/* ========================================================================= */
/* 4. Column Selection, Slicing & Relational Operations                       */
/* ========================================================================= */

/**
 * @brief Finds column index by name (case-sensitive).
 * @return Column index (0 .. num_cols-1) or -1 if not found.
 */
RCRE_NODISCARD intptr_t rcre_df_find_col(const RCRE_DataFrame* df, const char* name);

/**
 * @brief Creates a projection DataFrame containing a subset of columns.
 */
RCRE_NODISCARD int rcre_df_select(
    const RCRE_DataFrame* df,
    const char* const* col_names,
    size_t num_selected_cols,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
);

/**
 * @brief Creates a zero-copy row slice view of an existing DataFrame.
 */
RCRE_NODISCARD int rcre_df_slice(
    const RCRE_DataFrame* df,
    size_t start_row,
    size_t num_rows,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
);

/**
 * @brief Filters rows of a DataFrame using a boolean validity mask.
 *        Copies matching rows into a contiguous new DataFrame.
 * @param df Source DataFrame.
 * @param bool_mask Array of uint8_t boolean flags [num_rows] (1=keep, 0=drop).
 * @param arena Destination arena.
 * @param out_df Output filtered DataFrame.
 */
RCRE_NODISCARD int rcre_df_filter(
    const RCRE_DataFrame* df,
    const uint8_t* bool_mask,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
);

/**
 * @brief Sorts all columns of a DataFrame by a key column in O(N) using LSD Radix Sort.
 * @param df Source DataFrame.
 * @param sort_col_idx Index of column to sort by.
 * @param ascending True for ascending order, false for descending.
 * @param arena Destination arena for sorted DataFrame.
 * @param out_df Output sorted DataFrame.
 */
RCRE_NODISCARD int rcre_df_sort(
    const RCRE_DataFrame* df,
    size_t sort_col_idx,
    bool ascending,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
);

/* ========================================================================= */
/* 5. Parallel GroupBy & Aggregations                                        */
/* ========================================================================= */

typedef enum {
    RCRE_AGG_SUM   = 0,
    RCRE_AGG_MEAN  = 1,
    RCRE_AGG_COUNT = 2,
    RCRE_AGG_MIN   = 3,
    RCRE_AGG_MAX   = 4,
    RCRE_AGG_VAR   = 5,
    RCRE_AGG_STD   = 6
} RCRE_AggOp;

typedef struct RCRE_AggSpec {
    size_t      target_col_idx; /* Column to aggregate                       */
    RCRE_AggOp  op;             /* Aggregation operation                     */
    const char* out_name;       /* Output column name (e.g. "fare_mean")     */
} RCRE_AggSpec;

/**
 * @brief Performs high-performance hash-based GroupBy aggregation.
 * @param df Source DataFrame.
 * @param group_col_idx Column index to group by (typically integer / categorical).
 * @param specs Array of aggregation operations [num_specs].
 * @param num_specs Number of aggregation operations.
 * @param arena Destination arena.
 * @param out_df Output DataFrame with [group_key, agg_0, agg_1, ...].
 */
RCRE_NODISCARD int rcre_df_groupby(
    const RCRE_DataFrame* df,
    size_t group_col_idx,
    const RCRE_AggSpec* specs,
    size_t num_specs,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
);

/* ========================================================================= */
/* 6. Apache Arrow C Data Interface Zero-Copy Export                         */
/* ========================================================================= */

/**
 * @brief Exports a single column of a DataFrame as an Apache Arrow C Data Interface array.
 * @param col Source column.
 * @param arena Arena for ArrowSchema and ArrowArray pointers.
 * @param out_schema Output ArrowSchema.
 * @param out_array Output ArrowArray (references col->data and col->validity with 0 copies).
 */
RCRE_NODISCARD int rcre_col_to_arrow(
    const RCRE_Column* col,
    RCRE_Arena* arena,
    RCRE_ArrowSchema* out_schema,
    RCRE_ArrowArray* out_array
);

/**
 * @brief Prints an ASCII tabular preview of the DataFrame to stdout (like head/tail).
 * @param df Source DataFrame.
 * @param max_rows Maximum number of rows to display (e.g. 10).
 */
void rcre_df_print(const RCRE_DataFrame* df, size_t max_rows);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_DF_H */
