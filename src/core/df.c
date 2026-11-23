/**
 * @file df.c
 * @brief RCRE Columnar DataFrame Engine Implementation (ISO C23)
 */

#include "rcre_df.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================= */
/* 1. DataFrame Lifecycle & Construction                                     */
/* ========================================================================= */

RCRE_NODISCARD int rcre_df_create(
    const char* const* col_names,
    const RCRE_DType* col_dtypes,
    size_t num_cols,
    size_t num_rows,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
) {
    if (!col_dtypes || num_cols == 0 || num_cols > RCRE_DF_MAX_COLS || !arena || !out_df) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_df, 0, sizeof(RCRE_DataFrame));

    RCRE_Column* cols = (RCRE_Column*)rcre_arena_alloc(arena, num_cols * sizeof(RCRE_Column), alignof(RCRE_Column));
    if (!cols) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    size_t val_bytes = (num_rows + 7) >> 3;

    for (size_t c = 0; c < num_cols; ++c) {
        if (col_names && col_names[c]) {
            strncpy(cols[c].name, col_names[c], RCRE_DF_MAX_NAME - 1);
            cols[c].name[RCRE_DF_MAX_NAME - 1] = '\0';
        } else {
            snprintf(cols[c].name, sizeof(cols[c].name), "col_%zu", c);
        }

        cols[c].dtype = col_dtypes[c];
        cols[c].length = num_rows;
        cols[c].null_count = 0;
        cols[c].is_view = false;

        size_t elem_sz = rcre_dtype_size(cols[c].dtype);
        size_t col_bytes = num_rows * elem_sz;
        if (col_bytes > 0) {
            cols[c].data = rcre_arena_alloc(arena, col_bytes, 64);
            if (!cols[c].data) return RCRE_ERR_OUT_OF_MEMORY;
            memset(cols[c].data, 0, col_bytes);
        } else {
            cols[c].data = nullptr;
        }

        if (val_bytes > 0) {
            cols[c].validity = (uint8_t*)rcre_arena_alloc(arena, val_bytes, 8);
            if (!cols[c].validity) return RCRE_ERR_OUT_OF_MEMORY;
            memset(cols[c].validity, 0xFF, val_bytes); /* Default all valid */
        } else {
            cols[c].validity = nullptr;
        }
    }

    out_df->columns = cols;
    out_df->num_cols = num_cols;
    out_df->num_rows = num_rows;
    out_df->arena = arena;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_df_from_tensor(
    const RCRE_Tensor* tensor,
    const char* const* col_names,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
) {
    if (!tensor || !tensor->data || tensor->ndim != 2 || !arena || !out_df) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t num_rows = tensor->shape[0];
    size_t num_cols = tensor->shape[1];
    if (num_cols == 0 || num_cols > RCRE_DF_MAX_COLS) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_DType* dtypes = (RCRE_DType*)rcre_arena_alloc(arena, num_cols * sizeof(RCRE_DType), alignof(RCRE_DType));
    if (!dtypes) return RCRE_ERR_OUT_OF_MEMORY;
    for (size_t c = 0; c < num_cols; ++c) {
        dtypes[c] = tensor->dtype;
    }

    int rc = rcre_df_create(col_names, dtypes, num_cols, num_rows, arena, out_df);
    if (rc != RCRE_OK) return rc;

    size_t elem_sz = rcre_dtype_size(tensor->dtype);
    const uint8_t* src_base = (const uint8_t*)tensor->data;
    ptrdiff_t row_stride = tensor->strides[0] * (ptrdiff_t)elem_sz;
    ptrdiff_t col_stride = tensor->strides[1] * (ptrdiff_t)elem_sz;

    /* Copy into column-contiguous layout for SIMD memory locality */
    for (size_t c = 0; c < num_cols; ++c) {
        uint8_t* dst_col = (uint8_t*)out_df->columns[c].data;
        for (size_t r = 0; r < num_rows; ++r) {
            const uint8_t* elem_ptr = src_base + r * row_stride + c * col_stride;
            memcpy(dst_col + r * elem_sz, elem_ptr, elem_sz);
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_df_to_tensor(
    const RCRE_DataFrame* df,
    const size_t* col_indices,
    size_t num_cols,
    RCRE_DType target_dtype,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
) {
    if (!df || !arena || !out_tensor || num_cols == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t num_rows = df->num_rows;
    size_t shape[2] = { num_rows, num_cols };
    RCRE_Tensor t = rcre_tensor_create(shape, 2, target_dtype, RCRE_LAYOUT_TABULAR, arena);
    if (!t.data) return RCRE_ERR_OUT_OF_MEMORY;

    size_t elem_sz = rcre_dtype_size(target_dtype);
    uint8_t* dst_base = (uint8_t*)t.data;

    for (size_t c = 0; c < num_cols; ++c) {
        size_t c_idx = col_indices ? col_indices[c] : c;
        if (c_idx >= df->num_cols) return RCRE_ERR_INVALID_ARG;

        const RCRE_Column* col = &df->columns[c_idx];
        const uint8_t* src_col = (const uint8_t*)col->data;

        if (col->dtype == target_dtype) {
            for (size_t r = 0; r < num_rows; ++r) {
                uint8_t* dst_elem = dst_base + (r * num_cols + c) * elem_sz;
                memcpy(dst_elem, src_col + r * elem_sz, elem_sz);
            }
        } else if (col->dtype == RCRE_DTYPE_F32 && target_dtype == RCRE_DTYPE_F32) {
            const float* s = (const float*)col->data;
            float* d = (float*)t.data;
            for (size_t r = 0; r < num_rows; ++r) {
                d[r * num_cols + c] = s[r];
            }
        } else if (col->dtype == RCRE_DTYPE_I32 && target_dtype == RCRE_DTYPE_F32) {
            const int32_t* s = (const int32_t*)col->data;
            float* d = (float*)t.data;
            for (size_t r = 0; r < num_rows; ++r) {
                d[r * num_cols + c] = (float)s[r];
            }
        } else {
            /* General casting fallback */
            for (size_t r = 0; r < num_rows; ++r) {
                float val = 0.0f;
                if (col->dtype == RCRE_DTYPE_U8)  val = (float)((const uint8_t*)col->data)[r];
                if (col->dtype == RCRE_DTYPE_I32) val = (float)((const int32_t*)col->data)[r];
                if (col->dtype == RCRE_DTYPE_F32) val = ((const float*)col->data)[r];
                ((float*)t.data)[r * num_cols + c] = val;
            }
        }
    }

    *out_tensor = t;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_df_from_csv(
    const char* filepath,
    bool has_header,
    char delimiter,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
) {
    if (!filepath || !arena || !out_df) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_MappedFile map = {0};
    int rc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_SEQUENTIAL, &map);
    if (rc != RCRE_OK) return rc;

    RCRE_Slice s = rcre_io_mmap_slice(&map);
    if (s.len == 0) {
        rcre_io_mmap_close(&map);
        return RCRE_ERR_IO;
    }

    /* 1. Identify header line and count columns */
    size_t pos = 0;
    while (pos < s.len && (s.data[pos] == ' ' || s.data[pos] == '\t' || s.data[pos] == '\r' || s.data[pos] == '\n')) {
        pos++;
    }

    size_t line_start = pos;
    while (pos < s.len && s.data[pos] != '\n' && s.data[pos] != '\r') {
        pos++;
    }
    size_t line_end = pos;

    /* Count columns in first line */
    size_t num_cols = 1;
    for (size_t i = line_start; i < line_end; ++i) {
        if (s.data[i] == delimiter) num_cols++;
    }
    if (num_cols > RCRE_DF_MAX_COLS) {
        rcre_io_mmap_close(&map);
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    char col_names_buf[RCRE_DF_MAX_COLS][RCRE_DF_MAX_NAME];
    const char* col_names[RCRE_DF_MAX_COLS];

    if (has_header) {
        size_t tok_start = line_start;
        size_t c = 0;
        for (size_t i = line_start; i <= line_end && c < num_cols; ++i) {
            if (i == line_end || s.data[i] == delimiter) {
                size_t tok_len = i - tok_start;
                while (tok_len > 0 && (s.data[tok_start] == ' ' || s.data[tok_start] == '"')) {
                    tok_start++;
                    tok_len--;
                }
                while (tok_len > 0 && (s.data[tok_start + tok_len - 1] == ' ' || s.data[tok_start + tok_len - 1] == '"' || s.data[tok_start + tok_len - 1] == '\r')) {
                    tok_len--;
                }
                size_t cp_len = tok_len < (RCRE_DF_MAX_NAME - 1) ? tok_len : (RCRE_DF_MAX_NAME - 1);
                memcpy(col_names_buf[c], s.data + tok_start, cp_len);
                col_names_buf[c][cp_len] = '\0';
                col_names[c] = col_names_buf[c];
                c++;
                tok_start = i + 1;
            }
        }
        /* Advance pos to next line */
        while (pos < s.len && (s.data[pos] == '\n' || s.data[pos] == '\r')) pos++;
    } else {
        for (size_t c = 0; c < num_cols; ++c) {
            snprintf(col_names_buf[c], sizeof(col_names_buf[c]), "col_%zu", c);
            col_names[c] = col_names_buf[c];
        }
        pos = line_start;
    }

    /* 2. Count data rows */
    size_t data_start = pos;
    size_t num_rows = 0;
    while (pos < s.len) {
        if (s.data[pos] == '\n') {
            num_rows++;
        }
        pos++;
    }
    if (s.len > data_start && s.data[s.len - 1] != '\n') {
        num_rows++;
    }

    /* 3. Allocate DataFrame columns as F32 by default */
    RCRE_DType dtypes[RCRE_DF_MAX_COLS];
    for (size_t c = 0; c < num_cols; ++c) {
        dtypes[c] = RCRE_DTYPE_F32;
    }

    rc = rcre_df_create(col_names, dtypes, num_cols, num_rows, arena, out_df);
    if (rc != RCRE_OK) {
        rcre_io_mmap_close(&map);
        return rc;
    }

    /* 4. Parse rows into columns */
    pos = data_start;
    size_t r = 0;
    while (pos < s.len && r < num_rows) {
        /* Skip leading whitespace/newlines */
        if (s.data[pos] == '\r' || s.data[pos] == '\n') {
            pos++;
            continue;
        }

        size_t row_end = pos;
        while (row_end < s.len && s.data[row_end] != '\n' && s.data[row_end] != '\r') {
            row_end++;
        }

        size_t tok_start = pos;
        size_t c = 0;
        for (size_t i = pos; i <= row_end && c < num_cols; ++i) {
            if (i == row_end || s.data[i] == delimiter) {
                RCRE_Slice tok = rcre_slice_from_parts(s.data + tok_start, i - tok_start);
                tok = rcre_slice_trim_ascii(tok);

                float val = 0.0f;
                if (tok.len == 0 || rcre_slice_eq(tok, rcre_slice_from_cstr("NaN")) ||
                    rcre_slice_eq(tok, rcre_slice_from_cstr("null")) ||
                    rcre_slice_eq(tok, rcre_slice_from_cstr("NA"))) {
                    rcre_col_set_valid(&out_df->columns[c], r, false);
                } else if (rcre_slice_parse_f32(tok, &val)) {
                    ((float*)out_df->columns[c].data)[r] = val;
                } else {
                    rcre_col_set_valid(&out_df->columns[c], r, false);
                }

                c++;
                tok_start = i + 1;
            }
        }
        r++;
        pos = row_end;
        while (pos < s.len && (s.data[pos] == '\r' || s.data[pos] == '\n')) pos++;
    }

    out_df->num_rows = r;
    rcre_io_mmap_close(&map);
    return RCRE_OK;
}

/* ========================================================================= */
/* 2. Column Selection, Slicing & Relational Operations                       */
/* ========================================================================= */

RCRE_NODISCARD intptr_t rcre_df_find_col(const RCRE_DataFrame* df, const char* name) {
    if (!df || !name) return -1;
    for (size_t i = 0; i < df->num_cols; ++i) {
        if (strcmp(df->columns[i].name, name) == 0) {
            return (intptr_t)i;
        }
    }
    return -1;
}

RCRE_NODISCARD int rcre_df_select(
    const RCRE_DataFrame* df,
    const char* const* col_names,
    size_t num_selected_cols,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
) {
    if (!df || !col_names || num_selected_cols == 0 || !arena || !out_df) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_df, 0, sizeof(RCRE_DataFrame));

    RCRE_Column* cols = (RCRE_Column*)rcre_arena_alloc(arena, num_selected_cols * sizeof(RCRE_Column), alignof(RCRE_Column));
    if (!cols) return RCRE_ERR_OUT_OF_MEMORY;

    for (size_t i = 0; i < num_selected_cols; ++i) {
        intptr_t idx = rcre_df_find_col(df, col_names[i]);
        if (idx < 0) return RCRE_ERR_INVALID_ARG;
        cols[i] = df->columns[idx];
        cols[i].is_view = true;
    }

    out_df->columns = cols;
    out_df->num_cols = num_selected_cols;
    out_df->num_rows = df->num_rows;
    out_df->arena = arena;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_df_slice(
    const RCRE_DataFrame* df,
    size_t start_row,
    size_t num_rows,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
) {
    if (!df || !arena || !out_df || (start_row + num_rows > df->num_rows)) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_df, 0, sizeof(RCRE_DataFrame));

    RCRE_Column* cols = (RCRE_Column*)rcre_arena_alloc(arena, df->num_cols * sizeof(RCRE_Column), alignof(RCRE_Column));
    if (!cols) return RCRE_ERR_OUT_OF_MEMORY;

    size_t val_bytes = (num_rows + 7) >> 3;

    for (size_t c = 0; c < df->num_cols; ++c) {
        cols[c] = df->columns[c];
        cols[c].length = num_rows;
        cols[c].is_view = true;

        size_t elem_sz = rcre_dtype_size(cols[c].dtype);
        cols[c].data = (uint8_t*)df->columns[c].data + start_row * elem_sz;

        /* Re-slice validity mask */
        if (cols[c].validity && val_bytes > 0) {
            uint8_t* val_slice = (uint8_t*)rcre_arena_alloc(arena, val_bytes, 8);
            if (!val_slice) return RCRE_ERR_OUT_OF_MEMORY;
            memset(val_slice, 0xFF, val_bytes);

            size_t nulls = 0;
            for (size_t r = 0; r < num_rows; ++r) {
                if (rcre_col_is_null(&df->columns[c], start_row + r)) {
                    val_slice[r >> 3] &= ~(uint8_t)(1U << (r & 7U));
                    nulls++;
                }
            }
            cols[c].validity = val_slice;
            cols[c].null_count = nulls;
        }
    }

    out_df->columns = cols;
    out_df->num_cols = df->num_cols;
    out_df->num_rows = num_rows;
    out_df->arena = arena;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_df_filter(
    const RCRE_DataFrame* df,
    const uint8_t* bool_mask,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
) {
    if (!df || !bool_mask || !arena || !out_df) {
        return RCRE_ERR_INVALID_ARG;
    }

    /* 1. Count matching rows */
    size_t keep_count = 0;
    for (size_t r = 0; r < df->num_rows; ++r) {
        if (bool_mask[r]) keep_count++;
    }

    /* 2. Build schema and create filtered DataFrame */
    const char* col_names[RCRE_DF_MAX_COLS];
    RCRE_DType col_dtypes[RCRE_DF_MAX_COLS];
    for (size_t c = 0; c < df->num_cols; ++c) {
        col_names[c] = df->columns[c].name;
        col_dtypes[c] = df->columns[c].dtype;
    }

    int rc = rcre_df_create(col_names, col_dtypes, df->num_cols, keep_count, arena, out_df);
    if (rc != RCRE_OK) return rc;

    if (keep_count == 0) {
        return RCRE_OK;
    }

    /* 3. Copy filtered row elements across all columns */
    for (size_t c = 0; c < df->num_cols; ++c) {
        const RCRE_Column* src_col = &df->columns[c];
        RCRE_Column* dst_col = &out_df->columns[c];
        size_t elem_sz = rcre_dtype_size(src_col->dtype);

        const uint8_t* s_data = (const uint8_t*)src_col->data;
        uint8_t* d_data = (uint8_t*)dst_col->data;

        size_t dst_r = 0;
        for (size_t src_r = 0; src_r < df->num_rows; ++src_r) {
            if (bool_mask[src_r]) {
                memcpy(d_data + dst_r * elem_sz, s_data + src_r * elem_sz, elem_sz);
                if (rcre_col_is_null(src_col, src_r)) {
                    rcre_col_set_valid(dst_col, dst_r, false);
                }
                dst_r++;
            }
        }
    }

    return RCRE_OK;
}

/* ========================================================================= */
/* 3. LSD Radix Sort (O(N) Linear Time)                                      */
/* ========================================================================= */

RCRE_NODISCARD int rcre_df_sort(
    const RCRE_DataFrame* df,
    size_t sort_col_idx,
    bool ascending,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
) {
    if (!df || sort_col_idx >= df->num_cols || !arena || !out_df) {
        return RCRE_ERR_INVALID_ARG;
    }
    size_t N = df->num_rows;
    if (N == 0) {
        return rcre_df_slice(df, 0, 0, arena, out_df);
    }

    const RCRE_Column* key_col = &df->columns[sort_col_idx];
    if (key_col->dtype != RCRE_DTYPE_F32 && key_col->dtype != RCRE_DTYPE_I32 && key_col->dtype != RCRE_DTYPE_U8) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    /* Allocate permutation index array */
    uint32_t* perm = (uint32_t*)rcre_arena_alloc(arena, N * sizeof(uint32_t), 64);
    uint32_t* temp_perm = (uint32_t*)rcre_arena_alloc(arena, N * sizeof(uint32_t), 64);
    uint32_t* keys = (uint32_t*)rcre_arena_alloc(arena, N * sizeof(uint32_t), 64);
    uint32_t* temp_keys = (uint32_t*)rcre_arena_alloc(arena, N * sizeof(uint32_t), 64);
    if (!perm || !temp_perm || !keys || !temp_keys) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Convert keys to sortable unsigned 32-bit integers */
    if (key_col->dtype == RCRE_DTYPE_F32) {
        const float* fdata = (const float*)key_col->data;
        for (size_t i = 0; i < N; ++i) {
            perm[i] = (uint32_t)i;
            uint32_t u;
            memcpy(&u, &fdata[i], 4);
            /* Flip sign bit so IEEE-754 floats sort correctly as unsigned ints */
            keys[i] = (u & 0x80000000U) ? (~u) : (u ^ 0x80000000U);
        }
    } else if (key_col->dtype == RCRE_DTYPE_I32) {
        const int32_t* idata = (const int32_t*)key_col->data;
        for (size_t i = 0; i < N; ++i) {
            perm[i] = (uint32_t)i;
            keys[i] = (uint32_t)(idata[i] ^ 0x80000000U);
        }
    } else {
        const uint8_t* udata = (const uint8_t*)key_col->data;
        for (size_t i = 0; i < N; ++i) {
            perm[i] = (uint32_t)i;
            keys[i] = (uint32_t)udata[i];
        }
    }

    /* 4 passes of 8-bit radix sort */
    for (int pass = 0; pass < 4; ++pass) {
        uint32_t counts[256] = {0};
        int shift = pass * 8;

        for (size_t i = 0; i < N; ++i) {
            uint8_t byte = (uint8_t)((keys[i] >> shift) & 0xFFU);
            counts[byte]++;
        }

        uint32_t offsets[256] = {0};
        for (int i = 1; i < 256; ++i) {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }

        for (size_t i = 0; i < N; ++i) {
            uint8_t byte = (uint8_t)((keys[i] >> shift) & 0xFFU);
            uint32_t dst_idx = offsets[byte]++;
            temp_perm[dst_idx] = perm[i];
            temp_keys[dst_idx] = keys[i];
        }

        memcpy(perm, temp_perm, N * sizeof(uint32_t));
        memcpy(keys, temp_keys, N * sizeof(uint32_t));
    }

    /* Handle descending order */
    if (!ascending) {
        for (size_t i = 0; i < N / 2; ++i) {
            uint32_t tmp = perm[i];
            perm[i] = perm[N - 1 - i];
            perm[N - 1 - i] = tmp;
        }
    }

    /* Build sorted DataFrame */
    const char* col_names[RCRE_DF_MAX_COLS];
    RCRE_DType col_dtypes[RCRE_DF_MAX_COLS];
    for (size_t c = 0; c < df->num_cols; ++c) {
        col_names[c] = df->columns[c].name;
        col_dtypes[c] = df->columns[c].dtype;
    }

    int rc = rcre_df_create(col_names, col_dtypes, df->num_cols, N, arena, out_df);
    if (rc != RCRE_OK) return rc;

    for (size_t c = 0; c < df->num_cols; ++c) {
        const RCRE_Column* src_col = &df->columns[c];
        RCRE_Column* dst_col = &out_df->columns[c];
        size_t elem_sz = rcre_dtype_size(src_col->dtype);

        const uint8_t* s_data = (const uint8_t*)src_col->data;
        uint8_t* d_data = (uint8_t*)dst_col->data;

        for (size_t i = 0; i < N; ++i) {
            uint32_t src_idx = perm[i];
            memcpy(d_data + i * elem_sz, s_data + src_idx * elem_sz, elem_sz);
            if (rcre_col_is_null(src_col, src_idx)) {
                rcre_col_set_valid(dst_col, i, false);
            }
        }
    }

    return RCRE_OK;
}

/* ========================================================================= */
/* 4. Parallel GroupBy & Aggregations                                        */
/* ========================================================================= */

typedef struct RCRE_GroupNode {
    int32_t  key;
    size_t   count;
    double   sum;
    double   min;
    double   max;
    double   m2; /* For Welford variance */
} RCRE_GroupNode;

RCRE_NODISCARD int rcre_df_groupby(
    const RCRE_DataFrame* df,
    size_t group_col_idx,
    const RCRE_AggSpec* specs,
    size_t num_specs,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
) {
    if (!df || group_col_idx >= df->num_cols || !specs || num_specs == 0 || !arena || !out_df) {
        return RCRE_ERR_INVALID_ARG;
    }

    const RCRE_Column* gcol = &df->columns[group_col_idx];
    if (gcol->dtype != RCRE_DTYPE_I32 && gcol->dtype != RCRE_DTYPE_U8) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    /* Hash table to track distinct keys */
    size_t hash_cap = 2048;
    int32_t* hash_keys = (int32_t*)rcre_arena_alloc(arena, hash_cap * sizeof(int32_t), 16);
    size_t*  hash_entry_idx = (size_t*)rcre_arena_alloc(arena, hash_cap * sizeof(size_t), 16);
    bool*    hash_occupied = (bool*)rcre_arena_alloc(arena, hash_cap * sizeof(bool), 16);
    if (!hash_keys || !hash_entry_idx || !hash_occupied) return RCRE_ERR_OUT_OF_MEMORY;
    memset(hash_occupied, 0, hash_cap * sizeof(bool));

    size_t max_distinct = 1024;
    int32_t* distinct_keys = (int32_t*)rcre_arena_alloc(arena, max_distinct * sizeof(int32_t), 16);
    size_t num_distinct = 0;

    for (size_t r = 0; r < df->num_rows; ++r) {
        int32_t key = (gcol->dtype == RCRE_DTYPE_I32) ? ((const int32_t*)gcol->data)[r] : (int32_t)((const uint8_t*)gcol->data)[r];

        uint32_t h = (uint32_t)(key * 2654435761U) % (uint32_t)hash_cap;
        while (hash_occupied[h]) {
            if (hash_keys[h] == key) break;
            h = (h + 1) % (uint32_t)hash_cap;
        }

        if (!hash_occupied[h]) {
            hash_occupied[h] = true;
            hash_keys[h] = key;
            hash_entry_idx[h] = num_distinct;
            if (num_distinct < max_distinct) {
                distinct_keys[num_distinct++] = key;
            }
        }
    }

    /* Allocate output columns: 1 key col + num_specs aggregation cols */
    size_t total_out_cols = 1 + num_specs;
    const char* out_names[RCRE_DF_MAX_COLS];
    RCRE_DType  out_dtypes[RCRE_DF_MAX_COLS];

    out_names[0] = gcol->name;
    out_dtypes[0] = gcol->dtype;

    for (size_t s = 0; s < num_specs; ++s) {
        out_names[1 + s] = specs[s].out_name ? specs[s].out_name : "agg";
        out_dtypes[1 + s] = (specs[s].op == RCRE_AGG_COUNT) ? RCRE_DTYPE_I32 : RCRE_DTYPE_F32;
    }

    int rc = rcre_df_create(out_names, out_dtypes, total_out_cols, num_distinct, arena, out_df);
    if (rc != RCRE_OK) return rc;

    /* Write group keys into column 0 */
    for (size_t g = 0; g < num_distinct; ++g) {
        if (gcol->dtype == RCRE_DTYPE_I32) {
            ((int32_t*)out_df->columns[0].data)[g] = distinct_keys[g];
        } else {
            ((uint8_t*)out_df->columns[0].data)[g] = (uint8_t)distinct_keys[g];
        }
    }

    /* Compute aggregations per spec */
    for (size_t s = 0; s < num_specs; ++s) {
        size_t t_idx = specs[s].target_col_idx;
        if (t_idx >= df->num_cols) return RCRE_ERR_INVALID_ARG;
        const RCRE_Column* target_col = &df->columns[t_idx];

        RCRE_GroupNode* stats = (RCRE_GroupNode*)rcre_arena_alloc(arena, num_distinct * sizeof(RCRE_GroupNode), alignof(RCRE_GroupNode));
        if (!stats) return RCRE_ERR_OUT_OF_MEMORY;

        for (size_t g = 0; g < num_distinct; ++g) {
            stats[g].key = distinct_keys[g];
            stats[g].count = 0;
            stats[g].sum = 0.0;
            stats[g].min = INFINITY;
            stats[g].max = -INFINITY;
            stats[g].m2 = 0.0;
        }

        for (size_t r = 0; r < df->num_rows; ++r) {
            if (rcre_col_is_null(target_col, r)) continue;

            int32_t key = (gcol->dtype == RCRE_DTYPE_I32) ? ((const int32_t*)gcol->data)[r] : (int32_t)((const uint8_t*)gcol->data)[r];
            uint32_t h = (uint32_t)(key * 2654435761U) % (uint32_t)hash_cap;
            while (hash_occupied[h] && hash_keys[h] != key) {
                h = (h + 1) % (uint32_t)hash_cap;
            }
            size_t g_idx = hash_entry_idx[h];

            double val = (target_col->dtype == RCRE_DTYPE_F32) ? (double)((const float*)target_col->data)[r] : (double)((const int32_t*)target_col->data)[r];

            stats[g_idx].count++;
            stats[g_idx].sum += val;
            if (val < stats[g_idx].min) stats[g_idx].min = val;
            if (val > stats[g_idx].max) stats[g_idx].max = val;

            /* Welford update */
            double delta = val - (stats[g_idx].sum / (double)stats[g_idx].count);
            stats[g_idx].m2 += delta * (val - ((stats[g_idx].sum - val) / (stats[g_idx].count > 1 ? (double)(stats[g_idx].count - 1) : 1.0)));
        }

        /* Write results into output column */
        RCRE_Column* dst_col = &out_df->columns[1 + s];
        for (size_t g = 0; g < num_distinct; ++g) {
            if (stats[g].count == 0) {
                rcre_col_set_valid(dst_col, g, false);
                continue;
            }

            switch (specs[s].op) {
                case RCRE_AGG_COUNT:
                    ((int32_t*)dst_col->data)[g] = (int32_t)stats[g].count;
                    break;
                case RCRE_AGG_SUM:
                    ((float*)dst_col->data)[g] = (float)stats[g].sum;
                    break;
                case RCRE_AGG_MEAN:
                    ((float*)dst_col->data)[g] = (float)(stats[g].sum / (double)stats[g].count);
                    break;
                case RCRE_AGG_MIN:
                    ((float*)dst_col->data)[g] = (float)stats[g].min;
                    break;
                case RCRE_AGG_MAX:
                    ((float*)dst_col->data)[g] = (float)stats[g].max;
                    break;
                case RCRE_AGG_VAR:
                    ((float*)dst_col->data)[g] = (stats[g].count > 1) ? (float)(stats[g].m2 / (double)(stats[g].count - 1)) : 0.0f;
                    break;
                case RCRE_AGG_STD:
                    ((float*)dst_col->data)[g] = (stats[g].count > 1) ? sqrtf((float)(stats[g].m2 / (double)(stats[g].count - 1))) : 0.0f;
                    break;
            }
        }
    }

    return RCRE_OK;
}

/* ========================================================================= */
/* 5. Apache Arrow C Data Interface Zero-Copy Export                         */
/* ========================================================================= */

static void rcre_df_arrow_noop_release_schema(RCRE_ArrowSchema* s) { if (s) s->release = nullptr; }
static void rcre_df_arrow_noop_release_array(RCRE_ArrowArray* a) { if (a) a->release = nullptr; }

RCRE_NODISCARD int rcre_col_to_arrow(
    const RCRE_Column* col,
    RCRE_Arena* arena,
    RCRE_ArrowSchema* out_schema,
    RCRE_ArrowArray* out_array
) {
    if (!col || !arena || !out_schema || !out_array) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_schema, 0, sizeof(RCRE_ArrowSchema));
    memset(out_array, 0, sizeof(RCRE_ArrowArray));

    const char* fmt = nullptr;
    switch (col->dtype) {
        case RCRE_DTYPE_U8:   fmt = "C"; break;
        case RCRE_DTYPE_I32:  fmt = "i"; break;
        case RCRE_DTYPE_I64:  fmt = "l"; break;
        case RCRE_DTYPE_F32:  fmt = "f"; break;
        case RCRE_DTYPE_F64:  fmt = "g"; break;
        case RCRE_DTYPE_F16:  fmt = "e"; break;
        case RCRE_DTYPE_BF16: fmt = "b"; break;
        default: return RCRE_ERR_UNSUPPORTED_OP;
    }

    const void** bufs = (const void**)rcre_arena_alloc(arena, 2 * sizeof(void*), alignof(void*));
    if (!bufs) return RCRE_ERR_OUT_OF_MEMORY;

    bufs[0] = col->validity;
    bufs[1] = col->data;

    out_schema->format     = fmt;
    out_schema->name       = col->name;
    out_schema->flags      = 0;
    out_schema->n_children = 0;
    out_schema->release    = rcre_df_arrow_noop_release_schema;

    out_array->length      = (int64_t)col->length;
    out_array->null_count  = (int64_t)col->null_count;
    out_array->offset      = 0;
    out_array->n_buffers   = 2;
    out_array->n_children  = 0;
    out_array->buffers     = bufs;
    out_array->release     = rcre_df_arrow_noop_release_array;

    return RCRE_OK;
}

/* ========================================================================= */
/* 6. Formatted Terminal Preview                                             */
/* ========================================================================= */

void rcre_df_print(const RCRE_DataFrame* df, size_t max_rows) {
    if (!df || df->num_cols == 0) {
        printf("Empty DataFrame\n");
        return;
    }

    size_t display_rows = (df->num_rows < max_rows) ? df->num_rows : max_rows;

    printf("\nDataFrame: [%zu rows x %zu columns]\n", df->num_rows, df->num_cols);

    /* Print header */
    for (size_t c = 0; c < df->num_cols; ++c) {
        printf("%-16s ", df->columns[c].name);
    }
    printf("\n");

    for (size_t c = 0; c < df->num_cols; ++c) {
        printf("---------------- ");
    }
    printf("\n");

    /* Print rows */
    for (size_t r = 0; r < display_rows; ++r) {
        for (size_t c = 0; c < df->num_cols; ++c) {
            const RCRE_Column* col = &df->columns[c];
            if (rcre_col_is_null(col, r)) {
                printf("%-16s ", "null");
                continue;
            }

            char buf[32];
            if (col->dtype == RCRE_DTYPE_F32) {
                snprintf(buf, sizeof(buf), "%10.4f", (double)((const float*)col->data)[r]);
            } else if (col->dtype == RCRE_DTYPE_I32) {
                snprintf(buf, sizeof(buf), "%10d", ((const int32_t*)col->data)[r]);
            } else if (col->dtype == RCRE_DTYPE_U8) {
                snprintf(buf, sizeof(buf), "%10u", (unsigned)((const uint8_t*)col->data)[r]);
            } else {
                snprintf(buf, sizeof(buf), "%10p", col->data);
            }
            printf("%-16s ", buf);
        }
        printf("\n");
    }

    if (df->num_rows > display_rows) {
        printf("... (%zu more rows)\n", df->num_rows - display_rows);
    }
    printf("\n");
}
