/**
 * @file df.c
 * @brief RCRE Columnar DataFrame Engine Implementation (ISO C23)
 */

#include "rcre_df.h"
#include "rcre_prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#define RCRE_DF_HAVE_POSIX 1
#endif

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
            (void)rcre_cstr_copy_bounded(cols[c].name, sizeof(cols[c].name),
                                         rcre_slice_from_cstr(col_names[c]));
        } else {
            RCRE_StrBuf name_buf = RCRE_STRBUF_WRAP(cols[c].name);
            (void)rcre_strbuf_fmt(&name_buf, "col_%zu", c);
        }

        cols[c].dtype = col_dtypes[c];
        cols[c].length = num_rows;
        cols[c].null_count = 0;
        cols[c].is_view = false;
        cols[c].str_pool = nullptr;

        if (cols[c].dtype == RCRE_DTYPE_STR_ID) {
            RCRE_StrPool* sp = (RCRE_StrPool*)rcre_arena_alloc(arena, sizeof(RCRE_StrPool), alignof(RCRE_StrPool));
            if (!sp) return RCRE_ERR_OUT_OF_MEMORY;
            size_t init_cap = (num_rows > 16U) ? ((num_rows < 16384U) ? num_rows : 16384U) : 16U;
            int sprc = rcre_strpool_init(sp, arena, init_cap);
            if (sprc != RCRE_OK) return sprc;
            cols[c].str_pool = sp;
        }

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

RCRE_NODISCARD RCRE_Slice rcre_col_get_str(const RCRE_Column* col, size_t row_idx) {
    if (!col || col->dtype != RCRE_DTYPE_STR_ID || !col->str_pool ||
        !col->data || row_idx >= col->length || rcre_col_is_null(col, row_idx)) {
        return rcre_slice_empty();
    }
    RCRE_StrId id = ((const RCRE_StrId*)col->data)[row_idx];
    return rcre_strpool_get(col->str_pool, id);
}

RCRE_NODISCARD int rcre_col_set_str(RCRE_Column* col, size_t row_idx, RCRE_Slice val) {
    if (!col || col->dtype != RCRE_DTYPE_STR_ID || !col->str_pool || !col->data || row_idx >= col->length) {
        return RCRE_ERR_INVALID_ARG;
    }
    RCRE_StrId id = rcre_strpool_intern(col->str_pool, val);
    if (id == RCRE_STR_ID_INVALID) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    ((RCRE_StrId*)col->data)[row_idx] = id;
    rcre_col_set_valid(col, row_idx, true);
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
    if (gcol->dtype != RCRE_DTYPE_I32 && gcol->dtype != RCRE_DTYPE_U8 && gcol->dtype != RCRE_DTYPE_STR_ID) {
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
        int32_t key = (gcol->dtype == RCRE_DTYPE_I32)
            ? ((const int32_t*)gcol->data)[r]
            : ((gcol->dtype == RCRE_DTYPE_STR_ID)
                ? (int32_t)((const RCRE_StrId*)gcol->data)[r]
                : (int32_t)((const uint8_t*)gcol->data)[r]);

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
    if (gcol->dtype == RCRE_DTYPE_STR_ID) {
        out_df->columns[0].str_pool = gcol->str_pool;
    }
    for (size_t g = 0; g < num_distinct; ++g) {
        if (gcol->dtype == RCRE_DTYPE_I32) {
            ((int32_t*)out_df->columns[0].data)[g] = distinct_keys[g];
        } else if (gcol->dtype == RCRE_DTYPE_STR_ID) {
            ((RCRE_StrId*)out_df->columns[0].data)[g] = (RCRE_StrId)distinct_keys[g];
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

            int32_t key = (gcol->dtype == RCRE_DTYPE_I32)
                ? ((const int32_t*)gcol->data)[r]
                : ((gcol->dtype == RCRE_DTYPE_STR_ID)
                    ? (int32_t)((const RCRE_StrId*)gcol->data)[r]
                    : (int32_t)((const uint8_t*)gcol->data)[r]);
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
            if (col->dtype == RCRE_DTYPE_STR_ID) {
                RCRE_Slice s = rcre_col_get_str(col, r);
                printf("%-16.*s ", (int)(s.len > 16U ? 16U : s.len), s.data ? (const char*)s.data : "");
                continue;
            } else if (col->dtype == RCRE_DTYPE_F32) {
                snprintf(buf, sizeof(buf), "%10.4f", (double)((const float*)col->data)[r]);
            } else if (col->dtype == RCRE_DTYPE_F64) {
                snprintf(buf, sizeof(buf), "%10.4f", ((const double*)col->data)[r]);
            } else if (col->dtype == RCRE_DTYPE_I32) {
                snprintf(buf, sizeof(buf), "%10d", ((const int32_t*)col->data)[r]);
            } else if (col->dtype == RCRE_DTYPE_I64) {
                snprintf(buf, sizeof(buf), "%10lld", (long long)((const int64_t*)col->data)[r]);
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

/* ========================================================================= */
/* 7. Morsel-Driven Parallel Streaming CSV GroupBy (Out-of-Core ETL Engine)  */
/* ========================================================================= */

typedef struct rcre_df__stream_slot {
    uint64_t w0;        /* 8B: masked first <=8 bytes of UTF-8 key           */
    double   sum;       /* 8B: running sum                                   */
    double   min;       /* 8B: running minimum                               */
    double   max;       /* 8B: running maximum                               */
    double   m2;        /* 8B: Welford sum of squared deviations (VAR/STD)   */
    uint64_t count;     /* 8B: observation count (0 == empty slot)           */
    uint32_t name_off;  /* 4B: byte offset in worker contiguous key_buf      */
    uint16_t name_len;  /* 2B: key byte length                               */
    uint16_t _pad;      /* 2B: alignment padding                             */
} rcre_df__stream_slot;

#define RCRE_DF_FIB_MUL 0x9E3779B97F4A7C15ULL
#define RCRE_DF_MIX_MUL 0x517CC1B727220A95ULL

#define RCRE_DF_INVALID_KEY_OFF UINT32_MAX

typedef struct rcre_df__stream_worker {
    alignas(64) rcre_df__stream_slot* slots;
    uint8_t*     key_buf;
    size_t       key_used;
    size_t       key_cap;
    size_t       table_mask;
    unsigned int hash_shift;
    bool         need_var;
    uint8_t*     io_buf;
    size_t       io_buf_cap;
    size_t       num_distinct;
    atomic_int*  err_code;
} rcre_df__stream_worker;

typedef struct rcre_df__stream_slab {
    size_t start;
    size_t end;
} rcre_df__stream_slab;

typedef struct rcre_df__stream_ctx {
    const uint8_t*              base;
    size_t                      file_size;
    size_t                      data_offset; /* Byte offset after optional header */
    int                         fd;
    uint8_t                     delim;
    uint64_t                    delim_broadcast;
    size_t                      key_col_idx;
    size_t                      val_col_idx;
    bool                        fast_two_col;
    const rcre_df__stream_slab* slabs;
    rcre_df__stream_worker*     workers;
    size_t                      num_workers;
    atomic_int                  err_code;
} rcre_df__stream_ctx;

static inline uint64_t rcre_df__mask_low_bytes(uint64_t w, size_t nbytes) {
    return nbytes ? (w & ((1ULL << (nbytes * 8U)) - 1ULL)) : 0ULL;
}

static inline uint64_t rcre_df__swar_find_byte(uint64_t w, uint64_t broadcast) {
    uint64_t diff = w ^ broadcast;
    return (diff - 0x0101010101010101ULL) & ~diff & 0x8080808080808080ULL;
}

static inline uint64_t rcre_df__hash_w0(uint64_t w0) {
    return w0 * RCRE_DF_FIB_MUL;
}

static inline uint64_t rcre_df__hash_w0_w1(uint64_t w0, uint64_t w1) {
    return (w0 ^ (w1 * RCRE_DF_MIX_MUL)) * RCRE_DF_FIB_MUL;
}

static inline uint64_t rcre_df__hash_canonical(const uint8_t* data, size_t len, uint64_t* out_w0) {
    if (len < 8U) {
        uint64_t w0 = 0;
        if (len > 0) memcpy(&w0, data, len);
        *out_w0 = w0;
        return rcre_df__hash_w0(w0);
    }
    uint64_t w0;
    memcpy(&w0, data, 8);
    *out_w0 = w0;
    if (len < 16U) {
        uint64_t w1 = 0;
        if (len > 8U) memcpy(&w1, data + 8, len - 8U);
        return rcre_df__hash_w0_w1(w0, w1);
    }
    uint64_t w1;
    memcpy(&w1, data + 8, 8);
    uint64_t acc = w0 ^ (w1 * RCRE_DF_MIX_MUL);
    size_t i = 16U;
    while (i + 8U <= len) {
        uint64_t wn;
        memcpy(&wn, data + i, 8);
        acc = (acc * RCRE_DF_FIB_MUL) ^ wn;
        i += 8U;
    }
    if (i < len) {
        uint64_t tail = 0;
        memcpy(&tail, data + i, len - i);
        acc = (acc * RCRE_DF_FIB_MUL) ^ tail;
    }
    return acc * RCRE_DF_FIB_MUL;
}

static inline uint32_t rcre_df__store_key(rcre_df__stream_worker* ws, const uint8_t* name, size_t len) {
    size_t alloc_len = (len + 15U) & ~((size_t)7U);
    if (__builtin_expect(ws->key_used + alloc_len > ws->key_cap, 0)) {
        if (ws->err_code) {
            atomic_store(ws->err_code, RCRE_ERR_OUT_OF_MEMORY);
        }
        return RCRE_DF_INVALID_KEY_OFF;
    }
    uint32_t off = (uint32_t)ws->key_used;
    uint8_t* dst = ws->key_buf + off;
    if (len > 0) memcpy(dst, name, len);
    memset(dst + len, 0, alloc_len - len);
    ws->key_used += alloc_len;
    return off;
}

static inline void rcre_df__slot_accum(rcre_df__stream_slot* s, double val, bool need_var) {
    if (val < s->min) s->min = val;
    if (val > s->max) s->max = val;
    if (__builtin_expect(!need_var, 1)) {
        s->sum += val;
        s->count++;
    } else {
        double old_mean = s->sum / (double)s->count;
        s->sum += val;
        s->count++;
        double new_mean = s->sum / (double)s->count;
        s->m2 += (val - old_mean) * (val - new_mean);
    }
}

static inline void rcre_df__update_short(
    rcre_df__stream_worker* ws,
    const uint8_t* name,
    size_t name_len,
    uint64_t w0,
    double val
) {
    rcre_df__stream_slot* slots = ws->slots;
    size_t mask = ws->table_mask;
    size_t idx = (size_t)(rcre_df__hash_w0(w0) >> ws->hash_shift);
    uint16_t len16 = (uint16_t)name_len;
    size_t probes = 0;

    for (;;) {
        rcre_df__stream_slot* s = &slots[idx];
        if (__builtin_expect(s->w0 == w0 && s->name_len == len16 && s->count != 0, 1)) {
            rcre_df__slot_accum(s, val, ws->need_var);
            return;
        }
        if (__builtin_expect(s->count == 0, 1)) {
            if (__builtin_expect(ws->num_distinct * 4U >= (mask + 1U) * 3U, 0)) {
                if (ws->err_code) atomic_store(ws->err_code, RCRE_ERR_OUT_OF_MEMORY);
                return;
            }
            uint32_t off = rcre_df__store_key(ws, name, name_len);
            if (__builtin_expect(off == RCRE_DF_INVALID_KEY_OFF, 0)) {
                return;
            }
            s->w0       = w0;
            s->sum      = val;
            s->min      = val;
            s->max      = val;
            s->m2       = 0.0;
            s->count    = 1;
            s->name_off = off;
            s->name_len = len16;
            ws->num_distinct++;
            return;
        }
        idx = (idx + 1U) & mask;
        if (__builtin_expect(++probes > mask, 0)) {
            if (ws->err_code) atomic_store(ws->err_code, RCRE_ERR_OUT_OF_MEMORY);
            return;
        }
    }
}

static inline void rcre_df__update_medium(
    rcre_df__stream_worker* ws,
    const uint8_t* name,
    size_t name_len,
    uint64_t w0,
    uint64_t w1,
    double val
) {
    rcre_df__stream_slot* slots = ws->slots;
    size_t mask = ws->table_mask;
    size_t idx = (size_t)(rcre_df__hash_w0_w1(w0, w1) >> ws->hash_shift);
    uint16_t len16 = (uint16_t)name_len;
    size_t probes = 0;

    for (;;) {
        rcre_df__stream_slot* s = &slots[idx];
        if (__builtin_expect(s->w0 == w0 && s->name_len == len16 && s->count != 0, 1)) {
            uint64_t sw1;
            memcpy(&sw1, ws->key_buf + s->name_off + 8U, 8);
            if (__builtin_expect(sw1 == w1, 1)) {
                rcre_df__slot_accum(s, val, ws->need_var);
                return;
            }
        } else if (__builtin_expect(s->count == 0, 1)) {
            if (__builtin_expect(ws->num_distinct * 4U >= (mask + 1U) * 3U, 0)) {
                if (ws->err_code) atomic_store(ws->err_code, RCRE_ERR_OUT_OF_MEMORY);
                return;
            }
            uint32_t off = rcre_df__store_key(ws, name, name_len);
            if (__builtin_expect(off == RCRE_DF_INVALID_KEY_OFF, 0)) {
                return;
            }
            s->w0       = w0;
            s->sum      = val;
            s->min      = val;
            s->max      = val;
            s->m2       = 0.0;
            s->count    = 1;
            s->name_off = off;
            s->name_len = len16;
            ws->num_distinct++;
            return;
        }
        idx = (idx + 1U) & mask;
        if (__builtin_expect(++probes > mask, 0)) {
            if (ws->err_code) atomic_store(ws->err_code, RCRE_ERR_OUT_OF_MEMORY);
            return;
        }
    }
}

static inline void rcre_df__update_general(
    rcre_df__stream_worker* ws,
    const uint8_t* name,
    size_t name_len,
    uint64_t w0,
    uint64_t h,
    double val
) {
    rcre_df__stream_slot* slots = ws->slots;
    size_t mask = ws->table_mask;
    size_t idx = (size_t)(h >> ws->hash_shift);
    uint16_t len16 = (uint16_t)name_len;
    size_t probes = 0;

    for (;;) {
        rcre_df__stream_slot* s = &slots[idx];
        if (__builtin_expect(s->w0 == w0 && s->name_len == len16 && s->count != 0, 1)) {
            if (name_len <= 8U ||
                memcmp(ws->key_buf + s->name_off + 8U, name + 8U, name_len - 8U) == 0) {
                rcre_df__slot_accum(s, val, ws->need_var);
                return;
            }
        } else if (s->count == 0) {
            if (__builtin_expect(ws->num_distinct * 4U >= (mask + 1U) * 3U, 0)) {
                if (ws->err_code) atomic_store(ws->err_code, RCRE_ERR_OUT_OF_MEMORY);
                return;
            }
            uint32_t off = rcre_df__store_key(ws, name, name_len);
            if (__builtin_expect(off == RCRE_DF_INVALID_KEY_OFF, 0)) {
                return;
            }
            s->w0       = w0;
            s->sum      = val;
            s->min      = val;
            s->max      = val;
            s->m2       = 0.0;
            s->count    = 1;
            s->name_off = off;
            s->name_len = len16;
            ws->num_distinct++;
            return;
        }
        idx = (idx + 1U) & mask;
        if (__builtin_expect(++probes > mask, 0)) {
            if (ws->err_code) atomic_store(ws->err_code, RCRE_ERR_OUT_OF_MEMORY);
            return;
        }
    }
}

/* General fast ASCII decimal parser -> double (with rcre_slice_parse_f64 fallback) */
static inline bool rcre_df__parse_double_token(const uint8_t* start, size_t len, double* out_val) {
    static const double kInvPow10[10] = {
        1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9
    };
    const uint8_t* p = start;
    const uint8_t* end = start + len;
    while (p < end && (*p == ' ' || *p == '\t')) p++;
    while (end > p && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r')) end--;
    if (p >= end) return false;

    bool neg = false;
    if (*p == '-') { neg = true; p++; }
    else if (*p == '+') { p++; }
    if (p >= end) return false;

    /* Ultra-fast branchless path for standard 1-decimal measurements: D.D or DD.D */
    size_t rem = (size_t)(end - p);
    if (__builtin_expect((rem == 3U && p[1] == '.') || (rem == 4U && p[2] == '.'), 1)) {
        if (rem == 3U && (uint8_t)(p[0] - '0') <= 9U && (uint8_t)(p[2] - '0') <= 9U) {
            int32_t tenths = (p[0] - '0') * 10 + (p[2] - '0');
            *out_val = (double)(neg ? -tenths : tenths) * 0.1;
            return true;
        }
        if (rem == 4U && (uint8_t)(p[0] - '0') <= 9U &&
            (uint8_t)(p[1] - '0') <= 9U && (uint8_t)(p[3] - '0') <= 9U) {
            int32_t tenths = (p[0] - '0') * 100 + (p[1] - '0') * 10 + (p[3] - '0');
            *out_val = (double)(neg ? -tenths : tenths) * 0.1;
            return true;
        }
    }

    uint64_t mant = 0;
    size_t digits = 0;
    size_t frac = 0;
    bool seen_dot = false;
    const uint8_t* q = p;
    while (q < end) {
        uint8_t c = *q;
        if ((uint8_t)(c - '0') <= 9U) {
            if (digits >= 15U) {
                return rcre_slice_parse_f64(rcre_slice_from_parts(start, len), out_val);
            }
            mant = mant * 10ULL + (uint64_t)(c - '0');
            digits++;
            if (seen_dot) frac++;
            q++;
        } else if (c == '.' && !seen_dot) {
            seen_dot = true;
            q++;
        } else {
            return rcre_slice_parse_f64(rcre_slice_from_parts(start, len), out_val);
        }
    }
    if (digits == 0 || frac >= 10U) {
        return rcre_slice_parse_f64(rcre_slice_from_parts(start, len), out_val);
    }
    double v = (double)mant * kInvPow10[frac];
    *out_val = neg ? -v : v;
    return true;
}

/* Fast-path inline number parse ending at \n / \r\n (for 2-column key<delim>val rows) */
static inline bool rcre_df__parse_eol_num_fast(const uint8_t** pp, const uint8_t* end, double* out_val) {
    const uint8_t* p = *pp;
    /* Branchless fast path for [-]D.D\n or [-]DD.D\n (no extra columns on the row) */
    int32_t neg = (*p == '-');
    const uint8_t* q = p + neg;
    if (__builtin_expect(q + 5 <= end && (uint8_t)(q[0] - '0') <= 9U, 1)) {
        if (q[1] == '.' && (uint8_t)(q[2] - '0') <= 9U && (q[3] == '\n' || q[3] == '\r')) {
            int32_t tenths = (q[0] - '0') * 10 + (q[2] - '0');
            *out_val = (double)(neg ? -tenths : tenths) * 0.1;
            q += 3;
            if (*q == '\r') {
                q++;
                if (q < end && *q == '\n') q++;
            } else {
                q++;
            }
            *pp = q;
            return true;
        }
        if ((uint8_t)(q[1] - '0') <= 9U && q[2] == '.' && (uint8_t)(q[3] - '0') <= 9U &&
            (q[4] == '\n' || q[4] == '\r')) {
            int32_t tenths = (q[0] - '0') * 100 + (q[1] - '0') * 10 + (q[3] - '0');
            *out_val = (double)(neg ? -tenths : tenths) * 0.1;
            q += 4;
            if (*q == '\r') {
                q++;
                if (q < end && *q == '\n') q++;
            } else {
                q++;
            }
            *pp = q;
            return true;
        }
    }

    /* General fallback: find end of line (or next delimiter) */
    const uint8_t* line_end = p;
    while (line_end < end && *line_end != '\n') line_end++;
    const uint8_t* tok_end = line_end;
    if (tok_end > p && tok_end[-1] == '\r') tok_end--;
    bool ok = rcre_df__parse_double_token(p, (size_t)(tok_end - p), out_val);
    *pp = (line_end < end) ? (line_end + 1U) : end;
    return ok;
}

static inline bool rcre_df__step_two_col(
    rcre_df__stream_worker* ws,
    const uint8_t** pp,
    const uint8_t* end,
    uint8_t delim,
    uint64_t delim_bcast
) {
    const uint8_t* p = *pp;
    const uint8_t* name_start = p;

    uint64_t raw_w0;
    memcpy(&raw_w0, p, 8);
    uint64_t m0 = rcre_df__swar_find_byte(raw_w0, delim_bcast);
    if (m0 != 0) {
        size_t name_len = (size_t)(__builtin_ctzll(m0) >> 3);
        uint64_t w0 = rcre_df__mask_low_bytes(raw_w0, name_len);
        p += name_len + 1U;
        double val = 0.0;
        if (__builtin_expect(rcre_df__parse_eol_num_fast(&p, end, &val), 1)) {
            rcre_df__update_short(ws, name_start, name_len, w0, val);
        }
        *pp = p;
        return true;
    }

    uint64_t raw_w1;
    memcpy(&raw_w1, p + 8, 8);
    uint64_t m1 = rcre_df__swar_find_byte(raw_w1, delim_bcast);
    if (__builtin_expect(m1 != 0, 1)) {
        size_t l1 = (size_t)(__builtin_ctzll(m1) >> 3);
        size_t name_len = 8U + l1;
        uint64_t w1 = rcre_df__mask_low_bytes(raw_w1, l1);
        p += name_len + 1U;
        double val = 0.0;
        if (__builtin_expect(rcre_df__parse_eol_num_fast(&p, end, &val), 1)) {
            rcre_df__update_medium(ws, name_start, name_len, raw_w0, w1, val);
        }
        *pp = p;
        return true;
    }

    const uint8_t* q = p + 16;
    while (q + 8 <= end) {
        uint64_t wn;
        memcpy(&wn, q, 8);
        uint64_t mn = rcre_df__swar_find_byte(wn, delim_bcast);
        if (mn != 0) {
            q += (size_t)(__builtin_ctzll(mn) >> 3);
            break;
        }
        q += 8;
    }
    while (q < end && *q != delim && *q != '\n') q++;
    if (__builtin_expect(q >= end || *q != delim, 0)) {
        while (q < end && *q != '\n') q++;
        *pp = (q < end) ? (q + 1U) : end;
        return false;
    }
    size_t name_len = (size_t)(q - name_start);
    uint64_t w0_out = 0;
    uint64_t h = rcre_df__hash_canonical(name_start, name_len, &w0_out);
    p = q + 1U;
    double val = 0.0;
    if (rcre_df__parse_eol_num_fast(&p, end, &val)) {
        rcre_df__update_general(ws, name_start, name_len, w0_out, h, val);
    }
    *pp = p;
    return true;
}

static void rcre_df__drain_two_col(
    rcre_df__stream_worker* ws,
    const uint8_t* p,
    const uint8_t* end,
    uint8_t delim,
    uint64_t delim_bcast
) {
    const uint8_t* safe_end = (end - p >= 24) ? (end - 24) : p;
    while (p < safe_end) {
        (void)rcre_df__step_two_col(ws, &p, end, delim, delim_bcast);
    }
    while (p < end) {
        const uint8_t* line_start = p;
        const uint8_t* line_end = p;
        while (line_end < end && *line_end != '\n') line_end++;
        const uint8_t* sep = line_start;
        while (sep < line_end && *sep != delim) sep++;
        if (sep < line_end) {
            size_t klen = (size_t)(sep - line_start);
            const uint8_t* vstart = sep + 1U;
            const uint8_t* vend = line_end;
            if (vend > vstart && vend[-1] == '\r') vend--;
            double val = 0.0;
            if (klen > 0 && rcre_df__parse_double_token(vstart, (size_t)(vend - vstart), &val)) {
                uint64_t w0 = 0;
                uint64_t h = rcre_df__hash_canonical(line_start, klen, &w0);
                rcre_df__update_general(ws, line_start, klen, w0, h, val);
            }
        }
        p = (line_end < end) ? (line_end + 1U) : end;
    }
}

static void rcre_df__process_slab_general_cols(
    rcre_df__stream_worker* ws,
    const uint8_t* p,
    const uint8_t* end,
    uint8_t delim,
    size_t key_col_idx,
    size_t val_col_idx
) {
    size_t max_col = (key_col_idx > val_col_idx) ? key_col_idx : val_col_idx;
    while (p < end) {
        if (ws->err_code && atomic_load(ws->err_code) != RCRE_OK) return;
        const uint8_t* line_start = p;
        size_t nl = rcre_hal.find_byte(p, (size_t)(end - p), '\n');
        const uint8_t* line_end = (nl == RCRE_NPOS) ? end : (p + nl);
        p = (line_end < end) ? (line_end + 1U) : end;

        const uint8_t* content_end = line_end;
        if (content_end > line_start && content_end[-1] == '\r') content_end--;
        if (content_end <= line_start) continue;

        const uint8_t* key_ptr = nullptr;
        size_t key_len = 0;
        const uint8_t* val_ptr = nullptr;
        size_t val_len = 0;
        bool found_key = false;
        bool found_val = false;

        size_t col = 0;
        const uint8_t* field = line_start;
        const uint8_t* cur = line_start;
        while (cur <= content_end && col <= max_col) {
            if (cur == content_end || *cur == delim) {
                if (col == key_col_idx) {
                    key_ptr = field;
                    key_len = (size_t)(cur - field);
                    found_key = true;
                }
                if (col == val_col_idx) {
                    val_ptr = field;
                    val_len = (size_t)(cur - field);
                    found_val = true;
                }
                col++;
                field = cur + 1U;
            }
            cur++;
        }

        if (found_key && found_val && key_len > 0) {
            double val = 0.0;
            if (rcre_df__parse_double_token(val_ptr, val_len, &val)) {
                uint64_t w0 = 0;
                uint64_t h = rcre_df__hash_canonical(key_ptr, key_len, &w0);
                rcre_df__update_general(ws, key_ptr, key_len, w0, h, val);
            }
        }
    }
}

static void rcre_df__process_slab_bytes(
    rcre_df__stream_worker* ws,
    const uint8_t* start,
    const uint8_t* end,
    const rcre_df__stream_ctx* ctx
) {
    if (start >= end) return;
    if (ws->err_code && atomic_load(ws->err_code) != RCRE_OK) return;
    if (!ctx->fast_two_col) {
        rcre_df__process_slab_general_cols(ws, start, end, ctx->delim, ctx->key_col_idx, ctx->val_col_idx);
        return;
    }

    size_t len = (size_t)(end - start);
    if (len < 4096U) {
        rcre_df__drain_two_col(ws, start, end, ctx->delim, ctx->delim_broadcast);
        return;
    }

    /* Dual-cursor OoO ILP split */
    size_t half = len >> 1;
    size_t rel = rcre_hal.find_byte(start + half, len - half, '\n');
    if (rel == RCRE_NPOS) {
        rcre_df__drain_two_col(ws, start, end, ctx->delim, ctx->delim_broadcast);
        return;
    }

    const uint8_t* mid   = start + half + rel + 1U;
    const uint8_t* p0    = start;
    const uint8_t* p1    = mid;
    const uint8_t* safe0 = (mid - p0 >= 24) ? (mid - 24) : p0;
    const uint8_t* safe1 = (end - p1 >= 24) ? (end - 24) : p1;

    while (p0 < safe0 && p1 < safe1) {
        (void)rcre_df__step_two_col(ws, &p0, mid, ctx->delim, ctx->delim_broadcast);
        (void)rcre_df__step_two_col(ws, &p1, end, ctx->delim, ctx->delim_broadcast);
    }

    rcre_df__drain_two_col(ws, p0, mid, ctx->delim, ctx->delim_broadcast);
    rcre_df__drain_two_col(ws, p1, end, ctx->delim, ctx->delim_broadcast);
}

static void rcre_df__mmap_slab_task(void* arg, size_t task_id, size_t worker_id) {
    rcre_df__stream_ctx* ctx = (rcre_df__stream_ctx*)arg;
    rcre_df__stream_worker* ws = &ctx->workers[worker_id];
    const rcre_df__stream_slab slab = ctx->slabs[task_id];
    rcre_df__process_slab_bytes(ws, ctx->base + slab.start, ctx->base + slab.end, ctx);
}

#if defined(RCRE_DF_HAVE_POSIX)
static bool rcre_df__pread_exact(int fd, uint8_t* dst, size_t len, off_t offset) {
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(fd, dst + done, len - done, offset + (off_t)done);
        if (n <= 0) return false;
        done += (size_t)n;
    }
    return true;
}

static void rcre_df__pread_slab_task(void* arg, size_t task_id, size_t worker_id) {
    rcre_df__stream_ctx* ctx = (rcre_df__stream_ctx*)arg;
    rcre_df__stream_worker* ws = &ctx->workers[worker_id];
    const rcre_df__stream_slab slab = ctx->slabs[task_id];
    if (slab.start >= slab.end) return;

    constexpr size_t kLineMargin = 512U;
    size_t raw_len  = slab.end - slab.start;
    size_t read_end = (slab.end + kLineMargin < ctx->file_size) ? (slab.end + kLineMargin) : ctx->file_size;
    size_t read_len = read_end - slab.start;

    if (read_len > ws->io_buf_cap) {
        atomic_store(&ctx->err_code, RCRE_ERR_OUT_OF_MEMORY);
        return;
    }
    if (!rcre_df__pread_exact(ctx->fd, ws->io_buf, read_len, (off_t)slab.start)) {
        atomic_store(&ctx->err_code, RCRE_ERR_IO);
        return;
    }

    size_t local_start = 0;
    if (slab.start > ctx->data_offset) {
        size_t nl = rcre_hal.find_byte(ws->io_buf, read_len, '\n');
        if (nl == RCRE_NPOS) return;
        local_start = nl + 1U;
        if (local_start >= raw_len) return;
    }

    size_t local_end = raw_len;
    if (slab.end < ctx->file_size) {
        size_t tail_search = read_len - raw_len;
        size_t nl = rcre_hal.find_byte(ws->io_buf + raw_len, tail_search, '\n');
        local_end = (nl != RCRE_NPOS) ? (raw_len + nl + 1U) : read_len;
    } else {
        local_end = read_len;
    }

    if (local_start < local_end) {
        rcre_df__process_slab_bytes(ws, ws->io_buf + local_start, ws->io_buf + local_end, ctx);
    }
}
#endif

typedef struct rcre_df__master_group {
    const uint8_t* name;
    size_t         name_len;
    uint64_t       w0;
    double         sum;
    double         min;
    double         max;
    double         m2;
    uint64_t       count;
} rcre_df__master_group;

static int rcre_df__cmp_master_group(const void* a, const void* b) {
    const rcre_df__master_group* ga = *(const rcre_df__master_group* const*)a;
    const rcre_df__master_group* gb = *(const rcre_df__master_group* const*)b;
    return rcre_slice_cmp(
        rcre_slice_from_parts(ga->name, ga->name_len),
        rcre_slice_from_parts(gb->name, gb->name_len)
    );
}

static size_t rcre_df__resolve_pow2(size_t requested, unsigned int* out_shift) {
    size_t target = (requested == 0) ? 16384U : requested;
    size_t cap = 64U;
    unsigned int bits = 6U;
    while (cap < target && bits < 22U) {
        cap <<= 1;
        bits++;
    }
    *out_shift = 64U - bits;
    return cap;
}

RCRE_NODISCARD int rcre_df_stream_csv_groupby(
    const char* filepath,
    const RCRE_StreamGroupByOpts* opts_in,
    RCRE_Arena* arena,
    RCRE_DataFrame* out_df
) {
    if (!filepath || !arena || !out_df) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_df, 0, sizeof(*out_df));

    static const RCRE_AggSpec kDefaultSpecs[4] = {
        { .target_col_idx = 1, .op = RCRE_AGG_MIN,   .out_name = "min"   },
        { .target_col_idx = 1, .op = RCRE_AGG_MEAN,  .out_name = "mean"  },
        { .target_col_idx = 1, .op = RCRE_AGG_MAX,   .out_name = "max"   },
        { .target_col_idx = 1, .op = RCRE_AGG_COUNT, .out_name = "count" }
    };

    RCRE_StreamGroupByOpts opts = {
        .delimiter      = ',',
        .has_header     = false,
        .key_col_idx    = 0,
        .val_col_idx    = 1,
        .specs          = kDefaultSpecs,
        .num_specs      = 4,
        .pool           = nullptr,
        .slab_bytes     = 0,
        .table_cap      = 16384U,
        .bounded_ram    = false,
        .sort_by_key    = true,
        .use_f64_output = true,
        .key_col_name   = "key"
    };
    if (opts_in) {
        opts = *opts_in;
        if (opts.delimiter == 0) opts.delimiter = ',';
        if (!opts.specs || opts.num_specs == 0) {
            opts.specs = kDefaultSpecs;
            opts.num_specs = 4;
        }
        if (!opts.key_col_name) opts.key_col_name = "key";
    }
    if (opts.num_specs + 1U > RCRE_DF_MAX_COLS) {
        return RCRE_ERR_INVALID_ARG;
    }

    bool need_var = false;
    for (size_t s = 0; s < opts.num_specs; ++s) {
        if (opts.specs[s].op == RCRE_AGG_VAR || opts.specs[s].op == RCRE_AGG_STD) {
            need_var = true;
        }
    }

    size_t num_workers = rcre_threadpool_worker_count(opts.pool);
    unsigned int hash_shift = 50U;
    size_t table_cap = rcre_df__resolve_pow2(opts.table_cap, &hash_shift);
    uint64_t delim_bcast = (uint64_t)opts.delimiter * 0x0101010101010101ULL;

    /* Open input via either mmap or bounded-RAM POSIX fd */
    RCRE_MappedFile map = {0};
    int fd = -1;
    size_t file_size = 0;
    const uint8_t* mapped_base = nullptr;
    size_t data_offset = 0;
    bool use_bounded_pread = false;
    bool fast_two_col = (opts.key_col_idx == 0 && opts.val_col_idx == 1);

#if defined(RCRE_DF_HAVE_POSIX)
    use_bounded_pread = opts.bounded_ram;
#endif

    char header_key_name[RCRE_DF_MAX_NAME];
    header_key_name[0] = '\0';

    if (use_bounded_pread) {
#if defined(RCRE_DF_HAVE_POSIX)
        fd = open(filepath, O_RDONLY);
        if (fd < 0) return RCRE_ERR_IO;
        struct stat st;
        if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
            close(fd);
            return RCRE_ERR_IO;
        }
        file_size = (size_t)st.st_size;
        /* Inspect first line (up to 4KB) for header / column count check */
        if (file_size > 0) {
            uint8_t probe[4096];
            size_t plen = (file_size < sizeof(probe)) ? file_size : sizeof(probe);
            if (!rcre_df__pread_exact(fd, probe, plen, 0)) {
                close(fd);
                return RCRE_ERR_IO;
            }
            size_t nl = rcre_hal.find_byte(probe, plen, '\n');
            size_t line_len = (nl == RCRE_NPOS) ? plen : nl;
            size_t delim_count = 0;
            for (size_t i = 0; i < line_len; ++i) {
                if (probe[i] == opts.delimiter) delim_count++;
            }
            if (delim_count != 1U) fast_two_col = false;
            if (opts.has_header) {
                /* Extract header key column name if present */
                size_t c = 0, tok_s = 0;
                for (size_t i = 0; i <= line_len; ++i) {
                    if (i == line_len || probe[i] == opts.delimiter) {
                        if (c == opts.key_col_idx) {
                            size_t tlen = i - tok_s;
                            if (tlen > 0 && probe[tok_s + tlen - 1U] == '\r') tlen--;
                            (void)rcre_cstr_copy_bounded(header_key_name, sizeof(header_key_name),
                                                         rcre_slice_from_parts(probe + tok_s, tlen));
                            break;
                        }
                        c++;
                        tok_s = i + 1U;
                    }
                }
                data_offset = (nl == RCRE_NPOS) ? file_size : (nl + 1U);
            }
        }
#endif
    } else {
        int mrc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_SEQUENTIAL, &map);
        if (mrc != RCRE_OK) return mrc;
        mapped_base = map.data;
        file_size = map.size;
        if (file_size > 0) {
            size_t nl = rcre_hal.find_byte(mapped_base, file_size, '\n');
            size_t line_len = (nl == RCRE_NPOS) ? file_size : nl;
            size_t delim_count = 0;
            for (size_t i = 0; i < line_len; ++i) {
                if (mapped_base[i] == opts.delimiter) delim_count++;
            }
            if (delim_count != 1U) fast_two_col = false;
            if (opts.has_header) {
                size_t c = 0, tok_s = 0;
                for (size_t i = 0; i <= line_len; ++i) {
                    if (i == line_len || mapped_base[i] == opts.delimiter) {
                        if (c == opts.key_col_idx) {
                            size_t tlen = i - tok_s;
                            if (tlen > 0 && mapped_base[tok_s + tlen - 1U] == '\r') tlen--;
                            (void)rcre_cstr_copy_bounded(header_key_name, sizeof(header_key_name),
                                                         rcre_slice_from_parts(mapped_base + tok_s, tlen));
                            break;
                        }
                        c++;
                        tok_s = i + 1U;
                    }
                }
                data_offset = (nl == RCRE_NPOS) ? file_size : (nl + 1U);
            }
        }
    }

    const char* out_col_names[RCRE_DF_MAX_COLS];
    RCRE_DType  out_col_dtypes[RCRE_DF_MAX_COLS];
    out_col_names[0] = (header_key_name[0] != '\0') ? header_key_name : opts.key_col_name;
    out_col_dtypes[0] = RCRE_DTYPE_STR_ID;
    for (size_t s = 0; s < opts.num_specs; ++s) {
        out_col_names[1 + s] = opts.specs[s].out_name ? opts.specs[s].out_name : "agg";
        if (opts.specs[s].op == RCRE_AGG_COUNT) {
            out_col_dtypes[1 + s] = opts.use_f64_output ? RCRE_DTYPE_I64 : RCRE_DTYPE_I32;
        } else {
            out_col_dtypes[1 + s] = opts.use_f64_output ? RCRE_DTYPE_F64 : RCRE_DTYPE_F32;
        }
    }

    /* Empty or header-only file */
    if (data_offset >= file_size) {
        if (map.is_mapped || map.is_heap_buf) rcre_io_mmap_close(&map);
#if defined(RCRE_DF_HAVE_POSIX)
        if (fd >= 0) close(fd);
#endif
        return rcre_df_create(out_col_names, out_col_dtypes, 1U + opts.num_specs, 0, arena, out_df);
    }

    size_t data_bytes = file_size - data_offset;
    size_t slab_bytes = opts.slab_bytes;
    if (slab_bytes == 0) {
        slab_bytes = use_bounded_pread ? (4U * 1024U * 1024U) : (8U * 1024U * 1024U);
    }
    size_t max_slabs = (data_bytes + slab_bytes - 1U) / slab_bytes;
    if (max_slabs == 0) max_slabs = 1;

    constexpr size_t kKeyBufPerWorker = 1536U * 1024U;
    constexpr size_t kLineMargin = 512U;
    size_t per_worker_io = use_bounded_pread ? (slab_bytes + kLineMargin) : 0U;

    size_t work_arena_cap =
        num_workers * (table_cap * sizeof(rcre_df__stream_slot) + kKeyBufPerWorker + per_worker_io + 256U) +
        table_cap * (sizeof(rcre_df__master_group) + sizeof(rcre_df__master_group*)) +
        max_slabs * sizeof(rcre_df__stream_slab) +
        num_workers * sizeof(rcre_df__stream_worker) +
        (4U * 1024U * 1024U);

    RCRE_Arena* work_arena = rcre_arena_create(work_arena_cap);
    if (!work_arena) {
        if (map.is_mapped || map.is_heap_buf) rcre_io_mmap_close(&map);
#if defined(RCRE_DF_HAVE_POSIX)
        if (fd >= 0) close(fd);
#endif
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    rcre_df__stream_slab* slabs = (rcre_df__stream_slab*)rcre_arena_alloc(
        work_arena, max_slabs * sizeof(rcre_df__stream_slab), 64);
    rcre_df__stream_worker* workers = (rcre_df__stream_worker*)rcre_arena_alloc(
        work_arena, num_workers * sizeof(rcre_df__stream_worker), 64);
    if (!slabs || !workers) {
        rcre_arena_destroy(work_arena);
        if (map.is_mapped || map.is_heap_buf) rcre_io_mmap_close(&map);
#if defined(RCRE_DF_HAVE_POSIX)
        if (fd >= 0) close(fd);
#endif
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(workers, 0, num_workers * sizeof(rcre_df__stream_worker));

    size_t num_slabs = 0;
    if (use_bounded_pread) {
        for (size_t i = 0; i < max_slabs; ++i) {
            size_t s = data_offset + i * slab_bytes;
            size_t e = (s + slab_bytes < file_size) ? (s + slab_bytes) : file_size;
            slabs[i].start = s;
            slabs[i].end   = e;
        }
        num_slabs = max_slabs;
    } else {
        size_t cursor = data_offset;
        while (cursor < file_size && num_slabs < max_slabs) {
            size_t start = cursor;
            size_t target = start + slab_bytes;
            size_t end;
            if (target >= file_size || num_slabs + 1U == max_slabs) {
                end = file_size;
            } else {
                size_t rel = rcre_hal.find_byte(mapped_base + target, file_size - target, '\n');
                end = (rel == RCRE_NPOS) ? file_size : (target + rel + 1U);
            }
            slabs[num_slabs].start = start;
            slabs[num_slabs].end   = end;
            num_slabs++;
            cursor = end;
        }
    }

    rcre_df__stream_ctx ctx = {
        .base            = mapped_base,
        .file_size       = file_size,
        .data_offset     = data_offset,
        .fd              = fd,
        .delim           = opts.delimiter,
        .delim_broadcast = delim_bcast,
        .key_col_idx     = opts.key_col_idx,
        .val_col_idx     = opts.val_col_idx,
        .fast_two_col    = fast_two_col,
        .slabs           = slabs,
        .workers         = workers,
        .num_workers     = num_workers
    };
    atomic_init(&ctx.err_code, RCRE_OK);

    for (size_t w = 0; w < num_workers; ++w) {
        workers[w].slots = (rcre_df__stream_slot*)rcre_arena_alloc(
            work_arena, table_cap * sizeof(rcre_df__stream_slot), 64);
        workers[w].key_buf = (uint8_t*)rcre_arena_alloc(work_arena, kKeyBufPerWorker, 64);
        if (per_worker_io > 0) {
            workers[w].io_buf = (uint8_t*)rcre_arena_alloc(work_arena, per_worker_io, 64);
            workers[w].io_buf_cap = per_worker_io;
        }
        if (!workers[w].slots || !workers[w].key_buf || (per_worker_io > 0 && !workers[w].io_buf)) {
            rcre_arena_destroy(work_arena);
            if (map.is_mapped || map.is_heap_buf) rcre_io_mmap_close(&map);
#if defined(RCRE_DF_HAVE_POSIX)
            if (fd >= 0) close(fd);
#endif
            return RCRE_ERR_OUT_OF_MEMORY;
        }
        memset(workers[w].slots, 0, table_cap * sizeof(rcre_df__stream_slot));
        workers[w].key_used     = 0;
        workers[w].key_cap      = kKeyBufPerWorker;
        workers[w].table_mask   = table_cap - 1U;
        workers[w].hash_shift   = hash_shift;
        workers[w].need_var     = need_var;
        workers[w].num_distinct = 0;
        workers[w].err_code     = &ctx.err_code;
    }

    if (use_bounded_pread) {
#if defined(RCRE_DF_HAVE_POSIX)
        rcre_threadpool_parallel_for_indexed(opts.pool, num_slabs, rcre_df__pread_slab_task, &ctx);
#endif
    } else {
        rcre_threadpool_parallel_for_indexed(opts.pool, num_slabs, rcre_df__mmap_slab_task, &ctx);
    }

    int err = atomic_load(&ctx.err_code);
    if (err != RCRE_OK) {
        rcre_arena_destroy(work_arena);
        if (map.is_mapped || map.is_heap_buf) rcre_io_mmap_close(&map);
#if defined(RCRE_DF_HAVE_POSIX)
        if (fd >= 0) close(fd);
#endif
        return err;
    }

    /* Merge worker hash tables into master table (with Chan parallel Welford merge) */
    size_t mask = table_cap - 1U;
    rcre_df__master_group* master = (rcre_df__master_group*)rcre_arena_alloc(
        work_arena, table_cap * sizeof(rcre_df__master_group), 64);
    rcre_df__master_group** groups = (rcre_df__master_group**)rcre_arena_alloc(
        work_arena, table_cap * sizeof(rcre_df__master_group*), 64);
    if (!master || !groups) {
        rcre_arena_destroy(work_arena);
        if (map.is_mapped || map.is_heap_buf) rcre_io_mmap_close(&map);
#if defined(RCRE_DF_HAVE_POSIX)
        if (fd >= 0) close(fd);
#endif
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(master, 0, table_cap * sizeof(rcre_df__master_group));

    size_t master_distinct = 0;
    for (size_t w = 0; w < num_workers; ++w) {
        const rcre_df__stream_slot* wt = workers[w].slots;
        const uint8_t* kbuf = workers[w].key_buf;
        for (size_t j = 0; j < table_cap; ++j) {
            const rcre_df__stream_slot* we = &wt[j];
            if (we->count == 0) continue;

            const uint8_t* name = kbuf + we->name_off;
            size_t len = (size_t)we->name_len;
            uint64_t w0_chk = 0;
            uint64_t h = rcre_df__hash_canonical(name, len, &w0_chk);
            size_t idx = (size_t)(h >> hash_shift);
            size_t probes = 0;

            for (;;) {
                rcre_df__master_group* me = &master[idx];
                if (me->count == 0) {
                    if (master_distinct * 4U >= table_cap * 3U) {
                        rcre_arena_destroy(work_arena);
                        if (map.is_mapped || map.is_heap_buf) rcre_io_mmap_close(&map);
#if defined(RCRE_DF_HAVE_POSIX)
                        if (fd >= 0) close(fd);
#endif
                        return RCRE_ERR_OUT_OF_MEMORY;
                    }
                    me->name     = name;
                    me->name_len = len;
                    me->w0       = we->w0;
                    me->sum      = we->sum;
                    me->min      = we->min;
                    me->max      = we->max;
                    me->m2       = we->m2;
                    me->count    = we->count;
                    master_distinct++;
                    break;
                }
                if (me->w0 == we->w0 && me->name_len == len &&
                    (len <= 8U || memcmp(me->name + 8U, name + 8U, len - 8U) == 0)) {
                    if (need_var) {
                        double mean_a = me->sum / (double)me->count;
                        double mean_b = we->sum / (double)we->count;
                        double delta  = mean_b - mean_a;
                        double n_a    = (double)me->count;
                        double n_b    = (double)we->count;
                        me->m2 += we->m2 + delta * delta * (n_a * n_b / (n_a + n_b));
                    }
                    if (we->min < me->min) me->min = we->min;
                    if (we->max > me->max) me->max = we->max;
                    me->sum   += we->sum;
                    me->count += we->count;
                    break;
                }
                idx = (idx + 1U) & mask;
                if (++probes > mask) {
                    rcre_arena_destroy(work_arena);
                    if (map.is_mapped || map.is_heap_buf) rcre_io_mmap_close(&map);
#if defined(RCRE_DF_HAVE_POSIX)
                    if (fd >= 0) close(fd);
#endif
                    return RCRE_ERR_OUT_OF_MEMORY;
                }
            }
        }
    }

    size_t num_groups = 0;
    for (size_t j = 0; j < table_cap; ++j) {
        if (master[j].count > 0) {
            groups[num_groups++] = &master[j];
        }
    }
    if (opts.sort_by_key && num_groups > 1U) {
        qsort(groups, num_groups, sizeof(rcre_df__master_group*), rcre_df__cmp_master_group);
    }

    /* Emit standard RCRE_DataFrame into caller's arena */
    int rc = rcre_df_create(out_col_names, out_col_dtypes, 1U + opts.num_specs, num_groups, arena, out_df);
    if (rc == RCRE_OK) {
        RCRE_Column* key_col = &out_df->columns[0];
        for (size_t g = 0; g < num_groups; ++g) {
            const rcre_df__master_group* mg = groups[g];
            rc = rcre_col_set_str(key_col, g, rcre_slice_from_parts(mg->name, mg->name_len));
            if (rc != RCRE_OK) break;

            for (size_t s = 0; s < opts.num_specs; ++s) {
                RCRE_Column* dst = &out_df->columns[1 + s];
                if (opts.specs[s].op == RCRE_AGG_COUNT) {
                    if (opts.use_f64_output) {
                        ((int64_t*)dst->data)[g] = (int64_t)mg->count;
                    } else {
                        ((int32_t*)dst->data)[g] = (int32_t)mg->count;
                    }
                } else {
                    double v = 0.0;
                    switch (opts.specs[s].op) {
                        case RCRE_AGG_MIN:  v = mg->min; break;
                        case RCRE_AGG_MAX:  v = mg->max; break;
                        case RCRE_AGG_SUM:  v = mg->sum; break;
                        case RCRE_AGG_MEAN: v = mg->sum / (double)mg->count; break;
                        case RCRE_AGG_VAR:  v = (mg->count > 1U) ? (mg->m2 / (double)(mg->count - 1U)) : 0.0; break;
                        case RCRE_AGG_STD:  v = (mg->count > 1U) ? sqrt(mg->m2 / (double)(mg->count - 1U)) : 0.0; break;
                        default: break;
                    }
                    if (opts.use_f64_output) {
                        ((double*)dst->data)[g] = v;
                    } else {
                        ((float*)dst->data)[g] = (float)v;
                    }
                }
            }
        }
    }

    rcre_arena_destroy(work_arena);
    if (map.is_mapped || map.is_heap_buf) rcre_io_mmap_close(&map);
#if defined(RCRE_DF_HAVE_POSIX)
    if (fd >= 0) close(fd);
#endif
    return rc;
}
