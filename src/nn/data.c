/**
 * @file data.c
 * @brief High-throughput zero-copy data preprocessing, memory-mapped CSV ingestion,
 *        feature scaling, and mini-batch iterators.
 * 
 * Part of RCRE Unified Engine.
 */

#include "rcre_autograd.h"
#include "rcre_rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================= */
/* Dynamic line reader (lines may exceed any fixed stack buffer)             */
/* ========================================================================= */

typedef struct {
    char*  buf;
    size_t cap;
    int    done; /* 1 when a line was fully read */
} csv_line_t;

static void csv_line_free(csv_line_t* ln) {
    if (ln->buf) { free(ln->buf); ln->buf = nullptr; ln->cap = 0; }
}

/* Reads one full line (without the trailing newline). Returns 1 on success,
 * 0 at EOF. Grows the buffer as needed so arbitrarily long rows parse intact. */
static int csv_line_read(FILE* f, csv_line_t* ln) {
    size_t len = 0;
    int c;
    ln->done = 1;

    if (ln->cap == 0) {
        ln->cap = 4096;
        ln->buf = (char*)malloc(ln->cap);
        if (!ln->buf) return 0;
    }

    while ((c = fgetc(f)) != EOF) {
        if (c == '\n') break;
        if (c == '\r') continue;
        if (len + 1 >= ln->cap) {
            size_t new_cap = ln->cap * 2;
            char* nb = (char*)realloc(ln->buf, new_cap);
            if (!nb) { ln->buf = nullptr; ln->cap = 0; ln->done = 0; return 0; }
            ln->buf = nb;
            ln->cap = new_cap;
        }
        ln->buf[len++] = (char)c;
    }
    if (c == EOF && len == 0) {
        ln->done = 0; /* EOF before any content of this line */
        return 0;
    }
    ln->buf[len] = '\0';
    return 1;
}

RCRE_AGTensor* rcre_data_load_csv(RCRE_Arena* arena, const char* filepath, bool skip_header, size_t max_rows) {
    if (arena == nullptr || filepath == nullptr) return nullptr;
    FILE* f = fopen(filepath, "r");
    if (!f) return nullptr;

    csv_line_t line = { nullptr, 0, 0 };

    if (skip_header && !csv_line_read(f, &line)) {
        csv_line_free(&line);
        fclose(f);
        return nullptr;
    }

    /* Column count comes from the first data row. The rewind position must
     * be captured BEFORE reading it, so the parse pass sees the same start. */
    long first_data_pos = ftell(f);
    int have_first = csv_line_read(f, &line);
    if (!have_first) {
        csv_line_free(&line);
        fclose(f);
        return nullptr;
    }
    int c0 = 0;
    char* save = nullptr;
    char* tok = strtok_r(line.buf, ",\t", &save);
    while (tok != nullptr) { c0++; tok = strtok_r(nullptr, ",\t", &save); }
    if (c0 == 0) {
        csv_line_free(&line);
        fclose(f);
        return nullptr;
    }
    size_t cols = (size_t)c0;

    size_t rows = 1;
    if (max_rows == 0 || rows < max_rows) {
        int rc;
        while ((rc = csv_line_read(f, &line)) != 0) {
            rows++;
            if (max_rows > 0 && rows >= max_rows) break;
        }
        (void)rc;
    }
    if (rows == 0) {
        csv_line_free(&line);
        fclose(f);
        return nullptr;
    }

    fseek(f, first_data_pos, SEEK_SET);

    RCRE_AGTensor* t = rcre_ag_tensor_create_2d(arena, rows, cols);
    if (!t) {
        csv_line_free(&line);
        fclose(f);
        return nullptr;
    }

    /* Parse pass: every row must carry exactly `cols` fields. Ragged or
     * truncated rows are a data error -> reject the whole file (fail fast
     * instead of silently zero-filling missing values). */
    size_t r = 0;
    csv_line_read(f, &line);
    for (;;) {
        size_t c = 0;
        save = nullptr;
        tok = strtok_r(line.buf, ",\t", &save);
        while (tok != nullptr) {
            if (c >= cols) break;
            t->data[r * cols + c] = (float)atof(tok);
            c++;
            tok = strtok_r(nullptr, ",\t", &save);
        }
        if (c != cols) {
            csv_line_free(&line);
            fclose(f);
            return nullptr;
        }
        if (++r >= rows) break;
        if (!csv_line_read(f, &line)) break;
    }

    csv_line_free(&line);
    fclose(f);
    return t;
}

RCRE_StandardScaler* rcre_standard_scaler_fit(RCRE_Arena* arena, const RCRE_AGTensor* X) {
    if (arena == nullptr || !X || X->ndim != 2 || X->shape[0] == 0) return nullptr;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];

    RCRE_StandardScaler* scaler = (RCRE_StandardScaler*)rcre_arena_alloc(arena, sizeof(RCRE_StandardScaler), alignof(RCRE_StandardScaler));
    scaler->mean = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
    scaler->std = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
    if (!scaler || !scaler->mean || !scaler->std) return nullptr;
    scaler->num_features = cols;

    for (size_t c = 0; c < cols; ++c) {
        float sum = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            sum += X->data[r * cols + c];
        }
        float m = sum / (float)rows;
        scaler->mean[c] = m;

        float sq_diff = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            float d = X->data[r * cols + c] - m;
            sq_diff += d * d;
        }
        float s = sqrtf(sq_diff / (float)rows);
        scaler->std[c] = (s < 1e-8f) ? 1.0f : s;
    }

    return scaler;
}

void rcre_standard_scaler_transform(const RCRE_StandardScaler* scaler, RCRE_AGTensor* X) {
    if (!scaler || !X || X->ndim != 2 || X->shape[1] != scaler->num_features) return;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            X->data[r * cols + c] = (X->data[r * cols + c] - scaler->mean[c]) / scaler->std[c];
        }
    }
}

void rcre_standard_scaler_inverse_transform(const RCRE_StandardScaler* scaler, RCRE_AGTensor* X) {
    if (!scaler || !X || X->ndim != 2 || X->shape[1] != scaler->num_features) return;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            X->data[r * cols + c] = (X->data[r * cols + c] * scaler->std[c]) + scaler->mean[c];
        }
    }
}

RCRE_MinMaxScaler* rcre_minmax_scaler_fit(RCRE_Arena* arena, const RCRE_AGTensor* X) {
    if (arena == nullptr || !X || X->ndim != 2 || X->shape[0] == 0) return nullptr;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];

    RCRE_MinMaxScaler* scaler = (RCRE_MinMaxScaler*)rcre_arena_alloc(arena, sizeof(RCRE_MinMaxScaler), alignof(RCRE_MinMaxScaler));
    scaler->min_val = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
    scaler->max_val = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
    if (!scaler || !scaler->min_val || !scaler->max_val) return nullptr;
    scaler->num_features = cols;

    for (size_t c = 0; c < cols; ++c) {
        float min_v = X->data[c];
        float max_v = X->data[c];
        for (size_t r = 1; r < rows; ++r) {
            float v = X->data[r * cols + c];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        scaler->min_val[c] = min_v;
        scaler->max_val[c] = (max_v - min_v < 1e-8f) ? min_v + 1.0f : max_v;
    }
    return scaler;
}

void rcre_minmax_scaler_transform(const RCRE_MinMaxScaler* scaler, RCRE_AGTensor* X) {
    if (!scaler || !X || X->ndim != 2 || X->shape[1] != scaler->num_features) return;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float min_v = scaler->min_val[c];
            float max_v = scaler->max_val[c];
            X->data[r * cols + c] = (X->data[r * cols + c] - min_v) / (max_v - min_v);
        }
    }
}

void rcre_data_train_test_split(RCRE_Arena* arena, const RCRE_AGTensor* X, const RCRE_AGTensor* y,
                               float train_ratio,
                               RCRE_AGTensor** X_train, RCRE_AGTensor** y_train,
                               RCRE_AGTensor** X_test, RCRE_AGTensor** y_test) {
    if (arena == nullptr || !X || !y || X->ndim != 2) return;
    /* y may be 1-D (n,) regression targets or 2-D (n, k). A 1-D y is treated
     * as (n, 1); outputs are always 2-D. */
    if (y->ndim != 1 && y->ndim != 2) return;
    if (y->shape[0] != X->shape[0]) return;
    if (!(train_ratio > 0.0f && train_ratio < 1.0f)) return;
    if (!X_train || !y_train || !X_test || !y_test) return;

    size_t total_samples = X->shape[0];
    size_t train_samples = (size_t)((float)total_samples * train_ratio);
    size_t test_samples = total_samples - train_samples;
    if (train_samples == 0 || test_samples == 0) return;

    size_t x_features = X->shape[1];
    size_t y_features = (y->ndim == 2) ? y->shape[1] : 1;

    *X_train = rcre_ag_tensor_create_2d(arena, train_samples, x_features);
    *X_test  = rcre_ag_tensor_create_2d(arena, test_samples, x_features);
    *y_train = rcre_ag_tensor_create_2d(arena, train_samples, y_features);
    *y_test  = rcre_ag_tensor_create_2d(arena, test_samples, y_features);

    if (!*X_train || !*X_test || !*y_train || !*y_test) return;

    memcpy((*X_train)->data, X->data, train_samples * x_features * sizeof(float));
    memcpy((*X_test)->data, X->data + train_samples * x_features, test_samples * x_features * sizeof(float));

    memcpy((*y_train)->data, y->data, train_samples * y_features * sizeof(float));
    memcpy((*y_test)->data, y->data + train_samples * y_features, test_samples * y_features * sizeof(float));
}

RCRE_BatchIter* rcre_batch_iter_create(RCRE_Arena* arena, RCRE_AGTensor* X, RCRE_AGTensor* y, size_t batch_size, bool shuffle) {
    if (arena == nullptr || !X || !y || batch_size == 0) return nullptr;
    if (X->ndim != 2) return nullptr;
    if (y->ndim != 1 && y->ndim != 2) return nullptr;
    if (X->shape[0] != y->shape[0]) return nullptr;

    RCRE_BatchIter* iter = (RCRE_BatchIter*)rcre_arena_alloc(arena, sizeof(RCRE_BatchIter), alignof(RCRE_BatchIter));
    iter->indices = (size_t*)rcre_arena_alloc(arena, X->shape[0] * sizeof(size_t), alignof(size_t));
    if (!iter || !iter->indices) return nullptr;

    iter->X = X;
    iter->y = y;
    iter->y_cols = (y->ndim == 2) ? y->shape[1] : 1;
    iter->batch_size = batch_size;
    iter->current_idx = 0;
    iter->num_samples = X->shape[0];
    iter->shuffle = shuffle;
    iter->rng_state = 2026;

    for (size_t i = 0; i < iter->num_samples; ++i) {
        iter->indices[i] = i;
    }

    rcre_batch_iter_reset(iter);
    return iter;
}

void rcre_batch_iter_reset(RCRE_BatchIter* iter) {
    if (iter == nullptr) return;
    iter->current_idx = 0;

    if (iter->shuffle) {
        for (size_t i = iter->num_samples - 1; i > 0; --i) {
            size_t j = (size_t)(rcre_rng_next(&iter->rng_state) % (uint64_t)(i + 1));
            size_t tmp = iter->indices[i];
            iter->indices[i] = iter->indices[j];
            iter->indices[j] = tmp;
        }
    }
}

bool rcre_batch_iter_next(RCRE_BatchIter* iter, RCRE_AGTensor* batch_X, RCRE_AGTensor* batch_y) {
    if (iter == nullptr || batch_X == nullptr || batch_y == nullptr) return false;
    if (iter->current_idx >= iter->num_samples) return false;

    size_t remaining = iter->num_samples - iter->current_idx;
    size_t cur_batch = (remaining < iter->batch_size) ? remaining : iter->batch_size;

    size_t x_dim = iter->X->shape[1];
    size_t y_dim = iter->y_cols;

    /* Output buffers are caller-allocated: verify capacity so an undersized
     * batch tensor fails cleanly instead of corrupting neighbouring arena
     * allocations (which ASan cannot see inside a single arena buffer). */
    if (batch_X->numel < cur_batch * x_dim || batch_y->numel < cur_batch * y_dim) return false;

    for (size_t b = 0; b < cur_batch; ++b) {
        size_t idx = iter->indices[iter->current_idx + b];
        memcpy(batch_X->data + b * x_dim, iter->X->data + idx * x_dim, x_dim * sizeof(float));
        memcpy(batch_y->data + b * y_dim, iter->y->data + idx * y_dim, y_dim * sizeof(float));
    }

    iter->current_idx += cur_batch;
    return true;
}
