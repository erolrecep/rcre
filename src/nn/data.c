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

static bool data_is_f32(const RCRE_Tensor* t) {
    return t != nullptr && t->data != nullptr && t->dtype == RCRE_DTYPE_F32;
}

/* ========================================================================= */
/* High-Throughput Memory-Mapped & SIMD-Accelerated CSV Loader               */
/* ========================================================================= */

RCRE_Tensor* rcre_data_load_csv(RCRE_Arena* arena, const char* filepath, bool skip_header, size_t max_rows) {
    if (arena == nullptr || filepath == nullptr) return nullptr;

    RCRE_ArenaTemp checkpoint = rcre_arena_temp_begin(arena);
    RCRE_Tensor* t = (RCRE_Tensor*)rcre_arena_alloc(arena, sizeof(RCRE_Tensor), alignof(RCRE_Tensor));
    if (!t) {
        rcre_arena_temp_end(checkpoint);
        return nullptr;
    }

    RCRE_CSVOptions opts = {
        .delimiter      = 0, /* Auto: ',' or '\t' */
        .skip_header    = skip_header,
        .strict_numbers = false,
        .max_rows       = max_rows,
        .pool           = nullptr
    };

    if (rcre_io_load_csv_ex(filepath, &opts, arena, t) != RCRE_OK) {
        rcre_arena_temp_end(checkpoint);
        return nullptr;
    }

    t->grad = (float*)rcre_arena_alloc(arena, t->numel * sizeof(float), 64);
    if (!t->grad) {
        rcre_arena_temp_end(checkpoint);
        return nullptr;
    }
    memset(t->grad, 0, t->numel * sizeof(float));
    return t;
}

RCRE_StandardScaler* rcre_standard_scaler_fit(RCRE_Arena* arena, const RCRE_Tensor* X) {
    if (arena == nullptr || !data_is_f32(X) || X->ndim != 2 || X->shape[0] == 0) return nullptr;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];

    RCRE_StandardScaler* scaler = (RCRE_StandardScaler*)rcre_arena_alloc(arena, sizeof(RCRE_StandardScaler), alignof(RCRE_StandardScaler));
    scaler->mean = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
    scaler->std = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
    if (!scaler || !scaler->mean || !scaler->std) return nullptr;
    scaler->num_features = cols;
    const float* xd = rcre_tensor_f32_data(X);

    for (size_t c = 0; c < cols; ++c) {
        float sum = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            sum += xd[r * cols + c];
        }
        float m = sum / (float)rows;
        scaler->mean[c] = m;

        float sq_diff = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            float d = xd[r * cols + c] - m;
            sq_diff += d * d;
        }
        float s = sqrtf(sq_diff / (float)rows);
        scaler->std[c] = (s < 1e-8f) ? 1.0f : s;
    }

    return scaler;
}

void rcre_standard_scaler_transform(const RCRE_StandardScaler* scaler, RCRE_Tensor* X) {
    if (!scaler || !data_is_f32(X) || X->ndim != 2 || X->shape[1] != scaler->num_features) return;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];
    float* xd = rcre_tensor_f32_data(X);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            xd[r * cols + c] = (xd[r * cols + c] - scaler->mean[c]) / scaler->std[c];
        }
    }
}

void rcre_standard_scaler_inverse_transform(const RCRE_StandardScaler* scaler, RCRE_Tensor* X) {
    if (!scaler || !data_is_f32(X) || X->ndim != 2 || X->shape[1] != scaler->num_features) return;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];
    float* xd = rcre_tensor_f32_data(X);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            xd[r * cols + c] = (xd[r * cols + c] * scaler->std[c]) + scaler->mean[c];
        }
    }
}

RCRE_MinMaxScaler* rcre_minmax_scaler_fit(RCRE_Arena* arena, const RCRE_Tensor* X) {
    if (arena == nullptr || !data_is_f32(X) || X->ndim != 2 || X->shape[0] == 0) return nullptr;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];

    RCRE_MinMaxScaler* scaler = (RCRE_MinMaxScaler*)rcre_arena_alloc(arena, sizeof(RCRE_MinMaxScaler), alignof(RCRE_MinMaxScaler));
    scaler->min_val = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
    scaler->max_val = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
    if (!scaler || !scaler->min_val || !scaler->max_val) return nullptr;
    scaler->num_features = cols;

    const float* xd = rcre_tensor_f32_data(X);
    for (size_t c = 0; c < cols; ++c) {
        float min_v = xd[c];
        float max_v = xd[c];
        for (size_t r = 1; r < rows; ++r) {
            float v = xd[r * cols + c];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        scaler->min_val[c] = min_v;
        scaler->max_val[c] = (max_v - min_v < 1e-8f) ? min_v + 1.0f : max_v;
    }
    return scaler;
}

void rcre_minmax_scaler_transform(const RCRE_MinMaxScaler* scaler, RCRE_Tensor* X) {
    if (!scaler || !data_is_f32(X) || X->ndim != 2 || X->shape[1] != scaler->num_features) return;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];

    float* xd = rcre_tensor_f32_data(X);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float min_v = scaler->min_val[c];
            float max_v = scaler->max_val[c];
            xd[r * cols + c] = (xd[r * cols + c] - min_v) / (max_v - min_v);
        }
    }
}

void rcre_data_train_test_split(RCRE_Arena* arena, const RCRE_Tensor* X, const RCRE_Tensor* y,
                               float train_ratio,
                               RCRE_Tensor** X_train, RCRE_Tensor** y_train,
                               RCRE_Tensor** X_test, RCRE_Tensor** y_test) {
    if (arena == nullptr || !data_is_f32(X) || !data_is_f32(y) || X->ndim != 2) return;
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

    *X_train = rcre_tensor_alloc_2d(arena, train_samples, x_features);
    *X_test  = rcre_tensor_alloc_2d(arena, test_samples, x_features);
    *y_train = rcre_tensor_alloc_2d(arena, train_samples, y_features);
    *y_test  = rcre_tensor_alloc_2d(arena, test_samples, y_features);

    if (!*X_train || !*X_test || !*y_train || !*y_test) return;

    const float* xd = rcre_tensor_f32_data(X);
    const float* yd = rcre_tensor_f32_data(y);
    memcpy(rcre_tensor_f32_data(*X_train), xd, train_samples * x_features * sizeof(float));
    memcpy(rcre_tensor_f32_data(*X_test), xd + train_samples * x_features, test_samples * x_features * sizeof(float));

    memcpy(rcre_tensor_f32_data(*y_train), yd, train_samples * y_features * sizeof(float));
    memcpy(rcre_tensor_f32_data(*y_test), yd + train_samples * y_features, test_samples * y_features * sizeof(float));
}

RCRE_BatchIter* rcre_batch_iter_create(RCRE_Arena* arena, RCRE_Tensor* X, RCRE_Tensor* y, size_t batch_size, bool shuffle) {
    if (arena == nullptr || !data_is_f32(X) || !data_is_f32(y) || batch_size == 0) return nullptr;
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

bool rcre_batch_iter_next(RCRE_BatchIter* iter, RCRE_Tensor* batch_X, RCRE_Tensor* batch_y) {
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

    const float* xd = rcre_tensor_f32_data(iter->X);
    const float* yd = rcre_tensor_f32_data(iter->y);
    float* bxd = rcre_tensor_f32_data(batch_X);
    float* byd = rcre_tensor_f32_data(batch_y);
    for (size_t b = 0; b < cur_batch; ++b) {
        size_t idx = iter->indices[iter->current_idx + b];
        memcpy(bxd + b * x_dim, xd + idx * x_dim, x_dim * sizeof(float));
        memcpy(byd + b * y_dim, yd + idx * y_dim, y_dim * sizeof(float));
    }

    iter->current_idx += cur_batch;
    return true;
}
