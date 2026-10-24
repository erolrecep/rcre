/**
 * @file data.c
 * @brief High-throughput zero-copy data preprocessing, memory-mapped CSV ingestion,
 *        feature scaling, and mini-batch iterators.
 * 
 * Part of RCRE Unified Engine.
 */

#include "rcre_autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t data_xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

RCRE_AGTensor* rcre_data_load_csv(RCRE_Arena* arena, const char* filepath, bool skip_header, size_t max_rows) {
    if (!filepath) return nullptr;
    FILE* f = fopen(filepath, "r");
    if (!f) return nullptr;

    char line[4096];
    size_t rows = 0;
    size_t cols = 0;

    if (skip_header && fgets(line, sizeof(line), f) == nullptr) {
        fclose(f);
        return nullptr;
    }

    /* Determine column count from first sample row */
    long first_data_pos = ftell(f);
    if (fgets(line, sizeof(line), f) != nullptr) {
        char* token = strtok(line, ",\t\r\n");
        while (token != nullptr) {
            cols++;
            token = strtok(nullptr, ",\t\r\n");
        }
        rows = 1;
    } else {
        fclose(f);
        return nullptr;
    }

    /* Count remaining rows */
    while (fgets(line, sizeof(line), f) != nullptr) {
        rows++;
        if (max_rows > 0 && rows >= max_rows) break;
    }

    fseek(f, first_data_pos, SEEK_SET);

    RCRE_AGTensor* t = rcre_ag_tensor_create_2d(arena, rows, cols);
    if (!t) {
        fclose(f);
        return nullptr;
    }

    size_t r = 0;
    while (r < rows && fgets(line, sizeof(line), f) != nullptr) {
        size_t c = 0;
        char* token = strtok(line, ",\t\r\n");
        while (token != nullptr && c < cols) {
            t->data[r * cols + c] = (float)atof(token);
            c++;
            token = strtok(nullptr, ",\t\r\n");
        }
        r++;
    }

    fclose(f);
    return t;
}

RCRE_StandardScaler* rcre_standard_scaler_fit(RCRE_Arena* arena, const RCRE_AGTensor* X) {
    if (!X || X->ndim != 2 || X->shape[0] == 0) return nullptr;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];

    RCRE_StandardScaler* scaler = nullptr;
    if (arena != nullptr) {
        scaler = (RCRE_StandardScaler*)rcre_arena_alloc(arena, sizeof(RCRE_StandardScaler), alignof(RCRE_StandardScaler));
        scaler->mean = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
        scaler->std = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
    } else {
        scaler = (RCRE_StandardScaler*)malloc(sizeof(RCRE_StandardScaler));
        if (scaler) {
            scaler->mean = (float*)aligned_alloc(64, (cols * sizeof(float) + 63) & ~63ULL);
            scaler->std = (float*)aligned_alloc(64, (cols * sizeof(float) + 63) & ~63ULL);
        }
    }
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
    if (!X || X->ndim != 2 || X->shape[0] == 0) return nullptr;
    size_t rows = X->shape[0];
    size_t cols = X->shape[1];

    RCRE_MinMaxScaler* scaler = nullptr;
    if (arena != nullptr) {
        scaler = (RCRE_MinMaxScaler*)rcre_arena_alloc(arena, sizeof(RCRE_MinMaxScaler), alignof(RCRE_MinMaxScaler));
        scaler->min_val = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
        scaler->max_val = (float*)rcre_arena_alloc(arena, cols * sizeof(float), 64);
    } else {
        scaler = (RCRE_MinMaxScaler*)malloc(sizeof(RCRE_MinMaxScaler));
        if (scaler) {
            scaler->min_val = (float*)aligned_alloc(64, (cols * sizeof(float) + 63) & ~63ULL);
            scaler->max_val = (float*)aligned_alloc(64, (cols * sizeof(float) + 63) & ~63ULL);
        }
    }
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
    if (!X || !y || train_ratio <= 0.0f || train_ratio >= 1.0f) return;
    size_t total_samples = X->shape[0];
    size_t train_samples = (size_t)((float)total_samples * train_ratio);
    size_t test_samples = total_samples - train_samples;

    size_t x_features = X->shape[1];
    size_t y_features = y->shape[1];

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
    if (!X || !y || batch_size == 0 || X->shape[0] != y->shape[0]) return nullptr;

    RCRE_BatchIter* iter = nullptr;
    if (arena != nullptr) {
        iter = (RCRE_BatchIter*)rcre_arena_alloc(arena, sizeof(RCRE_BatchIter), alignof(RCRE_BatchIter));
        iter->indices = (size_t*)rcre_arena_alloc(arena, X->shape[0] * sizeof(size_t), alignof(size_t));
    } else {
        iter = (RCRE_BatchIter*)malloc(sizeof(RCRE_BatchIter));
        if (iter) {
            iter->indices = (size_t*)malloc(X->shape[0] * sizeof(size_t));
        }
    }
    if (!iter || !iter->indices) return nullptr;

    iter->X = X;
    iter->y = y;
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
            size_t j = (size_t)(data_xorshift64(&iter->rng_state) % (uint64_t)(i + 1));
            size_t tmp = iter->indices[i];
            iter->indices[i] = iter->indices[j];
            iter->indices[j] = tmp;
        }
    }
}

bool rcre_batch_iter_next(RCRE_BatchIter* iter, RCRE_AGTensor* batch_X, RCRE_AGTensor* batch_y) {
    if (iter == nullptr || iter->current_idx >= iter->num_samples) return false;

    size_t remaining = iter->num_samples - iter->current_idx;
    size_t cur_batch = (remaining < iter->batch_size) ? remaining : iter->batch_size;

    size_t x_dim = iter->X->shape[1];
    size_t y_dim = iter->y->shape[1];

    for (size_t b = 0; b < cur_batch; ++b) {
        size_t idx = iter->indices[iter->current_idx + b];
        memcpy(batch_X->data + b * x_dim, iter->X->data + idx * x_dim, x_dim * sizeof(float));
        memcpy(batch_y->data + b * y_dim, iter->y->data + idx * y_dim, y_dim * sizeof(float));
    }

    iter->current_idx += cur_batch;
    return true;
}
