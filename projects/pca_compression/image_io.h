/**
 * @file image_io.h
 * @brief High-Speed Image I/O and Test Pattern Generator for RCRE PCA Compression
 */

#ifndef RCRE_PCA_IMAGE_IO_H
#define RCRE_PCA_IMAGE_IO_H

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_SIMD
#include "stb_image.h"

static inline int rcre_save_bmp(const char* filename, const RCRE_Tensor* tensor) {
    if (!filename || !tensor || !tensor->data || tensor->ndim != 3) return -1;
    size_t H = tensor->shape[0];
    size_t W = tensor->shape[1];
    size_t C = tensor->shape[2];

    FILE* f = fopen(filename, "wb");
    if (!f) return -1;

    size_t row_stride = (W * 3 + 3) & ~3;
    size_t data_size = row_stride * H;
    size_t file_size = 54 + data_size;

    uint8_t header[54] = {
        'B', 'M',
        (uint8_t)(file_size), (uint8_t)(file_size >> 8), (uint8_t)(file_size >> 16), (uint8_t)(file_size >> 24),
        0, 0, 0, 0,
        54, 0, 0, 0,
        40, 0, 0, 0,
        (uint8_t)(W), (uint8_t)(W >> 8), (uint8_t)(W >> 16), (uint8_t)(W >> 24),
        (uint8_t)(H), (uint8_t)(H >> 8), (uint8_t)(H >> 16), (uint8_t)(H >> 24),
        1, 0,
        24, 0,
        0, 0, 0, 0,
        (uint8_t)(data_size), (uint8_t)(data_size >> 8), (uint8_t)(data_size >> 16), (uint8_t)(data_size >> 24),
        0x13, 0x0B, 0, 0,
        0x13, 0x0B, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    fwrite(header, 1, 54, f);
    const float* data = (const float*)tensor->data;
    uint8_t* row_buf = (uint8_t*)calloc(1, row_stride);

    for (int y = (int)H - 1; y >= 0; --y) {
        for (size_t x = 0; x < W; ++x) {
            size_t src_idx = (y * W + x) * C;
            float r = data[src_idx + 0];
            float g = (C >= 2) ? data[src_idx + 1] : r;
            float b = (C >= 3) ? data[src_idx + 2] : r;

            if (r < 0.0f) r = 0.0f; else if (r > 255.0f) r = 255.0f;
            if (g < 0.0f) g = 0.0f; else if (g > 255.0f) g = 255.0f;
            if (b < 0.0f) b = 0.0f; else if (b > 255.0f) b = 255.0f;

            row_buf[x * 3 + 0] = (uint8_t)b;
            row_buf[x * 3 + 1] = (uint8_t)g;
            row_buf[x * 3 + 2] = (uint8_t)r;
        }
        fwrite(row_buf, 1, row_stride, f);
    }

    free(row_buf);
    fclose(f);
    return 0;
}

static inline int rcre_load_image(const char* filename, RCRE_Tensor* out, RCRE_Arena* arena) {
    if (!filename || !out || !arena) return -1;
    int w = 0, h = 0, ch = 0;
    stbi_uc* raw = stbi_load(filename, &w, &h, &ch, 3);
    if (!raw) return -1;

    size_t shape[3] = {(size_t)h, (size_t)w, 3};
    *out = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!out->data) {
        stbi_image_free(raw);
        return -1;
    }

    float* dst = (float*)out->data;
    size_t total = (size_t)h * (size_t)w * 3;
    for (size_t i = 0; i < total; ++i) {
        dst[i] = (float)raw[i];
    }

    stbi_image_free(raw);
    return 0;
}

static inline void rcre_generate_test_image(RCRE_Tensor* out, size_t H, size_t W, RCRE_Arena* arena) {
    size_t shape[3] = {H, W, 3};
    *out = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    float* dst = (float*)out->data;

    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            float u = (float)x / (float)W;
            float v = (float)y / (float)H;

            /* Colorful radial gradient + smooth sinusoids */
            float dx = u - 0.5f;
            float dy = v - 0.5f;
            float dist = sqrtf(dx * dx + dy * dy);

            float r = (sinf(u * 12.0f) * 0.5f + 0.5f) * 255.0f;
            float g = (cosf(v * 16.0f) * 0.5f + 0.5f) * 255.0f;
            float b = (1.0f - dist * 1.4f) * 255.0f;
            if (b < 0.0f) b = 0.0f;

            /* Add some high-frequency checker detail */
            if (((x / 16) + (y / 16)) % 2 == 0) {
                r = r * 0.85f + 30.0f;
                g = g * 0.85f + 30.0f;
            }

            size_t idx = (y * W + x) * 3;
            dst[idx + 0] = r;
            dst[idx + 1] = g;
            dst[idx + 2] = b;
        }
    }
}

#endif /* RCRE_PCA_IMAGE_IO_H */
