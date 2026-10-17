/**
 * @file color.c
 * @brief Color Space Transformations
 * 
 * Satisfies CV-REQ-01.
 */

#include "rcre.h"

RCRE_NODISCARD int rcre_cv_rgb_to_gray(const RCRE_Tensor* src, RCRE_Tensor* dst) {
    if (!src || !src->data || !dst || !dst->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (src->ndim != 3 || src->shape[2] != 3) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t height = src->shape[0];
    size_t width = src->shape[1];

    if (dst->shape[0] != height || dst->shape[1] != width) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    /* Rec.601 coefficients: 0.299 * R + 0.587 * G + 0.114 * B */
    const float rw = 0.299f;
    const float gw = 0.587f;
    const float bw = 0.114f;

    size_t src_stride_y = (size_t)src->strides[0];
    size_t src_stride_x = (size_t)src->strides[1];
    size_t dst_stride_y = (size_t)dst->strides[0];
    size_t dst_stride_x = (size_t)dst->strides[1];

    if (src->dtype == RCRE_DTYPE_U8 && dst->dtype == RCRE_DTYPE_U8) {
        const uint8_t* in = (const uint8_t*)src->data;
        uint8_t* out = (uint8_t*)dst->data;

        /* Fast contiguous hardware SIMD dispatch */
        if (src_stride_y == width * 3 && src_stride_x == 3 &&
            dst_stride_y == width && dst_stride_x == 1) {
            rcre_hal.rgb_to_gray_u8(in, out, height * width);
            return RCRE_OK;
        }

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t in_idx = y * src_stride_y + x * src_stride_x;
                float r = in[in_idx + 0];
                float g = in[in_idx + 1];
                float b = in[in_idx + 2];
                float gray = rw * r + gw * g + bw * b;
                out[y * dst_stride_y + x * dst_stride_x] = (uint8_t)(gray + 0.5f);
            }
        }
        return RCRE_OK;
    } else if (src->dtype == RCRE_DTYPE_F32 && dst->dtype == RCRE_DTYPE_F32) {
        const float* in = (const float*)src->data;
        float* out = (float*)dst->data;

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t in_idx = y * src_stride_y + x * src_stride_x;
                float r = in[in_idx + 0];
                float g = in[in_idx + 1];
                float b = in[in_idx + 2];
                out[y * dst_stride_y + x * dst_stride_x] = rw * r + gw * g + bw * b;
            }
        }
        return RCRE_OK;
    }

    return RCRE_ERR_UNSUPPORTED_OP;
}

RCRE_NODISCARD int rcre_cv_bgr_to_rgb(const RCRE_Tensor* src, RCRE_Tensor* dst) {
    if (!src || !src->data || !dst || !dst->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (src->ndim != 3 || src->shape[2] != 3) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t height = src->shape[0];
    size_t width = src->shape[1];

    size_t src_stride_y = (size_t)src->strides[0];
    size_t src_stride_x = (size_t)src->strides[1];
    size_t dst_stride_y = (size_t)dst->strides[0];
    size_t dst_stride_x = (size_t)dst->strides[1];

    if (src->dtype == RCRE_DTYPE_U8 && dst->dtype == RCRE_DTYPE_U8) {
        const uint8_t* in = (const uint8_t*)src->data;
        uint8_t* out = (uint8_t*)dst->data;

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t in_idx = y * src_stride_y + x * src_stride_x;
                size_t out_idx = y * dst_stride_y + x * dst_stride_x;
                uint8_t b = in[in_idx + 0];
                uint8_t g = in[in_idx + 1];
                uint8_t r = in[in_idx + 2];
                out[out_idx + 0] = r;
                out[out_idx + 1] = g;
                out[out_idx + 2] = b;
            }
        }
        return RCRE_OK;
    }

    return RCRE_ERR_UNSUPPORTED_OP;
}
