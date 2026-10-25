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

#include <math.h>

RCRE_NODISCARD int rcre_cv_rgb_to_hsv(const RCRE_Tensor* src, RCRE_Tensor* dst) {
    if (!src || !src->data || !dst || !dst->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || dst->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }
    if (src->ndim != 3 || src->shape[2] != 3 || dst->ndim != 3 || dst->shape[2] != 3) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }
    size_t height = src->shape[0];
    size_t width = src->shape[1];
    if (dst->shape[0] != height || dst->shape[1] != width) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t src_sy = (size_t)src->strides[0];
    size_t src_sx = (size_t)src->strides[1];
    size_t dst_sy = (size_t)dst->strides[0];
    size_t dst_sx = (size_t)dst->strides[1];

    const float* in = (const float*)src->data;
    float* out = (float*)dst->data;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t in_idx = y * src_sy + x * src_sx;
            size_t out_idx = y * dst_sy + x * dst_sx;

            float r = in[in_idx + 0];
            float g = in[in_idx + 1];
            float b = in[in_idx + 2];

            float max_val = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
            float min_val = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);
            float c = max_val - min_val;
            float v = max_val;
            float s = (v == 0.0f) ? 0.0f : (c / v);

            float h = 0.0f;
            if (c != 0.0f) {
                if (max_val == r) {
                    h = (g - b) / c;
                } else if (max_val == g) {
                    h = (b - r) / c + 2.0f;
                } else {
                    h = (r - g) / c + 4.0f;
                }
                h /= 6.0f;
                if (h < 0.0f) h += 1.0f;
            }

            out[out_idx + 0] = h;
            out[out_idx + 1] = s;
            out[out_idx + 2] = v;
        }
    }
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_hsv_to_rgb(const RCRE_Tensor* src, RCRE_Tensor* dst) {
    if (!src || !src->data || !dst || !dst->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || dst->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }
    if (src->ndim != 3 || src->shape[2] != 3 || dst->ndim != 3 || dst->shape[2] != 3) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }
    size_t height = src->shape[0];
    size_t width = src->shape[1];
    if (dst->shape[0] != height || dst->shape[1] != width) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t src_sy = (size_t)src->strides[0];
    size_t src_sx = (size_t)src->strides[1];
    size_t dst_sy = (size_t)dst->strides[0];
    size_t dst_sx = (size_t)dst->strides[1];

    const float* in = (const float*)src->data;
    float* out = (float*)dst->data;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t in_idx = y * src_sy + x * src_sx;
            size_t out_idx = y * dst_sy + x * dst_sx;

            float h = in[in_idx + 0];
            float s = in[in_idx + 1];
            float v = in[in_idx + 2];

            float c = v * s;
            float h_prime = h * 6.0f;
            float mod_val = fmodf(h_prime, 2.0f);
            float x_val = c * (1.0f - fabsf(mod_val - 1.0f));
            float m = v - c;

            float r1 = 0.0f, g1 = 0.0f, b1 = 0.0f;
            if (h_prime >= 0.0f && h_prime < 1.0f) {
                r1 = c; g1 = x_val; b1 = 0.0f;
            } else if (h_prime >= 1.0f && h_prime < 2.0f) {
                r1 = x_val; g1 = c; b1 = 0.0f;
            } else if (h_prime >= 2.0f && h_prime < 3.0f) {
                r1 = 0.0f; g1 = c; b1 = x_val;
            } else if (h_prime >= 3.0f && h_prime < 4.0f) {
                r1 = 0.0f; g1 = x_val; b1 = c;
            } else if (h_prime >= 4.0f && h_prime < 5.0f) {
                r1 = x_val; g1 = 0.0f; b1 = c;
            } else if (h_prime >= 5.0f && h_prime <= 6.0f) {
                r1 = c; g1 = 0.0f; b1 = x_val;
            }

            out[out_idx + 0] = r1 + m;
            out[out_idx + 1] = g1 + m;
            out[out_idx + 2] = b1 + m;
        }
    }
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_scale_channel(RCRE_Tensor* tensor, size_t channel_idx, float scale) {
    if (!tensor || !tensor->data || tensor->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (tensor->ndim != 3) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }
    size_t height = tensor->shape[0];
    size_t width = tensor->shape[1];
    size_t channels = tensor->shape[2];
    if (channel_idx >= channels) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t sy = (size_t)tensor->strides[0];
    size_t sx = (size_t)tensor->strides[1];
    float* data = (float*)tensor->data;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            data[y * sy + x * sx + channel_idx] *= scale;
        }
    }
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_shift_channel(RCRE_Tensor* tensor, size_t channel_idx, float shift) {
    if (!tensor || !tensor->data || tensor->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (tensor->ndim != 3) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }
    size_t height = tensor->shape[0];
    size_t width = tensor->shape[1];
    size_t channels = tensor->shape[2];
    if (channel_idx >= channels) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t sy = (size_t)tensor->strides[0];
    size_t sx = (size_t)tensor->strides[1];
    float* data = (float*)tensor->data;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            data[y * sy + x * sx + channel_idx] += shift;
        }
    }
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_clamp(RCRE_Tensor* tensor, float min_val, float max_val) {
    if (!tensor || !tensor->data || tensor->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_INVALID_ARG;
    }
    size_t total = 1;
    for (size_t i = 0; i < tensor->ndim; ++i) {
        total *= tensor->shape[i];
    }
    float* data = (float*)tensor->data;
    for (size_t i = 0; i < total; ++i) {
        float val = data[i];
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        data[i] = val;
    }
    return RCRE_OK;
}

