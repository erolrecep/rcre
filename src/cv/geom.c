/**
 * @file geom.c
 * @brief Continuous Geometric Transforms and Image Resizing
 *
 * Implements Nearest-Neighbor and Bilinear interpolation with continuous
 * subpixel coordinate mapping, preserving aspect ratio and preventing boundary drift.
 */

#include "rcre.h"
#include <math.h>

static inline int geom_clamp_int(int val, int min_val, int max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static inline float geom_get_f32_pixel_hwc(const float* data, int x, int y, int c,
                                           int w, int h, int num_channels,
                                           int stride_y, int stride_x) {
    x = geom_clamp_int(x, 0, w - 1);
    y = geom_clamp_int(y, 0, h - 1);
    c = geom_clamp_int(c, 0, num_channels - 1);
    return data[y * stride_y + x * stride_x + c];
}

RCRE_NODISCARD int rcre_cv_resize_nn(const RCRE_Tensor* src, RCRE_Tensor* dst) {
    if (!src || !src->data || !dst || !dst->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || dst->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t in_h, in_w, in_c;
    size_t out_h, out_w, out_c;

    if (src->ndim == 2) {
        in_h = src->shape[0];
        in_w = src->shape[1];
        in_c = 1;
    } else if (src->ndim == 3) {
        in_h = src->shape[0];
        in_w = src->shape[1];
        in_c = src->shape[2];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (dst->ndim == 2) {
        out_h = dst->shape[0];
        out_w = dst->shape[1];
        out_c = 1;
    } else if (dst->ndim == 3) {
        out_h = dst->shape[0];
        out_w = dst->shape[1];
        out_c = dst->shape[2];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (in_c != out_c || in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* in = (const float*)src->data;
    float* out = (float*)dst->data;

    int src_sy = (src->ndim == 2) ? (int)src->strides[0] : (int)src->strides[0];
    int src_sx = (src->ndim == 2) ? 1 : (int)src->strides[1];
    int dst_sy = (dst->ndim == 2) ? (int)dst->strides[0] : (int)dst->strides[0];
    int dst_sx = (dst->ndim == 2) ? 1 : (int)dst->strides[1];

    float ax = (float)in_w / (float)out_w;
    float ay = (float)in_h / (float)out_h;

    for (size_t k = 0; k < out_c; ++k) {
        for (size_t j = 0; j < out_h; ++j) {
            float y = ((float)j + 0.5f) * ay - 0.5f;
            int iy = (int)roundf(y);

            for (size_t i = 0; i < out_w; ++i) {
                float x = ((float)i + 0.5f) * ax - 0.5f;
                int ix = (int)roundf(x);

                float val = geom_get_f32_pixel_hwc(in, ix, iy, (int)k, (int)in_w, (int)in_h,
                                                   (int)in_c, src_sy, src_sx);
                if (dst->ndim == 2) {
                    out[j * dst_sy + i * dst_sx] = val;
                } else {
                    out[j * dst_sy + i * dst_sx + k] = val;
                }
            }
        }
    }
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_resize_bilinear(const RCRE_Tensor* src, RCRE_Tensor* dst) {
    if (!src || !src->data || !dst || !dst->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || dst->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t in_h, in_w, in_c;
    size_t out_h, out_w, out_c;

    if (src->ndim == 2) {
        in_h = src->shape[0];
        in_w = src->shape[1];
        in_c = 1;
    } else if (src->ndim == 3) {
        in_h = src->shape[0];
        in_w = src->shape[1];
        in_c = src->shape[2];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (dst->ndim == 2) {
        out_h = dst->shape[0];
        out_w = dst->shape[1];
        out_c = 1;
    } else if (dst->ndim == 3) {
        out_h = dst->shape[0];
        out_w = dst->shape[1];
        out_c = dst->shape[2];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (in_c != out_c || in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* in = (const float*)src->data;
    float* out = (float*)dst->data;

    int src_sy = (src->ndim == 2) ? (int)src->strides[0] : (int)src->strides[0];
    int src_sx = (src->ndim == 2) ? 1 : (int)src->strides[1];
    int dst_sy = (dst->ndim == 2) ? (int)dst->strides[0] : (int)dst->strides[0];
    int dst_sx = (dst->ndim == 2) ? 1 : (int)dst->strides[1];

    float ax = (float)in_w / (float)out_w;
    float ay = (float)in_h / (float)out_h;

    for (size_t k = 0; k < out_c; ++k) {
        for (size_t j = 0; j < out_h; ++j) {
            float y = ((float)j + 0.5f) * ay - 0.5f;
            int y1 = (int)floorf(y);
            int y2 = y1 + 1;
            float d3 = y - (float)y1;
            float d4 = 1.0f - d3;

            for (size_t i = 0; i < out_w; ++i) {
                float x = ((float)i + 0.5f) * ax - 0.5f;
                int x1 = (int)floorf(x);
                int x2 = x1 + 1;
                float d1 = x - (float)x1;
                float d2 = 1.0f - d1;

                float v00 = geom_get_f32_pixel_hwc(in, x1, y1, (int)k, (int)in_w, (int)in_h,
                                                   (int)in_c, src_sy, src_sx);
                float v10 = geom_get_f32_pixel_hwc(in, x2, y1, (int)k, (int)in_w, (int)in_h,
                                                   (int)in_c, src_sy, src_sx);
                float v01 = geom_get_f32_pixel_hwc(in, x1, y2, (int)k, (int)in_w, (int)in_h,
                                                   (int)in_c, src_sy, src_sx);
                float v11 = geom_get_f32_pixel_hwc(in, x2, y2, (int)k, (int)in_w, (int)in_h,
                                                   (int)in_c, src_sy, src_sx);

                float val = d2 * d4 * v00 + d1 * d4 * v10 + d2 * d3 * v01 + d1 * d3 * v11;

                if (dst->ndim == 2) {
                    out[j * dst_sy + i * dst_sx] = val;
                } else {
                    out[j * dst_sy + i * dst_sx + k] = val;
                }
            }
        }
    }
    return RCRE_OK;
}
