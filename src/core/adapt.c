/**
 * @file adapt.c
 * @brief Input Data Recognition & Explicit Adaptation Pipeline
 * 
 * Satisfies ADP-REQ-01, ADP-REQ-02, and ADP-REQ-03.
 * Fuses aspect-ratio letterboxing, bilinear interpolation, layout transposition
 * (HWC <-> CHW), float conversion, and channel-wise normalization into a single pass.
 */

#include "rcre.h"
#include <string.h>
#include <math.h>

static inline float clampf(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static inline int clampi(int val, int min_val, int max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static inline float read_src_element(const RCRE_Tensor* src, size_t y, size_t x, size_t c, 
                                     size_t stride_y, size_t stride_x, size_t stride_c) {
    size_t elem_offset = y * stride_y + x * stride_x + c * stride_c;
    if (src->dtype == RCRE_DTYPE_U8) {
        return (float)((const uint8_t*)src->data)[elem_offset];
    } else if (src->dtype == RCRE_DTYPE_F32) {
        return ((const float*)src->data)[elem_offset];
    }
    return 0.0f;
}

RCRE_NODISCARD int rcre_tensor_adapt(const RCRE_Tensor* src, 
                                     const RCRE_PreprocessSpec* spec, 
                                     RCRE_Arena* arena, 
                                     RCRE_Tensor* dst) {
    if (!src || !src->data || !spec || !dst) {
        return RCRE_ERR_INVALID_ARG;
    }

    /* Extract source dimensions */
    size_t in_h = 0, in_w = 0, in_c = 1;
    size_t stride_y = 0, stride_x = 0, stride_c = 0;

    if (src->ndim == 2) {
        in_h = src->shape[0];
        in_w = src->shape[1];
        in_c = 1;
        stride_y = (size_t)src->strides[0];
        stride_x = (size_t)src->strides[1];
        stride_c = 0;
    } else if (src->ndim == 3) {
        if (src->layout == RCRE_LAYOUT_CHW) {
            in_c = src->shape[0];
            in_h = src->shape[1];
            in_w = src->shape[2];
            stride_c = (size_t)src->strides[0];
            stride_y = (size_t)src->strides[1];
            stride_x = (size_t)src->strides[2];
        } else {
            /* Default HWC */
            in_h = src->shape[0];
            in_w = src->shape[1];
            in_c = src->shape[2];
            stride_y = (size_t)src->strides[0];
            stride_x = (size_t)src->strides[1];
            stride_c = (size_t)src->strides[2];
        }
    } else if (src->ndim == 4) {
        /* Assume batch size = 1 for adaptation input */
        if (src->layout == RCRE_LAYOUT_CHW) {
            in_c = src->shape[1];
            in_h = src->shape[2];
            in_w = src->shape[3];
            stride_c = (size_t)src->strides[1];
            stride_y = (size_t)src->strides[2];
            stride_x = (size_t)src->strides[3];
        } else {
            in_h = src->shape[1];
            in_w = src->shape[2];
            in_c = src->shape[3];
            stride_y = (size_t)src->strides[1];
            stride_x = (size_t)src->strides[2];
            stride_c = (size_t)src->strides[3];
        }
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (in_h == 0 || in_w == 0 || in_c == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    /* Extract target spatial dimensions */
    size_t target_h = 0, target_w = 0, target_c = in_c;
    uint8_t target_ndim = 0;
    size_t target_shape[4] = {0};

    if (spec->target_layout == RCRE_LAYOUT_CHW) {
        target_ndim = 4;
        target_shape[0] = 1;
        target_shape[1] = in_c;
        target_shape[2] = spec->target_shape[2] ? spec->target_shape[2] : spec->target_shape[0];
        target_shape[3] = spec->target_shape[3] ? spec->target_shape[3] : spec->target_shape[1];
        target_h = target_shape[2];
        target_w = target_shape[3];
    } else {
        /* HWC */
        target_ndim = 3;
        target_shape[0] = spec->target_shape[0];
        target_shape[1] = spec->target_shape[1];
        target_shape[2] = in_c;
        target_h = target_shape[0];
        target_w = target_shape[1];
    }

    if (target_h == 0 || target_w == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    /* If destination not pre-allocated, allocate in arena */
    if (!dst->data) {
        if (!arena) return RCRE_ERR_OUT_OF_MEMORY;
        *dst = rcre_tensor_create(target_shape, target_ndim, spec->target_dtype, spec->target_layout, arena);
        if (!dst->data) {
            return RCRE_ERR_OUT_OF_MEMORY;
        }
    }

    /* Compute letterboxing geometry */
    float scale = 1.0f;
    float scaled_w = (float)target_w;
    float scaled_h = (float)target_h;
    float pad_x = 0.0f;
    float pad_y = 0.0f;

    if (spec->preserve_aspect) {
        float r_w = (float)target_w / (float)in_w;
        float r_h = (float)target_h / (float)in_h;
        scale = (r_w < r_h) ? r_w : r_h;

        scaled_w = roundf((float)in_w * scale);
        scaled_h = roundf((float)in_h * scale);
        pad_x = floorf(((float)target_w - scaled_w) * 0.5f);
        pad_y = floorf(((float)target_h - scaled_h) * 0.5f);
    }

    float inv_scale_x = (float)in_w / scaled_w;
    float inv_scale_y = (float)in_h / scaled_h;

    float* dst_f32 = (spec->target_dtype == RCRE_DTYPE_F32) ? (float*)dst->data : NULL;
    uint8_t* dst_u8 = (spec->target_dtype == RCRE_DTYPE_U8) ? (uint8_t*)dst->data : NULL;

    /* Precompute normalized padding value per channel if normalizer requested */
    float pad_per_channel[4];
    for (size_t c = 0; c < target_c && c < 4; ++c) {
        float std_c = (spec->std[c] != 0.0f) ? spec->std[c] : 1.0f;
        pad_per_channel[c] = (spec->target_dtype == RCRE_DTYPE_F32) 
            ? ((spec->pad_value - spec->mean[c]) / std_c) 
            : spec->pad_value;
    }

    /* Single-pass fused letterbox + resize + normalize + layout loop */
    for (size_t y_out = 0; y_out < target_h; ++y_out) {
        bool y_in_bounds = (y_out >= (size_t)pad_y) && (y_out < (size_t)(pad_y + scaled_h));
        float y_src = y_in_bounds ? (((float)y_out - pad_y + 0.5f) * inv_scale_y - 0.5f) : 0.0f;
        int y0 = clampi((int)floorf(y_src), 0, (int)in_h - 1);
        int y1 = clampi(y0 + 1, 0, (int)in_h - 1);
        float wy1 = y_src - floorf(y_src);
        float wy0 = 1.0f - wy1;

        for (size_t x_out = 0; x_out < target_w; ++x_out) {
            bool x_in_bounds = (x_out >= (size_t)pad_x) && (x_out < (size_t)(pad_x + scaled_w));
            bool is_inside = y_in_bounds && x_in_bounds;

            float x_src = is_inside ? (((float)x_out - pad_x + 0.5f) * inv_scale_x - 0.5f) : 0.0f;
            int x0 = clampi((int)floorf(x_src), 0, (int)in_w - 1);
            int x1 = clampi(x0 + 1, 0, (int)in_w - 1);
            float wx1 = x_src - floorf(x_src);
            float wx0 = 1.0f - wx1;

            for (size_t c = 0; c < target_c; ++c) {
                float val = 0.0f;
                if (!is_inside) {
                    val = (c < 4) ? pad_per_channel[c] : spec->pad_value;
                } else {
                    /* Bilinear sampling */
                    float p00 = read_src_element(src, (size_t)y0, (size_t)x0, c, stride_y, stride_x, stride_c);
                    float p01 = read_src_element(src, (size_t)y0, (size_t)x1, c, stride_y, stride_x, stride_c);
                    float p10 = read_src_element(src, (size_t)y1, (size_t)x0, c, stride_y, stride_x, stride_c);
                    float p11 = read_src_element(src, (size_t)y1, (size_t)x1, c, stride_y, stride_x, stride_c);

                    float top = p00 * wx0 + p01 * wx1;
                    float bot = p10 * wx0 + p11 * wx1;
                    float interpolated = top * wy0 + bot * wy1;

                    /* Fused normalization: (x - mean) / std */
                    float mean_c = (c < 4) ? spec->mean[c] : 0.0f;
                    float std_c = (c < 4 && spec->std[c] != 0.0f) ? spec->std[c] : 1.0f;

                    if (spec->target_dtype == RCRE_DTYPE_F32) {
                        val = (interpolated - mean_c) / std_c;
                    } else {
                        val = clampf(interpolated, 0.0f, 255.0f);
                    }
                }

                /* Compute destination memory offset based on target layout */
                size_t dst_offset = 0;
                if (spec->target_layout == RCRE_LAYOUT_CHW) {
                    /* [1, C, H, W] */
                    dst_offset = (c * target_h + y_out) * target_w + x_out;
                } else {
                    /* [H, W, C] */
                    dst_offset = (y_out * target_w + x_out) * target_c + c;
                }

                if (dst_f32) {
                    dst_f32[dst_offset] = val;
                } else if (dst_u8) {
                    dst_u8[dst_offset] = (uint8_t)clampf(val, 0.0f, 255.0f);
                }
            }
        }
    }

    return RCRE_OK;
}
