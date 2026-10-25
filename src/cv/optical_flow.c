/**
 * @file optical_flow.c
 * @brief Integral Images and Lucas-Kanade Optical Flow
 *
 * Implements Summed-Area Tables (O(1) box queries), spatiotemporal derivative tensors,
 * and Lucas-Kanade velocity field estimation with flow vector visualization.
 */

#include "rcre.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TWOPI 6.283185307179586476925286766559005f

RCRE_NODISCARD int rcre_cv_integral_image(const RCRE_Tensor* src, RCRE_Tensor* dst_integral) {
    if (!src || !src->data || !dst_integral || !dst_integral->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || dst_integral->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t height = src->shape[0];
    size_t width = src->shape[1];
    size_t channels = (src->ndim == 3) ? src->shape[2] : 1;

    if (dst_integral->shape[0] != height || dst_integral->shape[1] != width) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* in = (const float*)src->data;
    float* out = (float*)dst_integral->data;

    int src_sy = (src->ndim == 3) ? (int)src->strides[0] : (int)src->strides[0];
    int src_sx = (src->ndim == 3) ? (int)src->strides[1] : 1;
    int dst_sy = (dst_integral->ndim == 3) ? (int)dst_integral->strides[0] : (int)dst_integral->strides[0];
    int dst_sx = (dst_integral->ndim == 3) ? (int)dst_integral->strides[1] : 1;

    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float val = in[y * src_sy + x * src_sx + c];
                float left = (x > 0) ? out[y * dst_sy + (x - 1) * dst_sx + c] : 0.0f;
                float top  = (y > 0) ? out[(y - 1) * dst_sy + x * dst_sx + c] : 0.0f;
                float diag = (x > 0 && y > 0) ? out[(y - 1) * dst_sy + (x - 1) * dst_sx + c] : 0.0f;

                out[y * dst_sy + x * dst_sx + c] = val + left + top - diag;
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_box_filter_integral(const RCRE_Tensor* integral, RCRE_Tensor* dst,
                                              int box_radius) {
    if (!integral || !integral->data || !dst || !dst->data || box_radius < 0) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (integral->dtype != RCRE_DTYPE_F32 || dst->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t height = integral->shape[0];
    size_t width = integral->shape[1];
    size_t channels = (integral->ndim == 3) ? integral->shape[2] : 1;

    if (dst->shape[0] != height || dst->shape[1] != width) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* in = (const float*)integral->data;
    float* out = (float*)dst->data;

    int int_sy = (integral->ndim == 3) ? (int)integral->strides[0] : (int)integral->strides[0];
    int int_sx = (integral->ndim == 3) ? (int)integral->strides[1] : 1;
    int dst_sy = (dst->ndim == 3) ? (int)dst->strides[0] : (int)dst->strides[0];
    int dst_sx = (dst->ndim == 3) ? (int)dst->strides[1] : 1;

    int r = box_radius;

    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                int x1 = (int)x - r;
                int x2 = (int)x + r;
                int y1 = (int)y - r;
                int y2 = (int)y + r;

                if (x1 < 0) x1 = 0;
                if (x2 >= (int)width) x2 = (int)width - 1;
                if (y1 < 0) y1 = 0;
                if (y2 >= (int)height) y2 = (int)height - 1;

                float a = in[y2 * int_sy + x2 * int_sx + c];
                float b = (x1 > 0) ? in[y2 * int_sy + (x1 - 1) * int_sx + c] : 0.0f;
                float d = (y1 > 0) ? in[(y1 - 1) * int_sy + x2 * int_sx + c] : 0.0f;
                float diag = (x1 > 0 && y1 > 0) ? in[(y1 - 1) * int_sy + (x1 - 1) * int_sx + c] : 0.0f;

                float sum = a - b - d + diag;
                float area = (float)((x2 - x1 + 1) * (y2 - y1 + 1));
                out[y * dst_sy + x * dst_sx + c] = sum / area;
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_optical_flow_lk(const RCRE_Tensor* curr, const RCRE_Tensor* prev,
                                          RCRE_Tensor* dst_velocity, int window_size,
                                          int stride, float max_v, RCRE_Arena* arena) {
    if (!curr || !curr->data || !prev || !prev->data || !dst_velocity || !dst_velocity->data || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (curr->dtype != RCRE_DTYPE_F32 || prev->dtype != RCRE_DTYPE_F32 || dst_velocity->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t height = curr->shape[0];
    size_t width = curr->shape[1];

    if (prev->shape[0] != height || prev->shape[1] != width) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    int eff_stride = (stride < 1) ? 1 : stride;
    size_t v_h = height / (size_t)eff_stride;
    size_t v_w = width / (size_t)eff_stride;

    if (dst_velocity->shape[0] != v_h || dst_velocity->shape[1] != v_w || dst_velocity->shape[2] < 2) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    RCRE_ArenaTemp scratch = rcre_arena_temp_begin(arena);

    /* Grayscale conversion */
    size_t shape_2d[2] = {height, width};
    RCRE_Tensor curr_gray = rcre_tensor_create(shape_2d, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor prev_gray = rcre_tensor_create(shape_2d, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!curr_gray.data || !prev_gray.data) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    if (curr->ndim == 3 && curr->shape[2] >= 3) {
        (void)rcre_cv_rgb_to_gray(curr, &curr_gray);
    } else {
        memcpy(curr_gray.data, curr->data, height * width * sizeof(float));
    }

    if (prev->ndim == 3 && prev->shape[2] >= 3) {
        (void)rcre_cv_rgb_to_gray(prev, &prev_gray);
    } else {
        memcpy(prev_gray.data, prev->data, height * width * sizeof(float));
    }

    /* Spatial gradients Ix, Iy on curr */
    RCRE_Tensor gx = rcre_tensor_create(shape_2d, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor gy = rcre_tensor_create(shape_2d, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!gx.data || !gy.data) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    (void)rcre_cv_sobel(&curr_gray, &gx, &gy, arena);

    /* Construct 5-channel spatiotemporal tensor: [Ixx, Iyy, Ixy, Ixt, Iyt] */
    size_t shape_5d[3] = {height, width, 5};
    RCRE_Tensor S_raw = rcre_tensor_create(shape_5d, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!S_raw.data) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    const float* cg = (const float*)curr_gray.data;
    const float* pg = (const float*)prev_gray.data;
    const float* gx_d = (const float*)gx.data;
    const float* gy_d = (const float*)gy.data;
    float* s_data = (float*)S_raw.data;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t idx = y * width + x;
            float ix = gx_d[idx];
            float iy = gy_d[idx];
            float it = cg[idx] - pg[idx];

            size_t s_idx = idx * 5;
            s_data[s_idx + 0] = ix * ix; /* Ixx */
            s_data[s_idx + 1] = iy * iy; /* Iyy */
            s_data[s_idx + 2] = ix * iy; /* Ixy */
            s_data[s_idx + 3] = ix * it; /* Ixt */
            s_data[s_idx + 4] = iy * it; /* Iyt */
        }
    }

    /* Box filter spatiotemporal tensor via integral image */
    RCRE_Tensor S_integ = rcre_tensor_create(shape_5d, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    RCRE_Tensor S_smooth = rcre_tensor_create(shape_5d, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!S_integ.data || !S_smooth.data) {
        rcre_arena_temp_end(scratch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    (void)rcre_cv_integral_image(&S_raw, &S_integ);
    int r = (window_size <= 0) ? 7 : (window_size / 2);
    (void)rcre_cv_box_filter_integral(&S_integ, &S_smooth, r);

    /* Invert 2x2 structure tensor at each strided sample */
    const float* sm_d = (const float*)S_smooth.data;
    float* v_out = (float*)dst_velocity->data;
    int v_sy = (int)dst_velocity->strides[0];
    int v_sx = (int)dst_velocity->strides[1];
    float clamp_limit = (max_v <= 0.0f) ? 10.0f : max_v;

    for (size_t vj = 0; vj < v_h; ++vj) {
        size_t y = vj * (size_t)eff_stride + (size_t)(eff_stride / 2);
        if (y >= height) y = height - 1;

        for (size_t vi = 0; vi < v_w; ++vi) {
            size_t x = vi * (size_t)eff_stride + (size_t)(eff_stride / 2);
            if (x >= width) x = width - 1;

            size_t s_idx = (y * width + x) * 5;
            float Ixx = sm_d[s_idx + 0];
            float Iyy = sm_d[s_idx + 1];
            float Ixy = sm_d[s_idx + 2];
            float Ixt = sm_d[s_idx + 3];
            float Iyt = sm_d[s_idx + 4];

            float det = Ixx * Iyy - Ixy * Ixy;
            float vx = 0.0f;
            float vy = 0.0f;

            if (fabsf(det) > 1e-7f) {
                vx = -(Iyy * Ixt - Ixy * Iyt) / det;
                vy = -(Ixx * Iyt - Ixy * Ixt) / det;
            }

            if (vx < -clamp_limit) vx = -clamp_limit;
            if (vx > clamp_limit)  vx = clamp_limit;
            if (vy < -clamp_limit) vy = -clamp_limit;
            if (vy > clamp_limit)  vy = clamp_limit;

            v_out[vj * v_sy + vi * v_sx + 0] = vx;
            v_out[vj * v_sy + vi * v_sx + 1] = vy;
            if (dst_velocity->shape[2] >= 3) {
                v_out[vj * v_sy + vi * v_sx + 2] = 0.0f;
            }
        }
    }

    rcre_arena_temp_end(scratch);
    return RCRE_OK;
}

void rcre_cv_draw_flow(RCRE_Tensor* canvas, const RCRE_Tensor* velocity, int stride, float scale) {
    if (!canvas || !canvas->data || !velocity || !velocity->data) return;
    if (canvas->dtype != RCRE_DTYPE_F32 || canvas->ndim < 2) return;

    size_t c_h = canvas->shape[0];
    size_t c_w = canvas->shape[1];
    size_t c_c = (canvas->ndim == 3) ? canvas->shape[2] : 1;

    size_t v_h = velocity->shape[0];
    size_t v_w = velocity->shape[1];

    int eff_stride = (stride <= 0) ? (int)(c_w / v_w) : stride;
    float eff_scale = (scale <= 0.0f) ? 5.0f : scale;

    const float* v_data = (const float*)velocity->data;
    float* out = (float*)canvas->data;
    int v_sy = (int)velocity->strides[0];
    int v_sx = (int)velocity->strides[1];
    int c_sy = (int)canvas->strides[0];
    int c_sx = (int)canvas->strides[1];

    for (size_t j = 0; j < v_h; ++j) {
        int cy = (int)(j * (size_t)eff_stride + (size_t)(eff_stride / 2));
        if (cy >= (int)c_h) continue;

        for (size_t i = 0; i < v_w; ++i) {
            int cx = (int)(i * (size_t)eff_stride + (size_t)(eff_stride / 2));
            if (cx >= (int)c_w) continue;

            float vx = v_data[j * v_sy + i * v_sx + 0] * eff_scale;
            float vy = v_data[j * v_sy + i * v_sx + 1] * eff_scale;

            float d = sqrtf(vx * vx + vy * vy);
            if (d < 0.5f) continue;

            /* Draw line segment using DDA */
            int steps = (int)ceilf(d);
            for (int s = 0; s <= steps; ++s) {
                float t = (float)s / (float)steps;
                int px = (int)roundf((float)cx + vx * t);
                int py = (int)roundf((float)cy + vy * t);

                if (px >= 0 && px < (int)c_w && py >= 0 && py < (int)c_h) {
                    if (c_c >= 3) {
                        out[py * c_sy + px * c_sx + 0] = 0.0f; /* Red */
                        out[py * c_sy + px * c_sx + 1] = 1.0f; /* Green */
                        out[py * c_sy + px * c_sx + 2] = 0.0f; /* Blue */
                    } else {
                        out[py * c_sy + px * c_sx] = 1.0f;
                    }
                }
            }
        }
    }
}
