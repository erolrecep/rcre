/**
 * @file filter.c
 * @brief Spatial Convolutions & Image Filters (Gaussian Blur, Sobel)
 * 
 * Satisfies CV-REQ-03.
 * Implements O(2K) separable Gaussian blur and 3x3 Sobel gradient operators.
 */

#include "rcre.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline int reflect_101(int p, int len) {
    if (len <= 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) {
            p = -p;
        } else if (p >= len) {
            p = 2 * len - 2 - p;
        }
    }
    return p;
}

RCRE_NODISCARD int rcre_cv_gaussian_blur(const RCRE_Tensor* src, RCRE_Tensor* dst, 
                                         float sigma, RCRE_Arena* arena) {
    if (!src || !src->data || !dst || !dst->data || sigma <= 0.0f) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (src->ndim < 2 || dst->ndim < 2) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t height = src->shape[0];
    size_t width = src->shape[1];
    size_t channels = (src->ndim == 3) ? src->shape[2] : 1;

    if (dst->shape[0] != height || dst->shape[1] != width) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    /* Compute discrete 1D Gaussian kernel radius: radius = ceil(3 * sigma) */
    int radius = (int)ceilf(3.0f * sigma);
    if (radius < 1) radius = 1;
    if (radius > 15) radius = 15; /* Bound max kernel size for safety */
    int ksize = 2 * radius + 1;

    float kernel[31];
    float sum = 0.0f;
    float two_sigma_sq = 2.0f * sigma * sigma;

    for (int i = 0; i < ksize; ++i) {
        int d = i - radius;
        kernel[i] = expf(-((float)(d * d)) / two_sigma_sq);
        sum += kernel[i];
    }
    for (int i = 0; i < ksize; ++i) {
        kernel[i] /= sum;
    }

    /* Allocate intermediate float buffer for horizontal pass */
    size_t temp_elements = height * width * channels;
    float* temp = (float*)rcre_arena_alloc(arena, temp_elements * sizeof(float), 64);
    if (!temp) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Pass 1: Horizontal 1D Convolution (src -> temp) */
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            for (size_t c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int src_x = reflect_101((int)x + k, (int)width);
                    float val = 0.0f;
                    if (src->dtype == RCRE_DTYPE_U8) {
                        val = (float)((const uint8_t*)src->data)[y * src->strides[0] + (size_t)src_x * src->strides[1] + c];
                    } else if (src->dtype == RCRE_DTYPE_F32) {
                        val = ((const float*)src->data)[y * src->strides[0] + (size_t)src_x * src->strides[1] + c];
                    }
                    acc += val * kernel[k + radius];
                }
                temp[(y * width + x) * channels + c] = acc;
            }
        }
    }

    /* Pass 2: Vertical 1D Convolution (temp -> dst) */
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            for (size_t c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int src_y = reflect_101((int)y + k, (int)height);
                    acc += temp[((size_t)src_y * width + x) * channels + c] * kernel[k + radius];
                }

                size_t dst_idx = y * dst->strides[0] + x * dst->strides[1] + c;
                if (dst->dtype == RCRE_DTYPE_U8) {
                    float clamped = acc < 0.0f ? 0.0f : (acc > 255.0f ? 255.0f : acc);
                    ((uint8_t*)dst->data)[dst_idx] = (uint8_t)(clamped + 0.5f);
                } else if (dst->dtype == RCRE_DTYPE_F32) {
                    ((float*)dst->data)[dst_idx] = acc;
                }
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_sobel(const RCRE_Tensor* src, RCRE_Tensor* dst_dx, 
                                 RCRE_Tensor* dst_dy, RCRE_Arena* arena) {
    (void)arena;
    if (!src || !src->data || (!dst_dx && !dst_dy)) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t height = src->shape[0];
    size_t width = src->shape[1];
    size_t channels = (src->ndim == 3) ? src->shape[2] : 1;

    for (size_t y = 0; y < height; ++y) {
        int ym1 = reflect_101((int)y - 1, (int)height);
        int yp1 = reflect_101((int)y + 1, (int)height);
        int y0  = (int)y;

        for (size_t x = 0; x < width; ++x) {
            int xm1 = reflect_101((int)x - 1, (int)width);
            int xp1 = reflect_101((int)x + 1, (int)width);
            int x0  = (int)x;

            for (size_t c = 0; c < channels; ++c) {
                #define GET_PIXEL(r, col) \
                    ((src->dtype == RCRE_DTYPE_U8) \
                        ? (float)((const uint8_t*)src->data)[(r) * src->strides[0] + (col) * src->strides[1] + c] \
                        : ((const float*)src->data)[(r) * src->strides[0] + (col) * src->strides[1] + c])

                float p00 = GET_PIXEL((size_t)ym1, (size_t)xm1);
                float p01 = GET_PIXEL((size_t)ym1, (size_t)x0);
                float p02 = GET_PIXEL((size_t)ym1, (size_t)xp1);

                float p10 = GET_PIXEL((size_t)y0, (size_t)xm1);
                float p12 = GET_PIXEL((size_t)y0, (size_t)xp1);

                float p20 = GET_PIXEL((size_t)yp1, (size_t)xm1);
                float p21 = GET_PIXEL((size_t)yp1, (size_t)x0);
                float p22 = GET_PIXEL((size_t)yp1, (size_t)xp1);
                #undef GET_PIXEL

                /* Gx = [-1 0 1; -2 0 2; -1 0 1] */
                float gx = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);

                /* Gy = [-1 -2 -1; 0 0 0; 1 2 1] */
                float gy = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

                if (dst_dx && dst_dx->data) {
                    size_t idx = y * dst_dx->strides[0] + x * dst_dx->strides[1] + c;
                    if (dst_dx->dtype == RCRE_DTYPE_U8) {
                        float mag = fabsf(gx);
                        ((uint8_t*)dst_dx->data)[idx] = (uint8_t)(mag > 255.0f ? 255 : (uint8_t)mag);
                    } else if (dst_dx->dtype == RCRE_DTYPE_F32) {
                        ((float*)dst_dx->data)[idx] = gx;
                    }
                }
                if (dst_dy && dst_dy->data) {
                    size_t idx = y * dst_dy->strides[0] + x * dst_dy->strides[1] + c;
                    if (dst_dy->dtype == RCRE_DTYPE_U8) {
                        float mag = fabsf(gy);
                        ((uint8_t*)dst_dy->data)[idx] = (uint8_t)(mag > 255.0f ? 255 : (uint8_t)mag);
                    } else if (dst_dy->dtype == RCRE_DTYPE_F32) {
                        ((float*)dst_dy->data)[idx] = gy;
                    }
                }
            }
        }
    }

    return RCRE_OK;
}

static inline int filter_clamp_coord(int p, int max_val) {
    if (p < 0) return 0;
    if (p >= max_val) return max_val - 1;
    return p;
}

RCRE_NODISCARD int rcre_cv_convolve_2d(const RCRE_Tensor* src, const RCRE_Tensor* kernel,
                                      RCRE_Tensor* dst, bool preserve_channels, RCRE_Arena* arena) {
    (void)arena;
    if (!src || !src->data || !kernel || !kernel->data || !dst || !dst->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || kernel->dtype != RCRE_DTYPE_F32 || dst->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t in_h = src->shape[0];
    size_t in_w = src->shape[1];
    size_t in_c = (src->ndim == 3) ? src->shape[2] : 1;

    size_t kh = kernel->shape[0];
    size_t kw = kernel->shape[1];
    size_t kc = (kernel->ndim == 3) ? kernel->shape[2] : 1;

    if (kh == 0 || kw == 0 || (kh % 2 == 0) || (kw % 2 == 0)) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t out_h = dst->shape[0];
    size_t out_w = dst->shape[1];
    size_t out_c = (dst->ndim == 3) ? dst->shape[2] : 1;

    if (out_h != in_h || out_w != in_w) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (preserve_channels) {
        if (out_c != in_c) return RCRE_ERR_SHAPE_MISMATCH;
    } else {
        if (out_c != 1) return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* in = (const float*)src->data;
    const float* k_data = (const float*)kernel->data;
    float* out = (float*)dst->data;

    int r_y = (int)(kh / 2);
    int r_x = (int)(kw / 2);

    int src_sy = (src->ndim == 3) ? (int)src->strides[0] : (int)src->strides[0];
    int src_sx = (src->ndim == 3) ? (int)src->strides[1] : 1;
    int dst_sy = (dst->ndim == 3) ? (int)dst->strides[0] : (int)dst->strides[0];
    int dst_sx = (dst->ndim == 3) ? (int)dst->strides[1] : 1;

    if (preserve_channels) {
        for (size_t y = 0; y < in_h; ++y) {
            for (size_t x = 0; x < in_w; ++x) {
                for (size_t c = 0; c < in_c; ++c) {
                    size_t k_channel = (kc == in_c) ? c : 0;
                    float sum = 0.0f;

                    for (int dy = -r_y; dy <= r_y; ++dy) {
                        int sy_coord = filter_clamp_coord((int)y + dy, (int)in_h);
                        for (int dx = -r_x; dx <= r_x; ++dx) {
                            int sx_coord = filter_clamp_coord((int)x + dx, (int)in_w);

                            float pixel_val = in[sy_coord * src_sy + sx_coord * src_sx + c];
                            size_t k_idx = ((size_t)(dy + r_y) * kw + (size_t)(dx + r_x)) * kc + k_channel;
                            sum += pixel_val * k_data[k_idx];
                        }
                    }
                    if (dst->ndim == 3) {
                        out[y * dst_sy + x * dst_sx + c] = sum;
                    } else {
                        out[y * dst_sy + x * dst_sx] = sum;
                    }
                }
            }
        }
    } else {
        /* Merge / collapse across channels */
        for (size_t y = 0; y < in_h; ++y) {
            for (size_t x = 0; x < in_w; ++x) {
                float total_sum = 0.0f;
                for (size_t c = 0; c < in_c; ++c) {
                    size_t k_channel = (kc == in_c) ? c : 0;
                    for (int dy = -r_y; dy <= r_y; ++dy) {
                        int sy_coord = filter_clamp_coord((int)y + dy, (int)in_h);
                        for (int dx = -r_x; dx <= r_x; ++dx) {
                            int sx_coord = filter_clamp_coord((int)x + dx, (int)in_w);

                            float pixel_val = in[sy_coord * src_sy + sx_coord * src_sx + c];
                            size_t k_idx = ((size_t)(dy + r_y) * kw + (size_t)(dx + r_x)) * kc + k_channel;
                            total_sum += pixel_val * k_data[k_idx];
                        }
                    }
                }
                out[y * dst_sy + x * dst_sx] = total_sum;
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_filter_box(RCRE_Tensor* kernel, int size, RCRE_Arena* arena) {
    if (!kernel || size <= 0 || (size % 2 == 0) || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }
    size_t shape[3] = {(size_t)size, (size_t)size, 1};
    *kernel = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!kernel->data) return RCRE_ERR_OUT_OF_MEMORY;

    float val = 1.0f / (float)(size * size);
    float* data = (float*)kernel->data;
    size_t total = (size_t)(size * size);
    for (size_t i = 0; i < total; ++i) {
        data[i] = val;
    }
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_filter_sharpen(RCRE_Tensor* kernel, RCRE_Arena* arena) {
    if (!kernel || !arena) return RCRE_ERR_INVALID_ARG;
    size_t shape[3] = {3, 3, 1};
    *kernel = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!kernel->data) return RCRE_ERR_OUT_OF_MEMORY;

    float weights[9] = {
        0.0f, -1.0f,  0.0f,
       -1.0f,  5.0f, -1.0f,
        0.0f, -1.0f,  0.0f
    };
    memcpy(kernel->data, weights, sizeof(weights));
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_filter_emboss(RCRE_Tensor* kernel, RCRE_Arena* arena) {
    if (!kernel || !arena) return RCRE_ERR_INVALID_ARG;
    size_t shape[3] = {3, 3, 1};
    *kernel = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!kernel->data) return RCRE_ERR_OUT_OF_MEMORY;

    float weights[9] = {
       -2.0f, -1.0f, 0.0f,
       -1.0f,  1.0f, 1.0f,
        0.0f,  1.0f, 2.0f
    };
    memcpy(kernel->data, weights, sizeof(weights));
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_cv_filter_highpass(RCRE_Tensor* kernel, RCRE_Arena* arena) {
    if (!kernel || !arena) return RCRE_ERR_INVALID_ARG;
    size_t shape[3] = {3, 3, 1};
    *kernel = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    if (!kernel->data) return RCRE_ERR_OUT_OF_MEMORY;

    float weights[9] = {
        0.0f, -1.0f,  0.0f,
       -1.0f,  4.0f, -1.0f,
        0.0f, -1.0f,  0.0f
    };
    memcpy(kernel->data, weights, sizeof(weights));
    return RCRE_OK;
}

