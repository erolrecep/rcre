/**
 * @file fft.c
 * @brief Fast Fourier Transform & 2D Spectral Image Processing
 * 
 * Satisfies FFT-REQ-01 to FFT-REQ-04.
 */

#include "rcre.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Verify if an integer is a positive power of 2 */
static inline bool is_power_of_two(size_t n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

RCRE_NODISCARD int rcre_fft_1d(RCRE_Complex* data, size_t n, RCRE_Arena* arena) {
    (void)arena;
    if (!data || !is_power_of_two(n)) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (n <= 1) {
        return RCRE_OK;
    }

    /* Bit-reversal permutation */
    size_t j = 0;
    for (size_t i = 0; i < n - 1; ++i) {
        if (i < j) {
            RCRE_Complex temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
        size_t k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* Cooley-Tukey decimation-in-time butterfly stages */
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half = len >> 1;
        double angle = -2.0 * M_PI / (double)len;
        double w_step_re = cos(angle);
        double w_step_im = sin(angle);

        for (size_t i = 0; i < n; i += len) {
            double w_re = 1.0;
            double w_im = 0.0;

            for (size_t k = 0; k < half; ++k) {
                RCRE_Complex u = data[i + k];
                RCRE_Complex v_val = data[i + k + half];

                float v_re = (float)(v_val.re * w_re - v_val.im * w_im);
                float v_im = (float)(v_val.re * w_im + v_val.im * w_re);

                data[i + k].re = u.re + v_re;
                data[i + k].im = u.im + v_im;

                data[i + k + half].re = u.re - v_re;
                data[i + k + half].im = u.im - v_im;

                double next_w_re = w_re * w_step_re - w_im * w_step_im;
                double next_w_im = w_re * w_step_im + w_im * w_step_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_ifft_1d(RCRE_Complex* data, size_t n, RCRE_Arena* arena) {
    if (!data || !is_power_of_two(n)) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (n <= 1) {
        return RCRE_OK;
    }

    /* Conjugate input */
    for (size_t i = 0; i < n; ++i) {
        data[i].im = -data[i].im;
    }

    /* Forward transform */
    int status = rcre_fft_1d(data, n, arena);
    if (status != RCRE_OK) {
        return status;
    }

    /* Conjugate and scale by 1 / n */
    float inv_n = 1.0f / (float)n;
    for (size_t i = 0; i < n; ++i) {
        data[i].re *= inv_n;
        data[i].im = -data[i].im * inv_n;
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_fft_2d(const RCRE_Tensor* input,
                               RCRE_Tensor* output_re,
                               RCRE_Tensor* output_im,
                               RCRE_Arena* arena) {
    if (!input || !input->data || !output_re || !output_re->data ||
        !output_im || !output_im->data || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (input->ndim != 2 || output_re->ndim != 2 || output_im->ndim != 2) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t H = input->shape[0];
    size_t W = input->shape[1];

    if (!is_power_of_two(H) || !is_power_of_two(W)) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (output_re->shape[0] != H || output_re->shape[1] != W ||
        output_im->shape[0] != H || output_im->shape[1] != W) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* in_data = (const float*)input->data;
    float* out_re = (float*)output_re->data;
    float* out_im = (float*)output_im->data;

    /* Transient line buffers */
    RCRE_Complex* row_buf = (RCRE_Complex*)rcre_arena_alloc(arena, W * sizeof(RCRE_Complex), 64);
    RCRE_Complex* col_buf = (RCRE_Complex*)rcre_arena_alloc(arena, H * sizeof(RCRE_Complex), 64);

    if (!row_buf || !col_buf) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* 1. Transform rows */
    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            row_buf[w].re = in_data[h * W + w];
            row_buf[w].im = 0.0f;
        }

        int status = rcre_fft_1d(row_buf, W, arena);
        if (status != RCRE_OK) {
            return status;
        }

        for (size_t w = 0; w < W; ++w) {
            out_re[h * W + w] = row_buf[w].re;
            out_im[h * W + w] = row_buf[w].im;
        }
    }

    /* 2. Transform columns */
    for (size_t w = 0; w < W; ++w) {
        for (size_t h = 0; h < H; ++h) {
            col_buf[h].re = out_re[h * W + w];
            col_buf[h].im = out_im[h * W + w];
        }

        int status = rcre_fft_1d(col_buf, H, arena);
        if (status != RCRE_OK) {
            return status;
        }

        for (size_t h = 0; h < H; ++h) {
            out_re[h * W + w] = col_buf[h].re;
            out_im[h * W + w] = col_buf[h].im;
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_ifft_2d(const RCRE_Tensor* input_re,
                                const RCRE_Tensor* input_im,
                                RCRE_Tensor* output,
                                RCRE_Arena* arena) {
    if (!input_re || !input_re->data || !input_im || !input_im->data ||
        !output || !output->data || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (input_re->ndim != 2 || input_im->ndim != 2 || output->ndim != 2) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t H = input_re->shape[0];
    size_t W = input_re->shape[1];

    if (!is_power_of_two(H) || !is_power_of_two(W)) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (input_im->shape[0] != H || input_im->shape[1] != W ||
        output->shape[0] != H || output->shape[1] != W) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* in_re = (const float*)input_re->data;
    const float* in_im = (const float*)input_im->data;
    float* out_data = (float*)output->data;

    /* 2D working complex buffer */
    RCRE_Complex* grid = (RCRE_Complex*)rcre_arena_alloc(arena, H * W * sizeof(RCRE_Complex), 64);
    RCRE_Complex* row_buf = (RCRE_Complex*)rcre_arena_alloc(arena, W * sizeof(RCRE_Complex), 64);
    RCRE_Complex* col_buf = (RCRE_Complex*)rcre_arena_alloc(arena, H * sizeof(RCRE_Complex), 64);

    if (!grid || !row_buf || !col_buf) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < H * W; ++i) {
        grid[i].re = in_re[i];
        grid[i].im = in_im[i];
    }

    /* 1. Inverse transform columns */
    for (size_t w = 0; w < W; ++w) {
        for (size_t h = 0; h < H; ++h) {
            col_buf[h] = grid[h * W + w];
        }

        int status = rcre_ifft_1d(col_buf, H, arena);
        if (status != RCRE_OK) {
            return status;
        }

        for (size_t h = 0; h < H; ++h) {
            grid[h * W + w] = col_buf[h];
        }
    }

    /* 2. Inverse transform rows */
    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            row_buf[w] = grid[h * W + w];
        }

        int status = rcre_ifft_1d(row_buf, W, arena);
        if (status != RCRE_OK) {
            return status;
        }

        for (size_t w = 0; w < W; ++w) {
            out_data[h * W + w] = row_buf[w].re;
        }
    }

    return RCRE_OK;
}

static inline void swap_quadrants(float* data, size_t H, size_t W) {
    size_t half_H = H / 2;
    size_t half_W = W / 2;

    for (size_t h = 0; h < half_H; ++h) {
        for (size_t w = 0; w < half_W; ++w) {
            /* Top-left (h, w) <-> Bottom-right (h + half_H, w + half_W) */
            size_t idx_tl = h * W + w;
            size_t idx_br = (h + half_H) * W + (w + half_W);
            float tmp1 = data[idx_tl];
            data[idx_tl] = data[idx_br];
            data[idx_br] = tmp1;

            /* Top-right (h, w + half_W) <-> Bottom-left (h + half_H, w) */
            size_t idx_tr = h * W + (w + half_W);
            size_t idx_bl = (h + half_H) * W + w;
            float tmp2 = data[idx_tr];
            data[idx_tr] = data[idx_bl];
            data[idx_bl] = tmp2;
        }
    }
}

RCRE_NODISCARD int rcre_fftshift_2d(RCRE_Tensor* tensor_re, RCRE_Tensor* tensor_im) {
    if (!tensor_re || !tensor_re->data || tensor_re->ndim != 2) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t H = tensor_re->shape[0];
    size_t W = tensor_re->shape[1];

    if ((H % 2 != 0) || (W % 2 != 0)) {
        return RCRE_ERR_INVALID_ARG;
    }

    swap_quadrants((float*)tensor_re->data, H, W);

    if (tensor_im && tensor_im->data) {
        if (tensor_im->shape[0] != H || tensor_im->shape[1] != W) {
            return RCRE_ERR_SHAPE_MISMATCH;
        }
        swap_quadrants((float*)tensor_im->data, H, W);
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_ifftshift_2d(RCRE_Tensor* tensor_re, RCRE_Tensor* tensor_im) {
    /* For even dimensions, ifftshift is identical to fftshift */
    return rcre_fftshift_2d(tensor_re, tensor_im);
}

RCRE_NODISCARD int rcre_fft_filter_2d(RCRE_Tensor* spectrum_re,
                                      RCRE_Tensor* spectrum_im,
                                      float cutoff_radius,
                                      bool is_highpass,
                                      bool is_gaussian) {
    if (!spectrum_re || !spectrum_re->data || !spectrum_im || !spectrum_im->data || cutoff_radius <= 0.0f) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (spectrum_re->ndim != 2 || spectrum_im->ndim != 2) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t H = spectrum_re->shape[0];
    size_t W = spectrum_re->shape[1];

    if (spectrum_im->shape[0] != H || spectrum_im->shape[1] != W) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    float* re_data = (float*)spectrum_re->data;
    float* im_data = (float*)spectrum_im->data;

    float center_h = (float)H / 2.0f;
    float center_w = (float)W / 2.0f;
    float d0_sq = cutoff_radius * cutoff_radius;
    float two_d0_sq = 2.0f * d0_sq;

    for (size_t h = 0; h < H; ++h) {
        float dh = (float)h - center_h;
        float dh_sq = dh * dh;

        for (size_t w = 0; w < W; ++w) {
            float dw = (float)w - center_w;
            float dist_sq = dh_sq + dw * dw;

            float H_filter = 1.0f;

            if (is_gaussian) {
                float g = expf(-dist_sq / two_d0_sq);
                H_filter = is_highpass ? (1.0f - g) : g;
            } else {
                /* Ideal brickwall filter */
                bool inside = (dist_sq <= d0_sq);
                H_filter = is_highpass ? (inside ? 0.0f : 1.0f) : (inside ? 1.0f : 0.0f);
            }

            size_t idx = h * W + w;
            re_data[idx] *= H_filter;
            im_data[idx] *= H_filter;
        }
    }

    return RCRE_OK;
}
