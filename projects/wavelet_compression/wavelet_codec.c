/**
 * @file wavelet_codec.c
 * @brief Implementation of RCRE Reversible Integer Wavelet Image Codec (LeGall 5/3)
 */

#include "wavelet_codec.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

/* ========================================================================= */
/* 1D LeGall 5/3 Lifting Scheme                                              */
/* ========================================================================= */

static void dwt_1d_forward(const int32_t* src, int32_t* dst, size_t n, int32_t* temp) {
    if (n < 2) {
        if (n == 1) dst[0] = src[0];
        return;
    }

    size_t num_evens = (n + 1) / 2;
    size_t num_odds  = n / 2;

    /* Copy input into temporary buffer */
    memcpy(temp, src, n * sizeof(int32_t));

    /* 1. Predict Step (High-frequency odd details d[k]) */
    int32_t* d = dst + num_evens;
    for (size_t k = 0; k < num_odds; ++k) {
        int32_t x_left = temp[2 * k];
        int32_t x_right = (2 * k + 2 < n) ? temp[2 * k + 2] : temp[2 * k];
        int32_t predict = (x_left + x_right) >> 1;
        d[k] = temp[2 * k + 1] - predict;
    }

    /* 2. Update Step (Low-frequency even approximations s[k]) */
    int32_t* s = dst;
    for (size_t k = 0; k < num_evens; ++k) {
        int32_t d_left = (k > 0) ? d[k - 1] : d[0];
        int32_t d_right = (k < num_odds) ? d[k] : ((num_odds > 0) ? d[num_odds - 1] : 0);
        int32_t update = (d_left + d_right + 2) >> 2;
        s[k] = temp[2 * k] + update;
    }
}

static void dwt_1d_inverse(const int32_t* src, int32_t* dst, size_t n, int32_t* temp) {
    if (n < 2) {
        if (n == 1) dst[0] = src[0];
        return;
    }

    size_t num_evens = (n + 1) / 2;
    size_t num_odds  = n / 2;

    const int32_t* s = src;
    const int32_t* d = src + num_evens;

    /* 1. Invert Update Step: Recover even samples */
    for (size_t k = 0; k < num_evens; ++k) {
        int32_t d_left = (k > 0) ? d[k - 1] : d[0];
        int32_t d_right = (k < num_odds) ? d[k] : ((num_odds > 0) ? d[num_odds - 1] : 0);
        int32_t update = (d_left + d_right + 2) >> 2;
        temp[2 * k] = s[k] - update;
    }

    /* 2. Invert Predict Step: Recover odd samples */
    for (size_t k = 0; k < num_odds; ++k) {
        int32_t x_left = temp[2 * k];
        int32_t x_right = (2 * k + 2 < n) ? temp[2 * k + 2] : temp[2 * k];
        int32_t predict = (x_left + x_right) >> 1;
        temp[2 * k + 1] = d[k] + predict;
    }

    memcpy(dst, temp, n * sizeof(int32_t));
}

/* ========================================================================= */
/* 2D Multi-Resolution DWT                                                   */
/* ========================================================================= */

int rcre_wavelet_forward_2d(
    int32_t* buffer,
    size_t width,
    size_t height,
    size_t levels,
    RCRE_Arena* arena
) {
    if (!buffer || width == 0 || height == 0 || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t max_dim = (width > height) ? width : height;
    RCRE_ArenaTemp checkpoint = rcre_arena_temp_begin(arena);
    int32_t* line_src  = (int32_t*)rcre_arena_alloc(arena, max_dim * sizeof(int32_t), 64);
    int32_t* line_dst  = (int32_t*)rcre_arena_alloc(arena, max_dim * sizeof(int32_t), 64);
    int32_t* line_temp = (int32_t*)rcre_arena_alloc(arena, max_dim * sizeof(int32_t), 64);

    if (!line_src || !line_dst || !line_temp) {
        rcre_arena_temp_end(checkpoint);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    size_t cur_w = width;
    size_t cur_h = height;

    for (size_t l = 0; l < levels && cur_w >= 2 && cur_h >= 2; ++l) {
        /* Transform rows */
        for (size_t r = 0; r < cur_h; ++r) {
            int32_t* row_ptr = buffer + r * width;
            dwt_1d_forward(row_ptr, line_dst, cur_w, line_temp);
            memcpy(row_ptr, line_dst, cur_w * sizeof(int32_t));
        }

        /* Transform columns */
        for (size_t c = 0; c < cur_w; ++c) {
            for (size_t r = 0; r < cur_h; ++r) {
                line_src[r] = buffer[r * width + c];
            }
            dwt_1d_forward(line_src, line_dst, cur_h, line_temp);
            for (size_t r = 0; r < cur_h; ++r) {
                buffer[r * width + c] = line_dst[r];
            }
        }

        cur_w = (cur_w + 1) / 2;
        cur_h = (cur_h + 1) / 2;
    }

    rcre_arena_temp_end(checkpoint);
    return RCRE_OK;
}

int rcre_wavelet_inverse_2d(
    int32_t* buffer,
    size_t width,
    size_t height,
    size_t levels,
    RCRE_Arena* arena
) {
    if (!buffer || width == 0 || height == 0 || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t max_dim = (width > height) ? width : height;
    RCRE_ArenaTemp checkpoint = rcre_arena_temp_begin(arena);
    int32_t* line_src  = (int32_t*)rcre_arena_alloc(arena, max_dim * sizeof(int32_t), 64);
    int32_t* line_dst  = (int32_t*)rcre_arena_alloc(arena, max_dim * sizeof(int32_t), 64);
    int32_t* line_temp = (int32_t*)rcre_arena_alloc(arena, max_dim * sizeof(int32_t), 64);

    if (!line_src || !line_dst || !line_temp) {
        rcre_arena_temp_end(checkpoint);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Compute resolution boundaries per level */
    size_t dims_w[16];
    size_t dims_h[16];
    size_t actual_levels = 0;

    size_t cur_w = width;
    size_t cur_h = height;
    for (size_t l = 0; l < levels && cur_w >= 2 && cur_h >= 2; ++l) {
        dims_w[l] = cur_w;
        dims_h[l] = cur_h;
        actual_levels++;
        cur_w = (cur_w + 1) / 2;
        cur_h = (cur_h + 1) / 2;
    }

    /* Invert from innermost level back to level 0 */
    for (int l = (int)actual_levels - 1; l >= 0; --l) {
        size_t w = dims_w[l];
        size_t h = dims_h[l];

        /* Invert columns first */
        for (size_t c = 0; c < w; ++c) {
            for (size_t r = 0; r < h; ++r) {
                line_src[r] = buffer[r * width + c];
            }
            dwt_1d_inverse(line_src, line_dst, h, line_temp);
            for (size_t r = 0; r < h; ++r) {
                buffer[r * width + c] = line_dst[r];
            }
        }

        /* Invert rows */
        for (size_t r = 0; r < h; ++r) {
            int32_t* row_ptr = buffer + r * width;
            dwt_1d_inverse(row_ptr, line_dst, w, line_temp);
            memcpy(row_ptr, line_dst, w * sizeof(int32_t));
        }
    }

    rcre_arena_temp_end(checkpoint);
    return RCRE_OK;
}

/* ========================================================================= */
/* Run-Length Byte-Stream Entropy Coding                                     */
/* ========================================================================= */

/*
 * Bitstream Encoding Rules:
 *  - 0x00: Escape marker followed by int16_t (big-magnitude coefficient)
 *  - 0x01 .. 0x7E: Single small positive value (+1 .. +126)
 *  - 0x81 .. 0xFE: Single small negative value (-1 .. -126)
 *  - 0x7F: Zero-run marker followed by uint16_t count of zeros
 *  - 0xFF: End-of-subband marker
 */

static size_t compress_channel_coefficients(
    const int32_t* coeffs,
    size_t count,
    uint8_t* out_bytes,
    size_t max_out
) {
    size_t out_idx = 0;
    size_t i = 0;

    while (i < count && out_idx + 4 < max_out) {
        int32_t val = coeffs[i];

        if (val == 0) {
            /* Count consecutive zeros */
            size_t zero_run = 0;
            while (i < count && coeffs[i] == 0 && zero_run < 65535) {
                zero_run++;
                i++;
            }
            out_bytes[out_idx++] = 0x7F; /* Zero run marker */
            out_bytes[out_idx++] = (uint8_t)(zero_run & 0xFF);
            out_bytes[out_idx++] = (uint8_t)((zero_run >> 8) & 0xFF);
        } else {
            if (val >= 1 && val <= 126) {
                out_bytes[out_idx++] = (uint8_t)val;
            } else if (val >= -126 && val <= -1) {
                out_bytes[out_idx++] = (uint8_t)(0x80 | (-val));
            } else {
                /* Escape big value */
                out_bytes[out_idx++] = 0x00;
                int16_t clamped_val = (val > 32767) ? 32767 : ((val < -32768) ? -32768 : (int16_t)val);
                out_bytes[out_idx++] = (uint8_t)(clamped_val & 0xFF);
                out_bytes[out_idx++] = (uint8_t)((clamped_val >> 8) & 0xFF);
            }
            i++;
        }
    }

    return out_idx;
}

static size_t decompress_channel_coefficients(
    const uint8_t* in_bytes,
    size_t in_size,
    int32_t* out_coeffs,
    size_t expected_count
) {
    size_t in_idx = 0;
    size_t out_idx = 0;

    while (in_idx < in_size && out_idx < expected_count) {
        uint8_t tag = in_bytes[in_idx++];

        if (tag == 0x7F) {
            if (in_idx + 2 > in_size) break;
            uint16_t run = (uint16_t)in_bytes[in_idx] | ((uint16_t)in_bytes[in_idx + 1] << 8);
            in_idx += 2;
            for (uint16_t z = 0; z < run && out_idx < expected_count; ++z) {
                out_coeffs[out_idx++] = 0;
            }
        } else if (tag == 0x00) {
            if (in_idx + 2 > in_size) break;
            int16_t val = (int16_t)((uint16_t)in_bytes[in_idx] | ((uint16_t)in_bytes[in_idx + 1] << 8));
            in_idx += 2;
            out_coeffs[out_idx++] = (int32_t)val;
        } else if (tag < 0x7F) {
            out_coeffs[out_idx++] = (int32_t)tag;
        } else {
            out_coeffs[out_idx++] = -(int32_t)(tag & 0x7F);
        }
    }

    while (out_idx < expected_count) {
        out_coeffs[out_idx++] = 0;
    }

    return in_idx;
}

/* ========================================================================= */
/* End-to-End Image Encoding & Decoding Implementation                      */
/* ========================================================================= */

uint8_t* rcre_wavelet_encode(
    const RCRE_Tensor* image,
    size_t levels,
    uint16_t quant_step,
    RCRE_Arena* arena,
    size_t* out_bytes
) {
    if (!image || !image->data || !arena || !out_bytes) {
        return nullptr;
    }

    size_t H = image->shape[0];
    size_t W = image->shape[1];
    size_t C = (image->ndim >= 3) ? image->shape[2] : 1;

    if (quant_step == 0) quant_step = 1;
    if (levels == 0) levels = 3;

    /* Max output buffer size estimate */
    size_t max_compressed = sizeof(RCRE_WaveletHeader) + (H * W * C * 2);
    uint8_t* compressed = (uint8_t*)rcre_arena_alloc(arena, max_compressed, 64);
    if (!compressed) return nullptr;

    RCRE_WaveletHeader* hdr = (RCRE_WaveletHeader*)compressed;
    memcpy(hdr->magic, RCRE_RDWT_MAGIC, 4);
    hdr->version = RCRE_RDWT_VERSION;
    hdr->width = (uint16_t)W;
    hdr->height = (uint16_t)H;
    hdr->channels = (uint8_t)C;
    hdr->levels = (uint8_t)levels;
    hdr->quant_step = quant_step;
    memset(hdr->reserved, 0, sizeof(hdr->reserved));

    uint8_t* write_ptr = compressed + sizeof(RCRE_WaveletHeader);
    size_t remaining_bytes = max_compressed - sizeof(RCRE_WaveletHeader);

    /* Scratch arena for channel transforms */
    RCRE_ArenaTemp checkpoint = rcre_arena_temp_begin(arena);
    int32_t* ch_buffer = (int32_t*)rcre_arena_alloc(arena, H * W * sizeof(int32_t), 64);
    if (!ch_buffer) {
        rcre_arena_temp_end(checkpoint);
        return nullptr;
    }

    for (size_t ch = 0; ch < C; ++ch) {
        /* 1. Ingest pixel channel into signed 32-bit integer array */
        if (image->dtype == RCRE_DTYPE_U8) {
            const uint8_t* u8_data = (const uint8_t*)image->data;
            for (size_t r = 0; r < H; ++r) {
                for (size_t c = 0; c < W; ++c) {
                    size_t src_idx = (r * W + c) * C + ch;
                    ch_buffer[r * W + c] = (int32_t)u8_data[src_idx];
                }
            }
        } else {
            const float* f32_data = (const float*)image->data;
            for (size_t r = 0; r < H; ++r) {
                for (size_t c = 0; c < W; ++c) {
                    size_t src_idx = (r * W + c) * C + ch;
                    ch_buffer[r * W + c] = (int32_t)(f32_data[src_idx] * 255.0f);
                }
            }
        }

        /* 2. Execute Forward 2D LeGall 5/3 Integer Wavelet Transform */
        int rc = rcre_wavelet_forward_2d(ch_buffer, W, H, levels, arena);
        if (rc != RCRE_OK) {
            rcre_arena_temp_end(checkpoint);
            return nullptr;
        }

        /* 3. Apply Dead-Zone Quantization (if lossy) */
        if (quant_step > 1) {
            size_t ll_w = W;
            size_t ll_h = H;
            for (size_t l = 0; l < levels && ll_w >= 2 && ll_h >= 2; ++l) {
                ll_w = (ll_w + 1) / 2;
                ll_h = (ll_h + 1) / 2;
            }

            for (size_t r = 0; r < H; ++r) {
                for (size_t c = 0; c < W; ++c) {
                    /* Preserve low-frequency LL subband at full precision */
                    if (r < ll_h && c < ll_w) continue;

                    int32_t val = ch_buffer[r * W + c];
                    if (val > 0) {
                        ch_buffer[r * W + c] = val / quant_step;
                    } else if (val < 0) {
                        ch_buffer[r * W + c] = -((-val) / quant_step);
                    }
                }
            }
        }

        /* 4. Entropy Compress Channel */
        size_t written = compress_channel_coefficients(ch_buffer, H * W, write_ptr, remaining_bytes);
        write_ptr += written;
        remaining_bytes -= written;
    }

    rcre_arena_temp_end(checkpoint);

    size_t total_size = (size_t)(write_ptr - compressed);
    hdr->payload_bytes = (uint32_t)(total_size - sizeof(RCRE_WaveletHeader));
    *out_bytes = total_size;

    return compressed;
}

int rcre_wavelet_decode(
    const uint8_t* bitstream,
    size_t bitstream_size,
    RCRE_Arena* arena,
    RCRE_Tensor* out_image
) {
    if (!bitstream || bitstream_size < sizeof(RCRE_WaveletHeader) || !arena || !out_image) {
        return RCRE_ERR_INVALID_ARG;
    }

    const RCRE_WaveletHeader* hdr = (const RCRE_WaveletHeader*)bitstream;
    if (memcmp(hdr->magic, RCRE_RDWT_MAGIC, 4) != 0 || hdr->version != RCRE_RDWT_VERSION) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t W = hdr->width;
    size_t H = hdr->height;
    size_t C = hdr->channels;
    size_t levels = hdr->levels;
    uint16_t quant_step = hdr->quant_step;

    /* Allocate output tensor if needed */
    if (!out_image->data) {
        size_t shape[3] = {H, W, C};
        *out_image = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
        if (!out_image->data) return RCRE_ERR_OUT_OF_MEMORY;
    }

    uint8_t* dst_u8 = (uint8_t*)out_image->data;
    const uint8_t* read_ptr = bitstream + sizeof(RCRE_WaveletHeader);
    size_t remaining_bytes = bitstream_size - sizeof(RCRE_WaveletHeader);

    RCRE_ArenaTemp checkpoint = rcre_arena_temp_begin(arena);
    int32_t* ch_buffer = (int32_t*)rcre_arena_alloc(arena, H * W * sizeof(int32_t), 64);
    if (!ch_buffer) {
        rcre_arena_temp_end(checkpoint);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    for (size_t ch = 0; ch < C; ++ch) {
        /* 1. Decompress entropy run-length stream */
        size_t read_bytes = decompress_channel_coefficients(read_ptr, remaining_bytes, ch_buffer, H * W);
        read_ptr += read_bytes;
        remaining_bytes -= read_bytes;

        /* 2. Inverse Dead-Zone Quantization */
        if (quant_step > 1) {
            size_t ll_w = W;
            size_t ll_h = H;
            for (size_t l = 0; l < levels && ll_w >= 2 && ll_h >= 2; ++l) {
                ll_w = (ll_w + 1) / 2;
                ll_h = (ll_h + 1) / 2;
            }

            for (size_t r = 0; r < H; ++r) {
                for (size_t c = 0; c < W; ++c) {
                    if (r < ll_h && c < ll_w) continue;
                    ch_buffer[r * W + c] = ch_buffer[r * W + c] * quant_step;
                }
            }
        }

        /* 3. Invert 2D LeGall 5/3 Integer Wavelet Transform */
        int rc = rcre_wavelet_inverse_2d(ch_buffer, W, H, levels, arena);
        if (rc != RCRE_OK) {
            rcre_arena_temp_end(checkpoint);
            return rc;
        }

        /* 4. Write clamped pixels back to U8 output tensor */
        for (size_t r = 0; r < H; ++r) {
            for (size_t c = 0; c < W; ++c) {
                int32_t val = ch_buffer[r * W + c];
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                dst_u8[(r * W + c) * C + ch] = (uint8_t)val;
            }
        }
    }

    rcre_arena_temp_end(checkpoint);
    return RCRE_OK;
}

float rcre_wavelet_compute_psnr(
    const RCRE_Tensor* original,
    const RCRE_Tensor* reconstructed,
    float* out_mse
) {
    if (!original || !reconstructed || original->numel != reconstructed->numel) {
        if (out_mse) *out_mse = 9999.0f;
        return 0.0f;
    }

    size_t n = original->numel;
    const uint8_t* p_orig = (const uint8_t*)original->data;
    const uint8_t* p_recon = (const uint8_t*)reconstructed->data;

    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = (double)p_orig[i] - (double)p_recon[i];
        sum_sq += diff * diff;
    }

    float mse = (float)(sum_sq / (double)n);
    if (out_mse) *out_mse = mse;

    if (mse <= 1e-9f) {
        return 99.0f; /* Identical (Lossless) */
    }

    return 10.0f * log10f((255.0f * 255.0f) / mse);
}
