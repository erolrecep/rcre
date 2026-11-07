/**
 * @file kernels_neon.c
 * @brief ARM NEON 128-bit Vector Micro-Kernels
 * 
 * Satisfies HAL-REQ-03.
 * Targets Apple Silicon (M-series) and ARMv8-A / Cortex processors.
 */

#include "hal_internal.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

float rcre_neon_dot_f32(const float* x, const float* y, size_t n) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        float32x4_t x0 = vld1q_f32(&x[i + 0]);
        float32x4_t y0 = vld1q_f32(&y[i + 0]);
        sum0 = vfmaq_f32(sum0, x0, y0);

        float32x4_t x1 = vld1q_f32(&x[i + 4]);
        float32x4_t y1 = vld1q_f32(&y[i + 4]);
        sum1 = vfmaq_f32(sum1, x1, y1);

        float32x4_t x2 = vld1q_f32(&x[i + 8]);
        float32x4_t y2 = vld1q_f32(&y[i + 8]);
        sum2 = vfmaq_f32(sum2, x2, y2);

        float32x4_t x3 = vld1q_f32(&x[i + 12]);
        float32x4_t y3 = vld1q_f32(&y[i + 12]);
        sum3 = vfmaq_f32(sum3, x3, y3);
    }

    /* Accumulate 4 vector sums */
    float32x4_t sum01 = vaddq_f32(sum0, sum1);
    float32x4_t sum23 = vaddq_f32(sum2, sum3);
    float32x4_t sum_all = vaddq_f32(sum01, sum23);

#if defined(__aarch64__)
    float total = vaddvq_f32(sum_all);
#else
    float32x2_t sum_half = vadd_f32(vget_low_f32(sum_all), vget_high_f32(sum_all));
    float total = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);
#endif

    /* Scalar remainder */
    for (; i < n; ++i) {
        total += x[i] * y[i];
    }

    return total;
}

void rcre_neon_axpy_f32(size_t n, float alpha, const float* x, float* y) {
    float32x4_t a_vec = vdupq_n_f32(alpha);

    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        float32x4_t x0 = vld1q_f32(&x[i + 0]);
        float32x4_t y0 = vld1q_f32(&y[i + 0]);
        vst1q_f32(&y[i + 0], vfmaq_f32(y0, a_vec, x0));

        float32x4_t x1 = vld1q_f32(&x[i + 4]);
        float32x4_t y1 = vld1q_f32(&y[i + 4]);
        vst1q_f32(&y[i + 4], vfmaq_f32(y1, a_vec, x1));

        float32x4_t x2 = vld1q_f32(&x[i + 8]);
        float32x4_t y2 = vld1q_f32(&y[i + 8]);
        vst1q_f32(&y[i + 8], vfmaq_f32(y2, a_vec, x2));

        float32x4_t x3 = vld1q_f32(&x[i + 12]);
        float32x4_t y3 = vld1q_f32(&y[i + 12]);
        vst1q_f32(&y[i + 12], vfmaq_f32(y3, a_vec, x3));
    }

    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

/* 4x8 register-blocked GEMM micro-kernel on NEON */
static void neon_gemm_microkernel_4x8(size_t K, float alpha,
                                      const float* A, size_t lda,
                                      const float* B, size_t ldb,
                                      float beta, float* C, size_t ldc) {
    float32x4_t c00 = vdupq_n_f32(0.0f);
    float32x4_t c01 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f);
    float32x4_t c11 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f);
    float32x4_t c21 = vdupq_n_f32(0.0f);
    float32x4_t c30 = vdupq_n_f32(0.0f);
    float32x4_t c31 = vdupq_n_f32(0.0f);

    for (size_t k = 0; k < K; ++k) {
        float32x4_t b0 = vld1q_f32(&B[k * ldb + 0]);
        float32x4_t b1 = vld1q_f32(&B[k * ldb + 4]);

        float32x4_t a0 = vdupq_n_f32(A[0 * lda + k]);
        c00 = vfmaq_f32(c00, b0, a0);
        c01 = vfmaq_f32(c01, b1, a0);

        float32x4_t a1 = vdupq_n_f32(A[1 * lda + k]);
        c10 = vfmaq_f32(c10, b0, a1);
        c11 = vfmaq_f32(c11, b1, a1);

        float32x4_t a2 = vdupq_n_f32(A[2 * lda + k]);
        c20 = vfmaq_f32(c20, b0, a2);
        c21 = vfmaq_f32(c21, b1, a2);

        float32x4_t a3 = vdupq_n_f32(A[3 * lda + k]);
        c30 = vfmaq_f32(c30, b0, a3);
        c31 = vfmaq_f32(c31, b1, a3);
    }

    float32x4_t a_vec = vdupq_n_f32(alpha);
    float32x4_t b_vec = vdupq_n_f32(beta);

    #define STORE_ROW(row, acc0, acc1) do { \
        float32x4_t old0 = (beta == 0.0f) ? vdupq_n_f32(0.0f) : vld1q_f32(&C[(row) * ldc + 0]); \
        float32x4_t old1 = (beta == 0.0f) ? vdupq_n_f32(0.0f) : vld1q_f32(&C[(row) * ldc + 4]); \
        float32x4_t out0 = vfmaq_f32(vmulq_f32(old0, b_vec), acc0, a_vec); \
        float32x4_t out1 = vfmaq_f32(vmulq_f32(old1, b_vec), acc1, a_vec); \
        vst1q_f32(&C[(row) * ldc + 0], out0); \
        vst1q_f32(&C[(row) * ldc + 4], out1); \
    } while(0)

    STORE_ROW(0, c00, c01);
    STORE_ROW(1, c10, c11);
    STORE_ROW(2, c20, c21);
    STORE_ROW(3, c30, c31);
    #undef STORE_ROW
}

void rcre_neon_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc) {
    /* Cache-blocked driver for 4x8 micro-kernel */
    const size_t MB = 64;
    const size_t NB = 64;
    const size_t KB = 64;

    /* Initialize C with beta scaling if needed */
    if (beta != 1.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] = (beta == 0.0f) ? 0.0f : C[i * ldc + j] * beta;
            }
        }
    }

    if (alpha == 0.0f) return;

    for (size_t ii = 0; ii < M; ii += MB) {
        size_t i_end = (ii + MB < M) ? ii + MB : M;

        for (size_t jj = 0; jj < N; jj += NB) {
            size_t j_end = (jj + NB < N) ? jj + NB : N;

            for (size_t kk = 0; kk < K; kk += KB) {
                size_t k_end = (kk + KB < K) ? kk + KB : K;
                size_t sub_K = k_end - kk;

                /* 4x8 tiled blocks */
                size_t i = ii;
                for (; i + 3 < i_end; i += 4) {
                    size_t j = jj;
                    for (; j + 7 < j_end; j += 8) {
                        neon_gemm_microkernel_4x8(sub_K, alpha,
                                                  &A[i * lda + kk], lda,
                                                  &B[kk * ldb + j], ldb,
                                                  1.0f, /* C already pre-scaled by beta */
                                                  &C[i * ldc + j], ldc);
                    }
                    /* Remainder columns in 4-row block */
                    for (; j < j_end; ++j) {
                        for (size_t r = 0; r < 4; ++r) {
                            float acc = 0.0f;
                            for (size_t k = kk; k < k_end; ++k) {
                                acc += A[(i + r) * lda + k] * B[k * ldb + j];
                            }
                            C[(i + r) * ldc + j] += alpha * acc;
                        }
                    }
                }

                /* Remainder rows */
                for (; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        float acc = 0.0f;
                        for (size_t k = kk; k < k_end; ++k) {
                            acc += A[i * lda + k] * B[k * ldb + j];
                        }
                        C[i * ldc + j] += alpha * acc;
                    }
                }
            }
        }
    }
}

void rcre_neon_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels) {
    const uint16x4_t r_coeff = vdup_n_u16(9798);
    const uint16x4_t g_coeff = vdup_n_u16(19235);
    const uint16x4_t b_coeff = vdup_n_u16(3735);
    const uint32x4_t bias    = vdupq_n_u32(16384);

    size_t i = 0;
    for (; i + 7 < num_pixels; i += 8) {
        /* De-interleave 8 RGB pixels (24 bytes) into 3 8-byte registers */
        uint8x8x3_t rgb_vec = vld3_u8(&rgb[i * 3]);

        uint16x8_t r16 = vmovl_u8(rgb_vec.val[0]);
        uint16x8_t g16 = vmovl_u8(rgb_vec.val[1]);
        uint16x8_t b16 = vmovl_u8(rgb_vec.val[2]);

        /* Lower 4 pixels */
        uint32x4_t acc_l = vmull_u16(vget_low_u16(r16), r_coeff);
        acc_l = vmlal_u16(acc_l, vget_low_u16(g16), g_coeff);
        acc_l = vmlal_u16(acc_l, vget_low_u16(b16), b_coeff);
        acc_l = vaddq_u32(acc_l, bias);
        uint16x4_t res_l16 = vshrn_n_u32(acc_l, 15);

        /* Upper 4 pixels */
        uint32x4_t acc_h = vmull_u16(vget_high_u16(r16), r_coeff);
        acc_h = vmlal_u16(acc_h, vget_high_u16(g16), g_coeff);
        acc_h = vmlal_u16(acc_h, vget_high_u16(b16), b_coeff);
        acc_h = vaddq_u32(acc_h, bias);
        uint16x4_t res_h16 = vshrn_n_u32(acc_h, 15);

        uint16x8_t res16 = vcombine_u16(res_l16, res_h16);
        uint8x8_t res8 = vmovn_u16(res16);
        vst1_u8(&gray[i], res8);
    }

    /* Scalar remainder */
    for (; i < num_pixels; ++i) {
        uint32_t r = rgb[i * 3 + 0];
        uint32_t g = rgb[i * 3 + 1];
        uint32_t b = rgb[i * 3 + 2];
        gray[i] = (uint8_t)((9798 * r + 19235 * g + 3735 * b + 16384) >> 15);
    }
}

static inline size_t neon_first_match_16(uint8x16_t cmp) {
    /* Pack 16 0xFF/0x00 bytes into 16 4-bit nibbles in a single 64-bit word */
    uint8x8_t narrowed = vshrn_n_u16(vreinterpretq_u16_u8(cmp), 4);
    uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(narrowed), 0);
    return (size_t)(__builtin_ctzll(mask) >> 2);
}

size_t rcre_neon_find_byte(const uint8_t* hay, size_t len, uint8_t needle) {
    if (!hay || len == 0) {
        return RCRE_NPOS;
    }

    const uint8x16_t vn = vdupq_n_u8(needle);
    size_t i = 0;

    /* 64-byte unrolled fast scan across 4 NEON registers */
    for (; i + 63 < len; i += 64) {
        uint8x16_t c0 = vceqq_u8(vld1q_u8(&hay[i +  0]), vn);
        uint8x16_t c1 = vceqq_u8(vld1q_u8(&hay[i + 16]), vn);
        uint8x16_t c2 = vceqq_u8(vld1q_u8(&hay[i + 32]), vn);
        uint8x16_t c3 = vceqq_u8(vld1q_u8(&hay[i + 48]), vn);

        uint8x16_t any = vorrq_u8(vorrq_u8(c0, c1), vorrq_u8(c2, c3));
#if defined(__aarch64__)
        if (vmaxvq_u8(any) != 0) {
            if (vmaxvq_u8(c0) != 0) return i +  0 + neon_first_match_16(c0);
            if (vmaxvq_u8(c1) != 0) return i + 16 + neon_first_match_16(c1);
            if (vmaxvq_u8(c2) != 0) return i + 32 + neon_first_match_16(c2);
            return i + 48 + neon_first_match_16(c3);
        }
#else
        uint64x2_t any64 = vreinterpretq_u64_u8(any);
        if ((vgetq_lane_u64(any64, 0) | vgetq_lane_u64(any64, 1)) != 0) {
            for (size_t k = 0; k < 64; ++k) {
                if (hay[i + k] == needle) return i + k;
            }
        }
#endif
    }

    for (; i + 15 < len; i += 16) {
        uint8x16_t c0 = vceqq_u8(vld1q_u8(&hay[i]), vn);
#if defined(__aarch64__)
        if (vmaxvq_u8(c0) != 0) {
            return i + neon_first_match_16(c0);
        }
#else
        uint64x2_t c64 = vreinterpretq_u64_u8(c0);
        if ((vgetq_lane_u64(c64, 0) | vgetq_lane_u64(c64, 1)) != 0) {
            for (size_t k = 0; k < 16; ++k) {
                if (hay[i + k] == needle) return i + k;
            }
        }
#endif
    }

    for (; i < len; ++i) {
        if (hay[i] == needle) {
            return i;
        }
    }
    return RCRE_NPOS;
}

size_t rcre_neon_count_byte(const uint8_t* hay, size_t len, uint8_t needle) {
    if (!hay || len == 0) {
        return 0;
    }

    const uint8x16_t vn = vdupq_n_u8(needle);
    size_t total = 0;
    size_t i = 0;

    /* Accumulate up to 255 iterations in uint8 lanes before widening reduction */
    while (i + 63 < len) {
        uint8x16_t acc0 = vdupq_n_u8(0);
        uint8x16_t acc1 = vdupq_n_u8(0);
        uint8x16_t acc2 = vdupq_n_u8(0);
        uint8x16_t acc3 = vdupq_n_u8(0);

        size_t block_iters = (len - i) / 64;
        if (block_iters > 255) {
            block_iters = 255;
        }
        for (size_t b = 0; b < block_iters; ++b, i += 64) {
            /* vceqq_u8 yields 0xFF (-1 in mod 256) on match; subtracting adds +1 */
            acc0 = vsubq_u8(acc0, vceqq_u8(vld1q_u8(&hay[i +  0]), vn));
            acc1 = vsubq_u8(acc1, vceqq_u8(vld1q_u8(&hay[i + 16]), vn));
            acc2 = vsubq_u8(acc2, vceqq_u8(vld1q_u8(&hay[i + 32]), vn));
            acc3 = vsubq_u8(acc3, vceqq_u8(vld1q_u8(&hay[i + 48]), vn));
        }

        uint16x8_t s16 = vpaddlq_u8(acc0);
        s16 = vpadalq_u8(s16, acc1);
        s16 = vpadalq_u8(s16, acc2);
        s16 = vpadalq_u8(s16, acc3);

        uint32x4_t s32 = vpaddlq_u16(s16);
        uint64x2_t s64 = vpaddlq_u32(s32);
#if defined(__aarch64__)
        total += (size_t)vaddvq_u64(s64);
#else
        total += (size_t)(vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1));
#endif
    }

    for (; i < len; ++i) {
        total += (size_t)(hay[i] == needle);
    }
    return total;
}

#endif /* __ARM_NEON || __aarch64__ */
