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

void rcre_neon_gemv_f32(size_t M, size_t K, float alpha,
                        const float* A, size_t lda,
                        const float* x, float beta, float* y) {
    size_t i = 0;
    for (; i + 3 < M; i += 4) {
        const float* r0 = &A[(i + 0) * lda];
        const float* r1 = &A[(i + 1) * lda];
        const float* r2 = &A[(i + 2) * lda];
        const float* r3 = &A[(i + 3) * lda];
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        size_t k = 0;
        for (; k + 3 < K; k += 4) {
            float32x4_t xv = vld1q_f32(&x[k]);
            acc0 = vfmaq_f32(acc0, vld1q_f32(&r0[k]), xv);
            acc1 = vfmaq_f32(acc1, vld1q_f32(&r1[k]), xv);
            acc2 = vfmaq_f32(acc2, vld1q_f32(&r2[k]), xv);
            acc3 = vfmaq_f32(acc3, vld1q_f32(&r3[k]), xv);
        }
        float s0 = vaddvq_f32(acc0), s1 = vaddvq_f32(acc1), s2 = vaddvq_f32(acc2), s3 = vaddvq_f32(acc3);
        for (; k < K; ++k) {
            float xv = x[k];
            s0 += r0[k] * xv;
            s1 += r1[k] * xv;
            s2 += r2[k] * xv;
            s3 += r3[k] * xv;
        }
        y[i + 0] = (beta == 0.0f ? 0.0f : y[i + 0] * beta) + alpha * s0;
        y[i + 1] = (beta == 0.0f ? 0.0f : y[i + 1] * beta) + alpha * s1;
        y[i + 2] = (beta == 0.0f ? 0.0f : y[i + 2] * beta) + alpha * s2;
        y[i + 3] = (beta == 0.0f ? 0.0f : y[i + 3] * beta) + alpha * s3;
    }
    for (; i < M; ++i) {
        float acc = rcre_neon_dot_f32(&A[i * lda], x, K);
        y[i] = (beta == 0.0f ? 0.0f : y[i] * beta) + alpha * acc;
    }
}

int32_t rcre_neon_dot_i8(const int8_t* x, const int8_t* y, size_t n) {
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;
#if defined(__ARM_FEATURE_DOTPROD)
    for (; i + 15 < n; i += 16) {
        acc = vdotq_s32(acc, vld1q_s8(&x[i]), vld1q_s8(&y[i]));
    }
#else
    for (; i + 15 < n; i += 16) {
        int8x16_t vx = vld1q_s8(&x[i]);
        int8x16_t vy = vld1q_s8(&y[i]);
        int16x8_t p_lo = vmull_s8(vget_low_s8(vx), vget_low_s8(vy));
        int16x8_t p_hi = vmull_s8(vget_high_s8(vx), vget_high_s8(vy));
        acc = vpadalq_s16(acc, p_lo);
        acc = vpadalq_s16(acc, p_hi);
    }
#endif
    int32_t sum = vaddvq_s32(acc);
    for (; i < n; ++i) {
        sum += (int32_t)x[i] * (int32_t)y[i];
    }
    return sum;
}

void rcre_neon_gemm_i8(size_t M, size_t N, size_t K,
                       const int8_t* A, size_t lda,
                       const int8_t* B, size_t ldb,
                       int32_t* C, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * ldc + j] = 0;
        }
    }
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            int16_t a_val = (int16_t)A[i * lda + k];
            int16x8_t va = vdupq_n_s16(a_val);
            size_t j = 0;
            for (; j + 7 < N; j += 8) {
                int16x8_t vb = vmovl_s8(vld1_s8(&B[k * ldb + j]));
                int32x4_t c_lo = vld1q_s32(&C[i * ldc + j]);
                int32x4_t c_hi = vld1q_s32(&C[i * ldc + j + 4]);
                c_lo = vmlal_s16(c_lo, vget_low_s16(va), vget_low_s16(vb));
                c_hi = vmlal_s16(c_hi, vget_high_s16(va), vget_high_s16(vb));
                vst1q_s32(&C[i * ldc + j], c_lo);
                vst1q_s32(&C[i * ldc + j + 4], c_hi);
            }
            for (; j < N; ++j) {
                C[i * ldc + j] += (int32_t)a_val * (int32_t)B[k * ldb + j];
            }
        }
    }
}

/* H0 (2.1.0): F64 cells — NEON 128-bit double-precision (2-wide)           */
/* =========================================================================
 * aarch64 NEON's 128-bit double vector is float64x2_t (2 doubles per
 * register — unlike the 4-wide F32 case). Element-wise ops (dot/axpy/
 * scale/sum) use the q- intrinsics directly; GEMM uses a 4x2 micro-kernel
 * (4 rows x 2 cols per k-step) with the same 64/64/64 blocking as the F32
 * driver. Determinism: k-outer accumulation order matches the scalar
 * baseline; cross-ISA parity is tolerance-gated (docs/hpc.md).
 * ========================================================================= */

/* alpha scales the A load (F32-driver convention); the C part in the
 * accumulator keeps its beta*C0 value untouched. */
static void neon_gemm64_microkernel_4x2(size_t K, double alpha,
                                        const double* A, size_t lda,
                                        const double* B, size_t ldb,
                                        double* C, size_t ldc) {
    float64x2_t acc[4];
    for (size_t i = 0; i < 4; ++i) acc[i] = vld1q_f64(&C[i * ldc]);
    for (size_t k = 0; k < K; ++k) {
        float64x2_t b = vld1q_f64(&B[k * ldb]);      /* B[k, 0..1] */
        for (size_t i = 0; i < 4; ++i)
            acc[i] = vfmaq_n_f64(acc[i], b, A[i * lda + k] * alpha);
    }
    for (size_t i = 0; i < 4; ++i) vst1q_f64(&C[i * ldc], acc[i]);
}

void rcre_neon_gemm_f64(size_t M, size_t N, size_t K, double alpha,
                        const double* A, size_t lda, const double* B, size_t ldb,
                        double beta, double* C, size_t ldc) {
    if (beta != 1.0) {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                C[i * ldc + j] = (beta == 0.0) ? 0.0 : C[i * ldc + j] * beta;
    }
    if (alpha == 0.0 || M == 0 || N == 0 || K == 0) return;

    const size_t MB = 64, NB = 64, KB = 64;
    for (size_t ii = 0; ii < M; ii += MB) {
        size_t i_end = (ii + MB < M) ? ii + MB : M;
        for (size_t jj = 0; jj < N; jj += NB) {
            size_t j_end = (jj + NB < N) ? jj + NB : N;
            for (size_t kk = 0; kk < K; kk += KB) {
                size_t k_end = (kk + KB < K) ? kk + KB : K;
                size_t i = ii;
                for (; i + 3 < i_end; i += 4) {
                    size_t j = jj;
                    for (; j + 1 < j_end; j += 2)
                        neon_gemm64_microkernel_4x2(k_end - kk, alpha,
                                                    &A[i * lda + kk], lda,
                                                    &B[kk * ldb + j], ldb,
                                                    &C[i * ldc + j], ldc);
                    for (; j < j_end; ++j)
                        for (size_t r = 0; r < 4; ++r) {
                            double acc = 0.0;
                            for (size_t k = kk; k < k_end; ++k)
                                acc += (A[(i + r) * lda + k] * alpha) * B[k * ldb + j];
                            C[(i + r) * ldc + j] += acc;
                        }
                }
                for (; i < i_end; ++i)
                    for (size_t j = jj; j < j_end; ++j) {
                        double acc = 0.0;
                        for (size_t k = kk; k < k_end; ++k)
                            acc += (A[i * lda + k] * alpha) * B[k * ldb + j];
                        C[i * ldc + j] += acc;
                    }
            }
        }
    }
}

double rcre_neon_dot_f64(const double* x, const double* y, size_t n) {
    float64x2_t acc = vdupq_n_f64(0.0);
    size_t i = 0;
    for (; i + 2 <= n; i += 2)
        acc = vfmaq_f64(acc, vld1q_f64(&x[i]), vld1q_f64(&y[i]));
    double s = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
    for (; i < n; ++i) s += x[i] * y[i];
    return s;
}

void rcre_neon_axpy_f64(size_t n, double alpha, const double* x, double* y) {
    float64x2_t va = vdupq_n_f64(alpha);
    size_t i = 0;
    for (; i + 2 <= n; i += 2)
        vst1q_f64(&y[i], vmlaq_f64(vld1q_f64(&y[i]), va, vld1q_f64(&x[i])));
    for (; i < n; ++i) y[i] += alpha * x[i];
}

void rcre_neon_gemv_f64(size_t M, size_t N, double alpha,
                        const double* A, size_t lda, const double* x,
                        double beta, double* y) {
    for (size_t i = 0; i < M; ++i) {
        float64x2_t acc = vdupq_n_f64(0.0);
        const double* row = A + i * lda;
        size_t j = 0;
        for (; j + 2 <= N; j += 2)
            acc = vfmaq_f64(acc, vld1q_f64(&row[j]), vld1q_f64(&x[j]));
        double s = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for (; j < N; ++j) s += row[j] * x[j];
        y[i] = (beta == 0.0) ? alpha * s : beta * y[i] + alpha * s;
    }
}

void rcre_neon_scale_f64(size_t n, double alpha, double* x) {
    float64x2_t va = vdupq_n_f64(alpha);
    size_t i = 0;
    for (; i + 2 <= n; i += 2)
        vst1q_f64(&x[i], vmulq_f64(vld1q_f64(&x[i]), va));
    for (; i < n; ++i) x[i] *= alpha;
}

double rcre_neon_sum_f64(const double* x, size_t n) {
    float64x2_t acc = vdupq_n_f64(0.0);
    size_t i = 0;
    for (; i + 2 <= n; i += 2)
        acc = vaddq_f64(acc, vld1q_f64(&x[i]));
    double s = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
    for (; i < n; ++i) s += x[i];
    return s;
}

#endif /* __ARM_NEON || __aarch64__ */
