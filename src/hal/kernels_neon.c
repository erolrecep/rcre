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

#endif /* __ARM_NEON || __aarch64__ */
