/**
 * @file kernels_wasm.c
 * @brief WebAssembly 128-bit SIMD Micro-Kernels
 * 
 * Satisfies HAL-REQ-03.
 * Targets modern browsers with WASM SIMD enabled.
 */

#include "hal_internal.h"

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>

float rcre_wasm_dot_f32(const float* x, const float* y, size_t n) {
    v128_t sum0 = wasm_f32x4_splat(0.0f);
    v128_t sum1 = wasm_f32x4_splat(0.0f);

    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        v128_t x0 = wasm_v128_load(&x[i + 0]);
        v128_t y0 = wasm_v128_load(&y[i + 0]);
        sum0 = wasm_f32x4_add(sum0, wasm_f32x4_mul(x0, y0));

        v128_t x1 = wasm_v128_load(&x[i + 4]);
        v128_t y1 = wasm_v128_load(&y[i + 4]);
        sum1 = wasm_f32x4_add(sum1, wasm_f32x4_mul(x1, y1));
    }

    v128_t sum = wasm_f32x4_add(sum0, sum1);
    float total = wasm_f32x4_extract_lane(sum, 0) +
                  wasm_f32x4_extract_lane(sum, 1) +
                  wasm_f32x4_extract_lane(sum, 2) +
                  wasm_f32x4_extract_lane(sum, 3);

    for (; i < n; ++i) {
        total += x[i] * y[i];
    }
    return total;
}

void rcre_wasm_axpy_f32(size_t n, float alpha, const float* x, float* y) {
    v128_t a_vec = wasm_f32x4_splat(alpha);

    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        v128_t x0 = wasm_v128_load(&x[i + 0]);
        v128_t y0 = wasm_v128_load(&y[i + 0]);
        wasm_v128_store(&y[i + 0], wasm_f32x4_add(y0, wasm_f32x4_mul(a_vec, x0)));

        v128_t x1 = wasm_v128_load(&x[i + 4]);
        v128_t y1 = wasm_v128_load(&y[i + 4]);
        wasm_v128_store(&y[i + 4], wasm_f32x4_add(y1, wasm_f32x4_mul(a_vec, x1)));
    }

    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

/* 4x8 register-blocked GEMM micro-kernel on WebAssembly 128-bit SIMD */
static void wasm_gemm_microkernel_4x8(size_t K, float alpha,
                                      const float* A, size_t lda,
                                      const float* B, size_t ldb,
                                      float beta, float* C, size_t ldc) {
    v128_t c00 = wasm_f32x4_splat(0.0f);
    v128_t c01 = wasm_f32x4_splat(0.0f);
    v128_t c10 = wasm_f32x4_splat(0.0f);
    v128_t c11 = wasm_f32x4_splat(0.0f);
    v128_t c20 = wasm_f32x4_splat(0.0f);
    v128_t c21 = wasm_f32x4_splat(0.0f);
    v128_t c30 = wasm_f32x4_splat(0.0f);
    v128_t c31 = wasm_f32x4_splat(0.0f);

    for (size_t k = 0; k < K; ++k) {
        v128_t b0 = wasm_v128_load(&B[k * ldb + 0]);
        v128_t b1 = wasm_v128_load(&B[k * ldb + 4]);

        v128_t a0 = wasm_f32x4_splat(A[0 * lda + k]);
        c00 = wasm_f32x4_add(c00, wasm_f32x4_mul(b0, a0));
        c01 = wasm_f32x4_add(c01, wasm_f32x4_mul(b1, a0));

        v128_t a1 = wasm_f32x4_splat(A[1 * lda + k]);
        c10 = wasm_f32x4_add(c10, wasm_f32x4_mul(b0, a1));
        c11 = wasm_f32x4_add(c11, wasm_f32x4_mul(b1, a1));

        v128_t a2 = wasm_f32x4_splat(A[2 * lda + k]);
        c20 = wasm_f32x4_add(c20, wasm_f32x4_mul(b0, a2));
        c21 = wasm_f32x4_add(c21, wasm_f32x4_mul(b1, a2));

        v128_t a3 = wasm_f32x4_splat(A[3 * lda + k]);
        c30 = wasm_f32x4_add(c30, wasm_f32x4_mul(b0, a3));
        c31 = wasm_f32x4_add(c31, wasm_f32x4_mul(b1, a3));
    }

    v128_t a_vec = wasm_f32x4_splat(alpha);
    v128_t b_vec = wasm_f32x4_splat(beta);

    #define STORE_ROW_WASM(row, acc0, acc1) do { \
        v128_t old0 = (beta == 0.0f) ? wasm_f32x4_splat(0.0f) : wasm_v128_load(&C[(row) * ldc + 0]); \
        v128_t old1 = (beta == 0.0f) ? wasm_f32x4_splat(0.0f) : wasm_v128_load(&C[(row) * ldc + 4]); \
        v128_t out0 = wasm_f32x4_add(wasm_f32x4_mul(old0, b_vec), wasm_f32x4_mul(acc0, a_vec)); \
        v128_t out1 = wasm_f32x4_add(wasm_f32x4_mul(old1, b_vec), wasm_f32x4_mul(acc1, a_vec)); \
        wasm_v128_store(&C[(row) * ldc + 0], out0); \
        wasm_v128_store(&C[(row) * ldc + 4], out1); \
    } while(0)

    STORE_ROW_WASM(0, c00, c01);
    STORE_ROW_WASM(1, c10, c11);
    STORE_ROW_WASM(2, c20, c21);
    STORE_ROW_WASM(3, c30, c31);
    #undef STORE_ROW_WASM
}

void rcre_wasm_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
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
                        wasm_gemm_microkernel_4x8(sub_K, alpha,
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

void rcre_wasm_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels) {
    const v128_t r_coeff = wasm_i32x4_splat(9798);
    const v128_t g_coeff = wasm_i32x4_splat(19235);
    const v128_t b_coeff = wasm_i32x4_splat(3735);
    const v128_t bias    = wasm_i32x4_splat(16384);

    size_t i = 0;
    for (; i + 7 < num_pixels; i += 8) {
        v128_t r32_0 = wasm_i32x4_make(rgb[(i + 0) * 3 + 0], rgb[(i + 1) * 3 + 0], rgb[(i + 2) * 3 + 0], rgb[(i + 3) * 3 + 0]);
        v128_t g32_0 = wasm_i32x4_make(rgb[(i + 0) * 3 + 1], rgb[(i + 1) * 3 + 1], rgb[(i + 2) * 3 + 1], rgb[(i + 3) * 3 + 1]);
        v128_t b32_0 = wasm_i32x4_make(rgb[(i + 0) * 3 + 2], rgb[(i + 1) * 3 + 2], rgb[(i + 2) * 3 + 2], rgb[(i + 3) * 3 + 2]);

        v128_t r32_1 = wasm_i32x4_make(rgb[(i + 4) * 3 + 0], rgb[(i + 5) * 3 + 0], rgb[(i + 6) * 3 + 0], rgb[(i + 7) * 3 + 0]);
        v128_t g32_1 = wasm_i32x4_make(rgb[(i + 4) * 3 + 1], rgb[(i + 5) * 3 + 1], rgb[(i + 6) * 3 + 1], rgb[(i + 7) * 3 + 1]);
        v128_t b32_1 = wasm_i32x4_make(rgb[(i + 4) * 3 + 2], rgb[(i + 5) * 3 + 2], rgb[(i + 6) * 3 + 2], rgb[(i + 7) * 3 + 2]);

        v128_t acc0 = bias;
        acc0 = wasm_i32x4_add(acc0, wasm_i32x4_mul(r32_0, r_coeff));
        acc0 = wasm_i32x4_add(acc0, wasm_i32x4_mul(g32_0, g_coeff));
        acc0 = wasm_i32x4_add(acc0, wasm_i32x4_mul(b32_0, b_coeff));
        v128_t gray32_0 = wasm_u32x4_shr(acc0, 15);

        v128_t acc1 = bias;
        acc1 = wasm_i32x4_add(acc1, wasm_i32x4_mul(r32_1, r_coeff));
        acc1 = wasm_i32x4_add(acc1, wasm_i32x4_mul(g32_1, g_coeff));
        acc1 = wasm_i32x4_add(acc1, wasm_i32x4_mul(b32_1, b_coeff));
        v128_t gray32_1 = wasm_u32x4_shr(acc1, 15);

        v128_t gray16 = wasm_i16x8_narrow_i32x4(gray32_0, gray32_1);
        v128_t gray8  = wasm_u8x16_narrow_i16x8(gray16, gray16);

        uint64_t res8 = (uint64_t)(uint32_t)wasm_i32x4_extract_lane(gray8, 0) |
                        ((uint64_t)(uint32_t)wasm_i32x4_extract_lane(gray8, 1) << 32);
        __builtin_memcpy(&gray[i], &res8, 8);
    }

    /* Scalar remainder */
    for (; i < num_pixels; ++i) {
        uint32_t r = rgb[i * 3 + 0];
        uint32_t g = rgb[i * 3 + 1];
        uint32_t b = rgb[i * 3 + 2];
        gray[i] = (uint8_t)((9798 * r + 19235 * g + 3735 * b + 16384) >> 15);
    }
}

#endif /* __wasm_simd128__ */
