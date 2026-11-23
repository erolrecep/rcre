/**
 * @file hal_dispatch.c
 * @brief Dynamic Function Pointer Table & Hardware Acceleration Dispatcher
 *
 * Satisfies HAL-REQ-04.
 */

#include "hal_internal.h"
#include <stdatomic.h>

/* Default static initialization with pure ISO C23 scalar routines */
RCRE_HalDispatch rcre_hal = {
    .backend_name    = "ISO C23 Scalar Reference",
    .cpu_features    = RCRE_CPU_FEAT_SCALAR,
    .gemm_f32        = rcre_scalar_gemm_f32,
    .gemv_f32        = rcre_scalar_gemv_f32,
    .dot_f32         = rcre_scalar_dot_f32,
    .dot_i8          = rcre_scalar_dot_i8,
    .gemm_i8         = rcre_scalar_gemm_i8,
    .gemm_bf16       = rcre_scalar_gemm_bf16,
    .axpy_f32        = rcre_scalar_axpy_f32,
    .rgb_to_gray_u8  = rcre_scalar_rgb_to_gray_u8,
    .find_byte       = rcre_scalar_find_byte,
    .count_byte      = rcre_scalar_count_byte,
    /* H0 (2.1.0): F64 substrate — every cell has a scalar default, so
     * no algorithm ever branches on ISA; backends override below. */
    .gemm_f64        = rcre_scalar_gemm_f64,
    .dot_f64         = rcre_scalar_dot_f64,
    .axpy_f64        = rcre_scalar_axpy_f64,
    .gemv_f64        = rcre_scalar_gemv_f64,
    .scale_f32       = rcre_scalar_scale_f32,
    .scale_f64       = rcre_scalar_scale_f64,
    .sum_f32         = rcre_scalar_sum_f32,
    .sum_f64         = rcre_scalar_sum_f64
};

static atomic_bool s_hal_initialized = false;

void rcre_hal_init(void) {
    if (atomic_load(&s_hal_initialized)) {
        return;
    }

    uint32_t feats = rcre_cpu_get_features();

#if defined(__ARM_NEON) || defined(__aarch64__)
    if (feats & RCRE_CPU_FEAT_NEON) {
        rcre_hal.backend_name   = "ARM NEON 128-bit";
        rcre_hal.cpu_features   = feats;
        rcre_hal.gemm_f32       = rcre_neon_gemm_f32;
        rcre_hal.gemv_f32       = rcre_neon_gemv_f32;
        rcre_hal.dot_f32        = rcre_neon_dot_f32;
        rcre_hal.dot_i8         = rcre_neon_dot_i8;
        rcre_hal.gemm_i8        = rcre_neon_gemm_i8;
        rcre_hal.gemm_bf16      = rcre_scalar_gemm_bf16;
        rcre_hal.axpy_f32       = rcre_neon_axpy_f32;
        rcre_hal.rgb_to_gray_u8 = rcre_neon_rgb_to_gray_u8;
        rcre_hal.find_byte      = rcre_neon_find_byte;
        rcre_hal.count_byte     = rcre_neon_count_byte;
        /* H0: F64 cells (aarch64 NEON has the 64-bit datapath unconditionally) */
        rcre_hal.gemm_f64       = rcre_neon_gemm_f64;
        rcre_hal.dot_f64        = rcre_neon_dot_f64;
        rcre_hal.axpy_f64       = rcre_neon_axpy_f64;
        rcre_hal.gemv_f64       = rcre_neon_gemv_f64;
        rcre_hal.scale_f64      = rcre_neon_scale_f64;
        rcre_hal.sum_f64        = rcre_neon_sum_f64;
        atomic_store(&s_hal_initialized, true);
        return;
    }
#endif

/* H0: AVX-512 is checked BEFORE AVX2 — an AVX-512 host gets 8-wide f64,
 * keeping the 4-wide AVX2 F32 cells (F32 512-bit kernels arrive in H2). */
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX512F__) || defined(__clang__) || defined(__GNUC__))
    if (feats & RCRE_CPU_FEAT_AVX512F) {
        rcre_hal.backend_name   = "x86 AVX-512F 512-bit";
        rcre_hal.cpu_features   = feats;
        rcre_hal.gemm_f32       = rcre_avx2_gemm_f32;
        rcre_hal.dot_f32        = rcre_avx2_dot_f32;
        rcre_hal.axpy_f32       = rcre_avx2_axpy_f32;
        rcre_hal.rgb_to_gray_u8 = rcre_avx2_rgb_to_gray_u8;
        rcre_hal.find_byte      = rcre_avx2_find_byte;
        rcre_hal.count_byte     = rcre_avx2_count_byte;
        rcre_hal.gemm_f64       = rcre_avx512_gemm_f64;
        rcre_hal.dot_f64        = rcre_avx512_dot_f64;
        rcre_hal.axpy_f64       = rcre_avx512_axpy_f64;
        rcre_hal.gemv_f64       = rcre_avx512_gemv_f64;
        rcre_hal.scale_f64      = rcre_avx512_scale_f64;
        rcre_hal.sum_f64        = rcre_avx512_sum_f64;
        atomic_store(&s_hal_initialized, true);
        return;
    }
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX512F__) || defined(__clang__) || defined(__GNUC__))
    if (feats & RCRE_CPU_FEAT_AVX512F) {
        rcre_hal.backend_name   = "x86 AVX-512 512-bit";
        rcre_hal.cpu_features   = feats;
        rcre_hal.gemm_f32       = rcre_avx512_gemm_f32;
        rcre_hal.gemv_f32       = rcre_avx512_gemv_f32;
        rcre_hal.dot_f32        = rcre_avx512_dot_f32;
        rcre_hal.dot_i8         = rcre_avx2_dot_i8;
        rcre_hal.gemm_i8        = rcre_avx2_gemm_i8;
        rcre_hal.gemm_bf16      = rcre_scalar_gemm_bf16;
        rcre_hal.axpy_f32       = rcre_avx512_axpy_f32;
        rcre_hal.rgb_to_gray_u8 = rcre_avx2_rgb_to_gray_u8;
        rcre_hal.find_byte      = rcre_avx2_find_byte;
        rcre_hal.count_byte     = rcre_avx2_count_byte;
        atomic_store(&s_hal_initialized, true);
        return;
    }
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(__clang__) || defined(__GNUC__))
    if (feats & RCRE_CPU_FEAT_AVX2) {
        rcre_hal.backend_name   = "x86 AVX2 256-bit";
        rcre_hal.cpu_features   = feats;
        rcre_hal.gemm_f32       = rcre_avx2_gemm_f32;
        rcre_hal.gemv_f32       = rcre_avx2_gemv_f32;
        rcre_hal.dot_f32        = rcre_avx2_dot_f32;
        rcre_hal.dot_i8         = rcre_avx2_dot_i8;
        rcre_hal.gemm_i8        = rcre_avx2_gemm_i8;
        rcre_hal.gemm_bf16      = rcre_scalar_gemm_bf16;
        rcre_hal.axpy_f32       = rcre_avx2_axpy_f32;
        rcre_hal.rgb_to_gray_u8 = rcre_avx2_rgb_to_gray_u8;
        rcre_hal.find_byte      = rcre_avx2_find_byte;
        rcre_hal.count_byte     = rcre_avx2_count_byte;
        /* H0: F64 cells */
        rcre_hal.gemm_f64       = rcre_avx2_gemm_f64;
        rcre_hal.dot_f64        = rcre_avx2_dot_f64;
        rcre_hal.axpy_f64       = rcre_avx2_axpy_f64;
        rcre_hal.gemv_f64       = rcre_avx2_gemv_f64;
        rcre_hal.scale_f64      = rcre_avx2_scale_f64;
        rcre_hal.sum_f64        = rcre_avx2_sum_f64;
        atomic_store(&s_hal_initialized, true);
        return;
    }
#endif

#if defined(__wasm_simd128__)
    if (feats & RCRE_CPU_FEAT_WASM_SIMD) {
        rcre_hal.backend_name   = "WebAssembly SIMD 128-bit";
        rcre_hal.cpu_features   = feats;
        rcre_hal.gemm_f32       = rcre_wasm_gemm_f32;
        rcre_hal.gemv_f32       = rcre_wasm_gemv_f32;
        rcre_hal.dot_f32        = rcre_wasm_dot_f32;
        rcre_hal.dot_i8         = rcre_scalar_dot_i8;
        rcre_hal.gemm_i8        = rcre_scalar_gemm_i8;
        rcre_hal.gemm_bf16      = rcre_scalar_gemm_bf16;
        rcre_hal.axpy_f32       = rcre_wasm_axpy_f32;
        rcre_hal.rgb_to_gray_u8 = rcre_wasm_rgb_to_gray_u8;
        rcre_hal.find_byte      = rcre_wasm_find_byte;
        rcre_hal.count_byte     = rcre_wasm_count_byte;
        atomic_store(&s_hal_initialized, true);
        return;
    }
#endif

    rcre_hal.backend_name   = "ISO C23 Scalar Reference";
    rcre_hal.cpu_features   = feats;
    rcre_hal.gemm_f32       = rcre_scalar_gemm_f32;
    rcre_hal.gemv_f32       = rcre_scalar_gemv_f32;
    rcre_hal.dot_f32        = rcre_scalar_dot_f32;
    rcre_hal.dot_i8         = rcre_scalar_dot_i8;
    rcre_hal.gemm_i8        = rcre_scalar_gemm_i8;
    rcre_hal.gemm_bf16      = rcre_scalar_gemm_bf16;
    rcre_hal.axpy_f32       = rcre_scalar_axpy_f32;
    rcre_hal.rgb_to_gray_u8 = rcre_scalar_rgb_to_gray_u8;
    rcre_hal.find_byte      = rcre_scalar_find_byte;
    rcre_hal.count_byte     = rcre_scalar_count_byte;
    atomic_store(&s_hal_initialized, true);
}

typedef struct {
    size_t M, N, K;
    float alpha, beta;
    const float* A; size_t lda;
    const float* B; size_t ldb;
    float* C;       size_t ldc;
    size_t rows_per_task;
} RCRE_GemmMTCtx;

static void rcre_hal_gemm_mt_worker(void* arg, size_t task_id) {
    RCRE_GemmMTCtx* ctx = (RCRE_GemmMTCtx*)arg;
    size_t r0 = task_id * ctx->rows_per_task;
    if (r0 >= ctx->M) return;
    size_t r1 = r0 + ctx->rows_per_task;
    if (r1 > ctx->M) r1 = ctx->M;
    rcre_hal.gemm_f32(r1 - r0, ctx->N, ctx->K,
                      ctx->alpha,
                      ctx->A + r0 * ctx->lda, ctx->lda,
                      ctx->B, ctx->ldb,
                      ctx->beta,
                      ctx->C + r0 * ctx->ldc, ctx->ldc);
}

void rcre_hal_gemm_f32_mt(RCRE_ThreadPool* pool,
                          size_t M, size_t N, size_t K, float alpha,
                          const float* A, size_t lda,
                          const float* B, size_t ldb,
                          float beta, float* C, size_t ldc) {
    if (!pool || M < 16 || (M * N * K) < 16384) {
        rcre_hal.gemm_f32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    size_t num_tasks = 16;
    if (num_tasks > M / 4) num_tasks = M / 4;
    if (num_tasks < 2) {
        rcre_hal.gemm_f32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    size_t rows_per_task = (M + num_tasks - 1) / num_tasks;
    RCRE_GemmMTCtx ctx = {
        .M = M, .N = N, .K = K, .alpha = alpha, .beta = beta,
        .A = A, .lda = lda, .B = B, .ldb = ldb, .C = C, .ldc = ldc,
        .rows_per_task = rows_per_task
    };
    rcre_threadpool_parallel_for(pool, num_tasks, rcre_hal_gemm_mt_worker, &ctx);
}

/* Run dynamic probe and dispatch selection at program startup */
__attribute__((constructor)) static void rcre_hal_auto_init(void) {
    rcre_hal_init();
}
