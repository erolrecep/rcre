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
    .dot_f32         = rcre_scalar_dot_f32,
    .axpy_f32        = rcre_scalar_axpy_f32,
    .rgb_to_gray_u8  = rcre_scalar_rgb_to_gray_u8,
    .find_byte       = rcre_scalar_find_byte,
    .count_byte      = rcre_scalar_count_byte
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
        rcre_hal.dot_f32        = rcre_neon_dot_f32;
        rcre_hal.axpy_f32       = rcre_neon_axpy_f32;
        rcre_hal.rgb_to_gray_u8 = rcre_neon_rgb_to_gray_u8;
        rcre_hal.find_byte      = rcre_neon_find_byte;
        rcre_hal.count_byte     = rcre_neon_count_byte;
        atomic_store(&s_hal_initialized, true);
        return;
    }
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(__clang__) || defined(__GNUC__))
    if (feats & RCRE_CPU_FEAT_AVX2) {
        rcre_hal.backend_name   = "x86 AVX2 256-bit";
        rcre_hal.cpu_features   = feats;
        rcre_hal.gemm_f32       = rcre_avx2_gemm_f32;
        rcre_hal.dot_f32        = rcre_avx2_dot_f32;
        rcre_hal.axpy_f32       = rcre_avx2_axpy_f32;
        rcre_hal.rgb_to_gray_u8 = rcre_avx2_rgb_to_gray_u8;
        rcre_hal.find_byte      = rcre_avx2_find_byte;
        rcre_hal.count_byte     = rcre_avx2_count_byte;
        atomic_store(&s_hal_initialized, true);
        return;
    }
#endif

#if defined(__wasm_simd128__)
    if (feats & RCRE_CPU_FEAT_WASM_SIMD) {
        rcre_hal.backend_name   = "WebAssembly SIMD 128-bit";
        rcre_hal.cpu_features   = feats;
        rcre_hal.gemm_f32       = rcre_wasm_gemm_f32;
        rcre_hal.dot_f32        = rcre_wasm_dot_f32;
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
    rcre_hal.dot_f32        = rcre_scalar_dot_f32;
    rcre_hal.axpy_f32       = rcre_scalar_axpy_f32;
    rcre_hal.rgb_to_gray_u8 = rcre_scalar_rgb_to_gray_u8;
    rcre_hal.find_byte      = rcre_scalar_find_byte;
    rcre_hal.count_byte     = rcre_scalar_count_byte;
    atomic_store(&s_hal_initialized, true);
}

/* Run dynamic probe and dispatch selection at program startup */
__attribute__((constructor)) static void rcre_hal_auto_init(void) {
    rcre_hal_init();
}
