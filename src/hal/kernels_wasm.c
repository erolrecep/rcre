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

#endif /* __wasm_simd128__ */
