/**
 * @file galton_wasm.c
 * @brief WebAssembly export bindings for RCRE Galton Board C23 engine.
 * 
 * Can be compiled with Emscripten or clang wasm32:
 *   emcc -O3 -s WASM=1 -s EXPORTED_FUNCTIONS="['_wasm_galton_init', '_wasm_galton_drop_batch', ...]" ...
 */

#include "rcre_galton.h"
#include <stdlib.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define WASM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define WASM_EXPORT __attribute__((visibility("default")))
#endif

static galton_board_t* g_wasm_board = nullptr;
static galton_stats_t g_wasm_stats;

WASM_EXPORT int wasm_galton_init(uint32_t rows, double bias, uint64_t seed) {
    if (g_wasm_board) {
        galton_board_destroy(g_wasm_board);
        g_wasm_board = nullptr;
    }

    galton_config_t config = {
        .rows = rows,
        .bias = bias,
        .seed = seed,
        .record_paths = false
    };

    g_wasm_board = galton_board_create(&config);
    return g_wasm_board != nullptr ? 0 : -1;
}

WASM_EXPORT void wasm_galton_reset(void) {
    if (g_wasm_board) {
        galton_board_reset(g_wasm_board);
    }
}

WASM_EXPORT uint32_t wasm_galton_drop_single(void) {
    if (!g_wasm_board) return 0;
    return galton_drop_single(g_wasm_board, nullptr);
}

WASM_EXPORT void wasm_galton_drop_batch(uint32_t n_balls) {
    if (!g_wasm_board) return;
    galton_drop_batch(g_wasm_board, n_balls);
}

WASM_EXPORT const uint64_t* wasm_galton_get_bins_ptr(void) {
    if (!g_wasm_board) return nullptr;
    return g_wasm_board->bin_counts;
}

WASM_EXPORT uint32_t wasm_galton_get_num_bins(void) {
    if (!g_wasm_board) return 0;
    return g_wasm_board->num_bins;
}

WASM_EXPORT uint64_t wasm_galton_get_total_dropped(void) {
    if (!g_wasm_board) return 0;
    return g_wasm_board->total_dropped;
}

WASM_EXPORT double wasm_galton_get_empirical_mean(void) {
    if (!g_wasm_board) return 0.0;
    galton_calculate_stats(g_wasm_board, &g_wasm_stats);
    return g_wasm_stats.mean;
}

WASM_EXPORT double wasm_galton_get_empirical_variance(void) {
    if (!g_wasm_board) return 0.0;
    galton_calculate_stats(g_wasm_board, &g_wasm_stats);
    return g_wasm_stats.variance;
}

WASM_EXPORT double wasm_galton_get_chi_square(void) {
    if (!g_wasm_board) return 0.0;
    galton_calculate_stats(g_wasm_board, &g_wasm_stats);
    return g_wasm_stats.chi_square_stat;
}

WASM_EXPORT double wasm_galton_get_p_value(void) {
    if (!g_wasm_board) return 1.0;
    galton_calculate_stats(g_wasm_board, &g_wasm_stats);
    return g_wasm_stats.chi_square_p_value;
}
