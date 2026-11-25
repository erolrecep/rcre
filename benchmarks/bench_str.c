/**
 * @file bench_str.c
 * @brief Hardware Throughput Benchmarks for rcre_str (GB/s & Ops/s)
 */

#include "rcre.h"
#include "rcre_str.h"
#include "hal_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    rcre_hal_init();
    printf("==================================================================\n");
    printf("  RCRE String Library Benchmark (%s)\n", rcre_hal.backend_name);
    printf("  Host CPU: %s\n", rcre_cpu_get_brand());
    printf("==================================================================\n");

    const size_t N = 16 * 1024 * 1024; /* 16 MB buffer */
    RCRE_Arena* arena = rcre_arena_create(N * 3 + 4 * 1024 * 1024);
    if (!arena) {
        fprintf(stderr, "Failed to allocate benchmark arena\n");
        return 1;
    }

    uint8_t* ascii_buf = (uint8_t*)rcre_arena_alloc(arena, N + 1, 64);
    uint8_t* utf8_buf  = (uint8_t*)rcre_arena_alloc(arena, N + 1, 64);
    uint8_t* dst_buf   = (uint8_t*)rcre_arena_alloc(arena, N + 1, 64);

    for (size_t i = 0; i < N; ++i) {
        ascii_buf[i] = (uint8_t)('A' + (i % 52));
    }
    ascii_buf[N] = '\0';
    memcpy(ascii_buf + N - 16, "RCRE_NEEDLE_2026", 16);

    /* Repeating 4-byte UTF-8 pattern ("a" + "ü" + "世" + "🚀" = 10 bytes) */
    const uint8_t pat10[10] = { 'a', 0xC3, 0xBC, 0xE4, 0xB8, 0x96, 0xF0, 0x9F, 0x9A, 0x80 };
    for (size_t i = 0; i + 10 <= N; i += 10) {
        memcpy(utf8_buf + i, pat10, 10);
    }
    for (size_t i = (N / 10) * 10; i < N; ++i) utf8_buf[i] = 'a';
    utf8_buf[N] = '\0';

    volatile size_t sink = 0;
    const int iters = 32;

    /* 1. UTF-8 Validation (ASCII) */
    double t0 = now_sec();
    for (int r = 0; r < iters; ++r) sink += (size_t)rcre_scalar_utf8_validate(ascii_buf, N);
    double dt_s = now_sec() - t0;

    t0 = now_sec();
    for (int r = 0; r < iters; ++r) sink += (size_t)rcre_hal.utf8_validate(ascii_buf, N);
    double dt_v = now_sec() - t0;
    printf("UTF-8 Validate (16MB ASCII):   Scalar %6.2f GB/s | SIMD %6.2f GB/s (%.2fx)\n",
           ((double)N * iters / 1e9) / dt_s, ((double)N * iters / 1e9) / dt_v, dt_s / dt_v);

    /* 2. UTF-8 Validation (100% Multi-Byte UTF-8) */
    t0 = now_sec();
    for (int r = 0; r < iters; ++r) sink += (size_t)rcre_scalar_utf8_validate(utf8_buf, N);
    dt_s = now_sec() - t0;

    t0 = now_sec();
    for (int r = 0; r < iters; ++r) sink += (size_t)rcre_hal.utf8_validate(utf8_buf, N);
    dt_v = now_sec() - t0;
    printf("UTF-8 Validate (16MB UTF-8):   Scalar %6.2f GB/s | SIMD %6.2f GB/s (%.2fx)\n",
           ((double)N * iters / 1e9) / dt_s, ((double)N * iters / 1e9) / dt_v, dt_s / dt_v);

    /* 3. Substring Search */
    const uint8_t* needle = (const uint8_t*)"RCRE_NEEDLE_2026";
    t0 = now_sec();
    for (int r = 0; r < iters; ++r) sink += rcre_scalar_find_substr(ascii_buf, N, needle, 16);
    dt_s = now_sec() - t0;

    t0 = now_sec();
    for (int r = 0; r < iters; ++r) sink += rcre_hal.find_substr(ascii_buf, N, needle, 16);
    dt_v = now_sec() - t0;
    printf("Substring Search (16MB):       Scalar %6.2f GB/s | SIMD %6.2f GB/s (%.2fx)\n",
           ((double)N * iters / 1e9) / dt_s, ((double)N * iters / 1e9) / dt_v, dt_s / dt_v);

    /* 4. ASCII to_lower */
    t0 = now_sec();
    for (int r = 0; r < iters; ++r) rcre_scalar_ascii_to_lower(ascii_buf, dst_buf, N);
    dt_s = now_sec() - t0;

    t0 = now_sec();
    for (int r = 0; r < iters; ++r) rcre_hal.ascii_to_lower(ascii_buf, dst_buf, N);
    dt_v = now_sec() - t0;
    printf("ASCII to_lower (16MB):         Scalar %6.2f GB/s | SIMD %6.2f GB/s (%.2fx)\n",
           ((double)N * iters / 1e9) / dt_s, ((double)N * iters / 1e9) / dt_v, dt_s / dt_v);

    /* 5. Arena StrPool interning (100k lookups, 4096 unique keys) */
    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
    RCRE_StrPool pool;
    (void)rcre_strpool_init(&pool, arena, 4096);
    RCRE_Slice keys[4096];
    for (size_t i = 0; i < 4096; ++i) {
        keys[i] = rcre_str_fmt(arena, "tensor.layer_%zu.attn.q_proj.weight", i);
        (void)rcre_strpool_intern(&pool, keys[i]);
    }
    const size_t lookups = 1000000;
    t0 = now_sec();
    for (size_t i = 0; i < lookups; ++i) {
        sink += (size_t)rcre_strpool_find(&pool, keys[i & 4095U]);
    }
    double dt_pool = now_sec() - t0;
    printf("StrPool Lookup (4096 keys):    %6.2f M lookups/s (%.1f ns/op, 0 mallocs)\n",
           ((double)lookups / 1e6) / dt_pool, (dt_pool * 1e9) / (double)lookups);
    rcre_arena_temp_end(tmp);

    (void)sink;
    rcre_arena_destroy(arena);
    return 0;
}
