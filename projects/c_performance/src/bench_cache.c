/**
 * @file bench_cache.c
 * @brief Benchmark 01: Cache Locality & Memory Hierarchy Traversal
 */

#include "rcre_perf.h"
#include <inttypes.h>

void rcre_bench_cache_locality(const RCRE_PerfConfig* cfg, RCRE_ProfContext* prof, RCRE_Arena* arena) {
    const size_t N = cfg->matrix_size;
    const size_t tile_size = cfg->tile_size;
    const size_t num_elements = N * N;
    const size_t total_bytes = num_elements * sizeof(float);
    const double total_mb = (double)total_bytes / (1024.0 * 1024.0);
    const size_t cache_line = rcre_perf_get_cacheline_size();

    printf("\n===================================================================\n");
    printf(" 📊 [Benchmark 01] Cache Locality & Memory Hierarchy Traversal     \n");
    printf("===================================================================\n");
    printf(" Matrix Dimension       : %zu x %zu (%zu elements)\n", N, N, num_elements);
    printf(" Memory Footprint       : %.2f MB (Arena 64-byte aligned)\n", total_mb);
    printf(" Hardware Cache Line    : %zu bytes (%zu single-precision floats)\n", 
           cache_line, cache_line / sizeof(float));
    printf(" Sub-block Tile Size    : %zu x %zu (%.2f KB working set)\n", 
           tile_size, tile_size, (double)(tile_size * tile_size * sizeof(float)) / 1024.0);
    printf("-------------------------------------------------------------------\n");

    /* Allocate matrix within the arena */
    RCRE_ArenaTemp checkpoint = rcre_arena_temp_begin(arena);
    float* mat = (float*)rcre_arena_alloc(arena, total_bytes, 64);
    if (!mat) {
        fprintf(stderr, "Error: Insufficient arena capacity for %zu MB matrix.\n", (size_t)total_mb);
        return;
    }

    /* Initialize matrix */
    for (size_t i = 0; i < num_elements; ++i) {
        mat[i] = 1.0f;
    }

    /* Track initial RSS memory */
    if (prof) rcre_prof_sample_rss(prof);

    /* --------------------------------------------------------------------- */
    /* 1. Row-Major Traversal (High Spatial Locality)                         */
    /* Sequential access: mat[i * N + j] reads contiguous physical cachelines. */
    /* --------------------------------------------------------------------- */
    double sum_row = 0.0;
    uint64_t dur_row_ns = 0;

    /* Warm-up pass */
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            sum_row += mat[i * N + j];
        }
    }
    rcre_perf_do_not_optimize(&sum_row);
    sum_row = 0.0;

    /* Measured runs */
    for (size_t iter = 0; iter < cfg->iterations; ++iter) {
        RCRE_PROFILE_SCOPE(prof, "cache_row_major", z_row);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                sum_row += mat[i * N + j];
            }
        }
        dur_row_ns += RCRE_PROFILE_END(z_row);
    }
    dur_row_ns /= cfg->iterations;
    rcre_perf_do_not_optimize(&sum_row);

    double time_row_sec = (double)dur_row_ns * 1e-9;
    double bw_row = (total_mb / 1024.0) / time_row_sec;

    /* --------------------------------------------------------------------- */
    /* 2. Column-Major Traversal (Poor Spatial Locality)                      */
    /* Strided access: mat[j * N + i] jumps N*4 bytes every iteration.         */
    /* --------------------------------------------------------------------- */
    double sum_col = 0.0;
    uint64_t dur_col_ns = 0;

    for (size_t iter = 0; iter < cfg->iterations; ++iter) {
        RCRE_PROFILE_SCOPE(prof, "cache_col_major", z_col);
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                sum_col += mat[i * N + j];
            }
        }
        dur_col_ns += RCRE_PROFILE_END(z_col);
    }
    dur_col_ns /= cfg->iterations;
    rcre_perf_do_not_optimize(&sum_col);

    double time_col_sec = (double)dur_col_ns * 1e-9;
    double bw_col = (total_mb / 1024.0) / time_col_sec;

    /* --------------------------------------------------------------------- */
    /* 3. Tiled / Blocked Traversal (Temporal & Spatial Restructuring)        */
    /* Deconstructs traversal into BxB blocks that fit in CPU L1/L2 cache.   */
    /* --------------------------------------------------------------------- */
    double sum_tiled = 0.0;
    uint64_t dur_tiled_ns = 0;

    for (size_t iter = 0; iter < cfg->iterations; ++iter) {
        RCRE_PROFILE_SCOPE(prof, "cache_tiled_blocked", z_tile);
        for (size_t ii = 0; ii < N; ii += tile_size) {
            for (size_t jj = 0; jj < N; jj += tile_size) {
                size_t i_max = (ii + tile_size < N) ? ii + tile_size : N;
                size_t j_max = (jj + tile_size < N) ? jj + tile_size : N;
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t j = jj; j < j_max; ++j) {
                        sum_tiled += mat[i * N + j];
                    }
                }
            }
        }
        dur_tiled_ns += RCRE_PROFILE_END(z_tile);
    }
    dur_tiled_ns /= cfg->iterations;
    rcre_perf_do_not_optimize(&sum_tiled);

    double time_tiled_sec = (double)dur_tiled_ns * 1e-9;
    double bw_tiled = (total_mb / 1024.0) / time_tiled_sec;

    /* Report results */
    printf("\n [1] Row-Major Traversal (Sequential Streaming):\n");
    printf("     Latency    : %7.3f ms  (avg of %zu runs)\n", time_row_sec * 1000.0, cfg->iterations);
    printf("     Throughput : %7.2f GB/s\n", bw_row);
    printf("     Checksum   : %.1f\n", sum_row);

    printf("\n [2] Column-Major Traversal (Cacheline Thrashing):\n");
    printf("     Latency    : %7.3f ms  (avg of %zu runs)\n", time_col_sec * 1000.0, cfg->iterations);
    printf("     Throughput : %7.2f GB/s\n", bw_col);
    printf("     Checksum   : %.1f\n", sum_col);

    printf("\n [3] Tiled Traversal (%zux%zu Cache-Friendly Blocks):\n", tile_size, tile_size);
    printf("     Latency    : %7.3f ms  (avg of %zu runs)\n", time_tiled_sec * 1000.0, cfg->iterations);
    printf("     Throughput : %7.2f GB/s\n", bw_tiled);
    printf("     Checksum   : %.1f\n", sum_tiled);

    double speedup_row = time_col_sec / (time_row_sec > 1e-9 ? time_row_sec : 1e-9);
    double speedup_tiled = time_col_sec / (time_tiled_sec > 1e-9 ? time_tiled_sec : 1e-9);

    printf("\n Architectural Synthesis:\n");
    printf("   • Row-Major vs Column-Major Speedup : %6.2fx faster\n", speedup_row);
    printf("   • Tiled vs Column-Major Speedup     : %6.2fx faster\n", speedup_tiled);
    printf("   • Microarchitectural Mechanism      : Fetching 1 element in Row-Major brings\n");
    printf("     the entire %zu-byte line into L1 cache, granting zero-latency hits for\n", cache_line);
    printf("     the next %zu floats, whereas Column-Major jumps %zu bytes per read.\n",
           cache_line / sizeof(float), N * sizeof(float));

    if (prof) {
        rcre_prof_mark(prof, "CACHE_ROW_GBPS", bw_row);
        rcre_prof_mark(prof, "CACHE_COL_GBPS", bw_col);
        rcre_prof_mark(prof, "CACHE_TILED_GBPS", bw_tiled);
    }

    rcre_arena_temp_end(checkpoint);
}
