/**
 * @file main.c
 * @brief RCRE-PERF: Hardware Microarchitecture & Performance Benchmark Suite Entrypoint
 */

#include "rcre_perf.h"

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --all                  Run all benchmarks (default)\n");
    printf("  --cache                Run Benchmark 01: Cache locality & memory hierarchy\n");
    printf("  --strides              Run Benchmark 02: Strided tensor views vs contiguous\n");
    printf("  --simd                 Run Benchmark 03: SIMD vectorization & GEMM\n");
    printf("  --matrix-size <N>      Matrix dimension N for N x N workloads (default: 4096)\n");
    printf("  --tile-size <T>        Sub-block tile dimension for cache blocking (default: 64)\n");
    printf("  --iterations <I>       Number of measurement iterations (default: 3)\n");
    printf("  --perfetto <path.json> Export Chrome Tracing / Perfetto timeline trace\n");
    printf("  --html <path.html>     Export interactive self-contained HTML report\n");
    printf("  --help                 Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s --all --matrix-size 2048\n", prog);
    printf("  %s --cache --matrix-size 8192\n", prog);
    printf("  %s --all --perfetto trace.json --html dashboard.html\n", prog);
}

int main(int argc, char** argv) {
    RCRE_PerfConfig cfg = {
        .matrix_size  = 4096,
        .tile_size    = 64,
        .iterations   = 3,
        .perfetto_out = NULL,
        .html_out     = NULL,
        .run_cache    = false,
        .run_strides  = false,
        .run_simd     = false,
        .verbose      = false
    };

    bool explicitly_selected = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--all") == 0) {
            cfg.run_cache = true;
            cfg.run_strides = true;
            cfg.run_simd = true;
            explicitly_selected = true;
        } else if (strcmp(argv[i], "--cache") == 0) {
            cfg.run_cache = true;
            explicitly_selected = true;
        } else if (strcmp(argv[i], "--strides") == 0) {
            cfg.run_strides = true;
            explicitly_selected = true;
        } else if (strcmp(argv[i], "--simd") == 0) {
            cfg.run_simd = true;
            explicitly_selected = true;
        } else if (strcmp(argv[i], "--matrix-size") == 0 && i + 1 < argc) {
            cfg.matrix_size = (size_t)atoi(argv[++i]);
            if (cfg.matrix_size < 64) cfg.matrix_size = 64;
        } else if (strcmp(argv[i], "--tile-size") == 0 && i + 1 < argc) {
            cfg.tile_size = (size_t)atoi(argv[++i]);
            if (cfg.tile_size < 8) cfg.tile_size = 8;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            cfg.iterations = (size_t)atoi(argv[++i]);
            if (cfg.iterations < 1) cfg.iterations = 1;
        } else if (strcmp(argv[i], "--perfetto") == 0 && i + 1 < argc) {
            cfg.perfetto_out = argv[++i];
        } else if (strcmp(argv[i], "--html") == 0 && i + 1 < argc) {
            cfg.html_out = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s (see --help)\n", argv[i]);
            return 1;
        }
    }

    if (!explicitly_selected) {
        cfg.run_cache   = true;
        cfg.run_strides = true;
        cfg.run_simd    = true;
    }

    /* Initialize RCRE Hardware Abstraction Layer */
    rcre_hal_init();

    printf("===================================================================\n");
    printf("   ⚡ RCRE-PERF: Hardware Architecture & Performance Suite         \n");
    printf("===================================================================\n");
    printf(" Host Processor         : %s\n", rcre_cpu_get_brand());
    printf(" Active HAL Backend     : %s (SIMD Features: 0x%04x)\n", 
           rcre_hal.backend_name, rcre_cpu_get_features());
    printf(" Hardware Cacheline     : %zu bytes\n", rcre_perf_get_cacheline_size());
    printf(" Matrix Benchmark Size  : %zu x %zu\n", cfg.matrix_size, cfg.matrix_size);
    printf(" Tile Blocking Size     : %zu x %zu\n", cfg.tile_size, cfg.tile_size);
    printf(" Repetitions (Averaged) : %zu\n", cfg.iterations);
    printf("===================================================================\n");

    /* Allocate linear scratchpad arena (Capacity: 3 matrices of size N x N x sizeof(float) + 128MB headroom) */
    size_t needed_bytes = (cfg.matrix_size * cfg.matrix_size * sizeof(float) * 4) + (128 * 1024 * 1024);
    RCRE_Arena* arena = rcre_arena_create(needed_bytes);
    if (!arena) {
        fprintf(stderr, "Fatal: Failed to allocate %zu MB memory arena.\n", needed_bytes / (1024 * 1024));
        return 1;
    }

    /* Initialize RCRE telemetry profiler */
    RCRE_ProfContext* prof = rcre_prof_init(4096);

    /* Run selected benchmarks */
    if (cfg.run_cache) {
        rcre_bench_cache_locality(&cfg, prof, arena);
    }
    if (cfg.run_strides) {
        rcre_bench_strided_tensor(&cfg, prof, arena);
    }
    if (cfg.run_simd) {
        rcre_bench_simd_vectorization(&cfg, prof, arena);
    }

    /* Telemetry summary */
    printf("\n===================================================================\n");
    printf(" 📈 RCRE Profiler Telemetry Report (Pillars 1-4)                   \n");
    printf("===================================================================\n");
    if (prof) {
        rcre_prof_sample_rss(prof);
        rcre_prof_sample_arena(prof, arena);
        rcre_prof_print_summary(prof, stdout);

        if (cfg.perfetto_out) {
            if (rcre_prof_export_perfetto_json(prof, cfg.perfetto_out)) {
                printf("\n [Perfetto] Exported timeline trace to: %s\n", cfg.perfetto_out);
                printf("            Open https://ui.perfetto.dev and drag-and-drop this file!\n");
            } else {
                fprintf(stderr, "Error: Failed to export Perfetto trace to %s\n", cfg.perfetto_out);
            }
        }

        if (cfg.html_out) {
            if (rcre_prof_export_html(prof, cfg.html_out)) {
                printf("\n [Dashboard] Exported interactive HTML dashboard to: %s\n", cfg.html_out);
            } else {
                fprintf(stderr, "Error: Failed to export HTML dashboard to %s\n", cfg.html_out);
            }
        }
    }

    /* Teardown */
    if (prof) rcre_prof_free(prof);
    rcre_arena_destroy(arena);

    printf("\n✅ Benchmark execution complete.\n");
    return 0;
}
