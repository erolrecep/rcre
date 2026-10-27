/**
 * @file rcre_perf.h
 * @brief RCRE-PERF: Hardware Microarchitecture & Systems Performance Benchmark Suite
 * @version 1.0.0
 *
 * Built on RCRE ISO C23 primitives:
 *   - 64-byte cacheline-aligned RCRE_Arena
 *   - Strided N-D RCRE_Tensor & ShapeTracker views
 *   - RCRE_Prof hardware telemetry (cycles, RSS memory, Perfetto Chrome trace)
 *   - RCRE_HAL dynamic SIMD microkernel dispatch (ARM NEON / AVX2 / Scalar)
 */

#ifndef RCRE_PERF_H
#define RCRE_PERF_H

#include "rcre.h"
#include "rcre_prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Configuration & Benchmark Options                                         */
/* ========================================================================= */

typedef struct {
    size_t      matrix_size;    /* Dimension N for N x N matrix (default: 4096) */
    size_t      tile_size;      /* Sub-block tile dimension (default: 64)       */
    size_t      iterations;     /* Measurement repetitions (default: 3)         */
    const char* perfetto_out;   /* Filepath for Chrome/Perfetto JSON trace      */
    const char* html_out;       /* Filepath for self-contained HTML dashboard   */
    bool        run_cache;      /* Run cache locality benchmark                 */
    bool        run_strides;    /* Run strided tensor view benchmark            */
    bool        run_simd;       /* Run SIMD vs scalar benchmark                 */
    bool        verbose;        /* Verbose logging                              */
} RCRE_PerfConfig;

/* ========================================================================= */
/* Hardware Query & Optimization Barrier Utilities                          */
/* ========================================================================= */

/**
 * @brief Probes host CPU cache line size (e.g. 128 bytes on Apple Silicon, 64 on x86).
 */
static inline size_t rcre_perf_get_cacheline_size(void) {
#if defined(__APPLE__)
    size_t line_size = 0;
    size_t size = sizeof(line_size);
    if (sysctlbyname("hw.cachelinesize", &line_size, &size, NULL, 0) == 0 && line_size > 0) {
        return line_size;
    }
#endif
    return 64;
}

/**
 * @brief Compiler memory clobber preventing dead-code elimination.
 */
static inline void rcre_perf_do_not_optimize(void* p) {
    __asm__ volatile("" : : "g"(p) : "memory");
}

/* ========================================================================= */
/* Benchmark Subsystem Declarations                                          */
/* ========================================================================= */

/**
 * @brief Benchmark 01: Cache Locality & Memory Hierarchy Traversal.
 * Compares Row-Major (sequential streaming) vs Column-Major (cacheline thrashing)
 * vs Tiled/Blocked memory access using an RCRE_Arena.
 */
void rcre_bench_cache_locality(const RCRE_PerfConfig* cfg, RCRE_ProfContext* prof, RCRE_Arena* arena);

/**
 * @brief Benchmark 02: Strided Tensor Views vs Contiguous Physical Re-layout.
 * Demonstrates RCRE's ShapeTracker zero-copy transpositions vs contiguous repacking.
 */
void rcre_bench_strided_tensor(const RCRE_PerfConfig* cfg, RCRE_ProfContext* prof, RCRE_Arena* arena);

/**
 * @brief Benchmark 03: SIMD Vectorization & Arithmetic Intensity.
 * Evaluates RCRE HAL microkernels (ARM NEON / AVX2 axpy, dot, gemm) against
 * naive scalar C loops.
 */
void rcre_bench_simd_vectorization(const RCRE_PerfConfig* cfg, RCRE_ProfContext* prof, RCRE_Arena* arena);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_PERF_H */
