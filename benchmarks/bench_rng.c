/**
 * @file bench_rng.c
 * @brief H1 throughput bars: counter-based RNG and quasi-MC generation.
 *
 * Honest scope: these are generation-throughput bars (no statistical
 * work). SOTA context: PCG64 is a classic-algorithm benchmark —
 * comparisons of interest are (a) RCRE's own C implementation vs
 * numpy's C implementation if numpy is present, (b) skip-ahead and
 * substream derivation costs, (c) quasi-MC point generation rates.
 */

#include "rcre.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  RCRE H1 — RNG & Quasi-MC Generation Bars                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    RCRE_RNG r;
    rcre_rng_pcg_init(&r, 20261123);

    /* u64 draw throughput */
    const int N = 200000000;
    uint64_t acc = 0;
    double t0 = now_sec();
    for (int i = 0; i < N; ++i) acc ^= rcre_rng_pcg_next_u64(&r);
    double t1 = now_sec();
    printf("  PCG64 u64 draws      : %8.3f Ms/s  (2e8 draws)\n", (double)N / (t1 - t0) / 1e6);

    /* f64 stream throughput (GB/s of usable double values) */
    rcre_rng_pcg_init(&r, 1);
    double sum = 0.0;
    t0 = now_sec();
    for (int i = 0; i < N; ++i) sum += rcre_rng_pcg_next_f64(&r);
    t1 = now_sec();
    printf("  PCG64 f64 stream     : %8.3f GB/s  (8 B per value)\n",
           (double)N * 8.0 / (t1 - t0) / 1e9);
    printf("  (mean of f64 stream, sanity: ~0.5) = %.5f (sum of 2e8)\n", sum);
    (void)acc;

    /* skip-ahead cost: 10M random-size skips */
    rcre_rng_pcg_init(&r, 7);
    const uint64_t NSKIP = 10000000;
    t0 = now_sec();
    for (uint64_t i = 0; i < NSKIP; ++i)
        rcre_rng_pcg_skip(&r, (uint64_t)(i * 2654435761u) | 1);
    t1 = now_sec();
    printf("  PCG64 skip (avg ~27 bits) : %8.3f Mskips/s\n",
           (double)NSKIP / (t1 - t0) / 1e6);

    /* substream derivation cost */
    RCRE_RNG base;
    rcre_rng_pcg_init(&base, 42);
    const int NSW = 10000000;
    RCRE_RNG w;
    t0 = now_sec();
    for (int i = 0; i < NSW; ++i) rcre_rng_pcg_substream(&base, (uint64_t)i, &w);
    t1 = now_sec();
    printf("  PCG64 substream derive  : %8.3f M/s\n", (double)NSW / (t1 - t0) / 1e6);

    /* Sobol' point generation: 20 dims, 2e7 points */
    double p[RCRE_QRNG_SOBOL_DIM];
    const uint64_t NSOB = 20000000;
    double ssum = 0.0;
    t0 = now_sec();
    for (uint64_t n = 0; n < NSOB; ++n) {
        rcre_qrng_sobol_at(n, RCRE_QRNG_SOBOL_DIM, p);
        ssum += p[0] + p[19];
    }
    t1 = now_sec();
    printf("  Sobol at (d=20)        : %8.3f Mpts/s  (%.2f MB/s of points)\n",
           (double)NSOB / (t1 - t0) / 1e6,
           (double)NSOB * RCRE_QRNG_SOBOL_DIM * 8.0 / (t1 - t0) / 1e6);
    (void)ssum;

    /* Halton point generation: 3 dims */
    const uint64_t NHAL = 20000000;
    double h[RCRE_QRNG_HALTON_DIM];
    t0 = now_sec();
    for (uint64_t n = 0; n < NHAL; ++n) {
        rcre_qrng_halton_at(n, RCRE_QRNG_HALTON_DIM, h);
        ssum += h[0] + h[2];
    }
    t1 = now_sec();
    printf("  Halton at (k=3)        : %8.3f Mpts/s\n",
           (double)NHAL / (t1 - t0) / 1e6);

    return 0;
}
