/**
 * @file bench_e_means.c
 * @brief e-means vs RCRE Lloyd k-means at CV and embedding scales.
 *
 * Cases (see docs/benchmarks for numbers):
 *   4krgb    n=8,294,400  d=3    k=256   — 4K image pixels, palette fit
 *   sift100k n=100,000    d=128  k=4096  — SIFT-like descriptors
 *   embed1m  n=1,000,000  d=96   k=4096  — embedding scale (opt-in, --big)
 *
 * Deterministic data (splitmix-driven blobs + noise), so the table is
 * reproducible. Wall clock only (no clock_gettime portability games):
 * CLOCK_MONOTONIC on POSIX.
 */
#include "rcre.h"  /* core RCRE (incl. the e-means API) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* splitmix64 (same as RCRE's) for data generation */
static uint64_t sm64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double unif(uint64_t* s) { return (double)(sm64(s) >> 11) * 0x1p-53; }

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Generate structured data: nblob gaussian-ish blobs (uniform boxes +
 * jitter) so the problem is not degenerate but is not trivially solved by
 * random init either. */
static void gen_data(float* X, size_t n, size_t d, size_t nblob, uint64_t seed) {
    uint64_t s = seed;
    /* deterministic blob centers */
    double* cx = malloc(nblob * d * sizeof(double));
    double* sy = malloc(nblob * sizeof(double));
    for (size_t b = 0; b < nblob; b++) {
        sy[b] = 0.3 + 1.2 * unif(&s);
        for (size_t dd = 0; dd < d; dd++)
            cx[b * d + dd] = (unif(&s) * 2.0 - 1.0) * 20.0;
    }
    for (size_t i = 0; i < n; i++) {
        size_t b = (size_t)(unif(&s) * (double)nblob);
        for (size_t dd = 0; dd < d; dd++) {
            double u = unif(&s) * 2.0 - 1.0;
            double v = unif(&s) * 2.0 - 1.0;
            double g = sqrt(-2.0 * log(1.0e-12 + (u * u + v * v) * 0.5))
                       * cos(6.283185307179586 * unif(&s)); /* cheap polar */
            X[i * d + dd] = (float)(cx[b * d + dd] + sy[b] * g);
        }
    }
    free(cx); free(sy);
}

static void run_case(const char* name, size_t n, size_t d, size_t k,
                     size_t nblob, int with_pool) {
    printf("== %s: n=%zu d=%zu k=%zu %s ==\n", name, n, d, k,
           with_pool ? "(pool)" : "(serial)");

    /* Arena: X + label tensor + one full e-means state PER fit (the arena
     * is monotonic — serial and pool states both accumulate) + headroom. */
    size_t state = 2 * (n * 4 + 16u * (unsigned)k * (unsigned)d * 4u) + n * 4;
    size_t need = n * d * 4 + state + 64 * 1024 * 1024;
    RCRE_Arena* arena = rcre_arena_create(need > (size_t)512 * 1024 * 1024
                                              ? need : (size_t)512 * 1024 * 1024);
    if (!arena) { printf("arena OOM\n"); return; }

    size_t xshape[2] = { n, d };
    RCRE_Tensor X = rcre_tensor_create(xshape, 2, RCRE_DTYPE_F32,
                                       RCRE_LAYOUT_TABULAR, arena);
    gen_data((float*)X.data, n, d, nblob, 0xE5EEDULL);
    RCRE_Tensor C = rcre_tensor_create((size_t[2]){ k, d }, 2, RCRE_DTYPE_F32,
                                       RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor Cl = rcre_tensor_create((size_t[2]){ k, d }, 2, RCRE_DTYPE_F32,
                                        RCRE_LAYOUT_TABULAR, arena);
    size_t lshape[2] = { n, 1 };
    RCRE_Tensor L = rcre_tensor_create(lshape, 2, RCRE_DTYPE_I32,
                                       RCRE_LAYOUT_TABULAR, arena);

    RCRE_ThreadPool* pool = with_pool ? rcre_threadpool_create(0) : NULL;

    /* recipe */
    RCRE_EMMeansRecipe rec;
    if (rcre_ml_emeans_recipe(&X, k, NULL, &rec) != RCRE_OK)
        return;
    printf("  recipe: B=%zu alpha0=%.4f epochs=%zu steps/pass=%zu total=%zu\n",
           rec.batch_size, (double)rec.alpha0, rec.n_epochs,
           rec.steps_per_pass, rec.total_steps);

    /* e-means (serial — the comparison is at equal wall-clock budget,
     * and the vendored Lloyd reference is serial-only). */
    double t0 = now_sec();
    float em_mse = 0.0f;
    int splits = 0;
    int rc = rcre_ml_emeans_fit_ex(&X, k, NULL, &C, &L, &em_mse, &splits,
                                   arena, NULL);
    double t_em = now_sec() - t0;
    if (rc != RCRE_OK) { printf("  e-means FAILED (%d)\n", rc); return; }
    float em_final = 0.0f;
    if (rcre_ml_emeans_assign(&X, &C, &L, &em_final, arena) != RCRE_OK)
        return;

    /* e-means with the threadpool (when requested): threading effect. */
    double t_em_pool = 0.0;
    if (pool) {
        t0 = now_sec();
        rc = rcre_ml_emeans_fit_ex(&X, k, NULL, &C, &L, &em_mse, &splits,
                                   arena, pool);
        t_em_pool = now_sec() - t0;
        if (rc != RCRE_OK) { printf("  e-means pool FAILED (%d)\n", rc); return; }
    }

    /* Lloyd, the paper's headline budget: 25 full iterations vs e-means'
     * 5 epochs. */
    RCRE_KMeansParams kp;
    memset(&kp, 0, sizeof(kp));
    kp.max_iter = 25;
    kp.n_init = 1;
    kp.tol = 1e-6f;
    kp.seed = 42;
    t0 = now_sec();
    float lloyd_mse = 0.0f;
    rc = rcre_ml_kmeans_fit_ex(&X, k, &kp, &Cl, &L, &lloyd_mse, arena);
    double t_lloyd = now_sec() - t0;
    if (rc != RCRE_OK) { printf("  lloyd FAILED (%d)\n", rc); return; }
    float lloyd_final = 0.0f;
    /* exact final MSE for Lloyd too (fit returns moving-pass MSE) */
    if (rcre_ml_emeans_assign(&X, &Cl, &L, &lloyd_final, arena) != RCRE_OK)
        return;

    printf("  e-means : %8.1f ms  moving-MSE %.6f  final-MSE %.6f  splits=%d\n",
           t_em * 1000.0, (double)em_mse, (double)em_final, splits);
    if (pool)
        printf("  e-means P: %8.1f ms  (pool speedup %.2fx)\n",
               t_em_pool * 1000.0, t_em / (t_em_pool > 0 ? t_em_pool : 1e-12));
    printf("  lloyd-25: %8.1f ms  moving-MSE %.6f  final-MSE %.6f\n",
           t_lloyd * 1000.0, (double)lloyd_mse, (double)lloyd_final);
    printf("  speedup: %.2fx wall  |  MSE ratio e/lloyd: %.3f\n",
           t_lloyd / (t_em > 0 ? t_em : 1e-12),
           (double)em_final / (double)lloyd_final);

    if (pool) rcre_threadpool_destroy(pool);
    rcre_arena_destroy(arena);
}

int main(int argc, char** argv) {
    int big = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--big") == 0) big = 1;

    printf("rcre_e-means benchmark\n\n");
    run_case("4krgb",    8294400,   3,  256,  512, 1);
    run_case("sift100k", 100000,   128, 4096, 2048, 1);
    if (big)
        run_case("embed1m",  1000000,  96, 4096, 8192, 1);
    return 0;
}
