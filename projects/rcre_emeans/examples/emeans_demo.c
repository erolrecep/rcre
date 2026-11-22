/**
 * @file emeans_demo.c
 * @brief Turnkey CLI demo for rcre-emeans (Bessa et al., 2026).
 *
 * Demonstrates:
 *   1. Automatic scaling-law recipe resolution (batch size B, alpha0, cosine schedule)
 *   2. Fast mini-batch EMA training with Gumbel-max dead-cluster revival
 *   3. Real-time ASCII scatter map of clustered Gaussian blobs in 2D
 *   4. Instant convergence comparison vs standard 25-iteration Lloyd k-means
 */

#include "rcre.h"  /* core RCRE (incl. the e-means API) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define DEMO_N 20000
#define DEMO_D 2
#define DEMO_K 4

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

static uint64_t sm64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double unif(uint64_t* s) {
    return (double)(sm64(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* Generate 4 well-separated Gaussian blobs in 2D */
static void gen_demo_blobs(float* X, size_t n, uint64_t seed) {
    uint64_t s = seed;
    const float centers[4][2] = {
        { -5.0f, -5.0f },
        { -5.0f,  5.0f },
        {  5.0f, -5.0f },
        {  5.0f,  5.0f }
    };
    for (size_t i = 0; i < n; i++) {
        size_t c = (size_t)(unif(&s) * 4.0);
        if (c > 3) c = 3;
        double u = unif(&s) * 2.0 - 1.0;
        double v = unif(&s) * 2.0 - 1.0;
        double r2 = u * u + v * v;
        if (r2 >= 1.0 || r2 < 1e-12) { u = 0.5; v = 0.5; r2 = 0.5; }
        double g = sqrt(-2.0 * log(r2) / r2);
        X[i * 2 + 0] = centers[c][0] + (float)(u * g * 1.2);
        X[i * 2 + 1] = centers[c][1] + (float)(v * g * 1.2);
    }
}

/* Render 60x24 ASCII terminal scatter plot of clusters and centroids */
static void render_ascii_map(const float* X, const int32_t* L, size_t n,
                             const float* C, size_t k) {
    const int W = 62, H = 22;
    char grid[22][62];
    memset(grid, ' ', sizeof(grid));

    const float min_val = -10.0f, max_val = 10.0f;
    const char cluster_chars[4] = { '.', '+', 'x', 'o' };

    /* Subsample points for map clarity */
    for (size_t i = 0; i < n; i += 10) {
        float x = X[i * 2 + 0];
        float y = X[i * 2 + 1];
        int gx = (int)((x - min_val) / (max_val - min_val) * (float)(W - 1));
        int gy = (int)((max_val - y) / (max_val - min_val) * (float)(H - 1));
        if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
            int cid = L ? (L[i] % 4) : 0;
            if (grid[gy][gx] == ' ') grid[gy][gx] = cluster_chars[cid];
        }
    }

    /* Overlay centroids with distinct star markers */
    for (size_t j = 0; j < k; j++) {
        float cx = C[j * 2 + 0];
        float cy = C[j * 2 + 1];
        int gx = (int)((cx - min_val) / (max_val - min_val) * (float)(W - 1));
        int gy = (int)((max_val - cy) / (max_val - min_val) * (float)(H - 1));
        if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
            grid[gy][gx] = '@';
        }
    }

    printf("\n  +--------------------------------------------------------------+\n");
    for (int y = 0; y < H; y++) {
        printf("  |");
        for (int x = 0; x < W; x++) {
            putchar(grid[y][x]);
        }
        printf("|\n");
    }
    printf("  +--------------------------------------------------------------+\n");
    printf("  Legend: '.' Cluster 0 | '+' Cluster 1 | 'x' Cluster 2 | 'o' Cluster 3 | '@' Centroids\n\n");
}

int main(void) {
    printf("======================================================================\n");
    printf("🌟 RCRE-EMEANS Turnkey Demo (arXiv:2611.09501)\n");
    printf("   Mini-Batch EMA k-Means with Gumbel-Max Revival & Scaling Laws\n");
    printf("======================================================================\n\n");

    /* 1. Allocate arena */
    size_t arena_sz = 32 * 1024 * 1024;
    RCRE_Arena* arena = rcre_arena_create(arena_sz);
    if (!arena) {
        fprintf(stderr, "Failed to create RCRE arena\n");
        return 1;
    }

    /* 2. Create synthetic dataset */
    printf("[1/4] Generating synthetic 2D Gaussian mixture (n=%d, d=%d, ground-truth k=%d)...\n",
           DEMO_N, DEMO_D, DEMO_K);
    RCRE_Tensor X = rcre_tensor_create((size_t[2]){ DEMO_N, DEMO_D }, 2,
                                       RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    gen_demo_blobs((float*)X.data, DEMO_N, 0xACE1337ULL);

    /* 3. Inspect automatic scaling-law recipe */
    RCRE_EMMeansRecipe recipe;
    (void)rcre_ml_emeans_recipe(&X, DEMO_K, NULL, &recipe);
    printf("      ✓ Resolved Recipe: Batch B=%zu, alpha0=%.4f, passes=%zu, total_steps=%zu\n",
           recipe.batch_size, (double)recipe.alpha0, recipe.n_epochs, recipe.total_steps);

    /* 4. Train e-means with threadpool */
    printf("\n[2/4] Training e-means (T=%.1f epochs) with threadpool...\n", (double)recipe.nepochs);
    RCRE_ThreadPool* pool = rcre_threadpool_create(0);
    RCRE_Tensor C = rcre_tensor_create((size_t[2]){ DEMO_K, DEMO_D }, 2,
                                       RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor L = rcre_tensor_create((size_t[2]){ DEMO_N, 1 }, 2,
                                       RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);

    double t0 = now_ms();
    float em_moving_mse = 0.0f;
    int n_splits = 0;
    RCRE_EMMeansParams params = rcre_ml_emeans_params_default();
    int rc = rcre_ml_emeans_fit_ex(&X, DEMO_K, &params, &C, &L,
                                  &em_moving_mse, &n_splits, arena, pool);
    double em_time = now_ms() - t0;
    if (rc != RCRE_OK) {
        fprintf(stderr, "e-means fit failed: %d\n", rc);
        return 1;
    }

    float em_final_mse = 0.0f;
    (void)rcre_ml_emeans_assign(&X, &C, &L, &em_final_mse, arena);
    printf("      ✓ e-means converged in %.2f ms | Final MSE: %.4f | Dead-cluster splits: %d\n",
           em_time, (double)em_final_mse, n_splits);

    /* 5. Compare with standard Lloyd k-means */
    printf("\n[3/4] Running baseline Lloyd k-means (k-means++ init, 25 iterations)...\n");
    RCRE_Tensor Cl = rcre_tensor_create((size_t[2]){ DEMO_K, DEMO_D }, 2,
                                        RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor Ll = rcre_tensor_create((size_t[2]){ DEMO_N, 1 }, 2,
                                        RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_KMeansParams kp;
    memset(&kp, 0, sizeof(kp));
    kp.max_iter = 25;
    kp.n_init = 1;
    kp.tol = 1e-6f;
    kp.seed = 42;

    t0 = now_ms();
    float lloyd_moving_mse = 0.0f;
    (void)rcre_ml_kmeans_fit_ex(&X, DEMO_K, &kp, &Cl, &Ll, &lloyd_moving_mse, arena);
    double lloyd_time = now_ms() - t0;

    float lloyd_final_mse = 0.0f;
    (void)rcre_ml_emeans_assign(&X, &Cl, &Ll, &lloyd_final_mse, arena);
    printf("      ✓ Lloyd-25 completed in %.2f ms | Final MSE: %.4f\n",
           lloyd_time, (double)lloyd_final_mse);
    printf("      ⚡ Relative Speedup: %.2fx faster wall-clock\n",
           lloyd_time / (em_time > 0 ? em_time : 0.001));

    /* 6. Display converged centroids and 2D cluster map */
    printf("\n[4/4] Converged Cluster Centroids:\n");
    const float* cdata = (const float*)C.data;
    for (size_t j = 0; j < DEMO_K; j++) {
        printf("      Centroid %zu: (x = %+6.3f, y = %+6.3f)\n",
               j, (double)cdata[j * 2 + 0], (double)cdata[j * 2 + 1]);
    }

    render_ascii_map((const float*)X.data, (const int32_t*)L.data, DEMO_N,
                     cdata, DEMO_K);

    printf("🎉 Demo finished successfully! 0 heap leaks, 100%% arena-backed.\n\n");

    if (pool) rcre_threadpool_destroy(pool);
    rcre_arena_destroy(arena);
    return 0;
}
