/**
 * @file manifold_demo.c
 * @brief High-Dimensional Cluster Projection & ASCII Scatter Demo
 *
 * Demonstrates RCRE Manifold untangling 10-dimensional Gaussian clusters
 * into a 2D terminal scatter plot using UMAP, t-SNE, and PaCMAP.
 */

#include "rcre_manifold.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N_SAMPLES 320
#define IN_DIM    10
#define N_CLUSTERS 4
#define GRID_W    68
#define GRID_H    22

static const char CLUSTER_SYMBOLS[N_CLUSTERS] = {'@', '#', '*', '+'};
static const char* CLUSTER_NAMES[N_CLUSTERS] = {"Alpha", "Beta", "Gamma", "Delta"};

/* Generate 10-D synthetic Gaussian mixture */
static void generate_gaussian_mixture(float* X, int* labels, size_t n, size_t dim) {
    uint64_t rng = 987654321ULL;
    size_t per_cluster = n / N_CLUSTERS;

    float centers[N_CLUSTERS][10] = {
        { 4.0f,  4.0f,  2.0f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
        {-4.0f,  4.0f, -2.0f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
        {-4.0f, -4.0f,  0.0f, -2.0f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f},
        { 4.0f, -4.0f,  1.0f,  0.0f, -2.0f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f}
    };

    for (size_t c = 0; c < N_CLUSTERS; ++c) {
        for (size_t i = 0; i < per_cluster; ++i) {
            size_t idx = c * per_cluster + i;
            labels[idx] = (int)c;
            for (size_t d = 0; d < dim; ++d) {
                /* Box-Muller normal */
                uint64_t r1 = rng;
                rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; rng *= 0x2545F4914F6CDD1DULL;
                uint64_t r2 = rng;
                rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; rng *= 0x2545F4914F6CDD1DULL;
                double u1 = (double)(r1 >> 11) * (1.0 / 9007199254740992.0) + 1e-7;
                double u2 = (double)(r2 >> 11) * (1.0 / 9007199254740992.0);
                float noise = (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));

                X[idx * dim + d] = centers[c][d] + noise * 0.75f;
            }
        }
    }
}

/* Render 2D coordinates to ASCII grid */
static void render_ascii_scatter(const float* Y, const int* labels, size_t n, const char* title) {
    char grid[GRID_H][GRID_W];
    for (int y = 0; y < GRID_H; ++y) {
        for (int x = 0; x < GRID_W; ++x) {
            grid[y][x] = ' ';
        }
    }

    /* Find bounding box */
    float min_x = 1e9f, max_x = -1e9f;
    float min_y = 1e9f, max_y = -1e9f;
    for (size_t i = 0; i < n; ++i) {
        if (Y[i * 2 + 0] < min_x) min_x = Y[i * 2 + 0];
        if (Y[i * 2 + 0] > max_x) max_x = Y[i * 2 + 0];
        if (Y[i * 2 + 1] < min_y) min_y = Y[i * 2 + 1];
        if (Y[i * 2 + 1] > max_y) max_y = Y[i * 2 + 1];
    }

    float range_x = (max_x - min_x > 1e-5f) ? (max_x - min_x) : 1.0f;
    float range_y = (max_y - min_y > 1e-5f) ? (max_y - min_y) : 1.0f;

    for (size_t i = 0; i < n; ++i) {
        int gx = (int)(((Y[i * 2 + 0] - min_x) / range_x) * (GRID_W - 3)) + 1;
        int gy = (int)(((Y[i * 2 + 1] - min_y) / range_y) * (GRID_H - 3)) + 1;
        if (gx < 1) gx = 1;
        if (gx >= GRID_W - 1) gx = GRID_W - 2;
        if (gy < 1) gy = 1;
        if (gy >= GRID_H - 1) gy = GRID_H - 2;

        int c = labels[i];
        grid[gy][gx] = CLUSTER_SYMBOLS[c % N_CLUSTERS];
    }

    printf("\n┌────────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-66s │\n", title);
    printf("├────────────────────────────────────────────────────────────────────┤\n");
    for (int y = 0; y < GRID_H; ++y) {
        printf("│");
        for (int x = 0; x < GRID_W; ++x) {
            putchar(grid[y][x]);
        }
        printf("│\n");
    }
    printf("└────────────────────────────────────────────────────────────────────┘\n");
    printf("Clusters: ");
    for (int c = 0; c < N_CLUSTERS; ++c) {
        printf("[%c] %s   ", CLUSTER_SYMBOLS[c], CLUSTER_NAMES[c]);
    }
    printf("\n");
}

int main(void) {
    printf("====================================================================\n");
    printf("  RCRE Manifold Learning & Dimension Reduction Engine Demo          \n");
    printf("  10-Dimensional Gaussian Mixture Projection (320 Samples, 4 Blobs) \n");
    printf("====================================================================\n");

    rcre_hal_init();

    RCRE_Arena* arena = rcre_arena_create(32 * 1024 * 1024);
    if (!arena) {
        fprintf(stderr, "Failed to allocate RCRE arena\n");
        return 1;
    }

    float* X = (float*)rcre_arena_alloc(arena, N_SAMPLES * IN_DIM * sizeof(float), 64);
    int* labels = (int*)rcre_arena_alloc(arena, N_SAMPLES * sizeof(int), 64);
    generate_gaussian_mixture(X, labels, N_SAMPLES, IN_DIM);

    /* 1. UMAP Projection */
    printf("\n[1/3] Running UMAP Optimization (400 epochs, k=15, min_dist=0.1)...\n");
    RCRE_ManifoldParams umap_p = rcre_manifold_params_default(RCRE_MANIFOLD_UMAP);
    umap_p.n_epochs = 400;
    RCRE_ManifoldContext* ctx_umap = rcre_manifold_create(X, N_SAMPLES, IN_DIM, &umap_p, arena);
    if (ctx_umap) {
        rcre_manifold_fit(ctx_umap);
        const float* Y_umap = rcre_manifold_get_coords(ctx_umap);
        render_ascii_scatter(Y_umap, labels, N_SAMPLES, "UMAP 2D Projection (Preserving Neighborhood Topologies)");
        rcre_manifold_destroy(ctx_umap);
    }

    /* 2. PaCMAP Projection */
    printf("\n[2/3] Running PaCMAP 3-Tier Optimization (350 epochs)...\n");
    RCRE_ManifoldParams pacmap_p = rcre_manifold_params_default(RCRE_MANIFOLD_PACMAP);
    pacmap_p.n_epochs = 350;
    RCRE_ManifoldContext* ctx_pac = rcre_manifold_create(X, N_SAMPLES, IN_DIM, &pacmap_p, arena);
    if (ctx_pac) {
        rcre_manifold_fit(ctx_pac);
        const float* Y_pac = rcre_manifold_get_coords(ctx_pac);
        render_ascii_scatter(Y_pac, labels, N_SAMPLES, "PaCMAP 2D Projection (Preserving Local & Global Structure)");
        rcre_manifold_destroy(ctx_pac);
    }

    /* 3. t-SNE Projection */
    printf("\n[3/3] Running Barnes-Hut t-SNE (500 epochs, perp=25)...\n");
    RCRE_ManifoldParams tsne_p = rcre_manifold_params_default(RCRE_MANIFOLD_TSNE);
    tsne_p.n_epochs = 500;
    RCRE_ManifoldContext* ctx_tsne = rcre_manifold_create(X, N_SAMPLES, IN_DIM, &tsne_p, arena);
    if (ctx_tsne) {
        rcre_manifold_fit(ctx_tsne);
        const float* Y_tsne = rcre_manifold_get_coords(ctx_tsne);
        render_ascii_scatter(Y_tsne, labels, N_SAMPLES, "Barnes-Hut t-SNE 2D Projection (Student-t Low-D Cauchy)");
        rcre_manifold_destroy(ctx_tsne);
    }

    rcre_arena_destroy(arena);
    printf("\n✅ Manifold Demonstration Complete!\n");
    return 0;
}
