/**
 * @file manifold_animate.c
 * @brief Interactive Terminal Animation & Pedagogical Timeline Visualizer
 *
 * Demonstrates the step-by-step convergence and topological untangling of
 * high-dimensional datasets (10D Gaussians, 3D Swiss Roll, 3D Trefoil Knot)
 * using UMAP, PaCMAP, or Barnes-Hut t-SNE in pure ANSI terminal graphics.
 *
 * Standard: ISO C23
 */

#include "rcre_manifold.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#define GRID_W 74
#define GRID_H 24
#define MAX_SAMPLES 400

/* ANSI Color Palette */
#define COLOR_RESET   "\033[0m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_DIM     "\033[2m"
#define COLOR_CYAN    "\033[1;36m"
#define COLOR_GREEN   "\033[1;32m"
#define COLOR_YELLOW  "\033[1;33m"
#define COLOR_MAGENTA "\033[1;35m"
#define COLOR_BLUE    "\033[1;34m"
#define COLOR_RED     "\033[1;31m"
#define COLOR_WHITE   "\033[1;37m"

static const char* CLUSTER_COLORS[4] = {
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_MAGENTA
};

static const char CLUSTER_CHARS[4] = {'@', '#', '*', '+'};
static const char* CLUSTER_NAMES[4] = {"Alpha", "Beta ", "Gamma", "Delta"};

typedef enum {
    DATASET_BLOBS_10D = 0,
    DATASET_SWISS_ROLL = 1,
    DATASET_TREFOIL_KNOT = 2
} DatasetType;

/* PRNG Helpers */
static inline uint64_t anim_rng_next(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float anim_uniform(uint64_t* state, float lo, float hi) {
    double u = (double)(anim_rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
    return lo + (float)(u * (hi - lo));
}

static inline float anim_gaussian(uint64_t* state, float mean, float std) {
    float u1 = anim_uniform(state, 1e-7f, 1.0f);
    float u2 = anim_uniform(state, 0.0f, 1.0f);
    return mean + sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * std;
}

/* Dataset Generation */
static void generate_dataset(DatasetType type, float* X, int* labels, size_t n, size_t* out_dim) {
    uint64_t rng = 13374299ULL;

    if (type == DATASET_BLOBS_10D) {
        *out_dim = 10;
        size_t per = n / 4;
        float centers[4][10] = {
            { 4.0f,  4.0f,  2.0f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
            {-4.0f,  4.0f, -2.0f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
            {-4.0f, -4.0f,  0.0f, -2.0f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f},
            { 4.0f, -4.0f,  1.0f,  0.0f, -2.0f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f}
        };
        for (size_t c = 0; c < 4; ++c) {
            for (size_t i = 0; i < per; ++i) {
                size_t idx = c * per + i;
                labels[idx] = (int)c;
                for (size_t d = 0; d < 10; ++d) {
                    X[idx * 10 + d] = centers[c][d] + anim_gaussian(&rng, 0.0f, 0.75f);
                }
            }
        }
    } else if (type == DATASET_SWISS_ROLL) {
        *out_dim = 3;
        for (size_t i = 0; i < n; ++i) {
            float t = 1.5f * (float)M_PI * (1.0f + 2.0f * anim_uniform(&rng, 0.0f, 1.0f));
            float h = anim_uniform(&rng, 0.0f, 20.0f);
            X[i * 3 + 0] = t * cosf(t);
            X[i * 3 + 1] = h;
            X[i * 3 + 2] = t * sinf(t);

            /* Partition roll into 4 progressive segment bands */
            float norm_t = (t - 1.5f * (float)M_PI) / (3.0f * (float)M_PI);
            int band = (int)(norm_t * 4.0f);
            if (band < 0) band = 0;
            if (band > 3) band = 3;
            labels[i] = band;
        }
    } else if (type == DATASET_TREFOIL_KNOT) {
        *out_dim = 3;
        for (size_t i = 0; i < n; ++i) {
            float t = ((float)i / (float)n) * 2.0f * (float)M_PI;
            float r = 2.0f + cosf(3.0f * t);
            X[i * 3 + 0] = r * cosf(2.0f * t) + anim_gaussian(&rng, 0.0f, 0.1f);
            X[i * 3 + 1] = r * sinf(2.0f * t) + anim_gaussian(&rng, 0.0f, 0.1f);
            X[i * 3 + 2] = sinf(3.0f * t) + anim_gaussian(&rng, 0.0f, 0.1f);

            int seg = (int)((t / (2.0f * (float)M_PI)) * 4.0f);
            if (seg < 0) seg = 0;
            if (seg > 3) seg = 3;
            labels[i] = seg;
        }
    }
}

/* Cluster separation ratio metric */
static float compute_separation_ratio(const float* Y, const int* labels, size_t n) {
    float centers[4][2] = {{0}};
    int counts[4] = {0};

    for (size_t i = 0; i < n; ++i) {
        int c = labels[i] % 4;
        centers[c][0] += Y[i * 2 + 0];
        centers[c][1] += Y[i * 2 + 1];
        counts[c]++;
    }
    for (int c = 0; c < 4; ++c) {
        if (counts[c] > 0) {
            centers[c][0] /= (float)counts[c];
            centers[c][1] /= (float)counts[c];
        }
    }

    float intra_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        int c = labels[i] % 4;
        float dx = Y[i * 2 + 0] - centers[c][0];
        float dy = Y[i * 2 + 1] - centers[c][1];
        intra_sum += sqrtf(dx * dx + dy * dy);
    }
    float intra_mean = intra_sum / (float)n;

    float inter_sum = 0.0f;
    int pair_count = 0;
    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            float dx = centers[a][0] - centers[b][0];
            float dy = centers[a][1] - centers[b][1];
            inter_sum += sqrtf(dx * dx + dy * dy);
            pair_count++;
        }
    }
    float inter_mean = (pair_count > 0) ? (inter_sum / (float)pair_count) : 1.0f;
    return inter_mean / (intra_mean + 1e-5f);
}

/* Render animated frame */
static void render_frame(const float* Y, const int* labels, size_t n,
                         size_t current_epoch, size_t total_epochs,
                         float loss, float sep_ratio,
                         RCRE_ManifoldMethod method, DatasetType data_type) {
    char grid_char[GRID_H][GRID_W];
    int  grid_color[GRID_H][GRID_W];

    for (int y = 0; y < GRID_H; ++y) {
        for (int x = 0; x < GRID_W; ++x) {
            grid_char[y][x] = ' ';
            grid_color[y][x] = -1;
        }
    }

    /* Bounding box */
    float min_x = 1e9f, max_x = -1e9f;
    float min_y = 1e9f, max_y = -1e9f;
    for (size_t i = 0; i < n; ++i) {
        if (Y[i * 2 + 0] < min_x) min_x = Y[i * 2 + 0];
        if (Y[i * 2 + 0] > max_x) max_x = Y[i * 2 + 0];
        if (Y[i * 2 + 1] < min_y) min_y = Y[i * 2 + 1];
        if (Y[i * 2 + 1] > max_y) max_y = Y[i * 2 + 1];
    }
    float range_x = (max_x - min_x > 1e-4f) ? (max_x - min_x) : 1.0f;
    float range_y = (max_y - min_y > 1e-4f) ? (max_y - min_y) : 1.0f;

    for (size_t i = 0; i < n; ++i) {
        int gx = (int)(((Y[i * 2 + 0] - min_x) / range_x) * (GRID_W - 3)) + 1;
        int gy = (int)(((Y[i * 2 + 1] - min_y) / range_y) * (GRID_H - 3)) + 1;
        if (gx < 1) gx = 1;
        if (gx >= GRID_W - 1) gx = GRID_W - 2;
        if (gy < 1) gy = 1;
        if (gy >= GRID_H - 1) gy = GRID_H - 2;

        int c = labels[i] % 4;
        grid_char[gy][gx] = CLUSTER_CHARS[c];
        grid_color[gy][gx] = c;
    }

    /* Move cursor home and draw */
    printf("\033[H");

    /* Header banner */
    printf(COLOR_BOLD COLOR_WHITE "╔════════════════════════════════════════════════════════════════════════════╗\n" COLOR_RESET);
    const char* method_str = (method == RCRE_MANIFOLD_UMAP) ? "UMAP" :
                             (method == RCRE_MANIFOLD_PACMAP) ? "PaCMAP" :
                             (method == RCRE_MANIFOLD_TSNE) ? "Barnes-Hut t-SNE" : "PCA";

    const char* data_str = (data_type == DATASET_BLOBS_10D) ? "10D Gaussian Blobs (4 Clusters)" :
                           (data_type == DATASET_SWISS_ROLL) ? "3D Swiss Roll (Unrolling Manifold)" :
                           "3D Trefoil Knot (Topological Disentanglement)";

    printf(COLOR_BOLD "║  🔬 RCRE Manifold Teaching Laboratory — Interactive Timeline Visualizer   ║\n" COLOR_RESET);
    printf("║  Algorithm: " COLOR_CYAN "%-16s" COLOR_RESET " │ Dataset: " COLOR_YELLOW "%-33s" COLOR_RESET "║\n", method_str, data_str);
    printf("╠════════════════════════════════════════════════════════════════════════════╣\n");

    /* Render 2D Grid */
    for (int y = 0; y < GRID_H; ++y) {
        printf("║ ");
        for (int x = 0; x < GRID_W; ++x) {
            int c = grid_color[y][x];
            if (c >= 0) {
                printf("%s%c" COLOR_RESET, CLUSTER_COLORS[c], grid_char[y][x]);
            } else {
                putchar(grid_char[y][x]);
            }
        }
        printf(" ║\n");
    }

    printf("╠════════════════════════════════════════════════════════════════════════════╣\n");

    /* Progress bar */
    float pct = (total_epochs > 0) ? ((float)current_epoch / (float)total_epochs) : 1.0f;
    int bar_width = 30;
    int filled = (int)(pct * bar_width);
    if (filled > bar_width) filled = bar_width;

    printf("║ Progress: [");
    printf(COLOR_GREEN);
    for (int b = 0; b < filled; ++b) putchar('#');
    printf(COLOR_DIM);
    for (int b = filled; b < bar_width; ++b) putchar('-');
    printf(COLOR_RESET);
    printf("] %3.0f%% │ Epoch: " COLOR_BOLD "%4zu/%-4zu" COLOR_RESET " │ Step Loss: %6.3f ║\n",
           pct * 100.0f, current_epoch, total_epochs, loss);

    /* Cluster separation metric HUD */
    const char* sep_eval = (sep_ratio > 5.0f) ? COLOR_GREEN "Outstanding Separation" COLOR_RESET :
                           (sep_ratio > 2.0f) ? COLOR_YELLOW "Emerging Boundaries" COLOR_RESET :
                           COLOR_RED "Jumbled / Overlapping" COLOR_RESET;

    printf("║ Separation Ratio: " COLOR_CYAN "%5.2fx" COLOR_RESET " (%s) │ Points: " COLOR_WHITE "%zu" COLOR_RESET "             ║\n",
           sep_ratio, sep_eval, n);

    /* Pedagogical Phase Insight */
    const char* phase_insight = "";
    if (method == RCRE_MANIFOLD_UMAP) {
        if (pct < 0.20f) {
            phase_insight = "Phase 1: High Learning Rate — Coarse Global Untangling of Macro Nodes";
        } else if (pct < 0.70f) {
            phase_insight = "Phase 2: Negative Sampling Repulsion — Inter-Cluster Voids Opening";
        } else {
            phase_insight = "Phase 3: Smooth Simplicial Annealing — Local Neighborhood Fine Geometry";
        }
    } else if (method == RCRE_MANIFOLD_PACMAP) {
        if (current_epoch < 100) {
            phase_insight = "Stage 1: Strong Mid-Near Force (w_MN=2.0) — Establishing Global Layout";
        } else if (current_epoch < 200) {
            phase_insight = "Stage 2: Mid-Near Annealing Transition (w_MN decays 2.0 -> 0.0)";
        } else {
            phase_insight = "Stage 3: Local Neighbor Focus (w_NB=1.0) & Random Repulsion (w_FP=1.0)";
        }
    } else if (method == RCRE_MANIFOLD_TSNE) {
        if (current_epoch < 200) {
            phase_insight = "Early Exaggeration Phase (x12): Dense Clustered Protoplanets Form";
        } else {
            phase_insight = "Cauchy Student-t Heavy Tail Relaxation: Balanced Repulsive Field";
        }
    } else {
        phase_insight = "Orthogonal Eigenvector Projection: Closed-Form SVD Global Max-Variance";
    }

    printf("║ Teaching Note: " COLOR_YELLOW "%-59s" COLOR_RESET "║\n", phase_insight);

    /* Legend */
    printf("║ Legend: ");
    for (int c = 0; c < 4; ++c) {
        printf("[%s%c" COLOR_RESET "] %s%-6s" COLOR_RESET " ", CLUSTER_COLORS[c], CLUSTER_CHARS[c], CLUSTER_COLORS[c], CLUSTER_NAMES[c]);
    }
    printf("                 ║\n");
    printf(COLOR_BOLD COLOR_WHITE "╚════════════════════════════════════════════════════════════════════════════╝\n" COLOR_RESET);
    fflush(stdout);
}

int main(int argc, char** argv) {
    RCRE_ManifoldMethod method = RCRE_MANIFOLD_UMAP;
    DatasetType data_type = DATASET_BLOBS_10D;
    int step_size = 4;
    useconds_t frame_delay_us = 35000; /* ~28 FPS */
    bool interactive_step = false;

    /* Parse CLI arguments */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--pacmap") == 0) method = RCRE_MANIFOLD_PACMAP;
        else if (strcmp(argv[i], "--tsne") == 0) method = RCRE_MANIFOLD_TSNE;
        else if (strcmp(argv[i], "--pca") == 0) method = RCRE_MANIFOLD_PCA;
        else if (strcmp(argv[i], "--umap") == 0) method = RCRE_MANIFOLD_UMAP;
        else if (strcmp(argv[i], "--swiss") == 0) data_type = DATASET_SWISS_ROLL;
        else if (strcmp(argv[i], "--trefoil") == 0) data_type = DATASET_TREFOIL_KNOT;
        else if (strcmp(argv[i], "--blobs") == 0) data_type = DATASET_BLOBS_10D;
        else if (strcmp(argv[i], "--step") == 0) interactive_step = true;
        else if (strcmp(argv[i], "--fast") == 0) { step_size = 8; frame_delay_us = 20000; }
        else if (strcmp(argv[i], "--slow") == 0) { step_size = 2; frame_delay_us = 60000; }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("Options:\n");
            printf("  --umap       Run UMAP algorithm (default)\n");
            printf("  --pacmap     Run PaCMAP 3-tier algorithm\n");
            printf("  --tsne       Run Barnes-Hut t-SNE algorithm\n");
            printf("  --pca        Run Orthogonal PCA baseline\n");
            printf("  --blobs      10D Gaussian Blobs (default)\n");
            printf("  --swiss      3D Swiss Roll unrolling\n");
            printf("  --trefoil    3D Trefoil Knot untangling\n");
            printf("  --step       Interactive single-step mode (press Enter to advance)\n");
            printf("  --fast       Faster animation speed\n");
            printf("  --slow       Slower animation speed for teaching lectures\n");
            return 0;
        }
    }

    rcre_hal_init();

    RCRE_Arena* arena = rcre_arena_create(32 * 1024 * 1024);
    if (!arena) {
        fprintf(stderr, "Failed to allocate memory arena\n");
        return 1;
    }

    size_t n_samples = (data_type == DATASET_BLOBS_10D) ? 320 : 360;
    size_t in_dim = 10;
    float* X = (float*)rcre_arena_alloc(arena, n_samples * 10 * sizeof(float), 64);
    int* labels = (int*)rcre_arena_alloc(arena, n_samples * sizeof(int), 64);

    generate_dataset(data_type, X, labels, n_samples, &in_dim);

    RCRE_ManifoldParams p = rcre_manifold_params_default(method);
    if (data_type == DATASET_SWISS_ROLL || data_type == DATASET_TREFOIL_KNOT) {
        p.n_neighbors = 12;
    }

    RCRE_ManifoldContext* ctx = rcre_manifold_create(X, n_samples, in_dim, &p, arena);
    if (!ctx) {
        fprintf(stderr, "Failed to create manifold context\n");
        rcre_arena_destroy(arena);
        return 1;
    }

    /* Hide terminal cursor and clear screen */
    printf("\033[?25l\033[2J");

    size_t total_epochs = rcre_manifold_get_total_epochs(ctx);

    /* Render initial state (Epoch 0) */
    const float* Y = rcre_manifold_get_coords(ctx);
    float sep = compute_separation_ratio(Y, labels, n_samples);
    render_frame(Y, labels, n_samples, 0, total_epochs, 0.0f, sep, method, data_type);

    if (interactive_step) {
        printf("\n" COLOR_BOLD ">>> Press ENTER to step 10 epochs (or Ctrl+C to exit)... " COLOR_RESET);
        getchar();
    } else {
        usleep(frame_delay_us * 5);
    }

    /* Animation Loop */
    while (rcre_manifold_get_epoch(ctx) < total_epochs) {
        for (int s = 0; s < step_size && rcre_manifold_get_epoch(ctx) < total_epochs; ++s) {
            rcre_manifold_step(ctx);
        }

        size_t current_epoch = rcre_manifold_get_epoch(ctx);
        float loss = rcre_manifold_get_loss(ctx);
        sep = compute_separation_ratio(Y, labels, n_samples);

        render_frame(Y, labels, n_samples, current_epoch, total_epochs, loss, sep, method, data_type);

        if (interactive_step) {
            printf("\n" COLOR_BOLD ">>> Epoch %zu: Press ENTER to advance... " COLOR_RESET, current_epoch);
            getchar();
        } else {
            usleep(frame_delay_us);
        }
    }

    /* Final converged view */
    render_frame(Y, labels, n_samples, total_epochs, total_epochs, rcre_manifold_get_loss(ctx),
                 compute_separation_ratio(Y, labels, n_samples), method, data_type);

    /* Restore terminal cursor */
    printf("\033[?25h\n");
    printf(COLOR_BOLD COLOR_GREEN "✅ Manifold Timeline Convergence Completed Successfully!\n" COLOR_RESET);

    rcre_manifold_destroy(ctx);
    rcre_arena_destroy(arena);
    return 0;
}
