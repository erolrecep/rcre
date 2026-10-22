/**
 * @file main.c
 * @brief RCRE Lens — Fast, Self-Contained Reverse Image Search Engine CLI
 * 
 * Powered 100% by RCRE Computer Vision, HNSW Proximity Graph, and k-Means Clustering.
 * Pure C23, zero external machine learning dependencies.
 */

#include "rcre.h"
#include "feature_extractor.h"
#include "index_serializer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

#define RCRE_LENS_VERSION "1.1.0"
#define DEFAULT_TOP_K 5
#define MAX_SCANNED_IMAGES 65536
#define ARENA_MEMORY_CAPACITY (128 * 1024 * 1024) /* 128 MB */
#define SCRATCH_ARENA_CAPACITY (32 * 1024 * 1024)  /* 32 MB */

/* High-resolution timer */
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Check if file extension is a supported image format */
static bool is_image_file(const char* filename) {
    if (!filename) return false;
    const char* dot = strrchr(filename, '.');
    if (!dot) return false;

    if (strcasecmp(dot, ".jpg") == 0 ||
        strcasecmp(dot, ".jpeg") == 0 ||
        strcasecmp(dot, ".png") == 0 ||
        strcasecmp(dot, ".bmp") == 0) {
        return true;
    }
    return false;
}

/* Recursively collect all image paths from a directory */
static int scan_directory_images(const char* dir_path, 
                                 char** out_paths, 
                                 size_t* count, 
                                 size_t max_count, 
                                 RCRE_Arena* arena) {
    DIR* dir = opendir(dir_path);
    if (!dir) {
        return RCRE_ERR_IO;
    }

    struct dirent* entry;
    char path_buf[1024];

    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        /* Skip hidden files */
        if (entry->d_name[0] == '.') {
            continue;
        }

        snprintf(path_buf, sizeof(path_buf), "%s/%s", dir_path, entry->d_name);

        struct stat st;
        if (stat(path_buf, &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            scan_directory_images(path_buf, out_paths, count, max_count, arena);
        } else if (S_ISREG(st.st_mode) && is_image_file(entry->d_name)) {
            if (*count < max_count) {
                size_t len = strlen(path_buf) + 1;
                char* copy = (char*)rcre_arena_alloc(arena, len, 1);
                if (copy) {
                    memcpy(copy, path_buf, len);
                    out_paths[(*count)++] = copy;
                }
            }
        }
    }

    closedir(dir);
    return RCRE_OK;
}

/* Render terminal visual match bar */
static void render_similarity_bar(char* out_buf, size_t buf_size, float sim) {
    int total_blocks = 12;
    int filled = (int)(sim * total_blocks + 0.5f);
    if (filled < 0) filled = 0;
    if (filled > total_blocks) filled = total_blocks;

    int pos = 0;
    for (int i = 0; i < filled && pos + 3 < (int)buf_size; ++i) {
        out_buf[pos++] = (char)0xE2; /* UTF-8 for █ (U+2588) */
        out_buf[pos++] = (char)0x96;
        out_buf[pos++] = (char)0x88;
    }
    for (int i = filled; i < total_blocks && pos + 3 < (int)buf_size; ++i) {
        out_buf[pos++] = (char)0xE2; /* UTF-8 for ░ (U+2591) */
        out_buf[pos++] = (char)0x96;
        out_buf[pos++] = (char)0x91;
    }
    out_buf[pos] = '\0';
}

/* Compute visual theme from 64-D spatial pyramid centroid */
static void get_cluster_theme_name(const float* centroid, char* out_buf, size_t buf_size) {
    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f, tex_sum = 0.0f;
    for (int cell = 0; cell < 16; ++cell) {
        r_sum += centroid[cell * 4 + 0];
        g_sum += centroid[cell * 4 + 1];
        b_sum += centroid[cell * 4 + 2];
        tex_sum += centroid[cell * 4 + 3];
    }
    r_sum /= 16.0f; g_sum /= 16.0f; b_sum /= 16.0f; tex_sum /= 16.0f;

    if (r_sum > g_sum * 1.15f && r_sum > b_sum * 1.15f) {
        snprintf(out_buf, buf_size, "Warm Hues / Sunset & Gold");
    } else if (g_sum > r_sum * 1.15f && g_sum > b_sum * 1.15f) {
        snprintf(out_buf, buf_size, "Emerald Foliage & Nature");
    } else if (b_sum > r_sum * 1.15f && b_sum > g_sum * 1.15f) {
        snprintf(out_buf, buf_size, "Deep Indigo / Ocean & Sky");
    } else if (tex_sum > 0.35f) {
        snprintf(out_buf, buf_size, "High-Contrast Pattern & Texture");
    } else {
        snprintf(out_buf, buf_size, "Balanced & Monochromatic");
    }
}

/* Multi-Core ThreadPool Extraction Context */
typedef struct {
    char** image_paths;
    float* embeddings;
    int*   statuses;
} ParallelExtractCtx;

static void parallel_extract_task(void* arg, size_t i) {
    ParallelExtractCtx* ctx = (ParallelExtractCtx*)arg;
    RCRE_Arena* local_scratch = rcre_arena_create(8 * 1024 * 1024);
    if (!local_scratch) {
        ctx->statuses[i] = RCRE_ERR_OUT_OF_MEMORY;
        return;
    }
    float* emb = &ctx->embeddings[i * RCRE_FEATURE_DIM];
    ctx->statuses[i] = rcre_extract_features_file(ctx->image_paths[i], emb, local_scratch);
    rcre_arena_destroy(local_scratch);
}

/* ========================================================================= */
/* Command: Index a directory of images (Multi-Core Accelerated)             */
/* ========================================================================= */
static int cmd_index(const char* image_dir, const char* output_index, size_t num_threads) {
    printf("===================================================================\n");
    printf("🔍 RCRE Lens — Visual Indexer (v%s)\n", RCRE_LENS_VERSION);
    printf("===================================================================\n");
    printf("📂 Target Directory: %s\n", image_dir);
    printf("💾 Output Index:     %s\n", output_index);
    printf("⚡ Parallel Threads: %zu\n\n", num_threads == 0 ? (size_t)sysconf(_SC_NPROCESSORS_ONLN) : num_threads);

    RCRE_Arena* arena = rcre_arena_create(ARENA_MEMORY_CAPACITY);
    if (!arena) {
        fprintf(stderr, "❌ Error: Failed to allocate memory arena.\n");
        return 1;
    }

    /* 1. Discover all images */
    char** image_paths = (char**)rcre_arena_alloc(arena, MAX_SCANNED_IMAGES * sizeof(char*), 64);
    size_t num_images = 0;

    printf("🔎 Scanning directory for image files (.jpg, .png, .bmp)...\n");
    double scan_start = get_time_sec();
    scan_directory_images(image_dir, image_paths, &num_images, MAX_SCANNED_IMAGES, arena);
    double scan_time = get_time_sec() - scan_start;

    if (num_images == 0) {
        fprintf(stderr, "⚠️  No valid image files found in '%s'.\n", image_dir);
        rcre_arena_destroy(arena);
        return 1;
    }

    printf("✓ Found %zu images in %.3f seconds.\n\n", num_images, scan_time);
    printf("⚙️  Multi-Core Feature Extraction (64-D Spatial Pyramids) & HNSW Indexing...\n");

    /* 2. Create in-memory index */
    RCRE_ImageIndex* idx = rcre_index_create(num_images, RCRE_FEATURE_DIM, arena);
    if (!idx) {
        fprintf(stderr, "❌ Error: Failed to create index database.\n");
        rcre_arena_destroy(arena);
        return 1;
    }

    /* 3. Multi-threaded feature extraction */
    double index_start = get_time_sec();
    ParallelExtractCtx ctx = {
        .image_paths = image_paths,
        .embeddings = (float*)rcre_arena_alloc(arena, num_images * RCRE_FEATURE_DIM * sizeof(float), 64),
        .statuses = (int*)rcre_arena_alloc(arena, num_images * sizeof(int), 64)
    };

    RCRE_ThreadPool* pool = rcre_threadpool_create(num_threads);
    if (pool) {
        rcre_threadpool_parallel_for(pool, num_images, parallel_extract_task, &ctx);
        rcre_threadpool_destroy(pool);
    } else {
        /* Single-threaded fallback */
        for (size_t i = 0; i < num_images; ++i) {
            parallel_extract_task(&ctx, i);
        }
    }

    /* 4. Insert into HNSW graph index */
    size_t indexed_success = 0;
    for (size_t i = 0; i < num_images; ++i) {
        if (ctx.statuses[i] == RCRE_OK) {
            int rc = rcre_index_add(idx, image_paths[i], &ctx.embeddings[i * RCRE_FEATURE_DIM]);
            if (rc == RCRE_OK) {
                indexed_success++;
            }
        }
    }

    double total_index_time = get_time_sec() - index_start;
    printf("✓ Extracted & indexed %zu/%zu images in %.2f seconds (%.1f img/s)\n\n",
           indexed_success, num_images, total_index_time, (double)indexed_success / (total_index_time > 1e-4 ? total_index_time : 1e-4));

    /* 5. Serialize to disk */
    printf("💾 Persisting binary index to '%s'...\n", output_index);
    int save_rc = rcre_index_save(idx, output_index);
    if (save_rc != RCRE_OK) {
        fprintf(stderr, "❌ Error saving index file (code %d).\n", save_rc);
        rcre_arena_destroy(arena);
        return 1;
    }

    struct stat st;
    long file_size = 0;
    if (stat(output_index, &st) == 0) {
        file_size = (long)st.st_size;
    }

    printf("===================================================================\n");
    printf("🎉 Index Build Complete!\n");
    printf("  • Indexed Images:    %zu\n", indexed_success);
    printf("  • Feature Dimension: %d-D (Spatial Pyramid + Sobel Energy)\n", RCRE_FEATURE_DIM);
    printf("  • Output File Size:  %.2f KB (%ld bytes)\n", (double)file_size / 1024.0, file_size);
    printf("  • Storage per Image: %.1f bytes/image\n", (double)file_size / (double)(indexed_success > 0 ? indexed_success : 1));
    printf("===================================================================\n");

    rcre_arena_destroy(arena);
    return 0;
}

/* ========================================================================= */
/* Command: Unsupervised Visual Clustering (k-Means++)                       */
/* ========================================================================= */
static int cmd_cluster(const char* index_file, size_t k, bool output_json) {
    RCRE_Arena* arena = rcre_arena_create(ARENA_MEMORY_CAPACITY);
    if (!arena) return 1;

    RCRE_ImageIndex* idx = rcre_index_load(index_file, arena);
    if (!idx || idx->num_images == 0) {
        if (!output_json) fprintf(stderr, "❌ Error: Failed to load index from '%s'\n", index_file);
        rcre_arena_destroy(arena);
        return 1;
    }

    if (k > idx->num_images) k = idx->num_images;
    if (k == 0) k = 1;

    size_t N = idx->num_images;
    size_t D = idx->feature_dim;

    size_t X_shape[2] = {N, D};
    RCRE_Tensor X = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    memcpy(X.data, idx->embeddings, N * D * sizeof(float));

    size_t C_shape[2] = {k, D};
    RCRE_Tensor centroids = rcre_tensor_create(C_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    size_t L_shape[2] = {N, 1};
    RCRE_Tensor labels = rcre_tensor_create(L_shape, 2, RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);

    RCRE_KMeansParams params = {
        .max_iter = 100,
        .n_init = 5,
        .tol = 1e-4f,
        .seed = 42ULL
    };

    float inertia = 0.0f;
    int rc = rcre_ml_kmeans_fit_ex(&X, k, &params, &centroids, &labels, &inertia, arena);
    if (rc != RCRE_OK) {
        if (!output_json) fprintf(stderr, "Clustering failed with code: %d\n", rc);
        rcre_arena_destroy(arena);
        return 1;
    }

    const int32_t* l_data = (const int32_t*)labels.data;
    const float* c_data = (const float*)centroids.data;

    if (output_json) {
        printf("{\n");
        printf("  \"total_images\": %zu,\n", N);
        printf("  \"num_clusters\": %zu,\n", k);
        printf("  \"inertia\": %.4f,\n", inertia);
        printf("  \"clusters\": [\n");
        for (size_t c = 0; c < k; ++c) {
            char theme[64];
            get_cluster_theme_name(&c_data[c * D], theme, sizeof(theme));

            printf("    {\n");
            printf("      \"cluster_id\": %zu,\n", c);
            printf("      \"theme\": \"%s\",\n", theme);
            printf("      \"images\": [\n");

            bool first = true;
            for (size_t i = 0; i < N; ++i) {
                if (l_data[i] == (int32_t)c) {
                    if (!first) printf(",\n");
                    printf("        \"%s\"", idx->filepaths[i]);
                    first = false;
                }
            }
            printf("\n      ]\n");
            printf("    }%s\n", (c + 1 < k) ? "," : "");
        }
        printf("  ]\n");
        printf("}\n");
    } else {
        printf("===================================================================\n");
        printf(" 🔮 RCRE Lens — Unsupervised Visual Clustering (k-Means++)\n");
        printf("===================================================================\n");
        printf("📚 Index:        %s (%zu images, %zu-D features)\n", index_file, N, D);
        printf("🎯 Clusters (K): %zu (WCSS Inertia: %.3f)\n\n", k, inertia);

        for (size_t c = 0; c < k; ++c) {
            char theme[64];
            get_cluster_theme_name(&c_data[c * D], theme, sizeof(theme));

            size_t count = 0;
            for (size_t i = 0; i < N; ++i) {
                if (l_data[i] == (int32_t)c) count++;
            }

            printf("🏷️  Cluster %zu: [%s] (%zu images)\n", c, theme, count);
            size_t shown = 0;
            for (size_t i = 0; i < N && shown < 5; ++i) {
                if (l_data[i] == (int32_t)c) {
                    printf("    • %s\n", idx->filepaths[i]);
                    shown++;
                }
            }
            if (count > shown) printf("    • ... and %zu more\n", count - shown);
            printf("\n");
        }
        printf("===================================================================\n");
    }

    rcre_arena_destroy(arena);
    return 0;
}

/* ========================================================================= */
/* Command: Search visually similar images for a query image                 */
/* ========================================================================= */
static int cmd_search(const char* query_image, const char* index_file, size_t top_k, bool output_json) {
    RCRE_Arena* arena = rcre_arena_create(ARENA_MEMORY_CAPACITY);
    RCRE_Arena* scratch = rcre_arena_create(SCRATCH_ARENA_CAPACITY);
    if (!arena || !scratch) {
        if (!output_json) fprintf(stderr, "❌ Error: Failed to allocate memory arenas.\n");
        return 1;
    }

    /* 1. Load index and reconstruct HNSW graph */
    double load_start = get_time_sec();
    RCRE_ImageIndex* idx = rcre_index_load(index_file, arena);
    double load_time = (get_time_sec() - load_start) * 1000.0;

    if (!idx) {
        if (!output_json) fprintf(stderr, "❌ Error: Failed to load index from '%s'.\n", index_file);
        rcre_arena_destroy(arena);
        rcre_arena_destroy(scratch);
        return 1;
    }

    /* 2. Extract feature vector of query image */
    float query_embedding[RCRE_FEATURE_DIM];
    double feat_start = get_time_sec();
    int feat_rc = rcre_extract_features_file(query_image, query_embedding, scratch);
    double feat_time = (get_time_sec() - feat_start) * 1000.0;

    if (feat_rc != RCRE_OK) {
        if (!output_json) fprintf(stderr, "❌ Error: Failed to load or extract features from '%s'.\n", query_image);
        rcre_arena_destroy(arena);
        rcre_arena_destroy(scratch);
        return 1;
    }

    /* 3. Search top-K nearest neighbors */
    size_t k = top_k > 0 ? top_k : DEFAULT_TOP_K;
    RCRE_SearchResult* results = (RCRE_SearchResult*)rcre_arena_alloc(scratch, k * sizeof(RCRE_SearchResult), 64);

    double search_start = get_time_sec();
    int num_matches = rcre_index_search(idx, query_embedding, k, results, scratch);
    double search_time = (get_time_sec() - search_start) * 1000.0;

    if (num_matches < 0) {
        if (!output_json) fprintf(stderr, "❌ Error: Search execution failed (code %d).\n", num_matches);
        rcre_arena_destroy(arena);
        rcre_arena_destroy(scratch);
        return 1;
    }

    if (output_json) {
        printf("{\n");
        printf("  \"query_image\": \"%s\",\n", query_image);
        printf("  \"feat_extract_ms\": %.3f,\n", feat_time);
        printf("  \"search_time_ms\": %.3f,\n", search_time);
        printf("  \"total_indexed\": %zu,\n", idx->num_images);
        printf("  \"results\": [\n");
        for (int i = 0; i < num_matches; ++i) {
            float sim = results[i].similarity;
            if (sim < 0.0f) sim = 0.0f;
            if (sim > 1.0f) sim = 1.0f;

            printf("    {\"rank\": %d, \"filepath\": \"%s\", \"similarity\": %.4f, \"l2_distance\": %.4f}%s\n",
                   i + 1, results[i].filepath, sim, results[i].l2_distance,
                   (i + 1 < num_matches) ? "," : "");
        }
        printf("  ]\n");
        printf("}\n");
    } else {
        printf("===================================================================\n");
        printf("🔍 RCRE Lens — Visual Query Engine (v%s)\n", RCRE_LENS_VERSION);
        printf("===================================================================\n");
        printf("🖼️  Query Image: %s\n", query_image);
        printf("📚 Target Index: %s\n\n", index_file);
        printf("⚡ Loaded index (%zu images, %zu-D) in %.2f ms\n", idx->num_images, idx->feature_dim, load_time);
        printf("📸 Extracted 64-D query embedding in %.2f ms\n", feat_time);
        printf("\n⏱️  Search Latency: %.3f ms (HNSW Proximity Graph)\n", search_time);
        printf("🏆 Top-%d Visually Similar Results:\n", num_matches);
        printf("--------------------------------------------------------------------------------------------\n");
        printf(" #   Visual Match     Sim %%     L2 Dist   Image Path\n");
        printf("--------------------------------------------------------------------------------------------\n");

        char bar_buf[64];
        for (int i = 0; i < num_matches; ++i) {
            float sim = results[i].similarity;
            if (sim < 0.0f) sim = 0.0f;
            if (sim > 1.0f) sim = 1.0f;

            render_similarity_bar(bar_buf, sizeof(bar_buf), sim);

            printf(" %-2d  %-12s  %6.1f%%    %6.4f    %s\n",
                   i + 1,
                   bar_buf,
                   sim * 100.0f,
                   results[i].l2_distance,
                   results[i].filepath);
        }
        printf("--------------------------------------------------------------------------------------------\n");
    }

    rcre_arena_destroy(arena);
    rcre_arena_destroy(scratch);
    return 0;
}

/* ========================================================================= */
/* Command: List all indexed images in index                                 */
/* ========================================================================= */
static int cmd_list(const char* index_file, bool output_json) {
    RCRE_Arena* arena = rcre_arena_create(ARENA_MEMORY_CAPACITY);
    if (!arena) return 1;

    RCRE_ImageIndex* idx = rcre_index_load(index_file, arena);
    if (!idx) {
        if (!output_json) fprintf(stderr, "❌ Error: Could not read index file '%s'.\n", index_file);
        rcre_arena_destroy(arena);
        return 1;
    }

    if (output_json) {
        printf("{\n");
        printf("  \"total_images\": %zu,\n", idx->num_images);
        printf("  \"feature_dim\": %zu,\n", idx->feature_dim);
        printf("  \"images\": [\n");
        for (size_t i = 0; i < idx->num_images; ++i) {
            printf("    \"%s\"%s\n", idx->filepaths[i], (i + 1 < idx->num_images) ? "," : "");
        }
        printf("  ]\n");
        printf("}\n");
    } else {
        printf("Indexed images (%zu total):\n", idx->num_images);
        for (size_t i = 0; i < idx->num_images; ++i) {
            printf("  [%04zu] %s\n", i + 1, idx->filepaths[i]);
        }
    }

    rcre_arena_destroy(arena);
    return 0;
}

/* ========================================================================= */
/* Command: Inspect index metadata                                           */
/* ========================================================================= */
static int cmd_info(const char* index_file) {
    RCRE_Arena* arena = rcre_arena_create(ARENA_MEMORY_CAPACITY);
    if (!arena) return 1;

    double start = get_time_sec();
    RCRE_ImageIndex* idx = rcre_index_load(index_file, arena);
    double elapsed = (get_time_sec() - start) * 1000.0;

    if (!idx) {
        fprintf(stderr, "❌ Error: Could not read index file '%s'.\n", index_file);
        rcre_arena_destroy(arena);
        return 1;
    }

    struct stat st;
    long file_size = 0;
    if (stat(index_file, &st) == 0) file_size = (long)st.st_size;

    printf("===================================================================\n");
    printf("ℹ️  RCRE Lens Index Metadata: %s\n", index_file);
    printf("===================================================================\n");
    printf("  • Format:            RCRE Binary Index (.rcre_idx v1)\n");
    printf("  • Total Images:      %zu\n", idx->num_images);
    printf("  • Feature Vector:    %zu Dimensions\n", idx->feature_dim);
    printf("  • File Size:         %.2f KB (%ld bytes)\n", (double)file_size / 1024.0, file_size);
    printf("  • Rebuild Latency:   %.2f ms\n", elapsed);
    printf("\n📑 Sample Indexed Files (up to first 10):\n");
    size_t preview_count = idx->num_images < 10 ? idx->num_images : 10;
    for (size_t i = 0; i < preview_count; ++i) {
        printf("   [%04zu] %s\n", i + 1, idx->filepaths[i]);
    }
    if (idx->num_images > preview_count) {
        printf("   ... and %zu more items\n", idx->num_images - preview_count);
    }
    printf("===================================================================\n");

    rcre_arena_destroy(arena);
    return 0;
}

/* ========================================================================= */
/* Command: Self-Contained Demo with Synthetic Sample Images                 */
/* ========================================================================= */
static void write_bmp_image(const char* filepath, int width, int height, 
                           void (*draw_fn)(int x, int y, int w, int h, uint8_t* rgb)) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return;

    int row_padded = (width * 3 + 3) & (~3);
    int image_size = row_padded * height;
    int file_size = 54 + image_size;

    uint8_t header[54] = {
        'B', 'M',
        (uint8_t)(file_size & 0xFF), (uint8_t)((file_size >> 8) & 0xFF),
        (uint8_t)((file_size >> 16) & 0xFF), (uint8_t)((file_size >> 24) & 0xFF),
        0, 0, 0, 0,
        54, 0, 0, 0,
        40, 0, 0, 0,
        (uint8_t)(width & 0xFF), (uint8_t)((width >> 8) & 0xFF),
        (uint8_t)((width >> 16) & 0xFF), (uint8_t)((width >> 24) & 0xFF),
        (uint8_t)(height & 0xFF), (uint8_t)((height >> 8) & 0xFF),
        (uint8_t)((height >> 16) & 0xFF), (uint8_t)((height >> 24) & 0xFF),
        1, 0,
        24, 0,
        0, 0, 0, 0,
        (uint8_t)(image_size & 0xFF), (uint8_t)((image_size >> 8) & 0xFF),
        (uint8_t)((image_size >> 16) & 0xFF), (uint8_t)((image_size >> 24) & 0xFF),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    fwrite(header, 1, 54, fp);

    uint8_t* row = (uint8_t*)calloc(1, row_padded);
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            uint8_t rgb[3];
            draw_fn(x, y, width, height, rgb);
            row[x * 3 + 0] = rgb[2];
            row[x * 3 + 1] = rgb[1];
            row[x * 3 + 2] = rgb[0];
        }
        fwrite(row, 1, row_padded, fp);
    }
    free(row);
    fclose(fp);
}

/* Synthetic Image Generators */
static void draw_red_sun(int x, int y, int w, int h, uint8_t* rgb) {
    int cx = w / 2, cy = h / 2;
    int dx = x - cx, dy = y - cy;
    int r2 = dx * dx + dy * dy;
    if (r2 < 45 * 45) {
        rgb[0] = 240; rgb[1] = 40; rgb[2] = 20; /* Bright Red */
    } else {
        rgb[0] = 245; rgb[1] = 220; rgb[2] = 160; /* Warm Sky */
    }
}

static void draw_ocean(int x, int y, int w, int h, uint8_t* rgb) {
    rgb[0] = 10;
    rgb[1] = (uint8_t)(60 + (y * 80) / h);
    rgb[2] = (uint8_t)(160 + (x * 80) / w);
}

static void draw_forest(int x, int y, int w, int h, uint8_t* rgb) {
    (void)w; (void)h;
    rgb[0] = 25;
    rgb[1] = (uint8_t)(120 + ((x ^ y) % 60));
    rgb[2] = 35;
}

static void draw_checkerboard(int x, int y, int w, int h, uint8_t* rgb) {
    (void)w; (void)h;
    int cell = ((x / 16) + (y / 16)) % 2;
    uint8_t val = cell ? 230 : 25;
    rgb[0] = val; rgb[1] = val; rgb[2] = val;
}

static void draw_purple_neon(int x, int y, int w, int h, uint8_t* rgb) {
    rgb[0] = (uint8_t)(150 + (x * 80) / w);
    rgb[1] = 20;
    rgb[2] = (uint8_t)(180 + (y * 70) / h);
}

static void draw_query_sun(int x, int y, int w, int h, uint8_t* rgb) {
    int cx = w / 2, cy = h / 2;
    int dx = x - cx, dy = y - cy;
    int r2 = dx * dx + dy * dy;
    if (r2 < 42 * 42) {
        rgb[0] = 255; rgb[1] = 50; rgb[2] = 30; /* Red Sun with slight perturbation */
    } else {
        rgb[0] = 240; rgb[1] = 210; rgb[2] = 150;
    }
}

static int cmd_demo(void) {
    printf("===================================================================\n");
    printf("🚀 RCRE Lens Automated End-to-End Demo\n");
    printf("===================================================================\n");
    printf("1. Synthesizing Gallery & Query Images in 'demo_gallery/'...\n");

    const char* demo_dir = "demo_gallery";
    const char* demo_idx = "demo_gallery.rcre_idx";
    mkdir(demo_dir, 0755);

    char path[256];
    snprintf(path, sizeof(path), "%s/01_red_sun.bmp", demo_dir);
    write_bmp_image(path, 160, 120, draw_red_sun);

    snprintf(path, sizeof(path), "%s/02_deep_ocean.bmp", demo_dir);
    write_bmp_image(path, 160, 120, draw_ocean);

    snprintf(path, sizeof(path), "%s/03_pine_forest.bmp", demo_dir);
    write_bmp_image(path, 160, 120, draw_forest);

    snprintf(path, sizeof(path), "%s/04_checkerboard.bmp", demo_dir);
    write_bmp_image(path, 160, 120, draw_checkerboard);

    snprintf(path, sizeof(path), "%s/05_purple_neon.bmp", demo_dir);
    write_bmp_image(path, 160, 120, draw_purple_neon);

    const char* query_img = "demo_query_sun.bmp";
    write_bmp_image(query_img, 160, 120, draw_query_sun);

    printf("  ✓ Created 5 gallery images + 1 query probe image\n\n");

    /* Index the gallery */
    cmd_index(demo_dir, demo_idx, 0);

    printf("\n2. Discovering Visual Clusters (k-Means++, K=3)...\n");
    cmd_cluster(demo_idx, 3, false);

    printf("3. Querying Probe Image (HNSW Similarity Retrieval)...\n");
    cmd_search(query_img, demo_idx, 5, false);

    printf("\n🧹 Demo finished successfully!\n");
    return 0;
}

/* ========================================================================= */
/* Main CLI Dispatcher                                                       */
/* ========================================================================= */
static void print_usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s index   <image_dir> -o <index.rcre_idx> [--threads N]  Index image directory in parallel\n", prog);
    printf("  %s search  <query_img> -i <index.rcre_idx> [-k 5] [--json] Search visually similar images\n", prog);
    printf("  %s cluster -i <index.rcre_idx> [-k 4] [--json]            Unsupervised visual clustering (k-Means++)\n", prog);
    printf("  %s list    -i <index.rcre_idx> [--json]                   List all indexed images\n", prog);
    printf("  %s info    <index.rcre_idx>                               Inspect index metadata\n", prog);
    printf("  %s demo                                                   Run automated synthetic demo\n", prog);
    printf("\nSupported Formats: .jpg, .jpeg, .png, .bmp\n");
}

int main(int argc, char** argv) {
    rcre_hal_init();

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* command = argv[1];

    if (strcmp(command, "index") == 0) {
        if (argc < 5 || strcmp(argv[3], "-o") != 0) {
            fprintf(stderr, "Usage: %s index <image_dir> -o <output_index.rcre_idx> [--threads N]\n", argv[0]);
            return 1;
        }
        size_t threads = 0; /* Auto */
        if (argc >= 7 && strcmp(argv[5], "--threads") == 0) {
            threads = (size_t)atoi(argv[6]);
        }
        return cmd_index(argv[2], argv[4], threads);
    } else if (strcmp(command, "search") == 0) {
        if (argc < 5 || strcmp(argv[3], "-i") != 0) {
            fprintf(stderr, "Usage: %s search <query_img> -i <index.rcre_idx> [-k 5] [--json]\n", argv[0]);
            return 1;
        }
        size_t top_k = DEFAULT_TOP_K;
        bool json = false;
        for (int i = 5; i < argc; ++i) {
            if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) top_k = (size_t)atoi(argv[++i]);
            if (strcmp(argv[i], "--json") == 0) json = true;
        }
        return cmd_search(argv[2], argv[4], top_k, json);
    } else if (strcmp(command, "cluster") == 0) {
        if (argc < 4 || strcmp(argv[2], "-i") != 0) {
            fprintf(stderr, "Usage: %s cluster -i <index.rcre_idx> [-k 4] [--json]\n", argv[0]);
            return 1;
        }
        size_t k = 4;
        bool json = false;
        for (int i = 4; i < argc; ++i) {
            if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) k = (size_t)atoi(argv[++i]);
            if (strcmp(argv[i], "--json") == 0) json = true;
        }
        return cmd_cluster(argv[3], k, json);
    } else if (strcmp(command, "list") == 0) {
        if (argc < 4 || strcmp(argv[2], "-i") != 0) {
            fprintf(stderr, "Usage: %s list -i <index.rcre_idx> [--json]\n", argv[0]);
            return 1;
        }
        bool json = (argc >= 5 && strcmp(argv[4], "--json") == 0);
        return cmd_list(argv[3], json);
    } else if (strcmp(command, "info") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s info <index.rcre_idx>\n", argv[0]);
            return 1;
        }
        return cmd_info(argv[2]);
    } else if (strcmp(command, "demo") == 0) {
        return cmd_demo();
    } else if (strcmp(command, "--help") == 0 || strcmp(command, "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    } else {
        fprintf(stderr, "Unknown command: '%s'\n\n", command);
        print_usage(argv[0]);
        return 1;
    }
}
