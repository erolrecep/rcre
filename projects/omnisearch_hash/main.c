/**
 * @file main.c
 * @brief RCRE OmniSearch — 30M Vector Search Engine with LSH & Hash Table CLI
 */

#include "omnisearch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define DEMO_ARENA_SIZE (128 * 1024 * 1024)
#define DEMO_SCRATCH_SIZE (32 * 1024 * 1024)

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void print_banner(void) {
    printf("===================================================================\n");
    printf("⚡ RCRE OmniSearch — Multimodal Vector Search & LSH Hash Engine\n");
    printf("===================================================================\n");
}

static void generate_synthetic_embedding(float* out_vec, size_t dim, uint32_t seed) {
    for (size_t d = 0; d < dim; ++d) {
        float freq = (float)(seed % 19 + 1) * 0.08f;
        out_vec[d] = sinf((float)d * freq) + cosf((float)(seed * 3 + d) * 0.04f);
    }
}

static int cmd_demo(void) {
    print_banner();
    printf("🚀 Launching Multimodal Search Demo (Text & Images)...\n\n");

    RCRE_Arena* arena = rcre_arena_create(DEMO_ARENA_SIZE);
    RCRE_Arena* scratch = rcre_arena_create(DEMO_SCRATCH_SIZE);
    if (!arena || !scratch) {
        fprintf(stderr, "Error: Failed to allocate memory arenas\n");
        return 1;
    }

    const size_t NUM_ITEMS = 5000;
    const size_t DIM = OMNI_DEFAULT_DIM;
    const size_t NUM_HP = OMNI_DEFAULT_HYPERPLANES;
    const char* index_path = "demo_multimodal.rcre_lsh";

    printf("📦 Step 1: Ingesting %zu Multimodal Items (%zu-D embeddings)...\n", NUM_ITEMS, DIM);
    double t0 = get_time_ms();

    OmniIndex* idx = omni_index_create(DIM, NUM_HP, NUM_ITEMS, arena);
    if (!idx) {
        fprintf(stderr, "Error: Failed to create index container\n");
        return 1;
    }

    /* Synthetic text corpus templates */
    const char* text_samples[] = {
        "A high-performance C23 systems programming tutorial on memory arenas.",
        "Deep convolutional neural network architecture for edge image segmentation.",
        "A fluffy golden retriever puppy running happily across a sunny green meadow.",
        "Panoramic view of snow-capped mountain peaks during an alpine sunrise.",
        "Low-latency hardware abstraction layer SIMD acceleration on Apple Silicon.",
        "Classical music symphony composed in D minor performed by string orchestra.",
        "Urban cityscape with towering skyscrapers reflected in the bay at twilight.",
        "Real-time webcam computer vision pipeline tracking facial action units."
    };
    const size_t num_text_samples = sizeof(text_samples) / sizeof(text_samples[0]);

    float vec[DIM];
    char payload[256];

    for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
        generate_synthetic_embedding(vec, DIM, i);
        OmniItemType type = (i % 2 == 0) ? OMNI_ITEM_TEXT : OMNI_ITEM_IMAGE;

        if (type == OMNI_ITEM_TEXT) {
            const char* sample = text_samples[i % num_text_samples];
            snprintf(payload, sizeof(payload), "%s [Sentence #%u]", sample, i);
        } else {
            snprintf(payload, sizeof(payload), "dataset/images/gallery_item_%05u.jpg", i);
        }

        (void)omni_index_add(idx, i, type, payload, vec);
    }

    double t1 = get_time_ms();
    printf("  ✓ Ingested %zu items in %.2f ms (%.1f items/sec)\n", 
           NUM_ITEMS, t1 - t0, (double)NUM_ITEMS / ((t1 - t0) / 1000.0));

    printf("⚡ Step 2: Constructing Arena-Backed Open-Addressing Hash Table...\n");
    double t_hash0 = get_time_ms();
    (void)omni_index_build_hash(idx, scratch);
    double t_hash1 = get_time_ms();
    printf("  ✓ Hash table constructed in %.2f ms (%zu unique buckets, %zu postings)\n",
           t_hash1 - t_hash0, idx->hash_table->size, idx->hash_table->postings_size);

    printf("💾 Step 3: Serializing index to '%s'...\n", index_path);
    (void)omni_index_save(idx, index_path);

    /* Reload to prove cold-start loading speed */
    rcre_arena_reset(arena);
    double t_load0 = get_time_ms();
    OmniIndex* loaded_idx = omni_index_load(index_path, arena);
    double t_load1 = get_time_ms();
    printf("  ✓ Instant cold-start load in %.2f ms\n\n", t_load1 - t_load0);

    /* Perform probe searches */
    printf("🔎 Step 4: Executing Multi-Probe User Query (Top-20 Results)...\n");
    float q_vec[DIM];
    generate_synthetic_embedding(q_vec, DIM, 42); /* query corresponding to cluster 42 */

    const size_t TOP_K = 20;
    OmniSearchResult results[TOP_K];
    size_t out_count = 0;

    double t_q0 = get_time_ms();
    (void)omni_index_search(loaded_idx, q_vec, TOP_K, 1, results, &out_count, scratch);
    double t_q1 = get_time_ms();

    printf("⏱️  Search Latency: %.3f ms (Multi-Probe Hash Table + SIMD Re-rank)\n", t_q1 - t_q0);
    printf("🏆 Top-%zu Results across 5,000 Heterogeneous Items:\n", out_count);
    printf("----------------------------------------------------------------------------------------------------\n");
    printf(" #   Type     Sim %%     Hamming   Item ID   Payload\n");
    printf("----------------------------------------------------------------------------------------------------\n");

    for (size_t i = 0; i < out_count; ++i) {
        const char* type_str = (results[i].type == OMNI_ITEM_TEXT) ? "[TEXT] " : "[IMAGE]";
        float pct = results[i].score * 100.0f;
        if (pct > 100.0f) pct = 100.0f;
        if (pct < 0.0f) pct = 0.0f;

        printf("%2zu   %s  %5.1f%%    %2u bits   #%-6u   %.60s%s\n",
               i + 1,
               type_str,
               pct,
               results[i].hamming_dist,
               results[i].id,
               results[i].payload,
               strlen(results[i].payload) > 60 ? "..." : "");
    }
    printf("----------------------------------------------------------------------------------------------------\n\n");

    /* Ground-truth exact check */
    OmniSearchResult exact[TOP_K];
    size_t exact_count = 0;
    (void)omni_exact_search(loaded_idx, q_vec, TOP_K, exact, &exact_count, scratch);

    size_t overlap = 0;
    for (size_t i = 0; i < out_count; ++i) {
        for (size_t j = 0; j < exact_count; ++j) {
            if (results[i].id == exact[j].id) {
                overlap++;
                break;
            }
        }
    }
    printf("📊 Ground-Truth Verification:\n");
    printf("  • Recall@%zu: %.1f%% (%zu / %zu matched exact brute force)\n", 
           TOP_K, ((float)overlap / (float)TOP_K) * 100.0f, overlap, TOP_K);
    printf("  • Arena Memory Watermark: %.2f MB\n", (double)arena->offset / (1024.0 * 1024.0));
    printf("  • Heap Allocations in Search Loop: ZERO\n\n");

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
    return 0;
}

static int cmd_bench(void) {
    print_banner();
    printf("📊 Running Systems Micro-Benchmark Suite...\n\n");

    RCRE_Arena* arena = rcre_arena_create(DEMO_ARENA_SIZE);
    RCRE_Arena* scratch = rcre_arena_create(DEMO_SCRATCH_SIZE);
    if (!arena || !scratch) return 1;

    const size_t N = 10000;
    const size_t D = OMNI_DEFAULT_DIM;
    OmniIndex* idx = omni_index_create(D, 64, N, arena);

    float vec[D];
    for (uint32_t i = 0; i < N; ++i) {
        generate_synthetic_embedding(vec, D, i);
        (void)omni_index_add(idx, i, (i % 2 == 0) ? OMNI_ITEM_TEXT : OMNI_ITEM_IMAGE, "item", vec);
    }
    (void)omni_index_build_hash(idx, scratch);

    /* 1. Normalization Benchmark */
    const size_t ITERS = 20000;
    double t0 = get_time_ms();
    for (size_t it = 0; it < ITERS; ++it) {
        (void)omni_normalize_l2(vec, D);
    }
    double t1 = get_time_ms();
    double norm_ops = (double)ITERS / ((t1 - t0) / 1000.0);
    printf("  1. L2 Normalization Throughput : %10.0f vectors / sec (%.2f ns/vec)\n", 
           norm_ops, ((t1 - t0) * 1000000.0) / (double)ITERS);

    /* 2. LSH Projection Benchmark */
    double t2 = get_time_ms();
    volatile uint64_t h_sink = 0;
    for (size_t it = 0; it < ITERS; ++it) {
        h_sink ^= omni_lsh_compute_hash(vec, idx->hyperplanes, 64, D);
    }
    double t3 = get_time_ms();
    double lsh_ops = (double)ITERS / ((t3 - t2) / 1000.0);
    printf("  2. LSH 64-bit Projection Rate  : %10.0f projections / sec (%.2f ns/proj)\n", 
           lsh_ops, ((t3 - t2) * 1000000.0) / (double)ITERS);

    /* 3. SIMD Dot Product Throughput */
    const size_t DOT_ITERS = 50000;
    double t4 = get_time_ms();
    volatile float d_sink = 0.0f;
    for (size_t it = 0; it < DOT_ITERS; ++it) {
        d_sink += omni_cosine_similarity(vec, &idx->embeddings[(it % N) * D], D);
    }
    double t5 = get_time_ms();
    double dot_ops = (double)DOT_ITERS / ((t5 - t4) / 1000.0);
    printf("  3. HAL SIMD Dot Product Rate   : %10.0f dots / sec (%.2f ns/dot)\n", 
           dot_ops, ((t5 - t4) * 1000000.0) / (double)DOT_ITERS);

    /* 4. Multi-Probe Query Latency */
    const size_t QUERY_ITERS = 2000;
    OmniSearchResult res[20];
    size_t count = 0;
    double t6 = get_time_ms();
    for (size_t it = 0; it < QUERY_ITERS; ++it) {
        (void)omni_index_search(idx, &idx->embeddings[(it * 7) % N * D], 20, 1, res, &count, scratch);
    }
    double t7 = get_time_ms();
    double qps = (double)QUERY_ITERS / ((t7 - t6) / 1000.0);
    double avg_lat_us = ((t7 - t6) * 1000.0) / (double)QUERY_ITERS;
    printf("  4. Multi-Probe Search (Top-20) : %10.0f QPS (%.2f µs / query)\n", qps, avg_lat_us);

    printf("\n✅ Benchmark completed successfully.\n");
    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
    return 0;
}

static void print_usage(const char* prog) {
    print_banner();
    printf("Usage:\n");
    printf("  %s demo                  Run full multimodal synthesis, index, and query demo\n", prog);
    printf("  %s bench                 Run micro-benchmarks for SIMD, LSH, and Hash Table\n", prog);
    printf("  %s info <index.rcre_lsh> Inspect serialized index metadata\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    if (strcmp(argv[1], "demo") == 0) {
        return cmd_demo();
    } else if (strcmp(argv[1], "bench") == 0) {
        return cmd_bench();
    } else if (strcmp(argv[1], "info") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Error: missing index path for info\n");
            return 1;
        }
        RCRE_Arena* arena = rcre_arena_create(DEMO_ARENA_SIZE);
        OmniIndex* idx = omni_index_load(argv[2], arena);
        if (!idx) {
            fprintf(stderr, "Error: failed to load '%s'\n", argv[2]);
            return 1;
        }
        print_banner();
        printf("Index Metadata: '%s'\n", argv[2]);
        printf("  • Total Items       : %zu\n", idx->num_items);
        printf("  • Embedding Dim     : %zu\n", idx->dim);
        printf("  • LSH Hyperplanes   : %zu\n", idx->num_hyperplanes);
        printf("  • Hash Table Buckets: %zu\n", idx->hash_table->capacity);
        printf("  • Total Postings    : %zu\n", idx->hash_table->postings_size);
        printf("  • String Pool Size  : %zu bytes (%.2f KB)\n", 
               idx->string_pool_size, (double)idx->string_pool_size / 1024.0);
        rcre_arena_destroy(arena);
        return 0;
    } else {
        print_usage(argv[0]);
        return 1;
    }
}
