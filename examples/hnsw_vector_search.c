/**
 * @file hnsw_vector_search.c
 * @brief Approximate Nearest Neighbor Vector Search Demo & Benchmark
 * 
 * Demonstrates HNSW vector indexing and sub-millisecond query search
 * with Euclidean L2 distance on 64-dimensional embeddings.
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("===================================================================\n");
    printf("  RCRE Example: HNSW Vector Similarity Search Benchmark            \n");
    printf("===================================================================\n");

    const size_t num_vectors = 2000;
    const size_t dim = 64;
    const size_t num_queries = 200;
    const size_t k = 5;

    const size_t arena_size = 64 * 1024 * 1024; /* 64 MB */
    RCRE_Arena* arena = rcre_arena_create(arena_size);
    if (!arena) {
        fprintf(stderr, "Failed to create memory arena.\n");
        return 1;
    }

    /* Allocate vector database */
    size_t db_shape[2] = {num_vectors, dim};
    RCRE_Tensor db = rcre_tensor_create(db_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* db_data = (float*)db.data;

    srand(42);
    for (size_t i = 0; i < num_vectors * dim; ++i) {
        db_data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    /* Initialize HNSW index */
    RCRE_HNSWIndex* index = rcre_hnsw_create(dim, num_vectors, 16, 64, RCRE_METRIC_L2, arena);
    if (!index) {
        fprintf(stderr, "Failed to initialize HNSW index.\n");
        rcre_arena_destroy(arena);
        return 1;
    }

    printf("Database Configuration:\n");
    printf("  Vectors:          %zu\n", num_vectors);
    printf("  Dimensions:       %zu-D\n", dim);
    printf("  Graph M:          16\n");
    printf("  ef_construction:  64\n");

    /* Build HNSW index */
    printf("\nBuilding HNSW proximity graph...\n");
    double t_build_start = get_time_sec();
    for (size_t i = 0; i < num_vectors; ++i) {
        float* vec = db_data + i * dim;
        int status = rcre_hnsw_insert(index, vec, i, arena);
        if (status != RCRE_OK) {
            fprintf(stderr, "Failed to insert vector %zu into HNSW index\n", i);
            break;
        }
    }
    double t_build_end = get_time_sec();
    printf("  Index Built in %.3f seconds (%.1f vectors/sec)\n", 
           t_build_end - t_build_start, (double)num_vectors / (t_build_end - t_build_start));

    /* Generate queries */
    size_t q_shape[2] = {num_queries, dim};
    RCRE_Tensor queries = rcre_tensor_create(q_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* q_data = (float*)queries.data;
    for (size_t i = 0; i < num_queries * dim; ++i) {
        q_data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    /* Perform Search Benchmark */
    printf("\nExecuting %zu ANN top-%zu queries...\n", num_queries, k);
    size_t out_labels[5] = {0};
    float out_dists[5] = {0.0f};

    double t_query_start = get_time_sec();
    for (size_t q = 0; q < num_queries; ++q) {
        float* q_vec = q_data + q * dim;
        int status = rcre_hnsw_search(index, q_vec, k, out_labels, out_dists, 32, arena);
        if (status != RCRE_OK) {
            fprintf(stderr, "Failed search for query %zu\n", q);
            break;
        }
    }
    double t_query_end = get_time_sec();

    double query_total_sec = t_query_end - t_query_start;
    double qps = (double)num_queries / query_total_sec;
    double latency_us = (query_total_sec / (double)num_queries) * 1e6;

    /* Display results of last query */
    printf("\nTop-%zu Nearest Neighbors for Query #%zu:\n", k, num_queries - 1);
    for (size_t rank = 0; rank < k; ++rank) {
        printf("  Rank %zu: Vector ID %-5zu | L2 Distance = %.5f\n", 
               rank + 1, out_labels[rank], out_dists[rank]);
    }

    printf("\n===================================================================\n");
    printf("  HNSW Benchmark Results                                           \n");
    printf("===================================================================\n");
    printf("Queries Evaluated:    %zu\n", num_queries);
    printf("Query Throughput:     %.2f QPS (Queries / sec)\n", qps);
    printf("Mean Query Latency:   %.2f microseconds (%.4f ms)\n", latency_us, latency_us / 1000.0);
    printf("Arena Memory Used:    %zu / %zu bytes (%.1f%%)\n", 
           arena->offset, arena->capacity, ((double)arena->offset / (double)arena->capacity) * 100.0);
    printf("===================================================================\n");

    rcre_arena_destroy(arena);
    return 0;
}
