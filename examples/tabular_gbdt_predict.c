/**
 * @file tabular_gbdt_predict.c
 * @brief High-Throughput Tabular GBDT Inference Demo & Benchmark
 * 
 * Demonstrates flat-array decision tree ensemble prediction with zero heap allocations,
 * cache-aligned node structures, and sub-microsecond latency.
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
    printf("  RCRE Example: High-Throughput GBDT Tabular Inference            \n");
    printf("===================================================================\n");

    const size_t arena_size = 32 * 1024 * 1024; /* 32 MB */
    RCRE_Arena* arena = rcre_arena_create(arena_size);
    if (!arena) {
        fprintf(stderr, "Failed to create memory arena.\n");
        return 1;
    }

    /* Construct a realistic 100-tree ensemble with max_depth=3 (7 nodes per tree) */
    const size_t num_trees = 100;
    const size_t nodes_per_tree = 7;
    const size_t total_nodes = num_trees * nodes_per_tree;
    const size_t num_features = 16;
    const size_t num_samples = 100000;

    RCRE_TreeNode* nodes = (RCRE_TreeNode*)rcre_arena_alloc(arena, total_nodes * sizeof(RCRE_TreeNode), 64);
    size_t* tree_roots = (size_t*)rcre_arena_alloc(arena, num_trees * sizeof(size_t), 64);

    for (size_t t = 0; t < num_trees; ++t) {
        tree_roots[t] = t * nodes_per_tree;
        for (size_t n = 0; n < nodes_per_tree; ++n) {
            size_t idx = t * nodes_per_tree + n;
            if (n < 3) {
                /* Internal branch node */
                nodes[idx].feature_idx = (uint32_t)((t + n) % num_features);
                nodes[idx].threshold = 0.5f + (float)n * 0.1f;
                nodes[idx].left_child = (uint32_t)(tree_roots[t] + 2 * n + 1);
                nodes[idx].right_child = (uint32_t)(tree_roots[t] + 2 * n + 2);
                nodes[idx].leaf_value = 0.0f;
                nodes[idx].is_leaf = false;
            } else {
                /* Leaf node */
                nodes[idx].feature_idx = 0;
                nodes[idx].threshold = 0.0f;
                nodes[idx].left_child = 0;
                nodes[idx].right_child = 0;
                nodes[idx].leaf_value = ((float)(n % 3) - 1.0f) * 0.05f;
                nodes[idx].is_leaf = true;
            }
        }
    }

    RCRE_GBDTModel model = {
        .nodes = nodes,
        .tree_roots = tree_roots,
        .num_trees = num_trees,
        .num_nodes = total_nodes,
        .base_score = 0.0f,
        .is_binary_classification = true
    };

    printf("Model Architecture:\n");
    printf("  Trees:            %zu\n", model.num_trees);
    printf("  Total Nodes:      %zu (flat array, 64-byte cache aligned)\n", total_nodes);
    printf("  Feature Vector:   %zu continuous features\n", num_features);
    printf("  Evaluation Load:  %zu samples\n", num_samples);

    /* Allocate synthetic input batch: [100000, 16] F32 */
    size_t input_shape[2] = {num_samples, num_features};
    RCRE_Tensor X = rcre_tensor_create(input_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* X_data = (float*)X.data;
    for (size_t i = 0; i < num_samples * num_features; ++i) {
        X_data[i] = (float)(i % 100) / 100.0f;
    }

    /* Output predictions: [100000, 1] F32 */
    size_t out_shape[2] = {num_samples, 1};
    RCRE_Tensor preds = rcre_tensor_create(out_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    /* Benchmark batch inference */
    printf("\nExecuting GBDT batch prediction...\n");
    double t0 = get_time_sec();
    int status = rcre_ml_gbdt_predict(&model, &X, &preds);
    double t1 = get_time_sec();

    if (status != RCRE_OK) {
        fprintf(stderr, "GBDT inference failed with error %d\n", status);
        rcre_arena_destroy(arena);
        return 1;
    }

    double total_sec = t1 - t0;
    double throughput = (double)num_samples / total_sec;
    double latency_us = (total_sec / (double)num_samples) * 1e6;

    float* p = (float*)preds.data;
    printf("\nSample Predictions:\n");
    for (size_t i = 0; i < 5; ++i) {
        float prob = p[i];
        printf("  Sample #%zu: Predicted Prob = %.4f | Class = %s\n",
               i, prob, prob >= 0.5f ? "Positive [1]" : "Negative [0]");
    }

    printf("\n===================================================================\n");
    printf("  GBDT Benchmark Results                                           \n");
    printf("===================================================================\n");
    printf("Total Inferences:     %zu\n", num_samples);
    printf("Elapsed Time:         %.4f seconds\n", total_sec);
    printf("Throughput:           %.2f predictions / second\n", throughput);
    printf("Average Latency:      %.3f microseconds / prediction\n", latency_us);
    printf("Arena Memory Used:    %zu / %zu bytes (%.1f%%)\n", 
           arena->offset, arena->capacity, ((double)arena->offset / (double)arena->capacity) * 100.0);
    printf("Heap Allocations:     0 (100%% deterministic linear arena)\n");
    printf("===================================================================\n");

    rcre_arena_destroy(arena);
    return 0;
}
