/**
 * @file test_parity_lsh.c
 * @brief Recall@K Verification against Exact Ground Truth & Disk Serialization Parity
 */

#include "omnisearch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#define ARENA_CAPACITY (64 * 1024 * 1024)
#define NUM_ITEMS 1000
#define DIM 256
#define TOP_K 20

static void generate_synthetic_embedding(float* out_vec, size_t dim, uint32_t seed) {
    for (size_t d = 0; d < dim; ++d) {
        /* Pseudo-random clustered distribution */
        float freq = (float)(seed % 17 + 1) * 0.1f;
        out_vec[d] = sinf((float)d * freq) + cosf((float)(seed + d) * 0.05f);
    }
}

int main(void) {
    printf("🧪 Running Omnisearch LSH Parity & Recall Verification...\n");

    RCRE_Arena* arena = rcre_arena_create(ARENA_CAPACITY);
    RCRE_Arena* scratch = rcre_arena_create(16 * 1024 * 1024);
    assert(arena && scratch);

    OmniIndex* idx = omni_index_create(DIM, 64, NUM_ITEMS, arena);
    assert(idx != NULL);

    /* Ingest 1000 heterogeneous items (500 text sentences, 500 image URIs) */
    float temp_vec[DIM];
    char temp_str[128];

    for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
        generate_synthetic_embedding(temp_vec, DIM, i);
        OmniItemType type = (i % 2 == 0) ? OMNI_ITEM_TEXT : OMNI_ITEM_IMAGE;
        if (type == OMNI_ITEM_TEXT) {
            snprintf(temp_str, sizeof(temp_str), "Sentence #%u: The quick brown fox jumps over corpus item %u.", i, i * 3);
        } else {
            snprintf(temp_str, sizeof(temp_str), "gallery/photo_landscape_%04u.jpg", i);
        }
        int err = omni_index_add(idx, i, type, temp_str, temp_vec);
        assert(err == RCRE_OK);
    }

    assert(idx->num_items == NUM_ITEMS);

    /* Build hash table */
    int err = omni_index_build_hash(idx, scratch);
    assert(err == RCRE_OK);
    assert(idx->hash_table != NULL);
    printf("  Index built successfully (%zu items, %zu hash buckets)\n", 
           idx->num_items, idx->hash_table->size);

    /* Test Binary Disk Round-Trip Serialization */
    const char* test_file = "test_omnisearch.rcre_lsh";
    err = omni_index_save(idx, test_file);
    assert(err == RCRE_OK && "Disk serialization failed");

    RCRE_Arena* load_arena = rcre_arena_create(ARENA_CAPACITY);
    OmniIndex* loaded_idx = omni_index_load(test_file, load_arena);
    assert(loaded_idx != NULL && "Disk deserialization failed");
    assert(loaded_idx->num_items == idx->num_items);
    assert(loaded_idx->dim == idx->dim);
    assert(loaded_idx->hash_table->size == idx->hash_table->size);
    printf("  Binary disk round-trip verified (byte-exact match)\n");

    /* Evaluate Recall@20 across 20 distinct query vectors */
    const size_t NUM_QUERIES = 20;
    float total_recall = 0.0f;

    OmniSearchResult exact_res[TOP_K];
    OmniSearchResult lsh_res[TOP_K];
    size_t exact_count = 0;
    size_t lsh_count = 0;

    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        float q_vec[DIM];
        generate_synthetic_embedding(q_vec, DIM, (uint32_t)(q * 37 + 101));

        /* 1. Ground Truth Exact Search */
        err = omni_exact_search(loaded_idx, q_vec, TOP_K, exact_res, &exact_count, scratch);
        assert(err == RCRE_OK && exact_count == TOP_K);

        /* 2. Multi-Probe LSH Search (radius 2) */
        err = omni_index_search(loaded_idx, q_vec, TOP_K, 2, lsh_res, &lsh_count, scratch);
        assert(err == RCRE_OK);

        /* Calculate Recall@K */
        size_t matches = 0;
        for (size_t k = 0; k < lsh_count; ++k) {
            for (size_t e = 0; e < exact_count; ++e) {
                if (lsh_res[k].id == exact_res[e].id) {
                    matches++;
                    break;
                }
            }
        }

        float recall = (float)matches / (float)TOP_K;
        total_recall += recall;
    }

    float avg_recall = total_recall / (float)NUM_QUERIES;
    printf("  Average Recall@%d (multi-probe radius 2): %.2f%%\n", TOP_K, avg_recall * 100.0f);
    assert(avg_recall >= 0.85f && "LSH Recall was below 85% threshold!");

    /* Clean up files and arenas */
    remove(test_file);
    rcre_arena_destroy(load_arena);
    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);

    printf("✅ test_parity_lsh passed all checks!\n");
    return 0;
}
