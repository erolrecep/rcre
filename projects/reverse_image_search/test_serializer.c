/**
 * @file test_serializer.c
 * @brief Comprehensive Verification of .rcre_idx Binary Serialization, Deserialization & HNSW Retrieval
 */

#include "index_serializer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define TEST_ARENA_SIZE (32 * 1024 * 1024) /* 32 MB */
#define NUM_TEST_ITEMS  128
#define TEST_DIM        64

static void generate_synthetic_embedding(size_t id, float* vec, size_t dim) {
    /* High-dispersion pseudo-random vector with unique seed per id */
    uint32_t seed = (uint32_t)(id * 2654435761u + 1);
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        seed = seed * 1664525u + 1013904223u;
        float val = ((float)(seed % 2000) - 1000.0f) / 1000.0f;
        vec[i] = val;
        norm_sq += val * val;
    }
    float inv_norm = 1.0f / sqrtf(norm_sq);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] *= inv_norm;
    }
}

int main(void) {
    printf("===================================================================\n");
    printf("🧪 RCRE Lens: Binary Index Serialization & HNSW Search Verification\n");
    printf("===================================================================\n\n");

    /* Initialize hardware acceleration */
    rcre_hal_init();

    RCRE_Arena* arena = rcre_arena_create(TEST_ARENA_SIZE);
    assert(arena != NULL);

    const char* test_idx_file = "bin/test_dataset.rcre_idx";

    /* ------------------------------------------------------------- */
    /* Test 1: In-Memory Index Construction & Insertion              */
    /* ------------------------------------------------------------- */
    printf("[Test 1/5] Building in-memory index for %d items (%d-D)...\n", NUM_TEST_ITEMS, TEST_DIM);
    RCRE_ImageIndex* idx = rcre_index_create(NUM_TEST_ITEMS, TEST_DIM, arena);
    assert(idx != NULL);
    assert(idx->capacity == NUM_TEST_ITEMS);
    assert(idx->feature_dim == TEST_DIM);
    assert(idx->num_images == 0);

    float synth_embeddings[NUM_TEST_ITEMS][TEST_DIM];
    char path_buf[128];

    for (size_t i = 0; i < NUM_TEST_ITEMS; ++i) {
        generate_synthetic_embedding(i, synth_embeddings[i], TEST_DIM);
        snprintf(path_buf, sizeof(path_buf), "/dataset/test_images/sample_%04zu.jpg", i);
        int rc = rcre_index_add(idx, path_buf, synth_embeddings[i]);
        assert(rc == RCRE_OK);
    }
    assert(idx->num_images == NUM_TEST_ITEMS);
    printf("  ✓ Successfully indexed %d vectors into HNSW graph\n", NUM_TEST_ITEMS);

    /* ------------------------------------------------------------- */
    /* Test 2: Top-K Search on Original In-Memory Index              */
    /* ------------------------------------------------------------- */
    printf("[Test 2/5] Querying top-5 matches for vector #42...\n");
    RCRE_SearchResult results[5];
    int count = rcre_index_search(idx, synth_embeddings[42], 5, results, arena);
    assert(count == 5);
    assert(results[0].image_id == 42);
    assert(fabsf(results[0].similarity - 1.0f) < 1e-4f);
    assert(fabsf(results[0].l2_distance - 0.0f) < 1e-4f);
    assert(strcmp(results[0].filepath, "/dataset/test_images/sample_0042.jpg") == 0);
    printf("  ✓ Top-1 Match: ID %zu, Path: %s, Similarity: %.4f (L2: %.4f)\n",
           results[0].image_id, results[0].filepath, results[0].similarity, results[0].l2_distance);

    /* ------------------------------------------------------------- */
    /* Test 3: Binary Disk Serialization (.rcre_idx)                 */
    /* ------------------------------------------------------------- */
    printf("[Test 3/5] Serializing index to '%s'...\n", test_idx_file);
    int save_rc = rcre_index_save(idx, test_idx_file);
    assert(save_rc == RCRE_OK);

    FILE* fp = fopen(test_idx_file, "rb");
    assert(fp != NULL);
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fclose(fp);
    printf("  ✓ File successfully serialized (%ld bytes written)\n", file_size);

    /* ------------------------------------------------------------- */
    /* Test 4: Deserialization into Fresh Arena & Parity Check       */
    /* ------------------------------------------------------------- */
    printf("[Test 4/5] Deserializing into fresh arena & verifying parity...\n");
    RCRE_Arena* fresh_arena = rcre_arena_create(TEST_ARENA_SIZE);
    assert(fresh_arena != NULL);

    RCRE_ImageIndex* loaded_idx = rcre_index_load(test_idx_file, fresh_arena);
    assert(loaded_idx != NULL);
    assert(loaded_idx->num_images == NUM_TEST_ITEMS);
    assert(loaded_idx->feature_dim == TEST_DIM);

    /* Verify manifest filepaths and embeddings match verbatim */
    for (size_t i = 0; i < NUM_TEST_ITEMS; ++i) {
        snprintf(path_buf, sizeof(path_buf), "/dataset/test_images/sample_%04zu.jpg", i);
        assert(strcmp(loaded_idx->filepaths[i], path_buf) == 0);
        for (size_t d = 0; d < TEST_DIM; ++d) {
            float diff = fabsf(loaded_idx->embeddings[i * TEST_DIM + d] - synth_embeddings[i][d]);
            assert(diff < 1e-6f);
        }
    }
    printf("  ✓ 100%% byte-for-byte parity verified across all %d paths and embeddings\n", NUM_TEST_ITEMS);

    /* ------------------------------------------------------------- */
    /* Test 5: Search Retrieval Parity on Loaded Index               */
    /* ------------------------------------------------------------- */
    printf("[Test 5/5] Testing HNSW graph retrieval on loaded index...\n");
    RCRE_SearchResult loaded_results[5];
    int loaded_count = rcre_index_search(loaded_idx, synth_embeddings[42], 5, loaded_results, fresh_arena);
    assert(loaded_count == 5);
    assert(loaded_results[0].image_id == 42);
    assert(fabsf(loaded_results[0].similarity - 1.0f) < 1e-4f);
    assert(strcmp(loaded_results[0].filepath, "/dataset/test_images/sample_0042.jpg") == 0);

    /* Test perturbed query: vector 42 + 5% perturbation */
    float noisy_query[TEST_DIM];
    float noisy_norm = 0.0f;
    for (size_t d = 0; d < TEST_DIM; ++d) {
        noisy_query[d] = synth_embeddings[42][d] + 0.05f * sinf((float)d);
        noisy_norm += noisy_query[d] * noisy_query[d];
    }
    noisy_norm = 1.0f / sqrtf(noisy_norm);
    for (size_t d = 0; d < TEST_DIM; ++d) {
        noisy_query[d] *= noisy_norm;
    }

    loaded_count = rcre_index_search(loaded_idx, noisy_query, 5, loaded_results, fresh_arena);
    assert(loaded_count == 5);
    assert(loaded_results[0].image_id == 42);
    assert(loaded_results[0].similarity > 0.95f);
    printf("  ✓ Perturbed Query Match: ID %zu, Path: %s, Similarity: %.4f (L2: %.4f)\n",
           loaded_results[0].image_id, loaded_results[0].filepath, 
           loaded_results[0].similarity, loaded_results[0].l2_distance);

    /* Clean up temporary test file */
    remove(test_idx_file);
    rcre_arena_destroy(arena);
    rcre_arena_destroy(fresh_arena);

    printf("\n🎉 ALL 5 INDEX SERIALIZATION & RETRIEVAL INVARIANTS PASSED!\n");
    printf("===================================================================\n");
    return 0;
}
