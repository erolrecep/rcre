/**
 * @file test_hnsw.c
 * @brief Unit tests for RCRE Hierarchical Navigable Small World (HNSW) Vector Index
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-4f

static void test_hnsw_l2_search(void) {
    printf("  -> Testing HNSW L2 vector insertion and nearest-neighbor search...\n");
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    assert(arena != NULL);

    const size_t dim = 4;
    const size_t max_elements = 20;
    const size_t M = 4;
    const size_t ef = 16;

    RCRE_HNSWIndex* index = rcre_hnsw_create(dim, max_elements, M, ef, RCRE_METRIC_L2, arena);
    assert(index != NULL);

    /* Insert 10 distinct vectors */
    float vectors[10][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 0.0f, 1.0f},
        {2.0f, 2.0f, 2.0f, 2.0f},
        {-1.0f, -1.0f, -1.0f, -1.0f}
    };

    for (size_t i = 0; i < 10; ++i) {
        int status = rcre_hnsw_insert(index, vectors[i], i + 100, NULL);
        assert(status == RCRE_OK);
    }

    /* Query exact match for vector 5: [0, 1, 1, 0] */
    float query[4] = {0.0f, 1.0f, 1.0f, 0.0f};
    size_t out_ids[3] = {0};
    float out_dists[3] = {0.0f};

    int status = rcre_hnsw_search(index, query, 3, out_ids, out_dists, 16, NULL);
    assert(status == RCRE_OK);

    /* Top match must be vector 5 (id = 105) with distance 0 */
    assert(out_ids[0] == 105);
    assert(out_dists[0] < EPSILON);

    rcre_arena_destroy(arena);
    printf("     ✓ HNSW L2 search returned exact 0-distance query point\n");
}

static void test_hnsw_cosine_search(void) {
    printf("  -> Testing HNSW Cosine similarity search...\n");
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    assert(arena != NULL);

    const size_t dim = 3;
    const size_t max_elements = 10;
    const size_t M = 4;
    const size_t ef = 16;

    RCRE_HNSWIndex* index = rcre_hnsw_create(dim, max_elements, M, ef, RCRE_METRIC_COSINE, arena);
    assert(index != NULL);

    float v0[3] = {1.0f, 0.0f, 0.0f};
    float v1[3] = {0.0f, 1.0f, 0.0f};
    float v2[3] = {0.0f, 0.0f, 1.0f};

    int s0 = rcre_hnsw_insert(index, v0, 0, NULL); assert(s0 == RCRE_OK);
    int s1 = rcre_hnsw_insert(index, v1, 1, NULL); assert(s1 == RCRE_OK);
    int s2 = rcre_hnsw_insert(index, v2, 2, NULL); assert(s2 == RCRE_OK);

    float query[3] = {0.0f, 0.9f, 0.1f}; /* Very close to v1 */
    size_t out_ids[1] = {0};
    float out_dists[1] = {0.0f};

    int status = rcre_hnsw_search(index, query, 1, out_ids, out_dists, 16, NULL);
    assert(status == RCRE_OK);

    assert(out_ids[0] == 1); /* Should match v1 */

    rcre_arena_destroy(arena);
    printf("     ✓ HNSW Cosine search identified closest angular embedding\n");
}

int main(void) {
    printf("🧪 Running RCRE HNSW Vector Search Tests (ML-REQ-04)...\n");
    test_hnsw_l2_search();
    test_hnsw_cosine_search();
    printf("✅ All HNSW Vector Search Tests Passed Successfully!\n");
    return 0;
}
