/**
 * @file test_hnsw.c
 * @brief Unit tests for RCRE Hierarchical Navigable Small World (HNSW) Vector Index
 */

#include "rcre.h"
#include "test_framework.h"

#define EPSILON 1e-4f

static void test_hnsw_l2_search(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const size_t dim = 4;
    const size_t max_elements = 20;
    const size_t M = 4;
    const size_t ef = 16;

    RCRE_HNSWIndex* index = rcre_hnsw_create(dim, max_elements, M, ef, RCRE_METRIC_L2, arena);
    ASSERT_NOT_NULL(index);

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
        ASSERT_INT_EQ(RCRE_OK, status);
    }

    /* Query exact match for vector 5: [0, 1, 1, 0] */
    float query[4] = {0.0f, 1.0f, 1.0f, 0.0f};
    size_t out_ids[3] = {0};
    float out_dists[3] = {0.0f};

    int status = rcre_hnsw_search(index, query, 3, out_ids, out_dists, 16, NULL);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* Top match must be vector 5 (id = 105) with distance 0 */
    ASSERT_UINT_EQ(105, out_ids[0]);
    ASSERT_FLOAT_NEAR(0.0f, out_dists[0], EPSILON);

    rcre_arena_destroy(arena);
}

static void test_hnsw_cosine_search(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const size_t dim = 3;
    const size_t max_elements = 10;
    const size_t M = 4;
    const size_t ef = 16;

    RCRE_HNSWIndex* index = rcre_hnsw_create(dim, max_elements, M, ef, RCRE_METRIC_COSINE, arena);
    ASSERT_NOT_NULL(index);

    float v0[3] = {1.0f, 0.0f, 0.0f};
    float v1[3] = {0.0f, 1.0f, 0.0f};
    float v2[3] = {0.0f, 0.0f, 1.0f};

    int s0 = rcre_hnsw_insert(index, v0, 0, NULL); ASSERT_INT_EQ(RCRE_OK, s0);
    int s1 = rcre_hnsw_insert(index, v1, 1, NULL); ASSERT_INT_EQ(RCRE_OK, s1);
    int s2 = rcre_hnsw_insert(index, v2, 2, NULL); ASSERT_INT_EQ(RCRE_OK, s2);

    float query[3] = {0.0f, 0.9f, 0.1f}; /* Very close to v1 */
    size_t out_ids[1] = {0};
    float out_dists[1] = {0.0f};

    int status = rcre_hnsw_search(index, query, 1, out_ids, out_dists, 16, NULL);
    ASSERT_INT_EQ(RCRE_OK, status);

    ASSERT_UINT_EQ(1, out_ids[0]); /* Should match v1 */

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE HNSW Vector Search Index (ML-REQ-04)");
    TEST_RUN(test_hnsw_l2_search);
    TEST_RUN(test_hnsw_cosine_search);
    return test_suite_report();
}
