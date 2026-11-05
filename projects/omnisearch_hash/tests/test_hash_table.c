/**
 * @file test_hash_table.c
 * @brief Unit Tests for Arena-Backed Open-Addressing Hash Table
 */

#include "hash_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define TEST_ARENA_SIZE (16 * 1024 * 1024)

int main(void) {
    printf("🧪 Running RCRE Hash Table Unit Tests...\n");

    RCRE_Arena* arena = rcre_arena_create(TEST_ARENA_SIZE);
    assert(arena != NULL && "Arena creation failed");

    /* 1. Basic Builder Insert & Finalize */
    RCRE_HashBuilder* hb = rcre_hash_builder_create(64, arena);
    assert(hb != NULL);

    /* Insert 500 keys, some with multiple elements in the same bucket */
    for (uint32_t i = 0; i < 500; ++i) {
        uint64_t key = (uint64_t)(i % 50); /* 50 distinct keys, 10 elements each */
        int err = rcre_hash_builder_add(hb, key, i, arena);
        assert(err == RCRE_OK);
    }

    assert(hb->size == 50);
    assert(hb->total_elements == 500);

    /* Finalize into flat hash table */
    RCRE_HashTable* ht = rcre_hash_builder_finalize(hb, arena);
    assert(ht != NULL);
    assert(ht->size == 50);
    assert(ht->postings_size == 500);

    /* Verify all 50 keys exist and contain exactly 10 elements */
    for (uint64_t k = 0; k < 50; ++k) {
        uint32_t count = 0;
        const uint32_t* postings = rcre_hash_table_lookup(ht, k, &count);
        assert(postings != NULL);
        assert(count == 10);
        for (uint32_t j = 0; j < count; ++j) {
            assert(postings[j] % 50 == k);
        }
    }

    /* Verify non-existent key returns NULL */
    uint32_t missing_count = 999;
    const uint32_t* missing = rcre_hash_table_lookup(ht, 999999ULL, &missing_count);
    assert(missing == NULL);
    assert(missing_count == 0);

    /* Test Arena Rewind */
    size_t offset_before = arena->offset;
    (void)offset_before;
    RCRE_HashBuilder* hb2 = rcre_hash_builder_create(128, arena);
    assert(hb2 != NULL);
    rcre_arena_reset(arena);
    assert(arena->offset == 0);

    rcre_arena_destroy(arena);
    printf("✅ test_hash_table passed all checks!\n");
    return 0;
}
