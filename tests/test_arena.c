/**
 * @file test_arena.c
 * @brief Unit tests for RCRE linear memory arena allocator
 */

#include "rcre.h"
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

void test_arena_alignment_and_lifecycle(void) {
    printf("Running test_arena_alignment_and_lifecycle...\n");

    const size_t capacity = 1024 * 1024; /* 1MB */
    RCRE_Arena* arena = rcre_arena_create(capacity);
    assert(arena != NULL);
    assert(arena->capacity == capacity);
    assert(arena->offset == 0);

    /* Test 64-byte alignment */
    void* ptr1 = rcre_arena_alloc(arena, 13, 64);
    assert(ptr1 != NULL);
    assert(((uintptr_t)ptr1 % 64) == 0);

    /* Allocate odd size and verify next 64-byte alignment */
    void* ptr2 = rcre_arena_alloc(arena, 27, 64);
    assert(ptr2 != NULL);
    assert(((uintptr_t)ptr2 % 64) == 0);
    assert((uintptr_t)ptr2 >= (uintptr_t)ptr1 + 13);

    /* Test bulk reset in O(1) */
    rcre_arena_reset(arena);
    assert(arena->offset == 0);

    /* Allocate again after reset */
    void* ptr3 = rcre_arena_alloc(arena, 64, 64);
    assert(ptr3 != NULL);
    assert(ptr3 == ptr1); /* Reuses buffer start */

    /* Test capacity exhaustion */
    void* ptr_huge = rcre_arena_alloc(arena, capacity * 2, 64);
    assert(ptr_huge == NULL);

    rcre_arena_destroy(arena);
    printf("✓ test_arena_alignment_and_lifecycle passed!\n");
}

int main(void) {
    printf("=== RCRE Memory Arena Test Suite ===\n");
    test_arena_alignment_and_lifecycle();
    printf("=== All Arena Tests Passed ===\n");
    return 0;
}
