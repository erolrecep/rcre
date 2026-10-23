/**
 * @file test_arena.c
 * @brief Unit tests for RCRE linear memory arena allocator
 */

#include "rcre.h"
#include "test_framework.h"

static void test_arena_create_and_lifecycle(void) {
    const size_t capacity = 1024 * 1024; /* 1MB */
    RCRE_Arena* arena = rcre_arena_create(capacity);
    ASSERT_NOT_NULL(arena);
    ASSERT_UINT_EQ(capacity, arena->capacity);
    ASSERT_UINT_EQ(0, arena->offset);
    rcre_arena_destroy(arena);
}

static void test_arena_64byte_alignment(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Test 64-byte alignment on initial allocation */
    void* ptr1 = rcre_arena_alloc(arena, 13, 64);
    ASSERT_NOT_NULL(ptr1);
    ASSERT_UINT_EQ(0, (uintptr_t)ptr1 % 64);

    /* Allocate odd size (27 bytes) and verify next pointer remains 64-byte aligned */
    void* ptr2 = rcre_arena_alloc(arena, 27, 64);
    ASSERT_NOT_NULL(ptr2);
    ASSERT_UINT_EQ(0, (uintptr_t)ptr2 % 64);
    ASSERT_TRUE((uintptr_t)ptr2 >= (uintptr_t)ptr1 + 13);

    rcre_arena_destroy(arena);
}

static void test_arena_bulk_reset_o1(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    void* ptr1 = rcre_arena_alloc(arena, 128, 64);
    ASSERT_NOT_NULL(ptr1);
    ASSERT_TRUE(arena->offset > 0);

    /* O(1) bulk reset */
    rcre_arena_reset(arena);
    ASSERT_UINT_EQ(0, arena->offset);

    /* Reallocation should reuse initial buffer starting point */
    void* ptr2 = rcre_arena_alloc(arena, 64, 64);
    ASSERT_NOT_NULL(ptr2);
    ASSERT_TRUE(ptr2 == ptr1);

    rcre_arena_destroy(arena);
}

static void test_arena_capacity_exhaustion(void) {
    const size_t capacity = 64 * 1024;
    RCRE_Arena* arena = rcre_arena_create(capacity);
    ASSERT_NOT_NULL(arena);

    /* Allocate exact capacity */
    void* p1 = rcre_arena_alloc(arena, capacity, 1);
    ASSERT_NOT_NULL(p1);

    /* Further allocation must return NULL */
    void* p_overflow = rcre_arena_alloc(arena, 1, 1);
    ASSERT_NULL(p_overflow);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Linear Memory Arena Allocator (CORE-REQ-01)");
    TEST_RUN(test_arena_create_and_lifecycle);
    TEST_RUN(test_arena_64byte_alignment);
    TEST_RUN(test_arena_bulk_reset_o1);
    TEST_RUN(test_arena_capacity_exhaustion);
    return test_suite_report();
}
