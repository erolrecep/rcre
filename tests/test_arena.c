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

static void test_arena_temp_checkpoint(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    void* p1 = rcre_arena_alloc(arena, 256, 64);
    ASSERT_NOT_NULL(p1);
    size_t offset_before = arena->offset;

    /* Begin scoped checkpoint */
    RCRE_ArenaTemp temp = rcre_arena_temp_begin(arena);
    ASSERT_UINT_EQ(offset_before, temp.saved_offset);

    /* Allocate scratchpad memory */
    void* p_temp = rcre_arena_alloc(arena, 1024, 64);
    ASSERT_NOT_NULL(p_temp);
    ASSERT_TRUE(arena->offset > offset_before);

    /* End scoped checkpoint - rewinds offset */
    rcre_arena_temp_end(temp);
    ASSERT_UINT_EQ(offset_before, arena->offset);

    /* Next allocation reuses scratchpad space */
    void* p2 = rcre_arena_alloc(arena, 1024, 64);
    ASSERT_NOT_NULL(p2);
    ASSERT_TRUE(p2 == p_temp);

    rcre_arena_destroy(arena);
}

static void test_graph_ctx_lifecycle(void) {
    RCRE_GraphCtx* ctx = rcre_graph_ctx_create(512 * 1024, 1024 * 1024);
    ASSERT_NOT_NULL(ctx);
    ASSERT_NOT_NULL(ctx->persistent_arena);
    ASSERT_NOT_NULL(ctx->tape_arena);

    void* weight = rcre_arena_alloc(ctx->persistent_arena, 256, 64);
    void* act = rcre_arena_alloc(ctx->tape_arena, 512, 64);
    ASSERT_NOT_NULL(weight);
    ASSERT_NOT_NULL(act);

    size_t weight_offset = ctx->persistent_arena->offset;
    ASSERT_TRUE(weight_offset > 0);

    /* Reset ephemeral tape only */
    rcre_graph_ctx_reset_tape(ctx);
    ASSERT_UINT_EQ(0, ctx->tape_arena->offset);
    ASSERT_UINT_EQ(weight_offset, ctx->persistent_arena->offset);

    rcre_graph_ctx_destroy(ctx);
}

int main(void) {
    test_suite_begin("RCRE Linear Memory Arena Allocator (CORE-REQ-01)");
    TEST_RUN(test_arena_create_and_lifecycle);
    TEST_RUN(test_arena_64byte_alignment);
    TEST_RUN(test_arena_bulk_reset_o1);
    TEST_RUN(test_arena_capacity_exhaustion);
    TEST_RUN(test_arena_temp_checkpoint);
    TEST_RUN(test_graph_ctx_lifecycle);
    return test_suite_report();
}
