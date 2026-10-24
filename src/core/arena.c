/**
 * @file arena.c
 * @brief Linear Scratchpad Arena Allocator with Strict 64-byte Alignment
 * 
 * Satisfies MEM-REQ-01, MEM-REQ-02, and MEM-REQ-03.
 */

#include "rcre.h"
#include <stdlib.h>
#include <string.h>

#define RCRE_DEFAULT_ALIGNMENT 64

RCRE_NODISCARD RCRE_Arena* rcre_arena_create(size_t capacity) {
    if (capacity == 0) {
        return NULL;
    }

    RCRE_Arena* arena = (RCRE_Arena*)malloc(sizeof(RCRE_Arena));
    if (!arena) {
        return NULL;
    }

    /* Allocate underlying buffer with 64-byte alignment */
    void* buffer = NULL;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(&buffer, RCRE_DEFAULT_ALIGNMENT, capacity) != 0) {
        free(arena);
        return NULL;
    }
#elif defined(_MSC_VER)
    buffer = _aligned_malloc(capacity, RCRE_DEFAULT_ALIGNMENT);
    if (!buffer) {
        free(arena);
        return NULL;
    }
#else
    /* C11 / C23 aligned_alloc requires size to be multiple of alignment */
    size_t aligned_capacity = (capacity + (RCRE_DEFAULT_ALIGNMENT - 1)) & ~((size_t)(RCRE_DEFAULT_ALIGNMENT - 1));
    buffer = aligned_alloc(RCRE_DEFAULT_ALIGNMENT, aligned_capacity);
    if (!buffer) {
        free(arena);
        return NULL;
    }
#endif

    arena->buffer = (uint8_t*)buffer;
    arena->capacity = capacity;
    arena->offset = 0;

    return arena;
}

RCRE_NODISCARD void* rcre_arena_alloc(RCRE_Arena* arena, size_t size, size_t align) {
    if (!arena || !arena->buffer || size == 0) {
        return NULL;
    }

    if (align == 0) {
        align = RCRE_DEFAULT_ALIGNMENT;
    }

    /* Verify alignment is a power of two */
    if ((align & (align - 1)) != 0) {
        return NULL;
    }

    uintptr_t current_ptr = (uintptr_t)(arena->buffer + arena->offset);
    uintptr_t aligned_ptr = (current_ptr + (align - 1)) & ~((uintptr_t)(align - 1));
    size_t padding = (size_t)(aligned_ptr - current_ptr);

    if (arena->offset + padding + size > arena->capacity) {
        return NULL; /* Out of arena capacity */
    }

    arena->offset += padding + size;
    return (void*)aligned_ptr;
}

void rcre_arena_reset(RCRE_Arena* arena) {
    if (arena) {
        arena->offset = 0;
    }
}

void rcre_arena_destroy(RCRE_Arena* arena) {
    if (arena) {
        if (arena->buffer) {
#if defined(_MSC_VER)
            _aligned_free(arena->buffer);
#else
            free(arena->buffer);
#endif
            arena->buffer = NULL;
        }
        free(arena);
    }
}

RCRE_NODISCARD RCRE_ArenaTemp rcre_arena_temp_begin(RCRE_Arena* arena) {
    RCRE_ArenaTemp temp = {
        .arena = arena,
        .saved_offset = arena ? arena->offset : 0
    };
    return temp;
}

void rcre_arena_temp_end(RCRE_ArenaTemp temp) {
    if (temp.arena && temp.saved_offset <= temp.arena->capacity) {
        temp.arena->offset = temp.saved_offset;
    }
}

RCRE_NODISCARD RCRE_GraphCtx* rcre_graph_ctx_create(size_t persistent_cap, size_t tape_cap) {
    RCRE_GraphCtx* ctx = (RCRE_GraphCtx*)malloc(sizeof(RCRE_GraphCtx));
    if (!ctx) return NULL;

    ctx->persistent_arena = rcre_arena_create(persistent_cap);
    ctx->tape_arena = rcre_arena_create(tape_cap);

    if (!ctx->persistent_arena || !ctx->tape_arena) {
        if (ctx->persistent_arena) rcre_arena_destroy(ctx->persistent_arena);
        if (ctx->tape_arena) rcre_arena_destroy(ctx->tape_arena);
        free(ctx);
        return NULL;
    }

    return ctx;
}

void rcre_graph_ctx_reset_tape(RCRE_GraphCtx* ctx) {
    if (ctx && ctx->tape_arena) {
        rcre_arena_reset(ctx->tape_arena);
    }
}

void rcre_graph_ctx_destroy(RCRE_GraphCtx* ctx) {
    if (ctx) {
        if (ctx->persistent_arena) rcre_arena_destroy(ctx->persistent_arena);
        if (ctx->tape_arena) rcre_arena_destroy(ctx->tape_arena);
        free(ctx);
    }
}
