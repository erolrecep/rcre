/**
 * @file transformer_internal.h
 * @brief RCRE Internal Transformer Definitions, Helpers & GEMM Dispatch
 */

#ifndef RCRE_TRANSFORMER_INTERNAL_H
#define RCRE_TRANSFORMER_INTERNAL_H

#include "rcre.h"
#include "rcre_transformer.h"
#include "../core/rcre_rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

/* Multi-dimensional tensor row-major index helpers */
#define IDX2(r, c, cols) ((r) * (cols) + (c))
#define IDX3(b, t, c, T, C) (((b) * (T) * (C)) + ((t) * (C)) + (c))
#define IDX4(b, h, t, d, H, T, D) (((b) * (T) * (H) * (D)) + ((t) * (H) * (D)) + ((h) * (D)) + (d))
#define IDX_ATTN(b, h, tq, tk, H, T) (((b) * (H) * (T) * (T)) + ((h) * (T) * (T)) + ((tq) * (T)) + (tk))

/* Aligned dynamic memory allocation */
static inline void* rcre_transformer_alloc_mem(RCRE_Arena *arena, size_t size, size_t align) {
    if (arena != nullptr) {
        return rcre_arena_alloc(arena, size, align);
    }
    void *ptr = nullptr;
    if (posix_memalign(&ptr, align < sizeof(void*) ? sizeof(void*) : align, size) != 0) {
        return nullptr;
    }
    return ptr;
}

static inline void rcre_transformer_free_mem(RCRE_Arena *arena, void *ptr) {
    if (arena == nullptr && ptr != nullptr) {
        free(ptr);
    }
}

/* Transpose-capable GEMM using RCRE HAL for contiguous blocks */
void rcre_transformer_gemm(
    bool transA, bool transB,
    size_t M, size_t N, size_t K,
    float alpha,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    float beta,
    float *C, size_t ldc
);

#endif /* RCRE_TRANSFORMER_INTERNAL_H */

/* ========================================================================= */
/* Per-call scratch pool (backward / forward_token)                          */
/* ========================================================================= */
/*
 * Arena-resident models checkpoint their arena (one bump per step, zero
 * steady-state heap); heap models use checked malloc/free. 15 blocks is the
 * design maximum for any single call.
 */
typedef struct {
    RCRE_Arena *arena;
    RCRE_ArenaTemp tmp;
    void *heap[15];
    size_t n_heap;
} GPTScratch;

static inline void gpt_scratch_begin(GPTScratch *s, RCRE_Arena *arena) {
    s->arena = arena;
    s->n_heap = 0;
    if (arena) s->tmp = rcre_arena_temp_begin(arena);
}

static inline float *gpt_scratch_alloc(GPTScratch *s, size_t bytes) {
    float *p = (float*)rcre_transformer_alloc_mem(s->arena, bytes, 64);
    if (p && !s->arena) s->heap[s->n_heap++] = p;
    return p;
}

static inline void gpt_scratch_end(GPTScratch *s) {
    if (s->arena) { rcre_arena_temp_end(s->tmp); }
    else { for (size_t i = 0; i < s->n_heap; i++) free(s->heap[i]); }
}
