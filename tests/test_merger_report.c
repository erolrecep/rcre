/**
 * @file test_merger_report.c
 * @brief Regression & gap-demonstration tests mapped 1:1 to the findings in
 *        `rcre_c23mlwautograd_merger_review_by_pidev.md`.
 *
 * Every test carries a header block citing the exact report section it
 * corresponds to, so the mapping between "report finding" and "code
 * behaviour" stays auditable.
 *
 *   PART 1 — REPORT §2.3: fixed bugs, extra granular coverage
 *           (beyond the six regressions already in test_autograd_tensor.c /
 *            test_nn_train.c).
 *
 *   PART 2 — REPORT §3/§4: OPEN items that were deliberately NOT fixed in
 *           the hardening pass. These tests PASS today: they pin the current
 *           behaviour and document the ceiling/contract. Each block ends
 *           with a "CONTINUE HERE" note describing the follow-up work.
 */

#include "rcre.h"
#include "rcre_autograd.h"
#include "rcre_rng.h" /* internal header — pinned by report §2.3 (RNG copies) */
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define EPSILON 1e-4f

/* ========================================================================= */
/* PART 1 — REPORT §2.3: fixed bugs, granular regressions                    */
/* ========================================================================= */

/*
 * REPORT §2.3 (P1) — root cause of "train_test_split reads uninitialized
 * shape[1] = 13744632839234567870".
 *
 * Fix under test: RCRE_AGTensor is memset before field assignment, so every
 * shape[i]/strides[i] with i >= ndim reads back 0, never heap/arena garbage.
 */
static void test_agtensor_shape_zeroed_beyond_ndim(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_AGTensor* v1 = rcre_ag_tensor_create_1d(arena, 5);
    ASSERT_NOT_NULL(v1);
    ASSERT_UINT_EQ(1, v1->ndim);
    for (int i = 1; i < RCRE_AG_MAX_DIMS; ++i) {
        ASSERT_UINT_EQ(0, v1->shape[i]);
        ASSERT_UINT_EQ(0, v1->strides[i]);
    }
    ASSERT_UINT_EQ(5, v1->numel);

    RCRE_AGTensor* v2 = rcre_ag_tensor_create_2d(arena, 3, 7);
    ASSERT_NOT_NULL(v2);
    ASSERT_UINT_EQ(0, v2->shape[2]);
    ASSERT_UINT_EQ(0, v2->shape[3]);
    ASSERT_UINT_EQ(21, v2->numel);

    rcre_arena_destroy(arena);
}

/*
 * REPORT §2.3 (P1) — "CSV parser caps lines at 4096 bytes; long lines
 * silently truncated or split into phantom rows".
 *
 * Coverage beyond the existing 1000-column test:
 *   (a) a 4102-char line — just past the old 4096 cap,
 *   (b) a ~120KB line (20,000 columns) — proves unbounded dynamic growth,
 *   (c) a file whose last line has no trailing newline.
 */
static void test_csv_old_boundary_and_dynamic_growth(void) {
    RCRE_Arena* arena = rcre_arena_create(4 << 20);
    ASSERT_NOT_NULL(arena);

    /* (a) 373 columns of 10-digit tokens => line length 373*11 - 1 = 4102 */
    const char* p1 = "/tmp/rcre_mr_boundary.csv";
    FILE* f = fopen(p1, "w");
    ASSERT_NOT_NULL(f);
    for (int r = 0; r < 2; ++r) {
        fprintf(f, "%d", 1000000000 + r);
        for (int c = 1; c < 373; ++c) fprintf(f, ",%d", 1000000000 + c);
        fprintf(f, "\n");
    }
    fclose(f);
    RCRE_AGTensor* t1 = rcre_data_load_csv(arena, p1, false, 0);
    ASSERT_NOT_NULL(t1);
    ASSERT_UINT_EQ(2, t1->shape[0]);
    ASSERT_UINT_EQ(373, t1->shape[1]);
    ASSERT_FLOAT_NEAR(1000000372.0f, t1->data[372], 1.0f); /* row0 last col */
    remove(p1);

    /* (b) 20,000 columns, 3 rows (~360KB file, ~240KB tensor) */
    const char* p2 = "/tmp/rcre_mr_hugeline.csv";
    f = fopen(p2, "w");
    ASSERT_NOT_NULL(f);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 20000; ++c) {
            if (c > 0) fputc(',', f);
            fprintf(f, "%d", r * 100000 + c);
        }
        fputc('\n', f);
    }
    fclose(f);
    RCRE_AGTensor* t2 = rcre_data_load_csv(arena, p2, false, 0);
    ASSERT_NOT_NULL(t2);
    ASSERT_UINT_EQ(3, t2->shape[0]);
    ASSERT_UINT_EQ(20000, t2->shape[1]);
    ASSERT_FLOAT_NEAR(2 * 100000 + 19999, t2->data[2 * 20000 + 19999], 1.0f);
    remove(p2);

    /* (c) final line without trailing newline must still be parsed */
    const char* p3 = "/tmp/rcre_mr_nonewline.csv";
    f = fopen(p3, "w");
    ASSERT_NOT_NULL(f);
    fprintf(f, "1,2\n3,4\n5,6"); /* no \n after last row */
    fclose(f);
    RCRE_AGTensor* t3 = rcre_data_load_csv(arena, p3, false, 0);
    ASSERT_NOT_NULL(t3);
    ASSERT_UINT_EQ(3, t3->shape[0]);
    ASSERT_UINT_EQ(2, t3->shape[1]);
    ASSERT_FLOAT_NEAR(6.0f, t3->data[5], EPSILON);
    remove(p3);

    rcre_arena_destroy(arena);
}

/*
 * REPORT §2.3 (P2) — "rcre_ag_from_tensor reinterpreted ANY dtype as float"
 * (probe F: U8 {200,100,50,250} -> -2.3e35).
 *
 * Pinned contract: U8 and I32 are rejected (empty struct, data == NULL);
 * F32 is accepted with shape/strides/is_view carried over.
 */
static void test_from_tensor_dtype_matrix(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);
    float* grad = (float*)rcre_arena_alloc(arena, 16, 64);
    ASSERT_NOT_NULL(grad);

    uint8_t px[4] = { 200, 100, 50, 250 };
    RCRE_Tensor u8 = {
        .data = px, .ndim = 1, .dtype = RCRE_DTYPE_U8, .layout = RCRE_LAYOUT_TABULAR,
        .shape = { 4, 1, 1, 1 }, .strides = { 1, 0, 0, 0 }
    };
    RCRE_AGTensor a_u8 = rcre_ag_from_tensor(&u8, grad, "u8");
    ASSERT_NULL(a_u8.data);
    ASSERT_UINT_EQ(0, a_u8.numel);

    int32_t ix[4] = { 1, 2, 3, 4 };
    RCRE_Tensor i32 = {
        .data = ix, .ndim = 1, .dtype = RCRE_DTYPE_I32, .layout = RCRE_LAYOUT_TABULAR,
        .shape = { 4, 1, 1, 1 }, .strides = { 1, 0, 0, 0 }
    };
    RCRE_AGTensor a_i32 = rcre_ag_from_tensor(&i32, grad, "i32");
    ASSERT_NULL(a_i32.data);

    float fx[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    RCRE_Tensor f32 = {
        .data = fx, .ndim = 2, .dtype = RCRE_DTYPE_F32, .layout = RCRE_LAYOUT_TABULAR,
        .shape = { 2, 4, 1, 1 }, .strides = { 4, 1, 0, 0 }
    };
    RCRE_AGTensor a_f32 = rcre_ag_from_tensor(&f32, grad, "f32");
    ASSERT_NOT_NULL(a_f32.data);
    ASSERT_TRUE(a_f32.data == fx);
    ASSERT_TRUE(a_f32.grad == grad);
    ASSERT_UINT_EQ(2, a_f32.ndim);
    ASSERT_UINT_EQ(8, a_f32.numel);
    ASSERT_UINT_EQ(4, a_f32.shape[1]);
    ASSERT_UINT_EQ(4, a_f32.strides[0]);
    ASSERT_TRUE(a_f32.is_view);

    rcre_arena_destroy(arena);
}

/*
 * REPORT §2.3 (P0->P1) — "tape_push_node memcpied before its NULL check and
 * its return value was ignored by all rcre_autograd_* ops".
 *
 * Pinned contract: when the arena cannot service an activation allocation,
 * the op returns NULL (no crash, no torn state) and the arena remains
 * usable after a reset.
 */
static void test_tape_arena_exhaustion_graceful_null(void) {
    RCRE_Arena* arena = rcre_arena_create(4096);
    ASSERT_NOT_NULL(arena);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 8);
    ASSERT_NOT_NULL(tape);

    RCRE_AGTensor* X = rcre_ag_tensor_create_2d(arena, 4, 4); /* struct+data+grad */
    RCRE_AGTensor* W = rcre_ag_tensor_create_2d(arena, 4, 4);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(W);

    /* Fill the arena (leaving 128B) so the next activation — struct(64) +
     * data(64) + grad(64) = 192B — cannot fit and the op must return NULL.
     * (Note: rcre_arena_alloc returns NULL if even the alignment padding
     * would exceed capacity, so we never request the exact free remainder.) */
    size_t free_now = arena->capacity - arena->offset;
    if (free_now > 128) {
        void* sink = rcre_arena_alloc(arena, free_now - 128, 64);
        ASSERT_NOT_NULL(sink);
    }

    RCRE_AGTensor* out = rcre_autograd_matmul(tape, X, W);
    ASSERT_NULL(out); /* graceful NULL — pre-fix this path could crash */

    /* Destroy the tape BEFORE the reset: the tape struct is arena-resident,
     * so reset + fresh allocations would overwrite it (standard arena
     * semantics — same as any arena-allocated object). */
    rcre_tensor_tape_destroy(tape);

    /* Arena resets and serves allocations again. */
    rcre_arena_reset(arena);
    RCRE_AGTensor* again = rcre_ag_tensor_create_2d(arena, 4, 4);
    ASSERT_NOT_NULL(again);

    rcre_arena_destroy(arena);
}

/*
 * REPORT §2.3 (P1) — "rcre_batch_iter_next never checked the OUTPUT tensors"
 * (probe E: silent cross-arena OOB write, invisible to ASan).
 *
 * The existing regression covers both outputs undersized; this isolates
 * batch_y alone to pin each side of the guard independently.
 */
static void test_batch_iter_y_only_undersized(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_AGTensor* X = rcre_ag_tensor_create_2d(arena, 10, 4);
    RCRE_AGTensor* y = rcre_ag_tensor_create_2d(arena, 10, 2);
    RCRE_BatchIter* iter = rcre_batch_iter_create(arena, X, y, 8, false);
    ASSERT_NOT_NULL(iter);

    RCRE_AGTensor* bx = rcre_ag_tensor_create_2d(arena, 8, 4); /* fits */
    RCRE_AGTensor* by_small = rcre_ag_tensor_create_2d(arena, 4, 2); /* too small */
    ASSERT_FALSE(rcre_batch_iter_next(iter, bx, by_small));
    ASSERT_UINT_EQ(0, iter->current_idx); /* nothing consumed on failure */

    RCRE_AGTensor* by_ok = rcre_ag_tensor_create_2d(arena, 8, 2);
    ASSERT_TRUE(rcre_batch_iter_next(iter, bx, by_ok));
    ASSERT_UINT_EQ(8, iter->current_idx);

    rcre_arena_destroy(arena);
}

/*
 * REPORT §2.3 (P2) — "the merge added a third xorshift copy in
 * tensor_autograd.c and a fourth in data.c".
 *
 * Fix under test: one XorShift64* implementation in src/core/rcre_rng.h.
 * Pinned: identical seeds produce identical next/uniform/gaussian streams,
 * and XorShift64* is what powers the public tensor-init APIs (reproducible
 * initialization for deterministic training runs).
 */
static void test_unified_rng_determinism(void) {
    uint64_t s1 = 0x9E3779B97F4A7C15ULL, s2 = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 100; ++i) {
        ASSERT_UINT_EQ(rcre_rng_next(&s1), rcre_rng_next(&s2));
    }

    s1 = 42; s2 = 42;
    for (int i = 0; i < 50; ++i) {
        ASSERT_FLOAT_NEAR(rcre_rng_uniform_f32(&s1, -1.0f, 1.0f),
                          rcre_rng_uniform_f32(&s2, -1.0f, 1.0f), 0.0f);
        ASSERT_FLOAT_NEAR(rcre_rng_gaussian_f32(&s1, 0.0f, 1.0f),
                          rcre_rng_gaussian_f32(&s2, 0.0f, 1.0f), 0.0f);
    }

    /* Same seed -> bit-identical XorShift64* sequence (sanity of the mul). */
    uint64_t a = 7, b = 7;
    uint64_t pa = rcre_rng_next(&a), pb = rcre_rng_next(&b);
    ASSERT_UINT_EQ(pa, pb);
    ASSERT_UINT_EQ(0, pa == 0); /* xorshift64* never emits 0 from nonzero seed */

    /* Public init APIs stay reproducible through the unified RNG. */
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);
    RCRE_AGTensor* w1 = rcre_ag_tensor_create_2d(arena, 64, 64);
    RCRE_AGTensor* w2 = rcre_ag_tensor_create_2d(arena, 64, 64);
    uint64_t r1 = 12345, r2 = 12345;
    rcre_ag_tensor_init_xavier(w1, 64, 64, &r1);
    rcre_ag_tensor_init_xavier(w2, 64, 64, &r2);
    for (int i = 0; i < 4096; ++i) {
        ASSERT_FLOAT_NEAR(w1->data[i], w2->data[i], 0.0f);
    }
    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* PART 2 — REPORT §3/§4: OPEN items (pass today; continue the work here)    */
/* ========================================================================= */

/*
 * REPORT §3 (open item 1) — "The two-tensor-type split stays
 * (RCRE_Tensor + RCRE_AGTensor) … the merge left rcre_ag_to_tensor /
 * rcre_ag_from_tensor completely unused (dead code)".
 *
 * This test is the coverage that bridge was missing: a full
 * AGTensor -> RCRE_Tensor -> AGTensor -> RCRE_Tensor roundtrip proving the
 * bridge is data-pointer-identity and shape/stride faithful.
 *
 * CONTINUE HERE (consolidation plan, tensor unification):
 *   1. Add RCRE_DTYPE_F32 grad support to RCRE_Tensor (or an owning wrapper)
 *      and delete RCRE_AGTensor + the bridge in a single PR.
 *   2. Migrate src/autograd + src/nn to the unified type.
 *   3. Until then: every new autograd op that needs to touch core CV/ML
 *      (e.g. conv2d autograd) should go through these bridge functions so
 *      this roundtrip stays the canonical interop path.
 */
static void test_tensor_bridge_roundtrip(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_AGTensor* ag = rcre_ag_tensor_create_2d(arena, 3, 4);
    ASSERT_NOT_NULL(ag);
    for (int i = 0; i < 12; ++i) ag->data[i] = (float)i;

    /* AG -> core */
    RCRE_Tensor t = rcre_ag_to_tensor(ag);
    ASSERT_TRUE(t.data == ag->data);           /* zero-copy, same storage */
    ASSERT_UINT_EQ(2, t.ndim);
    ASSERT_UINT_EQ(RCRE_DTYPE_F32, t.dtype);
    ASSERT_UINT_EQ(3, t.shape[0]);
    ASSERT_UINT_EQ(4, t.shape[1]);
    ASSERT_UINT_EQ(4, (size_t)t.strides[0]);
    ASSERT_UINT_EQ(1, (size_t)t.strides[1]);
    ASSERT_TRUE(rcre_tensor_is_contiguous(&t));

    /* core -> AG (view over the same buffer, with a grad slot) */
    RCRE_AGTensor ag2 = rcre_ag_from_tensor(&t, ag->grad, "rt");
    ASSERT_TRUE(ag2.data == ag->data);
    ASSERT_TRUE(ag2.grad == ag->grad);
    ASSERT_UINT_EQ(2, ag2.ndim);
    ASSERT_UINT_EQ(12, ag2.numel);
    ASSERT_UINT_EQ(4, ag2.shape[1]);
    ASSERT_TRUE(ag2.is_view);

    /* AG (view) -> core again: metadata must survive the second hop */
    RCRE_Tensor t2 = rcre_ag_to_tensor(&ag2);
    ASSERT_TRUE(t2.data == ag->data);
    ASSERT_UINT_EQ(2, t2.ndim);
    ASSERT_UINT_EQ(3, t2.shape[0]);
    ASSERT_UINT_EQ(4, t2.shape[1]);
    ASSERT_UINT_EQ(4, (size_t)t2.strides[0]);

    /* The view must observe writes through the original buffer. */
    ag->data[0] = 99.0f;
    ASSERT_FLOAT_NEAR(99.0f, ((float*)t2.data)[0], EPSILON);

    rcre_arena_destroy(arena);
}

/*
 * REPORT §3 (open item 3) — "Recursive scalar topo-sort: fixed the leak,
 * kept the recursion. build_topo() is a recursive DFS; stack depth ==
 * graph depth."
 *
 * This test pins the CURRENT CEILING: a 20,000-deep differentiable chain
 * (40,001 DAG nodes) differentiates correctly on a default 8MB stack.
 *
 * CONTINUE HERE:
 *   Convert build_topo() to an iterative explicit-stack traversal
 *   (arena-backed node stack, same postorder semantics), then:
 *   - keep this test at 20k (it must still pass),
 *   - raise the depth (100k+) to prove the stack is no longer the limit,
 *   - that makes the scalar engine safe for WASM (64KB stacks) / embedded,
 *     matching rcre's "runs everywhere" promise.
 */
static void test_scalar_deep_chain_recursion_ceiling(void) {
    const size_t N = 20000; /* recursion depth; see "CONTINUE HERE" above */

    RCRE_GraphCtx* ctx = rcre_graph_ctx_create(1 << 20, 16 << 20);
    ASSERT_NOT_NULL(ctx);

    RCRE_ScalarVal* x0 = rcre_scalar_val_param(ctx, 0.0, "x0");
    ASSERT_NOT_NULL(x0);

    RCRE_ScalarVal* acc = x0;
    for (size_t i = 0; i < N; ++i) {
        RCRE_ScalarVal* one = rcre_scalar_val_const(ctx, 1.0);
        ASSERT_NOT_NULL(one);
        acc = rcre_scalar_add(ctx, acc, one);
        ASSERT_NOT_NULL(acc);
    }

    ASSERT_DOUBLE_NEAR((double)N, acc->data, 1e-9);

    rcre_scalar_backward(acc);
    ASSERT_DOUBLE_NEAR(1.0, acc->grad, 1e-12);
    ASSERT_DOUBLE_NEAR(1.0, x0->grad, 1e-12); /* gradient flows the whole chain */

    rcre_graph_ctx_destroy(ctx);
}

/*
 * REPORT §3 (open item 4) — "Loss functions' dual write (grad_out->grad AND
 * grad_out->data) — unusual, but the example and tests depend on it".
 *
 * This test pins the EXACT current contract so a future API split
 * (separate "compute gradient into buffer" vs "accumulate into tape")
 * breaks loudly instead of silently:
 *   - separate grad_out tensor: gradient is written to BOTH .grad and .data
 *   - grad_out == preds: gradient written to .grad only; preds->data (the
 *     forward output) is never clobbered.
 *
 * CONTINUE HERE: consider splitting into rcre_loss_mse_value() (loss only)
 * + rcre_loss_mse_grad_into() (explicit buffer), and retire the .data mirror.
 */
static void test_loss_dual_write_contract(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_AGTensor* preds = rcre_ag_tensor_create_1d(arena, 4);
    RCRE_AGTensor* targets = rcre_ag_tensor_create_1d(arena, 4);
    RCRE_AGTensor* g = rcre_ag_tensor_create_1d(arena, 4);
    for (int i = 0; i < 4; ++i) {
        preds->data[i] = (float)(i + 1);   /* 1,2,3,4 */
        g->data[i] = -123.45f;             /* sentinel: must be overwritten */
        g->grad[i] = -123.45f;
    }
    targets->data[0] = 2.0f; targets->data[1] = 2.0f;
    targets->data[2] = 4.0f; targets->data[3] = 4.0f;

    /* diff = [-1, 0, -1, 0], scale = 2/4 => g = [-0.5, 0, -0.5, 0] */
    float loss = rcre_loss_mse(preds, targets, g);
    ASSERT_FLOAT_NEAR(0.5f, loss, EPSILON);

    /* Contract A: separate tensor receives the gradient in BOTH slots */
    ASSERT_FLOAT_NEAR(-0.5f, g->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(-0.5f, g->data[0], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, g->grad[1], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, g->data[1], EPSILON);

    /* Contract B: grad_out == preds touches .grad only, never .data */
    for (int i = 0; i < 4; ++i) preds->grad[i] = 0.0f;
    float loss2 = rcre_loss_mse(preds, targets, preds);
    ASSERT_FLOAT_NEAR(0.5f, loss2, EPSILON);
    ASSERT_FLOAT_NEAR(-0.5f, preds->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(1.0f, preds->data[0], EPSILON); /* forward data intact */
    ASSERT_FLOAT_NEAR(4.0f, preds->data[3], EPSILON);

    rcre_arena_destroy(arena);
}

/*
 * REPORT §3 (open item 5) — "Cross-entropy silently clamps out-of-range
 * class indices — kept the clamp, but the precondition now belongs in the
 * docs; consider an assert in debug builds."
 *
 * This test pins the clamp: targets[b] = 999 with num_classes = 10 must not
 * crash, and must behave EXACTLY like target 9 (num_classes - 1).
 *
 * CONTINUE HERE:
 *   - document the precondition in include/rcre_autograd.h,
 *   - decide the failure policy: debug-assert + release-clamp is the
 *     suggested split (add #if defined(RCRE_DEBUG) around the clamp),
 *   - a dataset validator in src/nn/data.c could reject label columns with
 *     values >= num_classes at load time, before training starts.
 */
static void test_cross_entropy_out_of_range_clamp(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    const size_t B = 2, C = 10;
    RCRE_AGTensor* logits = rcre_ag_tensor_create_2d(arena, B, C); /* all zeros */
    RCRE_AGTensor* g = rcre_ag_tensor_create_2d(arena, B, C);

    size_t tgt_clamped[2] = { 0, 999 }; /* 999 -> clamped to 9 */
    size_t tgt_explicit[2] = { 0, 9 };

    /* Uniform logits: every prob = 0.1, loss per row = -log(0.1) */
    float loss_clamped = rcre_loss_cross_entropy_logits(logits, tgt_clamped, B, C, g);
    ASSERT_FLOAT_NEAR(2.302585f, loss_clamped, 1e-4f);

    /* The clamp must be behaviour-identical to the explicit boundary class */
    float loss_explicit = rcre_loss_cross_entropy_logits(logits, tgt_explicit, B, C, g);
    ASSERT_FLOAT_NEAR(loss_clamped, loss_explicit, 1e-6f);

    /* Gradient of row 1, clamped class 9: (p - 1)/B = (0.1 - 1)/2 */
    ASSERT_FLOAT_NEAR(-0.45f, g->grad[1 * C + 9], 1e-4f);
    /* ...and every other class of row 1: p/B = 0.05 */
    ASSERT_FLOAT_NEAR(0.05f, g->grad[1 * C + 0], 1e-4f);
    ASSERT_FLOAT_NEAR(0.05f, g->grad[1 * C + 8], 1e-4f);

    rcre_arena_destroy(arena);
}

/*
 * REPORT §4 — "Scalar engine's global visited-epoch: single-thread only
 * (now documented)."
 *
 * g_visited_epoch is a process-global reused across every backward/zero_grad
 * call. This test pins that the reuse is CORRECT single-threaded: two
 * consecutive backward passes over the same graph (with zero_grad between)
 * must produce identical gradients — i.e. the epoch mechanism re-enters
 * cleanly and stale visited_ids from the previous pass are never mistaken
 * for "already visited".
 *
 * CONTINUE HERE (if multi-threaded scalar AD is ever required):
 *   move the epoch into the RCRE_ScalarVal graph root (or make it
 *   thread_local) so concurrent backward() calls cannot interleave;
 *   the tensor-tape engine already has this property (per-tape storage).
 */
static void test_scalar_epoch_reuse_single_thread(void) {
    RCRE_GraphCtx* ctx = rcre_graph_ctx_create(1 << 20, 1 << 20);
    ASSERT_NOT_NULL(ctx);

    /* p = (x + y) * x  with x=2, y=3: dp=1 => dx = s + s' = 5 + 2 = 7, dy = 2 */
    RCRE_ScalarVal* x = rcre_scalar_val_param(ctx, 2.0, "x");
    RCRE_ScalarVal* y = rcre_scalar_val_param(ctx, 3.0, "y");
    RCRE_ScalarVal* s = rcre_scalar_add(ctx, x, y);
    RCRE_ScalarVal* p = rcre_scalar_mul(ctx, s, x);
    ASSERT_NOT_NULL(p);
    ASSERT_DOUBLE_NEAR(10.0, p->data, 1e-12);

    rcre_scalar_backward(p);
    double pass1_x = x->grad, pass1_y = y->grad;
    ASSERT_DOUBLE_NEAR(7.0, pass1_x, 1e-12);
    ASSERT_DOUBLE_NEAR(2.0, pass1_y, 1e-12);

    /* Second pass over the same graph must be bit-identical. */
    rcre_scalar_zero_grad(p);
    rcre_scalar_backward(p);
    ASSERT_DOUBLE_NEAR(pass1_x, x->grad, 1e-12);
    ASSERT_DOUBLE_NEAR(pass1_y, y->grad, 1e-12);
    ASSERT_DOUBLE_NEAR(1.0, p->grad, 1e-12); /* backward reseeds root to 1.0 */

    rcre_graph_ctx_destroy(ctx);
}

int main(void) {
    test_suite_begin("RCRE Merger-Report Findings (REGRESSION + OPEN GAPS)");

    /* PART 1 — §2.3 fixed bugs, granular regressions */
    TEST_RUN(test_agtensor_shape_zeroed_beyond_ndim);
    TEST_RUN(test_csv_old_boundary_and_dynamic_growth);
    TEST_RUN(test_from_tensor_dtype_matrix);
    TEST_RUN(test_tape_arena_exhaustion_graceful_null);
    TEST_RUN(test_batch_iter_y_only_undersized);
    TEST_RUN(test_unified_rng_determinism);

    /* PART 2 — §3/§4 open items (current behaviour pinned; see headers) */
    TEST_RUN(test_tensor_bridge_roundtrip);
    TEST_RUN(test_scalar_deep_chain_recursion_ceiling);
    TEST_RUN(test_loss_dual_write_contract);
    TEST_RUN(test_cross_entropy_out_of_range_clamp);
    TEST_RUN(test_scalar_epoch_reuse_single_thread);

    return test_suite_report();
}
