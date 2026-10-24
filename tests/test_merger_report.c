/**
 * @file test_merger_report.c
 * @brief Regression & contract tests mapped 1:1 to the findings in
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
 *   PART 2 — REPORT §3/§4: items that were OPEN after the hardening pass
 *           and were IMPLEMENTED in v0.2.2 (unified tensor type, iterative
 *           topo-sort, grad-only loss contract, CE clamp policy + label
 *           validator, thread_local visited epoch). These tests pin the NEW
 *           contracts so any future regression breaks loudly.
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
 * Fix under test: rcre_tensor_alloc memsets the struct before field
 * assignment, so every shape[i]/strides[i] with i >= ndim reads back 0,
 * never heap/arena garbage.
 */
static void test_tensor_alloc_shape_zeroed_beyond_ndim(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_Tensor* v1 = rcre_tensor_alloc_1d(arena, 5);
    ASSERT_NOT_NULL(v1);
    ASSERT_UINT_EQ(1, v1->ndim);
    for (int i = 1; i < 4; ++i) {
        ASSERT_UINT_EQ(0, v1->shape[i]);
        ASSERT_UINT_EQ(0, (size_t)v1->strides[i]);
    }
    ASSERT_UINT_EQ(5, v1->numel);
    ASSERT_TRUE(v1->dtype == RCRE_DTYPE_F32);

    RCRE_Tensor* v2 = rcre_tensor_alloc_2d(arena, 3, 7);
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
    RCRE_Tensor* t1 = rcre_data_load_csv(arena, p1, false, 0);
    ASSERT_NOT_NULL(t1);
    ASSERT_UINT_EQ(2, t1->shape[0]);
    ASSERT_UINT_EQ(373, t1->shape[1]);
    ASSERT_FLOAT_NEAR(1000000372.0f, rcre_tensor_f32_data(t1)[372], 1.0f); /* row0 last col */
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
    RCRE_Tensor* t2 = rcre_data_load_csv(arena, p2, false, 0);
    ASSERT_NOT_NULL(t2);
    ASSERT_UINT_EQ(3, t2->shape[0]);
    ASSERT_UINT_EQ(20000, t2->shape[1]);
    ASSERT_FLOAT_NEAR(2 * 100000 + 19999, rcre_tensor_f32_data(t2)[2 * 20000 + 19999], 1.0f);
    remove(p2);

    /* (c) final line without trailing newline must still be parsed */
    const char* p3 = "/tmp/rcre_mr_nonewline.csv";
    f = fopen(p3, "w");
    ASSERT_NOT_NULL(f);
    fprintf(f, "1,2\n3,4\n5,6"); /* no \n after last row */
    fclose(f);
    RCRE_Tensor* t3 = rcre_data_load_csv(arena, p3, false, 0);
    ASSERT_NOT_NULL(t3);
    ASSERT_UINT_EQ(3, t3->shape[0]);
    ASSERT_UINT_EQ(2, t3->shape[1]);
    ASSERT_FLOAT_NEAR(6.0f, rcre_tensor_f32_data(t3)[5], EPSILON);
    remove(p3);

    rcre_arena_destroy(arena);
}

/*
 * REPORT §2.3 (P2) — "rcre_ag_from_tensor reinterpreted ANY dtype as float"
 * (probe F: U8 {200,100,50,250} -> -2.3e35).
 *
 * v0.2.2: the bridge functions are GONE (type unification). The dtype gate
 * now lives in every autograd/training op itself: non-F32 RCRE_Tensors are
 * rejected with NULL instead of being reinterpreted. Pinned here for U8 and
 * I32, with F32 accepted.
 */
static void test_autograd_ops_reject_non_f32(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 8);
    ASSERT_NOT_NULL(tape);

    uint8_t px[4] = { 200, 100, 50, 250 };
    RCRE_Tensor u8 = {
        .data = px, .ndim = 1, .dtype = RCRE_DTYPE_U8, .layout = RCRE_LAYOUT_TABULAR,
        .shape = { 4, 1, 1, 1 }, .strides = { 1, 0, 0, 0 }, .numel = 4
    };
    int32_t ix[4] = { 1, 2, 3, 4 };
    RCRE_Tensor i32 = {
        .data = ix, .ndim = 1, .dtype = RCRE_DTYPE_I32, .layout = RCRE_LAYOUT_TABULAR,
        .shape = { 4, 1, 1, 1 }, .strides = { 1, 0, 0, 0 }, .numel = 4
    };

    ASSERT_NULL(rcre_autograd_relu(tape, &u8));
    ASSERT_NULL(rcre_autograd_relu(tape, &i32));

    /* F32 core tensor (value type) is accepted and differentiates. */
    float fx[4] = { -1.0f, 2.0f, -3.0f, 4.0f };
    size_t s2[2] = { 2, 2 };
    RCRE_Tensor f32 = rcre_tensor_from_buffer(fx, s2, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
    RCRE_Tensor* r = rcre_autograd_relu(tape, &f32);
    ASSERT_NOT_NULL(r);
    ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_f32_data(r)[0], EPSILON);
    ASSERT_FLOAT_NEAR(2.0f, rcre_tensor_f32_data(r)[1], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_f32_data(r)[2], EPSILON);
    ASSERT_FLOAT_NEAR(4.0f, rcre_tensor_f32_data(r)[3], EPSILON);

    /* Non-F32 params must also be rejected by the optimizers. */
    RCRE_Tensor* bad_params[1] = { &u8 };
    ASSERT_NULL(rcre_tensor_optim_sgd_create(arena, bad_params, 1, 0.1f, 0.9f, 0.0f));

    rcre_tensor_tape_destroy(tape);
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

    RCRE_Tensor* X = rcre_tensor_alloc_2d(arena, 4, 4); /* struct+data+grad */
    RCRE_Tensor* W = rcre_tensor_alloc_2d(arena, 4, 4);
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

    RCRE_Tensor* out = rcre_autograd_matmul(tape, X, W);
    ASSERT_NULL(out); /* graceful NULL — pre-fix this path could crash */

    /* Destroy the tape BEFORE the reset: the tape struct is arena-resident,
     * so reset + fresh allocations would overwrite it (standard arena
     * semantics — same as any arena-allocated object). */
    rcre_tensor_tape_destroy(tape);

    /* Arena resets and serves allocations again. */
    rcre_arena_reset(arena);
    RCRE_Tensor* again = rcre_tensor_alloc_2d(arena, 4, 4);
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

    RCRE_Tensor* X = rcre_tensor_alloc_2d(arena, 10, 4);
    RCRE_Tensor* y = rcre_tensor_alloc_2d(arena, 10, 2);
    RCRE_BatchIter* iter = rcre_batch_iter_create(arena, X, y, 8, false);
    ASSERT_NOT_NULL(iter);

    RCRE_Tensor* bx = rcre_tensor_alloc_2d(arena, 8, 4); /* fits */
    RCRE_Tensor* by_small = rcre_tensor_alloc_2d(arena, 4, 2); /* too small */
    ASSERT_FALSE(rcre_batch_iter_next(iter, bx, by_small));
    ASSERT_UINT_EQ(0, iter->current_idx); /* nothing consumed on failure */

    RCRE_Tensor* by_ok = rcre_tensor_alloc_2d(arena, 8, 2);
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
    RCRE_Tensor* w1 = rcre_tensor_alloc_2d(arena, 64, 64);
    RCRE_Tensor* w2 = rcre_tensor_alloc_2d(arena, 64, 64);
    uint64_t r1 = 12345, r2 = 12345;
    rcre_tensor_init_xavier(w1, 64, 64, &r1);
    rcre_tensor_init_xavier(w2, 64, 64, &r2);
    for (int i = 0; i < 4096; ++i) {
        ASSERT_FLOAT_NEAR(rcre_tensor_f32_data(w1)[i], rcre_tensor_f32_data(w2)[i], 0.0f);
    }
    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* PART 2 — REPORT §3/§4: implemented in v0.2.2 (contracts now pinned)      */
/* ========================================================================= */

/*
 * REPORT §3 (open item 1) — "The two-tensor-type split stays
 * (RCRE_Tensor + RCRE_AGTensor) … rcre_ag_to_tensor / rcre_ag_from_tensor
 * completely unused (dead code)".
 *
 * DONE in v0.2.2: RCRE_AGTensor and the bridge are DELETED. One type,
 * RCRE_Tensor, carries the optional `grad` buffer. This test pins the
 * payoff: core (value-style) F32 tensors feed autograd operators directly,
 * training tensors accept core view ops, and views never share the parent's
 * grad buffer.
 */
static void test_unified_core_view_feeds_autograd(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 8);
    ASSERT_NOT_NULL(tape);

    /* (1) A core value tensor (rcre_tensor_from_buffer) differentiates. */
    float data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    size_t cs[2] = { 2, 4 };
    RCRE_Tensor core = rcre_tensor_from_buffer(data, cs, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
    ASSERT_TRUE(core.is_view);

    RCRE_Tensor* W = rcre_tensor_alloc_2d(arena, 4, 2);
    for (int i = 0; i < 8; ++i) rcre_tensor_f32_data(W)[i] = 1.0f;

    RCRE_Tensor* C = rcre_autograd_matmul(tape, &core, W); /* &core: stack value, alive until backward */
    ASSERT_NOT_NULL(C);
    /* C[0,0] = 1+2+3+4 = 10 */
    ASSERT_FLOAT_NEAR(10.0f, rcre_tensor_f32_data(C)[0], EPSILON);
    /* C[1,0] = 5+6+7+8 = 26 */
    ASSERT_FLOAT_NEAR(26.0f, rcre_tensor_f32_data(C)[2], EPSILON);

    /* Seed dC = 1 and backprop. `core` has no grad buffer: backward must
     * silently skip it; W must receive its gradient.
     * dW[k][n] = core[0][k] + core[1][k]:  dW[0][0]=1+5=6, dW[3][1]=4+8=12 */
    for (int i = 0; i < 4; ++i) C->grad[i] = 1.0f;
    rcre_tensor_tape_backward(tape);
    ASSERT_FLOAT_NEAR(6.0f, W->grad[0], EPSILON);  /* dW[0][0] */
    ASSERT_FLOAT_NEAR(12.0f, W->grad[7], EPSILON); /* dW[3][1] */
    ASSERT_FLOAT_NEAR(8.0f, W->grad[2], EPSILON);  /* dW[1][0] */

    /* (2) Core view ops on a training tensor: views clear the grad slot. */
    RCRE_Tensor* T = rcre_tensor_alloc_2d(arena, 4, 4);
    ASSERT_NOT_NULL(T->grad);

    size_t starts[2] = { 1, 1 }, ends[2] = { 3, 3 };
    RCRE_Tensor slice = rcre_tensor_slice(T, starts, ends);
    ASSERT_TRUE(slice.is_view);
    ASSERT_NULL(slice.grad);  /* pinned: views never share parent grad */
    ASSERT_UINT_EQ(2, slice.shape[0]);
    ASSERT_TRUE(T->grad != nullptr);

    size_t rs[2] = { 2, 8 };
    RCRE_Tensor* rv = rcre_tensor_reshape_view(arena, T, rs, 2);
    ASSERT_NOT_NULL(rv);
    ASSERT_TRUE(rv->is_view);
    ASSERT_NULL(rv->grad);
    ASSERT_TRUE(rv->data == T->data); /* zero-copy */
    ASSERT_UINT_EQ(8, rv->shape[1]);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(arena);
}

/*
 * REPORT §3 (open item 3) — "Recursive scalar topo-sort: fixed the leak,
 * kept the recursion. build_topo() is a recursive DFS; stack depth ==
 * graph depth."
 *
 * DONE in v0.2.2: build_topo() is iterative (explicit heap frame stack,
 * same postorder semantics). This test pins the result: a 100,000-deep
 * differentiable chain (100,001 DAG nodes) differentiates correctly —
 * depth is no longer bounded by the C call stack, so the scalar engine is
 * safe on WASM (64KB stacks) / embedded targets.
 */
static void test_scalar_deep_chain_iterative_topo(void) {
    const size_t N = 100000; /* 5x the old 20k pinned ceiling */

    RCRE_GraphCtx* ctx = rcre_graph_ctx_create(1 << 20, 24 << 20);
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
 * DONE in v0.2.2: the .data mirror is RETIRED. Pinned contract:
 *   - the analytic gradient is written to grad_out->grad ONLY;
 *   - grad_out->data (the forward buffer) is NEVER modified, regardless of
 *     whether grad_out is a separate tensor or grad_out == preds;
 *   - borrowed views (grad == NULL) receive only the loss value.
 */
static void test_loss_grad_only_contract(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_Tensor* preds = rcre_tensor_alloc_1d(arena, 4);
    RCRE_Tensor* targets = rcre_tensor_alloc_1d(arena, 4);
    RCRE_Tensor* g = rcre_tensor_alloc_1d(arena, 4);
    for (int i = 0; i < 4; ++i) {
        rcre_tensor_f32_data(preds)[i] = (float)(i + 1);  /* 1,2,3,4 */
        rcre_tensor_f32_data(g)[i] = -123.45f;            /* sentinel: must SURVIVE */
        g->grad[i] = -123.45f;                            /* sentinel: must be overwritten */
    }
    rcre_tensor_f32_data(targets)[0] = 2.0f; rcre_tensor_f32_data(targets)[1] = 2.0f;
    rcre_tensor_f32_data(targets)[2] = 4.0f; rcre_tensor_f32_data(targets)[3] = 4.0f;

    /* diff = [-1, 0, -1, 0], scale = 2/4 => grad = [-0.5, 0, -0.5, 0] */
    float loss = rcre_loss_mse(preds, targets, g);
    ASSERT_FLOAT_NEAR(0.5f, loss, EPSILON);

    /* Gradient lands in ->grad ... */
    ASSERT_FLOAT_NEAR(-0.5f, g->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f,  g->grad[1], EPSILON);
    ASSERT_FLOAT_NEAR(-0.5f, g->grad[2], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f,  g->grad[3], EPSILON);
    /* ... and the forward buffer keeps its sentinel (mirror retired). */
    ASSERT_FLOAT_NEAR(-123.45f, rcre_tensor_f32_data(g)[0], 0.0f);
    ASSERT_FLOAT_NEAR(-123.45f, rcre_tensor_f32_data(g)[3], 0.0f);

    /* grad_out == preds: .grad written, .data (the forward output) intact. */
    for (int i = 0; i < 4; ++i) preds->grad[i] = 0.0f;
    float loss2 = rcre_loss_mse(preds, targets, preds);
    ASSERT_FLOAT_NEAR(0.5f, loss2, EPSILON);
    ASSERT_FLOAT_NEAR(-0.5f, preds->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(1.0f, rcre_tensor_f32_data(preds)[0], EPSILON);
    ASSERT_FLOAT_NEAR(4.0f, rcre_tensor_f32_data(preds)[3], EPSILON);

    /* Borrowed view (no grad buffer): loss value only, no crash. */
    float raw[4] = { 1, 2, 3, 4 };
    size_t vs[1] = { 4 };
    RCRE_Tensor borrowed = rcre_tensor_from_buffer(raw, vs, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
    float loss3 = rcre_loss_mse(&borrowed, targets, &borrowed);
    ASSERT_FLOAT_NEAR(0.5f, loss3, EPSILON);

    rcre_arena_destroy(arena);
}

/*
 * REPORT §3 (open item 5) — "Cross-entropy silently clamps out-of-range
 * class indices — kept the clamp, but the precondition now belongs in the
 * docs; consider an assert in debug builds."
 *
 * DONE in v0.2.2: failure policy implemented as
 *   - RCRE_DEBUG builds: one warning per call to stderr, then clamp;
 *   - release builds: clamp;
 *   - dataset hygiene: rcre_data_validate_labels() reports the first
 *     invalid index (or -1) so callers can reject bad label columns
 *     before training.
 * Pinned here: the clamp stays behaviour-identical to the boundary class,
 * and the validator finds exactly the planted bad index.
 */
static void test_cross_entropy_clamp_and_label_validation(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    const size_t B = 2, C = 10;
    RCRE_Tensor* logits = rcre_tensor_alloc_2d(arena, B, C); /* all zeros */
    RCRE_Tensor* g = rcre_tensor_alloc_2d(arena, B, C);

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

    /* Label validator: -1 when clean, first bad index when not. */
    ASSERT_INT_EQ(-1, rcre_data_validate_labels(tgt_explicit, 2, C));
    ASSERT_INT_EQ(1, rcre_data_validate_labels(tgt_clamped, 2, C));
    size_t clean[3] = { 0, 5, 9 };
    ASSERT_INT_EQ(-1, rcre_data_validate_labels(clean, 3, C));
    ASSERT_INT_EQ(-1, rcre_data_validate_labels(nullptr, 3, C));

    rcre_arena_destroy(arena);
}

/*
 * REPORT §4 — "Scalar engine's global visited-epoch: single-thread only".
 *
 * DONE in v0.2.2: the epoch counter is now `thread_local`, so concurrent
 * scalar backward() calls on DISJOINT graphs from different threads cannot
 * interleave epochs. (Traversing one graph from two threads is still
 * unsupported — grad accumulation is not atomic.)
 *
 * This test pins the reuse semantics: two consecutive backward passes over
 * the same graph (with zero_grad between) must produce identical gradients —
 * i.e. the epoch mechanism re-enters cleanly and stale visited_ids from the
 * previous pass are never mistaken for "already visited".
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
    test_suite_begin("RCRE Merger-Report Findings (REGRESSION + v0.2.2 CONTRACTS)");

    /* PART 1 — §2.3 fixed bugs, granular regressions */
    TEST_RUN(test_tensor_alloc_shape_zeroed_beyond_ndim);
    TEST_RUN(test_csv_old_boundary_and_dynamic_growth);
    TEST_RUN(test_autograd_ops_reject_non_f32);
    TEST_RUN(test_tape_arena_exhaustion_graceful_null);
    TEST_RUN(test_batch_iter_y_only_undersized);
    TEST_RUN(test_unified_rng_determinism);

    /* PART 2 — §3/§4 items implemented in v0.2.2 (new contracts pinned) */
    TEST_RUN(test_unified_core_view_feeds_autograd);
    TEST_RUN(test_scalar_deep_chain_iterative_topo);
    TEST_RUN(test_loss_grad_only_contract);
    TEST_RUN(test_cross_entropy_clamp_and_label_validation);
    TEST_RUN(test_scalar_epoch_reuse_single_thread);

    return test_suite_report();
}
