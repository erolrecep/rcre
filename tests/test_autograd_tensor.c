/**
 * @file test_autograd_tensor.c
 * @brief Unit tests for RCRE Dynamic Tensor Tape Autograd Engine
 */

#include "rcre.h"
#include "rcre_autograd.h"
#include "test_framework.h"
#include <math.h>

#define EPSILON 1e-4f

static void test_tensor_tape_matmul_and_backward(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 32);
    ASSERT_NOT_NULL(tape);

    /* A: [2, 3], B: [3, 2] */
    RCRE_Tensor* A = rcre_tensor_alloc_2d(arena, 2, 3);
    RCRE_Tensor* B = rcre_tensor_alloc_2d(arena, 3, 2);
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);

    /* Populate A */
    rcre_tensor_f32_data(A)[0] = 1.0f; rcre_tensor_f32_data(A)[1] = 2.0f; rcre_tensor_f32_data(A)[2] = 3.0f;
    rcre_tensor_f32_data(A)[3] = 4.0f; rcre_tensor_f32_data(A)[4] = 5.0f; rcre_tensor_f32_data(A)[5] = 6.0f;

    /* Populate B */
    rcre_tensor_f32_data(B)[0] = 7.0f;  rcre_tensor_f32_data(B)[1] = 8.0f;
    rcre_tensor_f32_data(B)[2] = 9.0f;  rcre_tensor_f32_data(B)[3] = 1.0f;
    rcre_tensor_f32_data(B)[4] = 2.0f;  rcre_tensor_f32_data(B)[5] = 3.0f;

    /* Forward: C = A * B */
    RCRE_Tensor* C = rcre_autograd_matmul(tape, A, B);
    ASSERT_NOT_NULL(C);
    ASSERT_UINT_EQ(2, C->shape[0]);
    ASSERT_UINT_EQ(2, C->shape[1]);

    /* Expected:
       C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
       C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
       C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
       C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    */
    ASSERT_FLOAT_NEAR(31.0f, rcre_tensor_f32_data(C)[0], EPSILON);
    ASSERT_FLOAT_NEAR(19.0f, rcre_tensor_f32_data(C)[1], EPSILON);
    ASSERT_FLOAT_NEAR(85.0f, rcre_tensor_f32_data(C)[2], EPSILON);
    ASSERT_FLOAT_NEAR(55.0f, rcre_tensor_f32_data(C)[3], EPSILON);

    /* Seed output gradients dC = 1 */
    C->grad[0] = 1.0f; C->grad[1] = 1.0f;
    C->grad[2] = 1.0f; C->grad[3] = 1.0f;

    /* Backprop */
    rcre_tensor_tape_backward(tape);

    /* dA = dC * B^T
       B^T:
       [7, 9, 2]
       [8, 1, 3]
       dA[0, :] = [1, 1] * B^T = [15, 10, 5]
       dA[1, :] = [1, 1] * B^T = [15, 10, 5]
    */
    ASSERT_FLOAT_NEAR(15.0f, A->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(10.0f, A->grad[1], EPSILON);
    ASSERT_FLOAT_NEAR(5.0f,  A->grad[2], EPSILON);
    ASSERT_FLOAT_NEAR(15.0f, A->grad[3], EPSILON);
    ASSERT_FLOAT_NEAR(10.0f, A->grad[4], EPSILON);
    ASSERT_FLOAT_NEAR(5.0f,  A->grad[5], EPSILON);

    /* dB = A^T * dC
       A^T:
       [1, 4]
       [2, 5]
       [3, 6]
       dB[0, :] = [1, 4] * [1, 1; 1, 1] = [5, 5]
       dB[1, :] = [2, 5] * [1, 1; 1, 1] = [7, 7]
       dB[2, :] = [3, 6] * [1, 1; 1, 1] = [9, 9]
    */
    ASSERT_FLOAT_NEAR(5.0f, B->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(5.0f, B->grad[1], EPSILON);
    ASSERT_FLOAT_NEAR(7.0f, B->grad[2], EPSILON);
    ASSERT_FLOAT_NEAR(7.0f, B->grad[3], EPSILON);
    ASSERT_FLOAT_NEAR(9.0f, B->grad[4], EPSILON);
    ASSERT_FLOAT_NEAR(9.0f, B->grad[5], EPSILON);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(arena);
}

static void test_tensor_tape_add_bias_and_activations(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 32);
    ASSERT_NOT_NULL(tape);

    /* X: [2, 2], bias: [2] */
    RCRE_Tensor* X = rcre_tensor_alloc_2d(arena, 2, 2);
    RCRE_Tensor* bias = rcre_tensor_alloc_1d(arena, 2);

    rcre_tensor_f32_data(X)[0] = -2.0f; rcre_tensor_f32_data(X)[1] = 1.0f;
    rcre_tensor_f32_data(X)[2] = 3.0f;  rcre_tensor_f32_data(X)[3] = -4.0f;

    rcre_tensor_f32_data(bias)[0] = 0.5f;
    rcre_tensor_f32_data(bias)[1] = -0.5f;

    /* Add bias */
    RCRE_Tensor* Z = rcre_autograd_add_bias(tape, X, bias);
    /* Z: [-1.5, 0.5; 3.5, -4.5] */
    ASSERT_FLOAT_NEAR(-1.5f, rcre_tensor_f32_data(Z)[0], EPSILON);
    ASSERT_FLOAT_NEAR(0.5f,  rcre_tensor_f32_data(Z)[1], EPSILON);
    ASSERT_FLOAT_NEAR(3.5f,  rcre_tensor_f32_data(Z)[2], EPSILON);
    ASSERT_FLOAT_NEAR(-4.5f, rcre_tensor_f32_data(Z)[3], EPSILON);

    /* ReLU */
    RCRE_Tensor* A = rcre_autograd_relu(tape, Z);
    /* A: [0.0, 0.5; 3.5, 0.0] */
    ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_f32_data(A)[0], EPSILON);
    ASSERT_FLOAT_NEAR(0.5f, rcre_tensor_f32_data(A)[1], EPSILON);
    ASSERT_FLOAT_NEAR(3.5f, rcre_tensor_f32_data(A)[2], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_f32_data(A)[3], EPSILON);

    /* Seed dA = 1 */
    A->grad[0] = 1.0f; A->grad[1] = 1.0f;
    A->grad[2] = 1.0f; A->grad[3] = 1.0f;

    rcre_tensor_tape_backward(tape);

    /* Z grad through ReLU:
       Z_grad: [0.0, 1.0; 1.0, 0.0]
    */
    ASSERT_FLOAT_NEAR(0.0f, X->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(1.0f, X->grad[1], EPSILON);
    ASSERT_FLOAT_NEAR(1.0f, X->grad[2], EPSILON);
    ASSERT_FLOAT_NEAR(0.0f, X->grad[3], EPSILON);

    /* bias grad: row sum of Z_grad = [1.0, 1.0] */
    ASSERT_FLOAT_NEAR(1.0f, bias->grad[0], EPSILON);
    ASSERT_FLOAT_NEAR(1.0f, bias->grad[1], EPSILON);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(arena);
}

static void test_tape_grow_then_rewind(void) {
    /* P0 regression: the tape node array must survive rcre_arena_temp_end()
     * rewinds even when it grows mid-batch. Node storage is heap-owned; only
     * activations live in the rewound arena. */
    RCRE_Arena* tape_arena = rcre_arena_create(1 << 20);
    RCRE_Arena* perma = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(tape_arena);
    ASSERT_NOT_NULL(perma);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(tape_arena, 4); /* tiny cap -> forces growth */
    ASSERT_NOT_NULL(tape);

    RCRE_Tensor* X = rcre_tensor_alloc_2d(perma, 4, 8);
    RCRE_Tensor* W = rcre_tensor_alloc_2d(perma, 8, 8);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(W);
    for (int i = 0; i < 32; ++i) rcre_tensor_f32_data(X)[i] = 0.25f;
    for (int i = 0; i < 64; ++i) rcre_tensor_f32_data(W)[i] = 0.1f;

    float run_grads[2];
    for (int batch = 0; batch < 2; ++batch) {
        RCRE_ArenaTemp temp = rcre_arena_temp_begin(tape_arena);
        rcre_tensor_tape_reset(tape);
        rcre_tensor_zero_grad(X); /* backward accumulates: start clean */

        RCRE_Tensor* cur = X;
        for (int i = 0; i < 8; ++i) {
            RCRE_Tensor* nxt = rcre_autograd_matmul(tape, cur, W);
            ASSERT_NOT_NULL(nxt);
            rcre_tensor_zero_grad(nxt);
            nxt->grad[0] = 1.0f;
            cur = nxt;
        }
        rcre_tensor_tape_backward(tape);
        run_grads[batch] = X->grad[0];

        rcre_arena_temp_end(temp);
    }

    /* Both batches are identical workloads: gradients must match exactly. */
    ASSERT_FLOAT_NEAR(run_grads[0], run_grads[1], 0.0f);
    ASSERT_TRUE(run_grads[0] != 0.0f);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(tape_arena);
    rcre_arena_destroy(perma);
}

static void test_autograd_shape_validation(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 8);
    ASSERT_NOT_NULL(tape);

    /* add with mismatched numel must be rejected */
    RCRE_Tensor* A = rcre_tensor_alloc_2d(arena, 2, 2); /* numel 4 */
    RCRE_Tensor* B = rcre_tensor_alloc_2d(arena, 3, 1); /* numel 3 */
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NULL(rcre_autograd_add(tape, A, B));

    /* add_bias with wrong bias width must be rejected */
    RCRE_Tensor* bias2 = rcre_tensor_alloc_1d(arena, 2);
    RCRE_Tensor* bias3 = rcre_tensor_alloc_1d(arena, 3);
    ASSERT_NOT_NULL(bias2);
    ASSERT_NOT_NULL(bias3);
    ASSERT_NOT_NULL(rcre_autograd_add_bias(tape, A, bias2)); /* valid: 2 cols */
    ASSERT_NULL(rcre_autograd_add_bias(tape, A, bias3));     /* invalid: 3 cols */

    /* Autograd ops must refuse non-F32 tensors (the dtype gate that used to
     * live in rcre_ag_from_tensor now lives in every op, via is_f32()). */
    uint8_t px[4] = { 200, 100, 50, 250 };
    RCRE_Tensor img = {
        .data = px, .ndim = 1, .dtype = RCRE_DTYPE_U8, .layout = RCRE_LAYOUT_TABULAR,
        .shape = { 4, 1, 1, 1 }, .strides = { 1, 0, 0, 0 }, .numel = 4
    };
    ASSERT_NULL(rcre_autograd_relu(tape, &img));
    ASSERT_NULL(rcre_autograd_add(tape, A, &img)); /* mixed numel too, but dtype fails first */

    /* A plain core F32 tensor (value type, from_buffer) feeds autograd ops
     * directly — the payoff of the v0.2.2 type unification. */
    float fx[4] = { -1.0f, 2.0f, -3.0f, 4.0f };
    size_t cshape[2] = { 2, 2 };
    RCRE_Tensor core = rcre_tensor_from_buffer(fx, cshape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR);
    RCRE_Tensor* ok = rcre_autograd_relu(tape, &core);
    ASSERT_NOT_NULL(ok);
    ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_f32_data(ok)[0], 1e-6f);
    ASSERT_FLOAT_NEAR(2.0f, rcre_tensor_f32_data(ok)[1], 1e-6f);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* Phase 0 Generative & Vision Autograd Gradient Verification                */
/* ========================================================================= */

typedef RCRE_Tensor* (*ag_test_fn)(RCRE_TensorTape* tape, RCRE_Tensor** ins, void* ctx);

static float ag_weighted_sum(const RCRE_Tensor* out, const float* weights) {
    const float* od = rcre_tensor_f32_data(out);
    double s = 0.0;
    for (size_t i = 0; i < out->numel; ++i) s += (double)od[i] * (double)weights[i];
    return (float)s;
}

static void ag_verify_grad_fd(
    ag_test_fn fn, void* ctx,
    RCRE_Tensor** ins, size_t n_ins,
    size_t out_numel, float h, float tol)
{
    RCRE_Arena* tape_arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(tape_arena);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(tape_arena, 32);
    ASSERT_NOT_NULL(tape);

    /* Deterministic arbitrary upstream weights w so every output element contributes */
    float* w = (float*)rcre_arena_alloc(tape_arena, out_numel * sizeof(float), 64);
    ASSERT_NOT_NULL(w);
    for (size_t i = 0; i < out_numel; ++i) {
        w[i] = 0.37f * sinf((float)(i + 1) * 1.13f) + 0.19f * cosf((float)(i + 2) * 0.71f);
    }

    for (size_t k = 0; k < n_ins; ++k) {
        if (ins[k]) rcre_tensor_zero_grad(ins[k]);
    }

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(tape_arena);
    rcre_tensor_tape_reset(tape);
    RCRE_Tensor* out = fn(tape, ins, ctx);
    ASSERT_NOT_NULL(out);
    ASSERT_UINT_EQ(out_numel, out->numel);
    for (size_t i = 0; i < out_numel; ++i) out->grad[i] = w[i];
    rcre_tensor_tape_backward(tape);
    rcre_arena_temp_end(tmp);

    for (size_t k = 0; k < n_ins; ++k) {
        if (!ins[k] || !ins[k]->grad) continue;
        float* x = rcre_tensor_f32_data(ins[k]);
        for (size_t idx = 0; idx < ins[k]->numel; ++idx) {
            float orig = x[idx];

            x[idx] = orig + h;
            tmp = rcre_arena_temp_begin(tape_arena);
            rcre_tensor_tape_reset(tape);
            RCRE_Tensor* o_plus = fn(tape, ins, ctx);
            ASSERT_NOT_NULL(o_plus);
            float f_plus = ag_weighted_sum(o_plus, w);
            rcre_arena_temp_end(tmp);

            x[idx] = orig - h;
            tmp = rcre_arena_temp_begin(tape_arena);
            rcre_tensor_tape_reset(tape);
            RCRE_Tensor* o_minus = fn(tape, ins, ctx);
            ASSERT_NOT_NULL(o_minus);
            float f_minus = ag_weighted_sum(o_minus, w);
            rcre_arena_temp_end(tmp);

            x[idx] = orig;
            float fd = (f_plus - f_minus) / (2.0f * h);
            ASSERT_FLOAT_NEAR(fd, ins[k]->grad[idx], tol);
        }
    }

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(tape_arena);
}

static void fill_det(RCRE_Tensor* t, float offset, float scale) {
    float* d = rcre_tensor_f32_data(t);
    for (size_t i = 0; i < t->numel; ++i) {
        d[i] = offset + scale * sinf((float)(i + 1) * 0.91f + offset);
    }
}

static RCRE_Tensor* fn_sub(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_sub(t, in[0], in[1]); }
static RCRE_Tensor* fn_mul(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_mul(t, in[0], in[1]); }
static RCRE_Tensor* fn_scale(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_scale(t, in[0], -1.75f); }
static RCRE_Tensor* fn_exp(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_exp(t, in[0]); }
static RCRE_Tensor* fn_lrelu(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_leaky_relu(t, in[0], 0.2f); }
static RCRE_Tensor* fn_tanh(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_tanh(t, in[0]); }
static RCRE_Tensor* fn_silu(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_silu(t, in[0]); }
static RCRE_Tensor* fn_gelu(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_gelu(t, in[0]); }
static RCRE_Tensor* fn_linear(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_linear(t, in[0], in[1], in[2]); }
static RCRE_Tensor* fn_bias4d(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_add_bias_4d(t, in[0], in[1]); }
static RCRE_Tensor* fn_ln(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_layernorm(t, in[0], in[1], in[2], 1e-5f); }
static RCRE_Tensor* fn_gn(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_groupnorm(t, in[0], in[1], in[2], 2, 1e-5f); }
static RCRE_Tensor* fn_film4d(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_film_4d(t, in[0], in[1], in[2]); }
static RCRE_Tensor* fn_adaln3d(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_adaln_3d(t, in[0], in[1], in[2]); }
static RCRE_Tensor* fn_gate3d(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_gate_3d(t, in[0], in[1]); }
static RCRE_Tensor* fn_reshape(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) {
    (void)c;
    size_t sh[2] = { 2, 6 };
    return rcre_autograd_reshape(t, in[0], sh, 2);
}
static RCRE_Tensor* fn_slice_cols(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_slice_cols(t, in[0], 2, 3); }
static RCRE_Tensor* fn_concat_ch(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_concat_channels(t, in[0], in[1]); }
static RCRE_Tensor* fn_upsample(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_upsample2d_nearest(t, in[0], 2); }
static RCRE_Tensor* fn_avgpool(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_avgpool2d(t, in[0], 2, 2); }
static RCRE_Tensor* fn_patchify(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_patchify(t, in[0], 2); }
static RCRE_Tensor* fn_unpatchify(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_unpatchify(t, in[0], 2, 4, 4, 2); }
static RCRE_Tensor* fn_mhsa(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_mhsa(t, in[0], in[1], in[2], 2); }
static RCRE_Tensor* fn_reparam(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_reparameterize(t, in[0], in[1], in[2]); }
static RCRE_Tensor* fn_pos_emb(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_add_pos_emb_3d(t, in[0], in[1]); }
static RCRE_Tensor* fn_mean_pool(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) { (void)c; return rcre_autograd_mean_pool_seq(t, in[0]); }
static RCRE_Tensor* fn_embed_lu(RCRE_TensorTape* t, RCRE_Tensor** in, void* c) {
    const size_t* idx = (const size_t*)c;
    return rcre_autograd_embed_lookup(t, in[0], idx, 3);
}

static void test_phase0_elementwise_and_activations_fd(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);

    RCRE_Tensor* A = rcre_tensor_alloc_2d(arena, 2, 4);
    RCRE_Tensor* B = rcre_tensor_alloc_2d(arena, 2, 4);
    fill_det(A, 0.25f, 0.8f);
    fill_det(B, -0.35f, 0.7f);
    /* Avoid exact 0 for leaky_relu FD check */
    for (size_t i = 0; i < A->numel; ++i) {
        if (fabsf(rcre_tensor_f32_data(A)[i]) < 0.1f) rcre_tensor_f32_data(A)[i] = 0.3f;
    }

    RCRE_Tensor* pair[2] = { A, B };
    RCRE_Tensor* single[1] = { A };
    ag_verify_grad_fd(fn_sub,   nullptr, pair,   2, 8, 1e-3f, 2e-3f);
    ag_verify_grad_fd(fn_mul,   nullptr, pair,   2, 8, 1e-3f, 2e-3f);
    ag_verify_grad_fd(fn_scale, nullptr, single, 1, 8, 1e-3f, 2e-3f);
    ag_verify_grad_fd(fn_exp,   nullptr, single, 1, 8, 1e-3f, 3e-3f);
    ag_verify_grad_fd(fn_lrelu, nullptr, single, 1, 8, 1e-3f, 2e-3f);
    ag_verify_grad_fd(fn_tanh,  nullptr, single, 1, 8, 1e-3f, 2e-3f);
    ag_verify_grad_fd(fn_silu,  nullptr, single, 1, 8, 1e-3f, 2e-3f);
    ag_verify_grad_fd(fn_gelu,  nullptr, single, 1, 8, 1e-3f, 2e-3f);

    rcre_arena_destroy(arena);
}

static void test_phase0_linear_norm_and_conditioning_fd(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 3D Linear: [2, 3, 4] x [4, 5] + [5] -> [2, 3, 5] */
    RCRE_Tensor* X3 = rcre_tensor_alloc_3d(arena, 2, 3, 4);
    RCRE_Tensor* W  = rcre_tensor_alloc_2d(arena, 4, 5);
    RCRE_Tensor* b  = rcre_tensor_alloc_1d(arena, 5);
    fill_det(X3, 0.1f, 0.6f);
    fill_det(W, -0.2f, 0.5f);
    fill_det(b, 0.05f, 0.3f);
    RCRE_Tensor* lin_ins[3] = { X3, W, b };
    ag_verify_grad_fd(fn_linear, nullptr, lin_ins, 3, 30, 1e-3f, 3e-3f);

    /* 4D channel bias: [2, 3, 2, 2] + [3] */
    RCRE_Tensor* X4 = rcre_tensor_alloc_4d(arena, 2, 3, 2, 2);
    RCRE_Tensor* b4 = rcre_tensor_alloc_1d(arena, 3);
    fill_det(X4, -0.1f, 0.7f);
    fill_det(b4, 0.2f, 0.4f);
    RCRE_Tensor* b4_ins[2] = { X4, b4 };
    ag_verify_grad_fd(fn_bias4d, nullptr, b4_ins, 2, 24, 1e-3f, 2e-3f);

    /* LayerNorm 3D: [2, 3, 4] with gamma/beta */
    RCRE_Tensor* ln_g = rcre_tensor_alloc_1d(arena, 4);
    RCRE_Tensor* ln_b = rcre_tensor_alloc_1d(arena, 4);
    fill_det(ln_g, 1.0f, 0.3f);
    fill_det(ln_b, 0.0f, 0.3f);
    RCRE_Tensor* ln_ins[3] = { X3, ln_g, ln_b };
    ag_verify_grad_fd(fn_ln, nullptr, ln_ins, 3, 24, 1e-3f, 3e-3f);

    /* GroupNorm 4D: [2, 4, 2, 2], G=2 */
    RCRE_Tensor* GNX = rcre_tensor_alloc_4d(arena, 2, 4, 2, 2);
    RCRE_Tensor* gn_g = rcre_tensor_alloc_1d(arena, 4);
    RCRE_Tensor* gn_b = rcre_tensor_alloc_1d(arena, 4);
    fill_det(GNX, 0.15f, 0.8f);
    fill_det(gn_g, 1.0f, 0.25f);
    fill_det(gn_b, -0.1f, 0.25f);
    RCRE_Tensor* gn_ins[3] = { GNX, gn_g, gn_b };
    ag_verify_grad_fd(fn_gn, nullptr, gn_ins, 3, 32, 1e-3f, 3e-3f);

    /* FiLM 4D: [2, 4, 2, 2] with [2, 4] scale & shift */
    RCRE_Tensor* f_sc = rcre_tensor_alloc_2d(arena, 2, 4);
    RCRE_Tensor* f_sh = rcre_tensor_alloc_2d(arena, 2, 4);
    fill_det(f_sc, 0.1f, 0.4f);
    fill_det(f_sh, -0.2f, 0.4f);
    RCRE_Tensor* film_ins[3] = { GNX, f_sc, f_sh };
    ag_verify_grad_fd(fn_film4d, nullptr, film_ins, 3, 32, 1e-3f, 3e-3f);

    /* AdaLN 3D & Gate 3D: [2, 3, 4] with [2, 4] */
    RCRE_Tensor* adaln_ins[3] = { X3, f_sc, f_sh };
    ag_verify_grad_fd(fn_adaln3d, nullptr, adaln_ins, 3, 24, 1e-3f, 3e-3f);
    RCRE_Tensor* gate_ins[2] = { X3, f_sc };
    ag_verify_grad_fd(fn_gate3d, nullptr, gate_ins, 2, 24, 1e-3f, 3e-3f);

    rcre_arena_destroy(arena);
}

static void test_phase0_spatial_vit_and_generative_fd(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Reshape: [2, 2, 1, 3] -> [2, 6] */
    RCRE_Tensor* R = rcre_tensor_alloc_4d(arena, 2, 2, 1, 3);
    fill_det(R, 0.2f, 0.5f);
    RCRE_Tensor* r_in[1] = { R };
    ag_verify_grad_fd(fn_reshape, nullptr, r_in, 1, 12, 1e-3f, 2e-3f);

    /* Slice cols: [3, 6] -> [3, 3] */
    RCRE_Tensor* S = rcre_tensor_alloc_2d(arena, 3, 6);
    fill_det(S, -0.1f, 0.6f);
    RCRE_Tensor* s_in[1] = { S };
    ag_verify_grad_fd(fn_slice_cols, nullptr, s_in, 1, 9, 1e-3f, 2e-3f);

    /* Concat channels: [2, 2, 2, 2] + [2, 3, 2, 2] -> [2, 5, 2, 2] */
    RCRE_Tensor* CA = rcre_tensor_alloc_4d(arena, 2, 2, 2, 2);
    RCRE_Tensor* CB = rcre_tensor_alloc_4d(arena, 2, 3, 2, 2);
    fill_det(CA, 0.1f, 0.5f);
    fill_det(CB, -0.3f, 0.5f);
    RCRE_Tensor* cc_in[2] = { CA, CB };
    ag_verify_grad_fd(fn_concat_ch, nullptr, cc_in, 2, 40, 1e-3f, 2e-3f);

    /* Upsample2D nearest (scale=2) & AvgPool2D (k=2, s=2) */
    RCRE_Tensor* U = rcre_tensor_alloc_4d(arena, 1, 2, 2, 2);
    fill_det(U, 0.3f, 0.6f);
    RCRE_Tensor* u_in[1] = { U };
    ag_verify_grad_fd(fn_upsample, nullptr, u_in, 1, 32, 1e-3f, 2e-3f);

    RCRE_Tensor* P = rcre_tensor_alloc_4d(arena, 1, 2, 4, 4);
    fill_det(P, -0.2f, 0.7f);
    RCRE_Tensor* p_in[1] = { P };
    ag_verify_grad_fd(fn_avgpool, nullptr, p_in, 1, 8, 1e-3f, 2e-3f);

    /* Patchify / Unpatchify roundtrip + FD */
    ag_verify_grad_fd(fn_patchify, nullptr, p_in, 1, 32, 1e-3f, 2e-3f);
    RCRE_TensorTape* rt_tape = rcre_tensor_tape_create(arena, 8);
    RCRE_Tensor* toks = rcre_autograd_patchify(rt_tape, P, 2);
    ASSERT_NOT_NULL(toks);
    ASSERT_UINT_EQ(1, toks->shape[0]);
    ASSERT_UINT_EQ(4, toks->shape[1]);
    ASSERT_UINT_EQ(8, toks->shape[2]);
    RCRE_Tensor* rec = rcre_autograd_unpatchify(rt_tape, toks, 2, 4, 4, 2);
    ASSERT_NOT_NULL(rec);
    for (size_t i = 0; i < P->numel; ++i) {
        ASSERT_FLOAT_NEAR(rcre_tensor_f32_data(P)[i], rcre_tensor_f32_data(rec)[i], 1e-6f);
    }
    RCRE_Tensor* tok_in[1] = { toks };
    ag_verify_grad_fd(fn_unpatchify, nullptr, tok_in, 1, 32, 1e-3f, 2e-3f);
    rcre_tensor_tape_destroy(rt_tape);

    /* Batched MHSA: [2, 4, 6], num_heads=2 */
    RCRE_Tensor* Q = rcre_tensor_alloc_3d(arena, 2, 4, 6);
    RCRE_Tensor* K = rcre_tensor_alloc_3d(arena, 2, 4, 6);
    RCRE_Tensor* V = rcre_tensor_alloc_3d(arena, 2, 4, 6);
    fill_det(Q, 0.1f, 0.5f);
    fill_det(K, -0.2f, 0.5f);
    fill_det(V, 0.3f, 0.5f);
    RCRE_Tensor* qkv[3] = { Q, K, V };
    ag_verify_grad_fd(fn_mhsa, nullptr, qkv, 3, 48, 1e-3f, 3e-3f);

    /* ViT/DiT helpers: add_pos_emb_3d, mean_pool_seq, embed_lookup */
    RCRE_Tensor* pos = rcre_tensor_alloc_2d(arena, 4, 6);
    fill_det(pos, -0.15f, 0.4f);
    RCRE_Tensor* pe_in[2] = { Q, pos };
    ag_verify_grad_fd(fn_pos_emb, nullptr, pe_in, 2, 48, 1e-3f, 2e-3f);
    RCRE_Tensor* mp_in[1] = { Q };
    ag_verify_grad_fd(fn_mean_pool, nullptr, mp_in, 1, 12, 1e-3f, 2e-3f);
    RCRE_Tensor* emb_tbl = rcre_tensor_alloc_2d(arena, 5, 4);
    fill_det(emb_tbl, 0.05f, 0.6f);
    size_t lu_idx[3] = { 1, 3, 1 }; /* repeated index tests += accumulation */
    RCRE_Tensor* el_in[1] = { emb_tbl };
    ag_verify_grad_fd(fn_embed_lu, lu_idx, el_in, 1, 12, 1e-3f, 2e-3f);

    /* VAE Reparameterize: z = mu + eps * exp(0.5 * logvar) */
    RCRE_Tensor* mu = rcre_tensor_alloc_2d(arena, 2, 4);
    RCRE_Tensor* lv = rcre_tensor_alloc_2d(arena, 2, 4);
    RCRE_Tensor* ep = rcre_tensor_alloc_2d(arena, 2, 4);
    fill_det(mu, 0.2f, 0.5f);
    fill_det(lv, -0.5f, 0.8f);
    fill_det(ep, 0.0f, 1.0f);
    RCRE_Tensor* vae_in[3] = { mu, lv, ep };
    ag_verify_grad_fd(fn_reparam, nullptr, vae_in, 3, 8, 1e-3f, 3e-3f);

    /* KL divergence accumulating (+=) with beta + pre-existing grad */
    rcre_tensor_zero_grad(mu);
    rcre_tensor_zero_grad(lv);
    mu->grad[0] = 0.5f;
    lv->grad[0] = -0.25f;
    float kl = rcre_loss_kl_divergence_add(mu, lv, 0.4f, mu, lv);
    ASSERT_TRUE(kl >= 0.0f);
    /* Verify += preserved existing 0.5f / -0.25f and added beta * dKL */
    float expected_gm0 = 0.5f + 0.4f * rcre_tensor_f32_data(mu)[0] / 8.0f;
    float expected_glv0 = -0.25f + 0.4f * 0.5f * (expf(rcre_tensor_f32_data(lv)[0]) - 1.0f) / 8.0f;
    ASSERT_FLOAT_NEAR(expected_gm0, mu->grad[0], 1e-5f);
    ASSERT_FLOAT_NEAR(expected_glv0, lv->grad[0], 1e-5f);

    /* Sinusoidal timestep embedding + Gaussian fill + grad norm clip */
    float ts[3] = { 0.0f, 10.0f, 500.0f };
    RCRE_Tensor* temb = rcre_sinusoidal_timestep_embed(arena, ts, 3, 8);
    ASSERT_NOT_NULL(temb);
    ASSERT_UINT_EQ(3, temb->shape[0]);
    ASSERT_UINT_EQ(8, temb->shape[1]);
    /* At t=0: sin(0)=0 for k<4, cos(0)=1 for k>=4 */
    for (size_t k = 0; k < 4; ++k) {
        ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_f32_data(temb)[k], 1e-6f);
        ASSERT_FLOAT_NEAR(1.0f, rcre_tensor_f32_data(temb)[4 + k], 1e-6f);
    }

    uint64_t rng = 42ULL;
    RCRE_Tensor* G = rcre_tensor_alloc_2d(arena, 2, 4);
    rcre_tensor_fill_normal(G, 0.0f, 1.0f, &rng);
    for (size_t i = 0; i < G->numel; ++i) G->grad[i] = 3.0f; /* norm = sqrt(8 * 9) = sqrt(72) */
    RCRE_Tensor* plist[1] = { G };
    float pre_norm = rcre_tensor_clip_grad_norm(plist, 1, 2.0f);
    ASSERT_FLOAT_NEAR(sqrtf(72.0f), pre_norm, 1e-4f);
    float post_sq = 0.0f;
    for (size_t i = 0; i < G->numel; ++i) post_sq += G->grad[i] * G->grad[i];
    ASSERT_FLOAT_NEAR(2.0f, sqrtf(post_sq), 1e-3f);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Dynamic Tensor Tape Autograd (AUTOGRAD-TENSOR-01)");
    TEST_RUN(test_tensor_tape_matmul_and_backward);
    TEST_RUN(test_tensor_tape_add_bias_and_activations);
    TEST_RUN(test_tape_grow_then_rewind);
    TEST_RUN(test_autograd_shape_validation);
    TEST_RUN(test_phase0_elementwise_and_activations_fd);
    TEST_RUN(test_phase0_linear_norm_and_conditioning_fd);
    TEST_RUN(test_phase0_spatial_vit_and_generative_fd);
    return test_suite_report();
}
