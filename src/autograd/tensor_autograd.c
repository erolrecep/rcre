/**
 * @file tensor_autograd.c
 * @brief Dynamic Tensor Tape and Automatic Differentiation Engine for RCRE.
 * 
 * Part of RCRE Unified Engine.
 */

#include "rcre_autograd.h"
#include "rcre_rng.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(__APPLE__) && defined(RCRE_USE_BLAS)
#include <Accelerate/Accelerate.h>
#endif

/* ========================================================================= */
/* Internal General GEMM with Transpose Support                              */
/* ========================================================================= */

static void rcre_ag_gemm(bool transA, bool transB, size_t M, size_t N, size_t K,
                         float alpha, const float* A, size_t lda,
                         const float* B, size_t ldb,
                         float beta, float* C, size_t ldc) {
#if defined(__APPLE__) && defined(RCRE_USE_BLAS)
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                (int)M, (int)N, (int)K,
                alpha, A, (int)lda,
                B, (int)ldb,
                beta, C, (int)ldc);
#else
    if (!transA && !transB) {
        rcre_hal.gemm_f32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    #define BLOCK_M 64
    #define BLOCK_N 64
    #define BLOCK_K 64

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (beta == 0.0f) {
                C[i * ldc + j] = 0.0f;
            } else {
                C[i * ldc + j] *= beta;
            }
        }
    }

    for (size_t ii = 0; ii < M; ii += BLOCK_M) {
        size_t m_end = (ii + BLOCK_M < M) ? ii + BLOCK_M : M;
        for (size_t kk = 0; kk < K; kk += BLOCK_K) {
            size_t k_end = (kk + BLOCK_K < K) ? kk + BLOCK_K : K;
            for (size_t jj = 0; jj < N; jj += BLOCK_N) {
                size_t j_end = (jj + BLOCK_N < N) ? jj + BLOCK_N : N;

                for (size_t i = ii; i < m_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        float a_val = alpha * (transA ? A[k * lda + i] : A[i * lda + k]);
                        for (size_t j = jj; j < j_end; ++j) {
                            float b_val = transB ? B[j * ldb + k] : B[k * ldb + j];
                            C[i * ldc + j] += a_val * b_val;
                        }
                    }
                }
            }
        }
    }
#endif
}

/* ========================================================================= */
/* Autograd Tensor Allocation & Initialization                               */
/* ========================================================================= */

static void rcre_ag_compute_contiguous_strides(RCRE_AGTensor* t) {
    if (t->ndim == 0) return;
    t->strides[t->ndim - 1] = 1;
    for (size_t i = t->ndim - 1; i > 0; --i) {
        t->strides[i - 1] = t->strides[i] * t->shape[i];
    }
}

RCRE_AGTensor* rcre_ag_tensor_create(RCRE_Arena* arena, const size_t* shape, size_t ndim) {
    if (arena == nullptr || ndim > RCRE_AG_MAX_DIMS || ndim == 0) return nullptr;

    RCRE_AGTensor* t = (RCRE_AGTensor*)rcre_arena_alloc(arena, sizeof(RCRE_AGTensor), alignof(RCRE_AGTensor));
    if (!t) return nullptr;

    /* Zero shape/strides first: fields beyond `ndim` must never read back
     * uninitialized memory (e.g. `y->shape[1]` on a 1-D tensor). */
    memset(t, 0, sizeof(RCRE_AGTensor));

    t->ndim = ndim;
    t->numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }
    rcre_ag_compute_contiguous_strides(t);
    t->is_view = false;
    t->label = nullptr;

    size_t byte_size = t->numel * sizeof(float);
    t->data = (float*)rcre_arena_alloc(arena, byte_size, 64);
    t->grad = (float*)rcre_arena_alloc(arena, byte_size, 64);
    if (!t->data || !t->grad) {
        return nullptr; /* arena space already consumed; rewind via temp checkpoint */
    }

    memset(t->data, 0, byte_size);
    memset(t->grad, 0, byte_size);
    return t;
}

RCRE_AGTensor* rcre_ag_tensor_create_1d(RCRE_Arena* arena, size_t d0) {
    size_t shape[1] = { d0 };
    return rcre_ag_tensor_create(arena, shape, 1);
}

RCRE_AGTensor* rcre_ag_tensor_create_2d(RCRE_Arena* arena, size_t rows, size_t cols) {
    size_t shape[2] = { rows, cols };
    return rcre_ag_tensor_create(arena, shape, 2);
}

RCRE_AGTensor* rcre_ag_tensor_create_4d(RCRE_Arena* arena, size_t n, size_t c, size_t h, size_t w) {
    size_t shape[4] = { n, c, h, w };
    return rcre_ag_tensor_create(arena, shape, 4);
}

RCRE_Tensor rcre_ag_to_tensor(const RCRE_AGTensor* ag) {
    RCRE_Tensor t = {
        .data = ag->data,
        .ndim = (uint8_t)ag->ndim,
        .dtype = RCRE_DTYPE_F32,
        .layout = RCRE_LAYOUT_TABULAR
    };
    for (size_t i = 0; i < 4; ++i) {
        t.shape[i] = (i < ag->ndim) ? ag->shape[i] : 1;
        t.strides[i] = (i < ag->ndim) ? (ptrdiff_t)ag->strides[i] : 0;
    }
    return t;
}

RCRE_AGTensor rcre_ag_from_tensor(const RCRE_Tensor* t, float* grad, const char* label) {
    RCRE_AGTensor ag;
    memset(&ag, 0, sizeof(ag));
    /* F32 only: reinterpreting a U8/I32 buffer as float would be silently wrong. */
    if (!t || !t->data || t->dtype != RCRE_DTYPE_F32 || t->ndim == 0 || t->ndim > RCRE_AG_MAX_DIMS) {
        return ag;
    }
    ag.data = (float*)t->data;
    ag.grad = grad;
    ag.ndim = t->ndim;
    ag.numel = 1;
    ag.is_view = true;
    ag.label = label;
    for (size_t i = 0; i < t->ndim; ++i) {
        ag.shape[i] = t->shape[i];
        ag.strides[i] = (size_t)t->strides[i];
        ag.numel *= t->shape[i];
    }
    return ag;
}

void rcre_ag_tensor_fill(RCRE_AGTensor* t, float val) {
    if (!t || !t->data) return;
    for (size_t i = 0; i < t->numel; ++i) {
        t->data[i] = val;
    }
}

void rcre_ag_tensor_zeros(RCRE_AGTensor* t) {
    rcre_ag_tensor_fill(t, 0.0f);
}

void rcre_ag_tensor_ones(RCRE_AGTensor* t) {
    rcre_ag_tensor_fill(t, 1.0f);
}

void rcre_ag_tensor_zero_grad(RCRE_AGTensor* t) {
    if (t && t->grad) {
        memset(t->grad, 0, t->numel * sizeof(float));
    }
}

void rcre_ag_tensor_init_xavier(RCRE_AGTensor* t, size_t fan_in, size_t fan_out, uint64_t* rng) {
    if (!t || !t->data || !rng || (fan_in + fan_out) == 0) return;
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < t->numel; ++i) {
        t->data[i] = rcre_rng_uniform_f32(rng, -limit, limit);
    }
}

void rcre_ag_tensor_init_he(RCRE_AGTensor* t, size_t fan_in, uint64_t* rng) {
    if (!t || !t->data || !rng || fan_in == 0) return;
    float std = sqrtf(2.0f / (float)fan_in);
    for (size_t i = 0; i < t->numel; ++i) {
        t->data[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
    }
}

RCRE_AGTensor* rcre_ag_tensor_view(RCRE_Arena* arena, const RCRE_AGTensor* src) {
    if (!src || arena == nullptr) return nullptr;
    RCRE_AGTensor* v = (RCRE_AGTensor*)rcre_arena_alloc(arena, sizeof(RCRE_AGTensor), alignof(RCRE_AGTensor));
    if (!v) return nullptr;
    memcpy(v, src, sizeof(RCRE_AGTensor));
    v->is_view = true;
    return v;
}

RCRE_AGTensor* rcre_ag_tensor_reshape(RCRE_Arena* arena, const RCRE_AGTensor* src, const size_t* new_shape, size_t new_ndim) {
    if (!src || new_ndim > RCRE_AG_MAX_DIMS || new_ndim == 0) return nullptr;
    size_t new_numel = 1;
    for (size_t i = 0; i < new_ndim; ++i) {
        new_numel *= new_shape[i];
    }
    if (new_numel != src->numel) return nullptr;

    RCRE_AGTensor* v = rcre_ag_tensor_view(arena, src);
    if (!v) return nullptr;
    v->ndim = new_ndim;
    for (size_t i = 0; i < new_ndim; ++i) {
        v->shape[i] = new_shape[i];
    }
    rcre_ag_compute_contiguous_strides(v);
    return v;
}

void rcre_ag_tensor_print(const RCRE_AGTensor* t, const char* name) {
    if (!t) return;
    printf("Tensor '%s' [ndim=%zu, numel=%zu]:\n", name ? name : "unnamed", t->ndim, t->numel);
    if (t->ndim == 1) {
        printf("  [");
        for (size_t i = 0; i < t->numel && i < 10; ++i) {
            printf("%.4f%s", t->data[i], (i + 1 < t->numel) ? ", " : "");
        }
        if (t->numel > 10) printf("...");
        printf("]\n");
    } else if (t->ndim == 2) {
        for (size_t r = 0; r < t->shape[0] && r < 5; ++r) {
            printf("  [");
            for (size_t c = 0; c < t->shape[1] && c < 8; ++c) {
                printf("%.4f%s", t->data[r * t->strides[0] + c], (c + 1 < t->shape[1]) ? ", " : "");
            }
            if (t->shape[1] > 8) printf("...");
            printf("]\n");
        }
        if (t->shape[0] > 5) printf("  ...\n");
    }
}

RCRE_AGTensor* rcre_ag_tensor_matmul(RCRE_Arena* arena, const RCRE_AGTensor* a, const RCRE_AGTensor* b) {
    if (!a || !b || a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) return nullptr;
    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];

    RCRE_AGTensor* c = rcre_ag_tensor_create_2d(arena, M, N);
    if (!c) return nullptr;
    rcre_ag_gemm(false, false, M, N, K, 1.0f, a->data, a->strides[0], b->data, b->strides[0], 0.0f, c->data, c->strides[0]);
    return c;
}

void rcre_ag_tensor_add_inplace(RCRE_AGTensor* a, const RCRE_AGTensor* b) {
    if (!a || !b || a->numel != b->numel) return;
    for (size_t i = 0; i < a->numel; ++i) {
        a->data[i] += b->data[i];
    }
}

/* ========================================================================= */
/* Dynamic Tensor Tape Autograd Engine                                       */
/* ========================================================================= */

RCRE_TensorTape* rcre_tensor_tape_create(RCRE_Arena* arena, size_t initial_cap) {
    if (arena == nullptr) return nullptr;
    if (initial_cap < 8) initial_cap = 8;

    RCRE_TensorTape* tape = (RCRE_TensorTape*)rcre_arena_alloc(arena, sizeof(RCRE_TensorTape), alignof(RCRE_TensorTape));
    /* The node array is deliberately heap-owned (malloc/realloc), NOT arena
     * owned: callers may rewind the tape arena with rcre_arena_temp_end()
     * between batches, which would invalidate (and later clobber) any node
     * storage placed inside it. The tape struct itself lives in the arena.
     * NOTE: create the tape BEFORE taking your first temp checkpoint. */
    tape->nodes = (RCRE_TensorNode*)malloc(initial_cap * sizeof(RCRE_TensorNode));
    if (!tape || !tape->nodes) {
        if (tape->nodes) free(tape->nodes);
        return nullptr;
    }
    tape->count = 0;
    tape->capacity = initial_cap;
    tape->arena = arena;
    return tape;
}

void rcre_tensor_tape_destroy(RCRE_TensorTape* tape) {
    if (tape != nullptr && tape->nodes != nullptr) {
        free(tape->nodes);
        tape->nodes = nullptr;
        tape->capacity = 0;
        tape->count = 0;
    }
}

void rcre_tensor_tape_reset(RCRE_TensorTape* tape) {
    if (tape != nullptr) {
        tape->count = 0;
    }
}

static RCRE_TensorNode* tape_push_node(RCRE_TensorTape* tape) {
    if (tape->count >= tape->capacity) {
        size_t new_cap = tape->capacity * 2;
        /* Realloc via temporary pointer: on failure the old array is intact. */
        RCRE_TensorNode* new_nodes = (RCRE_TensorNode*)realloc(tape->nodes, new_cap * sizeof(RCRE_TensorNode));
        if (!new_nodes) return nullptr; /* OOM: node array unchanged, caller must handle NULL */
        tape->nodes = new_nodes;
        tape->capacity = new_cap;
    }
    return &tape->nodes[tape->count++];
}

static void backward_matmul(RCRE_TensorNode* node) {
    RCRE_AGTensor* out = node->out;
    RCRE_AGTensor* a = node->inputs[0];
    RCRE_AGTensor* b = node->inputs[1];

    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];

    /* dA += dC * B^T */
    if (a->grad != nullptr) {
        rcre_ag_gemm(false, true, M, K, N, 1.0f, out->grad, N, b->data, N, 1.0f, a->grad, K);
    }

    /* dB += A^T * dC */
    if (b->grad != nullptr) {
        rcre_ag_gemm(true, false, K, N, M, 1.0f, a->data, K, out->grad, N, 1.0f, b->grad, N);
    }
}

static void backward_add(RCRE_TensorNode* node) {
    RCRE_AGTensor* out = node->out;
    RCRE_AGTensor* a = node->inputs[0];
    RCRE_AGTensor* b = node->inputs[1];

    for (size_t i = 0; i < out->numel; ++i) {
        if (a->grad != nullptr) a->grad[i] += out->grad[i];
        if (b->grad != nullptr) b->grad[i] += out->grad[i];
    }
}

static void backward_add_bias(RCRE_TensorNode* node) {
    RCRE_AGTensor* out = node->out;
    RCRE_AGTensor* x = node->inputs[0];
    RCRE_AGTensor* bias = node->inputs[1];

    size_t rows = out->shape[0];
    size_t cols = out->shape[1];

    for (size_t r = 0; r < rows; ++r) {
        const float* out_grad_row = out->grad + r * cols;
        float* x_grad_row = (x->grad != nullptr) ? (x->grad + r * cols) : nullptr;
        for (size_t c = 0; c < cols; ++c) {
            if (x_grad_row != nullptr) x_grad_row[c] += out_grad_row[c];
            if (bias->grad != nullptr) bias->grad[c] += out_grad_row[c];
        }
    }
}

static void backward_relu(RCRE_TensorNode* node) {
    RCRE_AGTensor* out = node->out;
    RCRE_AGTensor* x = node->inputs[0];

    if (x->grad == nullptr) return;

    for (size_t i = 0; i < out->numel; ++i) {
        if (x->data[i] > 0.0f) {
            x->grad[i] += out->grad[i];
        }
    }
}

static void backward_sigmoid(RCRE_TensorNode* node) {
    RCRE_AGTensor* out = node->out;
    RCRE_AGTensor* x = node->inputs[0];

    if (x->grad == nullptr) return;

    for (size_t i = 0; i < out->numel; ++i) {
        float s = out->data[i];
        x->grad[i] += s * (1.0f - s) * out->grad[i];
    }
}

void rcre_tensor_tape_backward(RCRE_TensorTape* tape) {
    if (!tape || tape->count == 0) return;
    for (size_t i = tape->count; i > 0; --i) {
        RCRE_TensorNode* node = &tape->nodes[i - 1];
        if (node->backward != nullptr) {
            node->backward(node);
        }
    }
}

RCRE_AGTensor* rcre_autograd_matmul(RCRE_TensorTape* tape, RCRE_AGTensor* a, RCRE_AGTensor* b) {
    RCRE_AGTensor* out = rcre_ag_tensor_matmul(tape->arena, a, b);
    if (!out) return nullptr;

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out; /* OOM: forward result valid, but graph is incomplete */
    node->out = out;
    node->inputs[0] = a;
    node->inputs[1] = b;
    node->num_inputs = 2;
    node->backward = backward_matmul;
    return out;
}

RCRE_AGTensor* rcre_autograd_add(RCRE_TensorTape* tape, RCRE_AGTensor* a, RCRE_AGTensor* b) {
    if (!a || !b || a->numel != b->numel) return nullptr;
    RCRE_AGTensor* out = rcre_ag_tensor_create(tape->arena, a->shape, a->ndim);
    if (!out) return nullptr;
    for (size_t i = 0; i < out->numel; ++i) {
        out->data[i] = a->data[i] + b->data[i];
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out;
    node->inputs[0] = a;
    node->inputs[1] = b;
    node->num_inputs = 2;
    node->backward = backward_add;
    return out;
}

RCRE_AGTensor* rcre_autograd_add_bias(RCRE_TensorTape* tape, RCRE_AGTensor* x, RCRE_AGTensor* bias) {
    if (!x || !bias || x->ndim != 2 || bias->ndim != 1 || bias->shape[0] != x->shape[1]) return nullptr;
    size_t rows = x->shape[0];
    size_t cols = x->shape[1];

    RCRE_AGTensor* out = rcre_ag_tensor_create_2d(tape->arena, rows, cols);
    if (!out) return nullptr;
    for (size_t r = 0; r < rows; ++r) {
        const float* x_row = x->data + r * cols;
        float* out_row = out->data + r * cols;
        for (size_t c = 0; c < cols; ++c) {
            out_row[c] = x_row[c] + bias->data[c];
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out;
    node->inputs[0] = x;
    node->inputs[1] = bias;
    node->num_inputs = 2;
    node->backward = backward_add_bias;
    return out;
}

RCRE_AGTensor* rcre_autograd_relu(RCRE_TensorTape* tape, RCRE_AGTensor* x) {
    if (!x) return nullptr;
    RCRE_AGTensor* out = rcre_ag_tensor_create(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;
    for (size_t i = 0; i < out->numel; ++i) {
        out->data[i] = (x->data[i] > 0.0f) ? x->data[i] : 0.0f;
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out;
    node->inputs[0] = x;
    node->num_inputs = 1;
    node->backward = backward_relu;
    return out;
}

RCRE_AGTensor* rcre_autograd_sigmoid(RCRE_TensorTape* tape, RCRE_AGTensor* x) {
    if (!x) return nullptr;
    RCRE_AGTensor* out = rcre_ag_tensor_create(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;
    for (size_t i = 0; i < out->numel; ++i) {
        out->data[i] = 1.0f / (1.0f + expf(-x->data[i]));
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out;
    node->inputs[0] = x;
    node->num_inputs = 1;
    node->backward = backward_sigmoid;
    return out;
}
