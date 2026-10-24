/**
 * @file tensor_autograd.c
 * @brief Dynamic Tensor Tape and Automatic Differentiation Engine for RCRE.
 *
 * Part of RCRE Unified Engine.
 *
 * Unified tensor model (v0.2.2): this file operates exclusively on
 * RCRE_Tensor (rcre.h). Training tensors are arena-owned RCRE_Tensor* with a
 * `grad` backprop buffer, produced by rcre_tensor_alloc*. All F32 element
 * access goes through rcre_tensor_f32_data(); operators validate dtype and
 * shape and return NULL (no tape node recorded) on any violation.
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
/* Training Tensor Allocation & Initialization (arena-owned RCRE_Tensor)     */
/* ========================================================================= */

static void compute_strides(const size_t* shape, uint8_t ndim, ptrdiff_t* strides) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (uint8_t i = ndim - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * (ptrdiff_t)shape[i];
    }
}

static bool is_f32(const RCRE_Tensor* t) {
    return t != nullptr && t->data != nullptr && t->dtype == RCRE_DTYPE_F32 && t->ndim != 0;
}

RCRE_Tensor* rcre_tensor_alloc(RCRE_Arena* arena, const size_t* shape, uint8_t ndim) {
    if (arena == nullptr || ndim == 0 || ndim > 4 || shape == nullptr) return nullptr;

    RCRE_Tensor* t = (RCRE_Tensor*)rcre_arena_alloc(arena, sizeof(RCRE_Tensor), alignof(RCRE_Tensor));
    if (!t) return nullptr;

    /* Zero-initialize: fields beyond `ndim` must never read back
     * uninitialized memory (e.g. `y->shape[1]` on a 1-D tensor). */
    memset(t, 0, sizeof(RCRE_Tensor));

    t->ndim = ndim;
    t->dtype = RCRE_DTYPE_F32;
    t->layout = RCRE_LAYOUT_TABULAR;
    for (size_t i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
    }
    t->numel = rcre_tensor_numel(t);
    compute_strides(t->shape, t->ndim, t->strides);
    t->is_view = false;
    t->label = nullptr;

    size_t byte_size = t->numel * sizeof(float);
    t->data = rcre_arena_alloc(arena, byte_size, 64);
    t->grad = (float*)rcre_arena_alloc(arena, byte_size, 64);
    if (!t->data || !t->grad) {
        return nullptr; /* arena space already consumed; rewind via temp checkpoint */
    }

    memset(t->data, 0, byte_size);
    memset(t->grad, 0, byte_size);
    return t;
}

RCRE_Tensor* rcre_tensor_alloc_1d(RCRE_Arena* arena, size_t d0) {
    size_t shape[1] = { d0 };
    return rcre_tensor_alloc(arena, shape, 1);
}

RCRE_Tensor* rcre_tensor_alloc_2d(RCRE_Arena* arena, size_t rows, size_t cols) {
    size_t shape[2] = { rows, cols };
    return rcre_tensor_alloc(arena, shape, 2);
}

RCRE_Tensor* rcre_tensor_alloc_4d(RCRE_Arena* arena, size_t n, size_t c, size_t h, size_t w) {
    size_t shape[4] = { n, c, h, w };
    return rcre_tensor_alloc(arena, shape, 4);
}

void rcre_tensor_fill(RCRE_Tensor* t, float val) {
    if (!is_f32(t)) return;
    for (size_t i = 0; i < t->numel; ++i) {
        rcre_tensor_f32_data(t)[i] = val;
    }
}

void rcre_tensor_zeros(RCRE_Tensor* t) {
    rcre_tensor_fill(t, 0.0f);
}

void rcre_tensor_ones(RCRE_Tensor* t) {
    rcre_tensor_fill(t, 1.0f);
}

void rcre_tensor_zero_grad(RCRE_Tensor* t) {
    if (t && t->grad) {
        memset(t->grad, 0, t->numel * sizeof(float));
    }
}

void rcre_tensor_init_xavier(RCRE_Tensor* t, size_t fan_in, size_t fan_out, uint64_t* rng) {
    if (!is_f32(t) || !rng || (fan_in + fan_out) == 0) return;
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < t->numel; ++i) {
        rcre_tensor_f32_data(t)[i] = rcre_rng_uniform_f32(rng, -limit, limit);
    }
}

void rcre_tensor_init_he(RCRE_Tensor* t, size_t fan_in, uint64_t* rng) {
    if (!is_f32(t) || !rng || fan_in == 0) return;
    float std = sqrtf(2.0f / (float)fan_in);
    for (size_t i = 0; i < t->numel; ++i) {
        rcre_tensor_f32_data(t)[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
    }
}

RCRE_Tensor* rcre_tensor_view(RCRE_Arena* arena, const RCRE_Tensor* src) {
    if (!src || arena == nullptr) return nullptr;
    RCRE_Tensor* v = (RCRE_Tensor*)rcre_arena_alloc(arena, sizeof(RCRE_Tensor), alignof(RCRE_Tensor));
    if (!v) return nullptr;
    *v = *src;
    v->grad = nullptr; /* parent grad is indexed in parent coordinates */
    v->is_view = true;
    return v;
}

RCRE_Tensor* rcre_tensor_reshape_view(RCRE_Arena* arena, const RCRE_Tensor* src, const size_t* new_shape, uint8_t new_ndim) {
    if (!src || new_ndim == 0 || new_ndim > 4 || !rcre_tensor_is_contiguous(src)) return nullptr;
    size_t new_numel = 1;
    for (size_t i = 0; i < new_ndim; ++i) {
        new_numel *= new_shape[i];
    }
    if (new_numel != rcre_tensor_numel(src)) return nullptr;

    RCRE_Tensor* v = rcre_tensor_view(arena, src);
    if (!v) return nullptr;
    v->ndim = new_ndim;
    for (size_t i = 0; i < new_ndim; ++i) {
        v->shape[i] = new_shape[i];
    }
    compute_strides(v->shape, v->ndim, v->strides);
    return v;
}

void rcre_tensor_print(const RCRE_Tensor* t, const char* name) {
    if (!t) return;
    printf("Tensor '%s' [ndim=%u, numel=%zu]:\n", name ? name : "unnamed", (unsigned)t->ndim, t->numel);
    if (t->ndim == 1) {
        printf("  [");
        for (size_t i = 0; i < t->numel && i < 10; ++i) {
            printf("%.4f%s", rcre_tensor_f32_data(t)[i], (i + 1 < t->numel) ? ", " : "");
        }
        if (t->numel > 10) printf("...");
        printf("]\n");
    } else if (t->ndim == 2) {
        for (size_t r = 0; r < t->shape[0] && r < 5; ++r) {
            printf("  [");
            for (size_t c = 0; c < t->shape[1] && c < 8; ++c) {
                printf("%.4f%s", rcre_tensor_f32_data(t)[r * (size_t)t->strides[0] + c], (c + 1 < t->shape[1]) ? ", " : "");
            }
            if (t->shape[1] > 8) printf("...");
            printf("]\n");
        }
        if (t->shape[0] > 5) printf("  ...\n");
    }
}

RCRE_Tensor* rcre_tensor_matmul(RCRE_Arena* arena, const RCRE_Tensor* a, const RCRE_Tensor* b) {
    if (!is_f32(a) || !is_f32(b) || a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) return nullptr;
    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];

    RCRE_Tensor* c = rcre_tensor_alloc_2d(arena, M, N);
    if (!c) return nullptr;
    rcre_ag_gemm(false, false, M, N, K, 1.0f,
                 rcre_tensor_f32_data(a), (size_t)a->strides[0],
                 rcre_tensor_f32_data(b), (size_t)b->strides[0],
                 0.0f, rcre_tensor_f32_data(c), (size_t)c->strides[0]);
    return c;
}

void rcre_tensor_add_inplace(RCRE_Tensor* a, const RCRE_Tensor* b) {
    if (!is_f32(a) || !is_f32(b) || a->numel != b->numel) return;
    float* ad = rcre_tensor_f32_data(a);
    const float* bd = rcre_tensor_f32_data(b);
    for (size_t i = 0; i < a->numel; ++i) {
        ad[i] += bd[i];
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
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* a = node->inputs[0];
    RCRE_Tensor* b = node->inputs[1];

    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];

    /* dA += dC * B^T */
    if (a->grad != nullptr) {
        rcre_ag_gemm(false, true, M, K, N, 1.0f, out->grad, N,
                     rcre_tensor_f32_data(b), N, 1.0f, a->grad, K);
    }

    /* dB += A^T * dC */
    if (b->grad != nullptr) {
        rcre_ag_gemm(true, false, K, N, M, 1.0f, rcre_tensor_f32_data(a), K,
                     out->grad, N, 1.0f, b->grad, N);
    }
}

static void backward_add(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* a = node->inputs[0];
    RCRE_Tensor* b = node->inputs[1];

    for (size_t i = 0; i < out->numel; ++i) {
        if (a->grad != nullptr) a->grad[i] += out->grad[i];
        if (b->grad != nullptr) b->grad[i] += out->grad[i];
    }
}

static void backward_add_bias(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x = node->inputs[0];
    RCRE_Tensor* bias = node->inputs[1];

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
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x = node->inputs[0];

    if (x->grad == nullptr) return;
    const float* xd = rcre_tensor_f32_data(x);
    const float* og = out->grad;
    float* xg = x->grad;
    for (size_t i = 0; i < out->numel; ++i) {
        if (xd[i] > 0.0f) {
            xg[i] += og[i];
        }
    }
}

static void backward_sigmoid(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x = node->inputs[0];

    if (x->grad == nullptr) return;
    const float* od = rcre_tensor_f32_data(out);
    const float* og = out->grad;
    float* xg = x->grad;
    for (size_t i = 0; i < out->numel; ++i) {
        float s = od[i];
        xg[i] += s * (1.0f - s) * og[i];
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

RCRE_Tensor* rcre_autograd_matmul(RCRE_TensorTape* tape, RCRE_Tensor* a, RCRE_Tensor* b) {
    RCRE_Tensor* out = rcre_tensor_matmul(tape->arena, a, b);
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

RCRE_Tensor* rcre_autograd_add(RCRE_TensorTape* tape, RCRE_Tensor* a, RCRE_Tensor* b) {
    if (!is_f32(a) || !is_f32(b) || a->numel != b->numel) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, a->shape, a->ndim);
    if (!out) return nullptr;
    const float* ad = rcre_tensor_f32_data(a);
    const float* bd = rcre_tensor_f32_data(b);
    float* od = rcre_tensor_f32_data(out);
    for (size_t i = 0; i < out->numel; ++i) {
        od[i] = ad[i] + bd[i];
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

RCRE_Tensor* rcre_autograd_add_bias(RCRE_TensorTape* tape, RCRE_Tensor* x, RCRE_Tensor* bias) {
    if (!is_f32(x) || !is_f32(bias) || x->ndim != 2 || bias->ndim != 1 || bias->shape[0] != x->shape[1]) return nullptr;
    size_t rows = x->shape[0];
    size_t cols = x->shape[1];

    RCRE_Tensor* out = rcre_tensor_alloc_2d(tape->arena, rows, cols);
    if (!out) return nullptr;
    const float* xd = rcre_tensor_f32_data(x);
    const float* bd = rcre_tensor_f32_data(bias);
    float* od = rcre_tensor_f32_data(out);
    for (size_t r = 0; r < rows; ++r) {
        const float* x_row = xd + r * cols;
        float* out_row = od + r * cols;
        for (size_t c = 0; c < cols; ++c) {
            out_row[c] = x_row[c] + bd[c];
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

RCRE_Tensor* rcre_autograd_relu(RCRE_TensorTape* tape, RCRE_Tensor* x) {
    if (!is_f32(x)) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;
    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t i = 0; i < out->numel; ++i) {
        od[i] = (xd[i] > 0.0f) ? xd[i] : 0.0f;
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out;
    node->inputs[0] = x;
    node->num_inputs = 1;
    node->backward = backward_relu;
    return out;
}

RCRE_Tensor* rcre_autograd_sigmoid(RCRE_TensorTape* tape, RCRE_Tensor* x) {
    if (!is_f32(x)) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;
    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t i = 0; i < out->numel; ++i) {
        od[i] = 1.0f / (1.0f + expf(-xd[i]));
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out;
    node->inputs[0] = x;
    node->num_inputs = 1;
    node->backward = backward_sigmoid;
    return out;
}
