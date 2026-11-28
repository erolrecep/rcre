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
    if (arena == nullptr || ndim == 0 || ndim > RCRE_MAX_DIMS || shape == nullptr) return nullptr;

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

RCRE_Tensor* rcre_tensor_alloc_3d(RCRE_Arena* arena, size_t d0, size_t d1, size_t d2) {
    size_t shape[3] = { d0, d1, d2 };
    return rcre_tensor_alloc(arena, shape, 3);
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

void rcre_tensor_fill_normal(RCRE_Tensor* t, float mean, float std, uint64_t* rng) {
    if (!is_f32(t) || !rng) return;
    float* td = rcre_tensor_f32_data(t);
    for (size_t i = 0; i < t->numel; ++i) {
        td[i] = rcre_rng_gaussian_f32(rng, mean, std);
    }
}

RCRE_Tensor* rcre_sinusoidal_timestep_embed(
    RCRE_Arena* arena, const float* timesteps, size_t batch_size, size_t embed_dim) {
    if (arena == nullptr || timesteps == nullptr || batch_size == 0 ||
        embed_dim < 2 || (embed_dim & 1U) != 0) {
        return nullptr;
    }
    RCRE_Tensor* out = rcre_tensor_alloc_2d(arena, batch_size, embed_dim);
    if (!out) return nullptr;

    float* od = rcre_tensor_f32_data(out);
    size_t half = embed_dim / 2U;
    const float log_max_period = logf(10000.0f);

    for (size_t k = 0; k < half; ++k) {
        float freq = expf(-log_max_period * (float)k / (float)half);
        for (size_t b = 0; b < batch_size; ++b) {
            float angle = timesteps[b] * freq;
            od[b * embed_dim + k]        = sinf(angle);
            od[b * embed_dim + half + k] = cosf(angle);
        }
    }
    return out;
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
    if (!src || new_ndim == 0 || new_ndim > RCRE_MAX_DIMS || !rcre_tensor_is_contiguous(src)) return nullptr;
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
    RCRE_TensorNode* n = &tape->nodes[tape->count++];
    memset(n, 0, sizeof(*n));
    return n;
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

static void backward_conv2d(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    RCRE_Tensor* w   = node->inputs[1];
    size_t stride  = node->aux[0];
    size_t padding = node->aux[1];

    size_t N = x->shape[0], C_in = x->shape[1], H = x->shape[2], W = x->shape[3];
    size_t C_out = w->shape[0], kH = w->shape[2], kW = w->shape[3];
    size_t H_out = out->shape[2], W_out = out->shape[3];

    const float* xd = rcre_tensor_f32_data(x);
    const float* wd = rcre_tensor_f32_data(w);
    const float* og = out->grad;

    for (size_t n = 0; n < N; ++n) {
        for (size_t co = 0; co < C_out; ++co) {
            for (size_t ho = 0; ho < H_out; ++ho) {
                for (size_t wo = 0; wo < W_out; ++wo) {
                    float g = og[((n * C_out + co) * H_out + ho) * W_out + wo];
                    if (g == 0.0f) continue;
                    for (size_t ci = 0; ci < C_in; ++ci) {
                        for (size_t kh = 0; kh < kH; ++kh) {
                            ptrdiff_t hi = (ptrdiff_t)(ho * stride + kh) - (ptrdiff_t)padding;
                            if (hi < 0 || hi >= (ptrdiff_t)H) continue;
                            for (size_t kw = 0; kw < kW; ++kw) {
                                ptrdiff_t wi = (ptrdiff_t)(wo * stride + kw) - (ptrdiff_t)padding;
                                if (wi < 0 || wi >= (ptrdiff_t)W) continue;
                                size_t x_idx = ((n * C_in + ci) * H + (size_t)hi) * W + (size_t)wi;
                                size_t w_idx = ((co * C_in + ci) * kH + kh) * kW + kw;
                                if (x->grad) x->grad[x_idx] += g * wd[w_idx];
                                if (w->grad) w->grad[w_idx] += g * xd[x_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

RCRE_Tensor* rcre_autograd_conv2d(RCRE_TensorTape* tape, RCRE_Tensor* x, RCRE_Tensor* weight,
                                  size_t stride, size_t padding) {
    if (!tape || !is_f32(x) || !is_f32(weight) || x->ndim != 4 || weight->ndim != 4 ||
        stride == 0 || x->shape[1] != weight->shape[1]) {
        return nullptr;
    }
    size_t N = x->shape[0], C_in = x->shape[1], H = x->shape[2], W = x->shape[3];
    size_t C_out = weight->shape[0], kH = weight->shape[2], kW = weight->shape[3];
    if (H + 2U * padding < kH || W + 2U * padding < kW) return nullptr;
    size_t H_out = (H + 2U * padding - kH) / stride + 1U;
    size_t W_out = (W + 2U * padding - kW) / stride + 1U;

    size_t out_shape[4] = { N, C_out, H_out, W_out };
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, out_shape, 4);
    if (!out) return nullptr;

    const float* xd = rcre_tensor_f32_data(x);
    const float* wd = rcre_tensor_f32_data(weight);
    float* od = rcre_tensor_f32_data(out);

    for (size_t n = 0; n < N; ++n) {
        for (size_t co = 0; co < C_out; ++co) {
            for (size_t ho = 0; ho < H_out; ++ho) {
                for (size_t wo = 0; wo < W_out; ++wo) {
                    float acc = 0.0f;
                    for (size_t ci = 0; ci < C_in; ++ci) {
                        for (size_t kh = 0; kh < kH; ++kh) {
                            ptrdiff_t hi = (ptrdiff_t)(ho * stride + kh) - (ptrdiff_t)padding;
                            if (hi < 0 || hi >= (ptrdiff_t)H) continue;
                            for (size_t kw = 0; kw < kW; ++kw) {
                                ptrdiff_t wi = (ptrdiff_t)(wo * stride + kw) - (ptrdiff_t)padding;
                                if (wi < 0 || wi >= (ptrdiff_t)W) continue;
                                acc += xd[((n * C_in + ci) * H + (size_t)hi) * W + (size_t)wi] *
                                       wd[((co * C_in + ci) * kH + kh) * kW + kw];
                            }
                        }
                    }
                    od[((n * C_out + co) * H_out + ho) * W_out + wo] = acc;
                }
            }
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out        = out;
    node->inputs[0]  = x;
    node->inputs[1]  = weight;
    node->num_inputs = 2;
    node->aux[0]     = stride;
    node->aux[1]     = padding;
    node->backward   = backward_conv2d;
    return out;
}

static void backward_attention(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* q   = node->inputs[0];
    RCRE_Tensor* k   = node->inputs[1];
    RCRE_Tensor* v   = node->inputs[2];

    size_t T = q->shape[0];
    size_t D = q->shape[1];
    float scale = 1.0f / sqrtf((float)D);
    const float* qd = rcre_tensor_f32_data(q);
    const float* kd = rcre_tensor_f32_data(k);
    const float* vd = rcre_tensor_f32_data(v);
    const float* og = out->grad;

    float* P = (float*)malloc(T * sizeof(float));
    float* dP = (float*)malloc(T * sizeof(float));
    if (!P || !dP) { free(P); free(dP); return; }

    for (size_t i = 0; i < T; ++i) {
        float mx = -1e30f;
        for (size_t j = 0; j < T; ++j) {
            float s = 0.0f;
            for (size_t d = 0; d < D; ++d) s += qd[i * D + d] * kd[j * D + d];
            P[j] = s * scale;
            if (P[j] > mx) mx = P[j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j < T; ++j) { P[j] = expf(P[j] - mx); sum += P[j]; }
        for (size_t j = 0; j < T; ++j) P[j] /= sum;

        /* dV += P^T dO, dP[j] = dot(dO[i], V[j]) */
        float row_dot = 0.0f;
        for (size_t j = 0; j < T; ++j) {
            float dp_ij = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                dp_ij += og[i * D + d] * vd[j * D + d];
                if (v->grad) v->grad[j * D + d] += P[j] * og[i * D + d];
            }
            dP[j] = dp_ij;
            row_dot += P[j] * dp_ij;
        }

        /* dS[j] = P[j] * (dP[j] - sum_k P[k]*dP[k]) */
        for (size_t j = 0; j < T; ++j) {
            float dS = P[j] * (dP[j] - row_dot) * scale;
            for (size_t d = 0; d < D; ++d) {
                if (q->grad) q->grad[i * D + d] += dS * kd[j * D + d];
                if (k->grad) k->grad[j * D + d] += dS * qd[i * D + d];
            }
        }
    }

    free(P);
    free(dP);
}

RCRE_Tensor* rcre_autograd_attention(RCRE_TensorTape* tape, RCRE_Tensor* q, RCRE_Tensor* k, RCRE_Tensor* v) {
    if (!tape || !is_f32(q) || !is_f32(k) || !is_f32(v) ||
        q->ndim != 2 || k->ndim != 2 || v->ndim != 2 ||
        q->shape[0] != k->shape[0] || q->shape[0] != v->shape[0] ||
        q->shape[1] != k->shape[1] || q->shape[1] != v->shape[1]) {
        return nullptr;
    }
    size_t T = q->shape[0], D = q->shape[1];
    RCRE_Tensor* out = rcre_tensor_alloc_2d(tape->arena, T, D);
    if (!out) return nullptr;

    float* P = (float*)malloc(T * sizeof(float));
    if (!P) return nullptr;

    const float* qd = rcre_tensor_f32_data(q);
    const float* kd = rcre_tensor_f32_data(k);
    const float* vd = rcre_tensor_f32_data(v);
    float* od = rcre_tensor_f32_data(out);
    float scale = 1.0f / sqrtf((float)D);

    for (size_t i = 0; i < T; ++i) {
        float mx = -1e30f;
        for (size_t j = 0; j < T; ++j) {
            float s = 0.0f;
            for (size_t d = 0; d < D; ++d) s += qd[i * D + d] * kd[j * D + d];
            P[j] = s * scale;
            if (P[j] > mx) mx = P[j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j < T; ++j) { P[j] = expf(P[j] - mx); sum += P[j]; }
        for (size_t j = 0; j < T; ++j) P[j] /= sum;
        for (size_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (size_t j = 0; j < T; ++j) acc += P[j] * vd[j * D + d];
            od[i * D + d] = acc;
        }
    }
    free(P);

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out        = out;
    node->inputs[0]  = q;
    node->inputs[1]  = k;
    node->inputs[2]  = v;
    node->num_inputs = 3;
    node->backward   = backward_attention;
    return out;
}

/* ========================================================================= */
/* Phase 0 Generative & Vision Autograd Primitives                           */
/* ========================================================================= */

static inline bool ag_is_f32_contig(const RCRE_Tensor* t) {
    return is_f32(t) && rcre_tensor_is_contiguous(t);
}

static inline size_t ag_pack_f32(float v) {
    uint32_t bits = 0;
    memcpy(&bits, &v, sizeof(bits));
    return (size_t)bits;
}

static inline float ag_unpack_f32(size_t u) {
    uint32_t bits = (uint32_t)u;
    float v = 0.0f;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

/* --- Elementwise: sub, mul, scale, exp --- */

static void ag_backward_sub(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* a   = node->inputs[0];
    RCRE_Tensor* b   = node->inputs[1];
    const float* og  = out->grad;
    for (size_t i = 0; i < out->numel; ++i) {
        if (a->grad) a->grad[i] += og[i];
        if (b->grad) b->grad[i] -= og[i];
    }
}

RCRE_Tensor* rcre_autograd_sub(RCRE_TensorTape* tape, RCRE_Tensor* a, RCRE_Tensor* b) {
    if (!tape || !ag_is_f32_contig(a) || !ag_is_f32_contig(b) || a->numel != b->numel) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, a->shape, a->ndim);
    if (!out) return nullptr;
    const float* ad = rcre_tensor_f32_data(a);
    const float* bd = rcre_tensor_f32_data(b);
    float* od = rcre_tensor_f32_data(out);
    for (size_t i = 0; i < out->numel; ++i) od[i] = ad[i] - bd[i];

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = a; node->inputs[1] = b; node->num_inputs = 2;
    node->backward = ag_backward_sub;
    return out;
}

static void ag_backward_mul(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* a   = node->inputs[0];
    RCRE_Tensor* b   = node->inputs[1];
    const float* ad  = rcre_tensor_f32_data(a);
    const float* bd  = rcre_tensor_f32_data(b);
    const float* og  = out->grad;
    for (size_t i = 0; i < out->numel; ++i) {
        if (a->grad) a->grad[i] += bd[i] * og[i];
        if (b->grad) b->grad[i] += ad[i] * og[i];
    }
}

RCRE_Tensor* rcre_autograd_mul(RCRE_TensorTape* tape, RCRE_Tensor* a, RCRE_Tensor* b) {
    if (!tape || !ag_is_f32_contig(a) || !ag_is_f32_contig(b) || a->numel != b->numel) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, a->shape, a->ndim);
    if (!out) return nullptr;
    const float* ad = rcre_tensor_f32_data(a);
    const float* bd = rcre_tensor_f32_data(b);
    float* od = rcre_tensor_f32_data(out);
    for (size_t i = 0; i < out->numel; ++i) od[i] = ad[i] * bd[i];

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = a; node->inputs[1] = b; node->num_inputs = 2;
    node->backward = ag_backward_mul;
    return out;
}

static void ag_backward_scale(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    float s = ag_unpack_f32(node->aux[0]);
    const float* og = out->grad;
    float* xg = x->grad;
    for (size_t i = 0; i < out->numel; ++i) xg[i] += og[i] * s;
}

RCRE_Tensor* rcre_autograd_scale(RCRE_TensorTape* tape, RCRE_Tensor* x, float scalar) {
    if (!tape || !ag_is_f32_contig(x)) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;
    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t i = 0; i < out->numel; ++i) od[i] = xd[i] * scalar;

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->aux[0] = ag_pack_f32(scalar);
    node->backward = ag_backward_scale;
    return out;
}

static void ag_backward_exp(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    const float* od = rcre_tensor_f32_data(out);
    const float* og = out->grad;
    float* xg = x->grad;
    for (size_t i = 0; i < out->numel; ++i) xg[i] += od[i] * og[i];
}

RCRE_Tensor* rcre_autograd_exp(RCRE_TensorTape* tape, RCRE_Tensor* x) {
    if (!tape || !ag_is_f32_contig(x)) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;
    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t i = 0; i < out->numel; ++i) od[i] = expf(xd[i]);

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->backward = ag_backward_exp;
    return out;
}

/* --- 4D Channel Bias & General 2D/3D Linear --- */

static void ag_backward_add_bias_4d(RCRE_TensorNode* node) {
    RCRE_Tensor* out  = node->out;
    RCRE_Tensor* x    = node->inputs[0];
    RCRE_Tensor* bias = node->inputs[1];
    size_t N = out->shape[0], C = out->shape[1], HW = out->shape[2] * out->shape[3];
    const float* og = out->grad;

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* g_ch = og + (n * C + c) * HW;
            float* x_ch = x->grad ? (x->grad + (n * C + c) * HW) : nullptr;
            float b_acc = 0.0f;
            for (size_t i = 0; i < HW; ++i) {
                if (x_ch) x_ch[i] += g_ch[i];
                b_acc += g_ch[i];
            }
            if (bias->grad) bias->grad[c] += b_acc;
        }
    }
}

RCRE_Tensor* rcre_autograd_add_bias_4d(RCRE_TensorTape* tape, RCRE_Tensor* x, RCRE_Tensor* bias) {
    if (!tape || !ag_is_f32_contig(x) || !ag_is_f32_contig(bias) ||
        x->ndim != 4 || bias->ndim != 1 || bias->shape[0] != x->shape[1]) {
        return nullptr;
    }
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, 4);
    if (!out) return nullptr;
    size_t N = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];
    const float* xd = rcre_tensor_f32_data(x);
    const float* bd = rcre_tensor_f32_data(bias);
    float* od = rcre_tensor_f32_data(out);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            float bv = bd[c];
            const float* x_ch = xd + (n * C + c) * HW;
            float* o_ch = od + (n * C + c) * HW;
            for (size_t i = 0; i < HW; ++i) o_ch[i] = x_ch[i] + bv;
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->inputs[1] = bias; node->num_inputs = 2;
    node->backward = ag_backward_add_bias_4d;
    return out;
}

static void ag_backward_linear(RCRE_TensorNode* node) {
    RCRE_Tensor* out  = node->out;
    RCRE_Tensor* x    = node->inputs[0];
    RCRE_Tensor* w    = node->inputs[1];
    RCRE_Tensor* bias = node->inputs[2];

    size_t Din  = w->shape[0];
    size_t Dout = w->shape[1];
    size_t M    = x->numel / Din;

    if (x->grad) {
        rcre_ag_gemm(false, true, M, Din, Dout, 1.0f,
                     out->grad, Dout, rcre_tensor_f32_data(w), Dout, 1.0f, x->grad, Din);
    }
    if (w->grad) {
        rcre_ag_gemm(true, false, Din, Dout, M, 1.0f,
                     rcre_tensor_f32_data(x), Din, out->grad, Dout, 1.0f, w->grad, Dout);
    }
    if (bias && bias->grad) {
        const float* og = out->grad;
        for (size_t r = 0; r < M; ++r) {
            const float* row = og + r * Dout;
            for (size_t c = 0; c < Dout; ++c) bias->grad[c] += row[c];
        }
    }
}

RCRE_Tensor* rcre_autograd_linear(RCRE_TensorTape* tape, RCRE_Tensor* x, RCRE_Tensor* weight, RCRE_Tensor* bias) {
    if (!tape || !ag_is_f32_contig(x) || !ag_is_f32_contig(weight) ||
        (x->ndim != 2 && x->ndim != 3) || weight->ndim != 2) {
        return nullptr;
    }
    size_t Din  = weight->shape[0];
    size_t Dout = weight->shape[1];
    if (x->shape[x->ndim - 1] != Din) return nullptr;
    if (bias != nullptr) {
        if (!ag_is_f32_contig(bias) || bias->ndim != 1 || bias->shape[0] != Dout) return nullptr;
    }

    size_t out_shape[3] = {0, 0, 0};
    if (x->ndim == 2) {
        out_shape[0] = x->shape[0];
        out_shape[1] = Dout;
    } else {
        out_shape[0] = x->shape[0];
        out_shape[1] = x->shape[1];
        out_shape[2] = Dout;
    }
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, out_shape, x->ndim);
    if (!out) return nullptr;

    size_t M = x->numel / Din;
    float* od = rcre_tensor_f32_data(out);
    rcre_ag_gemm(false, false, M, Dout, Din, 1.0f,
                 rcre_tensor_f32_data(x), Din,
                 rcre_tensor_f32_data(weight), Dout,
                 0.0f, od, Dout);

    if (bias != nullptr) {
        const float* bd = rcre_tensor_f32_data(bias);
        for (size_t r = 0; r < M; ++r) {
            float* row = od + r * Dout;
            for (size_t c = 0; c < Dout; ++c) row[c] += bd[c];
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out;
    node->inputs[0] = x;
    node->inputs[1] = weight;
    node->inputs[2] = bias;
    node->num_inputs = bias ? 3 : 2;
    node->backward = ag_backward_linear;
    return out;
}

/* --- Activations: LeakyReLU, Tanh, SiLU, GELU --- */

static void ag_backward_leaky_relu(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    float slope = ag_unpack_f32(node->aux[0]);
    const float* xd = rcre_tensor_f32_data(x);
    const float* og = out->grad;
    float* xg = x->grad;
    for (size_t i = 0; i < out->numel; ++i) {
        xg[i] += (xd[i] > 0.0f ? 1.0f : slope) * og[i];
    }
}

RCRE_Tensor* rcre_autograd_leaky_relu(RCRE_TensorTape* tape, RCRE_Tensor* x, float neg_slope) {
    if (!tape || !ag_is_f32_contig(x)) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;
    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t i = 0; i < out->numel; ++i) {
        od[i] = (xd[i] > 0.0f) ? xd[i] : (neg_slope * xd[i]);
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->aux[0] = ag_pack_f32(neg_slope);
    node->backward = ag_backward_leaky_relu;
    return out;
}

static void ag_backward_tanh(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    const float* od = rcre_tensor_f32_data(out);
    const float* og = out->grad;
    float* xg = x->grad;
    for (size_t i = 0; i < out->numel; ++i) {
        float t = od[i];
        xg[i] += (1.0f - t * t) * og[i];
    }
}

RCRE_Tensor* rcre_autograd_tanh(RCRE_TensorTape* tape, RCRE_Tensor* x) {
    if (!tape || !ag_is_f32_contig(x)) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;
    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t i = 0; i < out->numel; ++i) od[i] = tanhf(xd[i]);

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->backward = ag_backward_tanh;
    return out;
}

static void ag_backward_silu(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    const float* xd = rcre_tensor_f32_data(x);
    const float* od = rcre_tensor_f32_data(out);
    const float* og = out->grad;
    float* xg = x->grad;
    for (size_t i = 0; i < out->numel; ++i) {
        float sig = 1.0f / (1.0f + expf(-xd[i]));
        float deriv = sig + od[i] * (1.0f - sig);
        xg[i] += deriv * og[i];
    }
}

RCRE_Tensor* rcre_autograd_silu(RCRE_TensorTape* tape, RCRE_Tensor* x) {
    if (!tape || !ag_is_f32_contig(x)) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;
    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t i = 0; i < out->numel; ++i) {
        float v = xd[i];
        od[i] = v / (1.0f + expf(-v));
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->backward = ag_backward_silu;
    return out;
}

static void ag_backward_gelu(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    const float* xd = rcre_tensor_f32_data(x);
    const float* og = out->grad;
    float* xg = x->grad;
    const float c = 0.7978845608028654f; /* sqrt(2 / pi) */
    const float k = 0.044715f;
    for (size_t i = 0; i < out->numel; ++i) {
        float v  = xd[i];
        float v2 = v * v;
        float u  = c * (v + k * v * v2);
        float t  = tanhf(u);
        float du = c * (1.0f + 3.0f * k * v2);
        float deriv = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du;
        xg[i] += deriv * og[i];
    }
}

RCRE_Tensor* rcre_autograd_gelu(RCRE_TensorTape* tape, RCRE_Tensor* x) {
    if (!tape || !ag_is_f32_contig(x)) return nullptr;
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;
    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    const float c = 0.7978845608028654f;
    const float k = 0.044715f;
    for (size_t i = 0; i < out->numel; ++i) {
        float v = xd[i];
        float u = c * (v + k * v * v * v);
        od[i] = 0.5f * v * (1.0f + tanhf(u));
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->backward = ag_backward_gelu;
    return out;
}

/* --- Normalization: LayerNorm & GroupNorm --- */

static void ag_backward_layernorm(RCRE_TensorNode* node) {
    RCRE_Tensor* out   = node->out;
    RCRE_Tensor* x     = node->inputs[0];
    RCRE_Tensor* gamma = node->inputs[1];
    RCRE_Tensor* beta  = node->inputs[2];
    float eps = ag_unpack_f32(node->aux[0]);

    size_t C = x->shape[x->ndim - 1];
    size_t M = x->numel / C;
    float inv_C = 1.0f / (float)C;

    const float* xd = rcre_tensor_f32_data(x);
    const float* gd = gamma ? rcre_tensor_f32_data(gamma) : nullptr;
    const float* og = out->grad;

    for (size_t r = 0; r < M; ++r) {
        const float* xr = xd + r * C;
        const float* or_ = og + r * C;

        float mean = 0.0f;
        for (size_t c = 0; c < C; ++c) mean += xr[c];
        mean *= inv_C;

        float var = 0.0f;
        for (size_t c = 0; c < C; ++c) {
            float d = xr[c] - mean;
            var += d * d;
        }
        var *= inv_C;
        float rstd = 1.0f / sqrtf(var + eps);

        float s1 = 0.0f;
        float s2 = 0.0f;
        for (size_t c = 0; c < C; ++c) {
            float xhat = (xr[c] - mean) * rstd;
            float dy = or_[c];
            if (gamma && gamma->grad) gamma->grad[c] += dy * xhat;
            if (beta  && beta->grad)  beta->grad[c]  += dy;
            float g_val = gd ? (dy * gd[c]) : dy;
            s1 += g_val;
            s2 += g_val * xhat;
        }
        s1 *= inv_C;
        s2 *= inv_C;

        if (x->grad) {
            float* xgr = x->grad + r * C;
            for (size_t c = 0; c < C; ++c) {
                float xhat = (xr[c] - mean) * rstd;
                float g_val = gd ? (or_[c] * gd[c]) : or_[c];
                xgr[c] += rstd * (g_val - s1 - xhat * s2);
            }
        }
    }
}

RCRE_Tensor* rcre_autograd_layernorm(RCRE_TensorTape* tape, RCRE_Tensor* x,
                                     RCRE_Tensor* gamma, RCRE_Tensor* beta, float eps) {
    if (!tape || !ag_is_f32_contig(x) || (x->ndim != 2 && x->ndim != 3) || !(eps > 0.0f)) {
        return nullptr;
    }
    size_t C = x->shape[x->ndim - 1];
    if (C == 0) return nullptr;
    if (gamma && (!ag_is_f32_contig(gamma) || gamma->ndim != 1 || gamma->shape[0] != C)) return nullptr;
    if (beta  && (!ag_is_f32_contig(beta)  || beta->ndim  != 1 || beta->shape[0]  != C)) return nullptr;

    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, x->ndim);
    if (!out) return nullptr;

    size_t M = x->numel / C;
    float inv_C = 1.0f / (float)C;
    const float* xd = rcre_tensor_f32_data(x);
    const float* gd = gamma ? rcre_tensor_f32_data(gamma) : nullptr;
    const float* bd = beta  ? rcre_tensor_f32_data(beta)  : nullptr;
    float* od = rcre_tensor_f32_data(out);

    for (size_t r = 0; r < M; ++r) {
        const float* xr = xd + r * C;
        float* or_ = od + r * C;

        float mean = 0.0f;
        for (size_t c = 0; c < C; ++c) mean += xr[c];
        mean *= inv_C;

        float var = 0.0f;
        for (size_t c = 0; c < C; ++c) {
            float d = xr[c] - mean;
            var += d * d;
        }
        var *= inv_C;
        float rstd = 1.0f / sqrtf(var + eps);

        for (size_t c = 0; c < C; ++c) {
            float v = (xr[c] - mean) * rstd;
            if (gd) v *= gd[c];
            if (bd) v += bd[c];
            or_[c] = v;
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out;
    node->inputs[0] = x;
    node->inputs[1] = gamma;
    node->inputs[2] = beta;
    node->num_inputs = 3;
    node->aux[0] = ag_pack_f32(eps);
    node->backward = ag_backward_layernorm;
    return out;
}

static void ag_backward_groupnorm(RCRE_TensorNode* node) {
    RCRE_Tensor* out   = node->out;
    RCRE_Tensor* x     = node->inputs[0];
    RCRE_Tensor* gamma = node->inputs[1];
    RCRE_Tensor* beta  = node->inputs[2];
    size_t G  = node->aux[0];
    float eps = ag_unpack_f32(node->aux[1]);

    size_t N = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];
    size_t cpg = C / G;
    size_t group_sz = cpg * HW;
    float inv_gsz = 1.0f / (float)group_sz;

    const float* xd = rcre_tensor_f32_data(x);
    const float* gd = gamma ? rcre_tensor_f32_data(gamma) : nullptr;
    const float* og = out->grad;

    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < G; ++g) {
            size_t c0 = g * cpg;
            float mean = 0.0f;
            for (size_t ci = 0; ci < cpg; ++ci) {
                const float* ch = xd + (n * C + c0 + ci) * HW;
                for (size_t i = 0; i < HW; ++i) mean += ch[i];
            }
            mean *= inv_gsz;

            float var = 0.0f;
            for (size_t ci = 0; ci < cpg; ++ci) {
                const float* ch = xd + (n * C + c0 + ci) * HW;
                for (size_t i = 0; i < HW; ++i) {
                    float d = ch[i] - mean;
                    var += d * d;
                }
            }
            var *= inv_gsz;
            float rstd = 1.0f / sqrtf(var + eps);

            float s1 = 0.0f;
            float s2 = 0.0f;
            for (size_t ci = 0; ci < cpg; ++ci) {
                size_t c = c0 + ci;
                const float* xch = xd + (n * C + c) * HW;
                const float* gch = og + (n * C + c) * HW;
                float gscale = gd ? gd[c] : 1.0f;
                float dg_acc = 0.0f;
                float db_acc = 0.0f;
                for (size_t i = 0; i < HW; ++i) {
                    float xhat = (xch[i] - mean) * rstd;
                    float dy = gch[i];
                    dg_acc += dy * xhat;
                    db_acc += dy;
                    float dxhat = dy * gscale;
                    s1 += dxhat;
                    s2 += dxhat * xhat;
                }
                if (gamma && gamma->grad) gamma->grad[c] += dg_acc;
                if (beta  && beta->grad)  beta->grad[c]  += db_acc;
            }
            s1 *= inv_gsz;
            s2 *= inv_gsz;

            if (x->grad) {
                for (size_t ci = 0; ci < cpg; ++ci) {
                    size_t c = c0 + ci;
                    const float* xch = xd + (n * C + c) * HW;
                    const float* gch = og + (n * C + c) * HW;
                    float* dxch = x->grad + (n * C + c) * HW;
                    float gscale = gd ? gd[c] : 1.0f;
                    for (size_t i = 0; i < HW; ++i) {
                        float xhat = (xch[i] - mean) * rstd;
                        float dxhat = gch[i] * gscale;
                        dxch[i] += rstd * (dxhat - s1 - xhat * s2);
                    }
                }
            }
        }
    }
}

RCRE_Tensor* rcre_autograd_groupnorm(RCRE_TensorTape* tape, RCRE_Tensor* x,
                                     RCRE_Tensor* gamma, RCRE_Tensor* beta,
                                     size_t num_groups, float eps) {
    if (!tape || !ag_is_f32_contig(x) || x->ndim != 4 || num_groups == 0 ||
        x->shape[1] == 0 || (x->shape[1] % num_groups) != 0 || !(eps > 0.0f)) {
        return nullptr;
    }
    size_t N = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];
    if (HW == 0) return nullptr;
    if (gamma && (!ag_is_f32_contig(gamma) || gamma->ndim != 1 || gamma->shape[0] != C)) return nullptr;
    if (beta  && (!ag_is_f32_contig(beta)  || beta->ndim  != 1 || beta->shape[0]  != C)) return nullptr;

    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, 4);
    if (!out) return nullptr;

    size_t cpg = C / num_groups;
    size_t group_sz = cpg * HW;
    float inv_gsz = 1.0f / (float)group_sz;

    const float* xd = rcre_tensor_f32_data(x);
    const float* gd = gamma ? rcre_tensor_f32_data(gamma) : nullptr;
    const float* bd = beta  ? rcre_tensor_f32_data(beta)  : nullptr;
    float* od = rcre_tensor_f32_data(out);

    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            size_t c0 = g * cpg;
            float mean = 0.0f;
            for (size_t ci = 0; ci < cpg; ++ci) {
                const float* ch = xd + (n * C + c0 + ci) * HW;
                for (size_t i = 0; i < HW; ++i) mean += ch[i];
            }
            mean *= inv_gsz;

            float var = 0.0f;
            for (size_t ci = 0; ci < cpg; ++ci) {
                const float* ch = xd + (n * C + c0 + ci) * HW;
                for (size_t i = 0; i < HW; ++i) {
                    float d = ch[i] - mean;
                    var += d * d;
                }
            }
            var *= inv_gsz;
            float rstd = 1.0f / sqrtf(var + eps);

            for (size_t ci = 0; ci < cpg; ++ci) {
                size_t c = c0 + ci;
                const float* xch = xd + (n * C + c) * HW;
                float* och = od + (n * C + c) * HW;
                float gv = gd ? gd[c] : 1.0f;
                float bv = bd ? bd[c] : 0.0f;
                for (size_t i = 0; i < HW; ++i) {
                    och[i] = ((xch[i] - mean) * rstd) * gv + bv;
                }
            }
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out;
    node->inputs[0] = x;
    node->inputs[1] = gamma;
    node->inputs[2] = beta;
    node->num_inputs = 3;
    node->aux[0] = num_groups;
    node->aux[1] = ag_pack_f32(eps);
    node->backward = ag_backward_groupnorm;
    return out;
}

/* --- Conditioning: FiLM 4D (UNet), AdaLN 3D & Gate 3D (DiT) --- */

static void ag_backward_film_4d(RCRE_TensorNode* node) {
    RCRE_Tensor* out   = node->out;
    RCRE_Tensor* x     = node->inputs[0];
    RCRE_Tensor* scale = node->inputs[1];
    RCRE_Tensor* shift = node->inputs[2];
    size_t N = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];

    const float* xd = rcre_tensor_f32_data(x);
    const float* sd = scale ? rcre_tensor_f32_data(scale) : nullptr;
    const float* og = out->grad;

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            size_t nc = n * C + c;
            float mult = 1.0f + (sd ? sd[nc] : 0.0f);
            const float* xch = xd + nc * HW;
            const float* gch = og + nc * HW;
            float* dxch = x->grad ? (x->grad + nc * HW) : nullptr;
            float ds = 0.0f, db = 0.0f;
            for (size_t i = 0; i < HW; ++i) {
                if (dxch) dxch[i] += gch[i] * mult;
                ds += gch[i] * xch[i];
                db += gch[i];
            }
            if (scale && scale->grad) scale->grad[nc] += ds;
            if (shift && shift->grad) shift->grad[nc] += db;
        }
    }
}

RCRE_Tensor* rcre_autograd_film_4d(RCRE_TensorTape* tape, RCRE_Tensor* x,
                                   RCRE_Tensor* scale, RCRE_Tensor* shift) {
    if (!tape || !ag_is_f32_contig(x) || x->ndim != 4) return nullptr;
    size_t N = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];
    if (scale && (!ag_is_f32_contig(scale) || scale->ndim != 2 || scale->shape[0] != N || scale->shape[1] != C)) return nullptr;
    if (shift && (!ag_is_f32_contig(shift) || shift->ndim != 2 || shift->shape[0] != N || shift->shape[1] != C)) return nullptr;

    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, 4);
    if (!out) return nullptr;

    const float* xd  = rcre_tensor_f32_data(x);
    const float* sd  = scale ? rcre_tensor_f32_data(scale) : nullptr;
    const float* shd = shift ? rcre_tensor_f32_data(shift) : nullptr;
    float* od = rcre_tensor_f32_data(out);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            size_t nc = n * C + c;
            float mult = 1.0f + (sd ? sd[nc] : 0.0f);
            float add  = shd ? shd[nc] : 0.0f;
            const float* xch = xd + nc * HW;
            float* och = od + nc * HW;
            for (size_t i = 0; i < HW; ++i) och[i] = xch[i] * mult + add;
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->inputs[1] = scale; node->inputs[2] = shift;
    node->num_inputs = 3;
    node->backward = ag_backward_film_4d;
    return out;
}

static void ag_backward_adaln_3d(RCRE_TensorNode* node) {
    RCRE_Tensor* out   = node->out;
    RCRE_Tensor* x     = node->inputs[0];
    RCRE_Tensor* scale = node->inputs[1];
    RCRE_Tensor* shift = node->inputs[2];
    size_t B = x->shape[0], T = x->shape[1], C = x->shape[2];

    const float* xd = rcre_tensor_f32_data(x);
    const float* sd = scale ? rcre_tensor_f32_data(scale) : nullptr;
    const float* og = out->grad;

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t base = (b * T + t) * C;
            for (size_t c = 0; c < C; ++c) {
                size_t bc = b * C + c;
                float mult = 1.0f + (sd ? sd[bc] : 0.0f);
                float g = og[base + c];
                if (x->grad) x->grad[base + c] += g * mult;
                if (scale && scale->grad) scale->grad[bc] += g * xd[base + c];
                if (shift && shift->grad) shift->grad[bc] += g;
            }
        }
    }
}

RCRE_Tensor* rcre_autograd_adaln_3d(RCRE_TensorTape* tape, RCRE_Tensor* x,
                                    RCRE_Tensor* scale, RCRE_Tensor* shift) {
    if (!tape || !ag_is_f32_contig(x) || x->ndim != 3) return nullptr;
    size_t B = x->shape[0], T = x->shape[1], C = x->shape[2];
    if (scale && (!ag_is_f32_contig(scale) || scale->ndim != 2 || scale->shape[0] != B || scale->shape[1] != C)) return nullptr;
    if (shift && (!ag_is_f32_contig(shift) || shift->ndim != 2 || shift->shape[0] != B || shift->shape[1] != C)) return nullptr;

    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, 3);
    if (!out) return nullptr;

    const float* xd  = rcre_tensor_f32_data(x);
    const float* sd  = scale ? rcre_tensor_f32_data(scale) : nullptr;
    const float* shd = shift ? rcre_tensor_f32_data(shift) : nullptr;
    float* od = rcre_tensor_f32_data(out);

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t base = (b * T + t) * C;
            for (size_t c = 0; c < C; ++c) {
                size_t bc = b * C + c;
                float mult = 1.0f + (sd ? sd[bc] : 0.0f);
                float add  = shd ? shd[bc] : 0.0f;
                od[base + c] = xd[base + c] * mult + add;
            }
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->inputs[1] = scale; node->inputs[2] = shift;
    node->num_inputs = 3;
    node->backward = ag_backward_adaln_3d;
    return out;
}

static void ag_backward_gate_3d(RCRE_TensorNode* node) {
    RCRE_Tensor* out  = node->out;
    RCRE_Tensor* x    = node->inputs[0];
    RCRE_Tensor* gate = node->inputs[1];
    size_t B = x->shape[0], T = x->shape[1], C = x->shape[2];

    const float* xd = rcre_tensor_f32_data(x);
    const float* gd = rcre_tensor_f32_data(gate);
    const float* og = out->grad;

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t base = (b * T + t) * C;
            for (size_t c = 0; c < C; ++c) {
                size_t bc = b * C + c;
                float g = og[base + c];
                if (x->grad) x->grad[base + c] += g * gd[bc];
                if (gate->grad) gate->grad[bc] += g * xd[base + c];
            }
        }
    }
}

RCRE_Tensor* rcre_autograd_gate_3d(RCRE_TensorTape* tape, RCRE_Tensor* x, RCRE_Tensor* gate) {
    if (!tape || !ag_is_f32_contig(x) || !ag_is_f32_contig(gate) ||
        x->ndim != 3 || gate->ndim != 2 ||
        gate->shape[0] != x->shape[0] || gate->shape[1] != x->shape[2]) {
        return nullptr;
    }
    size_t B = x->shape[0], T = x->shape[1], C = x->shape[2];
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, 3);
    if (!out) return nullptr;

    const float* xd = rcre_tensor_f32_data(x);
    const float* gd = rcre_tensor_f32_data(gate);
    float* od = rcre_tensor_f32_data(out);

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t base = (b * T + t) * C;
            for (size_t c = 0; c < C; ++c) {
                od[base + c] = xd[base + c] * gd[b * C + c];
            }
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->inputs[1] = gate; node->num_inputs = 2;
    node->backward = ag_backward_gate_3d;
    return out;
}

/* --- Structural & Spatial: reshape, slice_cols, concat_channels, upsample2d, avgpool2d --- */

static void ag_backward_reshape(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    const float* og = out->grad;
    float* xg = x->grad;
    for (size_t i = 0; i < out->numel; ++i) xg[i] += og[i];
}

RCRE_Tensor* rcre_autograd_reshape(RCRE_TensorTape* tape, RCRE_Tensor* x,
                                   const size_t* new_shape, uint8_t new_ndim) {
    if (!tape || !ag_is_f32_contig(x) || !new_shape || new_ndim == 0 || new_ndim > RCRE_MAX_DIMS) {
        return nullptr;
    }
    size_t new_numel = 1;
    for (uint8_t i = 0; i < new_ndim; ++i) {
        if (new_shape[i] == 0) return nullptr;
        new_numel *= new_shape[i];
    }
    if (new_numel != x->numel) return nullptr;

    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, new_shape, new_ndim);
    if (!out) return nullptr;
    memcpy(out->data, x->data, x->numel * sizeof(float));

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->backward = ag_backward_reshape;
    return out;
}

static void ag_backward_slice_cols(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    size_t col_start = node->aux[0];
    size_t col_width = node->aux[1];
    size_t rows = x->shape[0];
    size_t D    = x->shape[1];
    const float* og = out->grad;
    for (size_t r = 0; r < rows; ++r) {
        float* xr = x->grad + r * D + col_start;
        const float* or_ = og + r * col_width;
        for (size_t c = 0; c < col_width; ++c) xr[c] += or_[c];
    }
}

RCRE_Tensor* rcre_autograd_slice_cols(RCRE_TensorTape* tape, RCRE_Tensor* x,
                                      size_t col_start, size_t col_width) {
    if (!tape || !ag_is_f32_contig(x) || x->ndim != 2 ||
        col_width == 0 || col_start > x->shape[1] || col_width > x->shape[1] - col_start) {
        return nullptr;
    }
    size_t rows = x->shape[0];
    size_t D    = x->shape[1];
    RCRE_Tensor* out = rcre_tensor_alloc_2d(tape->arena, rows, col_width);
    if (!out) return nullptr;

    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t r = 0; r < rows; ++r) {
        memcpy(od + r * col_width, xd + r * D + col_start, col_width * sizeof(float));
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->aux[0] = col_start;
    node->aux[1] = col_width;
    node->backward = ag_backward_slice_cols;
    return out;
}

static void ag_backward_concat_channels(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* a   = node->inputs[0];
    RCRE_Tensor* b   = node->inputs[1];
    size_t N  = out->shape[0];
    size_t C1 = a->shape[1];
    size_t C2 = b->shape[1];
    size_t C  = C1 + C2;
    size_t HW = out->shape[2] * out->shape[3];
    const float* og = out->grad;

    for (size_t n = 0; n < N; ++n) {
        if (a->grad) {
            size_t len = C1 * HW;
            float* ag = a->grad + n * len;
            const float* go = og + (n * C) * HW;
            for (size_t i = 0; i < len; ++i) ag[i] += go[i];
        }
        if (b->grad) {
            size_t len = C2 * HW;
            float* bg = b->grad + n * len;
            const float* go = og + (n * C + C1) * HW;
            for (size_t i = 0; i < len; ++i) bg[i] += go[i];
        }
    }
}

RCRE_Tensor* rcre_autograd_concat_channels(RCRE_TensorTape* tape, RCRE_Tensor* a, RCRE_Tensor* b) {
    if (!tape || !ag_is_f32_contig(a) || !ag_is_f32_contig(b) ||
        a->ndim != 4 || b->ndim != 4 ||
        a->shape[0] != b->shape[0] || a->shape[2] != b->shape[2] || a->shape[3] != b->shape[3]) {
        return nullptr;
    }
    size_t N = a->shape[0], C1 = a->shape[1], C2 = b->shape[1];
    size_t H = a->shape[2], W  = a->shape[3], HW = H * W;
    RCRE_Tensor* out = rcre_tensor_alloc_4d(tape->arena, N, C1 + C2, H, W);
    if (!out) return nullptr;

    const float* ad = rcre_tensor_f32_data(a);
    const float* bd = rcre_tensor_f32_data(b);
    float* od = rcre_tensor_f32_data(out);
    for (size_t n = 0; n < N; ++n) {
        memcpy(od + (n * (C1 + C2)) * HW,      ad + n * C1 * HW, C1 * HW * sizeof(float));
        memcpy(od + (n * (C1 + C2) + C1) * HW, bd + n * C2 * HW, C2 * HW * sizeof(float));
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = a; node->inputs[1] = b; node->num_inputs = 2;
    node->backward = ag_backward_concat_channels;
    return out;
}

static void ag_backward_upsample2d_nearest(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    size_t scale = node->aux[0];
    size_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    size_t Ho = H * scale, Wo = W * scale;
    const float* og = out->grad;

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            float* xch = x->grad + (n * C + c) * H * W;
            const float* och = og + (n * C + c) * Ho * Wo;
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    float acc = 0.0f;
                    for (size_t dh = 0; dh < scale; ++dh) {
                        const float* row = och + (h * scale + dh) * Wo + w * scale;
                        for (size_t dw = 0; dw < scale; ++dw) acc += row[dw];
                    }
                    xch[h * W + w] += acc;
                }
            }
        }
    }
}

RCRE_Tensor* rcre_autograd_upsample2d_nearest(RCRE_TensorTape* tape, RCRE_Tensor* x, size_t scale) {
    if (!tape || !ag_is_f32_contig(x) || x->ndim != 4 || scale == 0) return nullptr;
    size_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    size_t Ho = H * scale, Wo = W * scale;
    RCRE_Tensor* out = rcre_tensor_alloc_4d(tape->arena, N, C, Ho, Wo);
    if (!out) return nullptr;

    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* xch = xd + (n * C + c) * H * W;
            float* och = od + (n * C + c) * Ho * Wo;
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    float val = xch[h * W + w];
                    for (size_t dh = 0; dh < scale; ++dh) {
                        float* row = och + (h * scale + dh) * Wo + w * scale;
                        for (size_t dw = 0; dw < scale; ++dw) row[dw] = val;
                    }
                }
            }
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->aux[0] = scale;
    node->backward = ag_backward_upsample2d_nearest;
    return out;
}

static void ag_backward_avgpool2d(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    size_t k = node->aux[0];
    size_t s = node->aux[1];
    size_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    size_t Ho = out->shape[2], Wo = out->shape[3];
    float inv_k2 = 1.0f / (float)(k * k);
    const float* og = out->grad;

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            float* xch = x->grad + (n * C + c) * H * W;
            const float* och = og + (n * C + c) * Ho * Wo;
            for (size_t ho = 0; ho < Ho; ++ho) {
                for (size_t wo = 0; wo < Wo; ++wo) {
                    float g = och[ho * Wo + wo] * inv_k2;
                    for (size_t kh = 0; kh < k; ++kh) {
                        float* row = xch + (ho * s + kh) * W + wo * s;
                        for (size_t kw = 0; kw < k; ++kw) row[kw] += g;
                    }
                }
            }
        }
    }
}

RCRE_Tensor* rcre_autograd_avgpool2d(RCRE_TensorTape* tape, RCRE_Tensor* x,
                                     size_t kernel_size, size_t stride) {
    if (!tape || !ag_is_f32_contig(x) || x->ndim != 4 || kernel_size == 0 || stride == 0) {
        return nullptr;
    }
    size_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    if (H < kernel_size || W < kernel_size) return nullptr;
    size_t Ho = (H - kernel_size) / stride + 1U;
    size_t Wo = (W - kernel_size) / stride + 1U;
    RCRE_Tensor* out = rcre_tensor_alloc_4d(tape->arena, N, C, Ho, Wo);
    if (!out) return nullptr;

    float inv_k2 = 1.0f / (float)(kernel_size * kernel_size);
    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* xch = xd + (n * C + c) * H * W;
            float* och = od + (n * C + c) * Ho * Wo;
            for (size_t ho = 0; ho < Ho; ++ho) {
                for (size_t wo = 0; wo < Wo; ++wo) {
                    float acc = 0.0f;
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        const float* row = xch + (ho * stride + kh) * W + wo * stride;
                        for (size_t kw = 0; kw < kernel_size; ++kw) acc += row[kw];
                    }
                    och[ho * Wo + wo] = acc * inv_k2;
                }
            }
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->aux[0] = kernel_size;
    node->aux[1] = stride;
    node->backward = ag_backward_avgpool2d;
    return out;
}

/* --- Vision Transformer / DiT: patchify, unpatchify, batched MHSA --- */

static void ag_backward_patchify(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* img = node->inputs[0];
    if (!img->grad) return;
    size_t P  = node->aux[0];
    size_t N  = img->shape[0], C = img->shape[1], H = img->shape[2], W = img->shape[3];
    size_t gh = H / P, gw = W / P, T = gh * gw, D = C * P * P;
    const float* og = out->grad;

    for (size_t n = 0; n < N; ++n) {
        for (size_t py = 0; py < gh; ++py) {
            for (size_t px = 0; px < gw; ++px) {
                size_t t = py * gw + px;
                const float* gtok = og + (n * T + t) * D;
                for (size_t c = 0; c < C; ++c) {
                    for (size_t dy = 0; dy < P; ++dy) {
                        for (size_t dx = 0; dx < P; ++dx) {
                            size_t img_idx = ((n * C + c) * H + (py * P + dy)) * W + (px * P + dx);
                            size_t tok_idx = (c * P + dy) * P + dx;
                            img->grad[img_idx] += gtok[tok_idx];
                        }
                    }
                }
            }
        }
    }
}

RCRE_Tensor* rcre_autograd_patchify(RCRE_TensorTape* tape, RCRE_Tensor* img, size_t patch_size) {
    if (!tape || !ag_is_f32_contig(img) || img->ndim != 4 || patch_size == 0) return nullptr;
    size_t N = img->shape[0], C = img->shape[1], H = img->shape[2], W = img->shape[3];
    if (H == 0 || W == 0 || (H % patch_size) != 0 || (W % patch_size) != 0) return nullptr;
    size_t gh = H / patch_size, gw = W / patch_size;
    size_t T  = gh * gw;
    size_t D  = C * patch_size * patch_size;

    RCRE_Tensor* out = rcre_tensor_alloc_3d(tape->arena, N, T, D);
    if (!out) return nullptr;

    const float* id = rcre_tensor_f32_data(img);
    float* od = rcre_tensor_f32_data(out);
    for (size_t n = 0; n < N; ++n) {
        for (size_t py = 0; py < gh; ++py) {
            for (size_t px = 0; px < gw; ++px) {
                size_t t = py * gw + px;
                float* otok = od + (n * T + t) * D;
                for (size_t c = 0; c < C; ++c) {
                    for (size_t dy = 0; dy < patch_size; ++dy) {
                        for (size_t dx = 0; dx < patch_size; ++dx) {
                            size_t img_idx = ((n * C + c) * H + (py * patch_size + dy)) * W + (px * patch_size + dx);
                            otok[(c * patch_size + dy) * patch_size + dx] = id[img_idx];
                        }
                    }
                }
            }
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = img; node->num_inputs = 1;
    node->aux[0] = patch_size;
    node->backward = ag_backward_patchify;
    return out;
}

static void ag_backward_unpatchify(RCRE_TensorNode* node) {
    RCRE_Tensor* out    = node->out;
    RCRE_Tensor* tokens = node->inputs[0];
    if (!tokens->grad) return;
    size_t P  = node->aux[0];
    size_t N  = out->shape[0], C = out->shape[1], H = out->shape[2], W = out->shape[3];
    size_t gh = H / P, gw = W / P, T = gh * gw, D = C * P * P;
    const float* og = out->grad;

    for (size_t n = 0; n < N; ++n) {
        for (size_t py = 0; py < gh; ++py) {
            for (size_t px = 0; px < gw; ++px) {
                size_t t = py * gw + px;
                float* gtok = tokens->grad + (n * T + t) * D;
                for (size_t c = 0; c < C; ++c) {
                    for (size_t dy = 0; dy < P; ++dy) {
                        for (size_t dx = 0; dx < P; ++dx) {
                            size_t img_idx = ((n * C + c) * H + (py * P + dy)) * W + (px * P + dx);
                            gtok[(c * P + dy) * P + dx] += og[img_idx];
                        }
                    }
                }
            }
        }
    }
}

RCRE_Tensor* rcre_autograd_unpatchify(RCRE_TensorTape* tape, RCRE_Tensor* tokens,
                                      size_t channels, size_t height, size_t width, size_t patch_size) {
    if (!tape || !ag_is_f32_contig(tokens) || tokens->ndim != 3 ||
        channels == 0 || height == 0 || width == 0 || patch_size == 0 ||
        (height % patch_size) != 0 || (width % patch_size) != 0) {
        return nullptr;
    }
    size_t gh = height / patch_size, gw = width / patch_size;
    size_t T  = gh * gw;
    size_t D  = channels * patch_size * patch_size;
    if (tokens->shape[1] != T || tokens->shape[2] != D) return nullptr;

    size_t N = tokens->shape[0];
    RCRE_Tensor* out = rcre_tensor_alloc_4d(tape->arena, N, channels, height, width);
    if (!out) return nullptr;

    const float* td = rcre_tensor_f32_data(tokens);
    float* od = rcre_tensor_f32_data(out);
    for (size_t n = 0; n < N; ++n) {
        for (size_t py = 0; py < gh; ++py) {
            for (size_t px = 0; px < gw; ++px) {
                size_t t = py * gw + px;
                const float* itok = td + (n * T + t) * D;
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t dy = 0; dy < patch_size; ++dy) {
                        for (size_t dx = 0; dx < patch_size; ++dx) {
                            size_t img_idx = ((n * channels + c) * height + (py * patch_size + dy)) * width + (px * patch_size + dx);
                            od[img_idx] = itok[(c * patch_size + dy) * patch_size + dx];
                        }
                    }
                }
            }
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = tokens; node->num_inputs = 1;
    node->aux[0] = patch_size;
    node->backward = ag_backward_unpatchify;
    return out;
}

static void ag_backward_mhsa(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* q   = node->inputs[0];
    RCRE_Tensor* k   = node->inputs[1];
    RCRE_Tensor* v   = node->inputs[2];
    size_t H = node->aux[0];
    float* workspace = (float*)(uintptr_t)node->aux[1];

    size_t B  = q->shape[0];
    size_t T  = q->shape[1];
    size_t C  = q->shape[2];
    size_t Dh = C / H;
    float scale = 1.0f / sqrtf((float)Dh);

    const float* qd = rcre_tensor_f32_data(q);
    const float* kd = rcre_tensor_f32_data(k);
    const float* vd = rcre_tensor_f32_data(v);
    const float* og = out->grad;

    float* probs = workspace;
    float* dP    = workspace + B * H * T * T;

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            size_t h_off = h * Dh;
            const float* P_bh = probs + (b * H + h) * T * T;

            for (size_t i = 0; i < T; ++i) {
                const float* p_row = P_bh + i * T;
                const float* do_i  = og + (b * T + i) * C + h_off;
                const float* q_i   = qd + (b * T + i) * C + h_off;
                float* dq_i = q->grad ? (q->grad + (b * T + i) * C + h_off) : nullptr;

                float row_dot = 0.0f;
                for (size_t j = 0; j < T; ++j) {
                    const float* v_j = vd + (b * T + j) * C + h_off;
                    float* dv_j = v->grad ? (v->grad + (b * T + j) * C + h_off) : nullptr;
                    float pij = p_row[j];
                    float dp = 0.0f;
                    for (size_t d = 0; d < Dh; ++d) {
                        dp += do_i[d] * v_j[d];
                        if (dv_j) dv_j[d] += pij * do_i[d];
                    }
                    dP[j] = dp;
                    row_dot += pij * dp;
                }

                for (size_t j = 0; j < T; ++j) {
                    float dS = p_row[j] * (dP[j] - row_dot) * scale;
                    const float* k_j = kd + (b * T + j) * C + h_off;
                    float* dk_j = k->grad ? (k->grad + (b * T + j) * C + h_off) : nullptr;
                    for (size_t d = 0; d < Dh; ++d) {
                        if (dq_i) dq_i[d] += dS * k_j[d];
                        if (dk_j) dk_j[d] += dS * q_i[d];
                    }
                }
            }
        }
    }
}

RCRE_Tensor* rcre_autograd_mhsa(RCRE_TensorTape* tape, RCRE_Tensor* q, RCRE_Tensor* k,
                                RCRE_Tensor* v, size_t num_heads) {
    if (!tape || !ag_is_f32_contig(q) || !ag_is_f32_contig(k) || !ag_is_f32_contig(v) ||
        q->ndim != 3 || k->ndim != 3 || v->ndim != 3 || num_heads == 0 ||
        q->shape[0] != k->shape[0] || q->shape[0] != v->shape[0] ||
        q->shape[1] != k->shape[1] || q->shape[1] != v->shape[1] ||
        q->shape[2] != k->shape[2] || q->shape[2] != v->shape[2] ||
        q->shape[2] == 0 || (q->shape[2] % num_heads) != 0) {
        return nullptr;
    }
    size_t B  = q->shape[0];
    size_t T  = q->shape[1];
    size_t C  = q->shape[2];
    size_t Dh = C / num_heads;
    float scale = 1.0f / sqrtf((float)Dh);

    RCRE_Tensor* out = rcre_tensor_alloc_3d(tape->arena, B, T, C);
    if (!out) return nullptr;

    /* Zero-malloc arena workspace: [B * H * T * T] attention weights + [T] backward dP buffer */
    size_t ws_elems = B * num_heads * T * T + T;
    float* workspace = (float*)rcre_arena_alloc(tape->arena, ws_elems * sizeof(float), 64);
    if (!workspace) return nullptr;

    const float* qd = rcre_tensor_f32_data(q);
    const float* kd = rcre_tensor_f32_data(k);
    const float* vd = rcre_tensor_f32_data(v);
    float* od = rcre_tensor_f32_data(out);

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            size_t h_off = h * Dh;
            float* P_bh = workspace + (b * num_heads + h) * T * T;

            for (size_t i = 0; i < T; ++i) {
                const float* q_i = qd + (b * T + i) * C + h_off;
                float* p_row = P_bh + i * T;

                float mx = -1e30f;
                for (size_t j = 0; j < T; ++j) {
                    const float* k_j = kd + (b * T + j) * C + h_off;
                    float s = 0.0f;
                    for (size_t d = 0; d < Dh; ++d) s += q_i[d] * k_j[d];
                    s *= scale;
                    p_row[j] = s;
                    if (s > mx) mx = s;
                }

                float sum = 0.0f;
                for (size_t j = 0; j < T; ++j) {
                    p_row[j] = expf(p_row[j] - mx);
                    sum += p_row[j];
                }
                float inv_sum = 1.0f / sum;
                for (size_t j = 0; j < T; ++j) p_row[j] *= inv_sum;

                float* o_i = od + (b * T + i) * C + h_off;
                for (size_t d = 0; d < Dh; ++d) o_i[d] = 0.0f;
                for (size_t j = 0; j < T; ++j) {
                    float pij = p_row[j];
                    const float* v_j = vd + (b * T + j) * C + h_off;
                    for (size_t d = 0; d < Dh; ++d) o_i[d] += pij * v_j[d];
                }
            }
        }
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out        = out;
    node->inputs[0]  = q;
    node->inputs[1]  = k;
    node->inputs[2]  = v;
    node->num_inputs = 3;
    node->aux[0]     = num_heads;
    node->aux[1]     = (size_t)(uintptr_t)workspace;
    node->backward   = ag_backward_mhsa;
    return out;
}

/* --- VAE Reparameterization Trick: z = mu + eps * exp(0.5 * clamp(logvar, -20, 20)) --- */

static void ag_backward_reparameterize(RCRE_TensorNode* node) {
    RCRE_Tensor* out    = node->out;
    RCRE_Tensor* mu     = node->inputs[0];
    RCRE_Tensor* logvar = node->inputs[1];
    RCRE_Tensor* eps    = node->inputs[2];

    const float* lvd = rcre_tensor_f32_data(logvar);
    const float* epd = rcre_tensor_f32_data(eps);
    const float* og  = out->grad;

    for (size_t i = 0; i < out->numel; ++i) {
        float raw_lv = lvd[i];
        float lv = raw_lv;
        if (lv < -20.0f) lv = -20.0f;
        if (lv >  20.0f) lv =  20.0f;
        float std = expf(0.5f * lv);
        float g   = og[i];
        if (mu->grad) mu->grad[i] += g;
        if (logvar->grad && raw_lv > -20.0f && raw_lv < 20.0f) {
            logvar->grad[i] += g * epd[i] * 0.5f * std;
        }
        if (eps->grad) eps->grad[i] += g * std;
    }
}

RCRE_Tensor* rcre_autograd_reparameterize(RCRE_TensorTape* tape, RCRE_Tensor* mu,
                                          RCRE_Tensor* logvar, RCRE_Tensor* eps) {
    if (!tape || !ag_is_f32_contig(mu) || !ag_is_f32_contig(logvar) || !ag_is_f32_contig(eps) ||
        mu->numel != logvar->numel || mu->numel != eps->numel) {
        return nullptr;
    }
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, mu->shape, mu->ndim);
    if (!out) return nullptr;

    const float* md  = rcre_tensor_f32_data(mu);
    const float* lvd = rcre_tensor_f32_data(logvar);
    const float* epd = rcre_tensor_f32_data(eps);
    float* od = rcre_tensor_f32_data(out);

    for (size_t i = 0; i < out->numel; ++i) {
        float lv = lvd[i];
        if (lv < -20.0f) lv = -20.0f;
        if (lv >  20.0f) lv =  20.0f;
        od[i] = md[i] + epd[i] * expf(0.5f * lv);
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out;
    node->inputs[0] = mu;
    node->inputs[1] = logvar;
    node->inputs[2] = eps;
    node->num_inputs = 3;
    node->backward = ag_backward_reparameterize;
    return out;
}

/* --- ViT/DiT Helpers: add_pos_emb_3d, mean_pool_seq, embed_lookup --- */

static void ag_backward_add_pos_emb_3d(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    RCRE_Tensor* pos = node->inputs[1];
    size_t B = x->shape[0], TC = x->shape[1] * x->shape[2];
    const float* og = out->grad;

    for (size_t b = 0; b < B; ++b) {
        const float* gb = og + b * TC;
        float* xb = x->grad ? (x->grad + b * TC) : nullptr;
        for (size_t i = 0; i < TC; ++i) {
            if (xb) xb[i] += gb[i];
            if (pos->grad) pos->grad[i] += gb[i];
        }
    }
}

RCRE_Tensor* rcre_autograd_add_pos_emb_3d(RCRE_TensorTape* tape, RCRE_Tensor* x, RCRE_Tensor* pos_emb) {
    if (!tape || !ag_is_f32_contig(x) || !ag_is_f32_contig(pos_emb) ||
        x->ndim != 3 || pos_emb->ndim != 2 ||
        pos_emb->shape[0] != x->shape[1] || pos_emb->shape[1] != x->shape[2]) {
        return nullptr;
    }
    size_t B = x->shape[0], TC = x->shape[1] * x->shape[2];
    RCRE_Tensor* out = rcre_tensor_alloc(tape->arena, x->shape, 3);
    if (!out) return nullptr;

    const float* xd = rcre_tensor_f32_data(x);
    const float* pd = rcre_tensor_f32_data(pos_emb);
    float* od = rcre_tensor_f32_data(out);
    for (size_t b = 0; b < B; ++b) {
        const float* xb = xd + b * TC;
        float* ob = od + b * TC;
        for (size_t i = 0; i < TC; ++i) ob[i] = xb[i] + pd[i];
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->inputs[1] = pos_emb; node->num_inputs = 2;
    node->backward = ag_backward_add_pos_emb_3d;
    return out;
}

static void ag_backward_mean_pool_seq(RCRE_TensorNode* node) {
    RCRE_Tensor* out = node->out;
    RCRE_Tensor* x   = node->inputs[0];
    if (!x->grad) return;
    size_t B = x->shape[0], T = x->shape[1], C = x->shape[2];
    float inv_T = 1.0f / (float)T;
    const float* og = out->grad;

    for (size_t b = 0; b < B; ++b) {
        const float* gb = og + b * C;
        for (size_t t = 0; t < T; ++t) {
            float* xbt = x->grad + (b * T + t) * C;
            for (size_t c = 0; c < C; ++c) xbt[c] += gb[c] * inv_T;
        }
    }
}

RCRE_Tensor* rcre_autograd_mean_pool_seq(RCRE_TensorTape* tape, RCRE_Tensor* x) {
    if (!tape || !ag_is_f32_contig(x) || x->ndim != 3 || x->shape[1] == 0) return nullptr;
    size_t B = x->shape[0], T = x->shape[1], C = x->shape[2];
    RCRE_Tensor* out = rcre_tensor_alloc_2d(tape->arena, B, C);
    if (!out) return nullptr;

    float inv_T = 1.0f / (float)T;
    const float* xd = rcre_tensor_f32_data(x);
    float* od = rcre_tensor_f32_data(out);
    for (size_t b = 0; b < B; ++b) {
        float* ob = od + b * C;
        for (size_t c = 0; c < C; ++c) ob[c] = 0.0f;
        for (size_t t = 0; t < T; ++t) {
            const float* xbt = xd + (b * T + t) * C;
            for (size_t c = 0; c < C; ++c) ob[c] += xbt[c];
        }
        for (size_t c = 0; c < C; ++c) ob[c] *= inv_T;
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = x; node->num_inputs = 1;
    node->backward = ag_backward_mean_pool_seq;
    return out;
}

static void ag_backward_embed_lookup(RCRE_TensorNode* node) {
    RCRE_Tensor* out   = node->out;
    RCRE_Tensor* table = node->inputs[0];
    if (!table->grad) return;
    size_t B = out->shape[0], D = out->shape[1];
    const size_t* idx_copy = (const size_t*)(uintptr_t)node->aux[0];
    const float* og = out->grad;

    for (size_t b = 0; b < B; ++b) {
        size_t row = idx_copy[b];
        float* tg = table->grad + row * D;
        const float* gb = og + b * D;
        for (size_t d = 0; d < D; ++d) tg[d] += gb[d];
    }
}

RCRE_Tensor* rcre_autograd_embed_lookup(RCRE_TensorTape* tape, RCRE_Tensor* table,
                                        const size_t* indices, size_t batch_size) {
    if (!tape || !ag_is_f32_contig(table) || table->ndim != 2 || !indices || batch_size == 0) {
        return nullptr;
    }
    size_t V = table->shape[0], D = table->shape[1];
    for (size_t b = 0; b < batch_size; ++b) {
        if (indices[b] >= V) return nullptr;
    }

    RCRE_Tensor* out = rcre_tensor_alloc_2d(tape->arena, batch_size, D);
    if (!out) return nullptr;

    size_t* idx_copy = (size_t*)rcre_arena_alloc(tape->arena, batch_size * sizeof(size_t), alignof(size_t));
    if (!idx_copy) return nullptr;
    memcpy(idx_copy, indices, batch_size * sizeof(size_t));

    const float* td = rcre_tensor_f32_data(table);
    float* od = rcre_tensor_f32_data(out);
    for (size_t b = 0; b < batch_size; ++b) {
        memcpy(od + b * D, td + indices[b] * D, D * sizeof(float));
    }

    RCRE_TensorNode* node = tape_push_node(tape);
    if (!node) return out;
    node->out = out; node->inputs[0] = table; node->num_inputs = 1;
    node->aux[0] = (size_t)(uintptr_t)idx_copy;
    node->backward = ag_backward_embed_lookup;
    return out;
}
