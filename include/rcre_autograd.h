/**
 * @file rcre_autograd.h
 * @brief RCRE Unified Automatic Differentiation & Deep Learning Engine
 * 
 * Part of Recep's Code Run Everywhere (RCRE).
 * Strict ISO C23 compliant, zero external dependencies.
 *
 * Unified tensor model (v0.2.2): there is exactly ONE tensor type —
 * RCRE_Tensor (declared in rcre.h). Training tensors carry an optional
 * `grad` backprop buffer; the old dual-type model (RCRE_AGTensor +
 * rcre_ag_to/from_tensor bridge) has been removed. Autograd operators
 * require RCRE_DTYPE_F32 tensors and return NULL (no node recorded) on
 * any dtype/shape violation.
 */

#ifndef RCRE_AUTOGRAD_H
#define RCRE_AUTOGRAD_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Scalar Reverse-Mode Automatic Differentiation (DAG / Micrograd-style)   */
/* ========================================================================= */

typedef enum : uint8_t {
    RCRE_SCALAR_OP_CONST = 0,
    RCRE_SCALAR_OP_PARAM,
    RCRE_SCALAR_OP_ADD,
    RCRE_SCALAR_OP_SUB,
    RCRE_SCALAR_OP_MUL,
    RCRE_SCALAR_OP_DIV,
    RCRE_SCALAR_OP_POW,
    RCRE_SCALAR_OP_RELU,
    RCRE_SCALAR_OP_TANH,
    RCRE_SCALAR_OP_SIGMOID,
    RCRE_SCALAR_OP_EXP,
    RCRE_SCALAR_OP_LOG,
    RCRE_SCALAR_OP_NEG
} RCRE_ScalarOp;

typedef struct RCRE_ScalarVal RCRE_ScalarVal;

struct RCRE_ScalarVal {
    double data;
    double grad;
    void (*backward)(RCRE_ScalarVal* self);
    RCRE_ScalarVal* prev[2];
    RCRE_ScalarOp op;
    const char* label;
    uint32_t visited_id;
};

/* Scalar Value Constructors */
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_val_new(RCRE_GraphCtx* ctx, double data, const char* label);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_val_param(RCRE_GraphCtx* ctx, double data, const char* label);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_val_const(RCRE_GraphCtx* ctx, double data);

/* Scalar Elementary & Transcendental Operators */
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_add(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a, RCRE_ScalarVal* b);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_sub(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a, RCRE_ScalarVal* b);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_mul(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a, RCRE_ScalarVal* b);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_div(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a, RCRE_ScalarVal* b);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_pow(RCRE_GraphCtx* ctx, RCRE_ScalarVal* base, RCRE_ScalarVal* exponent);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_relu(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_tanh(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_sigmoid(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_exp(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_log(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_neg(RCRE_GraphCtx* ctx, RCRE_ScalarVal* a);

/* Scalar Reverse-Mode Execution */
/* Topological sort is iterative (explicit stack): chains of arbitrary depth
 * are safe; a DAG must be acyclic (re-entering an in-progress node aborts). */
void rcre_scalar_backward(RCRE_ScalarVal* root);
void rcre_scalar_zero_grad(RCRE_ScalarVal* root);
bool rcre_scalar_export_dot(RCRE_ScalarVal* root, const char* filename);

/* ========================================================================= */
/* 2. Training Tensors (arena-owned RCRE_Tensor with backprop buffer)         */
/* ========================================================================= */

/*
 * Unified tensor model: training tensors are ordinary RCRE_Tensor values
 * whose `data` holds F32 activations and whose `grad` buffer receives
 * backprop results. `rcre_tensor_alloc` returns a pointer (arena-owned);
 * the same struct layout works by value for core (inference) tensors, so
 * core views (slice/reshape/permute) can feed autograd operators directly.
 *
 * Arena-first API: every allocation is served from the caller-provided arena
 * (NULL is rejected). Objects are freed collectively by rcre_arena_reset() /
 * rcre_arena_destroy(), or by rewinding with rcre_arena_temp_begin/end().
 */
RCRE_NODISCARD RCRE_Tensor* rcre_tensor_alloc(RCRE_Arena* arena, const size_t* shape, uint8_t ndim);
RCRE_NODISCARD RCRE_Tensor* rcre_tensor_alloc_1d(RCRE_Arena* arena, size_t d0);
RCRE_NODISCARD RCRE_Tensor* rcre_tensor_alloc_2d(RCRE_Arena* arena, size_t rows, size_t cols);
RCRE_NODISCARD RCRE_Tensor* rcre_tensor_alloc_4d(RCRE_Arena* arena, size_t n, size_t c, size_t h, size_t w);

/* Initialization Routines (require F32; no-op otherwise) */
void rcre_tensor_fill(RCRE_Tensor* t, float val);
void rcre_tensor_zeros(RCRE_Tensor* t);
void rcre_tensor_ones(RCRE_Tensor* t);
void rcre_tensor_init_xavier(RCRE_Tensor* t, size_t fan_in, size_t fan_out, uint64_t* rng);
void rcre_tensor_init_he(RCRE_Tensor* t, size_t fan_in, uint64_t* rng);
void rcre_tensor_zero_grad(RCRE_Tensor* t);

/* View & Shape Manipulations */
/* Allocates a new arena struct referencing the source memory (O(1)). */
RCRE_NODISCARD RCRE_Tensor* rcre_tensor_view(RCRE_Arena* arena, const RCRE_Tensor* src);
/* As rcre_tensor_view with a new shape; requires contiguous source.
 * (Distinct from the core value-returning rcre_tensor_reshape in rcre.h.) */
RCRE_NODISCARD RCRE_Tensor* rcre_tensor_reshape_view(RCRE_Arena* arena, const RCRE_Tensor* src, const size_t* new_shape, uint8_t new_ndim);
void rcre_tensor_print(const RCRE_Tensor* t, const char* name);

/* Basic Tensor Algebraic Kernels (non-differentiable, F32 only) */
RCRE_NODISCARD RCRE_Tensor* rcre_tensor_matmul(RCRE_Arena* arena, const RCRE_Tensor* a, const RCRE_Tensor* b);
void rcre_tensor_add_inplace(RCRE_Tensor* a, const RCRE_Tensor* b);

/* ========================================================================= */
/* 3. Reverse-Mode Dynamic Tensor Tape Autograd Engine                       */
/* ========================================================================= */

typedef struct RCRE_TensorNode RCRE_TensorNode;

struct RCRE_TensorNode {
    RCRE_Tensor*    out;
    RCRE_Tensor*    inputs[3];
    size_t          num_inputs;
    void (*backward)(RCRE_TensorNode* node);
};

typedef struct {
    RCRE_TensorNode* nodes;
    size_t           count;
    size_t           capacity;
    RCRE_Arena*      arena;
} RCRE_TensorTape;

RCRE_NODISCARD RCRE_TensorTape* rcre_tensor_tape_create(RCRE_Arena* arena, size_t initial_cap);
void rcre_tensor_tape_reset(RCRE_TensorTape* tape);
/* Frees the tape's heap-owned node array. The tape struct itself lives in the
 * arena and is released by the arena — call destroy BEFORE any
 * rcre_arena_reset/re-allocation of the same arena region. */
void rcre_tensor_tape_destroy(RCRE_TensorTape* tape);
void rcre_tensor_tape_backward(RCRE_TensorTape* tape);

/*
 * Tape + arena rewinding contract: the tape's node storage is heap-owned, so
 * the documented pattern is safe —
 *   1. create the tape (before taking any temp checkpoint),
 *   2. per batch: temp_begin -> ops -> tape_reset -> temp_end,
 * even when the node array grows mid-batch.
 */

/*
 * Differentiable Tape Operators (F32 only).
 * Validation: matmul requires 2-D inputs with matching inner dims;
 * add requires equal numel; add_bias requires x: (R, C), bias: (C).
 * Any violation (including non-F32 inputs) returns NULL and records no node.
 * The recorded nodes reference the caller's tensor structs, which must stay
 * alive until rcre_tensor_tape_backward() has run.
 */
RCRE_NODISCARD RCRE_Tensor* rcre_autograd_matmul(RCRE_TensorTape* tape, RCRE_Tensor* a, RCRE_Tensor* b);
RCRE_NODISCARD RCRE_Tensor* rcre_autograd_add(RCRE_TensorTape* tape, RCRE_Tensor* a, RCRE_Tensor* b);
RCRE_NODISCARD RCRE_Tensor* rcre_autograd_add_bias(RCRE_TensorTape* tape, RCRE_Tensor* x, RCRE_Tensor* bias);
RCRE_NODISCARD RCRE_Tensor* rcre_autograd_relu(RCRE_TensorTape* tape, RCRE_Tensor* x);
RCRE_NODISCARD RCRE_Tensor* rcre_autograd_sigmoid(RCRE_TensorTape* tape, RCRE_Tensor* x);

/* ========================================================================= */
/* 4. Loss Functions & Gradients                                             */
/* ========================================================================= */

/*
 * Tensor-based Loss Functions (F32).
 *
 * Gradient contract: the analytic gradient is written to grad_out->grad.
 * For tensors produced by rcre_tensor_alloc this buffer always exists; for
 * borrowed core views (grad == NULL) only the scalar loss value is
 * computed. Targets/data are never modified by loss functions.
 *
 * rcre_loss_cross_entropy_logits: `targets` must contain class indices in
 * [0, num_classes). Out-of-range indices are clamped (and reported to
 * stderr when RCRE_DEBUG is defined) — validate labels with
 * rcre_data_validate_labels() before training.
 */
float rcre_loss_mse(const RCRE_Tensor* preds, const RCRE_Tensor* targets, RCRE_Tensor* grad_out);
float rcre_loss_bce(const RCRE_Tensor* preds, const RCRE_Tensor* targets, RCRE_Tensor* grad_out);
float rcre_loss_cross_entropy_logits(const RCRE_Tensor* logits, const size_t* targets, size_t batch_size, size_t num_classes, RCRE_Tensor* grad_out);
float rcre_loss_kl_divergence(const RCRE_Tensor* mu, const RCRE_Tensor* logvar, RCRE_Tensor* grad_mu, RCRE_Tensor* grad_logvar);

/*
 * Returns the index of the first target outside [0, num_classes), or -1 if
 * all labels are valid. Cheap dataset hygiene check before training.
 */
int rcre_data_validate_labels(const size_t* targets, size_t n, size_t num_classes);

/* Scalar-based Loss Functions */
/*
 * Note: the scalar DAG engine keeps a per-thread visited-epoch counter
 * (thread_local); a single DAG may be walked from any one thread. The
 * tensor-tape engine is the recommended path in multi-threaded contexts.
 */
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_loss_mse(RCRE_GraphCtx* ctx, RCRE_ScalarVal** preds, const double* targets, size_t count);
RCRE_NODISCARD RCRE_ScalarVal* rcre_scalar_loss_bce(RCRE_GraphCtx* ctx, RCRE_ScalarVal** preds, const double* targets, size_t count);

/* ========================================================================= */
/* 5. Optimizers: SGD w/ Momentum & Decoupled Weight Decay AdamW             */
/* ========================================================================= */

typedef enum : uint8_t {
    RCRE_OPTIM_SGD = 0,
    RCRE_OPTIM_ADAMW
} RCRE_OptimType;

typedef struct {
    RCRE_OptimType type;
    RCRE_Tensor** params;
    size_t         num_params;
    float          lr;
    float          weight_decay;

    /* SGD momentum */
    float          momentum;
    float**        v_momentum;

    /* AdamW moments */
    float          beta1;
    float          beta2;
    float          eps;
    size_t         step_count;
    float**        m; /* 1st moment */
    float**        v; /* 2nd moment */
} RCRE_TensorOptim;

RCRE_NODISCARD RCRE_TensorOptim* rcre_tensor_optim_sgd_create(RCRE_Arena* arena, RCRE_Tensor** params, size_t num_params, float lr, float momentum, float weight_decay);
RCRE_NODISCARD RCRE_TensorOptim* rcre_tensor_optim_adamw_create(RCRE_Arena* arena, RCRE_Tensor** params, size_t num_params, float lr, float beta1, float beta2, float eps, float weight_decay);
/* Both create() calls validate hyperparameters and return NULL on invalid
 * input: lr > 0, eps > 0, 0 <= momentum < 1, 0 <= beta1 < 1, 0 <= beta2 < 1
 * (beta == 1 would divide by zero in the bias correction). */
void rcre_tensor_optim_step(RCRE_TensorOptim* opt);
void rcre_tensor_optim_zero_grad(RCRE_TensorOptim* opt);

/* Scalar Autograd Optimizer */
typedef struct {
    RCRE_OptimType   type;
    RCRE_ScalarVal** params;
    size_t           num_params;
    double           lr;
    double           weight_decay;
    double           momentum;
    double*          v_momentum;
    double           beta1;
    double           beta2;
    double           eps;
    size_t           step_count;
    double*          m;
    double*          v;
} RCRE_ScalarOptim;

RCRE_NODISCARD RCRE_ScalarOptim* rcre_scalar_optim_adamw_create(RCRE_Arena* arena, RCRE_ScalarVal** params, size_t num_params, double lr, double beta1, double beta2, double eps, double weight_decay);
void rcre_scalar_optim_step(RCRE_ScalarOptim* opt);
void rcre_scalar_optim_zero_grad(RCRE_ScalarOptim* opt);

/* ========================================================================= */
/* 6. Data Ingestion, Streaming CSV Parser & Scalers                         */
/* ========================================================================= */

typedef struct {
    float* mean;
    float* std;
    size_t num_features;
} RCRE_StandardScaler;

typedef struct {
    float* min_val;
    float* max_val;
    size_t num_features;
} RCRE_MinMaxScaler;

typedef struct {
    RCRE_Tensor*   X;
    RCRE_Tensor*   y;
    size_t         y_cols; /* 1 for 1-D y, y->shape[1] for 2-D y */
    size_t         batch_size;
    size_t         current_idx;
    size_t         num_samples;
    bool           shuffle;
    uint64_t       rng_state;
    size_t*        indices;
} RCRE_BatchIter;

/*
 * Streaming line-based CSV parser (comma/tab separated). Rows longer than any
 * fixed buffer are read intact (dynamic line growth). Every row must contain
 * exactly the column count of the first data row; ragged input is rejected
 * (returns NULL) rather than silently zero-filled. Returns an arena-owned
 * 2-D F32 RCRE_Tensor (struct included), or NULL on error.
 */
RCRE_NODISCARD RCRE_Tensor* rcre_data_load_csv(RCRE_Arena* arena, const char* filepath, bool skip_header, size_t max_rows);

RCRE_NODISCARD RCRE_StandardScaler* rcre_standard_scaler_fit(RCRE_Arena* arena, const RCRE_Tensor* X);
void rcre_standard_scaler_transform(const RCRE_StandardScaler* scaler, RCRE_Tensor* X);
void rcre_standard_scaler_inverse_transform(const RCRE_StandardScaler* scaler, RCRE_Tensor* X);

RCRE_NODISCARD RCRE_MinMaxScaler* rcre_minmax_scaler_fit(RCRE_Arena* arena, const RCRE_Tensor* X);
void rcre_minmax_scaler_transform(const RCRE_MinMaxScaler* scaler, RCRE_Tensor* X);

/* X must be 2-D; y may be 1-D (n,) or 2-D (n, k). A 1-D y is treated as
 * (n, 1); all outputs are 2-D. Returns silently no-op on invalid input. */
void rcre_data_train_test_split(RCRE_Arena* arena, const RCRE_Tensor* X, const RCRE_Tensor* y,
                               float train_ratio,
                               RCRE_Tensor** X_train, RCRE_Tensor** y_train,
                               RCRE_Tensor** X_test, RCRE_Tensor** y_test);

/* X must be 2-D; y may be 1-D (n,) or 2-D (n, k). */
RCRE_NODISCARD RCRE_BatchIter* rcre_batch_iter_create(RCRE_Arena* arena, RCRE_Tensor* X, RCRE_Tensor* y, size_t batch_size, bool shuffle);
/* Copies one batch into caller-provided tensors. Returns false (and copies
 * nothing) at epoch end OR when batch_X/batch_y are too small. */
bool rcre_batch_iter_next(RCRE_BatchIter* iter, RCRE_Tensor* batch_X, RCRE_Tensor* batch_y);
void rcre_batch_iter_reset(RCRE_BatchIter* iter);

/* ========================================================================= */
/* 7. Visualizations: Terminal Loss Curves, Confusion Matrix & 3D WebGL      */
/* ========================================================================= */

void rcre_vis_loss_history(const float* loss_history, size_t count, const char* title);
void rcre_vis_confusion_matrix(const size_t* confusion_matrix, size_t num_classes, const char* const* class_names);
bool rcre_vis_export_svg_curve(const float* y_vals, size_t count, const char* filepath, const char* title);
bool rcre_vis_export_3d_clusters_html(const RCRE_Tensor* points, const size_t* assignments, const RCRE_Tensor* centroids, size_t k, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_AUTOGRAD_H */
