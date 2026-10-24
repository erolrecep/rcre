/**
 * @file rcre_autograd.h
 * @brief RCRE Unified Automatic Differentiation & Deep Learning Engine
 * 
 * Part of Recep's Code Run Everywhere (RCRE).
 * Strict ISO C23 compliant, zero external dependencies.
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
void rcre_scalar_backward(RCRE_ScalarVal* root);
void rcre_scalar_zero_grad(RCRE_ScalarVal* root);
bool rcre_scalar_export_dot(RCRE_ScalarVal* root, const char* filename);

/* ========================================================================= */
/* 2. Autograd Tensor Primitives & Operations                                */
/* ========================================================================= */

#define RCRE_AG_MAX_DIMS 4

typedef struct RCRE_AGTensor {
    float*      data;
    float*      grad;
    size_t      shape[RCRE_AG_MAX_DIMS];
    size_t      strides[RCRE_AG_MAX_DIMS];
    size_t      ndim;
    size_t      numel;
    bool        is_view;
    const char* label;
} RCRE_AGTensor;

typedef RCRE_AGTensor RCRE_TensorAG;

/* Allocation & Tensor Creation */
/*
 * Arena-first API: every allocation is served from the caller-provided arena
 * (NULL is rejected). Objects are freed collectively by rcre_arena_reset() /
 * rcre_arena_destroy(), or by rewinding with rcre_arena_temp_begin/end().
 */
RCRE_NODISCARD RCRE_AGTensor* rcre_ag_tensor_create(RCRE_Arena* arena, const size_t* shape, size_t ndim);
RCRE_NODISCARD RCRE_AGTensor* rcre_ag_tensor_create_1d(RCRE_Arena* arena, size_t d0);
RCRE_NODISCARD RCRE_AGTensor* rcre_ag_tensor_create_2d(RCRE_Arena* arena, size_t rows, size_t cols);
RCRE_NODISCARD RCRE_AGTensor* rcre_ag_tensor_create_4d(RCRE_Arena* arena, size_t n, size_t c, size_t h, size_t w);

/* Interoperability with core RCRE_Tensor */
/* rcre_ag_from_tensor accepts F32 tensors only; for any other dtype it
 * returns a zeroed, empty struct (ag.data == NULL) — check before use. */
RCRE_NODISCARD RCRE_Tensor   rcre_ag_to_tensor(const RCRE_AGTensor* ag);
RCRE_NODISCARD RCRE_AGTensor rcre_ag_from_tensor(const RCRE_Tensor* t, float* grad, const char* label);

/* Initialization Routines */
void rcre_ag_tensor_fill(RCRE_AGTensor* t, float val);
void rcre_ag_tensor_zeros(RCRE_AGTensor* t);
void rcre_ag_tensor_ones(RCRE_AGTensor* t);
void rcre_ag_tensor_init_xavier(RCRE_AGTensor* t, size_t fan_in, size_t fan_out, uint64_t* rng);
void rcre_ag_tensor_init_he(RCRE_AGTensor* t, size_t fan_in, uint64_t* rng);
void rcre_ag_tensor_zero_grad(RCRE_AGTensor* t);

/* View & Shape Manipulations */
RCRE_NODISCARD RCRE_AGTensor* rcre_ag_tensor_view(RCRE_Arena* arena, const RCRE_AGTensor* src);
RCRE_NODISCARD RCRE_AGTensor* rcre_ag_tensor_reshape(RCRE_Arena* arena, const RCRE_AGTensor* src, const size_t* new_shape, size_t new_ndim);
void rcre_ag_tensor_print(const RCRE_AGTensor* t, const char* name);

/* Basic Tensor Algebraic Kernels */
RCRE_NODISCARD RCRE_AGTensor* rcre_ag_tensor_matmul(RCRE_Arena* arena, const RCRE_AGTensor* a, const RCRE_AGTensor* b);
void rcre_ag_tensor_add_inplace(RCRE_AGTensor* a, const RCRE_AGTensor* b);

/* ========================================================================= */
/* 3. Reverse-Mode Dynamic Tensor Tape Autograd Engine                       */
/* ========================================================================= */

typedef struct RCRE_TensorNode RCRE_TensorNode;

struct RCRE_TensorNode {
    RCRE_AGTensor*  out;
    RCRE_AGTensor*  inputs[3];
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
 * arena and is released by the arena. Always call after create. */
void rcre_tensor_tape_destroy(RCRE_TensorTape* tape);
void rcre_tensor_tape_backward(RCRE_TensorTape* tape);

/*
 * Tape + arena rewinding contract: the tape's node storage is heap-owned, so
 * the documented pattern is safe —
 *   1. create the tape (before taking any temp checkpoint),
 *   2. per batch: temp_begin -> ops -> tape_reset -> temp_end,
 * even when the node array grows mid-batch.
 */

/* Differentiable Tape Operators */
/* Validation: matmul requires 2-D inputs with matching inner dims;
 * add requires equal numel; add_bias requires x: (R, C), bias: (C). */
RCRE_NODISCARD RCRE_AGTensor* rcre_autograd_matmul(RCRE_TensorTape* tape, RCRE_AGTensor* a, RCRE_AGTensor* b);
RCRE_NODISCARD RCRE_AGTensor* rcre_autograd_add(RCRE_TensorTape* tape, RCRE_AGTensor* a, RCRE_AGTensor* b);
RCRE_NODISCARD RCRE_AGTensor* rcre_autograd_add_bias(RCRE_TensorTape* tape, RCRE_AGTensor* x, RCRE_AGTensor* bias);
RCRE_NODISCARD RCRE_AGTensor* rcre_autograd_relu(RCRE_TensorTape* tape, RCRE_AGTensor* x);
RCRE_NODISCARD RCRE_AGTensor* rcre_autograd_sigmoid(RCRE_TensorTape* tape, RCRE_AGTensor* x);

/* ========================================================================= */
/* 4. Loss Functions & Gradients                                             */
/* ========================================================================= */

/* Tensor-based Loss Functions */
float rcre_loss_mse(const RCRE_AGTensor* preds, const RCRE_AGTensor* targets, RCRE_AGTensor* grad_out);
float rcre_loss_bce(const RCRE_AGTensor* preds, const RCRE_AGTensor* targets, RCRE_AGTensor* grad_out);
float rcre_loss_cross_entropy_logits(const RCRE_AGTensor* logits, const size_t* targets, size_t batch_size, size_t num_classes, RCRE_AGTensor* grad_out);
float rcre_loss_kl_divergence(const RCRE_AGTensor* mu, const RCRE_AGTensor* logvar, RCRE_AGTensor* grad_mu, RCRE_AGTensor* grad_logvar);

/* Scalar-based Loss Functions */
/*
 * Note: the scalar DAG engine keeps a process-global visited-epoch counter;
 * concurrent backward() calls from multiple threads are NOT supported.
 * Use the tensor-tape engine in multi-threaded contexts.
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
    RCRE_AGTensor** params;
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

RCRE_NODISCARD RCRE_TensorOptim* rcre_tensor_optim_sgd_create(RCRE_Arena* arena, RCRE_AGTensor** params, size_t num_params, float lr, float momentum, float weight_decay);
RCRE_NODISCARD RCRE_TensorOptim* rcre_tensor_optim_adamw_create(RCRE_Arena* arena, RCRE_AGTensor** params, size_t num_params, float lr, float beta1, float beta2, float eps, float weight_decay);
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
    RCRE_AGTensor* X;
    RCRE_AGTensor* y;
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
 * (returns NULL) rather than silently zero-filled.
 */
RCRE_NODISCARD RCRE_AGTensor* rcre_data_load_csv(RCRE_Arena* arena, const char* filepath, bool skip_header, size_t max_rows);

RCRE_NODISCARD RCRE_StandardScaler* rcre_standard_scaler_fit(RCRE_Arena* arena, const RCRE_AGTensor* X);
void rcre_standard_scaler_transform(const RCRE_StandardScaler* scaler, RCRE_AGTensor* X);
void rcre_standard_scaler_inverse_transform(const RCRE_StandardScaler* scaler, RCRE_AGTensor* X);

RCRE_NODISCARD RCRE_MinMaxScaler* rcre_minmax_scaler_fit(RCRE_Arena* arena, const RCRE_AGTensor* X);
void rcre_minmax_scaler_transform(const RCRE_MinMaxScaler* scaler, RCRE_AGTensor* X);

/* X must be 2-D; y may be 1-D (n,) or 2-D (n, k). A 1-D y is treated as
 * (n, 1); all outputs are 2-D. Returns silently no-op on invalid input. */
void rcre_data_train_test_split(RCRE_Arena* arena, const RCRE_AGTensor* X, const RCRE_AGTensor* y,
                               float train_ratio,
                               RCRE_AGTensor** X_train, RCRE_AGTensor** y_train,
                               RCRE_AGTensor** X_test, RCRE_AGTensor** y_test);

/* X must be 2-D; y may be 1-D (n,) or 2-D (n, k). */
RCRE_NODISCARD RCRE_BatchIter* rcre_batch_iter_create(RCRE_Arena* arena, RCRE_AGTensor* X, RCRE_AGTensor* y, size_t batch_size, bool shuffle);
/* Copies one batch into caller-provided tensors. Returns false (and copies
 * nothing) at epoch end OR when batch_X/batch_y are too small. */
bool rcre_batch_iter_next(RCRE_BatchIter* iter, RCRE_AGTensor* batch_X, RCRE_AGTensor* batch_y);
void rcre_batch_iter_reset(RCRE_BatchIter* iter);

/* ========================================================================= */
/* 7. Visualizations: Terminal Loss Curves, Confusion Matrix & 3D WebGL      */
/* ========================================================================= */

void rcre_vis_loss_history(const float* loss_history, size_t count, const char* title);
void rcre_vis_confusion_matrix(const size_t* confusion_matrix, size_t num_classes, const char* const* class_names);
bool rcre_vis_export_svg_curve(const float* y_vals, size_t count, const char* filepath, const char* title);
bool rcre_vis_export_3d_clusters_html(const RCRE_AGTensor* points, const size_t* assignments, const RCRE_AGTensor* centroids, size_t k, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_AUTOGRAD_H */
