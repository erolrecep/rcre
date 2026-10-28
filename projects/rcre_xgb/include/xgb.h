/**
 * @file xgb.h
 * @brief RCRE-XGB — XGBoost-style gradient boosted trees on the RCRE library.
 *
 * Zero external dependencies beyond RCRE. ISO C23, portable (clang/gcc).
 *
 * Implements the Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting
 * System" objective and the quantile-histogram tree construction:
 *
 *   - second-order (gradient + Hessian) Newton updates per leaf
 *   - quantile binning (default 256 bins/feature)
 *   - histogram greedy split finding with L1+L2-regularized structural gain
 *   - missing values: per-node default direction (larger-Hessian side)
 *   - row/column subsampling per tree, early stopping on a validation set
 *   - losses: squared error (regression), logistic (binary), softprob (multi)
 *
 * Trained models are stored as flat node arrays and can be handed to
 * RCRE's zero-allocation inference engine (rcre_ml_gbdt_predict) via
 * xgb_to_rcre_gbdt(), or predicted with xgb_predict() which handles
 * missing-value default directions and multi-class output natively.
 *
 * NOTE: include "rcre.h" (or the amalgamated dist header) BEFORE this
 * header — the RCRE interop signatures below use RCRE's public types.
 */

#ifndef XGB_H
#define XGB_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

/* RCRE_Arena is provided by the RCRE library (forward decl keeps xgb.h
 * standalone-includable). */
struct RCRE_Arena;

typedef enum {
    XGB_TASK_REGRESSION = 0,  /* squared error,  raw score           */
    XGB_TASK_BINARY     = 1,  /* logistic,       probability out     */
    XGB_TASK_MULTI      = 2   /* softmax over K classes, K-prob out  */
} XGBTask;

typedef struct {
    uint32_t feature;    /* split feature index (0 if leaf)          */
    float    threshold;  /* x <= threshold -> left (finite if split) */
    uint32_t left;       /* child node indices (invalid if leaf)     */
    uint32_t right;
    float    leaf_value; /* pre-eta leaf weight (eta applied in pred)*/
    bool     is_leaf;
    bool     default_left; /* where NaN/missing rows go              */
} XGBNode;

typedef struct XGBDataset XGBDataset;

typedef struct XGBModel {
    XGBTask  task;
    int      num_classes;     /* 1 for regression/binary */
    int      num_trees;       /* trees actually kept (early-stopped) */
    int      best_iteration;  /* 0-based best round, -1 if no val    */
    int      dim;
    float    eta;
    float    base_score;      /* scalar base (regression/binary)     */
    float    base_scores[16]; /* per-class base (multi)              */

    XGBNode*  nodes;          /* flat: all trees concatenated        */
    size_t    num_nodes;
    uint32_t* tree_roots;     /* num_trees entries into nodes[]      */

    /* validation loss curve (length = rounds attempted) */
    float*    val_curve;
    int       val_curve_len;
} XGBModel;

typedef struct {
    XGBTask   task;
    int       num_classes;        /* for XGB_TASK_MULTI; 1 otherwise */
    int       rounds;             /* max boosting rounds            */
    float     eta;                /* learning rate (default 0.3)    */
    int       max_depth;          /* default 6                      */
    float     min_child_weight;   /* min Hessian sum in a child (1) */
    float     gamma;              /* min gain to split (0)          */
    float     lambda;             /* L2 leaf regularization (1.0)   */
    float     alpha;              /* L1 leaf regularization (0)     */
    float     subsample;          /* row fraction per tree (1.0)    */
    float     colsample;          /* column fraction per tree (1.0) */
    int       max_bins;           /* quantile bins per feature (256)*/
    int       early_stop_rounds;  /* patience; 0 = disabled         */
    uint64_t  seed;               /* determinism (42)               */
    float     base_score;         /* 0 = auto (data-derived)        */
} XGBConfig;

void xgb_config_default(XGBConfig* cfg);

/* ------------------------------------------------------------------ */
/* Datasets                                                            */
/* ------------------------------------------------------------------ */

/**
 * Create a training dataset.
 *  X: row-major n x d floats; NaN marks a missing feature value.
 *  y: n labels. REGRESSION: target float. BINARY: 0/1. MULTI: class id
 *     0..K-1 (any integer encoding).
 *  cfg: used for bin count + seed (binning is deterministic).
 *  arena: owns the binning tables; X/y are copied in.
 * Returns NULL on allocation failure / bad args.
 */
XGBDataset* xgb_dataset_create(const float* X, const float* y,
                               size_t n, size_t d,
                               const XGBConfig* cfg,
                               struct RCRE_Arena* arena);

size_t xgb_dataset_rows(const XGBDataset* ds);
size_t xgb_dataset_dim(const XGBDataset* ds);
const float* xgb_dataset_x(const XGBDataset* ds);   /* n x d, NaN = missing */
const float* xgb_dataset_y(const XGBDataset* ds);

/* ------------------------------------------------------------------ */
/* Training                                                            */
/* ------------------------------------------------------------------ */

/**
 * Train a boosted model.
 *  val may be NULL (no early stopping; early_stop_rounds ignored).
 *  model is filled in place; all memory is arena-allocated (valid while
 *  the arena is alive). Returns 0 on success.
 * After training with a validation set, model->num_trees is set to the
 * best round (+1) — prediction uses the early-stopped model directly.
 */
int xgb_train(const XGBDataset* train, const XGBDataset* val,
              const XGBConfig* cfg, XGBModel* model,
              struct RCRE_Arena* arena);

/* ------------------------------------------------------------------ */
/* Prediction                                                          */
/* ------------------------------------------------------------------ */

/**
 * Predict.
 *  REGRESSION: out[i] = raw score.            (n outputs)
 *  BINARY:     out[i] = P(y=1 | x).           (n outputs)
 *  MULTI:      out[i*K + k] = P(y=k | x).     (n*K outputs)
 * Rows may contain NaNs (routed via each node's default direction).
 */
int xgb_predict(const XGBModel* model, const float* X,
                size_t n, size_t d, float* out);

/** Raw score (no sigmoid/softmax). Binary: logit. Multi: K logits. */
int xgb_predict_raw(const XGBModel* model, const float* X,
                    size_t n, size_t d, float* out);

/* ------------------------------------------------------------------ */
/* RCRE interop (RCRE_TreeNode / RCRE_GBDTModel come from rcre.h)      */
/* ------------------------------------------------------------------ */

/**
 * Convert a trained binary or regression model to RCRE's flat
 * inference layout. Leaf values are eta-scaled to match RCRE's
 * predictor convention (base_score + sum of leaf values, sigmoid for
 * binary). Multi-class models are rejected (RCRE predictor is
 * single-output).
 * NOTE: RCRE's predictor routes NaN to the RIGHT child; xgb_predict()
 * honors each node's default direction. Use xgb_predict for NaN-heavy
 * data, the RCRE path for NaN-free data.
 * out_nodes: caller-provided, model->num_nodes entries (arena-alloc).
 */
int xgb_to_rcre_gbdt(const XGBModel* model, RCRE_GBDTModel* out,
                     RCRE_TreeNode* out_nodes,
                     size_t* out_roots);

/* ------------------------------------------------------------------ */
/* Metrics                                                             */
/* ------------------------------------------------------------------ */

float xgb_logloss(const float* y, const float* p, size_t n);   /* binary */
float xgb_auc(const float* y, const float* p, size_t n);
float xgb_rmse(const float* y, const float* p, size_t n);
float xgb_mae(const float* y, const float* p, size_t n);
float xgb_accuracy(const float* y, const float* p, size_t n, int K);
/* multi-class: p is n x K probabilities (K columns) */

/* ------------------------------------------------------------------ */
/* I/O                                                                 */
/* ------------------------------------------------------------------ */

/**
 * Load a CSV/whitespace table. Header row auto-detected.
 *  label_col: 0-based column holding the label (-1 = no label column;
 *             y_out NULL).
 *  Missing values: empty field or "nan"/"NaN" -> NaN.
 *  X, y, n, d set on success (X is n x d with the label column removed).
 */
int xgb_csv_load(const char* path, float** X, float** y,
                 size_t* n, size_t* d, int label_col,
                 struct RCRE_Arena* arena);

int xgb_model_save(const XGBModel* model, const char* path);
int xgb_model_load(const char* path, XGBModel* model,
                   struct RCRE_Arena* arena);

/** Human-readable tree dump (for debugging / reports). feature_names may be NULL. */
void xgb_model_print(const XGBModel* model, const char* const* feature_names);

/* ------------------------------------------------------------------ */
/* Diagnostics                                                         */
/* ------------------------------------------------------------------ */

/** Number of internal nodes / leaves in the ensemble. */
void xgb_model_counts(const XGBModel* model, int* internal, int* leaves);

#ifdef __cplusplus
}
#endif

#endif /* XGB_H */
