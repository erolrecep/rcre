/**
 * @file rcre_metric.h
 * @brief RCRE Model Evaluation Metrics & Cross-Validation Protocol (ISO C23)
 *
 * Implements ROC-AUC, PR-AUC, Confusion Matrix, F1-Score, R2, RMSE, MAE,
 * Silhouette Score, Davies-Bouldin Index, and Stratified K-Fold CV splits.
 */

#ifndef RCRE_METRIC_H
#define RCRE_METRIC_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Classification Metrics                                                 */
/* ========================================================================= */

typedef struct {
    float  accuracy;
    float  precision;
    float  recall;       /* Sensitivity / True Positive Rate                 */
    float  specificity;  /* True Negative Rate                               */
    float  f1_score;     /* 2 * P * R / (P + R)                              */
    float  mcc;          /* Matthews Correlation Coefficient                 */
    size_t tp;
    size_t fp;
    size_t tn;
    size_t fn;
} RCRE_ClassificationReport;

/**
 * @brief Computes Area Under the Receiver Operating Characteristic Curve (ROC-AUC).
 * @param y_true Ground truth binary labels (0 or 1) [n].
 * @param y_score Predicted probability or continuous decision score [n].
 * @param n Sample count.
 * @param scratch Transient arena for sorting predictions.
 * @return Computed ROC-AUC score in [0.0, 1.0].
 */
RCRE_NODISCARD float rcre_metric_roc_auc(
    const float* y_true,
    const float* y_score,
    size_t n,
    RCRE_Arena* scratch
);

/**
 * @brief Computes Area Under the Precision-Recall Curve (PR-AUC).
 */
RCRE_NODISCARD float rcre_metric_pr_auc(
    const float* y_true,
    const float* y_score,
    size_t n,
    RCRE_Arena* scratch
);

/**
 * @brief Computes C x C confusion matrix for multi-class predictions.
 * @param y_true Ground truth labels (0 .. num_classes-1) [n].
 * @param y_pred Predicted labels (0 .. num_classes-1) [n].
 * @param n Sample count.
 * @param num_classes Class count (C).
 * @param arena Destination arena.
 * @return Output C x C tensor (int32).
 */
RCRE_NODISCARD RCRE_Tensor rcre_metric_confusion_matrix(
    const int32_t* y_true,
    const int32_t* y_pred,
    size_t n,
    size_t num_classes,
    RCRE_Arena* arena
);

/**
 * @brief Computes binary classification performance report (accuracy, F1, MCC).
 */
RCRE_NODISCARD RCRE_ClassificationReport rcre_metric_classification_summary(
    const int32_t* y_true,
    const int32_t* y_pred,
    size_t n,
    float threshold
);

/**
 * @brief Computes binary cross-entropy log-loss.
 */
RCRE_NODISCARD float rcre_metric_log_loss(
    const float* y_true,
    const float* y_prob,
    size_t n
);

/* ========================================================================= */
/* 2. Regression Metrics                                                     */
/* ========================================================================= */

RCRE_NODISCARD float rcre_metric_mse(
    const float* y_true,
    const float* y_pred,
    size_t n
);

RCRE_NODISCARD float rcre_metric_rmse(
    const float* y_true,
    const float* y_pred,
    size_t n
);

RCRE_NODISCARD float rcre_metric_mae(
    const float* y_true,
    const float* y_pred,
    size_t n
);

/**
 * @brief Computes R^2 (coefficient of determination): 1 - (SS_res / SS_tot).
 */
RCRE_NODISCARD float rcre_metric_r2_score(
    const float* y_true,
    const float* y_pred,
    size_t n
);

/* ========================================================================= */
/* 3. Unsupervised Clustering Validation                                     */
/* ========================================================================= */

/**
 * @brief Computes mean Silhouette Score over all clustered samples in [-1, 1].
 * @param X Feature matrix [n_samples, n_features].
 * @param labels Cluster label per sample (0 .. n_clusters-1) [n_samples].
 * @param scratch Transient arena.
 * @return Output mean silhouette score.
 */
RCRE_NODISCARD float rcre_metric_silhouette_score(
    const RCRE_Tensor* X,
    const int32_t* labels,
    RCRE_Arena* scratch
);

/**
 * @brief Computes Davies-Bouldin Index (lower is better, 0 is optimal separation).
 */
RCRE_NODISCARD float rcre_metric_davies_bouldin_score(
    const RCRE_Tensor* X,
    const int32_t* labels,
    size_t n_clusters,
    RCRE_Arena* scratch
);

/* ========================================================================= */
/* 4. Cross-Validation & Resampling Protocol                                 */
/* ========================================================================= */

typedef struct {
    size_t* train_indices;
    size_t  train_count;
    size_t* test_indices;
    size_t  test_count;
} RCRE_FoldSplit;

/**
 * @brief Generates K-Fold cross validation index splits.
 * @param n_samples Total dataset size.
 * @param k Number of folds (e.g. 5 or 10).
 * @param shuffle Whether to shuffle indices.
 * @param seed PRNG seed for shuffling.
 * @param arena Destination arena.
 * @param out_splits Output array of k fold splits [k].
 * @return RCRE_OK on success.
 */
RCRE_NODISCARD int rcre_cv_kfold(
    size_t n_samples,
    size_t k,
    bool shuffle,
    uint64_t seed,
    RCRE_Arena* arena,
    RCRE_FoldSplit* out_splits
);

/**
 * @brief Generates Stratified K-Fold cross validation index splits.
 */
RCRE_NODISCARD int rcre_cv_stratified_kfold(
    const int32_t* y_labels,
    size_t n_samples,
    size_t k,
    RCRE_Arena* arena,
    RCRE_FoldSplit* out_splits
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_METRIC_H */
