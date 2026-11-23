/**
 * @file rcre_prep.h
 * @brief RCRE Data Preprocessing & Feature Engineering Pipelines (ISO C23)
 *
 * Implements StandardScaler, MinMaxScaler, RobustScaler, SimpleImputer,
 * and One-Hot Encoding with zero-heap arena allocation.
 */

#ifndef RCRE_PREP_H
#define RCRE_PREP_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Feature Scalers: Standard, MinMax, Robust                              */
/* ========================================================================= */

typedef struct {
    float* mean;     /* Feature means [n_features]                           */
    float* scale;    /* Feature inverse standard deviations [n_features]     */
    size_t n_features;
} RCRE_PrepStandardScaler;

RCRE_NODISCARD RCRE_PrepStandardScaler rcre_prep_standard_scaler_fit(
    const RCRE_Tensor* X,
    RCRE_Arena* arena
);

RCRE_NODISCARD RCRE_Tensor rcre_prep_standard_scaler_transform(
    const RCRE_PrepStandardScaler* scaler,
    const RCRE_Tensor* X,
    RCRE_Arena* arena
);

typedef struct {
    float* min;      /* Feature minimums [n_features]                        */
    float* scale;    /* Inverse range: (feature_max - feature_min) / (max - min) */
    float  feature_min;
    float  feature_max;
    size_t n_features;
} RCRE_PrepMinMaxScaler;

RCRE_NODISCARD RCRE_PrepMinMaxScaler rcre_prep_minmax_scaler_fit(
    const RCRE_Tensor* X,
    float feature_min,
    float feature_max,
    RCRE_Arena* arena
);

RCRE_NODISCARD RCRE_Tensor rcre_prep_minmax_scaler_transform(
    const RCRE_PrepMinMaxScaler* scaler,
    const RCRE_Tensor* X,
    RCRE_Arena* arena
);

typedef struct {
    float* center;   /* Feature medians [n_features]                         */
    float* scale;    /* Feature inverse IQR: 1.0 / (Q75 - Q25) [n_features]  */
    size_t n_features;
} RCRE_RobustScaler;

RCRE_NODISCARD RCRE_RobustScaler rcre_prep_robust_scaler_fit(
    const RCRE_Tensor* X,
    RCRE_Arena* arena
);

RCRE_NODISCARD RCRE_Tensor rcre_prep_robust_scaler_transform(
    const RCRE_RobustScaler* scaler,
    const RCRE_Tensor* X,
    RCRE_Arena* arena
);

/* ========================================================================= */
/* 2. Missing Value Imputation                                               */
/* ========================================================================= */

typedef enum {
    RCRE_IMPUTE_MEAN     = 0,
    RCRE_IMPUTE_MEDIAN   = 1,
    RCRE_IMPUTE_CONSTANT = 2
} RCRE_ImputeStrategy;

typedef struct {
    RCRE_ImputeStrategy strategy;
    float*              statistics; /* Replacement value per column [n_cols] */
    size_t              n_features;
} RCRE_SimpleImputer;

RCRE_NODISCARD RCRE_SimpleImputer rcre_prep_simple_imputer_fit(
    const RCRE_Tensor* X,
    RCRE_ImputeStrategy strategy,
    RCRE_Arena* arena
);

RCRE_NODISCARD RCRE_Tensor rcre_prep_simple_imputer_transform(
    const RCRE_SimpleImputer* imputer,
    const RCRE_Tensor* X,
    RCRE_Arena* arena
);

/* ========================================================================= */
/* 3. Categorical Encoders                                                   */
/* ========================================================================= */

/**
 * @brief Expands an integer category column into an N x K one-hot float tensor.
 * @param categories Array of integer category labels in 0 .. num_classes-1 [n].
 * @param n Sample count.
 * @param num_classes Class count (K).
 * @param arena Destination arena.
 * @return Output dense float tensor [n, num_classes].
 */
RCRE_NODISCARD RCRE_Tensor rcre_one_hot_encode(
    const int32_t* categories,
    size_t n,
    size_t num_classes,
    RCRE_Arena* arena
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_PREP_H */
