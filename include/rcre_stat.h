/**
 * @file rcre_stat.h
 * @brief RCRE Exploratory Data Analysis & Scientific Statistics Core (ISO C23)
 *
 * Implements numerically stable Welford moments, SIMD quantiles,
 * Pearson & Spearman correlation matrices, and Principal Component Analysis (PCA).
 */

#ifndef RCRE_STAT_H
#define RCRE_STAT_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Descriptive Summary Statistics (Welford Single-Pass)                   */
/* ========================================================================= */

typedef struct RCRE_StatSummary {
    size_t count;      /* Valid element count                                */
    size_t null_count; /* Number of NaN/missing values                       */
    float  mean;       /* Arithmetic mean                                    */
    float  variance;   /* Unbiased sample variance (N-1 denominator)         */
    float  stddev;     /* Sample standard deviation                          */
    float  skewness;   /* Sample skewness (3rd standardized moment)          */
    float  kurtosis;   /* Sample excess kurtosis (4th standardized moment - 3)*/
    float  min;        /* Minimum value                                      */
    float  max;        /* Maximum value                                      */
    float  q25;        /* 25th percentile (1st quartile)                     */
    float  median;     /* Median value (50th percentile)                     */
    float  q75;        /* 75th percentile (3rd quartile)                     */
    float  iqr;        /* Interquartile range (q75 - q25)                    */
} RCRE_StatSummary;

/**
 * @brief Computes comprehensive descriptive statistics on a float array in O(N).
 * @param x Input float array [n].
 * @param n Element count.
 * @param scratch Transient arena for percentile quickselect scratch memory.
 * @return Computed summary structure.
 */
RCRE_NODISCARD RCRE_StatSummary rcre_stat_describe(
    const float* x,
    size_t n,
    RCRE_Arena* scratch
);

/**
 * @brief Computes exact quantile in O(N) using Quickselect.
 * @param x Input float array [n].
 * @param n Element count.
 * @param q Quantile target between 0.0 and 1.0 (e.g. 0.5 for median).
 * @param scratch Transient arena for copying elements before partitioning.
 * @return Computed quantile value.
 */
RCRE_NODISCARD float rcre_stat_quantile(
    const float* x,
    size_t n,
    float q,
    RCRE_Arena* scratch
);

/* ========================================================================= */
/* 2. Covariance & Correlation Matrices                                      */
/* ========================================================================= */

/**
 * @brief Computes the sample covariance matrix (P x P) for an N x P dataset.
 * @param X Input matrix [num_samples, num_features] (contiguous F32).
 * @param arena Destination arena.
 * @return Output covariance matrix [num_features, num_features].
 */
RCRE_NODISCARD RCRE_Tensor rcre_stat_cov_matrix(
    const RCRE_Tensor* X,
    RCRE_Arena* arena
);

/**
 * @brief Computes the Pearson correlation matrix (P x P) in [-1, 1].
 * @param X Input matrix [num_samples, num_features].
 * @param arena Destination arena.
 * @return Output correlation matrix [num_features, num_features].
 */
RCRE_NODISCARD RCRE_Tensor rcre_stat_corr_matrix(
    const RCRE_Tensor* X,
    RCRE_Arena* arena
);

/**
 * @brief Computes the Spearman rank correlation matrix (P x P) in [-1, 1].
 * @param X Input matrix [num_samples, num_features].
 * @param arena Destination arena.
 * @return Output rank correlation matrix.
 */
RCRE_NODISCARD RCRE_Tensor rcre_stat_spearman_matrix(
    const RCRE_Tensor* X,
    RCRE_Arena* arena
);

/* ========================================================================= */
/* 3. Principal Component Analysis (PCA)                                     */
/* ========================================================================= */

typedef struct RCRE_PCA {
    RCRE_Tensor components;               /* Eigenvectors [n_components, n_features] */
    float*      explained_variance;       /* Eigenvalues [n_components]              */
    float*      explained_variance_ratio; /* Variance ratio [n_components]           */
    float*      mean;                     /* Feature means [n_features]              */
    size_t      n_components;             /* Number of principal components          */
    size_t      n_features;               /* Input feature dimension                 */
} RCRE_PCA;

/**
 * @brief Fits Principal Component Analysis via power-iteration SVD on covariance.
 * @param X Input dataset [num_samples, num_features].
 * @param n_components Number of principal components to extract.
 * @param arena Destination arena.
 * @return Output fitted PCA model.
 */
RCRE_NODISCARD RCRE_PCA rcre_pca_fit(
    const RCRE_Tensor* X,
    size_t n_components,
    RCRE_Arena* arena
);

/**
 * @brief Projects high-dimensional data into PCA subspace: Y = (X - mean) * V^T.
 * @param pca Fitted PCA model.
 * @param X Input dataset [num_samples, n_features].
 * @param arena Destination arena.
 * @return Output coordinates [num_samples, n_components].
 */
RCRE_NODISCARD RCRE_Tensor rcre_pca_transform(
    const RCRE_PCA* pca,
    const RCRE_Tensor* X,
    RCRE_Arena* arena
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_STAT_H */
