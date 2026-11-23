/**
 * @file prep.c
 * @brief RCRE Preprocessing & Feature Engineering Implementation (ISO C23)
 */

#include "rcre_prep.h"
#include "rcre_stat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================= */
/* 1. StandardScaler                                                         */
/* ========================================================================= */

RCRE_PrepStandardScaler rcre_prep_standard_scaler_fit(
    const RCRE_Tensor* X,
    RCRE_Arena* arena
) {
    RCRE_PrepStandardScaler res = {0};
    if (!X || !X->data || X->ndim != 2 || X->dtype != RCRE_DTYPE_F32 || !arena) {
        return res;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (N < 2 || P == 0) return res;

    res.mean = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    res.scale = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    res.n_features = P;
    if (!res.mean || !res.scale) return res;

    const float* src = (const float*)X->data;

    for (size_t j = 0; j < P; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += (double)src[i * P + j];
        }
        double mean = sum / (double)N;
        res.mean[j] = (float)mean;

        double sum_sq = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double d = (double)src[i * P + j] - mean;
            sum_sq += d * d;
        }
        double var = sum_sq / (double)(N - 1);
        double std_val = (var > 1e-12) ? sqrt(var) : 1e-6;
        res.scale[j] = (float)(1.0 / std_val);
    }

    return res;
}

RCRE_Tensor rcre_prep_standard_scaler_transform(
    const RCRE_PrepStandardScaler* scaler,
    const RCRE_Tensor* X,
    RCRE_Arena* arena
) {
    RCRE_Tensor out = {0};
    if (!scaler || !X || !X->data || X->ndim != 2 || !arena) {
        return out;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (P != scaler->n_features) return out;

    size_t out_shape[2] = { N, P };
    out = rcre_tensor_create(out_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!out.data) return out;

    const float* src = (const float*)X->data;
    float* dst = (float*)out.data;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < P; ++j) {
            dst[i * P + j] = (src[i * P + j] - scaler->mean[j]) * scaler->scale[j];
        }
    }

    return out;
}

/* ========================================================================= */
/* 2. MinMaxScaler                                                           */
/* ========================================================================= */

RCRE_PrepMinMaxScaler rcre_prep_minmax_scaler_fit(
    const RCRE_Tensor* X,
    float feature_min,
    float feature_max,
    RCRE_Arena* arena
) {
    RCRE_PrepMinMaxScaler res = {0};
    if (!X || !X->data || X->ndim != 2 || feature_min >= feature_max || !arena) {
        return res;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (N == 0 || P == 0) return res;

    res.min = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    res.scale = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    res.feature_min = feature_min;
    res.feature_max = feature_max;
    res.n_features = P;
    if (!res.min || !res.scale) return res;

    const float* src = (const float*)X->data;
    float range_target = feature_max - feature_min;

    for (size_t j = 0; j < P; ++j) {
        float f_min = src[0 * P + j];
        float f_max = src[0 * P + j];
        for (size_t i = 1; i < N; ++i) {
            float val = src[i * P + j];
            if (val < f_min) f_min = val;
            if (val > f_max) f_max = val;
        }
        res.min[j] = f_min;
        float diff = f_max - f_min;
        res.scale[j] = (diff > 1e-8f) ? (range_target / diff) : 1.0f;
    }

    return res;
}

RCRE_Tensor rcre_prep_minmax_scaler_transform(
    const RCRE_PrepMinMaxScaler* scaler,
    const RCRE_Tensor* X,
    RCRE_Arena* arena
) {
    RCRE_Tensor out = {0};
    if (!scaler || !X || !X->data || X->ndim != 2 || !arena) {
        return out;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (P != scaler->n_features) return out;

    size_t out_shape[2] = { N, P };
    out = rcre_tensor_create(out_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!out.data) return out;

    const float* src = (const float*)X->data;
    float* dst = (float*)out.data;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < P; ++j) {
            dst[i * P + j] = (src[i * P + j] - scaler->min[j]) * scaler->scale[j] + scaler->feature_min;
        }
    }

    return out;
}

/* ========================================================================= */
/* 3. RobustScaler (Median & IQR)                                            */
/* ========================================================================= */

RCRE_RobustScaler rcre_prep_robust_scaler_fit(
    const RCRE_Tensor* X,
    RCRE_Arena* arena
) {
    RCRE_RobustScaler res = {0};
    if (!X || !X->data || X->ndim != 2 || !arena) {
        return res;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (N < 2 || P == 0) return res;

    res.center = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    res.scale = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    res.n_features = P;
    if (!res.center || !res.scale) return res;

    RCRE_ArenaTemp scratch = rcre_arena_temp_begin(arena);
    float* col_buf = (float*)rcre_arena_alloc(arena, N * sizeof(float), 64);
    const float* src = (const float*)X->data;

    for (size_t j = 0; j < P; ++j) {
        for (size_t i = 0; i < N; ++i) {
            col_buf[i] = src[i * P + j];
        }
        float q25 = rcre_stat_quantile(col_buf, N, 0.25f, arena);
        float med = rcre_stat_quantile(col_buf, N, 0.50f, arena);
        float q75 = rcre_stat_quantile(col_buf, N, 0.75f, arena);

        res.center[j] = med;
        float iqr = q75 - q25;
        res.scale[j] = (iqr > 1e-8f) ? (1.0f / iqr) : 1.0f;
    }

    rcre_arena_temp_end(scratch);
    return res;
}

RCRE_Tensor rcre_prep_robust_scaler_transform(
    const RCRE_RobustScaler* scaler,
    const RCRE_Tensor* X,
    RCRE_Arena* arena
) {
    RCRE_Tensor out = {0};
    if (!scaler || !X || !X->data || X->ndim != 2 || !arena) {
        return out;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (P != scaler->n_features) return out;

    size_t out_shape[2] = { N, P };
    out = rcre_tensor_create(out_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!out.data) return out;

    const float* src = (const float*)X->data;
    float* dst = (float*)out.data;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < P; ++j) {
            dst[i * P + j] = (src[i * P + j] - scaler->center[j]) * scaler->scale[j];
        }
    }

    return out;
}

/* ========================================================================= */
/* 4. SimpleImputer (Mean, Median, Constant)                                 */
/* ========================================================================= */

RCRE_SimpleImputer rcre_prep_simple_imputer_fit(
    const RCRE_Tensor* X,
    RCRE_ImputeStrategy strategy,
    RCRE_Arena* arena
) {
    RCRE_SimpleImputer res = {0};
    if (!X || !X->data || X->ndim != 2 || !arena) {
        return res;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (N == 0 || P == 0) return res;

    res.strategy = strategy;
    res.statistics = (float*)rcre_arena_alloc(arena, P * sizeof(float), 64);
    res.n_features = P;
    if (!res.statistics) return res;

    const float* src = (const float*)X->data;
    RCRE_ArenaTemp scratch = rcre_arena_temp_begin(arena);
    float* valid_vals = (float*)rcre_arena_alloc(arena, N * sizeof(float), 64);

    for (size_t j = 0; j < P; ++j) {
        size_t n_valid = 0;
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            float val = src[i * P + j];
            if (!isnan(val)) {
                valid_vals[n_valid++] = val;
                sum += (double)val;
            }
        }

        if (n_valid == 0) {
            res.statistics[j] = 0.0f;
        } else if (strategy == RCRE_IMPUTE_MEAN) {
            res.statistics[j] = (float)(sum / (double)n_valid);
        } else if (strategy == RCRE_IMPUTE_MEDIAN) {
            res.statistics[j] = rcre_stat_quantile(valid_vals, n_valid, 0.5f, arena);
        } else {
            res.statistics[j] = 0.0f;
        }
    }

    rcre_arena_temp_end(scratch);
    return res;
}

RCRE_Tensor rcre_prep_simple_imputer_transform(
    const RCRE_SimpleImputer* imputer,
    const RCRE_Tensor* X,
    RCRE_Arena* arena
) {
    RCRE_Tensor out = {0};
    if (!imputer || !X || !X->data || X->ndim != 2 || !arena) {
        return out;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];
    if (P != imputer->n_features) return out;

    size_t out_shape[2] = { N, P };
    out = rcre_tensor_create(out_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!out.data) return out;

    const float* src = (const float*)X->data;
    float* dst = (float*)out.data;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < P; ++j) {
            float val = src[i * P + j];
            dst[i * P + j] = isnan(val) ? imputer->statistics[j] : val;
        }
    }

    return out;
}

/* ========================================================================= */
/* 5. One-Hot Encoding                                                       */
/* ========================================================================= */

RCRE_Tensor rcre_one_hot_encode(
    const int32_t* categories,
    size_t n,
    size_t num_classes,
    RCRE_Arena* arena
) {
    RCRE_Tensor out = {0};
    if (!categories || n == 0 || num_classes == 0 || !arena) {
        return out;
    }

    size_t out_shape[2] = { n, num_classes };
    out = rcre_tensor_create(out_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!out.data) return out;

    float* dst = (float*)out.data;
    memset(dst, 0, n * num_classes * sizeof(float));

    for (size_t i = 0; i < n; ++i) {
        int32_t cat = categories[i];
        if (cat >= 0 && (size_t)cat < num_classes) {
            dst[i * num_classes + (size_t)cat] = 1.0f;
        }
    }

    return out;
}
