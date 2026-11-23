/**
 * @file metric.c
 * @brief RCRE Model Evaluation Metrics & Cross-Validation Implementation (ISO C23)
 */

#include "rcre_metric.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================= */
/* 1. Classification Metrics                                                 */
/* ========================================================================= */

typedef struct ScoreIndex {
    float   score;
    float   label;
    size_t  orig_idx;
} ScoreIndex;

static int compare_score_desc(const void* a, const void* b) {
    float sa = ((const ScoreIndex*)a)->score;
    float sb = ((const ScoreIndex*)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return 1;
    return 0;
}

float rcre_metric_roc_auc(
    const float* y_true,
    const float* y_score,
    size_t n,
    RCRE_Arena* scratch
) {
    if (!y_true || !y_score || n == 0 || !scratch) {
        return 0.5f;
    }

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
    ScoreIndex* items = (ScoreIndex*)rcre_arena_alloc(scratch, n * sizeof(ScoreIndex), alignof(ScoreIndex));
    if (!items) {
        rcre_arena_temp_end(tmp);
        return 0.5f;
    }

    size_t n_pos = 0;
    size_t n_neg = 0;
    for (size_t i = 0; i < n; ++i) {
        items[i].score = y_score[i];
        items[i].label = y_true[i];
        items[i].orig_idx = i;
        if (y_true[i] > 0.5f) n_pos++;
        else n_neg++;
    }

    if (n_pos == 0 || n_neg == 0) {
        rcre_arena_temp_end(tmp);
        return 0.5f;
    }

    /* Sort descending by score */
    qsort(items, n, sizeof(ScoreIndex), compare_score_desc);

    /* Trapezoidal AUC integration */
    double fp = 0.0;
    double tp = 0.0;
    double prev_fp = 0.0;
    double prev_tp = 0.0;
    double auc = 0.0;

    for (size_t i = 0; i < n; ++i) {
        if (items[i].label > 0.5f) {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        if (i + 1 == n || items[i].score != items[i + 1].score) {
            /* Add trapezoid area */
            auc += (fp - prev_fp) * (tp + prev_tp) * 0.5;
            prev_fp = fp;
            prev_tp = tp;
        }
    }

    float result = (float)(auc / (double)(n_pos * n_neg));
    rcre_arena_temp_end(tmp);
    return result;
}

float rcre_metric_pr_auc(
    const float* y_true,
    const float* y_score,
    size_t n,
    RCRE_Arena* scratch
) {
    if (!y_true || !y_score || n == 0 || !scratch) {
        return 0.0f;
    }

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
    ScoreIndex* items = (ScoreIndex*)rcre_arena_alloc(scratch, n * sizeof(ScoreIndex), alignof(ScoreIndex));
    if (!items) {
        rcre_arena_temp_end(tmp);
        return 0.0f;
    }

    size_t n_pos = 0;
    for (size_t i = 0; i < n; ++i) {
        items[i].score = y_score[i];
        items[i].label = y_true[i];
        items[i].orig_idx = i;
        if (y_true[i] > 0.5f) n_pos++;
    }

    if (n_pos == 0) {
        rcre_arena_temp_end(tmp);
        return 0.0f;
    }

    qsort(items, n, sizeof(ScoreIndex), compare_score_desc);

    double tp = 0.0;
    double fp = 0.0;
    double prev_recall = 0.0;
    double pr_auc = 0.0;

    for (size_t i = 0; i < n; ++i) {
        if (items[i].label > 0.5f) tp += 1.0;
        else fp += 1.0;

        double recall = tp / (double)n_pos;
        double precision = tp / (tp + fp);

        if (recall > prev_recall) {
            pr_auc += precision * (recall - prev_recall);
            prev_recall = recall;
        }
    }

    rcre_arena_temp_end(tmp);
    return (float)pr_auc;
}

RCRE_Tensor rcre_metric_confusion_matrix(
    const int32_t* y_true,
    const int32_t* y_pred,
    size_t n,
    size_t num_classes,
    RCRE_Arena* arena
) {
    RCRE_Tensor out_cm = {0};
    if (!y_true || !y_pred || n == 0 || num_classes == 0 || !arena) {
        return out_cm;
    }

    size_t cm_shape[2] = { num_classes, num_classes };
    out_cm = rcre_tensor_create(cm_shape, 2, RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);
    if (!out_cm.data) return out_cm;

    int32_t* cm = (int32_t*)out_cm.data;
    memset(cm, 0, num_classes * num_classes * sizeof(int32_t));

    for (size_t i = 0; i < n; ++i) {
        int32_t t = y_true[i];
        int32_t p = y_pred[i];
        if (t >= 0 && (size_t)t < num_classes && p >= 0 && (size_t)p < num_classes) {
            cm[(size_t)t * num_classes + (size_t)p]++;
        }
    }

    return out_cm;
}

RCRE_ClassificationReport rcre_metric_classification_summary(
    const int32_t* y_true,
    const int32_t* y_pred,
    size_t n,
    float threshold
) {
    (void)threshold;
    RCRE_ClassificationReport rep = {0};
    if (!y_true || !y_pred || n == 0) return rep;

    for (size_t i = 0; i < n; ++i) {
        int32_t t = y_true[i];
        int32_t p = y_pred[i];
        if (t == 1 && p == 1) rep.tp++;
        else if (t == 0 && p == 1) rep.fp++;
        else if (t == 0 && p == 0) rep.tn++;
        else if (t == 1 && p == 0) rep.fn++;
    }

    size_t total = rep.tp + rep.fp + rep.tn + rep.fn;
    rep.accuracy = (total > 0) ? (float)(rep.tp + rep.tn) / (float)total : 0.0f;
    rep.precision = (rep.tp + rep.fp > 0) ? (float)rep.tp / (float)(rep.tp + rep.fp) : 0.0f;
    rep.recall = (rep.tp + rep.fn > 0) ? (float)rep.tp / (float)(rep.tp + rep.fn) : 0.0f;
    rep.specificity = (rep.tn + rep.fp > 0) ? (float)rep.tn / (float)(rep.tn + rep.fp) : 0.0f;

    float p_plus_r = rep.precision + rep.recall;
    rep.f1_score = (p_plus_r > 1e-12f) ? (2.0f * rep.precision * rep.recall / p_plus_r) : 0.0f;

    double num = ((double)rep.tp * (double)rep.tn) - ((double)rep.fp * (double)rep.fn);
    double den = sqrt((double)(rep.tp + rep.fp) * (double)(rep.tp + rep.fn) * (double)(rep.tn + rep.fp) * (double)(rep.tn + rep.fn));
    rep.mcc = (den > 1e-12) ? (float)(num / den) : 0.0f;

    return rep;
}

float rcre_metric_log_loss(
    const float* y_true,
    const float* y_prob,
    size_t n
) {
    if (!y_true || !y_prob || n == 0) return 0.0f;

    double total_loss = 0.0;
    const double eps = 1e-7;

    for (size_t i = 0; i < n; ++i) {
        double y = (double)y_true[i];
        double p = (double)y_prob[i];
        if (p < eps) p = eps;
        if (p > 1.0 - eps) p = 1.0 - eps;
        total_loss += -(y * log(p) + (1.0 - y) * log(1.0 - p));
    }

    return (float)(total_loss / (double)n);
}

/* ========================================================================= */
/* 2. Regression Metrics                                                     */
/* ========================================================================= */

float rcre_metric_mse(
    const float* y_true,
    const float* y_pred,
    size_t n
) {
    if (!y_true || !y_pred || n == 0) return 0.0f;

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = (double)y_true[i] - (double)y_pred[i];
        sum += diff * diff;
    }
    return (float)(sum / (double)n);
}

float rcre_metric_rmse(
    const float* y_true,
    const float* y_pred,
    size_t n
) {
    return sqrtf(rcre_metric_mse(y_true, y_pred, n));
}

float rcre_metric_mae(
    const float* y_true,
    const float* y_pred,
    size_t n
) {
    if (!y_true || !y_pred || n == 0) return 0.0f;

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += fabs((double)y_true[i] - (double)y_pred[i]);
    }
    return (float)(sum / (double)n);
}

float rcre_metric_r2_score(
    const float* y_true,
    const float* y_pred,
    size_t n
) {
    if (!y_true || !y_pred || n < 2) return 0.0f;

    double sum_y = 0.0;
    for (size_t i = 0; i < n; ++i) sum_y += (double)y_true[i];
    double mean_y = sum_y / (double)n;

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = (double)y_true[i] - (double)y_pred[i];
        double dev = (double)y_true[i] - mean_y;
        ss_res += diff * diff;
        ss_tot += dev * dev;
    }

    if (ss_tot <= 1e-12) return 0.0f;
    return (float)(1.0 - (ss_res / ss_tot));
}

/* ========================================================================= */
/* 3. Unsupervised Clustering Validation                                     */
/* ========================================================================= */

float rcre_metric_silhouette_score(
    const RCRE_Tensor* X,
    const int32_t* labels,
    RCRE_Arena* scratch
) {
    if (!X || !X->data || X->ndim != 2 || !labels || !scratch) {
        return 0.0f;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];

    /* Find number of clusters */
    int32_t max_l = 0;
    for (size_t i = 0; i < N; ++i) {
        if (labels[i] > max_l) max_l = labels[i];
    }
    size_t n_clusters = (size_t)(max_l + 1);
    if (n_clusters < 2 || N < n_clusters) return 0.0f;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
    const float* src = (const float*)X->data;

    size_t* cluster_sizes = (size_t*)rcre_arena_alloc(scratch, n_clusters * sizeof(size_t), 16);
    if (!cluster_sizes) {
        rcre_arena_temp_end(tmp);
        return 0.0f;
    }
    memset(cluster_sizes, 0, n_clusters * sizeof(size_t));
    for (size_t i = 0; i < N; ++i) {
        int32_t l = labels[i];
        if (l >= 0 && (size_t)l < n_clusters) cluster_sizes[(size_t)l]++;
    }

    double total_silhouette = 0.0;
    size_t valid_samples = 0;

    for (size_t i = 0; i < N; ++i) {
        int32_t own_cluster = labels[i];
        if (own_cluster < 0 || (size_t)own_cluster >= n_clusters) continue;
        size_t own_size = cluster_sizes[(size_t)own_cluster];
        if (own_size <= 1) continue;

        double a_i = 0.0;
        for (size_t j = 0; j < N; ++j) {
            if (i == j || labels[j] != own_cluster) continue;
            double dist_sq = 0.0;
            for (size_t p = 0; p < P; ++p) {
                double diff = (double)src[i * P + p] - (double)src[j * P + p];
                dist_sq += diff * diff;
            }
            a_i += sqrt(dist_sq);
        }
        a_i /= (double)(own_size - 1);

        double b_i = INFINITY;
        for (size_t k = 0; k < n_clusters; ++k) {
            if ((int32_t)k == own_cluster || cluster_sizes[k] == 0) continue;
            double dist_k = 0.0;
            for (size_t j = 0; j < N; ++j) {
                if (labels[j] != (int32_t)k) continue;
                double dist_sq = 0.0;
                for (size_t p = 0; p < P; ++p) {
                    double diff = (double)src[i * P + p] - (double)src[j * P + p];
                    dist_sq += diff * diff;
                }
                dist_k += sqrt(dist_sq);
            }
            dist_k /= (double)cluster_sizes[k];
            if (dist_k < b_i) b_i = dist_k;
        }

        double max_ab = (a_i > b_i) ? a_i : b_i;
        if (max_ab > 1e-12) {
            total_silhouette += (b_i - a_i) / max_ab;
        }
        valid_samples++;
    }

    float res = (valid_samples > 0) ? (float)(total_silhouette / (double)valid_samples) : 0.0f;
    rcre_arena_temp_end(tmp);
    return res;
}

float rcre_metric_davies_bouldin_score(
    const RCRE_Tensor* X,
    const int32_t* labels,
    size_t n_clusters,
    RCRE_Arena* scratch
) {
    if (!X || !X->data || X->ndim != 2 || !labels || n_clusters < 2 || !scratch) {
        return 0.0f;
    }

    size_t N = X->shape[0];
    size_t P = X->shape[1];

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
    const float* src = (const float*)X->data;

    float* centroids = (float*)rcre_arena_alloc(scratch, n_clusters * P * sizeof(float), 64);
    size_t* cluster_sizes = (size_t*)rcre_arena_alloc(scratch, n_clusters * sizeof(size_t), 16);
    float* s_i = (float*)rcre_arena_alloc(scratch, n_clusters * sizeof(float), 16);
    if (!centroids || !cluster_sizes || !s_i) {
        rcre_arena_temp_end(tmp);
        return 0.0f;
    }
    memset(centroids, 0, n_clusters * P * sizeof(float));
    memset(cluster_sizes, 0, n_clusters * sizeof(size_t));
    memset(s_i, 0, n_clusters * sizeof(float));

    /* 1. Centroid sums */
    for (size_t i = 0; i < N; ++i) {
        int32_t l = labels[i];
        if (l >= 0 && (size_t)l < n_clusters) {
            cluster_sizes[(size_t)l]++;
            for (size_t p = 0; p < P; ++p) {
                centroids[(size_t)l * P + p] += src[i * P + p];
            }
        }
    }

    for (size_t k = 0; k < n_clusters; ++k) {
        if (cluster_sizes[k] > 0) {
            for (size_t p = 0; p < P; ++p) {
                centroids[k * P + p] /= (float)cluster_sizes[k];
            }
        }
    }

    /* 2. Mean intra-cluster distances s_i */
    for (size_t i = 0; i < N; ++i) {
        int32_t l = labels[i];
        if (l >= 0 && (size_t)l < n_clusters && cluster_sizes[(size_t)l] > 0) {
            double dist_sq = 0.0;
            for (size_t p = 0; p < P; ++p) {
                double diff = (double)src[i * P + p] - (double)centroids[(size_t)l * P + p];
                dist_sq += diff * diff;
            }
            s_i[(size_t)l] += (float)sqrt(dist_sq);
        }
    }
    for (size_t k = 0; k < n_clusters; ++k) {
        if (cluster_sizes[k] > 0) {
            s_i[k] /= (float)cluster_sizes[k];
        }
    }

    /* 3. Compute maximum R_ij */
    double db_sum = 0.0;
    for (size_t i = 0; i < n_clusters; ++i) {
        double max_r = 0.0;
        for (size_t j = 0; j < n_clusters; ++j) {
            if (i == j) continue;
            double d_ij_sq = 0.0;
            for (size_t p = 0; p < P; ++p) {
                double diff = (double)centroids[i * P + p] - (double)centroids[j * P + p];
                d_ij_sq += diff * diff;
            }
            double d_ij = sqrt(d_ij_sq);
            if (d_ij > 1e-12) {
                double r_ij = ((double)s_i[i] + (double)s_i[j]) / d_ij;
                if (r_ij > max_r) max_r = r_ij;
            }
        }
        db_sum += max_r;
    }

    float res = (float)(db_sum / (double)n_clusters);
    rcre_arena_temp_end(tmp);
    return res;
}

/* ========================================================================= */
/* 4. Cross-Validation & Resampling Protocol                                 */
/* ========================================================================= */

int rcre_cv_kfold(
    size_t n_samples,
    size_t k,
    bool shuffle,
    uint64_t seed,
    RCRE_Arena* arena,
    RCRE_FoldSplit* out_splits
) {
    if (n_samples < k || k < 2 || !arena || !out_splits) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
    size_t* indices = (size_t*)rcre_arena_alloc(arena, n_samples * sizeof(size_t), 16);
    if (!indices) {
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < n_samples; ++i) indices[i] = i;

    if (shuffle) {
        for (size_t i = n_samples - 1; i > 0; --i) {
            seed = seed * 6364136223846793005ULL + 1ULL;
            size_t j = (size_t)((seed >> 33) % (uint64_t)(i + 1));
            size_t swap_t = indices[i];
            indices[i] = indices[j];
            indices[j] = swap_t;
        }
    }

    size_t base_fold_sz = n_samples / k;
    size_t remainder = n_samples % k;

    size_t start = 0;
    for (size_t f = 0; f < k; ++f) {
        size_t test_sz = base_fold_sz + (f < remainder ? 1 : 0);
        size_t train_sz = n_samples - test_sz;

        out_splits[f].test_count = test_sz;
        out_splits[f].train_count = train_sz;
        out_splits[f].test_indices = (size_t*)rcre_arena_alloc(arena, test_sz * sizeof(size_t), 16);
        out_splits[f].train_indices = (size_t*)rcre_arena_alloc(arena, train_sz * sizeof(size_t), 16);

        if (!out_splits[f].test_indices || !out_splits[f].train_indices) {
            return RCRE_ERR_OUT_OF_MEMORY;
        }

        size_t train_pos = 0;
        size_t test_pos = 0;

        for (size_t i = 0; i < n_samples; ++i) {
            if (i >= start && i < start + test_sz) {
                out_splits[f].test_indices[test_pos++] = indices[i];
            } else {
                out_splits[f].train_indices[train_pos++] = indices[i];
            }
        }

        start += test_sz;
    }

    return RCRE_OK;
}

int rcre_cv_stratified_kfold(
    const int32_t* y_labels,
    size_t n_samples,
    size_t k,
    RCRE_Arena* arena,
    RCRE_FoldSplit* out_splits
) {
    if (!y_labels || n_samples < k || k < 2 || !arena || !out_splits) {
        return RCRE_ERR_INVALID_ARG;
    }

    int32_t max_class = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        if (y_labels[i] > max_class) max_class = y_labels[i];
    }
    size_t num_classes = (size_t)(max_class + 1);

    size_t* class_counts = (size_t*)rcre_arena_alloc(arena, num_classes * sizeof(size_t), 16);
    if (!class_counts) return RCRE_ERR_OUT_OF_MEMORY;
    memset(class_counts, 0, num_classes * sizeof(size_t));

    for (size_t i = 0; i < n_samples; ++i) {
        class_counts[(size_t)y_labels[i]]++;
    }

    for (size_t f = 0; f < k; ++f) {
        out_splits[f].test_count = 0;
        out_splits[f].train_count = 0;
        out_splits[f].test_indices = (size_t*)rcre_arena_alloc(arena, n_samples * sizeof(size_t), 16);
        out_splits[f].train_indices = (size_t*)rcre_arena_alloc(arena, n_samples * sizeof(size_t), 16);
        if (!out_splits[f].test_indices || !out_splits[f].train_indices) return RCRE_ERR_OUT_OF_MEMORY;
    }

    for (size_t c = 0; c < num_classes; ++c) {
        size_t c_n = class_counts[c];
        if (c_n == 0) continue;

        size_t* c_indices = (size_t*)rcre_arena_alloc(arena, c_n * sizeof(size_t), 16);
        size_t idx = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            if ((size_t)y_labels[i] == c) c_indices[idx++] = i;
        }

        size_t base_sz = c_n / k;
        size_t rem = c_n % k;
        size_t start = 0;

        for (size_t f = 0; f < k; ++f) {
            size_t test_sz = base_sz + (f < rem ? 1 : 0);
            for (size_t i = 0; i < c_n; ++i) {
                if (i >= start && i < start + test_sz) {
                    out_splits[f].test_indices[out_splits[f].test_count++] = c_indices[i];
                } else {
                    out_splits[f].train_indices[out_splits[f].train_count++] = c_indices[i];
                }
            }
            start += test_sz;
        }
    }

    return RCRE_OK;
}
