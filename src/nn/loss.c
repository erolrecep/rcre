/**
 * @file loss.c
 * @brief Numerical and autograd loss functions: MSE, BCE, Cross-Entropy, and KL Divergence.
 *
 * Part of RCRE Unified Engine.
 *
 * Gradient contract (v0.2.2): analytic gradients are written to
 * `grad_out->grad` ONLY. The legacy dual write into `grad_out->data` has
 * been retired: `.data` is the forward-pass buffer and must never be
 * clobbered by a loss function. When `grad_out->grad == NULL` (borrowed
 * core view), only the scalar loss value is computed.
 */

#include "rcre_autograd.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

static bool loss_is_f32(const RCRE_Tensor* t) {
    return t != nullptr && t->data != nullptr && t->dtype == RCRE_DTYPE_F32;
}

float rcre_loss_mse(const RCRE_Tensor* preds, const RCRE_Tensor* targets, RCRE_Tensor* grad_out) {
    if (!loss_is_f32(preds) || !loss_is_f32(targets) ||
        preds->numel != targets->numel || preds->numel == 0) return 0.0f;

    const float* pd = rcre_tensor_f32_data(preds);
    const float* td = rcre_tensor_f32_data(targets);
    float* g = (grad_out != nullptr) ? grad_out->grad : nullptr;

    size_t n = preds->numel;
    float sum_sq = 0.0f;
    float scale = 2.0f / (float)n;

    for (size_t i = 0; i < n; ++i) {
        float diff = pd[i] - td[i];
        sum_sq += diff * diff;
        if (g != nullptr) g[i] = diff * scale;
    }
    return sum_sq / (float)n;
}

float rcre_loss_bce(const RCRE_Tensor* preds, const RCRE_Tensor* targets, RCRE_Tensor* grad_out) {
    if (!loss_is_f32(preds) || !loss_is_f32(targets) ||
        preds->numel != targets->numel || preds->numel == 0) return 0.0f;

    const float* pd = rcre_tensor_f32_data(preds);
    const float* td = rcre_tensor_f32_data(targets);
    float* g = (grad_out != nullptr) ? grad_out->grad : nullptr;

    size_t n = preds->numel;
    float total_loss = 0.0f;
    const float eps = 1e-12f;

    for (size_t i = 0; i < n; ++i) {
        float p = pd[i];
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        float y = td[i];

        total_loss -= (y * logf(p) + (1.0f - y) * logf(1.0f - p));
        if (g != nullptr) g[i] = ((p - y) / (p * (1.0f - p))) / (float)n;
    }
    return total_loss / (float)n;
}

float rcre_loss_cross_entropy_logits(const RCRE_Tensor* logits, const size_t* targets, size_t batch_size, size_t num_classes, RCRE_Tensor* grad_out) {
    if (!loss_is_f32(logits) || !targets || batch_size == 0 || num_classes == 0) return 0.0f;

    const float* ld = rcre_tensor_f32_data(logits);
    float* g = (grad_out != nullptr) ? grad_out->grad : nullptr;

    float total_loss = 0.0f;
#if defined(RCRE_DEBUG)
    int warned_oob = 0;
#endif

    for (size_t b = 0; b < batch_size; ++b) {
        const float* row = ld + b * num_classes;
        float max_val = row[0];
        for (size_t c = 1; c < num_classes; ++c) {
            if (row[c] > max_val) max_val = row[c];
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            sum_exp += expf(row[c] - max_val);
        }

        /* Out-of-range labels: clamp (keeps training alive) and report once
         * per call under RCRE_DEBUG. Validate with rcre_data_validate_labels(). */
        size_t target_cls = targets[b];
        if (target_cls >= num_classes) {
            target_cls = num_classes - 1;
#if defined(RCRE_DEBUG)
            if (!warned_oob) {
                fprintf(stderr,
                        "[rcre] WARNING: rcre_loss_cross_entropy_logits: target[%zu]=%zu is out of "
                        "range [0,%zu); clamping. Validate labels before training.\n",
                        b, targets[b], num_classes);
                warned_oob = 1;
            }
#endif
        }

        float log_p = (row[target_cls] - max_val) - logf(sum_exp + 1e-12f);
        total_loss -= log_p;

        if (g != nullptr) {
            for (size_t c = 0; c < num_classes; ++c) {
                float prob = expf(row[c] - max_val) / (sum_exp + 1e-12f);
                g[b * num_classes + c] = (c == target_cls) ? (prob - 1.0f) / (float)batch_size : prob / (float)batch_size;
            }
        }
    }

    return total_loss / (float)batch_size;
}

float rcre_loss_kl_divergence(const RCRE_Tensor* mu, const RCRE_Tensor* logvar, RCRE_Tensor* grad_mu, RCRE_Tensor* grad_logvar) {
    if (!loss_is_f32(mu) || !loss_is_f32(logvar) ||
        mu->numel != logvar->numel || mu->numel == 0) return 0.0f;

    const float* md = rcre_tensor_f32_data(mu);
    const float* ldv = rcre_tensor_f32_data(logvar);
    float* gm = (grad_mu != nullptr) ? grad_mu->grad : nullptr;
    float* glv = (grad_logvar != nullptr) ? grad_logvar->grad : nullptr;

    size_t n = mu->numel;
    float kl_sum = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float m = md[i];
        float lv = ldv[i];
        float var = expf(lv);

        /* KL = -0.5 * sum(1 + log(var) - mu^2 - var) */
        kl_sum += -0.5f * (1.0f + lv - m * m - var);

        if (gm != nullptr) gm[i] = m / (float)n;
        if (glv != nullptr) glv[i] = 0.5f * (var - 1.0f) / (float)n;
    }
    return kl_sum / (float)n;
}

int rcre_data_validate_labels(const size_t* targets, size_t n, size_t num_classes) {
    if (!targets) return -1;
    for (size_t i = 0; i < n; ++i) {
        if (targets[i] >= num_classes) return (int)i;
    }
    return -1;
}

/* Scalar Autograd Losses */
RCRE_ScalarVal* rcre_scalar_loss_mse(RCRE_GraphCtx* ctx, RCRE_ScalarVal** preds, const double* targets, size_t count) {
    if (!preds || !targets || count == 0) return nullptr;
    RCRE_ScalarVal* total_loss = rcre_scalar_val_const(ctx, 0.0);
    RCRE_ScalarVal* two = rcre_scalar_val_const(ctx, 2.0);

    for (size_t i = 0; i < count; ++i) {
        RCRE_ScalarVal* t = rcre_scalar_val_const(ctx, targets[i]);
        RCRE_ScalarVal* diff = rcre_scalar_sub(ctx, preds[i], t);
        RCRE_ScalarVal* sq = rcre_scalar_pow(ctx, diff, two);
        total_loss = rcre_scalar_add(ctx, total_loss, sq);
    }
    RCRE_ScalarVal* n_val = rcre_scalar_val_const(ctx, (double)count);
    return rcre_scalar_div(ctx, total_loss, n_val);
}

RCRE_ScalarVal* rcre_scalar_loss_bce(RCRE_GraphCtx* ctx, RCRE_ScalarVal** preds, const double* targets, size_t count) {
    if (!preds || !targets || count == 0) return nullptr;
    RCRE_ScalarVal* total_loss = rcre_scalar_val_const(ctx, 0.0);
    RCRE_ScalarVal* one = rcre_scalar_val_const(ctx, 1.0);

    for (size_t i = 0; i < count; ++i) {
        RCRE_ScalarVal* p = preds[i];
        RCRE_ScalarVal* y = rcre_scalar_val_const(ctx, targets[i]);
        RCRE_ScalarVal* one_minus_y = rcre_scalar_val_const(ctx, 1.0 - targets[i]);

        RCRE_ScalarVal* log_p = rcre_scalar_log(ctx, p);
        RCRE_ScalarVal* one_minus_p = rcre_scalar_sub(ctx, one, p);
        RCRE_ScalarVal* log_one_minus_p = rcre_scalar_log(ctx, one_minus_p);

        RCRE_ScalarVal* term1 = rcre_scalar_mul(ctx, y, log_p);
        RCRE_ScalarVal* term2 = rcre_scalar_mul(ctx, one_minus_y, log_one_minus_p);
        RCRE_ScalarVal* sample_loss = rcre_scalar_neg(ctx, rcre_scalar_add(ctx, term1, term2));

        total_loss = rcre_scalar_add(ctx, total_loss, sample_loss);
    }
    RCRE_ScalarVal* n_val = rcre_scalar_val_const(ctx, (double)count);
    return rcre_scalar_div(ctx, total_loss, n_val);
}

RCRE_ScalarVal* rcre_scalar_loss_svm_max_margin(RCRE_GraphCtx* ctx, RCRE_ScalarVal** preds, const double* targets, size_t count) {
    if (!ctx || !preds || !targets || count == 0) return nullptr;
    RCRE_ScalarVal* total_loss = rcre_scalar_val_const(ctx, 0.0);
    RCRE_ScalarVal* one = rcre_scalar_val_const(ctx, 1.0);

    for (size_t i = 0; i < count; ++i) {
        RCRE_ScalarVal* y_target = rcre_scalar_val_const(ctx, targets[i]);
        RCRE_ScalarVal* score = preds[i];
        RCRE_ScalarVal* y_score = rcre_scalar_mul(ctx, y_target, score);
        RCRE_ScalarVal* diff = rcre_scalar_sub(ctx, one, y_score);
        RCRE_ScalarVal* margin = rcre_scalar_relu(ctx, diff);
        total_loss = rcre_scalar_add(ctx, total_loss, margin);
    }
    RCRE_ScalarVal* n_val = rcre_scalar_val_const(ctx, (double)count);
    return rcre_scalar_div(ctx, total_loss, n_val);
}
