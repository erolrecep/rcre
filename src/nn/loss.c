/**
 * @file loss.c
 * @brief Numerical and autograd loss functions: MSE, BCE, Cross-Entropy, and KL Divergence.
 * 
 * Part of RCRE Unified Engine.
 */

#include "rcre_autograd.h"
#include <math.h>
#include <string.h>

float rcre_loss_mse(const RCRE_AGTensor* preds, const RCRE_AGTensor* targets, RCRE_AGTensor* grad_out) {
    if (!preds || !targets || preds->numel != targets->numel || preds->numel == 0) return 0.0f;

    size_t n = preds->numel;
    float sum_sq = 0.0f;
    float scale = 2.0f / (float)n;

    for (size_t i = 0; i < n; ++i) {
        float diff = preds->data[i] - targets->data[i];
        sum_sq += diff * diff;
        float g = diff * scale;
        if (grad_out != nullptr) {
            if (grad_out->grad != nullptr) {
                grad_out->grad[i] = g;
            }
            if (grad_out->data != nullptr && grad_out != preds) {
                grad_out->data[i] = g;
            }
        }
    }
    return sum_sq / (float)n;
}

float rcre_loss_bce(const RCRE_AGTensor* preds, const RCRE_AGTensor* targets, RCRE_AGTensor* grad_out) {
    if (!preds || !targets || preds->numel != targets->numel || preds->numel == 0) return 0.0f;

    size_t n = preds->numel;
    float total_loss = 0.0f;
    const float eps = 1e-12f;

    for (size_t i = 0; i < n; ++i) {
        float p = preds->data[i];
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        float y = targets->data[i];

        total_loss -= (y * logf(p) + (1.0f - y) * logf(1.0f - p));
        if (grad_out != nullptr) {
            float g = ((p - y) / (p * (1.0f - p))) / (float)n;
            if (grad_out->grad != nullptr) {
                grad_out->grad[i] = g;
            }
            if (grad_out->data != nullptr && grad_out != preds) {
                grad_out->data[i] = g;
            }
        }
    }
    return total_loss / (float)n;
}

float rcre_loss_cross_entropy_logits(const RCRE_AGTensor* logits, const size_t* targets, size_t batch_size, size_t num_classes, RCRE_AGTensor* grad_out) {
    if (!logits || !targets || batch_size == 0 || num_classes == 0) return 0.0f;

    float total_loss = 0.0f;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* row = logits->data + b * num_classes;
        float max_val = row[0];
        for (size_t c = 1; c < num_classes; ++c) {
            if (row[c] > max_val) max_val = row[c];
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            sum_exp += expf(row[c] - max_val);
        }

        size_t target_cls = targets[b];
        if (target_cls >= num_classes) target_cls = num_classes - 1;

        float log_p = (row[target_cls] - max_val) - logf(sum_exp + 1e-12f);
        total_loss -= log_p;

        if (grad_out != nullptr) {
            for (size_t c = 0; c < num_classes; ++c) {
                float prob = expf(row[c] - max_val) / (sum_exp + 1e-12f);
                float g = (c == target_cls) ? (prob - 1.0f) / (float)batch_size : prob / (float)batch_size;
                if (grad_out->grad != nullptr) {
                    grad_out->grad[b * num_classes + c] = g;
                }
                if (grad_out->data != nullptr && grad_out != logits) {
                    grad_out->data[b * num_classes + c] = g;
                }
            }
        }
    }

    return total_loss / (float)batch_size;
}

float rcre_loss_kl_divergence(const RCRE_AGTensor* mu, const RCRE_AGTensor* logvar, RCRE_AGTensor* grad_mu, RCRE_AGTensor* grad_logvar) {
    if (!mu || !logvar || mu->numel != logvar->numel || mu->numel == 0) return 0.0f;

    size_t n = mu->numel;
    float kl_sum = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float m = mu->data[i];
        float lv = logvar->data[i];
        float var = expf(lv);

        /* KL = -0.5 * sum(1 + log(var) - mu^2 - var) */
        kl_sum += -0.5f * (1.0f + lv - m * m - var);

        if (grad_mu != nullptr) {
            float gm = m / (float)n;
            if (grad_mu->grad != nullptr) grad_mu->grad[i] = gm;
            if (grad_mu->data != nullptr && grad_mu != mu) grad_mu->data[i] = gm;
        }
        if (grad_logvar != nullptr) {
            float glv = 0.5f * (var - 1.0f) / (float)n;
            if (grad_logvar->grad != nullptr) grad_logvar->grad[i] = glv;
            if (grad_logvar->data != nullptr && grad_logvar != logvar) grad_logvar->data[i] = glv;
        }
    }
    return kl_sum / (float)n;
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
