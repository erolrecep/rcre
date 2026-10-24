/**
 * @file optim.c
 * @brief High-performance numerical and autograd optimizers: SGD w/ Momentum & AdamW.
 * 
 * Part of RCRE Unified Engine.
 */

#include "rcre_autograd.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

RCRE_TensorOptim* rcre_tensor_optim_sgd_create(RCRE_Arena* arena, RCRE_AGTensor** params, size_t num_params, float lr, float momentum, float weight_decay) {
    if (!params || num_params == 0) return nullptr;

    RCRE_TensorOptim* opt = nullptr;
    if (arena != nullptr) {
        opt = (RCRE_TensorOptim*)rcre_arena_alloc(arena, sizeof(RCRE_TensorOptim), alignof(RCRE_TensorOptim));
    } else {
        opt = (RCRE_TensorOptim*)malloc(sizeof(RCRE_TensorOptim));
    }
    if (!opt) return nullptr;

    opt->type = RCRE_OPTIM_SGD;
    opt->params = params;
    opt->num_params = num_params;
    opt->lr = lr;
    opt->weight_decay = weight_decay;
    opt->momentum = momentum;

    if (arena != nullptr) {
        opt->v_momentum = (float**)rcre_arena_alloc(arena, num_params * sizeof(float*), alignof(float*));
    } else {
        opt->v_momentum = (float**)malloc(num_params * sizeof(float*));
    }
    if (!opt->v_momentum) return nullptr;

    for (size_t p = 0; p < num_params; ++p) {
        size_t bytes = params[p]->numel * sizeof(float);
        if (arena != nullptr) {
            opt->v_momentum[p] = (float*)rcre_arena_alloc(arena, bytes, 64);
        } else {
            opt->v_momentum[p] = (float*)aligned_alloc(64, (bytes + 63) & ~63ULL);
        }
        if (opt->v_momentum[p]) {
            memset(opt->v_momentum[p], 0, bytes);
        }
    }
    return opt;
}

RCRE_TensorOptim* rcre_tensor_optim_adamw_create(RCRE_Arena* arena, RCRE_AGTensor** params, size_t num_params, float lr, float beta1, float beta2, float eps, float weight_decay) {
    if (!params || num_params == 0) return nullptr;

    RCRE_TensorOptim* opt = nullptr;
    if (arena != nullptr) {
        opt = (RCRE_TensorOptim*)rcre_arena_alloc(arena, sizeof(RCRE_TensorOptim), alignof(RCRE_TensorOptim));
    } else {
        opt = (RCRE_TensorOptim*)malloc(sizeof(RCRE_TensorOptim));
    }
    if (!opt) return nullptr;

    opt->type = RCRE_OPTIM_ADAMW;
    opt->params = params;
    opt->num_params = num_params;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->step_count = 0;

    if (arena != nullptr) {
        opt->m = (float**)rcre_arena_alloc(arena, num_params * sizeof(float*), alignof(float*));
        opt->v = (float**)rcre_arena_alloc(arena, num_params * sizeof(float*), alignof(float*));
    } else {
        opt->m = (float**)malloc(num_params * sizeof(float*));
        opt->v = (float**)malloc(num_params * sizeof(float*));
    }
    if (!opt->m || !opt->v) return nullptr;

    for (size_t p = 0; p < num_params; ++p) {
        size_t bytes = params[p]->numel * sizeof(float);
        if (arena != nullptr) {
            opt->m[p] = (float*)rcre_arena_alloc(arena, bytes, 64);
            opt->v[p] = (float*)rcre_arena_alloc(arena, bytes, 64);
        } else {
            opt->m[p] = (float*)aligned_alloc(64, (bytes + 63) & ~63ULL);
            opt->v[p] = (float*)aligned_alloc(64, (bytes + 63) & ~63ULL);
        }
        if (opt->m[p]) memset(opt->m[p], 0, bytes);
        if (opt->v[p]) memset(opt->v[p], 0, bytes);
    }
    return opt;
}

void rcre_tensor_optim_step(RCRE_TensorOptim* opt) {
    if (opt == nullptr) return;

    if (opt->type == RCRE_OPTIM_SGD) {
        for (size_t p = 0; p < opt->num_params; ++p) {
            RCRE_AGTensor* t = opt->params[p];
            float* v_mom = opt->v_momentum[p];
            for (size_t i = 0; i < t->numel; ++i) {
                float g = t->grad[i] + opt->weight_decay * t->data[i];
                v_mom[i] = opt->momentum * v_mom[i] + g;
                t->data[i] -= opt->lr * v_mom[i];
            }
        }
    } else if (opt->type == RCRE_OPTIM_ADAMW) {
        opt->step_count++;
        float b1 = opt->beta1;
        float b2 = opt->beta2;
        float lr = opt->lr;
        float wd = opt->weight_decay;
        float eps = opt->eps;

        float bias_corr1 = 1.0f - powf(b1, (float)opt->step_count);
        float bias_corr2 = 1.0f - powf(b2, (float)opt->step_count);

        for (size_t p = 0; p < opt->num_params; ++p) {
            RCRE_AGTensor* t = opt->params[p];
            float* m_arr = opt->m[p];
            float* v_arr = opt->v[p];

            for (size_t i = 0; i < t->numel; ++i) {
                float g = t->grad[i];
                /* Decoupled weight decay */
                t->data[i] -= lr * wd * t->data[i];

                m_arr[i] = b1 * m_arr[i] + (1.0f - b1) * g;
                v_arr[i] = b2 * v_arr[i] + (1.0f - b2) * (g * g);

                float m_hat = m_arr[i] / bias_corr1;
                float v_hat = v_arr[i] / bias_corr2;

                t->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
            }
        }
    }
}

void rcre_tensor_optim_zero_grad(RCRE_TensorOptim* opt) {
    if (opt == nullptr) return;
    for (size_t p = 0; p < opt->num_params; ++p) {
        rcre_ag_tensor_zero_grad(opt->params[p]);
    }
}

/* Scalar Autograd Optimizer */
RCRE_ScalarOptim* rcre_scalar_optim_adamw_create(RCRE_Arena* arena, RCRE_ScalarVal** params, size_t num_params, double lr, double beta1, double beta2, double eps, double weight_decay) {
    if (!params || num_params == 0) return nullptr;

    RCRE_ScalarOptim* opt = nullptr;
    if (arena != nullptr) {
        opt = (RCRE_ScalarOptim*)rcre_arena_alloc(arena, sizeof(RCRE_ScalarOptim), alignof(RCRE_ScalarOptim));
    } else {
        opt = (RCRE_ScalarOptim*)malloc(sizeof(RCRE_ScalarOptim));
    }
    if (!opt) return nullptr;

    opt->type = RCRE_OPTIM_ADAMW;
    opt->params = params;
    opt->num_params = num_params;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->step_count = 0;

    if (arena != nullptr) {
        opt->m = (double*)rcre_arena_alloc(arena, num_params * sizeof(double), 64);
        opt->v = (double*)rcre_arena_alloc(arena, num_params * sizeof(double), 64);
    } else {
        opt->m = (double*)aligned_alloc(64, (num_params * sizeof(double) + 63) & ~63ULL);
        opt->v = (double*)aligned_alloc(64, (num_params * sizeof(double) + 63) & ~63ULL);
    }
    if (!opt->m || !opt->v) return nullptr;

    memset(opt->m, 0, num_params * sizeof(double));
    memset(opt->v, 0, num_params * sizeof(double));
    return opt;
}

void rcre_scalar_optim_step(RCRE_ScalarOptim* opt) {
    if (opt == nullptr) return;
    opt->step_count++;

    double b1 = opt->beta1;
    double b2 = opt->beta2;
    double lr = opt->lr;
    double wd = opt->weight_decay;
    double eps = opt->eps;

    double bias_corr1 = 1.0 - pow(b1, (double)opt->step_count);
    double bias_corr2 = 1.0 - pow(b2, (double)opt->step_count);

    for (size_t i = 0; i < opt->num_params; ++i) {
        RCRE_ScalarVal* p = opt->params[i];
        double g = p->grad;

        /* Decoupled weight decay */
        p->data -= lr * wd * p->data;

        opt->m[i] = b1 * opt->m[i] + (1.0 - b1) * g;
        opt->v[i] = b2 * opt->v[i] + (1.0 - b2) * (g * g);

        double m_hat = opt->m[i] / bias_corr1;
        double v_hat = opt->v[i] / bias_corr2;

        p->data -= lr * m_hat / (sqrt(v_hat) + eps);
    }
}

void rcre_scalar_optim_zero_grad(RCRE_ScalarOptim* opt) {
    if (opt == nullptr) return;
    for (size_t i = 0; i < opt->num_params; ++i) {
        opt->params[i]->grad = 0.0;
    }
}
