/**
 * @file optim.c
 * @brief RCRE Decoupled Weight Decay AdamW Optimizer & Learning Rate Schedule (ISO C23)
 */

#include "transformer_internal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool rcre_gpt_adamw_config_valid(const RCRE_GPTAdamWConfig *cfg) {
    return cfg != nullptr &&
           isfinite(cfg->lr_max) && cfg->lr_max > 0.0f &&
           isfinite(cfg->lr_min) && cfg->lr_min >= 0.0f && cfg->lr_min <= cfg->lr_max &&
           isfinite(cfg->beta1) && cfg->beta1 >= 0.0f && cfg->beta1 < 1.0f &&
           isfinite(cfg->beta2) && cfg->beta2 >= 0.0f && cfg->beta2 < 1.0f &&
           isfinite(cfg->eps) && cfg->eps > 0.0f &&
           isfinite(cfg->weight_decay) &&
           isfinite(cfg->grad_clip) && cfg->grad_clip >= 0.0f &&
           cfg->warmup_steps >= 0 && cfg->max_steps > cfg->warmup_steps;
}

float rcre_gpt_get_scheduled_lr(const RCRE_GPTAdamWConfig *cfg, int step) {
    if (cfg->max_steps <= cfg->warmup_steps) {
        return cfg->lr_min; /* degenerate schedule: hold the floor */
    }
    if (step < cfg->warmup_steps) {
        return cfg->lr_max * ((float)(step + 1) / (float)cfg->warmup_steps);
    }
    if (step >= cfg->max_steps) {
        return cfg->lr_min;
    }
    float progress = (float)(step - cfg->warmup_steps) / (float)(cfg->max_steps - cfg->warmup_steps);
    float cosine_coeff = 0.5f * (1.0f + cosf((float)M_PI * progress));
    return cfg->lr_min + cosine_coeff * (cfg->lr_max - cfg->lr_min);
}

float rcre_gpt_clip_gradients(float *grads, size_t num_params, float max_norm) {
    float norm_sq = 0.0f;
    for (size_t i = 0; i < num_params; i++) {
        float g = grads[i];
        norm_sq += g * g;
    }
    float norm = sqrtf(norm_sq);
    if (norm > max_norm && norm > 1e-6f) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < num_params; i++) {
            grads[i] *= scale;
        }
    }
    return norm;
}

void rcre_gpt_adamw_step(RCRE_GPTModel *model, RCRE_GPTAdamWConfig *cfg) {
    if (!model || !cfg) return;
    if (!rcre_gpt_adamw_config_valid(cfg)) {
        /*
         * Invalid hyperparameters (e.g. beta1 == 1.0) would divide by zero in
         * the bias correction and turn every parameter NaN. Refuse the step
         * instead of corrupting the model.
         */
#ifdef RCRE_DEBUG
        fprintf(stderr, "[rcre][warn] rcre_gpt_adamw_step: invalid config; step skipped\n");
#endif
        return;
    }
    cfg->current_step++;
    int t = cfg->current_step;
    float lr = rcre_gpt_get_scheduled_lr(cfg, t);

    /* 1. Gradient Clipping */
    if (cfg->grad_clip > 0.0f) {
        rcre_gpt_clip_gradients(model->grads, model->num_params, cfg->grad_clip);
    }

    /* 2. Bias correction terms */
    float beta1 = cfg->beta1;
    float beta2 = cfg->beta2;
    float beta1_pow = powf(beta1, (float)t);
    float beta2_pow = powf(beta2, (float)t);
    float alpha = lr * sqrtf(1.0f - beta2_pow) / (1.0f - beta1_pow);
    float eps = cfg->eps;
    float wd = cfg->weight_decay;

    /* 3. Parameter updates */
    float *params = model->params;
    float *grads  = model->grads;
    float *m      = model->m;
    float *v      = model->v;
    size_t N      = model->num_params;

    for (size_t i = 0; i < N; i++) {
        float g = grads[i];
        float p = params[i];

        p -= lr * wd * p;
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);
        p -= alpha * m[i] / (sqrtf(v[i]) + eps);

        params[i] = p;
    }
}
