/**
 * @file gen_diff_sample_int.h
 * @brief Shared DDPM / DDIM reverse-diffusion steppers (internal, gen module).
 *
 * The Tiny U-Net (diffusion.c) and the DiT (vit_dit.c) share one exact
 * sampler implementation; the only per-architecture difference is the
 * epsilon-prediction forward pass, injected as a function pointer. Keeping
 * the DDPM/DDIM schedule arithmetic in exactly one place prevents the two
 * samplers from drifting apart (they must stay bit-identical for a fixed
 * seed — the determinism contract, verified in tests/test_gen.c).
 *
 * All temporary allocations use caller-supplied arenas; the per-step tape
 * is created on `scratch` and rewound (arena temp) after each step, so the
 * steady state is zero-allocation per step.
 */

#ifndef RCRE_GEN_DIFF_SAMPLE_INT_H
#define RCRE_GEN_DIFF_SAMPLE_INT_H

#include "rcre_gen.h"
#include "rcre_rng.h"
#include <math.h>

typedef RCRE_Tensor* (*gen_diff_fwd_fn)(const void* model, RCRE_TensorTape* tape,
                                        RCRE_Tensor* x, const float* timesteps, void* ctx);

/* Ancestral DDPM: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-abar_t) * eps)
 *                  + sqrt(posterior_variance_t) * z,   z ~ N(0, I) (t > 0)
 * The final step (t = 0) uses the deterministic mean clamped to [-1, 1]. */
static inline RCRE_Tensor* gen_diff_sample_ddpm(
    const void* model, gen_diff_fwd_fn fwd, void* ctx,
    const RCRE_DiffSchedule* sched, size_t in_ch, size_t H, size_t W,
    RCRE_Arena* out_arena, RCRE_Arena* scratch, size_t num_samples,
    uint64_t* rng, size_t tape_cap)
{
    if (!model || !sched || !out_arena || !scratch || num_samples == 0 || !rng) return nullptr;

    RCRE_Tensor* x = rcre_tensor_alloc_4d(out_arena, num_samples, in_ch, H, W);
    if (!x) return nullptr;
    rcre_tensor_fill_normal(x, 0.0f, 1.0f, rng);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(scratch, tape_cap);
    if (!tape) return nullptr;
    float* t_flt = (float*)rcre_arena_alloc(scratch, num_samples * sizeof(float), 16);
    if (!t_flt) {
        rcre_tensor_tape_destroy(tape);
        return nullptr;
    }

    float* xd = rcre_tensor_f32_data(x);
    for (size_t step = sched->num_steps; step > 0; --step) {
        size_t t = step - 1U;
        for (size_t n = 0; n < num_samples; ++n) t_flt[n] = (float)t;

        RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
        rcre_tensor_tape_reset(tape);
        RCRE_Tensor* eps = fwd(model, tape, x, t_flt, ctx);
        if (!eps) {
            rcre_arena_temp_end(tmp);
            rcre_tensor_tape_destroy(tape);
            return nullptr;
        }

        const float* ed = rcre_tensor_f32_data(eps);
        float inv_sqrt_alpha = 1.0f / sqrtf(sched->alphas[t]);
        float beta_over_som  = sched->betas[t] / (sched->sqrt_one_minus_alpha_bar[t] + 1e-12f);
        float sigma          = (t > 0) ? sqrtf(sched->posterior_variance[t]) : 0.0f;

        for (size_t i = 0; i < x->numel; ++i) {
            float mean = inv_sqrt_alpha * (xd[i] - beta_over_som * ed[i]);
            if (t > 0) {
                xd[i] = mean + sigma * rcre_rng_gaussian_f32(rng, 0.0f, 1.0f);
            } else {
                if (mean < -1.0f) mean = -1.0f;
                if (mean >  1.0f) mean =  1.0f;
                xd[i] = mean;
            }
        }
        rcre_arena_temp_end(tmp);
    }

    rcre_tensor_tape_destroy(tape);
    return x;
}

/* DDIM sampling (eta = 0 is the deterministic ODE). Sub-step mapping:
 * t_k = (T-1)(K-1-k)/(K-1); x0_hat clamped to [-1, 1];
 * x_{t_prev} = sqrt(abar_prev) * x0_hat + sqrt(1 - abar_prev - sigma^2) * eps
 *             + sigma * z,  sigma = eta * sqrt(((1-abar_prev)/(1-abar_t)) * (1 - abar_t/abar_prev)). */
static inline RCRE_Tensor* gen_diff_sample_ddim(
    const void* model, gen_diff_fwd_fn fwd, void* ctx,
    const RCRE_DiffSchedule* sched, size_t in_ch, size_t H, size_t W,
    RCRE_Arena* out_arena, RCRE_Arena* scratch, size_t num_samples,
    size_t ddim_steps, float eta, uint64_t* rng, size_t tape_cap)
{
    if (!model || !sched || !out_arena || !scratch || num_samples == 0 ||
        ddim_steps == 0 || ddim_steps > sched->num_steps || !rng) {
        return nullptr;
    }

    RCRE_Tensor* x = rcre_tensor_alloc_4d(out_arena, num_samples, in_ch, H, W);
    if (!x) return nullptr;
    rcre_tensor_fill_normal(x, 0.0f, 1.0f, rng);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(scratch, tape_cap);
    if (!tape) return nullptr;
    float* t_flt = (float*)rcre_arena_alloc(scratch, num_samples * sizeof(float), 16);
    if (!t_flt) {
        rcre_tensor_tape_destroy(tape);
        return nullptr;
    }

    float* xd = rcre_tensor_f32_data(x);
    size_t T = sched->num_steps;

    for (size_t k = 0; k < ddim_steps; ++k) {
        size_t t = (ddim_steps == 1)
                       ? (T - 1U)
                       : ((T - 1U) * (ddim_steps - 1U - k)) / (ddim_steps - 1U);
        bool is_last = (k + 1U == ddim_steps);
        size_t t_prev = is_last ? 0U : ((T - 1U) * (ddim_steps - 2U - k)) / (ddim_steps - 1U);

        for (size_t n = 0; n < num_samples; ++n) t_flt[n] = (float)t;

        RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
        rcre_tensor_tape_reset(tape);
        RCRE_Tensor* eps = fwd(model, tape, x, t_flt, ctx);
        if (!eps) {
            rcre_arena_temp_end(tmp);
            rcre_tensor_tape_destroy(tape);
            return nullptr;
        }

        const float* ed = rcre_tensor_f32_data(eps);
        float ab_t    = sched->alpha_bar[t];
        float ab_prev = is_last ? 1.0f : sched->alpha_bar[t_prev];
        float sa_t    = sqrtf(ab_t);
        float som_t   = sqrtf(fmaxf(0.0f, 1.0f - ab_t));
        float sa_prev = sqrtf(ab_prev);

        float sigma = 0.0f;
        if (!is_last && eta > 0.0f) {
            float ratio = (1.0f - ab_prev) / fmaxf(1e-12f, 1.0f - ab_t);
            float inner = ratio * fmaxf(0.0f, 1.0f - ab_t / fmaxf(1e-12f, ab_prev));
            sigma = eta * sqrtf(fmaxf(0.0f, inner));
        }
        float dir_coeff = sqrtf(fmaxf(0.0f, 1.0f - ab_prev - sigma * sigma));

        for (size_t i = 0; i < x->numel; ++i) {
            float x0_hat = (xd[i] - som_t * ed[i]) / fmaxf(1e-12f, sa_t);
            if (x0_hat < -1.0f) x0_hat = -1.0f;
            if (x0_hat >  1.0f) x0_hat =  1.0f;
            float nxt = sa_prev * x0_hat + dir_coeff * ed[i];
            if (sigma > 0.0f) nxt += sigma * rcre_rng_gaussian_f32(rng, 0.0f, 1.0f);
            xd[i] = nxt;
        }
        rcre_arena_temp_end(tmp);
    }

    rcre_tensor_tape_destroy(tape);
    return x;
}

#endif /* RCRE_GEN_DIFF_SAMPLE_INT_H */
