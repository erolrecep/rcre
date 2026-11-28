/**
 * @file diffusion.c
 * @brief Phase 3: DDPM / DDIM Noise Scheduler & Time-Conditioned Tiny U-Net
 *
 * Part of Recep's Code Run Everywhere (RCRE).
 * Strict ISO C23 compliant, zero external dependencies, zero hot-loop mallocs.
 */

#include "rcre_gen.h"
#include "gen_diff_sample_int.h"
#include "rcre_rng.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Epsilon-prediction forward adapter for the shared DDPM/DDIM steppers. */
static RCRE_Tensor* gen_tinyunet_fwd_wrap(const void* model, RCRE_TensorTape* tape,
                                          RCRE_Tensor* x, const float* t, void* ctx) {
    (void)ctx;
    return rcre_tinyunet_forward((const RCRE_TinyUNet*)model, tape, x, t);
}

/* ========================================================================= */
/* 1. Diffusion Noise Schedule (Linear & Cosine)                             */
/* ========================================================================= */

static float gen_diff_cosine_f(size_t t, size_t T) {
    const double s = 0.008;
    const double pi = 3.14159265358979323846;
    double frac = ((double)t / (double)T + s) / (1.0 + s);
    double c = cos(frac * (pi * 0.5));
    return (float)(c * c);
}

RCRE_DiffSchedule* rcre_diff_schedule_create(
    RCRE_Arena* arena, size_t num_steps, RCRE_DiffSchedType type,
    float beta_start, float beta_end)
{
    if (!arena || num_steps < 2) return nullptr;
    if (type == RCRE_DIFF_SCHED_LINEAR && (!(beta_start > 0.0f) || !(beta_end > beta_start) || beta_end >= 1.0f)) {
        return nullptr;
    }

    RCRE_DiffSchedule* s = (RCRE_DiffSchedule*)rcre_arena_alloc(
        arena, sizeof(RCRE_DiffSchedule), alignof(RCRE_DiffSchedule));
    if (!s) return nullptr;

    s->num_steps                = num_steps;
    s->type                     = type;
    s->betas                    = (float*)rcre_arena_alloc(arena, num_steps * sizeof(float), 64);
    s->alphas                   = (float*)rcre_arena_alloc(arena, num_steps * sizeof(float), 64);
    s->alpha_bar                = (float*)rcre_arena_alloc(arena, num_steps * sizeof(float), 64);
    s->sqrt_alpha_bar           = (float*)rcre_arena_alloc(arena, num_steps * sizeof(float), 64);
    s->sqrt_one_minus_alpha_bar = (float*)rcre_arena_alloc(arena, num_steps * sizeof(float), 64);
    s->posterior_variance       = (float*)rcre_arena_alloc(arena, num_steps * sizeof(float), 64);

    if (!s->betas || !s->alphas || !s->alpha_bar ||
        !s->sqrt_alpha_bar || !s->sqrt_one_minus_alpha_bar || !s->posterior_variance) {
        return nullptr;
    }

    if (type == RCRE_DIFF_SCHED_LINEAR) {
        for (size_t t = 0; t < num_steps; ++t) {
            float frac = (float)t / (float)(num_steps - 1U);
            s->betas[t] = beta_start + frac * (beta_end - beta_start);
        }
    } else {
        float f0 = gen_diff_cosine_f(0, num_steps);
        float ab_prev = 1.0f;
        for (size_t t = 0; t < num_steps; ++t) {
            float ab_next = gen_diff_cosine_f(t + 1U, num_steps) / f0;
            float b = 1.0f - (ab_next / ab_prev);
            if (b < 1e-4f) b = 1e-4f;
            if (b > 0.999f) b = 0.999f;
            s->betas[t] = b;
            ab_prev = ab_next;
        }
    }

    double prod = 1.0;
    for (size_t t = 0; t < num_steps; ++t) {
        s->alphas[t] = 1.0f - s->betas[t];
        prod *= (double)s->alphas[t];
        s->alpha_bar[t] = (float)prod;
        s->sqrt_alpha_bar[t] = sqrtf(s->alpha_bar[t]);
        float omab = 1.0f - s->alpha_bar[t];
        if (omab < 0.0f) omab = 0.0f;
        s->sqrt_one_minus_alpha_bar[t] = sqrtf(omab);

        if (t == 0) {
            s->posterior_variance[0] = s->betas[0];
        } else {
            float denom = 1.0f - s->alpha_bar[t];
            float pv = s->betas[t] * (1.0f - s->alpha_bar[t - 1U]) / (denom > 1e-12f ? denom : 1e-12f);
            s->posterior_variance[t] = pv > 1e-20f ? pv : 1e-20f;
        }
    }

    return s;
}

int rcre_diff_q_sample(
    const RCRE_DiffSchedule* sched,
    const RCRE_Tensor* x0,
    const size_t* t_indices,
    const RCRE_Tensor* noise,
    RCRE_Tensor* x_t_out)
{
    if (!sched || !x0 || !t_indices || !noise || !x_t_out ||
        x0->ndim != 4 || x0->numel != noise->numel || x0->numel != x_t_out->numel) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t N = x0->shape[0];
    size_t sample_elems = x0->numel / N;
    const float* x0d = rcre_tensor_f32_data(x0);
    const float* nd  = rcre_tensor_f32_data(noise);
    float* xtd       = rcre_tensor_f32_data(x_t_out);

    for (size_t n = 0; n < N; ++n) {
        size_t t = t_indices[n];
        if (t >= sched->num_steps) return RCRE_ERR_INVALID_ARG;
        float sa  = sched->sqrt_alpha_bar[t];
        float som = sched->sqrt_one_minus_alpha_bar[t];
        const float* x0_n = x0d + n * sample_elems;
        const float* e_n  = nd  + n * sample_elems;
        float* xt_n       = xtd + n * sample_elems;
        for (size_t i = 0; i < sample_elems; ++i) {
            xt_n[i] = sa * x0_n[i] + som * e_n[i];
        }
    }
    return RCRE_OK;
}

/* ========================================================================= */
/* 2. Time-Conditioned Tiny U-Net (GroupNorm + SiLU + FiLM + Skip Conns)     */
/* ========================================================================= */

static const char* gen_diff_fmt_label(RCRE_Arena* arena, const char* prefix, const char* suffix) {
    char* buf = (char*)rcre_arena_alloc(arena, 48, 8);
    if (!buf) return nullptr;
    snprintf(buf, 48, "unet.%s.%s", prefix, suffix);
    return buf;
}

static bool gen_diff_resblock_init(
    RCRE_Arena* arena, RCRE_UNetResBlock* blk,
    size_t in_ch, size_t out_ch, size_t num_groups, size_t time_dim,
    const char* name, uint64_t* rng)
{
    memset(blk, 0, sizeof(*blk));
    blk->in_ch      = in_ch;
    blk->out_ch     = out_ch;
    blk->num_groups = num_groups;

    blk->gn1_g   = rcre_tensor_alloc_1d(arena, in_ch);
    blk->gn1_b   = rcre_tensor_alloc_1d(arena, in_ch);
    blk->conv1_w = rcre_tensor_alloc_4d(arena, out_ch, in_ch, 3, 3);
    blk->conv1_b = rcre_tensor_alloc_1d(arena, out_ch);
    blk->time_w  = rcre_tensor_alloc_2d(arena, time_dim, 2U * out_ch);
    blk->time_b  = rcre_tensor_alloc_1d(arena, 2U * out_ch);
    blk->gn2_g   = rcre_tensor_alloc_1d(arena, out_ch);
    blk->gn2_b   = rcre_tensor_alloc_1d(arena, out_ch);
    blk->conv2_w = rcre_tensor_alloc_4d(arena, out_ch, out_ch, 3, 3);
    blk->conv2_b = rcre_tensor_alloc_1d(arena, out_ch);

    if (!blk->gn1_g || !blk->gn1_b || !blk->conv1_w || !blk->conv1_b ||
        !blk->time_w || !blk->time_b || !blk->gn2_g || !blk->gn2_b ||
        !blk->conv2_w || !blk->conv2_b) {
        return false;
    }

    rcre_tensor_ones(blk->gn1_g);
    rcre_tensor_ones(blk->gn2_g);
    rcre_tensor_init_he(blk->conv1_w, in_ch * 9U, rng);
    rcre_tensor_init_xavier(blk->time_w, time_dim, 2U * out_ch, rng);
    /* Small residual branch output init improves early U-Net stability */
    rcre_tensor_fill_normal(blk->conv2_w, 0.0f, 0.02f, rng);

    blk->gn1_g->label   = gen_diff_fmt_label(arena, name, "gn1.g");
    blk->gn1_b->label   = gen_diff_fmt_label(arena, name, "gn1.b");
    blk->conv1_w->label = gen_diff_fmt_label(arena, name, "conv1.w");
    blk->conv1_b->label = gen_diff_fmt_label(arena, name, "conv1.b");
    blk->time_w->label  = gen_diff_fmt_label(arena, name, "time.w");
    blk->time_b->label  = gen_diff_fmt_label(arena, name, "time.b");
    blk->gn2_g->label   = gen_diff_fmt_label(arena, name, "gn2.g");
    blk->gn2_b->label   = gen_diff_fmt_label(arena, name, "gn2.b");
    blk->conv2_w->label = gen_diff_fmt_label(arena, name, "conv2.w");
    blk->conv2_b->label = gen_diff_fmt_label(arena, name, "conv2.b");

    if (in_ch != out_ch) {
        blk->skip_w = rcre_tensor_alloc_4d(arena, out_ch, in_ch, 1, 1);
        blk->skip_b = rcre_tensor_alloc_1d(arena, out_ch);
        if (!blk->skip_w || !blk->skip_b) return false;
        rcre_tensor_init_he(blk->skip_w, in_ch, rng);
        blk->skip_w->label = gen_diff_fmt_label(arena, name, "skip.w");
        blk->skip_b->label = gen_diff_fmt_label(arena, name, "skip.b");
    }
    return true;
}

static size_t gen_diff_resblock_collect(const RCRE_UNetResBlock* blk, RCRE_Tensor** out, size_t pos, size_t max_p) {
    RCRE_Tensor* list[12] = {
        blk->gn1_g, blk->gn1_b, blk->conv1_w, blk->conv1_b,
        blk->time_w, blk->time_b, blk->gn2_g, blk->gn2_b,
        blk->conv2_w, blk->conv2_b, blk->skip_w, blk->skip_b
    };
    size_t count = (blk->skip_w != nullptr) ? 12U : 10U;
    for (size_t i = 0; i < count && pos < max_p; ++i) {
        out[pos++] = list[i];
    }
    return pos;
}

static RCRE_Tensor* gen_diff_resblock_forward(
    const RCRE_UNetResBlock* blk, RCRE_TensorTape* tape, RCRE_Tensor* x, RCRE_Tensor* t_emb)
{
    RCRE_Tensor* h = rcre_autograd_groupnorm(tape, x, blk->gn1_g, blk->gn1_b, blk->num_groups, 1e-5f);
    h = rcre_autograd_silu(tape, h);
    h = rcre_autograd_conv2d(tape, h, blk->conv1_w, 1, 1);
    h = rcre_autograd_add_bias_4d(tape, h, blk->conv1_b);

    /* FiLM time conditioning: scale & shift projected from SiLU(t_emb) */
    RCRE_Tensor* ts = rcre_autograd_silu(tape, t_emb);
    RCRE_Tensor* ss = rcre_autograd_linear(tape, ts, blk->time_w, blk->time_b);
    RCRE_Tensor* scale = rcre_autograd_slice_cols(tape, ss, 0, blk->out_ch);
    RCRE_Tensor* shift = rcre_autograd_slice_cols(tape, ss, blk->out_ch, blk->out_ch);

    h = rcre_autograd_groupnorm(tape, h, blk->gn2_g, blk->gn2_b, blk->num_groups, 1e-5f);
    h = rcre_autograd_film_4d(tape, h, scale, shift);
    h = rcre_autograd_silu(tape, h);
    h = rcre_autograd_conv2d(tape, h, blk->conv2_w, 1, 1);
    h = rcre_autograd_add_bias_4d(tape, h, blk->conv2_b);

    RCRE_Tensor* res = x;
    if (blk->skip_w != nullptr) {
        res = rcre_autograd_conv2d(tape, x, blk->skip_w, 1, 0);
        res = rcre_autograd_add_bias_4d(tape, res, blk->skip_b);
    }
    return rcre_autograd_add(tape, h, res);
}

RCRE_TinyUNet* rcre_tinyunet_create(RCRE_Arena* arena, RCRE_TinyUNetConfig cfg, uint64_t* rng) {
    if (!arena || !rng || cfg.in_channels == 0 || cfg.base_channels == 0 ||
        cfg.num_groups == 0 || (cfg.base_channels % cfg.num_groups) != 0 ||
        cfg.time_dim < 2 || (cfg.time_dim & 1U) != 0 ||
        cfg.height < 4 || cfg.width < 4 ||
        (cfg.height % 4U) != 0 || (cfg.width % 4U) != 0) {
        return nullptr;
    }

    RCRE_TinyUNet* m = (RCRE_TinyUNet*)rcre_arena_alloc(arena, sizeof(RCRE_TinyUNet), alignof(RCRE_TinyUNet));
    if (!m) return nullptr;
    memset(m, 0, sizeof(*m));
    m->cfg = cfg;

    size_t C  = cfg.base_channels;
    size_t C2 = C * 2U;
    size_t G  = cfg.num_groups;
    size_t TD = cfg.time_dim;

    m->time_fc1_w = rcre_tensor_alloc_2d(arena, TD, TD);
    m->time_fc1_b = rcre_tensor_alloc_1d(arena, TD);
    m->time_fc2_w = rcre_tensor_alloc_2d(arena, TD, TD);
    m->time_fc2_b = rcre_tensor_alloc_1d(arena, TD);
    m->in_conv_w  = rcre_tensor_alloc_4d(arena, C, cfg.in_channels, 3, 3);
    m->in_conv_b  = rcre_tensor_alloc_1d(arena, C);

    m->out_gn_g   = rcre_tensor_alloc_1d(arena, C);
    m->out_gn_b   = rcre_tensor_alloc_1d(arena, C);
    m->out_conv_w = rcre_tensor_alloc_4d(arena, cfg.in_channels, C, 3, 3);
    m->out_conv_b = rcre_tensor_alloc_1d(arena, cfg.in_channels);

    if (!m->time_fc1_w || !m->time_fc1_b || !m->time_fc2_w || !m->time_fc2_b ||
        !m->in_conv_w  || !m->in_conv_b  || !m->out_gn_g   || !m->out_gn_b   ||
        !m->out_conv_w || !m->out_conv_b) {
        return nullptr;
    }

    rcre_tensor_init_xavier(m->time_fc1_w, TD, TD, rng);
    rcre_tensor_init_xavier(m->time_fc2_w, TD, TD, rng);
    rcre_tensor_init_he(m->in_conv_w, cfg.in_channels * 9U, rng);
    rcre_tensor_ones(m->out_gn_g);
    rcre_tensor_fill_normal(m->out_conv_w, 0.0f, 0.02f, rng);

    m->time_fc1_w->label = "unet.time_fc1.w"; m->time_fc1_b->label = "unet.time_fc1.b";
    m->time_fc2_w->label = "unet.time_fc2.w"; m->time_fc2_b->label = "unet.time_fc2.b";
    m->in_conv_w->label  = "unet.in_conv.w";  m->in_conv_b->label  = "unet.in_conv.b";
    m->out_gn_g->label   = "unet.out_gn.g";   m->out_gn_b->label   = "unet.out_gn.b";
    m->out_conv_w->label = "unet.out_conv.w"; m->out_conv_b->label = "unet.out_conv.b";

    if (!gen_diff_resblock_init(arena, &m->enc1, C,      C,  G, TD, "enc1", rng) ||
        !gen_diff_resblock_init(arena, &m->enc2, C,      C2, G, TD, "enc2", rng) ||
        !gen_diff_resblock_init(arena, &m->mid,  C2,     C2, G, TD, "mid",  rng) ||
        !gen_diff_resblock_init(arena, &m->dec2, C2 + C2, C, G, TD, "dec2", rng) ||
        !gen_diff_resblock_init(arena, &m->dec1, C + C,  C,  G, TD, "dec1", rng)) {
        return nullptr;
    }
    return m;
}

size_t rcre_tinyunet_params(const RCRE_TinyUNet* m, RCRE_Tensor** out_params, size_t max_params) {
    if (!m || !out_params || max_params < 66U) return 0;
    size_t pos = 0;
    out_params[pos++] = m->time_fc1_w;
    out_params[pos++] = m->time_fc1_b;
    out_params[pos++] = m->time_fc2_w;
    out_params[pos++] = m->time_fc2_b;
    out_params[pos++] = m->in_conv_w;
    out_params[pos++] = m->in_conv_b;
    pos = gen_diff_resblock_collect(&m->enc1, out_params, pos, max_params);
    pos = gen_diff_resblock_collect(&m->enc2, out_params, pos, max_params);
    pos = gen_diff_resblock_collect(&m->mid,  out_params, pos, max_params);
    pos = gen_diff_resblock_collect(&m->dec2, out_params, pos, max_params);
    pos = gen_diff_resblock_collect(&m->dec1, out_params, pos, max_params);
    out_params[pos++] = m->out_gn_g;
    out_params[pos++] = m->out_gn_b;
    out_params[pos++] = m->out_conv_w;
    out_params[pos++] = m->out_conv_b;
    return pos;
}

RCRE_Tensor* rcre_tinyunet_forward(
    const RCRE_TinyUNet* m, RCRE_TensorTape* tape, RCRE_Tensor* x_t, const float* timesteps)
{
    if (!m || !tape || !x_t || !timesteps || x_t->ndim != 4 ||
        x_t->shape[1] != m->cfg.in_channels ||
        x_t->shape[2] != m->cfg.height ||
        x_t->shape[3] != m->cfg.width) {
        return nullptr;
    }

    size_t N = x_t->shape[0];
    RCRE_Tensor* temb = rcre_sinusoidal_timestep_embed(tape->arena, timesteps, N, m->cfg.time_dim);
    temb = rcre_autograd_linear(tape, temb, m->time_fc1_w, m->time_fc1_b);
    temb = rcre_autograd_silu(tape, temb);
    temb = rcre_autograd_linear(tape, temb, m->time_fc2_w, m->time_fc2_b);

    RCRE_Tensor* h0 = rcre_autograd_conv2d(tape, x_t, m->in_conv_w, 1, 1);
    h0 = rcre_autograd_add_bias_4d(tape, h0, m->in_conv_b);

    RCRE_Tensor* skip1 = gen_diff_resblock_forward(&m->enc1, tape, h0, temb);
    RCRE_Tensor* d1    = rcre_autograd_avgpool2d(tape, skip1, 2, 2);

    RCRE_Tensor* skip2 = gen_diff_resblock_forward(&m->enc2, tape, d1, temb);
    RCRE_Tensor* d2    = rcre_autograd_avgpool2d(tape, skip2, 2, 2);

    RCRE_Tensor* h     = gen_diff_resblock_forward(&m->mid, tape, d2, temb);

    h = rcre_autograd_upsample2d_nearest(tape, h, 2);
    h = rcre_autograd_concat_channels(tape, h, skip2);
    h = gen_diff_resblock_forward(&m->dec2, tape, h, temb);

    h = rcre_autograd_upsample2d_nearest(tape, h, 2);
    h = rcre_autograd_concat_channels(tape, h, skip1);
    h = gen_diff_resblock_forward(&m->dec1, tape, h, temb);

    h = rcre_autograd_groupnorm(tape, h, m->out_gn_g, m->out_gn_b, m->cfg.num_groups, 1e-5f);
    h = rcre_autograd_silu(tape, h);
    h = rcre_autograd_conv2d(tape, h, m->out_conv_w, 1, 1);
    h = rcre_autograd_add_bias_4d(tape, h, m->out_conv_b);
    return h;
}

float rcre_tinyunet_train_step(
    RCRE_TinyUNet* m,
    const RCRE_DiffSchedule* sched,
    RCRE_TensorTape* tape,
    RCRE_TensorOptim* opt,
    const RCRE_Tensor* x0,
    uint64_t* rng,
    float max_grad_norm)
{
    if (!m || !sched || !tape || !opt || !x0 || !rng || x0->ndim != 4) return -1.0f;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(tape->arena);
    rcre_tensor_tape_reset(tape);
    rcre_tensor_optim_zero_grad(opt);

    size_t N = x0->shape[0];
    size_t* t_idx = (size_t*)rcre_arena_alloc(tape->arena, N * sizeof(size_t), alignof(size_t));
    float*  t_flt = (float*)rcre_arena_alloc(tape->arena, N * sizeof(float), 16);
    RCRE_Tensor* noise = rcre_tensor_alloc(tape->arena, x0->shape, 4);
    RCRE_Tensor* x_t   = rcre_tensor_alloc(tape->arena, x0->shape, 4);
    if (!t_idx || !t_flt || !noise || !x_t) {
        rcre_arena_temp_end(tmp);
        return -1.0f;
    }

    for (size_t n = 0; n < N; ++n) {
        t_idx[n] = (size_t)(rcre_rng_next(rng) % (uint64_t)sched->num_steps);
        t_flt[n] = (float)t_idx[n];
    }
    rcre_tensor_fill_normal(noise, 0.0f, 1.0f, rng);
    if (rcre_diff_q_sample(sched, x0, t_idx, noise, x_t) != RCRE_OK) {
        rcre_arena_temp_end(tmp);
        return -1.0f;
    }

    RCRE_Tensor* eps_pred = rcre_tinyunet_forward(m, tape, x_t, t_flt);
    if (!eps_pred) {
        rcre_arena_temp_end(tmp);
        return -1.0f;
    }

    float loss = rcre_loss_mse(eps_pred, noise, eps_pred);
    rcre_tensor_tape_backward(tape);

    if (max_grad_norm > 0.0f) {
        (void)rcre_tensor_clip_grad_norm(opt->params, opt->num_params, max_grad_norm);
    }
    rcre_tensor_optim_step(opt);

    rcre_arena_temp_end(tmp);
    return loss;
}

RCRE_Tensor* rcre_tinyunet_sample_ddpm(
    const RCRE_TinyUNet* m, const RCRE_DiffSchedule* sched,
    RCRE_Arena* out_arena, RCRE_Arena* scratch,
    size_t num_samples, uint64_t* rng)
{
    if (!m) return nullptr;
    return gen_diff_sample_ddpm(m, gen_tinyunet_fwd_wrap, nullptr, sched,
                                m->cfg.in_channels, m->cfg.height, m->cfg.width,
                                out_arena, scratch, num_samples, rng, 64);
}

RCRE_Tensor* rcre_tinyunet_sample_ddim(
    const RCRE_TinyUNet* m, const RCRE_DiffSchedule* sched,
    RCRE_Arena* out_arena, RCRE_Arena* scratch,
    size_t num_samples, size_t ddim_steps, float eta, uint64_t* rng)
{
    if (!m) return nullptr;
    return gen_diff_sample_ddim(m, gen_tinyunet_fwd_wrap, nullptr, sched,
                                m->cfg.in_channels, m->cfg.height, m->cfg.width,
                                out_arena, scratch, num_samples, ddim_steps, eta, rng, 64);
}
