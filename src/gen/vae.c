/**
 * @file vae.c
 * @brief Phase 2: Convolutional Variational Autoencoder (RCRE_ConvVAE / beta-VAE)
 *
 * Part of Recep's Code Run Everywhere (RCRE).
 * Strict ISO C23 compliant, zero external dependencies, zero hot-loop mallocs.
 */

#include "rcre_gen.h"
#include "rcre_rng.h"
#include <math.h>
#include <string.h>

static void gen_vae_init_conv(RCRE_Tensor* w, size_t in_ch, size_t k, uint64_t* rng) {
    rcre_tensor_init_he(w, in_ch * k * k, rng);
}

RCRE_ConvVAE* rcre_convvae_create(RCRE_Arena* arena, RCRE_ConvVAEConfig cfg, uint64_t* rng) {
    if (!arena || !rng || cfg.in_channels == 0 || cfg.base_channels == 0 ||
        cfg.latent_dim == 0 || cfg.height < 4 || cfg.width < 4 ||
        (cfg.height % 4U) != 0 || (cfg.width % 4U) != 0) {
        return nullptr;
    }

    RCRE_ConvVAE* m = (RCRE_ConvVAE*)rcre_arena_alloc(arena, sizeof(RCRE_ConvVAE), alignof(RCRE_ConvVAE));
    if (!m) return nullptr;
    memset(m, 0, sizeof(*m));

    m->cfg      = cfg;
    m->feat_h   = cfg.height / 4U;
    m->feat_w   = cfg.width / 4U;
    size_t c1   = cfg.base_channels;
    size_t c2   = cfg.base_channels * 2U;
    m->flat_dim = c2 * m->feat_h * m->feat_w;

    m->enc_conv1_w = rcre_tensor_alloc_4d(arena, c1, cfg.in_channels, 3, 3);
    m->enc_conv1_b = rcre_tensor_alloc_1d(arena, c1);
    m->enc_conv2_w = rcre_tensor_alloc_4d(arena, c2, c1, 3, 3);
    m->enc_conv2_b = rcre_tensor_alloc_1d(arena, c2);
    m->fc_mu_w     = rcre_tensor_alloc_2d(arena, m->flat_dim, cfg.latent_dim);
    m->fc_mu_b     = rcre_tensor_alloc_1d(arena, cfg.latent_dim);
    m->fc_lv_w     = rcre_tensor_alloc_2d(arena, m->flat_dim, cfg.latent_dim);
    m->fc_lv_b     = rcre_tensor_alloc_1d(arena, cfg.latent_dim);

    m->dec_fc_w    = rcre_tensor_alloc_2d(arena, cfg.latent_dim, m->flat_dim);
    m->dec_fc_b    = rcre_tensor_alloc_1d(arena, m->flat_dim);
    m->dec_conv1_w = rcre_tensor_alloc_4d(arena, c1, c2, 3, 3);
    m->dec_conv1_b = rcre_tensor_alloc_1d(arena, c1);
    m->dec_conv2_w = rcre_tensor_alloc_4d(arena, cfg.in_channels, c1, 3, 3);
    m->dec_conv2_b = rcre_tensor_alloc_1d(arena, cfg.in_channels);

    if (!m->enc_conv1_w || !m->enc_conv1_b || !m->enc_conv2_w || !m->enc_conv2_b ||
        !m->fc_mu_w     || !m->fc_mu_b     || !m->fc_lv_w     || !m->fc_lv_b     ||
        !m->dec_fc_w    || !m->dec_fc_b    || !m->dec_conv1_w || !m->dec_conv1_b ||
        !m->dec_conv2_w || !m->dec_conv2_b) {
        return nullptr;
    }

    gen_vae_init_conv(m->enc_conv1_w, cfg.in_channels, 3, rng);
    gen_vae_init_conv(m->enc_conv2_w, c1, 3, rng);
    rcre_tensor_init_xavier(m->fc_mu_w, m->flat_dim, cfg.latent_dim, rng);
    /* Initialize logvar head small so initial posterior starts near N(0, I) */
    rcre_tensor_fill_normal(m->fc_lv_w, 0.0f, 0.02f, rng);
    rcre_tensor_fill(m->fc_lv_b, -0.5f);

    rcre_tensor_init_xavier(m->dec_fc_w, cfg.latent_dim, m->flat_dim, rng);
    gen_vae_init_conv(m->dec_conv1_w, c2, 3, rng);
    gen_vae_init_conv(m->dec_conv2_w, c1, 3, rng);

    m->enc_conv1_w->label = "vae.enc_conv1.w"; m->enc_conv1_b->label = "vae.enc_conv1.b";
    m->enc_conv2_w->label = "vae.enc_conv2.w"; m->enc_conv2_b->label = "vae.enc_conv2.b";
    m->fc_mu_w->label     = "vae.fc_mu.w";     m->fc_mu_b->label     = "vae.fc_mu.b";
    m->fc_lv_w->label     = "vae.fc_lv.w";     m->fc_lv_b->label     = "vae.fc_lv.b";
    m->dec_fc_w->label    = "vae.dec_fc.w";    m->dec_fc_b->label    = "vae.dec_fc.b";
    m->dec_conv1_w->label = "vae.dec_conv1.w"; m->dec_conv1_b->label = "vae.dec_conv1.b";
    m->dec_conv2_w->label = "vae.dec_conv2.w"; m->dec_conv2_b->label = "vae.dec_conv2.b";

    return m;
}

size_t rcre_convvae_params(const RCRE_ConvVAE* m, RCRE_Tensor** out_params, size_t max_params) {
    if (!m || !out_params || max_params < RCRE_CONVVAE_NUM_PARAMS) return 0;
    out_params[0]  = m->enc_conv1_w;
    out_params[1]  = m->enc_conv1_b;
    out_params[2]  = m->enc_conv2_w;
    out_params[3]  = m->enc_conv2_b;
    out_params[4]  = m->fc_mu_w;
    out_params[5]  = m->fc_mu_b;
    out_params[6]  = m->fc_lv_w;
    out_params[7]  = m->fc_lv_b;
    out_params[8]  = m->dec_fc_w;
    out_params[9]  = m->dec_fc_b;
    out_params[10] = m->dec_conv1_w;
    out_params[11] = m->dec_conv1_b;
    out_params[12] = m->dec_conv2_w;
    out_params[13] = m->dec_conv2_b;
    return RCRE_CONVVAE_NUM_PARAMS;
}

int rcre_convvae_encode(const RCRE_ConvVAE* m, RCRE_TensorTape* tape, RCRE_Tensor* x,
                        RCRE_Tensor** out_mu, RCRE_Tensor** out_logvar) {
    if (!m || !tape || !x || !out_mu || !out_logvar || x->ndim != 4 ||
        x->shape[1] != m->cfg.in_channels ||
        x->shape[2] != m->cfg.height ||
        x->shape[3] != m->cfg.width) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t N = x->shape[0];
    RCRE_Tensor* h = rcre_autograd_conv2d(tape, x, m->enc_conv1_w, 2, 1);
    h = rcre_autograd_add_bias_4d(tape, h, m->enc_conv1_b);
    h = rcre_autograd_silu(tape, h);

    h = rcre_autograd_conv2d(tape, h, m->enc_conv2_w, 2, 1);
    h = rcre_autograd_add_bias_4d(tape, h, m->enc_conv2_b);
    h = rcre_autograd_silu(tape, h);

    size_t flat_shape[2] = { N, m->flat_dim };
    h = rcre_autograd_reshape(tape, h, flat_shape, 2);

    RCRE_Tensor* mu = rcre_autograd_linear(tape, h, m->fc_mu_w, m->fc_mu_b);
    RCRE_Tensor* lv = rcre_autograd_linear(tape, h, m->fc_lv_w, m->fc_lv_b);
    if (!mu || !lv) return RCRE_ERR_OUT_OF_MEMORY;

    *out_mu     = mu;
    *out_logvar = lv;
    return RCRE_OK;
}

RCRE_Tensor* rcre_convvae_decode(const RCRE_ConvVAE* m, RCRE_TensorTape* tape, RCRE_Tensor* z) {
    if (!m || !tape || !z || z->ndim != 2 || z->shape[1] != m->cfg.latent_dim) {
        return nullptr;
    }
    size_t N = z->shape[0];
    size_t c2 = m->cfg.base_channels * 2U;

    RCRE_Tensor* h = rcre_autograd_linear(tape, z, m->dec_fc_w, m->dec_fc_b);
    h = rcre_autograd_silu(tape, h);

    size_t map_shape[4] = { N, c2, m->feat_h, m->feat_w };
    h = rcre_autograd_reshape(tape, h, map_shape, 4);

    h = rcre_autograd_upsample2d_nearest(tape, h, 2);
    h = rcre_autograd_conv2d(tape, h, m->dec_conv1_w, 1, 1);
    h = rcre_autograd_add_bias_4d(tape, h, m->dec_conv1_b);
    h = rcre_autograd_silu(tape, h);

    h = rcre_autograd_upsample2d_nearest(tape, h, 2);
    h = rcre_autograd_conv2d(tape, h, m->dec_conv2_w, 1, 1);
    h = rcre_autograd_add_bias_4d(tape, h, m->dec_conv2_b);

    return m->cfg.output_sigmoid ? rcre_autograd_sigmoid(tape, h)
                                 : rcre_autograd_tanh(tape, h);
}

RCRE_Tensor* rcre_convvae_forward(const RCRE_ConvVAE* m, RCRE_TensorTape* tape, RCRE_Tensor* x,
                                  uint64_t* rng, RCRE_Tensor** out_mu,
                                  RCRE_Tensor** out_logvar, RCRE_Tensor** out_z) {
    if (!m || !tape || !x || !rng) return nullptr;
    RCRE_Tensor* mu = nullptr;
    RCRE_Tensor* lv = nullptr;
    if (rcre_convvae_encode(m, tape, x, &mu, &lv) != RCRE_OK) return nullptr;

    RCRE_Tensor* eps = rcre_tensor_alloc_2d(tape->arena, x->shape[0], m->cfg.latent_dim);
    if (!eps) return nullptr;
    rcre_tensor_fill_normal(eps, 0.0f, 1.0f, rng);

    RCRE_Tensor* z = rcre_autograd_reparameterize(tape, mu, lv, eps);
    if (!z) return nullptr;

    RCRE_Tensor* x_hat = rcre_convvae_decode(m, tape, z);
    if (!x_hat) return nullptr;

    if (out_mu)     *out_mu     = mu;
    if (out_logvar) *out_logvar = lv;
    if (out_z)      *out_z      = z;
    return x_hat;
}

float rcre_convvae_train_step(
    RCRE_ConvVAE* m,
    RCRE_TensorTape* tape,
    RCRE_TensorOptim* opt,
    RCRE_Tensor* x,
    float beta,
    uint64_t* rng,
    float max_grad_norm,
    float* out_recon_loss,
    float* out_kl_loss)
{
    if (!m || !tape || !opt || !x || !rng) return -1.0f;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(tape->arena);
    rcre_tensor_tape_reset(tape);
    rcre_tensor_optim_zero_grad(opt);

    RCRE_Tensor* mu = nullptr;
    RCRE_Tensor* lv = nullptr;
    RCRE_Tensor* x_hat = rcre_convvae_forward(m, tape, x, rng, &mu, &lv, nullptr);
    if (!x_hat) {
        rcre_arena_temp_end(tmp);
        return -1.0f;
    }

    float recon = rcre_loss_mse(x_hat, x, x_hat);
    float kl    = rcre_loss_kl_divergence_add(mu, lv, beta, mu, lv);

    rcre_tensor_tape_backward(tape);

    if (max_grad_norm > 0.0f) {
        (void)rcre_tensor_clip_grad_norm(opt->params, opt->num_params, max_grad_norm);
    }
    rcre_tensor_optim_step(opt);

    rcre_arena_temp_end(tmp);
    if (out_recon_loss) *out_recon_loss = recon;
    if (out_kl_loss)    *out_kl_loss    = kl;
    return recon + beta * kl;
}

RCRE_Tensor* rcre_convvae_sample(
    const RCRE_ConvVAE* m, RCRE_Arena* scratch, size_t num_samples, uint64_t* rng)
{
    if (!m || !scratch || num_samples == 0 || !rng) return nullptr;
    RCRE_TensorTape* tape = rcre_tensor_tape_create(scratch, 32);
    if (!tape) return nullptr;

    RCRE_Tensor* z = rcre_tensor_alloc_2d(scratch, num_samples, m->cfg.latent_dim);
    if (!z) {
        rcre_tensor_tape_destroy(tape);
        return nullptr;
    }
    rcre_tensor_fill_normal(z, 0.0f, 1.0f, rng);
    RCRE_Tensor* imgs = rcre_convvae_decode(m, tape, z);
    rcre_tensor_tape_destroy(tape);
    return imgs;
}

RCRE_Tensor* rcre_convvae_interpolate(
    const RCRE_ConvVAE* m, RCRE_Arena* scratch,
    const float* z_start, const float* z_end, size_t num_steps)
{
    if (!m || !scratch || !z_start || !z_end || num_steps < 2) return nullptr;
    RCRE_TensorTape* tape = rcre_tensor_tape_create(scratch, 32);
    if (!tape) return nullptr;

    size_t D = m->cfg.latent_dim;
    RCRE_Tensor* z = rcre_tensor_alloc_2d(scratch, num_steps, D);
    if (!z) {
        rcre_tensor_tape_destroy(tape);
        return nullptr;
    }
    float* zd = rcre_tensor_f32_data(z);
    for (size_t s = 0; s < num_steps; ++s) {
        float alpha = (float)s / (float)(num_steps - 1U);
        for (size_t d = 0; d < D; ++d) {
            zd[s * D + d] = (1.0f - alpha) * z_start[d] + alpha * z_end[d];
        }
    }
    RCRE_Tensor* imgs = rcre_convvae_decode(m, tape, z);
    rcre_tensor_tape_destroy(tape);
    return imgs;
}
