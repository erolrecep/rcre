/**
 * @file ae.c
 * @brief Phase 1: Convolutional & Denoising Autoencoder (RCRE_ConvAE) + Shared Gen Utilities
 *
 * Part of Recep's Code Run Everywhere (RCRE).
 * Strict ISO C23 compliant, zero external dependencies, zero hot-loop mallocs.
 */

#include "rcre_gen.h"
#include "rcre_io.h"
#include "rcre_rng.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static void gen_ae_init_conv(RCRE_Tensor* w, size_t in_ch, size_t k, uint64_t* rng) {
    rcre_tensor_init_he(w, in_ch * k * k, rng);
}

RCRE_ConvAE* rcre_convae_create(RCRE_Arena* arena, RCRE_ConvAEConfig cfg, uint64_t* rng) {
    if (!arena || !rng || cfg.in_channels == 0 || cfg.base_channels == 0 ||
        cfg.latent_dim == 0 || cfg.height < 4 || cfg.width < 4 ||
        (cfg.height % 4U) != 0 || (cfg.width % 4U) != 0) {
        return nullptr;
    }

    RCRE_ConvAE* m = (RCRE_ConvAE*)rcre_arena_alloc(arena, sizeof(RCRE_ConvAE), alignof(RCRE_ConvAE));
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
    m->enc_fc_w    = rcre_tensor_alloc_2d(arena, m->flat_dim, cfg.latent_dim);
    m->enc_fc_b    = rcre_tensor_alloc_1d(arena, cfg.latent_dim);

    m->dec_fc_w    = rcre_tensor_alloc_2d(arena, cfg.latent_dim, m->flat_dim);
    m->dec_fc_b    = rcre_tensor_alloc_1d(arena, m->flat_dim);
    m->dec_conv1_w = rcre_tensor_alloc_4d(arena, c1, c2, 3, 3);
    m->dec_conv1_b = rcre_tensor_alloc_1d(arena, c1);
    m->dec_conv2_w = rcre_tensor_alloc_4d(arena, cfg.in_channels, c1, 3, 3);
    m->dec_conv2_b = rcre_tensor_alloc_1d(arena, cfg.in_channels);

    if (!m->enc_conv1_w || !m->enc_conv1_b || !m->enc_conv2_w || !m->enc_conv2_b ||
        !m->enc_fc_w    || !m->enc_fc_b    || !m->dec_fc_w    || !m->dec_fc_b    ||
        !m->dec_conv1_w || !m->dec_conv1_b || !m->dec_conv2_w || !m->dec_conv2_b) {
        return nullptr;
    }

    gen_ae_init_conv(m->enc_conv1_w, cfg.in_channels, 3, rng);
    gen_ae_init_conv(m->enc_conv2_w, c1, 3, rng);
    rcre_tensor_init_xavier(m->enc_fc_w, m->flat_dim, cfg.latent_dim, rng);
    rcre_tensor_init_xavier(m->dec_fc_w, cfg.latent_dim, m->flat_dim, rng);
    gen_ae_init_conv(m->dec_conv1_w, c2, 3, rng);
    gen_ae_init_conv(m->dec_conv2_w, c1, 3, rng);

    m->enc_conv1_w->label = "ae.enc_conv1.w"; m->enc_conv1_b->label = "ae.enc_conv1.b";
    m->enc_conv2_w->label = "ae.enc_conv2.w"; m->enc_conv2_b->label = "ae.enc_conv2.b";
    m->enc_fc_w->label    = "ae.enc_fc.w";    m->enc_fc_b->label    = "ae.enc_fc.b";
    m->dec_fc_w->label    = "ae.dec_fc.w";    m->dec_fc_b->label    = "ae.dec_fc.b";
    m->dec_conv1_w->label = "ae.dec_conv1.w"; m->dec_conv1_b->label = "ae.dec_conv1.b";
    m->dec_conv2_w->label = "ae.dec_conv2.w"; m->dec_conv2_b->label = "ae.dec_conv2.b";

    return m;
}

size_t rcre_convae_params(const RCRE_ConvAE* m, RCRE_Tensor** out_params, size_t max_params) {
    if (!m || !out_params || max_params < RCRE_CONVAE_NUM_PARAMS) return 0;
    out_params[0]  = m->enc_conv1_w;
    out_params[1]  = m->enc_conv1_b;
    out_params[2]  = m->enc_conv2_w;
    out_params[3]  = m->enc_conv2_b;
    out_params[4]  = m->enc_fc_w;
    out_params[5]  = m->enc_fc_b;
    out_params[6]  = m->dec_fc_w;
    out_params[7]  = m->dec_fc_b;
    out_params[8]  = m->dec_conv1_w;
    out_params[9]  = m->dec_conv1_b;
    out_params[10] = m->dec_conv2_w;
    out_params[11] = m->dec_conv2_b;
    return RCRE_CONVAE_NUM_PARAMS;
}

RCRE_Tensor* rcre_convae_encode(const RCRE_ConvAE* m, RCRE_TensorTape* tape, RCRE_Tensor* x) {
    if (!m || !tape || !x || x->ndim != 4 ||
        x->shape[1] != m->cfg.in_channels ||
        x->shape[2] != m->cfg.height ||
        x->shape[3] != m->cfg.width) {
        return nullptr;
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
    return rcre_autograd_linear(tape, h, m->enc_fc_w, m->enc_fc_b);
}

RCRE_Tensor* rcre_convae_decode(const RCRE_ConvAE* m, RCRE_TensorTape* tape, RCRE_Tensor* z) {
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

RCRE_Tensor* rcre_convae_forward(const RCRE_ConvAE* m, RCRE_TensorTape* tape, RCRE_Tensor* x) {
    RCRE_Tensor* z = rcre_convae_encode(m, tape, x);
    if (!z) return nullptr;
    return rcre_convae_decode(m, tape, z);
}

float rcre_convae_train_step(
    RCRE_ConvAE* m,
    RCRE_TensorTape* tape,
    RCRE_TensorOptim* opt,
    RCRE_Tensor* x_clean,
    float noise_std,
    uint64_t* rng,
    float max_grad_norm)
{
    if (!m || !tape || !opt || !x_clean) return -1.0f;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(tape->arena);
    rcre_tensor_tape_reset(tape);
    rcre_tensor_optim_zero_grad(opt);

    RCRE_Tensor* x_in = x_clean;
    if (noise_std > 0.0f && rng != nullptr) {
        x_in = rcre_tensor_alloc(tape->arena, x_clean->shape, x_clean->ndim);
        if (!x_in) {
            rcre_arena_temp_end(tmp);
            return -1.0f;
        }
        const float* cd = rcre_tensor_f32_data(x_clean);
        float* nd = rcre_tensor_f32_data(x_in);
        float lo = m->cfg.output_sigmoid ? 0.0f : -1.0f;
        float hi = 1.0f;
        for (size_t i = 0; i < x_clean->numel; ++i) {
            float v = cd[i] + rcre_rng_gaussian_f32(rng, 0.0f, noise_std);
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            nd[i] = v;
        }
    }

    RCRE_Tensor* x_hat = rcre_convae_forward(m, tape, x_in);
    if (!x_hat) {
        rcre_arena_temp_end(tmp);
        return -1.0f;
    }

    float loss = rcre_loss_mse(x_hat, x_clean, x_hat);
    rcre_tensor_tape_backward(tape);

    if (max_grad_norm > 0.0f) {
        (void)rcre_tensor_clip_grad_norm(opt->params, opt->num_params, max_grad_norm);
    }
    rcre_tensor_optim_step(opt);

    rcre_arena_temp_end(tmp);
    return loss;
}

float rcre_gen_psnr(const RCRE_Tensor* ref, const RCRE_Tensor* recon, float max_val) {
    if (!ref || !recon || ref->numel != recon->numel || ref->numel == 0 || max_val <= 0.0f) {
        return 0.0f;
    }
    const float* a = rcre_tensor_f32_data(ref);
    const float* b = rcre_tensor_f32_data(recon);
    double mse = 0.0;
    for (size_t i = 0; i < ref->numel; ++i) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= (double)ref->numel;
    if (mse <= 1e-12) return 99.0f;
    return (float)(10.0 * log10(((double)max_val * (double)max_val) / mse));
}

/* ========================================================================= */
/* Shared Generative Utilities: Synthetic Shapes Dataset & SafeTensors I/O   */
/* ========================================================================= */

int rcre_gen_make_shapes_dataset(
    RCRE_Arena* arena,
    size_t num_samples,
    size_t height,
    size_t width,
    bool zero_to_one,
    uint64_t* rng,
    RCRE_Tensor** out_images,
    size_t** out_labels)
{
    if (!arena || !rng || !out_images || num_samples == 0 || height < 8 || width < 8) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_Tensor* imgs = rcre_tensor_alloc_4d(arena, num_samples, 1, height, width);
    if (!imgs) return RCRE_ERR_OUT_OF_MEMORY;
    size_t* labels = (size_t*)rcre_arena_alloc(arena, num_samples * sizeof(size_t), alignof(size_t));
    if (!labels) return RCRE_ERR_OUT_OF_MEMORY;

    float* data = rcre_tensor_f32_data(imgs);
    float cy_base = 0.5f * (float)(height - 1U);
    float cx_base = 0.5f * (float)(width - 1U);
    float min_dim = (float)(height < width ? height : width);

    for (size_t n = 0; n < num_samples; ++n) {
        size_t cls = n % 4U;
        labels[n] = cls;

        /* Small controlled jitter for realistic generalization */
        float jx = rcre_rng_uniform_f32(rng, -0.08f, 0.08f) * min_dim;
        float jy = rcre_rng_uniform_f32(rng, -0.08f, 0.08f) * min_dim;
        float scale = rcre_rng_uniform_f32(rng, 0.88f, 1.08f);
        float cy = cy_base + jy;
        float cx = cx_base + jx;
        float rad = 0.30f * min_dim * scale;
        float thick = 0.11f * min_dim * scale;
        if (thick < 1.1f) thick = 1.1f;

        float* plane = data + n * height * width;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float dy = (float)y - cy;
                float dx = (float)x - cx;
                float r  = sqrtf(dx * dx + dy * dy);
                float val = 0.0f;

                if (cls == 0) {
                    /* Disk */
                    float edge = rad - r;
                    val = 1.0f / (1.0f + expf(-2.5f * edge));
                } else if (cls == 1) {
                    /* Square */
                    float dmax = fmaxf(fabsf(dx), fabsf(dy));
                    float edge = rad * 0.90f - dmax;
                    val = 1.0f / (1.0f + expf(-2.5f * edge));
                } else if (cls == 2) {
                    /* Plus / Cross */
                    float arm_h = fmaxf(fabsf(dy) - thick, fabsf(dx) - rad * 1.05f);
                    float arm_v = fmaxf(fabsf(dx) - thick, fabsf(dy) - rad * 1.05f);
                    float d = fminf(arm_h, arm_v);
                    val = 1.0f / (1.0f + expf(2.5f * d));
                } else {
                    /* Ring */
                    float d_ring = fabsf(r - rad * 0.85f) - thick * 0.85f;
                    val = 1.0f / (1.0f + expf(2.5f * d_ring));
                }

                plane[y * width + x] = zero_to_one ? val : (2.0f * val - 1.0f);
            }
        }
    }

    *out_images = imgs;
    if (out_labels) *out_labels = labels;
    return RCRE_OK;
}

int rcre_gen_save_safetensors(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath,
    RCRE_Arena* scratch)
{
    if (!params || num_params == 0 || !filepath || !scratch) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
    const char** names = (const char**)rcre_arena_alloc(scratch, num_params * sizeof(char*), alignof(char*));
    RCRE_Tensor* vals  = (RCRE_Tensor*)rcre_arena_alloc(scratch, num_params * sizeof(RCRE_Tensor), alignof(RCRE_Tensor));
    char* name_buf     = (char*)rcre_arena_alloc(scratch, num_params * 32U, 16);
    if (!names || !vals || !name_buf) {
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < num_params; ++i) {
        if (!params[i] || params[i]->dtype != RCRE_DTYPE_F32) {
            rcre_arena_temp_end(tmp);
            return RCRE_ERR_INVALID_ARG;
        }
        if (params[i]->label && params[i]->label[0] != '\0') {
            names[i] = params[i]->label;
        } else {
            snprintf(name_buf + i * 32U, 32U, "param_%zu", i);
            names[i] = name_buf + i * 32U;
        }
        vals[i] = *params[i];
    }

    int rc = rcre_io_safetensors_save(filepath, names, vals, num_params, scratch);
    rcre_arena_temp_end(tmp);
    return rc;
}

int rcre_gen_load_safetensors(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath)
{
    if (!params || num_params == 0 || !filepath) return RCRE_ERR_INVALID_ARG;

    RCRE_SafeTensorsArchive arch;
    int rc = rcre_io_safetensors_open(filepath, &arch);
    if (rc != RCRE_OK) return rc;

    RCRE_Arena* tmp_arena = rcre_arena_create(4 * 1024 * 1024);
    if (!tmp_arena) {
        rcre_io_safetensors_close(&arch);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    char fallback_name[32];
    for (size_t i = 0; i < num_params; ++i) {
        if (!params[i] || params[i]->dtype != RCRE_DTYPE_F32) {
            rc = RCRE_ERR_INVALID_ARG;
            break;
        }
        const char* nm = (params[i]->label && params[i]->label[0] != '\0')
                             ? params[i]->label
                             : (snprintf(fallback_name, sizeof(fallback_name), "param_%zu", i), fallback_name);
        RCRE_ArenaTemp t = rcre_arena_temp_begin(tmp_arena);
        RCRE_Tensor loaded = {0};
        rc = rcre_io_safetensors_get_tensor_f32(&arch, nm, tmp_arena, &loaded);
        if (rc != RCRE_OK || loaded.numel != params[i]->numel) {
            rcre_arena_temp_end(t);
            rc = (rc != RCRE_OK) ? rc : RCRE_ERR_SHAPE_MISMATCH;
            break;
        }
        memcpy(params[i]->data, loaded.data, loaded.numel * sizeof(float));
        rcre_arena_temp_end(t);
    }

    rcre_arena_destroy(tmp_arena);
    rcre_io_safetensors_close(&arch);
    return rc;
}
