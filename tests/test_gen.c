/**
 * @file test_gen.c
 * @brief Comprehensive Verification Suite for RCRE Generative & Vision Suite
 *        (Phase 1: ConvAE/Denoising AE, Phase 2: ConvVAE, Phase 3: DDPM/DDIM Tiny U-Net,
 *         Phase 4: ViT & DiT w/ AdaLN-Zero, plus SafeTensors round-trip)
 */

#include "rcre.h"
#include "rcre_gen.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void test_phase1_convae_and_denoising(void) {
    rcre_hal_init();
    RCRE_Arena* perma = rcre_arena_create(8 * 1024 * 1024);
    RCRE_Arena* tape_arena = rcre_arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(perma);
    ASSERT_NOT_NULL(tape_arena);

    uint64_t rng = 101ULL;
    RCRE_Tensor* imgs = nullptr;
    size_t* labels = nullptr;
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_make_shapes_dataset(perma, 8, 16, 16, true, &rng, &imgs, &labels));
    ASSERT_NOT_NULL(imgs);
    ASSERT_UINT_EQ(8, imgs->shape[0]);
    ASSERT_UINT_EQ(16, imgs->shape[2]);

    RCRE_ConvAEConfig cfg = {
        .in_channels    = 1,
        .height         = 16,
        .width          = 16,
        .base_channels  = 8,
        .latent_dim     = 12,
        .output_sigmoid = true
    };
    RCRE_ConvAE* ae = rcre_convae_create(perma, cfg, &rng);
    ASSERT_NOT_NULL(ae);

    RCRE_Tensor* params[RCRE_CONVAE_NUM_PARAMS];
    size_t np = rcre_convae_params(ae, params, RCRE_CONVAE_NUM_PARAMS);
    ASSERT_UINT_EQ(RCRE_CONVAE_NUM_PARAMS, np);

    RCRE_TensorOptim* opt = rcre_tensor_optim_adamw_create(perma, params, np, 8e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f);
    ASSERT_NOT_NULL(opt);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(tape_arena, 64);
    ASSERT_NOT_NULL(tape);

    float first_loss = 0.0f;
    float last_loss  = 0.0f;
    for (int step = 0; step < 60; ++step) {
        /* Denoising AE training with mild Gaussian corruption */
        float loss = rcre_convae_train_step(ae, tape, opt, imgs, 0.08f, &rng, 2.0f);
        ASSERT_TRUE(isfinite(loss));
        if (step == 0) first_loss = loss;
        last_loss = loss;
    }
    ASSERT_TRUE(last_loss < first_loss * 0.45f);

    /* Forward reconstruction & PSNR check */
    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(tape_arena);
    rcre_tensor_tape_reset(tape);
    RCRE_Tensor* recon = rcre_convae_forward(ae, tape, imgs);
    ASSERT_NOT_NULL(recon);
    float psnr = rcre_gen_psnr(imgs, recon, 1.0f);
    ASSERT_TRUE(psnr > 14.0f);
    rcre_arena_temp_end(tmp);

    /* SafeTensors checkpoint round-trip */
    const char* ckpt = "/tmp/rcre_test_convae.safetensors";
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_save_safetensors(params, np, ckpt, tape_arena));

    RCRE_ConvAE* ae2 = rcre_convae_create(perma, cfg, &rng);
    ASSERT_NOT_NULL(ae2);
    RCRE_Tensor* params2[RCRE_CONVAE_NUM_PARAMS];
    ASSERT_UINT_EQ(np, rcre_convae_params(ae2, params2, RCRE_CONVAE_NUM_PARAMS));
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_load_safetensors(params2, np, ckpt));
    for (size_t p = 0; p < np; ++p) {
        const float* a = rcre_tensor_f32_data(params[p]);
        const float* b = rcre_tensor_f32_data(params2[p]);
        for (size_t i = 0; i < params[p]->numel; ++i) {
            ASSERT_FLOAT_NEAR(a[i], b[i], 0.0f);
        }
    }
    remove(ckpt);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(tape_arena);
    rcre_arena_destroy(perma);
}

static void test_phase2_convvae_elbo_and_sampling(void) {
    rcre_hal_init();
    RCRE_Arena* perma = rcre_arena_create(8 * 1024 * 1024);
    RCRE_Arena* tape_arena = rcre_arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(perma);
    ASSERT_NOT_NULL(tape_arena);

    uint64_t rng = 202ULL;
    RCRE_Tensor* imgs = nullptr;
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_make_shapes_dataset(perma, 8, 16, 16, true, &rng, &imgs, nullptr));

    RCRE_ConvVAEConfig cfg = {
        .in_channels    = 1,
        .height         = 16,
        .width          = 16,
        .base_channels  = 8,
        .latent_dim     = 8,
        .output_sigmoid = true
    };
    RCRE_ConvVAE* vae = rcre_convvae_create(perma, cfg, &rng);
    ASSERT_NOT_NULL(vae);

    RCRE_Tensor* params[RCRE_CONVVAE_NUM_PARAMS];
    size_t np = rcre_convvae_params(vae, params, RCRE_CONVVAE_NUM_PARAMS);
    ASSERT_UINT_EQ(RCRE_CONVVAE_NUM_PARAMS, np);

    RCRE_TensorOptim* opt = rcre_tensor_optim_adamw_create(perma, params, np, 8e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(tape_arena, 64);

    float first_total = 0.0f, last_total = 0.0f;
    float last_kl = 0.0f;
    for (int step = 0; step < 65; ++step) {
        /* Warm up beta from 0.005 -> 0.05 */
        float beta = 0.005f + 0.045f * ((float)step / 64.0f);
        float recon = 0.0f, kl = 0.0f;
        float tot = rcre_convvae_train_step(vae, tape, opt, imgs, beta, &rng, 2.0f, &recon, &kl);
        ASSERT_TRUE(isfinite(tot));
        ASSERT_TRUE(kl >= -1e-5f);
        if (step == 0) first_total = tot;
        last_total = tot;
        last_kl = kl;
    }
    ASSERT_TRUE(last_total < first_total * 0.55f);
    ASSERT_TRUE(isfinite(last_kl));

    /* Sample from prior N(0, I) */
    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(tape_arena);
    RCRE_Tensor* samples = rcre_convvae_sample(vae, tape_arena, 4, &rng);
    ASSERT_NOT_NULL(samples);
    ASSERT_UINT_EQ(4, samples->shape[0]);
    ASSERT_UINT_EQ(16, samples->shape[2]);
    const float* sd = rcre_tensor_f32_data(samples);
    for (size_t i = 0; i < samples->numel; ++i) {
        ASSERT_TRUE(sd[i] >= 0.0f && sd[i] <= 1.0f);
    }

    /* Latent interpolation */
    float z0[8] = { -1.5f, 0.2f, -0.4f, 0.8f, 0.1f, -0.3f, 0.5f, -0.2f };
    float z1[8] = {  1.5f,-0.2f,  0.4f,-0.8f,-0.1f,  0.3f,-0.5f,  0.2f };
    RCRE_Tensor* interp = rcre_convvae_interpolate(vae, tape_arena, z0, z1, 5);
    ASSERT_NOT_NULL(interp);
    ASSERT_UINT_EQ(5, interp->shape[0]);
    rcre_arena_temp_end(tmp);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(tape_arena);
    rcre_arena_destroy(perma);
}

static void test_phase3_diffusion_schedule_and_tinyunet(void) {
    rcre_hal_init();
    RCRE_Arena* perma = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* tape_arena = rcre_arena_create(32 * 1024 * 1024);
    ASSERT_NOT_NULL(perma);
    ASSERT_NOT_NULL(tape_arena);

    /* Verify linear & cosine schedule invariants */
    RCRE_DiffSchedule* s_lin = rcre_diff_schedule_create(perma, 20, RCRE_DIFF_SCHED_LINEAR, 1e-4f, 0.02f);
    RCRE_DiffSchedule* s_cos = rcre_diff_schedule_create(perma, 20, RCRE_DIFF_SCHED_COSINE, 0.0f, 0.0f);
    ASSERT_NOT_NULL(s_lin);
    ASSERT_NOT_NULL(s_cos);
    for (size_t t = 1; t < 20; ++t) {
        ASSERT_TRUE(s_lin->alpha_bar[t] < s_lin->alpha_bar[t - 1]);
        ASSERT_TRUE(s_cos->alpha_bar[t] < s_cos->alpha_bar[t - 1]);
    }

    uint64_t rng = 303ULL;
    /* Images in [-1, 1] for diffusion */
    RCRE_Tensor* imgs = nullptr;
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_make_shapes_dataset(perma, 4, 16, 16, false, &rng, &imgs, nullptr));

    RCRE_TinyUNetConfig cfg = {
        .in_channels   = 1,
        .height        = 16,
        .width         = 16,
        .base_channels = 8,
        .num_groups    = 4,
        .time_dim      = 16
    };
    RCRE_TinyUNet* unet = rcre_tinyunet_create(perma, cfg, &rng);
    ASSERT_NOT_NULL(unet);

    RCRE_Tensor* params[RCRE_TINYUNET_MAX_PARAMS];
    size_t np = rcre_tinyunet_params(unet, params, RCRE_TINYUNET_MAX_PARAMS);
    ASSERT_TRUE(np >= 66U);

    RCRE_TensorOptim* opt = rcre_tensor_optim_adamw_create(perma, params, np, 5e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(tape_arena, 128);

    float avg_early = 0.0f, avg_late = 0.0f;
    for (int step = 0; step < 30; ++step) {
        float loss = rcre_tinyunet_train_step(unet, s_cos, tape, opt, imgs, &rng, 1.0f);
        ASSERT_TRUE(isfinite(loss));
        if (step < 5) avg_early += loss / 5.0f;
        if (step >= 25) avg_late += loss / 5.0f;
    }
    ASSERT_TRUE(avg_late < avg_early);

    /* DDPM ancestral sampling + deterministic DDIM sampling (eta=0) */
    uint64_t rng_s1 = 777ULL;
    uint64_t rng_s2 = 777ULL;
    RCRE_Tensor* ddim1 = rcre_tinyunet_sample_ddim(unet, s_cos, perma, tape_arena, 2, 5, 0.0f, &rng_s1);
    RCRE_Tensor* ddim2 = rcre_tinyunet_sample_ddim(unet, s_cos, perma, tape_arena, 2, 5, 0.0f, &rng_s2);
    ASSERT_NOT_NULL(ddim1);
    ASSERT_NOT_NULL(ddim2);
    /* Deterministic DDIM with identical seed must be bit-identical */
    for (size_t i = 0; i < ddim1->numel; ++i) {
        ASSERT_FLOAT_NEAR(rcre_tensor_f32_data(ddim1)[i], rcre_tensor_f32_data(ddim2)[i], 0.0f);
        ASSERT_TRUE(isfinite(rcre_tensor_f32_data(ddim1)[i]));
    }

    RCRE_Tensor* ddpm = rcre_tinyunet_sample_ddpm(unet, s_cos, perma, tape_arena, 1, &rng_s1);
    ASSERT_NOT_NULL(ddpm);
    for (size_t i = 0; i < ddpm->numel; ++i) {
        ASSERT_TRUE(rcre_tensor_f32_data(ddpm)[i] >= -1.0f && rcre_tensor_f32_data(ddpm)[i] <= 1.0f);
    }

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(tape_arena);
    rcre_arena_destroy(perma);
}

static void test_phase4_vit_classification_and_dit_diffusion(void) {
    rcre_hal_init();
    RCRE_Arena* perma = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* tape_arena = rcre_arena_create(32 * 1024 * 1024);
    ASSERT_NOT_NULL(perma);
    ASSERT_NOT_NULL(tape_arena);

    uint64_t rng = 404ULL;
    RCRE_Tensor* imgs = nullptr;
    size_t* labels = nullptr;
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_make_shapes_dataset(perma, 16, 16, 16, false, &rng, &imgs, &labels));

    /* --- 4A: Vision Transformer (ViT) 4-class shape classification --- */
    RCRE_ViTConfig vcfg = {
        .in_channels = 1,
        .height      = 16,
        .width       = 16,
        .patch_size  = 4,   /* 4x4 = 16 patch tokens */
        .embed_dim   = 24,
        .num_heads   = 4,
        .num_layers  = 2,
        .mlp_dim     = 48,
        .num_classes = 4
    };
    RCRE_ViT* vit = rcre_vit_create(perma, vcfg, &rng);
    ASSERT_NOT_NULL(vit);

    size_t v_np = rcre_vit_param_count(vit);
    RCRE_Tensor** v_params = (RCRE_Tensor**)rcre_arena_alloc(perma, v_np * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
    ASSERT_UINT_EQ(v_np, rcre_vit_params(vit, v_params, v_np));

    RCRE_TensorOptim* v_opt = rcre_tensor_optim_adamw_create(perma, v_params, v_np, 1e-2f, 0.9f, 0.999f, 1e-8f, 1e-4f);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(tape_arena, 256);

    float v_first = 0.0f, v_last = 0.0f;
    for (int step = 0; step < 50; ++step) {
        float loss = rcre_vit_train_step(vit, tape, v_opt, imgs, labels, 2.0f);
        ASSERT_TRUE(isfinite(loss));
        if (step == 0) v_first = loss;
        v_last = loss;
    }
    ASSERT_TRUE(v_last < v_first * 0.25f);

    size_t preds[16];
    ASSERT_INT_EQ(RCRE_OK, rcre_vit_predict(vit, tape_arena, imgs, preds));
    size_t correct = 0;
    for (size_t i = 0; i < 16; ++i) {
        if (preds[i] == labels[i]) correct++;
    }
    ASSERT_TRUE(correct >= 14U); /* >= 87.5% accuracy */

    /* --- 4B: Class-Conditional Diffusion Transformer (DiT w/ AdaLN-Zero) --- */
    RCRE_DiffSchedule* sched = rcre_diff_schedule_create(perma, 20, RCRE_DIFF_SCHED_COSINE, 0.0f, 0.0f);
    ASSERT_NOT_NULL(sched);

    RCRE_DiTConfig dcfg = {
        .in_channels = 1,
        .height      = 16,
        .width       = 16,
        .patch_size  = 4,
        .embed_dim   = 24,
        .num_heads   = 4,
        .num_layers  = 2,
        .mlp_dim     = 48,
        .num_classes = 4
    };
    RCRE_DiT* dit = rcre_dit_create(perma, dcfg, &rng);
    ASSERT_NOT_NULL(dit);

    size_t d_np = rcre_dit_param_count(dit);
    RCRE_Tensor** d_params = (RCRE_Tensor**)rcre_arena_alloc(perma, d_np * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
    ASSERT_UINT_EQ(d_np, rcre_dit_params(dit, d_params, d_np));

    RCRE_TensorOptim* d_opt = rcre_tensor_optim_adamw_create(perma, d_params, d_np, 6e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f);

    float d_early = 0.0f, d_late = 0.0f;
    for (int step = 0; step < 30; ++step) {
        float loss = rcre_dit_train_step(dit, sched, tape, d_opt, imgs, labels, &rng, 1.0f);
        ASSERT_TRUE(isfinite(loss));
        if (step < 5) d_early += loss / 5.0f;
        if (step >= 25) d_late += loss / 5.0f;
    }
    ASSERT_TRUE(d_late < d_early);

    /* Deterministic class-conditioned DDIM sampling */
    size_t gen_classes[4] = { 0, 1, 2, 3 };
    uint64_t s1 = 999ULL, s2 = 999ULL;
    RCRE_Tensor* g1 = rcre_dit_sample_ddim(dit, sched, perma, tape_arena, 4, gen_classes, 5, 0.0f, &s1);
    RCRE_Tensor* g2 = rcre_dit_sample_ddim(dit, sched, perma, tape_arena, 4, gen_classes, 5, 0.0f, &s2);
    ASSERT_NOT_NULL(g1);
    ASSERT_NOT_NULL(g2);
    for (size_t i = 0; i < g1->numel; ++i) {
        ASSERT_FLOAT_NEAR(rcre_tensor_f32_data(g1)[i], rcre_tensor_f32_data(g2)[i], 0.0f);
        ASSERT_TRUE(isfinite(rcre_tensor_f32_data(g1)[i]));
    }

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(tape_arena);
    rcre_arena_destroy(perma);
}

/* ========================================================================= */
/* Phase 0 (refactor additions): reference values, determinism, error paths,
               unconditional DiT, plain AE, unlabeled checkpoints            */
/* ========================================================================= */

/* Independent-reference values, generated by a separate Python
 * implementation of the Nichol-Dhariwal cosine schedule (same f32/f64
 * arithmetic classes as the C code) — see the refactor session report. */
static void test_phase0_schedule_reference_and_validation(void) {
    rcre_hal_init();
    RCRE_Arena* perma = rcre_arena_create(8 * 1024 * 1024);
    ASSERT_NOT_NULL(perma);

    /* --- Cosine schedule, T=1000: independent reference values --- */
    RCRE_DiffSchedule* s = rcre_diff_schedule_create(perma, 1000, RCRE_DIFF_SCHED_COSINE, 0.0f, 0.0f);
    ASSERT_NOT_NULL(s);
    ASSERT_FLOAT_NEAR(9.999999747378752e-05f, s->betas[0], 1e-9f);
    ASSERT_FLOAT_NEAR(0.9998999834060669f,  s->alpha_bar[0],   1e-7f);
    ASSERT_FLOAT_NEAR(0.9712021350860596f,  s->alpha_bar[100], 1e-7f);
    ASSERT_FLOAT_NEAR(0.49209585785865784f, s->alpha_bar[500], 1e-7f);
    ASSERT_FLOAT_NEAR(2.4278028476487634e-09f, s->alpha_bar[999], 1e-12f);
    ASSERT_TRUE(s->posterior_variance[0] > 0.0f);

    /* --- Linear schedule: exact closed-form beta_t --- */
    RCRE_DiffSchedule* l = rcre_diff_schedule_create(perma, 20, RCRE_DIFF_SCHED_LINEAR, 1e-4f, 0.02f);
    ASSERT_NOT_NULL(l);
    for (size_t t = 0; t < 20; ++t) {
        float expect = 1e-4f + (float)t / 19.0f * (0.02f - 1e-4f);
        ASSERT_FLOAT_NEAR(expect, l->betas[t], 1e-8f);
    }

    /* --- q_sample exact reference: T=20 cosine, t=7, x0=[0.5,-0.25], eps=[0.1,0.9] --- */
    RCRE_DiffSchedule* s20 = rcre_diff_schedule_create(perma, 20, RCRE_DIFF_SCHED_COSINE, 0.0f, 0.0f);
    ASSERT_NOT_NULL(s20);
    RCRE_Tensor* x0 = rcre_tensor_alloc_4d(perma, 1, 1, 1, 2);
    RCRE_Tensor* eps = rcre_tensor_alloc_4d(perma, 1, 1, 1, 2);
    RCRE_Tensor* xt = rcre_tensor_alloc_4d(perma, 1, 1, 1, 2);
    float x0v[2] = { 0.5f, -0.25f }, epv[2] = { 0.1f, 0.9f };
    memcpy(rcre_tensor_f32_data(x0), x0v, sizeof(x0v));
    memcpy(rcre_tensor_f32_data(eps), epv, sizeof(epv));
    size_t t7 = 7;
    ASSERT_INT_EQ(RCRE_OK, rcre_diff_q_sample(s20, x0, &t7, eps, xt));
    ASSERT_FLOAT_NEAR(0.46170371770858765f, rcre_tensor_f32_data(xt)[0], 1e-6f);
    ASSERT_FLOAT_NEAR(0.33319681882858276f,  rcre_tensor_f32_data(xt)[1], 1e-7f);

    /* --- Schedule validation rejections --- */
    ASSERT_NULL(rcre_diff_schedule_create(perma, 1, RCRE_DIFF_SCHED_LINEAR, 1e-4f, 0.02f));          /* T < 2 */
    ASSERT_NULL(rcre_diff_schedule_create(perma, 4, RCRE_DIFF_SCHED_LINEAR, 0.02f, 0.02f));          /* beta_end == beta_start */
    ASSERT_NULL(rcre_diff_schedule_create(perma, 4, RCRE_DIFF_SCHED_LINEAR, 1e-4f, 1.0f));           /* beta_end >= 1 */
    ASSERT_NULL(rcre_diff_schedule_create(perma, 4, RCRE_DIFF_SCHED_LINEAR, 0.0f, 0.02f));           /* beta_start <= 0 */

    /* --- q_sample validation rejections --- */
    size_t t_bad = 20; /* == num_steps -> out of range */
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_diff_q_sample(s20, x0, &t_bad, eps, xt));
    RCRE_Tensor* wrong = rcre_tensor_alloc_4d(perma, 1, 1, 2, 2); /* numel mismatch */
    size_t t7b = 7;
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_diff_q_sample(s20, x0, &t7b, eps, wrong));

    rcre_arena_destroy(perma);
}

/* The determinism contract for the stochastic entry points: identical seed,
 * identical model weights -> bit-identical output. */
static void test_phase0_determinism_suite(void) {
    rcre_hal_init();
    RCRE_Arena* perma = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(8 * 1024 * 1024);
    ASSERT_NOT_NULL(perma);
    ASSERT_NOT_NULL(scratch);

    uint64_t build_rng = 505ULL;

    /* --- Synthetic dataset: same seed -> bit-identical images --- */
    RCRE_Tensor* d1 = nullptr, *d2 = nullptr;
    size_t* l1 = nullptr, *l2 = nullptr;
    uint64_t ds1 = 7ULL, ds2 = 7ULL;
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_make_shapes_dataset(perma, 8, 16, 16, true, &ds1, &d1, &l1));
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_make_shapes_dataset(perma, 8, 16, 16, true, &ds2, &d2, &l2));
    for (size_t i = 0; i < d1->numel; ++i) {
        ASSERT_FLOAT_NEAR(rcre_tensor_f32_data(d1)[i], rcre_tensor_f32_data(d2)[i], 0.0f);
    }

    /* --- DDPM ancestral sampling: same seed -> bit-identical (the missing
     *     counterpart of the DDIM eta=0 determinism check) --- */
    RCRE_DiffSchedule* sched = rcre_diff_schedule_create(perma, 20, RCRE_DIFF_SCHED_COSINE, 0.0f, 0.0f);
    ASSERT_NOT_NULL(sched);
    RCRE_TinyUNetConfig cfg = {
        .in_channels = 1, .height = 16, .width = 16,
        .base_channels = 8, .num_groups = 4, .time_dim = 16
    };
    RCRE_TinyUNet* unet = rcre_tinyunet_create(perma, cfg, &build_rng);
    ASSERT_NOT_NULL(unet);
    uint64_t ss1 = 31337ULL, ss2 = 31337ULL;
    RCRE_Tensor* p1 = rcre_tinyunet_sample_ddpm(unet, sched, perma, scratch, 1, &ss1);
    RCRE_Tensor* p2 = rcre_tinyunet_sample_ddpm(unet, sched, perma, scratch, 1, &ss2);
    ASSERT_NOT_NULL(p1);
    ASSERT_NOT_NULL(p2);
    for (size_t i = 0; i < p1->numel; ++i) {
        ASSERT_FLOAT_NEAR(rcre_tensor_f32_data(p1)[i], rcre_tensor_f32_data(p2)[i], 0.0f);
    }

    /* --- VAE prior sampling: same seed -> bit-identical --- */
    RCRE_ConvVAEConfig vcfg = {
        .in_channels = 1, .height = 16, .width = 16,
        .base_channels = 8, .latent_dim = 8, .output_sigmoid = true
    };
    RCRE_ConvVAE* vae = rcre_convvae_create(perma, vcfg, &build_rng);
    ASSERT_NOT_NULL(vae);
    uint64_t vs1 = 4242ULL, vs2 = 4242ULL;
    RCRE_Tensor* smp1 = rcre_convvae_sample(vae, perma, 2, &vs1);
    RCRE_Tensor* smp2 = rcre_convvae_sample(vae, perma, 2, &vs2);
    ASSERT_NOT_NULL(smp1);
    ASSERT_NOT_NULL(smp2);
    for (size_t i = 0; i < smp1->numel; ++i) {
        ASSERT_FLOAT_NEAR(rcre_tensor_f32_data(smp1)[i], rcre_tensor_f32_data(smp2)[i], 0.0f);
    }

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(perma);
}

static void test_phase0_error_paths(void) {
    rcre_hal_init();
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 16);
    ASSERT_NOT_NULL(tape);
    uint64_t rng = 9ULL;

    /* --- Diffusion schedule / q_sample --- */
    ASSERT_NULL(rcre_diff_schedule_create(nullptr, 10, RCRE_DIFF_SCHED_LINEAR, 1e-4f, 0.02f));

    /* --- Gen model constructors --- */
    RCRE_ConvAEConfig bad = { .in_channels = 1, .height = 14, .width = 16,
                              .base_channels = 8, .latent_dim = 8, .output_sigmoid = true };
    ASSERT_NULL(rcre_convae_create(arena, bad, &rng));              /* height % 4 != 0 */
    RCRE_ViTConfig vbad = { .in_channels = 1, .height = 16, .width = 16, .patch_size = 4,
                            .embed_dim = 24, .num_heads = 5, .num_layers = 1,
                            .mlp_dim = 48, .num_classes = 4 };
    ASSERT_NULL(rcre_vit_create(arena, vbad, &rng));                /* embed_dim % heads != 0 */
    RCRE_DiTConfig dbad = { .in_channels = 1, .height = 16, .width = 16, .patch_size = 4,
                            .embed_dim = 25, .num_heads = 4, .num_layers = 1,
                            .mlp_dim = 48, .num_classes = 0 };
    ASSERT_NULL(rcre_dit_create(arena, dbad, &rng));                /* odd embed_dim */
    RCRE_TinyUNetConfig ubad = { .in_channels = 1, .height = 16, .width = 16,
                                 .base_channels = 8, .num_groups = 3, .time_dim = 16 };
    ASSERT_NULL(rcre_tinyunet_create(arena, ubad, &rng));           /* base_ch % groups != 0 */

    /* --- Sinusoidal embedding --- */
    float ts[1] = { 1.0f };
    ASSERT_NULL(rcre_sinusoidal_timestep_embed(arena, ts, 1, 7));   /* odd embed_dim */
    ASSERT_NULL(rcre_sinusoidal_timestep_embed(arena, nullptr, 1, 8));
    ASSERT_NULL(rcre_sinusoidal_timestep_embed(arena, ts, 0, 8));

    /* --- New autograd op validation (all return nullptr on invalid input) --- */
    RCRE_Tensor* x2 = rcre_tensor_alloc_2d(arena, 3, 6);
    ASSERT_NULL(rcre_autograd_slice_cols(tape, x2, 5, 2));          /* start+width > cols */
    ASSERT_NULL(rcre_autograd_slice_cols(tape, x2, 0, 0));          /* zero width */

    RCRE_Tensor* ca = rcre_tensor_alloc_4d(arena, 2, 2, 2, 2);
    RCRE_Tensor* cb = rcre_tensor_alloc_4d(arena, 2, 3, 2, 3);      /* W mismatch */
    ASSERT_NULL(rcre_autograd_concat_channels(tape, ca, cb));

    RCRE_Tensor* gn = rcre_tensor_alloc_4d(arena, 1, 3, 2, 2);      /* C % G != 0 */
    RCRE_Tensor* gn_g = rcre_tensor_alloc_1d(arena, 3);
    RCRE_Tensor* gn_b = rcre_tensor_alloc_1d(arena, 3);
    ASSERT_NULL(rcre_autograd_groupnorm(tape, gn, gn_g, gn_b, 2, 1e-5f));

    RCRE_Tensor* q = rcre_tensor_alloc_3d(arena, 1, 4, 6);          /* C % heads != 0 */
    ASSERT_NULL(rcre_autograd_mhsa(tape, q, q, q, 4));

    RCRE_Tensor* ap = rcre_tensor_alloc_4d(arena, 1, 1, 1, 1);      /* H < kernel */
    ASSERT_NULL(rcre_autograd_avgpool2d(tape, ap, 2, 2));

    RCRE_Tensor* pf = rcre_tensor_alloc_4d(arena, 1, 1, 5, 5);      /* H % patch != 0 */
    ASSERT_NULL(rcre_autograd_patchify(tape, pf, 2));

    /* --- KL divergence accumulating: shape mismatch + logvar clamp guard --- */
    RCRE_Tensor* mu = rcre_tensor_alloc_1d(arena, 2);
    RCRE_Tensor* lv = rcre_tensor_alloc_1d(arena, 3);               /* numel mismatch */
    ASSERT_FLOAT_NEAR(0.0f, rcre_loss_kl_divergence_add(mu, lv, 0.5f, mu, lv), 0.0f);

    RCRE_Tensor* lv2 = rcre_tensor_alloc_1d(arena, 2);
    rcre_tensor_zero_grad(mu);
    rcre_tensor_zero_grad(lv2);
    float muv[2] = { 1.0f, 2.0f }, lvv[2] = { -25.0f, 0.5f };
    memcpy(rcre_tensor_f32_data(mu), muv, sizeof(muv));
    memcpy(rcre_tensor_f32_data(lv2), lvv, sizeof(lvv));
    (void)rcre_loss_kl_divergence_add(mu, lv2, 0.5f, mu, lv2);
    /* raw_lv = -25 is outside (-20, 20): logvar gradient must get ZERO
     * contribution there; d=1 gets beta*0.5*(exp(0.5)-1)/2. */
    ASSERT_FLOAT_NEAR(0.0f, lv2->grad[0], 0.0f);
    ASSERT_FLOAT_NEAR(0.5f * 0.5f * (expf(0.5f) - 1.0f) / 2.0f, lv2->grad[1], 1e-7f);
    ASSERT_FLOAT_NEAR(0.5f * 1.0f / 2.0f, mu->grad[0], 1e-7f);      /* beta * m / n */

    /* --- Clip grad norm edge cases --- */
    RCRE_Tensor* g = rcre_tensor_alloc_1d(arena, 2);
    g->grad[0] = 3.0f; g->grad[1] = 4.0f;
    RCRE_Tensor* gl[1] = { g };
    ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_clip_grad_norm(nullptr, 0, 1.0f), 0.0f);
    float pre = rcre_tensor_clip_grad_norm(gl, 1, 0.0f);            /* max_norm<=0: no clip */
    ASSERT_FLOAT_NEAR(5.0f, pre, 1e-6f);
    ASSERT_FLOAT_NEAR(3.0f, g->grad[0], 0.0f);

    /* --- SafeTensors / dataset guards --- */
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_gen_save_safetensors(nullptr, 1, "/tmp/x.st", arena));
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_gen_make_shapes_dataset(arena, 4, 4, 16, true, &rng, nullptr, nullptr)); /* height < 8 */

    /* --- Model param list undersized --- */
    RCRE_TinyUNetConfig ok = { .in_channels = 1, .height = 16, .width = 16,
                               .base_channels = 8, .num_groups = 4, .time_dim = 16 };
    RCRE_TinyUNet* unet = rcre_tinyunet_create(arena, ok, &rng);
    ASSERT_NOT_NULL(unet);
    RCRE_Tensor* small[50];
    ASSERT_UINT_EQ(0, rcre_tinyunet_params(unet, small, 50));       /* needs >= 66 */

    /* --- ConvAE decode with wrong latent dim --- */
    RCRE_ConvAEConfig gcfg = { .in_channels = 1, .height = 16, .width = 16,
                               .base_channels = 8, .latent_dim = 12, .output_sigmoid = true };
    RCRE_ConvAE* ae = rcre_convae_create(arena, gcfg, &rng);
    ASSERT_NOT_NULL(ae);
    RCRE_Tensor* badz = rcre_tensor_alloc_2d(arena, 2, 5);          /* latent 5 != 12 */
    ASSERT_NULL(rcre_convae_decode(ae, tape, badz));

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(arena);
}

/* Paths the original phase tests did not reach: unconditional DiT
 * (num_classes = 0), plain (non-denoising) AE, and the unlabeled
 * param_%zu SafeTensors fallback round-trip. */
static void test_phase0_unconditional_dit_plain_ae_unlabeled_ckpt(void) {
    rcre_hal_init();
    RCRE_Arena* perma = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* tape_arena = rcre_arena_create(32 * 1024 * 1024);
    ASSERT_NOT_NULL(perma);
    ASSERT_NOT_NULL(tape_arena);

    uint64_t rng = 606ULL;
    RCRE_Tensor* imgs = nullptr;
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_make_shapes_dataset(perma, 4, 16, 16, false, &rng, &imgs, nullptr));

    /* --- Unconditional DiT (num_classes = 0, class_labels = NULL) --- */
    RCRE_DiffSchedule* sched = rcre_diff_schedule_create(perma, 20, RCRE_DIFF_SCHED_COSINE, 0.0f, 0.0f);
    ASSERT_NOT_NULL(sched);
    RCRE_DiTConfig dcfg = {
        .in_channels = 1, .height = 16, .width = 16, .patch_size = 4,
        .embed_dim = 24, .num_heads = 4, .num_layers = 2, .mlp_dim = 48,
        .num_classes = 0
    };
    RCRE_DiT* dit = rcre_dit_create(perma, dcfg, &rng);
    ASSERT_NOT_NULL(dit);
    ASSERT_NULL(dit->class_emb);

    size_t d_np = rcre_dit_param_count(dit);
    RCRE_Tensor** d_params = (RCRE_Tensor**)rcre_arena_alloc(perma, d_np * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
    ASSERT_UINT_EQ(d_np, rcre_dit_params(dit, d_params, d_np));
    RCRE_TensorOptim* d_opt = rcre_tensor_optim_adamw_create(perma, d_params, d_np, 6e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(tape_arena, 128);
    ASSERT_NOT_NULL(tape);

    float u_first = 0.0f, u_last = 0.0f;
    for (int step = 0; step < 15; ++step) {
        float loss = rcre_dit_train_step(dit, sched, tape, d_opt, imgs, nullptr, &rng, 1.0f);
        ASSERT_TRUE(isfinite(loss));
        if (step == 0) u_first = loss;
        u_last = loss;
    }
    ASSERT_TRUE(u_last < u_first);

    /* Conditional model must REJECT a NULL label list; unconditional must accept it. */
    RCRE_DiTConfig cdcfg = dcfg; cdcfg.num_classes = 4;
    RCRE_DiT* cdit = rcre_dit_create(perma, cdcfg, &rng);
    ASSERT_NOT_NULL(cdit);
    RCRE_ArenaTemp ctmp = rcre_arena_temp_begin(tape_arena);
    ASSERT_NULL(rcre_dit_forward(cdit, tape, imgs, (const float[]){ 5.0f, 5.0f, 5.0f, 5.0f }, nullptr));
    rcre_arena_temp_end(ctmp);

    uint64_t us1 = 555ULL, us2 = 555ULL;
    RCRE_Tensor* g1 = rcre_dit_sample_ddim(dit, sched, perma, tape_arena, 1, nullptr, 5, 0.0f, &us1);
    RCRE_Tensor* g2 = rcre_dit_sample_ddim(dit, sched, perma, tape_arena, 1, nullptr, 5, 0.0f, &us2);
    ASSERT_NOT_NULL(g1);
    ASSERT_NOT_NULL(g2);
    for (size_t i = 0; i < g1->numel; ++i) {
        ASSERT_FLOAT_NEAR(rcre_tensor_f32_data(g1)[i], rcre_tensor_f32_data(g2)[i], 0.0f);
        ASSERT_TRUE(isfinite(rcre_tensor_f32_data(g1)[i]));
    }

    /* --- Plain AE (noise_std = 0) trains on clean targets --- */
    RCRE_ConvAEConfig acfg = { .in_channels = 1, .height = 16, .width = 16,
                               .base_channels = 8, .latent_dim = 12, .output_sigmoid = true };
    RCRE_ConvAE* ae = rcre_convae_create(perma, acfg, &rng);
    ASSERT_NOT_NULL(ae);
    RCRE_Tensor* params[RCRE_CONVAE_NUM_PARAMS];
    size_t np = rcre_convae_params(ae, params, RCRE_CONVAE_NUM_PARAMS);
    RCRE_TensorOptim* opt = rcre_tensor_optim_adamw_create(perma, params, np, 8e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f);
    float a_first = 0.0f, a_last = 0.0f;
    for (int step = 0; step < 40; ++step) {
        float loss = rcre_convae_train_step(ae, tape, opt, imgs, 0.0f, &rng, 2.0f);
        ASSERT_TRUE(isfinite(loss));
        if (step == 0) a_first = loss;
        a_last = loss;
    }
    ASSERT_TRUE(a_last < a_first * 0.5f);

    /* --- Unlabeled param_%zu SafeTensors fallback round-trip --- */
    RCRE_Tensor* u0 = rcre_tensor_alloc_1d(perma, 4);
    RCRE_Tensor* u1 = rcre_tensor_alloc_2d(perma, 2, 2);
    float src0[4] = { 1.5f, -2.5f, 0.25f, 7.75f };
    float src1[4] = { 0.5f, 1.5f, -0.5f, 2.5f };
    memcpy(rcre_tensor_f32_data(u0), src0, sizeof(src0));
    memcpy(rcre_tensor_f32_data(u1), src1, sizeof(src1));
    RCRE_Tensor* up[2] = { u0, u1 };
    const char* ckpt = "/tmp/rcre_test_unlabeled.safetensors";
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_save_safetensors(up, 2, ckpt, tape_arena));

    RCRE_Tensor* r0 = rcre_tensor_alloc_1d(perma, 4);
    RCRE_Tensor* r1 = rcre_tensor_alloc_2d(perma, 2, 2);
    RCRE_Tensor* rp[2] = { r0, r1 };
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_load_safetensors(rp, 2, ckpt));
    ASSERT_FLOAT_NEAR(src0[3], rcre_tensor_f32_data(r0)[3], 0.0f);
    ASSERT_FLOAT_NEAR(src1[2], rcre_tensor_f32_data(r1)[2], 0.0f);
    /* Shape mismatch must fail with a shape error, not corrupt the target. */
    RCRE_Tensor* too_big = rcre_tensor_alloc_1d(perma, 8);
    RCRE_Tensor* tp[1] = { too_big };
    ASSERT_INT_EQ(RCRE_ERR_SHAPE_MISMATCH, rcre_gen_load_safetensors(tp, 1, ckpt));
    remove(ckpt);

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(tape_arena);
    rcre_arena_destroy(perma);
}

static void test_phase5_lifecycle_prune_compress_quantize(void) {
    rcre_hal_init();
    RCRE_Arena* perma = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(perma);
    ASSERT_NOT_NULL(scratch);

    uint64_t rng = 505ULL;
    RCRE_DiTConfig dcfg = {
        .in_channels = 1, .height = 16, .width = 16, .patch_size = 4,
        .embed_dim = 24, .num_heads = 4, .num_layers = 2, .mlp_dim = 48, .num_classes = 4
    };
    RCRE_DiT* m1 = rcre_dit_create(perma, dcfg, &rng);
    RCRE_DiT* m2 = rcre_dit_create(perma, dcfg, &rng);
    ASSERT_NOT_NULL(m1);
    ASSERT_NOT_NULL(m2);

    size_t np = rcre_dit_param_count(m1);
    RCRE_Tensor** p1 = (RCRE_Tensor**)rcre_arena_alloc(perma, np * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
    RCRE_Tensor** p2 = (RCRE_Tensor**)rcre_arena_alloc(perma, np * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
    ASSERT_UINT_EQ(np, rcre_dit_params(m1, p1, np));
    ASSERT_UINT_EQ(np, rcre_dit_params(m2, p2, np));

    /* 1. Unstructured Magnitude Pruning (50%) */
    RCRE_GenModelStats pst = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_prune_magnitude(p1, np, 0.50f, true, scratch, &pst));
    ASSERT_TRUE(pst.sparsity >= 0.40f && pst.sparsity <= 0.75f);

    /* 2. Lossless RCRZ Compressed Archive ("RGC1") — 0-ULP bit-exact round-trip */
    const char* rgc_path = "/tmp/rcre_test_dit.rgc1";
    RCRE_GenModelStats cst = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_save_compressed(p1, np, rgc_path, scratch, &cst));
    ASSERT_TRUE(cst.compression_ratio > 1.3f);
    ASSERT_FLOAT_NEAR(99.0f, cst.sqnr_db, 1e-4f);

    char fmt[32] = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_gen_load_any(p2, np, rgc_path, scratch, fmt, sizeof(fmt)));
    for (size_t k = 0; k < np; ++k) {
        const float* a = rcre_tensor_f32_data(p1[k]);
        const float* b = rcre_tensor_f32_data(p2[k]);
        for (size_t i = 0; i < p1[k]->numel; ++i) {
            ASSERT_FLOAT_NEAR(a[i], b[i], 0.0f);
        }
    }
    remove(rgc_path);

    /* 3. Post-Training Quantization ("RGQ1") across F16, BF16, Q8_0, Q4_0 */
    const RCRE_GenQuantType qtypes[4] = {
        RCRE_GEN_QUANT_F16, RCRE_GEN_QUANT_BF16, RCRE_GEN_QUANT_Q8_0, RCRE_GEN_QUANT_Q4_0
    };
    const float min_ratio[4] = { 1.85f, 1.85f, 3.4f, 6.0f };
    const float min_sqnr[4]  = { 50.0f, 35.0f, 30.0f, 12.0f };
    const char* rgq_path = "/tmp/rcre_test_dit.rgq1";

    for (int qi = 0; qi < 4; ++qi) {
        RCRE_GenModelStats qst = {0};
        ASSERT_INT_EQ(RCRE_OK, rcre_gen_save_quantized(p1, np, qtypes[qi], rgq_path, scratch, &qst));
        ASSERT_TRUE(qst.compression_ratio >= min_ratio[qi]);
        ASSERT_TRUE(qst.sqnr_db >= min_sqnr[qi]);

        RCRE_GenQuantType loaded_qt = 0;
        ASSERT_INT_EQ(RCRE_OK, rcre_gen_load_quantized(p2, np, rgq_path, &loaded_qt));
        ASSERT_INT_EQ((int)qtypes[qi], (int)loaded_qt);

        ASSERT_INT_EQ(RCRE_OK, rcre_gen_load_any(p2, np, rgq_path, scratch, fmt, sizeof(fmt)));
        remove(rgq_path);
    }

    /* 4. Error paths: invalid sparsity & nonexistent files */
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_gen_prune_magnitude(p1, np, 1.0f, true, scratch, &pst));
    ASSERT_INT_EQ(RCRE_ERR_IO, rcre_gen_load_any(p2, np, "/tmp/nonexistent_rcre_ckpt.rgq1", scratch, fmt, sizeof(fmt)));

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(perma);
}

int main(void) {
    test_suite_begin("RCRE Generative & Vision Suite (AE / VAE / Diffusion / ViT / DiT)");
    TEST_RUN(test_phase1_convae_and_denoising);
    TEST_RUN(test_phase2_convvae_elbo_and_sampling);
    TEST_RUN(test_phase3_diffusion_schedule_and_tinyunet);
    TEST_RUN(test_phase4_vit_classification_and_dit_diffusion);
    TEST_RUN(test_phase0_schedule_reference_and_validation);
    TEST_RUN(test_phase0_determinism_suite);
    TEST_RUN(test_phase0_error_paths);
    TEST_RUN(test_phase0_unconditional_dit_plain_ae_unlabeled_ckpt);
    TEST_RUN(test_phase5_lifecycle_prune_compress_quantize);
    return test_suite_report();
}
