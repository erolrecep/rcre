/**
 * @file deploy_infer.c
 * @brief Self-Contained Single-Header Edge Deployment Binary for RCRE Generative Models
 *
 * Compiles anywhere with ZERO external dependencies using ONLY dist/rcre.h:
 *   cc -std=c2x -O3 -ffp-contract=off deploy_infer.c -lm -lpthread -o rcre_infer
 *
 * Supports all 5 models (ae, vae, diffusion, vit, dit) and auto-detects
 * SafeTensors F32 (.safetensors), RCRZ Compressed (.rgc1), and Quantized (.rgq1: F16/BF16/Q8_0/Q4_0).
 */

#define RCRE_IMPLEMENTATION
#include "rcre.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double deploy_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

static void deploy_save_png_grid(const RCRE_Tensor* batch, bool signed_range,
                                 const char* path, RCRE_Arena* scratch) {
    if (!batch || batch->ndim != 4 || !path) return;
    size_t N = batch->shape[0], H = batch->shape[2], W = batch->shape[3];
    size_t cols = N < 4 ? N : 4;
    size_t rows = (N + cols - 1U) / cols;
    size_t scale = 4U, pad = 2U;
    size_t out_h = rows * (H * scale + pad) + pad;
    size_t out_w = cols * (W * scale + pad) + pad;
    size_t shape[3] = { out_h, out_w, 3 };
    RCRE_Tensor rgb = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch);
    if (!rgb.data) return;
    uint8_t* px = (uint8_t*)rgb.data;
    memset(px, 28, out_h * out_w * 3U);

    const float* d = rcre_tensor_f32_data(batch);
    for (size_t n = 0; n < N; ++n) {
        size_t r = n / cols, c = n % cols;
        size_t y0 = pad + r * (H * scale + pad);
        size_t x0 = pad + c * (W * scale + pad);
        const float* img = d + n * H * W;
        for (size_t y = 0; y < H; ++y) {
            for (size_t x = 0; x < W; ++x) {
                float v = img[y * W + x];
                if (signed_range) v = 0.5f * (v + 1.0f);
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                uint8_t g = (uint8_t)(v * 255.0f + 0.5f);
                for (size_t sy = 0; sy < scale; ++sy) {
                    for (size_t sx = 0; sx < scale; ++sx) {
                        size_t o = ((y0 + y * scale + sy) * out_w + (x0 + x * scale + sx)) * 3U;
                        px[o + 0] = g; px[o + 1] = g; px[o + 2] = g;
                    }
                }
            }
        }
    }
    (void)rcre_codec_save_png(&rgb, path);
}

int main(int argc, char** argv) {
    rcre_hal_init();

    const char* model   = nullptr;
    const char* weights = nullptr;
    const char* out_png = nullptr;
    size_t ddim_steps   = 10;
    uint64_t seed       = 777ULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)        model = argv[++i];
        else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) weights = argv[++i];
        else if (strcmp(argv[i], "--out-png") == 0 && i + 1 < argc) out_png = argv[++i];
        else if (strcmp(argv[i], "--ddim-steps") == 0 && i + 1 < argc) ddim_steps = (size_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)    seed = strtoull(argv[++i], nullptr, 10);
    }

    if (!model || !weights) {
        printf("Usage: %s --model <ae|vae|diffusion|vit|dit> --weights <file> [--out-png out.png] [--ddim-steps 10]\n", argv[0]);
        return 1;
    }

    RCRE_Arena* perma   = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(32 * 1024 * 1024);
    uint64_t rng = seed;
    char fmt[32] = {0};

    double t0 = deploy_now_ms();
    int rc = 0;

    if (strcmp(model, "ae") == 0) {
        RCRE_ConvAEConfig cfg = { 1, 16, 16, 8, 12, true };
        RCRE_ConvAE* m = rcre_convae_create(perma, cfg, &rng);
        RCRE_Tensor* p[RCRE_CONVAE_NUM_PARAMS];
        size_t np = rcre_convae_params(m, p, RCRE_CONVAE_NUM_PARAMS);
        if (rcre_gen_load_any(p, np, weights, scratch, fmt, sizeof(fmt)) != RCRE_OK) { rc = 2; goto done; }
        RCRE_Tensor* imgs = nullptr;
        (void)rcre_gen_make_shapes_dataset(perma, 4, 16, 16, true, &rng, &imgs, nullptr);
        RCRE_TensorTape* tape = rcre_tensor_tape_create(scratch, 64);
        RCRE_Tensor* out = rcre_convae_forward(m, tape, imgs);
        float psnr = rcre_gen_psnr(imgs, out, 1.0f);
        printf("{\"model\":\"ae\",\"format\":\"%s\",\"psnr_db\":%.2f,\"latency_ms\":%.2f}\n", fmt, psnr, deploy_now_ms() - t0);
        if (out_png) deploy_save_png_grid(out, false, out_png, scratch);
        rcre_tensor_tape_destroy(tape);
    } else if (strcmp(model, "vae") == 0) {
        RCRE_ConvVAEConfig cfg = { 1, 16, 16, 8, 8, true };
        RCRE_ConvVAE* m = rcre_convvae_create(perma, cfg, &rng);
        RCRE_Tensor* p[RCRE_CONVVAE_NUM_PARAMS];
        size_t np = rcre_convvae_params(m, p, RCRE_CONVVAE_NUM_PARAMS);
        if (rcre_gen_load_any(p, np, weights, scratch, fmt, sizeof(fmt)) != RCRE_OK) { rc = 2; goto done; }
        RCRE_Tensor* out = rcre_convvae_sample(m, scratch, 4, &rng);
        printf("{\"model\":\"vae\",\"format\":\"%s\",\"samples\":4,\"latency_ms\":%.2f}\n", fmt, deploy_now_ms() - t0);
        if (out_png) deploy_save_png_grid(out, false, out_png, scratch);
    } else if (strcmp(model, "diffusion") == 0 || strcmp(model, "unet") == 0) {
        RCRE_DiffSchedule* sc = rcre_diff_schedule_create(perma, 30, RCRE_DIFF_SCHED_COSINE, 0.0f, 0.0f);
        RCRE_TinyUNetConfig cfg = { 1, 16, 16, 8, 4, 16 };
        RCRE_TinyUNet* m = rcre_tinyunet_create(perma, cfg, &rng);
        RCRE_Tensor* p[RCRE_TINYUNET_MAX_PARAMS];
        size_t np = rcre_tinyunet_params(m, p, RCRE_TINYUNET_MAX_PARAMS);
        if (rcre_gen_load_any(p, np, weights, scratch, fmt, sizeof(fmt)) != RCRE_OK) { rc = 2; goto done; }
        RCRE_Tensor* out = rcre_tinyunet_sample_ddim(m, sc, perma, scratch, 4, ddim_steps, 0.0f, &rng);
        printf("{\"model\":\"diffusion\",\"format\":\"%s\",\"ddim_steps\":%zu,\"samples\":4,\"latency_ms\":%.2f}\n", fmt, ddim_steps, deploy_now_ms() - t0);
        if (out_png) deploy_save_png_grid(out, true, out_png, scratch);
    } else if (strcmp(model, "vit") == 0) {
        RCRE_ViTConfig cfg = { 1, 16, 16, 4, 24, 4, 2, 48, 4 };
        RCRE_ViT* m = rcre_vit_create(perma, cfg, &rng);
        size_t np = rcre_vit_param_count(m);
        RCRE_Tensor** p = (RCRE_Tensor**)rcre_arena_alloc(perma, np * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
        (void)rcre_vit_params(m, p, np);
        if (rcre_gen_load_any(p, np, weights, scratch, fmt, sizeof(fmt)) != RCRE_OK) { rc = 2; goto done; }
        RCRE_Tensor* imgs = nullptr; size_t* labels = nullptr;
        (void)rcre_gen_make_shapes_dataset(perma, 16, 16, 16, false, &rng, &imgs, &labels);
        size_t preds[16] = {0};
        (void)rcre_vit_predict(m, scratch, imgs, preds);
        size_t ok = 0; for (size_t i = 0; i < 16; ++i) if (preds[i] == labels[i]) ok++;
        printf("{\"model\":\"vit\",\"format\":\"%s\",\"correct\":%zu,\"total\":16,\"accuracy_pct\":%.1f,\"latency_ms\":%.2f}\n",
               fmt, ok, 100.0 * (double)ok / 16.0, deploy_now_ms() - t0);
    } else if (strcmp(model, "dit") == 0) {
        RCRE_DiffSchedule* sc = rcre_diff_schedule_create(perma, 30, RCRE_DIFF_SCHED_COSINE, 0.0f, 0.0f);
        RCRE_DiTConfig cfg = { 1, 16, 16, 4, 24, 4, 2, 48, 4 };
        RCRE_DiT* m = rcre_dit_create(perma, cfg, &rng);
        size_t np = rcre_dit_param_count(m);
        RCRE_Tensor** p = (RCRE_Tensor**)rcre_arena_alloc(perma, np * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
        (void)rcre_dit_params(m, p, np);
        if (rcre_gen_load_any(p, np, weights, scratch, fmt, sizeof(fmt)) != RCRE_OK) { rc = 2; goto done; }
        size_t cls[4] = { 0, 1, 2, 3 };
        RCRE_Tensor* out = rcre_dit_sample_ddim(m, sc, perma, scratch, 4, cls, ddim_steps, 0.0f, &rng);
        printf("{\"model\":\"dit\",\"format\":\"%s\",\"ddim_steps\":%zu,\"samples\":4,\"latency_ms\":%.2f}\n", fmt, ddim_steps, deploy_now_ms() - t0);
        if (out_png) deploy_save_png_grid(out, true, out_png, scratch);
    } else {
        fprintf(stderr, "Unknown model: %s\n", model);
        rc = 1;
    }

done:
    if (rc == 2) fprintf(stderr, "Failed to load weights: %s\n", weights);
    rcre_arena_destroy(scratch);
    rcre_arena_destroy(perma);
    return rc;
}
