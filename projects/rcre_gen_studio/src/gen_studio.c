/**
 * @file gen_studio.c
 * @brief RCRE Generative & Vision Studio — Full Lifecycle CLI
 *        (train | compress | quantize | infer | serve | demo)
 *        across ALL 5 architectures: ae, vae, diffusion (TinyUNet), vit, dit.
 *
 * Pure ISO C23, zero external dependencies.
 */

#include "rcre.h"
#include "rcre_gen.h"
#include "rcre_codec.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#define RCRE_STUDIO_POSIX 1
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

typedef enum {
    MODEL_AE = 0,
    MODEL_VAE,
    MODEL_DIFFUSION,
    MODEL_VIT,
    MODEL_DIT,
    MODEL_ALL
} StudioModelKind;

typedef struct {
    StudioModelKind kind;
    const char*     name;
    RCRE_ConvAE*    ae;
    RCRE_ConvVAE*   vae;
    RCRE_TinyUNet*  unet;
    RCRE_ViT*       vit;
    RCRE_DiT*       dit;
    RCRE_DiffSchedule* sched;
    RCRE_Tensor**   params;
    size_t          num_params;
    size_t          total_scalars;
} StudioModel;

static double studio_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

static bool studio_parse_model(const char* s, StudioModelKind* out) {
    if (!s || !out) return false;
    if (strcmp(s, "ae") == 0)                                     { *out = MODEL_AE; return true; }
    if (strcmp(s, "vae") == 0)                                    { *out = MODEL_VAE; return true; }
    if (strcmp(s, "diffusion") == 0 || strcmp(s, "unet") == 0)    { *out = MODEL_DIFFUSION; return true; }
    if (strcmp(s, "vit") == 0)                                    { *out = MODEL_VIT; return true; }
    if (strcmp(s, "dit") == 0)                                    { *out = MODEL_DIT; return true; }
    if (strcmp(s, "all") == 0)                                    { *out = MODEL_ALL; return true; }
    return false;
}

static bool studio_parse_qtype(const char* s, RCRE_GenQuantType* out) {
    if (!s || !out) return false;
    if (strcmp(s, "f16") == 0  || strcmp(s, "fp16") == 0)         { *out = RCRE_GEN_QUANT_F16;  return true; }
    if (strcmp(s, "bf16") == 0)                                   { *out = RCRE_GEN_QUANT_BF16; return true; }
    if (strcmp(s, "q8_0") == 0 || strcmp(s, "q8") == 0 || strcmp(s, "8") == 0) { *out = RCRE_GEN_QUANT_Q8_0; return true; }
    if (strcmp(s, "q4_0") == 0 || strcmp(s, "q4") == 0 || strcmp(s, "4") == 0) { *out = RCRE_GEN_QUANT_Q4_0; return true; }
    return false;
}

static const char* studio_qtype_name(RCRE_GenQuantType q) {
    switch (q) {
        case RCRE_GEN_QUANT_F16:  return "F16";
        case RCRE_GEN_QUANT_BF16: return "BF16";
        case RCRE_GEN_QUANT_Q8_0: return "Q8_0";
        case RCRE_GEN_QUANT_Q4_0: return "Q4_0";
        default: return "UNKNOWN";
    }
}

static bool studio_model_init(StudioModel* sm, StudioModelKind kind, RCRE_Arena* perma, uint64_t* rng) {
    memset(sm, 0, sizeof(*sm));
    sm->kind = kind;

    if (kind == MODEL_AE) {
        sm->name = "ae";
        RCRE_ConvAEConfig cfg = {
            .in_channels = 1, .height = 16, .width = 16,
            .base_channels = 8, .latent_dim = 12, .output_sigmoid = true
        };
        sm->ae = rcre_convae_create(perma, cfg, rng);
        if (!sm->ae) return false;
        sm->params = (RCRE_Tensor**)rcre_arena_alloc(perma, RCRE_CONVAE_NUM_PARAMS * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
        sm->num_params = rcre_convae_params(sm->ae, sm->params, RCRE_CONVAE_NUM_PARAMS);
    } else if (kind == MODEL_VAE) {
        sm->name = "vae";
        RCRE_ConvVAEConfig cfg = {
            .in_channels = 1, .height = 16, .width = 16,
            .base_channels = 8, .latent_dim = 8, .output_sigmoid = true
        };
        sm->vae = rcre_convvae_create(perma, cfg, rng);
        if (!sm->vae) return false;
        sm->params = (RCRE_Tensor**)rcre_arena_alloc(perma, RCRE_CONVVAE_NUM_PARAMS * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
        sm->num_params = rcre_convvae_params(sm->vae, sm->params, RCRE_CONVVAE_NUM_PARAMS);
    } else if (kind == MODEL_DIFFUSION) {
        sm->name = "diffusion";
        sm->sched = rcre_diff_schedule_create(perma, 30, RCRE_DIFF_SCHED_COSINE, 0.0f, 0.0f);
        RCRE_TinyUNetConfig cfg = {
            .in_channels = 1, .height = 16, .width = 16,
            .base_channels = 8, .num_groups = 4, .time_dim = 16
        };
        sm->unet = rcre_tinyunet_create(perma, cfg, rng);
        if (!sm->sched || !sm->unet) return false;
        sm->params = (RCRE_Tensor**)rcre_arena_alloc(perma, RCRE_TINYUNET_MAX_PARAMS * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
        sm->num_params = rcre_tinyunet_params(sm->unet, sm->params, RCRE_TINYUNET_MAX_PARAMS);
    } else if (kind == MODEL_VIT) {
        sm->name = "vit";
        RCRE_ViTConfig cfg = {
            .in_channels = 1, .height = 16, .width = 16, .patch_size = 4,
            .embed_dim = 24, .num_heads = 4, .num_layers = 2, .mlp_dim = 48, .num_classes = 4
        };
        sm->vit = rcre_vit_create(perma, cfg, rng);
        if (!sm->vit) return false;
        size_t cap = rcre_vit_param_count(sm->vit);
        sm->params = (RCRE_Tensor**)rcre_arena_alloc(perma, cap * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
        sm->num_params = rcre_vit_params(sm->vit, sm->params, cap);
    } else if (kind == MODEL_DIT) {
        sm->name = "dit";
        sm->sched = rcre_diff_schedule_create(perma, 30, RCRE_DIFF_SCHED_COSINE, 0.0f, 0.0f);
        RCRE_DiTConfig cfg = {
            .in_channels = 1, .height = 16, .width = 16, .patch_size = 4,
            .embed_dim = 24, .num_heads = 4, .num_layers = 2, .mlp_dim = 48, .num_classes = 4
        };
        sm->dit = rcre_dit_create(perma, cfg, rng);
        if (!sm->sched || !sm->dit) return false;
        size_t cap = rcre_dit_param_count(sm->dit);
        sm->params = (RCRE_Tensor**)rcre_arena_alloc(perma, cap * sizeof(RCRE_Tensor*), alignof(RCRE_Tensor*));
        sm->num_params = rcre_dit_params(sm->dit, sm->params, cap);
    } else {
        return false;
    }

    sm->total_scalars = 0;
    for (size_t i = 0; i < sm->num_params; ++i) sm->total_scalars += sm->params[i]->numel;
    return sm->num_params > 0;
}

static void print_ascii_strip(const RCRE_Tensor* batch, size_t max_imgs, bool signed_range, const char* title) {
    static const char* ramp = " .:-=+*#%@";
    size_t N = batch->shape[0] < max_imgs ? batch->shape[0] : max_imgs;
    size_t H = batch->shape[2];
    size_t W = batch->shape[3];
    const float* d = rcre_tensor_f32_data(batch);

    printf("  [%s]\n", title);
    for (size_t y = 0; y < H; y += 2) {
        printf("  ");
        for (size_t n = 0; n < N; ++n) {
            const float* img = d + n * H * W;
            for (size_t x = 0; x < W; ++x) {
                float v = img[y * W + x];
                if (signed_range) v = 0.5f * (v + 1.0f);
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                int idx = (int)(v * 9.0f + 0.5f);
                putchar(ramp[idx]);
            }
            printf("  ");
        }
        putchar('\n');
    }
}

static void save_png_grid(const RCRE_Tensor* batch, size_t cols, bool signed_range,
                          const char* path, RCRE_Arena* scratch) {
    if (!batch || batch->ndim != 4 || cols == 0 || !path || path[0] == '\0') return;
    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);

    size_t N = batch->shape[0];
    size_t H = batch->shape[2];
    size_t W = batch->shape[3];
    size_t rows = (N + cols - 1U) / cols;
    size_t scale = 4U;
    size_t pad = 2U;
    size_t out_h = rows * (H * scale + pad) + pad;
    size_t out_w = cols * (W * scale + pad) + pad;

    size_t shape[3] = { out_h, out_w, 3 };
    RCRE_Tensor rgb = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, scratch);
    if (!rgb.data) {
        rcre_arena_temp_end(tmp);
        return;
    }
    uint8_t* px = (uint8_t*)rgb.data;
    memset(px, 28, out_h * out_w * 3U);

    const float* d = rcre_tensor_f32_data(batch);
    for (size_t n = 0; n < N; ++n) {
        size_t r = n / cols;
        size_t c = n % cols;
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
                        size_t oidx = ((y0 + y * scale + sy) * out_w + (x0 + x * scale + sx)) * 3U;
                        px[oidx + 0] = g;
                        px[oidx + 1] = g;
                        px[oidx + 2] = g;
                    }
                }
            }
        }
    }

    if (rcre_codec_save_png(&rgb, path) == RCRE_OK) {
        printf("  -> Saved PNG grid: %s (%zux%zu)\n", path, out_w, out_h);
    }
    rcre_arena_temp_end(tmp);
}

/* ========================================================================= */
/* Unified Lifecycle Commands: train, compress, quantize, infer, serve       */
/* ========================================================================= */

static int cmd_train(StudioModelKind kind, int steps, float lr_override, uint64_t seed, const char* out_path) {
    RCRE_Arena* perma = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* tape_arena = rcre_arena_create(32 * 1024 * 1024);
    uint64_t rng = seed;

    StudioModel sm;
    if (!studio_model_init(&sm, kind, perma, &rng)) {
        fprintf(stderr, "[train] Failed to initialize model.\n");
        rcre_arena_destroy(tape_arena); rcre_arena_destroy(perma);
        return 1;
    }

    bool zero_to_one = (kind == MODEL_AE || kind == MODEL_VAE);
    size_t n_data = (kind == MODEL_VIT) ? 16U : (kind == MODEL_DIFFUSION ? 4U : 8U);
    RCRE_Tensor* imgs = nullptr;
    size_t* labels = nullptr;
    (void)rcre_gen_make_shapes_dataset(perma, n_data, 16, 16, zero_to_one, &rng, &imgs, &labels);

    float default_lr = (kind == MODEL_VIT) ? 1e-2f :
                       (kind == MODEL_DIFFUSION) ? 5e-3f :
                       (kind == MODEL_DIT) ? 6e-3f : 8e-3f;
    float lr = (lr_override > 0.0f) ? lr_override : default_lr;

    RCRE_TensorOptim* opt = rcre_tensor_optim_adamw_create(perma, sm.params, sm.num_params, lr, 0.9f, 0.999f, 1e-8f, 1e-4f);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(tape_arena, 256);

    double t0 = studio_now_ms();
    float l0 = 0.0f, l1 = 0.0f;
    for (int s = 0; s < steps; ++s) {
        float loss = 0.0f;
        if (kind == MODEL_AE) {
            loss = rcre_convae_train_step(sm.ae, tape, opt, imgs, 0.08f, &rng, 2.0f);
        } else if (kind == MODEL_VAE) {
            float beta = 0.005f + 0.045f * ((float)s / (float)(steps > 1 ? steps - 1 : 1));
            loss = rcre_convvae_train_step(sm.vae, tape, opt, imgs, beta, &rng, 2.0f, nullptr, nullptr);
        } else if (kind == MODEL_DIFFUSION) {
            loss = rcre_tinyunet_train_step(sm.unet, sm.sched, tape, opt, imgs, &rng, 1.0f);
        } else if (kind == MODEL_VIT) {
            loss = rcre_vit_train_step(sm.vit, tape, opt, imgs, labels, 2.0f);
        } else if (kind == MODEL_DIT) {
            loss = rcre_dit_train_step(sm.dit, sm.sched, tape, opt, imgs, labels, &rng, 1.0f);
        }
        if (s == 0) l0 = loss;
        l1 = loss;
    }
    double dt = studio_now_ms() - t0;

    printf("[train:%s] params=%zu (%zu B F32) | steps=%d | lr=%.4g | loss: %.5f -> %.5f | time=%.1f ms\n",
           sm.name, sm.total_scalars, sm.total_scalars * 4U, steps, lr, l0, l1, dt);

    int rc = 0;
    if (out_path && out_path[0] != '\0') {
        if (rcre_gen_save_safetensors(sm.params, sm.num_params, out_path, tape_arena) == RCRE_OK) {
            printf("  -> Saved SafeTensors checkpoint: %s\n", out_path);
        } else {
            fprintf(stderr, "  !! Failed to save checkpoint to %s\n", out_path);
            rc = 1;
        }
    }

    rcre_tensor_tape_destroy(tape);
    rcre_arena_destroy(tape_arena);
    rcre_arena_destroy(perma);
    return rc;
}

static int cmd_compress(StudioModelKind kind, const char* weights_in, float sparsity, const char* out_path) {
    if (!weights_in || !out_path) {
        fprintf(stderr, "[compress] --weights and --out are required.\n");
        return 1;
    }
    RCRE_Arena* perma = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(16 * 1024 * 1024);
    uint64_t rng = 1ULL;

    StudioModel sm;
    if (!studio_model_init(&sm, kind, perma, &rng)) return 1;

    char fmt[32] = {0};
    if (rcre_gen_load_any(sm.params, sm.num_params, weights_in, scratch, fmt, sizeof(fmt)) != RCRE_OK) {
        fprintf(stderr, "[compress:%s] Failed to load %s\n", sm.name, weights_in);
        rcre_arena_destroy(scratch); rcre_arena_destroy(perma);
        return 1;
    }

    RCRE_GenModelStats pst = {0}, cst = {0};
    if (rcre_gen_prune_magnitude(sm.params, sm.num_params, sparsity, true, scratch, &pst) != RCRE_OK ||
        rcre_gen_save_compressed(sm.params, sm.num_params, out_path, scratch, &cst) != RCRE_OK) {
        fprintf(stderr, "[compress:%s] Prune/compress failed.\n", sm.name);
        rcre_arena_destroy(scratch); rcre_arena_destroy(perma);
        return 1;
    }

    printf("[compress:%s] in=%s (%s) | pruned=%.1f%% (SQNR=%.2f dB) | raw=%zu B -> %zu B (%.2fx) | out=%s\n",
           sm.name, weights_in, fmt, pst.sparsity * 100.0f, pst.sqnr_db,
           cst.raw_bytes, cst.artifact_bytes, cst.compression_ratio, out_path);

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(perma);
    return 0;
}

static int cmd_quantize(StudioModelKind kind, const char* weights_in, RCRE_GenQuantType qtype, const char* out_path) {
    if (!weights_in || !out_path) {
        fprintf(stderr, "[quantize] --weights and --out are required.\n");
        return 1;
    }
    RCRE_Arena* perma = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(16 * 1024 * 1024);
    uint64_t rng = 1ULL;

    StudioModel sm;
    if (!studio_model_init(&sm, kind, perma, &rng)) return 1;

    char fmt[32] = {0};
    if (rcre_gen_load_any(sm.params, sm.num_params, weights_in, scratch, fmt, sizeof(fmt)) != RCRE_OK) {
        fprintf(stderr, "[quantize:%s] Failed to load %s\n", sm.name, weights_in);
        rcre_arena_destroy(scratch); rcre_arena_destroy(perma);
        return 1;
    }

    RCRE_GenModelStats qst = {0};
    if (rcre_gen_save_quantized(sm.params, sm.num_params, qtype, out_path, scratch, &qst) != RCRE_OK) {
        fprintf(stderr, "[quantize:%s] Quantization failed.\n", sm.name);
        rcre_arena_destroy(scratch); rcre_arena_destroy(perma);
        return 1;
    }

    printf("[quantize:%s] in=%s (%s) -> %s | raw=%zu B -> %zu B (%.2fx) | RMSE=%.5f | SQNR=%.2f dB | out=%s\n",
           sm.name, weights_in, fmt, studio_qtype_name(qtype),
           qst.raw_bytes, qst.artifact_bytes, qst.compression_ratio,
           qst.rmse, qst.sqnr_db, out_path);

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(perma);
    return 0;
}

static int run_model_eval(
    const StudioModel* sm, RCRE_Arena* perma, RCRE_Arena* scratch,
    size_t ddim_steps, uint64_t seed, bool ascii, const char* out_png,
    char* out_json, size_t max_json)
{
    uint64_t rng = seed;
    bool zero_to_one = (sm->kind == MODEL_AE || sm->kind == MODEL_VAE);
    size_t n_eval = (sm->kind == MODEL_VIT) ? 16U : 4U;
    RCRE_Tensor* imgs = nullptr;
    size_t* labels = nullptr;
    (void)rcre_gen_make_shapes_dataset(perma, n_eval, 16, 16, zero_to_one, &rng, &imgs, &labels);

    RCRE_TensorTape* tape = rcre_tensor_tape_create(scratch, 256);
    double t0 = studio_now_ms();
    RCRE_Tensor* vis_batch = nullptr;
    bool signed_range = !zero_to_one;
    char metric_str[256] = {0};

    if (sm->kind == MODEL_AE) {
        vis_batch = rcre_convae_forward(sm->ae, tape, imgs);
        float psnr = rcre_gen_psnr(imgs, vis_batch, 1.0f);
        float mse  = rcre_loss_mse(vis_batch, imgs, nullptr);
        double dt  = studio_now_ms() - t0;
        snprintf(metric_str, sizeof(metric_str), "mse=%.5f, psnr=%.2f dB, latency=%.2f ms", mse, psnr, dt);
        if (out_json) snprintf(out_json, max_json, "{\"model\":\"ae\",\"mse\":%.6f,\"psnr_db\":%.3f,\"latency_ms\":%.3f}", mse, psnr, dt);
    } else if (sm->kind == MODEL_VAE) {
        vis_batch = rcre_convvae_sample(sm->vae, scratch, 4, &rng);
        RCRE_Tensor* recon = rcre_convvae_forward(sm->vae, tape, imgs, &rng, nullptr, nullptr, nullptr);
        float psnr = rcre_gen_psnr(imgs, recon, 1.0f);
        double dt  = studio_now_ms() - t0;
        snprintf(metric_str, sizeof(metric_str), "recon_psnr=%.2f dB, sampled=4, latency=%.2f ms", psnr, dt);
        if (out_json) snprintf(out_json, max_json, "{\"model\":\"vae\",\"recon_psnr_db\":%.3f,\"samples\":4,\"latency_ms\":%.3f}", psnr, dt);
    } else if (sm->kind == MODEL_DIFFUSION) {
        vis_batch = rcre_tinyunet_sample_ddim(sm->unet, sm->sched, perma, scratch, 4, ddim_steps, 0.0f, &rng);
        double dt = studio_now_ms() - t0;
        float mean_abs = 0.0f;
        const float* d = rcre_tensor_f32_data(vis_batch);
        for (size_t i = 0; i < vis_batch->numel; ++i) mean_abs += fabsf(d[i]);
        mean_abs /= (float)vis_batch->numel;
        snprintf(metric_str, sizeof(metric_str), "ddim_steps=%zu, samples=4, mean_abs=%.3f, latency=%.2f ms", ddim_steps, mean_abs, dt);
        if (out_json) snprintf(out_json, max_json, "{\"model\":\"diffusion\",\"ddim_steps\":%zu,\"samples\":4,\"mean_abs\":%.4f,\"latency_ms\":%.3f}", ddim_steps, mean_abs, dt);
    } else if (sm->kind == MODEL_VIT) {
        size_t preds[16] = {0};
        (void)rcre_vit_predict(sm->vit, scratch, imgs, preds);
        double dt = studio_now_ms() - t0;
        size_t ok = 0;
        for (size_t i = 0; i < n_eval; ++i) if (preds[i] == labels[i]) ok++;
        double acc = 100.0 * (double)ok / (double)n_eval;
        snprintf(metric_str, sizeof(metric_str), "accuracy=%zu/%zu (%.1f%%), latency=%.2f ms", ok, n_eval, acc, dt);
        if (out_json) snprintf(out_json, max_json, "{\"model\":\"vit\",\"correct\":%zu,\"total\":%zu,\"accuracy_pct\":%.2f,\"latency_ms\":%.3f}", ok, n_eval, acc, dt);
    } else if (sm->kind == MODEL_DIT) {
        size_t cls[4] = { 0, 1, 2, 3 };
        vis_batch = rcre_dit_sample_ddim(sm->dit, sm->sched, perma, scratch, 4, cls, ddim_steps, 0.0f, &rng);
        double dt = studio_now_ms() - t0;
        float mean_abs = 0.0f;
        const float* d = rcre_tensor_f32_data(vis_batch);
        for (size_t i = 0; i < vis_batch->numel; ++i) mean_abs += fabsf(d[i]);
        mean_abs /= (float)vis_batch->numel;
        snprintf(metric_str, sizeof(metric_str), "ddim_steps=%zu, classes=[0,1,2,3], mean_abs=%.3f, latency=%.2f ms", ddim_steps, mean_abs, dt);
        if (out_json) snprintf(out_json, max_json, "{\"model\":\"dit\",\"ddim_steps\":%zu,\"classes\":[0,1,2,3],\"mean_abs\":%.4f,\"latency_ms\":%.3f}", ddim_steps, mean_abs, dt);
    }

    if (!out_json) {
        printf("  Result: %s\n", metric_str);
    }
    if (ascii && vis_batch) {
        print_ascii_strip(vis_batch, 4, signed_range, sm->name);
    }
    if (out_png && out_png[0] != '\0' && vis_batch) {
        save_png_grid(vis_batch, 4, signed_range, out_png, scratch);
    }

    rcre_tensor_tape_destroy(tape);
    return 0;
}

static int cmd_infer(StudioModelKind kind, const char* weights_in, size_t ddim_steps,
                     uint64_t seed, bool ascii, const char* out_png) {
    if (!weights_in) {
        fprintf(stderr, "[infer] --weights is required.\n");
        return 1;
    }
    RCRE_Arena* perma = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(32 * 1024 * 1024);
    uint64_t rng = seed;

    StudioModel sm;
    if (!studio_model_init(&sm, kind, perma, &rng)) return 1;

    char fmt[32] = {0};
    if (rcre_gen_load_any(sm.params, sm.num_params, weights_in, scratch, fmt, sizeof(fmt)) != RCRE_OK) {
        fprintf(stderr, "[infer:%s] Failed to load %s\n", sm.name, weights_in);
        rcre_arena_destroy(scratch); rcre_arena_destroy(perma);
        return 1;
    }

    printf("[infer:%s] loaded=%s (%s, %zu params)\n", sm.name, weights_in, fmt, sm.total_scalars);
    int rc = run_model_eval(&sm, perma, scratch, ddim_steps, seed, ascii, out_png, nullptr, 0);

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(perma);
    return rc;
}

static int cmd_serve(StudioModelKind kind, const char* weights_in, int port, int max_requests, size_t ddim_steps) {
#if !defined(RCRE_STUDIO_POSIX)
    (void)kind; (void)weights_in; (void)port; (void)max_requests; (void)ddim_steps;
    fprintf(stderr, "[serve] HTTP server requires POSIX sockets.\n");
    return 1;
#else
    if (!weights_in) {
        fprintf(stderr, "[serve] --weights is required.\n");
        return 1;
    }
    RCRE_Arena* perma = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(32 * 1024 * 1024);
    uint64_t rng = 42ULL;

    StudioModel sm;
    if (!studio_model_init(&sm, kind, perma, &rng)) return 1;

    char fmt[32] = {0};
    if (rcre_gen_load_any(sm.params, sm.num_params, weights_in, scratch, fmt, sizeof(fmt)) != RCRE_OK) {
        fprintf(stderr, "[serve:%s] Failed to load %s\n", sm.name, weights_in);
        rcre_arena_destroy(scratch); rcre_arena_destroy(perma);
        return 1;
    }

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) {
        perror("socket");
        rcre_arena_destroy(scratch); rcre_arena_destroy(perma);
        return 1;
    }
    int yes = 1;
    (void)setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons((uint16_t)port);

    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0 || listen(srv, 16) < 0) {
        perror("bind/listen");
        close(srv);
        rcre_arena_destroy(scratch); rcre_arena_destroy(perma);
        return 1;
    }

    printf("[serve:%s] Listening on http://127.0.0.1:%d (weights=%s, format=%s, params=%zu)\n",
           sm.name, port, weights_in, fmt, sm.total_scalars);
    printf("  Endpoints: GET /health | GET /v1/model | POST /v1/infer\n");
    fflush(stdout);

    int handled = 0;
    while (max_requests <= 0 || handled < max_requests) {
        int cli = accept(srv, nullptr, nullptr);
        if (cli < 0) break;

        char req[2048];
        ssize_t n = recv(cli, req, sizeof(req) - 1, 0);
        if (n <= 0) { close(cli); continue; }
        req[n] = '\0';

        char body[1024];
        if (strncmp(req, "GET /health", 11) == 0) {
            snprintf(body, sizeof(body), "{\"status\":\"ok\",\"model\":\"%s\",\"format\":\"%s\"}\n", sm.name, fmt);
        } else if (strncmp(req, "GET /v1/model", 13) == 0) {
            snprintf(body, sizeof(body), "{\"model\":\"%s\",\"weights\":\"%s\",\"format\":\"%s\",\"params\":%zu}\n",
                     sm.name, weights_in, fmt, sm.total_scalars);
        } else {
            RCRE_Arena* req_out = rcre_arena_create(4 * 1024 * 1024);
            rcre_arena_reset(scratch);
            (void)run_model_eval(&sm, req_out, scratch, ddim_steps, 777ULL + (uint64_t)handled,
                                 false, nullptr, body, sizeof(body));
            rcre_arena_destroy(req_out);
        }

        char resp[2048];
        int rlen = snprintf(resp, sizeof(resp),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n\r\n%s",
            strlen(body), body);
        (void)send(cli, resp, (size_t)rlen, 0);
        close(cli);
        handled++;
    }

    close(srv);
    rcre_arena_destroy(scratch);
    rcre_arena_destroy(perma);
    return 0;
#endif
}

static void run_full_demo_one(StudioModelKind kind, int steps, const char* out_dir) {
    char ckpt[256], rgc[256], q8[256], q4[256], png[256];
    const char* name = (kind == MODEL_AE) ? "ae" :
                       (kind == MODEL_VAE) ? "vae" :
                       (kind == MODEL_DIFFUSION) ? "diffusion" :
                       (kind == MODEL_VIT) ? "vit" : "dit";
    snprintf(ckpt, sizeof(ckpt), "%s/%s_f32.safetensors", out_dir, name);
    snprintf(rgc,  sizeof(rgc),  "%s/%s_pruned50.rgc1",   out_dir, name);
    snprintf(q8,   sizeof(q8),   "%s/%s_q8_0.rgq1",       out_dir, name);
    snprintf(q4,   sizeof(q4),   "%s/%s_q4_0.rgq1",       out_dir, name);
    snprintf(png,  sizeof(png),  "%s/%s_q8_infer.png",    out_dir, name);

    printf("\n----------------------------------------------------------------------\n");
    printf("  Model Lifecycle: %s (Train -> Compress -> Quantize -> Infer)\n", name);
    printf("----------------------------------------------------------------------\n");
    (void)cmd_train(kind, steps, 0.0f, 1001ULL + (uint64_t)kind * 1000ULL, ckpt);
    (void)cmd_compress(kind, ckpt, 0.50f, rgc);
    (void)cmd_quantize(kind, ckpt, RCRE_GEN_QUANT_Q8_0, q8);
    (void)cmd_quantize(kind, ckpt, RCRE_GEN_QUANT_Q4_0, q4);
    (void)cmd_infer(kind, ckpt, 10, 777ULL, false, nullptr);
    (void)cmd_infer(kind, rgc,  10, 777ULL, false, nullptr);
    (void)cmd_infer(kind, q8,   10, 777ULL, true,  png);
    (void)cmd_infer(kind, q4,   10, 777ULL, false, nullptr);
}

static void print_usage(const char* prog) {
    printf("RCRE Generative & Vision Studio — Full ML Lifecycle CLI (ISO C23)\n"
           "Usage:\n"
           "  %s train    --model <ae|vae|diffusion|vit|dit> [--steps 50] [--lr F] --out <file.safetensors>\n"
           "  %s compress --model <ae|vae|diffusion|vit|dit> --weights <in> [--sparsity 0.5] --out <file.rgc1>\n"
           "  %s quantize --model <ae|vae|diffusion|vit|dit> --weights <in> [--qtype f16|bf16|q8_0|q4_0] --out <file.rgq1>\n"
           "  %s infer    --model <ae|vae|diffusion|vit|dit> --weights <in> [--ddim-steps 10] [--out-png <out.png>] [--ascii]\n"
           "  %s serve    --model <ae|vae|diffusion|vit|dit> --weights <in> [--port 8090] [--max-requests 0]\n"
           "  %s demo     [--model all|ae|vae|diffusion|vit|dit] [--steps 50] [--out-dir build/outputs]\n",
           prog, prog, prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    rcre_hal_init();

    const char* cmd = "demo";
    int arg_start = 1;
    if (argc >= 2 && argv[1][0] != '-') {
        cmd = argv[1];
        arg_start = 2;
    }

    StudioModelKind kind = MODEL_ALL;
    RCRE_GenQuantType qtype = RCRE_GEN_QUANT_Q8_0;
    const char* weights  = nullptr;
    const char* out_path = nullptr;
    const char* out_png  = nullptr;
    const char* out_dir  = "build/outputs";
    int steps = 50;
    int port = 8090;
    int max_requests = 0;
    size_t ddim_steps = 10;
    float lr = 0.0f;
    float sparsity = 0.50f;
    uint64_t seed = 1001ULL;
    bool ascii = false;

    for (int i = arg_start; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            if (!studio_parse_model(argv[++i], &kind)) {
                fprintf(stderr, "Unknown model: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--out-png") == 0 && i + 1 < argc) {
            out_png = argv[++i];
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
            if (steps < 1) steps = 1;
        } else if (strcmp(argv[i], "--ddim-steps") == 0 && i + 1 < argc) {
            int ds = atoi(argv[++i]);
            ddim_steps = (ds > 0) ? (size_t)ds : 10U;
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            lr = strtof(argv[++i], nullptr);
        } else if (strcmp(argv[i], "--sparsity") == 0 && i + 1 < argc) {
            sparsity = strtof(argv[++i], nullptr);
        } else if (strcmp(argv[i], "--qtype") == 0 && i + 1 < argc) {
            if (!studio_parse_qtype(argv[++i], &qtype)) {
                fprintf(stderr, "Unknown qtype: %s (expected f16|bf16|q8_0|q4_0)\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-requests") == 0 && i + 1 < argc) {
            max_requests = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ascii") == 0) {
            ascii = true;
        }
    }

    if (strcmp(cmd, "train") == 0) {
        if (kind == MODEL_ALL) { fprintf(stderr, "[train] Specify a single --model <ae|vae|diffusion|vit|dit>\n"); return 1; }
        return cmd_train(kind, steps, lr, seed, out_path);
    }
    if (strcmp(cmd, "compress") == 0) {
        if (kind == MODEL_ALL) { fprintf(stderr, "[compress] Specify a single --model <ae|vae|diffusion|vit|dit>\n"); return 1; }
        return cmd_compress(kind, weights, sparsity, out_path);
    }
    if (strcmp(cmd, "quantize") == 0) {
        if (kind == MODEL_ALL) { fprintf(stderr, "[quantize] Specify a single --model <ae|vae|diffusion|vit|dit>\n"); return 1; }
        return cmd_quantize(kind, weights, qtype, out_path);
    }
    if (strcmp(cmd, "infer") == 0) {
        if (kind == MODEL_ALL) { fprintf(stderr, "[infer] Specify a single --model <ae|vae|diffusion|vit|dit>\n"); return 1; }
        return cmd_infer(kind, weights, ddim_steps, seed, ascii, out_png);
    }
    if (strcmp(cmd, "serve") == 0) {
        if (kind == MODEL_ALL) { fprintf(stderr, "[serve] Specify a single --model <ae|vae|diffusion|vit|dit>\n"); return 1; }
        return cmd_serve(kind, weights, port, max_requests, ddim_steps);
    }

    /* Default: demo mode */
    mkdir(out_dir, 0755);
    printf("======================================================================\n");
    printf("  RCRE Generative & Vision Studio — End-to-End Lifecycle Demo\n");
    printf("======================================================================\n");
    if (kind == MODEL_ALL) {
        run_full_demo_one(MODEL_AE,        steps, out_dir);
        run_full_demo_one(MODEL_VAE,       steps, out_dir);
        run_full_demo_one(MODEL_DIFFUSION, steps, out_dir);
        run_full_demo_one(MODEL_VIT,       steps, out_dir);
        run_full_demo_one(MODEL_DIT,       steps, out_dir);
    } else {
        run_full_demo_one(kind, steps, out_dir);
    }
    printf("\nDone.\n");
    return 0;
}
