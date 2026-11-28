/**
 * @file rcre_gen.h
 * @brief RCRE Generative & Vision Deep Learning Suite (ISO C23)
 *
 * Implements four unified, zero-dependency generative & vision architectures
 * built directly on RCRE's unified tensor tape autograd (rcre_autograd.h) and
 * 64-byte aligned linear memory arenas:
 *   1. Convolutional & Denoising Autoencoder (RCRE_ConvAE)
 *   2. Convolutional Variational Autoencoder (RCRE_ConvVAE / beta-VAE)
 *   3. DDPM / DDIM Diffusion Scheduler & Time-Conditioned Tiny U-Net (RCRE_TinyUNet)
 *   4. Vision Transformer (RCRE_ViT) & Diffusion Transformer w/ AdaLN-Zero (RCRE_DiT)
 *
 * Strict ISO C23 compliant, zero external dependencies, zero heap allocations
 * inside forward/backward/sampling hot loops.
 */

#ifndef RCRE_GEN_H
#define RCRE_GEN_H

#include "rcre.h"
#include "rcre_autograd.h"

typedef struct RCRE_TensorTape RCRE_TensorTape;
typedef struct RCRE_TensorOptim RCRE_TensorOptim;

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Phase 1: Convolutional & Denoising Autoencoder (RCRE_ConvAE)           */
/* ========================================================================= */

typedef struct {
    size_t in_channels;     /* e.g., 1 (grayscale) or 3 (RGB)               */
    size_t height;          /* Image height (must be divisible by 4)        */
    size_t width;           /* Image width  (must be divisible by 4)        */
    size_t base_channels;   /* First conv stage channels (stage 2 uses 2x)  */
    size_t latent_dim;      /* Bottleneck dimensionality                    */
    bool   output_sigmoid;  /* true: [0,1] sigmoid; false: [-1,1] tanh      */
} RCRE_ConvAEConfig;

typedef struct {
    RCRE_ConvAEConfig cfg;
    size_t feat_h;          /* height / 4                                   */
    size_t feat_w;          /* width  / 4                                   */
    size_t flat_dim;        /* (2 * base_channels) * feat_h * feat_w        */

    /* Encoder weights */
    RCRE_Tensor* enc_conv1_w; /* [C, in_ch, 3, 3] */
    RCRE_Tensor* enc_conv1_b; /* [C]              */
    RCRE_Tensor* enc_conv2_w; /* [2C, C, 3, 3]    */
    RCRE_Tensor* enc_conv2_b; /* [2C]             */
    RCRE_Tensor* enc_fc_w;    /* [flat, latent]   */
    RCRE_Tensor* enc_fc_b;    /* [latent]         */

    /* Decoder weights */
    RCRE_Tensor* dec_fc_w;    /* [latent, flat]   */
    RCRE_Tensor* dec_fc_b;    /* [flat]           */
    RCRE_Tensor* dec_conv1_w; /* [C, 2C, 3, 3]    */
    RCRE_Tensor* dec_conv1_b; /* [C]              */
    RCRE_Tensor* dec_conv2_w; /* [in_ch, C, 3, 3] */
    RCRE_Tensor* dec_conv2_b; /* [in_ch]          */
} RCRE_ConvAE;

#define RCRE_CONVAE_NUM_PARAMS 12

RCRE_NODISCARD RCRE_ConvAE* rcre_convae_create(RCRE_Arena* arena, RCRE_ConvAEConfig cfg, uint64_t* rng);
RCRE_NODISCARD size_t rcre_convae_params(const RCRE_ConvAE* m, RCRE_Tensor** out_params, size_t max_params);
RCRE_NODISCARD RCRE_Tensor* rcre_convae_encode(const RCRE_ConvAE* m, RCRE_TensorTape* tape, RCRE_Tensor* x);
RCRE_NODISCARD RCRE_Tensor* rcre_convae_decode(const RCRE_ConvAE* m, RCRE_TensorTape* tape, RCRE_Tensor* z);
RCRE_NODISCARD RCRE_Tensor* rcre_convae_forward(const RCRE_ConvAE* m, RCRE_TensorTape* tape, RCRE_Tensor* x);

/**
 * @brief Executes one training step (standard AE if noise_std <= 0,
 *        Denoising Autoencoder if noise_std > 0). Returns MSE reconstruction loss.
 */
RCRE_NODISCARD float rcre_convae_train_step(
    RCRE_ConvAE* m,
    RCRE_TensorTape* tape,
    RCRE_TensorOptim* opt,
    RCRE_Tensor* x_clean,
    float noise_std,
    uint64_t* rng,
    float max_grad_norm
);

RCRE_NODISCARD float rcre_gen_psnr(const RCRE_Tensor* ref, const RCRE_Tensor* recon, float max_val);

/* ========================================================================= */
/* 2. Phase 2: Convolutional Variational Autoencoder (RCRE_ConvVAE)          */
/* ========================================================================= */

typedef struct {
    size_t in_channels;
    size_t height;          /* Divisible by 4 */
    size_t width;           /* Divisible by 4 */
    size_t base_channels;
    size_t latent_dim;
    bool   output_sigmoid;  /* true: [0,1] sigmoid; false: [-1,1] tanh */
} RCRE_ConvVAEConfig;

typedef struct {
    RCRE_ConvVAEConfig cfg;
    size_t feat_h;
    size_t feat_w;
    size_t flat_dim;

    /* Encoder */
    RCRE_Tensor* enc_conv1_w;
    RCRE_Tensor* enc_conv1_b;
    RCRE_Tensor* enc_conv2_w;
    RCRE_Tensor* enc_conv2_b;
    RCRE_Tensor* fc_mu_w;     /* [flat, latent] */
    RCRE_Tensor* fc_mu_b;     /* [latent]       */
    RCRE_Tensor* fc_lv_w;     /* [flat, latent] */
    RCRE_Tensor* fc_lv_b;     /* [latent]       */

    /* Decoder */
    RCRE_Tensor* dec_fc_w;    /* [latent, flat] */
    RCRE_Tensor* dec_fc_b;    /* [flat]         */
    RCRE_Tensor* dec_conv1_w;
    RCRE_Tensor* dec_conv1_b;
    RCRE_Tensor* dec_conv2_w;
    RCRE_Tensor* dec_conv2_b;
} RCRE_ConvVAE;

#define RCRE_CONVVAE_NUM_PARAMS 14

RCRE_NODISCARD RCRE_ConvVAE* rcre_convvae_create(RCRE_Arena* arena, RCRE_ConvVAEConfig cfg, uint64_t* rng);
RCRE_NODISCARD size_t rcre_convvae_params(const RCRE_ConvVAE* m, RCRE_Tensor** out_params, size_t max_params);
RCRE_NODISCARD int rcre_convvae_encode(const RCRE_ConvVAE* m, RCRE_TensorTape* tape, RCRE_Tensor* x,
                                       RCRE_Tensor** out_mu, RCRE_Tensor** out_logvar);
RCRE_NODISCARD RCRE_Tensor* rcre_convvae_decode(const RCRE_ConvVAE* m, RCRE_TensorTape* tape, RCRE_Tensor* z);
RCRE_NODISCARD RCRE_Tensor* rcre_convvae_forward(const RCRE_ConvVAE* m, RCRE_TensorTape* tape, RCRE_Tensor* x,
                                                 uint64_t* rng, RCRE_Tensor** out_mu,
                                                 RCRE_Tensor** out_logvar, RCRE_Tensor** out_z);

/**
 * @brief Executes one beta-VAE ELBO step: loss = MSE(x_hat, x) + beta * KL(q||p).
 */
RCRE_NODISCARD float rcre_convvae_train_step(
    RCRE_ConvVAE* m,
    RCRE_TensorTape* tape,
    RCRE_TensorOptim* opt,
    RCRE_Tensor* x,
    float beta,
    uint64_t* rng,
    float max_grad_norm,
    float* out_recon_loss,
    float* out_kl_loss
);

/** Samples `num_samples` images from standard normal prior z ~ N(0, I). */
RCRE_NODISCARD RCRE_Tensor* rcre_convvae_sample(
    const RCRE_ConvVAE* m, RCRE_Arena* scratch, size_t num_samples, uint64_t* rng);

/** Linearly interpolates between latent vectors z_start and z_end ([latent_dim]). */
RCRE_NODISCARD RCRE_Tensor* rcre_convvae_interpolate(
    const RCRE_ConvVAE* m, RCRE_Arena* scratch,
    const float* z_start, const float* z_end, size_t num_steps);

/* ========================================================================= */
/* 3. Phase 3: Diffusion Schedule & Time-Conditioned Tiny U-Net              */
/* ========================================================================= */

typedef enum : uint8_t {
    RCRE_DIFF_SCHED_LINEAR = 0,
    RCRE_DIFF_SCHED_COSINE = 1
} RCRE_DiffSchedType;

typedef struct {
    size_t num_steps;                /* T (e.g., 50, 100, 1000) */
    RCRE_DiffSchedType type;
    float* betas;
    float* alphas;
    float* alpha_bar;
    float* sqrt_alpha_bar;
    float* sqrt_one_minus_alpha_bar;
    float* posterior_variance;
} RCRE_DiffSchedule;

RCRE_NODISCARD RCRE_DiffSchedule* rcre_diff_schedule_create(
    RCRE_Arena* arena, size_t num_steps, RCRE_DiffSchedType type,
    float beta_start, float beta_end);

/** Forward diffusion q(x_t | x_0): x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * noise */
RCRE_NODISCARD int rcre_diff_q_sample(
    const RCRE_DiffSchedule* sched,
    const RCRE_Tensor* x0,
    const size_t* t_indices,
    const RCRE_Tensor* noise,
    RCRE_Tensor* x_t_out
);

typedef struct {
    size_t in_channels;
    size_t height;          /* Divisible by 4 */
    size_t width;           /* Divisible by 4 */
    size_t base_channels;   /* Divisible by num_groups (e.g., 8, 16, 32) */
    size_t num_groups;      /* GroupNorm groups (e.g., 4) */
    size_t time_dim;        /* Even sinusoidal time embedding dim (e.g., 32) */
} RCRE_TinyUNetConfig;

typedef struct {
    size_t in_ch;
    size_t out_ch;
    size_t num_groups;
    RCRE_Tensor* gn1_g;     /* [in_ch]                  */
    RCRE_Tensor* gn1_b;     /* [in_ch]                  */
    RCRE_Tensor* conv1_w;   /* [out_ch, in_ch, 3, 3]    */
    RCRE_Tensor* conv1_b;   /* [out_ch]                 */
    RCRE_Tensor* time_w;    /* [time_dim, 2 * out_ch]   */
    RCRE_Tensor* time_b;    /* [2 * out_ch]             */
    RCRE_Tensor* gn2_g;     /* [out_ch]                 */
    RCRE_Tensor* gn2_b;     /* [out_ch]                 */
    RCRE_Tensor* conv2_w;   /* [out_ch, out_ch, 3, 3]   */
    RCRE_Tensor* conv2_b;   /* [out_ch]                 */
    RCRE_Tensor* skip_w;    /* [out_ch, in_ch, 1, 1] or NULL if in_ch == out_ch */
    RCRE_Tensor* skip_b;    /* [out_ch]              or NULL if in_ch == out_ch */
} RCRE_UNetResBlock;

typedef struct {
    RCRE_TinyUNetConfig cfg;

    /* Time embedding MLP: [time_dim -> time_dim -> time_dim] */
    RCRE_Tensor* time_fc1_w;
    RCRE_Tensor* time_fc1_b;
    RCRE_Tensor* time_fc2_w;
    RCRE_Tensor* time_fc2_b;

    /* Stem conv: [in_ch -> C] */
    RCRE_Tensor* in_conv_w;
    RCRE_Tensor* in_conv_b;

    /* U-Net blocks */
    RCRE_UNetResBlock enc1; /* C  -> C  (H, W)     */
    RCRE_UNetResBlock enc2; /* C  -> 2C (H/2, W/2) */
    RCRE_UNetResBlock mid;  /* 2C -> 2C (H/4, W/4) */
    RCRE_UNetResBlock dec2; /* 4C -> C  (H/2, W/2) */
    RCRE_UNetResBlock dec1; /* 2C -> C  (H, W)     */

    /* Output head: GroupNorm -> SiLU -> Conv(C -> in_ch) */
    RCRE_Tensor* out_gn_g;
    RCRE_Tensor* out_gn_b;
    RCRE_Tensor* out_conv_w;
    RCRE_Tensor* out_conv_b;
} RCRE_TinyUNet;

#define RCRE_TINYUNET_MAX_PARAMS 80

RCRE_NODISCARD RCRE_TinyUNet* rcre_tinyunet_create(RCRE_Arena* arena, RCRE_TinyUNetConfig cfg, uint64_t* rng);
RCRE_NODISCARD size_t rcre_tinyunet_params(const RCRE_TinyUNet* m, RCRE_Tensor** out_params, size_t max_params);
RCRE_NODISCARD RCRE_Tensor* rcre_tinyunet_forward(
    const RCRE_TinyUNet* m, RCRE_TensorTape* tape, RCRE_Tensor* x_t, const float* timesteps);

/** Executes one DDPM epsilon-prediction training step; returns MSE(eps_pred, eps). */
RCRE_NODISCARD float rcre_tinyunet_train_step(
    RCRE_TinyUNet* m,
    const RCRE_DiffSchedule* sched,
    RCRE_TensorTape* tape,
    RCRE_TensorOptim* opt,
    const RCRE_Tensor* x0,
    uint64_t* rng,
    float max_grad_norm
);

/** Full ancestral DDPM reverse diffusion sampling [num_samples, C, H, W]. */
RCRE_NODISCARD RCRE_Tensor* rcre_tinyunet_sample_ddpm(
    const RCRE_TinyUNet* m, const RCRE_DiffSchedule* sched,
    RCRE_Arena* out_arena, RCRE_Arena* scratch,
    size_t num_samples, uint64_t* rng);

/** Accelerated DDIM reverse sampling (eta = 0.0f is deterministic ODE). */
RCRE_NODISCARD RCRE_Tensor* rcre_tinyunet_sample_ddim(
    const RCRE_TinyUNet* m, const RCRE_DiffSchedule* sched,
    RCRE_Arena* out_arena, RCRE_Arena* scratch,
    size_t num_samples, size_t ddim_steps, float eta, uint64_t* rng);

/* ========================================================================= */
/* 4. Phase 4: Vision Transformer (RCRE_ViT) & Diffusion Transformer (DiT)   */
/* ========================================================================= */

typedef struct {
    size_t in_channels;
    size_t height;          /* Divisible by patch_size */
    size_t width;           /* Divisible by patch_size */
    size_t patch_size;
    size_t embed_dim;       /* Divisible by num_heads  */
    size_t num_heads;
    size_t num_layers;
    size_t mlp_dim;
    size_t num_classes;
} RCRE_ViTConfig;

typedef struct {
    RCRE_Tensor* ln1_g;
    RCRE_Tensor* ln1_b;
    RCRE_Tensor* wq; RCRE_Tensor* bq;
    RCRE_Tensor* wk; RCRE_Tensor* bk;
    RCRE_Tensor* wv; RCRE_Tensor* bv;
    RCRE_Tensor* wo; RCRE_Tensor* bo;
    RCRE_Tensor* ln2_g;
    RCRE_Tensor* ln2_b;
    RCRE_Tensor* fc1_w; RCRE_Tensor* fc1_b;
    RCRE_Tensor* fc2_w; RCRE_Tensor* fc2_b;
} RCRE_ViTBlock;

typedef struct {
    RCRE_ViTConfig cfg;
    size_t num_patches;
    size_t patch_dim;

    RCRE_Tensor* patch_w;   /* [patch_dim, embed_dim]  */
    RCRE_Tensor* patch_b;   /* [embed_dim]             */
    RCRE_Tensor* pos_emb;   /* [num_patches, embed_dim]*/
    RCRE_ViTBlock* blocks;  /* [num_layers]            */
    RCRE_Tensor* ln_f_g;    /* [embed_dim]             */
    RCRE_Tensor* ln_f_b;    /* [embed_dim]             */
    RCRE_Tensor* head_w;    /* [embed_dim, num_classes]*/
    RCRE_Tensor* head_b;    /* [num_classes]           */
} RCRE_ViT;

RCRE_NODISCARD RCRE_ViT* rcre_vit_create(RCRE_Arena* arena, RCRE_ViTConfig cfg, uint64_t* rng);
RCRE_NODISCARD size_t rcre_vit_param_count(const RCRE_ViT* m);
RCRE_NODISCARD size_t rcre_vit_params(const RCRE_ViT* m, RCRE_Tensor** out_params, size_t max_params);
RCRE_NODISCARD RCRE_Tensor* rcre_vit_forward(const RCRE_ViT* m, RCRE_TensorTape* tape, RCRE_Tensor* img);
RCRE_NODISCARD float rcre_vit_train_step(
    RCRE_ViT* m, RCRE_TensorTape* tape, RCRE_TensorOptim* opt,
    RCRE_Tensor* img, const size_t* labels, float max_grad_norm);
RCRE_NODISCARD int rcre_vit_predict(const RCRE_ViT* m, RCRE_Arena* scratch,
                                    RCRE_Tensor* img, size_t* out_preds);

/* --- Diffusion Transformer (DiT with AdaLN-Zero) --- */

typedef struct {
    size_t in_channels;
    size_t height;          /* Divisible by patch_size */
    size_t width;           /* Divisible by patch_size */
    size_t patch_size;
    size_t embed_dim;       /* Even & divisible by num_heads */
    size_t num_heads;
    size_t num_layers;
    size_t mlp_dim;
    size_t num_classes;     /* 0 = unconditional; >0 = class-conditioned */
} RCRE_DiTConfig;

typedef struct {
    RCRE_Tensor* ada_w;     /* [embed_dim, 6 * embed_dim] (zero-init) */
    RCRE_Tensor* ada_b;     /* [6 * embed_dim]            (zero-init) */
    RCRE_Tensor* wq; RCRE_Tensor* bq;
    RCRE_Tensor* wk; RCRE_Tensor* bk;
    RCRE_Tensor* wv; RCRE_Tensor* bv;
    RCRE_Tensor* wo; RCRE_Tensor* bo;
    RCRE_Tensor* fc1_w; RCRE_Tensor* fc1_b;
    RCRE_Tensor* fc2_w; RCRE_Tensor* fc2_b;
} RCRE_DiTBlock;

typedef struct {
    RCRE_DiTConfig cfg;
    size_t num_patches;
    size_t patch_dim;

    RCRE_Tensor* patch_w;       /* [patch_dim, embed_dim]     */
    RCRE_Tensor* patch_b;       /* [embed_dim]                */
    RCRE_Tensor* pos_emb;       /* [num_patches, embed_dim]   */

    RCRE_Tensor* time_fc1_w;    /* [embed_dim, embed_dim]     */
    RCRE_Tensor* time_fc1_b;    /* [embed_dim]                */
    RCRE_Tensor* time_fc2_w;    /* [embed_dim, embed_dim]     */
    RCRE_Tensor* time_fc2_b;    /* [embed_dim]                */
    RCRE_Tensor* class_emb;     /* [num_classes, embed_dim] or NULL */

    RCRE_DiTBlock* blocks;      /* [num_layers]               */

    RCRE_Tensor* final_ada_w;   /* [embed_dim, 2 * embed_dim] (zero-init) */
    RCRE_Tensor* final_ada_b;   /* [2 * embed_dim]            (zero-init) */
    RCRE_Tensor* final_w;       /* [embed_dim, patch_dim]     (zero-init) */
    RCRE_Tensor* final_b;       /* [patch_dim]                (zero-init) */
} RCRE_DiT;

RCRE_NODISCARD RCRE_DiT* rcre_dit_create(RCRE_Arena* arena, RCRE_DiTConfig cfg, uint64_t* rng);
RCRE_NODISCARD size_t rcre_dit_param_count(const RCRE_DiT* m);
RCRE_NODISCARD size_t rcre_dit_params(const RCRE_DiT* m, RCRE_Tensor** out_params, size_t max_params);

/** Forward pass: predicts noise epsilon [N, C, H, W] from noisy x_t, timesteps, optional class labels. */
RCRE_NODISCARD RCRE_Tensor* rcre_dit_forward(
    const RCRE_DiT* m, RCRE_TensorTape* tape, RCRE_Tensor* x_t,
    const float* timesteps, const size_t* class_labels);

RCRE_NODISCARD float rcre_dit_train_step(
    RCRE_DiT* m,
    const RCRE_DiffSchedule* sched,
    RCRE_TensorTape* tape,
    RCRE_TensorOptim* opt,
    const RCRE_Tensor* x0,
    const size_t* class_labels,
    uint64_t* rng,
    float max_grad_norm
);

RCRE_NODISCARD RCRE_Tensor* rcre_dit_sample_ddim(
    const RCRE_DiT* m, const RCRE_DiffSchedule* sched,
    RCRE_Arena* out_arena, RCRE_Arena* scratch,
    size_t num_samples, const size_t* class_labels,
    size_t ddim_steps, float eta, uint64_t* rng);

/* ========================================================================= */
/* 5. Shared Generative Utilities: Synthetic Sprites & SafeTensors I/O       */
/* ========================================================================= */

/**
 * @brief Generates a deterministic multi-class small-image dataset [N, 1, H, W]
 *        (4 geometric shape classes: 0=disk, 1=square, 2=cross, 3=ring)
 *        normalized to [0, 1] (if zero_to_one) or [-1, 1].
 */
RCRE_NODISCARD int rcre_gen_make_shapes_dataset(
    RCRE_Arena* arena,
    size_t num_samples,
    size_t height,
    size_t width,
    bool zero_to_one,
    uint64_t* rng,
    RCRE_Tensor** out_images,
    size_t** out_labels
);

/** Saves an array of F32 parameter tensors into a .safetensors archive. */
RCRE_NODISCARD int rcre_gen_save_safetensors(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath,
    RCRE_Arena* scratch
);

/** Loads an array of F32 parameter tensors from a .safetensors archive. */
RCRE_NODISCARD int rcre_gen_load_safetensors(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath
);

/* ========================================================================= */
/* 6. Model Lifecycle: Pruning, Lossless RCRZ Compression & Quantization     */
/* ========================================================================= */

typedef enum : uint8_t {
    RCRE_GEN_QUANT_F16  = 1, /* IEEE-754 FP16 (2.0x smaller)                 */
    RCRE_GEN_QUANT_BF16 = 2, /* Google BF16   (2.0x smaller, F32 range)      */
    RCRE_GEN_QUANT_Q8_0 = 3, /* Block-32 INT8 + FP16 scale (~3.76x smaller)  */
    RCRE_GEN_QUANT_Q4_0 = 4  /* Block-32 INT4 + FP16 scale (~7.11x smaller)  */
} RCRE_GenQuantType;

typedef struct {
    size_t total_params;      /* Total scalar weights across all tensors      */
    size_t zero_params;       /* Exact zero weights (after pruning/loading)   */
    float  sparsity;          /* zero_params / total_params in [0, 1]         */
    size_t raw_bytes;         /* Uncompressed F32 bytes (total_params * 4)    */
    size_t artifact_bytes;    /* Serialized file size on disk                 */
    float  compression_ratio; /* raw_bytes / artifact_bytes                   */
    float  rmse;              /* Root-mean-square error vs original F32       */
    float  max_abs_err;       /* Maximum absolute elementwise error           */
    float  sqnr_db;           /* Signal-to-Quantization-Noise Ratio (dB)      */
} RCRE_GenModelStats;

/**
 * @brief Applies unstructured magnitude pruning in-place across model parameters.
 *        When `skip_1d_tensors` is true, 1D bias/norm tensors are preserved.
 */
RCRE_NODISCARD int rcre_gen_prune_magnitude(
    RCRE_Tensor* const* params,
    size_t num_params,
    float target_sparsity,
    bool skip_1d_tensors,
    RCRE_Arena* scratch,
    RCRE_GenModelStats* out_stats
);

/**
 * @brief Saves model parameters as a lossless, 0-ULP bit-exact SIMD byte-shuffled
 *        RCRZ compressed archive ("RGC1", .rgc1). Ideal after magnitude pruning.
 */
RCRE_NODISCARD int rcre_gen_save_compressed(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath,
    RCRE_Arena* scratch,
    RCRE_GenModelStats* out_stats
);

/** Loads a lossless "RGC1" compressed parameter archive back into params[]. */
RCRE_NODISCARD int rcre_gen_load_compressed(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath,
    RCRE_Arena* scratch
);

/**
 * @brief Quantizes model parameters to F16, BF16, Q8_0, or Q4_0 and saves
 *        a compact "RGQ1" (.rgq1) quantized artifact with exact SQNR/RMSE stats.
 */
RCRE_NODISCARD int rcre_gen_save_quantized(
    RCRE_Tensor* const* params,
    size_t num_params,
    RCRE_GenQuantType qtype,
    const char* filepath,
    RCRE_Arena* scratch,
    RCRE_GenModelStats* out_stats
);

/** Loads and dequantizes an "RGQ1" quantized checkpoint back into params[]. */
RCRE_NODISCARD int rcre_gen_load_quantized(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath,
    RCRE_GenQuantType* out_qtype
);

/**
 * @brief Auto-detects file format ("RGC1", "RGQ1", or ".safetensors") and loads
 *        weights into params[]. Optionally writes human-readable format tag.
 */
RCRE_NODISCARD int rcre_gen_load_any(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath,
    RCRE_Arena* scratch,
    char* out_format,
    size_t max_fmt
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_GEN_H */
