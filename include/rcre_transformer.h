/**
 * @file rcre_transformer.h
 * @brief RCRE Generative AI & Autoregressive Transformer Architecture (ISO C23)
 *
 * Implements a complete Decoder-Only Transformer (Micro-GPT) engine in pure C23:
 *  - Causal Multi-Head Self-Attention with causal masking
 *  - Pre-LayerNorm architecture and GELU non-linearities
 *  - Inference Key-Value (KV) Cache for O(1) token generation
 *  - Subword Byte-Pair Encoding (BPE) & Character Tokenizers
 *  - Analytical backward passes and Decoupled AdamW optimization
 *  - Direct acceleration via RCRE dynamic Hardware Abstraction Layer (HAL)
 */

#ifndef RCRE_TRANSFORMER_H
#define RCRE_TRANSFORMER_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Architectural Hyperparameters & Configuration                             */
/* ========================================================================= */

#define RCRE_GPT_DEFAULT_BATCH_SIZE   4     /* Sequences per training step   */
#define RCRE_GPT_DEFAULT_SEQ_LEN      64    /* Maximum context length (T)    */
#define RCRE_GPT_DEFAULT_EMBED_DIM    64    /* Model dimension (d_model, C)  */
#define RCRE_GPT_DEFAULT_NUM_HEADS    4     /* Number of attention heads (H) */
#define RCRE_GPT_DEFAULT_NUM_LAYERS   2     /* Transformer blocks depth (L)  */
#define RCRE_GPT_DEFAULT_MLP_MULT     4     /* MLP expansion multiplier (4*C)*/
#define RCRE_MAX_CHAR_VOCAB           256   /* Max size for char tokenizer   */

typedef struct {
    int vocab_size;   /* V: Number of distinct tokens in vocabulary          */
    int seq_len;      /* T: Maximum context window                           */
    int embed_dim;    /* C: Hidden model dimension (d_model)                 */
    int num_heads;    /* H: Number of parallel attention heads               */
    int head_dim;     /* d_k = C / H                                         */
    int num_layers;   /* L: Depth (number of Transformer blocks)             */
    int mlp_dim;      /* Feed-forward network inner dimension (typically 4*C)*/
} RCRE_GPTConfig;

/* Transformer Block Parameters */
typedef struct {
    float *ln1_w;  float *ln1_b;   /* LayerNorm 1: (C), (C)                  */
    float *q_w;    float *q_b;     /* Query projection: (C, C), (C)          */
    float *k_w;    float *k_b;     /* Key projection: (C, C), (C)            */
    float *v_w;    float *v_b;     /* Value projection: (C, C), (C)          */
    float *proj_w; float *proj_b;  /* Multi-head output projection: (C, C),(C)*/
    float *ln2_w;  float *ln2_b;   /* LayerNorm 2: (C), (C)                  */
    float *fc1_w;  float *fc1_b;   /* MLP Expand: (C, mlp_dim), (mlp_dim)    */
    float *fc2_w;  float *fc2_b;   /* MLP Contract: (mlp_dim, C), (C)        */
} RCRE_GPTBlockWeights;

/* Forward Activations cached for backward pass */
typedef struct {
    float *ln1_out;
    float *ln1_mean;
    float *ln1_rstd;
    float *q, *k, *v;
    float *att_scores;
    float *att_out;
    float *proj_out;
    float *resid1_out;

    float *ln2_out;
    float *ln2_mean;
    float *ln2_rstd;
    float *fc1_out;
    float *gelu_out;
    float *fc2_out;
    float *resid2_out;
} RCRE_GPTBlockActs;

/* Full GPT Model Parameters */
typedef struct {
    RCRE_GPTConfig config;
    size_t num_params;

    /* Contiguous flat parameter buffers */
    float *params;
    float *grads;
    float *m; /* AdamW 1st moment vector */
    float *v; /* AdamW 2nd moment vector */

    /* Named pointers into params */
    float *wte;                      /* Token Embedding: (V, C)             */
    float *wpe;                      /* Positional Embedding: (T, C)        */
    RCRE_GPTBlockWeights *blocks;    /* Array of L blocks                   */
    float *ln_f_w;                   /* Final LayerNorm weight: (C)         */
    float *ln_f_b;                   /* Final LayerNorm bias: (C)           */
    float *head_w;                   /* LM Head Projection: (C, V)          */

    RCRE_Arena *arena;               /* Non-null if arena-resident          */
    bool owns_heap;                  /* True if heap-allocated              */
} RCRE_GPTModel;

/* Full Model Forward Activations */
typedef struct {
    float *emb_out;
    RCRE_GPTBlockActs *blocks;
    float *ln_f_out;
    float *ln_f_mean;
    float *ln_f_rstd;
    float *logits;
    float *probs;
    RCRE_Arena *arena;
    bool owns_heap;
} RCRE_GPTActs;

/* Inference Key-Value (KV) Cache for O(1) Token Generation */
typedef struct {
    float *k;          /* Key cache: (L, H, max_T, d_k)                      */
    float *v;          /* Value cache: (L, H, max_T, d_k)                    */
    int max_seq_len;
    int current_pos;
    RCRE_Arena *arena;
    bool owns_heap;
} RCRE_KVCache;

/* AdamW Optimizer Configuration for Transformers */
typedef struct {
    float lr_max;
    float lr_min;
    int warmup_steps;
    int max_steps;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float grad_clip;
    int current_step;
} RCRE_GPTAdamWConfig;

/* Generation & Sampling Configuration */
typedef struct {
    float temperature; /* Sampling temperature (> 0.0, default 0.8)          */
    int top_k;         /* Top-K filtering (0 to disable)                     */
    int max_new_tokens;/* Maximum number of tokens to generate               */
} RCRE_GPTGenerateConfig;

/* Character-Level Tokenizer */
typedef struct {
    int vocab_size;
    int char_to_id[256];
    unsigned char id_to_char[256];
} RCRE_CharTokenizer;

/* Subword Byte-Pair Encoding (BPE) Rule & Tokenizer */
typedef struct {
    uint16_t id1;
    uint16_t id2;
    uint16_t new_id;
} RCRE_BPEMergeRule;

typedef struct {
    int vocab_size;
    int num_rules;
    RCRE_BPEMergeRule *rules;
    char **id_to_token_str;
} RCRE_BPETokenizer;

/* Dataset / DataLoader */
typedef struct {
    int *tokens;
    size_t num_tokens;
    int batch_size;
    int seq_len;
    size_t current_idx;
} RCRE_GPTDataLoader;

/* ========================================================================= */
/* Core Neural Network Forward and Backward Kernels                          */
/* ========================================================================= */

/* 1. Linear Projection: Y = X * W + b */
void rcre_transformer_linear_forward(float *out, const float *x, const float *w, const float *b, int N, int Din, int Dout);
void rcre_transformer_linear_backward(float *dx, float *dw, float *db, const float *dout, const float *x, const float *w, int N, int Din, int Dout);

/* 2. GELU Non-linearity */
RCRE_NODISCARD float rcre_transformer_gelu_scalar(float x);
RCRE_NODISCARD float rcre_transformer_gelu_scalar_backward(float x, float dout);
void rcre_transformer_gelu_forward(float *out, const float *x, int N);
void rcre_transformer_gelu_backward(float *dx, const float *x, const float *dout, int N);

/* 3. Layer Normalization (Pre-norm Transformer standard) */
void rcre_transformer_layernorm_forward(float *out, float *mean, float *rstd, const float *x, const float *weight, const float *bias, int N, int C);
void rcre_transformer_layernorm_backward(float *dx, float *dweight, float *dbias, const float *dout, const float *x, const float *mean, const float *rstd, const float *weight, int N, int C);

/* 4. Numerically Stable Softmax */
void rcre_transformer_softmax_forward(float *out, const float *inp, int N, int C);
void rcre_transformer_softmax_backward(float *dinp, const float *dout, const float *out, int N, int C);

/* 5. Causal Multi-Head Self-Attention */
void rcre_transformer_attention_forward(
    float *out, float *att_scores,
    const float *q, const float *k, const float *v,
    int B, int T, int C, int H
);
void rcre_transformer_attention_backward(
    float *dq, float *dk, float *dv,
    float *datt_scores,
    const float *dout, const float *att_scores,
    const float *q, const float *k, const float *v,
    int B, int T, int C, int H
);

/* 6. Cross-Entropy Loss with Softmax over Vocabulary */
float rcre_transformer_cross_entropy_forward(float *probs, const float *logits, const int *targets, int N, int V);
void rcre_transformer_cross_entropy_backward(float *dlogits, const float *probs, const int *targets, int N, int V);

/* ========================================================================= */
/* Model Lifecycle, Forward, Backward & KV-Cache                             */
/* ========================================================================= */

RCRE_NODISCARD RCRE_GPTModel* rcre_gpt_create(RCRE_GPTConfig config, RCRE_Arena *arena);
void rcre_gpt_init_weights(RCRE_GPTModel *model, uint64_t *rng_state);
void rcre_gpt_zero_grads(RCRE_GPTModel *model);
void rcre_gpt_free(RCRE_GPTModel *model);

RCRE_NODISCARD RCRE_GPTActs* rcre_gpt_acts_create(RCRE_GPTConfig config, int B, int T, RCRE_Arena *arena);
void rcre_gpt_acts_free(RCRE_GPTActs *acts, int num_layers);

float rcre_gpt_forward(RCRE_GPTModel *model, RCRE_GPTActs *acts, const int *inputs, const int *targets, int B, int T);
void rcre_gpt_backward(RCRE_GPTModel *model, RCRE_GPTActs *acts, const int *inputs, const int *targets, int B, int T);

/* KV-Cache Functions */
RCRE_NODISCARD RCRE_KVCache* rcre_kv_cache_create(RCRE_GPTConfig config, RCRE_Arena *arena);
void rcre_kv_cache_reset(RCRE_KVCache *cache);
void rcre_kv_cache_free(RCRE_KVCache *cache);
void rcre_gpt_forward_token(RCRE_GPTModel *model, RCRE_KVCache *cache, int token, int pos, float *out_logits);

/* ========================================================================= */
/* Tokenization Subsystems                                                   */
/* ========================================================================= */

/* Character Tokenizer */
void rcre_char_tokenizer_init(RCRE_CharTokenizer *tok);
bool rcre_char_tokenizer_train_from_file(RCRE_CharTokenizer *tok, const char *filepath);
bool rcre_char_tokenizer_save(const RCRE_CharTokenizer *tok, const char *filepath);
bool rcre_char_tokenizer_load(RCRE_CharTokenizer *tok, const char *filepath);
int rcre_char_tokenizer_encode(const RCRE_CharTokenizer *tok, const char *text, size_t len, int *out_tokens);
size_t rcre_char_tokenizer_decode(const RCRE_CharTokenizer *tok, const int *tokens, size_t num_tokens, char *out_text, size_t max_out_len);
void rcre_char_tokenizer_print_info(const RCRE_CharTokenizer *tok);

/* Byte-Pair Encoding (BPE) Tokenizer */
RCRE_NODISCARD RCRE_BPETokenizer* rcre_bpe_create(void);
bool rcre_bpe_train_from_file(RCRE_BPETokenizer *tok, const char *filepath, int target_vocab_size);
bool rcre_bpe_save(const RCRE_BPETokenizer *tok, const char *filepath);
bool rcre_bpe_load(RCRE_BPETokenizer *tok, const char *filepath);
int rcre_bpe_encode(const RCRE_BPETokenizer *tok, const char *text, size_t text_len, int *out_tokens, int max_tokens);
size_t rcre_bpe_decode(const RCRE_BPETokenizer *tok, const int *tokens, size_t num_tokens, char *out_text, size_t max_out_len);
void rcre_bpe_print_summary(const RCRE_BPETokenizer *tok);
void rcre_bpe_free(RCRE_BPETokenizer *tok);

/* ========================================================================= */
/* Dataset & DataLoader                                                      */
/* ========================================================================= */

RCRE_NODISCARD RCRE_GPTDataLoader* rcre_gpt_dataloader_create(const char *filepath, const RCRE_CharTokenizer *tok, int batch_size, int seq_len);
void rcre_gpt_dataloader_next_batch_random(RCRE_GPTDataLoader *loader, int *x_batch, int *y_batch, uint64_t *rng_state);
void rcre_gpt_dataloader_next_batch_sequential(RCRE_GPTDataLoader *loader, int *x_batch, int *y_batch);
void rcre_gpt_dataloader_reset(RCRE_GPTDataLoader *loader);
void rcre_gpt_dataloader_free(RCRE_GPTDataLoader *loader);

/* ========================================================================= */
/* Optimization & Text Generation                                            */
/* ========================================================================= */

RCRE_NODISCARD float rcre_gpt_get_scheduled_lr(const RCRE_GPTAdamWConfig *cfg, int step);
float rcre_gpt_clip_gradients(float *grads, size_t num_params, float max_norm);
void rcre_gpt_adamw_step(RCRE_GPTModel *model, RCRE_GPTAdamWConfig *cfg);

int rcre_gpt_sample_token(const float *logits, int vocab_size, float temperature, int top_k, uint64_t *rng_state);
void rcre_gpt_generate(
    RCRE_GPTModel *model,
    const RCRE_CharTokenizer *tok,
    const char *prompt,
    const RCRE_GPTGenerateConfig *gen_cfg,
    uint64_t *rng_state
);

bool rcre_gpt_save_checkpoint(const RCRE_GPTModel *model, const char *filepath);
bool rcre_gpt_load_checkpoint(RCRE_GPTModel *model, const char *filepath);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_TRANSFORMER_H */
