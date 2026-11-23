/**
 * @file rcre_llm.h
 * @brief Modern LLM Architecture, GGUF v2/v3 Reader, Block Quantization (Q8_0/Q4_0),
 *        Byte-Level BPE, Sampler, LoRA Fine-Tuning & Sharded Streaming.
 *
 * Part of Recep's Code Run Everywhere (RCRE) v2.0.0.
 * Strict ISO C23 compliant, zero external dependencies.
 */

#ifndef RCRE_LLM_H
#define RCRE_LLM_H

#include "rcre.h"
#include "rcre_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Block Quantization Formats (GGML-Compatible Q8_0 & Q4_0)               */
/* ========================================================================= */

#define RCRE_QK8_0 32
#define RCRE_QK4_0 32

typedef struct {
    uint16_t d;                  /* IEEE FP16 delta scale                     */
    int8_t   qs[RCRE_QK8_0];     /* 32 signed 8-bit quants (34 bytes total)   */
} RCRE_BlockQ8_0;

typedef struct {
    uint16_t d;                  /* IEEE FP16 delta scale                     */
    uint8_t  qs[RCRE_QK4_0 / 2]; /* 32 4-bit nibbles packed low/high (18 B)   */
} RCRE_BlockQ4_0;

typedef enum {
    RCRE_LLM_W_F32  = 0,
    RCRE_LLM_W_Q8_0 = 8,
    RCRE_LLM_W_Q4_0 = 2
} RCRE_LLMWeightType;

void rcre_quantize_q8_0(const float* src, void* dst_blocks, size_t n);
void rcre_dequantize_q8_0(const void* src_blocks, float* dst, size_t n);
void rcre_quantize_q4_0(const float* src, void* dst_blocks, size_t n);
void rcre_dequantize_q4_0(const void* src_blocks, float* dst, size_t n);

RCRE_NODISCARD float rcre_vec_dot_q8_0(const void* q8_blocks, const float* f32_x, size_t n);
RCRE_NODISCARD float rcre_vec_dot_q4_0(const void* q4_blocks, const float* f32_x, size_t n);

void rcre_llm_gemv_quant(RCRE_LLMWeightType wtype, const void* w_data, const float* x,
                         float* y, size_t rows, size_t cols, RCRE_ThreadPool* pool);

/* ========================================================================= */
/* 2. Modern Tensor Math Primitives (RMSNorm, RoPE, SwiGLU, Flash-GQA)       */
/* ========================================================================= */

void rcre_llm_rmsnorm(float* out, const float* x, const float* weight, size_t n, float eps);
void rcre_llm_rope(float* q, float* k, size_t n_heads, size_t n_kv_heads,
                   size_t head_dim, size_t pos, float theta_base);
void rcre_llm_swiglu(float* out, const float* gate, const float* up, size_t n);

/**
 * @brief Flash-Attention style single-query online softmax attention over KV cache.
 *        Zero O(T) temporary score allocation (running max m + denominator l).
 */
void rcre_llm_flash_attention_gqa(
    float* out,
    const float* q,
    const float* k_cache,
    const float* v_cache,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    size_t seq_len,
    size_t kv_stride_elems
);

/* ========================================================================= */
/* 3. Zero-Copy GGUF v2/v3 Reader & Writer                                   */
/* ========================================================================= */

#define RCRE_GGUF_MAGIC 0x46554747U /* 'G','G','U','F' LE */

typedef enum {
    RCRE_GGUF_TYPE_UINT8   = 0,
    RCRE_GGUF_TYPE_INT8    = 1,
    RCRE_GGUF_TYPE_UINT16  = 2,
    RCRE_GGUF_TYPE_INT16   = 3,
    RCRE_GGUF_TYPE_UINT32  = 4,
    RCRE_GGUF_TYPE_INT32   = 5,
    RCRE_GGUF_TYPE_FLOAT32 = 6,
    RCRE_GGUF_TYPE_BOOL    = 7,
    RCRE_GGUF_TYPE_STRING  = 8,
    RCRE_GGUF_TYPE_ARRAY   = 9,
    RCRE_GGUF_TYPE_UINT64  = 10,
    RCRE_GGUF_TYPE_INT64   = 11,
    RCRE_GGUF_TYPE_FLOAT64 = 12
} RCRE_GGUFType;

typedef enum {
    RCRE_GGML_TYPE_F32  = 0,
    RCRE_GGML_TYPE_F16  = 1,
    RCRE_GGML_TYPE_Q4_0 = 2,
    RCRE_GGML_TYPE_Q8_0 = 8,
    RCRE_GGML_TYPE_I8   = 24,
    RCRE_GGML_TYPE_I16  = 25,
    RCRE_GGML_TYPE_I32  = 26,
    RCRE_GGML_TYPE_I64  = 27,
    RCRE_GGML_TYPE_F64  = 28,
    RCRE_GGML_TYPE_BF16 = 30
} RCRE_GGMLType;

typedef struct {
    RCRE_Slice    key;
    RCRE_GGUFType type;
    union {
        uint64_t   u64;
        int64_t    i64;
        double     f64;
        bool       b;
        RCRE_Slice str;
        struct {
            RCRE_GGUFType elem_type;
            uint64_t      len;
            const uint8_t* raw_data;
        } arr;
    } val;
} RCRE_GGUFKV;

typedef struct {
    RCRE_Slice    name;
    uint32_t      n_dims;
    uint64_t      dims[4];
    RCRE_GGMLType type;
    uint64_t      offset; /* Relative to archive data_offset */
    size_t        nbytes;
} RCRE_GGUFTensorInfo;

typedef struct {
    RCRE_MappedFile      map;
    uint32_t             version;
    uint32_t             alignment;
    uint64_t             n_tensors;
    uint64_t             n_kv;
    size_t               data_offset;
    RCRE_GGUFKV*         kvs;
    RCRE_GGUFTensorInfo* tensors;
} RCRE_GGUFFile;

RCRE_NODISCARD int rcre_gguf_open(const char* filepath, RCRE_GGUFFile* out_gguf);
RCRE_NODISCARD int rcre_gguf_open_from_buffer(const void* buf, size_t size, RCRE_GGUFFile* out_gguf);
void rcre_gguf_close(RCRE_GGUFFile* gguf);

RCRE_NODISCARD const RCRE_GGUFKV* rcre_gguf_find_kv(const RCRE_GGUFFile* gguf, const char* key);
RCRE_NODISCARD bool rcre_gguf_get_u32(const RCRE_GGUFFile* gguf, const char* key, uint32_t* out_v);
RCRE_NODISCARD bool rcre_gguf_get_f32(const RCRE_GGUFFile* gguf, const char* key, float* out_v);
RCRE_NODISCARD bool rcre_gguf_get_str(const RCRE_GGUFFile* gguf, const char* key, RCRE_Slice* out_s);
RCRE_NODISCARD const RCRE_GGUFTensorInfo* rcre_gguf_find_tensor(const RCRE_GGUFFile* gguf, const char* name);
RCRE_NODISCARD const void* rcre_gguf_tensor_data(const RCRE_GGUFFile* gguf, const RCRE_GGUFTensorInfo* info);

typedef struct {
    const char*   name;
    RCRE_GGMLType type;
    uint32_t      n_dims;
    uint64_t      dims[4];
    const void*   data;
    size_t        nbytes;
} RCRE_GGUFWriteTensor;

typedef struct {
    const char* key;
    uint32_t    u32_val;
    float       f32_val;
    const char* str_val;
    bool        is_float;
    bool        is_str;
} RCRE_GGUFWriteKV;

RCRE_NODISCARD int rcre_gguf_save_v3(
    const char* filepath,
    const RCRE_GGUFWriteKV* kvs,
    size_t n_kvs,
    const RCRE_GGUFWriteTensor* tensors,
    size_t n_tensors
);

/* ========================================================================= */
/* 4. Large-Vocabulary Byte-Level BPE Tokenizer                              */
/* ========================================================================= */

typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t merged;
} RCRE_ByteBPEMerge;

typedef struct {
    size_t             vocab_size;
    size_t             vocab_cap;
    size_t             n_merges;
    size_t             merges_cap;
    char**             pieces;
    size_t*            piece_lens;
    RCRE_ByteBPEMerge* merges;
    /* Open-addressing hash map for piece lookup & pair->merge lookup */
    uint32_t*          piece_hash_keys; /* 1-based token id, 0 = empty */
    size_t             piece_hash_cap;
    uint64_t*          pair_hash_keys;  /* (left<<32)|right, UINT64_MAX = empty */
    uint32_t*          pair_hash_vals;  /* index into merges[] */
    size_t             pair_hash_cap;
    uint32_t           bos_id;
    uint32_t           eos_id;
} RCRE_ByteBPE;

RCRE_NODISCARD RCRE_ByteBPE* rcre_byte_bpe_create(size_t max_vocab);
void rcre_byte_bpe_free(RCRE_ByteBPE* tok);
RCRE_NODISCARD int rcre_byte_bpe_train_corpus(RCRE_ByteBPE* tok, const uint8_t* corpus,
                                              size_t corpus_len, size_t target_vocab);
RCRE_NODISCARD int rcre_byte_bpe_add_token(RCRE_ByteBPE* tok, const uint8_t* bytes, size_t len);
RCRE_NODISCARD size_t rcre_byte_bpe_encode(const RCRE_ByteBPE* tok, const char* text,
                                           size_t len, int32_t* out_ids, size_t max_ids);
RCRE_NODISCARD size_t rcre_byte_bpe_decode(const RCRE_ByteBPE* tok, const int32_t* ids,
                                           size_t n_ids, char* out_text, size_t max_len);

/* ========================================================================= */
/* 5. Thread-Safe Advanced Sampler (Temp / Top-K / Top-P / Min-P / RepPen)   */
/* ========================================================================= */

typedef struct {
    float    temperature;
    int      top_k;
    float    top_p;
    float    min_p;
    float    repeat_penalty;
    size_t   repeat_window;
    uint64_t rng_state;
} RCRE_SamplerConfig;

typedef struct {
    RCRE_SamplerConfig cfg;
    float*             scratch_probs;
    int32_t*           scratch_idx;
    size_t             vocab_size;
} RCRE_Sampler;

RCRE_NODISCARD RCRE_Sampler* rcre_sampler_create(size_t vocab_size, RCRE_SamplerConfig cfg);
void rcre_sampler_free(RCRE_Sampler* sampler);
RCRE_NODISCARD int32_t rcre_sampler_sample(RCRE_Sampler* sampler, const float* logits,
                                           const int32_t* recent_tokens, size_t n_recent);

/* ========================================================================= */
/* 6. LLaMA Transformer Model, LoRA Adapters & Sharded Streaming Training    */
/* ========================================================================= */

typedef struct {
    size_t vocab_size;
    size_t dim;
    size_t hidden_dim;
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    size_t max_seq_len;
    float  rms_eps;
    float  rope_theta;
} RCRE_LlamaConfig;

typedef struct {
    RCRE_LLMWeightType wtype;
    void*              rms_att;    /* [dim] float */
    void*              rms_ffn;    /* [dim] float */
    void*              wq;         /* [dim, dim] */
    void*              wk;         /* [kv_dim, dim] */
    void*              wv;         /* [kv_dim, dim] */
    void*              wo;         /* [dim, dim] */
    void*              w_gate;     /* [hidden_dim, dim] */
    void*              w_up;       /* [hidden_dim, dim] */
    void*              w_down;     /* [dim, hidden_dim] */
} RCRE_LlamaLayerWeights;

typedef struct {
    RCRE_LlamaConfig        cfg;
    RCRE_LLMWeightType      wtype;
    float*                  tok_emb;    /* [vocab_size, dim] */
    float*                  rms_final;  /* [dim] */
    void*                   lm_head;    /* [vocab_size, dim] */
    RCRE_LlamaLayerWeights* layers;
    bool                    owns_memory;
    RCRE_GGUFFile*          backed_gguf;
    RCRE_ThreadPool*        pool;
} RCRE_LlamaModel;

typedef struct {
    RCRE_LlamaConfig cfg;
    float*           x;
    float*           xb;
    float*           xb2;
    float*           hb;
    float*           hb2;
    float*           q;
    float*           k;
    float*           v;
    float*           att_out;
    float*           logits;
    float*           key_cache;   /* [n_layers, max_seq_len, kv_dim] */
    float*           value_cache; /* [n_layers, max_seq_len, kv_dim] */
} RCRE_LlamaState;

typedef struct {
    size_t  rank;
    size_t  in_dim;
    size_t  out_dim;
    float   alpha;
    float*  A;      /* [rank, in_dim] */
    float*  B;      /* [out_dim, rank] */
    float*  grad_A;
    float*  grad_B;
    float*  m_A;
    float*  v_A;
    float*  m_B;
    float*  v_B;
    size_t  step;
} RCRE_LoRAAdapter;

RCRE_NODISCARD RCRE_LlamaModel* rcre_llama_create(RCRE_LlamaConfig cfg, RCRE_LLMWeightType wtype, uint64_t seed);
RCRE_NODISCARD RCRE_LlamaModel* rcre_llama_load_gguf(const char* filepath, int* out_err);
void rcre_llama_free(RCRE_LlamaModel* model);

RCRE_NODISCARD RCRE_LlamaState* rcre_llama_state_create(const RCRE_LlamaConfig* cfg);
void rcre_llama_state_reset(RCRE_LlamaState* state);
void rcre_llama_state_free(RCRE_LlamaState* state);

RCRE_NODISCARD int rcre_llama_forward_token(const RCRE_LlamaModel* model, RCRE_LlamaState* state,
                                            int32_t token, size_t pos);

RCRE_NODISCARD int rcre_llama_generate(
    const RCRE_LlamaModel* model,
    RCRE_LlamaState* state,
    RCRE_Sampler* sampler,
    const int32_t* prompt_tokens,
    size_t n_prompt,
    int32_t* out_tokens,
    size_t max_new_tokens,
    size_t* out_generated
);

RCRE_NODISCARD RCRE_LoRAAdapter* rcre_lora_create(size_t rank, size_t in_dim, size_t out_dim,
                                                  float alpha, uint64_t seed);
void rcre_lora_free(RCRE_LoRAAdapter* lora);
void rcre_lora_forward_add(const RCRE_LoRAAdapter* lora, const float* x, float* y);
RCRE_NODISCARD float rcre_lora_train_step(
    RCRE_LoRAAdapter* lora,
    const float* base_W_rowmajor,
    const float* x,
    const float* target_y,
    float lr
);

/**
 * @brief Trains a tiny F32 LLaMA step with gradient checkpointing and optional BF16 activation rounding.
 *        Returns cross-entropy loss across sequence [seq_len].
 */
RCRE_NODISCARD float rcre_llama_train_step_f32(
    RCRE_LlamaModel* model,
    const int32_t* tokens,
    size_t seq_len,
    float lr,
    bool bf16_activations,
    bool grad_checkpoint
);

/* --- Sharded Binary Token Streamer (mmap rotating shard files) --- */
typedef struct {
    const char**    shard_paths;
    size_t          num_shards;
    size_t          current_shard;
    size_t          seq_len;
    size_t          token_cursor;
    RCRE_MappedFile active_map;
} RCRE_ShardStream;

RCRE_NODISCARD int rcre_shard_stream_open(RCRE_ShardStream* stream, const char** shard_paths,
                                          size_t num_shards, size_t seq_len);
RCRE_NODISCARD bool rcre_shard_stream_next(RCRE_ShardStream* stream, int32_t* x_out, int32_t* y_out);
void rcre_shard_stream_close(RCRE_ShardStream* stream);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_LLM_H */
