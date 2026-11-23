/**
 * @file llm.c
 * @brief Modern LLaMA Inference, Q8_0/Q4_0 Block Quantization, Flash-GQA,
 *        Byte-Level BPE, Sampler, LoRA Fine-Tuning & Sharded Training (ISO C23)
 */

#include "rcre_llm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
#include <immintrin.h>
#endif

static uint64_t llm_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float llm_rand_normal(uint64_t* rng, float std) {
    double u1 = ((double)(llm_splitmix64(rng) >> 11) + 1.0) / 9007199254740993.0;
    double u2 = ((double)(llm_splitmix64(rng) >> 11)) / 9007199254740992.0;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2)) * std;
}

static inline float llm_fp16_to_f32_one(uint16_t h) {
    float out = 0.0f;
    rcre_f16_to_f32(&h, &out, 1);
    return out;
}

static inline uint16_t llm_f32_to_fp16_one(float f) {
    uint16_t out = 0;
    rcre_f32_to_f16(&f, &out, 1);
    return out;
}

static inline void llm_maybe_bf16_roundtrip(float* x, size_t n, bool enabled) {
    if (!enabled || !x || n == 0) return;
    uint16_t stack_buf[256];
    if (n <= 256) {
        rcre_f32_to_bf16(x, stack_buf, n);
        rcre_bf16_to_f32(stack_buf, x, n);
    } else {
        uint16_t* tmp = (uint16_t*)malloc(n * sizeof(uint16_t));
        if (tmp) {
            rcre_f32_to_bf16(x, tmp, n);
            rcre_bf16_to_f32(tmp, x, n);
            free(tmp);
        }
    }
}

/* ========================================================================= */
/* 1. Q8_0 & Q4_0 Block Quantization + SIMD Dot Products                     */
/* ========================================================================= */

void rcre_quantize_q8_0(const float* src, void* dst_blocks, size_t n) {
    if (!src || !dst_blocks || n == 0) return;
    RCRE_BlockQ8_0* blocks = (RCRE_BlockQ8_0*)dst_blocks;
    size_t nb = (n + RCRE_QK8_0 - 1U) / RCRE_QK8_0;
    for (size_t b = 0; b < nb; ++b) {
        size_t start = b * RCRE_QK8_0;
        size_t len = (n - start < RCRE_QK8_0) ? (n - start) : RCRE_QK8_0;
        float amax = 0.0f;
        for (size_t j = 0; j < len; ++j) {
            float a = fabsf(src[start + j]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        blocks[b].d = llm_f32_to_fp16_one(d);
        for (size_t j = 0; j < RCRE_QK8_0; ++j) {
            if (j < len) {
                int v = (int)lrintf(src[start + j] * id);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                blocks[b].qs[j] = (int8_t)v;
            } else {
                blocks[b].qs[j] = 0;
            }
        }
    }
}

void rcre_dequantize_q8_0(const void* src_blocks, float* dst, size_t n) {
    if (!src_blocks || !dst || n == 0) return;
    const RCRE_BlockQ8_0* blocks = (const RCRE_BlockQ8_0*)src_blocks;
    size_t nb = (n + RCRE_QK8_0 - 1U) / RCRE_QK8_0;
    for (size_t b = 0; b < nb; ++b) {
        size_t start = b * RCRE_QK8_0;
        size_t len = (n - start < RCRE_QK8_0) ? (n - start) : RCRE_QK8_0;
        float d = llm_fp16_to_f32_one(blocks[b].d);
        for (size_t j = 0; j < len; ++j) {
            dst[start + j] = d * (float)blocks[b].qs[j];
        }
    }
}

void rcre_quantize_q4_0(const float* src, void* dst_blocks, size_t n) {
    if (!src || !dst_blocks || n == 0) return;
    RCRE_BlockQ4_0* blocks = (RCRE_BlockQ4_0*)dst_blocks;
    size_t nb = (n + RCRE_QK4_0 - 1U) / RCRE_QK4_0;
    for (size_t b = 0; b < nb; ++b) {
        size_t start = b * RCRE_QK4_0;
        size_t len = (n - start < RCRE_QK4_0) ? (n - start) : RCRE_QK4_0;
        float amax = 0.0f;
        float max_signed = 0.0f;
        for (size_t j = 0; j < len; ++j) {
            float v = src[start + j];
            if (fabsf(v) > amax) {
                amax = fabsf(v);
                max_signed = v;
            }
        }
        float d = max_signed / -8.0f;
        float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        blocks[b].d = llm_f32_to_fp16_one(d);
        for (size_t j = 0; j < RCRE_QK4_0 / 2; ++j) {
            float x0 = (j < len) ? src[start + j] : 0.0f;
            float x1 = (j + 16U < len) ? src[start + j + 16U] : 0.0f;
            int xi0 = (int)truncf(x0 * id + 8.5f);
            int xi1 = (int)truncf(x1 * id + 8.5f);
            if (xi0 < 0) xi0 = 0;
            if (xi0 > 15) xi0 = 15;
            if (xi1 < 0) xi1 = 0;
            if (xi1 > 15) xi1 = 15;
            blocks[b].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
        }
    }
}

void rcre_dequantize_q4_0(const void* src_blocks, float* dst, size_t n) {
    if (!src_blocks || !dst || n == 0) return;
    const RCRE_BlockQ4_0* blocks = (const RCRE_BlockQ4_0*)src_blocks;
    size_t nb = (n + RCRE_QK4_0 - 1U) / RCRE_QK4_0;
    for (size_t b = 0; b < nb; ++b) {
        size_t start = b * RCRE_QK4_0;
        size_t len = (n - start < RCRE_QK4_0) ? (n - start) : RCRE_QK4_0;
        float d = llm_fp16_to_f32_one(blocks[b].d);
        for (size_t j = 0; j < RCRE_QK4_0 / 2; ++j) {
            uint8_t byte = blocks[b].qs[j];
            int q0 = (int)(byte & 0x0FU) - 8;
            int q1 = (int)(byte >> 4U) - 8;
            if (j < len) dst[start + j] = d * (float)q0;
            if (j + 16U < len) dst[start + j + 16U] = d * (float)q1;
        }
    }
}

RCRE_NODISCARD float rcre_vec_dot_q8_0(const void* q8_blocks, const float* f32_x, size_t n) {
    if (!q8_blocks || !f32_x || n == 0) return 0.0f;
    const RCRE_BlockQ8_0* blocks = (const RCRE_BlockQ8_0*)q8_blocks;
    size_t nb = (n + RCRE_QK8_0 - 1U) / RCRE_QK8_0;
    float sum = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        size_t start = b * RCRE_QK8_0;
        size_t len = (n - start < RCRE_QK8_0) ? (n - start) : RCRE_QK8_0;
        float d = llm_fp16_to_f32_one(blocks[b].d);
        float block_acc = 0.0f;
#if defined(__ARM_NEON) || defined(__aarch64__)
        if (len == RCRE_QK8_0) {
            float32x4_t accv = vdupq_n_f32(0.0f);
            for (size_t j = 0; j < RCRE_QK8_0; j += 8) {
                int8x8_t q8 = vld1_s8(blocks[b].qs + j);
                int16x8_t q16 = vmovl_s8(q8);
                int32x4_t q32_lo = vmovl_s16(vget_low_s16(q16));
                int32x4_t q32_hi = vmovl_s16(vget_high_s16(q16));
                float32x4_t f_lo = vcvtq_f32_s32(q32_lo);
                float32x4_t f_hi = vcvtq_f32_s32(q32_hi);
                accv = vfmaq_f32(accv, f_lo, vld1q_f32(f32_x + start + j));
                accv = vfmaq_f32(accv, f_hi, vld1q_f32(f32_x + start + j + 4));
            }
            block_acc = vaddvq_f32(accv);
        } else
#endif
        {
            for (size_t j = 0; j < len; ++j) {
                block_acc += (float)blocks[b].qs[j] * f32_x[start + j];
            }
        }
        sum += d * block_acc;
    }
    return sum;
}

RCRE_NODISCARD float rcre_vec_dot_q4_0(const void* q4_blocks, const float* f32_x, size_t n) {
    if (!q4_blocks || !f32_x || n == 0) return 0.0f;
    const RCRE_BlockQ4_0* blocks = (const RCRE_BlockQ4_0*)q4_blocks;
    size_t nb = (n + RCRE_QK4_0 - 1U) / RCRE_QK4_0;
    float sum = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        size_t start = b * RCRE_QK4_0;
        size_t len = (n - start < RCRE_QK4_0) ? (n - start) : RCRE_QK4_0;
        float d = llm_fp16_to_f32_one(blocks[b].d);
        float block_acc = 0.0f;
        for (size_t j = 0; j < RCRE_QK4_0 / 2; ++j) {
            uint8_t byte = blocks[b].qs[j];
            int q0 = (int)(byte & 0x0FU) - 8;
            int q1 = (int)(byte >> 4U) - 8;
            if (j < len) block_acc += (float)q0 * f32_x[start + j];
            if (j + 16U < len) block_acc += (float)q1 * f32_x[start + j + 16U];
        }
        sum += d * block_acc;
    }
    return sum;
}

typedef struct {
    RCRE_LLMWeightType wtype;
    const uint8_t*     w_bytes;
    const float*       x;
    float*             y;
    size_t             rows;
    size_t             cols;
    size_t             row_bytes;
} RCRE_LLMGemvTask;

static void llm_gemv_quant_worker(void* arg, size_t row) {
    RCRE_LLMGemvTask* t = (RCRE_LLMGemvTask*)arg;
    const void* row_ptr = t->w_bytes + row * t->row_bytes;
    if (t->wtype == RCRE_LLM_W_Q8_0) {
        t->y[row] = rcre_vec_dot_q8_0(row_ptr, t->x, t->cols);
    } else if (t->wtype == RCRE_LLM_W_Q4_0) {
        t->y[row] = rcre_vec_dot_q4_0(row_ptr, t->x, t->cols);
    } else {
        t->y[row] = rcre_hal.dot_f32((const float*)row_ptr, t->x, t->cols);
    }
}

void rcre_llm_gemv_quant(RCRE_LLMWeightType wtype, const void* w_data, const float* x,
                         float* y, size_t rows, size_t cols, RCRE_ThreadPool* pool) {
    if (!w_data || !x || !y || rows == 0 || cols == 0) return;
    if (wtype == RCRE_LLM_W_F32 && (!pool || rows < 64)) {
        rcre_hal.gemv_f32(rows, cols, 1.0f, (const float*)w_data, cols, x, 0.0f, y);
        return;
    }
    size_t row_bytes = 0;
    if (wtype == RCRE_LLM_W_Q8_0) {
        row_bytes = ((cols + 31U) / 32U) * sizeof(RCRE_BlockQ8_0);
    } else if (wtype == RCRE_LLM_W_Q4_0) {
        row_bytes = ((cols + 31U) / 32U) * sizeof(RCRE_BlockQ4_0);
    } else {
        row_bytes = cols * sizeof(float);
    }

    RCRE_LLMGemvTask task = {
        .wtype     = wtype,
        .w_bytes   = (const uint8_t*)w_data,
        .x         = x,
        .y         = y,
        .rows      = rows,
        .cols      = cols,
        .row_bytes = row_bytes
    };
    if (pool && rows >= 32) {
        rcre_threadpool_parallel_for(pool, rows, llm_gemv_quant_worker, &task);
    } else {
        for (size_t r = 0; r < rows; ++r) {
            llm_gemv_quant_worker(&task, r);
        }
    }
}

/* ========================================================================= */
/* 2. Modern Tensor Math Primitives                                          */
/* ========================================================================= */

void rcre_llm_rmsnorm(float* out, const float* x, const float* weight, size_t n, float eps) {
    if (!out || !x || !weight || n == 0) return;
    float ss = rcre_hal.dot_f32(x, x, n);
    float inv_rms = 1.0f / sqrtf((ss / (float)n) + eps);
    for (size_t i = 0; i < n; ++i) {
        out[i] = weight[i] * (x[i] * inv_rms);
    }
}

void rcre_llm_rope(float* q, float* k, size_t n_heads, size_t n_kv_heads,
                   size_t head_dim, size_t pos, float theta_base) {
    if (!q || head_dim < 2) return;
    for (size_t h = 0; h < n_heads; ++h) {
        float* qh = q + h * head_dim;
        for (size_t i = 0; i + 1 < head_dim; i += 2) {
            float freq = 1.0f / powf(theta_base, (float)i / (float)head_dim);
            float val = (float)pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            float v0 = qh[i];
            float v1 = qh[i + 1];
            qh[i]     = v0 * fcr - v1 * fci;
            qh[i + 1] = v0 * fci + v1 * fcr;
        }
    }
    if (k) {
        for (size_t h = 0; h < n_kv_heads; ++h) {
            float* kh = k + h * head_dim;
            for (size_t i = 0; i + 1 < head_dim; i += 2) {
                float freq = 1.0f / powf(theta_base, (float)i / (float)head_dim);
                float val = (float)pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                float v0 = kh[i];
                float v1 = kh[i + 1];
                kh[i]     = v0 * fcr - v1 * fci;
                kh[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }
}

void rcre_llm_swiglu(float* out, const float* gate, const float* up, size_t n) {
    if (!out || !gate || !up) return;
    for (size_t i = 0; i < n; ++i) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

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
) {
    if (!out || !q || !k_cache || !v_cache || n_heads == 0 || n_kv_heads == 0 || seq_len == 0) return;
    size_t kv_mul = n_heads / n_kv_heads;
    if (kv_mul == 0) kv_mul = 1;
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    /* Flash-Attention tiled online softmax: 0 score array allocation! */
    for (size_t h = 0; h < n_heads; ++h) {
        const float* qh = q + h * head_dim;
        float* oh = out + h * head_dim;
        size_t kv_h = h / kv_mul;
        for (size_t d = 0; d < head_dim; ++d) oh[d] = 0.0f;

        float m_prev = -1e30f;
        float l_prev = 0.0f;

        for (size_t t = 0; t < seq_len; ++t) {
            const float* kh = k_cache + t * kv_stride_elems + kv_h * head_dim;
            const float* vh = v_cache + t * kv_stride_elems + kv_h * head_dim;
            float score = rcre_hal.dot_f32(qh, kh, head_dim) * inv_sqrt_d;

            float m_new = (score > m_prev) ? score : m_prev;
            float exp_prev = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
            float exp_cur  = expf(score - m_new);
            float l_new    = l_prev * exp_prev + exp_cur;

            for (size_t d = 0; d < head_dim; ++d) {
                oh[d] = oh[d] * exp_prev + exp_cur * vh[d];
            }
            m_prev = m_new;
            l_prev = l_new;
        }
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            oh[d] *= inv_l;
        }
    }
}

/* ========================================================================= */
/* 3. Large-Vocabulary Byte-Level BPE Tokenizer                              */
/* ========================================================================= */

static inline uint64_t llm_fnv1a(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static size_t llm_next_pow2(size_t n) {
    size_t p = 64;
    while (p < n) p <<= 1U;
    return p;
}

static void llm_bpe_insert_piece_hash(RCRE_ByteBPE* tok, uint32_t id) {
    uint64_t h = llm_fnv1a(tok->pieces[id], tok->piece_lens[id]);
    size_t mask = tok->piece_hash_cap - 1U;
    size_t idx = (size_t)h & mask;
    while (tok->piece_hash_keys[idx] != 0) {
        uint32_t existing = tok->piece_hash_keys[idx] - 1U;
        if (tok->piece_lens[existing] == tok->piece_lens[id] &&
            memcmp(tok->pieces[existing], tok->pieces[id], tok->piece_lens[id]) == 0) {
            return;
        }
        idx = (idx + 1U) & mask;
    }
    tok->piece_hash_keys[idx] = id + 1U;
}

static int32_t llm_bpe_find_piece(const RCRE_ByteBPE* tok, const uint8_t* s, size_t len) {
    uint64_t h = llm_fnv1a(s, len);
    size_t mask = tok->piece_hash_cap - 1U;
    size_t idx = (size_t)h & mask;
    while (tok->piece_hash_keys[idx] != 0) {
        uint32_t id = tok->piece_hash_keys[idx] - 1U;
        if (tok->piece_lens[id] == len && (len == 0 || memcmp(tok->pieces[id], s, len) == 0)) {
            return (int32_t)id;
        }
        idx = (idx + 1U) & mask;
    }
    return -1;
}

static void llm_bpe_insert_pair_hash(RCRE_ByteBPE* tok, uint32_t l, uint32_t r, uint32_t merge_idx) {
    uint64_t key = ((uint64_t)l << 32) | (uint64_t)r;
    uint64_t h = llm_fnv1a(&key, sizeof(key));
    size_t mask = tok->pair_hash_cap - 1U;
    size_t idx = (size_t)h & mask;
    while (tok->pair_hash_keys[idx] != UINT64_MAX) {
        if (tok->pair_hash_keys[idx] == key) return;
        idx = (idx + 1U) & mask;
    }
    tok->pair_hash_keys[idx] = key;
    tok->pair_hash_vals[idx] = merge_idx;
}

static int32_t llm_bpe_find_pair_rank(const RCRE_ByteBPE* tok, uint32_t l, uint32_t r) {
    uint64_t key = ((uint64_t)l << 32) | (uint64_t)r;
    uint64_t h = llm_fnv1a(&key, sizeof(key));
    size_t mask = tok->pair_hash_cap - 1U;
    size_t idx = (size_t)h & mask;
    while (tok->pair_hash_keys[idx] != UINT64_MAX) {
        if (tok->pair_hash_keys[idx] == key) return (int32_t)tok->pair_hash_vals[idx];
        idx = (idx + 1U) & mask;
    }
    return -1;
}

RCRE_NODISCARD RCRE_ByteBPE* rcre_byte_bpe_create(size_t max_vocab) {
    if (max_vocab < 258U) max_vocab = 258U;
    RCRE_ByteBPE* tok = (RCRE_ByteBPE*)calloc(1, sizeof(RCRE_ByteBPE));
    if (!tok) return nullptr;

    tok->vocab_cap      = max_vocab;
    tok->merges_cap     = max_vocab;
    tok->piece_hash_cap = llm_next_pow2(max_vocab * 2U);
    tok->pair_hash_cap  = llm_next_pow2(max_vocab * 2U);

    tok->pieces          = (char**)calloc(max_vocab, sizeof(char*));
    tok->piece_lens      = (size_t*)calloc(max_vocab, sizeof(size_t));
    tok->merges          = (RCRE_ByteBPEMerge*)calloc(max_vocab, sizeof(RCRE_ByteBPEMerge));
    tok->piece_hash_keys = (uint32_t*)calloc(tok->piece_hash_cap, sizeof(uint32_t));
    tok->pair_hash_keys  = (uint64_t*)malloc(tok->pair_hash_cap * sizeof(uint64_t));
    tok->pair_hash_vals  = (uint32_t*)calloc(tok->pair_hash_cap, sizeof(uint32_t));

    if (!tok->pieces || !tok->piece_lens || !tok->merges ||
        !tok->piece_hash_keys || !tok->pair_hash_keys || !tok->pair_hash_vals) {
        rcre_byte_bpe_free(tok);
        return nullptr;
    }
    for (size_t i = 0; i < tok->pair_hash_cap; ++i) tok->pair_hash_keys[i] = UINT64_MAX;

    /* 256 raw byte tokens + BOS(256) + EOS(257) */
    for (int b = 0; b < 256; ++b) {
        uint8_t byte = (uint8_t)b;
        (void)rcre_byte_bpe_add_token(tok, &byte, 1);
    }
    tok->bos_id = (uint32_t)rcre_byte_bpe_add_token(tok, (const uint8_t*)"<bos>", 5);
    tok->eos_id = (uint32_t)rcre_byte_bpe_add_token(tok, (const uint8_t*)"<eos>", 5);
    return tok;
}

void rcre_byte_bpe_free(RCRE_ByteBPE* tok) {
    if (!tok) return;
    if (tok->pieces) {
        for (size_t i = 0; i < tok->vocab_size; ++i) free(tok->pieces[i]);
        free(tok->pieces);
    }
    free(tok->piece_lens);
    free(tok->merges);
    free(tok->piece_hash_keys);
    free(tok->pair_hash_keys);
    free(tok->pair_hash_vals);
    free(tok);
}

RCRE_NODISCARD int rcre_byte_bpe_add_token(RCRE_ByteBPE* tok, const uint8_t* bytes, size_t len) {
    if (!tok || (!bytes && len > 0) || tok->vocab_size >= tok->vocab_cap) return -1;
    uint32_t id = (uint32_t)tok->vocab_size;
    char* copy = (char*)malloc(len + 1U);
    if (!copy) return -1;
    if (len > 0) memcpy(copy, bytes, len);
    copy[len] = '\0';
    tok->pieces[id] = copy;
    tok->piece_lens[id] = len;
    tok->vocab_size++;
    llm_bpe_insert_piece_hash(tok, id);
    return (int)id;
}

RCRE_NODISCARD int rcre_byte_bpe_train_corpus(RCRE_ByteBPE* tok, const uint8_t* corpus,
                                              size_t corpus_len, size_t target_vocab) {
    if (!tok || !corpus || corpus_len < 2) return RCRE_ERR_INVALID_ARG;
    if (target_vocab > tok->vocab_cap) target_vocab = tok->vocab_cap;

    uint32_t* ids = (uint32_t*)malloc(corpus_len * sizeof(uint32_t));
    if (!ids) return RCRE_ERR_OUT_OF_MEMORY;
    for (size_t i = 0; i < corpus_len; ++i) ids[i] = (uint32_t)corpus[i];
    size_t cur_len = corpus_len;

    const size_t TBL_CAP = 16384U;
    uint64_t* keys = (uint64_t*)malloc(TBL_CAP * sizeof(uint64_t));
    uint32_t* cnts = (uint32_t*)malloc(TBL_CAP * sizeof(uint32_t));
    if (!keys || !cnts) {
        free(ids); free(keys); free(cnts);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    while (tok->vocab_size < target_vocab && cur_len >= 2) {
        for (size_t i = 0; i < TBL_CAP; ++i) {
            keys[i] = UINT64_MAX;
            cnts[i] = 0;
        }
        uint64_t best_key = UINT64_MAX;
        uint32_t best_cnt = 1; /* require frequency >= 2 */

        for (size_t i = 0; i + 1 < cur_len; ++i) {
            uint64_t k = ((uint64_t)ids[i] << 32) | (uint64_t)ids[i + 1];
            size_t slot = (size_t)(llm_fnv1a(&k, sizeof(k)) & (TBL_CAP - 1U));
            for (size_t probe = 0; probe < TBL_CAP; ++probe) {
                size_t p = (slot + probe) & (TBL_CAP - 1U);
                if (keys[p] == UINT64_MAX || keys[p] == k) {
                    keys[p] = k;
                    cnts[p]++;
                    if (cnts[p] > best_cnt) {
                        best_cnt = cnts[p];
                        best_key = k;
                    }
                    break;
                }
            }
        }
        if (best_key == UINT64_MAX) break;

        uint32_t l = (uint32_t)(best_key >> 32);
        uint32_t r = (uint32_t)(best_key & 0xFFFFFFFFULL);
        size_t new_len = tok->piece_lens[l] + tok->piece_lens[r];
        uint8_t* combined = (uint8_t*)malloc(new_len);
        if (!combined) break;
        memcpy(combined, tok->pieces[l], tok->piece_lens[l]);
        memcpy(combined + tok->piece_lens[l], tok->pieces[r], tok->piece_lens[r]);

        int merged_id = rcre_byte_bpe_add_token(tok, combined, new_len);
        free(combined);
        if (merged_id < 0) break;

        uint32_t m_idx = (uint32_t)tok->n_merges++;
        tok->merges[m_idx] = (RCRE_ByteBPEMerge){ l, r, (uint32_t)merged_id };
        llm_bpe_insert_pair_hash(tok, l, r, m_idx);

        /* Rewrite active token stream */
        size_t out_n = 0;
        for (size_t i = 0; i < cur_len; ) {
            if (i + 1 < cur_len && ids[i] == l && ids[i + 1] == r) {
                ids[out_n++] = (uint32_t)merged_id;
                i += 2;
            } else {
                ids[out_n++] = ids[i++];
            }
        }
        cur_len = out_n;
    }

    free(keys);
    free(cnts);
    free(ids);
    return RCRE_OK;
}

RCRE_NODISCARD size_t rcre_byte_bpe_encode(const RCRE_ByteBPE* tok, const char* text,
                                           size_t len, int32_t* out_ids, size_t max_ids) {
    if (!tok || !text || !out_ids || max_ids == 0 || len == 0) return 0;
    uint32_t* buf = (uint32_t*)malloc(len * sizeof(uint32_t));
    if (!buf) return 0;

    for (size_t i = 0; i < len; ++i) {
        buf[i] = (uint32_t)(uint8_t)text[i];
    }
    size_t n = len;

    for (;;) {
        int32_t best_rank = INT32_MAX;
        size_t best_pos = SIZE_MAX;
        uint32_t best_merged = 0;

        for (size_t i = 0; i + 1 < n; ++i) {
            int32_t r = llm_bpe_find_pair_rank(tok, buf[i], buf[i + 1]);
            if (r >= 0 && r < best_rank) {
                best_rank = r;
                best_pos = i;
                best_merged = tok->merges[r].merged;
            }
        }
        if (best_pos == SIZE_MAX) break;

        /* Apply lowest-rank merge across buffer */
        uint32_t l = buf[best_pos];
        uint32_t r = buf[best_pos + 1];
        size_t w = 0;
        for (size_t i = 0; i < n; ) {
            if (i + 1 < n && buf[i] == l && buf[i + 1] == r) {
                buf[w++] = best_merged;
                i += 2;
            } else {
                buf[w++] = buf[i++];
            }
        }
        n = w;
    }

    size_t copy_n = (n < max_ids) ? n : max_ids;
    for (size_t i = 0; i < copy_n; ++i) out_ids[i] = (int32_t)buf[i];
    free(buf);
    (void)llm_bpe_find_piece;
    return copy_n;
}

RCRE_NODISCARD size_t rcre_byte_bpe_decode(const RCRE_ByteBPE* tok, const int32_t* ids,
                                           size_t n_ids, char* out_text, size_t max_len) {
    if (!tok || !ids || !out_text || max_len == 0) return 0;
    size_t pos = 0;
    for (size_t i = 0; i < n_ids; ++i) {
        int32_t id = ids[i];
        if (id < 0 || (size_t)id >= tok->vocab_size) continue;
        size_t plen = tok->piece_lens[id];
        if (pos + plen >= max_len) break;
        memcpy(out_text + pos, tok->pieces[id], plen);
        pos += plen;
    }
    out_text[pos] = '\0';
    return pos;
}

/* ========================================================================= */
/* 4. Thread-Safe Advanced Sampler                                           */
/* ========================================================================= */

typedef struct {
    float   prob;
    int32_t id;
} RCRE_ProbIdx;

static int llm_cmp_prob_desc(const void* a, const void* b) {
    const RCRE_ProbIdx* pa = (const RCRE_ProbIdx*)a;
    const RCRE_ProbIdx* pb = (const RCRE_ProbIdx*)b;
    if (pb->prob > pa->prob) return 1;
    if (pb->prob < pa->prob) return -1;
    return 0;
}

RCRE_NODISCARD RCRE_Sampler* rcre_sampler_create(size_t vocab_size, RCRE_SamplerConfig cfg) {
    if (vocab_size == 0) return nullptr;
    RCRE_Sampler* s = (RCRE_Sampler*)calloc(1, sizeof(RCRE_Sampler));
    if (!s) return nullptr;
    s->cfg = cfg;
    if (s->cfg.rng_state == 0) s->cfg.rng_state = 0xCAFEBABE12345678ULL;
    s->vocab_size = vocab_size;
    s->scratch_probs = (float*)calloc(vocab_size, sizeof(float));
    s->scratch_idx   = (int32_t*)calloc(vocab_size, sizeof(int32_t));
    if (!s->scratch_probs || !s->scratch_idx) {
        rcre_sampler_free(s);
        return nullptr;
    }
    return s;
}

void rcre_sampler_free(RCRE_Sampler* sampler) {
    if (!sampler) return;
    free(sampler->scratch_probs);
    free(sampler->scratch_idx);
    free(sampler);
}

RCRE_NODISCARD int32_t rcre_sampler_sample(RCRE_Sampler* sampler, const float* logits,
                                           const int32_t* recent_tokens, size_t n_recent) {
    if (!sampler || !logits || sampler->vocab_size == 0) return 0;
    size_t V = sampler->vocab_size;
    float* p = sampler->scratch_probs;
    memcpy(p, logits, V * sizeof(float));

    /* 1. Repetition penalty over recent window */
    if (sampler->cfg.repeat_penalty > 1.0f && recent_tokens && n_recent > 0) {
        size_t w = (sampler->cfg.repeat_window > 0 && sampler->cfg.repeat_window < n_recent)
                       ? sampler->cfg.repeat_window
                       : n_recent;
        size_t start = n_recent - w;
        float rp = sampler->cfg.repeat_penalty;
        for (size_t i = start; i < n_recent; ++i) {
            int32_t tid = recent_tokens[i];
            if (tid >= 0 && (size_t)tid < V) {
                p[tid] = (p[tid] > 0.0f) ? (p[tid] / rp) : (p[tid] * rp);
            }
        }
    }

    /* Greedy argmax when temperature <= 0 */
    if (sampler->cfg.temperature <= 0.0f) {
        size_t best = 0;
        for (size_t i = 1; i < V; ++i) {
            if (p[i] > p[best]) best = i;
        }
        return (int32_t)best;
    }

    /* 2. Temperature scaling + numerically stable softmax */
    float inv_t = 1.0f / sampler->cfg.temperature;
    float max_l = p[0] * inv_t;
    for (size_t i = 1; i < V; ++i) {
        float v = p[i] * inv_t;
        if (v > max_l) max_l = v;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < V; ++i) {
        p[i] = expf(p[i] * inv_t - max_l);
        sum += p[i];
    }
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    for (size_t i = 0; i < V; ++i) p[i] *= inv_sum;

    /* 3. Sort candidate distribution when top-k / top-p / min-p active */
    RCRE_ProbIdx* pairs = (RCRE_ProbIdx*)malloc(V * sizeof(RCRE_ProbIdx));
    if (!pairs) return 0;
    for (size_t i = 0; i < V; ++i) {
        pairs[i].prob = p[i];
        pairs[i].id   = (int32_t)i;
    }
    qsort(pairs, V, sizeof(RCRE_ProbIdx), llm_cmp_prob_desc);

    size_t cutoff = V;
    if (sampler->cfg.top_k > 0 && (size_t)sampler->cfg.top_k < cutoff) {
        cutoff = (size_t)sampler->cfg.top_k;
    }

    if (sampler->cfg.min_p > 0.0f && cutoff > 1) {
        float thresh = pairs[0].prob * sampler->cfg.min_p;
        for (size_t i = 1; i < cutoff; ++i) {
            if (pairs[i].prob < thresh) {
                cutoff = i;
                break;
            }
        }
    }

    if (sampler->cfg.top_p > 0.0f && sampler->cfg.top_p < 1.0f && cutoff > 1) {
        float cum = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) {
            cum += pairs[i].prob;
            if (cum >= sampler->cfg.top_p) {
                cutoff = i + 1U;
                break;
            }
        }
    }

    float renorm = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) renorm += pairs[i].prob;
    double r = ((double)(llm_splitmix64(&sampler->cfg.rng_state) >> 11) / 9007199254740992.0) * (double)renorm;
    float cdf = 0.0f;
    int32_t chosen = pairs[0].id;
    for (size_t i = 0; i < cutoff; ++i) {
        cdf += pairs[i].prob;
        if ((double)cdf >= r) {
            chosen = pairs[i].id;
            break;
        }
    }

    free(pairs);
    return chosen;
}

/* ========================================================================= */
/* 5. Modern LLaMA Model Inference & GGUF Binding                            */
/* ========================================================================= */

static size_t llm_weight_matrix_bytes(RCRE_LLMWeightType wtype, size_t rows, size_t cols) {
    if (wtype == RCRE_LLM_W_Q8_0) {
        return rows * ((cols + 31U) / 32U) * sizeof(RCRE_BlockQ8_0);
    }
    if (wtype == RCRE_LLM_W_Q4_0) {
        return rows * ((cols + 31U) / 32U) * sizeof(RCRE_BlockQ4_0);
    }
    return rows * cols * sizeof(float);
}

RCRE_NODISCARD RCRE_LlamaModel* rcre_llama_create(RCRE_LlamaConfig cfg, RCRE_LLMWeightType wtype, uint64_t seed) {
    if (cfg.vocab_size == 0 || cfg.dim == 0 || cfg.hidden_dim == 0 ||
        cfg.n_layers == 0 || cfg.n_heads == 0 || cfg.n_kv_heads == 0 ||
        (cfg.dim % cfg.n_heads) != 0 || (cfg.n_heads % cfg.n_kv_heads) != 0) {
        return nullptr;
    }
    if (cfg.rms_eps <= 0.0f) cfg.rms_eps = 1e-5f;
    if (cfg.rope_theta <= 0.0f) cfg.rope_theta = 10000.0f;
    if (cfg.max_seq_len == 0) cfg.max_seq_len = 256;

    RCRE_LlamaModel* m = (RCRE_LlamaModel*)calloc(1, sizeof(RCRE_LlamaModel));
    if (!m) return nullptr;
    m->cfg = cfg;
    m->wtype = wtype;
    m->owns_memory = true;
    m->tok_emb   = (float*)calloc(cfg.vocab_size * cfg.dim, sizeof(float));
    m->rms_final = (float*)calloc(cfg.dim, sizeof(float));
    m->lm_head   = calloc(1, llm_weight_matrix_bytes(wtype, cfg.vocab_size, cfg.dim));
    m->layers    = (RCRE_LlamaLayerWeights*)calloc(cfg.n_layers, sizeof(RCRE_LlamaLayerWeights));

    if (!m->tok_emb || !m->rms_final || !m->lm_head || !m->layers) {
        rcre_llama_free(m);
        return nullptr;
    }

    size_t kv_dim = (cfg.dim / cfg.n_heads) * cfg.n_kv_heads;
    float* tmp_w = (float*)malloc(cfg.hidden_dim * cfg.dim * sizeof(float) + cfg.vocab_size * cfg.dim * sizeof(float));
    if (!tmp_w) { rcre_llama_free(m); return nullptr; }

    for (size_t i = 0; i < cfg.vocab_size * cfg.dim; ++i) {
        m->tok_emb[i] = llm_rand_normal(&seed, 0.02f);
        tmp_w[i]      = m->tok_emb[i];
    }
    if (wtype == RCRE_LLM_W_Q8_0) {
        for (size_t r = 0; r < cfg.vocab_size; ++r) {
            rcre_quantize_q8_0(tmp_w + r * cfg.dim,
                               (uint8_t*)m->lm_head + r * ((cfg.dim + 31U) / 32U) * sizeof(RCRE_BlockQ8_0),
                               cfg.dim);
        }
    } else if (wtype == RCRE_LLM_W_Q4_0) {
        for (size_t r = 0; r < cfg.vocab_size; ++r) {
            rcre_quantize_q4_0(tmp_w + r * cfg.dim,
                               (uint8_t*)m->lm_head + r * ((cfg.dim + 31U) / 32U) * sizeof(RCRE_BlockQ4_0),
                               cfg.dim);
        }
    } else {
        memcpy(m->lm_head, tmp_w, cfg.vocab_size * cfg.dim * sizeof(float));
    }

    for (size_t i = 0; i < cfg.dim; ++i) m->rms_final[i] = 1.0f;

    for (size_t l = 0; l < cfg.n_layers; ++l) {
        RCRE_LlamaLayerWeights* lw = &m->layers[l];
        lw->wtype   = wtype;
        lw->rms_att = malloc(cfg.dim * sizeof(float));
        lw->rms_ffn = malloc(cfg.dim * sizeof(float));
        lw->wq      = calloc(1, llm_weight_matrix_bytes(wtype, cfg.dim, cfg.dim));
        lw->wk      = calloc(1, llm_weight_matrix_bytes(wtype, kv_dim, cfg.dim));
        lw->wv      = calloc(1, llm_weight_matrix_bytes(wtype, kv_dim, cfg.dim));
        lw->wo      = calloc(1, llm_weight_matrix_bytes(wtype, cfg.dim, cfg.dim));
        lw->w_gate  = calloc(1, llm_weight_matrix_bytes(wtype, cfg.hidden_dim, cfg.dim));
        lw->w_up    = calloc(1, llm_weight_matrix_bytes(wtype, cfg.hidden_dim, cfg.dim));
        lw->w_down  = calloc(1, llm_weight_matrix_bytes(wtype, cfg.dim, cfg.hidden_dim));

        if (!lw->rms_att || !lw->rms_ffn || !lw->wq || !lw->wk || !lw->wv ||
            !lw->wo || !lw->w_gate || !lw->w_up || !lw->w_down) {
            free(tmp_w);
            rcre_llama_free(m);
            return nullptr;
        }
        float* ra = (float*)lw->rms_att;
        float* rf = (float*)lw->rms_ffn;
        for (size_t i = 0; i < cfg.dim; ++i) { ra[i] = 1.0f; rf[i] = 1.0f; }

        size_t specs[7][2] = {
            {cfg.dim, cfg.dim}, {kv_dim, cfg.dim}, {kv_dim, cfg.dim}, {cfg.dim, cfg.dim},
            {cfg.hidden_dim, cfg.dim}, {cfg.hidden_dim, cfg.dim}, {cfg.dim, cfg.hidden_dim}
        };
        void* dsts[7] = { lw->wq, lw->wk, lw->wv, lw->wo, lw->w_gate, lw->w_up, lw->w_down };
        for (int p = 0; p < 7; ++p) {
            size_t R = specs[p][0], C = specs[p][1];
            float std = 1.0f / sqrtf((float)C);
            for (size_t i = 0; i < R * C; ++i) tmp_w[i] = llm_rand_normal(&seed, std);
            if (wtype == RCRE_LLM_W_Q8_0) {
                size_t rb = ((C + 31U) / 32U) * sizeof(RCRE_BlockQ8_0);
                for (size_t r = 0; r < R; ++r) {
                    rcre_quantize_q8_0(tmp_w + r * C, (uint8_t*)dsts[p] + r * rb, C);
                }
            } else if (wtype == RCRE_LLM_W_Q4_0) {
                size_t rb = ((C + 31U) / 32U) * sizeof(RCRE_BlockQ4_0);
                for (size_t r = 0; r < R; ++r) {
                    rcre_quantize_q4_0(tmp_w + r * C, (uint8_t*)dsts[p] + r * rb, C);
                }
            } else {
                memcpy(dsts[p], tmp_w, R * C * sizeof(float));
            }
        }
    }

    free(tmp_w);
    return m;
}

static bool llm_gguf_read_f32_tensor(const RCRE_GGUFFile* g, const char* name, float* dst, size_t expected_elems) {
    const RCRE_GGUFTensorInfo* ti = rcre_gguf_find_tensor(g, name);
    if (!ti) return false;
    const void* raw = rcre_gguf_tensor_data(g, ti);
    if (!raw) return false;
    if (ti->type == RCRE_GGML_TYPE_F32 && ti->nbytes == expected_elems * 4U) {
        memcpy(dst, raw, expected_elems * 4U);
        return true;
    }
    if (ti->type == RCRE_GGML_TYPE_F16 && ti->nbytes == expected_elems * 2U) {
        rcre_f16_to_f32(raw, dst, expected_elems);
        return true;
    }
    if (ti->type == RCRE_GGML_TYPE_BF16 && ti->nbytes == expected_elems * 2U) {
        rcre_bf16_to_f32(raw, dst, expected_elems);
        return true;
    }
    if (ti->type == RCRE_GGML_TYPE_Q8_0) {
        rcre_dequantize_q8_0(raw, dst, expected_elems);
        return true;
    }
    if (ti->type == RCRE_GGML_TYPE_Q4_0) {
        rcre_dequantize_q4_0(raw, dst, expected_elems);
        return true;
    }
    return false;
}

static const void* llm_gguf_bind_weight(const RCRE_GGUFFile* g, const char* name, RCRE_LLMWeightType* out_wt) {
    const RCRE_GGUFTensorInfo* ti = rcre_gguf_find_tensor(g, name);
    if (!ti) return nullptr;
    if (ti->type == RCRE_GGML_TYPE_F32) *out_wt = RCRE_LLM_W_F32;
    else if (ti->type == RCRE_GGML_TYPE_Q8_0) *out_wt = RCRE_LLM_W_Q8_0;
    else if (ti->type == RCRE_GGML_TYPE_Q4_0) *out_wt = RCRE_LLM_W_Q4_0;
    else return nullptr;
    return rcre_gguf_tensor_data(g, ti);
}

RCRE_NODISCARD RCRE_LlamaModel* rcre_llama_load_gguf(const char* filepath, int* out_err) {
    if (out_err) *out_err = RCRE_ERR_INVALID_ARG;
    if (!filepath) return nullptr;

    RCRE_GGUFFile* g = (RCRE_GGUFFile*)calloc(1, sizeof(RCRE_GGUFFile));
    if (!g) { if (out_err) *out_err = RCRE_ERR_OUT_OF_MEMORY; return nullptr; }
    int rc = rcre_gguf_open(filepath, g);
    if (rc != RCRE_OK) { free(g); if (out_err) *out_err = rc; return nullptr; }

    uint32_t dim = 0, hidden = 0, n_layers = 0, n_heads = 0, n_kv = 0, ctx_len = 256, vocab = 0;
    float eps = 1e-5f, theta = 10000.0f;
    if (!rcre_gguf_get_u32(g, "llama.embedding_length", &dim) ||
        !rcre_gguf_get_u32(g, "llama.feed_forward_length", &hidden) ||
        !rcre_gguf_get_u32(g, "llama.block_count", &n_layers) ||
        !rcre_gguf_get_u32(g, "llama.attention.head_count", &n_heads) ||
        !rcre_gguf_get_u32(g, "llama.vocab_size", &vocab)) {
        rcre_gguf_close(g); free(g);
        if (out_err) *out_err = RCRE_ERR_IO;
        return nullptr;
    }
    if (!rcre_gguf_get_u32(g, "llama.attention.head_count_kv", &n_kv)) n_kv = n_heads;
    (void)rcre_gguf_get_u32(g, "llama.context_length", &ctx_len);
    (void)rcre_gguf_get_f32(g, "llama.attention.layer_norm_rms_epsilon", &eps);
    (void)rcre_gguf_get_f32(g, "llama.rope.freq_base", &theta);

    RCRE_LlamaModel* m = (RCRE_LlamaModel*)calloc(1, sizeof(RCRE_LlamaModel));
    if (!m) { rcre_gguf_close(g); free(g); if (out_err) *out_err = RCRE_ERR_OUT_OF_MEMORY; return nullptr; }
    m->cfg = (RCRE_LlamaConfig){
        .vocab_size  = vocab,
        .dim         = dim,
        .hidden_dim  = hidden,
        .n_layers    = n_layers,
        .n_heads     = n_heads,
        .n_kv_heads  = n_kv,
        .max_seq_len = ctx_len,
        .rms_eps     = eps,
        .rope_theta  = theta
    };
    m->owns_memory = false;
    m->backed_gguf = g;

    m->tok_emb   = (float*)malloc((size_t)vocab * dim * sizeof(float));
    m->rms_final = (float*)malloc(dim * sizeof(float));
    m->layers    = (RCRE_LlamaLayerWeights*)calloc(n_layers, sizeof(RCRE_LlamaLayerWeights));
    if (!m->tok_emb || !m->rms_final || !m->layers ||
        !llm_gguf_read_f32_tensor(g, "token_embd.weight", m->tok_emb, (size_t)vocab * dim) ||
        !llm_gguf_read_f32_tensor(g, "output_norm.weight", m->rms_final, dim)) {
        rcre_llama_free(m);
        if (out_err) *out_err = RCRE_ERR_IO;
        return nullptr;
    }
    RCRE_LLMWeightType head_wt = RCRE_LLM_W_F32;
    m->lm_head = (void*)(uintptr_t)llm_gguf_bind_weight(g, "output.weight", &head_wt);
    if (!m->lm_head) m->lm_head = m->tok_emb; /* Tied weights fallback */
    m->wtype = head_wt;

    for (uint32_t l = 0; l < n_layers; ++l) {
        RCRE_LlamaLayerWeights* lw = &m->layers[l];
        lw->rms_att = malloc(dim * sizeof(float));
        lw->rms_ffn = malloc(dim * sizeof(float));
        if (!lw->rms_att || !lw->rms_ffn) { rcre_llama_free(m); if (out_err) *out_err = RCRE_ERR_OUT_OF_MEMORY; return nullptr; }

        char nbuf[128];
        snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_norm.weight", l);
        if (!llm_gguf_read_f32_tensor(g, nbuf, (float*)lw->rms_att, dim)) { rcre_llama_free(m); if (out_err) *out_err = RCRE_ERR_IO; return nullptr; }
        snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_norm.weight", l);
        if (!llm_gguf_read_f32_tensor(g, nbuf, (float*)lw->rms_ffn, dim)) { rcre_llama_free(m); if (out_err) *out_err = RCRE_ERR_IO; return nullptr; }

        const char* suffixes[7] = { "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight" };
        void** targets[7] = { &lw->wq, &lw->wk, &lw->wv, &lw->wo, &lw->w_gate, &lw->w_up, &lw->w_down };
        for (int p = 0; p < 7; ++p) {
            RCRE_LLMWeightType wt = RCRE_LLM_W_F32;
            snprintf(nbuf, sizeof(nbuf), "blk.%u.%s", l, suffixes[p]);
            *targets[p] = (void*)(uintptr_t)llm_gguf_bind_weight(g, nbuf, &wt);
            if (!*targets[p]) { rcre_llama_free(m); if (out_err) *out_err = RCRE_ERR_IO; return nullptr; }
            lw->wtype = wt;
        }
    }

    if (out_err) *out_err = RCRE_OK;
    return m;
}

void rcre_llama_free(RCRE_LlamaModel* model) {
    if (!model) return;
    if (model->owns_memory) {
        free(model->tok_emb);
        free(model->rms_final);
        free(model->lm_head);
        if (model->layers) {
            for (size_t l = 0; l < model->cfg.n_layers; ++l) {
                RCRE_LlamaLayerWeights* lw = &model->layers[l];
                free(lw->rms_att);
                free(lw->rms_ffn);
                free(lw->wq); free(lw->wk); free(lw->wv); free(lw->wo);
                free(lw->w_gate); free(lw->w_up); free(lw->w_down);
            }
            free(model->layers);
        }
    } else {
        free(model->tok_emb);
        free(model->rms_final);
        if (model->layers) {
            for (size_t l = 0; l < model->cfg.n_layers; ++l) {
                free(model->layers[l].rms_att);
                free(model->layers[l].rms_ffn);
            }
            free(model->layers);
        }
        if (model->backed_gguf) {
            rcre_gguf_close(model->backed_gguf);
            free(model->backed_gguf);
        }
    }
    free(model);
}

RCRE_NODISCARD RCRE_LlamaState* rcre_llama_state_create(const RCRE_LlamaConfig* cfg) {
    if (!cfg || cfg->dim == 0 || cfg->n_heads == 0) return nullptr;
    size_t kv_dim = (cfg->dim / cfg->n_heads) * cfg->n_kv_heads;
    RCRE_LlamaState* s = (RCRE_LlamaState*)calloc(1, sizeof(RCRE_LlamaState));
    if (!s) return nullptr;
    s->cfg = *cfg;
    s->x           = (float*)calloc(cfg->dim, sizeof(float));
    s->xb          = (float*)calloc(cfg->dim, sizeof(float));
    s->xb2         = (float*)calloc(cfg->dim, sizeof(float));
    s->hb          = (float*)calloc(cfg->hidden_dim, sizeof(float));
    s->hb2         = (float*)calloc(cfg->hidden_dim, sizeof(float));
    s->q           = (float*)calloc(cfg->dim, sizeof(float));
    s->k           = (float*)calloc(kv_dim, sizeof(float));
    s->v           = (float*)calloc(kv_dim, sizeof(float));
    s->att_out     = (float*)calloc(cfg->dim, sizeof(float));
    s->logits      = (float*)calloc(cfg->vocab_size, sizeof(float));
    s->key_cache   = (float*)calloc(cfg->n_layers * cfg->max_seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(cfg->n_layers * cfg->max_seq_len * kv_dim, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v ||
        !s->att_out || !s->logits || !s->key_cache || !s->value_cache) {
        rcre_llama_state_free(s);
        return nullptr;
    }
    return s;
}

void rcre_llama_state_reset(RCRE_LlamaState* state) {
    if (!state) return;
    size_t kv_dim = (state->cfg.dim / state->cfg.n_heads) * state->cfg.n_kv_heads;
    memset(state->key_cache, 0, state->cfg.n_layers * state->cfg.max_seq_len * kv_dim * sizeof(float));
    memset(state->value_cache, 0, state->cfg.n_layers * state->cfg.max_seq_len * kv_dim * sizeof(float));
}

void rcre_llama_state_free(RCRE_LlamaState* state) {
    if (!state) return;
    free(state->x); free(state->xb); free(state->xb2);
    free(state->hb); free(state->hb2);
    free(state->q); free(state->k); free(state->v);
    free(state->att_out); free(state->logits);
    free(state->key_cache); free(state->value_cache);
    free(state);
}

RCRE_NODISCARD int rcre_llama_forward_token(const RCRE_LlamaModel* model, RCRE_LlamaState* state,
                                            int32_t token, size_t pos) {
    if (!model || !state || token < 0 || (size_t)token >= model->cfg.vocab_size ||
        pos >= model->cfg.max_seq_len) {
        return RCRE_ERR_INVALID_ARG;
    }

    const RCRE_LlamaConfig* c = &model->cfg;
    size_t dim = c->dim;
    size_t hd  = dim / c->n_heads;
    size_t kv_dim = hd * c->n_kv_heads;

    memcpy(state->x, model->tok_emb + (size_t)token * dim, dim * sizeof(float));

    for (size_t l = 0; l < c->n_layers; ++l) {
        const RCRE_LlamaLayerWeights* lw = &model->layers[l];
        rcre_llm_rmsnorm(state->xb, state->x, (const float*)lw->rms_att, dim, c->rms_eps);

        rcre_llm_gemv_quant(lw->wtype, lw->wq, state->xb, state->q, dim, dim, model->pool);
        rcre_llm_gemv_quant(lw->wtype, lw->wk, state->xb, state->k, kv_dim, dim, model->pool);
        rcre_llm_gemv_quant(lw->wtype, lw->wv, state->xb, state->v, kv_dim, dim, model->pool);

        rcre_llm_rope(state->q, state->k, c->n_heads, c->n_kv_heads, hd, pos, c->rope_theta);

        size_t loff = l * c->max_seq_len * kv_dim;
        float* k_cache_l = state->key_cache + loff;
        float* v_cache_l = state->value_cache + loff;
        memcpy(k_cache_l + pos * kv_dim, state->k, kv_dim * sizeof(float));
        memcpy(v_cache_l + pos * kv_dim, state->v, kv_dim * sizeof(float));

        rcre_llm_flash_attention_gqa(state->att_out, state->q, k_cache_l, v_cache_l,
                                     c->n_heads, c->n_kv_heads, hd, pos + 1U, kv_dim);

        rcre_llm_gemv_quant(lw->wtype, lw->wo, state->att_out, state->xb2, dim, dim, model->pool);
        for (size_t i = 0; i < dim; ++i) state->x[i] += state->xb2[i];

        /* SwiGLU FFN */
        rcre_llm_rmsnorm(state->xb, state->x, (const float*)lw->rms_ffn, dim, c->rms_eps);
        rcre_llm_gemv_quant(lw->wtype, lw->w_gate, state->xb, state->hb, c->hidden_dim, dim, model->pool);
        rcre_llm_gemv_quant(lw->wtype, lw->w_up, state->xb, state->hb2, c->hidden_dim, dim, model->pool);
        rcre_llm_swiglu(state->hb, state->hb, state->hb2, c->hidden_dim);
        rcre_llm_gemv_quant(lw->wtype, lw->w_down, state->hb, state->xb2, dim, c->hidden_dim, model->pool);
        for (size_t i = 0; i < dim; ++i) state->x[i] += state->xb2[i];
    }

    rcre_llm_rmsnorm(state->xb, state->x, model->rms_final, dim, c->rms_eps);
    rcre_llm_gemv_quant(model->wtype, model->lm_head, state->xb, state->logits, c->vocab_size, dim, model->pool);
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_llama_generate(
    const RCRE_LlamaModel* model,
    RCRE_LlamaState* state,
    RCRE_Sampler* sampler,
    const int32_t* prompt_tokens,
    size_t n_prompt,
    int32_t* out_tokens,
    size_t max_new_tokens,
    size_t* out_generated
) {
    if (!model || !state || !sampler || !prompt_tokens || n_prompt == 0 || !out_tokens) {
        return RCRE_ERR_INVALID_ARG;
    }
    rcre_llama_state_reset(state);

    size_t pos = 0;
    for (size_t i = 0; i < n_prompt && pos < model->cfg.max_seq_len; ++i) {
        int rc = rcre_llama_forward_token(model, state, prompt_tokens[i], pos++);
        if (rc != RCRE_OK) return rc;
    }

    size_t gen = 0;
    int32_t next_tok = rcre_sampler_sample(sampler, state->logits, prompt_tokens, n_prompt);
    while (gen < max_new_tokens && pos < model->cfg.max_seq_len) {
        out_tokens[gen++] = next_tok;
        if (gen == max_new_tokens || pos >= model->cfg.max_seq_len) break;
        int rc = rcre_llama_forward_token(model, state, next_tok, pos++);
        if (rc != RCRE_OK) return rc;
        next_tok = rcre_sampler_sample(sampler, state->logits, out_tokens, gen);
    }

    if (out_generated) *out_generated = gen;
    return RCRE_OK;
}

/* ========================================================================= */
/* 6. Phase 3: LoRA Adapters, Checkpointed/BF16 Training & Shard Stream      */
/* ========================================================================= */

RCRE_NODISCARD RCRE_LoRAAdapter* rcre_lora_create(size_t rank, size_t in_dim, size_t out_dim,
                                                  float alpha, uint64_t seed) {
    if (rank == 0 || in_dim == 0 || out_dim == 0) return nullptr;
    RCRE_LoRAAdapter* l = (RCRE_LoRAAdapter*)calloc(1, sizeof(RCRE_LoRAAdapter));
    if (!l) return nullptr;
    l->rank    = rank;
    l->in_dim  = in_dim;
    l->out_dim = out_dim;
    l->alpha   = alpha;
    l->A       = (float*)calloc(rank * in_dim, sizeof(float));
    l->B       = (float*)calloc(out_dim * rank, sizeof(float)); /* B starts at 0 per LoRA paper */
    l->grad_A  = (float*)calloc(rank * in_dim, sizeof(float));
    l->grad_B  = (float*)calloc(out_dim * rank, sizeof(float));
    l->m_A     = (float*)calloc(rank * in_dim, sizeof(float));
    l->v_A     = (float*)calloc(rank * in_dim, sizeof(float));
    l->m_B     = (float*)calloc(out_dim * rank, sizeof(float));
    l->v_B     = (float*)calloc(out_dim * rank, sizeof(float));
    if (!l->A || !l->B || !l->grad_A || !l->grad_B || !l->m_A || !l->v_A || !l->m_B || !l->v_B) {
        rcre_lora_free(l);
        return nullptr;
    }
    float std = 1.0f / sqrtf((float)in_dim);
    for (size_t i = 0; i < rank * in_dim; ++i) l->A[i] = llm_rand_normal(&seed, std);
    return l;
}

void rcre_lora_free(RCRE_LoRAAdapter* lora) {
    if (!lora) return;
    free(lora->A); free(lora->B);
    free(lora->grad_A); free(lora->grad_B);
    free(lora->m_A); free(lora->v_A);
    free(lora->m_B); free(lora->v_B);
    free(lora);
}

void rcre_lora_forward_add(const RCRE_LoRAAdapter* lora, const float* x, float* y) {
    if (!lora || !x || !y) return;
    float scale = lora->alpha / (float)lora->rank;
    float* h = (float*)malloc(lora->rank * sizeof(float));
    if (!h) return;
    for (size_t r = 0; r < lora->rank; ++r) {
        h[r] = rcre_hal.dot_f32(lora->A + r * lora->in_dim, x, lora->in_dim);
    }
    for (size_t o = 0; o < lora->out_dim; ++o) {
        y[o] += scale * rcre_hal.dot_f32(lora->B + o * lora->rank, h, lora->rank);
    }
    free(h);
}

RCRE_NODISCARD float rcre_lora_train_step(
    RCRE_LoRAAdapter* lora,
    const float* base_W_rowmajor,
    const float* x,
    const float* target_y,
    float lr
) {
    if (!lora || !base_W_rowmajor || !x || !target_y) return -1.0f;
    float* pred = (float*)calloc(lora->out_dim, sizeof(float));
    float* h    = (float*)calloc(lora->rank, sizeof(float));
    float* dy   = (float*)calloc(lora->out_dim, sizeof(float));
    float* dh   = (float*)calloc(lora->rank, sizeof(float));
    if (!pred || !h || !dy || !dh) {
        free(pred); free(h); free(dy); free(dh);
        return -1.0f;
    }

    rcre_hal.gemv_f32(lora->out_dim, lora->in_dim, 1.0f, base_W_rowmajor, lora->in_dim, x, 0.0f, pred);
    float scale = lora->alpha / (float)lora->rank;
    for (size_t r = 0; r < lora->rank; ++r) {
        h[r] = rcre_hal.dot_f32(lora->A + r * lora->in_dim, x, lora->in_dim);
    }
    float loss = 0.0f;
    for (size_t o = 0; o < lora->out_dim; ++o) {
        pred[o] += scale * rcre_hal.dot_f32(lora->B + o * lora->rank, h, lora->rank);
        float diff = pred[o] - target_y[o];
        loss += 0.5f * diff * diff;
        dy[o] = diff;
    }
    loss /= (float)lora->out_dim;

    for (size_t o = 0; o < lora->out_dim; ++o) {
        float g = scale * dy[o] / (float)lora->out_dim;
        for (size_t r = 0; r < lora->rank; ++r) {
            lora->grad_B[o * lora->rank + r] = g * h[r];
            dh[r] += g * lora->B[o * lora->rank + r];
        }
    }
    for (size_t r = 0; r < lora->rank; ++r) {
        for (size_t i = 0; i < lora->in_dim; ++i) {
            lora->grad_A[r * lora->in_dim + i] = dh[r] * x[i];
        }
    }

    lora->step++;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, (float)lora->step);
    float bc2 = 1.0f - powf(b2, (float)lora->step);

    for (size_t i = 0; i < lora->rank * lora->in_dim; ++i) {
        float g = lora->grad_A[i];
        lora->m_A[i] = b1 * lora->m_A[i] + (1.0f - b1) * g;
        lora->v_A[i] = b2 * lora->v_A[i] + (1.0f - b2) * g * g;
        lora->A[i] -= lr * (lora->m_A[i] / bc1) / (sqrtf(lora->v_A[i] / bc2) + eps);
    }
    for (size_t i = 0; i < lora->out_dim * lora->rank; ++i) {
        float g = lora->grad_B[i];
        lora->m_B[i] = b1 * lora->m_B[i] + (1.0f - b1) * g;
        lora->v_B[i] = b2 * lora->v_B[i] + (1.0f - b2) * g * g;
        lora->B[i] -= lr * (lora->m_B[i] / bc1) / (sqrtf(lora->v_B[i] / bc2) + eps);
    }

    free(pred); free(h); free(dy); free(dh);
    return loss;
}

static void llm_rmsnorm_backward(float* dx, float* dw, const float* dout,
                                 const float* x, const float* w, size_t n, float eps) {
    float ss = rcre_hal.dot_f32(x, x, n);
    float inv_rms = 1.0f / sqrtf((ss / (float)n) + eps);
    float inv_rms3_n = (inv_rms * inv_rms * inv_rms) / (float)n;
    float sum_wdoutx = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (dw) dw[i] += dout[i] * x[i] * inv_rms;
        sum_wdoutx += w[i] * dout[i] * x[i];
    }
    for (size_t i = 0; i < n; ++i) {
        dx[i] += (w[i] * dout[i] * inv_rms) - x[i] * sum_wdoutx * inv_rms3_n;
    }
}

RCRE_NODISCARD float rcre_llama_train_step_f32(
    RCRE_LlamaModel* model,
    const int32_t* tokens,
    size_t seq_len,
    float lr,
    bool bf16_activations,
    bool grad_checkpoint
) {
    if (!model || !tokens || seq_len < 2 || model->wtype != RCRE_LLM_W_F32) return -1.0f;
    const RCRE_LlamaConfig* c = &model->cfg;
    size_t T = seq_len - 1U;
    size_t D = c->dim;
    size_t H = c->hidden_dim;
    size_t V = c->vocab_size;

    /* Per-position F32 training buffers with true layer-boundary gradient checkpointing */
    float* layer_in = (float*)calloc((c->n_layers + 1U) * T * D, sizeof(float));
    float* final_x  = (float*)calloc(T * D, sizeof(float));
    float* logits   = (float*)calloc(T * V, sizeof(float));
    float* dlogits  = (float*)calloc(T * V, sizeof(float));
    float* dx_cur   = (float*)calloc(T * D, sizeof(float));
    float* dx_next  = (float*)calloc(T * D, sizeof(float));

    /* Single-layer recompute / checkpoint scratch */
    size_t save_layers = grad_checkpoint ? 1U : c->n_layers;
    float* blk_norm1 = (float*)calloc(save_layers * T * D, sizeof(float));
    float* blk_att   = (float*)calloc(save_layers * T * D, sizeof(float));
    float* blk_res1  = (float*)calloc(save_layers * T * D, sizeof(float));
    float* blk_norm2 = (float*)calloc(save_layers * T * D, sizeof(float));
    float* blk_gate  = (float*)calloc(save_layers * T * H, sizeof(float));
    float* blk_up    = (float*)calloc(save_layers * T * H, sizeof(float));
    float* blk_act   = (float*)calloc(save_layers * T * H, sizeof(float));

    if (!layer_in || !final_x || !logits || !dlogits || !dx_cur || !dx_next ||
        !blk_norm1 || !blk_att || !blk_res1 || !blk_norm2 || !blk_gate || !blk_up || !blk_act) {
        free(layer_in); free(final_x); free(logits); free(dlogits); free(dx_cur); free(dx_next);
        free(blk_norm1); free(blk_att); free(blk_res1); free(blk_norm2);
        free(blk_gate); free(blk_up); free(blk_act);
        return -1.0f;
    }

    for (size_t t = 0; t < T; ++t) {
        int32_t tok = tokens[t];
        if (tok < 0 || (size_t)tok >= V) tok = 0;
        memcpy(layer_in + t * D, model->tok_emb + (size_t)tok * D, D * sizeof(float));
    }

    for (size_t l = 0; l < c->n_layers; ++l) {
        const RCRE_LlamaLayerWeights* lw = &model->layers[l];
        size_t slot = grad_checkpoint ? 0U : l;
        float* cur_in = layer_in + l * T * D;
        float* cur_out = layer_in + (l + 1U) * T * D;
        float* n1 = blk_norm1 + slot * T * D;
        float* at = blk_att   + slot * T * D;
        float* r1 = blk_res1  + slot * T * D;
        float* n2 = blk_norm2 + slot * T * D;
        float* gt = blk_gate  + slot * T * H;
        float* up = blk_up    + slot * T * H;
        float* ac = blk_act   + slot * T * H;

        for (size_t t = 0; t < T; ++t) {
            rcre_llm_rmsnorm(n1 + t * D, cur_in + t * D, (const float*)lw->rms_att, D, c->rms_eps);
            llm_maybe_bf16_roundtrip(n1 + t * D, D, bf16_activations);
            rcre_hal.gemv_f32(D, D, 1.0f, (const float*)lw->wo, D, n1 + t * D, 0.0f, at + t * D);
            for (size_t i = 0; i < D; ++i) r1[t * D + i] = cur_in[t * D + i] + at[t * D + i];

            rcre_llm_rmsnorm(n2 + t * D, r1 + t * D, (const float*)lw->rms_ffn, D, c->rms_eps);
            llm_maybe_bf16_roundtrip(n2 + t * D, D, bf16_activations);
            rcre_hal.gemv_f32(H, D, 1.0f, (const float*)lw->w_gate, D, n2 + t * D, 0.0f, gt + t * H);
            rcre_hal.gemv_f32(H, D, 1.0f, (const float*)lw->w_up, D, n2 + t * D, 0.0f, up + t * H);
            rcre_llm_swiglu(ac + t * H, gt + t * H, up + t * H, H);
            rcre_hal.gemv_f32(D, H, 1.0f, (const float*)lw->w_down, H, ac + t * H, 0.0f, cur_out + t * D);
            for (size_t i = 0; i < D; ++i) cur_out[t * D + i] += r1[t * D + i];
        }
    }

    float total_loss = 0.0f;
    float* head_w = (float*)model->lm_head;
    for (size_t t = 0; t < T; ++t) {
        rcre_llm_rmsnorm(final_x + t * D, layer_in + c->n_layers * T * D + t * D,
                         model->rms_final, D, c->rms_eps);
        rcre_hal.gemv_f32(V, D, 1.0f, head_w, D, final_x + t * D, 0.0f, logits + t * V);

        float* lt = logits + t * V;
        float* dl = dlogits + t * V;
        float mx = lt[0];
        for (size_t v = 1; v < V; ++v) if (lt[v] > mx) mx = lt[v];
        float s = 0.0f;
        for (size_t v = 0; v < V; ++v) {
            dl[v] = expf(lt[v] - mx);
            s += dl[v];
        }
        int32_t tgt = tokens[t + 1U];
        if (tgt < 0 || (size_t)tgt >= V) tgt = 0;
        for (size_t v = 0; v < V; ++v) dl[v] /= s;
        total_loss += -logf(dl[tgt] + 1e-12f);
        dl[tgt] -= 1.0f;
        for (size_t v = 0; v < V; ++v) dl[v] /= (float)T;
    }
    total_loss /= (float)T;

    /* Head backward */
    float* d_final_w = (float*)calloc(D, sizeof(float));
    float* dfx = (float*)calloc(T * D, sizeof(float));
    if (d_final_w && dfx) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t v = 0; v < V; ++v) {
                float gv = dlogits[t * V + v];
                for (size_t d = 0; d < D; ++d) {
                    dfx[t * D + d] += gv * head_w[v * D + d];
                    head_w[v * D + d] -= lr * gv * final_x[t * D + d];
                }
            }
            llm_rmsnorm_backward(dx_cur + t * D, d_final_w, dfx + t * D,
                                 layer_in + c->n_layers * T * D + t * D,
                                 model->rms_final, D, c->rms_eps);
        }
        for (size_t d = 0; d < D; ++d) model->rms_final[d] -= lr * d_final_w[d];
    }
    free(d_final_w);
    free(dfx);

    /* Backward across transformer blocks (recomputing block activations if checkpointed) */
    float* d_act  = (float*)calloc(H, sizeof(float));
    float* d_gate = (float*)calloc(H, sizeof(float));
    float* d_up   = (float*)calloc(H, sizeof(float));
    float* d_n2   = (float*)calloc(D, sizeof(float));
    float* d_n1   = (float*)calloc(D, sizeof(float));

    for (size_t li = c->n_layers; li > 0; --li) {
        size_t l = li - 1U;
        RCRE_LlamaLayerWeights* lw = &model->layers[l];
        size_t slot = grad_checkpoint ? 0U : l;
        float* cur_in = layer_in + l * T * D;
        float* n1 = blk_norm1 + slot * T * D;
        float* at = blk_att   + slot * T * D;
        float* r1 = blk_res1  + slot * T * D;
        float* n2 = blk_norm2 + slot * T * D;
        float* gt = blk_gate  + slot * T * H;
        float* up = blk_up    + slot * T * H;
        float* ac = blk_act   + slot * T * H;

        if (grad_checkpoint) {
            for (size_t t = 0; t < T; ++t) {
                rcre_llm_rmsnorm(n1 + t * D, cur_in + t * D, (const float*)lw->rms_att, D, c->rms_eps);
                llm_maybe_bf16_roundtrip(n1 + t * D, D, bf16_activations);
                rcre_hal.gemv_f32(D, D, 1.0f, (const float*)lw->wo, D, n1 + t * D, 0.0f, at + t * D);
                for (size_t i = 0; i < D; ++i) r1[t * D + i] = cur_in[t * D + i] + at[t * D + i];
                rcre_llm_rmsnorm(n2 + t * D, r1 + t * D, (const float*)lw->rms_ffn, D, c->rms_eps);
                llm_maybe_bf16_roundtrip(n2 + t * D, D, bf16_activations);
                rcre_hal.gemv_f32(H, D, 1.0f, (const float*)lw->w_gate, D, n2 + t * D, 0.0f, gt + t * H);
                rcre_hal.gemv_f32(H, D, 1.0f, (const float*)lw->w_up, D, n2 + t * D, 0.0f, up + t * H);
                rcre_llm_swiglu(ac + t * H, gt + t * H, up + t * H, H);
            }
        }

        memset(dx_next, 0, T * D * sizeof(float));
        float* wo = (float*)lw->wo;
        float* wg = (float*)lw->w_gate;
        float* wu = (float*)lw->w_up;
        float* wd = (float*)lw->w_down;
        float* ra = (float*)lw->rms_att;
        float* rf = (float*)lw->rms_ffn;

        for (size_t t = 0; t < T; ++t) {
            const float* dy = dx_cur + t * D;
            memset(d_act, 0, H * sizeof(float));
            memset(d_n2, 0, D * sizeof(float));
            memset(d_n1, 0, D * sizeof(float));

            for (size_t d = 0; d < D; ++d) {
                float gd = dy[d];
                for (size_t h = 0; h < H; ++h) {
                    d_act[h] += gd * wd[d * H + h];
                    wd[d * H + h] -= lr * gd * ac[t * H + h];
                }
            }
            for (size_t h = 0; h < H; ++h) {
                float g = gt[t * H + h];
                float sig = 1.0f / (1.0f + expf(-g));
                float silu = g * sig;
                float dsilu = sig * (1.0f + g * (1.0f - sig));
                d_up[h]   = d_act[h] * silu;
                d_gate[h] = d_act[h] * up[t * H + h] * dsilu;
            }
            for (size_t h = 0; h < H; ++h) {
                float dg = d_gate[h], du = d_up[h];
                for (size_t d = 0; d < D; ++d) {
                    d_n2[d] += dg * wg[h * D + d] + du * wu[h * D + d];
                    wg[h * D + d] -= lr * dg * n2[t * D + d];
                    wu[h * D + d] -= lr * du * n2[t * D + d];
                }
            }
            /* dr1 accumulates residual path + FFN norm backward */
            memcpy(dx_next + t * D, dy, D * sizeof(float));
            llm_rmsnorm_backward(dx_next + t * D, nullptr, d_n2, r1 + t * D, rf, D, c->rms_eps);

            for (size_t d = 0; d < D; ++d) {
                float dr = dx_next[t * D + d];
                for (size_t c2 = 0; c2 < D; ++c2) {
                    d_n1[c2] += dr * wo[d * D + c2];
                    wo[d * D + c2] -= lr * dr * n1[t * D + c2];
                }
            }
            llm_rmsnorm_backward(dx_next + t * D, nullptr, d_n1, cur_in + t * D, ra, D, c->rms_eps);
        }
        memcpy(dx_cur, dx_next, T * D * sizeof(float));
    }

    for (size_t t = 0; t < T; ++t) {
        int32_t tok = tokens[t];
        if (tok >= 0 && (size_t)tok < V) {
            for (size_t d = 0; d < D; ++d) {
                model->tok_emb[(size_t)tok * D + d] -= lr * dx_cur[t * D + d];
            }
        }
    }

    free(d_act); free(d_gate); free(d_up); free(d_n2); free(d_n1);
    free(layer_in); free(final_x); free(logits); free(dlogits); free(dx_cur); free(dx_next);
    free(blk_norm1); free(blk_att); free(blk_res1); free(blk_norm2);
    free(blk_gate); free(blk_up); free(blk_act);
    return total_loss;
}

RCRE_NODISCARD int rcre_shard_stream_open(RCRE_ShardStream* stream, const char** shard_paths,
                                          size_t num_shards, size_t seq_len) {
    if (!stream || !shard_paths || num_shards == 0 || seq_len == 0) return RCRE_ERR_INVALID_ARG;
    memset(stream, 0, sizeof(*stream));
    stream->shard_paths = shard_paths;
    stream->num_shards  = num_shards;
    stream->seq_len     = seq_len;
    return rcre_io_mmap_open(shard_paths[0], RCRE_IO_HINT_SEQUENTIAL, &stream->active_map);
}

RCRE_NODISCARD bool rcre_shard_stream_next(RCRE_ShardStream* stream, int32_t* x_out, int32_t* y_out) {
    if (!stream || !x_out || !y_out) return false;
    size_t need_bytes = (stream->seq_len + 1U) * sizeof(int32_t);
    for (size_t attempts = 0; attempts < stream->num_shards; ++attempts) {
        size_t total_tokens = stream->active_map.size / sizeof(int32_t);
        if (stream->active_map.data && stream->token_cursor + stream->seq_len + 1U <= total_tokens) {
            const int32_t* toks = (const int32_t*)(const void*)stream->active_map.data;
            memcpy(x_out, toks + stream->token_cursor, stream->seq_len * sizeof(int32_t));
            memcpy(y_out, toks + stream->token_cursor + 1U, stream->seq_len * sizeof(int32_t));
            stream->token_cursor += stream->seq_len;
            return true;
        }
        rcre_io_mmap_close(&stream->active_map);
        stream->current_shard++;
        if (stream->current_shard >= stream->num_shards) return false;
        stream->token_cursor = 0;
        if (rcre_io_mmap_open(stream->shard_paths[stream->current_shard],
                              RCRE_IO_HINT_SEQUENTIAL, &stream->active_map) != RCRE_OK ||
            stream->active_map.size < need_bytes) {
            return false;
        }
    }
    return false;
}

void rcre_shard_stream_close(RCRE_ShardStream* stream) {
    if (!stream) return;
    rcre_io_mmap_close(&stream->active_map);
    memset(stream, 0, sizeof(*stream));
}
