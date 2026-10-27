/**
 * @file kv_cache.c
 * @brief RCRE Key-Value (KV) Cache Engine for O(1) Autoregressive Generation (ISO C23)
 */

#include "transformer_internal.h"
#ifdef RCRE_ENABLE_PROFILING
#include "rcre_prof.h"
#endif

RCRE_KVCache* rcre_kv_cache_create(RCRE_GPTConfig config, RCRE_Arena *arena) {
    /* The cache layout must match what the attention kernels index with
     * (d_k = embed_dim / num_heads); a mismatched head_dim here would cause
     * out-of-bounds cache writes during forward_token. */
    if (config.num_layers <= 0 || config.num_heads <= 0 || config.seq_len <= 0 ||
        config.embed_dim <= 0 || (config.embed_dim % config.num_heads) != 0 ||
        config.head_dim != config.embed_dim / config.num_heads) {
        return nullptr;
    }
    bool owns_heap = (arena == nullptr);
    RCRE_KVCache *cache = (RCRE_KVCache*)rcre_transformer_alloc_mem(arena, sizeof(RCRE_KVCache), alignof(RCRE_KVCache));
    if (!cache) return nullptr;

    int L = config.num_layers;
    int H = config.num_heads;
    int max_T = config.seq_len;
    int d_k = config.head_dim;
    size_t size = (size_t)L * H * max_T * d_k;

    cache->k = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * size, 64);
    cache->v = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * size, 64);
    cache->max_seq_len = max_T;
    cache->current_pos = 0;
    cache->arena = arena;
    cache->owns_heap = owns_heap;

    if (!cache->k || !cache->v) {
        rcre_kv_cache_free(cache);
        return nullptr;
    }
    memset(cache->k, 0, sizeof(float) * size);
    memset(cache->v, 0, sizeof(float) * size);
    return cache;
}

void rcre_kv_cache_reset(RCRE_KVCache *cache) {
    if (cache) {
        cache->current_pos = 0;
    }
}

void rcre_kv_cache_free(RCRE_KVCache *cache) {
    if (!cache) return;
    if (cache->owns_heap) {
        if (cache->k) free(cache->k);
        if (cache->v) free(cache->v);
        free(cache);
    }
}

bool rcre_gpt_forward_token(
    RCRE_GPTModel *model,
    RCRE_KVCache *cache,
    int token,
    int pos,
    float *out_logits
) {
    if (!model || !cache || !out_logits) return false;
#ifdef RCRE_ENABLE_PROFILING
    RCRE_ProfScope prof_z = rcre_prof_scope_begin(rcre_prof_global(), "gpt_forward_token");
#endif
    const RCRE_GPTConfig *cfg = &model->config;
    int C = cfg->embed_dim;
    int H = cfg->num_heads;
    int L = cfg->num_layers;
    int M = cfg->mlp_dim;
    int V = cfg->vocab_size;
    int max_T = cfg->seq_len;
    int d_k = cfg->head_dim;
    float scale = 1.0f / sqrtf((float)d_k);

    if (pos < 0 || pos >= max_T || token < 0 || token >= V) {
#ifdef RCRE_ENABLE_PROFILING
        rcre_prof_scope_end(&prof_z);
#endif
        return false;
    }

    /*
     * Local buffers via the scratch pool: arena-resident models bump the
     * model's arena (rolled back per token); heap models malloc with checks.
     */
    GPTScratch scratch;
    gpt_scratch_begin(&scratch, model->arena);
    float *x        = gpt_scratch_alloc(&scratch, sizeof(float) * C);
    float *x_norm   = gpt_scratch_alloc(&scratch, sizeof(float) * C);
    float *q        = gpt_scratch_alloc(&scratch, sizeof(float) * C);
    float *k        = gpt_scratch_alloc(&scratch, sizeof(float) * C);
    float *v        = gpt_scratch_alloc(&scratch, sizeof(float) * C);
    float *att_out  = gpt_scratch_alloc(&scratch, sizeof(float) * C);
    float *proj_out = gpt_scratch_alloc(&scratch, sizeof(float) * C);
    float *fc1_out  = gpt_scratch_alloc(&scratch, sizeof(float) * M);
    float *gelu_out = gpt_scratch_alloc(&scratch, sizeof(float) * M);
    float *fc2_out  = gpt_scratch_alloc(&scratch, sizeof(float) * C);
    float *scores   = gpt_scratch_alloc(&scratch, sizeof(float) * (size_t)(pos + 1));
    if (!x || !x_norm || !q || !k || !v || !att_out || !proj_out ||
        !fc1_out || !gelu_out || !fc2_out || !scores) {
        gpt_scratch_end(&scratch);
#ifdef RCRE_ENABLE_PROFILING
        rcre_prof_scope_end(&prof_z);
#endif
        return false;
    }
    float mean = 0.0f, rstd = 0.0f;

    /* 1. Embedding: wte[token] + wpe[pos] */
    const float *wte_row = model->wte + token * C;
    const float *wpe_row = model->wpe + pos * C;
    for (int c = 0; c < C; c++) {
        x[c] = wte_row[c] + wpe_row[c];
    }

    /* 2. Transformer Blocks */
    for (int l = 0; l < L; l++) {
        RCRE_GPTBlockWeights *w = &model->blocks[l];

        rcre_transformer_layernorm_forward(x_norm, &mean, &rstd, x, w->ln1_w, w->ln1_b, 1, C);

        rcre_transformer_linear_forward(q, x_norm, w->q_w, w->q_b, 1, C, C);
        rcre_transformer_linear_forward(k, x_norm, w->k_w, w->k_b, 1, C, C);
        rcre_transformer_linear_forward(v, x_norm, w->v_w, w->v_b, 1, C, C);

        /* Store k and v into KV-cache at position pos */
        for (int h = 0; h < H; h++) {
            size_t slot_idx = (size_t)l * (H * max_T * d_k) + (size_t)h * (max_T * d_k) + (size_t)pos * d_k;
            memcpy(cache->k + slot_idx, k + h * d_k, sizeof(float) * d_k);
            memcpy(cache->v + slot_idx, v + h * d_k, sizeof(float) * d_k);
        }

        /* Attention over cached keys and values */
        for (int h = 0; h < H; h++) {
            const float *q_head = q + h * d_k;
            float *out_head = att_out + h * d_k;

            float max_s = -1e9f;
            for (int s = 0; s <= pos; s++) {
                size_t k_idx = (size_t)l * (H * max_T * d_k) + (size_t)h * (max_T * d_k) + (size_t)s * d_k;
                const float *k_head = cache->k + k_idx;
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    dot += q_head[d] * k_head[d];
                }
                dot *= scale;
                scores[s] = dot;
                if (dot > max_s) max_s = dot;
            }

            float sum = 0.0f;
            for (int s = 0; s <= pos; s++) {
                float e = expf(scores[s] - max_s);
                scores[s] = e;
                sum += e;
            }
            float inv_sum = 1.0f / sum;

            for (int d = 0; d < d_k; d++) {
                float val = 0.0f;
                for (int s = 0; s <= pos; s++) {
                    size_t v_idx = (size_t)l * (H * max_T * d_k) + (size_t)h * (max_T * d_k) + (size_t)s * d_k;
                    val += (scores[s] * inv_sum) * cache->v[v_idx + d];
                }
                out_head[d] = val;
            }
        }

        rcre_transformer_linear_forward(proj_out, att_out, w->proj_w, w->proj_b, 1, C, C);

        for (int c = 0; c < C; c++) {
            x[c] += proj_out[c];
        }

        rcre_transformer_layernorm_forward(x_norm, &mean, &rstd, x, w->ln2_w, w->ln2_b, 1, C);

        rcre_transformer_linear_forward(fc1_out, x_norm, w->fc1_w, w->fc1_b, 1, C, M);
        rcre_transformer_gelu_forward(gelu_out, fc1_out, M);
        rcre_transformer_linear_forward(fc2_out, gelu_out, w->fc2_w, w->fc2_b, 1, M, C);

        for (int c = 0; c < C; c++) {
            x[c] += fc2_out[c];
        }
    }

    /* 3. Final LayerNorm */
    rcre_transformer_layernorm_forward(x_norm, &mean, &rstd, x, model->ln_f_w, model->ln_f_b, 1, C);

    /* 4. LM Head Projection */
    rcre_transformer_linear_forward(out_logits, x_norm, model->head_w, nullptr, 1, C, V);

    gpt_scratch_end(&scratch);
    cache->current_pos = pos + 1;
#ifdef RCRE_ENABLE_PROFILING
    rcre_prof_scope_end(&prof_z);
#endif
    return true;
}
