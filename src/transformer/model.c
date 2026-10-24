/**
 * @file model.c
 * @brief RCRE GPT Model Architecture, Lifecycle, Forward & Backward Passes (ISO C23)
 */

#include "transformer_internal.h"

static void map_block_pointers(RCRE_GPTBlockWeights *block, float *base, const RCRE_GPTConfig *cfg, size_t *offset) {
    int C = cfg->embed_dim;
    int M = cfg->mlp_dim;

    block->ln1_w = base + *offset; *offset += C;
    block->ln1_b = base + *offset; *offset += C;

    block->q_w = base + *offset; *offset += C * C;
    block->q_b = base + *offset; *offset += C;

    block->k_w = base + *offset; *offset += C * C;
    block->k_b = base + *offset; *offset += C;

    block->v_w = base + *offset; *offset += C * C;
    block->v_b = base + *offset; *offset += C;

    block->proj_w = base + *offset; *offset += C * C;
    block->proj_b = base + *offset; *offset += C;

    block->ln2_w = base + *offset; *offset += C;
    block->ln2_b = base + *offset; *offset += C;

    block->fc1_w = base + *offset; *offset += C * M;
    block->fc1_b = base + *offset; *offset += M;

    block->fc2_w = base + *offset; *offset += M * C;
    block->fc2_b = base + *offset; *offset += C;
}

RCRE_GPTModel* rcre_gpt_create(RCRE_GPTConfig config, RCRE_Arena *arena) {
    bool owns_heap = (arena == nullptr);
    RCRE_GPTModel *model = (RCRE_GPTModel*)rcre_transformer_alloc_mem(arena, sizeof(RCRE_GPTModel), alignof(RCRE_GPTModel));
    if (!model) return nullptr;

    model->config = config;
    model->arena = arena;
    model->owns_heap = owns_heap;

    int V = config.vocab_size;
    int T = config.seq_len;
    int C = config.embed_dim;
    int L = config.num_layers;
    int M = config.mlp_dim;

    /* Calculate total parameter count */
    size_t count = 0;
    count += (size_t)V * C; /* wte */
    count += (size_t)T * C; /* wpe */
    size_t block_params = (size_t)(2 * C) +
                          (size_t)(C * C + C) * 3 +
                          (size_t)(C * C + C) +
                          (size_t)(2 * C) +
                          (size_t)(C * M + M) +
                          (size_t)(M * C + C);
    count += (size_t)L * block_params;
    count += (size_t)(2 * C); /* ln_f */
    count += (size_t)(C * V); /* head_w */

    model->num_params = count;

    /* Aligned allocations for parameters, gradients, and optimizer state */
    model->params = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * count, 64);
    model->grads  = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * count, 64);
    model->m      = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * count, 64);
    model->v      = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * count, 64);

    if (!model->params || !model->grads || !model->m || !model->v) {
        rcre_gpt_free(model);
        return nullptr;
    }

    memset(model->grads, 0, sizeof(float) * count);
    memset(model->m, 0, sizeof(float) * count);
    memset(model->v, 0, sizeof(float) * count);

    /* Map flat parameter buffer into structured pointers */
    size_t offset = 0;
    model->wte = model->params + offset; offset += (size_t)V * C;
    model->wpe = model->params + offset; offset += (size_t)T * C;

    model->blocks = (RCRE_GPTBlockWeights*)rcre_transformer_alloc_mem(arena, sizeof(RCRE_GPTBlockWeights) * L, alignof(RCRE_GPTBlockWeights));
    if (!model->blocks) {
        rcre_gpt_free(model);
        return nullptr;
    }

    for (int l = 0; l < L; l++) {
        map_block_pointers(&model->blocks[l], model->params, &config, &offset);
    }

    model->ln_f_w = model->params + offset; offset += C;
    model->ln_f_b = model->params + offset; offset += C;
    model->head_w = model->params + offset; offset += (size_t)C * V;

    assert(offset == count);
    return model;
}

void rcre_gpt_init_weights(RCRE_GPTModel *model, uint64_t *rng_state) {
    uint64_t local_state = 1337ULL;
    uint64_t *rng = rng_state ? rng_state : &local_state;
    float std = 0.02f;

    memset(model->params, 0, sizeof(float) * model->num_params);

    /* 1. Embeddings */
    for (size_t i = 0; i < (size_t)model->config.vocab_size * model->config.embed_dim; i++) {
        model->wte[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
    }
    for (size_t i = 0; i < (size_t)model->config.seq_len * model->config.embed_dim; i++) {
        model->wpe[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
    }

    /* 2. Transformer Blocks */
    int C = model->config.embed_dim;
    int M = model->config.mlp_dim;

    for (int l = 0; l < model->config.num_layers; l++) {
        RCRE_GPTBlockWeights *b = &model->blocks[l];
        for (int i = 0; i < C; i++) {
            b->ln1_w[i] = 1.0f;
            b->ln1_b[i] = 0.0f;
            b->ln2_w[i] = 1.0f;
            b->ln2_b[i] = 0.0f;
        }
        for (int i = 0; i < C * C; i++) {
            b->q_w[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
            b->k_w[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
            b->v_w[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
            b->proj_w[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
        }
        for (int i = 0; i < C * M; i++) b->fc1_w[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
        for (int i = 0; i < M * C; i++) b->fc2_w[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
    }

    /* 3. Final LayerNorm & LM Head */
    for (int i = 0; i < C; i++) {
        model->ln_f_w[i] = 1.0f;
        model->ln_f_b[i] = 0.0f;
    }
    for (size_t i = 0; i < (size_t)C * model->config.vocab_size; i++) {
        model->head_w[i] = rcre_rng_gaussian_f32(rng, 0.0f, std);
    }
}

void rcre_gpt_zero_grads(RCRE_GPTModel *model) {
    if (model && model->grads) {
        memset(model->grads, 0, sizeof(float) * model->num_params);
    }
}

void rcre_gpt_free(RCRE_GPTModel *model) {
    if (!model) return;
    if (model->owns_heap) {
        if (model->params) free(model->params);
        if (model->grads)  free(model->grads);
        if (model->m)      free(model->m);
        if (model->v)      free(model->v);
        if (model->blocks) free(model->blocks);
        free(model);
    }
}

RCRE_GPTActs* rcre_gpt_acts_create(RCRE_GPTConfig config, int B, int T, RCRE_Arena *arena) {
    bool owns_heap = (arena == nullptr);
    RCRE_GPTActs *acts = (RCRE_GPTActs*)rcre_transformer_alloc_mem(arena, sizeof(RCRE_GPTActs), alignof(RCRE_GPTActs));
    if (!acts) return nullptr;

    acts->arena = arena;
    acts->owns_heap = owns_heap;

    int C = config.embed_dim;
    int H = config.num_heads;
    int L = config.num_layers;
    int M = config.mlp_dim;
    int V = config.vocab_size;
    int BT = B * T;

    acts->emb_out   = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
    acts->ln_f_out  = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
    acts->ln_f_mean = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT, 64);
    acts->ln_f_rstd = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT, 64);
    acts->logits    = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * V, 64);
    acts->probs     = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * V, 64);

    acts->blocks = (RCRE_GPTBlockActs*)rcre_transformer_alloc_mem(arena, sizeof(RCRE_GPTBlockActs) * L, alignof(RCRE_GPTBlockActs));
    if (!acts->blocks) {
        rcre_gpt_acts_free(acts, L);
        return nullptr;
    }

    for (int l = 0; l < L; l++) {
        RCRE_GPTBlockActs *ba = &acts->blocks[l];
        ba->ln1_out    = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
        ba->ln1_mean   = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT, 64);
        ba->ln1_rstd   = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT, 64);
        ba->q          = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
        ba->k          = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
        ba->v          = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
        ba->att_scores = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * B * H * T * T, 64);
        ba->att_out    = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
        ba->proj_out   = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
        ba->resid1_out = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);

        ba->ln2_out    = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
        ba->ln2_mean   = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT, 64);
        ba->ln2_rstd   = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT, 64);
        ba->fc1_out    = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * M, 64);
        ba->gelu_out   = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * M, 64);
        ba->fc2_out    = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
        ba->resid2_out = (float*)rcre_transformer_alloc_mem(arena, sizeof(float) * BT * C, 64);
    }
    return acts;
}

void rcre_gpt_acts_free(RCRE_GPTActs *acts, int num_layers) {
    if (!acts) return;
    if (acts->owns_heap) {
        if (acts->emb_out)   free(acts->emb_out);
        if (acts->ln_f_out)  free(acts->ln_f_out);
        if (acts->ln_f_mean) free(acts->ln_f_mean);
        if (acts->ln_f_rstd) free(acts->ln_f_rstd);
        if (acts->logits)    free(acts->logits);
        if (acts->probs)     free(acts->probs);

        if (acts->blocks) {
            for (int l = 0; l < num_layers; l++) {
                RCRE_GPTBlockActs *ba = &acts->blocks[l];
                if (ba->ln1_out)    free(ba->ln1_out);
                if (ba->ln1_mean)   free(ba->ln1_mean);
                if (ba->ln1_rstd)   free(ba->ln1_rstd);
                if (ba->q)          free(ba->q);
                if (ba->k)          free(ba->k);
                if (ba->v)          free(ba->v);
                if (ba->att_scores) free(ba->att_scores);
                if (ba->att_out)    free(ba->att_out);
                if (ba->proj_out)   free(ba->proj_out);
                if (ba->resid1_out) free(ba->resid1_out);
                if (ba->ln2_out)    free(ba->ln2_out);
                if (ba->ln2_mean)   free(ba->ln2_mean);
                if (ba->ln2_rstd)   free(ba->ln2_rstd);
                if (ba->fc1_out)    free(ba->fc1_out);
                if (ba->gelu_out)   free(ba->gelu_out);
                if (ba->fc2_out)    free(ba->fc2_out);
                if (ba->resid2_out) free(ba->resid2_out);
            }
            free(acts->blocks);
        }
        free(acts);
    }
}

float rcre_gpt_forward(RCRE_GPTModel *model, RCRE_GPTActs *acts, const int *inputs, const int *targets, int B, int T) {
    const RCRE_GPTConfig *cfg = &model->config;
    int C = cfg->embed_dim;
    int H = cfg->num_heads;
    int L = cfg->num_layers;
    int M = cfg->mlp_dim;
    int V = cfg->vocab_size;
    int BT = B * T;

    /* 1. Embedding: emb_out = wte[token] + wpe[t] */
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token = inputs[b * T + t];
            assert(token >= 0 && token < V);
            const float *wte_row = model->wte + token * C;
            const float *wpe_row = model->wpe + t * C;
            float *out_row = acts->emb_out + IDX3(b, t, 0, T, C);
            for (int c = 0; c < C; c++) {
                out_row[c] = wte_row[c] + wpe_row[c];
            }
        }
    }

    /* 2. Transformer Blocks */
    const float *curr_input = acts->emb_out;

    for (int l = 0; l < L; l++) {
        RCRE_GPTBlockWeights *w = &model->blocks[l];
        RCRE_GPTBlockActs *a = &acts->blocks[l];

        rcre_transformer_layernorm_forward(a->ln1_out, a->ln1_mean, a->ln1_rstd, curr_input, w->ln1_w, w->ln1_b, BT, C);

        rcre_transformer_linear_forward(a->q, a->ln1_out, w->q_w, w->q_b, BT, C, C);
        rcre_transformer_linear_forward(a->k, a->ln1_out, w->k_w, w->k_b, BT, C, C);
        rcre_transformer_linear_forward(a->v, a->ln1_out, w->v_w, w->v_b, BT, C, C);

        rcre_transformer_attention_forward(a->att_out, a->att_scores, a->q, a->k, a->v, B, T, C, H);

        rcre_transformer_linear_forward(a->proj_out, a->att_out, w->proj_w, w->proj_b, BT, C, C);

        for (int i = 0; i < BT * C; i++) {
            a->resid1_out[i] = curr_input[i] + a->proj_out[i];
        }

        rcre_transformer_layernorm_forward(a->ln2_out, a->ln2_mean, a->ln2_rstd, a->resid1_out, w->ln2_w, w->ln2_b, BT, C);

        rcre_transformer_linear_forward(a->fc1_out, a->ln2_out, w->fc1_w, w->fc1_b, BT, C, M);
        rcre_transformer_gelu_forward(a->gelu_out, a->fc1_out, BT * M);
        rcre_transformer_linear_forward(a->fc2_out, a->gelu_out, w->fc2_w, w->fc2_b, BT, M, C);

        for (int i = 0; i < BT * C; i++) {
            a->resid2_out[i] = a->resid1_out[i] + a->fc2_out[i];
        }

        curr_input = a->resid2_out;
    }

    /* 3. Final LayerNorm */
    rcre_transformer_layernorm_forward(acts->ln_f_out, acts->ln_f_mean, acts->ln_f_rstd, curr_input, model->ln_f_w, model->ln_f_b, BT, C);

    /* 4. LM Head Projection */
    rcre_transformer_linear_forward(acts->logits, acts->ln_f_out, model->head_w, nullptr, BT, C, V);

    /* 5. Loss */
    if (targets != nullptr) {
        return rcre_transformer_cross_entropy_forward(acts->probs, acts->logits, targets, BT, V);
    }
    return 0.0f;
}

void rcre_gpt_backward(RCRE_GPTModel *model, RCRE_GPTActs *acts, const int *inputs, const int *targets, int B, int T) {
    const RCRE_GPTConfig *cfg = &model->config;
    int C = cfg->embed_dim;
    int H = cfg->num_heads;
    int L = cfg->num_layers;
    int M = cfg->mlp_dim;
    int V = cfg->vocab_size;
    int BT = B * T;

    /* Temporary scratch buffers for backpropagation */
    float *d_logits = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * V, 64);
    float *d_ln_f   = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * C, 64);
    float *d_x      = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * C, 64);
    float *d_ln_out = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * C, 64);
    float *d_fc1    = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * M, 64);
    float *d_gelu   = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * M, 64);
    float *d_att    = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * C, 64);
    float *dq       = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * C, 64);
    float *dk       = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * C, 64);
    float *dv       = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * C, 64);
    float *datt_sc  = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * B * H * T * T, 64);
    float *d_qkv_in = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * C, 64);
    float *d_resid1 = (float*)rcre_transformer_alloc_mem(nullptr, sizeof(float) * BT * C, 64);

    size_t offset = 0;
    float *d_wte = model->grads + offset; offset += (size_t)V * C;
    float *d_wpe = model->grads + offset; offset += (size_t)T * C;

    RCRE_GPTBlockWeights d_blocks[L];
    for (int l = 0; l < L; l++) {
        map_block_pointers(&d_blocks[l], model->grads, cfg, &offset);
    }

    float *d_ln_f_w = model->grads + offset; offset += C;
    float *d_ln_f_b = model->grads + offset; offset += C;
    float *d_head_w = model->grads + offset; offset += (size_t)C * V;

    /* 1. Loss Gradient */
    rcre_transformer_cross_entropy_backward(d_logits, acts->probs, targets, BT, V);

    /* 2. LM Head Gradient */
    rcre_transformer_linear_backward(d_ln_f, d_head_w, nullptr, d_logits, acts->ln_f_out, model->head_w, BT, C, V);

    /* 3. Final LayerNorm Backward */
    const float *last_resid = acts->blocks[L - 1].resid2_out;
    rcre_transformer_layernorm_backward(d_x, d_ln_f_w, d_ln_f_b, d_ln_f, last_resid, acts->ln_f_mean, acts->ln_f_rstd, model->ln_f_w, BT, C);

    /* 4. Blocks Backward */
    for (int l = L - 1; l >= 0; l--) {
        RCRE_GPTBlockWeights *w = &model->blocks[l];
        RCRE_GPTBlockWeights *dw = &d_blocks[l];
        RCRE_GPTBlockActs *a = &acts->blocks[l];

        rcre_transformer_linear_backward(d_gelu, dw->fc2_w, dw->fc2_b, d_x, a->gelu_out, w->fc2_w, BT, M, C);
        rcre_transformer_gelu_backward(d_fc1, a->fc1_out, d_gelu, BT * M);
        rcre_transformer_linear_backward(d_ln_out, dw->fc1_w, dw->fc1_b, d_fc1, a->ln2_out, w->fc1_w, BT, C, M);

        rcre_transformer_layernorm_backward(d_resid1, dw->ln2_w, dw->ln2_b, d_ln_out, a->resid1_out, a->ln2_mean, a->ln2_rstd, w->ln2_w, BT, C);

        for (int i = 0; i < BT * C; i++) {
            d_resid1[i] += d_x[i];
        }

        rcre_transformer_linear_backward(d_att, dw->proj_w, dw->proj_b, d_resid1, a->att_out, w->proj_w, BT, C, C);
        rcre_transformer_attention_backward(dq, dk, dv, datt_sc, d_att, a->att_scores, a->q, a->k, a->v, B, T, C, H);

        memset(d_qkv_in, 0, sizeof(float) * BT * C);
        rcre_transformer_linear_backward(d_ln_out, dw->q_w, dw->q_b, dq, a->ln1_out, w->q_w, BT, C, C);
        for (int i = 0; i < BT * C; i++) d_qkv_in[i] += d_ln_out[i];

        rcre_transformer_linear_backward(d_ln_out, dw->k_w, dw->k_b, dk, a->ln1_out, w->k_w, BT, C, C);
        for (int i = 0; i < BT * C; i++) d_qkv_in[i] += d_ln_out[i];

        rcre_transformer_linear_backward(d_ln_out, dw->v_w, dw->v_b, dv, a->ln1_out, w->v_w, BT, C, C);
        for (int i = 0; i < BT * C; i++) d_qkv_in[i] += d_ln_out[i];

        const float *prev_input = (l > 0) ? acts->blocks[l - 1].resid2_out : acts->emb_out;
        rcre_transformer_layernorm_backward(d_x, dw->ln1_w, dw->ln1_b, d_qkv_in, prev_input, a->ln1_mean, a->ln1_rstd, w->ln1_w, BT, C);

        for (int i = 0; i < BT * C; i++) {
            d_x[i] += d_resid1[i];
        }
    }

    /* 5. Embedding Backward */
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token = inputs[b * T + t];
            const float *d_emb = d_x + IDX3(b, t, 0, T, C);
            float *dwte_row = d_wte + token * C;
            float *dwpe_row = d_wpe + t * C;
            for (int c = 0; c < C; c++) {
                dwte_row[c] += d_emb[c];
                dwpe_row[c] += d_emb[c];
            }
        }
    }

    /* Free temporary scratch buffers */
    free(d_logits);
    free(d_ln_f);
    free(d_x);
    free(d_ln_out);
    free(d_fc1);
    free(d_gelu);
    free(d_att);
    free(dq);
    free(dk);
    free(dv);
    free(datt_sc);
    free(d_qkv_in);
    free(d_resid1);
}
