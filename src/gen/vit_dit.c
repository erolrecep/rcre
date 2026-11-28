/**
 * @file vit_dit.c
 * @brief Phase 4: Vision Transformer (RCRE_ViT) & Diffusion Transformer (RCRE_DiT w/ AdaLN-Zero)
 *
 * Part of Recep's Code Run Everywhere (RCRE).
 * Strict ISO C23 compliant, zero external dependencies, zero hot-loop mallocs.
 */

#include "rcre_gen.h"
#include "gen_diff_sample_int.h"
#include "rcre_rng.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Epsilon-prediction forward adapter for the shared DDIM stepper
 * (class_labels arrive through the ctx pointer). */
static RCRE_Tensor* gen_dit_fwd_wrap(const void* model, RCRE_TensorTape* tape,
                                     RCRE_Tensor* x, const float* t, void* ctx) {
    return rcre_dit_forward((const RCRE_DiT*)model, tape, x, t,
                            (const size_t*)ctx);
}

static const char* gen_vt_fmt_label(RCRE_Arena* arena, const char* model, size_t layer, const char* name) {
    char* buf = (char*)rcre_arena_alloc(arena, 48, 8);
    if (!buf) return nullptr;
    snprintf(buf, 48, "%s.b%zu.%s", model, layer, name);
    return buf;
}

/* ========================================================================= */
/* 1. Vision Transformer (RCRE_ViT)                                          */
/* ========================================================================= */

RCRE_ViT* rcre_vit_create(RCRE_Arena* arena, RCRE_ViTConfig cfg, uint64_t* rng) {
    if (!arena || !rng || cfg.in_channels == 0 || cfg.patch_size == 0 ||
        cfg.height == 0 || cfg.width == 0 ||
        (cfg.height % cfg.patch_size) != 0 || (cfg.width % cfg.patch_size) != 0 ||
        cfg.embed_dim == 0 || cfg.num_heads == 0 || (cfg.embed_dim % cfg.num_heads) != 0 ||
        cfg.num_layers == 0 || cfg.mlp_dim == 0 || cfg.num_classes == 0) {
        return nullptr;
    }

    RCRE_ViT* m = (RCRE_ViT*)rcre_arena_alloc(arena, sizeof(RCRE_ViT), alignof(RCRE_ViT));
    if (!m) return nullptr;
    memset(m, 0, sizeof(*m));

    m->cfg         = cfg;
    m->num_patches = (cfg.height / cfg.patch_size) * (cfg.width / cfg.patch_size);
    m->patch_dim   = cfg.in_channels * cfg.patch_size * cfg.patch_size;
    size_t C       = cfg.embed_dim;
    size_t M       = cfg.mlp_dim;

    m->patch_w = rcre_tensor_alloc_2d(arena, m->patch_dim, C);
    m->patch_b = rcre_tensor_alloc_1d(arena, C);
    m->pos_emb = rcre_tensor_alloc_2d(arena, m->num_patches, C);
    m->blocks  = (RCRE_ViTBlock*)rcre_arena_alloc(arena, cfg.num_layers * sizeof(RCRE_ViTBlock), alignof(RCRE_ViTBlock));
    m->ln_f_g  = rcre_tensor_alloc_1d(arena, C);
    m->ln_f_b  = rcre_tensor_alloc_1d(arena, C);
    m->head_w  = rcre_tensor_alloc_2d(arena, C, cfg.num_classes);
    m->head_b  = rcre_tensor_alloc_1d(arena, cfg.num_classes);

    if (!m->patch_w || !m->patch_b || !m->pos_emb || !m->blocks ||
        !m->ln_f_g  || !m->ln_f_b  || !m->head_w  || !m->head_b) {
        return nullptr;
    }

    rcre_tensor_init_xavier(m->patch_w, m->patch_dim, C, rng);
    rcre_tensor_fill_normal(m->pos_emb, 0.0f, 0.02f, rng);
    rcre_tensor_ones(m->ln_f_g);
    rcre_tensor_init_xavier(m->head_w, C, cfg.num_classes, rng);

    m->patch_w->label = "vit.patch.w"; m->patch_b->label = "vit.patch.b";
    m->pos_emb->label = "vit.pos_emb";
    m->ln_f_g->label  = "vit.ln_f.g";  m->ln_f_b->label  = "vit.ln_f.b";
    m->head_w->label  = "vit.head.w";  m->head_b->label  = "vit.head.b";

    for (size_t l = 0; l < cfg.num_layers; ++l) {
        RCRE_ViTBlock* b = &m->blocks[l];
        b->ln1_g = rcre_tensor_alloc_1d(arena, C);
        b->ln1_b = rcre_tensor_alloc_1d(arena, C);
        b->wq    = rcre_tensor_alloc_2d(arena, C, C); b->bq = rcre_tensor_alloc_1d(arena, C);
        b->wk    = rcre_tensor_alloc_2d(arena, C, C); b->bk = rcre_tensor_alloc_1d(arena, C);
        b->wv    = rcre_tensor_alloc_2d(arena, C, C); b->bv = rcre_tensor_alloc_1d(arena, C);
        b->wo    = rcre_tensor_alloc_2d(arena, C, C); b->bo = rcre_tensor_alloc_1d(arena, C);
        b->ln2_g = rcre_tensor_alloc_1d(arena, C);
        b->ln2_b = rcre_tensor_alloc_1d(arena, C);
        b->fc1_w = rcre_tensor_alloc_2d(arena, C, M); b->fc1_b = rcre_tensor_alloc_1d(arena, M);
        b->fc2_w = rcre_tensor_alloc_2d(arena, M, C); b->fc2_b = rcre_tensor_alloc_1d(arena, C);

        if (!b->ln1_g || !b->ln1_b || !b->wq || !b->bq || !b->wk || !b->bk ||
            !b->wv || !b->bv || !b->wo || !b->bo || !b->ln2_g || !b->ln2_b ||
            !b->fc1_w || !b->fc1_b || !b->fc2_w || !b->fc2_b) {
            return nullptr;
        }

        rcre_tensor_ones(b->ln1_g);
        rcre_tensor_ones(b->ln2_g);
        rcre_tensor_init_xavier(b->wq, C, C, rng);
        rcre_tensor_init_xavier(b->wk, C, C, rng);
        rcre_tensor_init_xavier(b->wv, C, C, rng);
        rcre_tensor_init_xavier(b->wo, C, C, rng);
        rcre_tensor_init_xavier(b->fc1_w, C, M, rng);
        rcre_tensor_init_xavier(b->fc2_w, M, C, rng);

        b->ln1_g->label = gen_vt_fmt_label(arena, "vit", l, "ln1.g");
        b->ln1_b->label = gen_vt_fmt_label(arena, "vit", l, "ln1.b");
        b->wq->label    = gen_vt_fmt_label(arena, "vit", l, "wq");
        b->bq->label    = gen_vt_fmt_label(arena, "vit", l, "bq");
        b->wk->label    = gen_vt_fmt_label(arena, "vit", l, "wk");
        b->bk->label    = gen_vt_fmt_label(arena, "vit", l, "bk");
        b->wv->label    = gen_vt_fmt_label(arena, "vit", l, "wv");
        b->bv->label    = gen_vt_fmt_label(arena, "vit", l, "bv");
        b->wo->label    = gen_vt_fmt_label(arena, "vit", l, "wo");
        b->bo->label    = gen_vt_fmt_label(arena, "vit", l, "bo");
        b->ln2_g->label = gen_vt_fmt_label(arena, "vit", l, "ln2.g");
        b->ln2_b->label = gen_vt_fmt_label(arena, "vit", l, "ln2.b");
        b->fc1_w->label = gen_vt_fmt_label(arena, "vit", l, "fc1.w");
        b->fc1_b->label = gen_vt_fmt_label(arena, "vit", l, "fc1.b");
        b->fc2_w->label = gen_vt_fmt_label(arena, "vit", l, "fc2.w");
        b->fc2_b->label = gen_vt_fmt_label(arena, "vit", l, "fc2.b");
    }

    return m;
}

size_t rcre_vit_param_count(const RCRE_ViT* m) {
    if (!m) return 0;
    return 7U + m->cfg.num_layers * 16U;
}

size_t rcre_vit_params(const RCRE_ViT* m, RCRE_Tensor** out_params, size_t max_params) {
    size_t needed = rcre_vit_param_count(m);
    if (!m || !out_params || max_params < needed) return 0;
    size_t pos = 0;
    out_params[pos++] = m->patch_w;
    out_params[pos++] = m->patch_b;
    out_params[pos++] = m->pos_emb;
    for (size_t l = 0; l < m->cfg.num_layers; ++l) {
        const RCRE_ViTBlock* b = &m->blocks[l];
        out_params[pos++] = b->ln1_g; out_params[pos++] = b->ln1_b;
        out_params[pos++] = b->wq;    out_params[pos++] = b->bq;
        out_params[pos++] = b->wk;    out_params[pos++] = b->bk;
        out_params[pos++] = b->wv;    out_params[pos++] = b->bv;
        out_params[pos++] = b->wo;    out_params[pos++] = b->bo;
        out_params[pos++] = b->ln2_g; out_params[pos++] = b->ln2_b;
        out_params[pos++] = b->fc1_w; out_params[pos++] = b->fc1_b;
        out_params[pos++] = b->fc2_w; out_params[pos++] = b->fc2_b;
    }
    out_params[pos++] = m->ln_f_g;
    out_params[pos++] = m->ln_f_b;
    out_params[pos++] = m->head_w;
    out_params[pos++] = m->head_b;
    return pos;
}

RCRE_Tensor* rcre_vit_forward(const RCRE_ViT* m, RCRE_TensorTape* tape, RCRE_Tensor* img) {
    if (!m || !tape || !img || img->ndim != 4 ||
        img->shape[1] != m->cfg.in_channels ||
        img->shape[2] != m->cfg.height ||
        img->shape[3] != m->cfg.width) {
        return nullptr;
    }

    RCRE_Tensor* x = rcre_autograd_patchify(tape, img, m->cfg.patch_size);
    x = rcre_autograd_linear(tape, x, m->patch_w, m->patch_b);
    x = rcre_autograd_add_pos_emb_3d(tape, x, m->pos_emb);

    for (size_t l = 0; l < m->cfg.num_layers; ++l) {
        const RCRE_ViTBlock* b = &m->blocks[l];
        RCRE_Tensor* h = rcre_autograd_layernorm(tape, x, b->ln1_g, b->ln1_b, 1e-5f);
        RCRE_Tensor* q = rcre_autograd_linear(tape, h, b->wq, b->bq);
        RCRE_Tensor* k = rcre_autograd_linear(tape, h, b->wk, b->bk);
        RCRE_Tensor* v = rcre_autograd_linear(tape, h, b->wv, b->bv);
        RCRE_Tensor* a = rcre_autograd_mhsa(tape, q, k, v, m->cfg.num_heads);
        a = rcre_autograd_linear(tape, a, b->wo, b->bo);
        x = rcre_autograd_add(tape, x, a);

        RCRE_Tensor* h2 = rcre_autograd_layernorm(tape, x, b->ln2_g, b->ln2_b, 1e-5f);
        RCRE_Tensor* mlp = rcre_autograd_linear(tape, h2, b->fc1_w, b->fc1_b);
        mlp = rcre_autograd_gelu(tape, mlp);
        mlp = rcre_autograd_linear(tape, mlp, b->fc2_w, b->fc2_b);
        x = rcre_autograd_add(tape, x, mlp);
    }

    x = rcre_autograd_layernorm(tape, x, m->ln_f_g, m->ln_f_b, 1e-5f);
    RCRE_Tensor* pooled = rcre_autograd_mean_pool_seq(tape, x);
    return rcre_autograd_linear(tape, pooled, m->head_w, m->head_b);
}

float rcre_vit_train_step(
    RCRE_ViT* m, RCRE_TensorTape* tape, RCRE_TensorOptim* opt,
    RCRE_Tensor* img, const size_t* labels, float max_grad_norm)
{
    if (!m || !tape || !opt || !img || !labels || img->ndim != 4) return -1.0f;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(tape->arena);
    rcre_tensor_tape_reset(tape);
    rcre_tensor_optim_zero_grad(opt);

    RCRE_Tensor* logits = rcre_vit_forward(m, tape, img);
    if (!logits) {
        rcre_arena_temp_end(tmp);
        return -1.0f;
    }

    float loss = rcre_loss_cross_entropy_logits(logits, labels, img->shape[0], m->cfg.num_classes, logits);
    rcre_tensor_tape_backward(tape);

    if (max_grad_norm > 0.0f) {
        (void)rcre_tensor_clip_grad_norm(opt->params, opt->num_params, max_grad_norm);
    }
    rcre_tensor_optim_step(opt);

    rcre_arena_temp_end(tmp);
    return loss;
}

int rcre_vit_predict(const RCRE_ViT* m, RCRE_Arena* scratch, RCRE_Tensor* img, size_t* out_preds) {
    if (!m || !scratch || !img || !out_preds || img->ndim != 4) return RCRE_ERR_INVALID_ARG;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(scratch, 64);
    if (!tape) {
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    RCRE_Tensor* logits = rcre_vit_forward(m, tape, img);
    if (!logits) {
        rcre_tensor_tape_destroy(tape);
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    size_t N = img->shape[0];
    size_t K = m->cfg.num_classes;
    const float* ld = rcre_tensor_f32_data(logits);
    for (size_t n = 0; n < N; ++n) {
        size_t best = 0;
        float best_v = ld[n * K];
        for (size_t c = 1; c < K; ++c) {
            if (ld[n * K + c] > best_v) {
                best_v = ld[n * K + c];
                best = c;
            }
        }
        out_preds[n] = best;
    }

    rcre_tensor_tape_destroy(tape);
    rcre_arena_temp_end(tmp);
    return RCRE_OK;
}

/* ========================================================================= */
/* 2. Diffusion Transformer with AdaLN-Zero (RCRE_DiT)                       */
/* ========================================================================= */

RCRE_DiT* rcre_dit_create(RCRE_Arena* arena, RCRE_DiTConfig cfg, uint64_t* rng) {
    if (!arena || !rng || cfg.in_channels == 0 || cfg.patch_size == 0 ||
        cfg.height == 0 || cfg.width == 0 ||
        (cfg.height % cfg.patch_size) != 0 || (cfg.width % cfg.patch_size) != 0 ||
        cfg.embed_dim < 2 || (cfg.embed_dim & 1U) != 0 ||
        cfg.num_heads == 0 || (cfg.embed_dim % cfg.num_heads) != 0 ||
        cfg.num_layers == 0 || cfg.mlp_dim == 0) {
        return nullptr;
    }

    RCRE_DiT* m = (RCRE_DiT*)rcre_arena_alloc(arena, sizeof(RCRE_DiT), alignof(RCRE_DiT));
    if (!m) return nullptr;
    memset(m, 0, sizeof(*m));

    m->cfg         = cfg;
    m->num_patches = (cfg.height / cfg.patch_size) * (cfg.width / cfg.patch_size);
    m->patch_dim   = cfg.in_channels * cfg.patch_size * cfg.patch_size;
    size_t C       = cfg.embed_dim;
    size_t M       = cfg.mlp_dim;

    m->patch_w    = rcre_tensor_alloc_2d(arena, m->patch_dim, C);
    m->patch_b    = rcre_tensor_alloc_1d(arena, C);
    m->pos_emb    = rcre_tensor_alloc_2d(arena, m->num_patches, C);
    m->time_fc1_w = rcre_tensor_alloc_2d(arena, C, C);
    m->time_fc1_b = rcre_tensor_alloc_1d(arena, C);
    m->time_fc2_w = rcre_tensor_alloc_2d(arena, C, C);
    m->time_fc2_b = rcre_tensor_alloc_1d(arena, C);

    if (cfg.num_classes > 0) {
        m->class_emb = rcre_tensor_alloc_2d(arena, cfg.num_classes, C);
        if (!m->class_emb) return nullptr;
        rcre_tensor_fill_normal(m->class_emb, 0.0f, 0.02f, rng);
        m->class_emb->label = "dit.class_emb";
    }

    m->blocks      = (RCRE_DiTBlock*)rcre_arena_alloc(arena, cfg.num_layers * sizeof(RCRE_DiTBlock), alignof(RCRE_DiTBlock));
    m->final_ada_w = rcre_tensor_alloc_2d(arena, C, 2U * C);
    m->final_ada_b = rcre_tensor_alloc_1d(arena, 2U * C);
    m->final_w     = rcre_tensor_alloc_2d(arena, C, m->patch_dim);
    m->final_b     = rcre_tensor_alloc_1d(arena, m->patch_dim);

    if (!m->patch_w || !m->patch_b || !m->pos_emb ||
        !m->time_fc1_w || !m->time_fc1_b || !m->time_fc2_w || !m->time_fc2_b ||
        !m->blocks || !m->final_ada_w || !m->final_ada_b || !m->final_w || !m->final_b) {
        return nullptr;
    }

    rcre_tensor_init_xavier(m->patch_w, m->patch_dim, C, rng);
    rcre_tensor_fill_normal(m->pos_emb, 0.0f, 0.02f, rng);
    rcre_tensor_init_xavier(m->time_fc1_w, C, C, rng);
    rcre_tensor_init_xavier(m->time_fc2_w, C, C, rng);
    /* AdaLN-Zero: final_ada_w/b and final_w/b remain zero-initialized */

    m->patch_w->label     = "dit.patch.w";     m->patch_b->label     = "dit.patch.b";
    m->pos_emb->label     = "dit.pos_emb";
    m->time_fc1_w->label  = "dit.time_fc1.w";  m->time_fc1_b->label  = "dit.time_fc1.b";
    m->time_fc2_w->label  = "dit.time_fc2.w";  m->time_fc2_b->label  = "dit.time_fc2.b";
    m->final_ada_w->label = "dit.final_ada.w"; m->final_ada_b->label = "dit.final_ada.b";
    m->final_w->label     = "dit.final.w";     m->final_b->label     = "dit.final.b";

    for (size_t l = 0; l < cfg.num_layers; ++l) {
        RCRE_DiTBlock* b = &m->blocks[l];
        b->ada_w = rcre_tensor_alloc_2d(arena, C, 6U * C);
        b->ada_b = rcre_tensor_alloc_1d(arena, 6U * C);
        b->wq    = rcre_tensor_alloc_2d(arena, C, C); b->bq = rcre_tensor_alloc_1d(arena, C);
        b->wk    = rcre_tensor_alloc_2d(arena, C, C); b->bk = rcre_tensor_alloc_1d(arena, C);
        b->wv    = rcre_tensor_alloc_2d(arena, C, C); b->bv = rcre_tensor_alloc_1d(arena, C);
        b->wo    = rcre_tensor_alloc_2d(arena, C, C); b->bo = rcre_tensor_alloc_1d(arena, C);
        b->fc1_w = rcre_tensor_alloc_2d(arena, C, M); b->fc1_b = rcre_tensor_alloc_1d(arena, M);
        b->fc2_w = rcre_tensor_alloc_2d(arena, M, C); b->fc2_b = rcre_tensor_alloc_1d(arena, C);

        if (!b->ada_w || !b->ada_b || !b->wq || !b->bq || !b->wk || !b->bk ||
            !b->wv || !b->bv || !b->wo || !b->bo || !b->fc1_w || !b->fc1_b ||
            !b->fc2_w || !b->fc2_b) {
            return nullptr;
        }

        /* ada_w / ada_b stay zero-initialized (AdaLN-Zero identity initialization) */
        rcre_tensor_init_xavier(b->wq, C, C, rng);
        rcre_tensor_init_xavier(b->wk, C, C, rng);
        rcre_tensor_init_xavier(b->wv, C, C, rng);
        rcre_tensor_init_xavier(b->wo, C, C, rng);
        rcre_tensor_init_xavier(b->fc1_w, C, M, rng);
        rcre_tensor_init_xavier(b->fc2_w, M, C, rng);

        b->ada_w->label = gen_vt_fmt_label(arena, "dit", l, "ada.w");
        b->ada_b->label = gen_vt_fmt_label(arena, "dit", l, "ada.b");
        b->wq->label    = gen_vt_fmt_label(arena, "dit", l, "wq");
        b->bq->label    = gen_vt_fmt_label(arena, "dit", l, "bq");
        b->wk->label    = gen_vt_fmt_label(arena, "dit", l, "wk");
        b->bk->label    = gen_vt_fmt_label(arena, "dit", l, "bk");
        b->wv->label    = gen_vt_fmt_label(arena, "dit", l, "wv");
        b->bv->label    = gen_vt_fmt_label(arena, "dit", l, "bv");
        b->wo->label    = gen_vt_fmt_label(arena, "dit", l, "wo");
        b->bo->label    = gen_vt_fmt_label(arena, "dit", l, "bo");
        b->fc1_w->label = gen_vt_fmt_label(arena, "dit", l, "fc1.w");
        b->fc1_b->label = gen_vt_fmt_label(arena, "dit", l, "fc1.b");
        b->fc2_w->label = gen_vt_fmt_label(arena, "dit", l, "fc2.w");
        b->fc2_b->label = gen_vt_fmt_label(arena, "dit", l, "fc2.b");
    }

    return m;
}

size_t rcre_dit_param_count(const RCRE_DiT* m) {
    if (!m) return 0;
    size_t base = 11U + (m->class_emb != nullptr ? 1U : 0U);
    return base + m->cfg.num_layers * 14U;
}

size_t rcre_dit_params(const RCRE_DiT* m, RCRE_Tensor** out_params, size_t max_params) {
    size_t needed = rcre_dit_param_count(m);
    if (!m || !out_params || max_params < needed) return 0;
    size_t pos = 0;
    out_params[pos++] = m->patch_w;
    out_params[pos++] = m->patch_b;
    out_params[pos++] = m->pos_emb;
    out_params[pos++] = m->time_fc1_w;
    out_params[pos++] = m->time_fc1_b;
    out_params[pos++] = m->time_fc2_w;
    out_params[pos++] = m->time_fc2_b;
    if (m->class_emb != nullptr) {
        out_params[pos++] = m->class_emb;
    }
    for (size_t l = 0; l < m->cfg.num_layers; ++l) {
        const RCRE_DiTBlock* b = &m->blocks[l];
        out_params[pos++] = b->ada_w; out_params[pos++] = b->ada_b;
        out_params[pos++] = b->wq;    out_params[pos++] = b->bq;
        out_params[pos++] = b->wk;    out_params[pos++] = b->bk;
        out_params[pos++] = b->wv;    out_params[pos++] = b->bv;
        out_params[pos++] = b->wo;    out_params[pos++] = b->bo;
        out_params[pos++] = b->fc1_w; out_params[pos++] = b->fc1_b;
        out_params[pos++] = b->fc2_w; out_params[pos++] = b->fc2_b;
    }
    out_params[pos++] = m->final_ada_w;
    out_params[pos++] = m->final_ada_b;
    out_params[pos++] = m->final_w;
    out_params[pos++] = m->final_b;
    return pos;
}

RCRE_Tensor* rcre_dit_forward(
    const RCRE_DiT* m, RCRE_TensorTape* tape, RCRE_Tensor* x_t,
    const float* timesteps, const size_t* class_labels)
{
    if (!m || !tape || !x_t || !timesteps || x_t->ndim != 4 ||
        x_t->shape[1] != m->cfg.in_channels ||
        x_t->shape[2] != m->cfg.height ||
        x_t->shape[3] != m->cfg.width) {
        return nullptr;
    }
    if (m->class_emb != nullptr && class_labels == nullptr) return nullptr;

    size_t N = x_t->shape[0];
    size_t C = m->cfg.embed_dim;

    /* 1. Patchify + linear embed + positional embedding */
    RCRE_Tensor* x = rcre_autograd_patchify(tape, x_t, m->cfg.patch_size);
    x = rcre_autograd_linear(tape, x, m->patch_w, m->patch_b);
    x = rcre_autograd_add_pos_emb_3d(tape, x, m->pos_emb);

    /* 2. Conditioning vector c = MLP(sinusoidal(t)) [+ class_emb(y)] */
    RCRE_Tensor* c = rcre_sinusoidal_timestep_embed(tape->arena, timesteps, N, C);
    c = rcre_autograd_linear(tape, c, m->time_fc1_w, m->time_fc1_b);
    c = rcre_autograd_silu(tape, c);
    c = rcre_autograd_linear(tape, c, m->time_fc2_w, m->time_fc2_b);
    if (m->class_emb != nullptr) {
        RCRE_Tensor* y_emb = rcre_autograd_embed_lookup(tape, m->class_emb, class_labels, N);
        c = rcre_autograd_add(tape, c, y_emb);
    }
    RCRE_Tensor* c_act = rcre_autograd_silu(tape, c);

    /* 3. DiT Blocks with AdaLN-Zero modulation */
    for (size_t l = 0; l < m->cfg.num_layers; ++l) {
        const RCRE_DiTBlock* b = &m->blocks[l];
        RCRE_Tensor* mod = rcre_autograd_linear(tape, c_act, b->ada_w, b->ada_b);
        RCRE_Tensor* shift_msa = rcre_autograd_slice_cols(tape, mod, 0U * C, C);
        RCRE_Tensor* scale_msa = rcre_autograd_slice_cols(tape, mod, 1U * C, C);
        RCRE_Tensor* gate_msa  = rcre_autograd_slice_cols(tape, mod, 2U * C, C);
        RCRE_Tensor* shift_mlp = rcre_autograd_slice_cols(tape, mod, 3U * C, C);
        RCRE_Tensor* scale_mlp = rcre_autograd_slice_cols(tape, mod, 4U * C, C);
        RCRE_Tensor* gate_mlp  = rcre_autograd_slice_cols(tape, mod, 5U * C, C);

        RCRE_Tensor* h1 = rcre_autograd_layernorm(tape, x, nullptr, nullptr, 1e-5f);
        h1 = rcre_autograd_adaln_3d(tape, h1, scale_msa, shift_msa);
        RCRE_Tensor* q = rcre_autograd_linear(tape, h1, b->wq, b->bq);
        RCRE_Tensor* k = rcre_autograd_linear(tape, h1, b->wk, b->bk);
        RCRE_Tensor* v = rcre_autograd_linear(tape, h1, b->wv, b->bv);
        RCRE_Tensor* a = rcre_autograd_mhsa(tape, q, k, v, m->cfg.num_heads);
        a = rcre_autograd_linear(tape, a, b->wo, b->bo);
        x = rcre_autograd_add(tape, x, rcre_autograd_gate_3d(tape, a, gate_msa));

        RCRE_Tensor* h2 = rcre_autograd_layernorm(tape, x, nullptr, nullptr, 1e-5f);
        h2 = rcre_autograd_adaln_3d(tape, h2, scale_mlp, shift_mlp);
        RCRE_Tensor* mlp = rcre_autograd_linear(tape, h2, b->fc1_w, b->fc1_b);
        mlp = rcre_autograd_gelu(tape, mlp);
        mlp = rcre_autograd_linear(tape, mlp, b->fc2_w, b->fc2_b);
        x = rcre_autograd_add(tape, x, rcre_autograd_gate_3d(tape, mlp, gate_mlp));
    }

    /* 4. Final AdaLN-Zero + linear projection + unpatchify */
    RCRE_Tensor* fmod    = rcre_autograd_linear(tape, c_act, m->final_ada_w, m->final_ada_b);
    RCRE_Tensor* shift_f = rcre_autograd_slice_cols(tape, fmod, 0, C);
    RCRE_Tensor* scale_f = rcre_autograd_slice_cols(tape, fmod, C, C);

    RCRE_Tensor* hf = rcre_autograd_layernorm(tape, x, nullptr, nullptr, 1e-5f);
    hf = rcre_autograd_adaln_3d(tape, hf, scale_f, shift_f);
    RCRE_Tensor* patches = rcre_autograd_linear(tape, hf, m->final_w, m->final_b);
    return rcre_autograd_unpatchify(tape, patches, m->cfg.in_channels, m->cfg.height, m->cfg.width, m->cfg.patch_size);
}

float rcre_dit_train_step(
    RCRE_DiT* m,
    const RCRE_DiffSchedule* sched,
    RCRE_TensorTape* tape,
    RCRE_TensorOptim* opt,
    const RCRE_Tensor* x0,
    const size_t* class_labels,
    uint64_t* rng,
    float max_grad_norm)
{
    if (!m || !sched || !tape || !opt || !x0 || !rng || x0->ndim != 4) return -1.0f;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(tape->arena);
    rcre_tensor_tape_reset(tape);
    rcre_tensor_optim_zero_grad(opt);

    size_t N = x0->shape[0];
    size_t* t_idx = (size_t*)rcre_arena_alloc(tape->arena, N * sizeof(size_t), alignof(size_t));
    float*  t_flt = (float*)rcre_arena_alloc(tape->arena, N * sizeof(float), 16);
    RCRE_Tensor* noise = rcre_tensor_alloc(tape->arena, x0->shape, 4);
    RCRE_Tensor* x_t   = rcre_tensor_alloc(tape->arena, x0->shape, 4);
    if (!t_idx || !t_flt || !noise || !x_t) {
        rcre_arena_temp_end(tmp);
        return -1.0f;
    }

    for (size_t n = 0; n < N; ++n) {
        t_idx[n] = (size_t)(rcre_rng_next(rng) % (uint64_t)sched->num_steps);
        t_flt[n] = (float)t_idx[n];
    }
    rcre_tensor_fill_normal(noise, 0.0f, 1.0f, rng);
    if (rcre_diff_q_sample(sched, x0, t_idx, noise, x_t) != RCRE_OK) {
        rcre_arena_temp_end(tmp);
        return -1.0f;
    }

    RCRE_Tensor* eps_pred = rcre_dit_forward(m, tape, x_t, t_flt, class_labels);
    if (!eps_pred) {
        rcre_arena_temp_end(tmp);
        return -1.0f;
    }

    float loss = rcre_loss_mse(eps_pred, noise, eps_pred);
    rcre_tensor_tape_backward(tape);

    if (max_grad_norm > 0.0f) {
        (void)rcre_tensor_clip_grad_norm(opt->params, opt->num_params, max_grad_norm);
    }
    rcre_tensor_optim_step(opt);

    rcre_arena_temp_end(tmp);
    return loss;
}

RCRE_Tensor* rcre_dit_sample_ddim(
    const RCRE_DiT* m, const RCRE_DiffSchedule* sched,
    RCRE_Arena* out_arena, RCRE_Arena* scratch,
    size_t num_samples, const size_t* class_labels,
    size_t ddim_steps, float eta, uint64_t* rng)
{
    if (!m || !sched || !out_arena || !scratch || num_samples == 0 ||
        ddim_steps == 0 || ddim_steps > sched->num_steps || !rng) {
        return nullptr;
    }
    if (m->class_emb != nullptr && class_labels == nullptr) return nullptr;
    return gen_diff_sample_ddim(m, gen_dit_fwd_wrap, (void*)class_labels, sched,
                                m->cfg.in_channels, m->cfg.height, m->cfg.width,
                                out_arena, scratch, num_samples, ddim_steps, eta, rng, 128);
}
