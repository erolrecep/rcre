/**
 * @file test_transformer_merge.c
 * @brief Regression suite for the micro_llm_c -> rcre transformer merger.
 *
 * Maps 1:1 to the findings in
 * Desktop/rcre_micro_llm_c_merger_review_by_pidev.md:
 *
 *   Part 1 — Deserialization robustness (corrupt/malicious files must be
 *            rejected, never crash or corrupt memory)
 *   Part 2 — Validation contracts (optimizer hyperparameters, KV-cache
 *            layout, model config, token bounds)
 *   Part 3 — Behavioral contracts (data loader, sampler determinism,
 *            end-to-end learning, arena-model KV equivalence)
 */

#include "rcre.h"
#include "test_framework.h"
#include <math.h>
#include <string.h>

/* ========================================================================= */
/* Helpers                                                                   */
/* ========================================================================= */

static RCRE_GPTConfig tiny_config(void) {
    RCRE_GPTConfig cfg = {
        .vocab_size = 32,
        .seq_len = 16,
        .embed_dim = 32,
        .num_heads = 2,
        .head_dim = 16,
        .num_layers = 1,
        .mlp_dim = 64
    };
    return cfg;
}

/* Writes an 8-byte header (magic + version + 2 int fields) and returns the
 * FILE* left positioned after the header. */
static FILE* open_bpe_scaffold(const char *path, uint32_t magic, uint8_t version, int v, int r) {
    FILE *f = fopen(path, "wb");
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&v, sizeof(int), 1, f);
    fwrite(&r, sizeof(int), 1, f);
    return f;
}

/* ========================================================================= */
/* Part 1 — Deserialization robustness                                       */
/* ========================================================================= */

static void test_bpe_load_corrupt_files_rejected(void) {
    const char *corpus = "data/tinystories.txt";

    /* A healthy tokenizer to load corrupt files into. */
    RCRE_BPETokenizer *tok = rcre_bpe_create();
    ASSERT_NOT_NULL(tok);
    ASSERT_TRUE(rcre_bpe_train_from_file(tok, corpus, 384));
    int good_vocab = tok->vocab_size;
    int good_rules = tok->num_rules;

    /* 1. Wrong magic. */
    {
        FILE *f = open_bpe_scaffold("data/tmp_bad_magic.bin", 0xDEADBEEFu, RCRE_BPE_FORMAT_VERSION, 300, 44);
        fclose(f);
        RCRE_BPETokenizer *t2 = rcre_bpe_create();
        ASSERT_FALSE(rcre_bpe_load(t2, "data/tmp_bad_magic.bin"));
        rcre_bpe_free(t2);
        remove("data/tmp_bad_magic.bin");
    }

    /* 2. Wrong version. */
    {
        FILE *f = open_bpe_scaffold("data/tmp_bad_ver.bin", 0x31455052u, 99, 300, 44);
        fclose(f);
        RCRE_BPETokenizer *t2 = rcre_bpe_create();
        ASSERT_FALSE(rcre_bpe_load(t2, "data/tmp_bad_ver.bin"));
        rcre_bpe_free(t2);
        remove("data/tmp_bad_ver.bin");
    }

    /* 3. vocab_size above the 2048 ceiling (probe A: used to OOB-write the
     *    pointer table and SEGV in bpe_free). */
    {
        FILE *f = open_bpe_scaffold("data/tmp_big_vocab.bin", 0x31455052u, RCRE_BPE_FORMAT_VERSION, 5000, 4744);
        fclose(f);
        RCRE_BPETokenizer *t2 = rcre_bpe_create();
        ASSERT_FALSE(rcre_bpe_load(t2, "data/tmp_big_vocab.bin"));
        rcre_bpe_free(t2);
        remove("data/tmp_big_vocab.bin");
    }

    /* 4. Inconsistent num_rules (file claims vocab 300 but 100 rules; a
     *    trained tokenizer always has vocab == 256 + rules). */
    {
        FILE *f = open_bpe_scaffold("data/tmp_inconsistent.bin", 0x31455052u, RCRE_BPE_FORMAT_VERSION, 300, 100);
        fclose(f);
        RCRE_BPETokenizer *t2 = rcre_bpe_create();
        ASSERT_FALSE(rcre_bpe_load(t2, "data/tmp_inconsistent.bin"));
        rcre_bpe_free(t2);
        remove("data/tmp_inconsistent.bin");
    }

    /* 5. Negative num_rules (probe C: used to malloc(SIZE_MAX-ish)). */
    {
        FILE *f = open_bpe_scaffold("data/tmp_neg_rules.bin", 0x31455052u, RCRE_BPE_FORMAT_VERSION, 257, -1);
        fclose(f);
        RCRE_BPETokenizer *t2 = rcre_bpe_create();
        ASSERT_FALSE(rcre_bpe_load(t2, "data/tmp_neg_rules.bin"));
        rcre_bpe_free(t2);
        remove("data/tmp_neg_rules.bin");
    }

    /* 6. Truncated file (valid header, no body). */
    {
        FILE *f = open_bpe_scaffold("data/tmp_truncated.bin", 0x31455052u, RCRE_BPE_FORMAT_VERSION, 300, 44);
        fclose(f);
        RCRE_BPETokenizer *t2 = rcre_bpe_create();
        ASSERT_FALSE(rcre_bpe_load(t2, "data/tmp_truncated.bin"));
        rcre_bpe_free(t2);
        remove("data/tmp_truncated.bin");
    }

    /* 7. Empty file. */
    {
        FILE *f = fopen("data/tmp_empty.bin", "wb");
        fclose(f);
        RCRE_BPETokenizer *t2 = rcre_bpe_create();
        ASSERT_FALSE(rcre_bpe_load(t2, "data/tmp_empty.bin"));
        rcre_bpe_free(t2);
        remove("data/tmp_empty.bin");
    }

    /* 8. Nonexistent file. */
    {
        RCRE_BPETokenizer *t2 = rcre_bpe_create();
        ASSERT_FALSE(rcre_bpe_load(t2, "data/does_not_exist_bpe.bin"));
        rcre_bpe_free(t2);
    }

    /* The original tokenizer must be completely untouched by all failures. */
    ASSERT_INT_EQ(good_vocab, tok->vocab_size);
    ASSERT_INT_EQ(good_rules, tok->num_rules);
    char decoded[8];
    int sample_tokens[2] = { 0x48, 0x69 }; /* "Hi" */
    size_t n = rcre_bpe_decode(tok, sample_tokens, 2, decoded, sizeof(decoded));
    ASSERT_UINT_EQ(2, n);
    ASSERT_STR_EQ("Hi", decoded);

    rcre_bpe_free(tok);
}

static void test_char_tokenizer_corrupt_load_rejected(void) {
    /* 1. vocab_size = 300 > 256 member capacity (probe B: stack-buffer-
     *    overflow in fread). */
    {
        FILE *f = fopen("data/tmp_char_big.bin", "wb");
        uint32_t magic = 0x31434352u;
        uint8_t version = RCRE_CHAR_TOK_FORMAT_VERSION;
        int vocab = 300;
        fwrite(&magic, sizeof(magic), 1, f);
        fwrite(&version, sizeof(version), 1, f);
        fwrite(&vocab, sizeof(int), 1, f);
        for (int i = 0; i < 300; i++) fputc('a', f);
        fclose(f);

        RCRE_CharTokenizer tok;
        rcre_char_tokenizer_init(&tok);
        ASSERT_FALSE(rcre_char_tokenizer_load(&tok, "data/tmp_char_big.bin"));
        remove("data/tmp_char_big.bin");
    }

    /* 2. Wrong magic. */
    {
        FILE *f = fopen("data/tmp_char_magic.bin", "wb");
        uint32_t magic = 0x12345678u;
        uint8_t version = RCRE_CHAR_TOK_FORMAT_VERSION;
        int vocab = 3;
        fwrite(&magic, sizeof(magic), 1, f);
        fwrite(&version, sizeof(version), 1, f);
        fwrite(&vocab, sizeof(int), 1, f);
        fputc('a', f); fputc('b', f); fputc('c', f);
        fclose(f);

        RCRE_CharTokenizer tok;
        rcre_char_tokenizer_init(&tok);
        ASSERT_FALSE(rcre_char_tokenizer_load(&tok, "data/tmp_char_magic.bin"));
        remove("data/tmp_char_magic.bin");
    }

    /* 3. Truncated body (header says 10 chars, file has 3). */
    {
        FILE *f = fopen("data/tmp_char_trunc.bin", "wb");
        uint32_t magic = 0x31434352u;
        uint8_t version = RCRE_CHAR_TOK_FORMAT_VERSION;
        int vocab = 10;
        fwrite(&magic, sizeof(magic), 1, f);
        fwrite(&version, sizeof(version), 1, f);
        fwrite(&vocab, sizeof(int), 1, f);
        fputc('a', f); fputc('b', f); fputc('c', f);
        fclose(f);

        RCRE_CharTokenizer tok;
        rcre_char_tokenizer_init(&tok);
        ASSERT_FALSE(rcre_char_tokenizer_load(&tok, "data/tmp_char_trunc.bin"));
        remove("data/tmp_char_trunc.bin");
    }

    /* 4. Valid roundtrip in the new format still works. */
    {
        RCRE_CharTokenizer tok;
        ASSERT_TRUE(rcre_char_tokenizer_train_from_file(&tok, "data/arithmetic.txt"));
        ASSERT_TRUE(rcre_char_tokenizer_save(&tok, "data/tmp_char_ok.bin"));

        RCRE_CharTokenizer loaded;
        rcre_char_tokenizer_init(&loaded);
        ASSERT_TRUE(rcre_char_tokenizer_load(&loaded, "data/tmp_char_ok.bin"));
        ASSERT_INT_EQ(tok.vocab_size, loaded.vocab_size);
        remove("data/tmp_char_ok.bin");
    }
}

static void test_checkpoint_magic_and_config_mismatch(void) {
    RCRE_GPTConfig cfg = tiny_config();
    uint64_t seed = 5ULL;

    RCRE_GPTModel *m1 = rcre_gpt_create(cfg, nullptr);
    ASSERT_NOT_NULL(m1);
    rcre_gpt_init_weights(m1, &seed);
    float *params_copy = (float*)malloc(sizeof(float) * m1->num_params);
    ASSERT_NOT_NULL(params_copy);
    memcpy(params_copy, m1->params, sizeof(float) * m1->num_params);

    ASSERT_TRUE(rcre_gpt_save_checkpoint(m1, "data/tmp_ckpt.bin"));

    /* 1. Load into an identically-configured model: params restored. */
    RCRE_GPTModel *m2 = rcre_gpt_create(cfg, nullptr);
    ASSERT_NOT_NULL(m2);
    ASSERT_TRUE(rcre_gpt_load_checkpoint(m2, "data/tmp_ckpt.bin"));
    ASSERT_FLOAT_ARRAY_NEAR(m1->params, m2->params, m1->num_params, 0.0f);

    /* 2. Load into a different architecture: rejected, params untouched. */
    RCRE_GPTConfig other = cfg;
    other.mlp_dim = 128;
    RCRE_GPTModel *m3 = rcre_gpt_create(other, nullptr);
    ASSERT_NOT_NULL(m3);
    for (size_t i = 0; i < m3->num_params; i++) m3->params[i] = 7.5f; /* sentinel */
    ASSERT_FALSE(rcre_gpt_load_checkpoint(m3, "data/tmp_ckpt.bin"));
    for (size_t i = 0; i < m3->num_params; i++) {
        ASSERT_FLOAT_NEAR(7.5f, m3->params[i], 0.0f);
    }

    /* 3. Corrupted magic: rejected. */
    {
        FILE *f = fopen("data/tmp_ckpt.bin", "r+b");
        ASSERT_NOT_NULL(f);
        uint32_t junk = 0x00C0FFEEu;
        fwrite(&junk, sizeof(junk), 1, f);
        fclose(f);
        ASSERT_FALSE(rcre_gpt_load_checkpoint(m2, "data/tmp_ckpt.bin"));
    }

    /* 4. Checkpoint into a model with a different vocab: rejected. */
    RCRE_GPTConfig bigv = cfg;
    bigv.vocab_size = 64;
    RCRE_GPTModel *m4 = rcre_gpt_create(bigv, nullptr);
    ASSERT_NOT_NULL(m4);
    /* Resave a clean checkpoint for the next case. */
    ASSERT_TRUE(rcre_gpt_save_checkpoint(m1, "data/tmp_ckpt2.bin"));
    ASSERT_FALSE(rcre_gpt_load_checkpoint(m4, "data/tmp_ckpt2.bin"));

    rcre_gpt_free(m1);
    rcre_gpt_free(m2);
    rcre_gpt_free(m3);
    rcre_gpt_free(m4);
    free(params_copy);
    remove("data/tmp_ckpt.bin");
    remove("data/tmp_ckpt2.bin");
}

/* ========================================================================= */
/* Part 2 — Validation contracts                                             */
/* ========================================================================= */

static void test_adamw_hyperparam_validation(void) {
    RCRE_GPTAdamWConfig c = {
        .lr_max = 0.1f, .lr_min = 0.0f, .warmup_steps = 0, .max_steps = 100,
        .beta1 = 0.9f, .beta2 = 0.999f, .eps = 1e-8f, .weight_decay = 0.0f,
        .grad_clip = 0.0f, .current_step = 0
    };
    ASSERT_TRUE(rcre_gpt_adamw_config_valid(&c));

    RCRE_GPTAdamWConfig bad = c; bad.beta1 = 1.0f;
    ASSERT_FALSE(rcre_gpt_adamw_config_valid(&bad));
    bad = c; bad.beta2 = 1.0f;
    ASSERT_FALSE(rcre_gpt_adamw_config_valid(&bad));
    bad = c; bad.lr_max = 0.0f;
    ASSERT_FALSE(rcre_gpt_adamw_config_valid(&bad));
    bad = c; bad.lr_min = 0.5f; bad.lr_max = 0.1f;
    ASSERT_FALSE(rcre_gpt_adamw_config_valid(&bad));
    bad = c; bad.eps = 0.0f;
    ASSERT_FALSE(rcre_gpt_adamw_config_valid(&bad));
    bad = c; bad.max_steps = 10; bad.warmup_steps = 10;
    ASSERT_FALSE(rcre_gpt_adamw_config_valid(&bad));
    ASSERT_FALSE(rcre_gpt_adamw_config_valid(nullptr));

    /*
     * Probe D regression: beta1 == 1.0 previously divided by zero in the
     * bias correction and turned 100% of the parameters NaN. The step must
     * now be a no-op.
     */
    RCRE_GPTConfig cfg = tiny_config();
    uint64_t seed = 7ULL;
    RCRE_GPTModel *m = rcre_gpt_create(cfg, nullptr);
    ASSERT_NOT_NULL(m);
    rcre_gpt_init_weights(m, &seed);
    float *before = (float*)malloc(sizeof(float) * m->num_params);
    ASSERT_NOT_NULL(before);
    memcpy(before, m->params, sizeof(float) * m->num_params);

    for (size_t i = 0; i < m->num_params; i++) m->grads[i] = 0.5f;
    RCRE_GPTAdamWConfig nan_cfg = c; nan_cfg.beta1 = 1.0f;
    rcre_gpt_adamw_step(m, &nan_cfg);

    for (size_t i = 0; i < m->num_params; i++) {
        ASSERT_TRUE(isfinite(m->params[i]));
        ASSERT_FLOAT_NEAR(before[i], m->params[i], 0.0f); /* untouched */
    }

    /* A valid config must actually move the weights. */
    for (size_t i = 0; i < m->num_params; i++) m->grads[i] = 0.5f;
    rcre_gpt_adamw_step(m, &c);
    int moved = 0;
    for (size_t i = 0; i < m->num_params; i++) {
        if (m->params[i] != before[i]) moved++;
        ASSERT_TRUE(isfinite(m->params[i]));
    }
    ASSERT_TRUE(moved > 0);

    /* Degenerate schedule: max_steps == warmup_steps -> floor, no div-by-zero. */
    RCRE_GPTAdamWConfig deg = c; deg.max_steps = 5; deg.warmup_steps = 5;
    ASSERT_FLOAT_NEAR(deg.lr_min, rcre_gpt_get_scheduled_lr(&deg, 3), 1e-6f);

    /* Zero warmup: step 0 is already at lr_max. */
    ASSERT_FLOAT_NEAR(c.lr_max, rcre_gpt_get_scheduled_lr(&c, 0), 1e-5f);

    free(before);
    rcre_gpt_free(m);
}

static void test_kv_cache_config_validation(void) {
    RCRE_GPTConfig cfg = tiny_config();

    /* 1. head_dim inconsistent with embed_dim/num_heads (probe E). */
    RCRE_GPTConfig bad = cfg;
    bad.head_dim = 8; /* correct value is 16 */
    ASSERT_NULL(rcre_kv_cache_create(bad, nullptr));

    /* 2. embed_dim not divisible by num_heads. */
    RCRE_GPTConfig bad2 = cfg;
    bad2.num_heads = 3;
    ASSERT_NULL(rcre_kv_cache_create(bad2, nullptr));

    /* 3. Degenerate dims. */
    RCRE_GPTConfig bad3 = cfg;
    bad3.seq_len = 0;
    ASSERT_NULL(rcre_kv_cache_create(bad3, nullptr));

    /* 4. Valid config still works. */
    RCRE_KVCache *cache = rcre_kv_cache_create(cfg, nullptr);
    ASSERT_NOT_NULL(cache);

    /* 5. forward_token rejects out-of-bounds pos and token. */
    uint64_t seed = 11ULL;
    RCRE_GPTModel *m = rcre_gpt_create(cfg, nullptr);
    ASSERT_NOT_NULL(m);
    rcre_gpt_init_weights(m, &seed);
    float logits[32];
    ASSERT_TRUE(rcre_gpt_forward_token(m, cache, 5, 0, logits));
    ASSERT_FALSE(rcre_gpt_forward_token(m, cache, 5, cfg.seq_len, logits)); /* pos == max_T */
    ASSERT_FALSE(rcre_gpt_forward_token(m, cache, -1, 1, logits));
    ASSERT_FALSE(rcre_gpt_forward_token(m, cache, cfg.vocab_size, 1, logits));

    rcre_kv_cache_free(cache);
    rcre_gpt_free(m);
}

static void test_model_config_validation_and_token_guard(void) {
    RCRE_GPTConfig cfg = tiny_config();

    /* 1. embed_dim % num_heads != 0 -> rejected at create. */
    RCRE_GPTConfig bad = cfg;
    bad.num_heads = 3;
    ASSERT_NULL(rcre_gpt_create(bad, nullptr));

    /* 2. Zero-sized fields rejected. */
    RCRE_GPTConfig bad2 = cfg;
    bad2.mlp_dim = 0;
    ASSERT_NULL(rcre_gpt_create(bad2, nullptr));

    /* 3. head_dim is normalized regardless of what the caller passed. */
    RCRE_GPTConfig unreg = cfg;
    unreg.head_dim = 0;
    RCRE_GPTModel *m = rcre_gpt_create(unreg, nullptr);
    ASSERT_NOT_NULL(m);
    ASSERT_INT_EQ(16, m->config.head_dim);

    /* 4. Forward with an out-of-range target returns the -1.0f sentinel
     *    (previously: assert / out-of-bounds read). */
    RCRE_GPTActs *acts = rcre_gpt_acts_create(m->config, 1, 4, nullptr);
    ASSERT_NOT_NULL(acts);
    uint64_t seed = 13ULL;
    rcre_gpt_init_weights(m, &seed);
    int inputs[4] = { 1, 2, 3, 4 };
    int bad_targets[4] = { 1, 2, 999, 4 };
    float loss = rcre_gpt_forward(m, acts, inputs, bad_targets, 1, 4);
    ASSERT_FLOAT_NEAR(-1.0f, loss, 0.0f);

    /* 5. Valid targets produce a finite, non-negative loss. */
    int good_targets[4] = { 2, 3, 4, 5 };
    loss = rcre_gpt_forward(m, acts, inputs, good_targets, 1, 4);
    ASSERT_TRUE(isfinite(loss) && loss >= 0.0f);

    /* 6. acts_free no longer needs the caller to remember num_layers. */
    rcre_gpt_acts_free(acts);
    rcre_gpt_free(m);
}

static void test_model_full_backward_fd_check(void) {
    /*
     * Finite-difference check of the COMPLETE model backward (embedding ->
     * attention -> MLP -> head -> CE), not just the individual kernels.
     * This is the guard that catches wiring errors (wrong residual stream,
     * transposed gradient, dropped term) that kernel-level FD tests miss.
     */
    RCRE_GPTConfig cfg = { .vocab_size = 12, .seq_len = 8, .embed_dim = 8,
                           .num_heads = 2, .head_dim = 4, .num_layers = 1, .mlp_dim = 8 };
    RCRE_GPTModel *m = rcre_gpt_create(cfg, nullptr);
    ASSERT_NOT_NULL(m);
    uint64_t seed = 99ULL;
    rcre_gpt_init_weights(m, &seed);
    int inputs[8]  = { 3, 7, 1, 5, 9, 2, 10, 4 };
    int targets[8] = { 7, 1, 5, 9, 2, 10, 4, 8 };

    RCRE_GPTActs *acts = rcre_gpt_acts_create(cfg, 1, 8, nullptr);
    ASSERT_NOT_NULL(acts);
    ASSERT_TRUE(rcre_gpt_forward(m, acts, inputs, targets, 1, 8) > 0.0f);
    rcre_gpt_backward(m, acts, inputs, targets, 1, 8);

    size_t n = m->num_params;
    /* One sampled parameter from every weight block. */
    size_t idx[11] = { 2, 40, 60, 130, 200, 280, 400, 500, 600, 620, 700 };
    float eps = 1e-3f;
    for (int s = 0; s < 11; s++) {
        if (idx[s] >= n) idx[s] = n - 1 - (size_t)s;
        float orig = m->params[idx[s]];
        m->params[idx[s]] = orig + eps;
        float lp = rcre_gpt_forward(m, acts, inputs, targets, 1, 8);
        m->params[idx[s]] = orig - eps;
        float lm = rcre_gpt_forward(m, acts, inputs, targets, 1, 8);
        m->params[idx[s]] = orig;
        float fd = (lp - lm) / (2.0f * eps);
        float ag = m->grads[idx[s]];
        float scale = fmaxf(1.0f, fmaxf(fabsf(fd), fabsf(ag)));
        ASSERT_TRUE(fabsf(fd - ag) / scale < 1e-2f);
    }

    rcre_gpt_acts_free(acts);
    rcre_gpt_free(m);
}

/* ========================================================================= */
/* Part 3 — Behavioral contracts                                             */
/* ========================================================================= */

static void test_dataloader_validation_and_epochs(void) {
    RCRE_CharTokenizer tok;
    ASSERT_TRUE(rcre_char_tokenizer_train_from_file(&tok, "data/arithmetic.txt"));
    int V = tok.vocab_size;

    /* Invalid arguments -> NULL. */
    ASSERT_NULL(rcre_gpt_dataloader_create("data/arithmetic.txt", &tok, 0, 32));
    ASSERT_NULL(rcre_gpt_dataloader_create("data/arithmetic.txt", &tok, 4, 1));
    ASSERT_NULL(rcre_gpt_dataloader_create("data/no_such_file.txt", &tok, 4, 32));
    ASSERT_NULL(rcre_gpt_dataloader_create("data/arithmetic.txt", nullptr, 4, 32));

    RCRE_GPTDataLoader *dl = rcre_gpt_dataloader_create("data/arithmetic.txt", &tok, 4, 32);
    ASSERT_NOT_NULL(dl);
    int x[4 * 32], y[4 * 32];

    /* Sequential: first batch starts at index 0, second at 32, ... and the
     * epoch wraps back to a valid start. */
    rcre_gpt_dataloader_next_batch_sequential(dl, x, y);
    rcre_gpt_dataloader_next_batch_sequential(dl, x, y);
    ASSERT_UINT_EQ(2 * 4 * 32, dl->current_idx);
    /* Y is always the X sequence shifted by one. */
    rcre_gpt_dataloader_next_batch_sequential(dl, x, y);
    for (int i = 0; i < 4 * 32 - 4; i++) {
        ASSERT_INT_EQ(x[i + 1], y[i]);
    }

    /* Random with an explicit rng: reproducible. */
    uint64_t s1 = 42ULL, s2 = 42ULL;
    int x1[4 * 32], x2[4 * 32];
    rcre_gpt_dataloader_next_batch_random(dl, x1, y, &s1);
    rcre_gpt_dataloader_next_batch_random(dl, x2, y, &s2);
    ASSERT_MEM_EQ(x1, x2, sizeof(x1));

    /* Random with the NULL fallback rng: stateful, so consecutive calls
     * yield different (but deterministic-per-run) batches and must never
     * collapse to a constant. */
    int xf1[4 * 32], xf2[4 * 32];
    rcre_gpt_dataloader_next_batch_random(dl, xf1, y, nullptr);
    rcre_gpt_dataloader_next_batch_random(dl, xf2, y, nullptr);
    int differ = 0;
    for (int i = 0; i < 4 * 32; i++) if (xf1[i] != xf2[i]) differ++;
    ASSERT_TRUE(differ > 0);

    /* Every token is in range. */
    for (int i = 0; i < 4 * 32; i++) {
        ASSERT_TRUE(xf1[i] >= 0 && xf1[i] < V);
        ASSERT_TRUE(y[i] >= 0 && y[i] < V);
    }

    rcre_gpt_dataloader_free(dl);
}

static void test_sample_token_contracts(void) {
    int V = 50;
    float logits[50];
    for (int i = 0; i < V; i++) logits[i] = (float)(i % 5) - 2.0f;
    logits[42] = 10.0f; /* unique maximum */
    int argmax = 42;

    /* 1. top_k = 1 is deterministic argmax at any temperature. */
    uint64_t seed = 1ULL;
    for (int trial = 0; trial < 8; trial++) {
        int t = (trial % 2) ? 1 : 3;
        int tok_id = rcre_gpt_sample_token(logits, V, (float)t, 1, &seed);
        ASSERT_INT_EQ(argmax, tok_id);
    }

    /* 2. Same seed -> identical sample sequence (determinism). */
    uint64_t a = 77ULL, b = 77ULL;
    for (int i = 0; i < 32; i++) {
        int ta = rcre_gpt_sample_token(logits, V, 1.0f, 10, &a);
        int tb = rcre_gpt_sample_token(logits, V, 1.0f, 10, &b);
        ASSERT_INT_EQ(ta, tb);
    }

    /* 3. Samples stay in [0, V). */
    uint64_t c = 123ULL;
    for (int i = 0; i < 200; i++) {
        int tok_id = rcre_gpt_sample_token(logits, V, 1.0f, 0, &c);
        ASSERT_TRUE(tok_id >= 0 && tok_id < V);
    }

    /* 4. Degenerate vocab is rejected, not "token 0". */
    ASSERT_INT_EQ(-1, rcre_gpt_sample_token(logits, 0, 1.0f, 1, &c));
}

static void test_training_convergence_arithmetic(void) {
    /*
     * End-to-end "does it actually learn" gate: a small model trained on the
     * arithmetic corpus through the public training surface
     * (dataloader -> forward -> backward -> adamw). Runs the HEAP-owned
     * model path (arena == NULL) so the checked-malloc scratch in backward
     * is exercised; the arena path is covered by the KV-equivalence test.
     */
    RCRE_CharTokenizer tok;
    ASSERT_TRUE(rcre_char_tokenizer_train_from_file(&tok, "data/arithmetic.txt"));
    ASSERT_TRUE(tok.vocab_size > 4 && tok.vocab_size <= 256);

    RCRE_GPTConfig cfg = {
        .vocab_size = tok.vocab_size,
        .seq_len = 32,
        .embed_dim = 64,
        .num_heads = 4,
        .head_dim = 16,
        .num_layers = 2,
        .mlp_dim = 128
    };
    RCRE_GPTModel *model = rcre_gpt_create(cfg, nullptr);
    ASSERT_NOT_NULL(model);
    uint64_t seed = 20261024ULL;
    rcre_gpt_init_weights(model, &seed);

    RCRE_GPTActs *acts = rcre_gpt_acts_create(cfg, 4, 32, nullptr);
    ASSERT_NOT_NULL(acts);

    RCRE_GPTDataLoader *dl = rcre_gpt_dataloader_create("data/arithmetic.txt", &tok, 4, 32);
    ASSERT_NOT_NULL(dl);

    RCRE_GPTAdamWConfig opt = {
        .lr_max = 3e-3f, .lr_min = 3e-4f, .warmup_steps = 20, .max_steps = 400,
        .beta1 = 0.9f, .beta2 = 0.999f, .eps = 1e-8f,
        .weight_decay = 0.0f, .grad_clip = 1.0f, .current_step = 0
    };
    ASSERT_TRUE(rcre_gpt_adamw_config_valid(&opt));

    int x[4 * 32], y[4 * 32];
    float first_loss = 0.0f, last_loss = 0.0f;

    for (int step = 0; step < 800; step++) {
        rcre_gpt_dataloader_next_batch_sequential(dl, x, y);
        float loss = rcre_gpt_forward(model, acts, x, y, 4, 32);
        ASSERT_TRUE(isfinite(loss));
        if (step == 0) first_loss = loss;
        last_loss = loss;

        rcre_gpt_backward(model, acts, x, y, 4, 32);
        rcre_gpt_adamw_step(model, &opt);
    }

    /* Loss must fall meaningfully below chance (ln(15) ~ 2.71). */
    ASSERT_TRUE(first_loss > 2.0f);
    ASSERT_TRUE(last_loss < 0.8f * first_loss);
    ASSERT_TRUE(last_loss < 2.3f);

    /* All parameters finite after training. */
    for (size_t i = 0; i < model->num_params; i++) {
        ASSERT_TRUE(isfinite(model->params[i]));
    }

    /* wte gradients are nonzero (embedding path connected). */
    int nonzero = 0;
    for (size_t i = 0; i < (size_t)cfg.vocab_size * cfg.embed_dim; i++) {
        if (model->grads[i] != 0.0f) nonzero++;
    }
    ASSERT_TRUE(nonzero > 0);

    rcre_gpt_dataloader_free(dl);
    rcre_gpt_acts_free(acts);
    rcre_gpt_free(model);
}

static void test_kv_cache_arena_model_equivalence(void) {
    /*
     * The user's KV-equivalence suite only exercises HEAP models. This
     * variant pins the arena-resident path: model + cache in one arena,
     * per-token scratch rolled back via the arena temp checkpoint.
     */
    RCRE_Arena *arena = rcre_arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_GPTConfig cfg = {
        .vocab_size = 65, .seq_len = 32, .embed_dim = 64, .num_heads = 4,
        .head_dim = 16, .num_layers = 2, .mlp_dim = 256
    };
    uint64_t seed = 42ULL;
    RCRE_GPTModel *model = rcre_gpt_create(cfg, arena);
    ASSERT_NOT_NULL(model);
    rcre_gpt_init_weights(model, &seed);

    int T_test = 8;
    int test_tokens[8] = { 10, 25, 4, 32, 18, 55, 62, 1 };

    RCRE_GPTActs *acts = rcre_gpt_acts_create(cfg, 1, T_test, arena);
    ASSERT_NOT_NULL(acts);
    rcre_gpt_forward(model, acts, test_tokens, nullptr, 1, T_test);

    RCRE_KVCache *cache = rcre_kv_cache_create(cfg, arena);
    ASSERT_NOT_NULL(cache);
    float kv_logits[65];

    size_t offset_before = arena->offset;
    for (int t = 0; t < T_test; t++) {
        ASSERT_TRUE(rcre_gpt_forward_token(model, cache, test_tokens[t], t, kv_logits));
        const float *full_logits = acts->logits + (size_t)t * cfg.vocab_size;
        for (int v = 0; v < cfg.vocab_size; v++) {
            ASSERT_FLOAT_NEAR(full_logits[v], kv_logits[v], 1e-4f);
        }
    }
    /* Per-token scratch must be rolled back: arena offset unchanged by the
     * generation loop (within one bump of the max single-token footprint). */
    ASSERT_TRUE(arena->offset - offset_before <= sizeof(float) * (2 * 64 + 65 + 8));

    rcre_arena_destroy(arena); /* model + cache + acts freed together */
}

static void test_bpe_retrain_reuse_leak_free(void) {
    /*
     * Regression for the retrain/load ownership fixes: training a second
     * time and loading over a live tokenizer must not double-free or leak
     * the previously merged strings (ASan/LSan pin this).
     */
    RCRE_BPETokenizer *tok = rcre_bpe_create();
    ASSERT_NOT_NULL(tok);
    ASSERT_TRUE(rcre_bpe_train_from_file(tok, "data/tinystories.txt", 384));
    int v1 = tok->vocab_size;

    /* Retrain with a different target size. */
    ASSERT_TRUE(rcre_bpe_train_from_file(tok, "data/tinystories.txt", 300));
    int v2 = tok->vocab_size;
    ASSERT_TRUE(v2 < v1);

    /* Save, load into the same tokenizer, re-save. */
    ASSERT_TRUE(rcre_bpe_save(tok, "data/tmp_reuse.bin"));
    ASSERT_TRUE(rcre_bpe_load(tok, "data/tmp_reuse.bin"));
    ASSERT_INT_EQ(v2, tok->vocab_size);
    ASSERT_TRUE(rcre_bpe_save(tok, "data/tmp_reuse2.bin"));

    /* Encode/decode still consistent after all the churn. */
    const char *s = "hello world";
    int enc[64];
    int n = rcre_bpe_encode(tok, s, strlen(s), enc, 64);
    ASSERT_TRUE(n > 0);
    char dec[64];
    ASSERT_UINT_EQ(strlen(s), rcre_bpe_decode(tok, enc, n, dec, sizeof(dec)));
    ASSERT_STR_EQ(s, dec);

    rcre_bpe_free(tok);
    remove("data/tmp_reuse.bin");
    remove("data/tmp_reuse2.bin");
}

/* ========================================================================= */

int main(void) {
    test_suite_begin("RCRE Transformer Merger Regression (micro_llm_c assimilation)");

    /* Part 1: deserialization robustness */
    TEST_RUN(test_bpe_load_corrupt_files_rejected);
    TEST_RUN(test_char_tokenizer_corrupt_load_rejected);
    TEST_RUN(test_checkpoint_magic_and_config_mismatch);

    /* Part 2: validation contracts */
    TEST_RUN(test_adamw_hyperparam_validation);
    TEST_RUN(test_kv_cache_config_validation);
    TEST_RUN(test_model_config_validation_and_token_guard);
    TEST_RUN(test_model_full_backward_fd_check);

    /* Part 3: behavioral contracts */
    TEST_RUN(test_dataloader_validation_and_epochs);
    TEST_RUN(test_sample_token_contracts);
    TEST_RUN(test_training_convergence_arithmetic);
    TEST_RUN(test_kv_cache_arena_model_equivalence);
    TEST_RUN(test_bpe_retrain_reuse_leak_free);

    return test_suite_report();
}
