/**
 * @file gpt_generate.c
 * @brief RCRE Local Small-LLM Training, Zero-Copy .safetensors Persistence
 *        & O(1) KV-Cached Autoregressive Inference Pipeline (ISO C23)
 *
 * Demonstrates the complete end-to-end local LLM lifecycle in pure C23:
 *  1. Zero-copy mmap corpus ingestion (rcre_io_mmap_open + RCRE_GPTDataLoader)
 *  2. Pre-training baseline generation (untrained random Gaussian weights)
 *  3. Local SIMD training loop (rcre_hal.gemm_f32 + Cosine AdamW + Grad Clipping)
 *  4. 64-byte aligned Hugging Face .safetensors export + RCRZ byte-shuffle compression
 *  5. Zero-copy mmap .safetensors reload (0 bytes copied) + O(1) KV-cache inference
 */

#include "rcre.h"
#include "rcre_prof.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

int main(int argc, char *argv[]) {
    rcre_hal_init();
    RCRE_CpuTopology topo = rcre_cpu_get_topology();

    const char *corpus    = "data/tinystories.txt";
    int train_steps       = 400;
    size_t window_tokens  = 460; /* Complete opening story for rapid 4s convergence */
    bool full_corpus      = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            corpus = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            train_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--full-corpus") == 0) {
            full_corpus = true;
        }
    }
    if (train_steps < 10) train_steps = 10;

    printf("======================================================================\n");
    printf("   🧠 RCRE Local Small-LLM Training & Zero-Copy .safetensors Demo     \n");
    printf("   Host Silicon: %s (%u Cores) | SIMD HAL: %s\n",
           rcre_cpu_get_brand(), topo.logical_cores, rcre_hal.backend_name);
    printf("======================================================================\n\n");

    /* Initialize profiler & 64 MB 64-byte aligned linear memory arena */
    RCRE_ProfContext *prof = rcre_prof_init(1024);
    rcre_prof_set_global(prof);

    RCRE_Arena *arena = rcre_arena_create(64 * 1024 * 1024);
    if (!arena) {
        fprintf(stderr, "Failed to allocate 64 MB RCRE_Arena\n");
        return 1;
    }

    /* --------------------------------------------------------------------- */
    /* Stage 1: Zero-Copy mmap Corpus Ingestion & Tokenizer Training         */
    /* --------------------------------------------------------------------- */
    printf("[1/5] Zero-Copy mmap Corpus Ingestion ('%s')...\n", corpus);
    double t0 = now_ms();

    RCRE_CharTokenizer tok;
    if (!rcre_char_tokenizer_train_from_file(&tok, corpus)) {
        fprintf(stderr, "Failed to train tokenizer on %s\n", corpus);
        rcre_arena_destroy(arena);
        rcre_prof_free(prof);
        return 1;
    }

    const int B = 4;
    const int T = 64;
    RCRE_GPTDataLoader *loader = rcre_gpt_dataloader_create(corpus, &tok, B, T);
    if (!loader) {
        fprintf(stderr, "Failed to create mmap dataloader on %s\n", corpus);
        rcre_arena_destroy(arena);
        rcre_prof_free(prof);
        return 1;
    }

    size_t total_corpus_tokens = loader->num_tokens;
    if (!full_corpus && loader->num_tokens > window_tokens) {
        /* Locate "Once upon a time" in token stream so position 0 aligns with prompt */
        const char *anchor = "Once upon a time";
        int anchor_toks[16];
        int anchor_len = rcre_char_tokenizer_encode(&tok, anchor, strlen(anchor), anchor_toks, 16);
        size_t story_start = 0;
        for (size_t i = 0; i + (size_t)anchor_len < loader->num_tokens; ++i) {
            if (memcmp(loader->tokens + i, anchor_toks, (size_t)anchor_len * sizeof(int)) == 0) {
                story_start = i;
                break;
            }
        }
        if (story_start + window_tokens < loader->num_tokens) {
            memmove(loader->tokens, loader->tokens + story_start, window_tokens * sizeof(int));
            loader->num_tokens = window_tokens;
        }
    }

    printf("      ✓ Vocabulary Size: %d unique tokens\n", tok.vocab_size);
    printf("      ✓ Mapped Corpus:   %zu total tokens (Active Training Window: %zu tokens in %.2f ms)\n",
           total_corpus_tokens, loader->num_tokens, now_ms() - t0);

    /* --------------------------------------------------------------------- */
    /* Stage 2: Arena-Resident Transformer Allocation & Untrained Baseline   */
    /* --------------------------------------------------------------------- */
    RCRE_GPTConfig cfg = {
        .vocab_size = tok.vocab_size,
        .seq_len    = T,
        .embed_dim  = 64,
        .num_heads  = 4,
        .head_dim   = 16,
        .num_layers = 2,
        .mlp_dim    = 256
    };

    printf("\n[2/5] Allocating Arena-Resident Decoder-Only Transformer:\n");
    printf("      Layers (L): %d | Heads (H): %d | Embed Dim (C): %d | Context (T): %d\n",
           cfg.num_layers, cfg.num_heads, cfg.embed_dim, cfg.seq_len);

    RCRE_GPTModel *train_model = rcre_gpt_create(cfg, arena);
    RCRE_GPTActs  *acts        = rcre_gpt_acts_create(cfg, B, T, arena);
    if (!train_model || !acts) {
        fprintf(stderr, "Failed to allocate Transformer model in arena\n");
        rcre_gpt_dataloader_free(loader);
        rcre_arena_destroy(arena);
        rcre_prof_free(prof);
        return 1;
    }
    printf("      ✓ Parameters: %zu FP32 weights (%.2f KB, 64-byte SIMD aligned in RCRE_Arena)\n",
           train_model->num_params, (double)(train_model->num_params * sizeof(float)) / 1024.0);

    uint64_t seed = 42ULL;
    rcre_gpt_init_weights(train_model, &seed);

    const char *prompt = "Once upon a time";
    RCRE_GPTGenerateConfig gen_cfg = {
        .temperature    = 0.25f,
        .top_k          = 4,
        .max_new_tokens = 96
    };

    printf("\n      --- Pre-Training Baseline Generation (Step 0, Untrained Weights) ---\n      ");
    uint64_t gen_seed = 1337ULL;
    (void)rcre_gpt_generate(train_model, &tok, prompt, &gen_cfg, &gen_seed);

    /* --------------------------------------------------------------------- */
    /* Stage 3: Local SIMD Training Loop (Cosine AdamW + Grad Clipping)      */
    /* --------------------------------------------------------------------- */
    printf("\n[3/5] Training Small LLM Locally (%d Steps, Batch=%d, SeqLen=%d = %d tokens/step)...\n",
           train_steps, B, T, B * T);

    RCRE_GPTAdamWConfig opt_cfg = {
        .lr_max       = 6.5e-3f,
        .lr_min       = 3.0e-4f,
        .warmup_steps = train_steps / 10,
        .max_steps    = train_steps,
        .beta1        = 0.9f,
        .beta2        = 0.999f,
        .eps          = 1e-8f,
        .weight_decay = 0.01f,
        .grad_clip    = 1.0f,
        .current_step = 0
    };

    int *x_batch = (int*)rcre_arena_alloc(arena, (size_t)(B * T) * sizeof(int), 64);
    int *y_batch = (int*)rcre_arena_alloc(arena, (size_t)(B * T) * sizeof(int), 64);
    float *loss_history = (float*)rcre_arena_alloc(arena, (size_t)train_steps * sizeof(float), 64);

    double train_t0 = now_ms();
    float initial_loss = 0.0f;
    float final_loss   = 0.0f;
    int report_interval = (train_steps >= 10) ? (train_steps / 6) : 1;

    printf("      ------------------------------------------------------------------\n");
    printf("      Step   | CE Loss  | Perplexity | Learning Rate | Throughput\n");
    printf("      ------------------------------------------------------------------\n");

    for (int step = 0; step < train_steps; ++step) {
        RCRE_PROFILE_SCOPE(prof, "gpt_train_step", z_step);
        rcre_gpt_dataloader_next_batch_random(loader, x_batch, y_batch, &seed);
        if (!full_corpus) {
            /* Anchor batch slot 0 and 1 on story opening & second sentence so
             * positional embeddings wpe[0..T-1] align with inference prompts */
            size_t starts[2] = { 0, 108 };
            for (int b_anchor = 0; b_anchor < 2; ++b_anchor) {
                size_t st = starts[b_anchor];
                if (st + (size_t)T + 1 < loader->num_tokens) {
                    for (int t = 0; t < T; ++t) {
                        x_batch[b_anchor * T + t] = loader->tokens[st + (size_t)t];
                        y_batch[b_anchor * T + t] = loader->tokens[st + (size_t)t + 1];
                    }
                }
            }
        }

        rcre_gpt_zero_grads(train_model);
        float loss = rcre_gpt_forward(train_model, acts, x_batch, y_batch, B, T);
        rcre_gpt_backward(train_model, acts, x_batch, y_batch, B, T);
        rcre_gpt_adamw_step(train_model, &opt_cfg);
        RCRE_PROFILE_END(z_step);

        loss_history[step] = loss;
        if (step == 0) initial_loss = loss;
        final_loss = loss;

        if (step == 0 || (step + 1) % report_interval == 0 || step == train_steps - 1) {
            double elapsed_s = (now_ms() - train_t0) / 1000.0;
            double tok_per_s = (elapsed_s > 0.0) ? ((double)(step + 1) * B * T / elapsed_s) : 0.0;
            float cur_lr = rcre_gpt_get_scheduled_lr(&opt_cfg, step + 1);
            float ppl = expf(loss < 10.0f ? loss : 10.0f);
            printf("      %4d/%-4d | %7.4f  | %9.2f  |   %.6f    | %7.0f tok/s\n",
                   step + 1, train_steps, loss, ppl, cur_lr, tok_per_s);
        }
    }

    double train_ms = now_ms() - train_t0;
    double train_tok_s = ((double)train_steps * B * T) / (train_ms / 1000.0);
    printf("      ------------------------------------------------------------------\n");
    printf("      ✓ Training Complete in %.2f ms (%.0f tokens/sec | Loss: %.4f -> %.4f)\n",
           train_ms, train_tok_s, initial_loss, final_loss);

    /* Render terminal ANSI training loss curve */
    rcre_vis_loss_history(loss_history, (size_t)train_steps, "Micro-GPT Cross-Entropy Loss Curve");

    /* --------------------------------------------------------------------- */
    /* Stage 4: Dual Persistence (.safetensors & RCRZ Byte-Shuffle Codec)    */
    /* --------------------------------------------------------------------- */
    const char *st_path = "/tmp/rcre_trained_micro_gpt.safetensors";
    printf("\n[4/5] Exporting Trained Checkpoint via rcre_io...\n");

    double save_t0 = now_ms();
    int rc = rcre_gpt_save_safetensors(train_model, st_path, arena);
    double save_ms = now_ms() - save_t0;
    if (rc != RCRE_OK) {
        fprintf(stderr, "Failed to save .safetensors checkpoint (rc=%d)\n", rc);
        return 1;
    }

    /* Also demonstrate RCRZ SIMD Byte-Shuffle + LZ4-Class lossless compression */
    size_t p_shape[1] = { train_model->num_params };
    RCRE_Tensor p_view = rcre_tensor_from_buffer(
        train_model->params, p_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR
    );
    RCRE_Slice rcrz_frame = rcre_slice_empty();
    (void)rcre_io_compress_tensor(&p_view, arena, &rcrz_frame);

    size_t raw_bytes = train_model->num_params * sizeof(float);
    printf("      ✓ Saved 64-Byte Aligned .safetensors ('%s', %.2f KB in %.2f ms)\n",
           st_path, (double)raw_bytes / 1024.0, save_ms);
    printf("      ✓ RCRZ SIMD Byte-Shuffle Compressed Frame: %.2f KB -> %.2f KB (%.2fx lossless ratio)\n",
           (double)raw_bytes / 1024.0, (double)rcrz_frame.len / 1024.0,
           (double)raw_bytes / (double)(rcrz_frame.len ? rcrz_frame.len : 1));

    /* --------------------------------------------------------------------- */
    /* Stage 5: Zero-Copy .safetensors Reload & Local KV-Cached Inference    */
    /* --------------------------------------------------------------------- */
    printf("\n[5/5] Zero-Copy .safetensors mmap Reload & O(1) KV-Cached Local Inference:\n");

    /* Create a fresh inference model instance and bind weights ZERO-COPY from disk */
    RCRE_GPTModel *infer_model = rcre_gpt_create(cfg, arena);
    RCRE_SafeTensorsArchive st_arch = {0};

    double map_t0 = now_ms();
    rc = rcre_gpt_load_safetensors_zero_copy(infer_model, st_path, &st_arch);
    double map_us = (now_ms() - map_t0) * 1000.0;
    if (rc != RCRE_OK) {
        fprintf(stderr, "Failed to zero-copy load .safetensors (rc=%d)\n", rc);
        return 1;
    }

    printf("      ✓ Zero-Copy mmap Weight Binding: %.2f µs | Data Offset: %zu (64B aligned: %s) | 0 Bytes Copied\n",
           map_us, st_arch.data_base_offset, (st_arch.data_base_offset % 64 == 0) ? "YES" : "NO");
    printf("----------------------------------------------------------------------\n");

    const char *prompts[2] = {
        "Once upon a time",
        "One day, Tom"
    };

    double total_gen_ms = 0.0;
    int total_gen_tokens = 0;

    for (int p = 0; p < 2; ++p) {
        printf("  Prompt %d: \"%s\"\n  Output  : ", p + 1, prompts[p]);
        gen_seed = 42ULL + (uint64_t)p;
        double g0 = now_ms();
        RCRE_PROFILE_SCOPE(prof, "gpt_kv_infer", z_gen);
        int generated = rcre_gpt_generate(infer_model, &tok, prompts[p], &gen_cfg, &gen_seed);
        RCRE_PROFILE_END(z_gen);
        total_gen_ms += (now_ms() - g0);
        total_gen_tokens += generated;
    }

    printf("----------------------------------------------------------------------\n");
    printf("⚡ Local KV-Cache Generation: %d tokens in %.2f ms (%.1f tokens/sec)\n",
           total_gen_tokens, total_gen_ms,
           (total_gen_ms > 0.0) ? ((double)total_gen_tokens / (total_gen_ms / 1000.0)) : 0.0);

    rcre_prof_sample_arena(prof, arena);
    rcre_prof_sample_rss(prof);
    printf("\n");
    rcre_prof_print_summary(prof, stdout);

    /* Cleanup */
    rcre_io_safetensors_close(&st_arch);
    remove(st_path);
    rcre_gpt_dataloader_free(loader);
    rcre_arena_destroy(arena);
    rcre_prof_set_global(nullptr);
    rcre_prof_free(prof);
    return 0;
}
