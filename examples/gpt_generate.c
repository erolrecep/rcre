/**
 * @file gpt_generate.c
 * @brief RCRE Generative AI & Micro-GPT Autoregressive Generation Demo (ISO C23)
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    printf("======================================================================\n");
    printf("   🧠 RCRE Micro-GPT Autoregressive Transformer Generation Demo       \n");
    printf("======================================================================\n\n");

    const char *corpus = "data/tinystories.txt";
    if (argc > 1) {
        corpus = argv[1];
    }

    /* 1. Train Character Tokenizer */
    printf("[1/4] Training Character Tokenizer on '%s'...\n", corpus);
    RCRE_CharTokenizer tok;
    if (!rcre_char_tokenizer_train_from_file(&tok, corpus)) {
        fprintf(stderr, "Failed to train tokenizer on %s\n", corpus);
        return 1;
    }
    printf("      Vocabulary Size: %d unique tokens\n", tok.vocab_size);

    /* 2. Configure Model Architecture */
    RCRE_GPTConfig cfg = {
        .vocab_size = tok.vocab_size,
        .seq_len = 64,
        .embed_dim = 64,
        .num_heads = 4,
        .head_dim = 16,
        .num_layers = 2,
        .mlp_dim = 256
    };

    printf("\n[2/4] Allocating Transformer Model Architecture:\n");
    printf("      Layers (L): %d | Heads (H): %d | Embed Dim (C): %d | Context (T): %d\n",
           cfg.num_layers, cfg.num_heads, cfg.embed_dim, cfg.seq_len);

    RCRE_GPTModel *model = rcre_gpt_create(cfg, nullptr);
    if (!model) {
        fprintf(stderr, "Failed to allocate model\n");
        return 1;
    }
    printf("      Total Parameters: %zu floats (~%.2f KB)\n",
           model->num_params, (double)(model->num_params * sizeof(float)) / 1024.0);

    /* 3. Initialize Gaussian Weights */
    printf("\n[3/4] Initializing Weights (Gaussian std = 0.02)...\n");
    uint64_t seed = 42ULL;
    rcre_gpt_init_weights(model, &seed);

    /* 4. Autoregressive Text Generation with KV-Cache */
    const char *prompt = "Once upon a time";
    printf("\n[4/4] Generating Text with Inference KV-Cache:\n");
    printf("      Prompt: \"%s\"\n", prompt);
    printf("----------------------------------------------------------------------\n");

    RCRE_GPTGenerateConfig gen_cfg = {
        .temperature = 0.8f,
        .top_k = 10,
        .max_new_tokens = 40
    };

    clock_t start = clock();
    rcre_gpt_generate(model, &tok, prompt, &gen_cfg, &seed);
    clock_t end = clock();

    double elapsed_ms = ((double)(end - start) / (double)CLOCKS_PER_SEC) * 1000.0;
    printf("----------------------------------------------------------------------\n");
    printf("Generation Complete in %.2f ms (%.1f tokens/sec)\n\n",
           elapsed_ms, (double)gen_cfg.max_new_tokens / (elapsed_ms / 1000.0));

    rcre_gpt_free(model);
    return 0;
}
