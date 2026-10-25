/**
 * @file dataset.c
 * @brief RCRE Dataset & Autoregressive DataLoader Subsystem (ISO C23)
 */

#include "transformer_internal.h"

RCRE_GPTDataLoader* rcre_gpt_dataloader_create(const char *filepath, const RCRE_CharTokenizer *tok, int batch_size, int seq_len) {
    if (!tok || tok->vocab_size <= 0 || batch_size <= 0 || seq_len < 2) {
        return nullptr;
    }
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(f);
        return nullptr;
    }

    char *raw_text = (char*)malloc((size_t)file_size);
    if (!raw_text) {
        fclose(f);
        return nullptr;
    }

    size_t read_bytes = fread(raw_text, 1, (size_t)file_size, f);
    fclose(f);

    RCRE_GPTDataLoader *loader = (RCRE_GPTDataLoader*)malloc(sizeof(RCRE_GPTDataLoader));
    if (!loader) {
        free(raw_text);
        return nullptr;
    }

    loader->batch_size = batch_size;
    loader->seq_len = seq_len;
    loader->current_idx = 0;
    loader->tokens = (int*)malloc(sizeof(int) * read_bytes);
    if (!loader->tokens) {
        free(raw_text);
        free(loader);
        return nullptr;
    }

    int valid_tokens = rcre_char_tokenizer_encode(tok, raw_text, read_bytes, loader->tokens, (int)read_bytes);
    free(raw_text);

    loader->num_tokens = (size_t)valid_tokens;
    if (loader->num_tokens <= (size_t)(seq_len + 1)) {
        free(loader->tokens);
        free(loader);
        return nullptr;
    }

    return loader;
}

void rcre_gpt_dataloader_next_batch_random(RCRE_GPTDataLoader *loader, int *x_batch, int *y_batch, uint64_t *rng_state) {
    if (!loader || !x_batch || !y_batch) return;
    /* Deterministic internal fallback so rng_state == NULL is reproducible
     * (unified-RNG policy: no libc rand()). */
    static uint64_t fallback_rng_state = 0x1234567890ABCDEFULL;
    uint64_t *rng = rng_state ? rng_state : &fallback_rng_state;
    size_t max_start = loader->num_tokens - (size_t)loader->seq_len - 1;
    for (int b = 0; b < loader->batch_size; b++) {
        size_t start = (size_t)(rcre_rng_next(rng) % max_start);
        for (int t = 0; t < loader->seq_len; t++) {
            x_batch[b * loader->seq_len + t] = loader->tokens[start + t];
            y_batch[b * loader->seq_len + t] = loader->tokens[start + t + 1];
        }
    }
}

void rcre_gpt_dataloader_next_batch_sequential(RCRE_GPTDataLoader *loader, int *x_batch, int *y_batch) {
    if (!loader || !x_batch || !y_batch) return;
    size_t max_start = loader->num_tokens - (size_t)loader->seq_len - 1;
    for (int b = 0; b < loader->batch_size; b++) {
        if (loader->current_idx >= max_start) {
            loader->current_idx = 0;
        }
        size_t start = loader->current_idx;
        for (int t = 0; t < loader->seq_len; t++) {
            x_batch[b * loader->seq_len + t] = loader->tokens[start + t];
            y_batch[b * loader->seq_len + t] = loader->tokens[start + t + 1];
        }
        loader->current_idx += (size_t)loader->seq_len;
    }
}

void rcre_gpt_dataloader_reset(RCRE_GPTDataLoader *loader) {
    if (loader) {
        loader->current_idx = 0;
    }
}

void rcre_gpt_dataloader_free(RCRE_GPTDataLoader *loader) {
    if (loader) {
        if (loader->tokens) {
            free(loader->tokens);
        }
        free(loader);
    }
}
