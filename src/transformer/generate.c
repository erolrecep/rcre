/**
 * @file generate.c
 * @brief RCRE Autoregressive Sampling, Generation & Model Checkpointing (ISO C23)
 */

#include "transformer_internal.h"

int rcre_gpt_sample_token(const float *logits, int vocab_size, float temperature, int top_k, uint64_t *rng_state) {
    if (vocab_size <= 0) return -1;
    float *scaled = (float*)malloc(sizeof(float) * vocab_size);
    if (!scaled) return -1;

    float temp = (temperature > 0.0f) ? temperature : 1.0f;
    for (int v = 0; v < vocab_size; v++) {
        scaled[v] = logits[v] / temp;
    }

    /* Top-K filtering */
    if (top_k > 0 && top_k < vocab_size) {
        float *sorted = (float*)malloc(sizeof(float) * vocab_size);
        if (sorted) {
            memcpy(sorted, scaled, sizeof(float) * vocab_size);
            for (int i = 0; i < top_k; i++) {
                for (int j = i + 1; j < vocab_size; j++) {
                    if (sorted[j] > sorted[i]) {
                        float tmp = sorted[i];
                        sorted[i] = sorted[j];
                        sorted[j] = tmp;
                    }
                }
            }
            float threshold = sorted[top_k - 1];
            for (int v = 0; v < vocab_size; v++) {
                if (scaled[v] < threshold) {
                    scaled[v] = -1e9f;
                }
            }
            free(sorted);
        }
    }

    /* Softmax */
    float *probs = (float*)malloc(sizeof(float) * vocab_size);
    if (!probs) {
        free(scaled);
        return -1;
    }
    rcre_transformer_softmax_forward(probs, scaled, 1, vocab_size);

    /* Cumulative sampling */
    static uint64_t fallback_rng_state = 0x9E3779B97F4A7C15ULL; /* pi-64: deterministic fallback */
    float r;
    if (rng_state != nullptr) {
        r = rcre_rng_uniform_f32(rng_state, 0.0f, 1.0f);
    } else {
        r = rcre_rng_uniform_f32(&fallback_rng_state, 0.0f, 1.0f);
    }

    int selected = vocab_size - 1;
    float cum = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        cum += probs[v];
        if (cum >= r) {
            selected = v;
            break;
        }
    }

    free(scaled);
    free(probs);
    return selected;
}

int rcre_gpt_generate(
    RCRE_GPTModel *model,
    const RCRE_CharTokenizer *tok,
    const char *prompt,
    const RCRE_GPTGenerateConfig *gen_cfg,
    uint64_t *rng_state
) {
    if (!model || !tok || !gen_cfg || gen_cfg->max_new_tokens <= 0) return 0;
    int max_T = model->config.seq_len;
    int V = model->config.vocab_size;

    int *prompt_tokens = (int*)malloc(sizeof(int) * max_T);
    if (!prompt_tokens) return 0;
    int prompt_len = 0;

    if (prompt != nullptr && strlen(prompt) > 0) {
        prompt_len = rcre_char_tokenizer_encode(tok, prompt, strlen(prompt), prompt_tokens, max_T);
        if (prompt_len > max_T - 1) {
            prompt_len = max_T - 1;
        }
        printf("%s", prompt);
        fflush(stdout);
    } else {
        prompt_tokens[0] = 0;
        prompt_len = 1;
    }

    RCRE_KVCache *cache = rcre_kv_cache_create(model->config, model->arena);
    if (!cache) {
        free(prompt_tokens);
        return 0;
    }

    float *logits = (float*)malloc(sizeof(float) * V);
    if (!logits) {
        rcre_kv_cache_free(cache);
        free(prompt_tokens);
        return 0;
    }

    /* 1. Prefill Phase */
    for (int t = 0; t < prompt_len; t++) {
        if (!rcre_gpt_forward_token(model, cache, prompt_tokens[t], t, logits)) {
            printf("\n");
            free(logits);
            rcre_kv_cache_free(cache);
            free(prompt_tokens);
            return 0;
        }
    }

    /* 2. Autoregressive Generation */
    int current_pos = prompt_len;
    int generated = 0;
    for (int step = 0; step < gen_cfg->max_new_tokens && current_pos < max_T; step++) {
        int next_token = rcre_gpt_sample_token(logits, V, gen_cfg->temperature, gen_cfg->top_k, rng_state);
        if (next_token < 0) break; /* sampler failure (OOM): stop cleanly */

        char c_str[2] = {(char)tok->id_to_char[next_token], '\0'};
        printf("%s", c_str);
        fflush(stdout);
        generated++;

        if (!rcre_gpt_forward_token(model, cache, next_token, current_pos, logits)) break;
        current_pos++;
    }
    printf("\n");

    free(logits);
    rcre_kv_cache_free(cache);
    free(prompt_tokens);
    return generated;
}

bool rcre_gpt_save_checkpoint(const RCRE_GPTModel *model, const char *filepath) {
    if (!model) return false;
    FILE *f = fopen(filepath, "wb");
    if (!f) return false;
    const uint32_t magic = 0x31505447u; /* "GPT1" */
    const uint8_t version = RCRE_GPT_CKPT_FORMAT_VERSION;
    if (fwrite(&magic, sizeof(magic), 1, f) != 1) { fclose(f); return false; }
    if (fwrite(&version, sizeof(version), 1, f) != 1) { fclose(f); return false; }
    if (fwrite(&model->config, sizeof(RCRE_GPTConfig), 1, f) != 1) { fclose(f); return false; }
    if (fwrite(&model->num_params, sizeof(size_t), 1, f) != 1) { fclose(f); return false; }
    if (fwrite(model->params, sizeof(float), model->num_params, f) != model->num_params) { fclose(f); return false; }
    fclose(f);
    return true;
}

bool rcre_gpt_load_checkpoint(RCRE_GPTModel *model, const char *filepath) {
    if (!model) return false;
    FILE *f = fopen(filepath, "rb");
    if (!f) return false;
    uint32_t magic = 0;
    uint8_t version = 0;
    RCRE_GPTConfig cfg;
    bool ok = (fread(&magic, sizeof(magic), 1, f) == 1) &&
              (fread(&version, sizeof(version), 1, f) == 1) &&
              (fread(&cfg, sizeof(RCRE_GPTConfig), 1, f) == 1);
    if (ok) {
        /* The weights are laid out by the config: every architectural field
         * must match the live model before any float is written. */
        ok = (magic == 0x31505447u) && (version == RCRE_GPT_CKPT_FORMAT_VERSION) &&
             cfg.vocab_size == model->config.vocab_size &&
             cfg.seq_len == model->config.seq_len &&
             cfg.embed_dim == model->config.embed_dim &&
             cfg.num_heads == model->config.num_heads &&
             cfg.num_layers == model->config.num_layers &&
             cfg.mlp_dim == model->config.mlp_dim;
    }
    size_t count = 0;
    if (ok) ok = (fread(&count, sizeof(size_t), 1, f) == 1) && (count == model->num_params);
    if (ok) ok = (fread(model->params, sizeof(float), count, f) == count);
    fclose(f);
    return ok;
}
