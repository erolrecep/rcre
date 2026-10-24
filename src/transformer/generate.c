/**
 * @file generate.c
 * @brief RCRE Autoregressive Sampling, Generation & Model Checkpointing (ISO C23)
 */

#include "transformer_internal.h"

int rcre_gpt_sample_token(const float *logits, int vocab_size, float temperature, int top_k, uint64_t *rng_state) {
    float *scaled = (float*)malloc(sizeof(float) * vocab_size);
    if (!scaled) return 0;

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
        return 0;
    }
    rcre_transformer_softmax_forward(probs, scaled, 1, vocab_size);

    /* Cumulative sampling */
    float r;
    if (rng_state != nullptr) {
        r = rcre_rng_uniform_f32(rng_state, 0.0f, 1.0f);
    } else {
        r = ((float)rand()) / ((float)RAND_MAX + 1.0f);
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

void rcre_gpt_generate(
    RCRE_GPTModel *model,
    const RCRE_CharTokenizer *tok,
    const char *prompt,
    const RCRE_GPTGenerateConfig *gen_cfg,
    uint64_t *rng_state
) {
    int max_T = model->config.seq_len;
    int V = model->config.vocab_size;

    int *prompt_tokens = (int*)malloc(sizeof(int) * max_T);
    if (!prompt_tokens) return;
    int prompt_len = 0;

    if (prompt != nullptr && strlen(prompt) > 0) {
        prompt_len = rcre_char_tokenizer_encode(tok, prompt, strlen(prompt), prompt_tokens);
        if (prompt_len > max_T - 1) {
            prompt_len = max_T - 1;
        }
        printf("%s", prompt);
        fflush(stdout);
    } else {
        prompt_tokens[0] = 0;
        prompt_len = 1;
    }

    RCRE_KVCache *cache = rcre_kv_cache_create(model->config, nullptr);
    if (!cache) {
        free(prompt_tokens);
        return;
    }

    float *logits = (float*)malloc(sizeof(float) * V);
    if (!logits) {
        rcre_kv_cache_free(cache);
        free(prompt_tokens);
        return;
    }

    /* 1. Prefill Phase */
    for (int t = 0; t < prompt_len; t++) {
        rcre_gpt_forward_token(model, cache, prompt_tokens[t], t, logits);
    }

    /* 2. Autoregressive Generation */
    int current_pos = prompt_len;
    for (int step = 0; step < gen_cfg->max_new_tokens && current_pos < max_T; step++) {
        int next_token = rcre_gpt_sample_token(logits, V, gen_cfg->temperature, gen_cfg->top_k, rng_state);

        char c_str[2] = {(char)tok->id_to_char[next_token], '\0'};
        printf("%s", c_str);
        fflush(stdout);

        rcre_gpt_forward_token(model, cache, next_token, current_pos, logits);
        current_pos++;
    }
    printf("\n");

    free(logits);
    rcre_kv_cache_free(cache);
    free(prompt_tokens);
}

bool rcre_gpt_save_checkpoint(const RCRE_GPTModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return false;
    if (fwrite(&model->config, sizeof(RCRE_GPTConfig), 1, f) != 1) { fclose(f); return false; }
    if (fwrite(&model->num_params, sizeof(size_t), 1, f) != 1) { fclose(f); return false; }
    if (fwrite(model->params, sizeof(float), model->num_params, f) != model->num_params) { fclose(f); return false; }
    fclose(f);
    return true;
}

bool rcre_gpt_load_checkpoint(RCRE_GPTModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return false;
    RCRE_GPTConfig cfg;
    if (fread(&cfg, sizeof(RCRE_GPTConfig), 1, f) != 1) { fclose(f); return false; }
    size_t count;
    if (fread(&count, sizeof(size_t), 1, f) != 1 || count != model->num_params) { fclose(f); return false; }
    if (fread(model->params, sizeof(float), count, f) != count) { fclose(f); return false; }
    fclose(f);
    return true;
}
