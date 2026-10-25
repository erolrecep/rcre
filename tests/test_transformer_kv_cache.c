/**
 * @file test_transformer_kv_cache.c
 * @brief Numerical Equivalence Verification for KV-Cache vs Full Sequence Attention (ISO C23)
 */

#include "rcre.h"
#include "test_framework.h"

static void test_kv_cache_equivalence(void) {
    RCRE_GPTConfig cfg = {
        .vocab_size = 65,
        .seq_len = 32,
        .embed_dim = 64,
        .num_heads = 4,
        .head_dim = 16,
        .num_layers = 2,
        .mlp_dim = 256
    };

    uint64_t seed = 42ULL;
    RCRE_GPTModel *model = rcre_gpt_create(cfg, nullptr);
    ASSERT_NOT_NULL(model);
    rcre_gpt_init_weights(model, &seed);

    int T_test = 8;
    int test_tokens[8] = {10, 25, 4, 32, 18, 55, 62, 1};

    /* 1. Full Sequence Forward Pass */
    RCRE_GPTActs *acts = rcre_gpt_acts_create(cfg, 1, T_test, nullptr);
    ASSERT_NOT_NULL(acts);
    rcre_gpt_forward(model, acts, test_tokens, nullptr, 1, T_test);

    /* 2. Incremental KV-Cached Single-Token Forward Pass */
    RCRE_KVCache *cache = rcre_kv_cache_create(cfg, nullptr);
    ASSERT_NOT_NULL(cache);

    float *kv_logits = (float*)malloc(sizeof(float) * cfg.vocab_size);
    ASSERT_NOT_NULL(kv_logits);

    float max_diff = 0.0f;

    for (int t = 0; t < T_test; t++) {
        rcre_gpt_forward_token(model, cache, test_tokens[t], t, kv_logits);

        const float *full_logits = acts->logits + t * cfg.vocab_size;

        for (int v = 0; v < cfg.vocab_size; v++) {
            float diff = fabsf(kv_logits[v] - full_logits[v]);
            if (diff > max_diff) max_diff = diff;
            ASSERT_FLOAT_NEAR(full_logits[v], kv_logits[v], 1e-4f);
        }
    }

    ASSERT_TRUE(max_diff < 1e-4f);

    /* Test cache reset */
    rcre_kv_cache_reset(cache);
    ASSERT_INT_EQ(0, cache->current_pos);

    free(kv_logits);
    rcre_kv_cache_free(cache);
    rcre_gpt_acts_free(acts);
    rcre_gpt_free(model);
}

static void test_kv_cache_arena_resident(void) {
    RCRE_Arena *arena = rcre_arena_create(8 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_GPTConfig cfg = {
        .vocab_size = 32,
        .seq_len = 16,
        .embed_dim = 32,
        .num_heads = 2,
        .head_dim = 16,
        .num_layers = 1,
        .mlp_dim = 64
    };

    uint64_t seed = 999ULL;
    RCRE_GPTModel *model = rcre_gpt_create(cfg, arena);
    ASSERT_NOT_NULL(model);
    rcre_gpt_init_weights(model, &seed);

    RCRE_KVCache *cache = rcre_kv_cache_create(cfg, arena);
    ASSERT_NOT_NULL(cache);

    float logits[32];
    rcre_gpt_forward_token(model, cache, 5, 0, logits);

    /* Arena destruction safely frees model, cache and all memory at once */
    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Key-Value (KV) Cache Mathematical Equivalence & Arena Residency");
    TEST_RUN(test_kv_cache_equivalence);
    TEST_RUN(test_kv_cache_arena_resident);
    return test_suite_report();
}
