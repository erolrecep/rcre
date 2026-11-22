/**
 * @file test_amalgamated.c
 * @brief Verification of the single-header dist/rcre.h distribution
 */

#include "test_framework.h"

#define RCRE_IMPLEMENTATION
#include "../dist/rcre.h"

static void test_amalgamated_single_header(void) {
    /* Initialize hardware dispatch */
    rcre_hal_init();
    ASSERT_NOT_NULL(rcre_hal.backend_name);

    /* Create Arena */
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Tensor Allocation */
    size_t shape[3] = {10, 10, 3};
    RCRE_Tensor t = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(t.data);

    /* Rec.601 Grayscale Filter */
    size_t gray_shape[2] = {10, 10};
    RCRE_Tensor gray = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    int status = rcre_cv_rgb_to_gray(&t, &gray);
    ASSERT_INT_EQ(RCRE_OK, status);

    rcre_arena_destroy(arena);
}

static void test_amalgamated_transformer(void) {
    RCRE_GPTConfig cfg = {
        .vocab_size = 16,
        .seq_len = 8,
        .embed_dim = 16,
        .num_heads = 2,
        .head_dim = 8,
        .num_layers = 1,
        .mlp_dim = 32
    };

    RCRE_GPTModel *model = rcre_gpt_create(cfg, nullptr);
    ASSERT_NOT_NULL(model);

    uint64_t seed = 777ULL;
    rcre_gpt_init_weights(model, &seed);

    RCRE_KVCache *cache = rcre_kv_cache_create(cfg, nullptr);
    ASSERT_NOT_NULL(cache);

    float logits[16];
    rcre_gpt_forward_token(model, cache, 3, 0, logits);
    ASSERT_TRUE(logits[0] != 0.0f || logits[1] != 0.0f);

    rcre_kv_cache_free(cache);
    rcre_gpt_free(model);
}

static void test_amalgamated_scientific_io(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    float val = 0.0f;
    ASSERT_TRUE(rcre_slice_parse_f32(rcre_slice_from_cstr("-123.45e-2"), &val));
    ASSERT_FLOAT_NEAR(-1.2345f, val, 1e-6f);

    size_t shape[2] = {4, 4};
    RCRE_Tensor t = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    for (size_t i = 0; i < 16; ++i) rcre_tensor_f32_data(&t)[i] = (float)(i + 1);

    const char* npy_path = "/tmp/rcre_amalg_test.npy";
    ASSERT_INT_EQ(RCRE_OK, rcre_io_save_npy(&t, npy_path));

    RCRE_Tensor loaded = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_load_npy(npy_path, arena, &loaded));
    ASSERT_MEM_EQ(t.data, loaded.data, 16 * sizeof(float));
    remove(npy_path);

    RCRE_DLTensor dl = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_to_dlpack(&t, arena, &dl));
    ASSERT_TRUE(dl.data == t.data);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Single-Header (dist/rcre.h) Amalgamation Self-Test");
    TEST_RUN(test_amalgamated_single_header);
    TEST_RUN(test_amalgamated_transformer);
    TEST_RUN(test_amalgamated_scientific_io);
    return test_suite_report();
}
