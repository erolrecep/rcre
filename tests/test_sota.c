/**
 * @file test_sota.c
 * @brief Comprehensive Verification Suite for RCRE v2.0.0 SOTA Capabilities (Phases 0-5)
 */

#include "rcre.h"
#include "rcre_llm.h"
#include "rcre_codec.h"
#include "extras/rcre_gpu.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_phase0_1_abi_nd8_broadcast_hal(void) {
    assert(rcre_abi_version() == RCRE_ABI_VERSION);
    assert(rcre_abi_version() >= 0x020000U);

    RCRE_Arena* arena = rcre_arena_create(8 * 1024 * 1024);
    assert(arena != nullptr);

    /* 1. 6D Tensor + NumPy Broadcasting */
    size_t s6[6] = {2, 1, 2, 1, 3, 2};
    RCRE_Tensor t6 = rcre_tensor_create(s6, 6, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    assert(t6.data != nullptr && t6.ndim == 6 && rcre_tensor_numel(&t6) == 24);

    size_t sa[3] = {2, 1, 4};
    size_t sb[3] = {1, 3, 4};
    RCRE_Tensor a = rcre_tensor_create(sa, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor b = rcre_tensor_create(sb, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* ad = rcre_tensor_f32_data(&a);
    float* bd = rcre_tensor_f32_data(&b);
    for (size_t i = 0; i < 8; ++i) ad[i] = (float)(i + 1);
    for (size_t i = 0; i < 12; ++i) bd[i] = (float)(i + 1) * 0.5f;

    RCRE_Tensor out = {0};
    assert(rcre_tensor_add_broadcast(&a, &b, arena, &out) == RCRE_OK);
    assert(out.ndim == 3 && out.shape[0] == 2 && out.shape[1] == 3 && out.shape[2] == 4);
    float* od = rcre_tensor_f32_data(&out);
    assert(fabsf(od[0] - 1.5f) < 1e-5f);
    assert(fabsf(od[1 * 12 + 2 * 4 + 3] - (ad[7] + bd[11])) < 1e-5f);

    /* 2. HAL GEMV, INT8 DOT/GEMM, BF16 GEMM, Multi-Threaded SGEMM */
    float M_a[12] = {1,2,3,4, 5,6,7,8, -1,0,2,1};
    float x_v[4]  = {1,-1,2,0.5f};
    float y_v[3]  = {0};
    rcre_hal.gemv_f32(3, 4, 1.0f, M_a, 4, x_v, 0.0f, y_v);
    assert(fabsf(y_v[0] - (1 - 2 + 6 + 2)) < 1e-4f);
    assert(fabsf(y_v[1] - (5 - 6 + 14 + 4)) < 1e-4f);
    assert(fabsf(y_v[2] - (-1 + 0 + 4 + 0.5f)) < 1e-4f);

    int8_t iq_a[8] = {1,2,-3,4, -5,6,7,-8};
    int8_t iq_b[8] = {2,-1,4,0, 1,2,-1,3};
    assert(rcre_hal.dot_i8(iq_a, iq_b, 8) == (2 - 2 - 12 + 0 - 5 + 12 - 7 - 24));

    int8_t imA[4] = {1, 2, 3, 4};
    int8_t imB[4] = {5, 6, 7, 8};
    int32_t imC[4] = {0};
    rcre_hal.gemm_i8(2, 2, 2, imA, 2, imB, 2, imC, 2);
    assert(imC[0] == 19 && imC[1] == 22 && imC[2] == 43 && imC[3] == 50);

    uint16_t bfA[4], bfB[4];
    float fA[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float fB[4] = {0.5f, 1.0f, -1.0f, 2.0f};
    float bfC[4] = {0};
    rcre_f32_to_bf16(fA, bfA, 4);
    rcre_f32_to_bf16(fB, bfB, 4);
    rcre_hal.gemm_bf16(2, 2, 2, 1.0f, bfA, 2, bfB, 2, 0.0f, bfC, 2);
    assert(fabsf(bfC[0] - (-1.5f)) < 1e-3f);
    assert(fabsf(bfC[1] - 5.0f) < 1e-3f);

    RCRE_ThreadPool* pool = rcre_threadpool_create(2);
    const size_t N = 64;
    float* sgA = (float*)rcre_arena_alloc(arena, N * N * sizeof(float), 64);
    float* sgB = (float*)rcre_arena_alloc(arena, N * N * sizeof(float), 64);
    float* sgC1 = (float*)rcre_arena_alloc(arena, N * N * sizeof(float), 64);
    float* sgC2 = (float*)rcre_arena_alloc(arena, N * N * sizeof(float), 64);
    for (size_t i = 0; i < N * N; ++i) { sgA[i] = 0.01f * (float)(i % 13); sgB[i] = 0.02f * (float)(i % 7); }
    rcre_hal.gemm_f32(N, N, N, 1.0f, sgA, N, sgB, N, 0.0f, sgC1, N);
    rcre_hal_gemm_f32_mt(pool, N, N, N, 1.0f, sgA, N, sgB, N, 0.0f, sgC2, N);
    for (size_t i = 0; i < N * N; ++i) assert(fabsf(sgC1[i] - sgC2[i]) < 1e-4f);
    rcre_threadpool_destroy(pool);

    /* 3. CSV RFC-4180 Quotes */
    const char* qcsv = "\"1.5\",\"2.25\"\n\"-3.5\",\"4.0\"\n";
    RCRE_CSVOptions csv_opt = { .delimiter = ',', .quote_char = '"', .strict_numbers = true };
    RCRE_Tensor csv_t = {0};
    assert(rcre_io_parse_csv_slice(rcre_slice_from_cstr(qcsv), &csv_opt, arena, &csv_t) == RCRE_OK);
    float* cd = rcre_tensor_f32_data(&csv_t);
    assert(fabsf(cd[0] - 1.5f) < 1e-5f && fabsf(cd[2] - (-3.5f)) < 1e-5f);

    /* 4. Async POSIX TensorStream */
    size_t st_sh[2] = {10, 4};
    RCRE_Tensor st_src = rcre_tensor_create(st_sh, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    for (size_t i = 0; i < 40; ++i) rcre_tensor_f32_data(&st_src)[i] = (float)i;
    const char* npy_p = "/tmp/rcre_test_async_stream.npy";
    assert(rcre_io_save_npy(&st_src, npy_p) == RCRE_OK);

    RCRE_TensorStream ts;
    assert(rcre_io_stream_open_npy(npy_p, 3, arena, &ts) == RCRE_OK);
    assert(rcre_io_stream_enable_async(&ts) == RCRE_OK);
    size_t seen_rows = 0;
    RCRE_Tensor batch = {0};
    while (rcre_io_stream_next(&ts, &batch)) {
        const float* bd_s = (const float*)batch.data;
        for (size_t r = 0; r < batch.shape[0]; ++r) {
            assert(fabsf(bd_s[r * 4] - (float)((seen_rows + r) * 4)) < 1e-5f);
        }
        seen_rows += batch.shape[0];
    }
    assert(seen_rows == 10);
    rcre_io_stream_close(&ts);
    remove(npy_p);

    rcre_arena_destroy(arena);
    printf("  ✓ [PASS] test_phase0_1_abi_nd8_broadcast_hal\n");
}

static void test_phase2_quant_gguf_bpe_llm_sampler(void) {
    /* 1. Q8_0 & Q4_0 Quantization + Dot */
    float w[64], x[64];
    for (size_t i = 0; i < 64; ++i) {
        w[i] = sinf((float)i * 0.2f);
        x[i] = cosf((float)i * 0.1f);
    }
    float ref_dot = rcre_hal.dot_f32(w, x, 64);

    RCRE_BlockQ8_0 q8[2];
    rcre_quantize_q8_0(w, q8, 64);
    float d_q8 = rcre_vec_dot_q8_0(q8, x, 64);
    assert(fabsf(d_q8 - ref_dot) < 0.05f);

    RCRE_BlockQ4_0 q4[2];
    rcre_quantize_q4_0(w, q4, 64);
    float d_q4 = rcre_vec_dot_q4_0(q4, x, 64);
    assert(fabsf(d_q4 - ref_dot) < 0.35f);

    /* 2. Flash GQA + RMSNorm + RoPE + SwiGLU */
    float q[4] = {1.0f, 0.0f, 0.5f, -0.5f};
    float k[4] = {1.0f, 0.0f, 0.5f, -0.5f};
    float v[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float aout[4] = {0};
    rcre_llm_rope(q, k, 2, 2, 2, 3, 10000.0f);
    rcre_llm_flash_attention_gqa(aout, q, k, v, 2, 2, 2, 1, 4);
    assert(fabsf(aout[0] - 2.0f) < 1e-4f && fabsf(aout[3] - 5.0f) < 1e-4f);

    /* 3. GGUF v3 Save + Open + rcre_llama_load_gguf + Generation */
    RCRE_LlamaConfig cfg = {
        .vocab_size = 64, .dim = 32, .hidden_dim = 64,
        .n_layers = 1, .n_heads = 4, .n_kv_heads = 2,
        .max_seq_len = 32, .rms_eps = 1e-5f, .rope_theta = 10000.0f
    };
    RCRE_LlamaModel* orig = rcre_llama_create(cfg, RCRE_LLM_W_Q8_0, 77ULL);
    assert(orig != nullptr);

    size_t kv_dim = 16;
    size_t rb_dim = ((cfg.dim + 31U) / 32U) * sizeof(RCRE_BlockQ8_0);
    size_t rb_hid = ((cfg.hidden_dim + 31U) / 32U) * sizeof(RCRE_BlockQ8_0);

    RCRE_GGUFWriteKV kvs[8] = {
        { .key = "llama.embedding_length", .u32_val = 32 },
        { .key = "llama.feed_forward_length", .u32_val = 64 },
        { .key = "llama.block_count", .u32_val = 1 },
        { .key = "llama.attention.head_count", .u32_val = 4 },
        { .key = "llama.attention.head_count_kv", .u32_val = 2 },
        { .key = "llama.vocab_size", .u32_val = 64 },
        { .key = "llama.context_length", .u32_val = 32 },
        { .key = "general.name", .str_val = "rcre-tiny-q8", .is_str = true }
    };
    RCRE_GGUFWriteTensor ts[12] = {
        { "token_embd.weight", RCRE_GGML_TYPE_F32, 2, {32, 64, 0, 0}, orig->tok_emb, 64 * 32 * 4 },
        { "output_norm.weight", RCRE_GGML_TYPE_F32, 1, {32, 0, 0, 0}, orig->rms_final, 32 * 4 },
        { "output.weight", RCRE_GGML_TYPE_Q8_0, 2, {32, 64, 0, 0}, orig->lm_head, 64 * rb_dim },
        { "blk.0.attn_norm.weight", RCRE_GGML_TYPE_F32, 1, {32, 0, 0, 0}, orig->layers[0].rms_att, 32 * 4 },
        { "blk.0.ffn_norm.weight", RCRE_GGML_TYPE_F32, 1, {32, 0, 0, 0}, orig->layers[0].rms_ffn, 32 * 4 },
        { "blk.0.attn_q.weight", RCRE_GGML_TYPE_Q8_0, 2, {32, 32, 0, 0}, orig->layers[0].wq, 32 * rb_dim },
        { "blk.0.attn_k.weight", RCRE_GGML_TYPE_Q8_0, 2, {32, kv_dim, 0, 0}, orig->layers[0].wk, kv_dim * rb_dim },
        { "blk.0.attn_v.weight", RCRE_GGML_TYPE_Q8_0, 2, {32, kv_dim, 0, 0}, orig->layers[0].wv, kv_dim * rb_dim },
        { "blk.0.attn_output.weight", RCRE_GGML_TYPE_Q8_0, 2, {32, 32, 0, 0}, orig->layers[0].wo, 32 * rb_dim },
        { "blk.0.ffn_gate.weight", RCRE_GGML_TYPE_Q8_0, 2, {32, 64, 0, 0}, orig->layers[0].w_gate, 64 * rb_dim },
        { "blk.0.ffn_up.weight", RCRE_GGML_TYPE_Q8_0, 2, {32, 64, 0, 0}, orig->layers[0].w_up, 64 * rb_dim },
        { "blk.0.ffn_down.weight", RCRE_GGML_TYPE_Q8_0, 2, {64, 32, 0, 0}, orig->layers[0].w_down, 32 * rb_hid }
    };
    const char* gguf_p = "/tmp/rcre_test_model.gguf";
    assert(rcre_gguf_save_v3(gguf_p, kvs, 8, ts, 12) == RCRE_OK);
    rcre_llama_free(orig);

    int err = 0;
    RCRE_LlamaModel* loaded = rcre_llama_load_gguf(gguf_p, &err);
    assert(loaded != nullptr && err == RCRE_OK);
    RCRE_LlamaState* state = rcre_llama_state_create(&loaded->cfg);
    RCRE_Sampler* sampler = rcre_sampler_create(loaded->cfg.vocab_size, (RCRE_SamplerConfig){
        .temperature = 0.7f, .top_k = 16, .top_p = 0.9f, .min_p = 0.05f, .repeat_penalty = 1.1f, .rng_state = 99ULL
    });
    int32_t prompt[3] = {1, 5, 9};
    int32_t out_toks[8] = {0};
    size_t gen_n = 0;
    assert(rcre_llama_generate(loaded, state, sampler, prompt, 3, out_toks, 8, &gen_n) == RCRE_OK);
    assert(gen_n == 8);

    rcre_sampler_free(sampler);
    rcre_llama_state_free(state);
    rcre_llama_free(loaded);
    remove(gguf_p);

    /* 4. Large-Vocab ByteBPE */
    RCRE_ByteBPE* bpe = rcre_byte_bpe_create(2500);
    const char* corpus = "the quick brown fox jumps over the lazy dog the quick fox runs fast ";
    assert(rcre_byte_bpe_train_corpus(bpe, (const uint8_t*)corpus, strlen(corpus), 300) == RCRE_OK);
    int32_t ids[128];
    size_t n_ids = rcre_byte_bpe_encode(bpe, "the quick brown fox", 19, ids, 128);
    assert(n_ids > 0 && n_ids < 19);
    char dec[128] = {0};
    (void)rcre_byte_bpe_decode(bpe, ids, n_ids, dec, sizeof(dec));
    assert(strcmp(dec, "the quick brown fox") == 0);
    rcre_byte_bpe_free(bpe);

    printf("  ✓ [PASS] test_phase2_quant_gguf_bpe_llm_sampler\n");
}

static void test_phase3_lora_bf16_ckpt_autograd_optim(void) {
    /* 1. LoRA fine-tuning convergence */
    float base_W[16] = {0};
    for (int i = 0; i < 4; ++i) base_W[i * 4 + i] = 1.0f;
    float x[4] = {1.0f, -0.5f, 0.25f, 2.0f};
    float y_target[4] = {2.0f, 1.0f, -1.0f, 0.5f};

    RCRE_LoRAAdapter* lora = rcre_lora_create(2, 4, 4, 4.0f, 123ULL);
    float loss0 = rcre_lora_train_step(lora, base_W, x, y_target, 0.05f);
    float loss_end = loss0;
    for (int s = 0; s < 30; ++s) loss_end = rcre_lora_train_step(lora, base_W, x, y_target, 0.05f);
    assert(loss_end < loss0 * 0.2f);
    rcre_lora_free(lora);

    /* 2. LLaMA F32 Training with BF16 mixed precision + gradient checkpointing */
    RCRE_LlamaConfig cfg = {
        .vocab_size = 16, .dim = 16, .hidden_dim = 32,
        .n_layers = 2, .n_heads = 2, .n_kv_heads = 2,
        .max_seq_len = 16, .rms_eps = 1e-5f, .rope_theta = 10000.0f
    };
    RCRE_LlamaModel* lm = rcre_llama_create(cfg, RCRE_LLM_W_F32, 55ULL);
    int32_t seq[6] = {1, 2, 3, 4, 5, 6};
    float l0 = rcre_llama_train_step_f32(lm, seq, 6, 0.05f, true, true);
    float l_end = l0;
    for (int s = 0; s < 15; ++s) l_end = rcre_llama_train_step_f32(lm, seq, 6, 0.05f, true, true);
    assert(l_end < l0);
    rcre_llama_free(lm);

    /* 3. Conv2D + Attention Autograd + Lion + Adafactor Optimizers */
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    RCRE_TensorTape* tape = rcre_tensor_tape_create(arena, 16);

    RCRE_Tensor* conv_x = rcre_tensor_alloc_4d(arena, 1, 1, 4, 4);
    RCRE_Tensor* conv_w = rcre_tensor_alloc_4d(arena, 1, 1, 3, 3);
    for (size_t i = 0; i < 16; ++i) rcre_tensor_f32_data(conv_x)[i] = 0.1f * (float)i;
    for (size_t i = 0; i < 9; ++i) rcre_tensor_f32_data(conv_w)[i] = 0.2f;
    RCRE_Tensor* conv_y = rcre_autograd_conv2d(tape, conv_x, conv_w, 1, 1);
    assert(conv_y && conv_y->shape[2] == 4 && conv_y->shape[3] == 4);
    for (size_t i = 0; i < conv_y->numel; ++i) conv_y->grad[i] = 1.0f;

    RCRE_Tensor* Q = rcre_tensor_alloc_2d(arena, 3, 4);
    RCRE_Tensor* K = rcre_tensor_alloc_2d(arena, 3, 4);
    RCRE_Tensor* V = rcre_tensor_alloc_2d(arena, 3, 4);
    for (size_t i = 0; i < 12; ++i) {
        rcre_tensor_f32_data(Q)[i] = 0.1f * (float)(i + 1);
        rcre_tensor_f32_data(K)[i] = -0.05f * (float)(i + 1);
        rcre_tensor_f32_data(V)[i] = 0.2f * (float)i;
    }
    RCRE_Tensor* att = rcre_autograd_attention(tape, Q, K, V);
    assert(att != nullptr);
    for (size_t i = 0; i < att->numel; ++i) att->grad[i] = 1.0f;

    rcre_tensor_tape_backward(tape);
    assert(fabsf(conv_w->grad[4]) > 1.0f);
    float v_grad_sum = 0.0f;
    for (size_t i = 0; i < 12; ++i) v_grad_sum += V->grad[i];
    assert(fabsf(v_grad_sum - 12.0f) < 1e-3f); /* Row sums of softmax = 1 => dV col sum = 3 per feature * 4 = 12 */
    rcre_tensor_tape_destroy(tape);

    /* Lion & Adafactor */
    RCRE_Tensor* p_lion = rcre_tensor_alloc_2d(arena, 2, 2);
    RCRE_Tensor* p_ada  = rcre_tensor_alloc_2d(arena, 2, 2);
    for (int i = 0; i < 4; ++i) { rcre_tensor_f32_data(p_lion)[i] = 2.0f; rcre_tensor_f32_data(p_ada)[i] = 2.0f; }
    RCRE_TensorOptim* opt_lion = rcre_tensor_optim_lion_create(arena, &p_lion, 1, 0.1f, 0.9f, 0.99f, 0.0f);
    RCRE_TensorOptim* opt_ada  = rcre_tensor_optim_adafactor_create(arena, &p_ada, 1, 0.2f, 0.0f);
    for (int s = 0; s < 15; ++s) {
        for (int i = 0; i < 4; ++i) {
            p_lion->grad[i] = 2.0f * rcre_tensor_f32_data(p_lion)[i];
            p_ada->grad[i]  = 2.0f * rcre_tensor_f32_data(p_ada)[i];
        }
        rcre_tensor_optim_step(opt_lion);
        rcre_tensor_optim_step(opt_ada);
    }
    assert(fabsf(rcre_tensor_f32_data(p_lion)[0]) < 1.0f);
    assert(fabsf(rcre_tensor_f32_data(p_ada)[0]) < 1.0f);

    /* Sharded binary stream across 2 shards */
    int32_t s0[4] = {10, 11, 12, 13};
    int32_t s1[4] = {20, 21, 22, 23};
    assert(rcre_io_write_atomic("/tmp/rcre_shard_0.bin", s0, sizeof(s0)) == RCRE_OK);
    assert(rcre_io_write_atomic("/tmp/rcre_shard_1.bin", s1, sizeof(s1)) == RCRE_OK);
    const char* paths[2] = {"/tmp/rcre_shard_0.bin", "/tmp/rcre_shard_1.bin"};
    RCRE_ShardStream ss;
    assert(rcre_shard_stream_open(&ss, paths, 2, 2) == RCRE_OK);
    int32_t xb[2], yb[2];
    int batches = 0;
    while (rcre_shard_stream_next(&ss, xb, yb)) batches++;
    assert(batches == 2 && xb[0] == 20 && yb[1] == 22);
    rcre_shard_stream_close(&ss);
    remove("/tmp/rcre_shard_0.bin");
    remove("/tmp/rcre_shard_1.bin");

    rcre_arena_destroy(arena);
    printf("  ✓ [PASS] test_phase3_lora_bf16_ckpt_autograd_optim\n");
}

static void test_phase4_codecs_orb_onnx_rf_sparse(void) {
    RCRE_Arena* arena = rcre_arena_create(8 * 1024 * 1024);

    /* 1. PNG / BMP / PPM / JPEG codecs */
    size_t sh[3] = {16, 16, 3};
    RCRE_Tensor img = rcre_tensor_create(sh, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    uint8_t* px = (uint8_t*)img.data;
    for (size_t y = 0; y < 16; ++y) {
        for (size_t x = 0; x < 16; ++x) {
            size_t o = (y * 16 + x) * 3U;
            px[o + 0] = (uint8_t)(x * 16);
            px[o + 1] = (uint8_t)(y * 16);
            px[o + 2] = (uint8_t)((x + y) * 8);
        }
    }

    assert(rcre_codec_save_png(&img, "/tmp/rcre_test.png") == RCRE_OK);
    RCRE_Tensor dec_png = {0};
    assert(rcre_codec_load_png("/tmp/rcre_test.png", arena, &dec_png) == RCRE_OK);
    assert(dec_png.shape[0] == 16 && dec_png.shape[1] == 16 && dec_png.shape[2] == 3);
    assert(memcmp(dec_png.data, img.data, 16 * 16 * 3) == 0);
    remove("/tmp/rcre_test.png");

    assert(rcre_codec_save_bmp(&img, "/tmp/rcre_test.bmp") == RCRE_OK);
    RCRE_Tensor dec_bmp = {0};
    assert(rcre_codec_load_bmp("/tmp/rcre_test.bmp", arena, &dec_bmp) == RCRE_OK);
    assert(memcmp(dec_bmp.data, img.data, 16 * 16 * 3) == 0);
    remove("/tmp/rcre_test.bmp");

    assert(rcre_codec_save_ppm(&img, "/tmp/rcre_test.ppm") == RCRE_OK);
    RCRE_Tensor dec_ppm = {0};
    assert(rcre_codec_load_ppm("/tmp/rcre_test.ppm", arena, &dec_ppm) == RCRE_OK);
    assert(memcmp(dec_ppm.data, img.data, 16 * 16 * 3) == 0);
    remove("/tmp/rcre_test.ppm");

    assert(rcre_codec_save_jpeg(&img, "/tmp/rcre_test.jpg", 90) == RCRE_OK);
    RCRE_Tensor dec_jpg = {0};
    assert(rcre_codec_load_jpeg("/tmp/rcre_test.jpg", arena, &dec_jpg) == RCRE_OK);
    assert(dec_jpg.shape[0] == 16 && dec_jpg.shape[1] == 16 && dec_jpg.shape[2] == 3);
    remove("/tmp/rcre_test.jpg");

    /* 2. FAST-9 + ORB Binary Features */
    size_t gsh[2] = {48, 48};
    RCRE_Tensor gray = rcre_tensor_create(gsh, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    float* gd = rcre_tensor_f32_data(&gray);
    for (size_t y = 0; y < 48; ++y) {
        for (size_t x = 0; x < 48; ++x) {
            gd[y * 48 + x] = ((x >= 16 && x <= 32) && (y >= 16 && y <= 32)) ? 1.0f : 0.0f;
        }
    }
    RCRE_ORBKeypoint* kps = nullptr;
    int nk = 0;
    assert(rcre_cv_fast_detect(&gray, 0.3f, 64, &kps, &nk, arena) == RCRE_OK);
    assert(nk >= 4);
    assert(rcre_cv_orb_extract(&gray, kps, nk) == RCRE_OK);
    RCRE_CV_Match* matches = nullptr;
    int nm = 0;
    assert(rcre_cv_orb_match(kps, nk, kps, nk, 8, &matches, &nm, arena) == RCRE_OK);
    assert(nm == nk);

    /* 3. ONNX Protobuf Subset MLP Inference */
    float w1[4] = {1, -1, 0.5f, 2.0f}, b1[2] = {0.1f, -0.2f};
    float w2[2] = {1.5f, -0.5f}, b2[1] = {0.3f};
    uint8_t pb[4096];
    size_t pblen = rcre_onnx_build_mlp_protobuf(pb, sizeof(pb), w1, b1, 2, 2, w2, b2, 1);
    assert(pblen > 0);
    RCRE_ONNXModel onnx;
    assert(rcre_onnx_load_buffer(pb, pblen, &onnx) == RCRE_OK);
    size_t xsh[2] = {1, 2};
    RCRE_Tensor xin = rcre_tensor_create(xsh, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    rcre_tensor_f32_data(&xin)[0] = 2.0f;
    rcre_tensor_f32_data(&xin)[1] = 1.0f;
    RCRE_Tensor xout = {0};
    assert(rcre_onnx_run(&onnx, &xin, arena, &xout) == RCRE_OK);
    /* h0 = [2*1 + 1*0.5 + 0.1 = 2.6, 2*(-1) + 1*2.0 - 0.2 = -0.2] -> ReLU -> [2.6, 0] -> y = 2.6*1.5 + 0.3 = 4.2 */
    assert(fabsf(rcre_tensor_f32_data(&xout)[0] - 4.2f) < 1e-4f);
    rcre_onnx_free(&onnx);

    /* 4. Random Forest + Sparse L1 Linear */
    size_t rf_xsh[2] = {12, 2}, rf_ysh[1] = {12}, rf_wsh[1] = {2};
    RCRE_Tensor rfx = rcre_tensor_create(rf_xsh, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor rfy = rcre_tensor_create(rf_ysh, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor rfw = rcre_tensor_create(rf_wsh, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor rfout = rcre_tensor_create(rf_ysh, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    for (size_t i = 0; i < 12; ++i) {
        rcre_tensor_f32_data(&rfx)[i * 2 + 0] = (i < 6) ? -1.0f : 1.0f;
        rcre_tensor_f32_data(&rfx)[i * 2 + 1] = 0.0f;
        rcre_tensor_f32_data(&rfy)[i] = (i < 6) ? 0.0f : 1.0f;
    }
    RCRE_RandomForest rf;
    assert(rcre_ml_rf_fit(&rfx, &rfy, 8, 3, 42ULL, arena, &rf) == RCRE_OK);
    assert(rcre_ml_rf_predict(&rf, &rfx, &rfout) == RCRE_OK);
    assert(rcre_tensor_f32_data(&rfout)[0] < 0.2f && rcre_tensor_f32_data(&rfout)[11] > 0.8f);

    assert(rcre_ml_sparse_linear_fit(&rfx, &rfy, 0.2f, 0.01f, 30, &rfw) == RCRE_OK);
    assert(fabsf(rcre_tensor_f32_data(&rfw)[1]) < 1e-5f); /* Noise feature 1 zeroed out by L1 */

    rcre_arena_destroy(arena);
    printf("  ✓ [PASS] test_phase4_codecs_orb_onnx_rf_sparse\n");
}

static void test_phase5_freestanding_gpu(void) {
    /* 1. Static-buffer freestanding arena */
    uint8_t stack_backing[4096];
    RCRE_Arena local_arena;
    assert(rcre_arena_from_buffer(&local_arena, stack_backing, sizeof(stack_backing)) == RCRE_OK);
    size_t sh[2] = {4, 4};
    RCRE_Tensor t = rcre_tensor_create(sh, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, &local_arena);
    assert(t.data != nullptr && local_arena.offset > 0);

    /* 2. GPU Metal (or CPU HAL fallback) GEMM */
    RCRE_Arena* a = rcre_arena_create(1024 * 1024);
    const size_t N = 64; /* 64*64 = 4096 triggers Metal GPU dispatch path */
    float* A = (float*)rcre_arena_alloc(a, N * N * sizeof(float), 64);
    float* B = (float*)rcre_arena_alloc(a, N * N * sizeof(float), 64);
    float* C_gpu = (float*)rcre_arena_alloc(a, N * N * sizeof(float), 64);
    float* C_cpu = (float*)rcre_arena_alloc(a, N * N * sizeof(float), 64);
    for (size_t i = 0; i < N * N; ++i) {
        A[i] = (float)((i % 11) - 5) * 0.1f;
        B[i] = (float)((i % 7) - 3) * 0.1f;
    }
    /* H6 bridge: best available backend (cuda -> metal -> cpu). The cpu
     * backend delegates to the core HAL, so this cross-checks the bridge
     * on every platform and runs real GPU GEMM where one is present. */
    RCRE_GpuCtx* gctx = rcre_gpu_open();
    assert(gctx != NULL);
    assert(rcre_gpu_gemm_f32_host(gctx, N, N, N, 1.0f, A, N, B, N,
                                  0.0f, C_gpu, N) == RCRE_OK);
    rcre_hal.gemm_f32(N, N, N, 1.0f, A, N, B, N, 0.0f, C_cpu, N);
    for (size_t i = 0; i < N * N; ++i) {
        assert(fabsf(C_gpu[i] - C_cpu[i]) < 1e-3f);
    }
    printf("  ✓ [PASS] test_phase5_freestanding_gpu (gpu bridge backend: %s)\n",
           rcre_gpu_backend_name(gctx));
    rcre_gpu_close(gctx);
    rcre_arena_destroy(a);
}

int main(void) {
    rcre_hal_init();
    printf("======================================================================\n");
    printf("🚀 RCRE v2.0.0 SOTA Full-Stack Verification Suite\n");
    printf("======================================================================\n");
    test_phase0_1_abi_nd8_broadcast_hal();
    test_phase2_quant_gguf_bpe_llm_sampler();
    test_phase3_lora_bf16_ckpt_autograd_optim();
    test_phase4_codecs_orb_onnx_rf_sparse();
    test_phase5_freestanding_gpu();
    printf("======================================================================\n");
    printf("🎉 All RCRE v2.0.0 SOTA Verification Checks Passed!\n");
    printf("======================================================================\n");
    return 0;
}
