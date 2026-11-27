/**
 * @file fuzz_onnx.c
 * @brief LibFuzzer & Standalone Adversarial Fuzzer for RCRE ONNX Protobuf Subset Runner
 *
 * Exercises untrusted Protobuf varint decoding, length-delimited submessages,
 * initializer shapes, and graph execution under ASan+UBSan.
 */

#include "rcre_codec.h"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t s_arena_backing[512 * 1024];

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (!data || size == 0 || size > 65536U) return 0;

    RCRE_ONNXModel model;
    memset(&model, 0, sizeof(model));
    if (rcre_onnx_load_buffer(data, size, &model) == RCRE_OK) {
        RCRE_Arena arena;
        if (rcre_arena_from_buffer(&arena, s_arena_backing, sizeof(s_arena_backing)) == RCRE_OK) {
            float in_vals[4] = { 0.25f, -0.5f, 1.0f, 0.75f };
            RCRE_Tensor input = {0};
            input.data       = in_vals;
            input.ndim       = 2;
            input.shape[0]   = 1;
            input.shape[1]   = 4;
            input.strides[0] = 4;
            input.strides[1] = 1;
            input.dtype      = RCRE_DTYPE_F32;
            input.layout     = RCRE_LAYOUT_TABULAR;
            input.is_view    = true;

            RCRE_Tensor output = {0};
            (void)rcre_onnx_run(&model, &input, &arena, &output);
        }
        rcre_onnx_free(&model);
    }
    return 0;
}

#ifdef FUZZ_STANDALONE
static uint64_t s_rng = 0x0AA120261127ULL;
static uint64_t next_u64(void) {
    s_rng ^= s_rng << 13;
    s_rng ^= s_rng >> 7;
    s_rng ^= s_rng << 17;
    return s_rng;
}

int main(void) {
    rcre_hal_init();

    /* Build a valid 2-layer MLP ONNX protobuf seed using RCRE's own builder */
    const float w1[4 * 4] = {
        0.1f, -0.2f, 0.3f, 0.4f,
       -0.1f,  0.5f, 0.2f, 0.1f,
        0.3f,  0.1f,-0.4f, 0.2f,
        0.2f,  0.2f, 0.1f,-0.3f
    };
    const float b1[4] = { 0.01f, -0.02f, 0.03f, 0.0f };
    const float w2[4 * 2] = { 0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.2f };
    const float b2[2] = { 0.1f, -0.1f };

    uint8_t valid_seed[4096];
    size_t valid_len = rcre_onnx_build_mlp_protobuf(
        valid_seed, sizeof(valid_seed), w1, b1, 4, 4, w2, b2, 2);
    if (valid_len == 0) {
        fprintf(stderr, "FUZZ_ONNX: failed to generate valid MLP seed\n");
        return 1;
    }

    LLVMFuzzerTestOneInput(valid_seed, valid_len);

    uint8_t buf[4096];
    const size_t iters = 3000;
    for (size_t i = 0; i < iters; ++i) {
        size_t len = 0;
        if ((i % 3U) != 2U) {
            /* Mutate valid ONNX protobuf seed */
            len = valid_len;
            memcpy(buf, valid_seed, len);
            size_t flips = 1U + (size_t)(next_u64() % 8U);
            for (size_t f = 0; f < flips; ++f) {
                size_t pos = (size_t)(next_u64() % len);
                buf[pos] ^= (uint8_t)(next_u64() & 0xFFU);
            }
            if ((next_u64() & 3U) == 0 && len > 2) {
                len = 1U + (size_t)(next_u64() % len);
            }
        } else {
            /* Pure random protobuf byte stream */
            len = 1U + (size_t)(next_u64() % 256U);
            for (size_t j = 0; j < len; ++j) buf[j] = (uint8_t)(next_u64() & 0xFFU);
        }
        LLVMFuzzerTestOneInput(buf, len);
    }

    printf("[PASS] fuzz_onnx_standalone: %zu adversarial ONNX protobuf iterations clean (ASan+UBSan)\n", iters);
    return 0;
}
#endif
