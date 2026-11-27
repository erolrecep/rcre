/**
 * @file fuzz_gguf.c
 * @brief LibFuzzer & Standalone Adversarial Fuzzer for RCRE Zero-Copy GGUF v2/v3 Reader
 *
 * Exercises untrusted GGUF headers, nested KV arrays, extreme tensor/KV counts,
 * alignment edge cases, and tensor offset bounds under ASan+UBSan.
 */

#include "rcre_llm.h"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (!data || size == 0 || size > 65536U) return 0;

    RCRE_GGUFFile gguf;
    memset(&gguf, 0, sizeof(gguf));
    if (rcre_gguf_open_from_buffer(data, size, &gguf) == RCRE_OK) {
        /* Exercise metadata & tensor queries on any parsed archive */
        uint32_t u32_v = 0;
        float f32_v = 0.0f;
        RCRE_Slice s_v;
        (void)rcre_gguf_get_u32(&gguf, "general.alignment", &u32_v);
        (void)rcre_gguf_get_f32(&gguf, "llama.rope.freq_base", &f32_v);
        (void)rcre_gguf_get_str(&gguf, "general.architecture", &s_v);

        for (uint64_t t = 0; t < gguf.n_tensors && t < 32U; ++t) {
            const void* ptr = rcre_gguf_tensor_data(&gguf, &gguf.tensors[t]);
            if (ptr) {
                volatile uint8_t touch = ((const uint8_t*)ptr)[0];
                (void)touch;
            }
        }
        rcre_gguf_close(&gguf);
    }
    return 0;
}

#ifdef FUZZ_STANDALONE
static uint64_t s_rng = 0x660F20261127ULL;
static uint64_t next_u64(void) {
    s_rng ^= s_rng << 13;
    s_rng ^= s_rng >> 7;
    s_rng ^= s_rng << 17;
    return s_rng;
}

int main(void) {
    rcre_hal_init();

    /* Structured GGUF v3 minimal valid seed (magic='GGUF', ver=3, 0 tensors, 1 u32 KV) */
    static const uint8_t seed_gguf_v3[] = {
        'G', 'G', 'U', 'F',
        0x03, 0x00, 0x00, 0x00,                         /* version = 3 */
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* n_tensors = 0 */
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* n_kv = 1 */
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* key len = 3 */
        'f', 'o', 'o',
        0x04, 0x00, 0x00, 0x00,                         /* type = UINT32 */
        0x2A, 0x00, 0x00, 0x00                          /* value = 42 */
    };

    /* Adversarial seed: huge n_kv and n_tensors counts */
    static const uint8_t seed_gguf_huge[] = {
        'G', 'G', 'U', 'F',
        0x03, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0x7F, 0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0x7F, 0x00, 0x00, 0x00, 0x00
    };

    /* Adversarial seed: 1 tensor with out-of-bounds offset */
    static const uint8_t seed_gguf_oob_tensor[] = {
        'G', 'G', 'U', 'F',
        0x03, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* n_tensors = 1 */
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* n_kv = 0 */
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 'w',
        0x01, 0x00, 0x00, 0x00,                         /* n_dims = 1 */
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* dim[0] = 16 */
        0x00, 0x00, 0x00, 0x00,                         /* type = F32 */
        0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00  /* offset = 1MB (OOB) */
    };

    const uint8_t* seeds[] = { seed_gguf_v3, seed_gguf_huge, seed_gguf_oob_tensor };
    const size_t seed_lens[] = { sizeof(seed_gguf_v3), sizeof(seed_gguf_huge), sizeof(seed_gguf_oob_tensor) };

    for (size_t s = 0; s < 3; ++s) {
        LLVMFuzzerTestOneInput(seeds[s], seed_lens[s]);
    }

    uint8_t buf[512];
    const size_t iters = 3000;
    for (size_t i = 0; i < iters; ++i) {
        size_t len = 0;
        if ((i & 1U) == 0) {
            size_t sid = (size_t)(next_u64() % 3U);
            len = seed_lens[sid];
            memcpy(buf, seeds[sid], len);
            size_t extra = (size_t)(next_u64() % 128U);
            for (size_t j = 0; j < extra && len < sizeof(buf); ++j) {
                buf[len++] = (uint8_t)(next_u64() & 0xFFU);
            }
            size_t flips = 1U + (size_t)(next_u64() % 6U);
            for (size_t f = 0; f < flips; ++f) {
                size_t pos = (size_t)(next_u64() % len);
                buf[pos] ^= (uint8_t)(next_u64() & 0xFFU);
            }
            if ((next_u64() & 3U) == 0 && len > 4) {
                len = 4U + (size_t)(next_u64() % (len - 3U));
            }
        } else {
            len = 8U + (size_t)(next_u64() % 200U);
            for (size_t j = 0; j < len; ++j) buf[j] = (uint8_t)(next_u64() & 0xFFU);
            /* Keep GGUF magic intact to dive deep past header check */
            buf[0] = 'G'; buf[1] = 'G'; buf[2] = 'U'; buf[3] = 'F';
            buf[4] = (uint8_t)(2U + (next_u64() & 1U));
            buf[5] = 0; buf[6] = 0; buf[7] = 0;
        }
        LLVMFuzzerTestOneInput(buf, len);
    }

    printf("[PASS] fuzz_gguf_standalone: %zu adversarial GGUF v2/v3 iterations clean (ASan+UBSan)\n", iters);
    return 0;
}
#endif
