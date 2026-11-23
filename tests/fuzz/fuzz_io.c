/**
 * @file fuzz_io.c
 * @brief LibFuzzer & Deterministic Standalone Fuzz Target for RCRE Parsers & Codecs
 *
 * Build with libFuzzer:
 *   clang -fsanitize=fuzzer,address,undefined -Iinclude ... tests/fuzz/fuzz_io.c
 * Build deterministic regression runner:
 *   clang -DFUZZ_STANDALONE -fsanitize=address,undefined ... tests/fuzz/fuzz_io.c
 */

#include "rcre.h"
#include "rcre_io.h"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (!data || size == 0 || size > (1U << 20)) {
        return 0;
    }

    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    if (!arena) return 0;

    /* 1. Float32 parser */
    {
        float v = 0.0f;
        RCRE_Slice s = rcre_slice_from_parts(data, size);
        (void)rcre_slice_parse_f32(s, &v);
    }

    /* 2. CSV parser (raw + RFC-4180 quotes) */
    {
        RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
        RCRE_Tensor t = {0};
        RCRE_CSVOptions opts = { .delimiter = ',', .quote_char = '"' };
        (void)rcre_io_parse_csv_slice(rcre_slice_from_parts(data, size), &opts, arena, &t);
        rcre_arena_temp_end(tmp);
    }

    /* 3. Safetensors in-memory archive parser */
    {
        RCRE_SafeTensorsArchive arch;
        if (rcre_io_safetensors_open_from_buffer(data, size, &arch) == RCRE_OK) {
            rcre_io_safetensors_close(&arch);
        }
    }

    /* 4. .npy in-memory mmap wrapper */
    {
        RCRE_MappedFile mf;
        if (rcre_io_mmap_from_buffer(data, size, &mf) == RCRE_OK) {
            RCRE_Tensor view = {0};
            (void)rcre_io_view_npy(&mf, &view);
            rcre_io_mmap_close(&mf);
        }
    }

    /* 5. RCRZ decompressor (bounded 4 MB arena rejects corrupt/bomb frames safely) */
    {
        RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
        RCRE_Tensor t = {0};
        (void)rcre_io_decompress_tensor(rcre_slice_from_parts(data, size), arena, &t);
        rcre_arena_temp_end(tmp);
    }

    rcre_arena_destroy(arena);
    return 0;
}

#ifdef FUZZ_STANDALONE
static uint32_t xorshift32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

int main(void) {
    rcre_hal_init();

    const char* seeds[] = {
        "1.23,4.56\n7.89,-0.01\n",
        "\"1,234.5\",\"6.78\"\n\"-9.5\",\"10\"\n",
        "NaN", "inf", "-1e40", "1e-45", "0.000000000000000001",
        "\x93NUMPY\x01\x00\x10\x00{'descr':'<f4'}",
        "RCRZ\x00\x00\x00\x00\xFF\xFF\xFF\xFF",
        "\x08\x00\x00\x00\x00\x00\x00\x00{\"a\":{}}"
    };
    const size_t n_seeds = sizeof(seeds) / sizeof(seeds[0]);
    for (size_t i = 0; i < n_seeds; ++i) {
        LLVMFuzzerTestOneInput((const uint8_t*)seeds[i], strlen(seeds[i]));
    }

    uint8_t buf[512];
    uint32_t rng = 0xDEADBEEFU;
    const int ITERS = 1000;
    for (int it = 0; it < ITERS; ++it) {
        size_t seed_idx = (size_t)it % n_seeds;
        size_t slen = strlen(seeds[seed_idx]);
        size_t len = (xorshift32(&rng) % 250U) + 1U;
        if (it % 3 == 0 && slen < sizeof(buf)) {
            memcpy(buf, seeds[seed_idx], slen);
            len = slen;
            /* Bit-flip 1-3 random bytes */
            size_t flips = (xorshift32(&rng) % 3U) + 1U;
            for (size_t f = 0; f < flips; ++f) {
                buf[xorshift32(&rng) % len] ^= (uint8_t)(xorshift32(&rng) & 0xFFU);
            }
        } else {
            for (size_t j = 0; j < len; ++j) {
                buf[j] = (uint8_t)(xorshift32(&rng) & 0xFFU);
            }
        }
        LLVMFuzzerTestOneInput(buf, len);
    }

    printf("✅ Fuzz Regression Harness: %zu seeds + %d random/mutated inputs passed ASan/UBSan!\n",
           n_seeds, ITERS);
    return 0;
}
#endif
