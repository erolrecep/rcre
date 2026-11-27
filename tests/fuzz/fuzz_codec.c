/**
 * @file fuzz_codec.c
 * @brief LibFuzzer & Standalone Adversarial Fuzzer for RCRE Image Codecs (PNG / JPEG)
 *
 * Exercises untrusted image decoding paths (DEFLATE/zlib inflate, PNG chunk/IHDR
 * integer overflow checks, JPEG Huffman/SOF0/DQT/DHT markers) under ASan+UBSan.
 */

#include "rcre_codec.h"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t s_arena_backing[2 * 1024 * 1024];

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (!data || size == 0 || size > 65536U) return 0;

    RCRE_Arena arena;
    if (rcre_arena_from_buffer(&arena, s_arena_backing, sizeof(s_arena_backing)) != RCRE_OK) {
        return 0;
    }

    /* 1. Fuzz PNG buffer decoder */
    RCRE_Tensor img_png;
    memset(&img_png, 0, sizeof(img_png));
    if (rcre_codec_load_png_from_buffer(data, size, &arena, &img_png) == RCRE_OK) {
        if (!img_png.data || img_png.ndim != 3 || img_png.shape[0] == 0 || img_png.shape[1] == 0) {
            fprintf(stderr, "FUZZ_CODEC FAILURE: invalid PNG tensor invariants\n");
            abort();
        }
    }

    /* 2. Fuzz baseline JPEG buffer decoder */
    rcre_arena_reset(&arena);
    RCRE_Tensor img_jpg;
    memset(&img_jpg, 0, sizeof(img_jpg));
    if (rcre_codec_load_jpeg_from_buffer(data, size, &arena, &img_jpg) == RCRE_OK) {
        if (!img_jpg.data || img_jpg.ndim != 3 || img_jpg.shape[0] == 0 || img_jpg.shape[1] == 0) {
            fprintf(stderr, "FUZZ_CODEC FAILURE: invalid JPEG tensor invariants\n");
            abort();
        }
    }

    return 0;
}

#ifdef FUZZ_STANDALONE
static uint64_t s_rng = 0xC0DEC20261127ULL;
static uint64_t next_u64(void) {
    s_rng ^= s_rng << 13;
    s_rng ^= s_rng >> 7;
    s_rng ^= s_rng << 17;
    return s_rng;
}

int main(void) {
    rcre_hal_init();

    /* Structured adversarial seeds: valid PNG signature + corrupt/overflow IHDR,
     * valid JPEG SOI + SOF0/DHT/DQT/SOS markers, and pure random fuzzing */
    static const uint8_t seed_png_overflow[] = {
        0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 'I', 'H', 'D', 'R',
        0x7F, 0xFF, 0xFF, 0xFF, 0x7F, 0xFF, 0xFF, 0xFF,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    };
    static const uint8_t seed_png_tiny[] = {
        0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 'I', 'H', 'D', 'R',
        0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02,
        0x08, 0x02, 0x00, 0x00, 0x00, 0xFD, 0xD4, 0x9A, 0x73,
        0x00, 0x00, 0x00, 0x08, 'I', 'D', 'A', 'T',
        0x78, 0x9C, 0x63, 0x60, 0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x00, 'I', 'E', 'N', 'D', 0xAE, 0x42, 0x60, 0x82
    };
    static const uint8_t seed_jpg_hdr[] = {
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 'J', 'F', 'I', 'F', 0x00,
        0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x04, 0x00, 0x04, 0x01, 0x01, 0x11, 0x00,
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
        0x00, 0xFF, 0xD9
    };

    const uint8_t* seeds[] = { seed_png_overflow, seed_png_tiny, seed_jpg_hdr };
    const size_t seed_lens[] = { sizeof(seed_png_overflow), sizeof(seed_png_tiny), sizeof(seed_jpg_hdr) };

    for (size_t s = 0; s < 3; ++s) {
        LLVMFuzzerTestOneInput(seeds[s], seed_lens[s]);
    }

    uint8_t buf[1024];
    const size_t iters = 3000;
    for (size_t i = 0; i < iters; ++i) {
        size_t len = 0;
        if ((i % 3U) == 0) {
            /* Mutate a structured PNG/JPEG seed */
            size_t sid = (size_t)(next_u64() % 3U);
            len = seed_lens[sid];
            memcpy(buf, seeds[sid], len);
            size_t extra = (size_t)(next_u64() % 64U);
            for (size_t j = 0; j < extra && len < sizeof(buf); ++j) {
                buf[len++] = (uint8_t)(next_u64() & 0xFFU);
            }
            size_t n_flips = 1U + (size_t)(next_u64() % 6U);
            for (size_t f = 0; f < n_flips; ++f) {
                size_t pos = (size_t)(next_u64() % len);
                buf[pos] ^= (uint8_t)(next_u64() & 0xFFU);
            }
            if ((next_u64() & 3U) == 0 && len > 2) {
                len = 1U + (size_t)(next_u64() % len); /* Truncation fuzz */
            }
        } else {
            /* Pure random bytes with PNG or JPEG magic prefix */
            len = 4U + (size_t)(next_u64() % 256U);
            for (size_t j = 0; j < len; ++j) buf[j] = (uint8_t)(next_u64() & 0xFFU);
            if ((i % 3U) == 1 && len >= 8) {
                memcpy(buf, seed_png_tiny, 8);
            } else if (len >= 2) {
                buf[0] = 0xFF;
                buf[1] = 0xD8;
            }
        }
        LLVMFuzzerTestOneInput(buf, len);
    }

    printf("[PASS] fuzz_codec_standalone: %zu adversarial PNG/JPEG iterations clean (ASan+UBSan)\n", iters);
    return 0;
}
#endif
