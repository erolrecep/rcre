/**
 * @file fuzz_str.c
 * @brief LibFuzzer & Deterministic Standalone Differential Fuzzer for rcre_str
 *
 * Verifies:
 *   1. Bit-exact differential parity between scalar and HAL SIMD kernels
 *      (utf8_validate, find_substr, ascii_to_lower, ascii_to_upper).
 *   2. Zero ASan/UBSan violations across RCRE_StrBuf, UTF-8 decoding/truncation,
 *      strict parsers, pull splitters, JSON/CSV escaping, RCRE_StrPool, and paths.
 */

#include "rcre.h"
#include "rcre_str.h"
#include "hal_internal.h"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fuzz_abort(const char* msg) {
    fprintf(stderr, "FUZZ FAILURE: %s\n", msg);
    abort();
}

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (!data || size > (1U << 18)) {
        return 0;
    }

    RCRE_Slice s = rcre_slice_from_parts(data, size);

    /* 1. Differential SIMD vs Scalar UTF-8 validation */
    bool v_scalar = rcre_scalar_utf8_validate(data, size);
    bool v_simd = rcre_hal.utf8_validate(data, size);
    if (v_scalar != v_simd) {
        fuzz_abort("utf8_validate scalar vs SIMD mismatch");
    }
    size_t err_off = 0;
    bool v_api = rcre_utf8_validate(s, &err_off);
    if (v_api != v_scalar) {
        fuzz_abort("rcre_utf8_validate mismatch");
    }
    if (!v_api && err_off > size) {
        fuzz_abort("rcre_utf8_validate invalid err_off");
    }

    /* 2. UTF-8 truncation ALWAYS produces valid UTF-8 */
    for (size_t cap = 0; cap <= 32 && cap <= size + 1; cap += 5) {
        RCRE_Slice trunc = rcre_utf8_truncate_bytes(s, cap);
        if (trunc.len > cap || !rcre_utf8_validate(trunc, nullptr)) {
            fuzz_abort("rcre_utf8_truncate_bytes emitted invalid UTF-8");
        }
    }

    /* 3. Bounded UTF-8 copy ALWAYS produces valid NUL-terminated UTF-8 */
    char small[19];
    (void)rcre_cstr_copy_bounded(small, sizeof(small), s);
    if (!rcre_utf8_validate(rcre_slice_from_cstr(small), nullptr)) {
        fuzz_abort("rcre_cstr_copy_bounded emitted invalid UTF-8");
    }

    /* 4. Differential SIMD vs Scalar substring search */
    if (size >= 4) {
        size_t nlen = (size_t)(data[0] % 11U);
        if (nlen <= size) {
            const uint8_t* needle = data + (size - nlen);
            size_t f_scalar = rcre_scalar_find_substr(data, size, needle, nlen);
            size_t f_simd = rcre_hal.find_substr(data, size, needle, nlen);
            if (f_scalar != f_simd) {
                fuzz_abort("find_substr scalar vs SIMD mismatch");
            }
        }
    }

    /* 5. Differential SIMD vs Scalar ASCII case folding */
    uint8_t arena_mem[65536];
    RCRE_Arena arena;
    if (rcre_arena_from_buffer(&arena, arena_mem, sizeof(arena_mem)) == RCRE_OK) {
        if (size <= 4096) {
            uint8_t* b1 = (uint8_t*)rcre_arena_alloc(&arena, size > 0 ? size : 1, 64);
            uint8_t* b2 = (uint8_t*)rcre_arena_alloc(&arena, size > 0 ? size : 1, 64);
            if (b1 && b2) {
                rcre_scalar_ascii_to_lower(data, b1, size);
                rcre_hal.ascii_to_lower(data, b2, size);
                if (size > 0 && memcmp(b1, b2, size) != 0) {
                    fuzz_abort("ascii_to_lower scalar vs SIMD mismatch");
                }
                rcre_scalar_ascii_to_upper(data, b1, size);
                rcre_hal.ascii_to_upper(data, b2, size);
                if (size > 0 && memcmp(b1, b2, size) != 0) {
                    fuzz_abort("ascii_to_upper scalar vs SIMD mismatch");
                }
            }
        }

        /* 6. Strict parsers & pull iterators */
        double dv = 0.0;
        uint64_t uv = 0;
        bool bv = false;
        (void)rcre_slice_parse_f64(s, &dv);
        (void)rcre_slice_parse_hex_u64(s, &uv);
        (void)rcre_slice_parse_bool(s, &bv);

        RCRE_Slice cursor = s;
        RCRE_Slice line;
        size_t lines = 0;
        while (rcre_split_line_next(&cursor, &line) && lines < 128) {
            lines++;
        }

        /* 7. Path ops & JSON/CSV escaping + StrPool */
        (void)rcre_path_basename(s);
        (void)rcre_path_dirname(s);
        (void)rcre_path_ext(s);
        (void)rcre_path_stem(s);

        RCRE_StrBuilder sb = rcre_sb_init(&arena, 64);
        (void)rcre_sb_append_escape_json(&sb, s);
        rcre_sb_reset(&sb);
        (void)rcre_sb_append_escape_csv(&sb, s, ',');

        RCRE_StrPool pool;
        if (rcre_strpool_init(&pool, &arena, 16) == RCRE_OK) {
            RCRE_StrId id = rcre_strpool_intern(&pool, s);
            if (id != RCRE_STR_ID_INVALID && rcre_strpool_find(&pool, s) != id) {
                fuzz_abort("strpool intern/find mismatch");
            }
        }
    }

    return 0;
}

#ifdef FUZZ_STANDALONE
static uint64_t fuzz_xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

int main(void) {
    rcre_hal_init();

    const char* seeds[] = {
        "",
        "Hello, World!",
        "Merhaba D\xC3\xBCnya \xE4\xB8\x96\xE7\x95\x8C \xF0\x9F\x9A\x80",
        "\xC0\x80\xED\xA0\x80\xF4\x90\x80\x80\xF5\x80\x80\x80",
        "123.456e-78",
        "0xdeadbeefcafebabe",
        "/usr/local/bin//rcre.tar.gz",
        "a,b,\"c\"\"d\"\r\ne,f\n"
    };
    for (size_t i = 0; i < sizeof(seeds) / sizeof(seeds[0]); ++i) {
        LLVMFuzzerTestOneInput((const uint8_t*)seeds[i], strlen(seeds[i]));
    }

    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    uint8_t buf[512];
    for (size_t iter = 0; iter < 20000; ++iter) {
        size_t len = (size_t)(fuzz_xorshift64(&rng) % sizeof(buf));
        int mode = (int)(iter % 4);
        for (size_t j = 0; j < len; ++j) {
            uint64_t r = fuzz_xorshift64(&rng);
            if (mode == 0) {
                buf[j] = (uint8_t)(r & 0x7FU); /* Pure ASCII */
            } else if (mode == 1) {
                /* High-density UTF-8 boundary bytes */
                static const uint8_t interesting[16] = {
                    0x00, 0x41, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0,
                    0xBF, 0xC0, 0xC2, 0xE0, 0xED, 0xF0, 0xF4, 0xF5
                };
                buf[j] = interesting[r & 15U];
            } else if (mode == 2) {
                /* Mix ASCII runs with valid/invalid UTF-8 bursts */
                buf[j] = ((r & 7U) == 0) ? (uint8_t)(0x80U + (r >> 8) % 128U) : (uint8_t)('A' + (r % 58U));
            } else {
                buf[j] = (uint8_t)(r & 0xFFU); /* Uniform random bytes */
            }
        }
        LLVMFuzzerTestOneInput(buf, len);
    }

    printf("✅ fuzz_str passed 20,000+ differential SIMD/scalar & safety iterations!\n");
    return 0;
}
#endif
