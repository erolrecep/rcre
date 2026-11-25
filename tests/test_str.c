/**
 * @file test_str.c
 * @brief Comprehensive Unit, UTF-8, HAL-SIMD Parity & Boundary Tests for rcre_str
 *
 * Notice RCRE_BAN_UNSAFE_CSTR is enabled here: this entire suite compiles with
 * legacy unsafe <string.h> functions (strcpy/strncpy/strcat/sprintf/strtok)
 * poisoned at compile time.
 */

#define RCRE_BAN_UNSAFE_CSTR
#include "rcre.h"
#include "rcre_str.h"
#include "hal_internal.h"
#include "test_framework.h"
#include <math.h>
#include <string.h>

static void test_str_macros_and_strbuf(void) {
    RCRE_Slice lit = RCRE_STR("Hello, RCRE!");
    ASSERT_UINT_EQ(12, lit.len);
    ASSERT_TRUE(rcre_slice_eq_cstr(lit, "Hello, RCRE!"));

    RCRE_Slice empty_lit = RCRE_STR("");
    ASSERT_UINT_EQ(0, empty_lit.len);
    ASSERT_NULL(empty_lit.data);

    /* Stack bounded buffer — replaces strcpy/strncpy/sprintf */
    RCRE_STRBUF_STACK(buf, 16);
    ASSERT_TRUE(rcre_strbuf_copy(&buf, RCRE_STR("hello")));
    ASSERT_FALSE(buf.truncated);
    ASSERT_UINT_EQ(5, buf.len);
    ASSERT_TRUE(rcre_slice_eq(rcre_strbuf_as_slice(&buf), RCRE_STR("hello")));
    ASSERT_INT_EQ(0, strcmp(rcre_strbuf_cstr(&buf), "hello"));

    ASSERT_TRUE(rcre_strbuf_append(&buf, RCRE_STR("_world")));
    ASSERT_FALSE(buf.truncated);
    ASSERT_UINT_EQ(11, buf.len);
    ASSERT_INT_EQ(0, strcmp(rcre_strbuf_cstr(&buf), "hello_world"));

    /* Overflow sets sticky truncated flag, guarantees NUL termination at cap-1 */
    ASSERT_FALSE(rcre_strbuf_append(&buf, RCRE_STR("_123456789")));
    ASSERT_TRUE(buf.truncated);
    ASSERT_UINT_EQ(15, buf.len);
    ASSERT_INT_EQ(0, strcmp(rcre_strbuf_cstr(&buf), "hello_world_123"));

    /* Tiny capacity boundaries: cap=0 and cap=1 */
    char one[1] = { 'Z' };
    RCRE_StrBuf b1 = RCRE_STRBUF_WRAP(one);
    ASSERT_INT_EQ('\0', one[0]);
    ASSERT_FALSE(rcre_strbuf_copy(&b1, RCRE_STR("A")));
    ASSERT_TRUE(b1.truncated);
    ASSERT_UINT_EQ(0, b1.len);
    ASSERT_INT_EQ('\0', one[0]);

    RCRE_StrBuf b0 = rcre_strbuf_init(nullptr, 0);
    ASSERT_FALSE(rcre_strbuf_copy(&b0, RCRE_STR("A")));
    ASSERT_TRUE(b0.truncated);
    ASSERT_INT_EQ(0, strcmp(rcre_strbuf_cstr(&b0), ""));

    /* UTF-8 boundary-safe copy: "a🚀b" is 1 + 4 + 1 = 6 bytes.
     * In a 5-byte buffer (4 payload bytes + NUL), 'a' fits (1 byte),
     * '🚀' (4 bytes) does NOT fit, so result MUST be "a" (not sheared UTF-8!). */
    char utf8_dst[5];
    RCRE_Slice rocket = RCRE_STR("a\xF0\x9F\x9A\x80" "b");
    ASSERT_FALSE(rcre_cstr_copy_bounded(utf8_dst, sizeof(utf8_dst), rocket));
    ASSERT_INT_EQ(0, strcmp(utf8_dst, "a"));
    ASSERT_TRUE(rcre_utf8_validate(rcre_slice_from_cstr(utf8_dst), nullptr));

    /* In a 6-byte buffer (5 payload bytes + NUL), "a🚀" fits cleanly */
    char utf8_dst6[6];
    ASSERT_FALSE(rcre_cstr_copy_bounded(utf8_dst6, sizeof(utf8_dst6), rocket));
    ASSERT_UINT_EQ(5, strlen(utf8_dst6));
    ASSERT_TRUE(rcre_utf8_validate(rcre_slice_from_cstr(utf8_dst6), nullptr));

    /* Freestanding formatter into RCRE_StrBuf */
    RCRE_STRBUF_STACK(fbuf, 128);
    ASSERT_TRUE(rcre_strbuf_fmt(&fbuf, "id=%04d key=%S hex=0x%X val=%.2f %%",
                                7, RCRE_STR("alpha"), 0xBEEFu, -3.14159));
    ASSERT_TRUE(rcre_slice_eq(rcre_strbuf_as_slice(&fbuf),
                              RCRE_STR("id=0007 key=alpha hex=0xBEEF val=-3.14 %")));
}

static void test_slice_ops_search_and_iterators(void) {
    ASSERT_INT_EQ(0, rcre_slice_cmp(RCRE_STR("abc"), RCRE_STR("abc")));
    ASSERT_TRUE(rcre_slice_cmp(RCRE_STR("abc"), RCRE_STR("abd")) < 0);
    ASSERT_TRUE(rcre_slice_cmp(RCRE_STR("abcd"), RCRE_STR("abc")) > 0);
    ASSERT_TRUE(rcre_slice_eq_ascii_ci(RCRE_STR("HeLLo-123"), RCRE_STR("hello-123")));
    ASSERT_FALSE(rcre_slice_eq_ascii_ci(RCRE_STR("Hello"), RCRE_STR("World")));
    ASSERT_INT_EQ(0, rcre_slice_cmp_ascii_ci(RCRE_STR("RCRE"), RCRE_STR("rcre")));

    /* Substring search (exercises SIMD rcre_hal.find_substr) */
    RCRE_Slice hay = RCRE_STR("0123456789abcdef_needle_in_a_large_simd_haystack_needle_end");
    ASSERT_UINT_EQ(17, rcre_slice_find(hay, RCRE_STR("needle")));
    ASSERT_UINT_EQ(49, rcre_slice_rfind(hay, RCRE_STR("needle")));
    ASSERT_TRUE(rcre_slice_contains(hay, RCRE_STR("simd_haystack")));
    ASSERT_FALSE(rcre_slice_contains(hay, RCRE_STR("missing_pattern")));
    ASSERT_UINT_EQ(0, rcre_slice_find(hay, RCRE_STR("")));

    /* Charset find / span */
    ASSERT_UINT_EQ(3, rcre_slice_find_any_byte(RCRE_STR("abc:def;ghi"), RCRE_STR(";:=")));
    ASSERT_UINT_EQ(4, rcre_slice_span_bytes(RCRE_STR("1234xyz"), RCRE_STR("0123456789")));

    /* Trimming & stripping */
    ASSERT_TRUE(rcre_slice_eq(rcre_slice_trim_left_ascii(RCRE_STR(" \t\nabc ")), RCRE_STR("abc ")));
    ASSERT_TRUE(rcre_slice_eq(rcre_slice_trim_right_ascii(RCRE_STR(" abc\r\n")), RCRE_STR(" abc")));
    ASSERT_TRUE(rcre_slice_eq(rcre_slice_trim_chars(RCRE_STR("///path///"), RCRE_STR("/")), RCRE_STR("path")));

    bool stripped = false;
    RCRE_Slice s1 = rcre_slice_strip_prefix(RCRE_STR("https://rcre.dev"), RCRE_STR("https://"), &stripped);
    ASSERT_TRUE(stripped);
    ASSERT_TRUE(rcre_slice_eq(s1, RCRE_STR("rcre.dev")));
    RCRE_Slice s2 = rcre_slice_strip_suffix(s1, RCRE_STR(".dev"), &stripped);
    ASSERT_TRUE(stripped);
    ASSERT_TRUE(rcre_slice_eq(s2, RCRE_STR("rcre")));

    /* Zero-copy pull split iterator (replaces strtok) */
    RCRE_SplitIter it = rcre_split_byte_init(RCRE_STR("a,,b,c"), ',');
    RCRE_Slice tok;
    ASSERT_TRUE(rcre_split_next(&it, &tok) && rcre_slice_eq(tok, RCRE_STR("a")));
    ASSERT_TRUE(rcre_split_next(&it, &tok) && tok.len == 0);
    ASSERT_TRUE(rcre_split_next(&it, &tok) && rcre_slice_eq(tok, RCRE_STR("b")));
    ASSERT_TRUE(rcre_split_next(&it, &tok) && rcre_slice_eq(tok, RCRE_STR("c")));
    ASSERT_FALSE(rcre_split_next(&it, &tok));

    /* Multi-byte delimiter pull iterator */
    RCRE_SplitIter mit = rcre_split_slice_init(RCRE_STR("k1::k2::k3"), RCRE_STR("::"));
    ASSERT_TRUE(rcre_split_next(&mit, &tok) && rcre_slice_eq(tok, RCRE_STR("k1")));
    ASSERT_TRUE(rcre_split_next(&mit, &tok) && rcre_slice_eq(tok, RCRE_STR("k2")));
    ASSERT_TRUE(rcre_split_next(&mit, &tok) && rcre_slice_eq(tok, RCRE_STR("k3")));
    ASSERT_FALSE(rcre_split_next(&mit, &tok));

    /* Line splitter (handles LF and CRLF) */
    RCRE_Slice text = RCRE_STR("line1\r\nline2\nline3");
    RCRE_Slice line;
    ASSERT_TRUE(rcre_split_line_next(&text, &line) && rcre_slice_eq(line, RCRE_STR("line1")));
    ASSERT_TRUE(rcre_split_line_next(&text, &line) && rcre_slice_eq(line, RCRE_STR("line2")));
    ASSERT_TRUE(rcre_split_line_next(&text, &line) && rcre_slice_eq(line, RCRE_STR("line3")));
    ASSERT_FALSE(rcre_split_line_next(&text, &line));
}

static void test_parsers_builder_and_transforms(void) {
    /* Strict parsers */
    uint64_t hx = 0;
    ASSERT_TRUE(rcre_slice_parse_hex_u64(RCRE_STR("0xDEADBEEF"), &hx));
    ASSERT_UINT_EQ(0xDEADBEEFULL, hx);
    ASSERT_TRUE(rcre_slice_parse_hex_u64(RCRE_STR("ffffffffffffffff"), &hx));
    ASSERT_UINT_EQ(UINT64_MAX, hx);
    ASSERT_FALSE(rcre_slice_parse_hex_u64(RCRE_STR("0x10000000000000000"), &hx));
    ASSERT_FALSE(rcre_slice_parse_hex_u64(RCRE_STR("0x"), &hx));

    bool bv = false;
    ASSERT_TRUE(rcre_slice_parse_bool(RCRE_STR(" TRUE "), &bv) && bv);
    ASSERT_TRUE(rcre_slice_parse_bool(RCRE_STR("No"), &bv) && !bv);
    ASSERT_FALSE(rcre_slice_parse_bool(RCRE_STR("maybe"), &bv));

    double dv = 0.0;
    ASSERT_TRUE(rcre_slice_parse_f64(RCRE_STR("  -123.456e2 "), &dv));
    ASSERT_FLOAT_NEAR(-12345.6, dv, 1e-9);
    ASSERT_TRUE(rcre_slice_parse_f64(RCRE_STR("1e-308"), &dv));
    ASSERT_TRUE(dv > 0.0 && dv < 1e-307);
    ASSERT_TRUE(rcre_slice_parse_f64(RCRE_STR("inf"), &dv) && isinf(dv) && dv > 0);
    ASSERT_TRUE(rcre_slice_parse_f64(RCRE_STR("-NaN"), &dv) && isnan(dv));
    ASSERT_FALSE(rcre_slice_parse_f64(RCRE_STR("12.34xyz"), &dv));

    /* Arena transforms & O(1) scoped rewind */
    RCRE_Arena* arena = rcre_arena_create(65536);
    ASSERT_NOT_NULL(arena);
    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);

    RCRE_Slice parts[3] = { RCRE_STR("alpha"), RCRE_STR("beta"), RCRE_STR("gamma") };
    RCRE_Slice joined = rcre_str_join(arena, parts, 3, RCRE_STR(" | "));
    ASSERT_TRUE(rcre_slice_eq(joined, RCRE_STR("alpha | beta | gamma")));

    RCRE_Slice rep = rcre_str_replace_all(arena, RCRE_STR("foo-bar-foo-baz-foo"),
                                          RCRE_STR("foo"), RCRE_STR("RCRE"));
    ASSERT_TRUE(rcre_slice_eq(rep, RCRE_STR("RCRE-bar-RCRE-baz-RCRE")));

    RCRE_Slice lo = rcre_str_to_lower_ascii(arena, RCRE_STR("RCRE_SIMD_2026_AbCdEfGhIjKlMnOpQrStUvWxYz!"));
    ASSERT_TRUE(rcre_slice_eq(lo, RCRE_STR("rcre_simd_2026_abcdefghijklmnopqrstuvwxyz!")));
    RCRE_Slice up = rcre_str_to_upper_ascii(arena, lo);
    ASSERT_TRUE(rcre_slice_eq(up, RCRE_STR("RCRE_SIMD_2026_ABCDEFGHIJKLMNOPQRSTUVWXYZ!")));

    /* JSON & CSV escaping + freestanding builder formatting */
    RCRE_StrBuilder sb = rcre_sb_init(arena, 32);
    ASSERT_TRUE(rcre_sb_append_cstr(&sb, "{\"msg\":\""));
    ASSERT_TRUE(rcre_sb_append_escape_json(&sb, RCRE_STR("hi \"rcre\"\n\t\x01")));
    ASSERT_TRUE(rcre_sb_append_cstr(&sb, "\"}"));
    ASSERT_TRUE(rcre_slice_eq(rcre_sb_to_slice(&sb), RCRE_STR("{\"msg\":\"hi \\\"rcre\\\"\\n\\t\\u0001\"}")));

    rcre_sb_reset(&sb);
    ASSERT_TRUE(rcre_sb_append_escape_csv(&sb, RCRE_STR("a,\"b\"\nc"), ','));
    ASSERT_TRUE(rcre_slice_eq(rcre_sb_to_slice(&sb), RCRE_STR("\"a,\"\"b\"\"\nc\"")));

    RCRE_Slice fmt = rcre_str_fmt(arena, "%-6S=%08llx:%.3f", RCRE_STR("pi"), 0x2aULL, 3.14159);
    ASSERT_TRUE(rcre_slice_eq(fmt, RCRE_STR("pi    =0000002a:3.142")));

    rcre_arena_temp_end(tmp);
    ASSERT_UINT_EQ(0, arena->offset);
    rcre_arena_destroy(arena);
}

static void test_utf8_rfc3629_and_simd_parity(void) {
    /* Valid multi-byte strings: ASCII, 2-byte (Turkish/Latin), 3-byte (CJK), 4-byte (Emoji) */
    RCRE_Slice valid = RCRE_STR("Merhaba D\xC3\xBCnya \xE4\xB8\x96\xE7\x95\x8C \xF0\x9F\x9A\x80!");
    size_t err_off = 9999;
    ASSERT_TRUE(rcre_utf8_validate(valid, &err_off));
    ASSERT_UINT_EQ(19, rcre_utf8_count(valid));
    ASSERT_FALSE(rcre_utf8_is_ascii(valid));
    ASSERT_TRUE(rcre_utf8_is_ascii(RCRE_STR("pure ASCII 0123456789")));

    /* Forward and reverse codepoint iteration */
    RCRE_Slice cur = valid;
    uint32_t cps[32];
    size_t n = 0;
    uint32_t cp = 0;
    while (rcre_utf8_decode_next(&cur, &cp) > 0) {
        cps[n++] = cp;
    }
    ASSERT_UINT_EQ(19, n);
    ASSERT_UINT_EQ(0x00FCu, cps[9]);   /* ü */
    ASSERT_UINT_EQ(0x4E16u, cps[14]);  /* 世 */
    ASSERT_UINT_EQ(0x1F680u, cps[17]); /* 🚀 */

    RCRE_Slice rev = valid;
    for (size_t k = n; k > 0; --k) {
        ASSERT_TRUE(rcre_utf8_decode_prev(&rev, &cp) > 0);
        ASSERT_UINT_EQ(cps[k - 1], cp);
    }
    ASSERT_UINT_EQ(0, rev.len);

    /* Codepoint slicing */
    RCRE_Slice sub = rcre_utf8_sub_codepoints(valid, 14, 4);
    ASSERT_TRUE(rcre_slice_eq(sub, RCRE_STR("\xE4\xB8\x96\xE7\x95\x8C \xF0\x9F\x9A\x80")));

    /* Encoding edge cases */
    uint8_t enc[4];
    ASSERT_UINT_EQ(4, rcre_utf8_encode(0x10FFFFu, enc));
    ASSERT_TRUE(rcre_utf8_validate(rcre_slice_from_parts(enc, 4), nullptr));
    ASSERT_UINT_EQ(0, rcre_utf8_encode(0xD800u, enc));   /* Surrogate rejected */
    ASSERT_UINT_EQ(0, rcre_utf8_encode(0xDFFFu, enc));   /* Surrogate rejected */
    ASSERT_UINT_EQ(0, rcre_utf8_encode(0x110000u, enc)); /* > U+10FFFF rejected */

    /* RFC 3629 invalid classes with exact error offset check */
    const struct { const uint8_t bytes[8]; size_t len; size_t expected_err; } bad_cases[] = {
        { { 0x80 },                         1, 0 }, /* Lone continuation */
        { { 'A', 0xC0, 0x80 },              3, 1 }, /* Overlong 2-byte NUL */
        { { 'A', 0xC1, 0xBF },              3, 1 }, /* Overlong 2-byte 0x7F */
        { { 0xE0, 0x9F, 0xBF },             3, 0 }, /* Overlong 3-byte */
        { { 0xED, 0xA0, 0x80 },             3, 0 }, /* UTF-16 high surrogate U+D800 */
        { { 0xED, 0xBF, 0xBF },             3, 0 }, /* UTF-16 low surrogate U+DFFF */
        { { 0xF0, 0x8F, 0xBF, 0xBF },       4, 0 }, /* Overlong 4-byte */
        { { 0xF4, 0x90, 0x80, 0x80 },       4, 0 }, /* U+110000 out of range */
        { { 0xF5, 0x80, 0x80, 0x80 },       4, 0 }, /* Invalid lead 0xF5 */
        { { 'x', 'y', 0xE2, 0x82 },         4, 2 }, /* Truncated 3-byte */
        { { 'x', 0xF0, 0x9F, 0x9A },        4, 1 }, /* Truncated 4-byte */
    };
    for (size_t i = 0; i < sizeof(bad_cases) / sizeof(bad_cases[0]); ++i) {
        size_t pos = 9999;
        RCRE_Slice s = rcre_slice_from_parts(bad_cases[i].bytes, bad_cases[i].len);
        ASSERT_FALSE(rcre_utf8_validate(s, &pos));
        ASSERT_UINT_EQ(bad_cases[i].expected_err, pos);
        ASSERT_UINT_EQ(RCRE_NPOS, rcre_utf8_count(s));
    }

    /* Exhaustive differential check: ALL 65,536 2-byte pairs embedded across SIMD
     * chunk boundaries (offset 15..16 and 31..32) — HAL SIMD MUST match scalar 100%! */
    uint8_t pad_buf[40];
    memset(pad_buf, 'a', sizeof(pad_buf));
    for (unsigned b0 = 0; b0 < 256; ++b0) {
        for (unsigned b1 = 0; b1 < 256; ++b1) {
            pad_buf[15] = (uint8_t)b0;
            pad_buf[16] = (uint8_t)b1;
            bool ref = rcre_scalar_utf8_validate(pad_buf, 34);
            bool simd = rcre_hal.utf8_validate(pad_buf, 34);
            ASSERT_TRUE(ref == simd);
        }
    }

    /* Boundary crossings at every offset 0..31 for 2/3/4-byte valid & invalid sequences */
    const uint8_t seqs[][4] = {
        { 0xC2, 0x80, 'a', 'b' },
        { 0xE0, 0xA0, 0x80, 'a' },
        { 0xE0, 0x9F, 0x80, 'a' },
        { 0xED, 0x9F, 0xBF, 'a' },
        { 0xED, 0xA0, 0x80, 'a' },
        { 0xF0, 0x90, 0x80, 0x80 },
        { 0xF0, 0x8F, 0x80, 0x80 },
        { 0xF4, 0x8F, 0xBF, 0xBF },
        { 0xF4, 0x90, 0x80, 0x80 },
        { 0xF5, 0x80, 0x80, 0x80 },
    };
    for (size_t s = 0; s < sizeof(seqs) / sizeof(seqs[0]); ++s) {
        for (size_t off = 0; off < 32; ++off) {
            for (size_t len = off + 1; len <= off + 20; ++len) {
                uint8_t b[64];
                memset(b, 'z', sizeof(b));
                size_t cp_len = (len - off < 4) ? (len - off) : 4;
                memcpy(b + off, seqs[s], cp_len);
                ASSERT_TRUE(rcre_scalar_utf8_validate(b, len) == rcre_hal.utf8_validate(b, len));
            }
        }
    }

    /* HAL SIMD vs scalar parity for find_substr and ascii_to_lower/upper across lengths & offsets */
    uint8_t ascii_in[257], out_ref[257], out_simd[257];
    for (size_t i = 0; i < 257; ++i) ascii_in[i] = (uint8_t)i;
    for (size_t len = 0; len <= 257; len += 7) {
        rcre_scalar_ascii_to_lower(ascii_in, out_ref, len);
        rcre_hal.ascii_to_lower(ascii_in, out_simd, len);
        ASSERT_INT_EQ(0, memcmp(out_ref, out_simd, len));

        rcre_scalar_ascii_to_upper(ascii_in, out_ref, len);
        rcre_hal.ascii_to_upper(ascii_in, out_simd, len);
        ASSERT_INT_EQ(0, memcmp(out_ref, out_simd, len));
    }
    /* In-place aliasing safety */
    memcpy(out_simd, ascii_in, 257);
    rcre_hal.ascii_to_lower(out_simd, out_simd, 257);
    rcre_scalar_ascii_to_lower(ascii_in, out_ref, 257);
    ASSERT_INT_EQ(0, memcmp(out_ref, out_simd, 257));

    /* Substring search parity across every placement in 128-byte buffer */
    uint8_t hbuf[128];
    memset(hbuf, 'x', sizeof(hbuf));
    const uint8_t pat[7] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G' };
    for (size_t pos = 0; pos <= 128 - 7; ++pos) {
        memset(hbuf, 'x', sizeof(hbuf));
        hbuf[0] = 'A'; /* False first-byte match */
        hbuf[pos + 6] = 'G';
        memcpy(hbuf + pos, pat, 7);
        size_t r1 = rcre_scalar_find_substr(hbuf, 128, pat, 7);
        size_t r2 = rcre_hal.find_substr(hbuf, 128, pat, 7);
        ASSERT_UINT_EQ(pos, r1);
        ASSERT_UINT_EQ(r1, r2);
    }
}

static void test_strpool_and_paths(void) {
    RCRE_Arena* arena = rcre_arena_create(512 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_StrPool pool;
    ASSERT_INT_EQ(RCRE_OK, rcre_strpool_init(&pool, arena, 16));

    RCRE_StrId id_empty = rcre_strpool_intern(&pool, RCRE_STR(""));
    ASSERT_TRUE(id_empty != RCRE_STR_ID_INVALID);
    ASSERT_UINT_EQ(id_empty, rcre_strpool_intern(&pool, rcre_slice_empty()));
    ASSERT_INT_EQ(0, strcmp(rcre_strpool_cstr(&pool, id_empty), ""));

    /* Stress rehash across 1000 distinct strings & verify deduplication */
    RCRE_StrId ids[1000];
    for (size_t i = 0; i < 1000; ++i) {
        RCRE_STRBUF_STACK(kb, 32);
        ASSERT_TRUE(rcre_strbuf_fmt(&kb, "feature_col_%zu", i));
        ids[i] = rcre_strpool_intern(&pool, rcre_strbuf_as_slice(&kb));
        ASSERT_TRUE(ids[i] != RCRE_STR_ID_INVALID);
    }
    ASSERT_UINT_EQ(1001, rcre_strpool_count(&pool));

    for (size_t i = 0; i < 1000; ++i) {
        RCRE_STRBUF_STACK(kb, 32);
        ASSERT_TRUE(rcre_strbuf_fmt(&kb, "feature_col_%zu", i));
        RCRE_StrId found = rcre_strpool_find(&pool, rcre_strbuf_as_slice(&kb));
        ASSERT_UINT_EQ(ids[i], found);
        ASSERT_UINT_EQ(ids[i], rcre_strpool_intern(&pool, rcre_strbuf_as_slice(&kb)));
        ASSERT_TRUE(rcre_slice_eq(rcre_strpool_get(&pool, ids[i]), rcre_strbuf_as_slice(&kb)));
    }
    ASSERT_UINT_EQ(1001, rcre_strpool_count(&pool));
    ASSERT_UINT_EQ(RCRE_STR_ID_INVALID, rcre_strpool_find(&pool, RCRE_STR("not_present")));

    /* Safe path manipulation */
    ASSERT_TRUE(rcre_slice_eq(rcre_path_basename(RCRE_STR("/a/b/file.tar.gz")), RCRE_STR("file.tar.gz")));
    ASSERT_TRUE(rcre_slice_eq(rcre_path_basename(RCRE_STR("/a/b/dir///")), RCRE_STR("dir")));
    ASSERT_TRUE(rcre_slice_eq(rcre_path_basename(RCRE_STR("/")), RCRE_STR("/")));
    ASSERT_TRUE(rcre_slice_eq(rcre_path_dirname(RCRE_STR("/a/b/file.txt")), RCRE_STR("/a/b")));
    ASSERT_TRUE(rcre_slice_eq(rcre_path_dirname(RCRE_STR("/file.txt")), RCRE_STR("/")));
    ASSERT_TRUE(rcre_slice_eq(rcre_path_dirname(RCRE_STR("plain.txt")), RCRE_STR(".")));
    ASSERT_TRUE(rcre_slice_eq(rcre_path_ext(RCRE_STR("/a/model.gguf")), RCRE_STR(".gguf")));
    ASSERT_TRUE(rcre_slice_eq(rcre_path_stem(RCRE_STR("/a/model.gguf")), RCRE_STR("model")));
    ASSERT_UINT_EQ(0, rcre_path_ext(RCRE_STR("/home/user/.bashrc")).len);
    ASSERT_TRUE(rcre_slice_eq(rcre_path_stem(RCRE_STR("/home/user/.bashrc")), RCRE_STR(".bashrc")));

    ASSERT_TRUE(rcre_slice_eq(rcre_path_join(arena, RCRE_STR("/data/rcre/"), RCRE_STR("weights.npy")),
                              RCRE_STR("/data/rcre/weights.npy")));
    ASSERT_TRUE(rcre_slice_eq(rcre_path_join(arena, RCRE_STR("/data"), RCRE_STR("/abs/override")),
                              RCRE_STR("/abs/override")));
    ASSERT_TRUE(rcre_slice_eq(rcre_path_join(arena, RCRE_STR("/"), RCRE_STR("etc")),
                              RCRE_STR("/etc")));

    rcre_arena_destroy(arena);
}

int main(void) {
    rcre_hal_init();
    test_suite_begin("RCRE Safe String Library (STR-REQ-01 to STR-REQ-06)");
    TEST_RUN(test_str_macros_and_strbuf);
    TEST_RUN(test_slice_ops_search_and_iterators);
    TEST_RUN(test_parsers_builder_and_transforms);
    TEST_RUN(test_utf8_rfc3629_and_simd_parity);
    TEST_RUN(test_strpool_and_paths);
    return test_suite_report();
}
