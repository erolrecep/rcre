/**
 * @file test_slice.c
 * @brief Unit & boundary tests for RCRE_Slice & Arena-backed RCRE_StrBuilder
 */

#include "rcre.h"
#include "test_framework.h"
#include <limits.h>

static void test_slice_views_and_bounds(void) {
    RCRE_Slice empty = rcre_slice_empty();
    ASSERT_NULL(empty.data);
    ASSERT_UINT_EQ(0, empty.len);
    ASSERT_TRUE(rcre_slice_eq(empty, rcre_slice_from_cstr("")));
    ASSERT_TRUE(rcre_slice_eq(empty, rcre_slice_from_cstr(nullptr)));

    RCRE_Slice path = rcre_slice_from_cstr("/data/genomics/sample_01.fastq.gz");
    ASSERT_UINT_EQ(33, path.len);
    ASSERT_TRUE(rcre_slice_starts_with(path, rcre_slice_from_cstr("/data/genomics/")));
    ASSERT_TRUE(rcre_slice_ends_with(path, rcre_slice_from_cstr(".fastq.gz")));
    ASSERT_FALSE(rcre_slice_ends_with(path, rcre_slice_from_cstr(".bam")));

    /* Clamped O(1) sub-slices (zero UB on out-of-range indices) */
    RCRE_Slice clamped = rcre_slice_sub(path, 6, 9999);
    ASSERT_TRUE(rcre_slice_eq_cstr(clamped, "genomics/sample_01.fastq.gz"));
    ASSERT_UINT_EQ(0, rcre_slice_sub(path, 25, 10).len);
    ASSERT_UINT_EQ(0, rcre_slice_sub(path, 500, 600).len);

    /* Reverse byte lookup for basename {/} and extension stripping {.} */
    size_t slash = rcre_slice_rfind_byte(path, '/');
    ASSERT_UINT_EQ(14, slash);
    RCRE_Slice base = rcre_slice_sub(path, slash + 1, path.len);
    ASSERT_TRUE(rcre_slice_eq_cstr(base, "sample_01.fastq.gz"));

    size_t dot = rcre_slice_rfind_byte(base, '.');
    ASSERT_UINT_EQ(15, dot);
    RCRE_Slice stem = rcre_slice_sub(base, 0, dot);
    ASSERT_TRUE(rcre_slice_eq_cstr(stem, "sample_01.fastq"));
}

static void test_slice_embedded_nuls_and_splitting(void) {
    /* Binary --null (-0) stream containing embedded NULs */
    const uint8_t raw[] = { 'a', 'l', 'p', 'h', 'a', '\0', 'b', 'e', 't', 'a', '\0', 'g', 'a', 'm', 'm', 'a' };
    RCRE_Slice stream = rcre_slice_from_parts(raw, sizeof(raw));

    ASSERT_UINT_EQ(2, rcre_slice_count_byte(stream, '\0'));

    RCRE_SlicePair p1 = rcre_slice_split_once_byte(stream, '\0');
    ASSERT_TRUE(p1.found);
    ASSERT_TRUE(rcre_slice_eq_cstr(p1.head, "alpha"));

    RCRE_SlicePair p2 = rcre_slice_split_once_byte(p1.tail, '\0');
    ASSERT_TRUE(p2.found);
    ASSERT_TRUE(rcre_slice_eq_cstr(p2.head, "beta"));
    ASSERT_TRUE(rcre_slice_eq_cstr(p2.tail, "gamma"));

    RCRE_SlicePair p3 = rcre_slice_split_once_byte(p2.tail, '\0');
    ASSERT_FALSE(p3.found);
    ASSERT_TRUE(rcre_slice_eq_cstr(p3.head, "gamma"));
    ASSERT_UINT_EQ(0, p3.tail.len);

    /* Multi-byte delimiter splitting (e.g. GNU Parallel ::: separator) */
    RCRE_Slice cmdline = rcre_slice_from_cstr("  echo {1}:{2} ::: A B C ::: 1 2  ");
    RCRE_Slice trimmed = rcre_slice_trim_ascii(cmdline);
    ASSERT_TRUE(rcre_slice_eq_cstr(trimmed, "echo {1}:{2} ::: A B C ::: 1 2"));

    RCRE_Slice sep = rcre_slice_from_cstr(" ::: ");
    RCRE_SlicePair s1 = rcre_slice_split_once(trimmed, sep);
    ASSERT_TRUE(s1.found);
    ASSERT_TRUE(rcre_slice_eq_cstr(s1.head, "echo {1}:{2}"));

    RCRE_SlicePair s2 = rcre_slice_split_once(s1.tail, sep);
    ASSERT_TRUE(s2.found);
    ASSERT_TRUE(rcre_slice_eq_cstr(s2.head, "A B C"));
    ASSERT_TRUE(rcre_slice_eq_cstr(s2.tail, "1 2"));
}

static void test_slice_integer_parsing(void) {
    uint64_t u = 0;
    int64_t  v = 0;

    ASSERT_TRUE(rcre_slice_parse_u64(rcre_slice_from_cstr("0"), &u));
    ASSERT_UINT_EQ(0ULL, u);

    ASSERT_TRUE(rcre_slice_parse_u64(rcre_slice_from_cstr("18446744073709551615"), &u));
    ASSERT_UINT_EQ(UINT64_MAX, u);
    /* Overflow +1 must fail cleanly */
    ASSERT_FALSE(rcre_slice_parse_u64(rcre_slice_from_cstr("18446744073709551616"), &u));
    ASSERT_FALSE(rcre_slice_parse_u64(rcre_slice_from_cstr("123abc"), &u));
    ASSERT_FALSE(rcre_slice_parse_u64(rcre_slice_empty(), &u));

    ASSERT_TRUE(rcre_slice_parse_i64(rcre_slice_from_cstr("-9223372036854775808"), &v));
    ASSERT_TRUE(v == INT64_MIN);
    ASSERT_TRUE(rcre_slice_parse_i64(rcre_slice_from_cstr("9223372036854775807"), &v));
    ASSERT_TRUE(v == INT64_MAX);
    ASSERT_FALSE(rcre_slice_parse_i64(rcre_slice_from_cstr("-9223372036854775809"), &v));
}

static void test_arena_str_builder_and_rewind(void) {
    RCRE_Arena* arena = rcre_arena_create(4096);
    ASSERT_NOT_NULL(arena);

    for (size_t job = 1; job <= 100; ++job) {
        RCRE_ArenaTemp scratch = rcre_arena_temp_begin(arena);

        RCRE_StrBuilder sb = rcre_sb_init(arena, 8); /* Force multiple growths */
        ASSERT_TRUE(rcre_sb_append_cstr(&sb, "python3 infer.py --slot="));
        ASSERT_TRUE(rcre_sb_append_u64(&sb, (job - 1) % 8));
        ASSERT_TRUE(rcre_sb_append_cstr(&sb, " --seq="));
        ASSERT_TRUE(rcre_sb_append_i64(&sb, -(int64_t)job));
        ASSERT_TRUE(rcre_sb_append_byte(&sb, '!'));

        const char* rendered = rcre_sb_to_cstr(&sb);
        ASSERT_NOT_NULL(rendered);
        if (job == 42) {
            ASSERT_TRUE(rcre_slice_eq_cstr(rcre_sb_to_slice(&sb),
                        "python3 infer.py --slot=1 --seq=-42!"));
        }

        char* dup = rcre_arena_cstr_dup(arena, rcre_sb_to_slice(&sb));
        ASSERT_NOT_NULL(dup);
        ASSERT_INT_EQ(0, strcmp(rendered, dup));

        rcre_arena_temp_end(scratch);
        ASSERT_UINT_EQ(0, arena->offset); /* Zero arena leak across 100 job expansions */
    }

    /* Exhaustion test: builder fails gracefully without UB when arena is full */
    RCRE_StrBuilder tiny = rcre_sb_init(arena, 16);
    uint8_t big_chunk[5000];
    memset(big_chunk, 'X', sizeof(big_chunk));
    ASSERT_FALSE(rcre_sb_append_slice(&tiny, rcre_slice_from_parts(big_chunk, sizeof(big_chunk))));
    ASSERT_NULL(rcre_sb_to_cstr(&tiny));

    rcre_arena_destroy(arena);
}

int main(void) {
    rcre_hal_init();
    test_suite_begin("RCRE Zero-Copy Byte Slices & Arena String Builder (SLC-REQ-01 to 03)");
    TEST_RUN(test_slice_views_and_bounds);
    TEST_RUN(test_slice_embedded_nuls_and_splitting);
    TEST_RUN(test_slice_integer_parsing);
    TEST_RUN(test_arena_str_builder_and_rewind);
    return test_suite_report();
}
