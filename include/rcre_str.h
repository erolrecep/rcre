/**
 * @file rcre_str.h
 * @brief RCRE Safe, Zero-Malloc, SIMD-Accelerated C23 String Library
 *
 * Satisfies STR-REQ-01 through STR-REQ-06:
 *   - STR-REQ-01: Length-prefixed zero-copy views (RCRE_Slice / RCRE_StrView)
 *                 and compile-time literal macros (zero strlen on literals).
 *   - STR-REQ-02: Fixed-capacity stack/struct bounded buffer (RCRE_StrBuf) —
 *                 the direct, always-NUL-terminated, UTF-8-safe replacement
 *                 for strcpy / strncpy / strlcpy / sprintf.
 *   - STR-REQ-03: Rich slice comparison, search, trimming, pull-based splitting
 *                 (replaces strtok), strict numeric parsing, and freestanding
 *                 arena-backed RCRE_StrBuilder formatting & escaping.
 *   - STR-REQ-04: Strict RFC 3629 UTF-8 validation, decoding, encoding,
 *                 counting, and codepoint-boundary-safe truncation.
 *   - STR-REQ-05: Dynamic HAL SIMD acceleration (utf8_validate, find_substr,
 *                 ascii_to_lower, ascii_to_upper across NEON/AVX2/WASM/Scalar).
 *   - STR-REQ-06: Arena-backed string interning pool (RCRE_StrPool) and safe
 *                 path manipulation (rcre_path_*).
 *
 * Architectural guarantees:
 *   - Zero malloc/calloc/realloc/free anywhere in this module.
 *   - Pure, re-entrant, lock-free kernels safe for RCRE_ThreadPool slabs.
 *   - Compiles cleanly under -DRCRE_FREESTANDING (no <stdio.h> or <stdlib.h>).
 *   - Opt-in compile-time poisoning of legacy unsafe <string.h> functions via
 *     #define RCRE_BAN_UNSAFE_CSTR before including this header.
 */

#ifndef RCRE_STR_H
#define RCRE_STR_H

#include "rcre.h"
#include <stdarg.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Ergonomic Types & C23 Compile-Time Macros (STR-REQ-01)                 */
/* ========================================================================= */

/** Semantic alias for read-only non-owning string views (identical ABI to RCRE_Slice). */
typedef RCRE_Slice RCRE_StrView;

/**
 * @brief Constructs a zero-cost RCRE_Slice from a C string literal at compile time.
 *        Zero strlen() overhead; empty literal "" yields rcre_slice_empty().
 */
#define RCRE_STR(lit)                                                        \
    ((RCRE_Slice){                                                           \
        .data = (sizeof(lit) > 1U) ? (const uint8_t*)(lit) : nullptr,        \
        .len  = sizeof(lit) - 1U                                             \
    })

/**
 * @brief printf-family format specifier & argument pair for non-NUL slices.
 *        Example: printf("key=" RCRE_STR_FMT "\n", RCRE_STR_ARG(slice));
 */
#define RCRE_STR_FMT "%.*s"
#define RCRE_STR_ARG(s)                                                      \
    (int)((s).len > (size_t)INT_MAX ? (size_t)INT_MAX : (s).len),            \
    ((s).data ? (const char*)(s).data : "")

/* ========================================================================= */
/* 2. Fixed-Capacity Stack/Struct Bounded Buffer (STR-REQ-02)                */
/*    Direct replacement for strcpy / strncpy / strlcpy / snprintf.          */
/* ========================================================================= */

typedef struct RCRE_StrBuf {
    uint8_t* data;      /* Caller-owned backing storage                      */
    size_t   len;       /* Active byte length (strictly < cap when cap > 0)  */
    size_t   cap;       /* Total capacity in bytes including trailing NUL    */
    bool     truncated; /* Sticky flag: true if any write exceeded capacity  */
} RCRE_StrBuf;

/** Wraps a fixed-size char/uint8_t array into an initialized RCRE_StrBuf. */
#define RCRE_STRBUF_WRAP(arr) rcre_strbuf_init((arr), sizeof(arr))

/** Declares a stack-allocated backing buffer and its RCRE_StrBuf wrapper. */
#define RCRE_STRBUF_STACK(name, N)                                           \
    uint8_t name##_rcre_storage_[(N)];                                       \
    RCRE_StrBuf name = rcre_strbuf_init(name##_rcre_storage_, sizeof(name##_rcre_storage_))

RCRE_NODISCARD RCRE_StrBuf rcre_strbuf_init(void* backing, size_t cap);
void rcre_strbuf_reset(RCRE_StrBuf* sb);

RCRE_NODISCARD bool rcre_strbuf_copy(RCRE_StrBuf* sb, RCRE_Slice src);
RCRE_NODISCARD bool rcre_strbuf_copy_cstr(RCRE_StrBuf* sb, const char* cstr);
RCRE_NODISCARD bool rcre_strbuf_copy_utf8(RCRE_StrBuf* sb, RCRE_Slice src);

RCRE_NODISCARD bool rcre_strbuf_append(RCRE_StrBuf* sb, RCRE_Slice src);
RCRE_NODISCARD bool rcre_strbuf_append_cstr(RCRE_StrBuf* sb, const char* cstr);
RCRE_NODISCARD bool rcre_strbuf_append_utf8(RCRE_StrBuf* sb, RCRE_Slice src);
RCRE_NODISCARD bool rcre_strbuf_append_byte(RCRE_StrBuf* sb, uint8_t b);
RCRE_NODISCARD bool rcre_strbuf_append_cp(RCRE_StrBuf* sb, uint32_t cp);
RCRE_NODISCARD bool rcre_strbuf_append_u64(RCRE_StrBuf* sb, uint64_t val);
RCRE_NODISCARD bool rcre_strbuf_append_i64(RCRE_StrBuf* sb, int64_t val);
RCRE_NODISCARD bool rcre_strbuf_append_hex(RCRE_StrBuf* sb, uint64_t val, bool uppercase);
RCRE_NODISCARD bool rcre_strbuf_append_f64(RCRE_StrBuf* sb, double val, int precision);
RCRE_NODISCARD bool rcre_strbuf_vfmt(RCRE_StrBuf* sb, const char* fmt, va_list ap);
RCRE_NODISCARD bool rcre_strbuf_fmt(RCRE_StrBuf* sb, const char* fmt, ...);

RCRE_NODISCARD RCRE_Slice rcre_strbuf_as_slice(const RCRE_StrBuf* sb);
RCRE_NODISCARD const char* rcre_strbuf_cstr(const RCRE_StrBuf* sb);

/**
 * @brief One-call UTF-8-safe bounded copy into a legacy char[N] struct field.
 *        Always NUL-terminates when dst && dst_cap > 0; never reads past src.len;
 *        never shears multi-byte UTF-8 codepoints. Returns false if truncated or invalid.
 */
RCRE_NODISCARD bool rcre_cstr_copy_bounded(char* dst, size_t dst_cap, RCRE_Slice src);

/* ========================================================================= */
/* 3. Expanded Zero-Copy Slice Operations & Pull Iterators (STR-REQ-03)      */
/* ========================================================================= */

typedef struct RCRE_SplitIter {
    RCRE_Slice rest;
    RCRE_Slice delim;
    uint8_t    delim_byte;
    bool       use_byte;
    bool       done;
} RCRE_SplitIter;

/* Comparison & Hashing */
RCRE_NODISCARD int rcre_slice_cmp(RCRE_Slice a, RCRE_Slice b);
RCRE_NODISCARD bool rcre_slice_eq_ascii_ci(RCRE_Slice a, RCRE_Slice b);
RCRE_NODISCARD int rcre_slice_cmp_ascii_ci(RCRE_Slice a, RCRE_Slice b);
RCRE_NODISCARD uint64_t rcre_slice_hash64(RCRE_Slice s);
RCRE_NODISCARD uint64_t rcre_slice_hash64_seeded(RCRE_Slice s, uint64_t seed);

/* SIMD-Accelerated Substring & Charset Search */
RCRE_NODISCARD size_t rcre_slice_find(RCRE_Slice hay, RCRE_Slice needle);
RCRE_NODISCARD size_t rcre_slice_rfind(RCRE_Slice hay, RCRE_Slice needle);
RCRE_NODISCARD bool rcre_slice_contains(RCRE_Slice hay, RCRE_Slice needle);
RCRE_NODISCARD size_t rcre_slice_find_any_byte(RCRE_Slice s, RCRE_Slice charset);
RCRE_NODISCARD size_t rcre_slice_span_bytes(RCRE_Slice s, RCRE_Slice charset);

/* Trimming & Prefix/Suffix Stripping */
RCRE_NODISCARD RCRE_Slice rcre_slice_trim_left_ascii(RCRE_Slice s);
RCRE_NODISCARD RCRE_Slice rcre_slice_trim_right_ascii(RCRE_Slice s);
RCRE_NODISCARD RCRE_Slice rcre_slice_trim_chars(RCRE_Slice s, RCRE_Slice charset);
RCRE_NODISCARD RCRE_Slice rcre_slice_strip_prefix(RCRE_Slice s, RCRE_Slice prefix, bool* stripped);
RCRE_NODISCARD RCRE_Slice rcre_slice_strip_suffix(RCRE_Slice s, RCRE_Slice suffix, bool* stripped);

/* Zero-Copy Re-Entrant Splitting (replaces strtok) */
RCRE_NODISCARD RCRE_SplitIter rcre_split_byte_init(RCRE_Slice s, uint8_t delim);
RCRE_NODISCARD RCRE_SplitIter rcre_split_slice_init(RCRE_Slice s, RCRE_Slice delim);
RCRE_NODISCARD bool rcre_split_next(RCRE_SplitIter* it, RCRE_Slice* out_token);
RCRE_NODISCARD bool rcre_split_line_next(RCRE_Slice* cursor, RCRE_Slice* out_line);

/* Strict Zero-Allocation Parsers (freestanding, overflow-checked) */
RCRE_NODISCARD bool rcre_slice_parse_hex_u64(RCRE_Slice s, uint64_t* out_val);
RCRE_NODISCARD bool rcre_slice_parse_bool(RCRE_Slice s, bool* out_val);
RCRE_NODISCARD bool rcre_slice_parse_f64(RCRE_Slice s, double* out_val);

/* ========================================================================= */
/* 4. Expanded Arena String Builder & Transforms (STR-REQ-03)                */
/* ========================================================================= */

void rcre_sb_reset(RCRE_StrBuilder* sb);
bool rcre_sb_append_hex(RCRE_StrBuilder* sb, uint64_t val, bool uppercase);
bool rcre_sb_append_f64(RCRE_StrBuilder* sb, double val, int precision);
bool rcre_sb_append_cp(RCRE_StrBuilder* sb, uint32_t cp);
bool rcre_sb_append_repeat(RCRE_StrBuilder* sb, RCRE_Slice unit, size_t count);
bool rcre_sb_append_vfmt(RCRE_StrBuilder* sb, const char* fmt, va_list ap);
bool rcre_sb_append_fmt(RCRE_StrBuilder* sb, const char* fmt, ...);
bool rcre_sb_append_escape_json(RCRE_StrBuilder* sb, RCRE_Slice s);
bool rcre_sb_append_escape_csv(RCRE_StrBuilder* sb, RCRE_Slice s, uint8_t delim);

/* Arena-backed functional transforms (always NUL-terminated at data[len]) */
RCRE_NODISCARD RCRE_Slice rcre_str_dup(RCRE_Arena* arena, RCRE_Slice s);
RCRE_NODISCARD RCRE_Slice rcre_str_concat(RCRE_Arena* arena, RCRE_Slice a, RCRE_Slice b);
RCRE_NODISCARD RCRE_Slice rcre_str_join(RCRE_Arena* arena, const RCRE_Slice* parts, size_t count, RCRE_Slice sep);
RCRE_NODISCARD RCRE_Slice rcre_str_replace_all(RCRE_Arena* arena, RCRE_Slice s, RCRE_Slice old_sub, RCRE_Slice new_sub);
RCRE_NODISCARD RCRE_Slice rcre_str_to_lower_ascii(RCRE_Arena* arena, RCRE_Slice s);
RCRE_NODISCARD RCRE_Slice rcre_str_to_upper_ascii(RCRE_Arena* arena, RCRE_Slice s);
RCRE_NODISCARD RCRE_Slice rcre_str_fmt(RCRE_Arena* arena, const char* fmt, ...);

/* ========================================================================= */
/* 5. Strict RFC 3629 UTF-8 Layer (STR-REQ-04)                               */
/* ========================================================================= */

#define RCRE_UTF8_REPLACEMENT_CP 0xFFFDu

/**
 * @brief Validates strict RFC 3629 UTF-8 (rejects overlongs, UTF-16 surrogates,
 *        codepoints > U+10FFFF, and truncated sequences).
 *        Uses rcre_hal.utf8_validate SIMD fast path. If invalid and
 *        out_err_offset != nullptr, writes the exact failing byte index.
 */
RCRE_NODISCARD bool rcre_utf8_validate(RCRE_Slice s, size_t* out_err_offset);
RCRE_NODISCARD bool rcre_utf8_is_ascii(RCRE_Slice s);

/** Returns codepoint count, or RCRE_NPOS if s contains invalid UTF-8. */
RCRE_NODISCARD size_t rcre_utf8_count(RCRE_Slice s);

/**
 * @brief Decodes next UTF-8 codepoint from *cursor and advances *cursor.
 * @return 1..4 (bytes consumed) on valid codepoint, 0 on empty cursor,
 *         -1 on invalid UTF-8 (leaves cursor untouched; *out_cp = 0xFFFD).
 */
RCRE_NODISCARD int rcre_utf8_decode_next(RCRE_Slice* cursor, uint32_t* out_cp);

/**
 * @brief Decodes last UTF-8 codepoint from *cursor and shrinks cursor->len.
 * @return 1..4 (bytes consumed) on valid codepoint, 0 on empty cursor,
 *         -1 on invalid UTF-8 (leaves cursor untouched; *out_cp = 0xFFFD).
 */
RCRE_NODISCARD int rcre_utf8_decode_prev(RCRE_Slice* cursor, uint32_t* out_cp);

/** Encodes Unicode scalar value cp into out[4]. Returns 1..4, or 0 if invalid. */
RCRE_NODISCARD size_t rcre_utf8_encode(uint32_t cp, uint8_t out[4]);

/** Returns longest valid UTF-8 prefix of s whose byte length <= max_bytes. */
RCRE_NODISCARD RCRE_Slice rcre_utf8_truncate_bytes(RCRE_Slice s, size_t max_bytes);

/** Extracts [start_cp, start_cp + count_cp) codepoints as a zero-copy slice. */
RCRE_NODISCARD RCRE_Slice rcre_utf8_sub_codepoints(RCRE_Slice s, size_t start_cp, size_t count_cp);

/* ========================================================================= */
/* 6. Arena String Interning Pool (STR-REQ-06)                               */
/* ========================================================================= */

typedef uint32_t RCRE_StrId;
#define RCRE_STR_ID_INVALID ((RCRE_StrId)0xFFFFFFFFu)

typedef struct RCRE_StrPoolEntry {
    uint64_t   hash;
    RCRE_Slice slice;
    RCRE_StrId id;
} RCRE_StrPoolEntry;

typedef struct RCRE_StrPool {
    RCRE_Arena*        arena;
    RCRE_StrPoolEntry* slots;    /* Power-of-two open-addressed hash table */
    size_t             slot_cap;
    RCRE_Slice*        by_id;    /* Dense id -> interned slice table       */
    size_t             id_cap;
    size_t             count;
    uint64_t           seed;
} RCRE_StrPool;

RCRE_NODISCARD int rcre_strpool_init(RCRE_StrPool* pool, RCRE_Arena* arena, size_t initial_cap);
RCRE_NODISCARD RCRE_StrId rcre_strpool_intern(RCRE_StrPool* pool, RCRE_Slice s);
RCRE_NODISCARD RCRE_StrId rcre_strpool_find(const RCRE_StrPool* pool, RCRE_Slice s);
RCRE_NODISCARD RCRE_Slice rcre_strpool_get(const RCRE_StrPool* pool, RCRE_StrId id);
RCRE_NODISCARD const char* rcre_strpool_cstr(const RCRE_StrPool* pool, RCRE_StrId id);
RCRE_NODISCARD size_t rcre_strpool_count(const RCRE_StrPool* pool);

/* ========================================================================= */
/* 7. Safe Path Manipulation (STR-REQ-06)                                    */
/* ========================================================================= */

RCRE_NODISCARD RCRE_Slice rcre_path_basename(RCRE_Slice path);
RCRE_NODISCARD RCRE_Slice rcre_path_dirname(RCRE_Slice path);
RCRE_NODISCARD RCRE_Slice rcre_path_ext(RCRE_Slice path);
RCRE_NODISCARD RCRE_Slice rcre_path_stem(RCRE_Slice path);
RCRE_NODISCARD RCRE_Slice rcre_path_join(RCRE_Arena* arena, RCRE_Slice a, RCRE_Slice b);

#ifdef __cplusplus
}
#endif

/* ========================================================================= */
/* Optional Compile-Time Guardrail: Poison Legacy Unsafe <string.h> APIs     */
/* Define RCRE_BAN_UNSAFE_CSTR before including rcre_str.h to enforce at     */
/* compile time that strcpy/strcat/sprintf/gets/strncpy/strtok cannot build. */
/* ========================================================================= */
#ifdef RCRE_BAN_UNSAFE_CSTR
  #include <string.h>
  #ifndef RCRE_FREESTANDING
    #include <stdio.h>
  #endif
  #undef strcpy
  #undef strncpy
  #undef strcat
  #undef strncat
  #undef sprintf
  #undef vsprintf
  #undef gets
  #undef strtok
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC poison strcpy strncpy strcat strncat sprintf vsprintf gets strtok
  #endif
#endif

#endif /* RCRE_STR_H */
