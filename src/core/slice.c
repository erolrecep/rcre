/**
 * @file slice.c
 * @brief Zero-Copy Fat-Pointer Byte Slices & Arena-Backed String Builder
 *
 * Satisfies SLC-REQ-01, SLC-REQ-02, and SLC-REQ-03.
 * Pure ISO C23, zero heap allocations, UBSan-clean on empty/NULL views,
 * and accelerated by rcre_hal SIMD byte-stream scanners.
 */

#include "rcre.h"
#include <string.h>
#include <limits.h>

static inline bool slice_is_ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

RCRE_Slice rcre_slice_empty(void) {
    RCRE_Slice s = { .data = nullptr, .len = 0 };
    return s;
}

RCRE_Slice rcre_slice_from_parts(const void* data, size_t len) {
    if (!data || len == 0) {
        return rcre_slice_empty();
    }
    RCRE_Slice s = { .data = (const uint8_t*)data, .len = len };
    return s;
}

RCRE_Slice rcre_slice_from_cstr(const char* cstr) {
    if (!cstr) {
        return rcre_slice_empty();
    }
    size_t n = strlen(cstr);
    if (n == 0) {
        return rcre_slice_empty();
    }
    RCRE_Slice s = { .data = (const uint8_t*)cstr, .len = n };
    return s;
}

RCRE_Slice rcre_slice_sub(RCRE_Slice s, size_t start, size_t end) {
    if (!s.data || s.len == 0 || start >= s.len || start >= end) {
        return rcre_slice_empty();
    }
    if (end > s.len) {
        end = s.len;
    }
    RCRE_Slice out = { .data = s.data + start, .len = end - start };
    return out;
}

bool rcre_slice_eq(RCRE_Slice a, RCRE_Slice b) {
    if (a.len != b.len) {
        return false;
    }
    if (a.len == 0) {
        return true;
    }
    if (a.data == b.data) {
        return true;
    }
    return memcmp(a.data, b.data, a.len) == 0;
}

bool rcre_slice_eq_cstr(RCRE_Slice a, const char* cstr) {
    return rcre_slice_eq(a, rcre_slice_from_cstr(cstr));
}

bool rcre_slice_starts_with(RCRE_Slice s, RCRE_Slice prefix) {
    if (prefix.len == 0) {
        return true;
    }
    if (prefix.len > s.len) {
        return false;
    }
    return memcmp(s.data, prefix.data, prefix.len) == 0;
}

bool rcre_slice_ends_with(RCRE_Slice s, RCRE_Slice suffix) {
    if (suffix.len == 0) {
        return true;
    }
    if (suffix.len > s.len) {
        return false;
    }
    return memcmp(s.data + (s.len - suffix.len), suffix.data, suffix.len) == 0;
}

RCRE_Slice rcre_slice_trim_ascii(RCRE_Slice s) {
    if (!s.data || s.len == 0) {
        return rcre_slice_empty();
    }
    size_t start = 0;
    size_t end = s.len;
    while (start < end && slice_is_ascii_space(s.data[start])) {
        start++;
    }
    while (end > start && slice_is_ascii_space(s.data[end - 1])) {
        end--;
    }
    return rcre_slice_sub(s, start, end);
}

size_t rcre_slice_find_byte(RCRE_Slice s, uint8_t needle) {
    if (!s.data || s.len == 0) {
        return RCRE_NPOS;
    }
    return rcre_hal.find_byte(s.data, s.len, needle);
}

size_t rcre_slice_rfind_byte(RCRE_Slice s, uint8_t needle) {
    if (!s.data || s.len == 0) {
        return RCRE_NPOS;
    }
    size_t i = s.len;
    while (i > 0) {
        i--;
        if (s.data[i] == needle) {
            return i;
        }
    }
    return RCRE_NPOS;
}

size_t rcre_slice_count_byte(RCRE_Slice s, uint8_t needle) {
    if (!s.data || s.len == 0) {
        return 0;
    }
    return rcre_hal.count_byte(s.data, s.len, needle);
}

RCRE_SlicePair rcre_slice_split_once_byte(RCRE_Slice s, uint8_t delim) {
    size_t pos = rcre_slice_find_byte(s, delim);
    if (pos == RCRE_NPOS) {
        RCRE_SlicePair p = { .head = s, .tail = rcre_slice_empty(), .found = false };
        return p;
    }
    RCRE_SlicePair p = {
        .head  = rcre_slice_sub(s, 0, pos),
        .tail  = rcre_slice_sub(s, pos + 1, s.len),
        .found = true
    };
    return p;
}

RCRE_SlicePair rcre_slice_split_once(RCRE_Slice s, RCRE_Slice delim) {
    if (delim.len == 0) {
        RCRE_SlicePair p = { .head = rcre_slice_empty(), .tail = s, .found = true };
        return p;
    }
    if (delim.len == 1) {
        return rcre_slice_split_once_byte(s, delim.data[0]);
    }
    if (!s.data || s.len < delim.len) {
        RCRE_SlicePair p = { .head = s, .tail = rcre_slice_empty(), .found = false };
        return p;
    }

    uint8_t first = delim.data[0];
    size_t search_limit = s.len - delim.len + 1;
    size_t offset = 0;

    while (offset < search_limit) {
        size_t rel = rcre_hal.find_byte(s.data + offset, search_limit - offset, first);
        if (rel == RCRE_NPOS) {
            break;
        }
        size_t candidate = offset + rel;
        if (memcmp(s.data + candidate + 1, delim.data + 1, delim.len - 1) == 0) {
            RCRE_SlicePair p = {
                .head  = rcre_slice_sub(s, 0, candidate),
                .tail  = rcre_slice_sub(s, candidate + delim.len, s.len),
                .found = true
            };
            return p;
        }
        offset = candidate + 1;
    }

    RCRE_SlicePair p = { .head = s, .tail = rcre_slice_empty(), .found = false };
    return p;
}

bool rcre_slice_parse_u64(RCRE_Slice s, uint64_t* out_val) {
    if (!out_val || !s.data || s.len == 0) {
        return false;
    }
    size_t idx = 0;
    if (s.data[0] == '+') {
        idx++;
        if (idx == s.len) return false;
    }
    uint64_t acc = 0;
    for (; idx < s.len; ++idx) {
        uint8_t c = s.data[idx];
        if (c < '0' || c > '9') {
            return false;
        }
        uint64_t digit = (uint64_t)(c - '0');
        if (acc > (UINT64_MAX - digit) / 10ULL) {
            return false; /* Overflow */
        }
        acc = acc * 10ULL + digit;
    }
    *out_val = acc;
    return true;
}

bool rcre_slice_parse_i64(RCRE_Slice s, int64_t* out_val) {
    if (!out_val || !s.data || s.len == 0) {
        return false;
    }
    bool neg = false;
    size_t idx = 0;
    if (s.data[0] == '-') {
        neg = true;
        idx++;
    } else if (s.data[0] == '+') {
        idx++;
    }
    if (idx >= s.len) {
        return false;
    }
    uint64_t mag = 0;
    if (!rcre_slice_parse_u64(rcre_slice_sub(s, idx, s.len), &mag)) {
        return false;
    }
    if (neg) {
        const uint64_t max_neg = (uint64_t)INT64_MAX + 1ULL;
        if (mag > max_neg) return false;
        if (mag == max_neg) {
            *out_val = INT64_MIN;
        } else {
            *out_val = -(int64_t)mag;
        }
    } else {
        if (mag > (uint64_t)INT64_MAX) return false;
        *out_val = (int64_t)mag;
    }
    return true;
}

char* rcre_arena_cstr_dup(RCRE_Arena* arena, RCRE_Slice s) {
    if (!arena) {
        return nullptr;
    }
    char* out = (char*)rcre_arena_alloc(arena, s.len + 1, 8);
    if (!out) {
        return nullptr;
    }
    if (s.len > 0 && s.data) {
        memcpy(out, s.data, s.len);
    }
    out[s.len] = '\0';
    return out;
}

/* ========================================================================= */
/* Arena-Backed String Builder                                               */
/* ========================================================================= */

static bool slice_sb_reserve(RCRE_StrBuilder* sb, size_t extra) {
    if (!sb || sb->overflow) {
        return false;
    }
    if (extra > SIZE_MAX - sb->len - 1) {
        sb->overflow = true;
        return false;
    }
    size_t needed = sb->len + extra + 1; /* Always keep room for trailing NUL */
    if (needed <= sb->cap) {
        return true;
    }
    if (!sb->arena) {
        sb->overflow = true;
        return false;
    }

    size_t new_cap = (sb->cap == 0) ? 64 : sb->cap;
    while (new_cap < needed) {
        if (new_cap > SIZE_MAX / 2) {
            new_cap = needed;
            break;
        }
        new_cap *= 2;
    }

    /* Fast path: extend in-place if this builder sits at the tip of the arena */
    if (sb->buf && (sb->buf + sb->cap == sb->arena->buffer + sb->arena->offset)) {
        size_t delta = new_cap - sb->cap;
        if (sb->arena->offset + delta <= sb->arena->capacity) {
            sb->arena->offset += delta;
            if (sb->arena->offset > sb->arena->high_watermark) {
                sb->arena->high_watermark = sb->arena->offset;
            }
            sb->cap = new_cap;
            return true;
        }
    }

    uint8_t* next_buf = (uint8_t*)rcre_arena_alloc(sb->arena, new_cap, 8);
    if (!next_buf) {
        sb->overflow = true;
        return false;
    }
    if (sb->len > 0 && sb->buf) {
        memcpy(next_buf, sb->buf, sb->len);
    }
    next_buf[sb->len] = '\0';
    sb->buf = next_buf;
    sb->cap = new_cap;
    return true;
}

RCRE_StrBuilder rcre_sb_init(RCRE_Arena* arena, size_t initial_cap) {
    RCRE_StrBuilder sb = {
        .buf      = nullptr,
        .len      = 0,
        .cap      = 0,
        .arena    = arena,
        .overflow = (arena == nullptr)
    };
    if (arena && initial_cap > 0) {
        size_t alloc_cap = initial_cap + 1;
        uint8_t* mem = (uint8_t*)rcre_arena_alloc(arena, alloc_cap, 8);
        if (mem) {
            mem[0] = '\0';
            sb.buf = mem;
            sb.cap = alloc_cap;
        } else {
            sb.overflow = true;
        }
    }
    return sb;
}

bool rcre_sb_append_slice(RCRE_StrBuilder* sb, RCRE_Slice s) {
    if (!sb || sb->overflow) {
        return false;
    }
    if (s.len == 0) {
        return true;
    }
    if (!s.data) {
        sb->overflow = true;
        return false;
    }
    if (!slice_sb_reserve(sb, s.len)) {
        return false;
    }
    memcpy(sb->buf + sb->len, s.data, s.len);
    sb->len += s.len;
    sb->buf[sb->len] = '\0';
    return true;
}

bool rcre_sb_append_cstr(RCRE_StrBuilder* sb, const char* cstr) {
    if (!cstr) {
        return sb && !sb->overflow;
    }
    return rcre_sb_append_slice(sb, rcre_slice_from_cstr(cstr));
}

bool rcre_sb_append_byte(RCRE_StrBuilder* sb, uint8_t b) {
    if (!slice_sb_reserve(sb, 1)) {
        return false;
    }
    sb->buf[sb->len++] = b;
    sb->buf[sb->len] = '\0';
    return true;
}

bool rcre_sb_append_u64(RCRE_StrBuilder* sb, uint64_t val) {
    uint8_t tmp[24];
    size_t n = 0;
    do {
        tmp[n++] = (uint8_t)('0' + (val % 10ULL));
        val /= 10ULL;
    } while (val > 0);

    if (!slice_sb_reserve(sb, n)) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        sb->buf[sb->len++] = tmp[n - 1 - i];
    }
    sb->buf[sb->len] = '\0';
    return true;
}

bool rcre_sb_append_i64(RCRE_StrBuilder* sb, int64_t val) {
    if (val < 0) {
        if (!rcre_sb_append_byte(sb, '-')) {
            return false;
        }
        uint64_t mag = (val == INT64_MIN) ? ((uint64_t)INT64_MAX + 1ULL) : (uint64_t)(-val);
        return rcre_sb_append_u64(sb, mag);
    }
    return rcre_sb_append_u64(sb, (uint64_t)val);
}

RCRE_Slice rcre_sb_to_slice(const RCRE_StrBuilder* sb) {
    if (!sb || sb->overflow || !sb->buf || sb->len == 0) {
        return rcre_slice_empty();
    }
    RCRE_Slice s = { .data = sb->buf, .len = sb->len };
    return s;
}

char* rcre_sb_to_cstr(RCRE_StrBuilder* sb) {
    if (!sb || sb->overflow) {
        return nullptr;
    }
    if (!sb->buf) {
        if (!slice_sb_reserve(sb, 0)) {
            return nullptr;
        }
    }
    sb->buf[sb->len] = '\0';
    return (char*)sb->buf;
}
