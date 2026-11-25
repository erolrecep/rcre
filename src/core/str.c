/**
 * @file str.c
 * @brief RCRE Safe, Zero-Malloc, SIMD-Accelerated C23 String Library
 *
 * Satisfies STR-REQ-01 through STR-REQ-06.
 * Pure ISO C23, zero heap allocations, freestanding-clean (-DRCRE_FREESTANDING),
 * UBSan/ASan-clean on empty/NULL/unaligned slices, and HAL-SIMD accelerated.
 */

#include "rcre_str.h"
#include <string.h>

/* ========================================================================= */
/* Internal Helpers (namespaced with rcre_str__ for amalgamation uniqueness) */
/* ========================================================================= */

static inline uint64_t rcre_str__f64_to_bits(double v) {
    uint64_t bits = 0;
    memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static inline double rcre_str__bits_to_f64(uint64_t bits) {
    double v = 0.0;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

static inline bool rcre_str__f64_isnan(double v) {
    uint64_t b = rcre_str__f64_to_bits(v);
    return ((b >> 52) & 0x7FFULL) == 0x7FFULL && (b & 0x000FFFFFFFFFFFFFULL) != 0ULL;
}

static inline bool rcre_str__f64_isinf(double v) {
    uint64_t b = rcre_str__f64_to_bits(v);
    return ((b >> 52) & 0x7FFULL) == 0x7FFULL && (b & 0x000FFFFFFFFFFFFFULL) == 0ULL;
}

static inline bool rcre_str__f64_signbit(double v) {
    return (rcre_str__f64_to_bits(v) >> 63) != 0ULL;
}

static double rcre_str__pow10_u32(unsigned int exp) {
    double res = 1.0;
    double base = 10.0;
    while (exp > 0U) {
        if (exp & 1U) res *= base;
        base *= base;
        exp >>= 1;
    }
    return res;
}

static inline bool rcre_str__is_ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static inline uint8_t rcre_str__to_lower_byte(uint8_t c) {
    return (c >= 'A' && c <= 'Z') ? (uint8_t)(c + 32U) : c;
}

static inline bool rcre_str__is_path_sep(uint8_t c) {
    return c == '/' || c == '\\';
}

static inline size_t rcre_str__next_pow2(size_t v) {
    if (v <= 16U) return 16U;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if SIZE_MAX > 0xFFFFFFFFULL
    v |= v >> 32;
#endif
    return (v == SIZE_MAX) ? 0U : (v + 1U);
}

/* SplitMix64-inspired finalizer over fast 64-bit mixing (deterministic, zero-dep) */
static inline uint64_t rcre_str__mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline uint64_t rcre_str__load_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    memcpy(&v, p, sizeof(v));
    return v;
}

/* Sink abstraction for freestanding formatting into either RCRE_StrBuf or RCRE_StrBuilder */
typedef struct {
    RCRE_StrBuf*     strbuf;
    RCRE_StrBuilder* builder;
} rcre_str__sink;

static bool rcre_str__sink_write(rcre_str__sink* sink, const uint8_t* bytes, size_t n) {
    if (n == 0) {
        return true;
    }
    if (sink->strbuf) {
        return rcre_strbuf_append(sink->strbuf, rcre_slice_from_parts(bytes, n));
    }
    return rcre_sb_append_slice(sink->builder, rcre_slice_from_parts(bytes, n));
}

static bool rcre_str__sink_byte(rcre_str__sink* sink, uint8_t b) {
    return rcre_str__sink_write(sink, &b, 1);
}

static bool rcre_str__sink_pad(rcre_str__sink* sink, uint8_t ch, size_t count) {
    uint8_t chunk[32];
    memset(chunk, ch, sizeof(chunk));
    while (count > 0) {
        size_t step = (count < sizeof(chunk)) ? count : sizeof(chunk);
        if (!rcre_str__sink_write(sink, chunk, step)) {
            return false;
        }
        count -= step;
    }
    return true;
}

static bool rcre_str__sink_write_aligned(
    rcre_str__sink* sink,
    const uint8_t* bytes,
    size_t len,
    size_t width,
    bool left_align,
    uint8_t pad_ch
) {
    size_t pad = (width > len) ? (width - len) : 0U;
    if (!left_align && pad > 0) {
        if (!rcre_str__sink_pad(sink, pad_ch, pad)) return false;
    }
    if (!rcre_str__sink_write(sink, bytes, len)) return false;
    if (left_align && pad > 0) {
        if (!rcre_str__sink_pad(sink, ' ', pad)) return false;
    }
    return true;
}

static size_t rcre_str__u64_to_dec(uint64_t val, uint8_t out[24]) {
    uint8_t rev[24];
    size_t n = 0;
    do {
        rev[n++] = (uint8_t)('0' + (val % 10ULL));
        val /= 10ULL;
    } while (val > 0);
    for (size_t i = 0; i < n; ++i) {
        out[i] = rev[n - 1U - i];
    }
    return n;
}

static size_t rcre_str__u64_to_hex(uint64_t val, bool uppercase, uint8_t out[24]) {
    const char* digits = uppercase ? "0123456789ABCDEF" : "0123456789abcdef";
    uint8_t rev[24];
    size_t n = 0;
    do {
        rev[n++] = (uint8_t)digits[val & 0xFULL];
        val >>= 4;
    } while (val > 0);
    for (size_t i = 0; i < n; ++i) {
        out[i] = rev[n - 1U - i];
    }
    return n;
}

static bool rcre_str__sink_f64(rcre_str__sink* sink, double val, int precision) {
    if (precision < 0) precision = 6;
    if (precision > 15) precision = 15;

    if (rcre_str__f64_isnan(val)) {
        return rcre_str__sink_write(sink, (const uint8_t*)"nan", 3);
    }
    if (rcre_str__f64_signbit(val)) {
        if (!rcre_str__sink_byte(sink, '-')) return false;
        val = -val;
    }
    if (rcre_str__f64_isinf(val)) {
        return rcre_str__sink_write(sink, (const uint8_t*)"inf", 3);
    }

    /* Fall back to scientific notation for huge magnitudes that overflow uint64 */
    if (val >= 1e18) {
        int exp10 = 0;
        double m = val;
        while (m >= 10.0) { m /= 10.0; exp10++; }
        if (!rcre_str__sink_f64(sink, m, precision)) return false;
        if (!rcre_str__sink_byte(sink, 'e')) return false;
        if (!rcre_str__sink_byte(sink, '+')) return false;
        uint8_t ebuf[24];
        size_t elen = rcre_str__u64_to_dec((uint64_t)exp10, ebuf);
        if (elen < 2) {
            if (!rcre_str__sink_byte(sink, '0')) return false;
        }
        return rcre_str__sink_write(sink, ebuf, elen);
    }

    static const double pow10_tbl[16] = {
        1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7,
        1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15
    };
    double scale = pow10_tbl[precision];

    uint64_t whole = (uint64_t)val;
    double frac_d = (val - (double)whole) * scale + 0.5;
    uint64_t frac = (uint64_t)frac_d;
    uint64_t scale_u = (uint64_t)scale;
    if (precision == 0) {
        if (frac >= 1ULL) {
            whole += 1ULL;
        }
    } else if (frac >= scale_u) {
        whole += 1ULL;
        frac -= scale_u;
    }

    uint8_t wbuf[24];
    size_t wlen = rcre_str__u64_to_dec(whole, wbuf);
    if (!rcre_str__sink_write(sink, wbuf, wlen)) return false;

    if (precision > 0) {
        if (!rcre_str__sink_byte(sink, '.')) return false;
        uint8_t fbuf[24];
        size_t flen = rcre_str__u64_to_dec(frac, fbuf);
        if ((size_t)precision > flen) {
            if (!rcre_str__sink_pad(sink, '0', (size_t)precision - flen)) return false;
        }
        if (!rcre_str__sink_write(sink, fbuf, flen)) return false;
    }
    return true;
}

static bool rcre_str__sink_vfmt(rcre_str__sink* sink, const char* fmt, va_list ap) {
    if (!fmt) {
        return true;
    }
    bool ok = true;
    const uint8_t* p = (const uint8_t*)fmt;

    while (*p != '\0') {
        const uint8_t* lit_start = p;
        while (*p != '\0' && *p != '%') {
            p++;
        }
        if (p > lit_start) {
            if (!rcre_str__sink_write(sink, lit_start, (size_t)(p - lit_start))) {
                ok = false;
            }
        }
        if (*p == '\0') break;
        p++; /* skip '%' */

        if (*p == '%') {
            if (!rcre_str__sink_byte(sink, '%')) ok = false;
            p++;
            continue;
        }

        /* Flags */
        bool left_align = false;
        bool zero_pad = false;
        while (*p == '-' || *p == '0') {
            if (*p == '-') left_align = true;
            if (*p == '0') zero_pad = true;
            p++;
        }
        if (left_align) zero_pad = false;

        /* Width */
        size_t width = 0;
        while (*p >= '0' && *p <= '9') {
            if (width < 4096U) {
                width = width * 10U + (size_t)(*p - '0');
            }
            p++;
        }

        /* Precision */
        int precision = -1;
        if (*p == '.') {
            p++;
            precision = 0;
            while (*p >= '0' && *p <= '9') {
                if (precision < 1000) {
                    precision = precision * 10 + (int)(*p - '0');
                }
                p++;
            }
        }

        /* Length modifier: l, ll, z */
        int len_mod = 0; /* 0=default, 1=l, 2=ll, 3=z */
        if (*p == 'l') {
            p++;
            if (*p == 'l') { len_mod = 2; p++; }
            else           { len_mod = 1; }
        } else if (*p == 'z') {
            len_mod = 3;
            p++;
        }

        uint8_t spec = *p;
        if (spec == '\0') break;
        p++;

        uint8_t pad_ch = zero_pad ? (uint8_t)'0' : (uint8_t)' ';

        switch (spec) {
            case 'c': {
                uint8_t ch = (uint8_t)va_arg(ap, int);
                if (!rcre_str__sink_write_aligned(sink, &ch, 1, width, left_align, ' ')) ok = false;
                break;
            }
            case 's': {
                const char* cs = va_arg(ap, const char*);
                RCRE_Slice s = cs ? rcre_slice_from_cstr(cs) : RCRE_STR("(null)");
                if (precision >= 0 && (size_t)precision < s.len) {
                    s.len = (size_t)precision;
                }
                if (!rcre_str__sink_write_aligned(sink, s.data, s.len, width, left_align, ' ')) ok = false;
                break;
            }
            case 'S': {
                RCRE_Slice s = va_arg(ap, RCRE_Slice);
                if (precision >= 0 && (size_t)precision < s.len) {
                    s.len = (size_t)precision;
                }
                if (!rcre_str__sink_write_aligned(sink, s.data, s.len, width, left_align, ' ')) ok = false;
                break;
            }
            case 'd':
            case 'i': {
                int64_t val = 0;
                if (len_mod == 2)      val = (int64_t)va_arg(ap, long long);
                else if (len_mod == 1) val = (int64_t)va_arg(ap, long);
                else if (len_mod == 3) val = (int64_t)va_arg(ap, ptrdiff_t);
                else                   val = (int64_t)va_arg(ap, int);

                uint8_t num[32];
                size_t n = 0;
                uint64_t mag = 0;
                bool neg = (val < 0);
                if (neg) {
                    mag = (val == INT64_MIN) ? ((uint64_t)INT64_MAX + 1ULL) : (uint64_t)(-val);
                } else {
                    mag = (uint64_t)val;
                }
                if (neg && !zero_pad) {
                    num[n++] = '-';
                }
                n += rcre_str__u64_to_dec(mag, num + n);
                if (neg && zero_pad) {
                    if (!rcre_str__sink_byte(sink, '-')) ok = false;
                    size_t adj_w = (width > 0) ? (width - 1U) : 0U;
                    if (!rcre_str__sink_write_aligned(sink, num, n, adj_w, false, '0')) ok = false;
                } else {
                    if (!rcre_str__sink_write_aligned(sink, num, n, width, left_align, pad_ch)) ok = false;
                }
                break;
            }
            case 'u': {
                uint64_t val = 0;
                if (len_mod == 2)      val = (uint64_t)va_arg(ap, unsigned long long);
                else if (len_mod == 1) val = (uint64_t)va_arg(ap, unsigned long);
                else if (len_mod == 3) val = (uint64_t)va_arg(ap, size_t);
                else                   val = (uint64_t)va_arg(ap, unsigned int);

                uint8_t num[24];
                size_t n = rcre_str__u64_to_dec(val, num);
                if (!rcre_str__sink_write_aligned(sink, num, n, width, left_align, pad_ch)) ok = false;
                break;
            }
            case 'x':
            case 'X': {
                uint64_t val = 0;
                if (len_mod == 2)      val = (uint64_t)va_arg(ap, unsigned long long);
                else if (len_mod == 1) val = (uint64_t)va_arg(ap, unsigned long);
                else if (len_mod == 3) val = (uint64_t)va_arg(ap, size_t);
                else                   val = (uint64_t)va_arg(ap, unsigned int);

                uint8_t num[24];
                size_t n = rcre_str__u64_to_hex(val, spec == 'X', num);
                if (!rcre_str__sink_write_aligned(sink, num, n, width, left_align, pad_ch)) ok = false;
                break;
            }
            case 'p': {
                uintptr_t ptr = (uintptr_t)va_arg(ap, const void*);
                uint8_t num[26];
                num[0] = '0';
                num[1] = 'x';
                size_t n = 2U + rcre_str__u64_to_hex((uint64_t)ptr, false, num + 2);
                if (!rcre_str__sink_write_aligned(sink, num, n, width, left_align, ' ')) ok = false;
                break;
            }
            case 'f':
            case 'F': {
                double val = va_arg(ap, double);
                uint8_t tmp[96];
                RCRE_StrBuf local = rcre_strbuf_init(tmp, sizeof(tmp));
                rcre_str__sink local_sink = { .strbuf = &local, .builder = nullptr };
                (void)rcre_str__sink_f64(&local_sink, val, precision);
                if (!rcre_str__sink_write_aligned(sink, local.data, local.len, width, left_align, pad_ch)) {
                    ok = false;
                }
                break;
            }
            default: {
                uint8_t raw[2] = { '%', spec };
                if (!rcre_str__sink_write(sink, raw, 2)) ok = false;
                break;
            }
        }
    }
    return ok;
}

/* ========================================================================= */
/* 1. RCRE_StrBuf — Fixed-Capacity Bounded Buffer (STR-REQ-02)               */
/* ========================================================================= */

RCRE_StrBuf rcre_strbuf_init(void* backing, size_t cap) {
    RCRE_StrBuf sb = {
        .data      = (backing && cap > 0) ? (uint8_t*)backing : nullptr,
        .len       = 0,
        .cap       = (backing && cap > 0) ? cap : 0,
        .truncated = false
    };
    if (sb.data) {
        sb.data[0] = '\0';
    }
    return sb;
}

void rcre_strbuf_reset(RCRE_StrBuf* sb) {
    if (!sb) return;
    sb->len = 0;
    sb->truncated = false;
    if (sb->data && sb->cap > 0) {
        sb->data[0] = '\0';
    }
}

bool rcre_strbuf_copy(RCRE_StrBuf* sb, RCRE_Slice src) {
    if (!sb) return false;
    rcre_strbuf_reset(sb);
    return rcre_strbuf_append(sb, src);
}

bool rcre_strbuf_copy_cstr(RCRE_StrBuf* sb, const char* cstr) {
    return rcre_strbuf_copy(sb, rcre_slice_from_cstr(cstr));
}

bool rcre_strbuf_copy_utf8(RCRE_StrBuf* sb, RCRE_Slice src) {
    if (!sb) return false;
    rcre_strbuf_reset(sb);
    return rcre_strbuf_append_utf8(sb, src);
}

bool rcre_strbuf_append(RCRE_StrBuf* sb, RCRE_Slice src) {
    if (!sb) return false;
    if (src.len == 0) {
        return !sb->truncated;
    }
    if (!src.data) {
        sb->truncated = true;
        return false;
    }
    if (!sb->data || sb->cap == 0) {
        sb->truncated = true;
        return false;
    }
    size_t max_payload = sb->cap - 1U;
    size_t avail = (sb->len < max_payload) ? (max_payload - sb->len) : 0U;
    size_t to_copy = (src.len <= avail) ? src.len : avail;
    if (to_copy > 0) {
        memmove(sb->data + sb->len, src.data, to_copy);
        sb->len += to_copy;
    }
    sb->data[sb->len] = '\0';
    if (to_copy < src.len) {
        sb->truncated = true;
        return false;
    }
    return !sb->truncated;
}

bool rcre_strbuf_append_cstr(RCRE_StrBuf* sb, const char* cstr) {
    return rcre_strbuf_append(sb, rcre_slice_from_cstr(cstr));
}

bool rcre_strbuf_append_utf8(RCRE_StrBuf* sb, RCRE_Slice src) {
    if (!sb) return false;
    if (src.len == 0) return !sb->truncated;
    if (!src.data || !sb->data || sb->cap == 0) {
        sb->truncated = true;
        return false;
    }
    size_t max_payload = sb->cap - 1U;
    size_t avail = (sb->len < max_payload) ? (max_payload - sb->len) : 0U;
    RCRE_Slice fit = rcre_utf8_truncate_bytes(src, avail);
    if (fit.len > 0) {
        memmove(sb->data + sb->len, fit.data, fit.len);
        sb->len += fit.len;
    }
    sb->data[sb->len] = '\0';
    if (fit.len < src.len) {
        sb->truncated = true;
        return false;
    }
    return !sb->truncated;
}

bool rcre_strbuf_append_byte(RCRE_StrBuf* sb, uint8_t b) {
    return rcre_strbuf_append(sb, rcre_slice_from_parts(&b, 1));
}

bool rcre_strbuf_append_cp(RCRE_StrBuf* sb, uint32_t cp) {
    uint8_t enc[4];
    size_t n = rcre_utf8_encode(cp, enc);
    if (n == 0) {
        return false;
    }
    return rcre_strbuf_append(sb, rcre_slice_from_parts(enc, n));
}

bool rcre_strbuf_append_u64(RCRE_StrBuf* sb, uint64_t val) {
    uint8_t buf[24];
    size_t n = rcre_str__u64_to_dec(val, buf);
    return rcre_strbuf_append(sb, rcre_slice_from_parts(buf, n));
}

bool rcre_strbuf_append_i64(RCRE_StrBuf* sb, int64_t val) {
    if (val < 0) {
        if (!rcre_strbuf_append_byte(sb, '-')) return false;
        uint64_t mag = (val == INT64_MIN) ? ((uint64_t)INT64_MAX + 1ULL) : (uint64_t)(-val);
        return rcre_strbuf_append_u64(sb, mag);
    }
    return rcre_strbuf_append_u64(sb, (uint64_t)val);
}

bool rcre_strbuf_append_hex(RCRE_StrBuf* sb, uint64_t val, bool uppercase) {
    uint8_t buf[24];
    size_t n = rcre_str__u64_to_hex(val, uppercase, buf);
    return rcre_strbuf_append(sb, rcre_slice_from_parts(buf, n));
}

bool rcre_strbuf_append_f64(RCRE_StrBuf* sb, double val, int precision) {
    if (!sb) return false;
    rcre_str__sink sink = { .strbuf = sb, .builder = nullptr };
    return rcre_str__sink_f64(&sink, val, precision);
}

bool rcre_strbuf_vfmt(RCRE_StrBuf* sb, const char* fmt, va_list ap) {
    if (!sb) return false;
    rcre_str__sink sink = { .strbuf = sb, .builder = nullptr };
    return rcre_str__sink_vfmt(&sink, fmt, ap);
}

bool rcre_strbuf_fmt(RCRE_StrBuf* sb, const char* fmt, ...) {
    if (!sb) return false;
    va_list ap;
    va_start(ap, fmt);
    bool ok = rcre_strbuf_vfmt(sb, fmt, ap);
    va_end(ap);
    return ok;
}

RCRE_Slice rcre_strbuf_as_slice(const RCRE_StrBuf* sb) {
    if (!sb || !sb->data || sb->len == 0) {
        return rcre_slice_empty();
    }
    return rcre_slice_from_parts(sb->data, sb->len);
}

const char* rcre_strbuf_cstr(const RCRE_StrBuf* sb) {
    if (!sb || !sb->data || sb->cap == 0) {
        return "";
    }
    return (const char*)sb->data;
}

bool rcre_cstr_copy_bounded(char* dst, size_t dst_cap, RCRE_Slice src) {
    if (!dst || dst_cap == 0) {
        return false;
    }
    RCRE_StrBuf sb = rcre_strbuf_init(dst, dst_cap);
    return rcre_strbuf_copy_utf8(&sb, src);
}

/* ========================================================================= */
/* 2. Expanded Zero-Copy Slice Operations (STR-REQ-03)                       */
/* ========================================================================= */

int rcre_slice_cmp(RCRE_Slice a, RCRE_Slice b) {
    size_t min_len = (a.len < b.len) ? a.len : b.len;
    if (min_len > 0 && a.data != b.data) {
        int c = memcmp(a.data, b.data, min_len);
        if (c != 0) return (c < 0) ? -1 : 1;
    }
    if (a.len < b.len) return -1;
    if (a.len > b.len) return 1;
    return 0;
}

bool rcre_slice_eq_ascii_ci(RCRE_Slice a, RCRE_Slice b) {
    if (a.len != b.len) return false;
    if (a.len == 0 || a.data == b.data) return true;
    for (size_t i = 0; i < a.len; ++i) {
        if (rcre_str__to_lower_byte(a.data[i]) != rcre_str__to_lower_byte(b.data[i])) {
            return false;
        }
    }
    return true;
}

int rcre_slice_cmp_ascii_ci(RCRE_Slice a, RCRE_Slice b) {
    size_t min_len = (a.len < b.len) ? a.len : b.len;
    for (size_t i = 0; i < min_len; ++i) {
        uint8_t ca = rcre_str__to_lower_byte(a.data[i]);
        uint8_t cb = rcre_str__to_lower_byte(b.data[i]);
        if (ca != cb) return (ca < cb) ? -1 : 1;
    }
    if (a.len < b.len) return -1;
    if (a.len > b.len) return 1;
    return 0;
}

uint64_t rcre_slice_hash64_seeded(RCRE_Slice s, uint64_t seed) {
    uint64_t h = seed ^ (0x9e3779b97f4a7c15ULL * ((uint64_t)s.len + 1ULL));
    if (!s.data || s.len == 0) {
        return rcre_str__mix64(h);
    }
    size_t i = 0;
    for (; i + 8U <= s.len; i += 8U) {
        uint64_t k = rcre_str__load_u64_le(s.data + i);
        h = rcre_str__mix64(h ^ k);
    }
    if (i < s.len) {
        uint64_t tail = 0;
        size_t rem = s.len - i;
        for (size_t b = 0; b < rem; ++b) {
            tail |= ((uint64_t)s.data[i + b]) << (b * 8U);
        }
        h = rcre_str__mix64(h ^ tail);
    }
    return rcre_str__mix64(h);
}

uint64_t rcre_slice_hash64(RCRE_Slice s) {
    return rcre_slice_hash64_seeded(s, 0xcbf29ce484222325ULL);
}

size_t rcre_slice_find(RCRE_Slice hay, RCRE_Slice needle) {
    if (needle.len == 0) {
        return 0;
    }
    if (!hay.data || !needle.data || needle.len > hay.len) {
        return RCRE_NPOS;
    }
    if (needle.len == 1) {
        return rcre_hal.find_byte(hay.data, hay.len, needle.data[0]);
    }
    return rcre_hal.find_substr(hay.data, hay.len, needle.data, needle.len);
}

size_t rcre_slice_rfind(RCRE_Slice hay, RCRE_Slice needle) {
    if (needle.len == 0) {
        return hay.len;
    }
    if (!hay.data || !needle.data || needle.len > hay.len) {
        return RCRE_NPOS;
    }
    if (needle.len == 1) {
        return rcre_slice_rfind_byte(hay, needle.data[0]);
    }
    uint8_t first = needle.data[0];
    uint8_t last = needle.data[needle.len - 1U];
    size_t i = hay.len - needle.len + 1U;
    while (i > 0) {
        i--;
        if (hay.data[i] == first && hay.data[i + needle.len - 1U] == last) {
            if (needle.len == 2 || memcmp(hay.data + i + 1, needle.data + 1, needle.len - 2U) == 0) {
                return i;
            }
        }
    }
    return RCRE_NPOS;
}

bool rcre_slice_contains(RCRE_Slice hay, RCRE_Slice needle) {
    return rcre_slice_find(hay, needle) != RCRE_NPOS;
}

size_t rcre_slice_find_any_byte(RCRE_Slice s, RCRE_Slice charset) {
    if (!s.data || s.len == 0 || !charset.data || charset.len == 0) {
        return RCRE_NPOS;
    }
    uint8_t lut[256];
    memset(lut, 0, sizeof(lut));
    for (size_t j = 0; j < charset.len; ++j) {
        lut[charset.data[j]] = 1U;
    }
    for (size_t i = 0; i < s.len; ++i) {
        if (lut[s.data[i]]) return i;
    }
    return RCRE_NPOS;
}

size_t rcre_slice_span_bytes(RCRE_Slice s, RCRE_Slice charset) {
    if (!s.data || s.len == 0 || !charset.data || charset.len == 0) {
        return 0;
    }
    uint8_t lut[256];
    memset(lut, 0, sizeof(lut));
    for (size_t j = 0; j < charset.len; ++j) {
        lut[charset.data[j]] = 1U;
    }
    size_t i = 0;
    while (i < s.len && lut[s.data[i]]) {
        i++;
    }
    return i;
}

RCRE_Slice rcre_slice_trim_left_ascii(RCRE_Slice s) {
    if (!s.data || s.len == 0) return rcre_slice_empty();
    size_t start = 0;
    while (start < s.len && rcre_str__is_ascii_space(s.data[start])) {
        start++;
    }
    return rcre_slice_sub(s, start, s.len);
}

RCRE_Slice rcre_slice_trim_right_ascii(RCRE_Slice s) {
    if (!s.data || s.len == 0) return rcre_slice_empty();
    size_t end = s.len;
    while (end > 0 && rcre_str__is_ascii_space(s.data[end - 1U])) {
        end--;
    }
    return rcre_slice_sub(s, 0, end);
}

RCRE_Slice rcre_slice_trim_chars(RCRE_Slice s, RCRE_Slice charset) {
    if (!s.data || s.len == 0 || !charset.data || charset.len == 0) {
        return s;
    }
    uint8_t lut[256];
    memset(lut, 0, sizeof(lut));
    for (size_t j = 0; j < charset.len; ++j) {
        lut[charset.data[j]] = 1U;
    }
    size_t start = 0;
    size_t end = s.len;
    while (start < end && lut[s.data[start]]) start++;
    while (end > start && lut[s.data[end - 1U]]) end--;
    return rcre_slice_sub(s, start, end);
}

RCRE_Slice rcre_slice_strip_prefix(RCRE_Slice s, RCRE_Slice prefix, bool* stripped) {
    if (rcre_slice_starts_with(s, prefix)) {
        if (stripped) *stripped = true;
        return rcre_slice_sub(s, prefix.len, s.len);
    }
    if (stripped) *stripped = false;
    return s;
}

RCRE_Slice rcre_slice_strip_suffix(RCRE_Slice s, RCRE_Slice suffix, bool* stripped) {
    if (rcre_slice_ends_with(s, suffix)) {
        if (stripped) *stripped = true;
        return rcre_slice_sub(s, 0, s.len - suffix.len);
    }
    if (stripped) *stripped = false;
    return s;
}

RCRE_SplitIter rcre_split_byte_init(RCRE_Slice s, uint8_t delim) {
    RCRE_SplitIter it = {
        .rest       = s,
        .delim      = rcre_slice_empty(),
        .delim_byte = delim,
        .use_byte   = true,
        .done       = false
    };
    return it;
}

RCRE_SplitIter rcre_split_slice_init(RCRE_Slice s, RCRE_Slice delim) {
    RCRE_SplitIter it = {
        .rest       = s,
        .delim      = delim,
        .delim_byte = (delim.len == 1 && delim.data) ? delim.data[0] : 0,
        .use_byte   = (delim.len == 1 && delim.data != nullptr),
        .done       = false
    };
    return it;
}

bool rcre_split_next(RCRE_SplitIter* it, RCRE_Slice* out_token) {
    if (!it || !out_token || it->done) {
        return false;
    }
    if (it->use_byte) {
        RCRE_SlicePair p = rcre_slice_split_once_byte(it->rest, it->delim_byte);
        *out_token = p.head;
        if (p.found) {
            it->rest = p.tail;
        } else {
            it->rest = rcre_slice_empty();
            it->done = true;
        }
        return true;
    }
    if (it->delim.len == 0) {
        /* Empty delimiter yields the whole remaining slice once to avoid infinite loop */
        *out_token = it->rest;
        it->rest = rcre_slice_empty();
        it->done = true;
        return true;
    }
    size_t pos = rcre_slice_find(it->rest, it->delim);
    if (pos == RCRE_NPOS) {
        *out_token = it->rest;
        it->rest = rcre_slice_empty();
        it->done = true;
        return true;
    }
    *out_token = rcre_slice_sub(it->rest, 0, pos);
    it->rest = rcre_slice_sub(it->rest, pos + it->delim.len, it->rest.len);
    return true;
}

bool rcre_split_line_next(RCRE_Slice* cursor, RCRE_Slice* out_line) {
    if (!cursor || !out_line || !cursor->data || cursor->len == 0) {
        return false;
    }
    size_t nl = rcre_slice_find_byte(*cursor, '\n');
    RCRE_Slice line;
    if (nl == RCRE_NPOS) {
        line = *cursor;
        *cursor = rcre_slice_empty();
    } else {
        line = rcre_slice_sub(*cursor, 0, nl);
        *cursor = rcre_slice_sub(*cursor, nl + 1U, cursor->len);
    }
    if (line.len > 0 && line.data[line.len - 1U] == '\r') {
        line = rcre_slice_sub(line, 0, line.len - 1U);
    }
    *out_line = line;
    return true;
}

bool rcre_slice_parse_hex_u64(RCRE_Slice s, uint64_t* out_val) {
    if (!out_val || !s.data || s.len == 0) return false;
    size_t idx = 0;
    if (s.len >= 2 && s.data[0] == '0' && (s.data[1] == 'x' || s.data[1] == 'X')) {
        idx = 2;
    }
    if (idx >= s.len) return false;

    uint64_t acc = 0;
    for (; idx < s.len; ++idx) {
        uint8_t c = s.data[idx];
        uint64_t d = 0;
        if (c >= '0' && c <= '9')      d = (uint64_t)(c - '0');
        else if (c >= 'a' && c <= 'f') d = (uint64_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') d = (uint64_t)(c - 'A' + 10);
        else return false;

        if (acc > (UINT64_MAX >> 4)) return false;
        acc = (acc << 4) | d;
    }
    *out_val = acc;
    return true;
}

bool rcre_slice_parse_bool(RCRE_Slice s, bool* out_val) {
    if (!out_val) return false;
    s = rcre_slice_trim_ascii(s);
    if (rcre_slice_eq_ascii_ci(s, RCRE_STR("true")) ||
        rcre_slice_eq_ascii_ci(s, RCRE_STR("1")) ||
        rcre_slice_eq_ascii_ci(s, RCRE_STR("yes")) ||
        rcre_slice_eq_ascii_ci(s, RCRE_STR("on"))) {
        *out_val = true;
        return true;
    }
    if (rcre_slice_eq_ascii_ci(s, RCRE_STR("false")) ||
        rcre_slice_eq_ascii_ci(s, RCRE_STR("0")) ||
        rcre_slice_eq_ascii_ci(s, RCRE_STR("no")) ||
        rcre_slice_eq_ascii_ci(s, RCRE_STR("off"))) {
        *out_val = false;
        return true;
    }
    return false;
}

bool rcre_slice_parse_f64(RCRE_Slice s, double* out_val) {
    if (!out_val) return false;
    s = rcre_slice_trim_ascii(s);
    if (!s.data || s.len == 0) return false;

    size_t idx = 0;
    bool neg = false;
    if (s.data[idx] == '-') {
        neg = true;
        idx++;
    } else if (s.data[idx] == '+') {
        idx++;
    }
    if (idx >= s.len) return false;

    RCRE_Slice rem = rcre_slice_sub(s, idx, s.len);
    if (rcre_slice_eq_ascii_ci(rem, RCRE_STR("nan"))) {
        uint64_t qnan = neg ? 0xFFF8000000000000ULL : 0x7FF8000000000000ULL;
        *out_val = rcre_str__bits_to_f64(qnan);
        return true;
    }
    if (rcre_slice_eq_ascii_ci(rem, RCRE_STR("inf")) ||
        rcre_slice_eq_ascii_ci(rem, RCRE_STR("infinity"))) {
        uint64_t infb = neg ? 0xFFF0000000000000ULL : 0x7FF0000000000000ULL;
        *out_val = rcre_str__bits_to_f64(infb);
        return true;
    }

    uint64_t mant = 0;
    int digits = 0;
    int sig_digits = 0;
    int exp_adj = 0;
    bool seen_dot = false;

    while (idx < s.len) {
        uint8_t c = s.data[idx];
        if (c >= '0' && c <= '9') {
            digits++;
            if (sig_digits < 19) {
                if (mant != 0 || c != '0') {
                    mant = mant * 10ULL + (uint64_t)(c - '0');
                    sig_digits++;
                }
                if (seen_dot) exp_adj--;
            } else {
                if (!seen_dot) exp_adj++;
            }
            idx++;
        } else if (c == '.' && !seen_dot) {
            seen_dot = true;
            idx++;
        } else {
            break;
        }
    }

    if (digits == 0) return false;

    int exp10 = 0;
    if (idx < s.len && (s.data[idx] == 'e' || s.data[idx] == 'E')) {
        idx++;
        bool eneg = false;
        if (idx < s.len && (s.data[idx] == '-' || s.data[idx] == '+')) {
            eneg = (s.data[idx] == '-');
            idx++;
        }
        int edigits = 0;
        int eval = 0;
        while (idx < s.len && s.data[idx] >= '0' && s.data[idx] <= '9') {
            edigits++;
            if (eval < 10000) {
                eval = eval * 10 + (int)(s.data[idx] - '0');
            }
            idx++;
        }
        if (edigits == 0) return false;
        exp10 = eneg ? -eval : eval;
    }

    if (idx != s.len) return false;

    int total_exp = exp10 + exp_adj;
    if (mant == 0) {
        *out_val = neg ? -0.0 : 0.0;
        return true;
    }
    if (total_exp > 309) {
        *out_val = rcre_str__bits_to_f64(neg ? 0xFFF0000000000000ULL : 0x7FF0000000000000ULL);
        return true;
    }
    if (total_exp < -340) {
        *out_val = neg ? -0.0 : 0.0;
        return true;
    }

    /* Exact IEEE-754 double scaling for small exponents, binary exponentiation for large */
    static const double exact_pow10[23] = {
        1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
        1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
        1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
    };
    double res = (double)mant;
    if (total_exp >= 0 && total_exp <= 22) {
        res *= exact_pow10[total_exp];
    } else if (total_exp < 0 && total_exp >= -22) {
        res /= exact_pow10[-total_exp];
    } else if (total_exp > 22) {
        if (total_exp > 300) {
            res *= 1e300;
            res *= rcre_str__pow10_u32((unsigned int)(total_exp - 300));
        } else {
            res *= rcre_str__pow10_u32((unsigned int)total_exp);
        }
    } else {
        /* Split negative exponents past -300 to avoid intermediate 1e309 Inf */
        if (total_exp < -300) {
            res /= 1e300;
            res /= rcre_str__pow10_u32((unsigned int)(-total_exp - 300));
        } else {
            res /= rcre_str__pow10_u32((unsigned int)(-total_exp));
        }
    }

    *out_val = neg ? -res : res;
    return true;
}

/* ========================================================================= */
/* 3. Expanded Arena String Builder & Transforms (STR-REQ-03)                */
/* ========================================================================= */

void rcre_sb_reset(RCRE_StrBuilder* sb) {
    if (!sb) return;
    sb->len = 0;
    sb->overflow = (sb->arena == nullptr);
    if (sb->buf && sb->cap > 0) {
        sb->buf[0] = '\0';
    }
}

bool rcre_sb_append_hex(RCRE_StrBuilder* sb, uint64_t val, bool uppercase) {
    uint8_t buf[24];
    size_t n = rcre_str__u64_to_hex(val, uppercase, buf);
    return rcre_sb_append_slice(sb, rcre_slice_from_parts(buf, n));
}

bool rcre_sb_append_f64(RCRE_StrBuilder* sb, double val, int precision) {
    if (!sb || sb->overflow) return false;
    rcre_str__sink sink = { .strbuf = nullptr, .builder = sb };
    return rcre_str__sink_f64(&sink, val, precision);
}

bool rcre_sb_append_cp(RCRE_StrBuilder* sb, uint32_t cp) {
    uint8_t enc[4];
    size_t n = rcre_utf8_encode(cp, enc);
    if (n == 0) return false;
    return rcre_sb_append_slice(sb, rcre_slice_from_parts(enc, n));
}

bool rcre_sb_append_repeat(RCRE_StrBuilder* sb, RCRE_Slice unit, size_t count) {
    if (!sb || sb->overflow) return false;
    if (unit.len == 0 || count == 0) return true;
    for (size_t i = 0; i < count; ++i) {
        if (!rcre_sb_append_slice(sb, unit)) return false;
    }
    return true;
}

bool rcre_sb_append_vfmt(RCRE_StrBuilder* sb, const char* fmt, va_list ap) {
    if (!sb || sb->overflow) return false;
    rcre_str__sink sink = { .strbuf = nullptr, .builder = sb };
    return rcre_str__sink_vfmt(&sink, fmt, ap);
}

bool rcre_sb_append_fmt(RCRE_StrBuilder* sb, const char* fmt, ...) {
    if (!sb || sb->overflow) return false;
    va_list ap;
    va_start(ap, fmt);
    bool ok = rcre_sb_append_vfmt(sb, fmt, ap);
    va_end(ap);
    return ok;
}

bool rcre_sb_append_escape_json(RCRE_StrBuilder* sb, RCRE_Slice s) {
    if (!sb || sb->overflow) return false;
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < s.len; ++i) {
        uint8_t c = s.data[i];
        switch (c) {
            case '"':  if (!rcre_sb_append_slice(sb, RCRE_STR("\\\""))) return false; break;
            case '\\': if (!rcre_sb_append_slice(sb, RCRE_STR("\\\\"))) return false; break;
            case '\b': if (!rcre_sb_append_slice(sb, RCRE_STR("\\b")))  return false; break;
            case '\f': if (!rcre_sb_append_slice(sb, RCRE_STR("\\f")))  return false; break;
            case '\n': if (!rcre_sb_append_slice(sb, RCRE_STR("\\n")))  return false; break;
            case '\r': if (!rcre_sb_append_slice(sb, RCRE_STR("\\r")))  return false; break;
            case '\t': if (!rcre_sb_append_slice(sb, RCRE_STR("\\t")))  return false; break;
            default:
                if (c < 0x20U) {
                    uint8_t esc[6] = { '\\', 'u', '0', '0', (uint8_t)hex[c >> 4], (uint8_t)hex[c & 0xFU] };
                    if (!rcre_sb_append_slice(sb, rcre_slice_from_parts(esc, 6))) return false;
                } else {
                    if (!rcre_sb_append_byte(sb, c)) return false;
                }
                break;
        }
    }
    return true;
}

bool rcre_sb_append_escape_csv(RCRE_StrBuilder* sb, RCRE_Slice s, uint8_t delim) {
    if (!sb || sb->overflow) return false;
    bool needs_quote = false;
    for (size_t i = 0; i < s.len; ++i) {
        uint8_t c = s.data[i];
        if (c == delim || c == '"' || c == '\n' || c == '\r') {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote) {
        return rcre_sb_append_slice(sb, s);
    }
    if (!rcre_sb_append_byte(sb, '"')) return false;
    for (size_t i = 0; i < s.len; ++i) {
        if (s.data[i] == '"') {
            if (!rcre_sb_append_slice(sb, RCRE_STR("\"\""))) return false;
        } else {
            if (!rcre_sb_append_byte(sb, s.data[i])) return false;
        }
    }
    return rcre_sb_append_byte(sb, '"');
}

RCRE_Slice rcre_str_dup(RCRE_Arena* arena, RCRE_Slice s) {
    if (!arena || s.len == 0 || !s.data) {
        return rcre_slice_empty();
    }
    if (s.len > SIZE_MAX - 1U) {
        return rcre_slice_empty();
    }
    uint8_t* dst = (uint8_t*)rcre_arena_alloc(arena, s.len + 1U, 8);
    if (!dst) {
        return rcre_slice_empty();
    }
    memcpy(dst, s.data, s.len);
    dst[s.len] = '\0';
    return rcre_slice_from_parts(dst, s.len);
}

RCRE_Slice rcre_str_concat(RCRE_Arena* arena, RCRE_Slice a, RCRE_Slice b) {
    if (!arena) return rcre_slice_empty();
    if (a.len > SIZE_MAX - b.len - 1U) return rcre_slice_empty();
    size_t total = a.len + b.len;
    if (total == 0) return rcre_slice_empty();
    uint8_t* dst = (uint8_t*)rcre_arena_alloc(arena, total + 1U, 8);
    if (!dst) return rcre_slice_empty();
    if (a.len > 0 && a.data) memcpy(dst, a.data, a.len);
    if (b.len > 0 && b.data) memcpy(dst + a.len, b.data, b.len);
    dst[total] = '\0';
    return rcre_slice_from_parts(dst, total);
}

RCRE_Slice rcre_str_join(RCRE_Arena* arena, const RCRE_Slice* parts, size_t count, RCRE_Slice sep) {
    if (!arena || !parts || count == 0) return rcre_slice_empty();
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        if (parts[i].len > SIZE_MAX - total) return rcre_slice_empty();
        total += parts[i].len;
        if (i + 1U < count) {
            if (sep.len > SIZE_MAX - total) return rcre_slice_empty();
            total += sep.len;
        }
    }
    if (total == 0 || total > SIZE_MAX - 1U) return rcre_slice_empty();
    uint8_t* dst = (uint8_t*)rcre_arena_alloc(arena, total + 1U, 8);
    if (!dst) return rcre_slice_empty();

    size_t pos = 0;
    for (size_t i = 0; i < count; ++i) {
        if (parts[i].len > 0 && parts[i].data) {
            memcpy(dst + pos, parts[i].data, parts[i].len);
            pos += parts[i].len;
        }
        if (i + 1U < count && sep.len > 0 && sep.data) {
            memcpy(dst + pos, sep.data, sep.len);
            pos += sep.len;
        }
    }
    dst[pos] = '\0';
    return rcre_slice_from_parts(dst, pos);
}

RCRE_Slice rcre_str_replace_all(RCRE_Arena* arena, RCRE_Slice s, RCRE_Slice old_sub, RCRE_Slice new_sub) {
    if (!arena || s.len == 0 || !s.data) return rcre_slice_empty();
    if (old_sub.len == 0 || !old_sub.data) {
        return rcre_str_dup(arena, s);
    }

    RCRE_StrBuilder sb = rcre_sb_init(arena, s.len);
    RCRE_Slice cur = s;
    while (cur.len > 0) {
        size_t pos = rcre_slice_find(cur, old_sub);
        if (pos == RCRE_NPOS) {
            if (!rcre_sb_append_slice(&sb, cur)) return rcre_slice_empty();
            break;
        }
        if (pos > 0) {
            if (!rcre_sb_append_slice(&sb, rcre_slice_sub(cur, 0, pos))) return rcre_slice_empty();
        }
        if (new_sub.len > 0) {
            if (!rcre_sb_append_slice(&sb, new_sub)) return rcre_slice_empty();
        }
        cur = rcre_slice_sub(cur, pos + old_sub.len, cur.len);
    }
    return rcre_sb_to_slice(&sb);
}

RCRE_Slice rcre_str_to_lower_ascii(RCRE_Arena* arena, RCRE_Slice s) {
    if (!arena || s.len == 0 || !s.data || s.len > SIZE_MAX - 1U) {
        return rcre_slice_empty();
    }
    uint8_t* dst = (uint8_t*)rcre_arena_alloc(arena, s.len + 1U, 64);
    if (!dst) return rcre_slice_empty();
    rcre_hal.ascii_to_lower(s.data, dst, s.len);
    dst[s.len] = '\0';
    return rcre_slice_from_parts(dst, s.len);
}

RCRE_Slice rcre_str_to_upper_ascii(RCRE_Arena* arena, RCRE_Slice s) {
    if (!arena || s.len == 0 || !s.data || s.len > SIZE_MAX - 1U) {
        return rcre_slice_empty();
    }
    uint8_t* dst = (uint8_t*)rcre_arena_alloc(arena, s.len + 1U, 64);
    if (!dst) return rcre_slice_empty();
    rcre_hal.ascii_to_upper(s.data, dst, s.len);
    dst[s.len] = '\0';
    return rcre_slice_from_parts(dst, s.len);
}

RCRE_Slice rcre_str_fmt(RCRE_Arena* arena, const char* fmt, ...) {
    if (!arena || !fmt) return rcre_slice_empty();
    RCRE_StrBuilder sb = rcre_sb_init(arena, 64);
    va_list ap;
    va_start(ap, fmt);
    bool ok = rcre_sb_append_vfmt(&sb, fmt, ap);
    va_end(ap);
    if (!ok || sb.overflow) return rcre_slice_empty();
    return rcre_sb_to_slice(&sb);
}

/* ========================================================================= */
/* 4. Strict RFC 3629 UTF-8 Layer (STR-REQ-04)                               */
/* ========================================================================= */

static int rcre_str__utf8_decode_at(const uint8_t* p, size_t len, uint32_t* out_cp) {
    if (len == 0) return 0;
    uint8_t b0 = p[0];
    if (b0 < 0x80U) {
        if (out_cp) *out_cp = (uint32_t)b0;
        return 1;
    }
    if (b0 >= 0xC2U && b0 <= 0xDFU) {
        if (len < 2U) goto invalid;
        uint8_t b1 = p[1];
        if ((b1 & 0xC0U) != 0x80U) goto invalid;
        if (out_cp) *out_cp = ((uint32_t)(b0 & 0x1FU) << 6) | (uint32_t)(b1 & 0x3FU);
        return 2;
    }
    if (b0 >= 0xE0U && b0 <= 0xEFU) {
        if (len < 3U) goto invalid;
        uint8_t b1 = p[1];
        uint8_t b2 = p[2];
        if ((b1 & 0xC0U) != 0x80U || (b2 & 0xC0U) != 0x80U) goto invalid;
        if (b0 == 0xE0U && b1 < 0xA0U) goto invalid; /* Overlong 3-byte */
        if (b0 == 0xEDU && b1 >= 0xA0U) goto invalid; /* UTF-16 surrogates U+D800..U+DFFF */
        if (out_cp) {
            *out_cp = ((uint32_t)(b0 & 0x0FU) << 12) |
                      ((uint32_t)(b1 & 0x3FU) << 6)  |
                      (uint32_t)(b2 & 0x3FU);
        }
        return 3;
    }
    if (b0 >= 0xF0U && b0 <= 0xF4U) {
        if (len < 4U) goto invalid;
        uint8_t b1 = p[1];
        uint8_t b2 = p[2];
        uint8_t b3 = p[3];
        if ((b1 & 0xC0U) != 0x80U || (b2 & 0xC0U) != 0x80U || (b3 & 0xC0U) != 0x80U) goto invalid;
        if (b0 == 0xF0U && b1 < 0x90U) goto invalid;  /* Overlong 4-byte */
        if (b0 == 0xF4U && b1 >= 0x90U) goto invalid; /* > U+10FFFF */
        if (out_cp) {
            *out_cp = ((uint32_t)(b0 & 0x07U) << 18) |
                      ((uint32_t)(b1 & 0x3FU) << 12) |
                      ((uint32_t)(b2 & 0x3FU) << 6)  |
                      (uint32_t)(b3 & 0x3FU);
        }
        return 4;
    }

invalid:
    if (out_cp) *out_cp = RCRE_UTF8_REPLACEMENT_CP;
    return -1;
}

bool rcre_utf8_validate(RCRE_Slice s, size_t* out_err_offset) {
    if (!s.data || s.len == 0) {
        return true;
    }
    if (rcre_hal.utf8_validate(s.data, s.len)) {
        return true;
    }
    if (out_err_offset) {
        size_t i = 0;
        while (i < s.len) {
            int step = rcre_str__utf8_decode_at(s.data + i, s.len - i, nullptr);
            if (step <= 0) {
                *out_err_offset = i;
                return false;
            }
            i += (size_t)step;
        }
        *out_err_offset = s.len;
    }
    return false;
}

bool rcre_utf8_is_ascii(RCRE_Slice s) {
    if (!s.data || s.len == 0) return true;
    size_t i = 0;
    uint64_t acc = 0;
    for (; i + 8U <= s.len; i += 8U) {
        acc |= rcre_str__load_u64_le(s.data + i);
    }
    if ((acc & 0x8080808080808080ULL) != 0ULL) return false;
    for (; i < s.len; ++i) {
        if ((s.data[i] & 0x80U) != 0U) return false;
    }
    return true;
}

size_t rcre_utf8_count(RCRE_Slice s) {
    if (!s.data || s.len == 0) return 0;
    if (!rcre_hal.utf8_validate(s.data, s.len)) {
        return RCRE_NPOS;
    }
    /* For validated UTF-8, codepoint count == bytes where (b & 0xC0) != 0x80 */
    size_t count = 0;
    for (size_t i = 0; i < s.len; ++i) {
        count += (size_t)((s.data[i] & 0xC0U) != 0x80U);
    }
    return count;
}

int rcre_utf8_decode_next(RCRE_Slice* cursor, uint32_t* out_cp) {
    if (!cursor || !cursor->data || cursor->len == 0) {
        if (out_cp) *out_cp = 0;
        return 0;
    }
    int step = rcre_str__utf8_decode_at(cursor->data, cursor->len, out_cp);
    if (step > 0) {
        cursor->data += (size_t)step;
        cursor->len -= (size_t)step;
        if (cursor->len == 0) cursor->data = nullptr;
    }
    return step;
}

int rcre_utf8_decode_prev(RCRE_Slice* cursor, uint32_t* out_cp) {
    if (!cursor || !cursor->data || cursor->len == 0) {
        if (out_cp) *out_cp = 0;
        return 0;
    }
    size_t end = cursor->len;
    size_t start = end - 1U;
    size_t backtrack = 0;
    while (start > 0 && ((cursor->data[start] & 0xC0U) == 0x80U) && backtrack < 3U) {
        start--;
        backtrack++;
    }
    size_t seq_len = end - start;
    int step = rcre_str__utf8_decode_at(cursor->data + start, seq_len, out_cp);
    if (step != (int)seq_len) {
        if (out_cp) *out_cp = RCRE_UTF8_REPLACEMENT_CP;
        return -1;
    }
    cursor->len = start;
    if (cursor->len == 0) cursor->data = nullptr;
    return step;
}

size_t rcre_utf8_encode(uint32_t cp, uint8_t out[4]) {
    if (!out) return 0;
    if (cp <= 0x7FU) {
        out[0] = (uint8_t)cp;
        return 1;
    }
    if (cp <= 0x7FFU) {
        out[0] = (uint8_t)(0xC0U | (cp >> 6));
        out[1] = (uint8_t)(0x80U | (cp & 0x3FU));
        return 2;
    }
    if (cp >= 0xD800U && cp <= 0xDFFFU) {
        return 0; /* Reject UTF-16 surrogates */
    }
    if (cp <= 0xFFFFU) {
        out[0] = (uint8_t)(0xE0U | (cp >> 12));
        out[1] = (uint8_t)(0x80U | ((cp >> 6) & 0x3FU));
        out[2] = (uint8_t)(0x80U | (cp & 0x3FU));
        return 3;
    }
    if (cp <= 0x10FFFFU) {
        out[0] = (uint8_t)(0xF0U | (cp >> 18));
        out[1] = (uint8_t)(0x80U | ((cp >> 12) & 0x3FU));
        out[2] = (uint8_t)(0x80U | ((cp >> 6) & 0x3FU));
        out[3] = (uint8_t)(0x80U | (cp & 0x3FU));
        return 4;
    }
    return 0;
}

RCRE_Slice rcre_utf8_truncate_bytes(RCRE_Slice s, size_t max_bytes) {
    if (!s.data || s.len == 0 || max_bytes == 0) {
        return rcre_slice_empty();
    }
    size_t limit = (max_bytes < s.len) ? max_bytes : s.len;
    /* Fast path when the entire input is already valid UTF-8 and fits */
    if (limit == s.len && rcre_hal.utf8_validate(s.data, s.len)) {
        return s;
    }
    size_t i = 0;
    while (i < limit) {
        int step = rcre_str__utf8_decode_at(s.data + i, s.len - i, nullptr);
        if (step <= 0 || i + (size_t)step > limit) {
            break;
        }
        i += (size_t)step;
    }
    return rcre_slice_sub(s, 0, i);
}

RCRE_Slice rcre_utf8_sub_codepoints(RCRE_Slice s, size_t start_cp, size_t count_cp) {
    if (!s.data || s.len == 0 || count_cp == 0) {
        return rcre_slice_empty();
    }
    RCRE_Slice cur = s;
    size_t idx = 0;
    while (idx < start_cp && cur.len > 0) {
        int step = rcre_utf8_decode_next(&cur, nullptr);
        if (step <= 0) return rcre_slice_empty();
        idx++;
    }
    if (idx < start_cp || cur.len == 0) {
        return rcre_slice_empty();
    }
    const uint8_t* start_ptr = cur.data;
    size_t taken = 0;
    while (taken < count_cp && cur.len > 0) {
        int step = rcre_utf8_decode_next(&cur, nullptr);
        if (step <= 0) break;
        taken++;
    }
    size_t byte_len = (cur.data ? (size_t)(cur.data - start_ptr) : (size_t)((s.data + s.len) - start_ptr));
    return rcre_slice_from_parts(start_ptr, byte_len);
}

/* ========================================================================= */
/* 5. Arena String Interning Pool (STR-REQ-06)                               */
/* ========================================================================= */

static bool rcre_str__pool_grow(RCRE_StrPool* pool, size_t min_slot_cap, size_t min_id_cap) {
    size_t new_slot_cap = rcre_str__next_pow2(min_slot_cap);
    if (new_slot_cap == 0 || new_slot_cap > SIZE_MAX / sizeof(RCRE_StrPoolEntry)) {
        return false;
    }
    if (min_id_cap > SIZE_MAX / sizeof(RCRE_Slice)) {
        return false;
    }

    RCRE_StrPoolEntry* new_slots = (RCRE_StrPoolEntry*)rcre_arena_alloc(
        pool->arena, new_slot_cap * sizeof(RCRE_StrPoolEntry), alignof(RCRE_StrPoolEntry));
    if (!new_slots) return false;
    for (size_t i = 0; i < new_slot_cap; ++i) {
        new_slots[i].hash = 0;
        new_slots[i].slice = rcre_slice_empty();
        new_slots[i].id = RCRE_STR_ID_INVALID;
    }

    RCRE_Slice* new_by_id = pool->by_id;
    size_t new_id_cap = pool->id_cap;
    if (min_id_cap > pool->id_cap) {
        new_id_cap = (pool->id_cap == 0) ? 16U : pool->id_cap;
        while (new_id_cap < min_id_cap) {
            if (new_id_cap > SIZE_MAX / 2U) { new_id_cap = min_id_cap; break; }
            new_id_cap *= 2U;
        }
        new_by_id = (RCRE_Slice*)rcre_arena_alloc(
            pool->arena, new_id_cap * sizeof(RCRE_Slice), alignof(RCRE_Slice));
        if (!new_by_id) return false;
        if (pool->count > 0 && pool->by_id) {
            memcpy(new_by_id, pool->by_id, pool->count * sizeof(RCRE_Slice));
        }
    }

    /* Reinsert existing entries */
    size_t mask = new_slot_cap - 1U;
    for (size_t i = 0; i < pool->slot_cap; ++i) {
        if (pool->slots[i].id != RCRE_STR_ID_INVALID) {
            size_t idx = (size_t)(pool->slots[i].hash & (uint64_t)mask);
            while (new_slots[idx].id != RCRE_STR_ID_INVALID) {
                idx = (idx + 1U) & mask;
            }
            new_slots[idx] = pool->slots[i];
        }
    }

    pool->slots = new_slots;
    pool->slot_cap = new_slot_cap;
    pool->by_id = new_by_id;
    pool->id_cap = new_id_cap;
    return true;
}

int rcre_strpool_init(RCRE_StrPool* pool, RCRE_Arena* arena, size_t initial_cap) {
    if (!pool || !arena) return RCRE_ERR_INVALID_ARG;
    memset(pool, 0, sizeof(*pool));
    pool->arena = arena;
    pool->seed = 0x517cc1b727220a95ULL;

    size_t id_cap = (initial_cap < 16U) ? 16U : initial_cap;
    size_t slot_cap = rcre_str__next_pow2(id_cap * 2U);
    if (!rcre_str__pool_grow(pool, slot_cap, id_cap)) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    return RCRE_OK;
}

RCRE_StrId rcre_strpool_find(const RCRE_StrPool* pool, RCRE_Slice s) {
    if (!pool || !pool->slots || pool->slot_cap == 0) {
        return RCRE_STR_ID_INVALID;
    }
    uint64_t h = rcre_slice_hash64_seeded(s, pool->seed);
    size_t mask = pool->slot_cap - 1U;
    size_t idx = (size_t)(h & (uint64_t)mask);
    for (;;) {
        const RCRE_StrPoolEntry* e = &pool->slots[idx];
        if (e->id == RCRE_STR_ID_INVALID) {
            return RCRE_STR_ID_INVALID;
        }
        if (e->hash == h && rcre_slice_eq(e->slice, s)) {
            return e->id;
        }
        idx = (idx + 1U) & mask;
    }
}

RCRE_StrId rcre_strpool_intern(RCRE_StrPool* pool, RCRE_Slice s) {
    if (!pool || !pool->arena || !pool->slots || pool->slot_cap == 0) {
        return RCRE_STR_ID_INVALID;
    }
    if (pool->count >= (size_t)RCRE_STR_ID_INVALID) {
        return RCRE_STR_ID_INVALID;
    }

    RCRE_StrId existing = rcre_strpool_find(pool, s);
    if (existing != RCRE_STR_ID_INVALID) {
        return existing;
    }

    /* Keep load factor <= 70% */
    if ((pool->count + 1U) * 10U >= pool->slot_cap * 7U || pool->count >= pool->id_cap) {
        size_t next_slots = (pool->slot_cap <= SIZE_MAX / 2U) ? (pool->slot_cap * 2U) : 0U;
        size_t next_ids = (pool->count + 1U > pool->id_cap) ? (pool->id_cap * 2U) : pool->id_cap;
        if (next_slots == 0 || !rcre_str__pool_grow(pool, next_slots, next_ids)) {
            return RCRE_STR_ID_INVALID;
        }
    }

    /* Always store a NUL-terminated copy in the arena (even for empty slice) */
    if (s.len > SIZE_MAX - 1U) return RCRE_STR_ID_INVALID;
    uint8_t* copy = (uint8_t*)rcre_arena_alloc(pool->arena, s.len + 1U, 8);
    if (!copy) return RCRE_STR_ID_INVALID;
    if (s.len > 0 && s.data) {
        memcpy(copy, s.data, s.len);
    }
    copy[s.len] = '\0';

    RCRE_Slice stored = { .data = copy, .len = s.len };
    uint64_t h = rcre_slice_hash64_seeded(stored, pool->seed);
    RCRE_StrId id = (RCRE_StrId)pool->count;

    size_t mask = pool->slot_cap - 1U;
    size_t idx = (size_t)(h & (uint64_t)mask);
    while (pool->slots[idx].id != RCRE_STR_ID_INVALID) {
        idx = (idx + 1U) & mask;
    }
    pool->slots[idx].hash = h;
    pool->slots[idx].slice = stored;
    pool->slots[idx].id = id;
    pool->by_id[id] = stored;
    pool->count++;
    return id;
}

RCRE_Slice rcre_strpool_get(const RCRE_StrPool* pool, RCRE_StrId id) {
    if (!pool || !pool->by_id || (size_t)id >= pool->count) {
        return rcre_slice_empty();
    }
    RCRE_Slice s = pool->by_id[id];
    if (s.len == 0) {
        return rcre_slice_empty();
    }
    return s;
}

const char* rcre_strpool_cstr(const RCRE_StrPool* pool, RCRE_StrId id) {
    if (!pool || !pool->by_id || (size_t)id >= pool->count) {
        return "";
    }
    return pool->by_id[id].data ? (const char*)pool->by_id[id].data : "";
}

size_t rcre_strpool_count(const RCRE_StrPool* pool) {
    return pool ? pool->count : 0;
}

/* ========================================================================= */
/* 6. Safe Path Manipulation (STR-REQ-06)                                    */
/* ========================================================================= */

static RCRE_Slice rcre_str__trim_trailing_seps(RCRE_Slice path) {
    size_t end = path.len;
    while (end > 1U && rcre_str__is_path_sep(path.data[end - 1U])) {
        end--;
    }
    return rcre_slice_sub(path, 0, end);
}

RCRE_Slice rcre_path_basename(RCRE_Slice path) {
    if (!path.data || path.len == 0) return rcre_slice_empty();
    RCRE_Slice p = rcre_str__trim_trailing_seps(path);
    if (p.len == 1U && rcre_str__is_path_sep(p.data[0])) {
        return p;
    }
    size_t i = p.len;
    while (i > 0) {
        i--;
        if (rcre_str__is_path_sep(p.data[i])) {
            return rcre_slice_sub(p, i + 1U, p.len);
        }
    }
    return p;
}

RCRE_Slice rcre_path_dirname(RCRE_Slice path) {
    if (!path.data || path.len == 0) return RCRE_STR(".");
    RCRE_Slice p = rcre_str__trim_trailing_seps(path);
    if (p.len == 1U && rcre_str__is_path_sep(p.data[0])) {
        return p;
    }
    size_t i = p.len;
    while (i > 0) {
        i--;
        if (rcre_str__is_path_sep(p.data[i])) {
            if (i == 0) return rcre_slice_sub(p, 0, 1);
            return rcre_str__trim_trailing_seps(rcre_slice_sub(p, 0, i));
        }
    }
    return RCRE_STR(".");
}

RCRE_Slice rcre_path_ext(RCRE_Slice path) {
    RCRE_Slice base = rcre_path_basename(path);
    if (base.len <= 1U) return rcre_slice_empty();
    size_t dot = rcre_slice_rfind_byte(base, '.');
    /* Hidden file like ".bashrc" (dot == 0) has no extension */
    if (dot == RCRE_NPOS || dot == 0) {
        return rcre_slice_empty();
    }
    return rcre_slice_sub(base, dot, base.len);
}

RCRE_Slice rcre_path_stem(RCRE_Slice path) {
    RCRE_Slice base = rcre_path_basename(path);
    RCRE_Slice ext = rcre_path_ext(base);
    if (ext.len == 0) return base;
    return rcre_slice_sub(base, 0, base.len - ext.len);
}

RCRE_Slice rcre_path_join(RCRE_Arena* arena, RCRE_Slice a, RCRE_Slice b) {
    if (!arena) return rcre_slice_empty();
    if (a.len == 0) return rcre_str_dup(arena, b);
    if (b.len == 0) return rcre_str_dup(arena, a);
    /* Absolute RHS overrides LHS (POSIX convention) */
    if (rcre_str__is_path_sep(b.data[0])) {
        return rcre_str_dup(arena, b);
    }
    size_t a_end = a.len;
    while (a_end > 1U && rcre_str__is_path_sep(a.data[a_end - 1U])) {
        a_end--;
    }
    RCRE_Slice a_trim = rcre_slice_sub(a, 0, a_end);
    if (a_trim.len == 1U && rcre_str__is_path_sep(a_trim.data[0])) {
        return rcre_str_concat(arena, a_trim, b);
    }
    RCRE_Slice parts[2] = { a_trim, b };
    return rcre_str_join(arena, parts, 2, RCRE_STR("/"));
}
