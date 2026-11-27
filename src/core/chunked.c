/* RCRE H5 — Zarr v3 chunked ND arrays (out-of-core substrate, 2.2.0)
 *
 * Zero-dependency core: the "bytes" (identity) codec is always available;
 * the "zstd" codec is an opt-in translation unit (src/core/zarr_zstd.c,
 * built when RCRE_WITH_ZSTD is defined and libzstd is present) that
 * registers itself into the codec table below — the D1 bridge/extras
 * pattern, infrastructure-immune (a build without libzstd loses only the
 * zstd codec; everything else is unaffected).
 *
 * Format conformance (zarr-python 3.x, Zarr v3 spec):
 *   <root>/<name>/zarr.json          array metadata (v3)
 *   <root>/<name>/c/<i>/<j>/...      chunk files, no extension; keys are
 *                                     the chunk indices joined with "/"
 *                                     (the v3 "default" chunk-key encoding)
 *   <root>/zarr.json                 optional group root
 * Codec pipeline: codecs are applied in LIST ORDER on write and REVERSE
 * list order on read (v3 semantics). The identity pipeline is
 * [{"bytes": {"endian": "little"}}]; the zstd pipeline appends
 * [{"zstd": {"level": L, "checksum": false}}].
 * A missing chunk file reads as the fill value (v3 semantics).
 *
 * Threading: array-level operations dispatch one task per chunk through
 * rcre_threadpool_parallel_for (NULL pool = serial). Per-chunk tasks touch
 * disjoint buffer regions, so pooled runs are bit-identical to serial and
 * to each other regardless of work-stealing order (library-wide slab
 * determinism contract). rcre_zarr_reduce_sum keeps one Neumaier partial
 * per chunk (task id) and combines them in fixed chunk order —
 * bit-identical across pool sizes and machines (the H0 reduction
 * contract).
 */
#include "rcre.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(_WIN32)
  /* POSIX file store only: Zarr v3 on Windows is out of scope for H5
   * (the io layer documents the same boundary). */
  #error "rcre chunked (Zarr v3) requires a POSIX filesystem"
#endif

/* ========================================================================= */
/* dtype table                                                               */
/* ========================================================================= */

typedef struct {
    const char* name;   /* numpy-style name (v3 "data_type" field) */
    size_t      itemsize;
    int         cls;    /* 1 float, 0 integer, -1 bool */
} ZarrDType;

static const ZarrDType ZDT[] = {
    { "float64", 8,  1 },
    { "float32", 4,  1 },
    { "float16", 2,  1 },
    { "int64",   8,  0 },
    { "int32",   4,  0 },
    { "int16",   2,  0 },
    { "int8",    1,  0 },
    { "uint64",  8,  0 },
    { "uint32",  4,  0 },
    { "uint16",  2,  0 },
    { "uint8",   1,  0 },
    { "bool",    1, -1 },
};
static const int ZDT_N = (int)(sizeof(ZDT) / sizeof(ZDT[0]));

/* ---- float16 (IEEE 754 binary16) encode/decode --------------------------- */

/* Decode a binary16 bit pattern to double. */
static double f16_to_f64(uint16_t u) {
    int neg = (int)(u >> 15);
    int exp = (int)((u >> 10) & 0x1F);
    uint32_t mant = u & 0x3FF;
    double v;
    if (exp == 0) {
        v = mant ? ldexp((double)mant, -24) : 0.0; /* subnormal or zero */
    } else if (exp == 31) {
        v = mant ? NAN : INFINITY;
    } else {
        v = (1.0 + (double)mant * 0x1.0p-10) * ldexp(1.0, exp - 15);
    }
    return neg ? -v : v;
}

/* Encode a double to a binary16 bit pattern (round to nearest, ties to
 * even). Overflow -> ±inf, underflow -> flush to (possibly signed) zero. */
static uint16_t f64_to_f16(double v) {
    if (isnan(v)) return 0x7E00;
    int neg = signbit(v);
    v = fabs(v);
    if (isinf(v)) return (uint16_t)(neg ? 0xFC00 : 0x7C00);
    if (v >= 65504.0 + 32.0) return (uint16_t)(neg ? 0xFC00 : 0x7C00); /* overflow */
    if (v == 0.0) return (uint16_t)(neg ? 0x8000 : 0x0000);
    int exp;
    double m = frexp(v, &exp); /* v = m * 2^exp, m in [0.5, 1) */
    /* normalized half: v = 1.f * 2^(E-15), E in 1..30 */
    if (exp < -13) {
        /* subnormal: value = round(v / 2^-24) * 2^-24, 10-bit field */
        uint32_t bits = (uint32_t)llround(v / 0x1.0p-24);
        if (bits > 0x3FF) bits = 0x3FF;
        return (uint16_t)((neg ? 0x8000u : 0u) | bits);
    }
    int E = exp - 1; /* v = 1.xxx * 2^(E-15) */
    if (E < 1) { /* very small normalized -> subnormal path */
        uint32_t bits = (uint32_t)llround(v / 0x1.0p-24);
        if (bits > 0x3FF) bits = 0x3FF;
        return (uint16_t)((neg ? 0x8000u : 0u) | bits);
    }
    /* mantissa: 10 bits + rounding bit + sticky */
    double frac = (m - 0.5) * 2.0; /* in [0,1): the 1.xxx fraction */
    uint32_t full = (uint32_t)llround(frac * 2048.0); /* 11-bit field */
    if (full > 2047) { /* round-up into next exponent */
        full = 1024;
        E++;
        if (E > 30) return (uint16_t)(neg ? 0xFC00 : 0x7C00);
    }
    return (uint16_t)((neg ? 0x8000u : 0u) | ((uint32_t)(E - 1) << 10) | (full & 0x3FF));
}

/* ========================================================================= */
/* codec table (identity always; zstd self-registers from its own TU)        */
/* ========================================================================= */

typedef struct {
    const char*  name;
    int (*decode)(const uint8_t* in, size_t in_len, uint8_t* out, size_t out_cap);
    int (*encode)(const uint8_t* in, size_t in_len, int level,
                  uint8_t** out, size_t* out_len);
    int          registered;
} ZarrCodec;

static ZarrCodec g_codecs[8];
static int       g_codecs_init = 0;

static void zarr_codec_init(void) {
    if (g_codecs_init) return;
    g_codecs_init = 1;
    for (int i = 0; i < 8; ++i) {
        g_codecs[i].name = NULL;
        g_codecs[i].decode = NULL;
        g_codecs[i].encode = NULL;
        g_codecs[i].registered = 0;
    }
    g_codecs[0].name = "bytes";
    g_codecs[0].registered = 1; /* identity: decode/encode are no-ops */
}

/* Registered by src/core/zarr_zstd.c at first use (also callable directly). */
RCRE_NODISCARD int rcre_zarr_register_codec(
    const char* name,
    int (*decode)(const uint8_t*, size_t, uint8_t*, size_t),
    int (*encode)(const uint8_t*, size_t, int, uint8_t**, size_t*)) {
    zarr_codec_init();
    for (int i = 0; i < 8; ++i) {
        if (g_codecs[i].name && strcmp(g_codecs[i].name, name) == 0) {
            g_codecs[i].decode = decode;
            g_codecs[i].encode = encode;
            g_codecs[i].registered = 1;
            return RCRE_OK;
        }
    }
    for (int i = 0; i < 8; ++i) {
        if (!g_codecs[i].name) {
            g_codecs[i].name = strdup(name);
            g_codecs[i].decode = decode;
            g_codecs[i].encode = encode;
            g_codecs[i].registered = 1;
            return RCRE_OK;
        }
    }
    return RCRE_ERR_INVALID_ARG; /* table full */
}

static ZarrCodec* zarr_codec_find(const char* name) {
    zarr_codec_init();
    for (int i = 0; i < 8; ++i)
        if (g_codecs[i].name && strcmp(g_codecs[i].name, name) == 0)
            return &g_codecs[i];
    return NULL;
}

RCRE_NODISCARD int rcre_zarr_codec_available(const char* name) {
    ZarrCodec* c = zarr_codec_find(name);
    return (c && c->registered) ? 1 : 0;
}

/* ========================================================================= */
/* minimal JSON value tree (parser + writer)                                 */
/* ========================================================================= */

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JKind;

typedef struct JVal JVal;
struct JVal {
    JKind     kind;
    int       b;
    int       is_int; /* J_NUM: emit as a JSON integer (no ".0") */
    double    num;
    char*     str;   /* J_STR */
    JVal**    keys;  /* J_ARR items / J_OBJ keys */
    JVal**    vals;  /* J_OBJ values (parallel to keys) */
    int       n;
    int       cap;
};

static JVal* jv_new(JKind k) {
    JVal* v = (JVal*)calloc(1, sizeof(JVal));
    if (v) v->kind = k;
    return v;
}

static void jv_free(JVal* v) {
    if (!v) return;
    free(v->str);
    for (int i = 0; i < v->n; ++i) {
        jv_free(v->keys[i]);
        if (v->vals) jv_free(v->vals[i]);
    }
    free(v->keys);
    free(v->vals);
    free(v);
}

static int jv_push(JVal* o, JVal* key, JVal* val) {
    if (o->n >= o->cap) {
        int nc = o->cap ? o->cap * 2 : 4;
        JVal** nk = (JVal**)realloc(o->keys, (size_t)nc * sizeof(JVal*));
        if (!nk) return -1;
        o->keys = nk;
        if (o->kind == J_OBJ) {
            JVal** nv = (JVal**)realloc(o->vals, (size_t)nc * sizeof(JVal*));
            if (!nv) return -1;
            o->vals = nv;
        }
        o->cap = nc;
    }
    o->keys[o->n] = (o->kind == J_OBJ) ? key : val;
    if (o->vals) o->vals[o->n] = val;
    o->n++;
    return 0;
}

/* ---- parser (recursive descent; sufficient for v3 metadata) --------------- */

typedef struct { char* p; char* end; int err; } JParse;

static void jskip(JParse* s) {
    while (s->p < s->end) {
        char c = *s->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') s->p++;
        else break;
    }
}

static char* jstring_raw(JParse* s) {
    if (s->p >= s->end || *s->p != '"') { s->err = 1; return NULL; }
    s->p++;
    size_t cap = 16, n = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) { s->err = 1; return NULL; }
    while (s->p < s->end && *s->p != '"') {
        char c = *s->p++;
        if (c == '\\') {
            if (s->p >= s->end) break;
            char e = *s->p++;
            switch (e) {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case 'u': {
                    /* v3 metadata fields this library reads are ASCII;
                     * non-ASCII escapes are rejected rather than
                     * encoding-guessed (dimension_names etc. are
                     * optional). */
                    if (s->p + 4 > s->end) { s->err = 1; free(buf); return NULL; }
                    int code = 0;
                    for (int i = 0; i < 4; ++i) {
                        char h = s->p[i];
                        code *= 16;
                        if (h >= '0' && h <= '9') code += h - '0';
                        else if (h >= 'a' && h <= 'f') code += 10 + h - 'a';
                        else if (h >= 'A' && h <= 'F') code += 10 + h - 'A';
                        else { s->err = 1; free(buf); return NULL; }
                    }
                    s->p += 4;
                    if (code < 0x80) c = (char)code;
                    else { s->err = 1; free(buf); return NULL; }
                    break;
                }
                default: s->err = 1; free(buf); return NULL;
            }
        }
        if (n + 1 >= cap) {
            cap *= 2;
            char* nb = (char*)realloc(buf, cap);
            if (!nb) { s->err = 1; free(buf); return NULL; }
            buf = nb;
        }
        buf[n++] = c;
    }
    if (s->p >= s->end) { s->err = 1; free(buf); return NULL; }
    s->p++; /* closing quote */
    buf[n] = '\0';
    return buf;
}

static JVal* jvalue(JParse* s);

static JVal* jobject(JParse* s) {
    s->p++; /* '{' */
    JVal* o = jv_new(J_OBJ);
    if (!o) { s->err = 1; return NULL; }
    o->vals = (JVal**)calloc(1, sizeof(JVal*));
    jskip(s);
    if (s->p < s->end && *s->p == '}') { s->p++; return o; }
    for (;;) {
        jskip(s);
        char* key = jstring_raw(s);
        if (s->err) { jv_free(o); return NULL; }
        jskip(s);
        if (s->p >= s->end || *s->p != ':') {
            s->err = 1;
            free(key);
            jv_free(o);
            return NULL;
        }
        s->p++;
        JVal* val = jvalue(s);
        if (s->err) { free(key); jv_free(o); return NULL; }
        JVal* kv = jv_new(J_STR);
        if (!kv) { jv_free(val); jv_free(o); s->err = 1; return NULL; }
        kv->str = key;
        if (jv_push(o, kv, val) < 0) { jv_free(kv); jv_free(val); jv_free(o); return NULL; }
        jskip(s);
        if (s->p < s->end && *s->p == ',') { s->p++; continue; }
        if (s->p < s->end && *s->p == '}') { s->p++; return o; }
        s->err = 1;
        jv_free(o);
        return NULL;
    }
}

static JVal* jarray(JParse* s) {
    s->p++; /* '[' */
    JVal* a = jv_new(J_ARR);
    if (!a) { s->err = 1; return NULL; }
    jskip(s);
    if (s->p < s->end && *s->p == ']') { s->p++; return a; }
    for (;;) {
        JVal* v = jvalue(s);
        if (s->err) { jv_free(a); return NULL; }
        if (jv_push(a, NULL, v) < 0) { jv_free(v); jv_free(a); return NULL; }
        jskip(s);
        if (s->p < s->end && *s->p == ',') { s->p++; continue; }
        if (s->p < s->end && *s->p == ']') { s->p++; return a; }
        s->err = 1;
        jv_free(a);
        return NULL;
    }
}

static JVal* jvalue(JParse* s) {
    jskip(s);
    if (s->p >= s->end) { s->err = 1; return NULL; }
    char c = *s->p;
    if (c == '{') return jobject(s);
    if (c == '[') return jarray(s);
    if (c == '"') {
        char* str = jstring_raw(s);
        if (s->err) return NULL;
        JVal* v = jv_new(J_STR);
        if (!v) { free(str); s->err = 1; return NULL; }
        v->str = str;
        return v;
    }
    if (c == 't' && s->end - s->p >= 4 && memcmp(s->p, "true", 4) == 0) {
        s->p += 4;
        JVal* v = jv_new(J_BOOL);
        v->b = 1;
        return v;
    }
    if (c == 'f' && s->end - s->p >= 5 && memcmp(s->p, "false", 5) == 0) {
        s->p += 5;
        JVal* v = jv_new(J_BOOL);
        v->b = 0;
        return v;
    }
    if (c == 'n' && s->end - s->p >= 4 && memcmp(s->p, "null", 4) == 0) {
        s->p += 4;
        return jv_new(J_NULL);
    }
    /* number (Zarr's NaN/Infinity fills are JSON strings, so plain JSON
     * number grammar suffices here) */
    char* start = s->p;
    if (c == '-') s->p++;
    int digits = 0;
    while (s->p < s->end) {
        char d = *s->p;
        if ((d >= '0' && d <= '9') || d == '.' || d == 'e' || d == 'E' ||
            d == '+' || d == '-') {
            if (d >= '0' && d <= '9') digits++;
            s->p++;
        } else break;
    }
    if (!digits) { s->err = 1; return NULL; }
    char saved = *s->p;
    *s->p = '\0';
    JVal* v = jv_new(J_NUM);
    if (!v) { s->err = 1; return NULL; }
    v->num = strtod(start, NULL);
    *s->p = saved; /* restore the terminator character */
    return v;
}

static JVal* jparse(const char* text, size_t len) {
    char* copy = (char*)malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, text, len);
    copy[len] = '\0';
    JParse s = { copy, copy + (int)len, 0 };
    JVal* v = jvalue(&s);
    int err = s.err;
    jskip(&s);
    if (!err && s.p != s.end) err = 1; /* trailing garbage */
    free(copy);
    if (err) { jv_free(v); return NULL; }
    return v;
}

static JVal* jget(JVal* obj, const char* key) {
    if (!obj || obj->kind != J_OBJ) return NULL;
    for (int i = 0; i < obj->n; ++i)
        if (obj->keys[i]->kind == J_STR && strcmp(obj->keys[i]->str, key) == 0)
            return obj->vals[i];
    return NULL;
}

/* ---- writer (mimics zarr-python's 2-space pretty print) ------------------ */

typedef struct { char* buf; size_t n, cap; } JOut;

static int jo_reserve(JOut* o, size_t extra) {
    if (o->n + extra + 1 > o->cap) {
        size_t nc = o->cap ? o->cap : 64;
        while (nc < o->n + extra + 1) nc *= 2;
        char* nb = (char*)realloc(o->buf, nc);
        if (!nb) return -1;
        o->buf = nb;
        o->cap = nc;
    }
    return 0;
}

static int jo_put(JOut* o, const char* s, size_t l) {
    if (jo_reserve(o, l) < 0) return -1;
    memcpy(o->buf + o->n, s, l);
    o->n += l;
    o->buf[o->n] = '\0';
    return 0;
}

static int jo_puts(JOut* o, const char* s) { return jo_put(o, s, strlen(s)); }

/* Shortest round-trip decimal for a double, matching Python's repr for the
 * values v3 metadata carries: integer-valued floats keep ".0" (0.0, not 0),
 * non-finite values are emitted as the STRING forms "NaN" / "Infinity" /
 * "-Infinity" by the caller. */
static int jo_f64(JOut* o, double v, int is_int) {
    char tmp[64];
    if (is_int) {
        snprintf(tmp, sizeof tmp, "%lld", (long long)v);
    } else {
        int prec = 17;
        for (int p = 1; p <= 17; ++p) {
            snprintf(tmp, sizeof tmp, "%.*g", p, v);
            if (strtod(tmp, NULL) == v) { prec = p; break; }
        }
        (void)prec;
        if (!is_int && !strpbrk(tmp, ".eE")) {
            size_t l = strlen(tmp);
            if (l + 2 < sizeof tmp) {
                tmp[l] = '.';
                tmp[l + 1] = '0';
                tmp[l + 2] = '\0';
            }
        }
    }
    return jo_puts(o, tmp);
}

static int jwrite_val(JOut* o, const JVal* v, int depth, int pretty);

static int jwrite_indent(JOut* o, int depth) {
    for (int i = 0; i < depth; ++i)
        if (jo_puts(o, "  ") < 0) return -1;
    return 0;
}

static int jwrite_val(JOut* o, const JVal* v, int depth, int pretty) {
    switch (v->kind) {
        case J_NULL: return jo_puts(o, "null");
        case J_BOOL: return jo_puts(o, v->b ? "true" : "false");
        case J_NUM:  return jo_f64(o, v->num, v->is_int);
        case J_STR: {
            if (jo_puts(o, "\"") < 0) return -1;
            for (const char* c = v->str; *c; ++c) {
                char ch = *c;
                if (ch == '"' || ch == '\\') {
                    if (jo_puts(o, "\\") < 0) return -1;
                    if (jo_put(o, &ch, 1) < 0) return -1;
                } else if (ch == '\n') {
                    if (jo_puts(o, "\\n") < 0) return -1;
                } else if (ch == '\t') {
                    if (jo_puts(o, "\\t") < 0) return -1;
                } else if (ch == '\r') {
                    if (jo_puts(o, "\\r") < 0) return -1;
                }
                else if ((unsigned char)ch < 0x20) {
                    char esc[8];
                    snprintf(esc, sizeof esc, "\\u%04x", (unsigned char)ch);
                    if (jo_puts(o, esc) < 0) return -1;
                } else if (jo_put(o, &ch, 1) < 0) return -1;
            }
            return jo_puts(o, "\"");
        }
        case J_ARR: {
            if (!v->n) return jo_puts(o, "[]");
            if (jo_puts(o, "[") < 0) return -1;
            for (int i = 0; i < v->n; ++i) {
                if (pretty) {
                    if (jo_puts(o, "\n") < 0) return -1;
                    if (jwrite_indent(o, depth + 1) < 0) return -1;
                } else if (i) {
                    if (jo_puts(o, ", ") < 0) return -1;
                }
                if (jwrite_val(o, v->keys[i], depth + 1, pretty) < 0) return -1;
                if (i + 1 < v->n && jo_puts(o, ",") < 0) return -1;
            }
            if (pretty) {
                if (jo_puts(o, "\n") < 0) return -1;
                if (jwrite_indent(o, depth) < 0) return -1;
            }
            return jo_puts(o, "]");
        }
        case J_OBJ: {
            if (!v->n) return jo_puts(o, "{}");
            if (jo_puts(o, "{") < 0) return -1;
            for (int i = 0; i < v->n; ++i) {
                if (pretty) {
                    if (jo_puts(o, "\n") < 0) return -1;
                    if (jwrite_indent(o, depth + 1) < 0) return -1;
                } else if (i) {
                    if (jo_puts(o, ", ") < 0) return -1;
                }
                if (jwrite_val(o, v->keys[i], depth + 1, pretty) < 0) return -1;
                if (jo_puts(o, pretty ? ": " : ":") < 0) return -1;
                if (jwrite_val(o, v->vals[i], depth + 1, pretty) < 0) return -1;
                if (i + 1 < v->n && jo_puts(o, ",") < 0) return -1;
            }
            if (pretty) {
                if (jo_puts(o, "\n") < 0) return -1;
                if (jwrite_indent(o, depth) < 0) return -1;
            }
            return jo_puts(o, "}");
        }
    }
    return -1;
}

/* ========================================================================= */
/* array handle                                                              */
/* ========================================================================= */

struct RCRE_ZarrArray {
    char*      root;          /* full array directory (<root>/<name>) */
    size_t     ndim;
    size_t     shape[8];
    size_t     chunk[8];
    size_t     nchunks[8];    /* per-dim chunk counts (ceil) */
    size_t     nchunks_total; /* product */
    size_t     chunk_elems;   /* product of chunk dims */
    size_t     chunk_bytes;   /* chunk_elems * itemsize (identity form) */
    int        dt;            /* RCRE_ZarrDType */
    size_t     itemsize;
    double     fill_f64;      /* fill for float dtypes */
    int64_t    fill_int;      /* fill for int/bool dtypes */
    char       endian;        /* 'l' / 'b' (storage endianness) */
    ZarrCodec* codecs[4];     /* v3 list order (= write order) */
    int        ncodecs;
    int        has_zstd;
    int        zstd_level;
};

/* ---- small helpers -------------------------------------------------------- */

static int path_join(char* out, size_t cap, const char* a, const char* b) {
    int n = snprintf(out, cap, "%s/%s", a, b);
    return (n > 0 && (size_t)n < cap) ? 0 : -1;
}

/* mkdir -p for chunk directories. */
static int mkdirs(const char* path) {
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char* p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) < 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) < 0 && errno != EEXIST) return -1;
    return 0;
}

/* Chunk file path for cidx: <root>/c/<i>/<j>/... (v3 default encoding,
 * separator "/"). */
static int chunk_path(char* out, size_t cap, const RCRE_ZarrArray* arr,
                      const size_t* cidx) {
    size_t off = 0;
    int n = snprintf(out + off, cap - off, "%s/c", arr->root);
    if (n < 0 || (size_t)n >= cap - off) return -1;
    off += (size_t)n;
    for (size_t d = 0; d < arr->ndim; ++d) {
        n = snprintf(out + off, cap - off, "/%zu", cidx[d]);
        if (n < 0 || (size_t)n >= cap - off) return -1;
        off += (size_t)n;
    }
    return 0;
}

/* Fill a full chunk buffer with the array's fill value. */
static void fill_chunk(void* buf, const RCRE_ZarrArray* arr) {
    const ZarrDType* t = &ZDT[arr->dt];
    const size_t e = arr->chunk_elems;
    if (t->cls == 1) {
        if (arr->dt == RCRE_ZARR_F64) {
            double* p = (double*)buf;
            for (size_t i = 0; i < e; ++i) p[i] = arr->fill_f64;
        } else if (arr->dt == RCRE_ZARR_F32) {
            float* p = (float*)buf;
            float f = (float)arr->fill_f64;
            for (size_t i = 0; i < e; ++i) p[i] = f;
        } else { /* f16 */
            uint16_t bits = f64_to_f16(arr->fill_f64);
            uint16_t* p = (uint16_t*)buf;
            for (size_t i = 0; i < e; ++i) p[i] = bits;
        }
    } else if (t->cls == -1) {
        memset(buf, arr->fill_int ? 1 : 0, e);
    } else {
        switch (t->itemsize) {
            case 1: memset(buf, (int)(uint8_t)arr->fill_int, e); break;
            case 2: {
                int16_t* p = (int16_t*)buf;
                for (size_t i = 0; i < e; ++i) p[i] = (int16_t)arr->fill_int;
                break;
            }
            case 4: {
                int32_t* p = (int32_t*)buf;
                for (size_t i = 0; i < e; ++i) p[i] = (int32_t)arr->fill_int;
                break;
            }
            default: {
                int64_t* p = (int64_t*)buf;
                for (size_t i = 0; i < e; ++i) p[i] = arr->fill_int;
            }
        }
        /* uint dtypes share the int storage layout (same bit pattern) */
    }
}

/* Storage -> native endianness (no-op when they match). */
static void bytes_to_native(void* buf, size_t elems, size_t itemsize,
                            int storage_big) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    const int native_big = 0;
#else
    const int native_big = 1;
#endif
    if (storage_big == native_big || itemsize < 2) return;
    uint8_t* p = (uint8_t*)buf;
    for (size_t i = 0; i < elems; ++i) {
        uint8_t* e = p + i * itemsize;
        for (size_t a = 0, b = itemsize - 1; a < b; ++a, --b) {
            uint8_t tmp = e[a];
            e[a] = e[b];
            e[b] = tmp;
        }
    }
}

/* ---- file I/O -------------------------------------------------------------- */

static int read_file(const char* path, uint8_t** out, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return RCRE_ERR_IO;
    if (fseek(f, 0, SEEK_END) < 0) { fclose(f); return RCRE_ERR_IO; }
    long len = ftell(f);
    if (len < 0) { fclose(f); return RCRE_ERR_IO; }
    if (fseek(f, 0, SEEK_SET) < 0) { fclose(f); return RCRE_ERR_IO; }
    uint8_t* buf = (uint8_t*)malloc(len ? (size_t)len : 1);
    if (!buf) { fclose(f); return RCRE_ERR_OUT_OF_MEMORY; }
    size_t rd = len ? fread(buf, 1, (size_t)len, f) : 0;
    fclose(f);
    if (rd != (size_t)len) { free(buf); return RCRE_ERR_IO; }
    *out = buf;
    *out_len = (size_t)len;
    return RCRE_OK;
}

/* ========================================================================= */
/* metadata: build / write / parse                                           */
/* ========================================================================= */

/* Append key/value to an object (error-propagating helper). */
static int obj_put(JVal* obj, const char* key, JVal* val) {
    JVal* kv = jv_new(J_STR);
    if (!kv || !(kv->str = strdup(key))) {
        jv_free(kv);
        jv_free(val);
        return -1;
    }
    if (jv_push(obj, kv, val) < 0) {
        jv_free(kv);
        jv_free(val);
        return -1;
    }
    return 0;
}

static JVal* jnum(double v) {
    JVal* n = jv_new(J_NUM);
    if (n) n->num = v;
    return n;
}

/* JSON-integer number (shape/chunk dims, zarr_format, codec level, ...) */
static JVal* jnum_int(long long v) {
    JVal* n = jv_new(J_NUM);
    if (n) {
        n->num = (double)v;
        n->is_int = 1;
    }
    return n;
}

static JVal* jstr(const char* s) {
    JVal* v = jv_new(J_STR);
    if (v && !(v->str = strdup(s))) {
        jv_free(v);
        v = NULL;
    }
    return v;
}

static JVal* meta_build(const RCRE_ZarrArray* arr, int with_zstd, int zstd_level) {
    JVal* root = jv_new(J_OBJ);
    if (!root || !(root->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
        jv_free(root);
        return NULL;
    }
    /* shape */
    {
        JVal* a = jv_new(J_ARR);
        for (size_t d = 0; a && d < arr->ndim; ++d) {
            JVal* n = jnum_int((long long)arr->shape[d]);
            if (!n || jv_push(a, NULL, n) < 0) {
                if (n) jv_free(n);
                jv_free(a);
                a = NULL;
            }
        }
        if (!a || obj_put(root, "shape", a) < 0) return NULL;
    }
    /* data_type */
    if (obj_put(root, "data_type", jstr(ZDT[arr->dt].name)) < 0) return NULL;
    /* chunk_grid */
    {
        JVal* g = jv_new(J_OBJ);
        if (!g || !(g->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
            jv_free(g);
            return NULL;
        }
        JVal* cf = jv_new(J_OBJ);
        if (!cf || !(cf->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
            jv_free(cf);
            jv_free(g);
            return NULL;
        }
        JVal* a = jv_new(J_ARR);
        for (size_t d = 0; a && d < arr->ndim; ++d) {
            JVal* n = jnum_int((long long)arr->chunk[d]);
            if (!n || jv_push(a, NULL, n) < 0) {
                if (n) jv_free(n);
                jv_free(a);
                a = NULL;
            }
        }
        if (!a || obj_put(cf, "chunk_shape", a) < 0 ||
            obj_put(g, "name", jstr("regular")) < 0 ||
            obj_put(g, "configuration", cf) < 0 ||
            obj_put(root, "chunk_grid", g) < 0)
            return NULL;
    }
    /* chunk_key_encoding */
    {
        JVal* g = jv_new(J_OBJ);
        if (!g || !(g->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
            jv_free(g);
            return NULL;
        }
        JVal* cf = jv_new(J_OBJ);
        if (!cf || !(cf->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
            jv_free(cf);
            jv_free(g);
            return NULL;
        }
        if (obj_put(cf, "separator", jstr("/")) < 0 ||
            obj_put(g, "name", jstr("default")) < 0 ||
            obj_put(g, "configuration", cf) < 0 ||
            obj_put(root, "chunk_key_encoding", g) < 0)
            return NULL;
    }
    /* fill_value (v3: NaN/±Infinity are the STRING forms) */
    {
        JVal* fv = NULL;
        const ZarrDType* t = &ZDT[arr->dt];
        if (t->cls == 1) {
            if (isnan(arr->fill_f64)) fv = jstr("NaN");
            else if (isinf(arr->fill_f64))
                fv = jstr(arr->fill_f64 > 0 ? "Infinity" : "-Infinity");
            else fv = jnum(arr->fill_f64);
        } else if (t->cls == -1) {
            JVal* b = jv_new(J_BOOL);
            if (b) b->b = arr->fill_int ? 1 : 0;
            fv = b;
        } else {
            fv = jnum((double)arr->fill_int);
        }
        if (!fv || obj_put(root, "fill_value", fv) < 0) return NULL;
    }
    /* codecs (v3 list order = write order) */
    {
        JVal* a = jv_new(J_ARR);
        if (!a) return NULL;
        /* bytes codec */
        {
            JVal* c = jv_new(J_OBJ);
            if (!c || !(c->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
                jv_free(c);
                jv_free(a);
                return NULL;
            }
            JVal* cf = jv_new(J_OBJ);
            if (!cf || !(cf->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
                jv_free(cf);
                jv_free(c);
                jv_free(a);
                return NULL;
            }
            if (obj_put(cf, "endian", jstr("little")) < 0 ||
                obj_put(c, "name", jstr("bytes")) < 0 ||
                obj_put(c, "configuration", cf) < 0 ||
                jv_push(a, NULL, c) < 0) {
                jv_free(a);
                return NULL;
            }
        }
        if (with_zstd) {
            JVal* c = jv_new(J_OBJ);
            if (!c || !(c->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
                jv_free(c);
                jv_free(a);
                return NULL;
            }
            JVal* cf = jv_new(J_OBJ);
            if (!cf || !(cf->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
                jv_free(cf);
                jv_free(c);
                jv_free(a);
                return NULL;
            }
            JVal* lv = jnum_int(zstd_level);
            JVal* ck = jv_new(J_BOOL);
            if (ck) ck->b = 0;
            if (!lv || !ck || obj_put(cf, "level", lv) < 0 ||
                obj_put(cf, "checksum", ck) < 0 ||
                obj_put(c, "name", jstr("zstd")) < 0 ||
                obj_put(c, "configuration", cf) < 0 ||
                jv_push(a, NULL, c) < 0) {
                jv_free(a);
                return NULL;
            }
        }
        if (obj_put(root, "codecs", a) < 0) return NULL;
    }
    /* attributes */
    {
        JVal* a = jv_new(J_OBJ);
        if (!a || !(a->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
            jv_free(a);
            return NULL;
        }
        if (obj_put(root, "attributes", a) < 0) return NULL;
    }
    /* zarr_format, node_type, storage_transformers */
    if (obj_put(root, "zarr_format", jnum_int(3)) < 0) return NULL;
    if (obj_put(root, "node_type", jstr("array")) < 0) return NULL;
    {
        JVal* a = jv_new(J_ARR);
        if (!a || obj_put(root, "storage_transformers", a) < 0) return NULL;
    }
    return root;
}

static int meta_write(const char* path, const JVal* root) {
    JOut o = {0};
    if (jwrite_val(&o, root, 0, 1) < 0) { free(o.buf); return RCRE_ERR_INVALID_ARG; }
    if (jo_puts(&o, "\n") < 0) { free(o.buf); return RCRE_ERR_INVALID_ARG; }
    int rc = rcre_io_write_atomic(path, o.buf, o.n);
    free(o.buf);
    return rc;
}

/* Parse <arr>/zarr.json into the handle. */
static int meta_parse(RCRE_ZarrArray* arr, const char* path) {
    uint8_t* raw;
    size_t rawlen;
    int rc = read_file(path, &raw, &rawlen);
    if (rc != RCRE_OK) return rc;
    JVal* root = jparse((const char*)raw, rawlen);
    free(raw);
    if (!root) return RCRE_ERR_INVALID_ARG;

    JVal* shape = jget(root, "shape");
    if (!shape || shape->kind != J_ARR || shape->n < 1 || shape->n > 8) {
        jv_free(root);
        return RCRE_ERR_INVALID_ARG;
    }
    arr->ndim = (size_t)shape->n;
    for (int i = 0; i < shape->n; ++i) {
        if (shape->keys[i]->kind != J_NUM) { jv_free(root); return RCRE_ERR_INVALID_ARG; }
        arr->shape[i] = (size_t)shape->keys[i]->num;
    }
    JVal* dt = jget(root, "data_type");
    if (!dt || dt->kind != J_STR) { jv_free(root); return RCRE_ERR_INVALID_ARG; }
    arr->dt = -1;
    for (int i = 0; i < ZDT_N; ++i)
        if (strcmp(ZDT[i].name, dt->str) == 0) { arr->dt = i; break; }
    if (arr->dt < 0) { jv_free(root); return RCRE_ERR_INVALID_ARG; }
    arr->itemsize = ZDT[arr->dt].itemsize;

    JVal* grid = jget(root, "chunk_grid");
    JVal* cfg = grid ? jget(grid, "configuration") : NULL;
    JVal* cs = cfg ? jget(cfg, "chunk_shape") : NULL;
    if (!cs || cs->kind != J_ARR || cs->n != (int)arr->ndim) {
        jv_free(root);
        return RCRE_ERR_INVALID_ARG;
    }
    for (size_t d = 0; d < arr->ndim; ++d) {
        if (cs->keys[d]->kind != J_NUM) { jv_free(root); return RCRE_ERR_INVALID_ARG; }
        arr->chunk[d] = (size_t)cs->keys[d]->num;
        if (arr->chunk[d] == 0) { jv_free(root); return RCRE_ERR_INVALID_ARG; }
    }
    /* fill_value */
    JVal* fv = jget(root, "fill_value");
    if (!fv) { jv_free(root); return RCRE_ERR_INVALID_ARG; }
    const ZarrDType* t = &ZDT[arr->dt];
    if (t->cls == 1) {
        if (fv->kind == J_STR) {
            if (strcmp(fv->str, "NaN") == 0) arr->fill_f64 = NAN;
            else if (strcmp(fv->str, "Infinity") == 0) arr->fill_f64 = INFINITY;
            else if (strcmp(fv->str, "-Infinity") == 0) arr->fill_f64 = -INFINITY;
            else { jv_free(root); return RCRE_ERR_INVALID_ARG; }
        } else if (fv->kind == J_NUM) {
            arr->fill_f64 = fv->num;
        } else { jv_free(root); return RCRE_ERR_INVALID_ARG; }
    } else if (t->cls == -1) {
        arr->fill_int = (fv->kind == J_BOOL) ? (fv->b ? 1 : 0)
                                             : (fv->kind == J_NUM ? (int64_t)fv->num : 0);
    } else {
        arr->fill_int = (fv->kind == J_NUM) ? (int64_t)fv->num : 0;
    }
    /* codecs (v3 list order = write order; read = reverse) */
    JVal* codecs = jget(root, "codecs");
    if (!codecs || codecs->kind != J_ARR || codecs->n < 1 || codecs->n > 4) {
        jv_free(root);
        return RCRE_ERR_INVALID_ARG;
    }
    arr->ncodecs = 0;
    arr->has_zstd = 0;
    arr->zstd_level = 0;
    arr->endian = 'l';
    for (int i = 0; i < codecs->n; ++i) {
        JVal* c = codecs->keys[i];
        JVal* nm = jget(c, "name");
        JVal* cc = jget(c, "configuration");
        if (!nm || nm->kind != J_STR) { jv_free(root); return RCRE_ERR_INVALID_ARG; }
        ZarrCodec* zc = zarr_codec_find(nm->str);
        if (!zc || !zc->registered) { jv_free(root); return RCRE_ERR_INVALID_ARG; }
        arr->codecs[arr->ncodecs++] = zc;
        if (strcmp(nm->str, "bytes") == 0) {
            JVal* eo = cc ? jget(cc, "endian") : NULL;
            if (eo && eo->kind == J_STR)
                arr->endian = strcmp(eo->str, "big") == 0 ? 'b' : 'l';
        } else if (strcmp(nm->str, "zstd") == 0) {
            arr->has_zstd = 1;
            JVal* lv = cc ? jget(cc, "level") : NULL;
            if (lv && lv->kind == J_NUM) arr->zstd_level = (int)lv->num;
        }
    }
    /* derived sizes */
    arr->nchunks_total = 1;
    arr->chunk_elems = 1;
    for (size_t d = 0; d < arr->ndim; ++d) {
        arr->nchunks[d] = (arr->shape[d] + arr->chunk[d] - 1) / arr->chunk[d];
        arr->nchunks_total *= arr->nchunks[d];
        arr->chunk_elems *= arr->chunk[d];
    }
    arr->chunk_bytes = arr->chunk_elems * arr->itemsize;
    jv_free(root);
    return RCRE_OK;
}

/* ========================================================================= */
/* create / open / close                                                     */
/* ========================================================================= */

RCRE_NODISCARD int rcre_zarr_create_group_root(const char* root_dir) {
    if (!root_dir) return RCRE_ERR_INVALID_ARG;
    if (mkdirs(root_dir) < 0) return RCRE_ERR_IO;
    JVal* root = jv_new(J_OBJ);
    if (!root || !(root->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
        jv_free(root);
        return RCRE_ERR_INVALID_ARG;
    }
    {
        JVal* a = jv_new(J_OBJ);
        if (!a || !(a->vals = (JVal**)calloc(1, sizeof(JVal*)))) {
            jv_free(a);
            jv_free(root);
            return RCRE_ERR_INVALID_ARG;
        }
        if (obj_put(root, "attributes", a) < 0) return RCRE_ERR_INVALID_ARG;
    }
    if (obj_put(root, "zarr_format", jnum_int(3)) < 0) return RCRE_ERR_INVALID_ARG;
    if (obj_put(root, "node_type", jstr("group")) < 0) return RCRE_ERR_INVALID_ARG;
    char path[4096];
    int rc = (path_join(path, sizeof path, root_dir, "zarr.json") < 0)
                 ? RCRE_ERR_IO
                 : meta_write(path, root);
    jv_free(root);
    return rc;
}

RCRE_NODISCARD RCRE_ZarrArray* rcre_zarr_create(const char* root_dir,
                                                const char* name,
                                                const RCRE_ZarrCreate* opt) {
    if (!root_dir || !name || !opt) return NULL;
    rcre_zarr_zstd_ensure_registered();
    if (opt->ndim < 1 || opt->ndim > 8) return NULL;
    if (!opt->shape || !opt->chunk_shape) return NULL;
    if ((int)opt->dtype < 0 || (int)opt->dtype >= ZDT_N) return NULL;
    int zstd = opt->zstd_level > 0;
    if (zstd && opt->zstd_level > 22) return NULL;
    if (zstd && !rcre_zarr_codec_available("zstd")) return NULL;

    RCRE_ZarrArray* arr = (RCRE_ZarrArray*)calloc(1, sizeof(RCRE_ZarrArray));
    if (!arr) return NULL;
    char dir[4096];
    if (path_join(dir, sizeof dir, root_dir, name) < 0) { free(arr); return NULL; }
    if (mkdirs(dir) < 0) { free(arr); return NULL; }
    arr->root = strdup(dir);
    if (!arr->root) { free(arr); return NULL; }

    arr->ndim = opt->ndim;
    for (size_t d = 0; d < opt->ndim; ++d) {
        if (opt->chunk_shape[d] == 0) { free(arr->root); free(arr); return NULL; }
        arr->shape[d] = opt->shape[d];
        arr->chunk[d] = opt->chunk_shape[d];
    }
    arr->dt = (int)opt->dtype;
    arr->itemsize = ZDT[arr->dt].itemsize;
    arr->fill_f64 = opt->fill_f64;
    arr->fill_int = opt->fill_int;
    if (opt->dtype == RCRE_ZARR_BOOL) arr->fill_int = opt->fill_bool ? 1 : 0;
    arr->endian = 'l';
    arr->nchunks_total = 1;
    arr->chunk_elems = 1;
    for (size_t d = 0; d < arr->ndim; ++d) {
        arr->nchunks[d] = (arr->shape[d] + arr->chunk[d] - 1) / arr->chunk[d];
        arr->nchunks_total *= arr->nchunks[d];
        arr->chunk_elems *= arr->chunk[d];
    }
    arr->chunk_bytes = arr->chunk_elems * arr->itemsize;
    arr->codecs[0] = zarr_codec_find("bytes");
    arr->ncodecs = 1 + zstd; /* pipeline length: bytes + optional zstd */
    arr->has_zstd = zstd;
    arr->zstd_level = zstd ? opt->zstd_level : 0;
    if (zstd) arr->codecs[1] = zarr_codec_find("zstd");

    JVal* meta = meta_build(arr, zstd, opt->zstd_level);
    if (!meta) { free(arr->root); free(arr); return NULL; }
    char path[4096];
    int ok = path_join(path, sizeof path, arr->root, "zarr.json") == 0 &&
             meta_write(path, meta) == RCRE_OK;
    jv_free(meta);
    if (!ok) { free(arr->root); free(arr); return NULL; }
    return arr;
}

RCRE_NODISCARD RCRE_ZarrArray* rcre_zarr_open(const char* root_dir,
                                              const char* name) {
    if (!root_dir || !name) return NULL;
    rcre_zarr_zstd_ensure_registered();
    char dir[4096];
    if (path_join(dir, sizeof dir, root_dir, name) < 0) return NULL;
    char path[4096];
    if (path_join(path, sizeof path, dir, "zarr.json") < 0) return NULL;
    RCRE_ZarrArray* arr = (RCRE_ZarrArray*)calloc(1, sizeof(RCRE_ZarrArray));
    if (!arr) return NULL;
    int rc = meta_parse(arr, path);
    if (rc != RCRE_OK) { free(arr); return NULL; }
    arr->root = strdup(dir);
    if (!arr->root) { free(arr); return NULL; }
    return arr;
}

void rcre_zarr_close(RCRE_ZarrArray* arr) {
    if (!arr) return;
    free(arr->root);
    free(arr);
}

/* ========================================================================= */
/* metadata accessors                                                        */
/* ========================================================================= */

int rcre_zarr_ndim(const RCRE_ZarrArray* arr) {
    return arr ? (int)arr->ndim : -1;
}
RCRE_NODISCARD int rcre_zarr_shape(const RCRE_ZarrArray* arr, size_t* out) {
    if (!arr || !out) return RCRE_ERR_INVALID_ARG;
    memcpy(out, arr->shape, arr->ndim * sizeof(size_t));
    return RCRE_OK;
}
RCRE_NODISCARD int rcre_zarr_chunk_shape(const RCRE_ZarrArray* arr, size_t* out) {
    if (!arr || !out) return RCRE_ERR_INVALID_ARG;
    memcpy(out, arr->chunk, arr->ndim * sizeof(size_t));
    return RCRE_OK;
}
RCRE_ZarrDType rcre_zarr_dtype(const RCRE_ZarrArray* arr) {
    return arr ? (RCRE_ZarrDType)arr->dt : (RCRE_ZarrDType)-1;
}
size_t rcre_zarr_itemsize(const RCRE_ZarrArray* arr) {
    return arr ? arr->itemsize : 0;
}
RCRE_NODISCARD int rcre_zarr_nchunks(const RCRE_ZarrArray* arr, size_t* out) {
    if (!arr || !out) return RCRE_ERR_INVALID_ARG;
    memcpy(out, arr->nchunks, arr->ndim * sizeof(size_t));
    return RCRE_OK;
}
RCRE_NODISCARD int rcre_zarr_fill_value(const RCRE_ZarrArray* arr, void* out) {
    if (!arr || !out) return RCRE_ERR_INVALID_ARG;
    const ZarrDType* t = &ZDT[arr->dt];
    if (t->cls == 1) {
        switch (t->itemsize) {
            case 8: *(double*)out = arr->fill_f64; break;
            case 4: *(float*)out = (float)arr->fill_f64; break;
            case 2: *(uint16_t*)out = f64_to_f16(arr->fill_f64); break;
        }
    } else if (t->cls == -1) {
        *(uint8_t*)out = arr->fill_int ? 1 : 0;
    } else {
        switch (t->itemsize) {
            case 1: *(int8_t*)out = (int8_t)arr->fill_int; break;
            case 2: *(int16_t*)out = (int16_t)arr->fill_int; break;
            case 4: *(int32_t*)out = (int32_t)arr->fill_int; break;
            default: *(int64_t*)out = arr->fill_int;
        }
    }
    return RCRE_OK;
}

/* ========================================================================= */
/* chunk codec pipeline                                                      */
/* ========================================================================= */

/* native chunk bytes -> stored bytes (v3 LIST order: bytes -> zstd). The
 * bytes codec is a no-op for our little-endian writer (big-endian storage
 * is decode-only in H5). `out` is malloc'd. */
static int chunk_encode(const RCRE_ZarrArray* arr, const uint8_t* native,
                        size_t native_len, uint8_t** out, size_t* out_len) {
    uint8_t* cur = (uint8_t*)malloc(native_len ? native_len : 1);
    if (!cur) return RCRE_ERR_OUT_OF_MEMORY;
    memcpy(cur, native, native_len);
    size_t len = native_len;
    for (int i = 0; i < arr->ncodecs; ++i) {
        ZarrCodec* c = arr->codecs[i];
        if (strcmp(c->name, "bytes") == 0) continue;
        if (strcmp(c->name, "zstd") != 0) { free(cur); return RCRE_ERR_INVALID_ARG; }
        uint8_t* next = NULL;
        size_t next_len = 0;
        if (c->encode(cur, len, arr->zstd_level, &next, &next_len) != 0) {
            free(cur);
            return RCRE_ERR_INVALID_ARG;
        }
        free(cur);
        cur = next;
        len = next_len;
    }
    *out = cur;
    *out_len = len;
    return RCRE_OK;
}

/* stored bytes -> native chunk bytes (REVERSE v3 list order). */
static int chunk_decode(const RCRE_ZarrArray* arr, const uint8_t* stored,
                        size_t stored_len, uint8_t* native, size_t native_cap) {
    uint8_t* cur = (uint8_t*)malloc(stored_len ? stored_len : 1);
    if (!cur) return RCRE_ERR_OUT_OF_MEMORY;
    memcpy(cur, stored, stored_len);
    size_t len = stored_len;
    int rc = RCRE_OK;
    for (int i = arr->ncodecs - 1; i >= 0; --i) {
        ZarrCodec* c = arr->codecs[i];
        if (strcmp(c->name, "bytes") == 0) {
            bytes_to_native(cur, len / arr->itemsize, arr->itemsize,
                            arr->endian == 'b');
            continue;
        }
        if (strcmp(c->name, "zstd") != 0) { rc = RCRE_ERR_INVALID_ARG; break; }
        uint8_t* next = (uint8_t*)malloc(native_cap ? native_cap : 1);
        if (!next) { rc = RCRE_ERR_OUT_OF_MEMORY; break; }
        if (c->decode(cur, len, next, native_cap) != 0) {
            free(next);
            rc = RCRE_ERR_INVALID_ARG;
            break;
        }
        free(cur);
        cur = next;
        len = native_cap;
    }
    if (rc == RCRE_OK && len <= native_cap) memcpy(native, cur, len);
    free(cur);
    return rc;
}

/* ========================================================================= */
/* chunk I/O                                                                 */
/* ========================================================================= */

RCRE_NODISCARD int rcre_zarr_read_chunk(const RCRE_ZarrArray* arr,
                                        const size_t* cidx, void* out) {
    if (!arr || !cidx || !out) return RCRE_ERR_INVALID_ARG;
    for (size_t d = 0; d < arr->ndim; ++d)
        if (cidx[d] >= arr->nchunks[d]) return RCRE_ERR_INVALID_ARG;
    char path[4096];
    if (chunk_path(path, sizeof path, arr, cidx) < 0) return RCRE_ERR_INVALID_ARG;
    struct stat st;
    if (stat(path, &st) != 0) {
        if (errno == ENOENT) {
            /* missing chunk file = fill value (v3 semantics) */
            fill_chunk(out, arr);
            return RCRE_OK;
        }
        return RCRE_ERR_IO;
    }
    uint8_t* stored;
    size_t stored_len;
    int rc = read_file(path, &stored, &stored_len);
    if (rc != RCRE_OK) return rc;
    rc = chunk_decode(arr, stored, stored_len, (uint8_t*)out, arr->chunk_bytes);
    free(stored);
    return rc;
}

RCRE_NODISCARD int rcre_zarr_write_chunk(const RCRE_ZarrArray* arr,
                                         const size_t* cidx, const void* in) {
    if (!arr || !cidx || !in) return RCRE_ERR_INVALID_ARG;
    for (size_t d = 0; d < arr->ndim; ++d)
        if (cidx[d] >= arr->nchunks[d]) return RCRE_ERR_INVALID_ARG;
    char path[4096];
    if (chunk_path(path, sizeof path, arr, cidx) < 0) return RCRE_ERR_INVALID_ARG;
    /* ensure the directory tree exists */
    char dirp[4096];
    snprintf(dirp, sizeof dirp, "%s", path);
    char* slash = strrchr(dirp, '/');
    if (slash) {
        *slash = '\0';
        if (mkdirs(dirp) < 0) return RCRE_ERR_IO;
    }
    uint8_t* stored;
    size_t stored_len;
    int rc = chunk_encode(arr, (const uint8_t*)in, arr->chunk_bytes, &stored,
                          &stored_len);
    if (rc != RCRE_OK) return rc;
    rc = rcre_io_write_atomic(path, stored, stored_len);
    free(stored);
    return rc;
}

RCRE_NODISCARD int rcre_zarr_chunk_exists(const RCRE_ZarrArray* arr,
                                          const size_t* cidx) {
    if (!arr || !cidx) return -1;
    for (size_t d = 0; d < arr->ndim; ++d)
        if (cidx[d] >= arr->nchunks[d]) return -1;
    char path[4096];
    if (chunk_path(path, sizeof path, arr, cidx) < 0) return -1;
    struct stat st;
    if (stat(path, &st) == 0) return 1;
    return 0;
}

/* ========================================================================= */
/* array I/O (one task per chunk through the slab scheduler)                 */
/* ========================================================================= */

/* Linear chunk index -> multi-index (row-major over the chunk grid). */
static void cidx_from_linear(size_t linear, const RCRE_ZarrArray* arr,
                             size_t* cidx) {
    for (int d = (int)arr->ndim - 1; d >= 0; --d) {
        cidx[d] = linear % arr->nchunks[d];
        linear /= arr->nchunks[d];
    }
}

/* Valid (in-bounds) per-dim extents for a chunk. */
static void chunk_extents(const RCRE_ZarrArray* arr, const size_t* cidx,
                          size_t* ext) {
    for (size_t d = 0; d < arr->ndim; ++d) {
        size_t lo = cidx[d] * arr->chunk[d];
        size_t hi = lo + arr->chunk[d];
        ext[d] = hi > arr->shape[d] ? arr->shape[d] - lo : arr->chunk[d];
    }
}

/* Copy the valid region of a full chunk into the row-major array buffer
 * (innermost-dim runs for the fast path). */
static void chunk_to_array(const RCRE_ZarrArray* arr, const uint8_t* chunk,
                           const size_t* cidx, uint8_t* array) {
    size_t ext[8];
    chunk_extents(arr, cidx, ext);
    size_t cstride[8], dstride[8];
    cstride[arr->ndim - 1] = arr->itemsize;
    dstride[arr->ndim - 1] = arr->itemsize;
    for (int d = (int)arr->ndim - 2; d >= 0; --d) {
        cstride[d] = cstride[d + 1] * arr->chunk[d + 1];
        dstride[d] = dstride[d + 1] * arr->shape[d + 1];
    }
    size_t idx[8] = {0};
    const size_t last = arr->ndim - 1;
    /* innermost dimension is copied as a run: its chunk origin is a
     * constant dst offset (src always starts at the chunk origin) */
    const size_t dst_base = cidx[last] * arr->chunk[last] * dstride[last];
    for (;;) {
        size_t src_off = 0, dst_off = dst_base;
        for (size_t d = 0; d < last; ++d) {
            src_off += idx[d] * cstride[d];
            dst_off += (cidx[d] * arr->chunk[d] + idx[d]) * dstride[d];
        }
        memcpy(array + dst_off, chunk + src_off, ext[last] * arr->itemsize);
        if (arr->ndim == 1) break; /* single row, done */
        int d = (int)last - 1;
        for (; d >= 0; --d) {
            idx[d]++;
            if (idx[d] < ext[d]) break;
            idx[d] = 0;
        }
        if (d < 0) break;
    }
}

/* Copy the valid region of the row-major array buffer into a full chunk
 * (inverse of chunk_to_array). */
static void array_to_chunk(const RCRE_ZarrArray* arr, uint8_t* chunk,
                           const size_t* cidx, const uint8_t* array) {
    size_t ext[8];
    chunk_extents(arr, cidx, ext);
    size_t cstride[8], dstride[8];
    cstride[arr->ndim - 1] = arr->itemsize;
    dstride[arr->ndim - 1] = arr->itemsize;
    for (int d = (int)arr->ndim - 2; d >= 0; --d) {
        cstride[d] = cstride[d + 1] * arr->chunk[d + 1];
        dstride[d] = dstride[d + 1] * arr->shape[d + 1];
    }
    size_t idx[8] = {0};
    const size_t last = arr->ndim - 1;
    /* innermost run: its array origin is a constant src offset */
    const size_t src_base = cidx[last] * arr->chunk[last] * dstride[last];
    for (;;) {
        size_t src_off = src_base, dst_off = 0;
        for (size_t d = 0; d < last; ++d) {
            src_off += (cidx[d] * arr->chunk[d] + idx[d]) * dstride[d];
            dst_off += idx[d] * cstride[d];
        }
        memcpy(chunk + dst_off, array + src_off, ext[last] * arr->itemsize);
        if (arr->ndim == 1) break;
        int d = (int)last - 1;
        for (; d >= 0; --d) {
            idx[d]++;
            if (idx[d] < ext[d]) break;
            idx[d] = 0;
        }
        if (d < 0) break;
    }
}

/* ---- read_array ----------------------------------------------------------- */

typedef struct {
    const RCRE_ZarrArray* arr;
    uint8_t*              array;
    atomic_int            err; /* first chunk read failure (atomic: work
                                  stealing may interleave tasks) */
} ReadTask;

static void read_task(void* arg, size_t task_id) {
    ReadTask* rt = (ReadTask*)arg;
    const RCRE_ZarrArray* arr = rt->arr;
    size_t cidx[8];
    cidx_from_linear(task_id, arr, cidx);
    uint8_t* chunk = (uint8_t*)malloc(arr->chunk_bytes ? arr->chunk_bytes : 1);
    if (!chunk) { atomic_store(&rt->err, RCRE_ERR_OUT_OF_MEMORY); return; }
    int rc = rcre_zarr_read_chunk(arr, cidx, chunk);
    if (rc == RCRE_OK)
        chunk_to_array(arr, chunk, cidx, rt->array);
    else if (atomic_load(&rt->err) == 0) atomic_store(&rt->err, rc);
    free(chunk);
}

RCRE_NODISCARD int rcre_zarr_read_array(const RCRE_ZarrArray* arr, void* out,
                                        RCRE_ThreadPool* pool) {
    if (!arr || !out) return RCRE_ERR_INVALID_ARG;
    if (arr->nchunks_total == 0) return RCRE_OK;
    ReadTask rt;
    rt.arr = arr;
    rt.array = (uint8_t*)out;
    atomic_init(&rt.err, 0);
    rcre_threadpool_parallel_for(pool, arr->nchunks_total, read_task, &rt);
    return atomic_load(&rt.err) != 0 ? RCRE_ERR_IO : RCRE_OK;
}

/* ---- write_array (read-modify-write per chunk) ---------------------------- */

typedef struct {
    const RCRE_ZarrArray* arr;
    const uint8_t*        array;
    atomic_int            err;
} WriteTask;

static void write_task(void* arg, size_t task_id) {
    WriteTask* wt = (WriteTask*)arg;
    const RCRE_ZarrArray* arr = wt->arr;
    size_t cidx[8];
    cidx_from_linear(task_id, arr, cidx);
    uint8_t* chunk = (uint8_t*)malloc(arr->chunk_bytes ? arr->chunk_bytes : 1);
    if (!chunk) { atomic_store(&wt->err, RCRE_ERR_OUT_OF_MEMORY); return; }
    /* start from the existing chunk (or the fill) so edge regions and
     * previously-written content of the chunk survive */
    if (rcre_zarr_read_chunk(arr, cidx, chunk) != RCRE_OK) {
        fill_chunk(chunk, arr);
    }
    array_to_chunk(arr, chunk, cidx, wt->array);
    int rc = rcre_zarr_write_chunk(arr, cidx, chunk);
    if (rc != RCRE_OK && atomic_load(&wt->err) == 0) atomic_store(&wt->err, rc);
    free(chunk);
}

RCRE_NODISCARD int rcre_zarr_write_array(const RCRE_ZarrArray* arr,
                                         const void* in, RCRE_ThreadPool* pool) {
    if (!arr || !in) return RCRE_ERR_INVALID_ARG;
    if (arr->nchunks_total == 0) return RCRE_OK;
    WriteTask wt;
    wt.arr = arr;
    wt.array = (const uint8_t*)in;
    atomic_init(&wt.err, 0);
    rcre_threadpool_parallel_for(pool, arr->nchunks_total, write_task, &wt);
    return atomic_load(&wt.err) != 0 ? RCRE_ERR_IO : RCRE_OK;
}

/* ---- write_block ------------------------------------------------------------ */

typedef struct {
    const RCRE_ZarrArray* arr;
    const size_t* lo;
    const size_t* hi;
    const uint8_t* in;
    size_t         in_stride0; /* 0 = tight logical strides */
    atomic_int     err;
} BlockTask;

static int block_touched(const RCRE_ZarrArray* arr, const size_t* cidx,
                         const size_t* lo, const size_t* hi) {
    for (size_t d = 0; d < arr->ndim; ++d) {
        size_t c0 = cidx[d] * arr->chunk[d];
        size_t c1 = c0 + arr->chunk[d];
        if (c1 <= lo[d] || c0 >= hi[d]) return 0;
    }
    return 1;
}

static void block_task(void* arg, size_t task_id) {
    BlockTask* bt = (BlockTask*)arg;
    const RCRE_ZarrArray* arr = bt->arr;
    size_t cidx[8];
    cidx_from_linear(task_id, arr, cidx);
    if (!block_touched(arr, cidx, bt->lo, bt->hi)) return;

    uint8_t* chunk = (uint8_t*)malloc(arr->chunk_bytes ? arr->chunk_bytes : 1);
    if (!chunk) { atomic_store(&bt->err, RCRE_ERR_OUT_OF_MEMORY); return; }
    if (rcre_zarr_read_chunk(arr, cidx, chunk) != RCRE_OK)
        fill_chunk(chunk, arr);

    /* local (chunk-relative) extents of the intersection */
    size_t ext[8], lo_l[8];
    for (size_t d = 0; d < arr->ndim; ++d) {
        size_t c0 = cidx[d] * arr->chunk[d];
        size_t c1 = c0 + arr->chunk[d];
        lo_l[d] = bt->lo[d] > c0 ? bt->lo[d] - c0 : 0;
        size_t hi_l = bt->hi[d] < c1 ? bt->hi[d] - c0 : arr->chunk[d];
        ext[d] = hi_l - lo_l[d];
    }
    /* in-buffer strides: d==0 uses in_stride0 (or tight), others tight over
     * the BLOCK extents */
    size_t bext[8];
    for (size_t d = 0; d < arr->ndim; ++d) bext[d] = bt->hi[d] - bt->lo[d];
    size_t in_str[8];
    in_str[arr->ndim - 1] = arr->itemsize;
    for (int d = (int)arr->ndim - 2; d >= 0; --d)
        in_str[d] = in_str[d + 1] * bext[d + 1];
    const size_t in_str0 = bt->in_stride0 ? bt->in_stride0 * arr->itemsize
                                          : in_str[0];
    /* chunk strides (full chunk) */
    size_t cstride[8];
    cstride[arr->ndim - 1] = arr->itemsize;
    for (int d = (int)arr->ndim - 2; d >= 0; --d)
        cstride[d] = cstride[d + 1] * arr->chunk[d + 1];
    /* in-buffer offset of the intersection corner, relative to bt->in
     * (the user buffer START, not the block origin): buffer start ->
     * block origin (lo*d) + block origin -> intersection start (from*d). */
    size_t in_base = 0;
    for (size_t d = 0; d < arr->ndim; ++d) {
        size_t corner = cidx[d] * arr->chunk[d];
        size_t from = corner > bt->lo[d] ? corner - bt->lo[d] : 0;
        in_base += (bt->lo[d] + from) * (d == 0 ? in_str0 : in_str[d]);
    }
    /* walk (innermost-dim fast copy). in_base already includes the
     * innermost corner offset (it was summed over ALL dims); the chunk
     * side needs the innermost intersection start explicitly. */
    size_t idx[8] = {0};
    const size_t last = arr->ndim - 1;
    const size_t c_base = lo_l[last] * cstride[last];
    for (;;) {
        size_t c_off = c_base, in_off = 0;
        for (size_t d = 0; d < last; ++d) {
            c_off += (lo_l[d] + idx[d]) * cstride[d];
            in_off += idx[d] * (d == 0 ? in_str0 : in_str[d]);
        }
        memcpy(chunk + c_off, bt->in + in_base + in_off, ext[last] * arr->itemsize);
        if (arr->ndim == 1) break;
        int d = (int)last - 1;
        for (; d >= 0; --d) {
            idx[d]++;
            if (idx[d] < ext[d]) break;
            idx[d] = 0;
        }
        if (d < 0) break;
    }
    int rc = rcre_zarr_write_chunk(arr, cidx, chunk);
    if (rc != RCRE_OK && atomic_load(&bt->err) == 0) atomic_store(&bt->err, rc);
    free(chunk);
}

RCRE_NODISCARD int rcre_zarr_write_block(const RCRE_ZarrArray* arr,
                                         const size_t* lo, const size_t* hi,
                                         const void* in, size_t in_stride0,
                                         RCRE_ThreadPool* pool) {
    if (!arr || !lo || !hi || !in) return RCRE_ERR_INVALID_ARG;
    for (size_t d = 0; d < arr->ndim; ++d) {
        if (lo[d] > arr->shape[d] || hi[d] > arr->shape[d] || lo[d] >= hi[d])
            return RCRE_ERR_INVALID_ARG;
    }
    if (arr->nchunks_total == 0) return RCRE_OK;
    BlockTask bt;
    bt.arr = arr;
    bt.lo = lo;
    bt.hi = hi;
    bt.in = (const uint8_t*)in;
    bt.in_stride0 = in_stride0;
    atomic_init(&bt.err, 0);
    rcre_threadpool_parallel_for(pool, arr->nchunks_total, block_task, &bt);
    return atomic_load(&bt.err) != 0 ? RCRE_ERR_IO : RCRE_OK;
}

/* ---- for_each_chunk --------------------------------------------------------- */

typedef struct {
    const RCRE_ZarrArray* arr;
    RCRE_ZarrChunkFn      fn;
    void*                 user;
    atomic_int            first_error;
} EachTask;

static void each_task(void* arg, size_t task_id) {
    EachTask* et = (EachTask*)arg;
    const RCRE_ZarrArray* arr = et->arr;
    size_t cidx[8];
    cidx_from_linear(task_id, arr, cidx);
    uint8_t* chunk = (uint8_t*)malloc(arr->chunk_bytes ? arr->chunk_bytes : 1);
    if (!chunk) { atomic_store(&et->first_error, RCRE_ERR_OUT_OF_MEMORY); return; }
    int rc = rcre_zarr_read_chunk(arr, cidx, chunk);
    if (rc == RCRE_OK) rc = et->fn(cidx, chunk, et->user);
    free(chunk);
    if (rc != 0) {
        int expected = 0;
        atomic_compare_exchange_strong(&et->first_error, &expected, rc);
    }
}

RCRE_NODISCARD int rcre_zarr_for_each_chunk(const RCRE_ZarrArray* arr,
                                            RCRE_ZarrChunkFn fn, void* user,
                                            RCRE_ThreadPool* pool) {
    if (!arr || !fn) return RCRE_ERR_INVALID_ARG;
    if (arr->nchunks_total == 0) return RCRE_OK;
    EachTask et;
    et.arr = arr;
    et.fn = fn;
    et.user = user;
    atomic_init(&et.first_error, 0);
    rcre_threadpool_parallel_for(pool, arr->nchunks_total, each_task, &et);
    return atomic_load(&et.first_error) ? RCRE_ERR_IO : RCRE_OK;
}

/* ---- slab-parallel reduction ------------------------------------------------ */
/* One Neumaier (float) / exact (int) partial per CHUNK (task id); partials
 * combine in fixed chunk order — bit-identical across pool sizes and
 * machines regardless of work-stealing order. */

typedef struct {
    const RCRE_ZarrArray* arr;
    double*  part_f64; /* [2 * nchunks]: per-chunk Neumaier (sum, comp) */
    int64_t* part_i64; /* [nchunks] */
    atomic_int err;
} SumTask;

static void sum_task(void* arg, size_t task_id) {
    SumTask* st = (SumTask*)arg;
    const RCRE_ZarrArray* arr = st->arr;
    size_t cidx[8];
    cidx_from_linear(task_id, arr, cidx);
    uint8_t* chunk = (uint8_t*)malloc(arr->chunk_bytes ? arr->chunk_bytes : 1);
    if (!chunk) { atomic_store(&st->err, RCRE_ERR_OUT_OF_MEMORY); return; }
    if (rcre_zarr_read_chunk(arr, cidx, chunk) != RCRE_OK) {
        if (atomic_load(&st->err) == 0)
            atomic_store(&st->err, RCRE_ERR_IO);
        free(chunk);
        return;
    }
    size_t ext[8];
    chunk_extents(arr, cidx, ext);
    const ZarrDType* t = &ZDT[arr->dt];
    if (t->cls == 1) {
        double* part = st->part_f64 + 2 * task_id;
        size_t cstride[8];
        cstride[arr->ndim - 1] = 1;
        for (int d = (int)arr->ndim - 2; d >= 0; --d)
            cstride[d] = cstride[d + 1] * arr->chunk[d + 1];
        size_t idx[8] = {0};
        const size_t last = arr->ndim - 1;
        for (;;) {
            size_t off = 0;
            for (size_t d = 0; d < last; ++d) off += idx[d] * cstride[d];
            if (arr->dt == RCRE_ZARR_F64) {
                const double* run = (const double*)chunk + off;
                for (size_t j = 0; j < ext[last]; ++j) {
                    double x = run[j];
                    double s = part[0] + x;
                    if (fabs(part[0]) >= fabs(x)) part[1] += (part[0] - s) + x;
                    else part[1] += (x - s) + part[0];
                    part[0] = s;
                }
            } else if (arr->dt == RCRE_ZARR_F32) {
                const float* run = (const float*)chunk + off;
                for (size_t j = 0; j < ext[last]; ++j) {
                    double x = (double)run[j];
                    double s = part[0] + x;
                    if (fabs(part[0]) >= fabs(x)) part[1] += (part[0] - s) + x;
                    else part[1] += (x - s) + part[0];
                    part[0] = s;
                }
            } else { /* f16 */
                const uint16_t* run = (const uint16_t*)chunk + off;
                for (size_t j = 0; j < ext[last]; ++j) {
                    double x = f16_to_f64(run[j]);
                    double s = part[0] + x;
                    if (fabs(part[0]) >= fabs(x)) part[1] += (part[0] - s) + x;
                    else part[1] += (x - s) + part[0];
                    part[0] = s;
                }
            }
            if (arr->ndim == 1) break;
            int d = (int)last - 1;
            for (; d >= 0; --d) {
                idx[d]++;
                if (idx[d] < ext[d]) break;
                idx[d] = 0;
            }
            if (d < 0) break;
        }
    } else {
        int64_t* part = st->part_i64 + task_id;
        size_t cstride[8];
        cstride[arr->ndim - 1] = 1;
        for (int d = (int)arr->ndim - 2; d >= 0; --d)
            cstride[d] = cstride[d + 1] * arr->chunk[d + 1];
        size_t idx[8] = {0};
        const size_t last = arr->ndim - 1;
        for (;;) {
            size_t off = 0;
            for (size_t d = 0; d < last; ++d) off += idx[d] * cstride[d];
            if (t->cls == -1) {
                const uint8_t* run = (const uint8_t*)chunk + off;
                for (size_t j = 0; j < ext[last]; ++j) *part += run[j];
            } else if (t->itemsize == 1) {
                const int8_t* run = (const int8_t*)chunk + off;
                for (size_t j = 0; j < ext[last]; ++j) *part += run[j];
            } else if (t->itemsize == 2) {
                const int16_t* run = (const int16_t*)chunk + off;
                for (size_t j = 0; j < ext[last]; ++j) *part += run[j];
            } else if (t->itemsize == 4) {
                const int32_t* run = (const int32_t*)chunk + off;
                for (size_t j = 0; j < ext[last]; ++j) *part += run[j];
            } else {
                const int64_t* run = (const int64_t*)chunk + off;
                for (size_t j = 0; j < ext[last]; ++j) *part += run[j];
            }
            if (arr->ndim == 1) break;
            int d = (int)last - 1;
            for (; d >= 0; --d) {
                idx[d]++;
                if (idx[d] < ext[d]) break;
                idx[d] = 0;
            }
            if (d < 0) break;
        }
    }
    free(chunk);
}

RCRE_NODISCARD int rcre_zarr_reduce_sum(const RCRE_ZarrArray* arr, void* out,
                                        RCRE_ThreadPool* pool) {
    if (!arr || !out) return RCRE_ERR_INVALID_ARG;
    if (arr->nchunks_total == 0) {
        if (ZDT[arr->dt].cls == 1) *(double*)out = 0.0;
        else *(int64_t*)out = 0;
        return RCRE_OK;
    }
    SumTask st;
    st.arr = arr;
    st.part_f64 = NULL;
    st.part_i64 = NULL;
    atomic_init(&st.err, 0);
    if (ZDT[arr->dt].cls == 1)
        st.part_f64 = (double*)calloc(arr->nchunks_total * 2, sizeof(double));
    else
        st.part_i64 = (int64_t*)calloc(arr->nchunks_total, sizeof(int64_t));
    if (!st.part_f64 && !st.part_i64) return RCRE_ERR_OUT_OF_MEMORY;
    rcre_threadpool_parallel_for(pool, arr->nchunks_total, sum_task, &st);
    if (atomic_load(&st.err) != 0) {
        free(st.part_f64);
        free(st.part_i64);
        return RCRE_ERR_IO;
    }
    if (ZDT[arr->dt].cls == 1) {
        /* combine per-chunk partials in fixed chunk order (Neumaier) */
        double s = 0.0, c = 0.0;
        for (size_t i = 0; i < arr->nchunks_total; ++i) {
            double x = st.part_f64[2 * i] + st.part_f64[2 * i + 1];
            double sum = s + x;
            if (fabs(s) >= fabs(x)) c += (s - sum) + x;
            else c += (x - sum) + s;
            s = sum;
        }
        *(double*)out = s + c;
        free(st.part_f64);
    } else {
        int64_t s = 0;
        for (size_t i = 0; i < arr->nchunks_total; ++i) s += st.part_i64[i];
        *(int64_t*)out = s;
        free(st.part_i64);
    }
    return RCRE_OK;
}

/* ========================================================================= */
/* public IEEE 754 binary16 wrappers (shared with the Zarr float16 path)     */
/* ========================================================================= */

double rcre_f16_to_f64(uint16_t bits) {
    return f16_to_f64(bits);
}

uint16_t rcre_f16_from_f64(double v) {
    return f64_to_f16(v);
}

/* ========================================================================= */
/* zstd registration (hook design — link-order immune)                       */
/* ========================================================================= */

/* The zstd codec lives in an OPTIONAL translation unit (src/core/
 * zarr_zstd.c, compiled with -DRCRE_WITH_ZSTD when libzstd is present).
 * To avoid any weak-symbol / archive-extraction subtleties, chunked.c
 * holds a function-pointer hook: the zstd TU installs its real
 * implementation via rcre_zarr_zstd_set_ensure_hook() (called from a
 * C constructor). Without the TU (including single-TU amalgamations
 * that exclude it), the hook stays NULL and ensure is a no-op, so the
 * Zarr API remains fully usable with the identity codec. */
static void (*zstd_ensure_hook)(void) = NULL;

void rcre_zarr_zstd_set_ensure_hook(void (*fn)(void)) {
    zstd_ensure_hook = fn;
}

void rcre_zarr_zstd_ensure_registered(void) {
    if (zstd_ensure_hook) zstd_ensure_hook();
}
