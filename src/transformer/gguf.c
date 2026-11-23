/**
 * @file gguf.c
 * @brief Zero-Copy GGUF v2/v3 Binary Reader & v3 Writer (ISO C23)
 */

#include "rcre_llm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t* p;
    size_t         len;
    size_t         pos;
} RCRE_GGUFCursor;

static inline bool gguf_read_bytes(RCRE_GGUFCursor* c, size_t n, const uint8_t** out) {
    if (n > c->len - c->pos) return false;
    *out = c->p + c->pos;
    c->pos += n;
    return true;
}

static inline bool gguf_read_u8(RCRE_GGUFCursor* c, uint8_t* out) {
    const uint8_t* p = nullptr;
    if (!gguf_read_bytes(c, 1, &p)) return false;
    *out = p[0];
    return true;
}

static inline bool gguf_read_u16(RCRE_GGUFCursor* c, uint16_t* out) {
    const uint8_t* p = nullptr;
    if (!gguf_read_bytes(c, 2, &p)) return false;
    *out = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return true;
}

static inline bool gguf_read_u32(RCRE_GGUFCursor* c, uint32_t* out) {
    const uint8_t* p = nullptr;
    if (!gguf_read_bytes(c, 4, &p)) return false;
    *out = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return true;
}

static inline bool gguf_read_u64(RCRE_GGUFCursor* c, uint64_t* out) {
    const uint8_t* p = nullptr;
    if (!gguf_read_bytes(c, 8, &p)) return false;
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= ((uint64_t)p[i] << (i * 8));
    *out = v;
    return true;
}

static inline bool gguf_read_str(RCRE_GGUFCursor* c, RCRE_Slice* out) {
    uint64_t slen = 0;
    if (!gguf_read_u64(c, &slen) || slen > (uint64_t)(c->len - c->pos)) return false;
    const uint8_t* ptr = nullptr;
    if (!gguf_read_bytes(c, (size_t)slen, &ptr)) return false;
    *out = rcre_slice_from_parts(ptr, (size_t)slen);
    return true;
}

static size_t gguf_scalar_type_size(RCRE_GGUFType t) {
    switch (t) {
        case RCRE_GGUF_TYPE_UINT8:
        case RCRE_GGUF_TYPE_INT8:
        case RCRE_GGUF_TYPE_BOOL:    return 1;
        case RCRE_GGUF_TYPE_UINT16:
        case RCRE_GGUF_TYPE_INT16:   return 2;
        case RCRE_GGUF_TYPE_UINT32:
        case RCRE_GGUF_TYPE_INT32:
        case RCRE_GGUF_TYPE_FLOAT32: return 4;
        case RCRE_GGUF_TYPE_UINT64:
        case RCRE_GGUF_TYPE_INT64:
        case RCRE_GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static bool gguf_skip_value(RCRE_GGUFCursor* c, RCRE_GGUFType t, int depth) {
    if (depth > 8) return false;
    size_t sz = gguf_scalar_type_size(t);
    if (sz > 0) {
        const uint8_t* dummy = nullptr;
        return gguf_read_bytes(c, sz, &dummy);
    }
    if (t == RCRE_GGUF_TYPE_STRING) {
        RCRE_Slice s;
        return gguf_read_str(c, &s);
    }
    if (t == RCRE_GGUF_TYPE_ARRAY) {
        uint32_t etype_u = 0;
        uint64_t alen = 0;
        if (!gguf_read_u32(c, &etype_u) || !gguf_read_u64(c, &alen)) return false;
        RCRE_GGUFType et = (RCRE_GGUFType)etype_u;
        size_t esz = gguf_scalar_type_size(et);
        if (esz > 0) {
            if (alen > (uint64_t)((c->len - c->pos) / esz)) return false;
            c->pos += (size_t)alen * esz;
            return true;
        }
        for (uint64_t i = 0; i < alen; ++i) {
            if (!gguf_skip_value(c, et, depth + 1)) return false;
        }
        return true;
    }
    return false;
}

static bool gguf_read_kv_entry(RCRE_GGUFCursor* c, RCRE_GGUFKV* out) {
    memset(out, 0, sizeof(*out));
    if (!gguf_read_str(c, &out->key)) return false;
    uint32_t t_u32 = 0;
    if (!gguf_read_u32(c, &t_u32)) return false;
    out->type = (RCRE_GGUFType)t_u32;

    switch (out->type) {
        case RCRE_GGUF_TYPE_UINT8: {
            uint8_t v = 0;
            if (!gguf_read_u8(c, &v)) return false;
            out->val.u64 = v;
            return true;
        }
        case RCRE_GGUF_TYPE_INT8: {
            uint8_t v = 0;
            if (!gguf_read_u8(c, &v)) return false;
            out->val.i64 = (int8_t)v;
            return true;
        }
        case RCRE_GGUF_TYPE_BOOL: {
            uint8_t v = 0;
            if (!gguf_read_u8(c, &v)) return false;
            out->val.b = (v != 0);
            return true;
        }
        case RCRE_GGUF_TYPE_UINT16: {
            uint16_t v = 0;
            if (!gguf_read_u16(c, &v)) return false;
            out->val.u64 = v;
            return true;
        }
        case RCRE_GGUF_TYPE_INT16: {
            uint16_t v = 0;
            if (!gguf_read_u16(c, &v)) return false;
            out->val.i64 = (int16_t)v;
            return true;
        }
        case RCRE_GGUF_TYPE_UINT32: {
            uint32_t v = 0;
            if (!gguf_read_u32(c, &v)) return false;
            out->val.u64 = v;
            return true;
        }
        case RCRE_GGUF_TYPE_INT32: {
            uint32_t v = 0;
            if (!gguf_read_u32(c, &v)) return false;
            out->val.i64 = (int32_t)v;
            return true;
        }
        case RCRE_GGUF_TYPE_FLOAT32: {
            uint32_t v = 0;
            if (!gguf_read_u32(c, &v)) return false;
            float f = 0.0f;
            memcpy(&f, &v, 4);
            out->val.f64 = (double)f;
            return true;
        }
        case RCRE_GGUF_TYPE_UINT64: {
            if (!gguf_read_u64(c, &out->val.u64)) return false;
            return true;
        }
        case RCRE_GGUF_TYPE_INT64: {
            uint64_t v = 0;
            if (!gguf_read_u64(c, &v)) return false;
            out->val.i64 = (int64_t)v;
            return true;
        }
        case RCRE_GGUF_TYPE_FLOAT64: {
            uint64_t v = 0;
            if (!gguf_read_u64(c, &v)) return false;
            memcpy(&out->val.f64, &v, 8);
            return true;
        }
        case RCRE_GGUF_TYPE_STRING:
            return gguf_read_str(c, &out->val.str);
        case RCRE_GGUF_TYPE_ARRAY: {
            uint32_t et = 0;
            uint64_t alen = 0;
            if (!gguf_read_u32(c, &et) || !gguf_read_u64(c, &alen)) return false;
            out->val.arr.elem_type = (RCRE_GGUFType)et;
            out->val.arr.len       = alen;
            out->val.arr.raw_data  = c->p + c->pos;
            size_t esz = gguf_scalar_type_size(out->val.arr.elem_type);
            if (esz > 0) {
                if (alen > (uint64_t)((c->len - c->pos) / esz)) return false;
                c->pos += (size_t)alen * esz;
                return true;
            }
            for (uint64_t i = 0; i < alen; ++i) {
                if (!gguf_skip_value(c, out->val.arr.elem_type, 1)) return false;
            }
            return true;
        }
        default:
            return false;
    }
}

static size_t gguf_tensor_byte_size(RCRE_GGMLType t, const uint64_t dims[4], uint32_t n_dims) {
    size_t numel = 1;
    for (uint32_t i = 0; i < n_dims; ++i) {
        if (dims[i] == 0 || dims[i] > (uint64_t)SIZE_MAX / numel) return 0;
        numel *= (size_t)dims[i];
    }
    switch (t) {
        case RCRE_GGML_TYPE_F32:
        case RCRE_GGML_TYPE_I32:  return numel * 4;
        case RCRE_GGML_TYPE_F16:
        case RCRE_GGML_TYPE_BF16:
        case RCRE_GGML_TYPE_I16:  return numel * 2;
        case RCRE_GGML_TYPE_I8:   return numel;
        case RCRE_GGML_TYPE_I64:
        case RCRE_GGML_TYPE_F64:  return numel * 8;
        case RCRE_GGML_TYPE_Q8_0: return ((numel + 31U) / 32U) * sizeof(RCRE_BlockQ8_0);
        case RCRE_GGML_TYPE_Q4_0: return ((numel + 31U) / 32U) * sizeof(RCRE_BlockQ4_0);
        default: return 0;
    }
}

static int gguf_parse_mapped(RCRE_GGUFFile* g) {
    if (!g || !g->map.data || g->map.size < 24) return RCRE_ERR_IO;
    RCRE_GGUFCursor c = { .p = g->map.data, .len = g->map.size, .pos = 0 };

    uint32_t magic = 0;
    if (!gguf_read_u32(&c, &magic) || magic != RCRE_GGUF_MAGIC) return RCRE_ERR_IO;
    if (!gguf_read_u32(&c, &g->version) || (g->version != 2 && g->version != 3)) return RCRE_ERR_IO;
    if (!gguf_read_u64(&c, &g->n_tensors) || !gguf_read_u64(&c, &g->n_kv)) return RCRE_ERR_IO;
    if (g->n_tensors > 16384U || g->n_kv > 65536U) return RCRE_ERR_IO;

    g->alignment = 32U;

    if (g->n_kv > 0) {
        g->kvs = (RCRE_GGUFKV*)calloc((size_t)g->n_kv, sizeof(RCRE_GGUFKV));
        if (!g->kvs) return RCRE_ERR_OUT_OF_MEMORY;
        for (uint64_t i = 0; i < g->n_kv; ++i) {
            if (!gguf_read_kv_entry(&c, &g->kvs[i])) return RCRE_ERR_IO;
            if (rcre_slice_eq_cstr(g->kvs[i].key, "general.alignment") &&
                g->kvs[i].type == RCRE_GGUF_TYPE_UINT32 &&
                g->kvs[i].val.u64 >= 1 && (g->kvs[i].val.u64 & (g->kvs[i].val.u64 - 1)) == 0) {
                g->alignment = (uint32_t)g->kvs[i].val.u64;
            }
        }
    }

    if (g->n_tensors > 0) {
        g->tensors = (RCRE_GGUFTensorInfo*)calloc((size_t)g->n_tensors, sizeof(RCRE_GGUFTensorInfo));
        if (!g->tensors) return RCRE_ERR_OUT_OF_MEMORY;
        for (uint64_t i = 0; i < g->n_tensors; ++i) {
            RCRE_GGUFTensorInfo* ti = &g->tensors[i];
            if (!gguf_read_str(&c, &ti->name)) return RCRE_ERR_IO;
            if (!gguf_read_u32(&c, &ti->n_dims) || ti->n_dims == 0 || ti->n_dims > 4) return RCRE_ERR_IO;
            for (uint32_t d = 0; d < ti->n_dims; ++d) {
                if (!gguf_read_u64(&c, &ti->dims[d])) return RCRE_ERR_IO;
            }
            uint32_t tt = 0;
            if (!gguf_read_u32(&c, &tt) || !gguf_read_u64(&c, &ti->offset)) return RCRE_ERR_IO;
            ti->type = (RCRE_GGMLType)tt;
            ti->nbytes = gguf_tensor_byte_size(ti->type, ti->dims, ti->n_dims);
            if (ti->nbytes == 0) return RCRE_ERR_UNSUPPORTED_OP;
        }
    }

    size_t align = (size_t)g->alignment;
    g->data_offset = (c.pos + align - 1U) & ~(align - 1U);
    if (g->data_offset > g->map.size) return RCRE_ERR_IO;

    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        const RCRE_GGUFTensorInfo* ti = &g->tensors[i];
        if (ti->offset > (uint64_t)(g->map.size - g->data_offset) ||
            ti->nbytes > g->map.size - g->data_offset - (size_t)ti->offset) {
            return RCRE_ERR_IO;
        }
    }
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_gguf_open(const char* filepath, RCRE_GGUFFile* out_gguf) {
    if (!filepath || !out_gguf) return RCRE_ERR_INVALID_ARG;
    memset(out_gguf, 0, sizeof(*out_gguf));
    int rc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_SEQUENTIAL, &out_gguf->map);
    if (rc != RCRE_OK) return rc;
    rc = gguf_parse_mapped(out_gguf);
    if (rc != RCRE_OK) rcre_gguf_close(out_gguf);
    return rc;
}

RCRE_NODISCARD int rcre_gguf_open_from_buffer(const void* buf, size_t size, RCRE_GGUFFile* out_gguf) {
    if (!buf || size == 0 || !out_gguf) return RCRE_ERR_INVALID_ARG;
    memset(out_gguf, 0, sizeof(*out_gguf));
    int rc = rcre_io_mmap_from_buffer(buf, size, &out_gguf->map);
    if (rc != RCRE_OK) return rc;
    rc = gguf_parse_mapped(out_gguf);
    if (rc != RCRE_OK) rcre_gguf_close(out_gguf);
    return rc;
}

void rcre_gguf_close(RCRE_GGUFFile* gguf) {
    if (!gguf) return;
    free(gguf->kvs);
    free(gguf->tensors);
    rcre_io_mmap_close(&gguf->map);
    memset(gguf, 0, sizeof(*gguf));
}

RCRE_NODISCARD const RCRE_GGUFKV* rcre_gguf_find_kv(const RCRE_GGUFFile* gguf, const char* key) {
    if (!gguf || !key) return nullptr;
    for (uint64_t i = 0; i < gguf->n_kv; ++i) {
        if (rcre_slice_eq_cstr(gguf->kvs[i].key, key)) return &gguf->kvs[i];
    }
    return nullptr;
}

RCRE_NODISCARD bool rcre_gguf_get_u32(const RCRE_GGUFFile* gguf, const char* key, uint32_t* out_v) {
    const RCRE_GGUFKV* kv = rcre_gguf_find_kv(gguf, key);
    if (!kv || !out_v) return false;
    if (kv->type == RCRE_GGUF_TYPE_UINT32 || kv->type == RCRE_GGUF_TYPE_UINT64) {
        *out_v = (uint32_t)kv->val.u64;
        return true;
    }
    if (kv->type == RCRE_GGUF_TYPE_INT32 && kv->val.i64 >= 0) {
        *out_v = (uint32_t)kv->val.i64;
        return true;
    }
    return false;
}

RCRE_NODISCARD bool rcre_gguf_get_f32(const RCRE_GGUFFile* gguf, const char* key, float* out_v) {
    const RCRE_GGUFKV* kv = rcre_gguf_find_kv(gguf, key);
    if (!kv || !out_v) return false;
    if (kv->type == RCRE_GGUF_TYPE_FLOAT32 || kv->type == RCRE_GGUF_TYPE_FLOAT64) {
        *out_v = (float)kv->val.f64;
        return true;
    }
    return false;
}

RCRE_NODISCARD bool rcre_gguf_get_str(const RCRE_GGUFFile* gguf, const char* key, RCRE_Slice* out_s) {
    const RCRE_GGUFKV* kv = rcre_gguf_find_kv(gguf, key);
    if (!kv || !out_s || kv->type != RCRE_GGUF_TYPE_STRING) return false;
    *out_s = kv->val.str;
    return true;
}

RCRE_NODISCARD const RCRE_GGUFTensorInfo* rcre_gguf_find_tensor(const RCRE_GGUFFile* gguf, const char* name) {
    if (!gguf || !name) return nullptr;
    for (uint64_t i = 0; i < gguf->n_tensors; ++i) {
        if (rcre_slice_eq_cstr(gguf->tensors[i].name, name)) return &gguf->tensors[i];
    }
    return nullptr;
}

RCRE_NODISCARD const void* rcre_gguf_tensor_data(const RCRE_GGUFFile* gguf, const RCRE_GGUFTensorInfo* info) {
    if (!gguf || !info || !gguf->map.data) return nullptr;
    return gguf->map.data + gguf->data_offset + (size_t)info->offset;
}

static bool gguf_write_u32(FILE* f, uint32_t v) {
    uint8_t b[4] = { (uint8_t)(v & 0xFFU), (uint8_t)((v >> 8) & 0xFFU),
                     (uint8_t)((v >> 16) & 0xFFU), (uint8_t)((v >> 24) & 0xFFU) };
    return fwrite(b, 1, 4, f) == 4;
}

static bool gguf_write_u64(FILE* f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = (uint8_t)((v >> (i * 8)) & 0xFFULL);
    return fwrite(b, 1, 8, f) == 8;
}

static bool gguf_write_str(FILE* f, const char* s) {
    size_t len = s ? strlen(s) : 0;
    if (!gguf_write_u64(f, (uint64_t)len)) return false;
    return (len == 0) || fwrite(s, 1, len, f) == len;
}

RCRE_NODISCARD int rcre_gguf_save_v3(
    const char* filepath,
    const RCRE_GGUFWriteKV* kvs,
    size_t n_kvs,
    const RCRE_GGUFWriteTensor* tensors,
    size_t n_tensors
) {
    if (!filepath) return RCRE_ERR_INVALID_ARG;
    FILE* f = fopen(filepath, "wb");
    if (!f) return RCRE_ERR_IO;

    bool ok = gguf_write_u32(f, RCRE_GGUF_MAGIC) &&
              gguf_write_u32(f, 3U) &&
              gguf_write_u64(f, (uint64_t)n_tensors) &&
              gguf_write_u64(f, (uint64_t)(n_kvs + 1U));

    ok = ok && gguf_write_str(f, "general.alignment") &&
         gguf_write_u32(f, (uint32_t)RCRE_GGUF_TYPE_UINT32) &&
         gguf_write_u32(f, 32U);

    for (size_t i = 0; ok && i < n_kvs; ++i) {
        ok = gguf_write_str(f, kvs[i].key);
        if (kvs[i].is_str) {
            ok = ok && gguf_write_u32(f, (uint32_t)RCRE_GGUF_TYPE_STRING) && gguf_write_str(f, kvs[i].str_val);
        } else if (kvs[i].is_float) {
            uint32_t bits = 0;
            memcpy(&bits, &kvs[i].f32_val, 4);
            ok = ok && gguf_write_u32(f, (uint32_t)RCRE_GGUF_TYPE_FLOAT32) && gguf_write_u32(f, bits);
        } else {
            ok = ok && gguf_write_u32(f, (uint32_t)RCRE_GGUF_TYPE_UINT32) && gguf_write_u32(f, kvs[i].u32_val);
        }
    }

    uint64_t cur_off = 0;
    for (size_t i = 0; ok && i < n_tensors; ++i) {
        ok = gguf_write_str(f, tensors[i].name) &&
             gguf_write_u32(f, tensors[i].n_dims);
        for (uint32_t d = 0; ok && d < tensors[i].n_dims; ++d) {
            ok = gguf_write_u64(f, tensors[i].dims[d]);
        }
        ok = ok && gguf_write_u32(f, (uint32_t)tensors[i].type) && gguf_write_u64(f, cur_off);
        size_t padded = (tensors[i].nbytes + 31U) & ~(size_t)31U;
        cur_off += (uint64_t)padded;
    }

    long hdr_end = ftell(f);
    if (hdr_end < 0) ok = false;
    if (ok) {
        size_t pos = (size_t)hdr_end;
        size_t aligned = (pos + 31U) & ~(size_t)31U;
        uint8_t zero[32] = {0};
        if (aligned > pos && fwrite(zero, 1, aligned - pos, f) != aligned - pos) ok = false;
        for (size_t i = 0; ok && i < n_tensors; ++i) {
            if (tensors[i].nbytes > 0 && fwrite(tensors[i].data, 1, tensors[i].nbytes, f) != tensors[i].nbytes) {
                ok = false;
                break;
            }
            size_t padded = (tensors[i].nbytes + 31U) & ~(size_t)31U;
            if (padded > tensors[i].nbytes && fwrite(zero, 1, padded - tensors[i].nbytes, f) != padded - tensors[i].nbytes) {
                ok = false;
                break;
            }
        }
    }

    fclose(f);
    return ok ? RCRE_OK : RCRE_ERR_IO;
}
