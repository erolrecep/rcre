/**
 * @file codec.c
 * @brief Zero-Dependency Raster Image Codecs: PPM, BMP, PNG (zlib/DEFLATE), Baseline JPEG (ISO C23)
 */

#include "rcre_codec.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t codec_crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b) {
            uint32_t mask = (uint32_t)(-(int32_t)(crc & 1U));
            crc = (crc >> 1U) ^ (0xEDB88320U & mask);
        }
    }
    return ~crc;
}

static uint32_t codec_adler32(const uint8_t* data, size_t len) {
    uint32_t s1 = 1, s2 = 0;
    for (size_t i = 0; i < len; ++i) {
        s1 = (s1 + data[i]) % 65521U;
        s2 = (s2 + s1) % 65521U;
    }
    return (s2 << 16) | s1;
}

static inline uint32_t codec_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static inline void codec_wbe32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)((v >> 24) & 0xFFU);
    p[1] = (uint8_t)((v >> 16) & 0xFFU);
    p[2] = (uint8_t)((v >> 8) & 0xFFU);
    p[3] = (uint8_t)(v & 0xFFU);
}

static inline uint16_t codec_le16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline uint32_t codec_le32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* ========================================================================= */
/* 1. PPM P6 & BMP 24-bit BI_RGB                                             */
/* ========================================================================= */

RCRE_NODISCARD int rcre_codec_load_ppm(const char* filepath, RCRE_Arena* arena, RCRE_Tensor* out_img) {
    if (!filepath || !arena || !out_img) return RCRE_ERR_INVALID_ARG;
    RCRE_MappedFile map;
    int rc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_SEQUENTIAL, &map);
    if (rc != RCRE_OK) return rc;

    const uint8_t* p = map.data;
    size_t len = map.size;
    size_t pos = 0;
    if (len < 10 || p[0] != 'P' || p[1] != '6') { rcre_io_mmap_close(&map); return RCRE_ERR_IO; }
    pos = 2;

    uint32_t vals[3] = {0, 0, 0};
    for (int k = 0; k < 3; ++k) {
        while (pos < len) {
            if (p[pos] == '#') { while (pos < len && p[pos] != '\n') pos++; continue; }
            if (p[pos] == ' ' || p[pos] == '\n' || p[pos] == '\r' || p[pos] == '\t') { pos++; continue; }
            break;
        }
        if (pos >= len || p[pos] < '0' || p[pos] > '9') { rcre_io_mmap_close(&map); return RCRE_ERR_IO; }
        uint64_t acc = 0;
        while (pos < len && p[pos] >= '0' && p[pos] <= '9') {
            acc = acc * 10ULL + (uint64_t)(p[pos++] - '0');
            if (acc > 65535ULL) { rcre_io_mmap_close(&map); return RCRE_ERR_IO; }
        }
        vals[k] = (uint32_t)acc;
    }
    if (pos >= len || vals[0] == 0 || vals[1] == 0 || vals[2] != 255U) { rcre_io_mmap_close(&map); return RCRE_ERR_IO; }
    pos++; /* single whitespace byte before pixel stream */
    size_t need = (size_t)vals[0] * (size_t)vals[1] * 3U;
    if (len - pos < need) { rcre_io_mmap_close(&map); return RCRE_ERR_IO; }

    size_t sh[3] = { vals[1], vals[0], 3 };
    RCRE_Tensor t = rcre_tensor_create(sh, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    if (!t.data) { rcre_io_mmap_close(&map); return RCRE_ERR_OUT_OF_MEMORY; }
    memcpy(t.data, p + pos, need);
    rcre_io_mmap_close(&map);
    *out_img = t;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_codec_save_ppm(const RCRE_Tensor* img, const char* filepath) {
    if (!img || !img->data || !filepath || img->ndim != 3 || img->shape[2] != 3 || img->dtype != RCRE_DTYPE_U8) {
        return RCRE_ERR_INVALID_ARG;
    }
    char hdr[64];
    int hlen = snprintf(hdr, sizeof(hdr), "P6\n%zu %zu\n255\n", img->shape[1], img->shape[0]);
    if (hlen <= 0) return RCRE_ERR_IO;
    size_t pix = img->shape[0] * img->shape[1] * 3U;
    uint8_t* buf = (uint8_t*)malloc((size_t)hlen + pix);
    if (!buf) return RCRE_ERR_OUT_OF_MEMORY;
    memcpy(buf, hdr, (size_t)hlen);
    memcpy(buf + hlen, img->data, pix);
    int rc = rcre_io_write_atomic(filepath, buf, (size_t)hlen + pix);
    free(buf);
    return rc;
}

RCRE_NODISCARD int rcre_codec_load_bmp(const char* filepath, RCRE_Arena* arena, RCRE_Tensor* out_img) {
    if (!filepath || !arena || !out_img) return RCRE_ERR_INVALID_ARG;
    RCRE_MappedFile map;
    int rc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_SEQUENTIAL, &map);
    if (rc != RCRE_OK) return rc;

    const uint8_t* p = map.data;
    if (map.size < 54 || p[0] != 'B' || p[1] != 'M') { rcre_io_mmap_close(&map); return RCRE_ERR_IO; }
    uint32_t pix_off = codec_le32(p + 10);
    int32_t w = (int32_t)codec_le32(p + 18);
    int32_t h_raw = (int32_t)codec_le32(p + 22);
    uint16_t bpp = codec_le16(p + 28);
    uint32_t comp = codec_le32(p + 30);
    if (w <= 0 || h_raw == 0 || bpp != 24 || comp != 0) { rcre_io_mmap_close(&map); return RCRE_ERR_UNSUPPORTED_OP; }

    bool top_down = (h_raw < 0);
    size_t width = (size_t)w;
    size_t height = (size_t)(top_down ? -h_raw : h_raw);
    size_t row_stride = ((width * 3U) + 3U) & ~(size_t)3U;
    if (pix_off > map.size || row_stride * height > map.size - pix_off) {
        rcre_io_mmap_close(&map); return RCRE_ERR_IO;
    }

    size_t sh[3] = { height, width, 3 };
    RCRE_Tensor t = rcre_tensor_create(sh, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    if (!t.data) { rcre_io_mmap_close(&map); return RCRE_ERR_OUT_OF_MEMORY; }
    uint8_t* dst = (uint8_t*)t.data;
    const uint8_t* src_base = p + pix_off;

    for (size_t y = 0; y < height; ++y) {
        size_t sy = top_down ? y : (height - 1U - y);
        const uint8_t* sr = src_base + sy * row_stride;
        uint8_t* dr = dst + y * width * 3U;
        for (size_t x = 0; x < width; ++x) {
            dr[x * 3U + 0] = sr[x * 3U + 2];
            dr[x * 3U + 1] = sr[x * 3U + 1];
            dr[x * 3U + 2] = sr[x * 3U + 0];
        }
    }
    rcre_io_mmap_close(&map);
    *out_img = t;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_codec_save_bmp(const RCRE_Tensor* img, const char* filepath) {
    if (!img || !img->data || !filepath || img->ndim != 3 || img->shape[2] != 3 || img->dtype != RCRE_DTYPE_U8) {
        return RCRE_ERR_INVALID_ARG;
    }
    size_t h = img->shape[0], w = img->shape[1];
    size_t row_stride = ((w * 3U) + 3U) & ~(size_t)3U;
    size_t total = 54U + row_stride * h;
    uint8_t* buf = (uint8_t*)calloc(1, total);
    if (!buf) return RCRE_ERR_OUT_OF_MEMORY;

    buf[0] = 'B'; buf[1] = 'M';
    buf[2] = (uint8_t)(total & 0xFFU); buf[3] = (uint8_t)((total >> 8) & 0xFFU);
    buf[4] = (uint8_t)((total >> 16) & 0xFFU); buf[5] = (uint8_t)((total >> 24) & 0xFFU);
    buf[10] = 54;
    buf[14] = 40;
    buf[18] = (uint8_t)(w & 0xFFU); buf[19] = (uint8_t)((w >> 8) & 0xFFU);
    buf[22] = (uint8_t)(h & 0xFFU); buf[23] = (uint8_t)((h >> 8) & 0xFFU);
    buf[26] = 1; buf[28] = 24;

    const uint8_t* src = (const uint8_t*)img->data;
    for (size_t y = 0; y < h; ++y) {
        const uint8_t* sr = src + (h - 1U - y) * w * 3U;
        uint8_t* dr = buf + 54U + y * row_stride;
        for (size_t x = 0; x < w; ++x) {
            dr[x * 3U + 0] = sr[x * 3U + 2];
            dr[x * 3U + 1] = sr[x * 3U + 1];
            dr[x * 3U + 2] = sr[x * 3U + 0];
        }
    }
    int rc = rcre_io_write_atomic(filepath, buf, total);
    free(buf);
    return rc;
}

/* ========================================================================= */
/* 2. RFC 1950 / 1951 DEFLATE Inflate & Deflate + PNG Encoder/Decoder        */
/* ========================================================================= */

typedef struct {
    const uint8_t* p;
    size_t         len;
    size_t         byte_pos;
    uint32_t       bit_buf;
    int            bit_cnt;
} RCRE_ZBitReader;

static uint32_t codec_z_read_bits(RCRE_ZBitReader* r, int n) {
    while (r->bit_cnt < n) {
        if (r->byte_pos >= r->len) return 0;
        r->bit_buf |= ((uint32_t)r->p[r->byte_pos++]) << r->bit_cnt;
        r->bit_cnt += 8;
    }
    uint32_t v = r->bit_buf & ((1U << n) - 1U);
    r->bit_buf >>= n;
    r->bit_cnt -= n;
    return v;
}

typedef struct {
    uint16_t counts[16];
    uint16_t symbols[288];
    int      max_bits;
} RCRE_ZHuff;

static void codec_zhuff_build(RCRE_ZHuff* h, const uint8_t* lens, int n) {
    memset(h, 0, sizeof(*h));
    uint16_t offs[16] = {0};
    for (int i = 0; i < n; ++i) {
        if (lens[i] > 0 && lens[i] < 16) {
            h->counts[lens[i]]++;
            if (lens[i] > h->max_bits) h->max_bits = lens[i];
        }
    }
    uint16_t sum = 0;
    for (int b = 1; b <= 15; ++b) {
        offs[b] = sum;
        sum = (uint16_t)(sum + h->counts[b]);
    }
    for (int i = 0; i < n; ++i) {
        if (lens[i] > 0 && lens[i] < 16) {
            h->symbols[offs[lens[i]]++] = (uint16_t)i;
        }
    }
}

static int codec_zhuff_decode(RCRE_ZBitReader* r, const RCRE_ZHuff* h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= h->max_bits; ++len) {
        code |= (int)codec_z_read_bits(r, 1);
        int count = h->counts[len];
        if (code - count < first) return h->symbols[index + (code - first)];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    return -1;
}

static int codec_zlib_inflate(const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_cap, size_t* out_len) {
    if (src_len < 6 || (src[0] & 0x0FU) != 8U || (((uint32_t)src[0] << 8) | src[1]) % 31U != 0U) {
        return RCRE_ERR_IO;
    }
    RCRE_ZBitReader br = { src + 2, src_len - 2, 0, 0, 0 };
    size_t out_pos = 0;

    static const uint16_t len_base[29] = {
        3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,
        35,43,51,59,67,83,99,115,131,163,195,227,258
    };
    static const uint8_t len_extra[29] = {
        0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0
    };
    static const uint16_t dist_base[30] = {
        1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,
        1025,1537,2049,3073,4097,6145,8193,12289,16385,24577
    };
    static const uint8_t dist_extra[30] = {
        0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13
    };
    static const uint8_t cl_order[19] = {
        16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15
    };

    int bfinal = 0;
    do {
        bfinal = (int)codec_z_read_bits(&br, 1);
        int btype = (int)codec_z_read_bits(&br, 2);

        if (btype == 0) {
            br.bit_buf = 0; br.bit_cnt = 0;
            if (br.byte_pos + 4 > br.len) return RCRE_ERR_IO;
            uint16_t len = codec_le16(br.p + br.byte_pos);
            uint16_t nlen = codec_le16(br.p + br.byte_pos + 2);
            br.byte_pos += 4;
            if ((uint16_t)(len ^ 0xFFFFU) != nlen || br.byte_pos + len > br.len || out_pos + len > dst_cap) {
                return RCRE_ERR_IO;
            }
            memcpy(dst + out_pos, br.p + br.byte_pos, len);
            br.byte_pos += len;
            out_pos += len;
        } else if (btype == 1 || btype == 2) {
            RCRE_ZHuff lit_h, dist_h;
            if (btype == 1) {
                uint8_t ll[288], dl[32];
                for (int i = 0; i <= 143; ++i) ll[i] = 8;
                for (int i = 144; i <= 255; ++i) ll[i] = 9;
                for (int i = 256; i <= 279; ++i) ll[i] = 7;
                for (int i = 280; i <= 287; ++i) ll[i] = 8;
                for (int i = 0; i < 32; ++i) dl[i] = 5;
                codec_zhuff_build(&lit_h, ll, 288);
                codec_zhuff_build(&dist_h, dl, 32);
            } else {
                int hlit  = (int)codec_z_read_bits(&br, 5) + 257;
                int hdist = (int)codec_z_read_bits(&br, 5) + 1;
                int hclen = (int)codec_z_read_bits(&br, 4) + 4;
                uint8_t cl_lens[19] = {0};
                for (int i = 0; i < hclen; ++i) cl_lens[cl_order[i]] = (uint8_t)codec_z_read_bits(&br, 3);
                RCRE_ZHuff cl_h;
                codec_zhuff_build(&cl_h, cl_lens, 19);

                uint8_t all_lens[320] = {0};
                int idx = 0;
                while (idx < hlit + hdist) {
                    int sym = codec_zhuff_decode(&br, &cl_h);
                    if (sym < 0) return RCRE_ERR_IO;
                    if (sym < 16) {
                        all_lens[idx++] = (uint8_t)sym;
                    } else {
                        int rep = 0;
                        uint8_t val = 0;
                        if (sym == 16) {
                            if (idx == 0) return RCRE_ERR_IO;
                            val = all_lens[idx - 1];
                            rep = 3 + (int)codec_z_read_bits(&br, 2);
                        } else if (sym == 17) {
                            rep = 3 + (int)codec_z_read_bits(&br, 3);
                        } else {
                            rep = 11 + (int)codec_z_read_bits(&br, 7);
                        }
                        if (idx + rep > hlit + hdist) return RCRE_ERR_IO;
                        while (rep--) all_lens[idx++] = val;
                    }
                }
                codec_zhuff_build(&lit_h, all_lens, hlit);
                codec_zhuff_build(&dist_h, all_lens + hlit, hdist);
            }

            for (;;) {
                int sym = codec_zhuff_decode(&br, &lit_h);
                if (sym < 0) return RCRE_ERR_IO;
                if (sym < 256) {
                    if (out_pos >= dst_cap) return RCRE_ERR_IO;
                    dst[out_pos++] = (uint8_t)sym;
                } else if (sym == 256) {
                    break;
                } else {
                    int li = sym - 257;
                    if (li >= 29) return RCRE_ERR_IO;
                    size_t match_len = len_base[li] + codec_z_read_bits(&br, len_extra[li]);
                    int di = codec_zhuff_decode(&br, &dist_h);
                    if (di < 0 || di >= 30) return RCRE_ERR_IO;
                    size_t dist = dist_base[di] + codec_z_read_bits(&br, dist_extra[di]);
                    if (dist == 0 || dist > out_pos || out_pos + match_len > dst_cap) return RCRE_ERR_IO;
                    for (size_t k = 0; k < match_len; ++k) {
                        dst[out_pos] = dst[out_pos - dist];
                        out_pos++;
                    }
                }
            }
        } else {
            return RCRE_ERR_IO;
        }
    } while (!bfinal);

    *out_len = out_pos;
    return RCRE_OK;
}

static uint8_t codec_paeth(uint8_t a, uint8_t b, uint8_t c) {
    int p = (int)a + (int)b - (int)c;
    int pa = abs(p - (int)a), pb = abs(p - (int)b), pc = abs(p - (int)c);
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

RCRE_NODISCARD int rcre_codec_load_png_from_buffer(const void* buf, size_t size, RCRE_Arena* arena, RCRE_Tensor* out_img) {
    if (!buf || size < 33 || !arena || !out_img) return RCRE_ERR_INVALID_ARG;
    const uint8_t* p = (const uint8_t*)buf;
    static const uint8_t sig[8] = {137,80,78,71,13,10,26,10};
    if (memcmp(p, sig, 8) != 0) return RCRE_ERR_IO;

    size_t pos = 8;
    uint32_t w = 0, h = 0;
    uint8_t bit_depth = 0, color_type = 0;
    uint8_t* idat = nullptr;
    size_t idat_len = 0, idat_cap = 0;
    bool got_ihdr = false, got_iend = false;

    while (pos + 12 <= size) {
        uint32_t clen = codec_be32(p + pos);
        const uint8_t* ctype = p + pos + 4;
        if (pos + 12U + clen > size) { free(idat); return RCRE_ERR_IO; }
        const uint8_t* cdata = p + pos + 8;

        if (memcmp(ctype, "IHDR", 4) == 0 && clen == 13) {
            w = codec_be32(cdata);
            h = codec_be32(cdata + 4);
            bit_depth  = cdata[8];
            color_type = cdata[9];
            got_ihdr = true;
        } else if (memcmp(ctype, "IDAT", 4) == 0) {
            if (idat_len + clen > idat_cap) {
                size_t nc = (idat_cap ? idat_cap * 2U : 4096U) + clen;
                uint8_t* nb = (uint8_t*)realloc(idat, nc);
                if (!nb) { free(idat); return RCRE_ERR_OUT_OF_MEMORY; }
                idat = nb;
                idat_cap = nc;
            }
            if (clen > 0) memcpy(idat + idat_len, cdata, clen);
            idat_len += clen;
        } else if (memcmp(ctype, "IEND", 4) == 0) {
            got_iend = true;
            break;
        }
        pos += 12U + clen;
    }

    if (!got_ihdr || !got_iend || !idat || w == 0 || h == 0 || bit_depth != 8) {
        free(idat);
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t ch = 0;
    if (color_type == 0) ch = 1;
    else if (color_type == 2) ch = 3;
    else if (color_type == 6) ch = 4;
    else { free(idat); return RCRE_ERR_UNSUPPORTED_OP; }

    size_t row_bytes = (size_t)w * ch;
    size_t raw_cap = (size_t)h * (row_bytes + 1U);
    uint8_t* raw = (uint8_t*)malloc(raw_cap);
    if (!raw) { free(idat); return RCRE_ERR_OUT_OF_MEMORY; }

    size_t raw_len = 0;
    int zrc = codec_zlib_inflate(idat, idat_len, raw, raw_cap, &raw_len);
    free(idat);
    if (zrc != RCRE_OK || raw_len != raw_cap) {
        free(raw);
        return RCRE_ERR_IO;
    }

    size_t sh[3] = { h, w, ch };
    RCRE_Tensor t = rcre_tensor_create(sh, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    if (!t.data) { free(raw); return RCRE_ERR_OUT_OF_MEMORY; }
    uint8_t* dst = (uint8_t*)t.data;

    for (size_t y = 0; y < h; ++y) {
        const uint8_t* srow = raw + y * (row_bytes + 1U);
        uint8_t ftype = srow[0];
        const uint8_t* xrow = srow + 1;
        uint8_t* drow = dst + y * row_bytes;
        const uint8_t* prow = (y > 0) ? (dst + (y - 1U) * row_bytes) : nullptr;

        for (size_t i = 0; i < row_bytes; ++i) {
            uint8_t a = (i >= ch) ? drow[i - ch] : 0;
            uint8_t b = prow ? prow[i] : 0;
            uint8_t c = (prow && i >= ch) ? prow[i - ch] : 0;
            uint8_t val = xrow[i];
            switch (ftype) {
                case 0: drow[i] = val; break;
                case 1: drow[i] = (uint8_t)(val + a); break;
                case 2: drow[i] = (uint8_t)(val + b); break;
                case 3: drow[i] = (uint8_t)(val + (((int)a + (int)b) >> 1)); break;
                case 4: drow[i] = (uint8_t)(val + codec_paeth(a, b, c)); break;
                default: free(raw); return RCRE_ERR_IO;
            }
        }
    }

    free(raw);
    *out_img = t;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_codec_load_png(const char* filepath, RCRE_Arena* arena, RCRE_Tensor* out_img) {
    if (!filepath || !arena || !out_img) return RCRE_ERR_INVALID_ARG;
    RCRE_MappedFile map;
    int rc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_SEQUENTIAL, &map);
    if (rc != RCRE_OK) return rc;
    rc = rcre_codec_load_png_from_buffer(map.data, map.size, arena, out_img);
    rcre_io_mmap_close(&map);
    return rc;
}

RCRE_NODISCARD int rcre_codec_save_png(const RCRE_Tensor* img, const char* filepath) {
    if (!img || !img->data || !filepath || img->ndim != 3 || img->dtype != RCRE_DTYPE_U8) {
        return RCRE_ERR_INVALID_ARG;
    }
    size_t h = img->shape[0], w = img->shape[1], ch = img->shape[2];
    uint8_t ctype = (ch == 1) ? 0 : ((ch == 3) ? 2 : ((ch == 4) ? 6 : 255));
    if (ctype == 255 || w == 0 || h == 0) return RCRE_ERR_INVALID_ARG;

    size_t row_bytes = w * ch;
    size_t raw_len = h * (row_bytes + 1U);
    uint8_t* raw = (uint8_t*)malloc(raw_len);
    if (!raw) return RCRE_ERR_OUT_OF_MEMORY;
    const uint8_t* src = (const uint8_t*)img->data;
    for (size_t y = 0; y < h; ++y) {
        raw[y * (row_bytes + 1U)] = 0; /* Filter type 0 (None) */
        memcpy(raw + y * (row_bytes + 1U) + 1U, src + y * row_bytes, row_bytes);
    }

    /* Valid RFC 1950 zlib wrapper using uncompressed DEFLATE blocks (65535B max per block) */
    size_t n_blocks = (raw_len + 65534U) / 65535U;
    if (n_blocks == 0) n_blocks = 1;
    size_t zlib_len = 2U + n_blocks * 5U + raw_len + 4U;
    size_t total_len = 8U + 25U + (12U + zlib_len) + 12U;
    uint8_t* png = (uint8_t*)malloc(total_len);
    if (!png) { free(raw); return RCRE_ERR_OUT_OF_MEMORY; }

    size_t p = 0;
    static const uint8_t sig[8] = {137,80,78,71,13,10,26,10};
    memcpy(png + p, sig, 8); p += 8;

    /* IHDR */
    codec_wbe32(png + p, 13); p += 4;
    size_t ihdr_crc_start = p;
    memcpy(png + p, "IHDR", 4); p += 4;
    codec_wbe32(png + p, (uint32_t)w); p += 4;
    codec_wbe32(png + p, (uint32_t)h); p += 4;
    png[p++] = 8; png[p++] = ctype; png[p++] = 0; png[p++] = 0; png[p++] = 0;
    codec_wbe32(png + p, codec_crc32(png + ihdr_crc_start, 17)); p += 4;

    /* IDAT */
    codec_wbe32(png + p, (uint32_t)zlib_len); p += 4;
    size_t idat_crc_start = p;
    memcpy(png + p, "IDAT", 4); p += 4;
    png[p++] = 0x78; png[p++] = 0x01; /* zlib CMF=8, FLG=1 */
    size_t rem = raw_len, rpos = 0;
    while (rem > 0 || rpos == 0) {
        size_t chunk = (rem > 65535U) ? 65535U : rem;
        uint8_t last = (rem == chunk) ? 1U : 0U;
        png[p++] = last;
        png[p++] = (uint8_t)(chunk & 0xFFU);
        png[p++] = (uint8_t)((chunk >> 8) & 0xFFU);
        uint16_t nlen = (uint16_t)(~((uint16_t)chunk));
        png[p++] = (uint8_t)(nlen & 0xFFU);
        png[p++] = (uint8_t)((nlen >> 8) & 0xFFU);
        if (chunk > 0) memcpy(png + p, raw + rpos, chunk);
        p += chunk;
        rpos += chunk;
        rem -= chunk;
        if (rem == 0) break;
    }
    codec_wbe32(png + p, codec_adler32(raw, raw_len)); p += 4;
    codec_wbe32(png + p, codec_crc32(png + idat_crc_start, 4U + zlib_len)); p += 4;

    /* IEND */
    codec_wbe32(png + p, 0); p += 4;
    memcpy(png + p, "IEND", 4); p += 4;
    codec_wbe32(png + p, codec_crc32(png + p - 4, 4)); p += 4;

    int rc = rcre_io_write_atomic(filepath, png, p);
    free(png);
    free(raw);
    return rc;
}

/* ========================================================================= */
/* 3. Baseline JPEG (8x8 DCT/IDCT + Huffman + YCbCr<->RGB)                   */
/* ========================================================================= */

static const uint8_t codec_zigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

static const uint8_t codec_std_luma_qt[64] = {
    16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56, 14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77, 24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101, 72,92,95,98,112,100,103,99
};

static const uint8_t codec_dc_bits[17] = {0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0};
static const uint8_t codec_dc_vals[12] = {0,1,2,3,4,5,6,7,8,9,10,11};

static const uint8_t codec_ac_bits[17] = {0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125};
static const uint8_t codec_ac_vals[162] = {
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,
    0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,
    0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa
};

typedef struct {
    uint8_t  bits[17];
    uint8_t  vals[256];
    uint16_t maxcode[18];
    int32_t  valptr[17];
    int32_t  mincode[17];
    /* Encoder lookup */
    uint16_t enc_code[256];
    uint8_t  enc_size[256];
} RCRE_JHuff;

static void codec_jhuff_build(RCRE_JHuff* h) {
    int k = 0;
    uint16_t code = 0;
    for (int l = 1; l <= 16; ++l) {
        if (h->bits[l] == 0) {
            h->maxcode[l] = 0;
            h->mincode[l] = 0x7FFFFFFF;
            h->valptr[l]  = 0;
        } else {
            h->valptr[l] = k;
            h->mincode[l] = code;
            for (int i = 0; i < h->bits[l]; ++i) {
                uint8_t sym = h->vals[k + i];
                h->enc_code[sym] = (uint16_t)(code + i);
                h->enc_size[sym] = (uint8_t)l;
            }
            code = (uint16_t)(code + h->bits[l]);
            h->maxcode[l] = (uint16_t)(code - 1U);
            k += h->bits[l];
        }
        code <<= 1;
    }
}

typedef struct {
    const uint8_t* p;
    size_t         len;
    size_t         pos;
    uint32_t       bit_buf;
    int            bit_cnt;
} RCRE_JBitReader;

static int codec_j_next_bit(RCRE_JBitReader* r) {
    if (r->bit_cnt == 0) {
        if (r->pos >= r->len) return 0;
        uint8_t b = r->p[r->pos++];
        if (b == 0xFF && r->pos < r->len && r->p[r->pos] == 0x00) r->pos++;
        r->bit_buf = b;
        r->bit_cnt = 8;
    }
    r->bit_cnt--;
    return (int)((r->bit_buf >> r->bit_cnt) & 1U);
}

static int codec_j_read_bits(RCRE_JBitReader* r, int n) {
    int v = 0;
    for (int i = 0; i < n; ++i) v = (v << 1) | codec_j_next_bit(r);
    return v;
}

static int codec_j_extend(int v, int t) {
    if (t == 0) return 0;
    int vt = 1 << (t - 1);
    return (v < vt) ? (v + (-(1 << t)) + 1) : v;
}

static int codec_j_huff_decode(RCRE_JBitReader* r, const RCRE_JHuff* h) {
    int code = 0;
    for (int l = 1; l <= 16; ++l) {
        code = (code << 1) | codec_j_next_bit(r);
        if (h->bits[l] > 0 && code <= (int)h->maxcode[l]) {
            int idx = h->valptr[l] + (code - h->mincode[l]);
            return h->vals[idx];
        }
    }
    return 0;
}

static void codec_idct8x8(const float block[64], float out[64]) {
    float tmp[64];
    const float pi8 = 3.14159265358979323846f / 16.0f;
    for (int r = 0; r < 8; ++r) {
        for (int x = 0; x < 8; ++x) {
            float s = 0.0f;
            for (int u = 0; u < 8; ++u) {
                float cu = (u == 0) ? 0.707106781f : 1.0f;
                s += cu * block[r * 8 + u] * cosf((float)(2 * x + 1) * (float)u * pi8);
            }
            tmp[r * 8 + x] = 0.5f * s;
        }
    }
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            float s = 0.0f;
            for (int v = 0; v < 8; ++v) {
                float cv = (v == 0) ? 0.707106781f : 1.0f;
                s += cv * tmp[v * 8 + x] * cosf((float)(2 * y + 1) * (float)v * pi8);
            }
            out[y * 8 + x] = 0.5f * s + 128.0f;
        }
    }
}

static void codec_fdct8x8(const float in[64], float out[64]) {
    const float pi8 = 3.14159265358979323846f / 16.0f;
    for (int v = 0; v < 8; ++v) {
        float cv = (v == 0) ? 0.707106781f : 1.0f;
        for (int u = 0; u < 8; ++u) {
            float cu = (u == 0) ? 0.707106781f : 1.0f;
            float s = 0.0f;
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    s += (in[y * 8 + x] - 128.0f) *
                         cosf((float)(2 * x + 1) * (float)u * pi8) *
                         cosf((float)(2 * y + 1) * (float)v * pi8);
                }
            }
            out[v * 8 + u] = 0.25f * cu * cv * s;
        }
    }
}

RCRE_NODISCARD int rcre_codec_load_jpeg_from_buffer(const void* buf, size_t size, RCRE_Arena* arena, RCRE_Tensor* out_img) {
    if (!buf || size < 16 || !arena || !out_img) return RCRE_ERR_INVALID_ARG;
    const uint8_t* p = (const uint8_t*)buf;
    if (p[0] != 0xFF || p[1] != 0xD8) return RCRE_ERR_IO;

    uint16_t qt[4][64] = {0};
    RCRE_JHuff dc_h[4], ac_h[4];
    memset(dc_h, 0, sizeof(dc_h));
    memset(ac_h, 0, sizeof(ac_h));

    uint32_t w = 0, h = 0, n_comp = 0;
    uint8_t comp_id[4] = {0}, comp_hsamp[4] = {1,1,1,1}, comp_vsamp[4] = {1,1,1,1}, comp_qt[4] = {0};
    uint8_t scan_dc[4] = {0}, scan_ac[4] = {0};

    size_t pos = 2;
    size_t sos_data_start = 0;
    while (pos + 4 <= size) {
        if (p[pos] != 0xFF) { pos++; continue; }
        while (pos < size && p[pos] == 0xFF) pos++;
        if (pos >= size) break;
        uint8_t marker = p[pos++];
        if (marker == 0xD9) break;
        if (marker == 0x00 || (marker >= 0xD0 && marker <= 0xD7)) continue;
        if (pos + 2 > size) return RCRE_ERR_IO;
        uint16_t seg_len = (uint16_t)(((uint16_t)p[pos] << 8) | p[pos + 1]);
        if (seg_len < 2 || pos + seg_len > size) return RCRE_ERR_IO;
        const uint8_t* s = p + pos + 2;
        size_t slen = seg_len - 2U;

        if (marker == 0xDB) { /* DQT */
            size_t k = 0;
            while (k < slen) {
                uint8_t info = s[k++];
                uint8_t pq = info >> 4, tq = info & 0x0FU;
                if (tq >= 4 || pq > 1) return RCRE_ERR_UNSUPPORTED_OP;
                size_t nb = pq ? 128U : 64U;
                if (k + nb > slen) return RCRE_ERR_IO;
                for (int i = 0; i < 64; ++i) {
                    qt[tq][codec_zigzag[i]] = pq ? (uint16_t)((s[k + i * 2] << 8) | s[k + i * 2 + 1]) : s[k + i];
                }
                k += nb;
            }
        } else if (marker == 0xC0) { /* SOF0 Baseline */
            if (slen < 6) return RCRE_ERR_IO;
            h = (uint32_t)((s[1] << 8) | s[2]);
            w = (uint32_t)((s[3] << 8) | s[4]);
            n_comp = s[5];
            if (w == 0 || h == 0 || (n_comp != 1 && n_comp != 3) || slen < 6U + 3U * n_comp) return RCRE_ERR_UNSUPPORTED_OP;
            for (uint32_t i = 0; i < n_comp; ++i) {
                comp_id[i]    = s[6 + i * 3];
                comp_hsamp[i] = s[6 + i * 3 + 1] >> 4;
                comp_vsamp[i] = s[6 + i * 3 + 1] & 0x0FU;
                comp_qt[i]    = s[6 + i * 3 + 2];
            }
        } else if (marker == 0xC4) { /* DHT */
            size_t k = 0;
            while (k + 17 <= slen) {
                uint8_t info = s[k++];
                uint8_t tc = info >> 4, th = info & 0x0FU;
                if (th >= 4 || tc > 1) return RCRE_ERR_UNSUPPORTED_OP;
                RCRE_JHuff* target = tc ? &ac_h[th] : &dc_h[th];
                int sum = 0;
                for (int l = 1; l <= 16; ++l) { target->bits[l] = s[k++]; sum += target->bits[l]; }
                if (k + (size_t)sum > slen || sum > 256) return RCRE_ERR_IO;
                for (int i = 0; i < sum; ++i) target->vals[i] = s[k++];
                codec_jhuff_build(target);
            }
        } else if (marker == 0xDA) { /* SOS */
            if (slen < 1U + 2U * n_comp) return RCRE_ERR_IO;
            for (uint32_t i = 0; i < s[0] && i < 4; ++i) {
                uint8_t cid = s[1 + i * 2];
                for (uint32_t c = 0; c < n_comp; ++c) {
                    if (comp_id[c] == cid) {
                        scan_dc[c] = s[1 + i * 2 + 1] >> 4;
                        scan_ac[c] = s[1 + i * 2 + 1] & 0x0FU;
                    }
                }
            }
            sos_data_start = pos + seg_len;
            break;
        }
        pos += seg_len;
    }

    if (w == 0 || h == 0 || sos_data_start == 0) return RCRE_ERR_IO;

    size_t max_h = 1, max_v = 1;
    for (uint32_t c = 0; c < n_comp; ++c) {
        if (comp_hsamp[c] > max_h) max_h = comp_hsamp[c];
        if (comp_vsamp[c] > max_v) max_v = comp_vsamp[c];
    }
    size_t mcu_w = max_h * 8U, mcu_h = max_v * 8U;
    size_t n_mcu_x = (w + mcu_w - 1U) / mcu_w;
    size_t n_mcu_y = (h + mcu_h - 1U) / mcu_h;

    size_t sh[3] = { h, w, 3 };
    RCRE_Tensor t = rcre_tensor_create(sh, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    if (!t.data) return RCRE_ERR_OUT_OF_MEMORY;
    uint8_t* dst = (uint8_t*)t.data;

    RCRE_JBitReader br = { p, size, sos_data_start, 0, 0 };
    int dc_pred[4] = {0};

    float* cbuf[3] = {
        (float*)malloc(mcu_w * mcu_h * sizeof(float)),
        (float*)malloc(mcu_w * mcu_h * sizeof(float)),
        (float*)malloc(mcu_w * mcu_h * sizeof(float))
    };
    if (!cbuf[0] || !cbuf[1] || !cbuf[2]) {
        free(cbuf[0]); free(cbuf[1]); free(cbuf[2]);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    for (size_t my = 0; my < n_mcu_y; ++my) {
        for (size_t mx = 0; mx < n_mcu_x; ++mx) {
            for (uint32_t c = 0; c < n_comp; ++c) {
                for (uint8_t vs = 0; vs < comp_vsamp[c]; ++vs) {
                    for (uint8_t hs = 0; hs < comp_hsamp[c]; ++hs) {
                        float coef[64] = {0};
                        int cat = codec_j_huff_decode(&br, &dc_h[scan_dc[c] & 3U]);
                        int diff = codec_j_extend(codec_j_read_bits(&br, cat & 15), cat & 15);
                        dc_pred[c] += diff;
                        coef[0] = (float)dc_pred[c] * (float)qt[comp_qt[c] & 3U][0];

                        int k = 1;
                        while (k < 64) {
                            int rs = codec_j_huff_decode(&br, &ac_h[scan_ac[c] & 3U]);
                            int r = (rs >> 4) & 0x0F;
                            int ssz = rs & 0x0F;
                            if (ssz == 0) {
                                if (r == 15) k += 16;
                                else break;
                            } else {
                                k += r;
                                if (k >= 64) break;
                                int val = codec_j_extend(codec_j_read_bits(&br, ssz), ssz);
                                int zz = codec_zigzag[k];
                                coef[zz] = (float)val * (float)qt[comp_qt[c] & 3U][zz];
                                k++;
                            }
                        }

                        float pix[64];
                        codec_idct8x8(coef, pix);
                        size_t scale_x = max_h / (comp_hsamp[c] ? comp_hsamp[c] : 1);
                        size_t scale_y = max_v / (comp_vsamp[c] ? comp_vsamp[c] : 1);
                        for (size_t py = 0; py < 8U * scale_y; ++py) {
                            for (size_t px = 0; px < 8U * scale_x; ++px) {
                                cbuf[c][(vs * 8U * scale_y + py) * mcu_w + (hs * 8U * scale_x + px)] =
                                    pix[(py / scale_y) * 8U + (px / scale_x)];
                            }
                        }
                    }
                }
            }

            for (size_t py = 0; py < mcu_h; ++py) {
                size_t iy = my * mcu_h + py;
                if (iy >= h) continue;
                for (size_t px = 0; px < mcu_w; ++px) {
                    size_t ix = mx * mcu_w + px;
                    if (ix >= w) continue;
                    float yv = cbuf[0][py * mcu_w + px];
                    float cb = (n_comp == 3) ? (cbuf[1][py * mcu_w + px] - 128.0f) : 0.0f;
                    float cr = (n_comp == 3) ? (cbuf[2][py * mcu_w + px] - 128.0f) : 0.0f;
                    float r = yv + 1.402f * cr;
                    float g = yv - 0.344136f * cb - 0.714136f * cr;
                    float b = yv + 1.772f * cb;
                    int ri = (int)lrintf(r); ri = ri < 0 ? 0 : (ri > 255 ? 255 : ri);
                    int gi = (int)lrintf(g); gi = gi < 0 ? 0 : (gi > 255 ? 255 : gi);
                    int bi = (int)lrintf(b); bi = bi < 0 ? 0 : (bi > 255 ? 255 : bi);
                    size_t o = (iy * w + ix) * 3U;
                    dst[o + 0] = (uint8_t)ri;
                    dst[o + 1] = (uint8_t)gi;
                    dst[o + 2] = (uint8_t)bi;
                }
            }
        }
    }

    free(cbuf[0]); free(cbuf[1]); free(cbuf[2]);
    *out_img = t;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_codec_load_jpeg(const char* filepath, RCRE_Arena* arena, RCRE_Tensor* out_img) {
    if (!filepath || !arena || !out_img) return RCRE_ERR_INVALID_ARG;
    RCRE_MappedFile map;
    int rc = rcre_io_mmap_open(filepath, RCRE_IO_HINT_SEQUENTIAL, &map);
    if (rc != RCRE_OK) return rc;
    rc = rcre_codec_load_jpeg_from_buffer(map.data, map.size, arena, out_img);
    rcre_io_mmap_close(&map);
    return rc;
}

typedef struct {
    uint8_t* buf;
    size_t   pos;
    size_t   cap;
    uint32_t bit_buf;
    int      bit_cnt;
} RCRE_JBitWriter;

static void codec_jw_byte(RCRE_JBitWriter* w, uint8_t b, bool stuff) {
    if (w->pos + 2 < w->cap) {
        w->buf[w->pos++] = b;
        if (stuff && b == 0xFF) w->buf[w->pos++] = 0x00;
    }
}

static void codec_jw_bits(RCRE_JBitWriter* w, uint32_t code, int n) {
    for (int i = n - 1; i >= 0; --i) {
        w->bit_buf = (w->bit_buf << 1U) | ((code >> i) & 1U);
        w->bit_cnt++;
        if (w->bit_cnt == 8) {
            codec_jw_byte(w, (uint8_t)(w->bit_buf & 0xFFU), true);
            w->bit_buf = 0;
            w->bit_cnt = 0;
        }
    }
}

static int codec_j_bit_cat(int v) {
    int av = (v < 0) ? -v : v;
    int bits = 0;
    while (av > 0) { av >>= 1; bits++; }
    return bits;
}

static void codec_jw_flush(RCRE_JBitWriter* w) {
    if (w->bit_cnt > 0) {
        uint8_t b = (uint8_t)(((w->bit_buf << (8 - w->bit_cnt)) | ((1U << (8 - w->bit_cnt)) - 1U)) & 0xFFU);
        codec_jw_byte(w, b, true);
        w->bit_buf = 0;
        w->bit_cnt = 0;
    }
}

RCRE_NODISCARD int rcre_codec_save_jpeg(const RCRE_Tensor* img, const char* filepath, int quality) {
    if (!img || !img->data || !filepath || img->ndim != 3 || img->shape[2] != 3 || img->dtype != RCRE_DTYPE_U8) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    int scale = (quality < 50) ? (5000 / quality) : (200 - quality * 2);

    uint8_t qt[64];
    for (int i = 0; i < 64; ++i) {
        int v = ((int)codec_std_luma_qt[i] * scale + 50) / 100;
        qt[i] = (uint8_t)(v < 1 ? 1 : (v > 255 ? 255 : v));
    }

    RCRE_JHuff dc_h = {0}, ac_h = {0};
    memcpy(dc_h.bits, codec_dc_bits, 17); memcpy(dc_h.vals, codec_dc_vals, 12);
    memcpy(ac_h.bits, codec_ac_bits, 17); memcpy(ac_h.vals, codec_ac_vals, 162);
    codec_jhuff_build(&dc_h);
    codec_jhuff_build(&ac_h);

    size_t h = img->shape[0], w = img->shape[1];
    size_t cap = w * h * 4U + 4096U;
    RCRE_JBitWriter bw = { (uint8_t*)malloc(cap), 0, cap, 0, 0 };
    if (!bw.buf) return RCRE_ERR_OUT_OF_MEMORY;

    /* SOI */
    codec_jw_byte(&bw, 0xFF, false); codec_jw_byte(&bw, 0xD8, false);
    /* DQT (table 0) */
    codec_jw_byte(&bw, 0xFF, false); codec_jw_byte(&bw, 0xDB, false);
    codec_jw_byte(&bw, 0x00, false); codec_jw_byte(&bw, 67, false);
    codec_jw_byte(&bw, 0x00, false);
    for (int i = 0; i < 64; ++i) codec_jw_byte(&bw, qt[codec_zigzag[i]], false);

    /* SOF0 (4:4:4 3-component baseline) */
    codec_jw_byte(&bw, 0xFF, false); codec_jw_byte(&bw, 0xC0, false);
    codec_jw_byte(&bw, 0x00, false); codec_jw_byte(&bw, 17, false);
    codec_jw_byte(&bw, 8, false);
    codec_jw_byte(&bw, (uint8_t)(h >> 8), false); codec_jw_byte(&bw, (uint8_t)(h & 0xFFU), false);
    codec_jw_byte(&bw, (uint8_t)(w >> 8), false); codec_jw_byte(&bw, (uint8_t)(w & 0xFFU), false);
    codec_jw_byte(&bw, 3, false);
    for (int c = 1; c <= 3; ++c) {
        codec_jw_byte(&bw, (uint8_t)c, false);
        codec_jw_byte(&bw, 0x11, false);
        codec_jw_byte(&bw, 0, false);
    }

    /* DHT DC0 & AC0 */
    codec_jw_byte(&bw, 0xFF, false); codec_jw_byte(&bw, 0xC4, false);
    codec_jw_byte(&bw, 0x00, false); codec_jw_byte(&bw, 19 + 12, false);
    codec_jw_byte(&bw, 0x00, false);
    for (int i = 1; i <= 16; ++i) codec_jw_byte(&bw, codec_dc_bits[i], false);
    for (int i = 0; i < 12; ++i) codec_jw_byte(&bw, codec_dc_vals[i], false);

    codec_jw_byte(&bw, 0xFF, false); codec_jw_byte(&bw, 0xC4, false);
    codec_jw_byte(&bw, 0x00, false); codec_jw_byte(&bw, (uint8_t)(19 + 162), false);
    codec_jw_byte(&bw, 0x10, false);
    for (int i = 1; i <= 16; ++i) codec_jw_byte(&bw, codec_ac_bits[i], false);
    for (int i = 0; i < 162; ++i) codec_jw_byte(&bw, codec_ac_vals[i], false);

    /* SOS */
    codec_jw_byte(&bw, 0xFF, false); codec_jw_byte(&bw, 0xDA, false);
    codec_jw_byte(&bw, 0x00, false); codec_jw_byte(&bw, 12, false);
    codec_jw_byte(&bw, 3, false);
    for (int c = 1; c <= 3; ++c) {
        codec_jw_byte(&bw, (uint8_t)c, false);
        codec_jw_byte(&bw, 0x00, false);
    }
    codec_jw_byte(&bw, 0, false); codec_jw_byte(&bw, 63, false); codec_jw_byte(&bw, 0, false);

    const uint8_t* src = (const uint8_t*)img->data;
    int dc_prev[3] = {0};
    for (size_t my = 0; my < h; my += 8) {
        for (size_t mx = 0; mx < w; mx += 8) {
            for (int c = 0; c < 3; ++c) {
                float blk[64], dct[64];
                for (size_t py = 0; py < 8; ++py) {
                    size_t iy = (my + py < h) ? (my + py) : (h - 1U);
                    for (size_t px = 0; px < 8; ++px) {
                        size_t ix = (mx + px < w) ? (mx + px) : (w - 1U);
                        const uint8_t* rgb = src + (iy * w + ix) * 3U;
                        float R = rgb[0], G = rgb[1], B = rgb[2];
                        if (c == 0) blk[py * 8 + px] = 0.299f * R + 0.587f * G + 0.114f * B;
                        else if (c == 1) blk[py * 8 + px] = -0.168736f * R - 0.331264f * G + 0.5f * B + 128.0f;
                        else blk[py * 8 + px] = 0.5f * R - 0.418688f * G - 0.081312f * B + 128.0f;
                    }
                }
                codec_fdct8x8(blk, dct);
                int qz[64];
                for (int i = 0; i < 64; ++i) {
                    int zz = codec_zigzag[i];
                    qz[i] = (int)lrintf(dct[zz] / (float)qt[zz]);
                }
                int diff = qz[0] - dc_prev[c];
                dc_prev[c] = qz[0];
                int cat = codec_j_bit_cat(diff);
                if (cat > 11) cat = 11;
                codec_jw_bits(&bw, dc_h.enc_code[cat], dc_h.enc_size[cat]);
                if (cat > 0) {
                    int bits = (diff < 0) ? (diff - 1 + (1 << cat)) : diff;
                    codec_jw_bits(&bw, (uint32_t)bits, cat);
                }
                int last_nz = 63;
                while (last_nz > 0 && qz[last_nz] == 0) last_nz--;
                int run = 0;
                for (int k = 1; k <= last_nz; ++k) {
                    if (qz[k] == 0) {
                        run++;
                        if (run == 16) {
                            codec_jw_bits(&bw, ac_h.enc_code[0xF0], ac_h.enc_size[0xF0]);
                            run = 0;
                        }
                    } else {
                        int ac_cat = codec_j_bit_cat(qz[k]);
                        if (ac_cat > 10) ac_cat = 10;
                        uint8_t sym = (uint8_t)((run << 4) | ac_cat);
                        codec_jw_bits(&bw, ac_h.enc_code[sym], ac_h.enc_size[sym]);
                        int ab = (qz[k] < 0) ? (qz[k] - 1 + (1 << ac_cat)) : qz[k];
                        codec_jw_bits(&bw, (uint32_t)ab, ac_cat);
                        run = 0;
                    }
                }
                if (last_nz < 63) codec_jw_bits(&bw, ac_h.enc_code[0x00], ac_h.enc_size[0x00]);
            }
        }
    }
    codec_jw_flush(&bw);
    codec_jw_byte(&bw, 0xFF, false); codec_jw_byte(&bw, 0xD9, false);

    int rc = rcre_io_write_atomic(filepath, bw.buf, bw.pos);
    free(bw.buf);
    return rc;
}
