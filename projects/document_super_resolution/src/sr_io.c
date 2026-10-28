/**
 * @file sr_io.c
 * @brief Zero-dependency PGM/PPM (P2/P3/P5/P6) I/O for RCRE-SR.
 *
 * Binary P5/P6 used for writing; all four common encodings accepted on read.
 * Images are U8 HWC tensors (shape {h, w, c}).
 */
#include "sr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read next non-comment token as int. Returns 1 on success.
 * The first non-digit char after the number is pushed back (ungetc), so
 * binary payloads that follow the header are not consumed. */
static int pnm_next_int(FILE* f, int* out) {
    int c;
    for (;;) {
        c = fgetc(f);
        if (c == EOF) return 0;
        if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n') {} continue; }
        if (c >= '0' && c <= '9') break;
        if (c <= 32) continue;
        return 0;
    }
    int v = c - '0';
    while ((c = fgetc(f)) != EOF && c >= '0' && c <= '9') v = v * 10 + (c - '0');
    if (c != EOF) ungetc(c, f);
    *out = v;
    return 1;
}

int sr_pnm_read(const char* path, RCRE_Tensor* out, RCRE_Arena* arena) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    int c1 = fgetc(f), c2 = fgetc(f);
    if (c1 != 'P' || (c2 < '2' || c2 > '6')) { fclose(f); return 0; }
    int binary = (c2 >= '5');
    int channels;
    switch (c2) {
        case '2': case '5': channels = 1; break;
        case '3': case '6': channels = 3; break;
        default: fclose(f); return 0;
    }

    int w = 0, h = 0, maxval = 255;
    if (!pnm_next_int(f, &w) || !pnm_next_int(f, &h) || !pnm_next_int(f, &maxval) ||
        w <= 0 || h <= 0 || w > 65536 || h > 65536) {
        fclose(f); return 0;
    }

    size_t n = (size_t)w * (size_t)h * (size_t)channels;
    uint8_t* buf = (uint8_t*)rcre_arena_alloc(arena, n, 64);
    if (!buf) { fclose(f); return 0; }

    size_t got;
    if (binary) {
        fgetc(f); /* single whitespace after maxval */
        got = fread(buf, 1, n, f);
    } else {
        got = 0;
        for (size_t i = 0; i < n; i++) {
            int v;
            if (!pnm_next_int(f, &v)) { fclose(f); return 0; }
            if (v > maxval) v = maxval;
            buf[i] = (uint8_t)v;
            got++;
        }
    }
    fclose(f);
    if (got != n) return 0;

    size_t shape[3] = {(size_t)h, (size_t)w, (size_t)channels};
    *out = rcre_tensor_from_buffer(buf, shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC);
    return 1;
}

int sr_pnm_write(const char* path, const RCRE_Tensor* img) {
    if (!img || !img->data) return 0;
    if (img->dtype != RCRE_DTYPE_U8) return 0;
    size_t h = img->shape[0], w = img->shape[1];
    size_t c = img->ndim >= 3 ? img->shape[2] : 1;
    if (c != 1 && c != 3) return 0;

    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P%c\n%zu %zu\n255\n", c == 1 ? '5' : '6', w, h);
    size_t n = w * h * c;
    int ok = (fwrite(img->data, 1, n, f) == n);
    fclose(f);
    return ok;
}

/* ========================================================================= */
/* Minimal 8-bit PNG writer (grayscale or RGB). No zlib dependency:          */
/* uses raw DEFLATE "stored" blocks, which are trivially valid.              */
/* ========================================================================= */

static uint32_t png_crc_table[256];
static int png_crc_init = 0;

static void png_make_crc_table(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        png_crc_table[n] = c;
    }
    png_crc_init = 1;
}

static uint32_t png_crc(uint32_t crc, const uint8_t* buf, size_t len) {
    if (!png_crc_init) png_make_crc_table();
    crc = ~crc;
    for (size_t i = 0; i < len; i++)
        crc = png_crc_table[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

static uint32_t adler32(uint32_t adler, const uint8_t* buf, size_t len) {
    uint32_t a = adler & 0xFFFF, b = (adler >> 16) & 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        a += buf[i];
        if (a >= 65521) a -= 65521;
        b += a;
        if (b >= 65521) b -= 65521;
    }
    return (b << 16) | a;
}

static void be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

int sr_png_write(const char* path, const RCRE_Tensor* img) {
    if (!img || !img->data) return 0;
    if (img->dtype != RCRE_DTYPE_U8) return 0;
    size_t h = img->shape[0], w = img->shape[1];
    size_t c = img->ndim >= 3 ? img->shape[2] : 1;
    if (c != 1 && c != 3) return 0;

    FILE* f = fopen(path, "wb");
    if (!f) return 0;

    /* signature */
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    if (fwrite(sig, 1, 8, f) != 8) { fclose(f); return 0; }

    uint8_t chunk[12];
    /* IHDR */
    uint8_t ihdr[13];
    be32(ihdr, (uint32_t)w);
    be32(ihdr + 4, (uint32_t)h);
    ihdr[8] = 8;   /* bit depth */
    ihdr[9] = (c == 1) ? 0 : 2; /* color type */
    ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    be32(chunk, 13);
    memcpy(chunk + 4, "IHDR", 4);
    if (fwrite(chunk, 1, 8, f) != 8) { fclose(f); return 0; }
    uint32_t crc = png_crc(0, chunk + 4, 4);
    crc = png_crc(crc, ihdr, 13);
    be32(chunk, crc);
    if (fwrite(ihdr, 1, 13, f) != 13 || fwrite(chunk, 1, 4, f) != 4) { fclose(f); return 0; }

    /* IDAT: stored DEFLATE blocks over raw scanlines (filter 0 per row) */
    size_t stride = w * c;
    size_t rowbytes = stride + 1;
    size_t rawlen = rowbytes * h;
    uint8_t* raw = (uint8_t*)malloc(rawlen);
    if (!raw) { fclose(f); return 0; }
    for (size_t y = 0; y < h; y++) {
        raw[y * rowbytes] = 0; /* filter: none */
        memcpy(raw + y * rowbytes + 1, (const uint8_t*)img->data + y * stride, stride);
    }

    int ok = 1;

    /* PNG IDAT is a zlib (RFC 1950) stream: 0x78 0x01 header + DEFLATE
     * (stored, uncompressed blocks) + ADLER32. Build it fully, then frame
     * into IDAT chunks of at most 1MB. */
    size_t zlib_len = 2 + rawlen + ((rawlen / 65535) + 1) * 5 + 4;
    uint8_t* zs = (uint8_t*)malloc(zlib_len);
    if (!zs) { free(raw); fclose(f); return 0; }
    size_t z = 0;
    zs[z++] = 0x78;
    zs[z++] = 0x01;
    {
        size_t pos = 0;
        while (pos < rawlen) {
            size_t take = rawlen - pos;
            if (take > 65535) take = 65535;
            int last = (pos + take >= rawlen) ? 1 : 0;
            zs[z] = (uint8_t)last;
            zs[z + 1] = (uint8_t)(take & 0xFF);
            zs[z + 2] = (uint8_t)((take >> 8) & 0xFF);
            zs[z + 3] = (uint8_t)(~take & 0xFF);
            zs[z + 4] = (uint8_t)((~take >> 8) & 0xFF);
            memcpy(zs + z + 5, raw + pos, take);
            z += 5 + take;
            pos += take;
        }
        uint32_t ad = adler32(1, raw, rawlen);
        /* RFC 1950: ADLER32 trailer is big-endian (verified vs reference zlib) */
        zs[z++] = (uint8_t)((ad >> 24) & 0xFF);
        zs[z++] = (uint8_t)((ad >> 16) & 0xFF);
        zs[z++] = (uint8_t)((ad >> 8) & 0xFF);
        zs[z++] = (uint8_t)(ad & 0xFF);
    }
    free(raw);
    raw = NULL;

    size_t cpos = 0;
    while (ok && cpos < z) {
        size_t seg = z - cpos;
        if (seg > (1u << 20)) seg = (1u << 20);
        be32(chunk, (uint32_t)seg);
        memcpy(chunk + 4, "IDAT", 4);
        if (fwrite(chunk, 1, 8, f) != 8) { ok = 0; break; }
        if (fwrite(zs + cpos, 1, seg, f) != seg) { ok = 0; break; }
        uint32_t crc = png_crc(0, chunk + 4, 4);
        crc = png_crc(crc, zs + cpos, seg);
        be32(chunk, crc);
        if (fwrite(chunk, 1, 4, f) != 4) { ok = 0; break; }
        cpos += seg;
    }
    free(zs);

    /* IEND */
    if (ok) {
        be32(chunk, 0);
        memcpy(chunk + 4, "IEND", 4);
        crc = png_crc(0, chunk + 4, 4);
        be32(chunk + 8, crc);
        ok = (fwrite(chunk, 1, 12, f) == 12);
    }
    fclose(f);
    return ok;
}
