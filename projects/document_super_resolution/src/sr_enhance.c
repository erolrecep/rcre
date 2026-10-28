/**
 * @file sr_enhance.c
 * @brief Enhancement primitives: edge-masked unsharp, CLAHE, Otsu,
 *        adaptive threshold, binary morphology. All HWC, U8 in/out.
 */
#include "sr.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static size_t C_of(const RCRE_Tensor* t) { return (t->ndim >= 3) ? t->shape[2] : 1; }

static float* to_f32(const RCRE_Tensor* t, RCRE_Arena* a) {
    size_t n = t->shape[0] * t->shape[1] * C_of(t);
    float* buf = (float*)rcre_arena_alloc(a, n * sizeof(float), 64);
    if (!buf) return NULL;
    if (t->dtype == RCRE_DTYPE_U8) {
        const uint8_t* p = (const uint8_t*)t->data;
        for (size_t i = 0; i < n; i++) buf[i] = (float)p[i];
    } else {
        memcpy(buf, t->data, n * sizeof(float));
    }
    return buf;
}

static void u8_write(RCRE_Tensor* t, const float* buf, size_t n) {
    if (t->dtype == RCRE_DTYPE_U8) {
        uint8_t* p = (uint8_t*)t->data;
        for (size_t i = 0; i < n; i++) {
            int v = (int)lrintf(buf[i]);
            p[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    } else {
        memcpy(t->data, buf, n * sizeof(float));
    }
}

/* Separable Gaussian on a single-channel float buffer (in-place via tmp). */
static void gauss_f32(float* data, size_t h, size_t w, float sigma, float* tmp) {
    int r = (int)ceilf(sigma * 3.0f);
    if (r < 1) r = 1;
    if (r > 16) r = 16;
    float k[33];
    float ksum = 0;
    for (int i = -r; i <= r; i++) {
        k[i + r] = expf(-(float)(i * i) / (2.0f * sigma * sigma));
        ksum += k[i + r];
    }
    for (int i = 0; i < 2 * r + 1; i++) k[i] /= ksum;

    for (size_t y = 0; y < h; y++) {
        float* row = data + y * w;
        float* orow = tmp + y * w;
        for (size_t x = 0; x < w; x++) {
            float s = 0;
            for (int i = -r; i <= r; i++) {
                int xi = (int)x + i;
                if (xi < 0) xi = 0;
                if (xi >= (int)w) xi = (int)w - 1;
                s += row[xi] * k[i + r];
            }
            orow[x] = s;
        }
    }
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            float s = 0;
            for (int i = -r; i <= r; i++) {
                int yi = (int)y + i;
                if (yi < 0) yi = 0;
                if (yi >= (int)h) yi = (int)h - 1;
                s += tmp[(size_t)yi * w + x] * k[i + r];
            }
            data[y * w + x] = s;
        }
    }
}

/* ========================================================================= */
/* Edge-masked unsharp masking                                                */
/*   out = f + amount * (f - blur(f)) * mask(gradient)                        */
/*        -> sharpening only where structure exists; flats stay clean.        */
/* ========================================================================= */

int sr_sharpen_edge(const RCRE_Tensor* src, RCRE_Tensor* dst,
                    float amount, RCRE_Arena* arena) {
    if (!src || !dst || !src->data || !dst->data) return 0;
    if (amount <= 0.0f) { /* amount 0 = passthrough copy */
        size_t n = src->shape[0] * src->shape[1] * C_of(src);
        if (src->dtype == RCRE_DTYPE_U8 && dst->dtype == RCRE_DTYPE_U8) {
            memcpy(dst->data, src->data, n);
        } else {
            float* f = to_f32(src, arena);
            if (!f) return 0;
            u8_write(dst, f, n);
        }
        return 1;
    }
    size_t h = src->shape[0], w = src->shape[1], C = C_of(src);
    if (h < 3 || w < 3) return 0;
    if (C != C_of(dst) || dst->shape[0] != h || dst->shape[1] != w) return 0;

    float* f = to_f32(src, arena);
    if (!f) return 0;
    size_t n = h * w;

    float* mag = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    if (!mag) return 0;
    memset(mag, 0, n * sizeof(float));

    /* 1) gradient magnitude (channel-averaged) on the ORIGINAL image */
    for (size_t y = 1; y + 1 < h; y++) {
        for (size_t x = 1; x + 1 < w; x++) {
            float gx = 0, gy = 0;
            for (size_t c = 0; c < C; c++) {
                gx += f[(y * w + x) * C + c] - f[(y * w + x - 1) * C + c];
                gy += f[(y * w + x) * C + c] - f[((y - 1) * w + x) * C + c];
            }
            gx /= (float)C;
            gy /= (float)C;
            mag[y * w + x] = sqrtf(gx * gx + gy * gy);
        }
    }
    float mg = 0;
    for (size_t i = 0; i < n; i++) mg += mag[i];
    mg /= (float)n;
    float gate = mg * 4.0f + 2.0f; /* soft gate around typical edge strength */

    /* 2) per-channel unsharp with mask */
    float* out = (float*)rcre_arena_alloc(arena, (size_t)n * C * sizeof(float), 64);
    if (!out) return 0;
    float* ch = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    float* bch = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    if (!ch || !bch) return 0;
    for (size_t c = 0; c < C; c++) {
        for (size_t i = 0; i < n; i++) {
            ch[i] = f[c * n + i];
            bch[i] = f[c * n + i];
        }
        gauss_f32(bch, h, w, 1.3f, ch /* scratch */);
        for (size_t i = 0; i < n; i++) {
            float m = mag[i] / gate;
            if (m > 1.0f) m = 1.0f;
            m = m * m * (3.0f - 2.0f * m);
            out[i * C + c] = f[c * n + i] + amount * (f[c * n + i] - bch[i]) * m;
        }
    }
    u8_write(dst, out, (size_t)n * C);
    return 1;
}

/* ========================================================================= */
/* CLAHE — Contrast Limited Adaptive Histogram Equalization                   */
/* ========================================================================= */

int sr_clahe(const RCRE_Tensor* src, RCRE_Tensor* dst,
             int grid, float clip_limit, RCRE_Arena* arena) {
    if (!src || !dst || !src->data || !dst->data) return 0;
    size_t h = src->shape[0], w = src->shape[1], C = C_of(src);
    if (h < 8 || w < 8 || C != C_of(dst) || dst->shape[0] != h || dst->shape[1] != w) return 0;
    if (grid < 2) grid = 2;
    if (grid > 16) grid = 16;

    float* f = to_f32(src, arena);
    if (!f) return 0;
    size_t n = h * w;
    float* out = (float*)rcre_arena_alloc(arena, (size_t)n * C * sizeof(float), 64);
    size_t* lut = (size_t*)rcre_arena_alloc(
        arena, (size_t)grid * grid * 256 * sizeof(size_t), 64);
    if (!out || !lut) return 0;

    for (size_t c = 0; c < C; c++) {
        const float* ch = f + (size_t)c * n;

        /* tile pixel counts */
        size_t th = (h + grid / 2) / grid;
        size_t tw = (w + grid / 2) / grid;

        for (int ty = 0; ty < grid; ty++) {
            for (int tx = 0; tx < grid; tx++) {
                size_t hist[256];
                memset(hist, 0, sizeof(hist));
                size_t count = 0;
                size_t y0 = (size_t)ty * th, y1 = y0 + th;
                size_t x0 = (size_t)tx * tw, x1 = x0 + tw;
                if (y1 > h) y1 = h;
                if (x1 > w) x1 = w;
                for (size_t y = y0; y < y1; y++)
                    for (size_t x = x0; x < x1; x++) {
                        int v = (int)lrintf(ch[y * w + x]);
                        if (v < 0) v = 0;
                        if (v > 255) v = 255;
                        hist[v]++;
                        count++;
                    }
                if (count == 0) {
                    for (int v = 0; v < 256; v++) lut[(size_t)(ty * grid + tx) * 256 + v] = (size_t)v;
                    continue;
                }
                /* clip & redistribute */
                size_t clip = (size_t)((double)clip_limit * (double)count / 256.0);
                if (clip < 1) clip = 1;
                size_t excess = 0;
                for (int v = 0; v < 256; v++) {
                    if (hist[v] > clip) { excess += hist[v] - clip; hist[v] = clip; }
                }
                size_t per = excess / 256;
                for (int v = 0; v < 256; v++) hist[v] += per;
                /* CDF -> LUT */
                size_t cdf = 0;
                for (int v = 0; v < 256; v++) {
                    cdf += hist[v];
                    size_t mapped = (size_t)((double)cdf / (double)count * 255.0 + 0.5);
                    if (mapped > 255) mapped = 255;
                    lut[(size_t)(ty * grid + tx) * 256 + v] = mapped;
                }
            }
        }

        /* per-pixel: bilinear blend of 4 tile LUTs (tile centers) */
        for (size_t y = 0; y < h; y++) {
            float fy = (double)y / (double)h * (double)grid - 0.5;
            int ty0 = (int)floor(fy);
            float sy = fy - ty0;
            if (ty0 < 0) ty0 = 0;
            if (ty0 >= grid - 1) ty0 = grid - 2;
            int ty1 = ty0 + 1;
            for (size_t x = 0; x < w; x++) {
                float fx = (double)x / (double)w * (double)grid - 0.5;
                int tx0 = (int)floor(fx);
                float sx = fx - tx0;
                if (tx0 < 0) tx0 = 0;
                if (tx0 >= grid - 1) tx0 = grid - 2;
                int tx1 = tx0 + 1;
                int v = (int)lrintf(ch[y * w + x]);
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                const size_t* l00 = lut + (size_t)(ty0 * grid + tx0) * 256;
                const size_t* l01 = lut + (size_t)(ty0 * grid + tx1) * 256;
                const size_t* l10 = lut + (size_t)(ty1 * grid + tx0) * 256;
                const size_t* l11 = lut + (size_t)(ty1 * grid + tx1) * 256;
                float top = (float)(l00[v] * (1.0 - sx) + l01[v] * sx);
                float bot = (float)(l10[v] * (1.0 - sx) + l11[v] * sx);
                out[(size_t)c * n + y * w + x] = top * (1.0f - sy) + bot * sy;
            }
        }
    }
    u8_write(dst, out, (size_t)n * C);
    return 1;
}

/* ========================================================================= */
/* Otsu + adaptive threshold                                                  */
/* ========================================================================= */

int sr_otsu_threshold(const RCRE_Tensor* src, float* threshold) {
    if (!src || !src->data || !threshold) return 0;
    size_t n = src->shape[0] * src->shape[1] * C_of(src);
    size_t hist[256];
    memset(hist, 0, sizeof(hist));
    if (src->dtype == RCRE_DTYPE_U8) {
        const uint8_t* p = (const uint8_t*)src->data;
        for (size_t i = 0; i < n; i++) hist[p[i]]++;
    } else {
        const float* p = (const float*)src->data;
        for (size_t i = 0; i < n; i++) {
            int v = (int)lrintf(p[i]);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            hist[v]++;
        }
    }
    float hf[256];
    double total = 0;
    for (int i = 0; i < 256; i++) { hf[i] = (float)hist[i]; total += hist[i]; }
    if (total <= 0) { *threshold = 127.0f; return 1; }
    double mu = 0;
    for (int t = 0; t < 256; t++) mu += (double)t * hist[t];
    mu /= total;
    double sumB = 0, wB = 0, best = -1.0;
    int tFirst = 127, tLast = 127;
    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB <= 0) continue;
        double wF = total - wB;
        if (wF <= 0) break;
        sumB += (double)t * hist[t];
        double mB = sumB / wB, mF = (mu * total - sumB) / wF;
        double var = wB * wF * (mB - mF) * (mB - mF);
        if (var > best) { best = var; tFirst = t; tLast = t; }
        else if (var == best) { tLast = t; }
    }
    (void)hf;
    *threshold = (float)((double)tFirst + (double)tLast) / 2.0f;
    return 1;
}

int sr_adaptive_threshold(const RCRE_Tensor* src, RCRE_Tensor* dst,
                          int block_size, float const_c, RCRE_Arena* arena) {
    if (!src || !dst || !src->data || !dst->data) return 0;
    size_t h = src->shape[0], w = src->shape[1];
    if (dst->shape[0] != h || dst->shape[1] != w) return 0;
    if (block_size < 3 || (block_size % 2) == 0) block_size = 25;
    int r = block_size / 2;

    float* f = to_f32(src, arena);
    if (!f) return 0;
    size_t n = h * w;

    /* integral image (sum & sum-of-squares) with 1-row/col padding */
    size_t ih = h + 1, iw = w + 1;
    double* ii  = (double*)rcre_arena_alloc(arena, ih * iw * sizeof(double), 64);
    double* iis = (double*)rcre_arena_alloc(arena, ih * iw * sizeof(double), 64);
    if (!ii || !iis) return 0;
    memset(ii, 0, ih * iw * sizeof(double));
    memset(iis, 0, ih * iw * sizeof(double));
    for (size_t y = 1; y <= h; y++) {
        double row = 0, rowsq = 0;
        for (size_t x = 1; x <= w; x++) {
            double v = f[(y - 1) * w + (x - 1)];
            row += v;
            rowsq += v * v;
            ii[y * iw + x] = ii[(y - 1) * iw + x] + row;
            iis[y * iw + x] = iis[(y - 1) * iw + x] + rowsq;
        }
    }

    float* out = (float*)rcre_arena_alloc(arena, n * C_of(src) * sizeof(float), 64);
    if (!out) return 0;
    size_t C = C_of(src);
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            size_t y0 = y < (size_t)r ? 0 : y - (size_t)r + 1;
            size_t x0 = x < (size_t)r ? 0 : x - (size_t)r + 1;
            size_t y1 = y + (size_t)r + 1 > h ? h : y + (size_t)r + 1;
            size_t x1 = x + (size_t)r + 1 > w ? w : x + (size_t)r + 1;
            double area = (double)((y1 - y0) * (x1 - x0));
            double s  = ii[(y1) * iw + x1] - ii[y0 * iw + x1] - ii[y1 * iw + x0] + ii[y0 * iw + x0];
            double sq = iis[y1 * iw + x1] - iis[y0 * iw + x1] - iis[y1 * iw + x0] + iis[y0 * iw + x0];
            double mean = s / area;
            double var = sq / area - mean * mean;
            if (var < 0) var = 0;
            double thr = mean - const_c * sqrt(var);
            double v = f[y * w + x];
            float bin = (v < thr) ? 0.0f : 255.0f; /* dark text on light paper */
            for (size_t c = 0; c < C; c++) out[(y * w + x) * C + c] = bin;
        }
    }
    u8_write(dst, out, (size_t)n * C);
    return 1;
}

/* ========================================================================= */
/* 3x3 binary morphology (open/close), in-place on U8 {0,255}                 */
/* ========================================================================= */

static void morph_dilate_u8(uint8_t* img, size_t h, size_t w, size_t C, uint8_t* tmp) {
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++)
            for (size_t c = 0; c < C; c++) {
                uint8_t m = img[(y * w + x) * C + c];
                for (int dy = -1; dy <= 1 && m < 255; dy++)
                    for (int dx = -1; dx <= 1 && m < 255; dx++) {
                        int yy = (int)y + dy, xx = (int)x + dx;
                        if (yy < 0 || xx < 0 || yy >= (int)h || xx >= (int)w) continue;
                        uint8_t v = img[(size_t)yy * w * C + (size_t)xx * C + c];
                        if (v > m) m = v;
                    }
                tmp[(y * w + x) * C + c] = m;
            }
    memcpy(img, tmp, h * w * C);
}

static void morph_erode_u8(uint8_t* img, size_t h, size_t w, size_t C, uint8_t* tmp) {
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++)
            for (size_t c = 0; c < C; c++) {
                uint8_t m = img[(y * w + x) * C + c];
                for (int dy = -1; dy <= 1 && m > 0; dy++)
                    for (int dx = -1; dx <= 1 && m > 0; dx++) {
                        int yy = (int)y + dy, xx = (int)x + dx;
                        if (yy < 0 || xx < 0 || yy >= (int)h || xx >= (int)w) continue;
                        uint8_t v = img[(size_t)yy * w * C + (size_t)xx * C + c];
                        if (v < m) m = v;
                    }
                tmp[(y * w + x) * C + c] = m;
            }
    memcpy(img, tmp, h * w * C);
}

int sr_morphology_open_close(RCRE_Tensor* bin, int iterations, RCRE_Arena* arena) {
    if (!bin || !bin->data || iterations <= 0) return 1;
    if (bin->dtype != RCRE_DTYPE_U8) return 0;
    size_t h = bin->shape[0], w = bin->shape[1], C = C_of(bin);
    uint8_t* tmp = (uint8_t*)rcre_arena_alloc(arena, h * w * C, 64);
    if (!tmp) return 0;
    uint8_t* img = (uint8_t*)bin->data;
    for (int it = 0; it < iterations; it++) {
        morph_erode_u8(img, h, w, C, tmp);   /* open: remove specks   */
        morph_dilate_u8(img, h, w, C, tmp);  /* ...keep text strokes  */
        morph_dilate_u8(img, h, w, C, tmp);  /* close: bridge gaps    */
        morph_erode_u8(img, h, w, C, tmp);   /* ...without fattening  */
    }
    return 1;
}
