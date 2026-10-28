/**
 * @file sr_resize.c
 * @brief High-quality arbitrary-scale resize: separable Lanczos-3 & cubic.
 *
 * Lanczos-3 is the document-imaging standard (PDF tooling, scan
 * re-sampling): it preserves stroke edges far better than bilinear at
 * 2..4x, which is exactly what OCR needs. Implemented as two separable
 * passes (H then V) through one float temp buffer.
 *
 * Performance: kernel weights depend only on the output coordinate, so
 * each pass precomputes (src_index, weight) tables once and the inner
 * loop is pure FMA — no per-pixel transcendental calls.
 * Edge handling clamps the sample index and renormalizes weights per
 * sample so borders never darken.
 */
#include "sr.h"

#include <math.h>
#include <string.h>

typedef float (*KernelFn)(float d);

static float sinc_pi(float x) {
    if (x == 0.0f) return 1.0f;
    float px = (float)M_PI * x;
    return sinf(px) / px;
}

static float lanczos3(float d) {
    float ad = fabsf(d);
    if (ad >= 3.0f) return 0.0f;
    return sinc_pi(d) * sinc_pi(d / 3.0f);
}

/* Cubic convolution, a = -0.5 (Catmull-Rom). */
static float cubic(float d) {
    const float a = -0.5f;
    float ad = fabsf(d);
    if (ad >= 2.0f) return 0.0f;
    if (ad < 1.0f) return (a + 2.0f) * ad * ad * ad - (a + 3.0f) * ad * ad + 1.0f;
    return a * ad * ad * ad - 5.0f * a * ad * ad + 8.0f * a * ad - 4.0f * a;
}

/* Precomputed tap: clamped source index + weight. */
typedef struct {
    size_t idx;  /* clamped into [0, len-1] */
    float w;
} Tap;

typedef struct {
    Tap taps[2 * 3 + 1]; /* max support = 7 (Lanczos-3) */
    int n;
    float wsum;
} SampleRow;

/* Build the sample table for one axis.
 * out_len output pixels -> in_len input pixels, scale = in_len/out_len. */
static void build_samples(SampleRow* tbl, size_t out_len, size_t in_len,
                          float scale, KernelFn kern, int radius) {
    for (size_t p = 0; p < out_len; p++) {
        /* source coordinate of output pixel p (pixel-center mapping) */
        float c = ((float)p + 0.5f) * scale - 0.5f;
        int i0 = (int)floorf(c) - radius + 1;
        SampleRow* r = &tbl[p];
        r->n = 0;
        r->wsum = 0.0f;
        for (int k = 0; k <= 2 * radius; k++) {
            int xi = i0 + k;
            float w = kern(c - (float)xi);
            if (w == 0.0f) continue;
            int ci = (xi < 0) ? 0 : (xi >= (int)in_len ? (int)in_len - 1 : xi);
            r->taps[r->n].idx = (size_t)ci;
            r->taps[r->n].w = w;
            r->wsum += w;
            r->n++;
        }
        if (r->wsum == 0.0f) r->wsum = 1.0f;
    }
}

/**
 * Generic separable resize with precomputed weight tables.
 * temp: in_h * out_w * C floats (caller).
 */
static int resize_separable(const RCRE_Tensor* src, RCRE_Tensor* dst,
                            KernelFn kern, int S,
                            float* temp, RCRE_Arena* arena) {
    size_t sh = src->shape[0], sw = src->shape[1];
    size_t dh = dst->shape[0], dw = dst->shape[1];
    size_t C  = (src->ndim >= 3) ? src->shape[2] : 1;
    if (C != ((dst->ndim >= 3) ? dst->shape[2] : 1)) return 0;

    if (sw == dw && sh == dh) {
        /* same size: identity copy */
        size_t n = sh * sw * C;
        if (src->dtype == RCRE_DTYPE_U8 && dst->dtype == RCRE_DTYPE_U8) {
            memcpy(dst->data, src->data, n);
        } else {
            for (size_t i = 0; i < n; i++) {
                float v = (src->dtype == RCRE_DTYPE_U8)
                              ? (float)((const uint8_t*)src->data)[i]
                              : ((const float*)src->data)[i];
                if (dst->dtype == RCRE_DTYPE_U8) {
                    int iv = (int)lrintf(v);
                    ((uint8_t*)dst->data)[i] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
                } else {
                    ((float*)dst->data)[i] = v;
                }
            }
        }
        return 1;
    }

    float sx = (float)sw / (float)dw;
    float sy = (float)sh / (float)dh;

    /* weight tables live on the arena (a few KB) */
    SampleRow* tx = (SampleRow*)rcre_arena_alloc(arena, dw * sizeof(SampleRow), 64);
    SampleRow* ty = (SampleRow*)rcre_arena_alloc(arena, dh * sizeof(SampleRow), 64);
    if (!tx || !ty) return 0;
    build_samples(tx, dw, sw, sx, kern, S);
    build_samples(ty, dh, sh, sy, kern, S);

    if (src->dtype == RCRE_DTYPE_U8) {
        const uint8_t* s = (const uint8_t*)src->data;
        /* ---- pass 1: horizontal (src rows -> temp rows of width dw) ---- */
        for (size_t y = 0; y < sh; y++) {
            const uint8_t* row = s + y * sw * C;
            float* trow = temp + y * dw * C;
            for (size_t x = 0; x < dw; x++) {
                const SampleRow* r = &tx[x];
                float inv = 1.0f / r->wsum;
                for (size_t ch = 0; ch < C; ch++) {
                    float acc = 0.0f;
                    for (int t = 0; t < r->n; t++)
                        acc += r->taps[t].w * (float)row[r->taps[t].idx * C + ch];
                    trow[x * C + ch] = acc * inv;
                }
            }
        }
        /* ---- pass 2: vertical (temp -> dst U8) ---- */
        uint8_t* d = (uint8_t*)dst->data;
        for (size_t x = 0; x < dw; x++) {
            for (size_t y = 0; y < dh; y++) {
                const SampleRow* ry = &ty[y];
                float inv = 1.0f / ry->wsum;
                for (size_t ch = 0; ch < C; ch++) {
                    float acc = 0.0f;
                    for (int t = 0; t < ry->n; t++)
                        acc += ry->taps[t].w * temp[(ry->taps[t].idx * dw + x) * C + ch];
                    int v = (int)lrintf(acc * inv);
                    d[(y * dw + x) * C + ch] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
        }
    } else {
        const float* s = (const float*)src->data;
        for (size_t y = 0; y < sh; y++) {
            const float* row = s + y * sw * C;
            float* trow = temp + y * dw * C;
            for (size_t x = 0; x < dw; x++) {
                const SampleRow* r = &tx[x];
                float inv = 1.0f / r->wsum;
                for (size_t ch = 0; ch < C; ch++) {
                    float acc = 0.0f;
                    for (int t = 0; t < r->n; t++)
                        acc += r->taps[t].w * row[r->taps[t].idx * C + ch];
                    trow[x * C + ch] = acc * inv;
                }
            }
        }
        float* d = (float*)dst->data;
        for (size_t x = 0; x < dw; x++) {
            for (size_t y = 0; y < dh; y++) {
                const SampleRow* ry = &ty[y];
                float inv = 1.0f / ry->wsum;
                for (size_t ch = 0; ch < C; ch++) {
                    float acc = 0.0f;
                    for (int t = 0; t < ry->n; t++)
                        acc += ry->taps[t].w * temp[(ry->taps[t].idx * dw + x) * C + ch];
                    d[(y * dw + x) * C + ch] = acc * inv;
                }
            }
        }
    }
    return 1;
}

int sr_resize_lanczos(const RCRE_Tensor* src, RCRE_Tensor* dst, RCRE_Arena* arena) {
    if (!src || !dst || !src->data || !dst->data || !arena) return 0;
    size_t sh = src->shape[0], sw = src->shape[1];
    size_t dh = dst->shape[0], dw = dst->shape[1];
    size_t C  = (src->ndim >= 3) ? src->shape[2] : 1;
    if (!sh || !sw || !dh || !dw) return 0;
    float* temp = (float*)rcre_arena_alloc(arena, sh * dw * C * sizeof(float), 64);
    if (!temp) return 0;
    return resize_separable(src, dst, lanczos3, 3, temp, arena);
}

int sr_resize_bicubic(const RCRE_Tensor* src, RCRE_Tensor* dst, RCRE_Arena* arena) {
    if (!src || !dst || !src->data || !dst->data || !arena) return 0;
    size_t sh = src->shape[0], sw = src->shape[1];
    size_t dh = dst->shape[0], dw = dst->shape[1];
    size_t C  = (src->ndim >= 3) ? src->shape[2] : 1;
    if (!sh || !sw || !dh || !dw) return 0;
    float* temp = (float*)rcre_arena_alloc(arena, sh * dw * C * sizeof(float), 64);
    if (!temp) return 0;
    return resize_separable(src, dst, cubic, 2, temp, arena);
}
