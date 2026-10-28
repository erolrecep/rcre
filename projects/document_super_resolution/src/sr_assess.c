/**
 * @file sr_assess.c
 * @brief Measured image statistics that drive the adaptive pipeline.
 *
 *   blur_score  : variance of the 5-tap Laplacian (HIGHER = sharper)
 *   noise_sigma : robust estimator  sigma = 1.4826 * MAD(L) / sqrt(20)
 *   contrast    : (p98 - p2) percentile range, normalized 0..1
 *   skew_deg    : projection-profile maximization on the ink mask
 *                 (coarse 0.25° scan then 0.02° refine)
 *
 * All estimates run on a subsampled copy (<= 1000 px) so assessment stays
 * cheap (~10-30 ms) even on 4K input.
 */
#include "sr.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SR_ASSESS_MAX_SIDE 1000

static float px8(const RCRE_Tensor* t, size_t y, size_t x) {
    size_t c = (t->ndim >= 3) ? t->shape[2] : 1;
    if (t->dtype == RCRE_DTYPE_U8) {
        const uint8_t* p = (const uint8_t*)t->data;
        return (float)p[(y * t->shape[1] + x) * c];
    }
    const float* p = (const float*)t->data;
    return p[(y * t->shape[1] + x) * c];
}

/* Downscale to <= max_side with box averaging (F32 gray out, 1ch). */
static int subsample_gray(const RCRE_Tensor* src, size_t* out_w, size_t* out_h,
                          float** out_buf, RCRE_Arena* arena) {
    size_t sw = src->shape[1], sh = src->shape[0];
    size_t max_side = (sw > sh) ? sw : sh;
    float ratio = (max_side <= SR_ASSESS_MAX_SIDE)
                      ? 1.0f
                      : (float)SR_ASSESS_MAX_SIDE / (float)max_side;
    size_t ow = (size_t)ceil((double)sw * ratio);
    size_t oh = (size_t)ceil((double)sh * ratio);
    if (ow < 4) ow = 4;
    if (oh < 4) oh = 4;

    float* buf = (float*)rcre_arena_alloc(arena, ow * oh * sizeof(float), 64);
    if (!buf) return 0;

    for (size_t y = 0; y < oh; y++) {
        size_t y0 = (size_t)((double)y / ratio);
        size_t y1 = (size_t)((double)(y + 1) / ratio);
        if (y1 <= y0) y1 = y0 + 1;
        if (y1 > sh) y1 = sh;
        for (size_t x = 0; x < ow; x++) {
            size_t x0 = (size_t)((double)x / ratio);
            size_t x1 = (size_t)((double)(x + 1) / ratio);
            if (x1 <= x0) x1 = x0 + 1;
            if (x1 > sw) x1 = sw;
            double s = 0.0;
            int cnt = 0;
            for (size_t yy = y0; yy < y1; yy += (y1 - y0 > 8 ? (y1 - y0) / 8 : 1)) {
                for (size_t xx = x0; xx < x1; xx += (x1 - x0 > 8 ? (x1 - x0) / 8 : 1)) {
                    s += px8(src, yy, xx);
                    cnt++;
                }
            }
            buf[y * ow + x] = (float)(s / (cnt ? cnt : 1));
        }
    }
    *out_w = ow; *out_h = oh; *out_buf = buf;
    return 1;
}

/* 5-tap Laplacian response, clamped border (reflect by clamp). */
static void laplacian(const float* img, size_t w, size_t h, float* out) {
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            float c  = img[y * w + x];
            float up = img[(y > 0 ? y - 1 : y) * w + x];
            float dn = img[(y + 1 < h ? y + 1 : y) * w + x];
            float lf = img[y * w + (x > 0 ? x - 1 : x)];
            float rt = img[y * w + (x + 1 < w ? x + 1 : x)];
            out[y * w + x] = up + dn + lf + rt - 4.0f * c;
        }
    }
}

/* Otsu threshold on a 256-bin histogram (textbook between-class variance).
 * Returns the MIDPOINT of the maximizing plateau — robust for bimodal
 * histograms where a range of thresholds ties. */
static float otsu_of_hist(const size_t* hist256) {
    double total = 0;
    for (int i = 0; i < 256; i++) total += hist256[i];
    if (total <= 0) return 127.0f;
    double mu_total = 0;
    for (int t = 0; t < 256; t++) mu_total += (double)t * hist256[t];
    mu_total /= total;
    double sumB = 0, wB = 0, bestVar = -1.0;
    int tFirst = 127, tLast = 127;
    for (int t = 0; t < 256; t++) {
        wB += hist256[t];
        if (wB <= 0) continue;
        double wF = total - wB;
        if (wF <= 0) break;
        sumB += (double)t * hist256[t];
        double mB = sumB / wB;
        double mF = (mu_total * total - sumB) / wF;
        double var = wB * wF * (mB - mF) * (mB - mF);
        if (var > bestVar) { bestVar = var; tFirst = t; tLast = t; }
        else if (var == bestVar) { tLast = t; }
    }
    return (float)((double)tFirst + (double)tLast) / 2.0f;
}

/* Score an angle by sum of squared row sums of the rotated ink mask. */
static double skew_score_at(const uint8_t* mask, size_t w, size_t h,
                            float cos_a, float sin_a, float* row_acc) {
    memset(row_acc, 0, h * sizeof(float));
    double cx = (double)w / 2.0, cy = (double)h / 2.0;
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            if (!mask[y * w + x]) continue;
            /* map dst -> src: rotate by -a around center */
            double dx = (double)x - cx, dy = (double)y - cy;
            double sx = cos_a * dx + sin_a * dy;
            double sy = -sin_a * dx + cos_a * dy;
            int ix = (int)lrint(cx + sx);
            int iy = (int)lrint(cy + sy);
            if (ix < 0 || iy < 0 || ix >= (int)w || iy >= (int)h) continue;
            if (mask[(size_t)iy * w + (size_t)ix]) row_acc[y] += 1.0f;
        }
    }
    double score = 0;
    for (size_t y = 0; y < h; y++) score += (double)row_acc[y] * (double)row_acc[y];
    return score;
}

int sr_assess(const RCRE_Tensor* gray, SRAssessment* out) {
    if (!gray || !gray->data || !out) return 0;
    if (gray->shape[0] < 4 || gray->shape[1] < 4) return 0;

    size_t w = 0, h = 0;
    float* g = NULL;
    RCRE_Arena* scratch = rcre_arena_create(1 << 24);
    if (!scratch) return 0;
    if (!subsample_gray(gray, &w, &h, &g, scratch)) {
        rcre_arena_destroy(scratch);
        return 0;
    }
    size_t n = w * h;

    float* lap = (float*)rcre_arena_alloc(scratch, n * sizeof(float), 64);
    if (!lap) { rcre_arena_destroy(scratch); return 0; }
    laplacian(g, w, h, lap);

    /* blur score: variance of L */
    double mean = 0;
    for (size_t i = 0; i < n; i++) mean += lap[i];
    mean /= (double)n;
    double var = 0;
    for (size_t i = 0; i < n; i++) var += (double)(lap[i] - mean) * (lap[i] - mean);
    var /= (double)n;

    /* noise: robust sigma from MAD of L */
    float* absL = (float*)rcre_arena_alloc(scratch, n * sizeof(float), 64);
    double maxAbs = 0;
    for (size_t i = 0; i < n; i++) {
        absL[i] = fabsf(lap[i]);
        if (absL[i] > maxAbs) maxAbs = absL[i];
    }
    /* median via 256-bin histogram of |L| scaled to [0, 1] */
    float med = 0;
    if (maxAbs > 0) {
        size_t bins[512];
        memset(bins, 0, sizeof(bins));
        for (size_t i = 0; i < n; i++) {
            size_t b = (size_t)((absL[i] / maxAbs) * 511.0);
            bins[b]++;
        }
        size_t half = n / 2, acc = 0;
        for (size_t b = 0; b < 512; b++) {
            acc += bins[b];
            if (acc >= half) { med = (float)(b + 0.5) / 512.0f * (float)maxAbs; break; }
        }
    }
    float noise_sigma = (float)(1.4826 * (double)med / sqrt(20.0));
    if (noise_sigma > 40.0f) noise_sigma = 40.0f; /* sanity clamp */

    /* contrast: p2..p98 */
    {
        size_t hist[256];
        memset(hist, 0, sizeof(hist));
        for (size_t i = 0; i < n; i++) {
            int v = (int)lrint(g[i]);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            hist[v]++;
        }
        size_t p2 = 0, p98 = 0, acc = 0;
        for (int v = 0; v < 256; v++) {
            acc += hist[v];
            if (p2 == 0 && acc >= (size_t)(0.02 * n)) p2 = (size_t)v;
            if (acc >= (size_t)(0.98 * n)) { p98 = (size_t)v; break; }
        }
        out->contrast = (float)((double)p98 - (double)p2) / 255.0;
    }

    /* skew: ink mask via Otsu on subsampled gray */
    float skew = 0.0f;
    if (w >= 32 && h >= 32) {
        size_t hist[256];
        memset(hist, 0, sizeof(hist));
        for (size_t i = 0; i < n; i++) {
            int v = (int)lrint(g[i]);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            hist[v]++;
        }
        float thr = otsu_of_hist(hist);
        uint8_t* mask = (uint8_t*)rcre_arena_alloc(scratch, n, 64);
        size_t ink = 0;
        for (size_t i = 0; i < n; i++) {
            mask[i] = (g[i] < thr) ? 1 : 0; /* dark text on light paper */
            ink += mask[i];
        }
        /* only meaningful when ink is a modest fraction (a document, not a photo of a wall) */
        if (ink > n * 0.005 && ink < n * 0.5) {
            float* row_acc = (float*)rcre_arena_alloc(scratch, h * sizeof(float), 64);
            if (row_acc) {
                /* coarse scan */
                double best = -1.0;
                float best_a = 0.0f;
                for (float a = -7.5f; a <= 7.51f; a += 0.25f) {
                    double sc = skew_score_at(mask, w, h,
                                              cosf(a * (float)M_PI / 180.0f),
                                              sinf(a * (float)M_PI / 180.0f), row_acc);
                    if (sc > best) { best = sc; best_a = a; }
                }
                /* fine refine */
                for (float a = best_a - 0.25f; a <= best_a + 0.26f; a += 0.02f) {
                    double sc = skew_score_at(mask, w, h,
                                              cosf(a * (float)M_PI / 180.0f),
                                              sinf(a * (float)M_PI / 180.0f), row_acc);
                    if (sc > best) { best = sc; best_a = a; }
                }
                skew = best_a;
            }
        }
    }

    out->w = (int)w;
    out->h = (int)h;
    out->blur_score = (float)var;
    out->noise_sigma = noise_sigma;
    out->skew_deg = skew;

    rcre_arena_destroy(scratch);
    return 1;
}
