/**
 * @file sr_cli.c
 * @brief RCRE-SR command-line demo & tool.
 *
 *   sr_cli generate <orig> <degraded> [w] [h]
 *       Renders a sharp synthetic document (embedded 5x7 font) and a
 *       degraded "bad photo" of it (downscale + blur + noise + skew +
 *       illumination tilt + bilinear upscale back).
 *
 *   sr_cli enhance <in> <out> [--scale N] [--style auto|sharp|clean|binarize|upscale]
 *                   [--no-deskew] [--no-deblur] [--sharpen F] [--gt REF]
 *       Runs the adaptive pipeline, writes the result, prints the report;
 *       with --gt also prints PSNR/SSIM vs the reference (auto-resized).
 *
 *   sr_cli assess <in>
 *       Prints the measured image statistics.
 *
 *   sr_cli demo
 *       Self-contained: generate → enhance → PSNR/SSIM → Tesseract OCR
 *       comparison (when tesseract is on PATH) → full report.
 */
#include "sr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ========================================================================= */
/* Embedded 5x7 bitmap font (uppercase + digits + punctuation)               */
/* ========================================================================= */

typedef struct { char ch; char rows[7][7]; } Glyph; /* row: 5 chars + NUL */

/* =========================================================================
 * Mini vector stroke font — characters as line segments in a unit box
 * (x: 0..1 across 4.2 cells, y: 0..1 across 6 cells), rasterized with
 * anti-aliased thick strokes. Much closer to real type than the old 5x7
 * bitmap, so OCR engines (Tesseract) can actually read the document.
 * ========================================================================= */

typedef struct { float u0, v0, u1, v1; } VSeg;   /* segment, unit box */
typedef struct { float u, v, r; }      VDot;     /* circle, r in cell units */
typedef struct {
    char ch;
    const VSeg* segs;
    int nsegs;
    const VDot* dots;
    int ndots;
} VGlyph;

#define VS(a,b,c,d) { a, b, c, d }
#define VD(a,b,c)   { a, b, c }

static const VSeg G_A[] = { VS(0,1,.5,0), VS(.5,0,1,1), VS(.22,.62,.78,.62) };
static const VSeg G_B[] = { VS(0,0,0,1), VS(0,0,.78,0), VS(.78,0,.78,.42), VS(.78,.42,0,.42),
                            VS(0,.42,.92,.42), VS(.92,.42,.92,1), VS(.92,1,0,1) };
static const VSeg G_C[] = { VS(1,.14,.72,0), VS(.38,0,.14,.14), VS(.04,.3,.04,.7),
                            VS(.14,.86,.38,1), VS(.72,1,1,.86) };
static const VSeg G_D[] = { VS(0,0,0,1), VS(0,0,.5,0), VS(.82,0,1,.28), VS(1,.32,1,.68),
                            VS(1,.72,.82,1), VS(.5,1,0,1) };
static const VSeg G_E[] = { VS(1,0,0,0), VS(0,0,0,1), VS(0,1,1,1), VS(0,.5,.78,.5) };
static const VSeg G_F[] = { VS(1,0,0,0), VS(0,0,0,1), VS(0,.5,.78,.5) };
static const VSeg G_G[] = { VS(1,.14,.72,0), VS(.38,0,.14,.14), VS(.04,.3,.04,.7),
                            VS(.14,.86,.38,1), VS(.78,1,1,.8), VS(1,.8,1,.55), VS(1,.55,.55,.55) };
static const VSeg G_H[] = { VS(0,0,0,1), VS(1,0,1,1), VS(0,.5,1,.5) };
static const VSeg G_I[] = { VS(.5,0,.5,1), VS(.28,0,.72,0), VS(.28,1,.72,1) };
static const VSeg G_K[] = { VS(0,0,0,1), VS(1,0,.06,.52), VS(.3,.58,1,1) };
static const VSeg G_L[] = { VS(0,0,0,1), VS(0,1,1,1) };
static const VSeg G_M[] = { VS(0,1,0,0), VS(0,0,.5,.55), VS(.5,.55,1,0), VS(1,0,1,1) };
static const VSeg G_N[] = { VS(0,1,0,0), VS(0,0,1,1), VS(1,1,1,0) };
static const VSeg G_O[] = { VS(.5,0,.9,.2), VS(1,.3,1,.7), VS(.9,.8,.5,1),
                            VS(.1,.8,0,.7), VS(0,.3,0,.2), VS(.1,0,.5,0) };
static const VSeg G_P[] = { VS(0,1,0,0), VS(0,0,.8,0), VS(1,.16,1,.4), VS(.8,.5,0,.5) };
static const VSeg G_Q[] = { VS(.5,0,.9,.2), VS(1,.3,1,.7), VS(.9,.8,.5,1), VS(.1,.8,0,.7),
                            VS(0,.3,0,.2), VS(.1,0,.5,0), VS(.6,.72,1,1) };
static const VSeg G_R[] = { VS(0,1,0,0), VS(0,0,.8,0), VS(1,.16,1,.4), VS(.8,.5,0,.5),
                            VS(.42,.5,1,1) };
static const VSeg G_S[] = { VS(1,.12,.8,0), VS(.4,0,.1,.12), VS(0,.26,.14,.4),
                            VS(.4,.5,.72,.55), VS(1,.66,1,.8), VS(.7,1,.35,1), VS(.02,.88,0,.74) };
static const VSeg G_T[] = { VS(0,0,1,0), VS(.5,0,.5,1) };
static const VSeg G_U[] = { VS(0,0,0,.7), VS(.1,.9,.3,1), VS(.7,1,.9,.9), VS(1,.7,1,0) };
static const VSeg G_W[] = { VS(0,0,.2,1), VS(.34,.5,.66,.5), VS(.8,1,1,0) };
static const VSeg G_X[] = { VS(0,0,1,1), VS(1,0,0,1) };
static const VSeg G_Y[] = { VS(0,0,.5,.5), VS(.5,.5,1,0), VS(.5,.5,.5,1) };
static const VSeg G_0[] = { VS(.5,0,.9,.2), VS(1,.3,1,.7), VS(.9,.8,.5,1), VS(.1,.8,0,.7),
                            VS(0,.3,0,.2), VS(.1,0,.5,0), VS(.72,.28,.28,.72) };
static const VSeg G_1[] = { VS(.28,.14,.5,0), VS(.5,0,.5,1), VS(.3,1,.7,1) };
static const VSeg G_2[] = { VS(.1,.14,.32,0), VS(.7,0,.9,.14), VS(1,.3,.6,.55),
                            VS(.18,.82,0,.92), VS(0,1,1,1) };
static const VSeg G_3[] = { VS(.1,.1,.36,0), VS(.74,0,.9,.14), VS(1,.3,.72,.42),
                            VS(.48,.5,.74,.58), VS(1,.7,.9,.84), VS(.7,1,.36,1), VS(.1,.9,.06,.76) };
static const VSeg G_4[] = { VS(.78,0,.78,1), VS(.78,0,0,.72), VS(0,.72,1,.72) };
static const VSeg G_5[] = { VS(1,0,0,0), VS(0,0,.06,.42), VS(.3,.52,1,.56),
                            VS(1,.62,1,.8), VS(.7,1,.38,1), VS(.1,.9,.06,.74) };
static const VSeg G_6[] = { VS(.74,.06,.4,0), VS(.12,.1,0,.34), VS(.04,.5,.08,.74),
                            VS(.2,1,.5,1), VS(.82,1,.94,.84), VS(1,.64,.86,.46), VS(.6,.32,.12,.4) };
static const VSeg G_7[] = { VS(0,0,1,0), VS(.62,0,.3,1) };
static const VSeg G_8[] = { VS(.5,0,.84,.1), VS(1,.26,.8,.42), VS(.2,.42,0,.26), VS(.16,.1,.5,0),
                            VS(.2,.5,.8,.5), VS(1,.66,1,.8), VS(.8,1,.2,1), VS(0,.8,0,.66) };
static const VSeg G_9[] = { VS(.26,0,.78,0), VS(.98,.14,1,.4), VS(.9,.64,.5,1), VS(.2,.9,.1,.76) };
static const VSeg G_DASH[] = { VS(.25,.5,.75,.5) };
static const VSeg G_SLASH[] = { VS(.75,0,.25,1) };
static const VSeg G_PLUS[] = { VS(.5,.28,.5,.72), VS(.28,.5,.72,.5) };
static const VSeg G_PCT_S[] = { VS(0,1,1,0) };
static const VDot G_PCT_D[] = { VD(.22,.24,.16), VD(.78,.76,.16) };
static const VDot G_DOT_D[] = { VD(.5,.85,.14) };
static const VDot G_COL_D[] = { VD(.5,.3,.14), VD(.5,.68,.14) };
static const VDot G_COM_D[] = { VD(.5,.8,.14) };
static const VSeg G_PAREN_L[] = { VS(.62,.1,.34,.3), VS(.24,.45,.24,.55), VS(.34,.7,.62,.9) };
static const VSeg G_PAREN_R[] = { VS(.38,.1,.66,.3), VS(.76,.45,.76,.55), VS(.66,.7,.38,.9) };

static const VGlyph VFONT[] = {
    {'A', G_A, 3, NULL, 0}, {'B', G_B, 7, NULL, 0}, {'C', G_C, 5, NULL, 0},
    {'D', G_D, 6, NULL, 0}, {'E', G_E, 4, NULL, 0}, {'F', G_F, 3, NULL, 0},
    {'G', G_G, 7, NULL, 0}, {'H', G_H, 3, NULL, 0}, {'I', G_I, 3, NULL, 0},
    {'K', G_K, 3, NULL, 0}, {'L', G_L, 2, NULL, 0}, {'M', G_M, 4, NULL, 0},
    {'N', G_N, 3, NULL, 0}, {'O', G_O, 6, NULL, 0}, {'P', G_P, 4, NULL, 0},
    {'Q', G_Q, 7, NULL, 0}, {'R', G_R, 5, NULL, 0}, {'S', G_S, 7, NULL, 0},
    {'T', G_T, 2, NULL, 0}, {'U', G_U, 4, NULL, 0}, {'W', G_W, 3, NULL, 0},
    {'X', G_X, 2, NULL, 0}, {'Y', G_Y, 3, NULL, 0},
    {'0', G_0, 7, NULL, 0}, {'1', G_1, 3, NULL, 0}, {'2', G_2, 5, NULL, 0},
    {'3', G_3, 7, NULL, 0}, {'4', G_4, 3, NULL, 0}, {'5', G_5, 6, NULL, 0},
    {'6', G_6, 7, NULL, 0}, {'7', G_7, 2, NULL, 0}, {'8', G_8, 8, NULL, 0},
    {'9', G_9, 4, NULL, 0},
    {'-', G_DASH, 1, NULL, 0}, {'/', G_SLASH, 1, NULL, 0}, {'+', G_PLUS, 2, NULL, 0},
    {'%', G_PCT_S, 1, G_PCT_D, 2}, {'.', NULL, 0, G_DOT_D, 1},
    {':', NULL, 0, G_COL_D, 2}, {',', NULL, 0, G_COM_D, 1},
    {'(', G_PAREN_L, 3, NULL, 0}, {')', G_PAREN_R, 3, NULL, 0},
    {' ', NULL, 0, NULL, 0}, {'?', NULL, 0, NULL, 0},
};
#define VFONT_N ((int)(sizeof(VFONT) / sizeof(VFONT[0])))

static const VGlyph* vglyph_for(char ch) {
    if (ch >= 'a' && ch <= 'z') ch = (char)(ch - 'a' + 'A');
    for (int i = 0; i < VFONT_N; i++)
        if (VFONT[i].ch == ch) return &VFONT[i];
    for (int i = 0; i < VFONT_N; i++)
        if (VFONT[i].ch == '?') return &VFONT[i];
    return &VFONT[0];
}


/* ========================================================================= */
/* Document rendering (vector-free: 5x7 bitmap, supersampled & AA-rasterized) */
/* ========================================================================= */

#define SS 4 /* supersampling factor for crisp AA edges */

static uint64_t rnd_state = 12345;
static double frand(void) {
    rnd_state = rnd_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rnd_state >> 11) / (double)0x10000000000000ULL;
}
static double frandn(void) { /* approx gaussian via 3-uniform */
    return (frand() + frand() + frand()) / 3.0 * 2.0 - 1.0;
}

typedef struct {
    const char* text;
    int x, y;        /* top-left in doc px */
    int scale;       /* font scale (px per font pixel) */
    int ink;         /* 0..255 */
} TextItem;

static float seg_dist_px(float px, float py, float ax, float ay, float bx, float by) {
    float dx = bx - ax, dy = by - ay;
    float len2 = dx * dx + dy * dy;
    float t = (len2 > 1e-6f) ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0.0f;
    if (t < 0.0f) t = 0.0f;
    else if (t > 1.0f) t = 1.0f;
    float ex = px - (ax + t * dx), ey = py - (ay + t * dy);
    return sqrtf(ex * ex + ey * ey);
}

static void render_doc(float* img, size_t W, size_t H,
                       const TextItem* items, int n_items,
                       int doc_pad, int scale_base) {
    for (size_t i = 0; i < W * H; i++) img[i] = 255.0f; /* white paper (NOT memset!) */
    (void)doc_pad;
    for (int it = 0; it < n_items; it++) {
        const char* s = items[it].text;
        int px = items[it].x, py = items[it].y;
        int sc = items[it].scale > 0 ? items[it].scale : scale_base;
        int ink = items[it].ink;
        float boxw = 4.2f * (float)sc, boxh = 6.0f * (float)sc;
        int adv = 5 * sc;
        float halfw = 0.52f * (float)sc; /* stroke half-width */
        float aa = 0.75f;                /* AA ramp (px) */
        for (const char* p = s; *p; p++) {
            const VGlyph* g = vglyph_for(*p);
            if (g->nsegs == 0 && g->ndots == 0) { px += adv; continue; }
            int x0 = px, y0 = py;
            int x1 = px + (int)boxw, y1 = py + (int)boxh;
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (x1 >= (int)W) x1 = (int)W - 1;
            if (y1 >= (int)H) y1 = (int)H - 1;
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    float fu = (float)x - px + 0.5f;
                    float fv = (float)y - py + 0.5f;
                    float best = 0.0f;
                    for (int k = 0; k < g->nsegs; k++) {
                        const VSeg* sg = &g->segs[k];
                        float d = seg_dist_px(fu, fv,
                                              sg->u0 * boxw, sg->v0 * boxh,
                                              sg->u1 * boxw, sg->v1 * boxh);
                        float c = halfw + aa - d;
                        if (c > best) best = c;
                    }
                    for (int k = 0; k < g->ndots; k++) {
                        const VDot* dt = &g->dots[k];
                        float dx = fu - dt->u * boxw, dy = fv - dt->v * boxh;
                        float d = sqrtf(dx * dx + dy * dy);
                        float c = dt->r * (float)sc + aa - d;
                        if (c > best) best = c;
                    }
                    if (best <= 0.0f) continue;
                    if (best > 1.0f) best = 1.0f;
                    float cur = img[y * W + x];
                    img[y * W + x] = cur * (1.0f - best) + (float)ink * best;
                }
            }
            px += adv;
        }
    }
}

/* ========================================================================= */
/* Degradation: downscale -> blur -> noise -> skew -> illumination -> up      */
/* ========================================================================= */

static float* box_downscale(const float* in, size_t iw, size_t ih,
                            size_t ow, size_t oh, RCRE_Arena* a) {
    float* out = (float*)rcre_arena_alloc(a, ow * oh * sizeof(float), 64);
    if (!out) return NULL;
    for (size_t y = 0; y < oh; y++) {
        size_t y0 = ih * y / oh, y1 = ih * (y + 1) / oh;
        if (y1 <= y0) y1 = y0 + 1;
        for (size_t x = 0; x < ow; x++) {
            size_t x0 = iw * x / ow, x1 = iw * (x + 1) / ow;
            if (x1 <= x0) x1 = x0 + 1;
            double s = 0;
            size_t cnt = 0;
            for (size_t yy = y0; yy < y1; yy += (y1 - y0 > 16 ? (y1 - y0) / 16 : 1))
                for (size_t xx = x0; xx < x1; xx += (x1 - x0 > 16 ? (x1 - x0) / 16 : 1)) {
                    s += in[yy * iw + xx];
                    cnt++;
                }
            out[y * ow + x] = (float)(s / (cnt ? cnt : 1));
        }
    }
    return out;
}

static void gauss2d(float* img, size_t w, size_t h, float sigma, RCRE_Arena* a) {
    int r = (int)ceil(sigma * 3.0f);
    if (r < 1) r = 1;
    if (r > 12) r = 12;
    float* k = (float*)rcre_arena_alloc(a, (2 * r + 1) * sizeof(float), 8);
    float* tmp = (float*)rcre_arena_alloc(a, w * h * sizeof(float), 64);
    if (!k || !tmp) return;
    float ks = 0;
    for (int i = -r; i <= r; i++) { k[i + r] = expf(-(i * i) / (2 * sigma * sigma)); ks += k[i + r]; }
    for (int i = -r; i <= r; i++) k[i + r] /= ks;
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++) {
            float s = 0;
            for (int i = -r; i <= r; i++) {
                int xx = (int)x + i;
                if (xx < 0) xx = 0;
                if (xx >= (int)w) xx = (int)w - 1;
                s += img[y * w + xx] * k[i + r];
            }
            tmp[y * w + x] = s;
        }
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++) {
            float s = 0;
            for (int i = -r; i <= r; i++) {
                int yy = (int)y + i;
                if (yy < 0) yy = 0;
                if (yy >= (int)h) yy = (int)h - 1;
                s += tmp[yy * w + x] * k[i + r];
            }
            img[y * w + x] = s;
        }
}

static void rotate_bilinear(const float* in, size_t w, size_t h, float angle_deg,
                            float* out, RCRE_Arena* a) {
    (void)a;
    double cx = w / 2.0, cy = h / 2.0;
    double ca = cos(angle_deg * M_PI / 180.0), sa = sin(angle_deg * M_PI / 180.0);
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++) {
            double dx = x - cx, dy = y - cy;
            double sx = ca * dx + sa * dy;
            double sy = -sa * dx + ca * dy;
            double fx = cx + sx, fy = cy + sy;
            float v = 255.0f; /* white background */
            if (fx >= 0 && fy >= 0 && fx < (double)w - 1 && fy < (double)h - 1) {
                int x0 = (int)fx, y0 = (int)fy;
                float tx = (float)(fx - x0), ty = (float)(fy - y0);
                float p00 = in[y0 * w + x0], p10 = in[y0 * w + x0 + 1];
                float p01 = in[(y0 + 1) * w + x0], p11 = in[(y0 + 1) * w + x0 + 1];
                v = p00 * (1 - tx) * (1 - ty) + p10 * tx * (1 - ty) +
                    p01 * (1 - tx) * ty + p11 * tx * ty;
            }
            out[y * w + x] = v;
        }
}

/* Degradation recipe; returns float image of size W x H (arena). */
static float* degrade(const float* orig, size_t W, size_t H, RCRE_Arena* a,
                      uint64_t seed) {
    rnd_state = (double)seed;
    size_t lw = (W * 3) / 5, lh = (H * 3) / 5;   /* ~60% "photo" */
    float* low = box_downscale(orig, W, H, lw, lh, a);
    if (!low) return NULL;
    gauss2d(low, lw, lh, 0.8f, a);
    for (size_t i = 0; i < lw * lh; i++) {
        low[i] += frandn() * 8.0;       /* sensor-ish noise */
        if (low[i] < 0) low[i] = 0;
        if (low[i] > 255) low[i] = 255;
    }
    /* slight skew */
    float skew = (float)(frand() * 4.0 - 2.0);
    float* rot = (float*)rcre_arena_alloc(a, lw * lh * sizeof(float), 64);
    rotate_bilinear(low, lw, lh, skew, rot, a);

    /* back up to full size with bilinear (soft) */
    float* up = (float*)rcre_arena_alloc(a, W * H * sizeof(float), 64);
    if (!up) return NULL;
    for (size_t y = 0; y < H; y++)
        for (size_t x = 0; x < W; x++) {
            double fx = (double)x * ((double)lw / (double)W);
            double fy = (double)y * ((double)lh / (double)H);
            int x0 = (int)fx, y0 = (int)fy;
            if (x0 >= (int)lw - 1) x0 = (int)lw - 2;
            if (y0 >= (int)lh - 1) y0 = (int)lh - 2;
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            float tx = (float)(fx - x0), ty = (float)(fy - y0);
            up[y * W + x] = rot[y0 * lw + x0] * (1 - tx) * (1 - ty) +
                            rot[y0 * lw + x0 + 1] * tx * (1 - ty) +
                            rot[(y0 + 1) * lw + x0] * (1 - tx) * ty +
                            rot[(y0 + 1) * lw + x0 + 1] * tx * ty;
        }
    /* uneven illumination (light falloff) */
    for (size_t y = 0; y < H; y++)
        for (size_t x = 0; x < W; x++) {
            double gx = (double)x / (double)W - 0.5;
            double gy = (double)y / (double)H - 0.5;
            double fall = 1.0 + 0.15 * (gx * gx * 2.0) + 0.12 * (gy * gy * 2.0);
            up[y * W + x] *= (float)fall;
            if (up[y * W + x] > 255) up[y * W + x] = 255;
        }
    return up;
}

/* ========================================================================= */
/* Document content                                                           */
/* ========================================================================= */

static void build_content(TextItem* items, int* n, size_t W) {
    int pad = (int)(W / 30);
    int sc  = (int)(W / 100);
    if (sc < 3) sc = 3;
    int y = pad;
    int i = 0;
    #define ADD(txt_, xx_, yy_, sc_, ink_) do { \
        items[i].text = (txt_); items[i].x = (xx_); items[i].y = (yy_); \
        items[i].scale = (sc_); items[i].ink = (ink_); i++; } while (0)

    ADD("RCRE-SR TEST DOCUMENT", pad, y, sc * 2, 20); y += sc * 2 * 7 + sc;
    ADD("SPEC SHEET - IMAGE SUPER RESOLUTION BENCHMARK", pad, y, sc, 40);
    y += sc * 7 + sc * 2;

    const char* lines[] = {
        "1. PURPOSE: VERIFY OCR ACCURACY AFTER ENHANCEMENT.",
        "2. INPUT SCAN QUALITY: 60% RESOLUTION, MOTION BLUR",
        "   AND SENSOR NOISE WITH UNEVEN ILLUMINATION.",
        "3. EXPECTED: DESKEW, DENOISE, DEBLUR, LANCZOS X2",
        "   UPSCALE, EDGE-MASKED SHARPEN, CLAHE CONTRAST.",
        "4. METRICS: PSNR, SSIM, AND TESSERACT CER/ACC.",
        "5. GATE: PIPELINE MUST NOT REGRESS OCR VS INPUT.",
    };
    for (unsigned l = 0; l < sizeof(lines) / sizeof(lines[0]); l++) {
        ADD(lines[l], pad, y, sc, 30);
        y += sc * 7 + sc;
    }
    y += sc;
    const char* kv[] = {
        "ORDER   : RCRE-2026-0042",
        "MODEL   : QWEN3-CODER-NEXT (REFERENCE)",
        "SCHEMA  : 1200 X 1600 PX  GRAY  U8",
        "YIELD   : 98.2 %   TOL  : +-0.05",
        "SN      : 0042-77-88-99",
        "RATE    : 42.50 / UNIT   TOTAL : 1,785.00",
    };
    for (unsigned l = 0; l < sizeof(kv) / sizeof(kv[0]); l++) {
        ADD(kv[l], pad, y, sc, 30);
        y += sc * 7 + sc / 2;
    }
    *n = i;
    #undef ADD
}

/* ========================================================================= */
/* Commands                                                                   */
/* ========================================================================= */

int cmd_generate(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: generate <orig.pgm> <degraded.pgm> [w] [h]\n"); return 1; }
    size_t W = argc >= 3 ? atoi(argv[2]) : 1200;
    size_t H = argc >= 4 ? atoi(argv[3]) : 1600;
    if (W < 100 || H < 100) { fprintf(stderr, "size too small\n"); return 1; }

    RCRE_Arena* a = rcre_arena_create((size_t)W * H * 12); /* scratch only */
    if (!a) return 1;

    /* full-frame buffers live off-arena (arena holds scratch + the degraded
     * result only, so the working set stays ~1.5x the frame size) */
    float* orig = (float*)malloc(W * H * sizeof(float));
    if (!orig) { rcre_arena_destroy(a); return 1; }
    for (size_t i = 0; i < W * H; i++) orig[i] = 255.0f;
    TextItem items[64];
    int n = 0;
    build_content(items, &n, W);
    render_doc(orig, W, H, items, n, 0, 4);

    float* deg = degrade(orig, W, H, a, 0xC0FFEE);
    if (!deg) {
        fprintf(stderr, "degrade: arena scratch exhausted\n");
        free(orig);
        rcre_arena_destroy(a);
        return 1;
    }

    /* U8 views for PNM write */
    size_t shape[3] = {H, W, 1};
    RCRE_Tensor uO = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
    RCRE_Tensor uD = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
    if (!uO.data || !uD.data) {
        fprintf(stderr, "U8 tensor alloc failed\n");
        free(orig);
        rcre_arena_destroy(a);
        return 1;
    }
    for (size_t i = 0; i < W * H; i++) {
        int vo = (int)lrintf(orig[i]);
        int vd = (int)lrintf(deg[i]);
        ((uint8_t*)uO.data)[i] = (uint8_t)(vo < 0 ? 0 : (vo > 255 ? 255 : vo));
        ((uint8_t*)uD.data)[i] = (uint8_t)(vd < 0 ? 0 : (vd > 255 ? 255 : vd));
    }
    free(orig);
    if (!sr_pnm_write(argv[0], &uO) || !sr_pnm_write(argv[1], &uD)) {
        rcre_arena_destroy(a);
        return 1;
    }
    rcre_arena_destroy(a);
    printf("✅ wrote %s (sharp) and %s (degraded: 60%% res + blur + noise + skew + light fall)\n",
           argv[0], argv[1]);
    return 0;
}

static int parse_flags(int argc, char** argv, int start,
                       int* scale, SR_Style* style, bool* deskew,
                       bool* deblur, float* sharpen, const char** gt) {
    for (int i = start; i < argc; i++) {
        if (!strcmp(argv[i], "--scale") && i + 1 < argc) *scale = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--style") && i + 1 < argc) {
            const char* s = argv[++i];
            if (!strcmp(s, "auto")) *style = SR_STYLE_AUTO;
            else if (!strcmp(s, "sharp")) *style = SR_STYLE_SHARP;
            else if (!strcmp(s, "clean")) *style = SR_STYLE_CLEAN;
            else if (!strcmp(s, "binarize")) *style = SR_STYLE_BINARIZE;
            else if (!strcmp(s, "upscale")) *style = SR_STYLE_UPSCALE;
        }
        else if (!strcmp(argv[i], "--no-deskew")) *deskew = false;
        else if (!strcmp(argv[i], "--no-deblur")) *deblur = false;
        else if (!strcmp(argv[i], "--sharpen") && i + 1 < argc) *sharpen = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--gt") && i + 1 < argc) *gt = argv[++i];
    }
    return 0;
}

int cmd_enhance(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: enhance <in.pgm> <out.pgm> [--scale 2] [--style auto] "
                        "[--no-deskew] [--no-deblur] [--sharpen F] [--gt REF]\n");
        return 1;
    }
    int scale = 2;
    SR_Style style = SR_STYLE_AUTO;
    bool deskew = true, deblur = true;
    float sharpen = -1.0f;
    const char* gt = NULL;
    parse_flags(argc, argv, 2, &scale, &style, &deskew, &deblur, &sharpen, &gt);

    RCRE_Arena* a = rcre_arena_create(1ULL << 30);
    if (!a) { fprintf(stderr, "arena alloc failed\n"); return 1; }

    RCRE_Tensor in;
    if (!sr_pnm_read(argv[0], &in, a)) {
        fprintf(stderr, "failed to read %s (PGM/PPM P2/P3/P5/P6)\n", argv[0]);
        return 1;
    }
    size_t h = in.shape[0], w = in.shape[1], c = in.shape[2];

    SRConfig cfg;
    sr_config_default(&cfg);
    cfg.scale = scale;
    cfg.style = style;
    cfg.deskew = deskew;
    cfg.deblur = deblur;
    cfg.sharpen_amount = sharpen;

    size_t oh = h * (size_t)scale, ow = w * (size_t)scale;
    size_t shape[3] = {oh, ow, c};
    RCRE_Tensor out = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
    if (!out.data) { fprintf(stderr, "output tensor alloc failed\n"); return 1; }

    SRReport rep;
    if (!sr_enhance(&in, &out, &cfg, a, &rep)) {
        fprintf(stderr, "pipeline failed\n");
        return 1;
    }
    if (!sr_pnm_write(argv[1], &out)) {
        fprintf(stderr, "failed to write %s\n", argv[1]);
        return 1;
    }
    sr_print_report(&rep, stdout);

    if (gt) {
        RCRE_Tensor g;
        if (sr_pnm_read(gt, &g, a)) {
            size_t gshape[3] = {oh, ow, c};
            RCRE_Tensor gres = rcre_tensor_create(gshape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
            if (gres.data && sr_resize_lanczos(&g, &gres, a)) {
                printf("  vs GT:  PSNR = %.2f dB | SSIM = %.4f\n",
                       sr_psnr_db(&gres, &out, 255.0),
                       sr_ssim(&gres, &out));
            }
        } else {
            fprintf(stderr, "  (could not read GT %s)\n", gt);
        }
    }
    return 0;
}

int cmd_assess(int argc, char** argv) {
    if (argc < 1) { fprintf(stderr, "usage: assess <in.pgm>\n"); return 1; }
    RCRE_Arena* a = rcre_arena_create(1 << 26);
    RCRE_Tensor in;
    if (!sr_pnm_read(argv[0], &in, a)) return 1;
    SRAssessment s;
    if (!sr_assess(&in, &s)) return 1;
    printf("image     : %zux%zux%zu\n", in.shape[0], in.shape[1], in.shape[2]);
    printf("blur      : %.1f (Laplacian var; higher = sharper)\n", s.blur_score);
    printf("noise σ   : %.2f\n", s.noise_sigma);
    printf("contrast  : %.3f\n", s.contrast);
    printf("skew      : %.2f deg\n", s.skew_deg);
    return 0;
}

/* ---- tesseract helpers (demo command) ---- */

static int run_tesseract(const char* img, char* out_text, size_t cap) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "tesseract %s stdout -l eng --psm 6 2>/dev/null", img);
    FILE* p = popen(cmd, "r");
    if (!p) return -1;
    size_t n = 0;
    int ch;
    while ((ch = fgetc(p)) != EOF && n + 2 < cap) out_text[n++] = (char)ch;
    out_text[n] = 0;
    int code = pclose(p);
    return code;
}

static size_t levenshtein(const char* a, const char* b) {
    size_t la = strlen(a), lb = strlen(b);
    size_t* prev = (size_t*)malloc((lb + 1) * sizeof(size_t));
    size_t* cur = (size_t*)malloc((lb + 1) * sizeof(size_t));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (size_t j = 0; j <= lb; j++) prev[j] = j;
    for (size_t i = 1; i <= la; i++) {
        cur[0] = i;
        for (size_t j = 1; j <= lb; j++) {
            size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            size_t d = prev[j] + 1;
            if (cur[j - 1] + 1 < d) d = cur[j - 1] + 1;
            if (prev[j - 1] + cost < d) d = prev[j - 1] + cost;
            cur[j] = d;
        }
        size_t* t = prev; prev = cur; cur = t;
    }
    size_t d = prev[lb];
    free(prev); free(cur);
    return d;
}

static void norm_text(char* s) {
    /* uppercase, collapse spaces */
    char* w = s;
    for (char* p = s; *p; p++) {
        if (*p >= 'a' && *p <= 'z') *p = (char)(*p - 'a' + 'A');
        if (*p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') *w++ = *p;
        else if (*p == ' ') { if (w > s && w[-1] != ' ') *w++ = ' '; }
    }
    *w = 0;
}

static double cer(const char* a, const char* b) {
    size_t d = levenshtein(a, b);
    size_t m = strlen(a) > strlen(b) ? strlen(a) : strlen(b);
    return m ? (double)d / (double)m : 0.0;
}

int cmd_demo(int argc, char** argv) {
    (void)argc; (void)argv;
    /* macOS: /tmp is a symlink; this tesseract/leptonica build cannot fopen
     * through it — use the real directory when it exists. */
    const char* tmpl = (access("/private/tmp", W_OK) == 0)
                          ? "/private/tmp/sr_demo_XXXXXX"
                          : "/tmp/sr_demo_XXXXXX";
    char dir[280];
    strncpy(dir, tmpl, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = 0;
    if (!mkdtemp(dir)) { fprintf(stderr, "mkdtemp failed\n"); return 1; }
    char orig[256], deg[256], enh[256];
    snprintf(orig, sizeof(orig), "%s/original.pgm", dir);
    snprintf(deg, sizeof(deg), "%s/degraded.pgm", dir);
    snprintf(enh, sizeof(enh), "%s/enhanced.pgm", dir);

    printf("================================================================\n");
    printf("   RCRE-SR DEMO — generate → enhance → measure → OCR\n");
    printf("================================================================\n\n");
    if (cmd_generate(3, (char*[]){orig, deg, "1200", "1600"})) return 1;
    printf("\n");

    /* enhance (2x) */
    RCRE_Arena* a = rcre_arena_create(1ULL << 31);
    if (!a) return 1;
    RCRE_Tensor in;
    if (!sr_pnm_read(deg, &in, a)) return 1;
    SRConfig cfg;
    sr_config_default(&cfg);
    cfg.scale = 2;
    size_t shape[3] = {in.shape[0] * 2, in.shape[1] * 2, in.shape[2]};
    RCRE_Tensor out = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
    SRReport rep;
    if (!sr_enhance(&in, &out, &cfg, a, &rep)) return 1;
    sr_pnm_write(enh, &out);

    /* Tesseract (this build) cannot read PGM — write PNGs for OCR */
    char deg_png[300], enh_png[300];
    snprintf(deg_png, sizeof deg_png, "%s/degraded.png", dir);
    snprintf(enh_png, sizeof enh_png, "%s/enhanced.png", dir);
    (void)sr_png_write(deg_png, &in);
    (void)sr_png_write(enh_png, &out);
    sr_print_report(&rep, stdout);

    /* metrics vs GT (original resized to out size) */
    RCRE_Tensor g;
    if (sr_pnm_read(orig, &g, a)) {
        RCRE_Tensor gres = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
        if (gres.data && sr_resize_lanczos(&g, &gres, a)) {
            double psnr_in = 0, ssim_in = 0;
            /* input (deg) resized to out size for a fair compare */
            RCRE_Tensor din = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
            if (din.data && sr_resize_lanczos(&in, &din, a)) {
                psnr_in = sr_psnr_db(&gres, &din, 255.0);
                ssim_in = sr_ssim(&gres, &din);
            }
            printf("\n  QUALITY vs ground truth:\n");
            printf("    input   (bilinear 2x) : PSNR %6.2f dB | SSIM %.4f\n", psnr_in, ssim_in);
            printf("    pipeline (rcre-sr)    : PSNR %6.2f dB | SSIM %.4f\n",
                   sr_psnr_db(&gres, &out, 255.0), sr_ssim(&gres, &out));
        }
    }

    /* OCR comparison */
    char gt_text[8192], ocr_deg[8192], ocr_enh[8192], ocr_bin[8192];
    /* ground-truth text = the document content (we generated it) */
    TextItem items[64];
    int n = 0;
    build_content(items, &n, 1200);
    gt_text[0] = 0;
    for (int i = 0; i < n; i++) {
        strcat(gt_text, items[i].text);
        strcat(gt_text, " ");
    }
    norm_text(gt_text);

    int has_tess = (system("command -v tesseract >/dev/null 2>&1") == 0);
    if (has_tess) {
        /* binarized output: the OCR-optimized style */
        char bin_enh[300], bin_png[300];
        snprintf(bin_enh, sizeof bin_enh, "%s/enhanced_bin.pgm", dir);
        snprintf(bin_png, sizeof bin_png, "%s/enhanced_bin.png", dir);
        SRConfig cb = cfg;
        cb.style = SR_STYLE_BINARIZE;
        SRReport repb;
        if (sr_enhance(&in, &out, &cb, a, &repb)) {
            sr_pnm_write(bin_enh, &out);
            (void)sr_png_write(bin_png, &out);
        }

        printf("\n  OCR (tesseract, PSM 6):\n");
        int rc1 = run_tesseract(deg_png, ocr_deg, sizeof(ocr_deg));
        int rc2 = run_tesseract(enh_png, ocr_enh, sizeof(ocr_enh));
        int rc3 = run_tesseract(bin_png, ocr_bin, sizeof(ocr_bin));
        if (rc1 == 0 || rc1 == -1) {
            norm_text(ocr_deg);
        }
        if (rc2 == 0 || rc2 == -1) {
            norm_text(ocr_enh);
        }
        if (rc3 == 0 || rc3 == -1) {
            norm_text(ocr_bin);
        }
        double c1 = cer(gt_text, ocr_deg);
        double c2 = cer(gt_text, ocr_enh);
        double c3 = cer(gt_text, ocr_bin);
        printf("    input   (degraded)       : CER %7.2f %%  (%zu chars ocr'd)\n", c1 * 100, strlen(ocr_deg));
        printf("    pipeline (x2)            : CER %7.2f %%  (%zu chars ocr'd)\n", c2 * 100, strlen(ocr_enh));
        printf("    pipeline (x2, binarize)  : CER %7.2f %%  (%zu chars ocr'd)\n", c3 * 100, strlen(ocr_bin));
        double cbest = c2 < c3 ? c2 : c3;
        if (cbest < c1)
            printf("    → OCR accuracy improved by %.2f pts (CER %+.2f %%)\n\n",
                   (c1 - cbest) * 100, (cbest - c1) * 100);
        else
            printf("    → no OCR improvement this run (Δ CER %+.2f %%)\n\n", (cbest - c1) * 100);
        printf("    sample (pipeline) : \"%.*s...\"\n\n", 96, c3 <= c2 ? ocr_bin : ocr_enh);
    } else {
        printf("\n  (tesseract not found on PATH — OCR comparison skipped)\n\n");
    }

    printf("  artifacts: %s\n", dir);
    printf("  enhanced copy: %s\n", enh);
    printf("✅ demo complete.\n");
    return 0;
}

/* ========================================================================= */

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("RCRE-SR — Document Image Super-Resolution & OCR enhancement\n\n");
        printf("usage:\n");
        printf("  sr_cli generate <orig.pgm> <degraded.pgm> [w] [h]\n");
        printf("  sr_cli enhance  <in> <out> [--scale 2] [--style auto|sharp|clean|binarize|upscale]\n");
        printf("                  [--no-deskew] [--no-deblur] [--sharpen F] [--gt REF]\n");
        printf("  sr_cli assess   <in>\n");
        printf("  sr_cli demo     (full self-contained pipeline + OCR report)\n");
        return 1;
    }
    if (!strcmp(argv[1], "generate")) return cmd_generate(argc - 2, argv + 2);
    if (!strcmp(argv[1], "enhance")) return cmd_enhance(argc - 2, argv + 2);
    if (!strcmp(argv[1], "assess")) return cmd_assess(argc - 2, argv + 2);
    if (!strcmp(argv[1], "demo")) return cmd_demo(argc - 2, argv + 2);
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    return 1;
}
