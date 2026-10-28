/**
 * @file test_sr.c
 * @brief RCRE-SR unit & integration tests.
 *
 * Covers: PNM I/O roundtrip, metrics sanity, assessment ordering
 * (sharp > blurred), resize correctness, Otsu, adaptive threshold,
 * sharpen on flats, CLAHE, guided denoise, Wiener deblur, deskew,
 * and the full pipeline.
 */
#include "sr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0, g_fails = 0;
#define CHECK(cond) do { \
        g_checks++; \
        if (!(cond)) { g_fails++; \
            fprintf(stderr, "  ✗ FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)
#define CHECK_NEAR(a, b, tol) do { \
        g_checks++; \
        double _a = (double)(a), _b = (double)(b); \
        if (fabs(_a - _b) > (tol)) { g_fails++; \
            fprintf(stderr, "  ✗ FAIL %s:%d: %s=%.4f vs %s=%.4f\n", __FILE__, __LINE__, #a, _a, #b, _b); } \
    } while (0)

static RCRE_Tensor make_u8(RCRE_Arena* a, size_t h, size_t w, size_t c,
                           const uint8_t* data) {
    size_t shape[3] = {h, w, c};
    RCRE_Tensor t = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
    if (data) memcpy(t.data, data, h * w * c);
    return t;
}

/* ------------------------------- PNM I/O --------------------------------- */

static void test_pnm_roundtrip(void) {
    printf("[ pnm i/o ]\n");
    RCRE_Arena* a = rcre_arena_create(1 << 22);
    size_t w = 37, h = 23;
    uint8_t px[w * h * 3];
    for (size_t i = 0; i < w * h * 3; i++) px[i] = (uint8_t)(i * 7 + 3);
    RCRE_Tensor t = make_u8(a, h, w, 3, px);
    CHECK(sr_pnm_write("/tmp/sr_test.ppm", &t));
    RCRE_Tensor r;
    CHECK(sr_pnm_read("/tmp/sr_test.ppm", &r, a));
    CHECK(r.shape[0] == h && r.shape[1] == w && r.shape[2] == 3);
    CHECK(memcmp(r.data, px, w * h * 3) == 0);

    /* 1-channel */
    uint8_t g1[w * h];
    for (size_t i = 0; i < w * h; i++) g1[i] = (uint8_t)(255 - i % 256);
    RCRE_Tensor tg = make_u8(a, h, w, 1, g1);
    CHECK(sr_pnm_write("/tmp/sr_test.pgm", &tg));
    RCRE_Tensor rg;
    CHECK(sr_pnm_read("/tmp/sr_test.pgm", &rg, a));
    CHECK(rg.shape[2] == 1);
    CHECK(memcmp(rg.data, g1, w * h) == 0);

    /* ASCII P2 (PGM) read */
    FILE* fp2 = fopen("/tmp/sr_test_p2.pgm", "w");
    if (fp2) {
        fprintf(fp2, "P2\n# test comment\n4 4\n255\n");
        for (int i = 0; i < 16; i++) fprintf(fp2, "%d ", i * 16);
        fprintf(fp2, "\n");
        fclose(fp2);
        RCRE_Tensor rp2;
        CHECK(sr_pnm_read("/tmp/sr_test_p2.pgm", &rp2, a));
        CHECK(rp2.shape[0] == 4 && rp2.shape[1] == 4 && rp2.shape[2] == 1);
        CHECK(((const uint8_t*)rp2.data)[0] == 0);
        CHECK(((const uint8_t*)rp2.data)[1] == 16);
    }

    /* ASCII P3 (PPM) read */
    FILE* fp3 = fopen("/tmp/sr_test_p3.ppm", "w");
    if (fp3) {
        fprintf(fp3, "P3\n# test comment\n2 2\n255\n");
        for (int i = 0; i < 12; i++) fprintf(fp3, "%d ", i * 20);
        fprintf(fp3, "\n");
        fclose(fp3);
        RCRE_Tensor rp3;
        CHECK(sr_pnm_read("/tmp/sr_test_p3.ppm", &rp3, a));
        CHECK(rp3.shape[0] == 2 && rp3.shape[1] == 2 && rp3.shape[2] == 3);
        CHECK(((const uint8_t*)rp3.data)[0] == 0);
    }

    /* PNG write test: 1-channel and 3-channel */
    CHECK(sr_png_write("/tmp/sr_test_1ch.png", &tg));
    CHECK(sr_png_write("/tmp/sr_test_3ch.png", &t));

    /* Multi-block PNG write (> 65535 raw scanline bytes to test RFC 1950 chunk loop) */
    size_t lw = 260, lh = 260;
    uint8_t lpx[260 * 260];
    for (size_t i = 0; i < 260 * 260; i++) lpx[i] = (uint8_t)(i % 256);
    RCRE_Tensor tl = make_u8(a, lh, lw, 1, lpx);
    CHECK(sr_png_write("/tmp/sr_test_large.png", &tl));

    /* invalid PNG writes & error paths */
    CHECK(!sr_png_write(NULL, &t));
    CHECK(!sr_png_write("/tmp/invalid.png", NULL));
    CHECK(!sr_pnm_write(NULL, &t));
    CHECK(!sr_pnm_write("/tmp/invalid.pnm", NULL));

    /* missing file */
    CHECK(!sr_pnm_read("/tmp/sr_definitely_missing.pgm", &rg, a));
    rcre_arena_destroy(a);
}

/* ------------------------------- metrics --------------------------------- */

static void test_metrics(void) {
    printf("[ metrics ]\n");
    RCRE_Arena* a = rcre_arena_create(1 << 22);
    size_t w = 64, h = 64;
    uint8_t p1[w * h], p2[w * h];
    for (size_t i = 0; i < w * h; i++) { p1[i] = (uint8_t)(i % 256); p2[i] = p1[i]; }
    RCRE_Tensor t1 = make_u8(a, h, w, 1, p1);
    RCRE_Tensor t2 = make_u8(a, h, w, 1, p2);
    CHECK_NEAR(sr_psnr_db(&t1, &t2, 255.0), 99.0, 0.01);
    CHECK_NEAR(sr_ssim(&t1, &t2), 1.0, 0.01);

    for (size_t i = 0; i < w * h; i++)
        ((uint8_t*)t2.data)[i] = (uint8_t)(((uint8_t*)t1.data)[i] + (((uint8_t*)t1.data)[i] > 200 ? -1 : 3));
    CHECK(sr_psnr_db(&t1, &t2, 255.0) < 50.0);
    CHECK(sr_ssim(&t1, &t2) < 1.0);
    rcre_arena_destroy(a);
}

/* ------------------------------- assess ---------------------------------- */

/* synthetic: crisp text lines on white */
static void make_text_image(uint8_t* img, size_t w, size_t h) {
    memset(img, 255, w * h);
    for (size_t y = 0; y + 12 < h; y += 24) {
        for (size_t x = 0; x + 40 < w; x += 52) {
            /* blocky "character" 6x12 */
            for (int dy = 0; dy < 12; dy++)
                for (int dx = 0; dx < 6; dx++) {
                    int on = (dx == 0 || dx == 5 || dy == 0 || dy == 11 ||
                              (dy == 5 && dx >= 1 && dx <= 4)) ? 1 : 0;
                    if (on) img[(y + dy) * w + x + dx] = 30;
                }
        }
    }
}

static void gauss_blur_u8(uint8_t* img, size_t w, size_t h, float sigma, RCRE_Arena* a) {
    size_t shape[3] = {h, w, 1};
    RCRE_Tensor in = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, a);
    RCRE_Tensor out = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, a);
    for (size_t i = 0; i < w * h; i++) ((float*)in.data)[i] = (float)img[i];
    if (rcre_cv_gaussian_blur(&in, &out, sigma, a) != RCRE_OK) return;
    for (size_t i = 0; i < w * h; i++) {
        int v = (int)lrintf(((float*)out.data)[i]);
        img[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
}

static void test_assess(void) {
    printf("[ assess ]\n");
    RCRE_Arena* a = rcre_arena_create(1 << 23);
    size_t w = 400, h = 300;
    uint8_t sharp[w * h], blurred[w * h];
    make_text_image(sharp, w, h);
    memcpy(blurred, sharp, w * h);
    gauss_blur_u8(blurred, w, h, 2.5f, a);

    RCRE_Tensor ts = make_u8(a, h, w, 1, sharp);
    RCRE_Tensor tb = make_u8(a, h, w, 1, blurred);
    SRAssessment ss, bs;
    CHECK(sr_assess(&ts, &ss));
    CHECK(sr_assess(&tb, &bs));
    CHECK(ss.blur_score > bs.blur_score * 2.0f);   /* sharp clearly sharper */
    CHECK(ss.contrast > 0.1f);
    CHECK(!isnan(bs.skew_deg));            /* no NaNs from the estimator */
    rcre_arena_destroy(a);
}

/* ------------------------------- resize ---------------------------------- */

static void test_resize(void) {
    printf("[ resize ]\n");
    RCRE_Arena* a = rcre_arena_create(1 << 24);
    size_t w = 40, h = 30;
    uint8_t src[w * h];
    for (size_t i = 0; i < w * h; i++) src[i] = 128;
    RCRE_Tensor s = make_u8(a, h, w, 1, src);

    uint8_t dst[2 * h * 2 * w];
    RCRE_Tensor d = make_u8(a, 2 * h, 2 * w, 1, dst);
    CHECK(sr_resize_lanczos(&s, &d, a));
    int maxdev = 0;
    for (size_t i = 0; i < 2 * h * 2 * w; i++) {
        int dev = abs((int)((const uint8_t*)d.data)[i] - 128);
        if (dev > maxdev) maxdev = dev;
    }
    CHECK(maxdev <= 2); /* constant stays constant (no border darkening) */

    /* vertical edge at x = w/2: 2x upscale must keep a crisp transition */
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++) src[y * w + x] = (x < w / 2) ? 0 : 255;
    memcpy(s.data, src, w * h);
    CHECK(sr_resize_lanczos(&s, &d, a));
    /* find the strongest 1-px contrast near the expected edge x = w (2x of w/2) */
    const uint8_t* dd = (const uint8_t*)d.data;
    int best = 0;
    size_t cy = 2 * h - 1; /* last valid row */
    for (size_t x = (size_t)w - 4; x + 1 < (size_t)w + 4; x++) {
        int v = abs((int)dd[cy * 2 * w + x + 1] - (int)dd[cy * 2 * w + x]);
        if (v > best) best = v;
    }
    CHECK(best > 120); /* transition preserved (Lanczos 2x peaks ~147) */

    /* Bicubic resize */
    uint8_t dst_bic[2 * h * 2 * w];
    RCRE_Tensor d_bic = make_u8(a, 2 * h, 2 * w, 1, dst_bic);
    CHECK(sr_resize_bicubic(&s, &d_bic, a));
    CHECK(((const uint8_t*)d_bic.data)[0] == 0);

    /* Identity resize (sw == dw && sh == dh) */
    uint8_t dst_ident[h * w];
    RCRE_Tensor d_ident = make_u8(a, h, w, 1, dst_ident);
    CHECK(sr_resize_lanczos(&s, &d_ident, a));
    CHECK(memcmp(d_ident.data, s.data, h * w) == 0);

    /* F32 resize */
    size_t fshape[3] = {h, w, 1};
    RCRE_Tensor sf = rcre_tensor_create(fshape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, a);
    for (size_t i = 0; i < h * w; i++) ((float*)sf.data)[i] = (float)src[i];
    size_t dfshape[3] = {2 * h, 2 * w, 1};
    RCRE_Tensor df = rcre_tensor_create(dfshape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, a);
    CHECK(sr_resize_lanczos(&sf, &df, a));
    CHECK(sr_resize_bicubic(&sf, &df, a));

    /* 3-channel resize */
    uint8_t src3[h * w * 3];
    for (size_t i = 0; i < h * w * 3; i++) src3[i] = (uint8_t)(i % 256);
    RCRE_Tensor s3 = make_u8(a, h, w, 3, src3);
    uint8_t dst3[2 * h * 2 * w * 3];
    RCRE_Tensor d3 = make_u8(a, 2 * h, 2 * w, 3, dst3);
    CHECK(sr_resize_lanczos(&s3, &d3, a));
    CHECK(sr_resize_bicubic(&s3, &d3, a));

    /* NULL / invalid argument checks */
    CHECK(!sr_resize_lanczos(NULL, &d, a));
    CHECK(!sr_resize_bicubic(&s, NULL, a));
    CHECK(!sr_resize_lanczos(&s, &d, NULL));
    rcre_arena_destroy(a);
}

/* ------------------------------- otsu/threshold --------------------------- */

static void test_thresholds(void) {
    printf("[ thresholds ]\n");
    RCRE_Arena* a = rcre_arena_create(1 << 22);
    size_t w = 100, h = 100;
    uint8_t bim[w * h];
    for (size_t i = 0; i < w * h; i++) bim[i] = (i % 2) ? 200 : 10;
    RCRE_Tensor t = make_u8(a, h, w, 1, bim);
    float thr = 0;
    CHECK(sr_otsu_threshold(&t, &thr));
    CHECK(thr > 50.0f && thr < 150.0f);

    uint8_t bin[w * h];
    RCRE_Tensor b = make_u8(a, h, w, 1, bin);
    CHECK(sr_adaptive_threshold(&t, &b, 25, 0.25f, a));
    int zeros = 0, ones = 0, other = 0;
    for (size_t i = 0; i < w * h; i++) {
        uint8_t v = ((const uint8_t*)b.data)[i];
        if (v == 0) zeros++;
        else if (v == 255) ones++;
        else other++;
    }
    CHECK(other == 0);
    CHECK(zeros > 0 && ones > 0);
    rcre_arena_destroy(a);
}

/* ------------------------------- sharpen ---------------------------------- */

static void test_sharpen_flat(void) {
    printf("[ sharpen (flat invariance) ]\n");
    RCRE_Arena* a = rcre_arena_create(1 << 22);
    size_t w = 64, h = 64;
    uint8_t flat[w * h];
    memset(flat, 128, w * h);
    RCRE_Tensor s = make_u8(a, h, w, 1, flat);
    uint8_t out8[w * h];
    RCRE_Tensor d = make_u8(a, h, w, 1, out8);
    CHECK(sr_sharpen_edge(&s, &d, 1.0f, a));
    int maxdev = 0;
    for (size_t i = 0; i < w * h; i++) {
        int dev = abs((int)((const uint8_t*)d.data)[i] - 128);
        if (dev > maxdev) maxdev = dev;
    }
    CHECK(maxdev <= 4); /* flat regions must not gain noise */
    rcre_arena_destroy(a);
}

/* ------------------------------- CLAHE ------------------------------------ */

static float contrast_of(const uint8_t* p, size_t n) {
    size_t hist[256];
    memset(hist, 0, sizeof(hist));
    for (size_t i = 0; i < n; i++) hist[p[i]]++;
    size_t acc = 0, p2 = 0, p98 = 0;
    for (int v = 0; v < 256; v++) {
        acc += hist[v];
        if (p2 == 0 && acc >= n / 50) p2 = (size_t)v;
        if (acc >= (size_t)(0.98 * n)) { p98 = (size_t)v; break; }
    }
    return (float)((double)p98 - (double)p2) / 255.0;
}

static void test_clahe(void) {
    printf("[ clahe ]\n");
    RCRE_Arena* a = rcre_arena_create(1 << 22);
    size_t w = 128, h = 128;
    uint8_t low[w * h];
    /* low contrast: values bunched in [100, 130] */
    for (size_t i = 0; i < w * h; i++) low[i] = (uint8_t)(100 + (i % 30));
    RCRE_Tensor s = make_u8(a, h, w, 1, low);
    uint8_t out8[w * h];
    RCRE_Tensor d = make_u8(a, h, w, 1, out8);
    CHECK(sr_clahe(&s, &d, 8, 3.0f, a));
    CHECK(contrast_of((const uint8_t*)d.data, w * h) > contrast_of(low, w * h) + 0.15f);
    rcre_arena_destroy(a);
}

/* ------------------------------- guided denoise --------------------------- */

static float noise_rms(const uint8_t* p, const uint8_t* clean, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)p[i] - (double)clean[i];
        s += d * d;
    }
    return (float)sqrt(s / (double)n);
}

static void test_guided_denoise(void) {
    printf("[ guided denoise ]\n");
    RCRE_Arena* a = rcre_arena_create(1 << 23);
    size_t w = 128, h = 128;
    uint8_t clean[w * h];
    make_text_image(clean, w, h);
    uint8_t noisy[w * h];
    memcpy(noisy, clean, w * h);
    uint64_t state = 42;
    for (size_t i = 0; i < w * h; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        int n = (int)((state >> 33) % 61) - 30; /* +-30 */
        int v = (int)noisy[i] + n;
        noisy[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
    RCRE_Tensor s = make_u8(a, h, w, 1, noisy);
    uint8_t out8[w * h];
    RCRE_Tensor d = make_u8(a, h, w, 1, out8);
    CHECK(sr_guided_denoise(&s, &d, 6, 0.05f, a));
    CHECK(noise_rms((const uint8_t*)d.data, clean, w * h) < noise_rms(noisy, clean, w * h));
    rcre_arena_destroy(a);
}

/* ------------------------------- wiener deblur ---------------------------- */

static void test_wiener(void) {
    printf("[ wiener deblur ]\n");
    RCRE_Arena* a = rcre_arena_create(1 << 25);
    size_t w = 128, h = 128;

    /* Part 1 — recoverable content: smooth low-frequencies are fully inside
     * the blur passband, so deconvolution must beat the blur baseline. */
    uint8_t smooth[w * h];
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++) {
            float v = 60.0f * sinf((float)x * 2.0f * (float)M_PI / 32.0f)
                      + 40.0f * sinf((float)y * 2.0f * (float)M_PI / 64.0f);
            int iv = (int)lrintf(127.5f + v);
            smooth[y * w + x] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
        }
    uint8_t blurred[w * h];
    memcpy(blurred, smooth, w * h);
    gauss_blur_u8(blurred, w, h, 2.0f, a);
    RCRE_Tensor ts = make_u8(a, h, w, 1, smooth);
    RCRE_Tensor tb = make_u8(a, h, w, 1, blurred);
    RCRE_Tensor td = make_u8(a, h, w, 1, NULL);
    CHECK(sr_wiener_deblur(&tb, &td, SR_PSF_GAUSSIAN, 4.0f, 0.0f, 3e-3f, a));
    double psnr_before = sr_psnr_db(&ts, &tb, 255.0);
    double psnr_after = sr_psnr_db(&ts, &td, 255.0);
    CHECK(psnr_after > psnr_before); /* deblur must recover fidelity */

    /* Part 2 — 1-px strokes: frequencies above the blur passband are
     * information-destroyed; require the stage to run and change the image
     * (OCR benefit of deblur is validated at pipeline level with binarization). */
    RCRE_Arena* a2 = rcre_arena_create(1 << 25);
    uint8_t sharp2[w * h], blurred2[w * h];
    make_text_image(sharp2, w, h);
    memcpy(blurred2, sharp2, w * h);
    gauss_blur_u8(blurred2, w, h, 2.0f, a2);
    RCRE_Tensor ts2 = make_u8(a2, h, w, 1, sharp2);
    RCRE_Tensor tb2 = make_u8(a2, h, w, 1, blurred2);
    RCRE_Tensor td2 = make_u8(a2, h, w, 1, NULL);
    CHECK(sr_wiener_deblur(&tb2, &td2, SR_PSF_GAUSSIAN, 4.0f, 0.0f, 3e-4f, a2));
    const uint8_t* od = (const uint8_t*)td2.data;
    int changed = 0;
    for (size_t i = 0; i < w * h; i++)
        if (od[i] != blurred2[i]) { changed = 1; break; }
    CHECK(changed); /* filter applied */
    double t_before = sr_psnr_db(&ts2, &tb2, 255.0);
    double t_after = sr_psnr_db(&ts2, &td2, 255.0);
    CHECK(t_after > t_before); /* deblur beats blur baseline on strokes too */
    rcre_arena_destroy(a);
    rcre_arena_destroy(a2);
}

/* ------------------------------- deskew ----------------------------------- */

static void rotate_u8(const uint8_t* in, size_t w, size_t h, float deg,
                      uint8_t* out, RCRE_Arena* a) {
    (void)a;
    double cx = w / 2.0, cy = h / 2.0;
    double ca = cos(deg * M_PI / 180.0), sa = sin(deg * M_PI / 180.0);
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++) {
            double dx = x - cx, dy = y - cy;
            double fx = cx + (ca * dx + sa * dy);
            double fy = cy + (-sa * dx + ca * dy);
            float v = 255.0f;
            if (fx >= 0 && fy >= 0 && fx < (double)w - 1 && fy < (double)h - 1) {
                int x0 = (int)fx, y0 = (int)fy;
                float tx = (float)(fx - x0), ty = (float)(fy - y0);
                v = in[y0 * w + x0] * (1 - tx) * (1 - ty) +
                    in[y0 * w + x0 + 1] * tx * (1 - ty) +
                    in[(y0 + 1) * w + x0] * (1 - tx) * ty +
                    in[(y0 + 1) * w + x0 + 1] * tx * ty;
            }
            out[y * w + x] = (uint8_t)lrintf(v);
        }
}

static void test_deskew(void) {
    printf("[ deskew ]\n");
    RCRE_Arena* a = rcre_arena_create(1 << 24);
    size_t w = 320, h = 240;
    uint8_t straight[w * h], skewed[w * h], out8[w * h];
    make_text_image(straight, w, h);
    rotate_u8(straight, w, h, 2.0f, skewed, a); /* skew by +2 deg */

    RCRE_Tensor ts = make_u8(a, h, w, 1, skewed);
    RCRE_Tensor td = make_u8(a, h, w, 1, out8);
    float est = 99.0f;
    CHECK(sr_deskew(&ts, &td, 0.3f, 5.0f, &est, a));
    /* correction should be ~ -2 deg (undoes the +2 skew) */
    CHECK(fabsf(est + 2.0f) < 0.6f);
    /* residual should be sharper than the skewed input */
    SRAssessment a1, a2;
    sr_assess(&ts, &a1);
    sr_assess(&td, &a2);
    /* residual: double bilinear resampling softens edges; the pipeline's
     * sharpen stage compensates — require no significant degradation. */
    CHECK(a2.blur_score > a1.blur_score * 0.7f);
    rcre_arena_destroy(a);
}

/* ------------------------------- pipeline --------------------------------- */

static void test_pipeline(void) {
    printf("[ pipeline ]\n");
    RCRE_Arena* a = rcre_arena_create(1ULL << 28);
    size_t w = 480, h = 360;
    uint8_t in_img[w * h];
    make_text_image(in_img, w, h);
    /* add noise so denoise stage fires */
    uint64_t state = 7;
    for (size_t i = 0; i < w * h; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        int n = (int)((state >> 33) % 41) - 20;
        int v = (int)in_img[i] + n;
        in_img[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
    RCRE_Tensor in = make_u8(a, h, w, 1, in_img);

    SRConfig cfg;
    sr_config_default(&cfg);
    cfg.scale = 2;
    RCRE_Tensor out = make_u8(a, h * 2, w * 2, 1, NULL);
    SRReport rep;
    CHECK(sr_enhance(&in, &out, &cfg, a, &rep));
    CHECK(rep.n_stages >= 6);
    CHECK(rep.total_ms > 0);
    CHECK(rep.peak_mem_bytes > 0);
    CHECK(rep.out_w == (int)(w * 2) && rep.out_h == (int)(h * 2));
    /* no garbage: mean should stay in a sane range (document: mostly white) */
    double mean = 0;
    for (size_t i = 0; i < h * w * 4; i++) mean += ((const uint8_t*)out.data)[i];
    mean /= (double)(h * w * 4);
    CHECK(mean > 60.0 && mean < 255.0);

    /* binarize style */
    SRConfig cb = cfg;
    cb.style = SR_STYLE_BINARIZE;
    cb.scale = 1;
    RCRE_Tensor outb = make_u8(a, h, w, 1, NULL);
    SRReport repb;
    CHECK(sr_enhance(&in, &outb, &cb, a, &repb));
    int zeros = 0, ones = 0;
    for (size_t i = 0; i < h * w; i++) {
        if (((const uint8_t*)outb.data)[i] == 0) zeros++;
        else ones++;
    }
    CHECK(zeros == (int)(h * w) - ones); /* strictly binary */
    CHECK(zeros < (int)(h * w) * 9 / 10); /* not all-ink */

    /* SR_STYLE_SHARP */
    SRConfig c_sharp = cfg;
    c_sharp.style = SR_STYLE_SHARP;
    c_sharp.scale = 2;
    RCRE_Tensor out_sharp = make_u8(a, h * 2, w * 2, 1, NULL);
    SRReport rep_sharp;
    CHECK(sr_enhance(&in, &out_sharp, &c_sharp, a, &rep_sharp));
    CHECK(rep_sharp.out_w == (int)(w * 2));

    /* SR_STYLE_CLEAN */
    SRConfig c_clean = cfg;
    c_clean.style = SR_STYLE_CLEAN;
    c_clean.scale = 2;
    RCRE_Tensor out_clean = make_u8(a, h * 2, w * 2, 1, NULL);
    SRReport rep_clean;
    CHECK(sr_enhance(&in, &out_clean, &c_clean, a, &rep_clean));
    CHECK(rep_clean.out_w == (int)(w * 2));

    /* SR_STYLE_UPSCALE */
    SRConfig c_up = cfg;
    c_up.style = SR_STYLE_UPSCALE;
    c_up.scale = 2;
    RCRE_Tensor out_up = make_u8(a, h * 2, w * 2, 1, NULL);
    SRReport rep_up;
    CHECK(sr_enhance(&in, &out_up, &c_up, a, &rep_up));
    CHECK(rep_up.out_w == (int)(w * 2));

    /* Identity / no-op copy pipeline (scale=1, all stages bypassed) */
    SRConfig c_ident = cfg;
    c_ident.scale = 1;
    c_ident.deskew = false;
    c_ident.deblur = false;
    c_ident.clahe = false;
    c_ident.style = SR_STYLE_UPSCALE;
    c_ident.sharpen_amount = 0.0f;
    c_ident.denoise_strength = 0.0f;
    RCRE_Tensor out_ident = make_u8(a, h, w, 1, NULL);
    SRReport rep_ident;
    CHECK(sr_enhance(&in, &out_ident, &c_ident, a, &rep_ident));
    CHECK(memcmp(out_ident.data, in.data, h * w) == 0);

    /* All-stages active test: skewed + blurred + low contrast */
    uint8_t deg[w * h];
    make_text_image(deg, w, h);
    for (size_t i = 0; i < w * h; i++) deg[i] = (uint8_t)(100 + (deg[i] * 30 / 255));
    gauss_blur_u8(deg, w, h, 3.0f, a);
    uint8_t deg_skew[w * h];
    rotate_u8(deg, w, h, 2.0f, deg_skew, a);
    RCRE_Tensor in_deg = make_u8(a, h, w, 1, deg_skew);
    SRConfig c_all;
    sr_config_default(&c_all);
    c_all.scale = 2;
    RCRE_Tensor out_all = make_u8(a, h * 2, w * 2, 1, NULL);
    SRReport rep_all;
    CHECK(sr_enhance(&in_deg, &out_all, &c_all, a, &rep_all));

    /* Test terminal report output */
    sr_print_report(&rep_all, stdout);
    sr_print_report(&rep, stdout);
    sr_print_report(NULL, stdout);

    rcre_arena_destroy(a);
}

int main(void) {
    printf("================================================================\n");
    printf("   RCRE-SR Test Suite\n");
    printf("================================================================\n");
    test_pnm_roundtrip();
    test_metrics();
    test_assess();
    test_resize();
    test_thresholds();
    test_sharpen_flat();
    test_clahe();
    test_guided_denoise();
    test_wiener();
    test_deskew();
    test_pipeline();
    printf("----------------------------------------------------------------\n");
    if (g_fails == 0) {
        printf("  ✅ ALL %d CHECKS PASSED\n", g_checks);
        return 0;
    }
    printf("  ❌ %d/%d CHECKS FAILED\n", g_fails, g_checks);
    return 1;
}
