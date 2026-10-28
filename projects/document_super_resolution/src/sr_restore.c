/**
 * @file sr_restore.c
 * @brief Restoration primitives: guided-filter denoise (O(N)) and
 *        Wiener deconvolution (rcre_fft_2d, zero-padded to power-of-2).
 *
 * Both are *gated* by the pipeline: denoise runs only when the measured
 * noise sigma exceeds a floor, deblur only when the measured sharpness
 * (Laplacian variance) is low. Running either on a clean image measurably
 * HURTS OCR, so the gates are part of the correctness contract.
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

/* ========================================================================= */
/* Guided filter (self-guided) — O(N) via 4 integral images per channel       */
/* ========================================================================= */

static void integral(const float* v, size_t h, size_t w, double* ii) {
    /* ii has shape (h+1) x (w+1), row 0 / col 0 zero */
    for (size_t y = 0; y <= h; y++) ii[y * (w + 1)] = 0.0;
    for (size_t x = 0; x <= w; x++) ii[x] = 0.0;
    for (size_t y = 1; y <= h; y++) {
        double row = 0.0;
        for (size_t x = 1; x <= w; x++) {
            row += v[(y - 1) * w + (x - 1)];
            ii[y * (w + 1) + x] = ii[(y - 1) * (w + 1) + x] + row;
        }
    }
}

/* Box mean of v over (2r+1)^2 with clamped borders, O(1) per pixel. */
static void box_mean(const double* ii, size_t h, size_t w, int r, float* out) {
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            size_t y0 = y < (size_t)r ? 0 : y - (size_t)r + 1;
            size_t x0 = x < (size_t)r ? 0 : x - (size_t)r + 1;
            size_t y1 = y + (size_t)r + 1 > h ? h : y + (size_t)r + 1;
            size_t x1 = x + (size_t)r + 1 > w ? w : x + (size_t)r + 1;
            double area = (double)((y1 - y0) * (x1 - x0));
            double s = ii[y1 * (w + 1) + x1] - ii[y0 * (w + 1) + x1]
                     - ii[y1 * (w + 1) + x0] + ii[y0 * (w + 1) + x0];
            out[y * w + x] = (float)(s / area);
        }
    }
}

int sr_guided_denoise(const RCRE_Tensor* src, RCRE_Tensor* dst,
                      int radius, float eps, RCRE_Arena* arena) {
    if (!src || !dst || !src->data || !dst->data) return 0;
    size_t h = src->shape[0], w = src->shape[1], C = C_of(src);
    if (h < 4 || w < 4) return 0;
    if (C != C_of(dst) || dst->shape[0] != h || dst->shape[1] != w) return 0;
    if (radius < 1) radius = 4;
    if (radius > 24) radius = 24;
    if (eps < 0.0f) eps = 0.01f;
    if (eps > 1.0f) eps = 1.0f;

    float* f = to_f32(src, arena);
    if (!f) return 0;
    size_t n = h * w;

    size_t iis = (h + 1) * (w + 1);
    double* iiI  = (double*)rcre_arena_alloc(arena, iis * sizeof(double), 64);
    double* iiI2 = (double*)rcre_arena_alloc(arena, iis * sizeof(double), 64);
    double* iia  = (double*)rcre_arena_alloc(arena, iis * sizeof(double), 64);
    double* iib  = (double*)rcre_arena_alloc(arena, iis * sizeof(double), 64);
    float* a = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    float* b = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    float* mI  = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    float* mI2 = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    float* ma  = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    float* mb  = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    float* i2  = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    if (!iiI || !iiI2 || !iia || !iib || !a || !b || !mI || !mI2 || !ma || !mb || !i2)
        return 0;

    float* out = (float*)rcre_arena_alloc(arena, (size_t)n * C * sizeof(float), 64);
    if (!out) return 0;

    const double eps_d = (double)eps * 65025.0; /* eps on 0..1 scale -> 0..255^2 */

    for (size_t c = 0; c < C; c++) {
        const float* ch = f + (size_t)c * n;
        for (size_t i = 0; i < n; i++) i2[i] = ch[i] * ch[i];

        integral(ch, h, w, iiI);
        integral(i2, h, w, iiI2);
        box_mean(iiI, h, w, radius, mI);
        box_mean(iiI2, h, w, radius, mI2);

        for (size_t i = 0; i < n; i++) {
            double var = (double)mI2[i] - (double)mI[i] * (double)mI[i];
            if (var < 0) var = 0;
            double ai = var / (var + eps_d);
            a[i] = (float)ai;
            b[i] = (float)((1.0 - ai) * mI[i]);
        }

        integral(a, h, w, iia);
        integral(b, h, w, iib);
        box_mean(iia, h, w, radius, ma);
        box_mean(iib, h, w, radius, mb);

        for (size_t i = 0; i < n; i++) {
            out[(size_t)c * n + i] = ma[i] * ch[i] + mb[i];
        }
    }

    /* Edge retention: the guided filter softens 1-px strokes (fatal for OCR).
     * Blend back the original wherever structure exists: flat regions get the
     * full denoise, edges keep their pixels. */
    {
        float* mag = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
        if (mag) {
            memset(mag, 0, n * sizeof(float));
            for (size_t y = 1; y + 1 < h; y++)
                for (size_t x = 1; x + 1 < w; x++) {
                    float gx = 0, gy = 0;
                    for (size_t c = 0; c < C; c++) {
                        gx += f[(y * w + x) * C + c] - f[(y * w + x - 1) * C + c];
                        gy += f[(y * w + x) * C + c] - f[((y - 1) * w + x) * C + c];
                    }
                    mag[y * w + x] = sqrtf(gx * gx + gy * gy) / (float)C;
                }
            float mg = 0;
            for (size_t i = 0; i < n; i++) mg += mag[i];
            mg /= (float)n;
            float gate = mg * 4.0f + 4.0f;
            for (size_t i = 0; i < n; i++) {
                float m = mag[i] / gate;
                if (m > 1.0f) m = 1.0f;
                m = m * m * (3.0f - 2.0f * m);
                for (size_t c = 0; c < C; c++) {
                    float oi = out[(size_t)c * n + i];
                    out[(size_t)c * n + i] = f[(size_t)c * n + i] * m + oi * (1.0f - m);
                }
            }
        }
    }
    u8_write(dst, out, (size_t)n * C);
    return 1;
}

/* ========================================================================= */
/* Wiener deconvolution (frequency domain)                                    */
/* ========================================================================= */

static size_t next_pow2(size_t v) {
    size_t p = 2;
    while (p < v) p <<= 1;
    return p;
}

/* Same mirror boundary as rcre_cv_gaussian_blur (reflect_101). The Wiener
 * model must extend the image the same way the blur was applied, or the
 * padded boundary (zero vs mirror) corrupts an h/k-wide edge band. */
static size_t reflect_101_sz(int64_t p, int64_t len) {
    if (len <= 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) p = -p;
        else p = 2 * len - 2 - p;
    }
    return (size_t)p;
}

/* Build a k x k PSF (row-major, normalized sum 1) into psf[]. */
static void build_psf(float* psf, size_t k, SR_PSFKind kind, float size, float angle_deg) {
    memset(psf, 0, k * k * sizeof(float));
    float c = (float)(k - 1) / 2.0f;
    double s = 0.0;
    if (kind == SR_PSF_MOTION) {
        float L = (size < 1.0f) ? 1.0f : size;
        float rad = angle_deg * (float)M_PI / 180.0f;
        float dx = cosf(rad), dy = sinf(rad);
        int steps = (int)lrintf(L);
        if (steps < 1) steps = 1;
        for (int i = -steps / 2; i <= steps / 2; i++) {
            float t = (float)i / (float)steps * L;
            int x = (int)lrintf(c + dx * t);
            int y = (int)lrintf(c + dy * t);
            if (x >= 0 && y >= 0 && x < (int)k && y < (int)k) { psf[y * k + x] += 1.0f; s += 1.0f; }
        }
    } else if (kind == SR_PSF_GAUSSIAN) {
        float sigma = (size < 2.0f) ? 1.0f : size * 0.5f;
        for (size_t y = 0; y < k; y++)
            for (size_t x = 0; x < k; x++) {
                double d2 = (double)(x - c) * (x - c) + (double)(y - c) * (y - c);
                float g = expf(-(float)d2 / (2.0f * sigma * sigma));
                psf[y * k + x] = g;
                s += g;
            }
    } else { /* defocus disk */
        float r = (size < 1.0f) ? 0.5f : size * 0.5f;
        for (size_t y = 0; y < k; y++)
            for (size_t x = 0; x < k; x++) {
                double d = sqrt((double)(x - c) * (x - c) + (double)(y - c) * (y - c));
                if (d <= r) { psf[y * k + x] = 1.0f; s += 1.0; }
            }
    }
    if (s <= 0) { psf[(size_t)(k / 2) * k + (k / 2)] = 1.0f; s = 1.0; }
    for (size_t i = 0; i < k * k; i++) psf[i] = (float)(psf[i] / s);
}

static RCRE_Tensor mk_tensor(void* data, size_t h, size_t w, RCRE_Arena* a) {
    (void)a;
    size_t shape[2] = {h, w};
    return rcre_tensor_from_buffer(data, shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC);
}

int sr_wiener_deblur(const RCRE_Tensor* src, RCRE_Tensor* dst,
                     SR_PSFKind psf_kind, float psf_size, float psf_angle_deg,
                     float lambda, RCRE_Arena* arena) {
    if (!src || !dst || !src->data || !dst->data) return 0;
    size_t h = src->shape[0], w = src->shape[1];
    if (dst->shape[0] != h || dst->shape[1] != w) return 0;
    if (psf_size < 1.0f) psf_size = 3.0f;
    if (psf_size > 25.0f) psf_size = 25.0f;
    if (lambda <= 0.0f) lambda = 0.01f;

    size_t k = (size_t)ceil((double)psf_size) | 1u; /* odd kernel */
    if (psf_kind == SR_PSF_GAUSSIAN) {
        /* cover 3 sigma to avoid truncation artifacts */
        float sigma = (psf_size < 2.0f) ? 1.0f : psf_size * 0.5f;
        size_t kg = 2u * ((size_t)ceil((double)(3.0f * sigma)) + 1u);
        if ((kg & 1u) == 0u) kg += 1u;
        if (kg > 25u) kg = 25u;
        if (kg > k) k = kg;
    }
    if (k < 3) k = 3;

    size_t Hp = next_pow2(h + k - 1);
    size_t Wp = next_pow2(w + k - 1);
    size_t np = Hp * Wp;

    float* Ip  = (float*)rcre_arena_alloc(arena, np * sizeof(float), 64);
    float* Fr  = (float*)rcre_arena_alloc(arena, np * sizeof(float), 64);
    float* Fi  = (float*)rcre_arena_alloc(arena, np * sizeof(float), 64);
    float* Hr  = (float*)rcre_arena_alloc(arena, np * sizeof(float), 64);
    float* Hi  = (float*)rcre_arena_alloc(arena, np * sizeof(float), 64);
    float* Sr  = (float*)rcre_arena_alloc(arena, np * sizeof(float), 64);
    float* Si  = (float*)rcre_arena_alloc(arena, np * sizeof(float), 64);
    float* Op  = (float*)rcre_arena_alloc(arena, np * sizeof(float), 64);
    float* Hs  = (float*)rcre_arena_alloc(arena, np * sizeof(float), 64); /* PSF real input */
    float* psf = (float*)rcre_arena_alloc(arena, k * k * sizeof(float), 64);
    if (!Ip || !Fr || !Fi || !Hr || !Hi || !Sr || !Si || !Op || !Hs || !psf) return 0;

    memset(Ip, 0, np * sizeof(float));
    memset(Hs, 0, np * sizeof(float));
    memset(Hr, 0, np * sizeof(float));
    memset(Hi, 0, np * sizeof(float));

    /* place image & PSF: the image is centered at (ky,kx) with mirror-extended
     * (reflect_101, as the blur uses) coordinates, so the linear convolution
     * at the centered offset reproduces the blur's boundary behavior exactly.
     * A top-left placement cannot represent the negative coordinates the
     * boundary kernel needs and corrupts a k/2-wide edge band. */
    {
        size_t ky0 = k / 2, kx0 = k / 2;
        if (src->dtype == RCRE_DTYPE_U8) {
            const uint8_t* p = (const uint8_t*)src->data;
            for (size_t y = 0; y < Hp; y++)
                for (size_t x = 0; x < Wp; x++)
                    Ip[y * Wp + x] = (float)p[reflect_101_sz((int64_t)y - (int64_t)ky0, (int64_t)h) * w
                                                    + reflect_101_sz((int64_t)x - (int64_t)kx0, (int64_t)w)];
        } else {
            const float* p = (const float*)src->data;
            for (size_t y = 0; y < Hp; y++)
                for (size_t x = 0; x < Wp; x++)
                    Ip[y * Wp + x] = p[reflect_101_sz((int64_t)y - (int64_t)ky0, (int64_t)h) * w
                                            + reflect_101_sz((int64_t)x - (int64_t)kx0, (int64_t)w)];
        }
    }
    build_psf(psf, k, psf_kind, psf_size, psf_angle_deg);
    for (size_t y = 0; y < k; y++)
        for (size_t x = 0; x < k; x++)
            Hs[y * Wp + x] = psf[y * k + x];

    RCRE_Tensor tI  = mk_tensor(Ip, Hp, Wp, arena);
    RCRE_Tensor tHs = mk_tensor(Hs, Hp, Wp, arena);
    RCRE_Tensor tFr = mk_tensor(Fr, Hp, Wp, arena);
    RCRE_Tensor tFi = mk_tensor(Fi, Hp, Wp, arena);
    RCRE_Tensor tHr = mk_tensor(Hr, Hp, Wp, arena);
    RCRE_Tensor tHi = mk_tensor(Hi, Hp, Wp, arena);
    RCRE_Tensor tSr = mk_tensor(Sr, Hp, Wp, arena);
    RCRE_Tensor tSi = mk_tensor(Si, Hp, Wp, arena);
    RCRE_Tensor tOp = mk_tensor(Op, Hp, Wp, arena);

    if (rcre_fft_2d(&tI, &tFr, &tFi, arena) != RCRE_OK) return 0;
    if (rcre_fft_2d(&tHs, &tHr, &tHi, arena) != RCRE_OK) return 0;

    for (size_t i = 0; i < np; i++) {
        float hr = Hr[i], hi = Hi[i];
        float fr = Fr[i], fi = Fi[i];
        float h2 = hr * hr + hi * hi;
        float den = h2 + lambda;
        /* (H* * F) / (|H|^2 + lam) */
        float nr = hr * fr + hi * fi;
        float ni = hr * fi - hi * fr;
        Sr[i] = nr / den;
        Si[i] = ni / den;
    }
    if (rcre_ifft_2d(&tSr, &tSi, &tOp, arena) != RCRE_OK) return 0;

    float* out = (float*)rcre_arena_alloc(arena, h * w * sizeof(float), 64);
    if (!out) return 0;
    /* Empirical alignment rule (delta probe): the Wiener result on the padded
     * canvas lands at (content - k/2); with the image placed at +k/2 that is
     * exactly the canvas index n. (The centered-PSF phase and the content
     * shift cancel on the finite padded canvas.) */
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++) {
            float v = Op[y * Wp + x];
            out[y * w + x] = v;
        }
    u8_write(dst, out, h * w);
    return 1;
}
