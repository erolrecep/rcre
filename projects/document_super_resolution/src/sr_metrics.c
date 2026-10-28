/**
 * @file sr_metrics.c
 * @brief Objective image-quality metrics: PSNR & SSIM (gray, first channel).
 *
 * SSIM uses a 3x3 box average as the window approximation (standard fast
 * SSIM). L = 255 for U8 inputs; F32 inputs normalized by their max if <= 1.
 */
#include "sr.h"

#include <math.h>
#include <stdlib.h>

static float px(const RCRE_Tensor* t, size_t y, size_t x) {
    if (t->dtype == RCRE_DTYPE_U8) {
        const uint8_t* p = (const uint8_t*)t->data;
        size_t c = t->ndim >= 3 ? t->shape[2] : 1;
        return (float)p[(y * t->shape[1] + x) * c];
    }
    const float* p = (const float*)t->data;
    size_t c = t->ndim >= 3 ? t->shape[2] : 1;
    return p[(y * t->shape[1] + x) * c];
}

double sr_psnr_db(const RCRE_Tensor* a, const RCRE_Tensor* b, double max_val) {
    if (!a || !b || !a->data || !b->data) return -1.0;
    if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1]) return -1.0;
    if (max_val <= 0) max_val = 255.0;
    size_t h = a->shape[0], w = a->shape[1];
    double se = 0.0;
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            double d = (double)px(a, y, x) - (double)px(b, y, x);
            se += d * d;
        }
    }
    double mse = se / ((double)h * (double)w);
    if (mse < 1e-12) return 99.0; /* numerically identical */
    return 10.0 * log10(max_val * max_val / mse);
}

double sr_ssim(const RCRE_Tensor* a, const RCRE_Tensor* b) {
    if (!a || !b || !a->data || !b->data) return -1.0;
    if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1]) return -1.0;
    size_t h = a->shape[0], w = a->shape[1];
    if (h < 3 || w < 3) return -1.0;

    const double L = 255.0;
    const double C1 = (0.01 * L) * (0.01 * L);
    const double C2 = (0.03 * L) * (0.03 * L);

    double ssim_sum = 0.0;
    size_t valid = 0;
    for (size_t y = 1; y + 1 < h; y++) {
        for (size_t x = 1; x + 1 < w; x++) {
            double sa = 0, sb = 0, sa2 = 0, sb2 = 0, sab = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    float pa = px(a, y + (size_t)dy, x + (size_t)dx);
                    float pb = px(b, y + (size_t)dy, x + (size_t)dx);
                    sa += pa; sb += pb;
                    sa2 += (double)pa * pa;
                    sb2 += (double)pb * pb;
                    sab += (double)pa * pb;
                }
            }
            double inv = 1.0 / 9.0;
            double mxa = sa * inv, mxb = sb * inv;
            double vara = sa2 * inv - mxa * mxa;
            double varb = sb2 * inv - mxb * mxb;
            double covab = sab * inv - mxa * mxb;
            double num = (2 * mxa * mxb + C1) * (2 * covab + C2);
            double den = (mxa * mxa + mxb * mxb + C1) * (vara + varb + C2);
            ssim_sum += num / den;
            valid++;
        }
    }
    return valid ? ssim_sum / (double)valid : -1.0;
}
