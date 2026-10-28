/**
 * @file sr_pipeline.c
 * @brief The adaptive document super-resolution pipeline.
 *
 *   assess → [deskew] → [denoise] → [deblur] → Lanczos-3 ×scale
 *          → [edge-masked unsharp] → [CLAHE] → [adaptive threshold]
 *
 * Gating policy (correctness contract: a stage must never run unless the
 * assessment shows the image needs it):
 *   deskew   : |skew| > cfg.min_skew_deg
 *   denoise  : noise_sigma > 3.0
 *   deblur   : blur_score < 60 (Laplacian-variance scale)
 *   sharpen  : always after upscale, amount adaptive from blur_score
 *   CLAHE    : contrast < 0.55
 *   threshold: only SR_STYLE_BINARIZE
 *
 * Memory: each stage allocates its output tensor from the caller's arena
 * (cumulative; peak reported in SRReport.peak_mem_bytes). Size the arena
 * ~ (input_bytes * 6) * scale^2 for 3ch input.
 */
#include "sr.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void sr_config_default(SRConfig* cfg) {
    cfg->scale = 2;
    cfg->style = SR_STYLE_AUTO;
    cfg->deskew = true;
    cfg->deblur = true;
    cfg->psf = SR_PSF_GAUSSIAN; /* camera-style blur model */
    cfg->psf_size = 4.0f;
    cfg->sharpen_amount = -1.0f;   /* auto */
    cfg->denoise_strength = -1.0f; /* auto */
    cfg->min_skew_deg = 0.3f;
    cfg->max_skew_deg = 7.5f;
    cfg->clahe = true;
    cfg->clahe_grid = 8;
    cfg->clahe_clip = 3.0f;
}

/* ========================================================================= */
/* Deskew — projection-profile search + perspective warp                     */
/* ========================================================================= */

static float px_u8(const RCRE_Tensor* t, size_t y, size_t x) {
    size_t c = (t->ndim >= 3) ? t->shape[2] : 1;
    if (t->dtype == RCRE_DTYPE_U8) return (float)((const uint8_t*)t->data)[(y * t->shape[1] + x) * c];
    return ((const float*)t->data)[(y * t->shape[1] + x) * c];
}

/* Row-sum-of-squares score for rotating src by angle a (dst->src mapping). */
static double deskew_score(const uint8_t* mask, size_t w, size_t h,
                           float a_deg, float* row_acc) {
    double cx = (double)w / 2.0, cy = (double)h / 2.0;
    float ca = cosf(a_deg * (float)M_PI / 180.0f);
    float sa = sinf(a_deg * (float)M_PI / 180.0f);
    memset(row_acc, 0, h * sizeof(float));
    for (size_t y = 0; y < h; y++) {
        double dy = (double)y - cy;
        for (size_t x = 0; x < w; x++) {
            double dx = (double)x - cx;
            /* same dst->src mapping as assessment: rotate sample point by +a */
            int ix = (int)lrint(cx + (double)ca * dx + (double)sa * dy);
            int iy = (int)lrint(cy - (double)sa * dx + (double)ca * dy);
            if (ix < 0 || iy < 0 || ix >= (int)w || iy >= (int)h) continue;
            if (mask[(size_t)iy * w + (size_t)ix]) row_acc[y] += 1.0f;
        }
    }
    double s = 0;
    for (size_t y = 0; y < h; y++) s += (double)row_acc[y] * (double)row_acc[y];
    return s;
}

int sr_deskew(const RCRE_Tensor* src, RCRE_Tensor* dst,
              float min_angle, float max_angle, float* angle_out,
              RCRE_Arena* arena) {
    if (!src || !dst || !src->data || !dst->data || !arena) return 0;
    size_t h = src->shape[0], w = src->shape[1];
    if (h < 16 || w < 16) return 0;
    if (max_angle < 0.5f) max_angle = 7.5f;
    if (max_angle > 15.0f) max_angle = 15.0f;
    if (min_angle < 0.1f) min_angle = 0.3f;

    /* subsample mask for the search (<= 800 px) */
    float ratio = 1.0f;
    if (w > 800) ratio = 800.0f / (float)w;
    if (h * ratio > 800.0f) ratio = 800.0f / (float)h;
    size_t mw = (size_t)ceil((double)w * ratio);
    size_t mh = (size_t)ceil((double)h * ratio);
    if (mw < 16 || mh < 16) mw = 16, mh = 16;

    uint8_t* mask = (uint8_t*)rcre_arena_alloc(arena, mw * mh, 64);
    if (!mask) return 0;
    for (size_t y = 0; y < mh; y++) {
        size_t sy = (size_t)((double)y / ratio);
        if (sy >= h) sy = h - 1;
        for (size_t x = 0; x < mw; x++) {
            size_t sx = (size_t)((double)x / ratio);
            if (sx >= w) sx = w - 1;
            mask[y * mw + x] = (px_u8(src, sy, sx) < 128.0f) ? 1 : 0;
        }
    }

    float* row_acc = (float*)rcre_arena_alloc(arena, mh * sizeof(float), 64);
    if (!row_acc) return 0;

    double best = -1.0;
    float best_a = 0.0f;
    for (float a = -max_angle; a <= max_angle + 1e-4f; a += 0.25f) {
        double sc = deskew_score(mask, mw, mh, a, row_acc);
        if (sc > best) { best = sc; best_a = a; }
    }
    for (float a = best_a - 0.25f; a <= best_a + 0.25f + 1e-4f; a += 0.02f) {
        double sc = deskew_score(mask, mw, mh, a, row_acc);
        if (sc > best) { best = sc; best_a = a; }
    }

    if (angle_out) *angle_out = best_a;
    if (fabsf(best_a) < min_angle) {
        /* below threshold: copy src */
        size_t c = (src->ndim >= 3) ? src->shape[2] : 1;
        size_t n = h * w * c;
        if (src->dtype == RCRE_DTYPE_U8 && dst->dtype == RCRE_DTYPE_U8) {
            memcpy(dst->data, src->data, n);
        } else {
            for (size_t y = 0; y < h; y++)
                for (size_t x = 0; x < w; x++)
                    for (size_t ch = 0; ch < c; ch++) {
                        float v = (src->dtype == RCRE_DTYPE_U8)
                                      ? (float)((const uint8_t*)src->data)[(y * w + x) * c + ch]
                                      : ((const float*)src->data)[(y * w + x) * c + ch];
                        if (dst->dtype == RCRE_DTYPE_U8) {
                            int iv = (int)lrintf(v);
                            ((uint8_t*)dst->data)[(y * w + x) * c + ch] =
                                (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
                        } else {
                            ((float*)dst->data)[(y * w + x) * c + ch] = v;
                        }
                    }
        }
        return 1;
    }

    /* rotation matrix implementing the winning dst->src mapping */
    double cx = (double)w / 2.0, cy = (double)h / 2.0;
    float a = (float)((double)best_a * (float)M_PI / 180.0);
    float ca = cosf(a), sa = sinf(a);
    float H[3][3] = {
        { ca, sa, (float)(cx - (double)ca * cx - (double)sa * cy) },
        { -sa, ca, (float)(cy + (double)sa * cx - (double)ca * cy) },
        { 0.0f, 0.0f, 1.0f }
    };

    /* rcre_cv_warp_perspective requires F32 tensors and fills out-of-bounds
     * dst pixels with 0 — convert, warp, then repaint corners as paper white. */
    size_t c = (src->ndim >= 3) ? src->shape[2] : 1;
    size_t n = h * w * c;
    float* fsrc = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    float* fdst = (float*)rcre_arena_alloc(arena, n * sizeof(float), 64);
    if (!fsrc || !fdst) return 0;
    if (src->dtype == RCRE_DTYPE_U8) {
        const uint8_t* p = (const uint8_t*)src->data;
        for (size_t i = 0; i < n; i++) fsrc[i] = (float)p[i];
    } else {
        memcpy(fsrc, src->data, n * sizeof(float));
    }
    size_t shape[3] = {h, w, c};
    RCRE_Tensor ts = rcre_tensor_from_buffer(fsrc, shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC);
    RCRE_Tensor td = rcre_tensor_from_buffer(fdst, shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC);
    if (rcre_cv_warp_perspective(&ts, H, &td, arena) != RCRE_OK) return 0;

    /* repaint out-of-bounds (black) corners as paper white for documents */
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            double dx = (double)x - cx, dy = (double)y - cy;
            double sx = (double)ca * dx + (double)sa * dy;
            double sy = -((double)sa) * dx + (double)ca * dy;
            double fx = cx + sx, fy = cy + sy;
            if (fx < 0.0 || fy < 0.0 || fx >= (double)w || fy >= (double)h) {
                for (size_t ch = 0; ch < c; ch++) fdst[(y * w + x) * c + ch] = 255.0f;
            }
        }
    }

    if (dst->dtype == RCRE_DTYPE_U8) {
        uint8_t* p = (uint8_t*)dst->data;
        for (size_t i = 0; i < n; i++) {
            int v = (int)lrintf(fdst[i]);
            p[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    } else {
        memcpy(dst->data, fdst, n * sizeof(float));
    }
    return 1;
}

/* ========================================================================= */
/* Pipeline                                                                   */
/* ========================================================================= */

static RCRE_Tensor alloc_like(RCRE_Arena* a, size_t h, size_t w, size_t c,
                              RCRE_DType dt) {
    size_t shape[3] = {h, w, c};
    return rcre_tensor_create(shape, 3, dt, RCRE_LAYOUT_HWC, a);
}

int sr_enhance(const RCRE_Tensor* src, RCRE_Tensor* dst,
               const SRConfig* cfg, RCRE_Arena* arena, SRReport* report) {
    if (!src || !dst || !cfg || !arena) return 0;
    size_t h = src->shape[0], w = src->shape[1];
    size_t C = (src->ndim >= 3) ? src->shape[2] : 1;
    if (h < 8 || w < 8) return 0;

    SRConfig c = *cfg;
    if (c.scale < 1) c.scale = 1;
    if (c.scale > 8) c.scale = 8;

    double t_start = now_s();
    SRReport rep;
    memset(&rep, 0, sizeof(rep));

    /* internal report storage for the assessment */
    SRAssessment* assess_own = (SRAssessment*)rcre_arena_alloc(
        arena, sizeof(SRAssessment), 8);
    if (!assess_own) return 0;
    rep.before = assess_own;
    if (!sr_assess(src, assess_own)) return 0;

    const RCRE_Tensor* cur = src;
    size_t ch = h, cw = w;
    int stage_i = 0;
    double t0 = 0.0;

    /* Stage buffers live at function scope: `cur` aliases them across stage
     * boundaries; block-scoped tensors would leave `cur` dangling (UB). */
    RCRE_Tensor buf_dskew, buf_denoise, buf_deblur, buf_upscale;
    RCRE_Tensor buf_sharpen, buf_clahe, buf_bin;
    memset(&buf_dskew, 0, sizeof(buf_dskew));
    memset(&buf_denoise, 0, sizeof(buf_denoise));
    memset(&buf_deblur, 0, sizeof(buf_deblur));
    memset(&buf_upscale, 0, sizeof(buf_upscale));
    memset(&buf_sharpen, 0, sizeof(buf_sharpen));
    memset(&buf_clahe, 0, sizeof(buf_clahe));
    memset(&buf_bin, 0, sizeof(buf_bin));

    #define STAGE_BEGIN(name) (t0 = now_s())
    #define STAGE_END(stage_name, ran_, param_) do { \
        if (stage_i < 16) { \
            rep.stages[stage_i].name = (stage_name); \
            rep.stages[stage_i].ran = (ran_); \
            rep.stages[stage_i].ms = (now_s() - t0) * 1000.0; \
            rep.stages[stage_i].param = (param_); \
            stage_i++; \
        } \
    } while (0)

    /* ---- 1. deskew ---- */
    STAGE_BEGIN("deskew");
    if (c.deskew && fabsf(assess_own->skew_deg) >= c.min_skew_deg) {
        buf_dskew = alloc_like(arena, ch, cw, C, src->dtype);
        if (!buf_dskew.data) return 0;
        float ang = 0;
        if (sr_deskew(cur, &buf_dskew, c.min_skew_deg, c.max_skew_deg, &ang, arena)) {
            cur = &buf_dskew;
            STAGE_END("deskew", true, ang);
        } else {
            STAGE_END("deskew", false, 0.0f);
        }
    } else {
        STAGE_END("deskew", false, 0.0f);
    }

    /* ---- 2. denoise ---- */
    STAGE_BEGIN("denoise");
    float strength = c.denoise_strength;
    if (strength < 0.0f) {
        strength = (assess_own->noise_sigma - 3.0f) / 15.0f;
        if (strength < 0.2f) strength = 0.2f;
        if (strength > 1.0f) strength = 1.0f;
    }
    if (assess_own->noise_sigma > 3.0f &&
        c.style != SR_STYLE_SHARP && c.style != SR_STYLE_UPSCALE) {
        buf_denoise = alloc_like(arena, ch, cw, C, src->dtype);
        if (!buf_denoise.data) return 0;
        float eps = 0.002f + 0.18f * strength;
        if (sr_guided_denoise(cur, &buf_denoise, 6, eps, arena)) {
            cur = &buf_denoise;
            STAGE_END("denoise", true, eps);
        } else {
            STAGE_END("denoise", false, 0.0f);
        }
    } else {
        STAGE_END("denoise", false, 0.0f);
    }

    /* ---- 3. deblur (Wiener, FFT) ---- */
    STAGE_BEGIN("deblur");
    if (c.deblur && assess_own->blur_score < 60.0f &&
        c.style != SR_STYLE_SHARP && c.style != SR_STYLE_UPSCALE) {
        float lam = (assess_own->noise_sigma / 255.0f);
        lam = lam * lam * 0.5f;      /* relative Wiener constant (~1e-3..1e-2) */
        buf_deblur = alloc_like(arena, ch, cw, C, src->dtype);
        if (!buf_deblur.data) return 0;
        if (lam < 1e-4f) lam = 1e-4f;
        if (sr_wiener_deblur(cur, &buf_deblur, c.psf, c.psf_size, 0.0f, lam, arena)) {
            cur = &buf_deblur;
            STAGE_END("deblur", true, lam);
        } else {
            STAGE_END("deblur", false, 0.0f);
        }
    } else {
        STAGE_END("deblur", false, 0.0f);
    }

    /* ---- 4. upscale (Lanczos-3) — runs for every style when scale > 1 ---- */
    STAGE_BEGIN("upscale");
    if (c.scale > 1) {
        buf_upscale = alloc_like(arena, ch * (size_t)c.scale, cw * (size_t)c.scale,
                                 C, src->dtype);
        if (!buf_upscale.data) return 0;
        if (sr_resize_lanczos(cur, &buf_upscale, arena)) {
            cur = &buf_upscale;
            ch = ch * (size_t)c.scale;
            cw = cw * (size_t)c.scale;
            STAGE_END("upscale", true, (float)c.scale);
        } else {
            STAGE_END("upscale", false, 0.0f);
        }
    } else {
        STAGE_END("upscale", false, (float)c.scale);
    }

    /* ---- 5. sharpen (edge-masked unsharp) ---- */
    STAGE_BEGIN("sharpen");
    float amount = c.sharpen_amount;
    if (amount < 0.0f) {
        if (assess_own->blur_score < 60.0f) amount = 0.8f;
        else if (assess_own->blur_score < 200.0f) amount = 0.5f;
        else amount = 0.25f;
    }
    if (amount > 0.0f && c.style != SR_STYLE_CLEAN) {
        buf_sharpen = alloc_like(arena, ch, cw, C, src->dtype);
        if (!buf_sharpen.data) return 0;
        if (sr_sharpen_edge(cur, &buf_sharpen, amount, arena)) {
            cur = &buf_sharpen;
            STAGE_END("sharpen", true, amount);
        } else {
            STAGE_END("sharpen", false, 0.0f);
        }
    } else {
        STAGE_END("sharpen", false, amount);
    }

    /* ---- 6. CLAHE ---- */
    STAGE_BEGIN("clahe");
    if (c.clahe && assess_own->contrast < 0.55f) {
        buf_clahe = alloc_like(arena, ch, cw, C, src->dtype);
        if (!buf_clahe.data) return 0;
        if (sr_clahe(cur, &buf_clahe, c.clahe_grid, c.clahe_clip, arena)) {
            cur = &buf_clahe;
            STAGE_END("clahe", true, assess_own->contrast);
        } else {
            STAGE_END("clahe", false, 0.0f);
        }
    } else {
        STAGE_END("clahe", false, 0.0f);
    }

    /* ---- 7. binarize ---- */
    STAGE_BEGIN("binarize");
    if (c.style == SR_STYLE_BINARIZE) {
        buf_bin = alloc_like(arena, ch, cw, C, RCRE_DTYPE_U8);
        if (!buf_bin.data) return 0;
        int block = (int)(2.0 * (ch < cw ? ch : cw) / 40.0) | 1;
        if (block < 15) block = 15;
        if (block > 51) block = 51;
        if (sr_adaptive_threshold(cur, &buf_bin, block, 0.25f, arena)) {
            sr_morphology_open_close(&buf_bin, 1, arena);
            cur = &buf_bin;
            STAGE_END("binarize", true, (float)block);
        } else {
            STAGE_END("binarize", false, 0.0f);
        }
    } else {
        STAGE_END("binarize", false, 0.0f);
    }

    /* ---- write to dst ---- */
    if (cur == src) {
        /* no stage ran: copy (dst dims must equal src) */
        if (dst->shape[0] != src->shape[0] || dst->shape[1] != src->shape[1]) return 0;
        size_t n = h * w * C;
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
    } else {
        if (dst->shape[0] != ch || dst->shape[1] != cw) return 0;
        size_t n = ch * cw * C;
        if (cur->dtype == RCRE_DTYPE_U8 && dst->dtype == RCRE_DTYPE_U8) {
            memcpy(dst->data, cur->data, n);
        } else {
            for (size_t i = 0; i < n; i++) {
                float v = (cur->dtype == RCRE_DTYPE_U8)
                              ? (float)((const uint8_t*)cur->data)[i]
                              : ((const float*)cur->data)[i];
                if (dst->dtype == RCRE_DTYPE_U8) {
                    int iv = (int)lrintf(v);
                    ((uint8_t*)dst->data)[i] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
                } else {
                    ((float*)dst->data)[i] = v;
                }
            }
        }
    }

    rep.n_stages = stage_i;
    rep.total_ms = (now_s() - t_start) * 1000.0;
    rep.peak_mem_bytes = arena->high_watermark;
    rep.out_w = (int)cw;
    rep.out_h = (int)ch;

    if (report) *report = rep;
    return 1;
}

/* ========================================================================= */
/* Terminal report                                                            */
/* ========================================================================= */

void sr_print_report(const SRReport* rep, FILE* out) {
    if (!rep) return;
    fprintf(out, "================================================================\n");
    fprintf(out, "   RCRE-SR Pipeline Report\n");
    fprintf(out, "================================================================\n");
    if (rep->before) {
        fprintf(out, "  Input assessment (subsampled %dx%d):\n",
                rep->before->w, rep->before->h);
        fprintf(out, "    blur score   : %8.1f  (Laplacian variance; higher = sharper)\n",
                rep->before->blur_score);
        fprintf(out, "    noise sigma  : %8.2f  (0..255 robust estimate)\n",
                rep->before->noise_sigma);
        fprintf(out, "    contrast     : %8.3f  (p2..p98 range, 0..1)\n",
                rep->before->contrast);
        fprintf(out, "    skew angle   : %8.2f deg\n", rep->before->skew_deg);
    }
    fprintf(out, "  Stages:\n");
    for (int i = 0; i < rep->n_stages; i++) {
        const SRStageStat* s = &rep->stages[i];
        fprintf(out, "    %-10s %s  %8.3f ms   param=%6.3f\n",
                s->name, s->ran ? "RUN " : "skip", s->ms, s->param);
    }
    fprintf(out, "  ----------------------------------------------------------------\n");
    fprintf(out, "  TOTAL: %.2f ms | output %dx%d | arena HWM %.1f MB\n",
            rep->total_ms, rep->out_w, rep->out_h,
            (double)rep->peak_mem_bytes / 1e6);
    fprintf(out, "================================================================\n");
}
