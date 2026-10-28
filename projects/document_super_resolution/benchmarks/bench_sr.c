/**
 * @file bench_sr.c
 * @brief RCRE-SR benchmarks: per-stage pipeline timing + memory on a
 *        realistic document, plus kernel micro-benchmarks.
 *
 * Usage:
 *   bench_sr [input.pgm]
 * Without an input file a synthetic 1200x1600 document is generated.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "sr.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* --- synthetic document (text-like strokes on white paper) -------------- */
static void make_doc(uint8_t* img, size_t w, size_t h) {
    memset(img, 255, w * h);
    uint64_t state = 123456789u;
    for (size_t y = 48; y + 40 < h; y += 64) {
        size_t x = 48;
        while (x + 48 < w) {
            /* a "word": 3..8 strokes with serif-ish caps */
            int nst = 3 + (int)((state = state * 6364136223846793005ULL +
                                 1442695040888963407ULL) % 6);
            for (int s = 0; s < nst; s++) {
                int sw = 4 + (int)((state = state * 6364136223846793005ULL +
                                    1442695040888963407ULL) % 6);
                int top = (int)(y + 4 + (state >> 8) % 8);
                for (int dy = 0; dy < 28; dy++)
                    for (int dx = 0; dx < sw; dx++) {
                        int on = (dx == 0 || dx == sw - 1 || dy == 0 ||
                                  dy == 27 || (dy == 13 && dx > 0 && dx < sw - 1) ||
                                  (dy < 4 && dx == sw / 2));
                        if (on) img[(size_t)(top + dy) * w + x + dx] = 24;
                    }
                x += (size_t)sw + 6;
                if (x + 48 >= w) break;
            }
            x += 24; /* word gap */
        }
    }
}

/* --- light degradation (blur + downscale + noise) for deblur stage ------ */
static void degrade(uint8_t* img, size_t w, size_t h, RCRE_Arena* a) {
    size_t shape[3] = {h, w, 1};
    RCRE_Tensor f = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, a);
    RCRE_Tensor g = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, a);
    for (size_t i = 0; i < w * h; i++) ((float*)f.data)[i] = (float)img[i];
    if (rcre_cv_gaussian_blur(&f, &g, 1.2f, a) == RCRE_OK) {
        for (size_t i = 0; i < w * h; i++) {
            int v = (int)lrintf(((float*)g.data)[i]);
            img[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    uint64_t st = 987654321u;
    for (size_t i = 0; i < w * h; i++) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        int n = (int)((st >> 33) % 15) - 7;
        int v = (int)img[i] + n;
        img[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
}

int main(int argc, char** argv) {
    printf("================================================================\n");
    printf("   RCRE-SR Benchmarks\n");
    printf("================================================================\n");

    size_t w = 1200, h = 1600;
    uint8_t* doc = (uint8_t*)malloc(w * h);
    if (!doc) return 1;
    if (argc > 1) {
        RCRE_Arena* tmp = rcre_arena_create(1 << 28);
        RCRE_Tensor tin;
        if (sr_pnm_read(argv[1], &tin, tmp)) {
            w = tin.shape[1];
            h = tin.shape[0];
            free(doc);
            doc = (uint8_t*)malloc(w * h);
            for (size_t i = 0; i < w * h; i++)
                doc[i] = (tin.dtype == RCRE_DTYPE_U8)
                            ? ((const uint8_t*)tin.data)[i]
                            : (uint8_t)((const float*)tin.data)[i];
            printf("  input: %s (%zux%zu)\n", argv[1], w, h);
        } else {
            fprintf(stderr, "cannot read %s, using synthetic document\n", argv[1]);
            make_doc(doc, w, h);
        }
        rcre_arena_destroy(tmp);
    } else {
        make_doc(doc, w, h);
        printf("  input: synthetic document (%zux%zu)\n", w, h);
    }

    RCRE_Arena* a = rcre_arena_create(1ULL << 31);
    size_t shape[3] = {h, w, 1};
    RCRE_Tensor in = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
    memcpy(in.data, doc, w * h);

    /* ---- 1. full pipeline, 2x upscale (all stages) --------------------- */
    SRConfig cfg;
    sr_config_default(&cfg);
    cfg.scale = 2;
    size_t oh = h * 2, ow = w * 2;
    size_t oshape[3] = {oh, ow, 1};
    RCRE_Tensor out = rcre_tensor_create(oshape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);

    SRReport rep;
    double t0 = now_ms();
    if (!sr_enhance(&in, &out, &cfg, a, &rep)) {
        fprintf(stderr, "pipeline failed\n");
        return 1;
    }
    double wall = now_ms() - t0;

    double mpps_in = (double)(w * h) / 1e6 / (wall / 1e3);
    double mpps_out = (double)(w * h * 4) / 1e6 / (wall / 1e3);
    printf("\n  [pipeline 2x, sharp input]  wall = %8.2f ms   "
           "(%.1f MPix/s in | %.1f MPix/s out)\n", wall, mpps_in, mpps_out);
    for (int i = 0; i < rep.n_stages; i++) {
        const SRStageStat* s = &rep.stages[i];
        printf("    %-10s %s  %8.3f ms   param=%6.3f\n",
               s->name, s->ran ? "on " : "off", s->ms, s->param);
    }
    printf("    arena HWM: %.1f MB\n",
           (double)rep.peak_mem_bytes / (1024.0 * 1024.0));

    /* ---- 1b. same pipeline on a degraded copy (restoration stages fire) */
    uint8_t* ddoc = (uint8_t*)malloc(w * h);
    memcpy(ddoc, doc, w * h);
    degrade(ddoc, w, h, a);
    RCRE_Tensor din = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
    memcpy(din.data, ddoc, w * h);
    SRReport drep;
    double t1 = now_ms();
    sr_enhance(&din, &out, &cfg, a, &drep);
    double dwall = now_ms() - t1;
    printf("\n  [pipeline 2x, degraded input]  wall = %8.2f ms\n", dwall);
    for (int i = 0; i < drep.n_stages; i++) {
        const SRStageStat* s = &drep.stages[i];
        printf("    %-10s %s  %8.3f ms   param=%6.3f\n",
               s->name, s->ran ? "on " : "off", s->ms, s->param);
    }
    free(ddoc);

    /* ---- 2. kernel micro-benchmarks (median of 5) ---------------------- */
    struct { const char* name; double* t; } kb[6];
    double t_assess = 0, t_resize = 0, t_clahe = 0, t_sharp = 0,
           t_denoise = 0, t_wiener = 0;
    kb[0].name = "assess        "; kb[0].t = &t_assess;
    kb[1].name = "lanczos 2x    "; kb[1].t = &t_resize;
    kb[2].name = "clahe         "; kb[2].t = &t_clahe;
    kb[3].name = "sharpen       "; kb[3].t = &t_sharp;
    kb[4].name = "guided denoise"; kb[4].t = &t_denoise;
    kb[5].name = "wiener deblur "; kb[5].t = &t_wiener;

    RCRE_Tensor up = rcre_tensor_create(oshape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
    RCRE_Tensor mid = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, a);
    SRAssessment as;
    for (int it = 0; it < 5; it++) {
        double t;
        t = now_ms(); sr_assess(&in, &as); t = now_ms() - t; if (it) t_assess += t;
        t = now_ms(); sr_resize_lanczos(&in, &up, a); t = now_ms() - t; if (it) t_resize += t;
        t = now_ms(); sr_clahe(&in, &mid, 8, 2.0f, a); t = now_ms() - t; if (it) t_clahe += t;
        t = now_ms(); sr_sharpen_edge(&in, &mid, 0.5f, a); t = now_ms() - t; if (it) t_sharp += t;
        t = now_ms(); sr_guided_denoise(&in, &mid, 6, 0.05f, a); t = now_ms() - t; if (it) t_denoise += t;
        t = now_ms(); sr_wiener_deblur(&in, &mid, SR_PSF_GAUSSIAN, 4.0f, 0.0f, 1e-3f, a);
        t = now_ms() - t; if (it) t_wiener += t;
    }
    printf("\n  [kernels, %zux%zu, median of 5]\n", w, h);
    for (int i = 0; i < 6; i++) {
        double ms = (*kb[i].t) / 4.0; /* 4 iters accumulated */
        double mpx = (double)(w * h) / 1e6;
        printf("    %-16s %9.2f ms   (%.1f MPix/s)\n", kb[i].name, ms, mpx / (ms / 1e3));
    }

    printf("\n  arena HWM (all): %.1f MB\n",
           (double)a->high_watermark / (1024.0 * 1024.0));
    printf("================================================================\n");
    rcre_arena_destroy(a);
    free(doc);
    return 0;
}
