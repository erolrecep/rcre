/**
 * @file sr.h
 * @brief RCRE-SR — Document Image Super-Resolution & Enhancement (ISO C23)
 *
 * Purpose:
 *   Turn degraded document/photo images (blur, noise, low resolution, skew,
 *   low contrast) into crystal-clear, sharp outputs that are dramatically
 *   easier for OCR engines to read.
 *
 * Design contract (professional guarantees):
 *   1. Structure-preserving: no stage may hallucinate ink. Every enhancement
 *      is edge/contrast/resolution-preserving and is gated by measured image
 *      statistics (sr_assess), so an already-sharp image is never degraded.
 *   2. Built entirely on RCRE primitives (arena, tensors, FFT, integral
 *      images, warp). Zero external dependencies.
 *   3. Objective: quality is measured by PSNR/SSIM + OCR accuracy (CER)
 *      against ground truth, not by eye. See benchmarks/bench_sr.c.
 *
 * Usage (one-call):
 *   #define RCRE_IMPLEMENTATION
 *   #include "rcre.h"
 *   #include "sr.h"
 *
 *   RCRE_Arena* a = rcre_arena_create(1 << 28);
 *   RCRE_Tensor in  = sr_pnm_read("scan.pgm", a);
 *   RCRE_Tensor out;                       // caller-created, HWC U8
 *   SRConfig cfg; sr_config_default(&cfg);
 *   SRReport rep;
 *   sr_enhance(&in, &out, &cfg, a, &rep);
 *   sr_print_report(&rep, stdout);
 *
 *   @note All images are RCRE_Tensor, layout HWC, dtype U8 (pixel data) or
 *   F32 (internal/float paths). Width = shape[1], Height = shape[0].
 */
#ifndef SR_H
#define SR_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Configuration & report                                                     */
/* ========================================================================= */

typedef enum {
    SR_STYLE_AUTO      = 0, /* adaptive: denoise/deblur/sharpen/CLAHE as measured */
    SR_STYLE_SHARP     = 1, /* upscale + edge-masked unsharp only                 */
    SR_STYLE_CLEAN     = 2, /* denoise + deblur + gentle sharpen, no binarize     */
    SR_STYLE_BINARIZE  = 3, /* full pipeline + adaptive threshold (black/white)   */
    SR_STYLE_UPSCALE   = 4  /* pure Lanczos-3 upscale, no enhancement (baseline)  */
} SR_Style;

typedef enum {
    SR_PSF_NONE = 0,
    SR_PSF_DEFOCUS = 1,  /* uniform disk, radius = psf_size / 2        */
    SR_PSF_MOTION  = 2,  /* motion line, length = psf_size, angle deg  */
    SR_PSF_GAUSSIAN = 3  /* gaussian blob, sigma = psf_size / 2 (camera blur) */
} SR_PSFKind;

typedef struct {
    int      scale;              /* upscale factor 1..8 (default 2)            */
    SR_Style style;              /* default SR_STYLE_AUTO                      */
    bool     deskew;             /* estimate & correct skew (default true)     */
    bool     deblur;             /* FFT-Wiener deblur, gated by blur score     */
    SR_PSFKind psf;              /* PSF model when deblurring (defocus default)*/
    float    psf_size;           /* PSF diameter in source pixels (3..11)      */
    float    sharpen_amount;     /* 0..2, -1 = auto from residual blur         */
    float    denoise_strength;   /* 0..1,  -1 = auto from noise estimate       */
    float    min_skew_deg;       /* correct skew only if |angle| > this (0.3)  */
    float    max_skew_deg;       /* search range ± (7.5)                       */
    bool     clahe;              /* adaptive histogram contrast (default true) */
    int      clahe_grid;         /* tiles per axis (8)                         */
    float    clahe_clip;         /* clip limit, 4*median histogram (3.0)       */
} SRConfig;

void sr_config_default(SRConfig* cfg);

typedef struct {
    const char* name;    /* stage id                                          */
    bool        ran;     /* did the stage actually execute (gating)           */
    double      ms;      /* wall time                                         */
    float       param;   /* key parameter used (angle, threshold, sigma, ...) */
} SRStageStat;

/* forward declaration; full definition below (SRAssessment struct) */
typedef struct SRAssessment SRAssessment;

typedef struct {
    SRAssessment* before;     /* assessment of the input (owned by report)    */
    SRStageStat   stages[16];
    int           n_stages;
    double        total_ms;
    size_t        peak_mem_bytes;  /* arena high-water mark during pipeline   */
    int           out_w, out_h;
} SRReport;

/* ========================================================================= */
/* Assessment (measured image statistics — drives adaptive gating)           */
/* ========================================================================= */

struct SRAssessment {
    int     w, h;
    float   blur_score;    /* variance of Laplacian; HIGHER = sharper         */
    float   noise_sigma;   /* robust noise estimate, 0..255 scale             */
    float   contrast;      /* (p98-p2) percentile range, 0..1                 */
    float   skew_deg;      /* estimated skew (deg, + = CCW)                    */
};

/**
 * @brief Measures blur / noise / contrast / skew of a single-channel image.
 * @param gray HWC U8 or F32 tensor (first channel used).
 */
int sr_assess(const RCRE_Tensor* gray, SRAssessment* out);

/* ========================================================================= */
/* Core super-resolution primitives                                           */
/* ========================================================================= */

/**
 * @brief High-quality arbitrary-scale resize using separable Lanczos-3.
 *        The document-imaging standard: preserves stroke edges at 2..4x.
 * @param src  HWC U8 or F32
 * @param dst  caller-created HWC tensor of size (h*scale_h, w*scale_w)
 */
int sr_resize_lanczos(const RCRE_Tensor* src, RCRE_Tensor* dst, RCRE_Arena* arena);

/**
 * @brief Cubic-convolution (a = -0.5) resize — faster, slightly softer.
 */
int sr_resize_bicubic(const RCRE_Tensor* src, RCRE_Tensor* dst, RCRE_Arena* arena);

/**
 * @brief Edge-masked unsharp masking: sharpens only where structure exists
 *        (Sobel magnitude gate), so flat regions stay noise-free.
 * @param amount 0..2 (0.8 typical)
 */
int sr_sharpen_edge(const RCRE_Tensor* src, RCRE_Tensor* dst,
                    float amount, RCRE_Arena* arena);

/**
 * @brief Contrast-Limited Adaptive Histogram Equalization (CLAHE).
 *        Tile-wise histogram, clip + redistribute, bilinear CDF blending.
 */
int sr_clahe(const RCRE_Tensor* src, RCRE_Tensor* dst,
             int grid, float clip_limit, RCRE_Arena* arena);

/**
 * @brief Otsu global threshold; returns 1 on success, stores threshold.
 */
int sr_otsu_threshold(const RCRE_Tensor* src, float* threshold);

/**
 * @brief Adaptive (local mean) thresholding — robust to uneven illumination.
 *        dst is U8 {0,255}, inverted=false means dark text on light paper.
 */
int sr_adaptive_threshold(const RCRE_Tensor* src, RCRE_Tensor* dst,
                          int block_size, float const_c, RCRE_Arena* arena);

/**
 * @brief 3x3 morphological open/close cleanup of a binary image (in-place ok).
 */
int sr_morphology_open_close(RCRE_Tensor* bin, int iterations, RCRE_Arena* arena);

/* ========================================================================= */
/* Restoration (gated — only run when assessment says the image needs it)     */
/* ========================================================================= */

/**
 * @brief Edge-preserving guided filter denoise, O(N) via integral images.
 * @param radius  support radius in px (4..12)
 * @param eps    regularization, 0..0.05 (bigger = less detail kept)
 */
int sr_guided_denoise(const RCRE_Tensor* src, RCRE_Tensor* dst,
                      int radius, float eps, RCRE_Arena* arena);

/**
 * @brief Wiener deconvolution in the frequency domain (rcre_fft_2d).
 *        Zero-pads to power-of-2 to avoid circular wrap-around.
 * @param psf      model of the blur kernel
 * @param lambda   RELATIVE Wiener regularization (1e-4..0.05 typical;
 *                 lambda ~= (sigma_noise/255)^2 * 0.5)
 */
int sr_wiener_deblur(const RCRE_Tensor* src, RCRE_Tensor* dst,
                     SR_PSFKind psf, float psf_size, float psf_angle_deg,
                     float lambda, RCRE_Arena* arena);

/**
 * @brief Estimates & corrects skew via projection-profile maximization
 *        (coarse-to-fine). Bakes the rotation into dst (bilinear warp).
 */
int sr_deskew(const RCRE_Tensor* src, RCRE_Tensor* dst,
              float min_angle, float max_angle, float* angle_out,
              RCRE_Arena* arena);

/* ========================================================================= */
/* Pipeline                                                                   */
/* ========================================================================= */

/**
 * @brief The one-call document super-resolution pipeline.
 *
 *   assess → [deskew] → [denoise] → [deblur] → Lanczos-3 upscale ×scale
 *          → edge-masked unsharp → [CLAHE] → [adaptive threshold]
 *
 * Gating: denoise runs iff noise_sigma > 3.0; deblur iff blur_score < 60
 * (Laplacian-variance scale) AND cfg->deblur; deskew iff |skew| > min;
 * CLAHE iff contrast < 0.55. See sr_pipeline.c for the exact policy.
 *
 * @param src   input HWC U8
 * @param dst   caller-created HWC U8 tensor (any size; pipeline fills it)
 * @param report optional (NULL ok)
 * @return 1 on success
 */
int sr_enhance(const RCRE_Tensor* src, RCRE_Tensor* dst,
               const SRConfig* cfg, RCRE_Arena* arena, SRReport* report);

void sr_print_report(const SRReport* rep, FILE* out);

/* ========================================================================= */
/* Metrics (objective quality)                                                */
/* ========================================================================= */

double sr_psnr_db(const RCRE_Tensor* a, const RCRE_Tensor* b, double max_val);
double sr_ssim(const RCRE_Tensor* a, const RCRE_Tensor* b);  /* gray, 0..1 */

/* ========================================================================= */
/* I/O — PGM/PPM (P2/P3 ASCII, P5/P6 binary); keeps the stack zero-dep       */
/* ========================================================================= */

/**
 * @brief Reads PGM (1ch) / PPM (3ch) P2/P3/P5/P6 into a U8 HWC tensor.
 */
int sr_pnm_read(const char* path, RCRE_Tensor* out, RCRE_Arena* arena);

/**
 * @brief Writes U8 HWC tensor as PGM (1ch) / PPM (3ch) binary (P5/P6).
 */
int sr_pnm_write(const char* path, const RCRE_Tensor* img);

/**
 * @brief Write an 8-bit PNG (grayscale or RGB), no external dependencies.
 *        Uses raw DEFLATE "stored" blocks — valid PNG, zero compression.
 *        Needed e.g. to hand images to Tesseract (no PGM support there).
 */
int sr_png_write(const char* path, const RCRE_Tensor* img);

#ifdef __cplusplus
}
#endif

#endif /* SR_H */
