/**
 * @file resize_gm.c
 * @brief GraphicsMagick-Exact Resampling Tier for RCRE Tensors
 *
 * Provides a bit-for-bit mathematical port of GraphicsMagick 1.3.49's
 * HorizontalFilter/VerticalFilter resamplers, operating directly on RCRE
 * continuous F32 tensors across 16 continuous windowed filter kernels.
 */

#include "rcre.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const double RCRE_GM_EPS = 1.0e-12;
static const double RCRE_GM_PI  = 3.14159265358979323846264338327950288419716939937510;

/* ---------------------------------------------------------------------
 * Bessel J1 & Asymptotic Approximations (GM 1.3.49 Cephes port)
 * ------------------------------------------------------------------ */
static double rcre_gm_j1(double x) {
    static const double Pone[] = {
        0.581199354001606143928050809e+21,
       -0.6672106568924916298020941484e+20,
        0.2316433580634002297931815435e+19,
       -0.3588817569910106050743641413e+17,
        0.2908795263834775409737601689e+15,
       -0.1322983480332126453125473247e+13,
        0.3413234182301700539091292655e+10,
       -0.4695753530642995859767162166e+7,
        0.270112271089232341485679099e+4
    };
    static const double Qone[] = {
        0.11623987080032122878585294e+22,
        0.1185770712190320999837113348e+20,
        0.6092061398917521746105196863e+17,
        0.2081661221307607351240184229e+15,
        0.5243710262167649715406728642e+12,
        0.1013863514358673989967045588e+10,
        0.1501793594998585505921097578e+7,
        0.1606931573481487801970916749e+4,
        0.1e+1
    };
    double p = Pone[8], q = Qone[8];
    for (long i = 7; i >= 0; i--) {
        p = p * x * x + Pone[i];
        q = q * x * x + Qone[i];
    }
    return p / q;
}

static double rcre_gm_p1(double x) {
    static const double Pone[] = {
        0.352246649133679798341724373e+5,
        0.62758845247161281269005675e+5,
        0.313539631109159574238669888e+5,
        0.49854832060594338434500455e+4,
        0.2111529182853962382105718e+3,
        0.12571716929145341558495e+1
    };
    static const double Qone[] = {
        0.352246649133679798068390431e+5,
        0.626943469593560511888833731e+5,
        0.312404063819041039923015703e+5,
        0.4930396490181088979386097e+4,
        0.2030775189134759322293574e+3,
        0.1e+1
    };
    const double r = 8.0 / x;
    double p = Pone[5], q = Qone[5];
    for (long i = 4; i >= 0; i--) {
        p = p * r * r + Pone[i];
        q = q * r * r + Qone[i];
    }
    return p / q;
}

static double rcre_gm_q1(double x) {
    static const double Pone[] = {
        0.3511751914303552822533318e+3,
        0.7210391804904475039280863e+3,
        0.4259873011654442389886993e+3,
        0.831898957673850827325226e+2,
        0.45681716295512267064405e+1,
        0.3532840052740123642735e-1
    };
    static const double Qone[] = {
        0.74917374171809127714519505e+4,
        0.154141773392650970499848051e+5,
        0.91522317015169922705904727e+4,
        0.18111867005523513506724158e+4,
        0.1038187585462133728776636e+3,
        0.1e+1
    };
    const double r = 8.0 / x;
    double p = Pone[5], q = Qone[5];
    for (long i = 4; i >= 0; i--) {
        p = p * r * r + Pone[i];
        q = q * r * r + Qone[i];
    }
    return p / q;
}

static double rcre_gm_bessel1(double x) {
    if (x == 0.0) return 0.0;
    const double p = x;
    if (x < 0.0) x = -x;
    if (x < 8.0) return p * rcre_gm_j1(x);
    const double sq2 = 1.0 / sqrt(2.0);
    const double s = sin(x), c = cos(x);
    const double q = sqrt(2.0 / (RCRE_GM_PI * x)) *
        (rcre_gm_p1(x) * (sq2 * (s - c)) -
         8.0 / x * rcre_gm_q1(x) * (-sq2 * (s + c)));
    return (p < 0.0) ? -q : q;
}

/* ---------------------------------------------------------------------
 * 16 GM Filter Kernel Functions
 * ------------------------------------------------------------------ */
static double rcre_gm_filter(int which, double x, double sup) {
    switch (which) {
    case RCRE_CV_INTERP_GM_POINT:
    case RCRE_CV_INTERP_GM_BOX:
        return (x < -0.5) ? 0.0 : ((x < 0.5) ? 1.0 : 0.0);
    case RCRE_CV_INTERP_GM_TRIANGLE:
        if (x < -1.0) return 0.0;
        if (x < 0.0)  return 1.0 + x;
        if (x < 1.0)  return 1.0 - x;
        return 0.0;
    case RCRE_CV_INTERP_GM_HERMITE:
        if (x < -1.0) return 0.0;
        if (x < 0.0)  { const double u = -x; return (2.0 * u - 3.0) * u * u + 1.0; }
        if (x < 1.0)  return (2.0 * x - 3.0) * x * x + 1.0;
        return 0.0;
    case RCRE_CV_INTERP_GM_HANNING:
        return 0.5 + 0.5 * cos(RCRE_GM_PI * x);
    case RCRE_CV_INTERP_GM_HAMMING:
        return 0.54 + 0.46 * cos(RCRE_GM_PI * x);
    case RCRE_CV_INTERP_GM_BLACKMAN:
        return 0.42 + 0.5 * cos(RCRE_GM_PI * x) + 0.08 * cos(2.0 * RCRE_GM_PI * x);
    case RCRE_CV_INTERP_GM_GAUSSIAN:
        return exp(-2.0 * x * x) * sqrt(2.0 / RCRE_GM_PI);
    case RCRE_CV_INTERP_GM_QUADRATIC:
        if (x < -1.5) return 0.0;
        if (x < -0.5) return 0.5 * (x + 1.5) * (x + 1.5);
        if (x <  0.5) return 0.75 - x * x;
        if (x <  1.5) return 0.5 * (x - 1.5) * (x - 1.5);
        return 0.0;
    case RCRE_CV_INTERP_GM_CUBIC:
        if (x < -2.0) return 0.0;
        if (x < -1.0) { const double u = 2.0 + x; return u * u * u / 6.0; }
        if (x <  0.0) return (4.0 + x * x * (-6.0 - 3.0 * x)) / 6.0;
        if (x <  1.0) return (4.0 + x * x * (-6.0 + 3.0 * x)) / 6.0;
        if (x <  2.0) { const double u = 2.0 - x; return u * u * u / 6.0; }
        return 0.0;
    case RCRE_CV_INTERP_GM_CATROM:
        if (x < -2.0) return 0.0;
        if (x < -1.0) return 0.5 * (4.0 + x * (8.0 + x * (5.0 + x)));
        if (x <  0.0) return 0.5 * (2.0 + x * x * (-5.0 - 3.0 * x));
        if (x <  1.0) return 0.5 * (2.0 + x * x * (-5.0 + 3.0 * x));
        if (x <  2.0) return 0.5 * (4.0 + x * (-8.0 + x * (5.0 - x)));
        return 0.0;
    case RCRE_CV_INTERP_GM_MITCHELL: {
        const double B = 1.0 / 3.0, C = 1.0 / 3.0;
        const double P0 = (6.0 - 2.0 * B) / 6.0;
        const double P2 = (-18.0 + 12.0 * B + 6.0 * C) / 6.0;
        const double P3 = (12.0 - 9.0 * B - 6.0 * C) / 6.0;
        const double Q0 = (8.0 * B + 24.0 * C) / 6.0;
        const double Q1 = (-12.0 * B - 48.0 * C) / 6.0;
        const double Q2 = (6.0 * B + 30.0 * C) / 6.0;
        const double Q3 = (-1.0 * B - 6.0 * C) / 6.0;
        if (x < -2.0) return 0.0;
        if (x < -1.0) return Q0 - x * (Q1 - x * (Q2 - x * Q3));
        if (x <  0.0) return P0 + x * x * (P2 - x * P3);
        if (x <  1.0) return P0 + x * x * (P2 + x * P3);
        if (x <  2.0) return Q0 + x * (Q1 + x * (Q2 + x * Q3));
        return 0.0;
    }
    case RCRE_CV_INTERP_GM_LANCZOS: {
        if (x < -3.0 || x >= 3.0) return 0.0;
        const double a = (x == 0.0) ? 1.0 : sin(RCRE_GM_PI * x) / (RCRE_GM_PI * x);
        const double x3 = x / 3.0;
        const double b = (x3 == 0.0) ? 1.0 : sin(RCRE_GM_PI * x3) / (RCRE_GM_PI * x3);
        return a * b;
    }
    case RCRE_CV_INTERP_GM_BESSEL: {
        const double bm = 0.42 + 0.5 * cos(RCRE_GM_PI * (x / sup)) + 0.08 * cos(2.0 * RCRE_GM_PI * (x / sup));
        const double bs = (x == 0.0) ? (RCRE_GM_PI / 4.0) : (rcre_gm_bessel1(RCRE_GM_PI * x) / (2.0 * x));
        return bm * bs;
    }
    case RCRE_CV_INTERP_GM_SINC: {
        const double bm = 0.42 + 0.5 * cos(RCRE_GM_PI * (x / sup)) + 0.08 * cos(2.0 * RCRE_GM_PI * (x / sup));
        const double sc = (x == 0.0) ? 1.0 : sin(RCRE_GM_PI * x) / (RCRE_GM_PI * x);
        return bm * sc;
    }
    default:
        return 0.0;
    }
}

static double rcre_gm_support(int which) {
    switch (which) {
    case RCRE_CV_INTERP_GM_POINT:     return 0.0;
    case RCRE_CV_INTERP_GM_BOX:       return 0.5;
    case RCRE_CV_INTERP_GM_TRIANGLE:  return 1.0;
    case RCRE_CV_INTERP_GM_HERMITE:   return 1.0;
    case RCRE_CV_INTERP_GM_HANNING:   return 1.0;
    case RCRE_CV_INTERP_GM_HAMMING:   return 1.0;
    case RCRE_CV_INTERP_GM_BLACKMAN:  return 1.0;
    case RCRE_CV_INTERP_GM_GAUSSIAN:  return 1.25;
    case RCRE_CV_INTERP_GM_QUADRATIC: return 1.5;
    case RCRE_CV_INTERP_GM_CUBIC:     return 2.0;
    case RCRE_CV_INTERP_GM_CATROM:    return 2.0;
    case RCRE_CV_INTERP_GM_MITCHELL:  return 2.0;
    case RCRE_CV_INTERP_GM_LANCZOS:   return 3.0;
    case RCRE_CV_INTERP_GM_BESSEL:    return 3.2383;
    case RCRE_CV_INTERP_GM_SINC:      return 4.0;
    default:                          return 1.0;
    }
}

/* ---------------------------------------------------------------------
 * Sampling Tap Weights
 * ------------------------------------------------------------------ */
typedef struct {
    int start;
    int nt;
    const double* wgt;
} RcreGmTap;

static int rcre_gm_build_taps(RcreGmTap** out_taps, double** out_weights,
                              unsigned dst_len, unsigned src_len,
                              int filter, RCRE_Arena* arena) {
    const double f = (double)dst_len / (double)src_len;
    const double invf = 1.0 / f;
    const double s_pre = (invf > 1.0) ? invf : 1.0;
    const double fsup = rcre_gm_support(filter);
    double support = s_pre * fsup;
    double scale = 0.0;

    if (support <= 0.5) {
        support = 0.5 + RCRE_GM_EPS;
        scale = 1.0;
    } else {
        scale = 1.0 / s_pre;
    }

    const size_t max_nt = (size_t)ceil(2.0 * support + 2.0);
    if (max_nt < 1) return RCRE_ERR_INVALID_ARG;

    size_t taps_sz = (size_t)dst_len * sizeof(RcreGmTap);
    size_t wgt_sz  = (size_t)dst_len * max_nt * sizeof(double);

    RcreGmTap* taps = arena ? (RcreGmTap*)rcre_arena_alloc(arena, taps_sz, alignof(RcreGmTap)) : (RcreGmTap*)malloc(taps_sz);
    double* wpool   = arena ? (double*)rcre_arena_alloc(arena, wgt_sz, alignof(double))    : (double*)malloc(wgt_sz);

    if (!taps || !wpool) {
        if (!arena) {
            free(taps);
            free(wpool);
        }
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    for (unsigned x = 0; x < dst_len; x++) {
        const double center = ((double)x + 0.5) * invf;
        long start = (long)(center - support + 0.5);
        long stop  = (long)(center + support + 0.5);
        if (start < 0) start = 0;
        if (stop > (long)src_len) stop = (long)src_len;

        const long nt = stop - start;
        taps[x].start = (int)start;
        taps[x].nt    = (int)nt;
        taps[x].wgt   = wpool + (size_t)x * max_nt;

        if (nt <= 0) continue;

        double* wgt = wpool + (size_t)x * max_nt;
        double density = 0.0;
        for (long i = 0; i < nt; i++) {
            const double arg = scale * ((double)(start + i) + 0.5 - center);
            wgt[i] = rcre_gm_filter(filter, arg, fsup);
            density += wgt[i];
        }
        if (density != 0.0 && density != 1.0) {
            const double inv = 1.0 / density;
            for (long i = 0; i < nt; i++)
                wgt[i] *= inv;
        }
    }

    *out_taps = taps;
    *out_weights = wpool;
    return RCRE_OK;
}

/* ---------------------------------------------------------------------
 * Horizontal & Vertical Resampling Passes (Float)
 * ------------------------------------------------------------------ */
static void rcre_gm_pass_h(const float* src, float* dst, unsigned w_in, unsigned w_out,
                           unsigned h, const RcreGmTap* taps) {
    for (unsigned y = 0; y < h; y++) {
        const float* srow = src + (size_t)y * w_in;
        float* drow = dst + (size_t)y * w_out;

        for (unsigned x = 0; x < w_out; x++) {
            const RcreGmTap* t = &taps[x];
            const int nt = t->nt;
            if (nt <= 0) {
                drow[x] = 0.0f;
                continue;
            }
            const int start = t->start;
            const double* wgt = t->wgt;

            double acc = 0.0;
            for (int i = 0; i < nt; i++) {
                acc += wgt[i] * (double)srow[start + i];
            }
            drow[x] = (float)acc;
        }
    }
}

static void rcre_gm_pass_v(const float* src, float* dst, unsigned w,
                           unsigned h_out, const RcreGmTap* taps) {
    for (unsigned y = 0; y < h_out; y++) {
        const RcreGmTap* t = &taps[y];
        const int nt = t->nt;
        float* drow = dst + (size_t)y * w;

        if (nt <= 0) {
            memset(drow, 0, (size_t)w * sizeof(float));
            continue;
        }
        const int start = t->start;
        const double* wgt = t->wgt;

        for (unsigned x = 0; x < w; x++) {
            double acc = 0.0;
            for (int i = 0; i < nt; i++) {
                acc += wgt[i] * (double)src[(size_t)(start + i) * w + x];
            }
            drow[x] = (float)acc;
        }
    }
}

/* ---------------------------------------------------------------------
 * Public API: rcre_cv_resize_gm
 * ------------------------------------------------------------------ */
RCRE_NODISCARD int rcre_cv_resize_gm(const RCRE_Tensor* src, RCRE_Tensor* dst,
                                     RCRE_CV_Interp interp, RCRE_Arena* arena) {
    if (!src || !src->data || !dst || !dst->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (src->dtype != RCRE_DTYPE_F32 || dst->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    size_t in_h, in_w, in_c;
    size_t out_h, out_w, out_c;
    bool is_chw = (src->layout == RCRE_LAYOUT_CHW);

    if (src->ndim == 2) {
        in_h = src->shape[0];
        in_w = src->shape[1];
        in_c = 1;
        is_chw = true;
    } else if (src->ndim == 3) {
        if (is_chw) {
            in_c = src->shape[0];
            in_h = src->shape[1];
            in_w = src->shape[2];
        } else {
            in_h = src->shape[0];
            in_w = src->shape[1];
            in_c = src->shape[2];
        }
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (dst->ndim == 2) {
        out_h = dst->shape[0];
        out_w = dst->shape[1];
        out_c = 1;
    } else if (dst->ndim == 3) {
        if (dst->layout == RCRE_LAYOUT_CHW) {
            out_c = dst->shape[0];
            out_h = dst->shape[1];
            out_w = dst->shape[2];
        } else {
            out_h = dst->shape[0];
            out_w = dst->shape[1];
            out_c = dst->shape[2];
        }
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (in_c != out_c || in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    int filter = (int)interp;
    /* Default to GM Triangle if an unsupported / invalid enum is passed */
    if (filter < (int)RCRE_CV_INTERP_GM_POINT || filter > (int)RCRE_CV_INTERP_GM_SINC) {
        filter = (int)RCRE_CV_INTERP_GM_TRIANGLE;
    }

    /* Precompute horizontal taps (in_w -> out_w) */
    RcreGmTap* h_taps = NULL;
    double* h_wgts = NULL;
    int err = rcre_gm_build_taps(&h_taps, &h_wgts, (unsigned)out_w, (unsigned)in_w, filter, arena);
    if (err != RCRE_OK) return err;

    /* Precompute vertical taps (in_h -> out_h) */
    RcreGmTap* v_taps = NULL;
    double* v_wgts = NULL;
    err = rcre_gm_build_taps(&v_taps, &v_wgts, (unsigned)out_h, (unsigned)in_h, filter, arena);
    if (err != RCRE_OK) {
        if (!arena) { free(h_taps); free(h_wgts); }
        return err;
    }

    /* Determine pass order: H first if out_w * (in_h + out_h) > out_h * (in_w + out_w) */
    bool h_first = (out_w * (in_h + out_h) > out_h * (in_w + out_w));
    size_t tmp_w = h_first ? out_w : in_w;
    size_t tmp_h = h_first ? in_h : out_h;
    size_t tmp_sz = tmp_w * tmp_h * sizeof(float);

    float* tmp_buf = arena ? (float*)rcre_arena_alloc(arena, tmp_sz, alignof(float)) : (float*)malloc(tmp_sz);
    if (!tmp_buf) {
        if (!arena) {
            free(h_taps); free(h_wgts);
            free(v_taps); free(v_wgts);
        }
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Planar buffers for HWC conversion if needed */
    float* in_planar = NULL;
    float* out_planar = NULL;
    if (!is_chw && in_c > 1) {
        size_t in_sz = in_w * in_h * sizeof(float);
        size_t out_sz = out_w * out_h * sizeof(float);
        in_planar = arena ? (float*)rcre_arena_alloc(arena, in_sz, alignof(float)) : (float*)malloc(in_sz);
        out_planar = arena ? (float*)rcre_arena_alloc(arena, out_sz, alignof(float)) : (float*)malloc(out_sz);
    }

    const float* src_data = (const float*)src->data;
    float* dst_data = (float*)dst->data;
    const size_t in_plane = in_w * in_h;
    const size_t out_plane = out_w * out_h;

    for (size_t c = 0; c < in_c; c++) {
        const float* cur_src = NULL;
        float* cur_dst = NULL;

        if (is_chw || in_c == 1) {
            cur_src = src_data + c * in_plane;
            cur_dst = dst_data + c * out_plane;
        } else {
            /* Deinterleave channel c */
            for (size_t i = 0; i < in_plane; i++) {
                in_planar[i] = src_data[i * in_c + c];
            }
            cur_src = in_planar;
            cur_dst = out_planar;
        }

        if (h_first) {
            /* Pass 1: Horizontal (in_w -> out_w, height in_h) */
            rcre_gm_pass_h(cur_src, tmp_buf, (unsigned)in_w, (unsigned)out_w, (unsigned)in_h, h_taps);
            /* Pass 2: Vertical (in_h -> out_h, width out_w) */
            rcre_gm_pass_v(tmp_buf, cur_dst, (unsigned)out_w, (unsigned)out_h, v_taps);
        } else {
            /* Pass 1: Vertical (in_h -> out_h, width in_w) */
            rcre_gm_pass_v(cur_src, tmp_buf, (unsigned)in_w, (unsigned)out_h, v_taps);
            /* Pass 2: Horizontal (in_w -> out_w, height out_h) */
            rcre_gm_pass_h(tmp_buf, cur_dst, (unsigned)in_w, (unsigned)out_w, (unsigned)out_h, h_taps);
        }

        if (!is_chw && in_c > 1) {
            /* Interleave channel c back into dst */
            for (size_t i = 0; i < out_plane; i++) {
                dst_data[i * out_c + c] = out_planar[i];
            }
        }
    }

    if (!arena) {
        free(tmp_buf);
        free(h_taps);
        free(h_wgts);
        free(v_taps);
        free(v_wgts);
        if (in_planar) free(in_planar);
        if (out_planar) free(out_planar);
    }

    return RCRE_OK;
}
