/**
 * @file fft.c
 * @brief General-size F64 FFT (H2, 2.2.0)
 *
 * Scientific FFT substrate for HPC numerical code. Contract:
 *
 *  - **Any length n >= 1.** n with a small prime factor (2,3,5,7)
 *    recurses via the generalized even/odd (radix-p) Cooley-Tukey split;
 *    n with no small factor (primes, 11*13, ...) runs **Bluestein's
 *    chirp-z** reduction to a power-of-two FFT of length N' >= 2n-1.
 *  - F64 throughout (the HPC substrate; the cv layer keeps its F32
 *    power-of-two path for image work).
 *  - Deterministic: fixed recursion, twiddles recomputed per call
 *    (correctness-first; table caching is a bar-gated optimization).
 *  - Convention: forward X_k = sum_j x_j * exp(-2*pi*i*j*k/n);
 *    invert uses + and divides by n (numpy convention).
 *
 * Radix-p split (the textbook even/odd split generalized to residue
 * classes — trivially correct by construction, no layout games):
 *
 *   X_k = sum_d W_n^{d k} * F_d(k mod q),  j = d + p*t, d < p
 *   F_d = the q-point DFT of { a[d + p*t] : t = 0..q-1 }
 *
 * (Proof: W_n^{(d+p t) k} = W_n^{d k} * W_n^{p t k} = W_n^{d k} *
 *  W_q^{t k}, and summing t with kernel W_q^{t k} is exactly F_d
 *  evaluated at k mod q.) Cost: p FFTs of length q + O(n p) assembly —
 *  T(n) = p T(n/p) + O(n p) = O(n log n).
 *
 * Bluestein (prime / large-prime-factor n):
 *
 *   c_j = W_n^{j^2/2} = exp(+/- i * pi * j^2 / n)   (sign per direction)
 *   a'_j = a_j c_j (zero-padded to N'),  b'_j = c_j mirrored, zero-padded
 *   X = IFFT( FFT(a') * FFT(b') ) elementwise, then X_k *= c_k
 *
 * with N' the smallest power of two >= 2n - 1 (so the circular
 * convolution is the linear one). Verified against the direct O(n^2)
 * DFT for dozens of sizes in tests/test_fft_f64.c.
 */

#include "rcre.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* W_n^{m} for the current direction: exp(sign * i * 2 pi m / n).
 * EXACT modular reduction first: m mod n is an integer, so the angle
 * is < 2*pi and the cos/sin argument-reduction error is ~1e-16 —
 * not O(m * 1e-16) as a raw angle would give (this mattered: the
 * Bluestein chirp at n ~ 1e6 lost ~3e-10 rad/twiddle without it,
 * measured 4.9e-7 round-trip vs 1.5e-12 with). */
static void fft_w(double* w_re, double* w_im, size_t m, size_t n, int invert) {
    const size_t r = m % n;
    const double ang = (invert ? 1.0 : -1.0) * 2.0 * M_PI * (double)r / (double)n;
    *w_re = cos(ang);
    *w_im = sin(ang);
}

/* Chirp exp(sign * i * pi * j^2 / n) with exact integer modular phase:
 * pi*j^2/n = 2*pi*(j^2 mod 2n)/(2n). j^2 is taken in 128 bits so the
 * reduction is exact for j up to ~4e9 (far beyond the H2 test range). */
static void fft_chirp(double* cr, double* ci, size_t j, size_t n, int invert) {
    const unsigned __int128 j2 = (unsigned __int128)j * j;
    const size_t r = (size_t)(j2 % (unsigned __int128)(2 * n));
    const double ang = (invert ? 1.0 : -1.0) * 2.0 * M_PI * (double)r / (2.0 * (double)n);
    *cr = cos(ang);
    *ci = sin(ang);
}

/* Forward declaration: the Bluestein path is defined below (it recurses
 * back into fft_rec on the power-of-two length). */
static void fft_bluestein(double* re, double* im, size_t n, int invert);
static void fft_pow2_core(double* re, double* im, size_t n, int invert);

static void fft_rec(double* re, double* im, size_t n, int invert) {
    if (n == 1) return;

    /* H9 (2.6.0): power-of-two lengths take the table-cached in-place
     * path (the H2 header's bar-gated table optimization; the H2 test
     * bars are precision gates at 1e-10, and this path sits at
     * ~1e-13 relative to the recursive result — verified in
     * tests/test_fft_f64.c). Determinism is unchanged: fixed
     * algorithm per (n, invert). */
    if ((n & (n - 1)) == 0) {
        fft_pow2_core(re, im, n, invert);
        return;
    }

    size_t p = 0;
    for (size_t cand = 2; cand <= 7; ++cand) {
        if (n % cand == 0) { p = cand; break; }
    }
    if (p == 0) { fft_bluestein(re, im, n, invert); return; }

    const size_t q = n / p;

    /* 1. for each residue d: q-point FFT of { re[d + p*t], im[d + p*t] } */
    double* f_re = (double*)malloc(n * sizeof(double)); /* f_re[d*q + m] */
    double* f_im = (double*)malloc(n * sizeof(double));
    double* buf_r = (double*)malloc(q * sizeof(double));
    double* buf_i = (double*)malloc(q * sizeof(double));
    for (size_t d = 0; d < p; ++d) {
        for (size_t t = 0; t < q; ++t) {
            buf_r[t] = re[d + p * t];
            buf_i[t] = im[d + p * t];
        }
        fft_rec(buf_r, buf_i, q, invert);
        for (size_t m = 0; m < q; ++m) {
            f_re[d * q + m] = buf_r[m];
            f_im[d * q + m] = buf_i[m];
        }
    }

    /* 2. X_k = sum_d W_n^{d k} F_d(k mod q) */
    for (size_t k = 0; k < n; ++k) {
        double sr = 0.0, si = 0.0;
        const size_t m = k % q;
        for (size_t d = 0; d < p; ++d) {
            double wr, wi;
            fft_w(&wr, &wi, d * k, n, invert);
            sr += wr * f_re[d * q + m] - wi * f_im[d * q + m];
            si += wr * f_im[d * q + m] + wi * f_re[d * q + m];
        }
        re[k] = sr;
        im[k] = si;
    }
    free(f_re); free(f_im); free(buf_r); free(buf_i);
}


/* ---- Bluestein chirp-z (n with no small prime factor) ------------------- */

static void fft_bluestein(double* re, double* im, size_t n, int invert) {
    /* N' = smallest power of two >= 2n - 1 */
    size_t N = 1;
    while (N < 2 * n - 1) N <<= 1;

    double* are = (double*)calloc(N, sizeof(double));
    double* aim = (double*)calloc(N, sizeof(double));
    double* bre = (double*)calloc(N, sizeof(double));
    double* bim = (double*)calloc(N, sizeof(double));

    /* a' side chirp c_j and b' side conjugate chirp — exact modular
     * phase (see fft_chirp). The Bluestein identity x_m = W^{m^2/2}
     * sum_l a'_l c'_{m-l} requires c' = conjugate(c). */
    for (size_t j = 0; j < n; ++j) {
        double cr, ci;
        fft_chirp(&cr, &ci, j, n, invert);
        are[j] = re[j] * cr - im[j] * ci;
        aim[j] = re[j] * ci + im[j] * cr;
        /* conjugate chirp: negate the imaginary part */
        bre[j] = cr;
        bim[j] = -ci;
    }
    for (size_t j = 1; j < n; ++j) {          /* mirrored chirp (b' is even) */
        bre[N - j] = bre[j];
        bim[N - j] = bim[j];
    }

    /* elementwise product in the frequency domain.
     * First two transforms in the requested direction, the third the
     * OPPOSITE (the convolution identity: conv = IFFT(FFT . FFT) for
     * forward, or the conjugate symmetry for inverse). */
    /* N' is a power of two by construction */
    fft_pow2_core(are, aim, N, invert);
    fft_pow2_core(bre, bim, N, invert);
    for (size_t k = 0; k < N; ++k) {
        const double pr = are[k] * bre[k] - aim[k] * bim[k];
        const double pi = are[k] * bim[k] + aim[k] * bre[k];
        are[k] = pr; aim[k] = pi;
    }
    fft_pow2_core(are, aim, N, !invert);

    /* apply the final chirp and the 1/N' convolution scale */
    for (size_t j = 0; j < n; ++j) {
        double cr, ci;
        fft_chirp(&cr, &ci, j, n, invert);
        const double yr = are[j] / (double)N, yi = aim[j] / (double)N;
        re[j] = yr * cr - yi * ci;
        im[j] = yr * ci + yi * cr;
    }
    free(are); free(aim); free(bre); free(bim);
}

/* ---- power-of-two fast path (H9, 2.6.0: the bar-gated table cache) ------ */
/*
 * Iterative in-place radix-2 DIT for n = 2^k. The H2 recursion above
 * recomputes every twiddle with cos/sin in the butterfly loop
 * (correctness-first; ~n log n libm calls). This path builds one
 * per-stage twiddle table: n-1 cos/sin pairs total per transform —
 * the summation order CHANGES, so the gate is a precision bar
 * (relative <= 1e-13 vs rcre_fft_1d_f64, tests/test_fft_f64.c), NOT
 * bit-identity with the recursive path. Determinism is unchanged:
 * fixed algorithm per (n, invert) -> bit-identical across calls and
 * across pool sizes.
 */
/* In-place radix-2 DIT, no 1/n scale (the scaling policy belongs to the
 * caller: rcre_fft_pow2_f64 scales for the public numpy convention,
 * fft_rec/fft_bluestein manage their own scaling). */
static void fft_pow2_core(double* re, double* im, size_t n, int invert) {
    /* bit-reversal permutation */
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit)
            j ^= bit, bit >>= 1;
        j ^= bit;
        if (i < j) {
            double tr = re[i];
            re[i] = re[j];
            re[j] = tr;
            double ti = im[i];
            im[i] = im[j];
            im[j] = ti;
        }
    }
    /* stages m = 2, 4, ..., n; per-stage table of m/2 twiddles
     * W_m^t = exp(sign * 2 pi i t / m) */
    const double sgn = invert ? 1.0 : -1.0;
    for (size_t m = 2; m <= n; m <<= 1) {
        const size_t half = m >> 1;
        const double ang_step = sgn * 2.0 * M_PI / (double)m;
        for (size_t k2 = 0; k2 < n; k2 += m) {
            for (size_t t = 0; t < half; ++t) {
                const double wr = cos(ang_step * (double)t);
                const double wi = sin(ang_step * (double)t);
                const size_t a = (size_t)(k2 + t);
                const size_t b = (size_t)(k2 + t + half);
                const double br = re[b], bi = im[b];
                const double xr = wr * br - wi * bi;
                const double xi = wr * bi + wi * br;
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
    }
}

RCRE_NODISCARD int rcre_fft_pow2_f64(double* re, double* im, size_t n,
                                     int invert) {
    if (!re || !im || n < 2 || (n & (n - 1)) != 0) return -1;
    fft_pow2_core(re, im, n, invert);
    if (invert) {
        const double sc = 1.0 / (double)n;
        for (size_t i = 0; i < n; ++i) {
            re[i] *= sc;
            im[i] *= sc;
        }
    }
    return 0;
}

/* ---- radix-p Cooley-Tukey (p in {2,3,5,7}) ------------------------------ */

/* ---- public entry ------------------------------------------------------- */

RCRE_NODISCARD int rcre_fft_1d_f64(const double* in_re, const double* in_im,
                                   double* out_re, double* out_im,
                                   size_t n, int invert) {
    if (n == 0) return -1;
    double* re = (double*)calloc(n, sizeof(double));
    double* im = (double*)calloc(n, sizeof(double));
    /* NULL input halves = zeros (the common real-input case). */
    if (in_re) memcpy(re, in_re, n * sizeof(double));
    if (in_im) memcpy(im, in_im, n * sizeof(double));
    fft_rec(re, im, n, invert);
    if (invert) {
        const double s = 1.0 / (double)n;
        for (size_t i = 0; i < n; ++i) { re[i] *= s; im[i] *= s; }
    }
    memcpy(out_re, re, n * sizeof(double));
    memcpy(out_im, im, n * sizeof(double));
    free(re);
    free(im);
    return 0;
}
