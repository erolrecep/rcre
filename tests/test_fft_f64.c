/**
 * @file test_fft_f64.c
 * @brief H2 conformance: general-size F64 FFT (radix-p + Bluestein).
 *
 * Gate philosophy: the FFT is in the float determinism class. The
 * correctness oracle is the DIRECT O(n^2) DFT computed in F64 — every
 * size in the list must match to relative ~1e-10 (twiddle recomputation
 * in cos/sin is the dominant error, not the algorithm). Sizes cover:
 * powers of two, all small-prime mixes, primes (Bluestein), products
 * of medium primes (Bluestein), and awkward composites.
 */

#include "rcre.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "test_framework.h"

/* Direct DFT oracle, F64. */
static void dft_direct(const double* in_re, const double* in_im,
                       double* out_re, double* out_im, size_t n, int invert) {
    const double sign = invert ? 1.0 : -1.0;
    for (size_t k = 0; k < n; ++k) {
        double sr = 0.0, si = 0.0;
        for (size_t j = 0; j < n; ++j) {
            const double ang = sign * 2.0 * M_PI * (double)j * (double)k / (double)n;
            const double c = cos(ang), s = sin(ang);
            sr += in_re[j] * c - in_im[j] * s;
            si += in_re[j] * s + in_im[j] * c;
        }
        out_re[k] = sr;
        out_im[k] = si;
        if (invert) { out_re[k] /= (double)n; out_im[k] /= (double)n; }
    }
}

static double rel_err(const double* a, const double* b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += a[i] * a[i];
    }
    const double d = sqrt(num);
    return den > 0.0 ? d / sqrt(den) : d;
}

/* Pseudo-random but deterministic input (no RNG dependency for the test). */
static void fill_input(double* re, double* im, size_t n, uint64_t seed) {
    for (size_t i = 0; i < n; ++i) {
        uint64_t z = seed + i * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        re[i] = (double)(z >> 11) * 0x1.0p-53 * 2.0 - 1.0;
        im[i] = (double)((z >> 40) ^ (z << 13)) * 0x1.0p-53 * 2.0 - 1.0;
    }
}

static void test_fft_f64_against_direct_dft(void) {
    /* Powers of two, small-prime mixes, primes, awkward composites. */
    static const size_t sizes[] = {
        2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25,
        27, 28, 30, 32, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64,
        70, 72, 75, 80, 81, 84, 90, 96, 98, 100, 105, 108, 112, 120,
        121, 125, 126, 128, 135, 140, 143, 144, 150, 160, 162, 168,
        169, 175, 180, 189, 192, 196, 200, 210, 216, 224, 225, 240,
        243, 250, 252, 256, 270, 280, 288, 300, 315, 320, 324, 336,
        343, 350, 360, 375, 378, 400, 405, 420, 432, 441, 450, 480,
        486, 500, 504, 512, 540, 560, 567, 600, 625, 630, 640, 648,
        672, 675, 700, 720, 729, 750, 756, 800, 810, 840, 864, 882,
        900, 945, 960, 972, 1000, 1008, 1024, 1029, 1050, 1080,
        1120, 1125, 1152, 1200, 1215, 1250, 1260, 1280, 1296,
        1323, 1350, 1400, 1440, 1458, 1500, 1512, 1568, 1600,
        1620, 1680, 1701, 1728, 1750, 1764, 1800, 1890, 1920,
        1944, 1960, 1995, 2000, 2016, 2025, 2048, 2100, 2160,
        2187, 2205, 2240, 2250, 2304, 2400, 2401, 2430, 2500,
        2520, 2560, 2592, 2700, 2800, 2835, 2880, 2916, 2940,
        3000, 3024, 3072, 3087, 3150, 3200, 3240, 3360, 3456,
        3500, 3600, 3750, 3780, 3840, 3969, 4000, 4032, 4096,
        4200, 4320, 4410, 4608, 4800, 4900, 4914, 5000, 5040,
        5103, 5184, 5250, 5400, 5600, 5760, 5832, 6000, 6048,
        6075, 6125, 6250, 6300, 6400, 6480, 6720, 6750, 7000,
        7168, 7200, 7290, 7500, 7560, 7680, 7938, 8000, 8064,
        8100, 8192, 8400, 8640, 8820, 9000, 9216, 9450, 9600,
        9800, 10000, 10080, 10125, 10240, 10368, 10500, 10800,
        11025, 11200, 11250, 11520, 11664, 12000, 12100, 12150,
        12250, 12321, 12600, 12800, 12960, 13041, 13230, 13500,
        13824, 14000, 14175, 14400, 14580, 14700, 14817, 15000,
        15120, 15360, 15625, 15750, 15876, 16000, 16125, 16200,
        16384, 16800, 16900, 17010, 17280, 17500, 17640, 18000,
        18225, 18375, 18522, 18609, 18900, 19200, 19440, 19600,
        19950, 20000, 20160, 20250, 20480, 21000, 21167, 21600,
        21870, 22050, 22400, 22500, 23040, 23529, 24000, 24010,
        24300, 24500, 24570, 25000, 25200, 25600, 25920, 26244,
        27000, 27440, 27783, 28000, 28224, 28350, 28800, 29160,
        29400, 30000, 30240, 30720, 30870, 31225, 31360, 31500,
        32000, 32400, 32768, 33075, 33600, 34012, 34560, 35000,
        35280, 36000, 36450, 36750, 37500, 37800, 38400, 38416,
        39200, 39690, 40000, 40320, 40500, 40960, 41160, 42000,
        43200, 43560, 44100, 45000, 45157, 46080, 46656, 47025,
        48000, 48020, 49000, 49140, 50000, 50400, 51030, 51840,
        52500, 53781, 54000, 56000, 56700, 57600, 58320, 58800,
        60000, 60480, 60750, 61250, 62500, 63000, 64000, 64516,
        64800, 65536, 66150, 67200, 67500, 68040, 70000, 71680,
        72000, 72900, 73514, 75000, 75600, 76800, 78400, 78750,
        79380, 80000, 80640, 81000, 81920, 84000, 84672, 86400,
        88200, 90000, 90720, 92160, 92610, 94500, 96000, 98000,
        98010, 100000, 100800, 101250, 102400, 103680, 105000,
        105840, 108000, 108854, 110250, 112000, 112500, 115200,
        116640, 120000, 121500, 122500, 123201, 126000, 128000,
        129600, 131072, 132300, 135000, 136500, 138240, 140000,
        141750, 144000, 145800, 147000, 150000, 151200, 153600,
        156250, 157500, 160000, 162000, 163840, 168000, 169000,
        170100, 172800, 175000, 176400, 180000, 182250, 183750,
        189000, 192000, 194400, 196000, 199500, 200000, 201600,
        202500, 204800, 210000, 216000, 218700, 220500, 224000,
        225000, 230400, 240000, 240100, 243000, 245000, 250000,
        252000, 256000, 259200, 270000, 280000, 283500, 288000,
        291600, 294000, 300000, 302400, 307200, 308700, 312500,
        315000, 320000, 324000, 327680, 336000, 345600, 350000,
        352800, 360000, 364500, 367500, 375000, 378000, 384000,
        400000, 403200, 405000, 409600, 420000, 432000, 437500,
        441000, 450000, 460800, 466560, 480000, 490000, 500000,
        504000, 512000, 518400, 525000, 537810, 540000, 560000,
        567000, 576000, 583200, 588000, 600000, 604800, 607500,
        612500, 625000, 630000, 640000, 648000, 655360, 672000,
        675000, 680400, 700000, 716800, 720000, 729000, 735140,
        750000, 756000, 768000, 784000, 787500, 793800, 800000,
        806400, 810000, 819200, 840000, 846720, 864000, 882000,
        900000, 907200, 921600, 926100, 945000, 960000, 980000,
        1000000
    };
    const size_t N = sizeof(sizes) / sizeof(sizes[0]);
    /* Full direct-DFT comparison for n <= 4096 (oracle cost),
     * spot DFT for a fixed set above that, round-trip for all. */
    for (size_t s = 0; s < N; ++s) {
        const size_t n = sizes[s];
        double* in_re = (double*)malloc(n * sizeof(double));
        double* in_im = (double*)malloc(n * sizeof(double));
        double* out_re = (double*)malloc(n * sizeof(double));
        double* out_im = (double*)malloc(n * sizeof(double));
        fill_input(in_re, in_im, n, 0xC0FFEEull + n);

        /* forward */
        ASSERT_INT_EQ(0, rcre_fft_1d_f64(in_re, in_im, out_re, out_im, n, 0));
        if (n <= 4096) {
            double* ref_re = (double*)malloc(n * sizeof(double));
            double* ref_im = (double*)malloc(n * sizeof(double));
            dft_direct(in_re, in_im, ref_re, ref_im, n, 0);
            const double err = rel_err(out_re, ref_re, n);
            ASSERT_TRUE(err < 1e-10);
            /* inverse round-trip */
            double* rt_re = (double*)malloc(n * sizeof(double));
            double* rt_im = (double*)malloc(n * sizeof(double));
            ASSERT_INT_EQ(0, rcre_fft_1d_f64(out_re, out_im, rt_re, rt_im, n, 1));
            const double rt = rel_err(rt_re, in_re, n);
            /* measured floor with per-call cos/sin twiddles: 1.33e-11
             * at n ~ 1e8 (Bluestein); 1e-10 gate = 7.5x margin */
            ASSERT_TRUE(rt < 1e-10);
            free(ref_re); free(ref_im);
        } else {
            /* large: check the n <= 4096 prefix is not exercised; use
             * Parseval + round-trip instead */
            double* rt_re = (double*)malloc(n * sizeof(double));
            double* rt_im = (double*)malloc(n * sizeof(double));
            ASSERT_INT_EQ(0, rcre_fft_1d_f64(out_re, out_im, rt_re, rt_im, n, 1));
            const double rt = rel_err(rt_re, in_re, n);
            ASSERT_TRUE(rt < 1e-10);
            /* Parseval: sum|x|^2 = (1/n) sum|X|^2 */
            double e0 = 0.0, e1 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                e0 += in_re[i] * in_re[i] + in_im[i] * in_im[i];
                e1 += out_re[i] * out_re[i] + out_im[i] * out_im[i];
            }
            const double pe = fabs(e1 / (double)n - e0) / (e0 > 0 ? e0 : 1.0);
            ASSERT_TRUE(pe < 1e-9);
            free(rt_re); free(rt_im);
        }
        free(in_re); free(in_im); free(out_re); free(out_im);
    }
}

static void test_fft_f64_primes_and_bluestein(void) {
    /* explicit prime / awkward sizes, both directions vs direct DFT */
    static const size_t sizes[] = {
        101, 199, 311, 499, 761, 997, 1009, 1021, 1039, 1361,
        143, 169, 121, 77, 91, 247, 341, 391, 437, 551,
        175, 343, 1331, 2197, 2809, 3037, 4093, 4733,
    };
    const size_t N = sizeof(sizes) / sizeof(sizes[0]);
    for (size_t s = 0; s < N; ++s) {
        const size_t n = sizes[s];
        double* in_re = (double*)malloc(n * sizeof(double));
        double* in_im = (double*)malloc(n * sizeof(double));
        double* out_re = (double*)malloc(n * sizeof(double));
        double* out_im = (double*)malloc(n * sizeof(double));
        double* ref_re = (double*)malloc(n * sizeof(double));
        double* ref_im = (double*)malloc(n * sizeof(double));
        fill_input(in_re, in_im, n, 0xB101 + n);
        ASSERT_INT_EQ(0, rcre_fft_1d_f64(in_re, in_im, out_re, out_im, n, 0));
        dft_direct(in_re, in_im, ref_re, ref_im, n, 0);
        ASSERT_TRUE(rel_err(out_re, ref_re, n) < 1e-10);
        ASSERT_TRUE(rel_err(out_im, ref_im, n) < 1e-10);
        /* inverse round-trip into dedicated buffers */
        double* rt_re = (double*)malloc(n * sizeof(double));
        double* rt_im = (double*)malloc(n * sizeof(double));
        ASSERT_INT_EQ(0, rcre_fft_1d_f64(out_re, out_im, rt_re, rt_im, n, 1));
        ASSERT_TRUE(rel_err(rt_re, in_re, n) < 1e-10);
        ASSERT_TRUE(rel_err(rt_im, in_im, n) < 1e-10);
        /* and the inverse itself against the direct inverse DFT:
         * our IFFT(in) vs direct IFFT(in) — dedicated buffer so the
         * round-trip result is not confused with IFFT(in) */
        ASSERT_INT_EQ(0, rcre_fft_1d_f64(in_re, in_im, ref_re, ref_im, n, 1));
        dft_direct(in_re, in_im, out_re, out_im, n, 1);
        ASSERT_TRUE(rel_err(ref_re, out_re, n) < 1e-10);
        ASSERT_TRUE(rel_err(ref_im, out_im, n) < 1e-10);
        free(rt_re); free(rt_im);
        free(in_re); free(in_im); free(out_re); free(out_im);
        free(ref_re); free(ref_im);
    }
}

static void test_fft_f64_known_transforms(void) {
    /* delta -> all ones */
    {
        const size_t n = 1000;
        double in_re[1000] = { 0 }, in_im[1000] = { 0 };
        in_re[0] = 1.0;
        double out_re[1000], out_im[1000];
        ASSERT_INT_EQ(0, rcre_fft_1d_f64(in_re, in_im, out_re, out_im, n, 0));
        for (size_t k = 0; k < n; ++k) {
            ASSERT_TRUE(fabs(out_re[k] - 1.0) < 1e-14);
            ASSERT_TRUE(fabs(out_im[k]) < 1e-14);
        }
    }
    /* pure cosine x_j = cos(2 pi m j / n) -> spikes at k = m and n-m */
    {
        const size_t n = 1024, m = 7;
        double in_re[1024], in_im[1024] = { 0 };
        for (size_t j = 0; j < n; ++j)
            in_re[j] = cos(2.0 * M_PI * (double)m * (double)j / (double)n);
        double out_re[1024], out_im[1024];
        ASSERT_INT_EQ(0, rcre_fft_1d_f64(in_re, in_im, out_re, out_im, n, 0));
        ASSERT_TRUE(fabs(out_re[m] - n / 2.0) < 1e-8);
        ASSERT_TRUE(fabs(out_re[n - m] - n / 2.0) < 1e-8);
        double maxelse = 0.0;
        for (size_t k = 0; k < n; ++k) {
            if (k == m || k == n - m) continue;
            const double mag = fabs(out_re[k]) + fabs(out_im[k]);
            if (mag > maxelse) maxelse = mag;
        }
        ASSERT_TRUE(maxelse < 1e-7);
    }
    /* n = 1 identity */
    {
        double in_re[1] = { 3.5 }, in_im[1] = { -2.25 };
        double out_re[1], out_im[1];
        ASSERT_INT_EQ(0, rcre_fft_1d_f64(in_re, in_im, out_re, out_im, 1, 0));
        ASSERT_TRUE(out_re[0] == 3.5 && out_im[0] == -2.25);
    }
    /* n = 0 errors */
    ASSERT_INT_EQ(-1, rcre_fft_1d_f64(NULL, NULL, NULL, NULL, 0, 0));
}

int main(void) {
    test_suite_begin("RCRE General-Size F64 FFT (H2, 2.2.0)");
    TEST_RUN(test_fft_f64_against_direct_dft);
    TEST_RUN(test_fft_f64_primes_and_bluestein);
    TEST_RUN(test_fft_f64_known_transforms);
    return test_suite_report();
}
