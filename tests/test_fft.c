/**
 * @file test_fft.c
 * @brief Unit tests for RCRE Fast Fourier Transform & Spectral Image Processing
 * 
 * Satisfies FFT-REQ-01 to FFT-REQ-04.
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-5f

static void test_fft_1d(void) {
    printf("  -> Testing 1D FFT and Inverse IFFT round-trip (N=8)...\n");
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    const size_t N = 8;
    RCRE_Complex sig[8] = {
        {1.0f, 0.0f},
        {2.0f, 0.0f},
        {3.0f, 0.0f},
        {4.0f, 0.0f},
        {5.0f, 0.0f},
        {6.0f, 0.0f},
        {7.0f, 0.0f},
        {8.0f, 0.0f}
    };
    RCRE_Complex orig[8];
    memcpy(orig, sig, sizeof(sig));

    /* Forward FFT */
    int status = rcre_fft_1d(sig, N, arena);
    assert(status == RCRE_OK);

    /* DC component for 1+2+...+8 = 36 */
    assert(fabsf(sig[0].re - 36.0f) < 1e-4f);
    assert(fabsf(sig[0].im - 0.0f) < 1e-4f);

    /* Inverse FFT */
    status = rcre_ifft_1d(sig, N, arena);
    assert(status == RCRE_OK);

    /* Assert exact reconstruction */
    for (size_t i = 0; i < N; ++i) {
        assert(fabsf(sig[i].re - orig[i].re) < EPSILON);
        assert(fabsf(sig[i].im - 0.0f) < EPSILON);
    }

    rcre_arena_destroy(arena);
    printf("     ✓ 1D FFT/IFFT exact inversion verified (reconstruction error < 1e-5)\n");
}

static void test_fft_2d(void) {
    printf("  -> Testing 2D FFT and Inverse 2D IFFT on image matrix (16x16)...\n");
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    assert(arena != NULL);

    const size_t H = 16;
    const size_t W = 16;
    size_t shape[2] = {H, W};

    RCRE_Tensor img = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor out_re = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor out_im = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor recon = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float* img_data = (float*)img.data;
    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            img_data[h * W + w] = sinf((float)h * 0.5f) * cosf((float)w * 0.3f);
        }
    }

    /* 1. Forward 2D FFT */
    int status = rcre_fft_2d(&img, &out_re, &out_im, arena);
    assert(status == RCRE_OK);

    /* 2. Inverse 2D IFFT */
    status = rcre_ifft_2d(&out_re, &out_im, &recon, arena);
    assert(status == RCRE_OK);

    /* 3. Check reconstruction fidelity */
    float* recon_data = (float*)recon.data;
    float max_diff = 0.0f;
    for (size_t i = 0; i < H * W; ++i) {
        float diff = fabsf(img_data[i] - recon_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    assert(max_diff < EPSILON);

    rcre_arena_destroy(arena);
    printf("     ✓ 2D FFT/IFFT round-trip reconstruction verified (max diff: %e)\n", max_diff);
}

static void test_fftshift(void) {
    printf("  -> Testing 2D FFTShift and IFFTShift quadrant permutation (4x4)...\n");
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    const size_t H = 4;
    const size_t W = 4;
    size_t shape[2] = {H, W};

    RCRE_Tensor mat_re = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor mat_im = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float* re = (float*)mat_re.data;
    memset(re, 0, H * W * sizeof(float));
    /* Place DC spike at top-left (0, 0) */
    re[0] = 99.0f;

    int status = rcre_fftshift_2d(&mat_re, &mat_im);
    assert(status == RCRE_OK);

    /* After fftshift on 4x4, (0, 0) shifts to (2, 2) */
    assert(re[2 * W + 2] == 99.0f);
    assert(re[0] == 0.0f);

    /* Invert shift */
    status = rcre_ifftshift_2d(&mat_re, &mat_im);
    assert(status == RCRE_OK);

    /* Back to (0, 0) */
    assert(re[0] == 99.0f);
    assert(re[2 * W + 2] == 0.0f);

    rcre_arena_destroy(arena);
    printf("     ✓ 2D FFTShift quadrant rotation centered DC frequency component\n");
}

static void test_fft_filter(void) {
    printf("  -> Testing 2D frequency-domain Gaussian filtering...\n");
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    assert(arena != NULL);

    const size_t H = 8;
    const size_t W = 8;
    size_t shape[2] = {H, W};

    RCRE_Tensor spec_re = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor spec_im = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    float* re = (float*)spec_re.data;
    for (size_t i = 0; i < H * W; ++i) {
        re[i] = 1.0f;
    }

    /* Apply Gaussian low-pass filter with cutoff radius 2.0 */
    int status = rcre_fft_filter_2d(&spec_re, &spec_im, 2.0f, false, true);
    assert(status == RCRE_OK);

    /* Center (4, 4) distance = 0 -> filter value must be 1.0 */
    size_t center_idx = 4 * W + 4;
    assert(fabsf(re[center_idx] - 1.0f) < 1e-4f);

    /* Corner (0, 0) distance = sqrt(32) ~ 5.65 -> filter value should be significantly attenuated */
    assert(re[0] < 0.1f);

    rcre_arena_destroy(arena);
    printf("     ✓ 2D Gaussian frequency-domain low-pass filter verified\n");
}

int main(void) {
    printf("🧪 Running RCRE Fast Fourier Transform Tests (FFT-REQ-01 to 04)...\n");
    test_fft_1d();
    test_fft_2d();
    test_fftshift();
    test_fft_filter();
    printf("✅ All Fast Fourier Transform Tests Passed Successfully!\n");
    return 0;
}
