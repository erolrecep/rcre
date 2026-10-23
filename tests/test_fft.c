/**
 * @file test_fft.c
 * @brief Unit tests for RCRE Fast Fourier Transform & Spectral Image Processing
 * 
 * Satisfies FFT-REQ-01 to FFT-REQ-04.
 */

#include "rcre.h"
#include "test_framework.h"

#define EPSILON 1e-5f

static void test_fft_1d(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

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
    ASSERT_INT_EQ(RCRE_OK, status);

    /* DC component for 1+2+...+8 = 36 */
    ASSERT_FLOAT_NEAR(36.0f, sig[0].re, 1e-4f);
    ASSERT_FLOAT_NEAR(0.0f, sig[0].im, 1e-4f);

    /* Inverse FFT */
    status = rcre_ifft_1d(sig, N, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* Assert exact reconstruction */
    for (size_t i = 0; i < N; ++i) {
        ASSERT_FLOAT_NEAR(orig[i].re, sig[i].re, EPSILON);
        ASSERT_FLOAT_NEAR(0.0f, sig[i].im, EPSILON);
    }

    rcre_arena_destroy(arena);
}

static void test_fft_2d(void) {
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

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
    ASSERT_INT_EQ(RCRE_OK, status);

    /* 2. Inverse 2D IFFT */
    status = rcre_ifft_2d(&out_re, &out_im, &recon, arena);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* 3. Check reconstruction fidelity */
    float* recon_data = (float*)recon.data;
    ASSERT_FLOAT_ARRAY_NEAR(img_data, recon_data, H * W, EPSILON);

    rcre_arena_destroy(arena);
}

static void test_fftshift(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

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
    ASSERT_INT_EQ(RCRE_OK, status);

    /* After fftshift on 4x4, (0, 0) shifts to (2, 2) */
    ASSERT_FLOAT_NEAR(99.0f, re[2 * W + 2], 1e-5f);
    ASSERT_FLOAT_NEAR(0.0f, re[0], 1e-5f);

    /* Invert shift */
    status = rcre_ifftshift_2d(&mat_re, &mat_im);
    ASSERT_INT_EQ(RCRE_OK, status);

    /* Back to (0, 0) */
    ASSERT_FLOAT_NEAR(99.0f, re[0], 1e-5f);
    ASSERT_FLOAT_NEAR(0.0f, re[2 * W + 2], 1e-5f);

    rcre_arena_destroy(arena);
}

static void test_fft_filter(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

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
    ASSERT_INT_EQ(RCRE_OK, status);

    /* Center (4, 4) distance = 0 -> filter value must be 1.0 */
    size_t center_idx = 4 * W + 4;
    ASSERT_FLOAT_NEAR(1.0f, re[center_idx], 1e-4f);

    /* Corner (0, 0) distance = sqrt(32) ~ 5.65 -> filter value should be significantly attenuated */
    ASSERT_TRUE(re[0] < 0.1f);

    rcre_arena_destroy(arena);
}

int main(void) {
    test_suite_begin("RCRE Fast Fourier Transform & Spectral Filter (FFT-REQ-01 to 04)");
    TEST_RUN(test_fft_1d);
    TEST_RUN(test_fft_2d);
    TEST_RUN(test_fftshift);
    TEST_RUN(test_fft_filter);
    return test_suite_report();
}
