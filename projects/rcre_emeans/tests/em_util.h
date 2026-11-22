/**
 * @file em_util.h
 * @brief Shared helpers for the rcre_e-means test suites.
 */
#ifndef EM_UTIL_H
#define EM_UTIL_H

#include "rcre.h"              /* core RCRE (incl. the e-means API) */
#include "e_means_internal.h"  /* core src/ml (EMState, em_* helpers) */
#include "test_framework.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Stream availability sentinel (written by the Makefile; 1 when the
 * oracle artifacts are present, 0 = degraded-coverage skip). */
#include "streams/generated.h"

/* --- file I/O ------------------------------------------------------------- */

[[maybe_unused]] static void *em_read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    void* buf = malloc((size_t)sz);
    if (buf && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    if (out_size) *out_size = (size_t)sz;
    return buf;
}

[[maybe_unused]] static RCRE_Tensor em_tensor2(RCRE_Arena* arena, size_t n, size_t d,
                                              RCRE_DType dtype) {
    size_t shape[2] = { n, d };
    return rcre_tensor_create(shape, 2, dtype, RCRE_LAYOUT_TABULAR, arena);
}

/* --- ULP arithmetic -------------------------------------------------------- */

[[maybe_unused]] static int32_t em_ulp_order(int32_t x) {
    return (x < 0) ? (int32_t)(~(uint32_t)x) : (x ^ 0x80000000);
}

/** Absolute ULP distance between two floats (0 when equal). */
[[maybe_unused]] static int em_ulp_diff(float a, float b) {
    if (a == b) return 0;
    int32_t ia, ib;
    memcpy(&ia, &a, 4);
    memcpy(&ib, &b, 4);
    int32_t oa = em_ulp_order(ia), ob = em_ulp_order(ib);
    int d = oa - ob;
    return d < 0 ? -d : d;
}

[[maybe_unused]] static size_t em_max_ulp(const float* a, const float* b, size_t n) {
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        int d = em_ulp_diff(a[i], b[i]);
        if ((size_t)d > m) m = (size_t)d;
    }
    return m;
}

/* --- synthetic data (deterministic, no ties) --------------------------------
 * Gaussian blobs around splitmix64-drawn centers; used by the quality,
 * determinism and assign suites (NOT by the parity suites, which read the
 * oracle's X_<case>.f32 files so both sides see identical data). */

[[maybe_unused]] static void em_make_blobs(float* X, size_t n, size_t d, size_t n_blobs,
                                          uint64_t seed) {
    uint64_t rs = seed;
    float* centers = malloc(n_blobs * d * sizeof(float));
    for (size_t i = 0; i < n_blobs * d; i++) {
        uint64_t r = rcre_splitmix64(&rs);
        float u = (float)((r >> 11) * (1.0 / 9007199254740992.0));
        uint64_t r2 = rcre_splitmix64(&rs);
        float v = (float)((r2 >> 11) * (1.0 / 9007199254740992.0));
        centers[i] = (float)sqrt(-2.0 * log((double)u + 1e-12))
                     * (float)cos(2.0 * M_PI * v) * 6.0f;
    }
    for (size_t i = 0; i < n; i++) {
        uint64_t r = rcre_splitmix64(&rs);
        size_t b = (size_t)(r % n_blobs);
        for (size_t dd = 0; dd < d; dd++) {
            uint64_t r1 = rcre_splitmix64(&rs);
            float u = (float)((r1 >> 11) * (1.0 / 9007199254740992.0));
            uint64_t r2 = rcre_splitmix64(&rs);
            float v = (float)((r2 >> 11) * (1.0 / 9007199254740992.0));
            float g = (float)sqrt(-2.0 * log((double)u + 1e-12))
                      * (float)cos(2.0 * M_PI * v) * 0.35f;
            X[i * d + dd] = centers[b * d + dd] + g;
        }
    }
    free(centers);
}

#endif /* EM_UTIL_H */
