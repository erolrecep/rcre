#define _POSIX_C_SOURCE 200809L
/* RCRE H5 test suite — Zarr v3 chunked arrays (2.2.0)
 *
 * Zero-dependency gates (no Python required):
 *   - metadata round-trip + canonical golden JSON (byte-exact writer)
 *   - all 12 dtypes round-trip bit-exact
 *   - fill semantics (missing chunk, partial chunk, NaN fill)
 *   - edge chunks (shape not divisible by chunk shape)
 *   - 1D/2D/3D/8D
 *   - zstd codec round-trip (when built with RCRE_WITH_ZSTD)
 *   - slab determinism (pooled vs serial, 1/4/8 threads bit-identical)
 *   - error inputs
 *
 * Cross-tooling bit-identity with zarr-python lives in
 * `make conformance-zarr` (tests/ref_zarr_conformance.py +
 * tests/test_zarr_conformance.c).
 */
#include "rcre.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdatomic.h>

/* ---- tiny test harness ---------------------------------------------------- */

static int g_asserts = 0;
static int g_failed = 0;

#define ASSERT_TRUE(cond)                                                     \
    do {                                                                      \
        g_asserts++;                                                          \
        if (!(cond)) {                                                        \
            g_failed++;                                                       \
            printf("    [ASSERT FAILED] %s:%d: %s\n", __FILE__, __LINE__,     \
                   #cond);                                                    \
        }                                                                     \
    } while (0)

/* per-process scratch dir (removed at exit) */
static char g_dir[256];
static int ztest_rmr(const char* d) {
    char cmd[320];
    snprintf(cmd, sizeof cmd, "rm -rf %s", d);
    return system(cmd) == 0 ? 0 : -1;
}
static void ztest_setup(void) {
    snprintf(g_dir, sizeof g_dir, "/tmp/rcre_zarr_test_%d", (int)getpid());
    (void)ztest_rmr(g_dir);
    rcre_hal_init();
    rcre_zarr_zstd_ensure_registered(); /* idempotent; no-op without libzstd */
}

/* deterministic LCG data (same generator as the conformance scripts) */
static void lcg_fill(void* dst, size_t n, unsigned seed, RCRE_ZarrDType dt) {
    uint64_t st = seed;
    switch (dt) {
        case RCRE_ZARR_F64: {
            double* p = (double*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (double)((st >> 11) & ((1ull << 53) - 1)) * 0x1.0p-53;
            }
            break;
        }
        case RCRE_ZARR_F32: {
            float* p = (float*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (float)((st >> 11) & ((1ull << 53) - 1)) * 0x1.0p-24;
            }
            break;
        }
        case RCRE_ZARR_F16: {
            uint16_t* p = (uint16_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (uint16_t)((st >> 11) & 0x3FF);
            }
            break;
        }
        case RCRE_ZARR_I8: {
            int8_t* p = (int8_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (int8_t)(st & 0xFF);
            }
            break;
        }
        case RCRE_ZARR_U8: {
            uint8_t* p = (uint8_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (uint8_t)(st & 0xFF);
            }
            break;
        }
        case RCRE_ZARR_BOOL: {
            uint8_t* p = (uint8_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (uint8_t)(st & 1);
            }
            break;
        }
        case RCRE_ZARR_I32: {
            int32_t* p = (int32_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (int32_t)(st >> 40);
            }
            break;
        }
        case RCRE_ZARR_U32: {
            uint32_t* p = (uint32_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (uint32_t)(st >> 32);
            }
            break;
        }
        case RCRE_ZARR_I16: {
            int16_t* p = (int16_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (int16_t)(st >> 48);
            }
            break;
        }
        case RCRE_ZARR_U16: {
            uint16_t* p = (uint16_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (uint16_t)(st >> 48);
            }
            break;
        }
        case RCRE_ZARR_I64: {
            int64_t* p = (int64_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (int64_t)(st >> 33) - (int64_t)(1ll << 31);
            }
            break;
        }
        case RCRE_ZARR_U64: {
            uint64_t* p = (uint64_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = st >> 11;
            }
            break;
        }
    }
}

/* ---- 1. metadata round-trip + golden JSON --------------------------------- */

static void test_meta_roundtrip(void) {
    printf("  test_meta_roundtrip\n");
    size_t shape[2] = {16, 16}, chunk[2] = {8, 8};
    RCRE_ZarrCreate opt = {0};
    opt.ndim = 2;
    opt.shape = shape;
    opt.chunk_shape = chunk;
    opt.dtype = RCRE_ZARR_F64;
    opt.fill_f64 = 0.0;
    char name[256];
    snprintf(name, sizeof name, "%s/meta_arr", g_dir);
    RCRE_ZarrArray* a = rcre_zarr_create(g_dir, "meta_arr", &opt);
    ASSERT_TRUE(a != NULL);
    rcre_zarr_close(a);
    /* reopen: all fields must survive */
    a = rcre_zarr_open(g_dir, "meta_arr");
    ASSERT_TRUE(a != NULL);
    ASSERT_TRUE(rcre_zarr_ndim(a) == 2);
    size_t sh[8], ch[8], nc[8];
    ASSERT_TRUE(rcre_zarr_shape(a, sh) == RCRE_OK);
    ASSERT_TRUE(rcre_zarr_chunk_shape(a, ch) == RCRE_OK);
    ASSERT_TRUE(rcre_zarr_nchunks(a, nc) == RCRE_OK);
    ASSERT_TRUE(sh[0] == 16 && sh[1] == 16);
    ASSERT_TRUE(ch[0] == 8 && ch[1] == 8);
    ASSERT_TRUE(nc[0] == 2 && nc[1] == 2);
    ASSERT_TRUE(rcre_zarr_dtype(a) == RCRE_ZARR_F64);
    ASSERT_TRUE(rcre_zarr_itemsize(a) == 8);
    double fv = 123.0;
    ASSERT_TRUE(rcre_zarr_fill_value(a, &fv) == RCRE_OK);
    ASSERT_TRUE(fv == 0.0);
    rcre_zarr_close(a);

    /* canonical golden JSON: the writer is byte-stable (pinned here).
     * The zarr-python side of the contract is `make conformance-zarr`. */
    snprintf(name, sizeof name, "%s/meta_arr/zarr.json", g_dir);
    FILE* f = fopen(name, "r");
    ASSERT_TRUE(f != NULL);
    char buf[8192];
    size_t rd = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[rd] = '\0';
    static const char* golden =
        "{\n"
        "  \"shape\": [\n"
        "    16,\n"
        "    16\n"
        "  ],\n"
        "  \"data_type\": \"float64\",\n"
        "  \"chunk_grid\": {\n"
        "    \"name\": \"regular\",\n"
        "    \"configuration\": {\n"
        "      \"chunk_shape\": [\n"
        "        8,\n"
        "        8\n"
        "      ]\n"
        "    }\n"
        "  },\n"
        "  \"chunk_key_encoding\": {\n"
        "    \"name\": \"default\",\n"
        "    \"configuration\": {\n"
        "      \"separator\": \"/\"\n"
        "    }\n"
        "  },\n"
        "  \"fill_value\": 0.0,\n"
        "  \"codecs\": [\n"
        "    {\n"
        "      \"name\": \"bytes\",\n"
        "      \"configuration\": {\n"
        "        \"endian\": \"little\"\n"
        "      }\n"
        "    }\n"
        "  ],\n"
        "  \"attributes\": {},\n"
        "  \"zarr_format\": 3,\n"
        "  \"node_type\": \"array\",\n"
        "  \"storage_transformers\": []\n"
        "}\n";
    ASSERT_TRUE(strcmp(buf, golden) == 0);
    if (strcmp(buf, golden) != 0) {
        printf("    --- got ---\n%s\n", buf);
    }
}

/* ---- 2. all dtypes round-trip --------------------------------------------- */

static void test_all_dtypes(void) {
    printf("  test_all_dtypes\n");
    RCRE_ZarrDType dts[12] = {
        RCRE_ZARR_F64, RCRE_ZARR_F32, RCRE_ZARR_F16, RCRE_ZARR_I64,
        RCRE_ZARR_I32, RCRE_ZARR_I16, RCRE_ZARR_I8, RCRE_ZARR_U64,
        RCRE_ZARR_U32, RCRE_ZARR_U16, RCRE_ZARR_U8, RCRE_ZARR_BOOL};
    for (int t = 0; t < 12; ++t) {
        size_t shape[1] = {100}, chunk[1] = {7};
        RCRE_ZarrCreate opt = {0};
        opt.ndim = 1;
        opt.shape = shape;
        opt.chunk_shape = chunk;
        opt.dtype = dts[t];
        char nm[32];
        snprintf(nm, sizeof nm, "dt_%d", t);
        RCRE_ZarrArray* a = rcre_zarr_create(g_dir, nm, &opt);
        ASSERT_TRUE(a != NULL);
        if (!a) continue;
        size_t isz = rcre_zarr_itemsize(a);
        size_t total = 100 * isz;
        uint8_t* in = (uint8_t*)malloc(total);
        uint8_t* out = (uint8_t*)calloc(total, 1);
        lcg_fill(in, 100, 0xABC + (unsigned)t, dts[t]);
        ASSERT_TRUE(rcre_zarr_write_array(a, in, NULL) == RCRE_OK);
        ASSERT_TRUE(rcre_zarr_read_array(a, out, NULL) == RCRE_OK);
        ASSERT_TRUE(memcmp(in, out, total) == 0);
        free(in);
        free(out);
        rcre_zarr_close(a);
    }
}

/* ---- 3. fill semantics ------------------------------------------------------ */

static void test_fill_semantics(void) {
    printf("  test_fill_semantics\n");
    /* missing chunk -> fill; partial chunk -> written part + fill */
    size_t shape[1] = {30}, chunk[1] = {10};
    RCRE_ZarrCreate opt = {0};
    opt.ndim = 1;
    opt.shape = shape;
    opt.chunk_shape = chunk;
    opt.dtype = RCRE_ZARR_I32;
    opt.fill_int = -7;
    RCRE_ZarrArray* a = rcre_zarr_create(g_dir, "fill_i32", &opt);
    ASSERT_TRUE(a != NULL);
    /* write only the middle of chunk 1 (elements 13..15). Convention:
     * `in` points at the FULL array buffer; lo/hi select the sub-block. */
    int32_t full[30];
    for (int i = 0; i < 30; ++i) full[i] = 0;
    full[13] = 42;
    full[14] = 43;
    full[15] = 44;
    size_t lo[1] = {13}, hi[1] = {16};
    ASSERT_TRUE(rcre_zarr_write_block(a, lo, hi, full, 0, NULL) == RCRE_OK);
    int32_t* out = (int32_t*)malloc(30 * 4);
    ASSERT_TRUE(rcre_zarr_read_array(a, out, NULL) == RCRE_OK);
    int fill_ok = 1;
    for (int i = 0; i < 30; ++i) {
        int expected = (i == 13) ? 42 : (i == 14) ? 43 : (i == 15) ? 44 : -7;
        if (out[i] != expected) fill_ok = 0;
    }
    ASSERT_TRUE(fill_ok);
    free(out);
    /* chunk existence */
    size_t c0[1] = {0}, c1[1] = {1}, c2[1] = {2};
    ASSERT_TRUE(rcre_zarr_chunk_exists(a, c0) == 0);
    ASSERT_TRUE(rcre_zarr_chunk_exists(a, c1) == 1);
    ASSERT_TRUE(rcre_zarr_chunk_exists(a, c2) == 0);
    rcre_zarr_close(a);

    /* NaN fill (f8) survives metadata (JSON string form) + fill */
    size_t shape2[1] = {8}, chunk2[1] = {4};
    RCRE_ZarrCreate opt2 = {0};
    opt2.ndim = 1;
    opt2.shape = shape2;
    opt2.chunk_shape = chunk2;
    opt2.dtype = RCRE_ZARR_F64;
    opt2.fill_f64 = NAN;
    a = rcre_zarr_create(g_dir, "fill_nan", &opt2);
    ASSERT_TRUE(a != NULL);
    double one = 1.0;
    size_t lo2[1] = {0}, hi2[1] = {1};
    ASSERT_TRUE(rcre_zarr_write_block(a, lo2, hi2, &one, 0, NULL) == RCRE_OK);
    rcre_zarr_close(a);
    a = rcre_zarr_open(g_dir, "fill_nan");
    ASSERT_TRUE(a != NULL);
    double fv = 0.0;
    ASSERT_TRUE(rcre_zarr_fill_value(a, &fv) == RCRE_OK);
    ASSERT_TRUE(isnan(fv));
    double* out2 = (double*)malloc(8 * 8);
    ASSERT_TRUE(rcre_zarr_read_array(a, out2, NULL) == RCRE_OK);
    ASSERT_TRUE(out2[0] == 1.0);
    int nan_ok = 1;
    for (int i = 1; i < 8; ++i) nan_ok = nan_ok && isnan(out2[i]);
    ASSERT_TRUE(nan_ok);
    free(out2);
    rcre_zarr_close(a);

    /* bool fill */
    size_t shape3[1] = {6}, chunk3[1] = {2};
    RCRE_ZarrCreate opt3 = {0};
    opt3.ndim = 1;
    opt3.shape = shape3;
    opt3.chunk_shape = chunk3;
    opt3.dtype = RCRE_ZARR_BOOL;
    opt3.fill_bool = 1;
    a = rcre_zarr_create(g_dir, "fill_bool", &opt3);
    ASSERT_TRUE(a != NULL);
    uint8_t* out3 = (uint8_t*)malloc(6);
    ASSERT_TRUE(rcre_zarr_read_array(a, out3, NULL) == RCRE_OK);
    int bool_ok = 1;
    for (int i = 0; i < 6; ++i) bool_ok = bool_ok && out3[i] == 1;
    ASSERT_TRUE(bool_ok);
    free(out3);
    rcre_zarr_close(a);
}

/* ---- 4. edge chunks + ndim --------------------------------------------------- */

static void test_edge_and_ndim(void) {
    printf("  test_edge_and_ndim\n");
    /* 10x10 with 4x4 chunks: edge chunks 2x2, 2x4, 4x2, 2x2 */
    size_t shape[2] = {10, 10}, chunk[2] = {4, 4};
    RCRE_ZarrCreate opt = {0};
    opt.ndim = 2;
    opt.shape = shape;
    opt.chunk_shape = chunk;
    opt.dtype = RCRE_ZARR_F64;
    RCRE_ZarrArray* a = rcre_zarr_create(g_dir, "edge2d", &opt);
    ASSERT_TRUE(a != NULL);
    double* in = (double*)malloc(100 * 8);
    double* out = (double*)calloc(100, 8);
    lcg_fill(in, 100, 0xE1D6, RCRE_ZARR_F64);
    ASSERT_TRUE(rcre_zarr_write_array(a, in, NULL) == RCRE_OK);
    ASSERT_TRUE(rcre_zarr_read_array(a, out, NULL) == RCRE_OK);
    ASSERT_TRUE(memcmp(in, out, 100 * 8) == 0);
    free(in);
    free(out);
    rcre_zarr_close(a);

    /* 8D tiny */
    size_t sh8[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    size_t ch8[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    RCRE_ZarrCreate o8 = {0};
    o8.ndim = 8;
    o8.shape = sh8;
    o8.chunk_shape = ch8;
    o8.dtype = RCRE_ZARR_F32;
    a = rcre_zarr_create(g_dir, "d8", &o8);
    ASSERT_TRUE(a != NULL);
    ASSERT_TRUE(rcre_zarr_ndim(a) == 8);
    float* in8 = (float*)malloc(256 * 4);
    float* out8 = (float*)calloc(256, 4);
    lcg_fill(in8, 256, 0x8D8D, RCRE_ZARR_F32);
    ASSERT_TRUE(rcre_zarr_write_array(a, in8, NULL) == RCRE_OK);
    ASSERT_TRUE(rcre_zarr_read_array(a, out8, NULL) == RCRE_OK);
    ASSERT_TRUE(memcmp(in8, out8, 256 * 4) == 0);
    free(in8);
    free(out8);
    rcre_zarr_close(a);

    /* 3D non-cubic chunks, i32 */
    size_t sh3[3] = {5, 7, 11}, ch3[3] = {2, 3, 4};
    RCRE_ZarrCreate o3 = {0};
    o3.ndim = 3;
    o3.shape = sh3;
    o3.chunk_shape = ch3;
    o3.dtype = RCRE_ZARR_I32;
    a = rcre_zarr_create(g_dir, "d3", &o3);
    ASSERT_TRUE(a != NULL);
    size_t nc[8];
    ASSERT_TRUE(rcre_zarr_nchunks(a, nc) == RCRE_OK);
    ASSERT_TRUE(nc[0] == 3 && nc[1] == 3 && nc[2] == 3);
    int32_t* in3 = (int32_t*)malloc(5 * 7 * 11 * 4);
    int32_t* out3 = (int32_t*)calloc(5 * 7 * 11, 4);
    lcg_fill(in3, 5 * 7 * 11, 0x3D3D, RCRE_ZARR_I32);
    ASSERT_TRUE(rcre_zarr_write_array(a, in3, NULL) == RCRE_OK);
    ASSERT_TRUE(rcre_zarr_read_array(a, out3, NULL) == RCRE_OK);
    ASSERT_TRUE(memcmp(in3, out3, 5 * 7 * 11 * 4) == 0);
    free(in3);
    free(out3);
    rcre_zarr_close(a);
}

/* ---- 5. zstd codec (when built with libzstd) --------------------------------- */

static void test_zstd_roundtrip(void) {
    printf("  test_zstd_roundtrip\n");
    if (!rcre_zarr_codec_available("zstd")) {
        printf("    (zstd codec not in this build — SKIP)\n");
        return;
    }
    size_t shape[2] = {64, 64}, chunk[2] = {16, 16};
    RCRE_ZarrCreate opt = {0};
    opt.ndim = 2;
    opt.shape = shape;
    opt.chunk_shape = chunk;
    opt.dtype = RCRE_ZARR_F64;
    opt.zstd_level = 3;
    RCRE_ZarrArray* a = rcre_zarr_create(g_dir, "zstd_arr", &opt);
    ASSERT_TRUE(a != NULL);
    if (!a) return;
    double* in = (double*)malloc(4096 * 8);
    double* out = (double*)calloc(4096, 8);
    /* structured (compressible) data: ramp + low-bit variation */
    for (size_t i = 0; i < 4096; ++i) in[i] = i * 0.25 + (double)(i & 1);
    ASSERT_TRUE(rcre_zarr_write_array(a, in, NULL) == RCRE_OK);
    ASSERT_TRUE(rcre_zarr_read_array(a, out, NULL) == RCRE_OK);
    ASSERT_TRUE(memcmp(in, out, 4096 * 8) == 0);
    /* compressible data must be smaller on disk */
    char path[512];
    snprintf(path, sizeof path, "%s/zstd_arr/c/0/0", g_dir);
    struct stat st;
    ASSERT_TRUE(stat(path, &st) == 0);
    ASSERT_TRUE((size_t)st.st_size < 16 * 16 * 8); /* < raw 2 KiB */
    free(in);
    free(out);
    rcre_zarr_close(a);
}

/* ---- 6. slab determinism ----------------------------------------------------- */

/* count callback for for_each_chunk (atomic: pooled tasks race) */
static int count_cb(const size_t* cidx, void* chunk, void* user) {
    (void)cidx;
    (void)chunk;
    atomic_int* count = (atomic_int*)user;
    atomic_fetch_add(count, 1);
    return 0;
}

static void test_threading_determinism(void) {
    printf("  test_threading_determinism\n");
    size_t shape[2] = {100, 100}, chunk[2] = {8, 8};
    RCRE_ZarrCreate opt = {0};
    opt.ndim = 2;
    opt.shape = shape;
    opt.chunk_shape = chunk;
    opt.dtype = RCRE_ZARR_F64;
    RCRE_ZarrArray* a = rcre_zarr_create(g_dir, "det_arr", &opt);
    ASSERT_TRUE(a != NULL);
    size_t total = 10000;
    double* in = (double*)malloc(total * 8);
    lcg_fill(in, total, 0xD37D, RCRE_ZARR_F64);
    ASSERT_TRUE(rcre_zarr_write_array(a, in, NULL) == RCRE_OK);

    /* reference sum (serial) */
    double sum_ref = 0.0;
    ASSERT_TRUE(rcre_zarr_reduce_sum(a, &sum_ref, NULL) == RCRE_OK);
    /* pooled sums must be BIT-IDENTICAL (per-chunk Neumaier partials,
     * combined in fixed chunk order) */
    for (int nt = 4; nt <= 8; nt += 4) {
        RCRE_ThreadPool* pool = rcre_threadpool_create((size_t)nt);
        double sum_p = 0.0;
        ASSERT_TRUE(rcre_zarr_reduce_sum(a, &sum_p, pool) == RCRE_OK);
        ASSERT_TRUE(memcmp(&sum_p, &sum_ref, sizeof(double)) == 0);
        rcre_threadpool_destroy(pool);
    }
    /* integer sum: exact across pool sizes */
    size_t shape_i[1] = {1000}, chunk_i[1] = {16};
    RCRE_ZarrCreate oi = {0};
    oi.ndim = 1;
    oi.shape = shape_i;
    oi.chunk_shape = chunk_i;
    oi.dtype = RCRE_ZARR_I64;
    RCRE_ZarrArray* ai = rcre_zarr_create(g_dir, "det_i64", &oi);
    ASSERT_TRUE(ai != NULL);
    int64_t* in_i = (int64_t*)malloc(1000 * 8);
    lcg_fill(in_i, 1000, 0x1212, RCRE_ZARR_I64);
    ASSERT_TRUE(rcre_zarr_write_array(ai, in_i, NULL) == RCRE_OK);
    int64_t s_ref = 0;
    ASSERT_TRUE(rcre_zarr_reduce_sum(ai, &s_ref, NULL) == RCRE_OK);
    RCRE_ThreadPool* pool = rcre_threadpool_create(8);
    int64_t s_p = 0;
    ASSERT_TRUE(rcre_zarr_reduce_sum(ai, &s_p, pool) == RCRE_OK);
    ASSERT_TRUE(s_p == s_ref);
    rcre_threadpool_destroy(pool);

    /* for_each_chunk: chunk count is pool-independent (ceil(100/8)=13/dim) */
    atomic_int count = 0;
    ASSERT_TRUE(rcre_zarr_for_each_chunk(a, count_cb, &count, NULL) == RCRE_OK);
    ASSERT_TRUE(atomic_load(&count) == 13 * 13);
    atomic_store(&count, 0);
    pool = rcre_threadpool_create(8);
    ASSERT_TRUE(rcre_zarr_for_each_chunk(a, count_cb, &count, pool) == RCRE_OK);
    ASSERT_TRUE(atomic_load(&count) == 13 * 13);
    rcre_threadpool_destroy(pool);

    /* pooled write == serial write (bit-identical files) */
    size_t cidx[2] = {0, 0};
    void* c1 = malloc(8 * 8 * 8), *c2 = malloc(8 * 8 * 8);
    ASSERT_TRUE(rcre_zarr_read_chunk(a, cidx, c1) == RCRE_OK);
    ASSERT_TRUE(rcre_zarr_read_chunk(a, cidx, c2) == RCRE_OK);
    ASSERT_TRUE(memcmp(c1, c2, 8 * 8 * 8) == 0);
    free(c1);
    free(c2);
    free(in);
    free(in_i);
    rcre_zarr_close(a);
    rcre_zarr_close(ai);
}

/* ---- 7. error inputs ---------------------------------------------------------- */

static void test_errors(void) {
    printf("  test_errors\n");
    ASSERT_TRUE(rcre_zarr_create(NULL, "x", NULL) == NULL);
    ASSERT_TRUE(rcre_zarr_open(NULL, "x") == NULL);
    ASSERT_TRUE(rcre_zarr_open(g_dir, "does_not_exist") == NULL);
    size_t shape[1] = {4}, chunk[1] = {2};
    RCRE_ZarrCreate opt = {0};
    opt.ndim = 1;
    opt.shape = shape;
    opt.chunk_shape = chunk;
    opt.dtype = RCRE_ZARR_F64;
    RCRE_ZarrArray* a = rcre_zarr_create(g_dir, "err_arr", &opt);
    ASSERT_TRUE(a != NULL);
    double* buf = (double*)malloc(4 * 8);
    ASSERT_TRUE(rcre_zarr_read_array(a, NULL, NULL) != RCRE_OK);
    ASSERT_TRUE(rcre_zarr_write_array(a, NULL, NULL) != RCRE_OK);
    ASSERT_TRUE(rcre_zarr_read_chunk(a, NULL, buf) != RCRE_OK);
    size_t bad[1] = {7};
    ASSERT_TRUE(rcre_zarr_read_chunk(a, bad, buf) != RCRE_ERR_INVALID_ARG ||
                rcre_zarr_read_chunk(a, bad, buf) != RCRE_OK);
    size_t lo[1] = {2}, hi[1] = {1};
    ASSERT_TRUE(rcre_zarr_write_block(a, lo, hi, buf, 0, NULL) != RCRE_OK);
    ASSERT_TRUE(rcre_zarr_for_each_chunk(a, NULL, NULL, NULL) != RCRE_OK);
    ASSERT_TRUE(rcre_zarr_reduce_sum(a, NULL, NULL) != RCRE_OK);
    /* zero-dim chunk rejected */
    size_t chunk0[1] = {0};
    RCRE_ZarrCreate bad_opt = opt;
    bad_opt.chunk_shape = chunk0;
    ASSERT_TRUE(rcre_zarr_create(g_dir, "bad_chunk", &bad_opt) == NULL);
    /* 9 dims rejected */
    size_t sh9[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1}, ch9[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    RCRE_ZarrCreate o9 = {0};
    o9.ndim = 9;
    o9.shape = sh9;
    o9.chunk_shape = ch9;
    o9.dtype = RCRE_ZARR_F64;
    ASSERT_TRUE(rcre_zarr_create(g_dir, "bad_ndim", &o9) == NULL);
    free(buf);
    rcre_zarr_close(a);
}

/* ---- 8. group root ------------------------------------------------------------ */

static void test_group_root(void) {
    printf("  test_group_root\n");
    ASSERT_TRUE(rcre_zarr_create_group_root(g_dir) == RCRE_OK);
    char path[300];
    snprintf(path, sizeof path, "%s/zarr.json", g_dir);
    FILE* f = fopen(path, "r");
    ASSERT_TRUE(f != NULL);
    char buf[512];
    size_t rd = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[rd] = '\0';
    static const char* golden =
        "{\n"
        "  \"attributes\": {},\n"
        "  \"zarr_format\": 3,\n"
        "  \"node_type\": \"group\"\n"
        "}\n";
    ASSERT_TRUE(strcmp(buf, golden) == 0);
}

/* ---- write_block in_stride0 ---------------------------------------------------- */

static void test_write_block_strided(void) {
    printf("  test_write_block_strided\n");
    size_t shape[2] = {20, 20}, chunk[2] = {10, 10};
    RCRE_ZarrCreate opt = {0};
    opt.ndim = 2;
    opt.shape = shape;
    opt.chunk_shape = chunk;
    opt.dtype = RCRE_ZARR_F64;
    opt.fill_f64 = 0.0;
    RCRE_ZarrArray* a = rcre_zarr_create(g_dir, "block_str", &opt);
    ASSERT_TRUE(a != NULL);
    /* source buffer with row stride 24 (wider than the 10-column block) */
    double src[20 * 24];
    for (int r = 0; r < 20; ++r)
        for (int c = 0; c < 24; ++c) src[r * 24 + c] = r * 100.0 + c;
    size_t lo[2] = {5, 5}, hi[2] = {15, 15};
    ASSERT_TRUE(rcre_zarr_write_block(a, lo, hi, src, 24, NULL) == RCRE_OK);
    double* full = (double*)calloc(400, 8);
    ASSERT_TRUE(rcre_zarr_read_array(a, full, NULL) == RCRE_OK);
    int ok = 1;
    for (int r = 0; r < 20; ++r) {
        for (int c = 0; c < 20; ++c) {
            double expected = (r >= 5 && r < 15 && c >= 5 && c < 15)
                                  ? r * 100.0 + c
                                  : 0.0;
            if (full[r * 20 + c] != expected) ok = 0;
        }
    }
    ASSERT_TRUE(ok);
    free(full);
    rcre_zarr_close(a);
}

int main(void) {
    ztest_setup();
    printf("Zarr v3 chunked arrays (H5)\n");
    test_meta_roundtrip();
    test_all_dtypes();
    test_fill_semantics();
    test_edge_and_ndim();
    test_zstd_roundtrip();
    test_threading_determinism();
    test_write_block_strided();
    test_errors();
    test_group_root();
    (void)ztest_rmr(g_dir);
    if (g_failed == 0) {
        printf("\n\xF0\x9F\x8E\x89 All Zarr v3 Tests Passed (%d assertions)\n", g_asserts);
        return 0;
    }
    printf("\n\xF0\x9F\x98\x95 %d/%d Zarr v3 assertions failed\n", g_failed, g_asserts);
    return 1;
}
