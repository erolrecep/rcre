/**
 * @file test_io.c
 * @brief Unit, boundary, adversarial & performance verification suite for
 *        RCRE High-Performance Scientific Data I/O (IO-REQ-01 to IO-REQ-06).
 */

#include "rcre.h"
#include "rcre_io.h"
#include "rcre_prof.h"
#include "test_framework.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================= */
/* 1. Virtual Memory Mapping, Atomic Writes & Profiler Telemetry (IO-REQ-01) */
/* ========================================================================= */

static void test_io_mmap_atomic_write_and_telemetry(void) {
    RCRE_ProfContext* prof = rcre_prof_init(64);
    ASSERT_NOT_NULL(prof);
    rcre_prof_set_global(prof);

    const char* path = "/tmp/rcre_test_io_atomic.bin";
    uint8_t payload[4096];
    for (size_t i = 0; i < sizeof(payload); ++i) {
        payload[i] = (uint8_t)((i * 37 + 13) & 0xFF);
    }

    int rc = rcre_io_write_atomic(path, payload, sizeof(payload));
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(sizeof(payload), (size_t)prof->io_write_app);

    /* Open via mmap with SEQUENTIAL hint */
    RCRE_MappedFile map = {0};
    rc = rcre_io_mmap_open(path, RCRE_IO_HINT_SEQUENTIAL, &map);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_NOT_NULL(map.data);
    ASSERT_UINT_EQ(sizeof(payload), map.size);
    ASSERT_UINT_EQ(0, (uintptr_t)map.data % 64); /* OS page is >= 64B aligned */
    ASSERT_UINT_EQ(sizeof(payload), (size_t)prof->io_read_app);

    RCRE_Slice s = rcre_io_mmap_slice(&map);
    ASSERT_UINT_EQ(sizeof(payload), s.len);
    ASSERT_MEM_EQ(payload, s.data, sizeof(payload));

    /* Sub-range madvise hints */
    ASSERT_INT_EQ(RCRE_OK, rcre_io_madvise_range(&map, 128, 1024, RCRE_IO_HINT_RANDOM));
    ASSERT_INT_EQ(RCRE_OK, rcre_io_madvise_range(&map, 0, sizeof(payload), RCRE_IO_HINT_STREAMING));
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_io_madvise_range(&map, sizeof(payload) + 1, 64, RCRE_IO_HINT_DEFAULT));

    rcre_io_mmap_close(&map);
    ASSERT_NULL(map.data);
    ASSERT_UINT_EQ(0, map.size);

    /* Test in-memory buffer wrapper (WASM / embedded path) */
    RCRE_MappedFile mem_map = {0};
    rc = rcre_io_mmap_from_buffer(payload, sizeof(payload), &mem_map);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_FALSE(mem_map.is_mapped);
    ASSERT_TRUE(mem_map.data == payload);
    rcre_io_mmap_close(&mem_map);

    /* Error path checks */
    ASSERT_INT_EQ(RCRE_ERR_IO, rcre_io_mmap_open("/tmp/non_existent_rcre_file_99999.bin", RCRE_IO_HINT_DEFAULT, &map));
    ASSERT_INT_EQ(RCRE_ERR_IO, rcre_io_mmap_open("/tmp", RCRE_IO_HINT_DEFAULT, &map)); /* Directory rejected */
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_io_mmap_open(nullptr, RCRE_IO_HINT_DEFAULT, &map));

    remove(path);
    rcre_prof_set_global(nullptr);
    rcre_prof_free(prof);
}

/* ========================================================================= */
/* 2. Clinger / Eisel-Lemire 0-ULP Fast Float32 Parser (IO-REQ-02)           */
/* ========================================================================= */

static void test_io_fast_float_eisel_lemire_0_ulp_parity(void) {
    /* 1. Exact special & boundary values */
    float val = 0.0f;
    ASSERT_TRUE(rcre_slice_parse_f32(rcre_slice_from_cstr("0"), &val));
    ASSERT_FLOAT_NEAR(0.0f, val, 0.0f);

    ASSERT_TRUE(rcre_slice_parse_f32(rcre_slice_from_cstr("  -0.000  "), &val));
    ASSERT_FLOAT_NEAR(0.0f, val, 0.0f);
    ASSERT_TRUE(signbit(val) != 0);

    ASSERT_TRUE(rcre_slice_parse_f32(rcre_slice_from_cstr(".5"), &val));
    ASSERT_FLOAT_NEAR(0.5f, val, 0.0f);

    ASSERT_TRUE(rcre_slice_parse_f32(rcre_slice_from_cstr("5."), &val));
    ASSERT_FLOAT_NEAR(5.0f, val, 0.0f);

    ASSERT_TRUE(rcre_slice_parse_f32(rcre_slice_from_cstr("+3.14159265"), &val));
    ASSERT_FLOAT_NEAR(3.14159265f, val, 0.0f);

    ASSERT_TRUE(rcre_slice_parse_f32(rcre_slice_from_cstr("NaN"), &val));
    ASSERT_TRUE(isnan(val));

    ASSERT_TRUE(rcre_slice_parse_f32(rcre_slice_from_cstr("-Inf"), &val));
    ASSERT_TRUE(isinf(val) && val < 0.0f);

    ASSERT_TRUE(rcre_slice_parse_f32(rcre_slice_from_cstr("INFINITY"), &val));
    ASSERT_TRUE(isinf(val) && val > 0.0f);

    /* 2. Malformed input rejection */
    ASSERT_FALSE(rcre_slice_parse_f32(rcre_slice_empty(), &val));
    ASSERT_FALSE(rcre_slice_parse_f32(rcre_slice_from_cstr("   "), &val));
    ASSERT_FALSE(rcre_slice_parse_f32(rcre_slice_from_cstr("."), &val));
    ASSERT_FALSE(rcre_slice_parse_f32(rcre_slice_from_cstr("-"), &val));
    ASSERT_FALSE(rcre_slice_parse_f32(rcre_slice_from_cstr("+"), &val));
    ASSERT_FALSE(rcre_slice_parse_f32(rcre_slice_from_cstr("1.2.3"), &val));
    ASSERT_FALSE(rcre_slice_parse_f32(rcre_slice_from_cstr("123.45abc"), &val));
    ASSERT_FALSE(rcre_slice_parse_f32(rcre_slice_from_cstr("1.0e"), &val));
    ASSERT_FALSE(rcre_slice_parse_f32(rcre_slice_from_cstr("1.0e+"), &val));
    ASSERT_FALSE(rcre_slice_parse_f32(rcre_slice_from_cstr("e10"), &val));

    /* 3. 10,000-number 0-ULP strict bit-exact sweep vs libc strtof across
     *    normal, scientific, tiny (1e-38), and large (1e+38) ranges */
    uint64_t rng = 0xCAFEBABE2026ULL;
    char buf[64];
    for (int i = 0; i < 10000; ++i) {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        uint64_t r = rng * 2685821657736338717ULL;

        int exp_choice = (int)(r % 77) - 38; /* -38 .. +38 */
        uint32_t mant = (uint32_t)((r >> 16) % 10000000ULL);
        bool is_neg = ((r >> 48) & 1ULL) != 0;

        int len = snprintf(buf, sizeof(buf), "%s%u.%06ue%+d",
                           is_neg ? "-" : "",
                           mant / 1000000U,
                           mant % 1000000U,
                           exp_choice);
        ASSERT_TRUE(len > 0);

        float expected = strtof(buf, nullptr);
        float actual = 0.0f;
        ASSERT_TRUE(rcre_slice_parse_f32(rcre_slice_from_parts(buf, (size_t)len), &actual));

        uint32_t bits_exp = 0;
        uint32_t bits_act = 0;
        memcpy(&bits_exp, &expected, 4);
        memcpy(&bits_act, &actual, 4);
        ASSERT_UINT_EQ(bits_exp, bits_act); /* 0-ULP bit-exact match */
    }
}

/* ========================================================================= */
/* 3. SIMD-Accelerated & Threadpool-Parallel CSV Loader (IO-REQ-02)          */
/* ========================================================================= */

static void test_io_simd_csv_and_parallel_threadpool(void) {
    RCRE_Arena* arena = rcre_arena_create(8 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const char* csv_path = "/tmp/rcre_test_parallel_io.csv";
    FILE* f = fopen(csv_path, "w");
    ASSERT_NOT_NULL(f);

    const size_t ROWS = 256;
    const size_t COLS = 16;

    /* Header row */
    for (size_t c = 0; c < COLS; ++c) {
        fprintf(f, "%scol_%zu", (c == 0) ? "" : ",", c);
    }
    fprintf(f, "\r\n"); /* Test CRLF handling */

    for (size_t r = 0; r < ROWS; ++r) {
        for (size_t c = 0; c < COLS; ++c) {
            float v = (float)r * 10.0f + (float)c * 0.25f - 50.5f;
            fprintf(f, "%s%.4f", (c == 0) ? "" : ",", v);
        }
        fprintf(f, "\r\n");
    }
    fclose(f);

    /* 1. Single-threaded load */
    RCRE_CSVOptions opts_serial = {
        .delimiter      = ',',
        .skip_header    = true,
        .strict_numbers = true,
        .max_rows       = 0,
        .pool           = nullptr
    };
    RCRE_Tensor t_serial = {0};
    int rc = rcre_io_load_csv_ex(csv_path, &opts_serial, arena, &t_serial);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(ROWS, t_serial.shape[0]);
    ASSERT_UINT_EQ(COLS, t_serial.shape[1]);
    ASSERT_UINT_EQ(0, (uintptr_t)t_serial.data % 64);

    /* 2. Multi-threaded parallel slab load via RCRE_ThreadPool */
    RCRE_ThreadPool* pool = rcre_threadpool_create(4);
    ASSERT_NOT_NULL(pool);

    RCRE_CSVOptions opts_parallel = opts_serial;
    opts_parallel.pool = pool;

    RCRE_Tensor t_parallel = {0};
    size_t offset_before = arena->offset;
    rc = rcre_io_load_csv_ex(csv_path, &opts_parallel, arena, &t_parallel);
    size_t offset_after = arena->offset;
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(ROWS, t_parallel.shape[0]);
    ASSERT_UINT_EQ(COLS, t_parallel.shape[1]);

    /* Verify scratch line index was cleanly rewound from arena */
    ASSERT_TRUE((offset_after - offset_before) <= (ROWS * COLS * sizeof(float) + 64));

    /* Assert bit-exact parity between serial and parallel CSV ingestion */
    ASSERT_MEM_EQ(t_serial.data, t_parallel.data, ROWS * COLS * sizeof(float));

    /* 3. Strict number check rejects invalid token in parallel mode */
    const char* bad_csv = "1.0,2.0\n3.0,NOT_A_FLOAT\n5.0,6.0\n";
    RCRE_Tensor t_bad = {0};
    ASSERT_INT_EQ(RCRE_ERR_IO, rcre_io_parse_csv_slice(
        rcre_slice_from_cstr(bad_csv), &opts_serial, arena, &t_bad));

    remove(csv_path);
    rcre_threadpool_destroy(pool);
    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* 4. NumPy (.npy v1.0 & v2.0) Zero-Copy Views & Adversarial Tests (IO-REQ-03)*/
/* ========================================================================= */

static void test_io_npy_roundtrip_zero_copy_and_adversarial(void) {
    RCRE_Arena* arena = rcre_arena_create(8 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 1. 3D F32 Tensor Round-Trip & Zero-Copy View */
    size_t shape3d[3] = { 8, 16, 3 };
    RCRE_Tensor orig = rcre_tensor_create(shape3d, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    ASSERT_NOT_NULL(orig.data);
    float* od = rcre_tensor_f32_data(&orig);
    for (size_t i = 0; i < orig.numel; ++i) {
        od[i] = (float)i * 0.125f - 12.0f;
    }

    const char* npy_path = "/tmp/rcre_test_tensor.npy";
    int rc = rcre_io_save_npy(&orig, npy_path);
    ASSERT_INT_EQ(RCRE_OK, rc);

    /* Verify zero-copy mmap view */
    RCRE_MappedFile map = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_mmap_open(npy_path, RCRE_IO_HINT_DEFAULT, &map));
    RCRE_Tensor view = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_view_npy(&map, &view));
    ASSERT_TRUE(view.is_view);
    ASSERT_UINT_EQ(3, view.ndim);
    ASSERT_UINT_EQ(8, view.shape[0]);
    ASSERT_UINT_EQ(16, view.shape[1]);
    ASSERT_UINT_EQ(3, view.shape[2]);
    ASSERT_UINT_EQ(0, (uintptr_t)view.data % 64); /* Strict 64-byte SIMD alignment */
    ASSERT_MEM_EQ(orig.data, view.data, rcre_tensor_nbytes(&orig));
    rcre_io_mmap_close(&map);

    /* Verify arena copy load */
    RCRE_Tensor loaded = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_load_npy(npy_path, arena, &loaded));
    ASSERT_FALSE(loaded.is_view);
    ASSERT_UINT_EQ(0, (uintptr_t)loaded.data % 64);
    ASSERT_MEM_EQ(orig.data, loaded.data, rcre_tensor_nbytes(&orig));
    remove(npy_path);

    /* 2. 1D I32 & 2D U8 Round-Trip */
    size_t shape1d[1] = { 33 };
    RCRE_Tensor t_i32 = rcre_tensor_create(shape1d, 1, RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);
    int32_t* idata = (int32_t*)t_i32.data;
    for (size_t i = 0; i < 33; ++i) idata[i] = (int32_t)(i * 100 - 500);
    ASSERT_INT_EQ(RCRE_OK, rcre_io_save_npy(&t_i32, npy_path));
    RCRE_Tensor l_i32 = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_load_npy(npy_path, arena, &l_i32));
    ASSERT_INT_EQ(RCRE_DTYPE_I32, l_i32.dtype);
    ASSERT_UINT_EQ(1, l_i32.ndim);
    ASSERT_UINT_EQ(33, l_i32.shape[0]);
    ASSERT_MEM_EQ(t_i32.data, l_i32.data, 33 * sizeof(int32_t));
    remove(npy_path);

    /* 3. Adversarial & Corrupt .npy Validation */
    uint8_t bad_npy[128];
    memset(bad_npy, 0, sizeof(bad_npy));
    RCRE_MappedFile bad_map = {0};
    RCRE_Tensor bad_view = {0};

    /* (a) Wrong magic */
    memcpy(bad_npy, "BADMAG", 6);
    ASSERT_INT_EQ(RCRE_OK, rcre_io_mmap_from_buffer(bad_npy, sizeof(bad_npy), &bad_map));
    ASSERT_INT_EQ(RCRE_ERR_IO, rcre_io_view_npy(&bad_map, &bad_view));

    /* (b) Fortran order = True rejected */
    const char* fort_dict = "{'descr': '<f4', 'fortran_order': True, 'shape': (4,), }       \n";
    memcpy(bad_npy, "\x93NUMPY\x01\x00", 8);
    uint16_t fort_len = (uint16_t)strlen(fort_dict);
    bad_npy[8] = (uint8_t)(fort_len & 0xFF);
    bad_npy[9] = (uint8_t)(fort_len >> 8);
    memcpy(bad_npy + 10, fort_dict, fort_len);
    ASSERT_INT_EQ(RCRE_OK, rcre_io_mmap_from_buffer(bad_npy, 10 + fort_len + 16, &bad_map));
    ASSERT_INT_EQ(RCRE_ERR_UNSUPPORTED_OP, rcre_io_view_npy(&bad_map, &bad_view));

    /* (c) Truncated payload (shape claims 1000 floats, only 16 bytes follow) */
    const char* trunc_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (1000,), }   \n";
    uint16_t trunc_len = (uint16_t)strlen(trunc_dict);
    bad_npy[8] = (uint8_t)(trunc_len & 0xFF);
    bad_npy[9] = (uint8_t)(trunc_len >> 8);
    memcpy(bad_npy + 10, trunc_dict, trunc_len);
    ASSERT_INT_EQ(RCRE_OK, rcre_io_mmap_from_buffer(bad_npy, 10 + trunc_len + 16, &bad_map));
    ASSERT_INT_EQ(RCRE_ERR_IO, rcre_io_view_npy(&bad_map, &bad_view));

    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* 5. Multi-Tensor Safetensors (.safetensors) Archive Engine (IO-REQ-04)     */
/* ========================================================================= */

static void test_io_safetensors_archive_and_adversarial(void) {
    RCRE_Arena* arena = rcre_arena_create(8 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Create 3 tensors of different ranks and dtypes:
     *  - "weight": [16, 16] F32 (1024 bytes, multiple of 64)
     *  - "bias":   [16]     F32 (64 bytes, multiple of 64)
     *  - "tokens": [7]      U8  (7 bytes, intentionally non-multiple of 64)
     *  - "tail":   [4]      I32 (16 bytes, follows 7-byte tensor -> unaligned offset!) */
    size_t s_w[2] = { 16, 16 };
    size_t s_b[1] = { 16 };
    size_t s_u[1] = { 7 };
    size_t s_t[1] = { 4 };

    RCRE_Tensor tensors[4];
    tensors[0] = rcre_tensor_create(s_w, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    tensors[1] = rcre_tensor_create(s_b, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    tensors[2] = rcre_tensor_create(s_u, 1, RCRE_DTYPE_U8,  RCRE_LAYOUT_TABULAR, arena);
    tensors[3] = rcre_tensor_create(s_t, 1, RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);

    for (size_t i = 0; i < 256; ++i) rcre_tensor_f32_data(&tensors[0])[i] = (float)i * 0.5f;
    for (size_t i = 0; i < 16; ++i)  rcre_tensor_f32_data(&tensors[1])[i] = -(float)i;
    for (size_t i = 0; i < 7; ++i)   ((uint8_t*)tensors[2].data)[i] = (uint8_t)(i + 10);
    for (size_t i = 0; i < 4; ++i)   ((int32_t*)tensors[3].data)[i] = (int32_t)(i * 1000);

    const char* names[4] = { "weight", "bias", "tokens", "tail" };
    const char* st_path = "/tmp/rcre_test_model.safetensors";

    int rc = rcre_io_safetensors_save(st_path, names, tensors, 4, arena);
    ASSERT_INT_EQ(RCRE_OK, rc);

    RCRE_SafeTensorsArchive arch = {0};
    rc = rcre_io_safetensors_open(st_path, &arch);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(4, arch.num_tensors);
    /* Verify RCRE padded (8 + header_nbytes) to an exact 64-byte boundary */
    ASSERT_UINT_EQ(0, arch.data_base_offset % 64);

    /* "weight" (offset 0) and "bias" (offset 1024) are 64B-aligned -> Zero-Copy Views! */
    RCRE_Tensor v_weight = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_safetensors_get_tensor(&arch, "weight", nullptr, &v_weight));
    ASSERT_TRUE(v_weight.is_view);
    ASSERT_UINT_EQ(0, (uintptr_t)v_weight.data % 64);
    ASSERT_MEM_EQ(tensors[0].data, v_weight.data, 256 * sizeof(float));

    RCRE_Tensor v_bias = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_safetensors_get_tensor(&arch, "bias", nullptr, &v_bias));
    ASSERT_TRUE(v_bias.is_view);
    ASSERT_UINT_EQ(0, (uintptr_t)v_bias.data % 64);
    ASSERT_MEM_EQ(tensors[1].data, v_bias.data, 16 * sizeof(float));

    /* "tail" sits at relative offset 1024 + 64 + 7 = 1095 (not divisible by 64):
     *  - without fallback_arena -> rejected to protect SIMD kernels from unaligned ptr
     *  - with fallback_arena    -> automatically copied to 64B-aligned arena buffer! */
    RCRE_Tensor v_tail = {0};
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_io_safetensors_get_tensor(&arch, "tail", nullptr, &v_tail));
    ASSERT_INT_EQ(RCRE_OK, rcre_io_safetensors_get_tensor(&arch, "tail", arena, &v_tail));
    ASSERT_FALSE(v_tail.is_view);
    ASSERT_UINT_EQ(0, (uintptr_t)v_tail.data % 64);
    ASSERT_MEM_EQ(tensors[3].data, v_tail.data, 4 * sizeof(int32_t));

    /* Missing tensor name */
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_io_safetensors_get_tensor(&arch, "nonexistent", arena, &v_tail));

    rcre_io_safetensors_close(&arch);
    remove(st_path);

    /* Adversarial .safetensors tests */
    /* (a) Overlapping data_offsets between two tensors */
    const char* overlap_json =
        "{\"__metadata__\":{\"format\":\"pt\"},"
        "\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]},"
        "\"b\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[8,24]}}";
    uint8_t adv_buf[512];
    memset(adv_buf, 0, sizeof(adv_buf));
    uint64_t jlen = (uint64_t)strlen(overlap_json);
    memcpy(adv_buf, &jlen, 8);
    memcpy(adv_buf + 8, overlap_json, (size_t)jlen);

    RCRE_SafeTensorsArchive bad_arch = {0};
    ASSERT_INT_EQ(RCRE_ERR_IO, rcre_io_safetensors_open_from_buffer(adv_buf, 8 + (size_t)jlen + 32, &bad_arch));

    /* (b) Shape byte-count mismatch (shape=[4] F32 = 16 bytes, but offsets=[0, 32]) */
    const char* mismatch_json =
        "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,32]}}";
    jlen = (uint64_t)strlen(mismatch_json);
    memcpy(adv_buf, &jlen, 8);
    memcpy(adv_buf + 8, mismatch_json, (size_t)jlen);
    ASSERT_INT_EQ(RCRE_ERR_IO, rcre_io_safetensors_open_from_buffer(adv_buf, 8 + (size_t)jlen + 32, &bad_arch));

    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* 6. SIMD Byte-Shuffle, Block Compression & Ping-Pong Streamer (IO-REQ-05)  */
/* ========================================================================= */

static void test_io_simd_byte_shuffle_compression_and_streamer(void) {
    RCRE_Arena* arena = rcre_arena_create(8 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 1. SIMD Byte-Shuffle & Unshuffle across odd and vector-aligned lengths */
    const size_t counts[6] = { 1, 3, 4, 16, 19, 257 };
    for (size_t c_idx = 0; c_idx < 6; ++c_idx) {
        size_t n = counts[c_idx];
        RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
        uint32_t* orig = (uint32_t*)rcre_arena_alloc(arena, n * sizeof(uint32_t), 64);
        uint8_t*  shuf = (uint8_t*)rcre_arena_alloc(arena, n * sizeof(uint32_t), 64);
        uint32_t* rec  = (uint32_t*)rcre_arena_alloc(arena, n * sizeof(uint32_t), 64);

        for (size_t i = 0; i < n; ++i) {
            orig[i] = 0x01020304U + (uint32_t)(i * 0x01010101U);
        }

        rcre_io_byte_shuffle_32(orig, shuf, n);

        /* Verify stream 0 holds LSBs and stream 3 holds MSBs */
        const uint8_t* raw_orig = (const uint8_t*)orig;
        for (size_t i = 0; i < n; ++i) {
            ASSERT_UINT_EQ(raw_orig[i * 4 + 0], shuf[0 * n + i]);
            ASSERT_UINT_EQ(raw_orig[i * 4 + 1], shuf[1 * n + i]);
            ASSERT_UINT_EQ(raw_orig[i * 4 + 2], shuf[2 * n + i]);
            ASSERT_UINT_EQ(raw_orig[i * 4 + 3], shuf[3 * n + i]);
        }

        rcre_io_byte_unshuffle_32(shuf, rec, n);
        ASSERT_MEM_EQ(orig, rec, n * sizeof(uint32_t));
        rcre_arena_temp_end(tmp);
    }

    /* 2. Byte-Shuffled Tensor Block Compression ("RCRZ") */
    size_t c_shape[2] = { 64, 64 };
    RCRE_Tensor t_src = rcre_tensor_create(c_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* sd = rcre_tensor_f32_data(&t_src);
    /* Populate with typical narrow-range neural activations (0.5 .. 0.99)
     * sharing identical IEEE-754 exponent bytes + sparse zeros */
    for (size_t i = 0; i < t_src.numel; ++i) {
        sd[i] = (i % 4 == 0) ? 0.0f : (0.5f + (float)(i % 16) * (1.0f / 64.0f));
    }

    RCRE_Slice compressed = rcre_slice_empty();
    ASSERT_INT_EQ(RCRE_OK, rcre_io_compress_tensor(&t_src, arena, &compressed));
    ASSERT_NOT_NULL(compressed.data);
    /* Expect strong compression (< 50% of raw 16,384 bytes) */
    ASSERT_TRUE(compressed.len < rcre_tensor_nbytes(&t_src) / 2);

    RCRE_Tensor t_dec = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_decompress_tensor(compressed, arena, &t_dec));
    ASSERT_UINT_EQ(2, t_dec.ndim);
    ASSERT_UINT_EQ(64, t_dec.shape[0]);
    ASSERT_UINT_EQ(64, t_dec.shape[1]);
    ASSERT_UINT_EQ(0, (uintptr_t)t_dec.data % 64);
    /* 0-ULP bit-exact lossless roundtrip */
    ASSERT_MEM_EQ(t_src.data, t_dec.data, rcre_tensor_nbytes(&t_src));

    /* 3. Out-of-Core Ping-Pong Streamer over .npy */
    size_t s_shape[2] = { 105, 8 };
    RCRE_Tensor stream_src = rcre_tensor_create(s_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* ssd = rcre_tensor_f32_data(&stream_src);
    for (size_t i = 0; i < stream_src.numel; ++i) {
        ssd[i] = (float)i;
    }

    const char* stream_path = "/tmp/rcre_test_stream.npy";
    ASSERT_INT_EQ(RCRE_OK, rcre_io_save_npy(&stream_src, stream_path));

    RCRE_TensorStream stream = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_stream_open_npy(stream_path, 32, arena, &stream));

    RCRE_Tensor batch = {0};
    size_t batches_seen = 0;
    size_t rows_accum = 0;
    const void* prev_buf_ptr = nullptr;

    while (rcre_io_stream_next(&stream, &batch)) {
        ASSERT_TRUE(batch.is_view);
        ASSERT_UINT_EQ(0, (uintptr_t)batch.data % 64);
        /* Verify ping-pong buffer pointer alternates on every step */
        if (prev_buf_ptr != nullptr) {
            ASSERT_TRUE(batch.data != prev_buf_ptr);
        }
        prev_buf_ptr = batch.data;

        size_t expected_rows = (batches_seen < 3) ? 32 : 9;
        ASSERT_UINT_EQ(expected_rows, batch.shape[0]);
        ASSERT_UINT_EQ(8, batch.shape[1]);
        ASSERT_MEM_EQ(ssd + rows_accum * 8, batch.data, expected_rows * 8 * sizeof(float));

        rows_accum += expected_rows;
        batches_seen++;
    }
    ASSERT_UINT_EQ(4, batches_seen);
    ASSERT_UINT_EQ(105, rows_accum);

    /* Reset stream for Epoch 2 */
    rcre_io_stream_reset(&stream);
    ASSERT_TRUE(rcre_io_stream_next(&stream, &batch));
    ASSERT_UINT_EQ(32, batch.shape[0]);
    ASSERT_FLOAT_NEAR(0.0f, rcre_tensor_f32_data(&batch)[0], 0.0f);

    rcre_io_stream_close(&stream);
    remove(stream_path);
    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* 7. Zero-Copy Cross-Framework C ABI Bridges: DLPack & Arrow (IO-REQ-06)    */
/* ========================================================================= */

static void test_io_dlpack_and_arrow_zero_copy_bridges(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 1. DLPack v1.0 Zero-Copy Export & Import (including strided transpose view) */
    size_t shape2d[2] = { 4, 8 };
    RCRE_Tensor mat = rcre_tensor_create(shape2d, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    for (size_t i = 0; i < 32; ++i) rcre_tensor_f32_data(&mat)[i] = (float)i;

    RCRE_Tensor mat_T = rcre_tensor_transpose(&mat, 0, 1); /* shape [8, 4], strides [1, 8] */
    ASSERT_FALSE(rcre_tensor_is_contiguous(&mat_T));

    RCRE_DLTensor dl = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_to_dlpack(&mat_T, arena, &dl));
    ASSERT_TRUE(dl.data == mat.data); /* Zero-copy pointer identity */
    ASSERT_INT_EQ(RCRE_DL_DEVICE_CPU, dl.device_type);
    ASSERT_INT_EQ(2, dl.ndim);
    ASSERT_UINT_EQ(RCRE_DL_FLOAT, dl.dtype_code);
    ASSERT_UINT_EQ(32, dl.dtype_bits);
    ASSERT_INT_EQ(8, (int)dl.shape[0]);
    ASSERT_INT_EQ(4, (int)dl.shape[1]);
    ASSERT_INT_EQ(1, (int)dl.strides[0]);
    ASSERT_INT_EQ(8, (int)dl.strides[1]);

    RCRE_Tensor imported_dl = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_from_dlpack(&dl, &imported_dl));
    ASSERT_TRUE(imported_dl.is_view);
    ASSERT_TRUE(imported_dl.data == mat.data);
    ASSERT_UINT_EQ(8, imported_dl.shape[0]);
    ASSERT_UINT_EQ(4, imported_dl.shape[1]);
    ASSERT_INT_EQ(1, (int)imported_dl.strides[0]);
    ASSERT_INT_EQ(8, (int)imported_dl.strides[1]);

    /* Reject GPU device type in CPU runtime */
    dl.device_type = RCRE_DL_DEVICE_CUDA;
    ASSERT_INT_EQ(RCRE_ERR_UNSUPPORTED_OP, rcre_io_from_dlpack(&dl, &imported_dl));

    /* 2. Apache Arrow C Data Interface Zero-Copy Export & Import */
    size_t shape1d[1] = { 16 };
    RCRE_Tensor col = rcre_tensor_create(shape1d, 1, RCRE_DTYPE_I32, RCRE_LAYOUT_TABULAR, arena);
    col.label = "feature_id";
    int32_t* cdata = (int32_t*)col.data;
    for (int i = 0; i < 16; ++i) cdata[i] = i * 10;

    RCRE_ArrowSchema schema = {0};
    RCRE_ArrowArray  arr    = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_to_arrow(&col, arena, &schema, &arr));
    ASSERT_STR_EQ("i", schema.format);
    ASSERT_STR_EQ("feature_id", schema.name);
    ASSERT_INT_EQ(16, (int)arr.length);
    ASSERT_INT_EQ(0, (int)arr.null_count);
    ASSERT_TRUE(arr.buffers[1] == col.data); /* Zero-copy buffer identity */

    /* Test importing with non-zero Arrow slice offset */
    arr.offset = 4;
    arr.length = 12;
    RCRE_Tensor arrow_view = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_from_arrow(&schema, &arr, &arrow_view));
    ASSERT_TRUE(arrow_view.is_view);
    ASSERT_INT_EQ(RCRE_DTYPE_I32, arrow_view.dtype);
    ASSERT_UINT_EQ(12, arrow_view.shape[0]);
    ASSERT_INT_EQ(40, ((const int32_t*)arrow_view.data)[0]);
    ASSERT_INT_EQ(150, ((const int32_t*)arrow_view.data)[11]);

    schema.release(&schema);
    arr.release(&arr);
    ASSERT_NULL(schema.release);
    ASSERT_NULL(arr.release);

    /* 3. DLPack & Arrow Half-Precision (F16 & BF16) Bridges */
    size_t shape16[1] = { 8 };
    RCRE_Tensor t_f16 = rcre_tensor_create(shape16, 1, RCRE_DTYPE_F16, RCRE_LAYOUT_TABULAR, arena);
    RCRE_DLTensor dl_f16 = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_to_dlpack(&t_f16, arena, &dl_f16));
    ASSERT_UINT_EQ(RCRE_DL_FLOAT, dl_f16.dtype_code);
    ASSERT_UINT_EQ(16, dl_f16.dtype_bits);
    RCRE_Tensor imp_f16 = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_from_dlpack(&dl_f16, &imp_f16));
    ASSERT_INT_EQ(RCRE_DTYPE_F16, imp_f16.dtype);

    RCRE_Tensor t_bf16 = rcre_tensor_create(shape16, 1, RCRE_DTYPE_BF16, RCRE_LAYOUT_TABULAR, arena);
    RCRE_DLTensor dl_bf16 = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_to_dlpack(&t_bf16, arena, &dl_bf16));
    ASSERT_UINT_EQ(4, dl_bf16.dtype_code); /* kDLBfloat = 4 */
    ASSERT_UINT_EQ(16, dl_bf16.dtype_bits);
    RCRE_Tensor imp_bf16 = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_from_dlpack(&dl_bf16, &imp_bf16));
    ASSERT_INT_EQ(RCRE_DTYPE_BF16, imp_bf16.dtype);

    RCRE_ArrowSchema schema_f16 = {0};
    RCRE_ArrowArray  arr_f16    = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_to_arrow(&t_f16, arena, &schema_f16, &arr_f16));
    ASSERT_STR_EQ("e", schema_f16.format);
    RCRE_Tensor arrow_f16 = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_from_arrow(&schema_f16, &arr_f16, &arrow_f16));
    ASSERT_INT_EQ(RCRE_DTYPE_F16, arrow_f16.dtype);
    schema_f16.release(&schema_f16);
    arr_f16.release(&arr_f16);

    RCRE_ArrowSchema schema_bf16 = {0};
    RCRE_ArrowArray  arr_bf16    = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_to_arrow(&t_bf16, arena, &schema_bf16, &arr_bf16));
    ASSERT_STR_EQ("b", schema_bf16.format);
    RCRE_Tensor arrow_bf16 = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_from_arrow(&schema_bf16, &arr_bf16, &arrow_bf16));
    ASSERT_INT_EQ(RCRE_DTYPE_BF16, arrow_bf16.dtype);
    schema_bf16.release(&schema_bf16);
    arr_bf16.release(&arr_bf16);

    /* 4. Zero-Copy GPT Model .safetensors Save, Bind & KV-Cache Parity */
    RCRE_GPTConfig gpt_cfg = {
        .vocab_size = 16,
        .seq_len    = 8,
        .embed_dim  = 16,
        .num_heads  = 2,
        .head_dim   = 8,
        .num_layers = 1,
        .mlp_dim    = 32
    };
    RCRE_GPTModel* gpt_orig = rcre_gpt_create(gpt_cfg, arena);
    ASSERT_NOT_NULL(gpt_orig);
    uint64_t rng_seed = 2026ULL;
    rcre_gpt_init_weights(gpt_orig, &rng_seed);

    const char* gpt_st_path = "/tmp/rcre_test_gpt_zero_copy.safetensors";
    ASSERT_INT_EQ(RCRE_OK, rcre_gpt_save_safetensors(gpt_orig, gpt_st_path, arena));

    RCRE_GPTModel* gpt_mapped = rcre_gpt_create(gpt_cfg, arena);
    ASSERT_NOT_NULL(gpt_mapped);
    RCRE_SafeTensorsArchive gpt_arch = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_gpt_load_safetensors_zero_copy(gpt_mapped, gpt_st_path, &gpt_arch));

    /* Verify wte pointer points directly into mapped file pages (0 copies!) */
    ASSERT_TRUE((const uint8_t*)gpt_mapped->wte == gpt_arch.map.data + gpt_arch.data_base_offset);
    ASSERT_UINT_EQ(0, (uintptr_t)gpt_mapped->wte % 64);

    /* Compare KV-cached token logits between original and zero-copy mapped model */
    RCRE_KVCache* kv1 = rcre_kv_cache_create(gpt_cfg, arena);
    RCRE_KVCache* kv2 = rcre_kv_cache_create(gpt_cfg, arena);
    float logits1[16] = {0};
    float logits2[16] = {0};
    ASSERT_TRUE(rcre_gpt_forward_token(gpt_orig,   kv1, 3, 0, logits1));
    ASSERT_TRUE(rcre_gpt_forward_token(gpt_mapped, kv2, 3, 0, logits2));
    ASSERT_MEM_EQ(logits1, logits2, sizeof(logits1));

    rcre_io_safetensors_close(&gpt_arch);
    remove(gpt_st_path);
    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* 8. Half-Precision (IEEE FP16 & Google BF16) SIMD & Dequantization         */
/* ========================================================================= */

static void test_io_half_precision_fp16_bf16_simd_and_dequantization(void) {
    RCRE_Arena* arena = rcre_arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 1. Scalar & SIMD round-trip conversions */
    const size_t N = 1024;
    float* orig_f32 = (float*)rcre_arena_alloc(arena, N * sizeof(float), 64);
    uint16_t* fp16_buf = (uint16_t*)rcre_arena_alloc(arena, N * sizeof(uint16_t), 64);
    uint16_t* bf16_buf = (uint16_t*)rcre_arena_alloc(arena, N * sizeof(uint16_t), 64);
    float* recon_fp16 = (float*)rcre_arena_alloc(arena, N * sizeof(float), 64);
    float* recon_bf16 = (float*)rcre_arena_alloc(arena, N * sizeof(float), 64);

    for (size_t i = 0; i < N; ++i) {
        orig_f32[i] = ((float)i - 512.0f) * 0.125f;
    }
    orig_f32[0] = 0.0f;
    orig_f32[1] = -0.0f;
    orig_f32[2] = 1.0f;
    orig_f32[3] = -1.0f;
    orig_f32[4] = 65504.0f; /* Max finite FP16 */
    orig_f32[5] = -65504.0f;

    /* Test FP16 conversion */
    rcre_f32_to_f16(orig_f32, fp16_buf, N);
    rcre_f16_to_f32(fp16_buf, recon_fp16, N);

    ASSERT_FLOAT_NEAR(0.0f, recon_fp16[0], 0.0f);
    ASSERT_FLOAT_NEAR(0.0f, recon_fp16[1], 0.0f);
    ASSERT_TRUE(signbit(recon_fp16[1]) != 0);
    ASSERT_FLOAT_NEAR(1.0f, recon_fp16[2], 0.0f);
    ASSERT_FLOAT_NEAR(-1.0f, recon_fp16[3], 0.0f);
    ASSERT_FLOAT_NEAR(65504.0f, recon_fp16[4], 0.0f);
    ASSERT_FLOAT_NEAR(-65504.0f, recon_fp16[5], 0.0f);

    for (size_t i = 6; i < N; ++i) {
        float expected = orig_f32[i];
        float actual = recon_fp16[i];
        float diff = fabsf(actual - expected);
        float tol = fabsf(expected) * 0.002f + 1e-4f;
        ASSERT_TRUE(diff <= tol);
    }

    /* Test BF16 conversion */
    rcre_f32_to_bf16(orig_f32, bf16_buf, N);
    rcre_bf16_to_f32(bf16_buf, recon_bf16, N);

    ASSERT_FLOAT_NEAR(0.0f, recon_bf16[0], 0.0f);
    ASSERT_FLOAT_NEAR(0.0f, recon_bf16[1], 0.0f);
    ASSERT_TRUE(signbit(recon_bf16[1]) != 0);
    ASSERT_FLOAT_NEAR(1.0f, recon_bf16[2], 0.0f);
    ASSERT_FLOAT_NEAR(-1.0f, recon_bf16[3], 0.0f);

    for (size_t i = 6; i < N; ++i) {
        float expected = orig_f32[i];
        float actual = recon_bf16[i];
        float diff = fabsf(actual - expected);
        float tol = fabsf(expected) * 0.01f + 1e-3f; /* BF16 has 7 mantissa bits */
        ASSERT_TRUE(diff <= tol);
    }

    /* 2. rcre_tensor_to_f32 tests */
    size_t shape[2] = { 16, 16 };
    RCRE_Tensor t_f16 = rcre_tensor_create(shape, 2, RCRE_DTYPE_F16, RCRE_LAYOUT_TABULAR, arena);
    uint16_t* f16_raw = (uint16_t*)t_f16.data;
    for (size_t i = 0; i < 256; ++i) {
        f16_raw[i] = fp16_buf[i];
    }
    RCRE_Tensor deq_f16 = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_tensor_to_f32(&t_f16, arena, &deq_f16));
    ASSERT_INT_EQ(RCRE_DTYPE_F32, deq_f16.dtype);
    ASSERT_UINT_EQ(2, deq_f16.ndim);
    ASSERT_UINT_EQ(16, deq_f16.shape[0]);
    ASSERT_UINT_EQ(16, deq_f16.shape[1]);
    const float* deq_f16_ptr = rcre_tensor_f32_data(&deq_f16);
    ASSERT_FLOAT_NEAR(0.0f, deq_f16_ptr[0], 0.0f);
    ASSERT_FLOAT_NEAR(1.0f, deq_f16_ptr[2], 0.0f);

    RCRE_Tensor t_bf16 = rcre_tensor_create(shape, 2, RCRE_DTYPE_BF16, RCRE_LAYOUT_TABULAR, arena);
    uint16_t* bf16_raw = (uint16_t*)t_bf16.data;
    for (size_t i = 0; i < 256; ++i) {
        bf16_raw[i] = bf16_buf[i];
    }
    RCRE_Tensor deq_bf16 = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_tensor_to_f32(&t_bf16, arena, &deq_bf16));
    ASSERT_INT_EQ(RCRE_DTYPE_F32, deq_bf16.dtype);
    ASSERT_UINT_EQ(2, deq_bf16.ndim);
    ASSERT_UINT_EQ(16, deq_bf16.shape[0]);
    ASSERT_UINT_EQ(16, deq_bf16.shape[1]);
    const float* deq_bf16_ptr = rcre_tensor_f32_data(&deq_bf16);
    ASSERT_FLOAT_NEAR(0.0f, deq_bf16_ptr[0], 0.0f);
    ASSERT_FLOAT_NEAR(1.0f, deq_bf16_ptr[2], 0.0f);

    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* 9. Safetensors FP16/BF16 Storage, Dequantization & Chunk Slicing          */
/* ========================================================================= */

static void test_io_safetensors_f16_bf16_and_chunked_slicing(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Create F16 and BF16 tensors */
    size_t f16_shape[2] = { 32, 16 }; /* 512 elements */
    size_t bf16_shape[2] = { 64, 8 };  /* 512 elements */

    RCRE_Tensor t_f16 = rcre_tensor_create(f16_shape, 2, RCRE_DTYPE_F16, RCRE_LAYOUT_TABULAR, arena);
    RCRE_Tensor t_bf16 = rcre_tensor_create(bf16_shape, 2, RCRE_DTYPE_BF16, RCRE_LAYOUT_TABULAR, arena);

    float orig_vals[512];
    for (size_t i = 0; i < 512; ++i) {
        orig_vals[i] = (float)i * 0.25f + 1.0f;
    }
    rcre_f32_to_f16(orig_vals, (uint16_t*)t_f16.data, 512);
    rcre_f32_to_bf16(orig_vals, (uint16_t*)t_bf16.data, 512);

    const char* names[2] = { "weights.fp16", "weights.bf16" };
    RCRE_Tensor tensors[2] = { t_f16, t_bf16 };

    const char* st_path = "/tmp/rcre_test_half_prec.safetensors";
    int rc = rcre_io_safetensors_save(st_path, names, tensors, 2, arena);
    ASSERT_INT_EQ(RCRE_OK, rc);

    /* Open archive */
    RCRE_SafeTensorsArchive arch = {0};
    rc = rcre_io_safetensors_open(st_path, &arch);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_UINT_EQ(2, arch.num_tensors);

    /* Retrieve tensor views and verify dtypes */
    RCRE_Tensor v_f16 = {0};
    rc = rcre_io_safetensors_get_tensor(&arch, "weights.fp16", nullptr, &v_f16);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_INT_EQ(RCRE_DTYPE_F16, v_f16.dtype);
    ASSERT_UINT_EQ(32, v_f16.shape[0]);
    ASSERT_UINT_EQ(16, v_f16.shape[1]);

    RCRE_Tensor v_bf16 = {0};
    rc = rcre_io_safetensors_get_tensor(&arch, "weights.bf16", nullptr, &v_bf16);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_INT_EQ(RCRE_DTYPE_BF16, v_bf16.dtype);
    ASSERT_UINT_EQ(64, v_bf16.shape[0]);
    ASSERT_UINT_EQ(8, v_bf16.shape[1]);

    /* Test full dequantization to F32 */
    RCRE_Tensor out_f16_f32 = {0};
    rc = rcre_io_safetensors_get_tensor_f32(&arch, "weights.fp16", arena, &out_f16_f32);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_INT_EQ(RCRE_DTYPE_F32, out_f16_f32.dtype);
    ASSERT_UINT_EQ(32, out_f16_f32.shape[0]);
    ASSERT_UINT_EQ(16, out_f16_f32.shape[1]);
    const float* f16_f32_ptr = rcre_tensor_f32_data(&out_f16_f32);
    ASSERT_FLOAT_NEAR(orig_vals[0], f16_f32_ptr[0], 0.01f);
    ASSERT_FLOAT_NEAR(orig_vals[100], f16_f32_ptr[100], 0.5f);

    RCRE_Tensor out_bf16_f32 = {0};
    rc = rcre_io_safetensors_get_tensor_f32(&arch, "weights.bf16", arena, &out_bf16_f32);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_INT_EQ(RCRE_DTYPE_F32, out_bf16_f32.dtype);
    ASSERT_UINT_EQ(64, out_bf16_f32.shape[0]);
    ASSERT_UINT_EQ(8, out_bf16_f32.shape[1]);
    const float* bf16_f32_ptr = rcre_tensor_f32_data(&out_bf16_f32);
    ASSERT_FLOAT_NEAR(orig_vals[0], bf16_f32_ptr[0], 0.05f);
    ASSERT_FLOAT_NEAR(orig_vals[100], bf16_f32_ptr[100], 0.5f);

    /* Test chunked row slicing with on-the-fly dequantization */
    RCRE_Tensor slice = {0};
    rc = rcre_io_safetensors_get_tensor_slice_f32(&arch, "weights.fp16", 10, 4, arena, &slice);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_INT_EQ(RCRE_DTYPE_F32, slice.dtype);
    ASSERT_UINT_EQ(4, slice.shape[0]);
    ASSERT_UINT_EQ(16, slice.shape[1]);
    const float* slice_ptr = rcre_tensor_f32_data(&slice);
    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 0; c < 16; ++c) {
            float exp_val = orig_vals[(10 + r) * 16 + c];
            float act_val = slice_ptr[r * 16 + c];
            ASSERT_TRUE(fabsf(act_val - exp_val) <= 0.05f);
        }
    }

    /* Test boundary and error handling */
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_io_safetensors_get_tensor_slice_f32(&arch, "weights.fp16", 30, 5, arena, &slice)); /* 30+5 > 32 */
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_io_safetensors_get_tensor_slice_f32(&arch, "non_existent", 0, 1, arena, &slice));
    ASSERT_INT_EQ(RCRE_ERR_INVALID_ARG, rcre_io_safetensors_get_tensor_slice_f32(&arch, "weights.fp16", 0, 0, arena, &slice));

    rcre_io_safetensors_close(&arch);
    remove(st_path);
    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* 10. SIMD 16-Bit Byte-Shuffle & Lossless RCRZ Frame Compression            */
/* ========================================================================= */

static void test_io_simd_16bit_byte_shuffle_and_rcrz_compression(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    const size_t N = 2048;
    uint16_t* src = (uint16_t*)rcre_arena_alloc(arena, N * sizeof(uint16_t), 64);
    uint16_t* shuffled = (uint16_t*)rcre_arena_alloc(arena, N * sizeof(uint16_t), 64);
    uint16_t* unshuffled = (uint16_t*)rcre_arena_alloc(arena, N * sizeof(uint16_t), 64);

    for (size_t i = 0; i < N; ++i) {
        src[i] = (uint16_t)(i * 31 + 7);
    }

    /* 1. 16-bit Byte Shuffle & Unshuffle parity */
    rcre_io_byte_shuffle_16(src, shuffled, N);
    rcre_io_byte_unshuffle_16(shuffled, unshuffled, N);
    ASSERT_MEM_EQ(src, unshuffled, N * sizeof(uint16_t));

    /* 2. RCRZ Compression round-trip for F16 tensor */
    size_t shape[2] = { 64, 32 }; /* 2048 elements */
    RCRE_Tensor t_f16 = rcre_tensor_create(shape, 2, RCRE_DTYPE_F16, RCRE_LAYOUT_TABULAR, arena);
    memcpy(t_f16.data, src, N * sizeof(uint16_t));

    RCRE_Slice frame_f16 = {0};
    int rc = rcre_io_compress_tensor(&t_f16, arena, &frame_f16);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_NOT_NULL(frame_f16.data);
    ASSERT_TRUE(frame_f16.len > 64);

    RCRE_Tensor decomp_f16 = {0};
    rc = rcre_io_decompress_tensor(frame_f16, arena, &decomp_f16);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_INT_EQ(RCRE_DTYPE_F16, decomp_f16.dtype);
    ASSERT_UINT_EQ(2, decomp_f16.ndim);
    ASSERT_UINT_EQ(64, decomp_f16.shape[0]);
    ASSERT_UINT_EQ(32, decomp_f16.shape[1]);
    ASSERT_MEM_EQ(t_f16.data, decomp_f16.data, N * sizeof(uint16_t));

    /* 3. RCRZ Compression round-trip for BF16 tensor */
    RCRE_Tensor t_bf16 = rcre_tensor_create(shape, 2, RCRE_DTYPE_BF16, RCRE_LAYOUT_TABULAR, arena);
    memcpy(t_bf16.data, src, N * sizeof(uint16_t));

    RCRE_Slice frame_bf16 = {0};
    rc = rcre_io_compress_tensor(&t_bf16, arena, &frame_bf16);
    ASSERT_INT_EQ(RCRE_OK, rc);

    RCRE_Tensor decomp_bf16 = {0};
    rc = rcre_io_decompress_tensor(frame_bf16, arena, &decomp_bf16);
    ASSERT_INT_EQ(RCRE_OK, rc);
    ASSERT_INT_EQ(RCRE_DTYPE_BF16, decomp_bf16.dtype);
    ASSERT_MEM_EQ(t_bf16.data, decomp_bf16.data, N * sizeof(uint16_t));

    rcre_arena_destroy(arena);
}

/* ========================================================================= */
/* 11. Linux io_uring Probe & Double-Buffered Asynchronous Streaming         */
/* ========================================================================= */

static void test_io_uring_probe_and_async_streaming(void) {
    RCRE_Arena* arena = rcre_arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 1. Verify io_uring probing returns a valid boolean */
    bool has_uring = rcre_io_is_io_uring_available();
#if defined(__linux__)
    (void)has_uring; /* May be true or false depending on kernel config */
#else
    ASSERT_FALSE(has_uring); /* Non-Linux always returns false */
#endif

    /* 2. Create test .npy file */
    const char* path = "/tmp/rcre_test_async_stream.npy";
    size_t full_shape[2] = { 128, 16 };
    RCRE_Tensor full_mat = rcre_tensor_create(full_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* fptr = rcre_tensor_f32_data(&full_mat);
    for (size_t i = 0; i < 128 * 16; ++i) {
        fptr[i] = (float)i;
    }
    ASSERT_INT_EQ(RCRE_OK, rcre_io_save_npy(&full_mat, path));

    /* 3. Open stream with batch_rows = 32 */
    RCRE_TensorStream stream = {0};
    ASSERT_INT_EQ(RCRE_OK, rcre_io_stream_open_npy(path, 32, arena, &stream));
    ASSERT_FALSE(stream.async_enabled);

    /* Enable async streaming */
    ASSERT_INT_EQ(RCRE_OK, rcre_io_stream_enable_async(&stream));
    ASSERT_TRUE(stream.async_enabled);

    size_t batches_read = 0;
    size_t rows_read = 0;
    RCRE_Tensor batch = {0};
    while (rcre_io_stream_next(&stream, &batch)) {
        ASSERT_INT_EQ(RCRE_DTYPE_F32, batch.dtype);
        ASSERT_UINT_EQ(16, batch.shape[1]);
        const float* bdata = rcre_tensor_f32_data(&batch);
        ASSERT_FLOAT_NEAR((float)(rows_read * 16), bdata[0], 0.0f);
        rows_read += batch.shape[0];
        batches_read++;
    }
    ASSERT_UINT_EQ(4, batches_read);
    ASSERT_UINT_EQ(128, rows_read);

    /* Reset and re-stream */
    rcre_io_stream_reset(&stream);
    ASSERT_UINT_EQ(0, stream.current_row);
    ASSERT_TRUE(rcre_io_stream_next(&stream, &batch));
    ASSERT_UINT_EQ(32, batch.shape[0]);

    rcre_io_stream_close(&stream);
    remove(path);
    rcre_arena_destroy(arena);
}

int main(void) {
    rcre_hal_init();
    test_suite_begin("RCRE High-Performance Scientific Data I/O Engine (IO-REQ-01 to 06)");
    TEST_RUN(test_io_mmap_atomic_write_and_telemetry);
    TEST_RUN(test_io_fast_float_eisel_lemire_0_ulp_parity);
    TEST_RUN(test_io_simd_csv_and_parallel_threadpool);
    TEST_RUN(test_io_npy_roundtrip_zero_copy_and_adversarial);
    TEST_RUN(test_io_safetensors_archive_and_adversarial);
    TEST_RUN(test_io_simd_byte_shuffle_compression_and_streamer);
    TEST_RUN(test_io_dlpack_and_arrow_zero_copy_bridges);
    TEST_RUN(test_io_half_precision_fp16_bf16_simd_and_dequantization);
    TEST_RUN(test_io_safetensors_f16_bf16_and_chunked_slicing);
    TEST_RUN(test_io_simd_16bit_byte_shuffle_and_rcrz_compression);
    TEST_RUN(test_io_uring_probe_and_async_streaming);
    return test_suite_report();
}
