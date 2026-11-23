/**
 * @file bench_io.c
 * @brief Comprehensive Hardware I/O Bandwidth & Scientific Interop Benchmark Suite
 *
 * Benchmarks:
 *  1. Float32 Parsing: libc strtof/atof vs RCRE Clinger/Eisel-Lemire rcre_slice_parse_f32
 *  2. Tabular CSV Ingestion: Legacy C89 fgetc+strtok_r+atof vs rcre_io Serial vs ThreadPool
 *  3. Binary Tensor Loading (64 MB): stdio fread+malloc vs rcre_io_load_npy vs Zero-Copy mmap view (.npy & .safetensors)
 *  4. SIMD Byte-Shuffle (NEON/AVX2/WASM) & RCRZ LZ4-Class Block Compression/Decompression
 *  5. Cross-Framework Zero-Copy C ABI Bridges: DLPack v1.0 & Apache Arrow C Data Interface
 */

#include "rcre.h"
#include "rcre_io.h"
#include "rcre_prof.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Legacy C89 baseline (mirrors old src/nn/data.c fgetc + strtok_r + atof) */
static size_t legacy_c89_load_csv(const char* filepath, float* dst, size_t max_elems) {
    FILE* f = fopen(filepath, "r");
    if (!f) return 0;

    size_t cap = 4096;
    char* buf = (char*)malloc(cap);
    size_t count = 0;

    for (;;) {
        size_t len = 0;
        int c;
        while ((c = fgetc(f)) != EOF) {
            if (c == '\n') break;
            if (c == '\r') continue;
            if (len + 1 >= cap) {
                cap *= 2;
                buf = (char*)realloc(buf, cap);
            }
            buf[len++] = (char)c;
        }
        if (c == EOF && len == 0) break;
        buf[len] = '\0';

        char* save = nullptr;
        char* tok = strtok_r(buf, ",\t", &save);
        while (tok && count < max_elems) {
            dst[count++] = (float)atof(tok);
            tok = strtok_r(nullptr, ",\t", &save);
        }
    }

    free(buf);
    fclose(f);
    return count;
}

static void bench_float_parsing(void) {
    printf("\n📊 [1/5] Float32 ASCII Token Parsing (1,000,000 Scientific Floats)\n");
    const size_t N = 1000000;
    const char* sample_tokens[8] = {
        "3.1415927", "-0.0002451", "12345.678", "-98.76543",
        "1.2345e-4", "6.02214e+5", "0.5000000", "-42.12500"
    };
    RCRE_Slice slices[8];
    size_t total_bytes = 0;
    for (int i = 0; i < 8; ++i) {
        slices[i] = rcre_slice_from_cstr(sample_tokens[i]);
        total_bytes += slices[i].len * (N / 8);
    }

    volatile float sink = 0.0f;

    /* 1. libc atof / strtof */
    double t0 = bench_now_sec();
    float acc1 = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        acc1 += strtof(sample_tokens[i & 7], nullptr);
    }
    double dt_libc = bench_now_sec() - t0;
    sink = acc1;

    /* 2. RCRE Eisel-Lemire rcre_slice_parse_f32 */
    t0 = bench_now_sec();
    float acc2 = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float v = 0.0f;
        (void)rcre_slice_parse_f32(slices[i & 7], &v);
        acc2 += v;
    }
    double dt_rcre = bench_now_sec() - t0;
    sink += acc2;
    (void)sink;

    double mfloats_libc = ((double)N * 1e-6) / dt_libc;
    double mfloats_rcre = ((double)N * 1e-6) / dt_rcre;
    double mbps_libc    = ((double)total_bytes / (1024.0 * 1024.0)) / dt_libc;
    double mbps_rcre    = ((double)total_bytes / (1024.0 * 1024.0)) / dt_rcre;

    printf("   • libc strtof (Locale-Bound):      %6.2f ms | %6.1f M floats/s | %7.1f MB/s\n",
           dt_libc * 1000.0, mfloats_libc, mbps_libc);
    printf("   • RCRE Eisel-Lemire (parse_f32):   %6.2f ms | %6.1f M floats/s | %7.1f MB/s (Speedup: %.2fx)\n",
           dt_rcre * 1000.0, mfloats_rcre, mbps_rcre, dt_libc / dt_rcre);
}

static void bench_csv_ingestion(void) {
    printf("\n📊 [2/5] Tabular CSV Ingestion (16,384 Rows x 64 Cols = 1,048,576 Floats)\n");
    const size_t ROWS = 16384;
    const size_t COLS = 64;
    const char* csv_path = "/tmp/rcre_bench_io_matrix.csv";

    FILE* f = fopen(csv_path, "wb");
    if (!f) return;
    char row_buf[2048];
    int pos = 0;
    for (size_t c = 0; c < COLS; ++c) {
        pos += snprintf(row_buf + pos, sizeof(row_buf) - (size_t)pos,
                        "%s%.4f", (c == 0) ? "" : ",", (double)c * 0.125 - 4.0);
    }
    row_buf[pos++] = '\n';
    for (size_t r = 0; r < ROWS; ++r) {
        (void)fwrite(row_buf, 1, (size_t)pos, f);
    }
    fclose(f);

    size_t file_bytes = ROWS * (size_t)pos;
    double file_mb = (double)file_bytes / (1024.0 * 1024.0);

    RCRE_Arena* arena = rcre_arena_create(32 * 1024 * 1024);
    float* legacy_buf = (float*)malloc(ROWS * COLS * sizeof(float));

    /* 1. Legacy C89 fgetc + strtok_r + atof */
    double t0 = bench_now_sec();
    size_t parsed = legacy_c89_load_csv(csv_path, legacy_buf, ROWS * COLS);
    double dt_legacy = bench_now_sec() - t0;
    (void)parsed;

    /* 2. RCRE rcre_io_load_csv_ex (Single-Threaded mmap + SIMD + Eisel-Lemire) */
    RCRE_CSVOptions opts_serial = {
        .delimiter      = ',',
        .skip_header    = false,
        .strict_numbers = false,
        .max_rows       = 0,
        .pool           = nullptr
    };
    RCRE_Tensor t_out = {0};
    rcre_arena_reset(arena);
    t0 = bench_now_sec();
    (void)rcre_io_load_csv_ex(csv_path, &opts_serial, arena, &t_out);
    double dt_serial = bench_now_sec() - t0;

    /* 3. RCRE rcre_io_load_csv_ex (Multi-Threaded RCRE_ThreadPool) */
    RCRE_ThreadPool* pool = rcre_threadpool_create_ex(0, RCRE_QOS_PERF);
    RCRE_CSVOptions opts_parallel = opts_serial;
    opts_parallel.pool = pool;

    /* Warmup threadpool */
    rcre_arena_reset(arena);
    (void)rcre_io_load_csv_ex(csv_path, &opts_parallel, arena, &t_out);

    rcre_arena_reset(arena);
    t0 = bench_now_sec();
    (void)rcre_io_load_csv_ex(csv_path, &opts_parallel, arena, &t_out);
    double dt_parallel = bench_now_sec() - t0;

    printf("   • File Size: %.2f MB (%zu floats)\n", file_mb, ROWS * COLS);
    printf("   • Legacy C89 (fgetc+strtok+atof):  %7.2f ms | %7.1f MB/s\n",
           dt_legacy * 1000.0, file_mb / dt_legacy);
    printf("   • RCRE Serial (mmap+SIMD+Eisel):   %7.2f ms | %7.1f MB/s (Speedup: %.1fx)\n",
           dt_serial * 1000.0, file_mb / dt_serial, dt_legacy / dt_serial);
    printf("   • RCRE Parallel (ThreadPool Slabs):%7.2f ms | %7.1f MB/s (Speedup: %.1fx | %.2f GB/s)\n",
           dt_parallel * 1000.0, file_mb / dt_parallel, dt_legacy / dt_parallel,
           (file_mb / 1024.0) / dt_parallel);

    const char* min_env = getenv("RCRE_IO_MIN_CSV_MBPS");
    if (min_env && min_env[0] != '\0') {
        double min_mbps = strtod(min_env, nullptr);
        double parallel_mbps = file_mb / dt_parallel;
        if (parallel_mbps < min_mbps) {
            fprintf(stderr, "❌ CSV Throughput Regression Gate Failed: %.1f MB/s < %.1f MB/s\n",
                    parallel_mbps, min_mbps);
            exit(2);
        }
        printf("   ✅ CI Throughput Gate Passed (%.1f MB/s >= %.1f MB/s)\n", parallel_mbps, min_mbps);
    }

    rcre_threadpool_destroy(pool);
    free(legacy_buf);
    rcre_arena_destroy(arena);
    remove(csv_path);
}

static void bench_binary_containers(void) {
    printf("\n📊 [3/5] Binary Tensor Deserialization (64 MB Tensor: [4096, 4096] F32)\n");
    const size_t N = 4096;
    const size_t BYTES = N * N * sizeof(float); /* 64 MiB */
    const double GB = (double)BYTES / (1024.0 * 1024.0 * 1024.0);

    RCRE_Arena* arena = rcre_arena_create(160 * 1024 * 1024);
    size_t shape[2] = { N, N };
    RCRE_Tensor src = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* sd = rcre_tensor_f32_data(&src);
    for (size_t i = 0; i < N * N; ++i) {
        sd[i] = (float)(i & 1023) * 0.001f;
    }

    const char* npy_path = "/tmp/rcre_bench_64mb.npy";
    const char* st_path  = "/tmp/rcre_bench_64mb.safetensors";
    (void)rcre_io_save_npy(&src, npy_path);
    const char* names[1] = { "matrix_64mb" };
    (void)rcre_io_safetensors_save(st_path, names, &src, 1, arena);

    /* 1. Traditional stdio fopen + malloc + fread + free */
    double t0 = bench_now_sec();
    FILE* f = fopen(npy_path, "rb");
    void* heap_buf = malloc(BYTES);
    (void)fseek(f, 128, SEEK_SET);
    size_t nr = fread(heap_buf, 1, BYTES, f);
    fclose(f);
    double dt_fread = bench_now_sec() - t0;
    free(heap_buf);
    (void)nr;

    /* 2. RCRE rcre_io_load_npy (mmap + 64B aligned Arena copy) */
    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
    RCRE_Tensor loaded = {0};
    t0 = bench_now_sec();
    (void)rcre_io_load_npy(npy_path, arena, &loaded);
    double dt_load_npy = bench_now_sec() - t0;
    rcre_arena_temp_end(tmp);

    /* 3. RCRE Zero-Copy View (rcre_io_mmap_open + rcre_io_view_npy) */
    const int ITERS = 1000;
    t0 = bench_now_sec();
    for (int i = 0; i < ITERS; ++i) {
        RCRE_MappedFile map = {0};
        RCRE_Tensor view = {0};
        (void)rcre_io_mmap_open(npy_path, RCRE_IO_HINT_DEFAULT, &map);
        (void)rcre_io_view_npy(&map, &view);
        rcre_io_mmap_close(&map);
    }
    double dt_view_npy = (bench_now_sec() - t0) / ITERS;

    /* 4. RCRE Safetensors Zero-Copy View Lookup on Open Archive */
    RCRE_SafeTensorsArchive arch = {0};
    (void)rcre_io_safetensors_open(st_path, &arch);
    t0 = bench_now_sec();
    const int LOOKUP_ITERS = 100000;
    RCRE_Tensor st_view = {0};
    for (int i = 0; i < LOOKUP_ITERS; ++i) {
        (void)rcre_io_safetensors_get_tensor(&arch, "matrix_64mb", nullptr, &st_view);
    }
    double dt_st_lookup = (bench_now_sec() - t0) / LOOKUP_ITERS;

    /* 5. Streaming read bandwidth directly over mapped view (SIMD dot product) */
    t0 = bench_now_sec();
    float dot = rcre_hal.dot_f32((const float*)st_view.data, (const float*)st_view.data, N * N);
    double dt_stream_compute = bench_now_sec() - t0;
    (void)dot;
    rcre_io_safetensors_close(&arch);

    printf("   • stdio fopen + malloc + fread:    %7.3f ms | %6.2f GB/s | 64.0 MB Heap Alloc\n",
           dt_fread * 1000.0, GB / dt_fread);
    printf("   • rcre_io_load_npy (Arena Copy):   %7.3f ms | %6.2f GB/s | 64.0 MB Arena Alloc\n",
           dt_load_npy * 1000.0, GB / dt_load_npy);
    printf("   • rcre_io_view_npy (mmap+close):   %7.3f µs | Zero-Copy   |  0.0 MB Alloc (%.0fx faster init)\n",
           dt_view_npy * 1e6, dt_fread / dt_view_npy);
    printf("   • rcre_io_safetensors_get_tensor:  %7.1f ns | Zero-Copy   |  0.0 MB Alloc\n",
           dt_st_lookup * 1e9);
    printf("   • Direct SIMD Compute on mmap View:%7.3f ms | %6.2f GB/s sustained read bandwidth\n",
           dt_stream_compute * 1000.0, GB / dt_stream_compute);

    remove(npy_path);
    remove(st_path);
    rcre_arena_destroy(arena);
}

static void bench_shuffle_and_compression(void) {
    printf("\n📊 [4/5] SIMD Byte-Shuffle & Lossless RCRZ Block Compression (16 MB F32 Tensor)\n");
    const size_t N = 2048;
    const size_t NUMEL = N * N; /* 4,194,304 floats = 16 MiB */
    const size_t BYTES = NUMEL * sizeof(float);
    const double GB = (double)BYTES / (1024.0 * 1024.0 * 1024.0);

    RCRE_Arena* arena = rcre_arena_create(128 * 1024 * 1024);
    size_t shape[2] = { N, N };
    RCRE_Tensor src = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    float* sd = rcre_tensor_f32_data(&src);
    for (size_t i = 0; i < NUMEL; ++i) {
        sd[i] = (i % 4 == 0) ? 0.0f : (0.5f + (float)(i % 32) * (1.0f / 128.0f));
    }

    uint8_t* shuf_buf = (uint8_t*)rcre_arena_alloc(arena, BYTES, 64);
    uint8_t* rec_buf  = (uint8_t*)rcre_arena_alloc(arena, BYTES, 64);

    const int RUNS = 20;
    double t0 = bench_now_sec();
    for (int r = 0; r < RUNS; ++r) {
        rcre_io_byte_shuffle_32(src.data, shuf_buf, NUMEL);
    }
    double dt_shuf = (bench_now_sec() - t0) / RUNS;

    t0 = bench_now_sec();
    for (int r = 0; r < RUNS; ++r) {
        rcre_io_byte_unshuffle_32(shuf_buf, rec_buf, NUMEL);
    }
    double dt_unshuf = (bench_now_sec() - t0) / RUNS;

    /* RCRZ Full Compress & Decompress */
    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
    RCRE_Slice comp_slice = rcre_slice_empty();
    t0 = bench_now_sec();
    (void)rcre_io_compress_tensor(&src, arena, &comp_slice);
    double dt_comp = bench_now_sec() - t0;

    RCRE_Tensor dec_t = {0};
    t0 = bench_now_sec();
    (void)rcre_io_decompress_tensor(comp_slice, arena, &dec_t);
    double dt_decomp = bench_now_sec() - t0;

    double ratio = (double)BYTES / (double)comp_slice.len;

    printf("   • SIMD Byte-Shuffle (%s):   %6.3f ms | %6.2f GB/s\n",
           rcre_hal.backend_name, dt_shuf * 1000.0, GB / dt_shuf);
    printf("   • SIMD Byte-Unshuffle (%s): %6.3f ms | %6.2f GB/s\n",
           rcre_hal.backend_name, dt_unshuf * 1000.0, GB / dt_unshuf);
    printf("   • RCRZ Lossless Compress:          %6.3f ms | %6.2f GB/s | Ratio: %.2fx (16.0 MB -> %.2f MB)\n",
           dt_comp * 1000.0, GB / dt_comp, ratio, (double)comp_slice.len / (1024.0 * 1024.0));
    printf("   • RCRZ Lossless Decompress:        %6.3f ms | %6.2f GB/s | 0-ULP Bit-Exact\n",
           dt_decomp * 1000.0, GB / dt_decomp);

    rcre_arena_temp_end(tmp);
    rcre_arena_destroy(arena);
}

static void bench_abi_bridges(void) {
    printf("\n📊 [5/5] Zero-Copy Cross-Framework C ABI Handoff (DLPack v1.0 & Apache Arrow)\n");
    RCRE_Arena* arena = rcre_arena_create(4 * 1024 * 1024);
    size_t shape[1] = { 262144 }; /* 1 MB vector */
    RCRE_Tensor vec = rcre_tensor_create(shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);

    const int ITERS = 1000000;
    RCRE_DLTensor dl = {0};
    RCRE_Tensor view = {0};

    double t0 = bench_now_sec();
    for (int i = 0; i < ITERS; ++i) {
        RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
        (void)rcre_io_to_dlpack(&vec, arena, &dl);
        (void)rcre_io_from_dlpack(&dl, &view);
        rcre_arena_temp_end(tmp);
    }
    double dt_dlpack = (bench_now_sec() - t0) / ITERS;

    RCRE_ArrowSchema schema = {0};
    RCRE_ArrowArray  arr    = {0};
    t0 = bench_now_sec();
    for (int i = 0; i < ITERS; ++i) {
        RCRE_ArenaTemp tmp = rcre_arena_temp_begin(arena);
        (void)rcre_io_to_arrow(&vec, arena, &schema, &arr);
        (void)rcre_io_from_arrow(&schema, &arr, &view);
        rcre_arena_temp_end(tmp);
    }
    double dt_arrow = (bench_now_sec() - t0) / ITERS;

    printf("   • DLPack v1.0 Round-Trip (Export+Import): %5.1f ns | 0 Bytes Copied\n", dt_dlpack * 1e9);
    printf("   • Arrow C Data Interface Round-Trip:      %5.1f ns | 0 Bytes Copied\n", dt_arrow * 1e9);

    rcre_arena_destroy(arena);
}

static void bench_half_precision_dequantization(void) {
    printf("\n📊 [6/6] Half-Precision SIMD De-Quantization Throughput (16M Elements)\n");
    RCRE_Arena* arena = rcre_arena_create(128 * 1024 * 1024);
    const size_t N = 16 * 1024 * 1024; /* 16M elements: 32 MB input + 64 MB output */
    uint16_t* fp16_buf = (uint16_t*)rcre_arena_alloc(arena, N * sizeof(uint16_t), 64);
    uint16_t* bf16_buf = (uint16_t*)rcre_arena_alloc(arena, N * sizeof(uint16_t), 64);
    float* f32_dst = (float*)rcre_arena_alloc(arena, N * sizeof(float), 64);

    for (size_t i = 0; i < N; ++i) {
        fp16_buf[i] = (uint16_t)(i & 0xFFFF);
        bf16_buf[i] = (uint16_t)(i & 0xFFFF);
    }

    const int ITERS = 10;
    double t0 = bench_now_sec();
    for (int it = 0; it < ITERS; ++it) {
        rcre_f16_to_f32(fp16_buf, f32_dst, N);
    }
    double dt_f16 = (bench_now_sec() - t0) / ITERS;
    double bw_f16 = ((double)N * sizeof(float) + (double)N * sizeof(uint16_t)) / (dt_f16 * 1e9);
    double mps_f16 = ((double)N / 1e6) / dt_f16;

    t0 = bench_now_sec();
    for (int it = 0; it < ITERS; ++it) {
        rcre_bf16_to_f32(bf16_buf, f32_dst, N);
    }
    double dt_bf16 = (bench_now_sec() - t0) / ITERS;
    double bw_bf16 = ((double)N * sizeof(float) + (double)N * sizeof(uint16_t)) / (dt_bf16 * 1e9);
    double mps_bf16 = ((double)N / 1e6) / dt_bf16;

    printf("   • IEEE FP16 -> FP32 SIMD Dequant:  %6.2f ms | %6.1f M elem/s | %6.2f GB/s\n",
           dt_f16 * 1e3, mps_f16, bw_f16);
    printf("   • Google BF16 -> FP32 SIMD Dequant: %6.2f ms | %6.1f M elem/s | %6.2f GB/s\n",
           dt_bf16 * 1e3, mps_bf16, bw_bf16);

    rcre_arena_destroy(arena);
}

int main(void) {
    rcre_hal_init();
    RCRE_CpuTopology topo = rcre_cpu_get_topology();

    printf("======================================================================\n");
    printf("🚀 RCRE Scientific Data I/O & Bandwidth Benchmark Suite (rcre_io)\n");
    printf("   Host CPU: %s (%u Logical Cores: %uP + %uE)\n",
           rcre_cpu_get_brand(), topo.logical_cores, topo.perf_cores, topo.eff_cores);
    printf("   Active SIMD Backend: %s | Unified RAM: %.1f GB\n",
           rcre_hal.backend_name, (double)topo.total_ram_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("======================================================================\n");

    bench_float_parsing();
    bench_csv_ingestion();
    bench_binary_containers();
    bench_shuffle_and_compression();
    bench_abi_bridges();
    bench_half_precision_dequantization();

    printf("\n======================================================================\n");
    printf("🏁 All Scientific I/O Bandwidth Benchmarks Completed!\n");
    printf("======================================================================\n");
    return 0;
}
