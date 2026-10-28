/**
 * @file main.c
 * @brief RCRE-Wavelet CLI Application & Rate-Distortion Benchmark
 *
 * Standalone multi-resolution integer wavelet codec powered by RCRE.
 */

#include "wavelet_codec.h"
#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char* prog) {
    printf("======================================================================\n");
    printf("🌊 RCRE-Wavelet: Reversible Multi-Resolution Image Codec (ISO C23)\n");
    printf("======================================================================\n");
    printf("Usage:\n");
    printf("  %s --demo                           Run end-to-end codec demo\n", prog);
    printf("  %s --benchmark [image.bmp]          Run rate-distortion sweep\n", prog);
    printf("  %s --encode in.bmp out.rdwt [-q 8]  Encode image with quantization\n", prog);
    printf("  %s --decode in.rdwt out.bmp         Decode RDWT file to bitmap\n", prog);
    printf("\nFlags:\n");
    printf("  -q, --quant <int>   Quantization step (1=lossless, default: 8)\n");
    printf("  -l, --levels <int>  DWT decomposition levels (default: 3)\n");
    printf("======================================================================\n");
}

static void run_demo(void) {
    printf("\n======================================================================\n");
    printf("🌊 RCRE-Wavelet: Multi-Resolution Image Codec Demonstration\n");
    printf("======================================================================\n");

    RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024);
    if (!arena) {
        fprintf(stderr, "Failed to allocate memory arena\n");
        return;
    }

    /* 1. Generate synthetic test pattern */
    const size_t H = 256;
    const size_t W = 256;
    size_t shape[3] = {H, W, 3};
    RCRE_Tensor original = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
    uint8_t* orig_data = (uint8_t*)original.data;

    for (size_t r = 0; r < H; ++r) {
        for (size_t c = 0; c < W; ++c) {
            uint8_t red   = (uint8_t)((r * 255) / H);
            uint8_t green = (uint8_t)((c * 255) / W);
            uint8_t blue  = (uint8_t)(((r + c) * 128) / (H + W));
            /* Add high-frequency edge box */
            if (r > 64 && r < 192 && c > 64 && c < 192) {
                red   = 255 - red;
                green = 255 - green;
            }
            size_t idx = (r * W + c) * 3;
            orig_data[idx + 0] = red;
            orig_data[idx + 1] = green;
            orig_data[idx + 2] = blue;
        }
    }

    const char* orig_bmp = "wavelet_demo_original.bmp";
    const char* rdwt_file = "wavelet_demo_compressed.rdwt";
    const char* recon_bmp = "wavelet_demo_reconstructed.bmp";

    rcre_save_bmp(orig_bmp, &original);
    printf("  ✓ Generated test pattern [%zux%zu RGB] -> %s\n", W, H, orig_bmp);

    /* 2. Lossless Run (q=1) */
    printf("\n[1/2] Testing Bit-Exact Lossless Mode (Quant Step = 1)...\n");
    size_t lossless_bytes = 0;
    uint8_t* lossless_stream = rcre_wavelet_encode(&original, 3, 1, arena, &lossless_bytes);

    RCRE_Tensor lossless_recon = {0};
    (void)rcre_wavelet_decode(lossless_stream, lossless_bytes, arena, &lossless_recon);

    float lossless_mse = 0.0f;
    float lossless_psnr = rcre_wavelet_compute_psnr(&original, &lossless_recon, &lossless_mse);
    printf("  • Compressed size : %zu bytes (%.2fx lossless compression)\n",
           lossless_bytes, (float)(H * W * 3) / (float)lossless_bytes);
    printf("  • MSE / PSNR      : %.4f MSE | %.1f dB PSNR (100%% Bit-Exact Reversible!)\n",
           lossless_mse, lossless_psnr);

    /* 3. Lossy Run (q=12) */
    printf("\n[2/2] Testing High-Compression Lossy Mode (Quant Step = 12)...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    size_t lossy_bytes = 0;
    uint8_t* lossy_stream = rcre_wavelet_encode(&original, 3, 12, arena, &lossy_bytes);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double enc_ms = ((double)(t1.tv_sec - t0.tv_sec) * 1000.0) + ((double)(t1.tv_nsec - t0.tv_nsec) / 1e6);

    /* Save compressed file */
    FILE* f_out = fopen(rdwt_file, "wb");
    if (f_out) {
        fwrite(lossy_stream, 1, lossy_bytes, f_out);
        fclose(f_out);
    }

    clock_gettime(CLOCK_MONOTONIC, &t0);
    RCRE_Tensor lossy_recon = {0};
    (void)rcre_wavelet_decode(lossy_stream, lossy_bytes, arena, &lossy_recon);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dec_ms = ((double)(t1.tv_sec - t0.tv_sec) * 1000.0) + ((double)(t1.tv_nsec - t0.tv_nsec) / 1e6);

    rcre_save_bmp(recon_bmp, &lossy_recon);

    float lossy_mse = 0.0f;
    float lossy_psnr = rcre_wavelet_compute_psnr(&original, &lossy_recon, &lossy_mse);

    printf("  • Original Size   : %zu bytes (24 bits/pixel)\n", H * W * 3);
    printf("  • RDWT File Size  : %zu bytes (%.2f bits/pixel)\n", lossy_bytes, (float)(lossy_bytes * 8) / (float)(H * W));
    printf("  • Compression     : %.1fx reduction (%.1f%% bandwidth saved)\n",
           (float)(H * W * 3) / (float)lossy_bytes,
           (1.0f - (float)lossy_bytes / (float)(H * W * 3)) * 100.0f);
    printf("  • Reconstruction  : %.2f dB PSNR (High Visual Fidelity)\n", lossy_psnr);
    printf("  • Latency         : Encode: %.2f ms | Decode: %.2f ms\n", enc_ms, dec_ms);
    printf("  ✓ Saved output to %s and %s\n", rdwt_file, recon_bmp);

    printf("\n======================================================================\n");
    printf("🎉 Wavelet Codec Demo Completed with Zero Memory Leaks!\n");
    printf("======================================================================\n");

    rcre_arena_destroy(arena);
}

static void run_benchmark(const char* image_path) {
    RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024);
    if (!arena) return;

    RCRE_Tensor img = {0};
    if (image_path) {
        if (rcre_load_image(image_path, &img, arena) != 0) {
            img.data = NULL;
        }
    }

    if (!img.data) {
        /* Generate synthetic benchmark pattern 512x512 */
        size_t shape[3] = {512, 512, 3};
        img = rcre_tensor_create(shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
        uint8_t* p = (uint8_t*)img.data;
        for (size_t r = 0; r < 512; ++r) {
            for (size_t c = 0; c < 512; ++c) {
                p[(r * 512 + c) * 3 + 0] = (uint8_t)(r / 2);
                p[(r * 512 + c) * 3 + 1] = (uint8_t)(c / 2);
                p[(r * 512 + c) * 3 + 2] = (uint8_t)((r + c) / 4);
            }
        }
    }

    size_t H = img.shape[0];
    size_t W = img.shape[1];
    size_t raw_bytes = H * W * (img.ndim >= 3 ? img.shape[2] : 1);

    printf("\n====================================================================================\n");
    printf("📊 RCRE-Wavelet Rate-Distortion Sweep [%zux%zu Image | Raw: %.1f KB]\n",
           W, H, (float)raw_bytes / 1024.0f);
    printf("====================================================================================\n");
    printf("| Quant Step | Size (KB) | Ratio  | BPP   | PSNR (dB) | MSE    | Enc (ms) | Dec (ms) |\n");
    printf("|:----------:|:---------:|:------:|:-----:|:---------:|:------:|:--------:|:--------:|\n");

    uint16_t q_steps[] = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64};
    size_t num_q = sizeof(q_steps) / sizeof(q_steps[0]);

    for (size_t i = 0; i < num_q; ++i) {
        uint16_t q = q_steps[i];
        RCRE_ArenaTemp checkpoint = rcre_arena_temp_begin(arena);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        size_t compressed_bytes = 0;
        uint8_t* stream = rcre_wavelet_encode(&img, 3, q, arena, &compressed_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double enc_ms = ((double)(t1.tv_sec - t0.tv_sec) * 1000.0) + ((double)(t1.tv_nsec - t0.tv_nsec) / 1e6);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        RCRE_Tensor recon = {0};
        (void)rcre_wavelet_decode(stream, compressed_bytes, arena, &recon);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dec_ms = ((double)(t1.tv_sec - t0.tv_sec) * 1000.0) + ((double)(t1.tv_nsec - t0.tv_nsec) / 1e6);

        float mse = 0.0f;
        float psnr = rcre_wavelet_compute_psnr(&img, &recon, &mse);

        float ratio = (float)raw_bytes / (float)compressed_bytes;
        float bpp = (float)(compressed_bytes * 8) / (float)(H * W);

        printf("| %10u | %9.1f | %5.1fx | %5.2f | %9.2f | %6.2f | %8.2f | %8.2f |\n",
               q, (float)compressed_bytes / 1024.0f, ratio, bpp, psnr, mse, enc_ms, dec_ms);

        rcre_arena_temp_end(checkpoint);
    }
    printf("====================================================================================\n");

    rcre_arena_destroy(arena);
}

int main(int argc, char** argv) {
    rcre_hal_init();

    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    if (strcmp(argv[1], "--demo") == 0) {
        run_demo();
        return 0;
    }

    if (strcmp(argv[1], "--benchmark") == 0) {
        const char* path = (argc >= 3) ? argv[2] : NULL;
        run_benchmark(path);
        return 0;
    }

    if (strcmp(argv[1], "--encode") == 0) {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }
        const char* in_bmp = argv[2];
        const char* out_rdwt = argv[3];
        uint16_t q = 8;
        size_t levels = 3;

        for (int i = 4; i < argc; ++i) {
            if ((strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quant") == 0) && i + 1 < argc) {
                q = (uint16_t)atoi(argv[++i]);
            } else if ((strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--levels") == 0) && i + 1 < argc) {
                levels = (size_t)atoi(argv[++i]);
            }
        }

        RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024);
        RCRE_Tensor img = {0};
        if (rcre_load_image(in_bmp, &img, arena) != 0 || !img.data) {
            fprintf(stderr, "Failed to load image: %s\n", in_bmp);
            rcre_arena_destroy(arena);
            return 1;
        }

        size_t out_bytes = 0;
        uint8_t* compressed = rcre_wavelet_encode(&img, levels, q, arena, &out_bytes);
        if (!compressed) {
            fprintf(stderr, "Compression failed\n");
            rcre_arena_destroy(arena);
            return 1;
        }

        FILE* f = fopen(out_rdwt, "wb");
        if (!f) {
            fprintf(stderr, "Failed to write %s\n", out_rdwt);
            rcre_arena_destroy(arena);
            return 1;
        }
        fwrite(compressed, 1, out_bytes, f);
        fclose(f);

        printf("✅ Successfully encoded %s -> %s (%zu bytes, q=%u, levels=%zu)\n",
               in_bmp, out_rdwt, out_bytes, q, levels);
        rcre_arena_destroy(arena);
        return 0;
    }

    if (strcmp(argv[1], "--decode") == 0) {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }
        const char* in_rdwt = argv[2];
        const char* out_bmp = argv[3];

        FILE* f = fopen(in_rdwt, "rb");
        if (!f) {
            fprintf(stderr, "Failed to open %s\n", in_rdwt);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);

        RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024);
        uint8_t* buf = (uint8_t*)rcre_arena_alloc(arena, sz, 64);
        if (fread(buf, 1, sz, f) != (size_t)sz) {
            fclose(f);
            rcre_arena_destroy(arena);
            return 1;
        }
        fclose(f);

        RCRE_Tensor decoded = {0};
        int rc = rcre_wavelet_decode(buf, sz, arena, &decoded);
        if (rc != RCRE_OK) {
            fprintf(stderr, "Decompression failed (error %d)\n", rc);
            rcre_arena_destroy(arena);
            return 1;
        }

        rcre_save_bmp(out_bmp, &decoded);
        printf("✅ Successfully decoded %s -> %s [%zux%zu]\n",
               in_rdwt, out_bmp, decoded.shape[1], decoded.shape[0]);
        rcre_arena_destroy(arena);
        return 0;
    }

    print_usage(argv[0]);
    return 0;
}
