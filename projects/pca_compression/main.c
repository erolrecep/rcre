/**
 * @file main.c
 * @brief Standalone CLI for RCRE PCA Image Compression & Quantization
 */

#include "pca_compressor.h"
#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void print_banner(void) {
    printf("===================================================================\n");
    printf(" 🎨 RCRE PCA Image Compression & Quantization Engine (v1.0.0)      \n");
    printf("    Pure C23 Zero-Dependency Dimensionality Reduction & Codec      \n");
    printf("===================================================================\n");
}

static void run_demo(void) {
    print_banner();
    printf("🚀 Executing Automated End-to-End PCA Compression Demo...\n\n");

    RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024); /* 64 MB arena */
    if (!arena) {
        fprintf(stderr, "Failed to create arena\n");
        return;
    }

    const size_t H = 256;
    const size_t W = 256;
    const uint32_t B = 8;
    const uint32_t K = 12;

    printf("1. Synthesizing High-Detail 256x256 RGB Test Pattern...\n");
    RCRE_Tensor orig_img;
    rcre_generate_test_image(&orig_img, H, W, arena);
    rcre_save_bmp("pca_demo_original.bmp", &orig_img);
    printf("   ✓ Saved original reference image: pca_demo_original.bmp\n");

    size_t uncompressed_bytes = H * W * 3 * sizeof(uint8_t);
    printf("   Uncompressed 24-bit RGB payload: %zu bytes (%.2f KB)\n\n", 
           uncompressed_bytes, (double)uncompressed_bytes / 1024.0);

    printf("2. Compressing with Spatial Tiling (Block=%ux%u, Components K=%u)...\n", B, B, K);
    double t0 = get_time_ms();
    RCRE_PCACompressedImage comp;
    int status = rcre_pca_compress(&orig_img, B, K, &comp, arena);
    double t_compress = get_time_ms() - t0;

    if (status != RCRE_OK) {
        fprintf(stderr, "Compression failed with error code: %d\n", status);
        rcre_arena_destroy(arena);
        return;
    }

    status = rcre_pca_save_file("pca_demo_compressed.rpca", &comp);
    if (status != RCRE_OK) {
        fprintf(stderr, "Failed to save .rpca file\n");
    }

    size_t comp_bytes = rcre_pca_compressed_byte_size(&comp);
    double comp_ratio = (double)uncompressed_bytes / (double)comp_bytes;
    double savings_pct = (1.0 - (double)comp_bytes / (double)uncompressed_bytes) * 100.0;

    printf("   ✓ Compression finished in %.2f ms\n", t_compress);
    printf("   ✓ Saved binary bitstream: pca_demo_compressed.rpca (%zu bytes, %.2f KB)\n", 
           comp_bytes, (double)comp_bytes / 1024.0);
    printf("   📊 Compression Ratio: %.2fx (%.1f%% bitstream savings!)\n\n", comp_ratio, savings_pct);

    printf("3. Decompressing & Reconstructing via Inverse PCA Transform...\n");
    t0 = get_time_ms();
    RCRE_PCACompressedImage loaded_comp;
    status = rcre_pca_load_file("pca_demo_compressed.rpca", &loaded_comp, arena);
    if (status != RCRE_OK) {
        fprintf(stderr, "Failed to load .rpca file\n");
    }

    size_t out_shape[3] = {H, W, 3};
    RCRE_Tensor recon_img = rcre_tensor_create(out_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
    status = rcre_pca_decompress(&loaded_comp, &recon_img, arena);
    double t_decompress = get_time_ms() - t0;

    if (status != RCRE_OK) {
        fprintf(stderr, "Decompression failed with error code: %d\n", status);
        rcre_arena_destroy(arena);
        return;
    }

    rcre_save_bmp("pca_demo_reconstructed.bmp", &recon_img);
    printf("   ✓ Decompression finished in %.2f ms\n", t_decompress);
    printf("   ✓ Saved reconstructed image: pca_demo_reconstructed.bmp\n\n");

    printf("4. Evaluating Reconstruction Fidelity (MSE & PSNR)...\n");
    float mse = 0.0f, psnr = 0.0f;
    status = rcre_pca_compute_metrics(&orig_img, &recon_img, &mse, &psnr);
    if (status != RCRE_OK) {
        fprintf(stderr, "Metric computation failed\n");
    }

    printf("   -------------------------------------------------------------\n");
    printf("   Metric             Value                 Status             \n");
    printf("   -------------------------------------------------------------\n");
    printf("   MSE (Mean Sq Err)  %-18.3f     %s\n", mse, (mse < 100.0f) ? "EXCELLENT" : "ACCEPTABLE");
    printf("   PSNR (Fidelity)    %-15.2f dB     %s\n", psnr, (psnr >= 28.0f) ? "HIGH QUALITY" : "LOSSY");
    printf("   Compression Ratio  %-15.2fx       SAVINGS: %.1f%%\n", comp_ratio, savings_pct);
    printf("   Encode Latency     %-15.2f ms\n", t_compress);
    printf("   Decode Latency     %-15.2f ms\n", t_decompress);
    printf("   -------------------------------------------------------------\n\n");

    rcre_arena_destroy(arena);
    printf("🎉 PCA Image Compression & Quantization Demo Completed Successfully!\n");
}

static void run_benchmark(const char* image_path, uint32_t block_size) {
    print_banner();
    printf("⚡ Running Comprehensive Rate-Distortion Benchmark (PSNR vs Bitrate)\n\n");

    RCRE_Arena* arena = rcre_arena_create(128 * 1024 * 1024);
    if (!arena) return;

    RCRE_Tensor orig_img;
    if (image_path && strlen(image_path) > 0) {
        printf("Loading input image: %s\n", image_path);
        if (rcre_load_image(image_path, &orig_img, arena) != 0) {
            fprintf(stderr, "Failed to load %s, falling back to synthetic image\n", image_path);
            rcre_generate_test_image(&orig_img, 256, 256, arena);
        }
    } else {
        printf("Using synthetic 256x256 test image\n");
        rcre_generate_test_image(&orig_img, 256, 256, arena);
    }

    size_t H = orig_img.shape[0];
    size_t W = orig_img.shape[1];
    size_t C = orig_img.shape[2];
    size_t uncompressed_bytes = H * W * C;

    printf("Image Dimensions: %zux%zux%zu | Uncompressed Payload: %zu bytes (%.1f KB)\n\n",
           H, W, C, uncompressed_bytes, (double)uncompressed_bytes / 1024.0);

    printf("-----------------------------------------------------------------------------------------------------\n");
    printf("  K Comp   Patch Dim   Comp Size (B)   Ratio   Savings %%     MSE        PSNR (dB)   Enc (ms)   Dec (ms)\n");
    printf("-----------------------------------------------------------------------------------------------------\n");

    uint32_t k_values[] = {2, 4, 8, 12, 16, 24, 32, 48, 64};
    size_t num_k = sizeof(k_values) / sizeof(k_values[0]);

    for (size_t i = 0; i < num_k; ++i) {
        uint32_t k = k_values[i];
        if (k > block_size * block_size * C) continue;

        RCRE_Arena* run_arena = rcre_arena_create(32 * 1024 * 1024);

        double t0 = get_time_ms();
        RCRE_PCACompressedImage comp;
        int status = rcre_pca_compress(&orig_img, block_size, k, &comp, run_arena);
        if (status != RCRE_OK) {
            rcre_arena_destroy(run_arena);
            continue;
        }
        double enc_time = get_time_ms() - t0;

        t0 = get_time_ms();
        size_t out_shape[3] = {H, W, C};
        RCRE_Tensor recon = rcre_tensor_create(out_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, run_arena);
        status = rcre_pca_decompress(&comp, &recon, run_arena);
        if (status != RCRE_OK) {
            rcre_arena_destroy(run_arena);
            continue;
        }
        double dec_time = get_time_ms() - t0;

        float mse = 0.0f, psnr = 0.0f;
        (void)rcre_pca_compute_metrics(&orig_img, &recon, &mse, &psnr);

        size_t c_bytes = rcre_pca_compressed_byte_size(&comp);
        double ratio = (double)uncompressed_bytes / (double)c_bytes;
        double savings = (1.0 - (double)c_bytes / (double)uncompressed_bytes) * 100.0;

        printf("  %-6u   %-9u   %-13zu   %-5.2fx   %-8.1f%%   %-9.2f   %-9.2f   %-8.2f   %-8.2f\n",
               k, comp.patch_dim, c_bytes, ratio, savings, mse, psnr, enc_time, dec_time);

        rcre_arena_destroy(run_arena);
    }

    printf("-----------------------------------------------------------------------------------------------------\n\n");
    rcre_arena_destroy(arena);
}

int main(int argc, char** argv) {
    if (argc < 2 || strcmp(argv[1], "--demo") == 0) {
        run_demo();
        return 0;
    }

    if (strcmp(argv[1], "--benchmark") == 0) {
        const char* img_path = (argc > 2) ? argv[2] : NULL;
        uint32_t b = 8;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) b = (uint32_t)atoi(argv[++i]);
        }
        run_benchmark(img_path, b);
        return 0;
    }

    if (strcmp(argv[1], "--compress") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s --compress <input> <output.rpca> [-k 12] [-b 8]\n", argv[0]);
            return 1;
        }
        const char* in_path = argv[2];
        const char* out_path = argv[3];
        uint32_t k = 12;
        uint32_t b = 8;

        for (int i = 4; i < argc; ++i) {
            if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) k = (uint32_t)atoi(argv[++i]);
            if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) b = (uint32_t)atoi(argv[++i]);
        }

        RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024);
        RCRE_Tensor img;
        if (rcre_load_image(in_path, &img, arena) != 0) {
            fprintf(stderr, "Error loading image: %s\n", in_path);
            rcre_arena_destroy(arena);
            return 1;
        }

        RCRE_PCACompressedImage comp;
        int status = rcre_pca_compress(&img, b, k, &comp, arena);
        if (status != RCRE_OK) {
            fprintf(stderr, "Compression failed: %d\n", status);
            rcre_arena_destroy(arena);
            return 1;
        }

        status = rcre_pca_save_file(out_path, &comp);
        if (status != RCRE_OK) {
            fprintf(stderr, "Failed to save .rpca to %s\n", out_path);
            rcre_arena_destroy(arena);
            return 1;
        }
        printf("✅ Compressed %s -> %s (K=%u, B=%u, Size=%zu bytes)\n",
               in_path, out_path, k, b, rcre_pca_compressed_byte_size(&comp));
        rcre_arena_destroy(arena);
        return 0;
    }

    if (strcmp(argv[1], "--decompress") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s --decompress <input.rpca> <output.bmp>\n", argv[0]);
            return 1;
        }
        const char* in_path = argv[2];
        const char* out_path = argv[3];

        RCRE_Arena* arena = rcre_arena_create(64 * 1024 * 1024);
        RCRE_PCACompressedImage comp;
        if (rcre_pca_load_file(in_path, &comp, arena) != 0) {
            fprintf(stderr, "Error loading .rpca: %s\n", in_path);
            rcre_arena_destroy(arena);
            return 1;
        }

        size_t shape[3] = {comp.height, comp.width, comp.channels};
        RCRE_Tensor recon = rcre_tensor_create(shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
        int status = rcre_pca_decompress(&comp, &recon, arena);
        if (status != RCRE_OK) {
            fprintf(stderr, "Decompression failed: %d\n", status);
            rcre_arena_destroy(arena);
            return 1;
        }
        if (rcre_save_bmp(out_path, &recon) != 0) {
            fprintf(stderr, "Failed to save BMP output to %s\n", out_path);
        }
        printf("✅ Decompressed %s -> %s (%ux%u, %u channels)\n",
               in_path, out_path, comp.width, comp.height, comp.channels);

        rcre_arena_destroy(arena);
        return 0;
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    fprintf(stderr, "Commands: --demo, --benchmark [img], --compress <in> <out.rpca>, --decompress <in.rpca> <out.bmp>\n");
    return 1;
}
