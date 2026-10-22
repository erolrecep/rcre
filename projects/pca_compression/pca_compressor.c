/**
 * @file pca_compressor.c
 * @brief Implementation of C23 PCA Image Compression & Quantization Engine
 * 
 * Part of RCRE (Recep's Code Run Everywhere)
 */

#include "pca_compressor.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

[[nodiscard]] int rcre_pca_compress(const RCRE_Tensor* img, uint32_t block_size, uint32_t k,
                                    RCRE_PCACompressedImage* out_comp, RCRE_Arena* arena) {
    if (!img || !img->data || !out_comp || !arena || block_size == 0 || k == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (img->ndim != 3 || img->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t H = img->shape[0];
    size_t W = img->shape[1];
    size_t C = img->shape[2];

    uint32_t n_patches_x = (uint32_t)((W + block_size - 1) / block_size);
    uint32_t n_patches_y = (uint32_t)((H + block_size - 1) / block_size);
    uint32_t N = n_patches_x * n_patches_y;
    uint32_t D = block_size * block_size * (uint32_t)C;

    if (k > D) {
        k = D;
    }

    out_comp->magic = RCRE_PCA_MAGIC;
    out_comp->width = (uint32_t)W;
    out_comp->height = (uint32_t)H;
    out_comp->channels = (uint32_t)C;
    out_comp->block_size = block_size;
    out_comp->num_components = k;
    out_comp->quant_bits = 8;
    out_comp->num_patches_x = n_patches_x;
    out_comp->num_patches_y = n_patches_y;
    out_comp->num_patches = N;
    out_comp->patch_dim = D;

    /* 1. Allocate Patch Matrix X [N x D] in arena */
    size_t X_shape[2] = {N, D};
    RCRE_Tensor X_tensor = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!X_tensor.data) return RCRE_ERR_OUT_OF_MEMORY;

    float* X_data = (float*)X_tensor.data;
    const float* img_data = (const float*)img->data;

    /* Extract patches with reflection/clamping at borders */
    size_t patch_idx = 0;
    for (uint32_t py = 0; py < n_patches_y; ++py) {
        for (uint32_t px = 0; px < n_patches_x; ++px) {
            float* patch_row = &X_data[patch_idx * D];
            size_t el_idx = 0;

            for (uint32_t ly = 0; ly < block_size; ++ly) {
                size_t y = py * block_size + ly;
                if (y >= H) y = H - 1; /* clamp */

                for (uint32_t lx = 0; lx < block_size; ++lx) {
                    size_t x = px * block_size + lx;
                    if (x >= W) x = W - 1; /* clamp */

                    const float* pixel = &img_data[(y * W + x) * C];
                    for (size_t c = 0; c < C; ++c) {
                        patch_row[el_idx++] = pixel[c];
                    }
                }
            }
            patch_idx++;
        }
    }

    /* 2. Allocate PCA Components [K x D] and Mean [D] */
    size_t W_shape[2] = {k, D};
    RCRE_Tensor comp_tensor = rcre_tensor_create(W_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    size_t M_shape[2] = {D, 1};
    RCRE_Tensor mean_tensor = rcre_tensor_create(M_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!comp_tensor.data || !mean_tensor.data) return RCRE_ERR_OUT_OF_MEMORY;

    /* 3. Fit PCA on Patch Matrix */
    int status = rcre_ml_pca_fit(&X_tensor, k, &comp_tensor, &mean_tensor, arena);
    if (status != RCRE_OK) return status;

    /* 4. Project Patches onto Principal Components: X_proj [N x K] */
    size_t P_shape[2] = {N, k};
    RCRE_Tensor proj_tensor = rcre_tensor_create(P_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!proj_tensor.data) return RCRE_ERR_OUT_OF_MEMORY;

    status = rcre_ml_pca_transform(&X_tensor, &comp_tensor, &mean_tensor, &proj_tensor);
    if (status != RCRE_OK) return status;

    /* 5. Quantize Coefficients to 8-bit integers with per-component min/max scales */
    out_comp->mean = (float*)rcre_arena_alloc(arena, D * sizeof(float), 64);
    out_comp->components = (float*)rcre_arena_alloc(arena, k * D * sizeof(float), 64);
    out_comp->scale_min = (float*)rcre_arena_alloc(arena, k * sizeof(float), 64);
    out_comp->scale_max = (float*)rcre_arena_alloc(arena, k * sizeof(float), 64);
    out_comp->quant_data = (uint8_t*)rcre_arena_alloc(arena, N * k * sizeof(uint8_t), 64);

    if (!out_comp->mean || !out_comp->components || !out_comp->scale_min || 
        !out_comp->scale_max || !out_comp->quant_data) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    memcpy(out_comp->mean, mean_tensor.data, D * sizeof(float));
    memcpy(out_comp->components, comp_tensor.data, k * D * sizeof(float));

    const float* P_data = (const float*)proj_tensor.data;

    for (uint32_t j = 0; j < k; ++j) {
        float min_val = P_data[0 * k + j];
        float max_val = min_val;

        for (uint32_t i = 1; i < N; ++i) {
            float v = P_data[i * k + j];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        out_comp->scale_min[j] = min_val;
        out_comp->scale_max[j] = max_val;

        float range = max_val - min_val;
        if (range < 1e-8f) range = 1.0f;
        float inv_range = 255.0f / range;

        for (uint32_t i = 0; i < N; ++i) {
            float v = P_data[i * k + j];
            float q = (v - min_val) * inv_range;
            if (q < 0.0f) q = 0.0f;
            if (q > 255.0f) q = 255.0f;
            out_comp->quant_data[i * k + j] = (uint8_t)(q + 0.5f);
        }
    }

    return RCRE_OK;
}

[[nodiscard]] int rcre_pca_decompress(const RCRE_PCACompressedImage* comp, RCRE_Tensor* out_img,
                                      RCRE_Arena* arena) {
    if (!comp || !out_img || !out_img->data || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (comp->magic != RCRE_PCA_MAGIC) {
        return RCRE_ERR_INVALID_ARG;
    }

    uint32_t W = comp->width;
    uint32_t H = comp->height;
    uint32_t C = comp->channels;
    uint32_t B = comp->block_size;
    uint32_t K = comp->num_components;
    uint32_t N = comp->num_patches;
    uint32_t D = comp->patch_dim;

    if (out_img->ndim != 3 || out_img->shape[0] != H || out_img->shape[1] != W || out_img->shape[2] != C) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    /* 1. Dequantize coefficients back to float32 X_proj [N x K] */
    size_t P_shape[2] = {N, K};
    RCRE_Tensor proj_tensor = rcre_tensor_create(P_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!proj_tensor.data) return RCRE_ERR_OUT_OF_MEMORY;

    float* P_data = (float*)proj_tensor.data;
    for (uint32_t j = 0; j < K; ++j) {
        float min_val = comp->scale_min[j];
        float max_val = comp->scale_max[j];
        float range = (max_val - min_val) / 255.0f;

        for (uint32_t i = 0; i < N; ++i) {
            uint8_t q = comp->quant_data[i * K + j];
            P_data[i * K + j] = min_val + ((float)q * range);
        }
    }

    /* 2. Wrap components and mean in RCRE_Tensor */
    size_t W_shape[2] = {K, D};
    RCRE_Tensor comp_tensor = rcre_tensor_create(W_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    memcpy(comp_tensor.data, comp->components, K * D * sizeof(float));

    size_t M_shape[2] = {D, 1};
    RCRE_Tensor mean_tensor = rcre_tensor_create(M_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    memcpy(mean_tensor.data, comp->mean, D * sizeof(float));

    /* 3. Reconstruct Patches via Inverse PCA Transform: X_recon [N x D] */
    size_t X_shape[2] = {N, D};
    RCRE_Tensor X_recon = rcre_tensor_create(X_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!X_recon.data) return RCRE_ERR_OUT_OF_MEMORY;

    int status = rcre_ml_pca_inverse_transform(&proj_tensor, &comp_tensor, &mean_tensor, &X_recon);
    if (status != RCRE_OK) return status;

    /* 4. Assemble Patches back into final image canvas */
    float* out_data = (float*)out_img->data;
    const float* X_data = (const float*)X_recon.data;

    uint32_t patch_idx = 0;
    for (uint32_t py = 0; py < comp->num_patches_y; ++py) {
        for (uint32_t px = 0; px < comp->num_patches_x; ++px) {
            const float* patch_row = &X_data[patch_idx * D];
            size_t el_idx = 0;

            for (uint32_t ly = 0; ly < B; ++ly) {
                size_t y = py * B + ly;
                if (y >= H) continue;

                for (uint32_t lx = 0; lx < B; ++lx) {
                    size_t x = px * B + lx;
                    if (x >= W) {
                        el_idx += C;
                        continue;
                    }

                    float* pixel = &out_data[(y * W + x) * C];
                    for (size_t c = 0; c < C; ++c) {
                        float val = patch_row[el_idx++];
                        /* Clamp pixel values to valid range */
                        if (val < 0.0f) val = 0.0f;
                        if (val > 255.0f) val = 255.0f;
                        pixel[c] = val;
                    }
                }
            }
            patch_idx++;
        }
    }

    return RCRE_OK;
}

[[nodiscard]] int rcre_pca_compute_metrics(const RCRE_Tensor* original, const RCRE_Tensor* reconstructed,
                                           float* out_mse, float* out_psnr) {
    if (!original || !original->data || !reconstructed || !reconstructed->data || !out_mse || !out_psnr) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t total_elements = 1;
    for (size_t i = 0; i < original->ndim; ++i) {
        total_elements *= original->shape[i];
    }

    const float* orig = (const float*)original->data;
    const float* recon = (const float*)reconstructed->data;

    double sse = 0.0;
    for (size_t i = 0; i < total_elements; ++i) {
        double diff = (double)orig[i] - (double)recon[i];
        sse += diff * diff;
    }

    float mse = (float)(sse / (double)total_elements);
    *out_mse = mse;

    if (mse <= 1e-12f) {
        *out_psnr = 99.0f; /* Near lossless */
    } else {
        *out_psnr = (float)(10.0 * log10((255.0 * 255.0) / (double)mse));
    }

    return RCRE_OK;
}

[[nodiscard]] size_t rcre_pca_compressed_byte_size(const RCRE_PCACompressedImage* comp) {
    if (!comp) return 0;
    size_t header_size = 11 * sizeof(uint32_t);
    size_t mean_size = comp->patch_dim * sizeof(float);
    size_t comp_size = comp->num_components * comp->patch_dim * sizeof(float);
    size_t scales_size = 2 * comp->num_components * sizeof(float);
    size_t quant_size = comp->num_patches * comp->num_components * sizeof(uint8_t);
    return header_size + mean_size + comp_size + scales_size + quant_size;
}

[[nodiscard]] int rcre_pca_save_file(const char* filepath, const RCRE_PCACompressedImage* comp) {
    if (!filepath || !comp) return RCRE_ERR_INVALID_ARG;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return RCRE_ERR_IO;

    uint32_t header[11] = {
        comp->magic,
        comp->width,
        comp->height,
        comp->channels,
        comp->block_size,
        comp->num_components,
        comp->quant_bits,
        comp->num_patches_x,
        comp->num_patches_y,
        comp->num_patches,
        comp->patch_dim
    };

    if (fwrite(header, sizeof(uint32_t), 11, fp) != 11) {
        fclose(fp);
        return RCRE_ERR_IO;
    }

    uint32_t D = comp->patch_dim;
    uint32_t K = comp->num_components;
    uint32_t N = comp->num_patches;

    fwrite(comp->mean, sizeof(float), D, fp);
    fwrite(comp->components, sizeof(float), K * D, fp);
    fwrite(comp->scale_min, sizeof(float), K, fp);
    fwrite(comp->scale_max, sizeof(float), K, fp);
    fwrite(comp->quant_data, sizeof(uint8_t), N * K, fp);

    fclose(fp);
    return RCRE_OK;
}

[[nodiscard]] int rcre_pca_load_file(const char* filepath, RCRE_PCACompressedImage* comp,
                                     RCRE_Arena* arena) {
    if (!filepath || !comp || !arena) return RCRE_ERR_INVALID_ARG;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return RCRE_ERR_IO;

    uint32_t header[11];
    if (fread(header, sizeof(uint32_t), 11, fp) != 11) {
        fclose(fp);
        return RCRE_ERR_IO;
    }

    if (header[0] != RCRE_PCA_MAGIC) {
        fclose(fp);
        return RCRE_ERR_INVALID_ARG;
    }

    comp->magic          = header[0];
    comp->width          = header[1];
    comp->height         = header[2];
    comp->channels       = header[3];
    comp->block_size     = header[4];
    comp->num_components = header[5];
    comp->quant_bits     = header[6];
    comp->num_patches_x  = header[7];
    comp->num_patches_y  = header[8];
    comp->num_patches    = header[9];
    comp->patch_dim      = header[10];

    uint32_t D = comp->patch_dim;
    uint32_t K = comp->num_components;
    uint32_t N = comp->num_patches;

    comp->mean        = (float*)rcre_arena_alloc(arena, D * sizeof(float), 64);
    comp->components  = (float*)rcre_arena_alloc(arena, K * D * sizeof(float), 64);
    comp->scale_min   = (float*)rcre_arena_alloc(arena, K * sizeof(float), 64);
    comp->scale_max   = (float*)rcre_arena_alloc(arena, K * sizeof(float), 64);
    comp->quant_data  = (uint8_t*)rcre_arena_alloc(arena, N * K * sizeof(uint8_t), 64);

    if (!comp->mean || !comp->components || !comp->scale_min || !comp->scale_max || !comp->quant_data) {
        fclose(fp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    fread(comp->mean, sizeof(float), D, fp);
    fread(comp->components, sizeof(float), K * D, fp);
    fread(comp->scale_min, sizeof(float), K, fp);
    fread(comp->scale_max, sizeof(float), K, fp);
    fread(comp->quant_data, sizeof(uint8_t), N * K, fp);

    fclose(fp);
    return RCRE_OK;
}
