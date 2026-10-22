/**
 * @file pca_compressor.h
 * @brief Zero-Dependency C23 PCA Image Compression & Quantization Engine
 * 
 * Part of RCRE (Recep's Code Run Everywhere)
 */

#ifndef RCRE_PCA_COMPRESSOR_H
#define RCRE_PCA_COMPRESSOR_H

#include "rcre.h"
#include <stdint.h>
#include <stdbool.h>

#define RCRE_PCA_MAGIC 0x41435052 /* "RPCA" in Little Endian */

/**
 * @brief In-memory container representing a PCA-compressed quantized image.
 */
typedef struct RCRE_PCACompressedImage {
    uint32_t magic;          /* RCRE_PCA_MAGIC */
    uint32_t width;          /* Original image width in pixels */
    uint32_t height;         /* Original image height in pixels */
    uint32_t channels;       /* Number of color channels (1=Gray, 3=RGB) */
    uint32_t block_size;     /* Patch dimension B (e.g. 8 for 8x8 patches) */
    uint32_t num_components; /* Number of principal components K */
    uint32_t quant_bits;     /* Quantization bit depth (8 bits) */
    uint32_t num_patches_x;  /* Patches along width: ceil(W / B) */
    uint32_t num_patches_y;  /* Patches along height: ceil(H / B) */
    uint32_t num_patches;    /* Total patches N = num_patches_x * num_patches_y */
    uint32_t patch_dim;      /* Patch vector length D = B * B * channels */
    
    float*   mean;           /* Patch mean vector [patch_dim] */
    float*   components;     /* Principal eigenvectors [K * patch_dim] */
    float*   scale_min;      /* Per-component minimum value [K] */
    float*   scale_max;      /* Per-component maximum value [K] */
    uint8_t* quant_data;     /* 8-bit quantized projection coefficients [N * K] */
} RCRE_PCACompressedImage;

/**
 * @brief Compress an image tensor (HWC, float32, range [0, 255] or [0, 1]) using PCA.
 *
 * @param img Input image tensor (ndim=3, shape=[H, W, C], dtype=F32)
 * @param block_size Patch dimension B (e.g. 8)
 * @param k Number of principal components to retain (K <= B * B * C)
 * @param out_comp Pointer to compressed structure to populate
 * @param arena High-speed memory arena for allocations
 * @return RCRE_OK on success, error code otherwise
 */
[[nodiscard]] int rcre_pca_compress(const RCRE_Tensor* img, uint32_t block_size, uint32_t k,
                                    RCRE_PCACompressedImage* out_comp, RCRE_Arena* arena);

/**
 * @brief Decompress a PCACompressedImage into a reconstructed image tensor.
 *
 * @param comp Compressed image structure
 * @param out_img Pre-allocated image tensor (ndim=3, shape=[H, W, C], dtype=F32)
 * @param arena Scratchpad arena for dequantization and reconstruction
 * @return RCRE_OK on success, error code otherwise
 */
[[nodiscard]] int rcre_pca_decompress(const RCRE_PCACompressedImage* comp, RCRE_Tensor* out_img,
                                      RCRE_Arena* arena);

/**
 * @brief Compute objective reconstruction fidelity metrics (MSE and PSNR).
 *
 * @param original Original reference image tensor [H, W, C]
 * @param reconstructed Reconstructed decompressed image tensor [H, W, C]
 * @param out_mse Pointer to float receiving Mean Squared Error
 * @param out_psnr Pointer to float receiving Peak Signal-to-Noise Ratio (dB)
 * @return RCRE_OK on success, error code otherwise
 */
[[nodiscard]] int rcre_pca_compute_metrics(const RCRE_Tensor* original, const RCRE_Tensor* reconstructed,
                                           float* out_mse, float* out_psnr);

/**
 * @brief Serialize compressed representation to a standalone binary file (.rpca).
 */
[[nodiscard]] int rcre_pca_save_file(const char* filepath, const RCRE_PCACompressedImage* comp);

/**
 * @brief Deserialize a .rpca file into memory using an arena allocator.
 */
[[nodiscard]] int rcre_pca_load_file(const char* filepath, RCRE_PCACompressedImage* comp,
                                     RCRE_Arena* arena);

/**
 * @brief Calculate the compressed bitstream size in bytes.
 */
[[nodiscard]] size_t rcre_pca_compressed_byte_size(const RCRE_PCACompressedImage* comp);

#endif /* RCRE_PCA_COMPRESSOR_H */
