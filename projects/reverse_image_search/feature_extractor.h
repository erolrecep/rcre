/**
 * @file feature_extractor.h
 * @brief RCRE 64-Dimensional Spatial Pyramid Feature Extractor for Reverse Image Search
 * 
 * Extracts compact, highly-discriminative 64-dimensional visual embeddings combining
 * spatial color moments and Sobel edge energy fields.
 * 
 * Powered 100% by RCRE core CV kernels (Rec.601 gray, Gaussian blur, 3x3 Sobel)
 * with zero heap allocations during extraction.
 */

#ifndef RCRE_FEATURE_EXTRACTOR_H
#define RCRE_FEATURE_EXTRACTOR_H

#include "rcre.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Standard embedding dimensionality: 4x4 spatial grid * 4 features/cell = 64 */
#define RCRE_FEATURE_DIM 64

/** Canonical processing resolution (normalized square) */
#define RCRE_CANONICAL_SIZE 256

/** Number of spatial grid divisions along X and Y */
#define RCRE_GRID_CELLS_X 4
#define RCRE_GRID_CELLS_Y 4
#define RCRE_NUM_CELLS (RCRE_GRID_CELLS_X * RCRE_GRID_CELLS_Y)

/**
 * @brief Extract a 64-D normalized visual embedding from raw RGB24 pixel buffer.
 * 
 * @param rgb_pixels Pointer to raw interleaved RGB24 pixels [H, W, 3].
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param out_embedding Destination buffer (must have capacity for at least RCRE_FEATURE_DIM floats).
 * @param scratch_arena Scratchpad arena for intermediate tensor operations.
 * @return RCRE_OK (0) on success, or an RCRE error code on failure.
 */
RCRE_NODISCARD int rcre_extract_features_rgb(const uint8_t* rgb_pixels, 
                                            size_t width, 
                                            size_t height, 
                                            float* out_embedding, 
                                            RCRE_Arena* scratch_arena);

/**
 * @brief Load an image from disk (JPG, PNG, BMP) and extract its 64-D visual embedding.
 * 
 * @param filepath Path to the image file.
 * @param out_embedding Destination buffer (at least RCRE_FEATURE_DIM floats).
 * @param scratch_arena Scratchpad arena for intermediate tensor operations.
 * @return RCRE_OK (0) on success, or an RCRE error code on failure.
 */
RCRE_NODISCARD int rcre_extract_features_file(const char* filepath, 
                                             float* out_embedding, 
                                             RCRE_Arena* scratch_arena);

/**
 * @brief Compute cosine similarity between two normalized feature vectors in [-1.0, 1.0].
 * (Returns 1.0 for identical vectors, 0.0 for orthogonal vectors).
 */
float rcre_compute_cosine_similarity(const float* a, const float* b, size_t dim);

/**
 * @brief Compute Euclidean L2 distance between two feature vectors.
 */
float rcre_compute_l2_distance(const float* a, const float* b, size_t dim);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_FEATURE_EXTRACTOR_H */
