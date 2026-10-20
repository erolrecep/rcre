/**
 * @file index_serializer.h
 * @brief Binary Serialization, Index Persistence & Fast Retrieval Engine for RCRE Lens (.rcre_idx)
 */

#ifndef RCRE_INDEX_SERIALIZER_H
#define RCRE_INDEX_SERIALIZER_H

#include "rcre.h"
#include "feature_extractor.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RCRE_IDX_MAGIC "RCRE_IDX"
#define RCRE_IDX_VERSION 1

/**
 * @brief Search match result for an image query
 */
typedef struct {
    size_t      image_id;
    const char* filepath;
    float       similarity;  /* Cosine similarity: 1.0 (100% match) to 0.0 */
    float       l2_distance; /* Euclidean L2 distance */
} RCRE_SearchResult;

/**
 * @brief In-memory Reverse Image Search Database
 */
typedef struct {
    char**            filepaths;       /* Array of image file paths */
    float*            embeddings;      /* Contiguous [num_images * feature_dim] floats */
    size_t            num_images;      /* Total number of indexed images */
    size_t            capacity;        /* Maximum allocated capacity */
    size_t            feature_dim;     /* Feature vector dimensionality (default 64) */
    RCRE_HNSWIndex*   hnsw;            /* High-speed proximity graph index */
    RCRE_Arena*       arena;           /* Arena where index data resides */
} RCRE_ImageIndex;

/**
 * @brief Create an empty in-memory image index.
 * 
 * @param capacity Maximum expected number of images.
 * @param feature_dim Dimensionality of feature vectors (e.g. 64).
 * @param arena Arena allocator for index structures.
 * @return RCRE_ImageIndex* or NULL on failure.
 */
RCRE_NODISCARD RCRE_ImageIndex* rcre_index_create(size_t capacity, 
                                                  size_t feature_dim, 
                                                  RCRE_Arena* arena);

/**
 * @brief Add an image and its feature embedding to the index.
 * 
 * @param index Target index.
 * @param filepath Path to the image file (copied into arena).
 * @param embedding Feature vector (RCRE_FEATURE_DIM floats).
 * @return RCRE_OK (0) on success, or error code.
 */
RCRE_NODISCARD int rcre_index_add(RCRE_ImageIndex* index, 
                                  const char* filepath, 
                                  const float* embedding);

/**
 * @brief Save the in-memory index to a binary .rcre_idx file on disk.
 * 
 * @param index Index to serialize.
 * @param output_filepath Destination file path (e.g., "photos.rcre_idx").
 * @return RCRE_OK (0) on success, or error code.
 */
RCRE_NODISCARD int rcre_index_save(const RCRE_ImageIndex* index, 
                                  const char* output_filepath);

/**
 * @brief Load an index from a binary .rcre_idx file into an arena and reconstruct HNSW graph.
 * 
 * @param input_filepath Source file path.
 * @param arena Arena for memory allocation.
 * @return RCRE_ImageIndex* or NULL on failure.
 */
RCRE_NODISCARD RCRE_ImageIndex* rcre_index_load(const char* input_filepath, 
                                                RCRE_Arena* arena);

/**
 * @brief Search top-K most visually similar images matching a query embedding.
 * 
 * @param index In-memory index.
 * @param query_embedding 64-D query feature vector.
 * @param top_k Maximum number of results to return.
 * @param out_results Array of capacity top_k to populate with ranked matches.
 * @param scratch_arena Scratchpad arena for query priority queues.
 * @return Number of results returned (<= top_k), or negative error code.
 */
int rcre_index_search(const RCRE_ImageIndex* index, 
                      const float* query_embedding, 
                      size_t top_k, 
                      RCRE_SearchResult* out_results, 
                      RCRE_Arena* scratch_arena);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_INDEX_SERIALIZER_H */
