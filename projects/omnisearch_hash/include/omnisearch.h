/**
 * @file omnisearch.h
 * @brief Turnkey Vector Search & LSH Hash Engine for 30M Multimodal Items
 *
 * Implements:
 * 1. L2 Spherical Normalization for high-throughput Cosine/Euclidean equivalence.
 * 2. 64-bit Locality-Sensitive Hashing (LSH) via Random Hyperplane projections.
 * 3. Arena-backed Open-Addressing Hash Table for O(1) multi-probe candidate retrieval.
 * 4. SIMD-accelerated distance kernels (Hamming, MIPS dot product, Squared L2).
 * 5. Bounded Min-Heap Priority Queue for Top-K extraction.
 * 6. Zero heap allocations in query loops via RCRE_Arena.
 */

#ifndef RCRE_OMNISEARCH_H
#define RCRE_OMNISEARCH_H

#include "rcre.h"
#include "hash_table.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OMNI_MAGIC "RCRE_LSH"
#define OMNI_VERSION 1
#define OMNI_DEFAULT_DIM 256
#define OMNI_DEFAULT_HYPERPLANES 64
#define OMNI_DEFAULT_TOP_K 20

typedef enum {
    OMNI_ITEM_TEXT  = 0,
    OMNI_ITEM_IMAGE = 1
} OmniItemType;

/**
 * @brief Metadata for a single indexed item.
 */
typedef struct {
    uint32_t id;             /**< Unique item identifier (0 .. N-1) */
    uint8_t  type;           /**< OMNI_ITEM_TEXT or OMNI_ITEM_IMAGE */
    uint8_t  reserved[3];
    uint32_t str_offset;     /**< Byte offset into contiguous string pool */
    uint32_t str_len;        /**< Length of text sentence or image path */
    uint64_t hash_sig;       /**< Precomputed 64-bit LSH signature */
} OmniItemMeta;

/**
 * @brief Single search match entry.
 */
typedef struct {
    uint32_t id;             /**< Item identifier */
    float    score;          /**< Cosine similarity (1.0 = identical, 0.0 = orthogonal) */
    float    distance;       /**< Metric distance (1.0 - score or squared L2) */
    uint32_t hamming_dist;   /**< Hamming distance between query hash and item hash */
    OmniItemType type;       /**< Text or Image */
    const char*  payload;    /**< Text sentence or image file path */
} OmniSearchResult;

/**
 * @brief Main In-Memory or Mapped Vector Search Index.
 */
typedef struct {
    size_t       num_items;        /**< Total number of indexed items */
    size_t       dim;              /**< Vector embedding dimensionality (e.g. 256) */
    size_t       num_hyperplanes;  /**< Number of LSH hyperplanes (e.g. 64) */
    size_t       table_bits;       /**< Number of bits used for hash table key (e.g. 10-18) */
    float*       hyperplanes;      /**< Hyperplane projection matrix [num_hyperplanes * dim] */
    RCRE_HashTable* hash_table;    /**< Hash table mapping LSH hash -> postings list */
    float*       embeddings;       /**< Contiguous L2-normalized float vectors [num_items * dim] */
    OmniItemMeta* items;           /**< Item metadata array [num_items] */
    char*        string_pool;      /**< Contiguous text/URI pool */
    size_t       string_pool_size; /**< Total byte size of string pool */
    size_t       string_pool_cap;  /**< Capacity of string pool */
    RCRE_Arena*  arena;            /**< Host arena backing index memory */
} OmniIndex;

/* ========================================================================= */
/* 1. Vector Normalization Algorithms                                         */
/* ========================================================================= */

/**
 * @brief In-place L2 Spherical Normalization: v = v / sqrt(sum(v_i^2) + eps).
 * Guarantees ||v||_2 == 1.0 within float precision.
 */
RCRE_NODISCARD int omni_normalize_l2(float* vec, size_t dim);

/**
 * @brief Computes vector norm ||v||_2.
 */
RCRE_NODISCARD float omni_vector_norm_l2(const float* vec, size_t dim);

/* ========================================================================= */
/* 2. LSH Hyperplane Projection & Hashing                                    */
/* ========================================================================= */

/**
 * @brief Initializes random Gaussian hyperplanes using deterministic PRNG.
 */
RCRE_NODISCARD int omni_lsh_init_hyperplanes(float* hyperplanes, size_t num_hyperplanes, 
                                             size_t dim, uint64_t seed);

/**
 * @brief Projects a normalized vector onto hyperplanes and extracts a 64-bit signature.
 */
RCRE_NODISCARD uint64_t omni_lsh_compute_hash(const float* vec, const float* hyperplanes, 
                                              size_t num_hyperplanes, size_t dim);

/* ========================================================================= */
/* 3. Distance & Similarity Kernels                                          */
/* ========================================================================= */

/**
 * @brief Hardware-accelerated bitwise Hamming distance between two 64-bit signatures.
 */
RCRE_NODISCARD static inline uint32_t omni_hamming_distance64(uint64_t a, uint64_t b) {
    return (uint32_t)__builtin_popcountll(a ^ b);
}

/**
 * @brief Cosine similarity between two unit-normalized vectors: <a, b>.
 * Executes via RCRE HAL SIMD (ARM NEON / AVX2).
 */
RCRE_NODISCARD float omni_cosine_similarity(const float* a, const float* b, size_t dim);

/**
 * @brief Direct Squared Euclidean Distance: ||a - b||^2.
 */
RCRE_NODISCARD float omni_squared_l2_distance(const float* a, const float* b, size_t dim);

/* ========================================================================= */
/* 4. Index Lifecycle & Disk Persistence                                     */
/* ========================================================================= */

/**
 * @brief Allocates an empty OmniIndex in the given arena.
 */
RCRE_NODISCARD OmniIndex* omni_index_create(size_t dim, size_t num_hyperplanes, 
                                            size_t max_items, RCRE_Arena* arena);

/**
 * @brief Adds an item to the index during construction.
 * Normalizes vector and computes LSH signature.
 */
RCRE_NODISCARD int omni_index_add(OmniIndex* idx, uint32_t id, OmniItemType type, 
                                  const char* payload, const float* embedding);

/**
 * @brief Builds the final open-addressing hash table from all ingested items.
 */
RCRE_NODISCARD int omni_index_build_hash(OmniIndex* idx, RCRE_Arena* scratch);

/**
 * @brief Serializes the entire index to a portable .rcre_lsh binary file.
 */
RCRE_NODISCARD int omni_index_save(const OmniIndex* idx, const char* filepath);

/**
 * @brief Deserializes an index from a .rcre_lsh file into the specified arena.
 */
RCRE_NODISCARD OmniIndex* omni_index_load(const char* filepath, RCRE_Arena* arena);

/* ========================================================================= */
/* 5. Online Search Queries                                                  */
/* ========================================================================= */

/**
 * @brief Fast two-tier vector search using Hash Table + Hamming filter + SIMD Re-ranking.
 * @param idx Loaded index.
 * @param query_vec Raw or normalized query vector.
 * @param top_k Number of top results to return (e.g. 20).
 * @param multi_probe_radius Hamming probe radius (0 = exact bucket, 1 = 64 adjacent 1-bit buckets).
 * @param out_results Buffer to receive top-k results.
 * @param out_count Pointer to receive number of returned items.
 * @param scratch Scratchpad arena for zero-heap query allocations.
 */
RCRE_NODISCARD int omni_index_search(const OmniIndex* idx, const float* query_vec, 
                                     size_t top_k, size_t multi_probe_radius, 
                                     OmniSearchResult* out_results, size_t* out_count, 
                                     RCRE_Arena* scratch);

/**
 * @brief Brute-force exhaustive search across all items (ground-truth reference for recall parity).
 */
RCRE_NODISCARD int omni_exact_search(const OmniIndex* idx, const float* query_vec, 
                                     size_t top_k, OmniSearchResult* out_results, 
                                     size_t* out_count, RCRE_Arena* scratch);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_OMNISEARCH_H */
