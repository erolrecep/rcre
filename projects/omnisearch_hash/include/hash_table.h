/**
 * @file hash_table.h
 * @brief Arena-Backed Open-Addressing Hash Table for Vector Search & LSH Buckets
 *
 * Implements a cache-line conscious open-addressing hash table with linear probing.
 * Designed for 100% zero-heap allocation inside RCRE memory arenas.
 */

#ifndef RCRE_OMNIS_HASH_TABLE_H
#define RCRE_OMNIS_HASH_TABLE_H

#include "rcre.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RCRE_HASH_EMPTY_KEY 0xFFFFFFFFFFFFFFFFULL

/**
 * @brief Individual hash table entry mapping a 64-bit key to an inverted postings slice.
 */
typedef struct {
    uint64_t key;              /**< 64-bit LSH signature or bucket key */
    uint32_t postings_offset;  /**< Byte or index offset into contiguous postings buffer */
    uint32_t postings_count;   /**< Number of candidate vector IDs in this bucket */
} RCRE_HashBucket;

/**
 * @brief Open-addressing hash table container.
 */
typedef struct {
    RCRE_HashBucket* buckets;  /**< Power-of-two bucket array */
    size_t capacity;           /**< Total number of bucket slots (power of 2) */
    size_t size;               /**< Number of distinct active keys */
    size_t mask;               /**< capacity - 1 */
    uint32_t* postings;        /**< Flat contiguous array of vector IDs */
    size_t postings_capacity;  /**< Allocated postings capacity */
    size_t postings_size;      /**< Current postings count */
} RCRE_HashTable;

/**
 * @brief Temporary builder node for dynamic postings list accumulation before finalization.
 */
typedef struct RCRE_HashNode {
    uint32_t val;
    struct RCRE_HashNode* next;
} RCRE_HashNode;

typedef struct {
    uint64_t key;
    uint32_t count;
    RCRE_HashNode* head;
} RCRE_HashBuilderEntry;

typedef struct {
    RCRE_HashBuilderEntry* entries;
    size_t capacity;
    size_t size;
    size_t mask;
    size_t total_elements;
} RCRE_HashBuilder;

/**
 * @brief Allocates an open-addressing hash builder in the given arena.
 */
RCRE_NODISCARD RCRE_HashBuilder* rcre_hash_builder_create(size_t initial_capacity, RCRE_Arena* arena);

/**
 * @brief Inserts a key-value pair into the hash builder.
 */
RCRE_NODISCARD int rcre_hash_builder_add(RCRE_HashBuilder* hb, uint64_t key, uint32_t val, RCRE_Arena* arena);

/**
 * @brief Compacts builder entries into a contiguous, cache-aligned RCRE_HashTable.
 */
RCRE_NODISCARD RCRE_HashTable* rcre_hash_builder_finalize(RCRE_HashBuilder* hb, RCRE_Arena* arena);

/**
 * @brief Queries the hash table for all postings matching the given key.
 * @param ht Hash table to search.
 * @param key 64-bit key to look up.
 * @param out_count Pointer to receive the number of matching elements.
 * @return Pointer to contiguous array of uint32_t IDs, or NULL if key not found.
 */
RCRE_NODISCARD const uint32_t* rcre_hash_table_lookup(const RCRE_HashTable* ht, uint64_t key, uint32_t* out_count);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_OMNIS_HASH_TABLE_H */

