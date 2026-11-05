/**
 * @file hash_table.c
 * @brief Arena-Backed Open-Addressing Hash Table Implementation
 */

#include "hash_table.h"
#include <string.h>

/* Murmur3 64-bit finalizer mix */
static inline uint64_t hash_mix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

static size_t next_power_of_two(size_t v) {
    if (v < 16) return 16;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

RCRE_NODISCARD RCRE_HashBuilder* rcre_hash_builder_create(size_t initial_capacity, RCRE_Arena* arena) {
    if (!arena) return NULL;
    size_t cap = next_power_of_two(initial_capacity < 64 ? 64 : initial_capacity);

    RCRE_HashBuilder* hb = (RCRE_HashBuilder*)rcre_arena_alloc(arena, sizeof(RCRE_HashBuilder), 16);
    if (!hb) return NULL;

    hb->entries = (RCRE_HashBuilderEntry*)rcre_arena_alloc(arena, cap * sizeof(RCRE_HashBuilderEntry), 64);
    if (!hb->entries) return NULL;

    for (size_t i = 0; i < cap; ++i) {
        hb->entries[i].key = RCRE_HASH_EMPTY_KEY;
        hb->entries[i].count = 0;
        hb->entries[i].head = NULL;
    }

    hb->capacity = cap;
    hb->size = 0;
    hb->mask = cap - 1;
    hb->total_elements = 0;
    return hb;
}

RCRE_NODISCARD int rcre_hash_builder_add(RCRE_HashBuilder* hb, uint64_t key, uint32_t val, RCRE_Arena* arena) {
    if (!hb || !arena) return RCRE_ERR_INVALID_ARG;

    /* Rehash if load factor > 70% */
    if (hb->size * 10 >= hb->capacity * 7) {
        size_t new_cap = hb->capacity * 2;
        RCRE_HashBuilderEntry* new_entries = (RCRE_HashBuilderEntry*)rcre_arena_alloc(
            arena, new_cap * sizeof(RCRE_HashBuilderEntry), 64);
        if (!new_entries) return RCRE_ERR_OUT_OF_MEMORY;

        for (size_t i = 0; i < new_cap; ++i) {
            new_entries[i].key = RCRE_HASH_EMPTY_KEY;
            new_entries[i].count = 0;
            new_entries[i].head = NULL;
        }

        size_t new_mask = new_cap - 1;
        for (size_t i = 0; i < hb->capacity; ++i) {
            if (hb->entries[i].key != RCRE_HASH_EMPTY_KEY) {
                uint64_t k = hb->entries[i].key;
                size_t slot = (size_t)(hash_mix64(k) & new_mask);
                while (new_entries[slot].key != RCRE_HASH_EMPTY_KEY) {
                    slot = (slot + 1) & new_mask;
                }
                new_entries[slot] = hb->entries[i];
            }
        }
        hb->entries = new_entries;
        hb->capacity = new_cap;
        hb->mask = new_mask;
    }

    /* Locate or create entry */
    size_t slot = (size_t)(hash_mix64(key) & hb->mask);
    while (hb->entries[slot].key != RCRE_HASH_EMPTY_KEY && hb->entries[slot].key != key) {
        slot = (slot + 1) & hb->mask;
    }

    if (hb->entries[slot].key == RCRE_HASH_EMPTY_KEY) {
        hb->entries[slot].key = key;
        hb->entries[slot].count = 0;
        hb->entries[slot].head = NULL;
        hb->size++;
    }

    /* Append node to linked list for this bucket */
    RCRE_HashNode* node = (RCRE_HashNode*)rcre_arena_alloc(arena, sizeof(RCRE_HashNode), 8);
    if (!node) return RCRE_ERR_OUT_OF_MEMORY;

    node->val = val;
    node->next = hb->entries[slot].head;
    hb->entries[slot].head = node;
    hb->entries[slot].count++;
    hb->total_elements++;

    return RCRE_OK;
}

RCRE_NODISCARD RCRE_HashTable* rcre_hash_builder_finalize(RCRE_HashBuilder* hb, RCRE_Arena* arena) {
    if (!hb || !arena) return NULL;

    RCRE_HashTable* ht = (RCRE_HashTable*)rcre_arena_alloc(arena, sizeof(RCRE_HashTable), 16);
    if (!ht) return NULL;

    /* Allocate final table with load factor ~50% for fast lookup */
    size_t cap = next_power_of_two(hb->size * 2 < 64 ? 64 : hb->size * 2);
    ht->buckets = (RCRE_HashBucket*)rcre_arena_alloc(arena, cap * sizeof(RCRE_HashBucket), 64);
    if (!ht->buckets) return NULL;

    for (size_t i = 0; i < cap; ++i) {
        ht->buckets[i].key = RCRE_HASH_EMPTY_KEY;
        ht->buckets[i].postings_offset = 0;
        ht->buckets[i].postings_count = 0;
    }

    ht->capacity = cap;
    ht->mask = cap - 1;
    ht->size = hb->size;
    ht->postings_capacity = hb->total_elements;
    ht->postings_size = hb->total_elements;

    /* Allocate single contiguous postings array */
    ht->postings = (uint32_t*)rcre_arena_alloc(arena, hb->total_elements * sizeof(uint32_t), 64);
    if (!ht->postings && hb->total_elements > 0) return NULL;

    /* Flatten all linked lists into contiguous buffer */
    uint32_t curr_offset = 0;
    for (size_t i = 0; i < hb->capacity; ++i) {
        if (hb->entries[i].key != RCRE_HASH_EMPTY_KEY) {
            uint64_t key = hb->entries[i].key;
            uint32_t count = hb->entries[i].count;

            /* Find slot in final hash table */
            size_t slot = (size_t)(hash_mix64(key) & ht->mask);
            while (ht->buckets[slot].key != RCRE_HASH_EMPTY_KEY) {
                slot = (slot + 1) & ht->mask;
            }

            ht->buckets[slot].key = key;
            ht->buckets[slot].postings_offset = curr_offset;
            ht->buckets[slot].postings_count = count;

            /* Copy IDs into contiguous postings array */
            RCRE_HashNode* curr = hb->entries[i].head;
            size_t pos = curr_offset + count;
            while (curr) {
                pos--;
                ht->postings[pos] = curr->val;
                curr = curr->next;
            }
            curr_offset += count;
        }
    }

    return ht;
}

RCRE_NODISCARD const uint32_t* rcre_hash_table_lookup(const RCRE_HashTable* ht, uint64_t key, uint32_t* out_count) {
    if (!ht || !out_count || key == RCRE_HASH_EMPTY_KEY) {
        if (out_count) *out_count = 0;
        return NULL;
    }

    size_t slot = (size_t)(hash_mix64(key) & ht->mask);
    size_t start_slot = slot;

    while (ht->buckets[slot].key != RCRE_HASH_EMPTY_KEY) {
        if (ht->buckets[slot].key == key) {
            *out_count = ht->buckets[slot].postings_count;
            return &ht->postings[ht->buckets[slot].postings_offset];
        }
        slot = (slot + 1) & ht->mask;
        if (slot == start_slot) break;
    }

    *out_count = 0;
    return NULL;
}

