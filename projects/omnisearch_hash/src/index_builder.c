/**
 * @file index_builder.c
 * @brief Index Construction, String Pool, and Binary Disk Persistence (.rcre_lsh)
 */

#include "omnisearch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

RCRE_NODISCARD OmniIndex* omni_index_create(size_t dim, size_t num_hyperplanes, 
                                            size_t max_items, RCRE_Arena* arena) {
    if (!arena || dim == 0 || num_hyperplanes == 0 || max_items == 0) {
        return NULL;
    }

    OmniIndex* idx = (OmniIndex*)rcre_arena_alloc(arena, sizeof(OmniIndex), 16);
    if (!idx) return NULL;

    idx->dim = dim;
    idx->num_hyperplanes = num_hyperplanes;
    idx->num_items = 0;
    idx->arena = arena;

    /* Auto-tune table bits based on max_items (e.g. 8 bits for 1000 items, up to 18 bits for 30M) */
    size_t bits = 8;
    while ((1ULL << (bits + 2)) <= max_items && bits < 18) {
        bits++;
    }
    idx->table_bits = bits;

    /* Allocate hyperplanes */
    idx->hyperplanes = (float*)rcre_arena_alloc(arena, num_hyperplanes * dim * sizeof(float), 64);
    if (!idx->hyperplanes) return NULL;

    /* Allocate item metadata array */
    idx->items = (OmniItemMeta*)rcre_arena_alloc(arena, max_items * sizeof(OmniItemMeta), 16);
    if (!idx->items) return NULL;

    /* Allocate embeddings matrix */
    idx->embeddings = (float*)rcre_arena_alloc(arena, max_items * dim * sizeof(float), 64);
    if (!idx->embeddings) return NULL;

    /* Allocate initial string pool (e.g. average 64 bytes per item) */
    idx->string_pool_cap = max_items * 64;
    idx->string_pool = (char*)rcre_arena_alloc(arena, idx->string_pool_cap, 16);
    idx->string_pool_size = 0;

    idx->hash_table = NULL;

    /* Initialize hyperplanes */
    if (omni_lsh_init_hyperplanes(idx->hyperplanes, num_hyperplanes, dim, 0xABCDEF0123456789ULL) != RCRE_OK) {
        return NULL;
    }

    return idx;
}

RCRE_NODISCARD int omni_index_add(OmniIndex* idx, uint32_t id, OmniItemType type, 
                                  const char* payload, const float* embedding) {
    if (!idx || !payload || !embedding) return RCRE_ERR_INVALID_ARG;

    size_t item_idx = idx->num_items;
    idx->items[item_idx].id = id;
    idx->items[item_idx].type = (uint8_t)type;

    /* Copy & normalize embedding */
    float* dst_vec = &idx->embeddings[item_idx * idx->dim];
    memcpy(dst_vec, embedding, idx->dim * sizeof(float));
    (void)omni_normalize_l2(dst_vec, idx->dim);

    /* Compute 64-bit LSH signature */
    idx->items[item_idx].hash_sig = omni_lsh_compute_hash(dst_vec, idx->hyperplanes, 
                                                           idx->num_hyperplanes, idx->dim);

    /* Append string payload to string pool */
    size_t p_len = strlen(payload);
    if (idx->string_pool_size + p_len + 1 > idx->string_pool_cap) {
        /* Allocate new string chunk from arena */
        size_t new_cap = (idx->string_pool_cap + p_len + 1) * 2;
        char* new_pool = (char*)rcre_arena_alloc(idx->arena, new_cap, 16);
        if (!new_pool) return RCRE_ERR_OUT_OF_MEMORY;
        memcpy(new_pool, idx->string_pool, idx->string_pool_size);
        idx->string_pool = new_pool;
        idx->string_pool_cap = new_cap;
    }

    uint32_t offset = (uint32_t)idx->string_pool_size;
    memcpy(&idx->string_pool[offset], payload, p_len);
    idx->string_pool[offset + p_len] = '\0';
    idx->string_pool_size += (p_len + 1);

    idx->items[item_idx].str_offset = offset;
    idx->items[item_idx].str_len = (uint32_t)p_len;

    idx->num_items++;
    return RCRE_OK;
}

RCRE_NODISCARD int omni_index_build_hash(OmniIndex* idx, RCRE_Arena* scratch) {
    if (!idx || idx->num_items == 0) return RCRE_ERR_INVALID_ARG;

    RCRE_Arena* build_arena = scratch ? scratch : idx->arena;
    RCRE_HashBuilder* hb = rcre_hash_builder_create(idx->num_items, build_arena);
    if (!hb) return RCRE_ERR_OUT_OF_MEMORY;

    uint64_t mask = (idx->table_bits >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << idx->table_bits) - 1);
    for (size_t i = 0; i < idx->num_items; ++i) {
        uint64_t key = idx->items[i].hash_sig & mask;
        int err = rcre_hash_builder_add(hb, key, (uint32_t)i, build_arena);
        if (err != RCRE_OK) return err;
    }

    idx->hash_table = rcre_hash_builder_finalize(hb, idx->arena);
    if (!idx->hash_table) return RCRE_ERR_OUT_OF_MEMORY;

    return RCRE_OK;
}

#pragma pack(push, 1)
typedef struct {
    char     magic[8];        /* "RCRE_LSH" */
    uint32_t version;         /* 1 */
    uint32_t dim;             /* 256 */
    uint32_t num_hyperplanes; /* 64 */
    uint32_t table_bits;      /* 10-18 */
    uint64_t num_items;       /* N */
    uint32_t num_buckets;     /* hash table capacity */
    uint32_t num_keys;        /* number of active keys */
    uint32_t num_postings;    /* total postings elements */
    uint64_t string_bytes;    /* size of string pool */
    uint8_t  reserved[8];
} OmniFileHeader;
#pragma pack(pop)

RCRE_NODISCARD int omni_index_save(const OmniIndex* idx, const char* filepath) {
    if (!idx || !filepath || !idx->hash_table) return RCRE_ERR_INVALID_ARG;

    FILE* f = fopen(filepath, "wb");
    if (!f) return RCRE_ERR_IO;

    OmniFileHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, OMNI_MAGIC, 8);
    hdr.version = OMNI_VERSION;
    hdr.dim = (uint32_t)idx->dim;
    hdr.num_hyperplanes = (uint32_t)idx->num_hyperplanes;
    hdr.table_bits = (uint32_t)idx->table_bits;
    hdr.num_items = (uint64_t)idx->num_items;
    hdr.num_buckets = (uint32_t)idx->hash_table->capacity;
    hdr.num_keys = (uint32_t)idx->hash_table->size;
    hdr.num_postings = (uint32_t)idx->hash_table->postings_size;
    hdr.string_bytes = (uint64_t)idx->string_pool_size;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return RCRE_ERR_IO; }

    /* Write hyperplanes */
    size_t hp_size = idx->num_hyperplanes * idx->dim;
    if (fwrite(idx->hyperplanes, sizeof(float), hp_size, f) != hp_size) {
        fclose(f); return RCRE_ERR_IO;
    }

    /* Write hash buckets */
    if (fwrite(idx->hash_table->buckets, sizeof(RCRE_HashBucket), idx->hash_table->capacity, f) != idx->hash_table->capacity) {
        fclose(f); return RCRE_ERR_IO;
    }

    /* Write postings array */
    if (fwrite(idx->hash_table->postings, sizeof(uint32_t), idx->hash_table->postings_size, f) != idx->hash_table->postings_size) {
        fclose(f); return RCRE_ERR_IO;
    }

    /* Write normalized embeddings */
    size_t emb_size = idx->num_items * idx->dim;
    if (fwrite(idx->embeddings, sizeof(float), emb_size, f) != emb_size) {
        fclose(f); return RCRE_ERR_IO;
    }

    /* Write item metadata */
    if (fwrite(idx->items, sizeof(OmniItemMeta), idx->num_items, f) != idx->num_items) {
        fclose(f); return RCRE_ERR_IO;
    }

    /* Write string pool */
    if (fwrite(idx->string_pool, 1, idx->string_pool_size, f) != idx->string_pool_size) {
        fclose(f); return RCRE_ERR_IO;
    }

    fclose(f);
    return RCRE_OK;
}

RCRE_NODISCARD OmniIndex* omni_index_load(const char* filepath, RCRE_Arena* arena) {
    if (!filepath || !arena) return NULL;

    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    OmniFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NULL; }

    if (memcmp(hdr.magic, OMNI_MAGIC, 8) != 0 || hdr.version != OMNI_VERSION) {
        fclose(f);
        return NULL;
    }

    OmniIndex* idx = (OmniIndex*)rcre_arena_alloc(arena, sizeof(OmniIndex), 16);
    if (!idx) { fclose(f); return NULL; }

    idx->dim = (size_t)hdr.dim;
    idx->num_hyperplanes = (size_t)hdr.num_hyperplanes;
    idx->table_bits = (size_t)hdr.table_bits;
    idx->num_items = (size_t)hdr.num_items;
    idx->arena = arena;

    /* Read hyperplanes */
    size_t hp_size = idx->num_hyperplanes * idx->dim;
    idx->hyperplanes = (float*)rcre_arena_alloc(arena, hp_size * sizeof(float), 64);
    if (!idx->hyperplanes || fread(idx->hyperplanes, sizeof(float), hp_size, f) != hp_size) {
        fclose(f); return NULL;
    }

    /* Read hash table */
    idx->hash_table = (RCRE_HashTable*)rcre_arena_alloc(arena, sizeof(RCRE_HashTable), 16);
    if (!idx->hash_table) { fclose(f); return NULL; }

    idx->hash_table->capacity = (size_t)hdr.num_buckets;
    idx->hash_table->mask = (size_t)hdr.num_buckets - 1;
    idx->hash_table->size = (size_t)hdr.num_keys;
    idx->hash_table->postings_capacity = (size_t)hdr.num_postings;
    idx->hash_table->postings_size = (size_t)hdr.num_postings;

    idx->hash_table->buckets = (RCRE_HashBucket*)rcre_arena_alloc(arena, hdr.num_buckets * sizeof(RCRE_HashBucket), 64);
    if (!idx->hash_table->buckets || 
        fread(idx->hash_table->buckets, sizeof(RCRE_HashBucket), hdr.num_buckets, f) != hdr.num_buckets) {
        fclose(f); return NULL;
    }

    idx->hash_table->postings = (uint32_t*)rcre_arena_alloc(arena, hdr.num_postings * sizeof(uint32_t), 64);
    if (!idx->hash_table->postings || 
        fread(idx->hash_table->postings, sizeof(uint32_t), hdr.num_postings, f) != hdr.num_postings) {
        fclose(f); return NULL;
    }

    /* Read embeddings */
    size_t emb_size = idx->num_items * idx->dim;
    idx->embeddings = (float*)rcre_arena_alloc(arena, emb_size * sizeof(float), 64);
    if (!idx->embeddings || fread(idx->embeddings, sizeof(float), emb_size, f) != emb_size) {
        fclose(f); return NULL;
    }

    /* Read metadata */
    idx->items = (OmniItemMeta*)rcre_arena_alloc(arena, idx->num_items * sizeof(OmniItemMeta), 16);
    if (!idx->items || fread(idx->items, sizeof(OmniItemMeta), idx->num_items, f) != idx->num_items) {
        fclose(f); return NULL;
    }

    /* Read string pool */
    idx->string_pool_size = (size_t)hdr.string_bytes;
    idx->string_pool_cap = idx->string_pool_size;
    idx->string_pool = (char*)rcre_arena_alloc(arena, idx->string_pool_size, 16);
    if (!idx->string_pool || fread(idx->string_pool, 1, idx->string_pool_size, f) != idx->string_pool_size) {
        fclose(f); return NULL;
    }

    fclose(f);
    return idx;
}
