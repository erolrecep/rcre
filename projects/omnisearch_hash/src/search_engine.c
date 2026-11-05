/**
 * @file search_engine.c
 * @brief Multi-Probe LSH Hash Search & Bounded Min-Heap Top-K Extraction
 */

#include "omnisearch.h"
#include <string.h>

typedef struct {
    uint32_t id;
    float    score;
    uint32_t hamming_dist;
} HeapNode;

static void heap_sift_down(HeapNode* heap, size_t size, size_t idx) {
    while (true) {
        size_t left = 2 * idx + 1;
        size_t right = 2 * idx + 2;
        size_t smallest = idx;

        if (left < size && heap[left].score < heap[smallest].score) {
            smallest = left;
        }
        if (right < size && heap[right].score < heap[smallest].score) {
            smallest = right;
        }

        if (smallest != idx) {
            HeapNode tmp = heap[idx];
            heap[idx] = heap[smallest];
            heap[smallest] = tmp;
            idx = smallest;
        } else {
            break;
        }
    }
}

static void heap_sift_up(HeapNode* heap, size_t idx) {
    while (idx > 0) {
        size_t parent = (idx - 1) / 2;
        if (heap[idx].score < heap[parent].score) {
            HeapNode tmp = heap[idx];
            heap[idx] = heap[parent];
            heap[parent] = tmp;
            idx = parent;
        } else {
            break;
        }
    }
}

static void heap_push(HeapNode* heap, size_t* heap_size, size_t k, 
                      uint32_t id, float score, uint32_t hamming_dist) {
    if (*heap_size < k) {
        heap[*heap_size] = (HeapNode){ .id = id, .score = score, .hamming_dist = hamming_dist };
        (*heap_size)++;
        heap_sift_up(heap, *heap_size - 1);
    } else if (score > heap[0].score) {
        heap[0] = (HeapNode){ .id = id, .score = score, .hamming_dist = hamming_dist };
        heap_sift_down(heap, k, 0);
    }
}

/* Sorts min-heap in descending order for output */
static void heap_sort_descending(HeapNode* heap, size_t size) {
    for (size_t i = size; i > 1; --i) {
        HeapNode tmp = heap[0];
        heap[0] = heap[i - 1];
        heap[i - 1] = tmp;
        heap_sift_down(heap, i - 1, 0);
    }
}

RCRE_NODISCARD int omni_index_search(const OmniIndex* idx, const float* query_vec, 
                                     size_t top_k, size_t multi_probe_radius, 
                                     OmniSearchResult* out_results, size_t* out_count, 
                                     RCRE_Arena* scratch) {
    if (!idx || !query_vec || top_k == 0 || !out_results || !out_count || idx->num_items == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t saved_offset = scratch ? scratch->offset : 0;

    /* 1. Normalize query vector in scratch arena */
    float* q_norm = (float*)rcre_arena_alloc(scratch, idx->dim * sizeof(float), 64);
    if (!q_norm) return RCRE_ERR_OUT_OF_MEMORY;
    memcpy(q_norm, query_vec, idx->dim * sizeof(float));
    (void)omni_normalize_l2(q_norm, idx->dim);

    /* 2. Compute 64-bit query LSH signature */
    uint64_t q_hash = omni_lsh_compute_hash(q_norm, idx->hyperplanes, idx->num_hyperplanes, idx->dim);

    /* 3. Visited bitset to deduplicate multi-probe candidates */
    size_t bitset_bytes = (idx->num_items + 7) / 8;
    uint8_t* visited = (uint8_t*)rcre_arena_alloc(scratch, bitset_bytes, 8);
    if (!visited) return RCRE_ERR_OUT_OF_MEMORY;
    memset(visited, 0, bitset_bytes);

    /* 4. Min-Heap for Top-K extraction */
    HeapNode* heap = (HeapNode*)rcre_arena_alloc(scratch, top_k * sizeof(HeapNode), 16);
    if (!heap) return RCRE_ERR_OUT_OF_MEMORY;
    size_t heap_size = 0;

    /* 5. Generate probe list using table_bits */
    uint64_t mask = (idx->table_bits >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << idx->table_bits) - 1);
    uint64_t base_key = q_hash & mask;
    size_t num_bits = idx->table_bits;

    size_t max_probes = 1;
    if (multi_probe_radius >= 1) max_probes += num_bits;
    if (multi_probe_radius >= 2) max_probes += (num_bits * (num_bits - 1)) / 2;

    uint64_t* probe_keys = (uint64_t*)rcre_arena_alloc(scratch, max_probes * sizeof(uint64_t), 8);
    if (!probe_keys) return RCRE_ERR_OUT_OF_MEMORY;

    size_t num_probes = 0;
    probe_keys[num_probes++] = base_key;

    if (multi_probe_radius >= 1) {
        for (size_t b = 0; b < num_bits; ++b) {
            probe_keys[num_probes++] = base_key ^ (1ULL << b);
        }
    }
    if (multi_probe_radius >= 2) {
        for (size_t b1 = 0; b1 < num_bits; ++b1) {
            for (size_t b2 = b1 + 1; b2 < num_bits; ++b2) {
                probe_keys[num_probes++] = base_key ^ (1ULL << b1) ^ (1ULL << b2);
            }
        }
    }

    /* 6. Multi-probe candidate scanning */
    for (size_t p = 0; p < num_probes; ++p) {
        uint32_t count = 0;
        const uint32_t* postings = rcre_hash_table_lookup(idx->hash_table, probe_keys[p], &count);
        if (!postings || count == 0) continue;

        for (uint32_t c = 0; c < count; ++c) {
            uint32_t doc_id = postings[c];
            if (doc_id >= idx->num_items) continue;

            /* Check bitset */
            size_t byte_idx = doc_id / 8;
            uint8_t bit_mask = (uint8_t)(1 << (doc_id % 8));
            if (visited[byte_idx] & bit_mask) {
                continue; /* already evaluated */
            }
            visited[byte_idx] |= bit_mask;

            /* Fast Hamming distance check */
            uint32_t h_dist = omni_hamming_distance64(q_hash, idx->items[doc_id].hash_sig);

            /* Compute exact normalized Cosine similarity via SIMD */
            const float* doc_vec = &idx->embeddings[doc_id * idx->dim];
            float sim = omni_cosine_similarity(q_norm, doc_vec, idx->dim);

            heap_push(heap, &heap_size, top_k, doc_id, sim, h_dist);
        }
    }

    /* Fallback: if multi-probe yielded fewer than top_k items, scan additional candidates */
    if (heap_size < top_k && multi_probe_radius > 0) {
        for (size_t i = 0; i < idx->num_items && heap_size < top_k; ++i) {
            size_t byte_idx = i / 8;
            uint8_t bit_mask = (uint8_t)(1 << (i % 8));
            if (!(visited[byte_idx] & bit_mask)) {
                visited[byte_idx] |= bit_mask;
                uint32_t h_dist = omni_hamming_distance64(q_hash, idx->items[i].hash_sig);
                const float* doc_vec = &idx->embeddings[i * idx->dim];
                float sim = omni_cosine_similarity(q_norm, doc_vec, idx->dim);
                heap_push(heap, &heap_size, top_k, (uint32_t)i, sim, h_dist);
            }
        }
    }

    /* 7. Sort Top-K descending */
    heap_sort_descending(heap, heap_size);

    /* 8. Format results */
    for (size_t i = 0; i < heap_size; ++i) {
        uint32_t id = heap[i].id;
        out_results[i].id = id;
        out_results[i].score = heap[i].score;
        out_results[i].distance = 1.0f - heap[i].score;
        out_results[i].hamming_dist = heap[i].hamming_dist;
        out_results[i].type = (OmniItemType)idx->items[id].type;
        out_results[i].payload = &idx->string_pool[idx->items[id].str_offset];
    }

    *out_count = heap_size;

    /* Rewind scratch arena */
    if (scratch) scratch->offset = saved_offset;
    return RCRE_OK;
}

RCRE_NODISCARD int omni_exact_search(const OmniIndex* idx, const float* query_vec, 
                                     size_t top_k, OmniSearchResult* out_results, 
                                     size_t* out_count, RCRE_Arena* scratch) {
    if (!idx || !query_vec || top_k == 0 || !out_results || !out_count || idx->num_items == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t saved_offset = scratch ? scratch->offset : 0;

    /* Normalize query vector */
    float* q_norm = (float*)rcre_arena_alloc(scratch, idx->dim * sizeof(float), 64);
    if (!q_norm) return RCRE_ERR_OUT_OF_MEMORY;
    memcpy(q_norm, query_vec, idx->dim * sizeof(float));
    (void)omni_normalize_l2(q_norm, idx->dim);

    uint64_t q_hash = omni_lsh_compute_hash(q_norm, idx->hyperplanes, idx->num_hyperplanes, idx->dim);

    HeapNode* heap = (HeapNode*)rcre_arena_alloc(scratch, top_k * sizeof(HeapNode), 16);
    if (!heap) return RCRE_ERR_OUT_OF_MEMORY;
    size_t heap_size = 0;

    for (size_t i = 0; i < idx->num_items; ++i) {
        const float* doc_vec = &idx->embeddings[i * idx->dim];
        float sim = omni_cosine_similarity(q_norm, doc_vec, idx->dim);
        uint32_t h_dist = omni_hamming_distance64(q_hash, idx->items[i].hash_sig);
        heap_push(heap, &heap_size, top_k, (uint32_t)i, sim, h_dist);
    }

    heap_sort_descending(heap, heap_size);

    for (size_t i = 0; i < heap_size; ++i) {
        uint32_t id = heap[i].id;
        out_results[i].id = id;
        out_results[i].score = heap[i].score;
        out_results[i].distance = 1.0f - heap[i].score;
        out_results[i].hamming_dist = heap[i].hamming_dist;
        out_results[i].type = (OmniItemType)idx->items[id].type;
        out_results[i].payload = &idx->string_pool[idx->items[id].str_offset];
    }

    *out_count = heap_size;

    if (scratch) scratch->offset = saved_offset;
    return RCRE_OK;
}
