/**
 * @file index_serializer.c
 * @brief Binary Serialization, Index Persistence & Fast Retrieval Engine for RCRE Lens (.rcre_idx)
 */

#include "index_serializer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

RCRE_NODISCARD RCRE_ImageIndex* rcre_index_create(size_t capacity, 
                                                  size_t feature_dim, 
                                                  RCRE_Arena* arena) {
    if (capacity == 0 || feature_dim == 0 || !arena) {
        return NULL;
    }

    RCRE_ImageIndex* idx = (RCRE_ImageIndex*)rcre_arena_alloc(arena, sizeof(RCRE_ImageIndex), 64);
    if (!idx) return NULL;

    idx->capacity = capacity;
    idx->feature_dim = feature_dim;
    idx->num_images = 0;
    idx->arena = arena;

    idx->filepaths = (char**)rcre_arena_alloc(arena, capacity * sizeof(char*), 64);
    idx->embeddings = (float*)rcre_arena_alloc(arena, capacity * feature_dim * sizeof(float), 64);

    if (!idx->filepaths || !idx->embeddings) {
        return NULL;
    }

    /* Allocate HNSW Proximity Graph using Cosine Distance Metric */
    idx->hnsw = rcre_hnsw_create(feature_dim, capacity, 16, 64, RCRE_METRIC_COSINE, arena);
    if (!idx->hnsw) {
        return NULL;
    }

    return idx;
}

RCRE_NODISCARD int rcre_index_add(RCRE_ImageIndex* index, 
                                  const char* filepath, 
                                  const float* embedding) {
    if (!index || !filepath || !embedding) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (index->num_images >= index->capacity) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    size_t id = index->num_images;

    /* Copy filepath string into arena */
    size_t path_len = strlen(filepath) + 1;
    char* path_copy = (char*)rcre_arena_alloc(index->arena, path_len, 1);
    if (!path_copy) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memcpy(path_copy, filepath, path_len);
    index->filepaths[id] = path_copy;

    /* Copy contiguous feature embedding vector */
    float* emb_dst = &index->embeddings[id * index->feature_dim];
    memcpy(emb_dst, embedding, index->feature_dim * sizeof(float));

    /* Insert vector into HNSW index */
    int rc = rcre_hnsw_insert(index->hnsw, emb_dst, id, index->arena);
    if (rc != RCRE_OK) {
        return rc;
    }

    index->num_images++;
    return RCRE_OK;
}

static const char RCRE_MAGIC[8] = {'R', 'C', 'R', 'E', '_', 'I', 'D', 'X'};

RCRE_NODISCARD int rcre_index_save(const RCRE_ImageIndex* index, 
                                   const char* output_filepath) {
    if (!index || !output_filepath) {
        return RCRE_ERR_INVALID_ARG;
    }

    FILE* fp = fopen(output_filepath, "wb");
    if (!fp) {
        return RCRE_ERR_IO;
    }

    /* 1. Header (32 bytes) */
    uint32_t version = RCRE_IDX_VERSION;
    uint32_t dim = (uint32_t)index->feature_dim;
    uint64_t num = (uint64_t)index->num_images;
    uint64_t reserved = 0;

    if (fwrite(RCRE_MAGIC, 1, 8, fp) != 8 ||
        fwrite(&version, sizeof(uint32_t), 1, fp) != 1 ||
        fwrite(&dim, sizeof(uint32_t), 1, fp) != 1 ||
        fwrite(&num, sizeof(uint64_t), 1, fp) != 1 ||
        fwrite(&reserved, sizeof(uint64_t), 1, fp) != 1) {
        fclose(fp);
        return RCRE_ERR_IO;
    }

    /* 2. Filepath Manifest */
    for (size_t i = 0; i < index->num_images; ++i) {
        const char* path = index->filepaths[i];
        uint32_t len = (uint32_t)(path ? strlen(path) : 0);
        if (fwrite(&len, sizeof(uint32_t), 1, fp) != 1) {
            fclose(fp);
            return RCRE_ERR_IO;
        }
        if (len > 0) {
            if (fwrite(path, 1, len, fp) != len) {
                fclose(fp);
                return RCRE_ERR_IO;
            }
        }
    }

    /* 3. Contiguous Float Embeddings */
    if (index->num_images > 0) {
        size_t total_floats = index->num_images * index->feature_dim;
        if (fwrite(index->embeddings, sizeof(float), total_floats, fp) != total_floats) {
            fclose(fp);
            return RCRE_ERR_IO;
        }
    }

    fclose(fp);
    return RCRE_OK;
}

RCRE_NODISCARD RCRE_ImageIndex* rcre_index_load(const char* input_filepath, 
                                                RCRE_Arena* arena) {
    if (!input_filepath || !arena) {
        return NULL;
    }

    FILE* fp = fopen(input_filepath, "rb");
    if (!fp) {
        return NULL;
    }

    /* 1. Read & Validate Header */
    char magic[8];
    uint32_t version = 0;
    uint32_t dim = 0;
    uint64_t num = 0;
    uint64_t reserved = 0;

    if (fread(magic, 1, 8, fp) != 8 ||
        fread(&version, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&dim, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&num, sizeof(uint64_t), 1, fp) != 1 ||
        fread(&reserved, sizeof(uint64_t), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    if (memcmp(magic, RCRE_MAGIC, 8) != 0 || version != RCRE_IDX_VERSION) {
        fclose(fp);
        return NULL;
    }

    /* Create index structure in arena */
    size_t count = (size_t)num;
    size_t feature_dim = (size_t)dim;
    /* Ensure capacity is at least 1 even if file is empty */
    size_t capacity = count > 0 ? count : 1;

    RCRE_ImageIndex* index = rcre_index_create(capacity, feature_dim, arena);
    if (!index) {
        fclose(fp);
        return NULL;
    }

    /* 2. Read Filepath Manifest */
    for (size_t i = 0; i < count; ++i) {
        uint32_t len = 0;
        if (fread(&len, sizeof(uint32_t), 1, fp) != 1) {
            fclose(fp);
            return NULL;
        }
        char* str = (char*)rcre_arena_alloc(arena, len + 1, 1);
        if (!str) {
            fclose(fp);
            return NULL;
        }
        if (len > 0) {
            if (fread(str, 1, len, fp) != len) {
                fclose(fp);
                return NULL;
            }
        }
        str[len] = '\0';
        index->filepaths[i] = str;
    }

    /* 3. Read Embeddings and Rebuild HNSW Proximity Graph */
    if (count > 0) {
        size_t total_floats = count * feature_dim;
        if (fread(index->embeddings, sizeof(float), total_floats, fp) != total_floats) {
            fclose(fp);
            return NULL;
        }

        for (size_t i = 0; i < count; ++i) {
            const float* vec = &index->embeddings[i * feature_dim];
            int rc = rcre_hnsw_insert(index->hnsw, vec, i, arena);
            if (rc != RCRE_OK) {
                fclose(fp);
                return NULL;
            }
        }
    }

    index->num_images = count;
    fclose(fp);
    return index;
}

int rcre_index_search(const RCRE_ImageIndex* index, 
                      const float* query_embedding, 
                      size_t top_k, 
                      RCRE_SearchResult* out_results, 
                      RCRE_Arena* scratch_arena) {
    if (!index || !query_embedding || !out_results || top_k == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (index->num_images == 0) {
        return 0;
    }

    size_t k = (top_k < index->num_images) ? top_k : index->num_images;

    /* Allocate temporary arrays for HNSW search output */
    size_t stack_ids[128];
    float stack_dists[128];
    size_t* ids = stack_ids;
    float* dists = stack_dists;

    if (k > 128) {
        if (!scratch_arena) {
            return RCRE_ERR_INVALID_ARG;
        }
        ids = (size_t*)rcre_arena_alloc(scratch_arena, k * sizeof(size_t), 64);
        dists = (float*)rcre_arena_alloc(scratch_arena, k * sizeof(float), 64);
        if (!ids || !dists) {
            return RCRE_ERR_OUT_OF_MEMORY;
        }
    }

    int rc = rcre_hnsw_search(index->hnsw, query_embedding, k, ids, dists, 64, scratch_arena);
    if (rc != RCRE_OK) {
        return rc;
    }

    /* Populate search results with cosine similarity and euclidean distance */
    for (size_t i = 0; i < k; ++i) {
        size_t img_id = ids[i];
        float dist = dists[i]; /* In RCRE_METRIC_COSINE, dist = 1.0 - cos_sim */

        float cos_sim = 1.0f - dist;
        if (cos_sim > 1.0f) cos_sim = 1.0f;
        if (cos_sim < -1.0f) cos_sim = -1.0f;

        /* For L2 distance between normalized vectors:
         * ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b) = 2*(1 - cos_sim) = 2*dist
         */
        float l2_sq = 2.0f * (1.0f - cos_sim);
        if (l2_sq < 0.0f) l2_sq = 0.0f;
        float l2_dist = sqrtf(l2_sq);

        out_results[i].image_id = img_id;
        out_results[i].filepath = (img_id < index->num_images) ? index->filepaths[img_id] : "";
        out_results[i].similarity = cos_sim;
        out_results[i].l2_distance = l2_dist;
    }

    return (int)k;
}
