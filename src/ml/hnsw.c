/**
 * @file hnsw.c
 * @brief Hierarchical Navigable Small World (HNSW) Vector Search Index
 * 
 * Satisfies ML-REQ-04.
 * Graph-based Approximate Nearest Neighbors (ANN) algorithm.
 */

#include "rcre.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define HNSW_MAX_LEVELS 16

typedef struct {
    size_t id;
    float  dist;
} HNSWDistPair;

typedef struct {
    size_t   id;
    float*   vector;
    int      level;
    size_t*  neighbors[HNSW_MAX_LEVELS];
    size_t   num_neighbors[HNSW_MAX_LEVELS];
} HNSWNode;

struct RCRE_HNSWIndex {
    size_t          dim;
    size_t          max_elements;
    size_t          num_elements;
    size_t          M;
    size_t          M0;
    size_t          ef_construction;
    RCRE_MetricType metric;
    int             max_level;
    size_t          enter_node;
    HNSWNode*       nodes;
    float*          vector_storage;
};

static float compute_distance(const float* a, const float* b, size_t dim, RCRE_MetricType metric) {
    if (metric == RCRE_METRIC_L2) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    } else {
        /* Cosine distance: 1.0 - (a . b) / (|a| * |b|) */
        float dot = rcre_hal.dot_f32(a, b, dim);
        float norm_a = rcre_hal.dot_f32(a, a, dim);
        float norm_b = rcre_hal.dot_f32(b, b, dim);
        if (norm_a <= 1e-12f || norm_b <= 1e-12f) return 1.0f;
        float cos_sim = dot / (sqrtf(norm_a) * sqrtf(norm_b));
        return 1.0f - cos_sim;
    }
}

RCRE_NODISCARD RCRE_HNSWIndex* rcre_hnsw_create(size_t dim, size_t max_elements, 
                                                size_t M, size_t ef_construction, 
                                                RCRE_MetricType metric, 
                                                RCRE_Arena* arena) {
    if (dim == 0 || max_elements == 0 || M == 0 || !arena) {
        return NULL;
    }

    RCRE_HNSWIndex* index = (RCRE_HNSWIndex*)rcre_arena_alloc(arena, sizeof(RCRE_HNSWIndex), 64);
    if (!index) return NULL;

    index->dim = dim;
    index->max_elements = max_elements;
    index->num_elements = 0;
    index->M = M;
    index->M0 = 2 * M;
    index->ef_construction = ef_construction > M ? ef_construction : M;
    index->metric = metric;
    index->max_level = -1;
    index->enter_node = 0;

    index->nodes = (HNSWNode*)rcre_arena_alloc(arena, max_elements * sizeof(HNSWNode), 64);
    index->vector_storage = (float*)rcre_arena_alloc(arena, max_elements * dim * sizeof(float), 64);

    if (!index->nodes || !index->vector_storage) {
        return NULL;
    }

    /* Pre-allocate neighbor arrays for all potential elements in arena */
    for (size_t i = 0; i < max_elements; ++i) {
        index->nodes[i].level = 0;
        index->nodes[i].vector = &index->vector_storage[i * dim];
        for (int l = 0; l < HNSW_MAX_LEVELS; ++l) {
            size_t max_m = (l == 0) ? index->M0 : index->M;
            index->nodes[i].neighbors[l] = (size_t*)rcre_arena_alloc(arena, max_m * sizeof(size_t), 64);
            index->nodes[i].num_neighbors[l] = 0;
        }
    }

    return index;
}

/* Sample random level with exponential decay parameter mL = 1 / ln(M) */
static int sample_level(size_t M) {
    double r = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double mL = 1.0 / log((double)M);
    int level = (int)(-log(r) * mL);
    if (level >= HNSW_MAX_LEVELS) level = HNSW_MAX_LEVELS - 1;
    return level;
}

RCRE_NODISCARD int rcre_hnsw_insert(RCRE_HNSWIndex* index, const float* vector, size_t id, 
                                    RCRE_Arena* scratch) {
    (void)scratch;
    if (!index || !vector || index->num_elements >= index->max_elements) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t new_idx = index->num_elements;
    index->nodes[new_idx].id = id;
    memcpy(index->nodes[new_idx].vector, vector, index->dim * sizeof(float));

    int new_level = sample_level(index->M);
    index->nodes[new_idx].level = new_level;

    if (index->num_elements == 0) {
        index->enter_node = new_idx;
        index->max_level = new_level;
        index->num_elements++;
        return RCRE_OK;
    }

    size_t curr_obj = index->enter_node;
    float curr_dist = compute_distance(vector, index->nodes[curr_obj].vector, index->dim, index->metric);

    /* 1. Greedy search from top level down to new_level + 1 */
    for (int l = index->max_level; l > new_level; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i < index->nodes[curr_obj].num_neighbors[l]; ++i) {
                size_t neighbor = index->nodes[curr_obj].neighbors[l][i];
                float d = compute_distance(vector, index->nodes[neighbor].vector, index->dim, index->metric);
                if (d < curr_dist) {
                    curr_dist = d;
                    curr_obj = neighbor;
                    changed = true;
                }
            }
        }
    }

    /* 2. Connect at levels min(max_level, new_level) down to 0 */
    int top_connect_level = (index->max_level < new_level) ? index->max_level : new_level;
    for (int l = top_connect_level; l >= 0; --l) {
        /* Simple greedy search at level l to find candidate neighbors */
        size_t candidates[128];
        float cand_dists[128];
        size_t num_candidates = 0;

        candidates[num_candidates] = curr_obj;
        cand_dists[num_candidates] = curr_dist;
        num_candidates++;

        for (size_t i = 0; i < index->nodes[curr_obj].num_neighbors[l]; ++i) {
            size_t n = index->nodes[curr_obj].neighbors[l][i];
            float d = compute_distance(vector, index->nodes[n].vector, index->dim, index->metric);
            if (num_candidates < 128) {
                candidates[num_candidates] = n;
                cand_dists[num_candidates] = d;
                num_candidates++;
            }
        }

        /* Sort candidates by distance (simple selection sort for small candidate sets) */
        for (size_t a = 0; a < num_candidates; ++a) {
            for (size_t b = a + 1; b < num_candidates; ++b) {
                if (cand_dists[b] < cand_dists[a]) {
                    float td = cand_dists[a]; cand_dists[a] = cand_dists[b]; cand_dists[b] = td;
                    size_t tc = candidates[a]; candidates[a] = candidates[b]; candidates[b] = tc;
                }
            }
        }

        /* Add bidirectional edges to closest M neighbors */
        size_t max_m = (l == 0) ? index->M0 : index->M;
        size_t to_add = (num_candidates < max_m) ? num_candidates : max_m;

        for (size_t i = 0; i < to_add; ++i) {
            size_t n = candidates[i];
            /* Connect new_idx -> n */
            if (index->nodes[new_idx].num_neighbors[l] < max_m) {
                index->nodes[new_idx].neighbors[l][index->nodes[new_idx].num_neighbors[l]++] = n;
            }
            /* Connect n -> new_idx */
            if (index->nodes[n].num_neighbors[l] < max_m) {
                index->nodes[n].neighbors[l][index->nodes[n].num_neighbors[l]++] = new_idx;
            }
        }

        curr_obj = candidates[0];
        curr_dist = cand_dists[0];
    }

    if (new_level > index->max_level) {
        index->max_level = new_level;
        index->enter_node = new_idx;
    }

    index->num_elements++;
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_hnsw_search(const RCRE_HNSWIndex* index, const float* query, 
                                    size_t k, size_t* out_ids, float* out_distances, 
                                    size_t ef_search, RCRE_Arena* scratch) {
    (void)scratch;
    (void)ef_search;
    if (!index || !query || k == 0 || !out_ids || index->num_elements == 0) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t curr_obj = index->enter_node;
    float curr_dist = compute_distance(query, index->nodes[curr_obj].vector, index->dim, index->metric);

    /* 1. Greedy search through upper levels */
    for (int l = index->max_level; l > 0; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i < index->nodes[curr_obj].num_neighbors[l]; ++i) {
                size_t neighbor = index->nodes[curr_obj].neighbors[l][i];
                float d = compute_distance(query, index->nodes[neighbor].vector, index->dim, index->metric);
                if (d < curr_dist) {
                    curr_dist = d;
                    curr_obj = neighbor;
                    changed = true;
                }
            }
        }
    }

    /* 2. Beam search at level 0 */
    size_t visited[512];
    size_t num_visited = 0;

    HNSWDistPair candidates[512];
    size_t num_cand = 0;

    candidates[num_cand++] = (HNSWDistPair){.id = curr_obj, .dist = curr_dist};
    visited[num_visited++] = curr_obj;

    bool changed = true;
    while (changed) {
        changed = false;
        size_t current_candidates_count = num_cand;
        for (size_t c = 0; c < current_candidates_count; ++c) {
            size_t cand_id = candidates[c].id;
            for (size_t i = 0; i < index->nodes[cand_id].num_neighbors[0]; ++i) {
                size_t neighbor = index->nodes[cand_id].neighbors[0][i];

                /* Check if already visited */
                bool seen = false;
                for (size_t v = 0; v < num_visited; ++v) {
                    if (visited[v] == neighbor) { seen = true; break; }
                }
                if (seen) continue;

                if (num_visited < 512) visited[num_visited++] = neighbor;

                float d = compute_distance(query, index->nodes[neighbor].vector, index->dim, index->metric);
                if (num_cand < 512) {
                    candidates[num_cand++] = (HNSWDistPair){.id = neighbor, .dist = d};
                    changed = true;
                }
            }
        }
    }

    /* Sort all accumulated candidates by distance */
    for (size_t a = 0; a < num_cand; ++a) {
        for (size_t b = a + 1; b < num_cand; ++b) {
            if (candidates[b].dist < candidates[a].dist) {
                HNSWDistPair tmp = candidates[a];
                candidates[a] = candidates[b];
                candidates[b] = tmp;
            }
        }
    }

    size_t results_to_return = (k < num_cand) ? k : num_cand;
    for (size_t i = 0; i < results_to_return; ++i) {
        out_ids[i] = index->nodes[candidates[i].id].id;
        if (out_distances) {
            out_distances[i] = candidates[i].dist;
        }
    }

    return RCRE_OK;
}
