/**
 * @file hnsw.c
 * @brief Hierarchical Navigable Small World (HNSW) Vector Search Index
 * 
 * Satisfies ML-REQ-04.
 * Graph-based Approximate Nearest Neighbors (ANN) algorithm.
 */

#include "rcre.h"
#include "ml_internal.h"
#include <stdio.h>
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
        return rcre_squared_l2_distance(a, b, dim);
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
        /* Greedy search at level l to find local minimum before connecting */
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

        /* Gather candidate neighbors around curr_obj at level l */
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
            } else {
                /* When neighbor list is full, replace the farthest neighbor if new_idx is closer */
                float d_new = compute_distance(index->nodes[n].vector, vector, index->dim, index->metric);
                size_t worst_pos = 0;
                float worst_d = compute_distance(index->nodes[n].vector, 
                                                 index->nodes[index->nodes[n].neighbors[l][0]].vector, 
                                                 index->dim, index->metric);
                for (size_t j = 1; j < index->nodes[n].num_neighbors[l]; ++j) {
                    size_t nb = index->nodes[n].neighbors[l][j];
                    float d_nb = compute_distance(index->nodes[n].vector, index->nodes[nb].vector, index->dim, index->metric);
                    if (d_nb > worst_d) {
                        worst_d = d_nb;
                        worst_pos = j;
                    }
                }
                if (d_new < worst_d) {
                    index->nodes[n].neighbors[l][worst_pos] = new_idx;
                }
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

    /* 2. Priority best-first beam search at level 0 */
    size_t ef = (ef_search > k) ? ef_search : k;
    if (ef < 32) ef = 32;
    if (ef > 128) ef = 128;

    typedef struct {
        size_t id;
        float  dist;
        bool   explored;
    } CandidateItem;

    CandidateItem W[128];
    size_t W_size = 0;

    W[W_size++] = (CandidateItem){.id = curr_obj, .dist = curr_dist, .explored = false};

    /* Visited tracking: use stack bitset for up to 4096 elements, or dynamic fallback */
    size_t max_nodes = index->num_elements;
    size_t bitset_bytes = (max_nodes + 7) / 8;
    uint8_t stack_visited[512];
    uint8_t* visited = stack_visited;
    bool allocated_visited = false;
    if (bitset_bytes > sizeof(stack_visited)) {
        if (scratch) {
            visited = (uint8_t*)rcre_arena_alloc(scratch, bitset_bytes, 8);
        } else {
            visited = (uint8_t*)calloc(1, bitset_bytes);
            allocated_visited = true;
        }
    }
    if (visited) {
        memset(visited, 0, bitset_bytes);
        visited[curr_obj / 8] |= (uint8_t)(1 << (curr_obj % 8));
    }

    while (true) {
        /* Find unexplored candidate with minimum distance */
        int best_idx = -1;
        float min_unexp_dist = 1e30f;
        for (size_t i = 0; i < W_size; ++i) {
            if (!W[i].explored && W[i].dist < min_unexp_dist) {
                min_unexp_dist = W[i].dist;
                best_idx = (int)i;
            }
        }

        if (best_idx < 0) {
            break; /* All candidates explored */
        }

        /* If closest unexplored is worse than the worst result in full capacity W, stop */
        if (W_size >= ef && min_unexp_dist > W[W_size - 1].dist) {
            break;
        }

        W[best_idx].explored = true;
        size_t cand_id = W[best_idx].id;

        /* Explore neighbors of cand_id at level 0 */
        for (size_t i = 0; i < index->nodes[cand_id].num_neighbors[0]; ++i) {
            size_t neighbor = index->nodes[cand_id].neighbors[0][i];

            if (visited) {
                if (visited[neighbor / 8] & (1 << (neighbor % 8))) {
                    continue;
                }
                visited[neighbor / 8] |= (uint8_t)(1 << (neighbor % 8));
            }

            float d = compute_distance(query, index->nodes[neighbor].vector, index->dim, index->metric);

            if (d < W[W_size - 1].dist || W_size < ef) {
                /* Insert (neighbor, d) in sorted order into W */
                size_t insert_pos = 0;
                while (insert_pos < W_size && W[insert_pos].dist <= d) {
                    insert_pos++;
                }

                if (insert_pos < ef) {
                    if (W_size < ef) {
                        W_size++;
                    }
                    for (size_t m = W_size - 1; m > insert_pos; --m) {
                        W[m] = W[m - 1];
                    }
                    W[insert_pos] = (CandidateItem){.id = neighbor, .dist = d, .explored = false};
                }
            }
        }
    }

    if (allocated_visited && visited) {
        free(visited);
    }

    size_t results_to_return = (k < W_size) ? k : W_size;
    for (size_t i = 0; i < results_to_return; ++i) {
        out_ids[i] = index->nodes[W[i].id].id;
        if (out_distances) {
            out_distances[i] = W[i].dist;
        }
    }

    return RCRE_OK;
}
