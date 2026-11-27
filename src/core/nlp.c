/**
 * @file nlp.c
 * @brief RCRE HPC-Native, Zero-Malloc C23 NLP Engine Implementation
 *
 * Satisfies NLP-REQ-01 through NLP-REQ-06.
 * Pure ISO C23, zero malloc/free/realloc, flat arena allocations,
 * lock-free re-entrant kernels, and HAL SIMD string acceleration.
 */

#include "rcre_nlp.h"
#include <math.h>
#include <string.h>

/* ========================================================================= */
/* Internal Helpers (namespaced with rcre_nlp__ for amalgamation safety)     */
/* ========================================================================= */

static inline bool rcre_nlp__is_ascii_alpha(uint8_t c) {
    return ((uint8_t)(c - (uint8_t)'A') <= 25U) || ((uint8_t)(c - (uint8_t)'a') <= 25U);
}

static inline bool rcre_nlp__is_ascii_digit(uint8_t c) {
    return (uint8_t)(c - (uint8_t)'0') <= 9U;
}

static inline bool rcre_nlp__is_word_byte(uint8_t c, bool keep_digits) {
    if (c >= 0x80U) return true; /* Preserve UTF-8 multi-byte word letters */
    if (rcre_nlp__is_ascii_alpha(c)) return true;
    if (keep_digits && rcre_nlp__is_ascii_digit(c)) return true;
    return false;
}

static inline bool rcre_nlp__is_ascii_punct(uint8_t c) {
    return (c >= 33U && c <= 47U) || (c >= 58U && c <= 64U) ||
           (c >= 91U && c <= 96U) || (c >= 123U && c <= 126U);
}

static inline bool rcre_nlp__is_ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static inline uint8_t rcre_nlp__lower_byte(uint8_t c) {
    return ((uint8_t)(c - (uint8_t)'A') <= 25U) ? (uint8_t)(c | 0x20U) : c;
}

static inline uint64_t rcre_nlp__splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline RCRE_Slice rcre_nlp__normalize_tok(
    RCRE_Slice tok,
    bool lowercase,
    uint8_t* stack_buf,
    size_t stack_cap,
    RCRE_Arena* scratch
) {
    if (!lowercase || tok.len == 0 || !tok.data) {
        return tok;
    }
    /* Fast check: if no uppercase ASCII byte exists, return zero-copy slice! */
    bool has_upper = false;
    for (size_t i = 0; i < tok.len; ++i) {
        if ((uint8_t)(tok.data[i] - (uint8_t)'A') <= 25U) {
            has_upper = true;
            break;
        }
    }
    if (!has_upper) {
        return tok;
    }
    if (tok.len <= stack_cap) {
        rcre_hal.ascii_to_lower(tok.data, stack_buf, tok.len);
        return rcre_slice_from_parts(stack_buf, tok.len);
    }
    if (scratch) {
        uint8_t* dst = (uint8_t*)rcre_arena_alloc(scratch, tok.len, 16);
        if (dst) {
            rcre_hal.ascii_to_lower(tok.data, dst, tok.len);
            return rcre_slice_from_parts(dst, tok.len);
        }
    }
    /* Fallback: truncate to stack_cap if no scratch arena */
    rcre_hal.ascii_to_lower(tok.data, stack_buf, stack_cap);
    return rcre_slice_from_parts(stack_buf, stack_cap);
}

/* Deterministic in-place heapsort for uint32_t (zero recursion, zero malloc) */
static void rcre_nlp__sort_u32(uint32_t* a, size_t n) {
    if (n <= 1U) return;
    for (size_t start = n / 2U; start > 0U; --start) {
        size_t root = start - 1U;
        for (;;) {
            size_t child = root * 2U + 1U;
            if (child >= n) break;
            if (child + 1U < n && a[child] < a[child + 1U]) child++;
            if (a[root] < a[child]) {
                uint32_t t = a[root]; a[root] = a[child]; a[child] = t;
                root = child;
            } else {
                break;
            }
        }
    }
    for (size_t end = n - 1U; end > 0U; --end) {
        uint32_t t = a[end]; a[end] = a[0]; a[0] = t;
        size_t root = 0U;
        for (;;) {
            size_t child = root * 2U + 1U;
            if (child >= end) break;
            if (child + 1U < end && a[child] < a[child + 1U]) child++;
            if (a[root] < a[child]) {
                uint32_t tmp = a[root]; a[root] = a[child]; a[child] = tmp;
                root = child;
            } else {
                break;
            }
        }
    }
}

/* Deterministic heapsort of (score desc, doc_id asc) */
static inline bool rcre_nlp__hit_better(RCRE_SearchHit a, RCRE_SearchHit b) {
    if (a.score != b.score) return a.score > b.score;
    return a.doc_id < b.doc_id;
}

static void rcre_nlp__sort_hits_desc(RCRE_SearchHit* a, size_t n) {
    if (n <= 1U) return;
    /* Build min-heap (worst at root) so popping yields descending order */
    for (size_t start = n / 2U; start > 0U; --start) {
        size_t root = start - 1U;
        for (;;) {
            size_t child = root * 2U + 1U;
            if (child >= n) break;
            if (child + 1U < n && rcre_nlp__hit_better(a[child], a[child + 1U])) child++;
            if (rcre_nlp__hit_better(a[root], a[child])) {
                RCRE_SearchHit t = a[root]; a[root] = a[child]; a[child] = t;
                root = child;
            } else {
                break;
            }
        }
    }
    for (size_t end = n - 1U; end > 0U; --end) {
        RCRE_SearchHit t = a[end]; a[end] = a[0]; a[0] = t;
        size_t root = 0U;
        for (;;) {
            size_t child = root * 2U + 1U;
            if (child >= end) break;
            if (child + 1U < end && rcre_nlp__hit_better(a[child], a[child + 1U])) child++;
            if (rcre_nlp__hit_better(a[root], a[child])) {
                RCRE_SearchHit tmp = a[root]; a[root] = a[child]; a[child] = tmp;
                root = child;
            } else {
                break;
            }
        }
    }
}

/* Sort vocab term IDs lexicographically by slice (matches scikit-learn) */
typedef struct {
    RCRE_Slice slice;
    uint32_t   old_id;
} rcre_nlp__term_pair;

static inline bool rcre_nlp__term_less(rcre_nlp__term_pair a, rcre_nlp__term_pair b) {
    int c = rcre_slice_cmp(a.slice, b.slice);
    return (c != 0) ? (c < 0) : (a.old_id < b.old_id);
}

static void rcre_nlp__sort_terms(rcre_nlp__term_pair* a, size_t n) {
    if (n <= 1U) return;
    for (size_t start = n / 2U; start > 0U; --start) {
        size_t root = start - 1U;
        for (;;) {
            size_t child = root * 2U + 1U;
            if (child >= n) break;
            if (child + 1U < n && rcre_nlp__term_less(a[child], a[child + 1U])) child++;
            if (rcre_nlp__term_less(a[root], a[child])) {
                rcre_nlp__term_pair t = a[root]; a[root] = a[child]; a[child] = t;
                root = child;
            } else {
                break;
            }
        }
    }
    for (size_t end = n - 1U; end > 0U; --end) {
        rcre_nlp__term_pair t = a[end]; a[end] = a[0]; a[0] = t;
        size_t root = 0U;
        for (;;) {
            size_t child = root * 2U + 1U;
            if (child >= end) break;
            if (child + 1U < end && rcre_nlp__term_less(a[child], a[child + 1U])) child++;
            if (rcre_nlp__term_less(a[root], a[child])) {
                rcre_nlp__term_pair tmp = a[root]; a[root] = a[child]; a[child] = tmp;
                root = child;
            } else {
                break;
            }
        }
    }
}

/* ========================================================================= */
/* 1. Zero-Copy Tokenizers & N-Gram Iterators (NLP-REQ-01)                   */
/* ========================================================================= */

RCRE_TokenizeOpts rcre_nlp_default_tokenize_opts(void) {
    RCRE_TokenizeOpts opts = {
        .split_punct = false,
        .keep_digits = true,
        .min_len     = 1,
        .max_len     = 0
    };
    return opts;
}

RCRE_WordIter rcre_nlp_word_iter_init(RCRE_Slice text, const RCRE_TokenizeOpts* opts) {
    RCRE_WordIter it = {
        .src  = text,
        .pos  = 0,
        .opts = opts ? *opts : rcre_nlp_default_tokenize_opts()
    };
    if (it.opts.min_len == 0) {
        it.opts.min_len = 1;
    }
    return it;
}

bool rcre_nlp_word_iter_next(RCRE_WordIter* it, RCRE_Slice* out_tok) {
    if (!it || !out_tok || !it->src.data) {
        return false;
    }
    while (it->pos < it->src.len) {
        uint8_t c = it->src.data[it->pos];
        if (rcre_nlp__is_word_byte(c, it->opts.keep_digits)) {
            size_t start = it->pos;
            it->pos++;
            while (it->pos < it->src.len && rcre_nlp__is_word_byte(it->src.data[it->pos], it->opts.keep_digits)) {
                it->pos++;
            }
            size_t tok_len = it->pos - start;
            if (tok_len >= it->opts.min_len && (it->opts.max_len == 0 || tok_len <= it->opts.max_len)) {
                *out_tok = rcre_slice_sub(it->src, start, it->pos);
                return true;
            }
        } else if (it->opts.split_punct && rcre_nlp__is_ascii_punct(c)) {
            size_t start = it->pos;
            it->pos++;
            if (1U >= it->opts.min_len) {
                *out_tok = rcre_slice_sub(it->src, start, it->pos);
                return true;
            }
        } else {
            it->pos++;
        }
    }
    return false;
}

RCRE_CharNGramIter rcre_nlp_char_ngram_init(RCRE_Slice text, size_t n) {
    RCRE_CharNGramIter it = {
        .src = text,
        .n   = (n == 0) ? 1U : n,
        .pos = 0
    };
    return it;
}

bool rcre_nlp_char_ngram_next(RCRE_CharNGramIter* it, RCRE_Slice* out_ngram) {
    if (!it || !out_ngram || !it->src.data || it->pos >= it->src.len) {
        return false;
    }
    RCRE_Slice cur = rcre_slice_sub(it->src, it->pos, it->src.len);
    size_t start_off = it->pos;
    size_t first_step = 0;

    for (size_t k = 0; k < it->n; ++k) {
        if (cur.len == 0) {
            it->pos = it->src.len;
            return false;
        }
        int step = rcre_utf8_decode_next(&cur, nullptr);
        if (step <= 0) {
            /* Advance 1 raw byte on malformed UTF-8 so iteration never stalls */
            cur = rcre_slice_sub(cur, 1U, cur.len);
            step = 1;
        }
        if (k == 0) first_step = (size_t)step;
    }

    size_t end_off = cur.data ? (size_t)(cur.data - it->src.data) : it->src.len;
    *out_ngram = rcre_slice_sub(it->src, start_off, end_off);
    it->pos = start_off + first_step;
    return true;
}

RCRE_WordNGramIter rcre_nlp_word_ngram_init(RCRE_Slice text, size_t n, const RCRE_TokenizeOpts* opts) {
    if (n == 0) n = 1U;
    if (n > RCRE_NLP_MAX_NGRAM) n = RCRE_NLP_MAX_NGRAM;
    RCRE_WordNGramIter it;
    memset(&it, 0, sizeof(it));
    it.words = rcre_nlp_word_iter_init(text, opts);
    it.n = n;
    return it;
}

bool rcre_nlp_word_ngram_next(RCRE_WordNGramIter* it, RCRE_Slice* out_span) {
    if (!it || !out_span || it->n == 0) {
        return false;
    }
    RCRE_Slice tok;
    while (it->filled < it->n) {
        if (!rcre_nlp_word_iter_next(&it->words, &tok)) {
            return false;
        }
        it->ring[(it->head + it->filled) % it->n] = tok;
        it->filled++;
    }
    RCRE_Slice first = it->ring[it->head];
    RCRE_Slice last  = it->ring[(it->head + it->n - 1U) % it->n];
    const uint8_t* start_ptr = first.data;
    const uint8_t* end_ptr = last.data + last.len;
    *out_span = rcre_slice_from_parts(start_ptr, (size_t)(end_ptr - start_ptr));

    /* Advance sliding window by 1 token for the next call */
    it->head = (it->head + 1U) % it->n;
    it->filled--;
    return true;
}

/* ========================================================================= */
/* 2. Flat-Arena BM25 Inverted Index & Hybrid RRF Fusion (NLP-REQ-02)        */
/* ========================================================================= */

RCRE_BM25Opts rcre_bm25_default_opts(void) {
    RCRE_BM25Opts opts = {
        .k1        = 1.5f,
        .b         = 0.75f,
        .lowercase = true,
        .tok_opts  = rcre_nlp_default_tokenize_opts()
    };
    return opts;
}

int rcre_bm25_build(
    RCRE_BM25Index* index,
    const RCRE_Slice* docs,
    size_t num_docs,
    const RCRE_BM25Opts* opts,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
) {
    if (!index || !arena || (num_docs > 0 && !docs) || num_docs > (size_t)UINT32_MAX - 1U) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(index, 0, sizeof(*index));
    RCRE_BM25Opts cfg = opts ? *opts : rcre_bm25_default_opts();
    index->num_docs  = (uint32_t)num_docs;
    index->k1        = cfg.k1;
    index->b         = cfg.b;
    index->lowercase = cfg.lowercase;
    index->tok_opts  = cfg.tok_opts;

    RCRE_Arena* tmp_arena = scratch ? scratch : arena;
    bool distinct_scratch = (scratch != nullptr && scratch != arena);
    RCRE_ArenaTemp tmp_mark = rcre_arena_temp_begin(tmp_arena);

    int rc = rcre_strpool_init(&index->vocab, arena, num_docs > 64 ? num_docs * 2U : 128U);
    if (rc != RCRE_OK) {
        if (distinct_scratch) rcre_arena_temp_end(tmp_mark);
        return rc;
    }

    if (num_docs == 0) {
        if (distinct_scratch) rcre_arena_temp_end(tmp_mark);
        return RCRE_OK;
    }

    index->doc_lens = (uint32_t*)rcre_arena_alloc(arena, num_docs * sizeof(uint32_t), 64);
    if (!index->doc_lens) {
        if (distinct_scratch) rcre_arena_temp_end(tmp_mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(index->doc_lens, 0, num_docs * sizeof(uint32_t));

    /* Pass 1: intern every term and compute doc lengths */
    uint8_t low_buf[256];
    uint64_t total_tokens = 0;
    for (size_t d = 0; d < num_docs; ++d) {
        RCRE_WordIter it = rcre_nlp_word_iter_init(docs[d], &cfg.tok_opts);
        RCRE_Slice tok;
        uint32_t dl = 0;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, cfg.lowercase, low_buf, sizeof(low_buf), tmp_arena);
            if (rcre_strpool_intern(&index->vocab, norm) == RCRE_STR_ID_INVALID) {
                if (distinct_scratch) rcre_arena_temp_end(tmp_mark);
                return RCRE_ERR_OUT_OF_MEMORY;
            }
            dl++;
        }
        index->doc_lens[d] = dl;
        total_tokens += dl;
    }

    size_t V = rcre_strpool_count(&index->vocab);
    index->num_terms = (uint32_t)V;
    index->avg_doc_len = (num_docs > 0) ? (float)((double)total_tokens / (double)num_docs) : 0.0f;

    if (V == 0) {
        if (distinct_scratch) rcre_arena_temp_end(tmp_mark);
        return RCRE_OK;
    }

    /* Count exact document frequencies df[t] in O(tokens) via last_doc[t] marker */
    uint32_t* df = (uint32_t*)rcre_arena_alloc(tmp_arena, V * sizeof(uint32_t), 64);
    uint32_t* last_doc = (uint32_t*)rcre_arena_alloc(tmp_arena, V * sizeof(uint32_t), 64);
    if (!df || !last_doc) {
        if (distinct_scratch) rcre_arena_temp_end(tmp_mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(df, 0, V * sizeof(uint32_t));
    for (size_t t = 0; t < V; ++t) last_doc[t] = UINT32_MAX;

    for (size_t d = 0; d < num_docs; ++d) {
        RCRE_WordIter it = rcre_nlp_word_iter_init(docs[d], &cfg.tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, cfg.lowercase, low_buf, sizeof(low_buf), tmp_arena);
            RCRE_StrId tid = rcre_strpool_find(&index->vocab, norm);
            if (tid != RCRE_STR_ID_INVALID && last_doc[tid] != (uint32_t)d) {
                last_doc[tid] = (uint32_t)d;
                df[tid]++;
            }
        }
    }

    /* Build CSR offsets & precompute Robertson-Sparck Jones BM25 IDF */
    index->term_offsets = (uint32_t*)rcre_arena_alloc(arena, (V + 1U) * sizeof(uint32_t), 64);
    index->term_idf     = (float*)rcre_arena_alloc(arena, V * sizeof(float), 64);
    if (!index->term_offsets || !index->term_idf) {
        if (distinct_scratch) rcre_arena_temp_end(tmp_mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    size_t total_postings = 0;
    double N = (double)num_docs;
    for (size_t t = 0; t < V; ++t) {
        index->term_offsets[t] = (uint32_t)total_postings;
        total_postings += df[t];
        double n_t = (double)df[t];
        index->term_idf[t] = (float)log(1.0 + (N - n_t + 0.5) / (n_t + 0.5));
    }
    index->term_offsets[V] = (uint32_t)total_postings;
    index->total_postings = total_postings;

    if (total_postings == 0) {
        if (distinct_scratch) rcre_arena_temp_end(tmp_mark);
        return RCRE_OK;
    }

    index->postings_doc = (uint32_t*)rcre_arena_alloc(arena, total_postings * sizeof(uint32_t), 64);
    index->postings_tf  = (uint32_t*)rcre_arena_alloc(arena, total_postings * sizeof(uint32_t), 64);
    uint32_t* write_cur = (uint32_t*)rcre_arena_alloc(tmp_arena, V * sizeof(uint32_t), 64);
    if (!index->postings_doc || !index->postings_tf || !write_cur) {
        if (distinct_scratch) rcre_arena_temp_end(tmp_mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memcpy(write_cur, index->term_offsets, V * sizeof(uint32_t));
    for (size_t t = 0; t < V; ++t) last_doc[t] = UINT32_MAX;

    /* Pass 2: scatter doc-sorted (doc_id, tf) postings in O(tokens) */
    for (size_t d = 0; d < num_docs; ++d) {
        RCRE_WordIter it = rcre_nlp_word_iter_init(docs[d], &cfg.tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, cfg.lowercase, low_buf, sizeof(low_buf), tmp_arena);
            RCRE_StrId tid = rcre_strpool_find(&index->vocab, norm);
            if (tid == RCRE_STR_ID_INVALID) continue;
            if (last_doc[tid] != (uint32_t)d) {
                uint32_t pos = write_cur[tid]++;
                index->postings_doc[pos] = (uint32_t)d;
                index->postings_tf[pos]  = 1U;
                last_doc[tid] = (uint32_t)d;
            } else {
                index->postings_tf[write_cur[tid] - 1U]++;
            }
        }
    }

    if (distinct_scratch) {
        rcre_arena_temp_end(tmp_mark);
    }
    return RCRE_OK;
}

int rcre_bm25_search(
    const RCRE_BM25Index* index,
    RCRE_Slice query,
    size_t top_k,
    RCRE_SearchHit* out_hits,
    size_t* out_count,
    RCRE_Arena* scratch
) {
    if (!index || !out_count || !scratch || (top_k > 0 && !out_hits)) {
        return RCRE_ERR_INVALID_ARG;
    }
    *out_count = 0;
    if (index->num_docs == 0 || index->num_terms == 0 || top_k == 0 || query.len == 0) {
        return RCRE_OK;
    }

    RCRE_ArenaTemp mark = rcre_arena_temp_begin(scratch);
    float* scores = (float*)rcre_arena_alloc(scratch, (size_t)index->num_docs * sizeof(float), 64);
    uint32_t* touched = (uint32_t*)rcre_arena_alloc(scratch, (size_t)index->num_docs * sizeof(uint32_t), 64);
    if (!scores || !touched) {
        rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(scores, 0, (size_t)index->num_docs * sizeof(float));
    size_t num_touched = 0;

    float k1 = index->k1;
    float b  = index->b;
    float avgdl = (index->avg_doc_len > 0.0f) ? index->avg_doc_len : 1.0f;

    uint8_t low_buf[256];
    RCRE_WordIter it = rcre_nlp_word_iter_init(query, &index->tok_opts);
    RCRE_Slice tok;
    while (rcre_nlp_word_iter_next(&it, &tok)) {
        RCRE_Slice norm = rcre_nlp__normalize_tok(tok, index->lowercase, low_buf, sizeof(low_buf), scratch);
        RCRE_StrId tid = rcre_strpool_find(&index->vocab, norm);
        if (tid == RCRE_STR_ID_INVALID) continue;

        float idf = index->term_idf[tid];
        uint32_t p0 = index->term_offsets[tid];
        uint32_t p1 = index->term_offsets[tid + 1U];
        for (uint32_t p = p0; p < p1; ++p) {
            uint32_t d = index->postings_doc[p];
            float tf = (float)index->postings_tf[p];
            float dl = (float)index->doc_lens[d];
            float denom = tf + k1 * (1.0f - b + b * (dl / avgdl));
            float contrib = idf * ((tf * (k1 + 1.0f)) / denom);
            if (scores[d] == 0.0f && contrib > 0.0f) {
                touched[num_touched++] = d;
            }
            scores[d] += contrib;
        }
    }

    if (num_touched == 0) {
        rcre_arena_temp_end(mark);
        return RCRE_OK;
    }

    RCRE_SearchHit* all_hits = (RCRE_SearchHit*)rcre_arena_alloc(
        scratch, num_touched * sizeof(RCRE_SearchHit), alignof(RCRE_SearchHit));
    if (!all_hits) {
        rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < num_touched; ++i) {
        uint32_t d = touched[i];
        all_hits[i].doc_id = d;
        all_hits[i].score = scores[d];
    }
    rcre_nlp__sort_hits_desc(all_hits, num_touched);

    size_t n_emit = (num_touched < top_k) ? num_touched : top_k;
    memcpy(out_hits, all_hits, n_emit * sizeof(RCRE_SearchHit));
    *out_count = n_emit;

    rcre_arena_temp_end(mark);
    return RCRE_OK;
}

int rcre_nlp_rrf_fuse(
    const RCRE_SearchHit* const* lists,
    const size_t* list_counts,
    size_t num_lists,
    float k_rrf,
    size_t top_k,
    RCRE_SearchHit* out_hits,
    size_t* out_count,
    RCRE_Arena* scratch
) {
    if (!out_count || !scratch || (num_lists > 0 && (!lists || !list_counts)) || (top_k > 0 && !out_hits)) {
        return RCRE_ERR_INVALID_ARG;
    }
    *out_count = 0;
    if (num_lists == 0 || top_k == 0) return RCRE_OK;
    if (k_rrf <= 0.0f) k_rrf = 60.0f;

    size_t total_entries = 0;
    for (size_t l = 0; l < num_lists; ++l) {
        total_entries += list_counts[l];
    }
    if (total_entries == 0) return RCRE_OK;

    RCRE_ArenaTemp mark = rcre_arena_temp_begin(scratch);
    /* Power-of-two open-addressed hash table keyed by doc_id */
    size_t cap = 16U;
    while (cap < total_entries * 2U) cap *= 2U;
    uint32_t* keys = (uint32_t*)rcre_arena_alloc(scratch, cap * sizeof(uint32_t), 64);
    float* vals    = (float*)rcre_arena_alloc(scratch, cap * sizeof(float), 64);
    if (!keys || !vals) {
        rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < cap; ++i) {
        keys[i] = UINT32_MAX;
        vals[i] = 0.0f;
    }

    size_t mask = cap - 1U;
    size_t unique = 0;
    for (size_t l = 0; l < num_lists; ++l) {
        const RCRE_SearchHit* lst = lists[l];
        size_t n = list_counts[l];
        if (!lst) continue;
        for (size_t r = 0; r < n; ++r) {
            uint32_t doc = lst[r].doc_id;
            if (doc == UINT32_MAX) continue;
            float inc = 1.0f / (k_rrf + (float)(r + 1U));
            size_t slot = (size_t)(rcre_nlp__splitmix64((uint64_t)doc) & (uint64_t)mask);
            for (;;) {
                if (keys[slot] == UINT32_MAX) {
                    keys[slot] = doc;
                    vals[slot] = inc;
                    unique++;
                    break;
                }
                if (keys[slot] == doc) {
                    vals[slot] += inc;
                    break;
                }
                slot = (slot + 1U) & mask;
            }
        }
    }

    RCRE_SearchHit* fused = (RCRE_SearchHit*)rcre_arena_alloc(
        scratch, unique * sizeof(RCRE_SearchHit), alignof(RCRE_SearchHit));
    if (!fused) {
        rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    size_t idx = 0;
    for (size_t i = 0; i < cap; ++i) {
        if (keys[i] != UINT32_MAX) {
            fused[idx].doc_id = keys[i];
            fused[idx].score = vals[i];
            idx++;
        }
    }
    rcre_nlp__sort_hits_desc(fused, unique);

    size_t n_emit = (unique < top_k) ? unique : top_k;
    memcpy(out_hits, fused, n_emit * sizeof(RCRE_SearchHit));
    *out_count = n_emit;
    rcre_arena_temp_end(mark);
    return RCRE_OK;
}

/* ========================================================================= */
/* 3. Scikit-Learn Parity TF-IDF Vectorizer & Feature Hashing (NLP-REQ-03)   */
/* ========================================================================= */

RCRE_TfidfOpts rcre_tfidf_default_opts(void) {
    RCRE_TfidfOpts opts = {
        .lowercase    = true,
        .smooth_idf   = true,
        .sublinear_tf = false,
        .norm         = RCRE_NLP_NORM_L2,
        .min_df       = 1,
        .max_df_ratio = 1.0f,
        .max_features = 0,
        .tok_opts     = rcre_nlp_default_tokenize_opts()
    };
    return opts;
}

int rcre_tfidf_fit(
    RCRE_TfidfVectorizer* vec,
    const RCRE_Slice* docs,
    size_t num_docs,
    const RCRE_TfidfOpts* opts,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
) {
    if (!vec || !arena || (num_docs > 0 && !docs)) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(vec, 0, sizeof(*vec));
    RCRE_TfidfOpts cfg = opts ? *opts : rcre_tfidf_default_opts();
    if (cfg.min_df == 0) cfg.min_df = 1U;
    if (cfg.max_df_ratio <= 0.0f || cfg.max_df_ratio > 1.0f) cfg.max_df_ratio = 1.0f;
    vec->opts = cfg;
    vec->num_fit_docs = num_docs;

    RCRE_Arena* tmp_arena = scratch ? scratch : arena;
    bool distinct_scratch = (scratch != nullptr && scratch != arena);
    RCRE_ArenaTemp mark = rcre_arena_temp_begin(tmp_arena);

    RCRE_StrPool raw_pool;
    int rc = rcre_strpool_init(&raw_pool, tmp_arena, num_docs > 64 ? num_docs * 2U : 128U);
    if (rc != RCRE_OK) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return rc;
    }

    uint8_t low_buf[256];
    for (size_t d = 0; d < num_docs; ++d) {
        RCRE_WordIter it = rcre_nlp_word_iter_init(docs[d], &cfg.tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, cfg.lowercase, low_buf, sizeof(low_buf), tmp_arena);
            if (rcre_strpool_intern(&raw_pool, norm) == RCRE_STR_ID_INVALID) {
                if (distinct_scratch) rcre_arena_temp_end(mark);
                return RCRE_ERR_OUT_OF_MEMORY;
            }
        }
    }

    size_t raw_V = rcre_strpool_count(&raw_pool);
    if (raw_V == 0) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return rcre_strpool_init(&vec->vocab, arena, 16);
    }

    uint32_t* df = (uint32_t*)rcre_arena_alloc(tmp_arena, raw_V * sizeof(uint32_t), 64);
    uint32_t* last_doc = (uint32_t*)rcre_arena_alloc(tmp_arena, raw_V * sizeof(uint32_t), 64);
    if (!df || !last_doc) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(df, 0, raw_V * sizeof(uint32_t));
    for (size_t t = 0; t < raw_V; ++t) last_doc[t] = UINT32_MAX;

    for (size_t d = 0; d < num_docs; ++d) {
        RCRE_WordIter it = rcre_nlp_word_iter_init(docs[d], &cfg.tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, cfg.lowercase, low_buf, sizeof(low_buf), tmp_arena);
            RCRE_StrId tid = rcre_strpool_find(&raw_pool, norm);
            if (tid != RCRE_STR_ID_INVALID && last_doc[tid] != (uint32_t)d) {
                last_doc[tid] = (uint32_t)d;
                df[tid]++;
            }
        }
    }

    /* Filter by min_df / max_df_ratio */
    uint32_t max_df = (uint32_t)floor((double)cfg.max_df_ratio * (double)num_docs + 1e-9);
    if (max_df < cfg.min_df) max_df = cfg.min_df;

    rcre_nlp__term_pair* kept = (rcre_nlp__term_pair*)rcre_arena_alloc(
        tmp_arena, raw_V * sizeof(rcre_nlp__term_pair), alignof(rcre_nlp__term_pair));
    if (!kept) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    size_t kept_n = 0;
    for (size_t t = 0; t < raw_V; ++t) {
        if (df[t] >= cfg.min_df && df[t] <= max_df) {
            kept[kept_n].slice = rcre_strpool_get(&raw_pool, (RCRE_StrId)t);
            kept[kept_n].old_id = (uint32_t)t;
            kept_n++;
        }
    }

    /* Sort alphabetically for deterministic scikit-learn column order */
    rcre_nlp__sort_terms(kept, kept_n);
    if (cfg.max_features > 0 && kept_n > cfg.max_features) {
        kept_n = cfg.max_features;
    }

    rc = rcre_strpool_init(&vec->vocab, arena, kept_n > 16 ? kept_n : 16);
    if (rc != RCRE_OK) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return rc;
    }

    vec->vocab_size = kept_n;
    if (kept_n > 0) {
        vec->idf = (float*)rcre_arena_alloc(arena, kept_n * sizeof(float), 64);
        if (!vec->idf) {
            if (distinct_scratch) rcre_arena_temp_end(mark);
            return RCRE_ERR_OUT_OF_MEMORY;
        }
        double N = (double)num_docs;
        for (size_t i = 0; i < kept_n; ++i) {
            RCRE_StrId new_id = rcre_strpool_intern(&vec->vocab, kept[i].slice);
            if (new_id == RCRE_STR_ID_INVALID) {
                if (distinct_scratch) rcre_arena_temp_end(mark);
                return RCRE_ERR_OUT_OF_MEMORY;
            }
            double df_i = (double)df[kept[i].old_id];
            if (cfg.smooth_idf) {
                vec->idf[new_id] = (float)(log((1.0 + N) / (1.0 + df_i)) + 1.0);
            } else {
                vec->idf[new_id] = (float)(log(N / df_i) + 1.0);
            }
        }
    }

    if (distinct_scratch) {
        rcre_arena_temp_end(mark);
    }
    return RCRE_OK;
}

int rcre_tfidf_transform(
    const RCRE_TfidfVectorizer* vec,
    const RCRE_Slice* docs,
    size_t num_docs,
    RCRE_Arena* arena,
    RCRE_Arena* scratch,
    RCRE_CSRMatrix* out_csr
) {
    if (!vec || !arena || !out_csr || (num_docs > 0 && !docs)) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(out_csr, 0, sizeof(*out_csr));
    out_csr->rows = num_docs;
    out_csr->cols = vec->vocab_size;

    uint32_t* row_ptrs = (uint32_t*)rcre_arena_alloc(arena, (num_docs + 1U) * sizeof(uint32_t), 64);
    if (!row_ptrs) return RCRE_ERR_OUT_OF_MEMORY;
    memset(row_ptrs, 0, (num_docs + 1U) * sizeof(uint32_t));
    out_csr->row_pointers = row_ptrs;

    if (num_docs == 0 || vec->vocab_size == 0) {
        return RCRE_OK;
    }

    RCRE_Arena* tmp_arena = scratch ? scratch : arena;
    bool distinct_scratch = (scratch != nullptr && scratch != arena);
    RCRE_ArenaTemp mark = rcre_arena_temp_begin(tmp_arena);

    size_t V = vec->vocab_size;
    uint32_t* last_doc = (uint32_t*)rcre_arena_alloc(tmp_arena, V * sizeof(uint32_t), 64);
    uint32_t* doc_tf   = (uint32_t*)rcre_arena_alloc(tmp_arena, V * sizeof(uint32_t), 64);
    uint32_t* active   = (uint32_t*)rcre_arena_alloc(tmp_arena, V * sizeof(uint32_t), 64);
    if (!last_doc || !doc_tf || !active) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    for (size_t t = 0; t < V; ++t) last_doc[t] = UINT32_MAX;

    /* Pass 1: exact nnz per row */
    uint8_t low_buf[256];
    size_t total_nnz = 0;
    for (size_t d = 0; d < num_docs; ++d) {
        row_ptrs[d] = (uint32_t)total_nnz;
        RCRE_WordIter it = rcre_nlp_word_iter_init(docs[d], &vec->opts.tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, vec->opts.lowercase, low_buf, sizeof(low_buf), tmp_arena);
            RCRE_StrId tid = rcre_strpool_find(&vec->vocab, norm);
            if (tid != RCRE_STR_ID_INVALID && last_doc[tid] != (uint32_t)d) {
                last_doc[tid] = (uint32_t)d;
                total_nnz++;
            }
        }
    }
    row_ptrs[num_docs] = (uint32_t)total_nnz;
    out_csr->nnz = total_nnz;

    if (total_nnz == 0) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_OK;
    }

    float* values = (float*)rcre_arena_alloc(arena, total_nnz * sizeof(float), 64);
    uint32_t* col_idx = (uint32_t*)rcre_arena_alloc(arena, total_nnz * sizeof(uint32_t), 64);
    if (!values || !col_idx) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    out_csr->values = values;
    out_csr->col_indices = col_idx;

    for (size_t t = 0; t < V; ++t) last_doc[t] = UINT32_MAX;

    /* Pass 2: populate sorted column indices & normalized TF-IDF weights */
    for (size_t d = 0; d < num_docs; ++d) {
        size_t n_act = 0;
        RCRE_WordIter it = rcre_nlp_word_iter_init(docs[d], &vec->opts.tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, vec->opts.lowercase, low_buf, sizeof(low_buf), tmp_arena);
            RCRE_StrId tid = rcre_strpool_find(&vec->vocab, norm);
            if (tid == RCRE_STR_ID_INVALID) continue;
            if (last_doc[tid] != (uint32_t)d) {
                last_doc[tid] = (uint32_t)d;
                doc_tf[tid] = 1U;
                active[n_act++] = tid;
            } else {
                doc_tf[tid]++;
            }
        }
        rcre_nlp__sort_u32(active, n_act);

        uint32_t base = row_ptrs[d];
        double norm_acc = 0.0;
        for (size_t k = 0; k < n_act; ++k) {
            uint32_t tid = active[k];
            double tf = (double)doc_tf[tid];
            if (vec->opts.sublinear_tf) {
                tf = 1.0 + log(tf);
            }
            double w = tf * (double)vec->idf[tid];
            col_idx[base + k] = tid;
            values[base + k] = (float)w;
            if (vec->opts.norm == RCRE_NLP_NORM_L2) {
                norm_acc += w * w;
            } else if (vec->opts.norm == RCRE_NLP_NORM_L1) {
                norm_acc += fabs(w);
            }
        }

        if (vec->opts.norm == RCRE_NLP_NORM_L2 && norm_acc > 0.0) {
            float inv = (float)(1.0 / sqrt(norm_acc));
            for (size_t k = 0; k < n_act; ++k) values[base + k] *= inv;
        } else if (vec->opts.norm == RCRE_NLP_NORM_L1 && norm_acc > 0.0) {
            float inv = (float)(1.0 / norm_acc);
            for (size_t k = 0; k < n_act; ++k) values[base + k] *= inv;
        }
    }

    if (distinct_scratch) {
        rcre_arena_temp_end(mark);
    }
    return RCRE_OK;
}

int rcre_nlp_hash_vectorize(
    const RCRE_Slice* docs,
    size_t num_docs,
    size_t n_features,
    bool lowercase,
    RCRE_NlpNorm norm,
    const RCRE_TokenizeOpts* tok_opts,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
) {
    if (!arena || !out_tensor || n_features == 0 || (num_docs > 0 && !docs)) {
        return RCRE_ERR_INVALID_ARG;
    }
    size_t shape[2] = { num_docs > 0 ? num_docs : 1U, n_features };
    RCRE_Tensor t = rcre_tensor_create(shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
    if (!t.data) return RCRE_ERR_OUT_OF_MEMORY;
    float* mat = rcre_tensor_f32_data(&t);
    memset(mat, 0, shape[0] * n_features * sizeof(float));

    uint8_t low_buf[256];
    for (size_t d = 0; d < num_docs; ++d) {
        float* row = mat + d * n_features;
        RCRE_WordIter it = rcre_nlp_word_iter_init(docs[d], tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm_tok = rcre_nlp__normalize_tok(tok, lowercase, low_buf, sizeof(low_buf), nullptr);
            uint64_t h = rcre_slice_hash64(norm_tok);
            size_t bucket = (size_t)(h % (uint64_t)n_features);
            float sign = ((h >> 63) != 0ULL) ? -1.0f : 1.0f;
            row[bucket] += sign;
        }
        if (norm == RCRE_NLP_NORM_L2) {
            float dot = rcre_hal.dot_f32(row, row, n_features);
            if (dot > 0.0f) {
                rcre_hal.scale_f32(n_features, 1.0f / sqrtf(dot), row);
            }
        } else if (norm == RCRE_NLP_NORM_L1) {
            double s = 0.0;
            for (size_t j = 0; j < n_features; ++j) s += fabs((double)row[j]);
            if (s > 0.0) {
                rcre_hal.scale_f32(n_features, (float)(1.0 / s), row);
            }
        }
    }

    *out_tensor = t;
    return RCRE_OK;
}

/* ========================================================================= */
/* 4. Aho-Corasick Flat DFA Multi-Pattern Automaton (NLP-REQ-04)             */
/* ========================================================================= */

int rcre_ac_build(
    RCRE_AhoCorasick* ac,
    const RCRE_Slice* patterns,
    size_t num_patterns,
    bool case_insensitive,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
) {
    if (!ac || !arena || (num_patterns > 0 && !patterns) || num_patterns > (size_t)UINT32_MAX - 1U) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(ac, 0, sizeof(*ac));
    ac->case_insensitive = case_insensitive;
    ac->num_patterns = (uint32_t)num_patterns;

    size_t max_states = 1U;
    for (size_t p = 0; p < num_patterns; ++p) {
        if (patterns[p].len > SIZE_MAX - max_states) return RCRE_ERR_INVALID_ARG;
        max_states += patterns[p].len;
    }
    if (max_states > (size_t)UINT32_MAX / 256U) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    RCRE_Arena* tmp_arena = scratch ? scratch : arena;
    bool distinct_scratch = (scratch != nullptr && scratch != arena);
    RCRE_ArenaTemp mark = rcre_arena_temp_begin(tmp_arena);

    ac->next = (uint32_t*)rcre_arena_alloc(arena, max_states * 256U * sizeof(uint32_t), 64);
    ac->out_head = (uint32_t*)rcre_arena_alloc(arena, max_states * sizeof(uint32_t), 64);
    ac->pat_lens = num_patterns > 0 ? (size_t*)rcre_arena_alloc(arena, num_patterns * sizeof(size_t), 64) : nullptr;
    /* Upper bound on direct output nodes before failure-link expansion: num_patterns */
    uint32_t* fail = (uint32_t*)rcre_arena_alloc(tmp_arena, max_states * sizeof(uint32_t), 64);
    uint32_t* queue = (uint32_t*)rcre_arena_alloc(tmp_arena, max_states * sizeof(uint32_t), 64);
    uint32_t* direct_head = (uint32_t*)rcre_arena_alloc(tmp_arena, max_states * sizeof(uint32_t), 64);
    uint32_t* direct_pat  = num_patterns > 0 ? (uint32_t*)rcre_arena_alloc(tmp_arena, num_patterns * sizeof(uint32_t), 64) : nullptr;
    uint32_t* direct_next = num_patterns > 0 ? (uint32_t*)rcre_arena_alloc(tmp_arena, num_patterns * sizeof(uint32_t), 64) : nullptr;

    if (!ac->next || !ac->out_head || !fail || !queue || !direct_head ||
        (num_patterns > 0 && (!ac->pat_lens || !direct_pat || !direct_next))) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    memset(ac->next, 0, max_states * 256U * sizeof(uint32_t));
    for (size_t s = 0; s < max_states; ++s) {
        ac->out_head[s] = RCRE_AC_NONE;
        direct_head[s] = RCRE_AC_NONE;
        fail[s] = 0;
    }

    /* Step 1: Build Trie (canonical lowercase byte if case_insensitive) */
    uint32_t num_states = 1U;
    uint32_t direct_cnt = 0U;
    for (size_t p = 0; p < num_patterns; ++p) {
        ac->pat_lens[p] = patterns[p].len;
        if (patterns[p].len == 0 || !patterns[p].data) continue;
        uint32_t cur = 0;
        for (size_t i = 0; i < patterns[p].len; ++i) {
            uint8_t c = patterns[p].data[i];
            if (case_insensitive) c = rcre_nlp__lower_byte(c);
            uint32_t nxt = ac->next[(size_t)cur * 256U + c];
            if (nxt == 0) {
                nxt = num_states++;
                ac->next[(size_t)cur * 256U + c] = nxt;
            }
            cur = nxt;
        }
        direct_pat[direct_cnt] = (uint32_t)p;
        direct_next[direct_cnt] = direct_head[cur];
        direct_head[cur] = direct_cnt++;
    }
    ac->num_states = num_states;

    /* Step 2: BFS failure links + full DFA transition completion */
    size_t qh = 0, qt = 0;
    for (size_t c = 0; c < 256U; ++c) {
        uint32_t nxt = ac->next[c];
        if (nxt != 0) {
            fail[nxt] = 0;
            queue[qt++] = nxt;
        }
    }

    while (qh < qt) {
        uint32_t u = queue[qh++];
        uint32_t f = fail[u];
        uint32_t* u_row = ac->next + (size_t)u * 256U;
        const uint32_t* f_row = ac->next + (size_t)f * 256U;
        for (size_t c = 0; c < 256U; ++c) {
            uint32_t v = u_row[c];
            if (v != 0) {
                fail[v] = f_row[c];
                queue[qt++] = v;
            } else {
                u_row[c] = f_row[c];
            }
        }
    }

    /* Bake ASCII uppercase -> lowercase transitions directly into DFA (0 runtime cost!) */
    if (case_insensitive) {
        for (uint32_t s = 0; s < num_states; ++s) {
            uint32_t* row = ac->next + (size_t)s * 256U;
            for (size_t c = 'A'; c <= 'Z'; ++c) {
                row[c] = row[c | 0x20U];
            }
        }
    }

    /* Step 3: Materialize flattened output lists per state (including failure outputs) */
    size_t total_out_nodes = 0;
    for (uint32_t s = 0; s < num_states; ++s) {
        uint32_t cur = s;
        while (cur != 0) {
            for (uint32_t n = direct_head[cur]; n != RCRE_AC_NONE; n = direct_next[n]) {
                total_out_nodes++;
            }
            cur = fail[cur];
        }
    }

    if (total_out_nodes > 0) {
        ac->out_pat_id = (uint32_t*)rcre_arena_alloc(arena, total_out_nodes * sizeof(uint32_t), 64);
        ac->out_next   = (uint32_t*)rcre_arena_alloc(arena, total_out_nodes * sizeof(uint32_t), 64);
        if (!ac->out_pat_id || !ac->out_next) {
            if (distinct_scratch) rcre_arena_temp_end(mark);
            return RCRE_ERR_OUT_OF_MEMORY;
        }
        uint32_t out_idx = 0;
        for (uint32_t s = 0; s < num_states; ++s) {
            uint32_t cur = s;
            while (cur != 0) {
                for (uint32_t n = direct_head[cur]; n != RCRE_AC_NONE; n = direct_next[n]) {
                    ac->out_pat_id[out_idx] = direct_pat[n];
                    ac->out_next[out_idx] = ac->out_head[s];
                    ac->out_head[s] = out_idx++;
                }
                cur = fail[cur];
            }
        }
        ac->num_out_nodes = out_idx;
    }

    if (distinct_scratch) {
        rcre_arena_temp_end(mark);
    }
    return RCRE_OK;
}

size_t rcre_ac_scan(
    const RCRE_AhoCorasick* ac,
    RCRE_Slice text,
    rcre_ac_match_cb cb,
    void* user_ctx
) {
    if (!ac || !ac->next || ac->num_states == 0 || !text.data || text.len == 0) {
        return 0;
    }
    uint32_t state = 0;
    size_t matches = 0;
    const uint32_t* next = ac->next;
    const uint32_t* out_head = ac->out_head;

    for (size_t i = 0; i < text.len; ++i) {
        state = next[(size_t)state * 256U + text.data[i]];
        uint32_t node = out_head[state];
        if (node != RCRE_AC_NONE) {
            do {
                uint32_t pid = ac->out_pat_id[node];
                size_t plen = ac->pat_lens[pid];
                matches++;
                if (cb) {
                    RCRE_ACMatch m = {
                        .start = i + 1U - plen,
                        .end   = i + 1U,
                        .pattern_id = pid
                    };
                    if (!cb(m, user_ctx)) {
                        return matches;
                    }
                }
                node = ac->out_next[node];
            } while (node != RCRE_AC_NONE);
        }
    }
    return matches;
}

typedef struct {
    RCRE_ACMatch* buf;
    size_t        cap;
    size_t        written;
} rcre_nlp__ac_collect_ctx;

static bool rcre_nlp__ac_collect_cb(RCRE_ACMatch m, void* ctx_ptr) {
    rcre_nlp__ac_collect_ctx* ctx = (rcre_nlp__ac_collect_ctx*)ctx_ptr;
    if (ctx->buf && ctx->written < ctx->cap) {
        ctx->buf[ctx->written++] = m;
    }
    return true;
}

size_t rcre_ac_find_all(
    const RCRE_AhoCorasick* ac,
    RCRE_Slice text,
    RCRE_ACMatch* out_matches,
    size_t max_matches
) {
    rcre_nlp__ac_collect_ctx ctx = { .buf = out_matches, .cap = max_matches, .written = 0 };
    return rcre_ac_scan(ac, text, rcre_nlp__ac_collect_cb, &ctx);
}

RCRE_Slice rcre_ac_redact(
    const RCRE_AhoCorasick* ac,
    RCRE_Slice text,
    RCRE_Slice replacement,
    RCRE_Arena* arena
) {
    if (!arena || text.len == 0 || !text.data) return rcre_slice_empty();
    if (!ac || !ac->next || ac->num_states == 0) {
        return rcre_str_dup(arena, text);
    }

    RCRE_StrBuilder sb = rcre_sb_init(arena, text.len);
    size_t emitted = 0;
    uint32_t state = 0;

    for (size_t i = 0; i < text.len; ++i) {
        state = ac->next[(size_t)state * 256U + text.data[i]];
        uint32_t node = ac->out_head[state];
        if (node != RCRE_AC_NONE) {
            /* Pick longest match starting at or after emitted boundary */
            size_t best_start = SIZE_MAX;
            while (node != RCRE_AC_NONE) {
                uint32_t pid = ac->out_pat_id[node];
                size_t start = i + 1U - ac->pat_lens[pid];
                if (start >= emitted && start < best_start) {
                    best_start = start;
                }
                node = ac->out_next[node];
            }
            if (best_start != SIZE_MAX) {
                if (best_start > emitted) {
                    if (!rcre_sb_append_slice(&sb, rcre_slice_sub(text, emitted, best_start))) {
                        return rcre_slice_empty();
                    }
                }
                if (!rcre_sb_append_slice(&sb, replacement)) {
                    return rcre_slice_empty();
                }
                emitted = i + 1U;
                state = 0; /* Non-overlapping reset after redaction */
            }
        }
    }
    if (emitted < text.len) {
        if (!rcre_sb_append_slice(&sb, rcre_slice_sub(text, emitted, text.len))) {
            return rcre_slice_empty();
        }
    }
    return rcre_sb_to_slice(&sb);
}

/* ========================================================================= */
/* 5. Bit-Parallel Myers Edit Distance & Fuzzy Metrics (NLP-REQ-05)          */
/* ========================================================================= */

static size_t rcre_nlp__levenshtein_myers64(const uint8_t* pat, size_t m, const uint8_t* txt, size_t n) {
    uint64_t peq[256];
    memset(peq, 0, sizeof(peq));
    for (size_t i = 0; i < m; ++i) {
        peq[pat[i]] |= (1ULL << i);
    }

    uint64_t pv = ~0ULL;
    uint64_t mv = 0ULL;
    size_t score = m;
    uint64_t high_bit = 1ULL << (m - 1U);

    for (size_t j = 0; j < n; ++j) {
        uint64_t eq = peq[txt[j]];
        uint64_t xv = eq | mv;
        uint64_t xh = (((eq & pv) + pv) ^ pv) | eq;

        uint64_t ph = mv | ~(xh | pv);
        uint64_t mh = pv & xh;

        if (ph & high_bit)      score++;
        else if (mh & high_bit) score--;

        ph = (ph << 1) | 1ULL;
        mh = (mh << 1);
        pv = mh | ~(xv | ph);
        mv = ph & xv;
    }
    return score;
}

size_t rcre_nlp_levenshtein_scalar(RCRE_Slice a, RCRE_Slice b, RCRE_Arena* scratch) {
    if (a.len == 0) return b.len;
    if (b.len == 0) return a.len;
    if (a.len > b.len) {
        RCRE_Slice tmp = a; a = b; b = tmp;
    }
    size_t m = a.len;
    size_t n = b.len;

    uint32_t stack_row[1025];
    uint32_t* prev = stack_row;
    RCRE_ArenaTemp mark = { .arena = nullptr, .saved_offset = 0 };
    if (m + 1U > sizeof(stack_row) / sizeof(stack_row[0])) {
        if (!scratch) return RCRE_NPOS;
        mark = rcre_arena_temp_begin(scratch);
        prev = (uint32_t*)rcre_arena_alloc(scratch, (m + 1U) * sizeof(uint32_t), 64);
        if (!prev) {
            rcre_arena_temp_end(mark);
            return RCRE_NPOS;
        }
    }

    for (size_t i = 0; i <= m; ++i) prev[i] = (uint32_t)i;
    for (size_t j = 1U; j <= n; ++j) {
        uint32_t diag = prev[0];
        prev[0] = (uint32_t)j;
        uint8_t bj = b.data[j - 1U];
        for (size_t i = 1U; i <= m; ++i) {
            uint32_t up = prev[i];
            uint32_t cost = (a.data[i - 1U] == bj) ? 0U : 1U;
            uint32_t ins = prev[i - 1U] + 1U;
            uint32_t del = up + 1U;
            uint32_t sub = diag + cost;
            uint32_t best = (ins < del) ? ins : del;
            if (sub < best) best = sub;
            prev[i] = best;
            diag = up;
        }
    }

    size_t ans = (size_t)prev[m];
    if (mark.arena) rcre_arena_temp_end(mark);
    return ans;
}

size_t rcre_nlp_levenshtein(RCRE_Slice a, RCRE_Slice b, RCRE_Arena* scratch) {
    if (a.len == 0) return b.len;
    if (b.len == 0) return a.len;
    if (a.len == b.len && memcmp(a.data, b.data, a.len) == 0) return 0;

    /* Strip common prefix & suffix in O(prefix + suffix) to shrink work */
    size_t pre = 0;
    size_t min_len = (a.len < b.len) ? a.len : b.len;
    while (pre < min_len && a.data[pre] == b.data[pre]) pre++;
    a = rcre_slice_sub(a, pre, a.len);
    b = rcre_slice_sub(b, pre, b.len);
    while (a.len > 0 && b.len > 0 && a.data[a.len - 1U] == b.data[b.len - 1U]) {
        a.len--;
        b.len--;
    }
    if (a.len == 0) return b.len;
    if (b.len == 0) return a.len;

    /* Put shorter string as pattern so Myers 64-bit kicks in whenever min(m,n) <= 64 */
    if (a.len > b.len) {
        RCRE_Slice tmp = a; a = b; b = tmp;
    }
    if (a.len <= 64U) {
        return rcre_nlp__levenshtein_myers64(a.data, a.len, b.data, b.len);
    }
    return rcre_nlp_levenshtein_scalar(a, b, scratch);
}

float rcre_nlp_levenshtein_sim(RCRE_Slice a, RCRE_Slice b, RCRE_Arena* scratch) {
    if (a.len == 0 && b.len == 0) return 1.0f;
    size_t d = rcre_nlp_levenshtein(a, b, scratch);
    if (d == RCRE_NPOS) return 0.0f;
    size_t denom = (a.len > b.len) ? a.len : b.len;
    return 1.0f - (float)((double)d / (double)denom);
}

float rcre_nlp_jaro_winkler(RCRE_Slice a, RCRE_Slice b, float prefix_scale, RCRE_Arena* scratch) {
    if (a.len == 0 && b.len == 0) return 1.0f;
    if (a.len == 0 || b.len == 0) return 0.0f;
    if (rcre_slice_eq(a, b)) return 1.0f;

    if (prefix_scale < 0.0f) prefix_scale = 0.0f;
    if (prefix_scale > 0.25f) prefix_scale = 0.25f;

    size_t max_len = (a.len > b.len) ? a.len : b.len;
    size_t match_dist = (max_len / 2U > 0U) ? (max_len / 2U - 1U) : 0U;

    uint8_t stack_flags[1024];
    uint8_t* a_match = stack_flags;
    uint8_t* b_match = stack_flags + a.len;
    RCRE_ArenaTemp mark = { .arena = nullptr, .saved_offset = 0 };

    if (a.len + b.len > sizeof(stack_flags)) {
        if (!scratch) return 0.0f;
        mark = rcre_arena_temp_begin(scratch);
        a_match = (uint8_t*)rcre_arena_alloc(scratch, a.len + b.len, 16);
        if (!a_match) {
            rcre_arena_temp_end(mark);
            return 0.0f;
        }
        b_match = a_match + a.len;
    }
    memset(a_match, 0, a.len + b.len);

    size_t matches = 0;
    for (size_t i = 0; i < a.len; ++i) {
        size_t start = (i > match_dist) ? (i - match_dist) : 0U;
        size_t end = (i + match_dist + 1U < b.len) ? (i + match_dist + 1U) : b.len;
        for (size_t j = start; j < end; ++j) {
            if (!b_match[j] && a.data[i] == b.data[j]) {
                a_match[i] = 1U;
                b_match[j] = 1U;
                matches++;
                break;
            }
        }
    }

    if (matches == 0) {
        if (mark.arena) rcre_arena_temp_end(mark);
        return 0.0f;
    }

    size_t transpositions = 0;
    size_t k = 0;
    for (size_t i = 0; i < a.len; ++i) {
        if (!a_match[i]) continue;
        while (!b_match[k]) k++;
        if (a.data[i] != b.data[k]) transpositions++;
        k++;
    }

    double m = (double)matches;
    double jaro = ((m / (double)a.len) + (m / (double)b.len) + ((m - (double)transpositions / 2.0) / m)) / 3.0;

    size_t prefix = 0;
    size_t max_pre = (a.len < b.len) ? a.len : b.len;
    if (max_pre > 4U) max_pre = 4U;
    while (prefix < max_pre && a.data[prefix] == b.data[prefix]) prefix++;

    double jw = jaro + (double)prefix * (double)prefix_scale * (1.0 - jaro);
    if (mark.arena) rcre_arena_temp_end(mark);
    return (float)jw;
}

float rcre_nlp_jaccard_char_ngrams(RCRE_Slice a, RCRE_Slice b, size_t n, RCRE_Arena* scratch) {
    if (!scratch) return 0.0f;
    if (n == 0) n = 1U;

    RCRE_ArenaTemp mark = rcre_arena_temp_begin(scratch);
    RCRE_StrPool set_a, set_b;
    if (rcre_strpool_init(&set_a, scratch, 64) != RCRE_OK ||
        rcre_strpool_init(&set_b, scratch, 64) != RCRE_OK) {
        rcre_arena_temp_end(mark);
        return 0.0f;
    }

    RCRE_CharNGramIter ita = rcre_nlp_char_ngram_init(a, n);
    RCRE_Slice ng;
    while (rcre_nlp_char_ngram_next(&ita, &ng)) {
        (void)rcre_strpool_intern(&set_a, ng);
    }
    RCRE_CharNGramIter itb = rcre_nlp_char_ngram_init(b, n);
    while (rcre_nlp_char_ngram_next(&itb, &ng)) {
        (void)rcre_strpool_intern(&set_b, ng);
    }

    size_t na = rcre_strpool_count(&set_a);
    size_t nb = rcre_strpool_count(&set_b);
    if (na == 0 && nb == 0) {
        rcre_arena_temp_end(mark);
        return 1.0f;
    }

    size_t inter = 0;
    for (size_t i = 0; i < na; ++i) {
        if (rcre_strpool_find(&set_b, rcre_strpool_get(&set_a, (RCRE_StrId)i)) != RCRE_STR_ID_INVALID) {
            inter++;
        }
    }
    size_t uni = na + nb - inter;
    float res = (uni > 0) ? (float)((double)inter / (double)uni) : 0.0f;
    rcre_arena_temp_end(mark);
    return res;
}

/* ========================================================================= */
/* 6. LLM Dataset Deduplication & UTF-8 RAG Chunker (NLP-REQ-06)             */
/* ========================================================================= */

uint64_t rcre_nlp_simhash64(RCRE_Slice text, const RCRE_TokenizeOpts* opts) {
    int32_t acc[64];
    memset(acc, 0, sizeof(acc));

    uint8_t low_buf[256];
    RCRE_WordIter it = rcre_nlp_word_iter_init(text, opts);
    RCRE_Slice tok;
    while (rcre_nlp_word_iter_next(&it, &tok)) {
        RCRE_Slice norm = rcre_nlp__normalize_tok(tok, true, low_buf, sizeof(low_buf), nullptr);
        uint64_t h = rcre_slice_hash64(norm);
        for (int b = 0; b < 64; ++b) {
            acc[b] += ((h >> b) & 1ULL) ? 1 : -1;
        }
    }

    uint64_t fp = 0;
    for (int b = 0; b < 64; ++b) {
        if (acc[b] > 0) {
            fp |= (1ULL << b);
        }
    }
    return fp;
}

uint32_t rcre_nlp_hamming64(uint64_t a, uint64_t b) {
    uint64_t x = a ^ b;
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_popcountll(x);
#else
    uint32_t c = 0;
    while (x) { x &= (x - 1ULL); c++; }
    return c;
#endif
}

int rcre_nlp_minhash_compute(
    RCRE_Slice text,
    size_t ngram_n,
    uint64_t* out_sig,
    size_t num_hashes,
    const RCRE_TokenizeOpts* opts
) {
    if (!out_sig || num_hashes == 0) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (ngram_n == 0) ngram_n = 1U;
    for (size_t k = 0; k < num_hashes; ++k) {
        out_sig[k] = UINT64_MAX;
    }

    uint8_t low_buf[512];
    RCRE_WordNGramIter it = rcre_nlp_word_ngram_init(text, ngram_n, opts);
    RCRE_Slice span;
    while (rcre_nlp_word_ngram_next(&it, &span)) {
        RCRE_Slice norm = rcre_nlp__normalize_tok(span, true, low_buf, sizeof(low_buf), nullptr);
        uint64_t base_h = rcre_slice_hash64(norm);
        for (size_t k = 0; k < num_hashes; ++k) {
            uint64_t seed = 0x243f6a8885a308d3ULL ^ ((uint64_t)(k + 1U) * 0x9e3779b97f4a7c15ULL);
            uint64_t hk = rcre_nlp__splitmix64(base_h ^ seed);
            if (hk < out_sig[k]) {
                out_sig[k] = hk;
            }
        }
    }
    return RCRE_OK;
}

float rcre_nlp_minhash_jaccard(const uint64_t* sig_a, const uint64_t* sig_b, size_t num_hashes) {
    if (!sig_a || !sig_b || num_hashes == 0) return 0.0f;
    size_t eq = 0;
    for (size_t k = 0; k < num_hashes; ++k) {
        if (sig_a[k] == sig_b[k]) eq++;
    }
    return (float)((double)eq / (double)num_hashes);
}

int rcre_nlp_chunk_utf8(
    RCRE_Slice text,
    size_t max_bytes,
    size_t overlap_bytes,
    RCRE_Arena* arena,
    RCRE_Slice** out_chunks,
    size_t* out_count
) {
    if (!arena || !out_chunks || !out_count || max_bytes == 0) {
        return RCRE_ERR_INVALID_ARG;
    }
    *out_chunks = nullptr;
    *out_count = 0;
    if (!text.data || text.len == 0) return RCRE_OK;
    if (overlap_bytes >= max_bytes) overlap_bytes = max_bytes - 1U;

    /* Pass 1: count chunks; Pass 2: fill slice array */
    for (int pass = 0; pass < 2; ++pass) {
        size_t pos = 0;
        size_t count = 0;
        while (pos < text.len) {
            RCRE_Slice rem = rcre_slice_sub(text, pos, text.len);
            RCRE_Slice fit = rcre_utf8_truncate_bytes(rem, max_bytes);
            if (fit.len == 0) {
                /* Single codepoint > max_bytes or malformed leading byte: consume 1 byte */
                fit = rcre_slice_sub(rem, 0, 1U);
            } else if (pos + fit.len < text.len) {
                /* Snap backward to last ASCII whitespace if within second half of chunk */
                size_t snap = fit.len;
                size_t floor_pos = fit.len / 2U;
                while (snap > floor_pos && !rcre_nlp__is_ascii_space(fit.data[snap - 1U])) {
                    snap--;
                }
                if (snap > floor_pos) {
                    fit = rcre_slice_trim_right_ascii(rcre_slice_sub(fit, 0, snap));
                    if (fit.len == 0) fit = rcre_utf8_truncate_bytes(rem, max_bytes);
                }
            }

            if (pass == 1) {
                (*out_chunks)[count] = fit;
            }
            count++;

            size_t end_pos = pos + fit.len;
            if (end_pos >= text.len) break;

            size_t next_pos = (fit.len > overlap_bytes) ? (end_pos - overlap_bytes) : (pos + 1U);
            /* Advance next_pos off UTF-8 continuation bytes (10xxxxxx) */
            while (next_pos < text.len && ((text.data[next_pos] & 0xC0U) == 0x80U)) {
                next_pos++;
            }
            /* Skip leading whitespace on fresh non-overlapping starts */
            if (overlap_bytes == 0) {
                while (next_pos < text.len && rcre_nlp__is_ascii_space(text.data[next_pos])) {
                    next_pos++;
                }
            }
            if (next_pos <= pos) next_pos = pos + 1U;
            pos = next_pos;
        }

        if (pass == 0) {
            *out_count = count;
            if (count == 0) return RCRE_OK;
            *out_chunks = (RCRE_Slice*)rcre_arena_alloc(arena, count * sizeof(RCRE_Slice), alignof(RCRE_Slice));
            if (!*out_chunks) {
                *out_count = 0;
                return RCRE_ERR_OUT_OF_MEMORY;
            }
        }
    }
    return RCRE_OK;
}

/* ========================================================================= */
/* 7. Fast Sparse-Sparse CSR Row Cosine Similarity (NLP-REQ-10)              */
/* ========================================================================= */

float rcre_nlp_csr_row_cosine(
    const RCRE_CSRMatrix* csr_a, size_t row_a,
    const RCRE_CSRMatrix* csr_b, size_t row_b
) {
    if (!csr_a || !csr_b || row_a >= csr_a->rows || row_b >= csr_b->rows ||
        !csr_a->row_pointers || !csr_b->row_pointers) {
        return 0.0f;
    }
    uint32_t pa = csr_a->row_pointers[row_a];
    uint32_t ea = csr_a->row_pointers[row_a + 1U];
    uint32_t pb = csr_b->row_pointers[row_b];
    uint32_t eb = csr_b->row_pointers[row_b + 1U];
    if (pa == ea || pb == eb) return 0.0f;

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint32_t i = pa; i < ea; ++i) {
        double v = (double)csr_a->values[i];
        na += v * v;
    }
    for (uint32_t j = pb; j < eb; ++j) {
        double v = (double)csr_b->values[j];
        nb += v * v;
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0f;

    /* Two-pointer sorted merge over CSR column indices */
    uint32_t i = pa, j = pb;
    while (i < ea && j < eb) {
        uint32_t ca = csr_a->col_indices[i];
        uint32_t cb = csr_b->col_indices[j];
        if (ca == cb) {
            dot += (double)csr_a->values[i] * (double)csr_b->values[j];
            i++;
            j++;
        } else if (ca < cb) {
            i++;
        } else {
            j++;
        }
    }
    return (float)(dot / sqrt(na * nb));
}

/* ========================================================================= */
/* 8. Topic Modeling: Sparse HALS NMF & Collapsed Gibbs LDA (NLP-REQ-07)     */
/* ========================================================================= */

RCRE_NmfOpts rcre_nmf_default_opts(size_t num_topics) {
    RCRE_NmfOpts opts = {
        .num_topics = (num_topics == 0) ? 8U : num_topics,
        .max_iter   = 50U,
        .tol        = 1e-4f,
        .seed       = 42ULL
    };
    return opts;
}

static void rcre_nlp__hals_update_W(
    const RCRE_CSRMatrix* X,
    const float* H,
    float* W,
    size_t K,
    float* B,       /* [N * K] scratch */
    float* P        /* [K * K] scratch */
) {
    const size_t N = X->rows;
    const size_t V = X->cols;
    const float eps = 1e-12f;

    /* P = H * H^T [K x K] using HAL SIMD dot */
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = k; j < K; ++j) {
            float d = rcre_hal.dot_f32(H + k * V, H + j * V, V);
            P[k * K + j] = d;
            P[j * K + k] = d;
        }
    }

    /* B = X * H^T [N x K] directly from sparse CSR (O(K * nnz)) */
    memset(B, 0, N * K * sizeof(float));
    for (size_t d = 0; d < N; ++d) {
        uint32_t p0 = X->row_pointers[d];
        uint32_t p1 = X->row_pointers[d + 1U];
        float* brow = B + d * K;
        for (uint32_t p = p0; p < p1; ++p) {
            uint32_t col = X->col_indices[p];
            float val = X->values[p];
            for (size_t k = 0; k < K; ++k) {
                brow[k] += val * H[k * V + col];
            }
        }
    }

    /* Coordinate-wise HALS update across topics k = 0..K-1 */
    for (size_t k = 0; k < K; ++k) {
        float denom = P[k * K + k];
        if (denom < eps) continue;
        float inv_denom = 1.0f / denom;
        for (size_t d = 0; d < N; ++d) {
            float* wrow = W + d * K;
            float proj = 0.0f;
            for (size_t j = 0; j < K; ++j) {
                proj += wrow[j] * P[j * K + k];
            }
            float upd = wrow[k] + (B[d * K + k] - proj) * inv_denom;
            wrow[k] = (upd > eps) ? upd : eps;
        }
    }
}

int rcre_nmf_fit(
    RCRE_NmfModel* model,
    const RCRE_CSRMatrix* X_csr,
    const RCRE_NmfOpts* opts,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
) {
    if (!model || !X_csr || !arena || X_csr->rows == 0 || X_csr->cols == 0 || !X_csr->row_pointers) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(model, 0, sizeof(*model));
    RCRE_NmfOpts cfg = opts ? *opts : rcre_nmf_default_opts(8);
    if (cfg.num_topics == 0) cfg.num_topics = 1U;
    if (cfg.max_iter == 0) cfg.max_iter = 1U;

    const size_t N = X_csr->rows;
    const size_t V = X_csr->cols;
    const size_t K = cfg.num_topics;
    const float eps = 1e-12f;

    model->num_docs   = N;
    model->vocab_size = V;
    model->num_topics = K;
    model->W = (float*)rcre_arena_alloc(arena, N * K * sizeof(float), 64);
    model->H = (float*)rcre_arena_alloc(arena, K * V * sizeof(float), 64);
    if (!model->W || !model->H) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    RCRE_Arena* tmp_arena = scratch ? scratch : arena;
    bool distinct_scratch = (scratch != nullptr && scratch != arena);
    RCRE_ArenaTemp mark = rcre_arena_temp_begin(tmp_arena);

    float* A = (float*)rcre_arena_alloc(tmp_arena, K * V * sizeof(float), 64);
    float* B = (float*)rcre_arena_alloc(tmp_arena, N * K * sizeof(float), 64);
    float* G = (float*)rcre_arena_alloc(tmp_arena, K * K * sizeof(float), 64);
    float* P = (float*)rcre_arena_alloc(tmp_arena, K * K * sizeof(float), 64);
    if (!A || !B || !G || !P) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Precompute ||X||_F^2 once for O(K*V + K^2) exact Frobenius tracking */
    double x_norm2 = 0.0;
    for (size_t p = 0; p < X_csr->nnz; ++p) {
        double v = (double)X_csr->values[p];
        x_norm2 += v * v;
    }
    double scale_init = (x_norm2 > 0.0) ? sqrt(sqrt(x_norm2 / (double)(N * V)) / (double)K) : 0.1;
    if (scale_init < 1e-4) scale_init = 1e-4;

    /* Deterministic PCG64 positive initialization */
    RCRE_RNG rng;
    rcre_rng_pcg_init(&rng, cfg.seed);
    for (size_t i = 0; i < N * K; ++i) {
        model->W[i] = (float)((0.1 + 0.9 * rcre_rng_pcg_next_f64(&rng)) * scale_init);
    }
    for (size_t i = 0; i < K * V; ++i) {
        model->H[i] = (float)((0.1 + 0.9 * rcre_rng_pcg_next_f64(&rng)) * scale_init);
    }

    double prev_err = 1e30;
    size_t iters = 0;

    for (size_t iter = 0; iter < cfg.max_iter; ++iter) {
        iters = iter + 1U;

        /* 1. Update H given W: A = W^T * X [K x V], G = W^T * W [K x K] */
        memset(A, 0, K * V * sizeof(float));
        for (size_t d = 0; d < N; ++d) {
            uint32_t p0 = X_csr->row_pointers[d];
            uint32_t p1 = X_csr->row_pointers[d + 1U];
            const float* wrow = model->W + d * K;
            for (uint32_t p = p0; p < p1; ++p) {
                uint32_t col = X_csr->col_indices[p];
                float val = X_csr->values[p];
                for (size_t k = 0; k < K; ++k) {
                    A[k * V + col] += wrow[k] * val;
                }
            }
        }

        memset(G, 0, K * K * sizeof(float));
        for (size_t d = 0; d < N; ++d) {
            const float* wrow = model->W + d * K;
            for (size_t k = 0; k < K; ++k) {
                float wk = wrow[k];
                for (size_t j = k; j < K; ++j) {
                    G[k * K + j] += wk * wrow[j];
                }
            }
        }
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = k + 1U; j < K; ++j) {
                G[j * K + k] = G[k * K + j];
            }
        }

        for (size_t k = 0; k < K; ++k) {
            float denom = G[k * K + k];
            if (denom < eps) continue;
            float inv_denom = 1.0f / denom;
            float* hk = model->H + k * V;
            const float* ak = A + k * V;
            /* Residual vector in-place using HAL axpy: H_k += (A_k - sum_j G_kj H_j) / G_kk */
            for (size_t v = 0; v < V; ++v) {
                float pred = 0.0f;
                for (size_t j = 0; j < K; ++j) {
                    pred += G[k * K + j] * model->H[j * V + v];
                }
                float upd = hk[v] + (ak[v] - pred) * inv_denom;
                hk[v] = (upd > eps) ? upd : eps;
            }
        }

        /* 2. Update W given H */
        rcre_nlp__hals_update_W(X_csr, model->H, model->W, K, B, P);

        /* 3. Normalize each H topic row to L2 unit norm, absorb into W columns */
        for (size_t k = 0; k < K; ++k) {
            float* hk = model->H + k * V;
            float nrm = sqrtf(rcre_hal.dot_f32(hk, hk, V));
            if (nrm > eps) {
                rcre_hal.scale_f32(V, 1.0f / nrm, hk);
                for (size_t d = 0; d < N; ++d) {
                    model->W[d * K + k] *= nrm;
                }
            }
        }

        /* 4. Exact Frobenius error: ||X - WH||_F^2 = ||X||^2 - 2<B, W> + <W^T W, H H^T> */
        if ((iter + 1U) % 5U == 0U || iter + 1U == cfg.max_iter) {
            /* Recompute exact B = X H^T, P = H H^T, G = W^T W */
            for (size_t k = 0; k < K; ++k) {
                for (size_t j = k; j < K; ++j) {
                    float d = rcre_hal.dot_f32(model->H + k * V, model->H + j * V, V);
                    P[k * K + j] = d;
                    P[j * K + k] = d;
                }
            }
            memset(G, 0, K * K * sizeof(float));
            for (size_t d = 0; d < N; ++d) {
                const float* wrow = model->W + d * K;
                for (size_t k = 0; k < K; ++k) {
                    for (size_t j = 0; j < K; ++j) {
                        G[k * K + j] += wrow[k] * wrow[j];
                    }
                }
            }
            double cross = 0.0;
            for (size_t d = 0; d < N; ++d) {
                uint32_t p0 = X_csr->row_pointers[d];
                uint32_t p1 = X_csr->row_pointers[d + 1U];
                const float* wrow = model->W + d * K;
                for (uint32_t p = p0; p < p1; ++p) {
                    uint32_t col = X_csr->col_indices[p];
                    double pred = 0.0;
                    for (size_t k = 0; k < K; ++k) pred += (double)wrow[k] * (double)model->H[k * V + col];
                    cross += (double)X_csr->values[p] * pred;
                }
            }
            double wh_norm2 = 0.0;
            for (size_t i = 0; i < K * K; ++i) wh_norm2 += (double)G[i] * (double)P[i];
            double err2 = x_norm2 - 2.0 * cross + wh_norm2;
            if (err2 < 0.0) err2 = 0.0;
            double err = sqrt(err2);
            model->recon_error = (float)err;
            if (prev_err < 1e29 && fabs(prev_err - err) <= (double)cfg.tol * (prev_err + 1e-12)) {
                break;
            }
            prev_err = err;
        }
    }

    model->iters_run = iters;
    if (distinct_scratch) rcre_arena_temp_end(mark);
    return RCRE_OK;
}

int rcre_nmf_transform(
    const RCRE_NmfModel* model,
    const RCRE_CSRMatrix* X_new,
    size_t max_iter,
    float* out_W,
    RCRE_Arena* scratch
) {
    if (!model || !X_new || !out_W || !scratch || X_new->cols != model->vocab_size || !model->H) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (X_new->rows == 0) return RCRE_OK;
    if (max_iter == 0) max_iter = 25U;

    const size_t N = X_new->rows;
    const size_t K = model->num_topics;
    for (size_t i = 0; i < N * K; ++i) out_W[i] = 0.1f;

    RCRE_ArenaTemp mark = rcre_arena_temp_begin(scratch);
    float* B = (float*)rcre_arena_alloc(scratch, N * K * sizeof(float), 64);
    float* P = (float*)rcre_arena_alloc(scratch, K * K * sizeof(float), 64);
    if (!B || !P) {
        rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    for (size_t iter = 0; iter < max_iter; ++iter) {
        rcre_nlp__hals_update_W(X_new, model->H, out_W, K, B, P);
    }

    rcre_arena_temp_end(mark);
    return RCRE_OK;
}

size_t rcre_nlp_topic_top_terms(
    const float* topic_row,
    size_t vocab_size,
    size_t top_k,
    RCRE_TopicTerm* out_terms,
    RCRE_Arena* scratch
) {
    if (!topic_row || !out_terms || !scratch || vocab_size == 0 || top_k == 0) {
        return 0;
    }
    RCRE_ArenaTemp mark = rcre_arena_temp_begin(scratch);
    RCRE_SearchHit* tmp = (RCRE_SearchHit*)rcre_arena_alloc(
        scratch, vocab_size * sizeof(RCRE_SearchHit), alignof(RCRE_SearchHit));
    if (!tmp) {
        rcre_arena_temp_end(mark);
        return 0;
    }
    for (size_t v = 0; v < vocab_size; ++v) {
        tmp[v].doc_id = (uint32_t)v;
        tmp[v].score  = topic_row[v];
    }
    rcre_nlp__sort_hits_desc(tmp, vocab_size);
    size_t n_emit = (vocab_size < top_k) ? vocab_size : top_k;
    for (size_t i = 0; i < n_emit; ++i) {
        out_terms[i].term_id = tmp[i].doc_id;
        out_terms[i].weight  = tmp[i].score;
    }
    rcre_arena_temp_end(mark);
    return n_emit;
}

RCRE_LdaOpts rcre_lda_default_opts(size_t num_topics) {
    RCRE_LdaOpts opts = {
        .num_topics = (num_topics == 0) ? 8U : num_topics,
        .num_iters  = 60U,
        .alpha      = 0.1f,
        .beta       = 0.01f,
        .seed       = 42ULL,
        .lowercase  = true,
        .tok_opts   = rcre_nlp_default_tokenize_opts()
    };
    return opts;
}

int rcre_lda_fit(
    RCRE_LdaModel* model,
    const RCRE_Slice* docs,
    size_t num_docs,
    const RCRE_LdaOpts* opts,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
) {
    if (!model || !arena || (num_docs > 0 && !docs)) {
        return RCRE_ERR_INVALID_ARG;
    }
    memset(model, 0, sizeof(*model));
    RCRE_LdaOpts cfg = opts ? *opts : rcre_lda_default_opts(8);
    if (cfg.num_topics == 0) cfg.num_topics = 1U;
    if (cfg.num_iters == 0) cfg.num_iters = 1U;
    if (cfg.alpha <= 0.0f) cfg.alpha = 0.1f;
    if (cfg.beta <= 0.0f) cfg.beta = 0.01f;

    model->num_docs   = num_docs;
    model->num_topics = cfg.num_topics;
    model->alpha      = cfg.alpha;
    model->beta       = cfg.beta;
    model->lowercase  = cfg.lowercase;
    model->tok_opts   = cfg.tok_opts;

    RCRE_Arena* tmp_arena = scratch ? scratch : arena;
    bool distinct_scratch = (scratch != nullptr && scratch != arena);
    RCRE_ArenaTemp mark = rcre_arena_temp_begin(tmp_arena);

    int rc = rcre_strpool_init(&model->vocab, arena, num_docs > 64 ? num_docs * 2U : 128U);
    if (rc != RCRE_OK) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return rc;
    }

    /* Pass 1: intern vocab & count total tokens */
    uint8_t low_buf[256];
    size_t total_tokens = 0;
    for (size_t d = 0; d < num_docs; ++d) {
        RCRE_WordIter it = rcre_nlp_word_iter_init(docs[d], &cfg.tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, cfg.lowercase, low_buf, sizeof(low_buf), tmp_arena);
            if (rcre_strpool_intern(&model->vocab, norm) == RCRE_STR_ID_INVALID) {
                if (distinct_scratch) rcre_arena_temp_end(mark);
                return RCRE_ERR_OUT_OF_MEMORY;
            }
            total_tokens++;
        }
    }

    const size_t V = rcre_strpool_count(&model->vocab);
    const size_t K = cfg.num_topics;
    model->vocab_size = V;
    if (num_docs == 0 || V == 0 || total_tokens == 0) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_OK;
    }

    /* Allocate flat Gibbs workspace in scratch */
    uint32_t* doc_off   = (uint32_t*)rcre_arena_alloc(tmp_arena, (num_docs + 1U) * sizeof(uint32_t), 64);
    uint32_t* tok_word  = (uint32_t*)rcre_arena_alloc(tmp_arena, total_tokens * sizeof(uint32_t), 64);
    uint32_t* tok_topic = (uint32_t*)rcre_arena_alloc(tmp_arena, total_tokens * sizeof(uint32_t), 64);
    uint32_t* n_dk      = (uint32_t*)rcre_arena_alloc(tmp_arena, num_docs * K * sizeof(uint32_t), 64);
    uint32_t* n_kw      = (uint32_t*)rcre_arena_alloc(tmp_arena, K * V * sizeof(uint32_t), 64);
    uint32_t* n_k       = (uint32_t*)rcre_arena_alloc(tmp_arena, K * sizeof(uint32_t), 64);
    double*   probs     = (double*)rcre_arena_alloc(tmp_arena, K * sizeof(double), 64);
    if (!doc_off || !tok_word || !tok_topic || !n_dk || !n_kw || !n_k || !probs) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(n_dk, 0, num_docs * K * sizeof(uint32_t));
    memset(n_kw, 0, K * V * sizeof(uint32_t));
    memset(n_k,  0, K * sizeof(uint32_t));

    /* Pass 2: flatten corpus into (tok_word, tok_topic) with PCG64 init */
    RCRE_RNG rng;
    rcre_rng_pcg_init(&rng, cfg.seed);
    size_t tpos = 0;
    for (size_t d = 0; d < num_docs; ++d) {
        doc_off[d] = (uint32_t)tpos;
        RCRE_WordIter it = rcre_nlp_word_iter_init(docs[d], &cfg.tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, cfg.lowercase, low_buf, sizeof(low_buf), tmp_arena);
            RCRE_StrId wid = rcre_strpool_find(&model->vocab, norm);
            uint32_t z = (uint32_t)(rcre_rng_pcg_next_u64(&rng) % (uint64_t)K);
            tok_word[tpos] = wid;
            tok_topic[tpos] = z;
            n_dk[d * K + z]++;
            n_kw[z * V + wid]++;
            n_k[z]++;
            tpos++;
        }
    }
    doc_off[num_docs] = (uint32_t)tpos;

    /* Collapsed Gibbs sweeps */
    const double alpha = (double)cfg.alpha;
    const double beta  = (double)cfg.beta;
    const double vbeta = (double)V * beta;

    for (size_t iter = 0; iter < cfg.num_iters; ++iter) {
        for (size_t d = 0; d < num_docs; ++d) {
            uint32_t* drow = n_dk + d * K;
            for (uint32_t i = doc_off[d]; i < doc_off[d + 1U]; ++i) {
                uint32_t w = tok_word[i];
                uint32_t old_z = tok_topic[i];

                drow[old_z]--;
                n_kw[old_z * V + w]--;
                n_k[old_z]--;

                double sum_p = 0.0;
                for (size_t k = 0; k < K; ++k) {
                    double p = ((double)drow[k] + alpha) *
                               (((double)n_kw[k * V + w] + beta) / ((double)n_k[k] + vbeta));
                    sum_p += p;
                    probs[k] = sum_p;
                }

                double u = rcre_rng_pcg_next_f64(&rng) * sum_p;
                uint32_t new_z = (uint32_t)(K - 1U);
                for (size_t k = 0; k < K; ++k) {
                    if (u < probs[k]) {
                        new_z = (uint32_t)k;
                        break;
                    }
                }

                tok_topic[i] = new_z;
                drow[new_z]++;
                n_kw[new_z * V + w]++;
                n_k[new_z]++;
            }
        }
    }

    /* Materialize posterior theta [N x K] and phi [K x V] in persistent arena */
    model->theta = (float*)rcre_arena_alloc(arena, num_docs * K * sizeof(float), 64);
    model->phi   = (float*)rcre_arena_alloc(arena, K * V * sizeof(float), 64);
    if (!model->theta || !model->phi) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    const double kalpha = (double)K * alpha;
    for (size_t d = 0; d < num_docs; ++d) {
        double denom = (double)(doc_off[d + 1U] - doc_off[d]) + kalpha;
        for (size_t k = 0; k < K; ++k) {
            model->theta[d * K + k] = (float)(((double)n_dk[d * K + k] + alpha) / denom);
        }
    }
    for (size_t k = 0; k < K; ++k) {
        double denom = (double)n_k[k] + vbeta;
        for (size_t w = 0; w < V; ++w) {
            model->phi[k * V + w] = (float)(((double)n_kw[k * V + w] + beta) / denom);
        }
    }

    /* Compute held-in log-perplexity = - (1 / total_tokens) * sum log P(w | d) */
    double log_ll = 0.0;
    for (size_t d = 0; d < num_docs; ++d) {
        const float* td = model->theta + d * K;
        for (uint32_t i = doc_off[d]; i < doc_off[d + 1U]; ++i) {
            uint32_t w = tok_word[i];
            double pw = 0.0;
            for (size_t k = 0; k < K; ++k) {
                pw += (double)td[k] * (double)model->phi[k * V + w];
            }
            if (pw < 1e-30) pw = 1e-30;
            log_ll += log(pw);
        }
    }
    model->log_perplexity = (float)(-log_ll / (double)total_tokens);

    if (distinct_scratch) rcre_arena_temp_end(mark);
    return RCRE_OK;
}

int rcre_lda_infer(
    const RCRE_LdaModel* model,
    RCRE_Slice doc,
    size_t num_iters,
    uint64_t seed,
    float* out_theta,
    RCRE_Arena* scratch
) {
    if (!model || !out_theta || !scratch || model->num_topics == 0) {
        return RCRE_ERR_INVALID_ARG;
    }
    const size_t K = model->num_topics;
    const size_t V = model->vocab_size;
    if (num_iters == 0) num_iters = 30U;

    if (V == 0 || !model->phi || doc.len == 0) {
        float unif = 1.0f / (float)K;
        for (size_t k = 0; k < K; ++k) out_theta[k] = unif;
        return RCRE_OK;
    }

    RCRE_ArenaTemp mark = rcre_arena_temp_begin(scratch);
    uint8_t low_buf[256];

    /* Pass 1: count known vocabulary tokens in doc */
    size_t n_toks = 0;
    {
        RCRE_WordIter it = rcre_nlp_word_iter_init(doc, &model->tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, model->lowercase, low_buf, sizeof(low_buf), scratch);
            if (rcre_strpool_find(&model->vocab, norm) != RCRE_STR_ID_INVALID) {
                n_toks++;
            }
        }
    }
    if (n_toks == 0) {
        float unif = 1.0f / (float)K;
        for (size_t k = 0; k < K; ++k) out_theta[k] = unif;
        rcre_arena_temp_end(mark);
        return RCRE_OK;
    }

    uint32_t* wids  = (uint32_t*)rcre_arena_alloc(scratch, n_toks * sizeof(uint32_t), 64);
    uint32_t* zids  = (uint32_t*)rcre_arena_alloc(scratch, n_toks * sizeof(uint32_t), 64);
    uint32_t* n_dk  = (uint32_t*)rcre_arena_alloc(scratch, K * sizeof(uint32_t), 64);
    double*   probs = (double*)rcre_arena_alloc(scratch, K * sizeof(double), 64);
    if (!wids || !zids || !n_dk || !probs) {
        rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(n_dk, 0, K * sizeof(uint32_t));

    RCRE_RNG rng;
    rcre_rng_pcg_init(&rng, seed);
    size_t pos = 0;
    {
        RCRE_WordIter it = rcre_nlp_word_iter_init(doc, &model->tok_opts);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, model->lowercase, low_buf, sizeof(low_buf), scratch);
            RCRE_StrId wid = rcre_strpool_find(&model->vocab, norm);
            if (wid != RCRE_STR_ID_INVALID) {
                uint32_t z = (uint32_t)(rcre_rng_pcg_next_u64(&rng) % (uint64_t)K);
                wids[pos] = wid;
                zids[pos] = z;
                n_dk[z]++;
                pos++;
            }
        }
    }

    const double alpha = (double)model->alpha;
    for (size_t iter = 0; iter < num_iters; ++iter) {
        for (size_t i = 0; i < n_toks; ++i) {
            uint32_t w = wids[i];
            uint32_t old_z = zids[i];
            n_dk[old_z]--;
            double sum_p = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum_p += ((double)n_dk[k] + alpha) * (double)model->phi[k * V + w];
                probs[k] = sum_p;
            }
            double u = rcre_rng_pcg_next_f64(&rng) * sum_p;
            uint32_t new_z = (uint32_t)(K - 1U);
            for (size_t k = 0; k < K; ++k) {
                if (u < probs[k]) { new_z = (uint32_t)k; break; }
            }
            zids[i] = new_z;
            n_dk[new_z]++;
        }
    }

    double denom = (double)n_toks + (double)K * alpha;
    for (size_t k = 0; k < K; ++k) {
        out_theta[k] = (float)(((double)n_dk[k] + alpha) / denom);
    }

    rcre_arena_temp_end(mark);
    return RCRE_OK;
}

/* ========================================================================= */
/* 9. Native Sparse Text Classifiers (NLP-REQ-08)                            */
/* ========================================================================= */

static void rcre_nlp__softmax_inplace(float* logits, size_t C) {
    float mx = logits[0];
    for (size_t c = 1; c < C; ++c) {
        if (logits[c] > mx) mx = logits[c];
    }
    double sum = 0.0;
    for (size_t c = 0; c < C; ++c) {
        double e = exp((double)(logits[c] - mx));
        logits[c] = (float)e;
        sum += e;
    }
    if (sum > 0.0) {
        float inv = (float)(1.0 / sum);
        for (size_t c = 0; c < C; ++c) logits[c] *= inv;
    }
}

int rcre_nlp_nb_fit(
    RCRE_NaiveBayesModel* model,
    const RCRE_CSRMatrix* X_csr,
    const uint32_t* labels,
    size_t num_classes,
    float alpha,
    bool complement,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
) {
    if (!model || !X_csr || !labels || !arena || num_classes < 2U ||
        X_csr->rows == 0 || X_csr->cols == 0 || !X_csr->row_pointers) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (alpha <= 0.0f) alpha = 1.0f;
    memset(model, 0, sizeof(*model));
    model->num_classes = num_classes;
    model->vocab_size  = X_csr->cols;
    model->alpha       = alpha;
    model->complement  = complement;

    const size_t N = X_csr->rows;
    const size_t V = X_csr->cols;
    const size_t C = num_classes;

    model->class_log_prior = (float*)rcre_arena_alloc(arena, C * sizeof(float), 64);
    model->feature_weights = (float*)rcre_arena_alloc(arena, C * V * sizeof(float), 64);
    if (!model->class_log_prior || !model->feature_weights) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    RCRE_Arena* tmp_arena = scratch ? scratch : arena;
    bool distinct_scratch = (scratch != nullptr && scratch != arena);
    RCRE_ArenaTemp mark = rcre_arena_temp_begin(tmp_arena);

    double* class_counts = (double*)rcre_arena_alloc(tmp_arena, C * sizeof(double), 64);
    double* class_tf_sum = (double*)rcre_arena_alloc(tmp_arena, C * sizeof(double), 64);
    double* feat_tf      = (double*)rcre_arena_alloc(tmp_arena, C * V * sizeof(double), 64);
    if (!class_counts || !class_tf_sum || !feat_tf) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(class_counts, 0, C * sizeof(double));
    memset(class_tf_sum, 0, C * sizeof(double));
    memset(feat_tf,      0, C * V * sizeof(double));

    /* Single O(nnz) accumulation pass */
    for (size_t d = 0; d < N; ++d) {
        uint32_t c = labels[d];
        if ((size_t)c >= C) {
            if (distinct_scratch) rcre_arena_temp_end(mark);
            return RCRE_ERR_INVALID_ARG;
        }
        class_counts[c] += 1.0;
        uint32_t p0 = X_csr->row_pointers[d];
        uint32_t p1 = X_csr->row_pointers[d + 1U];
        double* frow = feat_tf + (size_t)c * V;
        double row_sum = 0.0;
        for (uint32_t p = p0; p < p1; ++p) {
            double v = (double)X_csr->values[p];
            if (v > 0.0) {
                frow[X_csr->col_indices[p]] += v;
                row_sum += v;
            }
        }
        class_tf_sum[c] += row_sum;
    }

    for (size_t c = 0; c < C; ++c) {
        double p = (class_counts[c] > 0.0) ? (class_counts[c] / (double)N) : (1e-12 / (double)N);
        model->class_log_prior[c] = (float)log(p);
    }

    double a = (double)alpha;
    if (!complement) {
        for (size_t c = 0; c < C; ++c) {
            double denom = class_tf_sum[c] + a * (double)V;
            const double* frow = feat_tf + c * V;
            float* wrow = model->feature_weights + c * V;
            for (size_t v = 0; v < V; ++v) {
                wrow[v] = (float)log((frow[v] + a) / denom);
            }
        }
    } else {
        /* Complement Naive Bayes (Rennie et al. 2003) */
        double total_tf = 0.0;
        for (size_t c = 0; c < C; ++c) total_tf += class_tf_sum[c];

        double* total_feat = (double*)rcre_arena_alloc(tmp_arena, V * sizeof(double), 64);
        if (!total_feat) {
            if (distinct_scratch) rcre_arena_temp_end(mark);
            return RCRE_ERR_OUT_OF_MEMORY;
        }
        memset(total_feat, 0, V * sizeof(double));
        for (size_t c = 0; c < C; ++c) {
            const double* frow = feat_tf + c * V;
            for (size_t v = 0; v < V; ++v) total_feat[v] += frow[v];
        }

        for (size_t c = 0; c < C; ++c) {
            double comp_denom = (total_tf - class_tf_sum[c]) + a * (double)V;
            const double* frow = feat_tf + c * V;
            float* wrow = model->feature_weights + c * V;
            double l1 = 0.0;
            for (size_t v = 0; v < V; ++v) {
                double comp_tf = total_feat[v] - frow[v];
                double w = -log((comp_tf + a) / comp_denom);
                wrow[v] = (float)w;
                l1 += fabs(w);
            }
            if (l1 > 0.0) {
                float inv = (float)(1.0 / l1);
                for (size_t v = 0; v < V; ++v) wrow[v] *= inv;
            }
        }
    }

    if (distinct_scratch) rcre_arena_temp_end(mark);
    return RCRE_OK;
}

int rcre_nlp_nb_predict(
    const RCRE_NaiveBayesModel* model,
    const RCRE_CSRMatrix* X_csr,
    uint32_t* out_labels,
    float* out_probs
) {
    if (!model || !X_csr || !model->feature_weights || X_csr->cols != model->vocab_size) {
        return RCRE_ERR_INVALID_ARG;
    }
    const size_t N = X_csr->rows;
    const size_t V = model->vocab_size;
    const size_t C = model->num_classes;

    for (size_t d = 0; d < N; ++d) {
        uint32_t p0 = X_csr->row_pointers[d];
        uint32_t p1 = X_csr->row_pointers[d + 1U];

        uint32_t best_c = 0;
        float best_score = -1e30f;

        for (size_t c = 0; c < C; ++c) {
            double s = model->complement ? 0.0 : (double)model->class_log_prior[c];
            const float* wrow = model->feature_weights + c * V;
            for (uint32_t p = p0; p < p1; ++p) {
                s += (double)X_csr->values[p] * (double)wrow[X_csr->col_indices[p]];
            }
            float sf = (float)s;
            if (out_probs) out_probs[d * C + c] = sf;
            if (sf > best_score) {
                best_score = sf;
                best_c = (uint32_t)c;
            }
        }
        if (out_labels) out_labels[d] = best_c;
        if (out_probs)  rcre_nlp__softmax_inplace(out_probs + d * C, C);
    }
    return RCRE_OK;
}

RCRE_SparseClfOpts rcre_sparse_clf_default_opts(void) {
    RCRE_SparseClfOpts opts = {
        .epochs = 30U,
        .lr     = 0.5f,
        .l2_reg = 1e-4f,
        .seed   = 42ULL
    };
    return opts;
}

int rcre_nlp_sparse_clf_fit(
    RCRE_SparseLinearClf* clf,
    const RCRE_CSRMatrix* X_csr,
    const uint32_t* labels,
    size_t num_classes,
    const RCRE_SparseClfOpts* opts,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
) {
    if (!clf || !X_csr || !labels || !arena || num_classes < 2U ||
        X_csr->rows == 0 || X_csr->cols == 0 || !X_csr->row_pointers) {
        return RCRE_ERR_INVALID_ARG;
    }
    RCRE_SparseClfOpts cfg = opts ? *opts : rcre_sparse_clf_default_opts();
    if (cfg.epochs == 0) cfg.epochs = 1U;
    if (cfg.lr <= 0.0f) cfg.lr = 0.5f;

    memset(clf, 0, sizeof(*clf));
    const size_t N = X_csr->rows;
    const size_t V = X_csr->cols;
    const size_t C = num_classes;

    clf->num_classes = C;
    clf->vocab_size  = V;
    clf->weights = (float*)rcre_arena_alloc(arena, C * V * sizeof(float), 64);
    clf->bias    = (float*)rcre_arena_alloc(arena, C * sizeof(float), 64);
    if (!clf->weights || !clf->bias) return RCRE_ERR_OUT_OF_MEMORY;
    memset(clf->weights, 0, C * V * sizeof(float));
    memset(clf->bias,    0, C * sizeof(float));

    for (size_t d = 0; d < N; ++d) {
        if ((size_t)labels[d] >= C) return RCRE_ERR_INVALID_ARG;
    }

    RCRE_Arena* tmp_arena = scratch ? scratch : arena;
    bool distinct_scratch = (scratch != nullptr && scratch != arena);
    RCRE_ArenaTemp mark = rcre_arena_temp_begin(tmp_arena);

    uint32_t* order  = (uint32_t*)rcre_arena_alloc(tmp_arena, N * sizeof(uint32_t), 64);
    float*    logits = (float*)rcre_arena_alloc(tmp_arena, C * sizeof(float), 64);
    if (!order || !logits) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < N; ++i) order[i] = (uint32_t)i;

    RCRE_RNG rng;
    rcre_rng_pcg_init(&rng, cfg.seed);

    for (size_t ep = 0; ep < cfg.epochs; ++ep) {
        /* Deterministic Fisher-Yates shuffle */
        for (size_t i = N - 1U; i > 0U; --i) {
            size_t j = (size_t)(rcre_rng_pcg_next_u64(&rng) % (uint64_t)(i + 1U));
            uint32_t t = order[i]; order[i] = order[j]; order[j] = t;
        }
        float lr = cfg.lr / (1.0f + 0.05f * (float)ep);
        float decay = 1.0f - lr * cfg.l2_reg;
        if (decay < 0.5f) decay = 0.5f;

        for (size_t idx = 0; idx < N; ++idx) {
            uint32_t d = order[idx];
            uint32_t y = labels[d];
            uint32_t p0 = X_csr->row_pointers[d];
            uint32_t p1 = X_csr->row_pointers[d + 1U];

            for (size_t c = 0; c < C; ++c) {
                float s = clf->bias[c];
                const float* wrow = clf->weights + c * V;
                for (uint32_t p = p0; p < p1; ++p) {
                    s += X_csr->values[p] * wrow[X_csr->col_indices[p]];
                }
                logits[c] = s;
            }
            rcre_nlp__softmax_inplace(logits, C);

            for (size_t c = 0; c < C; ++c) {
                float grad = logits[c] - ((c == (size_t)y) ? 1.0f : 0.0f);
                clf->bias[c] -= lr * grad;
                float* wrow = clf->weights + c * V;
                float step = -lr * grad;
                for (uint32_t p = p0; p < p1; ++p) {
                    uint32_t col = X_csr->col_indices[p];
                    wrow[col] = wrow[col] * decay + step * X_csr->values[p];
                }
            }
        }
    }

    if (distinct_scratch) rcre_arena_temp_end(mark);
    return RCRE_OK;
}

int rcre_nlp_sparse_clf_predict(
    const RCRE_SparseLinearClf* clf,
    const RCRE_CSRMatrix* X_csr,
    uint32_t* out_labels,
    float* out_probs
) {
    if (!clf || !X_csr || !clf->weights || !clf->bias || X_csr->cols != clf->vocab_size) {
        return RCRE_ERR_INVALID_ARG;
    }
    const size_t N = X_csr->rows;
    const size_t V = clf->vocab_size;
    const size_t C = clf->num_classes;

    for (size_t d = 0; d < N; ++d) {
        uint32_t p0 = X_csr->row_pointers[d];
        uint32_t p1 = X_csr->row_pointers[d + 1U];
        uint32_t best_c = 0;
        float best_s = -1e30f;

        for (size_t c = 0; c < C; ++c) {
            float s = clf->bias[c];
            const float* wrow = clf->weights + c * V;
            for (uint32_t p = p0; p < p1; ++p) {
                s += X_csr->values[p] * wrow[X_csr->col_indices[p]];
            }
            if (out_probs) out_probs[d * C + c] = s;
            if (s > best_s) { best_s = s; best_c = (uint32_t)c; }
        }
        if (out_labels) out_labels[d] = best_c;
        if (out_probs)  rcre_nlp__softmax_inplace(out_probs + d * C, C);
    }
    return RCRE_OK;
}

/* ========================================================================= */
/* 10. Sentence Splitting, Extractive Summarization & Keywords (NLP-REQ-09)  */
/* ========================================================================= */

static bool rcre_nlp__is_abbrev(RCRE_Slice word) {
    static const char* abbrevs[] = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "fig", "no", "e.g", "i.e"
    };
    for (size_t i = 0; i < sizeof(abbrevs) / sizeof(abbrevs[0]); ++i) {
        if (rcre_slice_eq_ascii_ci(word, rcre_slice_from_cstr(abbrevs[i]))) {
            return true;
        }
    }
    return false;
}

static bool rcre_nlp__is_stopword_ascii(RCRE_Slice w) {
    if (w.len <= 2U) return true;
    static const char* stops[] = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
        "new", "now", "old", "see", "two", "way", "who", "did", "get", "let",
        "put", "say", "she", "too", "use", "that", "with", "have", "this", "will",
        "your", "from", "they", "know", "want", "been", "good", "much", "some",
        "time", "very", "when", "come", "here", "just", "like", "long", "make",
        "many", "more", "only", "over", "such", "take", "than", "them", "well",
        "were", "what", "into", "about", "after", "again", "being", "could",
        "other", "their", "there", "these", "those", "under", "where", "which",
        "while", "would", "should", "through", "between", "because"
    };
    for (size_t i = 0; i < sizeof(stops) / sizeof(stops[0]); ++i) {
        if (rcre_slice_eq_ascii_ci(w, rcre_slice_from_cstr(stops[i]))) {
            return true;
        }
    }
    return false;
}

RCRE_SentIter rcre_nlp_sent_iter_init(RCRE_Slice text) {
    RCRE_SentIter it = { .src = text, .pos = 0 };
    return it;
}

bool rcre_nlp_sent_iter_next(RCRE_SentIter* it, RCRE_Slice* out_sent) {
    if (!it || !out_sent || !it->src.data) return false;

    while (it->pos < it->src.len && rcre_nlp__is_ascii_space(it->src.data[it->pos])) {
        it->pos++;
    }
    if (it->pos >= it->src.len) return false;

    size_t start = it->pos;
    size_t i = start;
    while (i < it->src.len) {
        uint8_t c = it->src.data[i];
        /* Paragraph break (\n\n or \r\n\r\n) terminates sentence */
        if (c == '\n' && i + 1U < it->src.len &&
            (it->src.data[i + 1U] == '\n' ||
             (it->src.data[i + 1U] == '\r' && i + 2U < it->src.len && it->src.data[i + 2U] == '\n'))) {
            RCRE_Slice cand = rcre_slice_trim_ascii(rcre_slice_sub(it->src, start, i));
            it->pos = i + 1U;
            if (cand.len > 0) { *out_sent = cand; return true; }
            return rcre_nlp_sent_iter_next(it, out_sent);
        }

        if (c == '.' || c == '!' || c == '?') {
            /* Guard decimal numbers (e.g. 3.14) */
            if (c == '.' && i > start && i + 1U < it->src.len &&
                rcre_nlp__is_ascii_digit(it->src.data[i - 1U]) &&
                rcre_nlp__is_ascii_digit(it->src.data[i + 1U])) {
                i++;
                continue;
            }
            /* Guard common abbreviations (e.g., "Dr. Smith") */
            if (c == '.' && i > start) {
                size_t ws = i;
                while (ws > start && (rcre_nlp__is_ascii_alpha(it->src.data[ws - 1U]) || it->src.data[ws - 1U] == '.')) {
                    ws--;
                }
                if (rcre_nlp__is_abbrev(rcre_slice_sub(it->src, ws, i))) {
                    i++;
                    continue;
                }
            }
            /* Consume consecutive terminal punctuation (e.g. "?!", "...") & closing quotes */
            size_t end = i + 1U;
            while (end < it->src.len &&
                   (it->src.data[end] == '.' || it->src.data[end] == '!' ||
                    it->src.data[end] == '?' || it->src.data[end] == '"' ||
                    it->src.data[end] == '\'' || it->src.data[end] == ')')) {
                end++;
            }
            /* Must be at EOF or followed by whitespace to end sentence */
            if (end == it->src.len || rcre_nlp__is_ascii_space(it->src.data[end])) {
                RCRE_Slice cand = rcre_slice_trim_ascii(rcre_slice_sub(it->src, start, end));
                it->pos = end;
                if (cand.len > 0) { *out_sent = cand; return true; }
                return rcre_nlp_sent_iter_next(it, out_sent);
            }
        }
        i++;
    }

    RCRE_Slice tail = rcre_slice_trim_ascii(rcre_slice_sub(it->src, start, it->src.len));
    it->pos = it->src.len;
    if (tail.len == 0) return false;
    *out_sent = tail;
    return true;
}

RCRE_SummarizeOpts rcre_summarize_default_opts(void) {
    RCRE_SummarizeOpts opts = {
        .damping       = 0.85f,
        .sim_threshold = 0.05f,
        .mmr_lambda    = 0.7f,
        .max_iter      = 30U
    };
    return opts;
}

static void rcre_nlp__pagerank_dense(
    const float* adj,
    size_t M,
    float damping,
    size_t max_iter,
    float* ranks,
    float* next_ranks
) {
    if (M == 0) return;
    float unif = 1.0f / (float)M;
    for (size_t i = 0; i < M; ++i) ranks[i] = unif;

    for (size_t iter = 0; iter < max_iter; ++iter) {
        float base = (1.0f - damping) * unif;
        for (size_t i = 0; i < M; ++i) next_ranks[i] = base;

        double dangling_mass = 0.0;
        for (size_t i = 0; i < M; ++i) {
            const float* row = adj + i * M;
            double row_sum = 0.0;
            for (size_t j = 0; j < M; ++j) row_sum += (double)row[j];
            if (row_sum <= 1e-12) {
                dangling_mass += (double)ranks[i];
            } else {
                float scale = damping * ranks[i] / (float)row_sum;
                rcre_hal.axpy_f32(M, scale, row, next_ranks);
            }
        }
        if (dangling_mass > 0.0) {
            float d_add = damping * (float)dangling_mass * unif;
            for (size_t i = 0; i < M; ++i) next_ranks[i] += d_add;
        }

        double diff = 0.0;
        for (size_t i = 0; i < M; ++i) {
            diff += fabs((double)next_ranks[i] - (double)ranks[i]);
            ranks[i] = next_ranks[i];
        }
        if (diff < 1e-6) break;
    }
}

int rcre_nlp_summarize_extractive(
    RCRE_Slice text,
    size_t top_k,
    const RCRE_SummarizeOpts* opts,
    RCRE_SummarySent* out_sents,
    size_t* out_count,
    RCRE_Arena* scratch
) {
    if (!out_count || !scratch || (top_k > 0 && !out_sents)) {
        return RCRE_ERR_INVALID_ARG;
    }
    *out_count = 0;
    if (!text.data || text.len == 0 || top_k == 0) return RCRE_OK;

    RCRE_SummarizeOpts cfg = opts ? *opts : rcre_summarize_default_opts();
    if (cfg.damping <= 0.0f || cfg.damping >= 1.0f) cfg.damping = 0.85f;
    if (cfg.mmr_lambda < 0.0f) cfg.mmr_lambda = 0.0f;
    if (cfg.mmr_lambda > 1.0f) cfg.mmr_lambda = 1.0f;
    if (cfg.max_iter == 0) cfg.max_iter = 30U;

    RCRE_ArenaTemp mark = rcre_arena_temp_begin(scratch);

    /* Pass 1: count sentences */
    size_t M = 0;
    {
        RCRE_SentIter it = rcre_nlp_sent_iter_init(text);
        RCRE_Slice s;
        while (rcre_nlp_sent_iter_next(&it, &s)) M++;
    }
    if (M == 0) {
        rcre_arena_temp_end(mark);
        return RCRE_OK;
    }

    RCRE_Slice* sents = (RCRE_Slice*)rcre_arena_alloc(scratch, M * sizeof(RCRE_Slice), alignof(RCRE_Slice));
    if (!sents) { rcre_arena_temp_end(mark); return RCRE_ERR_OUT_OF_MEMORY; }
    {
        RCRE_SentIter it = rcre_nlp_sent_iter_init(text);
        size_t idx = 0;
        while (idx < M && rcre_nlp_sent_iter_next(&it, &sents[idx])) idx++;
    }

    if (M <= top_k) {
        for (size_t i = 0; i < M; ++i) {
            out_sents[i].sent_index = i;
            out_sents[i].score = 1.0f / (float)M;
            out_sents[i].sentence = sents[i];
        }
        *out_count = M;
        rcre_arena_temp_end(mark);
        return RCRE_OK;
    }

    /* Vectorize sentences via TF-IDF -> RCRE_CSRMatrix */
    RCRE_TfidfVectorizer vec;
    RCRE_CSRMatrix csr;
    int rc = rcre_tfidf_fit(&vec, sents, M, nullptr, scratch, scratch);
    if (rc != RCRE_OK) { rcre_arena_temp_end(mark); return rc; }
    rc = rcre_tfidf_transform(&vec, sents, M, scratch, scratch, &csr);
    if (rc != RCRE_OK) { rcre_arena_temp_end(mark); return rc; }

    /* Build symmetric cosine similarity graph [M x M] */
    float* sim   = (float*)rcre_arena_alloc(scratch, M * M * sizeof(float), 64);
    float* ranks = (float*)rcre_arena_alloc(scratch, M * sizeof(float), 64);
    float* next  = (float*)rcre_arena_alloc(scratch, M * sizeof(float), 64);
    uint8_t* chosen = (uint8_t*)rcre_arena_alloc(scratch, M, 16);
    size_t*  picked = (size_t*)rcre_arena_alloc(scratch, top_k * sizeof(size_t), 64);
    if (!sim || !ranks || !next || !chosen || !picked) {
        rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(sim, 0, M * M * sizeof(float));
    memset(chosen, 0, M);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = i + 1U; j < M; ++j) {
            float c = rcre_nlp_csr_row_cosine(&csr, i, &csr, j);
            if (c >= cfg.sim_threshold) {
                sim[i * M + j] = c;
                sim[j * M + i] = c;
            }
        }
    }

    rcre_nlp__pagerank_dense(sim, M, cfg.damping, cfg.max_iter, ranks, next);

    /* Normalize ranks to [0, 1] for balanced MMR selection */
    float max_r = 0.0f;
    for (size_t i = 0; i < M; ++i) if (ranks[i] > max_r) max_r = ranks[i];
    float inv_max = (max_r > 0.0f) ? (1.0f / max_r) : 1.0f;

    /* Greedy MMR selection */
    for (size_t step = 0; step < top_k; ++step) {
        size_t best_i = 0;
        float best_mmr = -1e30f;
        for (size_t i = 0; i < M; ++i) {
            if (chosen[i]) continue;
            float max_sim = 0.0f;
            for (size_t p = 0; p < step; ++p) {
                float s = sim[i * M + picked[p]];
                if (s > max_sim) max_sim = s;
            }
            float score = cfg.mmr_lambda * (ranks[i] * inv_max) - (1.0f - cfg.mmr_lambda) * max_sim;
            if (score > best_mmr) {
                best_mmr = score;
                best_i = i;
            }
        }
        chosen[best_i] = 1U;
        picked[step] = best_i;
    }

    /* Emit selected sentences in original document order */
    size_t emitted = 0;
    for (size_t i = 0; i < M && emitted < top_k; ++i) {
        if (chosen[i]) {
            out_sents[emitted].sent_index = i;
            out_sents[emitted].score      = ranks[i];
            out_sents[emitted].sentence   = sents[i];
            emitted++;
        }
    }
    *out_count = emitted;
    rcre_arena_temp_end(mark);
    return RCRE_OK;
}

int rcre_nlp_extract_keywords(
    RCRE_Slice text,
    size_t top_k,
    size_t window_size,
    RCRE_Keyword* out_keywords,
    size_t* out_count,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
) {
    if (!arena || !out_count || (top_k > 0 && !out_keywords)) {
        return RCRE_ERR_INVALID_ARG;
    }
    *out_count = 0;
    if (!text.data || text.len == 0 || top_k == 0) return RCRE_OK;
    if (window_size < 2U) window_size = 4U;

    RCRE_Arena* tmp_arena = scratch ? scratch : arena;
    bool distinct_scratch = (scratch != nullptr && scratch != arena);
    RCRE_ArenaTemp mark = rcre_arena_temp_begin(tmp_arena);

    RCRE_StrPool pool;
    int rc = rcre_strpool_init(&pool, tmp_arena, 128);
    if (rc != RCRE_OK) { if (distinct_scratch) rcre_arena_temp_end(mark); return rc; }

    /* Pass 1: intern non-stopword candidate terms & count sequence length */
    uint8_t low_buf[256];
    size_t seq_len = 0;
    {
        RCRE_WordIter it = rcre_nlp_word_iter_init(text, nullptr);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, true, low_buf, sizeof(low_buf), tmp_arena);
            if (rcre_nlp__is_stopword_ascii(norm)) continue;
            if (rcre_strpool_intern(&pool, norm) == RCRE_STR_ID_INVALID) {
                if (distinct_scratch) rcre_arena_temp_end(mark);
                return RCRE_ERR_OUT_OF_MEMORY;
            }
            seq_len++;
        }
    }

    size_t V = rcre_strpool_count(&pool);
    if (V == 0) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_OK;
    }

    uint32_t* seq   = (uint32_t*)rcre_arena_alloc(tmp_arena, seq_len * sizeof(uint32_t), 64);
    float*    adj   = (float*)rcre_arena_alloc(tmp_arena, V * V * sizeof(float), 64);
    float*    ranks = (float*)rcre_arena_alloc(tmp_arena, V * sizeof(float), 64);
    float*    next  = (float*)rcre_arena_alloc(tmp_arena, V * sizeof(float), 64);
    RCRE_SearchHit* scored = (RCRE_SearchHit*)rcre_arena_alloc(tmp_arena, V * sizeof(RCRE_SearchHit), alignof(RCRE_SearchHit));
    if (!seq || !adj || !ranks || !next || !scored) {
        if (distinct_scratch) rcre_arena_temp_end(mark);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(adj, 0, V * V * sizeof(float));

    /* Pass 2: collect sequence & populate sliding-window co-occurrence edges */
    size_t pos = 0;
    {
        RCRE_WordIter it = rcre_nlp_word_iter_init(text, nullptr);
        RCRE_Slice tok;
        while (rcre_nlp_word_iter_next(&it, &tok)) {
            RCRE_Slice norm = rcre_nlp__normalize_tok(tok, true, low_buf, sizeof(low_buf), tmp_arena);
            if (rcre_nlp__is_stopword_ascii(norm)) continue;
            seq[pos++] = rcre_strpool_find(&pool, norm);
        }
    }

    for (size_t i = 0; i < seq_len; ++i) {
        uint32_t u = seq[i];
        size_t j_end = (i + window_size < seq_len) ? (i + window_size) : seq_len;
        for (size_t j = i + 1U; j < j_end; ++j) {
            uint32_t v = seq[j];
            if (u != v) {
                adj[(size_t)u * V + v] += 1.0f;
                adj[(size_t)v * V + u] += 1.0f;
            }
        }
    }

    rcre_nlp__pagerank_dense(adj, V, 0.85f, 30U, ranks, next);

    for (size_t v = 0; v < V; ++v) {
        scored[v].doc_id = (uint32_t)v;
        scored[v].score  = ranks[v];
    }
    rcre_nlp__sort_hits_desc(scored, V);

    size_t n_emit = (V < top_k) ? V : top_k;
    for (size_t i = 0; i < n_emit; ++i) {
        out_keywords[i].term  = rcre_str_dup(arena, rcre_strpool_get(&pool, scored[i].doc_id));
        out_keywords[i].score = scored[i].score;
    }
    *out_count = n_emit;

    if (distinct_scratch) rcre_arena_temp_end(mark);
    return RCRE_OK;
}

