/**
 * @file rcre_nlp.h
 * @brief RCRE HPC-Native, Zero-Malloc C23 Natural Language Processing Engine
 *
 * Built directly on top of rcre_str.h (RCRE_Slice, RCRE_StrBuilder, RCRE_StrPool,
 * HAL SIMD) and rcre_sparse.h (RCRE_CSRMatrix).
 *
 * Satisfies NLP-REQ-01 through NLP-REQ-06:
 *   - NLP-REQ-01: Zero-copy pull tokenizers (RCRE_WordIter, RCRE_CharNGramIter,
 *                 RCRE_WordNGramIter) over raw or mmap'd RCRE_Slice buffers.
 *   - NLP-REQ-02: Flat CSR-layout arena BM25 inverted index (RCRE_BM25Index)
 *                 and deterministic Reciprocal Rank Fusion (rcre_nlp_rrf_fuse)
 *                 for hybrid lexical + HNSW vector search.
 *   - NLP-REQ-03: Scikit-Learn-parity TF-IDF vectorizer (RCRE_TfidfVectorizer)
 *                 emitting 64-byte aligned RCRE_CSRMatrix, plus lock-free signed
 *                 Feature Hashing (Weinberger trick).
 *   - NLP-REQ-04: Flat-table DFA Aho-Corasick multi-pattern matcher & single-pass
 *                 redactor (1 table load per byte, compile-time case folding).
 *   - NLP-REQ-05: 64-bit word-parallel Myers Levenshtein edit distance,
 *                 Jaro-Winkler similarity, and Jaccard n-gram overlap.
 *   - NLP-REQ-06: LLM dataset deduplication (SimHash64 + hardware popcount,
 *                 K-permutation MinHash) and zero-copy UTF-8 RAG document chunker.
 *   - NLP-REQ-07: Topic modeling — Sparse HALS NMF (RCRE_NmfModel) and
 *                 deterministic PCG64 Collapsed Gibbs LDA (RCRE_LdaModel).
 *   - NLP-REQ-08: Native sparse text classification directly on RCRE_CSRMatrix —
 *                 Multinomial / Complement Naive Bayes and Multi-Class Softmax SGD.
 *   - NLP-REQ-09: Zero-copy sentence splitter (RCRE_SentIter), LexRank/TextRank +
 *                 MMR extractive summarization, and graph keyword extraction.
 *   - NLP-REQ-10: Exact O(nnz_a + nnz_b) sparse-sparse CSR row cosine similarity.
 *
 * Architectural guarantees:
 *   - Zero malloc / calloc / realloc / free anywhere in this module.
 *   - Pure, re-entrant, lock-free algorithms safe for RCRE_ThreadPool slabs.
 *   - All dynamic structures reside contiguously inside caller RCRE_Arena pools.
 */

#ifndef RCRE_NLP_H
#define RCRE_NLP_H

#include "rcre.h"
#include "rcre_str.h"
#include "rcre_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Zero-Copy Tokenizers & N-Gram Iterators (NLP-REQ-01)                   */
/* ========================================================================= */

#define RCRE_NLP_MAX_NGRAM 8U

typedef struct RCRE_TokenizeOpts {
    bool   split_punct; /* Emit ASCII punctuation marks as standalone tokens */
    bool   keep_digits; /* Include ASCII digits [0-9] in word tokens         */
    size_t min_len;     /* Minimum token byte length (0 => default 1)        */
    size_t max_len;     /* Maximum token byte length (0 => unlimited)        */
} RCRE_TokenizeOpts;

typedef struct RCRE_WordIter {
    RCRE_Slice        src;
    size_t            pos;
    RCRE_TokenizeOpts opts;
} RCRE_WordIter;

typedef struct RCRE_CharNGramIter {
    RCRE_Slice src;
    size_t     n;
    size_t     pos;
} RCRE_CharNGramIter;

typedef struct RCRE_WordNGramIter {
    RCRE_WordIter words;
    size_t        n;
    size_t        filled;
    size_t        head;
    RCRE_Slice    ring[RCRE_NLP_MAX_NGRAM];
} RCRE_WordNGramIter;

RCRE_NODISCARD RCRE_TokenizeOpts rcre_nlp_default_tokenize_opts(void);
RCRE_NODISCARD RCRE_WordIter rcre_nlp_word_iter_init(RCRE_Slice text, const RCRE_TokenizeOpts* opts);
RCRE_NODISCARD bool rcre_nlp_word_iter_next(RCRE_WordIter* it, RCRE_Slice* out_tok);

RCRE_NODISCARD RCRE_CharNGramIter rcre_nlp_char_ngram_init(RCRE_Slice text, size_t n);
RCRE_NODISCARD bool rcre_nlp_char_ngram_next(RCRE_CharNGramIter* it, RCRE_Slice* out_ngram);

RCRE_NODISCARD RCRE_WordNGramIter rcre_nlp_word_ngram_init(RCRE_Slice text, size_t n, const RCRE_TokenizeOpts* opts);
/** Emits contiguous zero-copy slice spanning the current n-word window. */
RCRE_NODISCARD bool rcre_nlp_word_ngram_next(RCRE_WordNGramIter* it, RCRE_Slice* out_span);

/* ========================================================================= */
/* 2. Flat-Arena BM25 Inverted Index & Hybrid RRF Fusion (NLP-REQ-02)        */
/* ========================================================================= */

typedef struct RCRE_BM25Opts {
    float             k1;        /* Term-frequency saturation (default 1.5)  */
    float             b;         /* Document-length normalization (default 0.75) */
    bool              lowercase; /* ASCII lowercase normalization (default true) */
    RCRE_TokenizeOpts tok_opts;
} RCRE_BM25Opts;

typedef struct RCRE_SearchHit {
    uint32_t doc_id;
    float    score;
} RCRE_SearchHit;

typedef struct RCRE_BM25Index {
    RCRE_StrPool vocab;          /* Term -> dense RCRE_StrId [0..num_terms)  */
    uint32_t     num_docs;
    uint32_t     num_terms;
    float        avg_doc_len;
    float        k1;
    float        b;
    bool         lowercase;
    RCRE_TokenizeOpts tok_opts;

    uint32_t*    doc_lens;       /* [num_docs] token count per document      */
    float*       term_idf;       /* [num_terms] precomputed BM25 IDF         */
    uint32_t*    term_offsets;   /* [num_terms + 1] CSR posting offsets      */
    uint32_t*    postings_doc;   /* [total_postings] sorted doc IDs          */
    uint32_t*    postings_tf;    /* [total_postings] term frequency in doc   */
    size_t       total_postings;
} RCRE_BM25Index;

RCRE_NODISCARD RCRE_BM25Opts rcre_bm25_default_opts(void);

/**
 * @brief Builds a flat CSR-layout BM25 inverted index in two O(tokens) linear passes.
 * @param index   Output initialized index.
 * @param docs    Array of document slices [num_docs].
 * @param num_docs Number of documents.
 * @param opts    Optional BM25 options (nullptr => defaults).
 * @param arena   Persistent arena owning the index buffers.
 * @param scratch Ephemeral scratch arena (rewound before return; may equal arena).
 */
RCRE_NODISCARD int rcre_bm25_build(
    RCRE_BM25Index* index,
    const RCRE_Slice* docs,
    size_t num_docs,
    const RCRE_BM25Opts* opts,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
);

/**
 * @brief Executes Okapi BM25 top-k search using a transient scratch arena.
 * @param out_hits Caller array of capacity top_k.
 * @param out_count Actual number of hits written (<= top_k).
 */
RCRE_NODISCARD int rcre_bm25_search(
    const RCRE_BM25Index* index,
    RCRE_Slice query,
    size_t top_k,
    RCRE_SearchHit* out_hits,
    size_t* out_count,
    RCRE_Arena* scratch
);

/**
 * @brief Reusable per-worker BM25 query scratchpad for high-QPS online serving.
 *
 * Allocated once from a caller RCRE_Arena via rcre_bm25_scratch_init() (zero
 * malloc). Avoids O(num_docs) memset per query via O(touched) cleanup and uses
 * partial top-k selection when top_k << touched.
 */
typedef struct RCRE_BM25Scratch {
    float*          scores;
    uint32_t*       touched;
    RCRE_SearchHit* hits;
    size_t          num_docs;
} RCRE_BM25Scratch;

RCRE_NODISCARD int rcre_bm25_scratch_init(
    RCRE_BM25Scratch* scratch,
    const RCRE_BM25Index* index,
    RCRE_Arena* arena
);

/**
 * @brief High-throughput Okapi BM25 top-k search over a reusable RCRE_BM25Scratch.
 *        Produces identical rankings and scores to rcre_bm25_search() with
 *        O(touched) reset overhead and zero allocations per query.
 */
RCRE_NODISCARD int rcre_bm25_search_scratch(
    const RCRE_BM25Index* index,
    RCRE_Slice query,
    size_t top_k,
    RCRE_SearchHit* out_hits,
    size_t* out_count,
    RCRE_BM25Scratch* scratch
);

/**
 * @brief Reciprocal Rank Fusion (RRF): fuses multiple ranked hit lists
 *        (e.g., BM25 lexical + HNSW dense vector) via sum 1 / (k_rrf + rank).
 */
RCRE_NODISCARD int rcre_nlp_rrf_fuse(
    const RCRE_SearchHit* const* lists,
    const size_t* list_counts,
    size_t num_lists,
    float k_rrf,
    size_t top_k,
    RCRE_SearchHit* out_hits,
    size_t* out_count,
    RCRE_Arena* scratch
);

/* ========================================================================= */
/* 3. Scikit-Learn Parity TF-IDF Vectorizer & Feature Hashing (NLP-REQ-03)   */
/* ========================================================================= */

typedef enum RCRE_NlpNorm {
    RCRE_NLP_NORM_NONE = 0,
    RCRE_NLP_NORM_L2   = 1,
    RCRE_NLP_NORM_L1   = 2
} RCRE_NlpNorm;

typedef struct RCRE_TfidfOpts {
    bool              lowercase;    /* Default true                          */
    bool              smooth_idf;   /* log((1+N)/(1+df)) + 1 (default true)  */
    bool              sublinear_tf; /* 1 + log(tf) (default false)           */
    RCRE_NlpNorm      norm;         /* Default RCRE_NLP_NORM_L2              */
    uint32_t          min_df;       /* Minimum document freq (default 1)     */
    float             max_df_ratio; /* Max doc freq ratio in (0,1] (def 1.0) */
    size_t            max_features; /* 0 => unlimited                        */
    RCRE_TokenizeOpts tok_opts;
} RCRE_TfidfOpts;

typedef struct RCRE_TfidfVectorizer {
    RCRE_StrPool   vocab;        /* Alphabetically ordered feature IDs [0..V) */
    float*         idf;          /* [vocab_size]                              */
    size_t         vocab_size;
    size_t         num_fit_docs;
    RCRE_TfidfOpts opts;
} RCRE_TfidfVectorizer;

RCRE_NODISCARD RCRE_TfidfOpts rcre_tfidf_default_opts(void);

RCRE_NODISCARD int rcre_tfidf_fit(
    RCRE_TfidfVectorizer* vec,
    const RCRE_Slice* docs,
    size_t num_docs,
    const RCRE_TfidfOpts* opts,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
);

/**
 * @brief Transforms documents into a 64-byte aligned RCRE_CSRMatrix
 *        [num_docs, vocab_size] with sorted column indices per row.
 */
RCRE_NODISCARD int rcre_tfidf_transform(
    const RCRE_TfidfVectorizer* vec,
    const RCRE_Slice* docs,
    size_t num_docs,
    RCRE_Arena* arena,
    RCRE_Arena* scratch,
    RCRE_CSRMatrix* out_csr
);

/**
 * @brief Lock-free signed Feature Hashing (Weinberger trick) directly into
 *        a contiguous 2D F32 RCRE_Tensor [num_docs, n_features].
 */
RCRE_NODISCARD int rcre_nlp_hash_vectorize(
    const RCRE_Slice* docs,
    size_t num_docs,
    size_t n_features,
    bool lowercase,
    RCRE_NlpNorm norm,
    const RCRE_TokenizeOpts* tok_opts,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
);

/* ========================================================================= */
/* 4. Aho-Corasick Flat DFA Multi-Pattern Automaton (NLP-REQ-04)             */
/* ========================================================================= */

#define RCRE_AC_NONE ((uint32_t)0xFFFFFFFFu)

typedef struct RCRE_ACMatch {
    size_t   start;      /* Inclusive byte start offset in text */
    size_t   end;        /* Exclusive byte end offset in text   */
    uint32_t pattern_id; /* Index into original patterns array  */
} RCRE_ACMatch;

typedef bool (*rcre_ac_match_cb)(RCRE_ACMatch match, void* user_ctx);

typedef struct RCRE_AhoCorasick {
    uint32_t* next;          /* Flat DFA table: [num_states * 256]      */
    uint32_t* out_head;      /* [num_states] index into out_nodes       */
    uint32_t* out_pat_id;    /* [num_out_nodes] pattern index           */
    uint32_t* out_next;      /* [num_out_nodes] next output node        */
    size_t*   pat_lens;      /* [num_patterns] byte length per pattern  */
    uint32_t  num_states;
    uint32_t  num_out_nodes;
    uint32_t  num_patterns;
    bool      case_insensitive;
} RCRE_AhoCorasick;

RCRE_NODISCARD int rcre_ac_build(
    RCRE_AhoCorasick* ac,
    const RCRE_Slice* patterns,
    size_t num_patterns,
    bool case_insensitive,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
);

/** Streams every match to callback (return false from cb to stop early). Returns match count. */
size_t rcre_ac_scan(
    const RCRE_AhoCorasick* ac,
    RCRE_Slice text,
    rcre_ac_match_cb cb,
    void* user_ctx
);

/** Collects up to max_matches into out_matches; returns total matches found. */
size_t rcre_ac_find_all(
    const RCRE_AhoCorasick* ac,
    RCRE_Slice text,
    RCRE_ACMatch* out_matches,
    size_t max_matches
);

/** Single-pass leftmost-first non-overlapping redaction into arena-owned slice. */
RCRE_NODISCARD RCRE_Slice rcre_ac_redact(
    const RCRE_AhoCorasick* ac,
    RCRE_Slice text,
    RCRE_Slice replacement,
    RCRE_Arena* arena
);

/* ========================================================================= */
/* 5. Bit-Parallel Myers Edit Distance & Fuzzy Metrics (NLP-REQ-05)          */
/* ========================================================================= */

/**
 * @brief Exact Levenshtein edit distance using 64-bit word-parallel Myers
 *        bit-vectors when min(a.len, b.len) <= 64, and O(min(m,n)) 2-row DP
 *        for longer strings (scratch arena optional; stack handles up to 1024).
 */
RCRE_NODISCARD size_t rcre_nlp_levenshtein(RCRE_Slice a, RCRE_Slice b, RCRE_Arena* scratch);

/** Reference textbook O(m*n) Levenshtein oracle (for differential verification). */
RCRE_NODISCARD size_t rcre_nlp_levenshtein_scalar(RCRE_Slice a, RCRE_Slice b, RCRE_Arena* scratch);

/** Normalized Levenshtein similarity in [0.0, 1.0]. */
RCRE_NODISCARD float rcre_nlp_levenshtein_sim(RCRE_Slice a, RCRE_Slice b, RCRE_Arena* scratch);

/** Jaro-Winkler string similarity in [0.0, 1.0] (prefix_scale typically 0.1f). */
RCRE_NODISCARD float rcre_nlp_jaro_winkler(RCRE_Slice a, RCRE_Slice b, float prefix_scale, RCRE_Arena* scratch);

/** Exact byte/UTF-8 character n-gram Jaccard similarity in [0.0, 1.0]. */
RCRE_NODISCARD float rcre_nlp_jaccard_char_ngrams(RCRE_Slice a, RCRE_Slice b, size_t n, RCRE_Arena* scratch);

/* ========================================================================= */
/* 6. LLM Dataset Deduplication & UTF-8 RAG Chunker (NLP-REQ-06)             */
/* ========================================================================= */

/** 64-bit Charikar SimHash over tokenized words (lowercase ASCII normalized). */
RCRE_NODISCARD uint64_t rcre_nlp_simhash64(RCRE_Slice text, const RCRE_TokenizeOpts* opts);

/** Hardware-accelerated Hamming distance between two 64-bit SimHash fingerprints. */
RCRE_NODISCARD uint32_t rcre_nlp_hamming64(uint64_t a, uint64_t b);

/**
 * @brief Computes a K-permutation MinHash signature over word n-grams
 *        (ngram_n == 1 => unigrams; ngram_n >= 2 => word shingles).
 */
RCRE_NODISCARD int rcre_nlp_minhash_compute(
    RCRE_Slice text,
    size_t ngram_n,
    uint64_t* out_sig,
    size_t num_hashes,
    const RCRE_TokenizeOpts* opts
);

/** Estimates Jaccard similarity from two MinHash signatures in [0.0, 1.0]. */
RCRE_NODISCARD float rcre_nlp_minhash_jaccard(const uint64_t* sig_a, const uint64_t* sig_b, size_t num_hashes);

/**
 * @brief Zero-copy UTF-8 & word-boundary-safe RAG document chunker.
 *        Allocates only the RCRE_Slice* descriptor array in arena (0 text copies).
 */
RCRE_NODISCARD int rcre_nlp_chunk_utf8(
    RCRE_Slice text,
    size_t max_bytes,
    size_t overlap_bytes,
    RCRE_Arena* arena,
    RCRE_Slice** out_chunks,
    size_t* out_count
);

/* ========================================================================= */
/* 7. Topic Modeling: Sparse HALS NMF & Collapsed Gibbs LDA (NLP-REQ-07)     */
/* ========================================================================= */

typedef struct RCRE_NmfOpts {
    size_t   num_topics; /* K: latent topic rank (default 8)                 */
    size_t   max_iter;   /* HALS outer iterations (default 50)               */
    float    tol;        /* Relative Frobenius improvement stop (def 1e-4)   */
    uint64_t seed;       /* PCG64 deterministic seed (default 42)            */
} RCRE_NmfOpts;

typedef struct RCRE_NmfModel {
    size_t num_docs;
    size_t vocab_size;
    size_t num_topics;
    float* W;            /* [num_docs * num_topics] row-major doc-topic >= 0 */
    float* H;            /* [num_topics * vocab_size] row-major topic-term   */
    float  recon_error;  /* Final Frobenius norm ||X - W*H||_F               */
    size_t iters_run;
} RCRE_NmfModel;

RCRE_NODISCARD RCRE_NmfOpts rcre_nmf_default_opts(size_t num_topics);

/**
 * @brief Fits Non-Negative Matrix Factorization X ~ W * H directly on a
 *        non-negative RCRE_CSRMatrix using Fast HALS (O(K * nnz) per sweep).
 */
RCRE_NODISCARD int rcre_nmf_fit(
    RCRE_NmfModel* model,
    const RCRE_CSRMatrix* X_csr,
    const RCRE_NmfOpts* opts,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
);

/**
 * @brief Projects unseen documents X_new onto a fitted NMF basis H (fixed H,
 *        HALS W-updates). Writes out_W [X_new->rows * model->num_topics].
 */
RCRE_NODISCARD int rcre_nmf_transform(
    const RCRE_NmfModel* model,
    const RCRE_CSRMatrix* X_new,
    size_t max_iter,
    float* out_W,
    RCRE_Arena* scratch
);

typedef struct RCRE_TopicTerm {
    uint32_t term_id;
    float    weight;
} RCRE_TopicTerm;

/** Extracts top-k highest-weight term IDs from a topic row [vocab_size]. */
RCRE_NODISCARD size_t rcre_nlp_topic_top_terms(
    const float* topic_row,
    size_t vocab_size,
    size_t top_k,
    RCRE_TopicTerm* out_terms,
    RCRE_Arena* scratch
);

typedef struct RCRE_LdaOpts {
    size_t            num_topics; /* K topics (default 8)                    */
    size_t            num_iters;  /* Collapsed Gibbs sweeps (default 60)     */
    float             alpha;      /* Doc-topic Dirichlet prior (def 0.1)     */
    float             beta;       /* Topic-word Dirichlet prior (def 0.01)   */
    uint64_t          seed;       /* PCG64 deterministic seed (default 42)   */
    bool              lowercase;  /* ASCII lowercase tokens (default true)   */
    RCRE_TokenizeOpts tok_opts;
} RCRE_LdaOpts;

typedef struct RCRE_LdaModel {
    RCRE_StrPool vocab;
    size_t       num_docs;
    size_t       vocab_size;
    size_t       num_topics;
    float        alpha;
    float        beta;
    bool         lowercase;
    RCRE_TokenizeOpts tok_opts;

    float*       theta;       /* [num_docs * num_topics] doc-topic probs     */
    float*       phi;         /* [num_topics * vocab_size] topic-word probs  */
    float        log_perplexity;
} RCRE_LdaModel;

RCRE_NODISCARD RCRE_LdaOpts rcre_lda_default_opts(size_t num_topics);

/**
 * @brief Fits Latent Dirichlet Allocation via deterministic PCG64 Collapsed
 *        Gibbs Sampling in flat arena count tables (zero malloc).
 */
RCRE_NODISCARD int rcre_lda_fit(
    RCRE_LdaModel* model,
    const RCRE_Slice* docs,
    size_t num_docs,
    const RCRE_LdaOpts* opts,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
);

/**
 * @brief Infers normalized topic distribution out_theta[num_topics] for a new
 *        unseen document using fixed phi and deterministic fold-in Gibbs sweeps.
 */
RCRE_NODISCARD int rcre_lda_infer(
    const RCRE_LdaModel* model,
    RCRE_Slice doc,
    size_t num_iters,
    uint64_t seed,
    float* out_theta,
    RCRE_Arena* scratch
);

/* ========================================================================= */
/* 8. Native Sparse Text Classifiers (NLP-REQ-08)                            */
/* ========================================================================= */

typedef struct RCRE_NaiveBayesModel {
    size_t num_classes;
    size_t vocab_size;
    float  alpha;
    bool   complement;        /* false: Multinomial NB; true: Complement NB  */
    float* class_log_prior;   /* [num_classes]                               */
    float* feature_weights;   /* [num_classes * vocab_size] log-prob/weights */
} RCRE_NaiveBayesModel;

/**
 * @brief Trains Multinomial or Complement Naive Bayes directly on RCRE_CSRMatrix
 *        in a single O(nnz) linear pass (zero dense matrix materialization).
 */
RCRE_NODISCARD int rcre_nlp_nb_fit(
    RCRE_NaiveBayesModel* model,
    const RCRE_CSRMatrix* X_csr,
    const uint32_t* labels,
    size_t num_classes,
    float alpha,
    bool complement,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
);

/**
 * @brief Predicts class labels (and optional calibrated softmax probs [N*C])
 *        directly on RCRE_CSRMatrix rows.
 */
RCRE_NODISCARD int rcre_nlp_nb_predict(
    const RCRE_NaiveBayesModel* model,
    const RCRE_CSRMatrix* X_csr,
    uint32_t* out_labels,
    float* out_probs
);

typedef struct RCRE_SparseClfOpts {
    size_t   epochs;      /* Training passes over CSR rows (default 30)      */
    float    lr;          /* Initial learning rate (default 0.5)             */
    float    l2_reg;      /* L2 weight decay (default 1e-4)                  */
    uint64_t seed;        /* PCG64 deterministic shuffle seed (default 42)   */
} RCRE_SparseClfOpts;

typedef struct RCRE_SparseLinearClf {
    size_t num_classes;
    size_t vocab_size;
    float* weights;       /* [num_classes * vocab_size]                      */
    float* bias;          /* [num_classes]                                   */
} RCRE_SparseLinearClf;

RCRE_NODISCARD RCRE_SparseClfOpts rcre_sparse_clf_default_opts(void);

/**
 * @brief Trains an L2-regularized multi-class Softmax classifier directly on
 *        RCRE_CSRMatrix via PCG64-shuffled sparse SGD (O(epochs * nnz)).
 */
RCRE_NODISCARD int rcre_nlp_sparse_clf_fit(
    RCRE_SparseLinearClf* clf,
    const RCRE_CSRMatrix* X_csr,
    const uint32_t* labels,
    size_t num_classes,
    const RCRE_SparseClfOpts* opts,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
);

RCRE_NODISCARD int rcre_nlp_sparse_clf_predict(
    const RCRE_SparseLinearClf* clf,
    const RCRE_CSRMatrix* X_csr,
    uint32_t* out_labels,
    float* out_probs
);

/* ========================================================================= */
/* 9. Sentence Splitting, Extractive Summarization & Keywords (NLP-REQ-09)   */
/* ========================================================================= */

typedef struct RCRE_SentIter {
    RCRE_Slice src;
    size_t     pos;
} RCRE_SentIter;

RCRE_NODISCARD RCRE_SentIter rcre_nlp_sent_iter_init(RCRE_Slice text);
RCRE_NODISCARD bool rcre_nlp_sent_iter_next(RCRE_SentIter* it, RCRE_Slice* out_sent);

typedef struct RCRE_SummarizeOpts {
    float  damping;         /* PageRank damping factor (default 0.85)        */
    float  sim_threshold;   /* Min cosine edge threshold (default 0.05)      */
    float  mmr_lambda;      /* MMR relevance vs diversity in [0,1] (def 0.7) */
    size_t max_iter;        /* Power iterations (default 30)                 */
} RCRE_SummarizeOpts;

typedef struct RCRE_SummarySent {
    size_t     sent_index;  /* Original 0-based sentence index               */
    float      score;       /* LexRank/TextRank centrality score             */
    RCRE_Slice sentence;    /* Zero-copy slice into source document          */
} RCRE_SummarySent;

RCRE_NODISCARD RCRE_SummarizeOpts rcre_summarize_default_opts(void);

/**
 * @brief Extractive text summarization via TF-IDF sentence graph + LexRank
 *        PageRank power iteration + MMR diversity, returned in document order.
 */
RCRE_NODISCARD int rcre_nlp_summarize_extractive(
    RCRE_Slice text,
    size_t top_k,
    const RCRE_SummarizeOpts* opts,
    RCRE_SummarySent* out_sents,
    size_t* out_count,
    RCRE_Arena* scratch
);

typedef struct RCRE_Keyword {
    RCRE_Slice term;        /* Arena-backed lowercase keyword slice          */
    float      score;       /* TextRank co-occurrence centrality score       */
} RCRE_Keyword;

/**
 * @brief Extracts top-k keywords via sliding-window word co-occurrence
 *        TextRank graph (built-in English stopword filtering, zero malloc).
 */
RCRE_NODISCARD int rcre_nlp_extract_keywords(
    RCRE_Slice text,
    size_t top_k,
    size_t window_size,
    RCRE_Keyword* out_keywords,
    size_t* out_count,
    RCRE_Arena* arena,
    RCRE_Arena* scratch
);

/* ========================================================================= */
/* 10. Fast Sparse-Sparse CSR Row Cosine Similarity (NLP-REQ-10)             */
/* ========================================================================= */

/**
 * @brief Exact O(nnz_a + nnz_b) cosine similarity between two sorted CSR rows.
 *        Zero allocations.
 */
RCRE_NODISCARD float rcre_nlp_csr_row_cosine(
    const RCRE_CSRMatrix* csr_a, size_t row_a,
    const RCRE_CSRMatrix* csr_b, size_t row_b
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_NLP_H */
