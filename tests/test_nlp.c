/**
 * @file test_nlp.c
 * @brief Comprehensive Verification Suite for RCRE HPC NLP Engine (NLP-REQ-01..06)
 *
 * Compiles with RCRE_BAN_UNSAFE_CSTR active (zero legacy unsafe C-string calls).
 */

#define RCRE_BAN_UNSAFE_CSTR
#include "rcre_nlp.h"
#include "test_framework.h"
#include <math.h>
#include <string.h>

static void test_nlp_tokenizers_and_ngrams(void) {
    /* Word iterator with ASCII + UTF-8 words + punctuation split */
    RCRE_TokenizeOpts opts = rcre_nlp_default_tokenize_opts();
    opts.split_punct = true;
    RCRE_Slice text = RCRE_STR("Hello, D\xC3\xBCnya 2026!");
    RCRE_WordIter it = rcre_nlp_word_iter_init(text, &opts);
    RCRE_Slice tok;
    ASSERT_TRUE(rcre_nlp_word_iter_next(&it, &tok) && rcre_slice_eq(tok, RCRE_STR("Hello")));
    ASSERT_TRUE(rcre_nlp_word_iter_next(&it, &tok) && rcre_slice_eq(tok, RCRE_STR(",")));
    ASSERT_TRUE(rcre_nlp_word_iter_next(&it, &tok) && rcre_slice_eq(tok, RCRE_STR("D\xC3\xBCnya")));
    ASSERT_TRUE(rcre_nlp_word_iter_next(&it, &tok) && rcre_slice_eq(tok, RCRE_STR("2026")));
    ASSERT_TRUE(rcre_nlp_word_iter_next(&it, &tok) && rcre_slice_eq(tok, RCRE_STR("!")));
    ASSERT_FALSE(rcre_nlp_word_iter_next(&it, &tok));

    /* UTF-8 character 3-grams */
    RCRE_CharNGramIter cit = rcre_nlp_char_ngram_init(RCRE_STR("a\xC3\xBC\xE4\xB8\x96\xF0\x9F\x9A\x80"), 3);
    RCRE_Slice ng;
    ASSERT_TRUE(rcre_nlp_char_ngram_next(&cit, &ng) && rcre_slice_eq(ng, RCRE_STR("a\xC3\xBC\xE4\xB8\x96")));
    ASSERT_TRUE(rcre_nlp_char_ngram_next(&cit, &ng) && rcre_slice_eq(ng, RCRE_STR("\xC3\xBC\xE4\xB8\x96\xF0\x9F\x9A\x80")));
    ASSERT_FALSE(rcre_nlp_char_ngram_next(&cit, &ng));

    /* Zero-copy word bigrams */
    RCRE_WordNGramIter wit = rcre_nlp_word_ngram_init(RCRE_STR("fast zero copy c23 nlp"), 2, nullptr);
    ASSERT_TRUE(rcre_nlp_word_ngram_next(&wit, &ng) && rcre_slice_eq(ng, RCRE_STR("fast zero")));
    ASSERT_TRUE(rcre_nlp_word_ngram_next(&wit, &ng) && rcre_slice_eq(ng, RCRE_STR("zero copy")));
    ASSERT_TRUE(rcre_nlp_word_ngram_next(&wit, &ng) && rcre_slice_eq(ng, RCRE_STR("copy c23")));
    ASSERT_TRUE(rcre_nlp_word_ngram_next(&wit, &ng) && rcre_slice_eq(ng, RCRE_STR("c23 nlp")));
    ASSERT_FALSE(rcre_nlp_word_ngram_next(&wit, &ng));
}

static void test_nlp_bm25_and_rrf(void) {
    RCRE_Arena* arena = rcre_arena_create(512 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(256 * 1024);
    ASSERT_NOT_NULL(arena);
    ASSERT_NOT_NULL(scratch);

    const RCRE_Slice docs[] = {
        RCRE_STR("zero malloc c23 arena allocator simd neon kernel"),
        RCRE_STR("python pandas dataframe slow interpreter memory overhead"),
        RCRE_STR("bm25 inverted index lexical search with arena postings"),
        RCRE_STR("hybrid search fuses bm25 lexical and hnsw vector search"),
        RCRE_STR("neural network transformer attention flash kv cache")
    };

    RCRE_BM25Index idx;
    ASSERT_INT_EQ(RCRE_OK, rcre_bm25_build(&idx, docs, 5, nullptr, arena, scratch));
    ASSERT_UINT_EQ(0, scratch->offset); /* Scratch rewound cleanly */
    ASSERT_UINT_EQ(5, idx.num_docs);
    ASSERT_TRUE(idx.num_terms >= 25);

    RCRE_SearchHit hits[5];
    size_t n_hits = 0;
    ASSERT_INT_EQ(RCRE_OK, rcre_bm25_search(&idx, RCRE_STR("BM25 lexical search"), 3, hits, &n_hits, scratch));
    ASSERT_UINT_EQ(0, scratch->offset);
    ASSERT_UINT_EQ(2, n_hits);
    /* Doc 3 has all 3 query terms (bm25, lexical, search x2); doc 2 has 3 matches too */
    ASSERT_UINT_EQ(3, hits[0].doc_id);
    ASSERT_UINT_EQ(2, hits[1].doc_id);
    ASSERT_TRUE(hits[0].score > hits[1].score);

    /* OOV query returns 0 hits cleanly */
    ASSERT_INT_EQ(RCRE_OK, rcre_bm25_search(&idx, RCRE_STR("quantum_unobtainium"), 3, hits, &n_hits, scratch));
    ASSERT_UINT_EQ(0, n_hits);

    /* Hybrid Reciprocal Rank Fusion (BM25 + HNSW mock) */
    RCRE_SearchHit bm25_list[3] = { { 3, 4.2f }, { 2, 3.1f }, { 0, 1.0f } };
    RCRE_SearchHit hnsw_list[3] = { { 0, 0.95f }, { 3, 0.91f }, { 4, 0.80f } };
    const RCRE_SearchHit* lists[2] = { bm25_list, hnsw_list };
    const size_t counts[2] = { 3, 3 };
    RCRE_SearchHit fused[4];
    size_t n_fused = 0;
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_rrf_fuse(lists, counts, 2, 60.0f, 4, fused, &n_fused, scratch));
    ASSERT_UINT_EQ(0, scratch->offset);
    ASSERT_UINT_EQ(4, n_fused);
    /* Doc 3 is rank 1 in BM25 and rank 2 in HNSW -> highest RRF score */
    ASSERT_UINT_EQ(3, fused[0].doc_id);
    ASSERT_FLOAT_NEAR((1.0f / 61.0f) + (1.0f / 62.0f), fused[0].score, 1e-6f);
    /* Doc 0 is rank 3 in BM25 and rank 1 in HNSW -> second highest */
    ASSERT_UINT_EQ(0, fused[1].doc_id);

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
}

static void test_nlp_tfidf_and_feature_hashing(void) {
    RCRE_Arena* arena = rcre_arena_create(512 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(256 * 1024);
    ASSERT_NOT_NULL(arena);
    ASSERT_NOT_NULL(scratch);

    /* 3-doc mini corpus verified analytically against scikit-learn TfidfVectorizer:
     * vocab (alphabetical): 0="cat", 1="dog", 2="sat"
     * df: cat=2, dog=1, sat=2; N=3
     * smooth_idf:
     *   idf(cat) = ln((1+3)/(1+2)) + 1 = ln(4/3) + 1 = 1.28768207
     *   idf(dog) = ln((1+3)/(1+1)) + 1 = ln(2)   + 1 = 1.69314718
     *   idf(sat) = ln((1+3)/(1+2)) + 1 = ln(4/3) + 1 = 1.28768207
     */
    const RCRE_Slice docs[3] = {
        RCRE_STR("cat sat"),
        RCRE_STR("cat dog"),
        RCRE_STR("sat sat")
    };

    RCRE_TfidfVectorizer vec;
    ASSERT_INT_EQ(RCRE_OK, rcre_tfidf_fit(&vec, docs, 3, nullptr, arena, scratch));
    ASSERT_UINT_EQ(0, scratch->offset);
    ASSERT_UINT_EQ(3, vec.vocab_size);
    ASSERT_UINT_EQ(0, rcre_strpool_find(&vec.vocab, RCRE_STR("cat")));
    ASSERT_UINT_EQ(1, rcre_strpool_find(&vec.vocab, RCRE_STR("dog")));
    ASSERT_UINT_EQ(2, rcre_strpool_find(&vec.vocab, RCRE_STR("sat")));

    float idf_cat = (float)(log(4.0 / 3.0) + 1.0);
    float idf_dog = (float)(log(2.0) + 1.0);
    ASSERT_FLOAT_NEAR(idf_cat, vec.idf[0], 1e-6f);
    ASSERT_FLOAT_NEAR(idf_dog, vec.idf[1], 1e-6f);
    ASSERT_FLOAT_NEAR(idf_cat, vec.idf[2], 1e-6f);

    RCRE_CSRMatrix csr;
    ASSERT_INT_EQ(RCRE_OK, rcre_tfidf_transform(&vec, docs, 3, arena, scratch, &csr));
    ASSERT_UINT_EQ(0, scratch->offset);
    ASSERT_UINT_EQ(3, csr.rows);
    ASSERT_UINT_EQ(3, csr.cols);
    ASSERT_UINT_EQ(5, csr.nnz);

    /* Row 0 ("cat sat"): cols [0, 2], L2 normalized -> [1/sqrt(2), 1/sqrt(2)] */
    ASSERT_UINT_EQ(0, csr.row_pointers[0]);
    ASSERT_UINT_EQ(2, csr.row_pointers[1]);
    ASSERT_UINT_EQ(0, csr.col_indices[0]);
    ASSERT_UINT_EQ(2, csr.col_indices[1]);
    ASSERT_FLOAT_NEAR(0.70710678f, csr.values[0], 1e-5f);
    ASSERT_FLOAT_NEAR(0.70710678f, csr.values[1], 1e-5f);

    /* Row 1 ("cat dog"): cols [0, 1], L2 normalized */
    float n1 = sqrtf(idf_cat * idf_cat + idf_dog * idf_dog);
    ASSERT_UINT_EQ(0, csr.col_indices[2]);
    ASSERT_UINT_EQ(1, csr.col_indices[3]);
    ASSERT_FLOAT_NEAR(idf_cat / n1, csr.values[2], 1e-5f);
    ASSERT_FLOAT_NEAR(idf_dog / n1, csr.values[3], 1e-5f);

    /* Row 2 ("sat sat"): single term -> L2 normalized == 1.0 */
    ASSERT_UINT_EQ(2, csr.col_indices[4]);
    ASSERT_FLOAT_NEAR(1.0f, csr.values[4], 1e-6f);

    /* Signed Feature Hashing into 64B-aligned F32 tensor */
    RCRE_Tensor hashed;
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_hash_vectorize(docs, 3, 64, true, RCRE_NLP_NORM_L2, nullptr, arena, &hashed));
    ASSERT_UINT_EQ(3, hashed.shape[0]);
    ASSERT_UINT_EQ(64, hashed.shape[1]);
    const float* r0 = rcre_tensor_f32_data(&hashed);
    ASSERT_FLOAT_NEAR(1.0f, rcre_hal.dot_f32(r0, r0, 64), 1e-5f);

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
}

static void test_nlp_aho_corasick(void) {
    RCRE_Arena* arena = rcre_arena_create(512 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(256 * 1024);
    ASSERT_NOT_NULL(arena);
    ASSERT_NOT_NULL(scratch);

    /* Classic Aho-Corasick 1975 overlapping dictionary */
    const RCRE_Slice pats[4] = { RCRE_STR("he"), RCRE_STR("she"), RCRE_STR("his"), RCRE_STR("hers") };
    RCRE_AhoCorasick ac;
    ASSERT_INT_EQ(RCRE_OK, rcre_ac_build(&ac, pats, 4, false, arena, scratch));
    ASSERT_UINT_EQ(0, scratch->offset);

    RCRE_ACMatch matches[16];
    size_t n = rcre_ac_find_all(&ac, RCRE_STR("ushers"), matches, 16);
    /* Matches in "ushers": "she" [1,4), "he" [2,4), "hers" [2,6) */
    ASSERT_UINT_EQ(3, n);

    /* Case-insensitive redaction */
    const RCRE_Slice pii[2] = { RCRE_STR("secret"), RCRE_STR("token") };
    RCRE_AhoCorasick ac_ci;
    ASSERT_INT_EQ(RCRE_OK, rcre_ac_build(&ac_ci, pii, 2, true, arena, scratch));
    RCRE_Slice clean = rcre_ac_redact(&ac_ci, RCRE_STR("My SECRET api ToKeN is safe"), RCRE_STR("[REDACTED]"), arena);
    ASSERT_TRUE(rcre_slice_eq(clean, RCRE_STR("My [REDACTED] api [REDACTED] is safe")));

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
}

static void test_nlp_edit_distance_and_fuzzy(void) {
    RCRE_Arena* scratch = rcre_arena_create(64 * 1024);
    ASSERT_NOT_NULL(scratch);

    /* Myers 64-bit bit-parallel vs exact textbook examples */
    ASSERT_UINT_EQ(0, rcre_nlp_levenshtein(RCRE_STR("rcre"), RCRE_STR("rcre"), scratch));
    ASSERT_UINT_EQ(3, rcre_nlp_levenshtein(RCRE_STR("kitten"), RCRE_STR("sitting"), scratch));
    ASSERT_UINT_EQ(3, rcre_nlp_levenshtein(RCRE_STR("saturday"), RCRE_STR("sunday"), scratch));
    ASSERT_UINT_EQ(5, rcre_nlp_levenshtein(RCRE_STR(""), RCRE_STR("hello"), scratch));
    ASSERT_FLOAT_NEAR(1.0f - (3.0f / 7.0f), rcre_nlp_levenshtein_sim(RCRE_STR("kitten"), RCRE_STR("sitting"), scratch), 1e-6f);

    /* Differential parity: Myers 64-bit vs scalar DP on 64-byte and >64-byte strings */
    RCRE_Slice long_a = RCRE_STR("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef_TAIL_A");
    RCRE_Slice long_b = RCRE_STR("0123456789Xbcdef0123456789abcdef0123456789abcdeY0123456789abcdef_TAIL_BB");
    ASSERT_UINT_EQ(rcre_nlp_levenshtein_scalar(long_a, long_b, scratch),
                   rcre_nlp_levenshtein(long_a, long_b, scratch));

    /* Exact boundary m == 64 (Myers high_bit == 1ULL << 63) */
    RCRE_Slice s64a = rcre_slice_sub(long_a, 0, 64);
    RCRE_Slice s64b = rcre_slice_sub(long_b, 0, 64);
    ASSERT_UINT_EQ(2, rcre_nlp_levenshtein(s64a, s64b, scratch));
    ASSERT_UINT_EQ(rcre_nlp_levenshtein_scalar(s64a, s64b, scratch),
                   rcre_nlp_levenshtein(s64a, s64b, scratch));

    /* Classic Jaro-Winkler benchmark: MARTHA vs MARHTA == 0.961111... */
    float jw = rcre_nlp_jaro_winkler(RCRE_STR("MARTHA"), RCRE_STR("MARHTA"), 0.1f, scratch);
    ASSERT_FLOAT_NEAR(0.961111f, jw, 1e-4f);
    ASSERT_FLOAT_NEAR(1.0f, rcre_nlp_jaro_winkler(RCRE_STR("RCRE"), RCRE_STR("RCRE"), 0.1f, scratch), 1e-6f);

    /* Character 2-gram Jaccard */
    float jac = rcre_nlp_jaccard_char_ngrams(RCRE_STR("night"), RCRE_STR("nacht"), 2, scratch);
    /* {"ni","ig","gh","ht"} vs {"na","ac","ch","ht"} -> 1/7 */
    ASSERT_FLOAT_NEAR(1.0f / 7.0f, jac, 1e-5f);
    ASSERT_UINT_EQ(0, scratch->offset);

    rcre_arena_destroy(scratch);
}

static void test_nlp_dedup_and_rag_chunker(void) {
    RCRE_Arena* arena = rcre_arena_create(128 * 1024);
    ASSERT_NOT_NULL(arena);

    RCRE_Slice doc1 = RCRE_STR("the quick brown fox jumps over the lazy dog in the sunny park");
    RCRE_Slice doc2 = RCRE_STR("the quick brown fox jumps over the lazy cat in the sunny park");
    RCRE_Slice doc3 = RCRE_STR("sparse matrix multiplication uses arm neon intrinsics for hpc");

    uint64_t sh1 = rcre_nlp_simhash64(doc1, nullptr);
    uint64_t sh2 = rcre_nlp_simhash64(doc2, nullptr);
    uint64_t sh3 = rcre_nlp_simhash64(doc3, nullptr);
    ASSERT_UINT_EQ(0, rcre_nlp_hamming64(sh1, sh1));
    uint32_t d12 = rcre_nlp_hamming64(sh1, sh2);
    uint32_t d13 = rcre_nlp_hamming64(sh1, sh3);
    ASSERT_TRUE(d12 < d13);

    uint64_t mh1[64], mh2[64], mh3[64];
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_minhash_compute(doc1, 2, mh1, 64, nullptr));
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_minhash_compute(doc2, 2, mh2, 64, nullptr));
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_minhash_compute(doc3, 2, mh3, 64, nullptr));
    ASSERT_FLOAT_NEAR(1.0f, rcre_nlp_minhash_jaccard(mh1, mh1, 64), 1e-6f);
    ASSERT_TRUE(rcre_nlp_minhash_jaccard(mh1, mh2, 64) > 0.5f);
    ASSERT_FLOAT_NEAR(0.0f, rcre_nlp_minhash_jaccard(mh1, mh3, 64), 1e-6f);

    /* UTF-8 zero-copy RAG chunker with overlap */
    RCRE_Slice utf8_doc = RCRE_STR("Alpha \xF0\x9F\x9A\x80 Beta \xE4\xB8\x96\xE7\x95\x8C Gamma \xC3\xBC Delta Epsilon Zeta");
    RCRE_Slice* chunks = nullptr;
    size_t n_chunks = 0;
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_chunk_utf8(utf8_doc, 16, 4, arena, &chunks, &n_chunks));
    ASSERT_TRUE(n_chunks >= 3);
    for (size_t i = 0; i < n_chunks; ++i) {
        ASSERT_TRUE(chunks[i].len > 0 && chunks[i].len <= 16);
        ASSERT_TRUE(rcre_utf8_validate(chunks[i], nullptr));
        /* Verify zero-copy: chunk data points strictly inside utf8_doc */
        ASSERT_TRUE(chunks[i].data >= utf8_doc.data &&
                    chunks[i].data + chunks[i].len <= utf8_doc.data + utf8_doc.len);
    }

    rcre_arena_destroy(arena);
}

static void test_nlp_nmf_topic_modeling(void) {
    RCRE_Arena* arena   = rcre_arena_create(1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(512 * 1024);
    ASSERT_NOT_NULL(arena);
    ASSERT_NOT_NULL(scratch);

    /* Two clearly separable topic clusters: 0..3 HPC/SIMD, 4..7 Cooking */
    const RCRE_Slice docs[8] = {
        RCRE_STR("simd neon avx2 vector register cache kernel gemm"),
        RCRE_STR("arena memory allocator zero malloc cache simd kernel"),
        RCRE_STR("sparse matrix csr vector neon gemm throughput cache"),
        RCRE_STR("kernel vector avx2 neon memory arena throughput"),
        RCRE_STR("garlic olive oil pasta tomato basil simmer sauce"),
        RCRE_STR("flour butter sugar bake oven cake pastry dough"),
        RCRE_STR("tomato onion garlic simmer herb sauce olive pasta"),
        RCRE_STR("oven bake butter dough garlic herb olive sauce")
    };

    RCRE_TfidfVectorizer vec;
    RCRE_CSRMatrix csr;
    ASSERT_INT_EQ(RCRE_OK, rcre_tfidf_fit(&vec, docs, 8, nullptr, arena, scratch));
    ASSERT_INT_EQ(RCRE_OK, rcre_tfidf_transform(&vec, docs, 8, arena, scratch, &csr));
    ASSERT_UINT_EQ(0, scratch->offset);

    RCRE_NmfOpts opts = rcre_nmf_default_opts(2);
    opts.max_iter = 60;
    opts.seed = 12345ULL;

    RCRE_NmfModel nmf1, nmf2;
    ASSERT_INT_EQ(RCRE_OK, rcre_nmf_fit(&nmf1, &csr, &opts, arena, scratch));
    ASSERT_INT_EQ(RCRE_OK, rcre_nmf_fit(&nmf2, &csr, &opts, arena, scratch));
    ASSERT_UINT_EQ(0, scratch->offset);

    /* Bit-exact deterministic repeatability via PCG64 */
    ASSERT_FLOAT_NEAR(nmf1.recon_error, nmf2.recon_error, 1e-7f);
    for (size_t i = 0; i < 8 * 2; ++i) {
        ASSERT_TRUE(nmf1.W[i] >= 0.0f);
        ASSERT_FLOAT_NEAR(nmf1.W[i], nmf2.W[i], 1e-7f);
    }

    /* All HPC docs (0..3) share dominant topic; all Cooking docs (4..7) share the other */
    size_t hpc_topic = (nmf1.W[0 * 2 + 0] >= nmf1.W[0 * 2 + 1]) ? 0 : 1;
    size_t cook_topic = 1U - hpc_topic;
    for (size_t d = 0; d < 4; ++d) {
        ASSERT_TRUE(nmf1.W[d * 2 + hpc_topic] > nmf1.W[d * 2 + cook_topic]);
    }
    for (size_t d = 4; d < 8; ++d) {
        ASSERT_TRUE(nmf1.W[d * 2 + cook_topic] > nmf1.W[d * 2 + hpc_topic]);
    }

    /* Unseen document projection via rcre_nmf_transform */
    const RCRE_Slice unseen[2] = {
        RCRE_STR("neon simd gemm cache kernel"),
        RCRE_STR("garlic tomato olive pasta sauce")
    };
    RCRE_CSRMatrix csr_new;
    ASSERT_INT_EQ(RCRE_OK, rcre_tfidf_transform(&vec, unseen, 2, arena, scratch, &csr_new));
    float W_new[4];
    ASSERT_INT_EQ(RCRE_OK, rcre_nmf_transform(&nmf1, &csr_new, 30, W_new, scratch));
    ASSERT_TRUE(W_new[0 * 2 + hpc_topic] > W_new[0 * 2 + cook_topic]);
    ASSERT_TRUE(W_new[1 * 2 + cook_topic] > W_new[1 * 2 + hpc_topic]);

    /* Top terms extraction */
    RCRE_TopicTerm top3[3];
    size_t nt = rcre_nlp_topic_top_terms(nmf1.H + hpc_topic * nmf1.vocab_size, nmf1.vocab_size, 3, top3, scratch);
    ASSERT_UINT_EQ(3, nt);
    ASSERT_TRUE(top3[0].weight >= top3[1].weight && top3[1].weight >= top3[2].weight);

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
}

static void test_nlp_lda_collapsed_gibbs(void) {
    RCRE_Arena* arena   = rcre_arena_create(1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(512 * 1024);
    ASSERT_NOT_NULL(arena);
    ASSERT_NOT_NULL(scratch);

    const RCRE_Slice docs[8] = {
        RCRE_STR("simd neon avx2 vector register cache kernel gemm neon simd"),
        RCRE_STR("arena memory allocator zero malloc cache simd kernel arena"),
        RCRE_STR("sparse matrix csr vector neon gemm throughput cache vector"),
        RCRE_STR("kernel vector avx2 neon memory arena throughput cache gemm"),
        RCRE_STR("garlic olive oil pasta tomato basil simmer sauce pasta garlic"),
        RCRE_STR("flour butter sugar bake oven cake pastry dough bake butter"),
        RCRE_STR("tomato onion garlic simmer herb sauce olive pasta tomato"),
        RCRE_STR("oven bake butter dough garlic herb olive sauce bake oven")
    };

    RCRE_LdaOpts opts = rcre_lda_default_opts(2);
    opts.num_iters = 100;
    opts.alpha = 0.1f;
    opts.beta  = 0.05f;
    opts.seed  = 777ULL;

    RCRE_LdaModel lda1, lda2;
    ASSERT_INT_EQ(RCRE_OK, rcre_lda_fit(&lda1, docs, 8, &opts, arena, scratch));
    ASSERT_INT_EQ(RCRE_OK, rcre_lda_fit(&lda2, docs, 8, &opts, arena, scratch));
    ASSERT_UINT_EQ(0, scratch->offset);
    ASSERT_FLOAT_NEAR(lda1.log_perplexity, lda2.log_perplexity, 1e-6f);

    /* Simplex invariants: sum_k theta[d,k] == 1, sum_w phi[k,w] == 1 */
    for (size_t d = 0; d < 8; ++d) {
        float s = lda1.theta[d * 2 + 0] + lda1.theta[d * 2 + 1];
        ASSERT_FLOAT_NEAR(1.0f, s, 1e-5f);
        ASSERT_FLOAT_NEAR(lda1.theta[d * 2 + 0], lda2.theta[d * 2 + 0], 1e-6f);
    }
    for (size_t k = 0; k < 2; ++k) {
        float s = 0.0f;
        for (size_t w = 0; w < lda1.vocab_size; ++w) s += lda1.phi[k * lda1.vocab_size + w];
        ASSERT_FLOAT_NEAR(1.0f, s, 1e-4f);
    }

    /* Topic separation across the two clusters */
    size_t hpc_t = (lda1.theta[0 * 2 + 0] >= lda1.theta[0 * 2 + 1]) ? 0 : 1;
    size_t cook_t = 1U - hpc_t;
    for (size_t d = 0; d < 4; ++d) ASSERT_TRUE(lda1.theta[d * 2 + hpc_t] > 0.5f);
    for (size_t d = 4; d < 8; ++d) ASSERT_TRUE(lda1.theta[d * 2 + cook_t] > 0.5f);

    /* Unseen fold-in inference */
    float inf_theta[2] = { 0 };
    ASSERT_INT_EQ(RCRE_OK, rcre_lda_infer(&lda1, RCRE_STR("neon simd cache kernel gemm"), 40, 99ULL, inf_theta, scratch));
    ASSERT_FLOAT_NEAR(1.0f, inf_theta[0] + inf_theta[1], 1e-5f);
    ASSERT_TRUE(inf_theta[hpc_t] > inf_theta[cook_t]);

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
}

static void test_nlp_sparse_classifiers_and_cosine(void) {
    RCRE_Arena* arena   = rcre_arena_create(1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(512 * 1024);
    ASSERT_NOT_NULL(arena);
    ASSERT_NOT_NULL(scratch);

    /* 3-class corpus: 0=HPC, 1=Cooking, 2=Astronomy */
    const RCRE_Slice train_docs[9] = {
        RCRE_STR("simd neon avx2 cache kernel matrix gemm"),
        RCRE_STR("arena allocator zero malloc vector throughput"),
        RCRE_STR("sparse csr matrix dot product neon kernel"),
        RCRE_STR("garlic olive oil tomato pasta basil sauce"),
        RCRE_STR("butter flour sugar oven bake cake dough"),
        RCRE_STR("simmer onion herb garlic sauce olive pasta"),
        RCRE_STR("galaxy nebula supernova telescope orbit star"),
        RCRE_STR("planet asteroid comet gravity orbit telescope"),
        RCRE_STR("black hole quasar galaxy starlight nebula")
    };
    const uint32_t train_y[9] = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };

    const RCRE_Slice test_docs[3] = {
        RCRE_STR("neon matrix kernel arena cache"),
        RCRE_STR("olive garlic pasta bake sauce"),
        RCRE_STR("nebula galaxy telescope orbit star")
    };
    const uint32_t test_y[3] = { 0, 1, 2 };

    RCRE_TfidfVectorizer vec;
    RCRE_CSRMatrix X_train, X_test;
    ASSERT_INT_EQ(RCRE_OK, rcre_tfidf_fit(&vec, train_docs, 9, nullptr, arena, scratch));
    ASSERT_INT_EQ(RCRE_OK, rcre_tfidf_transform(&vec, train_docs, 9, arena, scratch, &X_train));
    ASSERT_INT_EQ(RCRE_OK, rcre_tfidf_transform(&vec, test_docs,  3, arena, scratch, &X_test));

    /* Exact CSR row cosine similarity check */
    ASSERT_FLOAT_NEAR(1.0f, rcre_nlp_csr_row_cosine(&X_train, 0, &X_train, 0), 1e-5f);
    ASSERT_TRUE(rcre_nlp_csr_row_cosine(&X_train, 0, &X_train, 2) >
                rcre_nlp_csr_row_cosine(&X_train, 0, &X_train, 3));
    ASSERT_FLOAT_NEAR(0.0f, rcre_nlp_csr_row_cosine(&X_train, 0, &X_train, 6), 1e-6f);

    /* 1. Multinomial Naive Bayes */
    RCRE_NaiveBayesModel mnb;
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_nb_fit(&mnb, &X_train, train_y, 3, 1.0f, false, arena, scratch));
    uint32_t preds[3] = { 99, 99, 99 };
    float probs[9] = { 0 };
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_nb_predict(&mnb, &X_test, preds, probs));
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_UINT_EQ(test_y[i], preds[i]);
        ASSERT_FLOAT_NEAR(1.0f, probs[i * 3 + 0] + probs[i * 3 + 1] + probs[i * 3 + 2], 1e-5f);
    }

    /* 2. Complement Naive Bayes */
    RCRE_NaiveBayesModel cnb;
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_nb_fit(&cnb, &X_train, train_y, 3, 1.0f, true, arena, scratch));
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_nb_predict(&cnb, &X_test, preds, probs));
    for (size_t i = 0; i < 3; ++i) ASSERT_UINT_EQ(test_y[i], preds[i]);

    /* 3. Native Sparse Multi-Class Softmax SGD */
    RCRE_SparseLinearClf sclf;
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_sparse_clf_fit(&sclf, &X_train, train_y, 3, nullptr, arena, scratch));
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_sparse_clf_predict(&sclf, &X_test, preds, probs));
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_UINT_EQ(test_y[i], preds[i]);
        ASSERT_FLOAT_NEAR(1.0f, probs[i * 3 + 0] + probs[i * 3 + 1] + probs[i * 3 + 2], 1e-5f);
    }
    ASSERT_UINT_EQ(0, scratch->offset);

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
}

static void test_nlp_summarization_and_keywords(void) {
    RCRE_Arena* arena   = rcre_arena_create(512 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(512 * 1024);
    ASSERT_NOT_NULL(arena);
    ASSERT_NOT_NULL(scratch);

    /* Zero-copy sentence splitter with abbreviation & decimal guards */
    RCRE_Slice sample = RCRE_STR("Dr. Smith measured 3.14 seconds. Is C23 fast?! Yes, zero malloc arenas rock.");
    RCRE_SentIter sit = rcre_nlp_sent_iter_init(sample);
    RCRE_Slice sent;
    ASSERT_TRUE(rcre_nlp_sent_iter_next(&sit, &sent) && rcre_slice_eq(sent, RCRE_STR("Dr. Smith measured 3.14 seconds.")));
    ASSERT_TRUE(rcre_nlp_sent_iter_next(&sit, &sent) && rcre_slice_eq(sent, RCRE_STR("Is C23 fast?!")));
    ASSERT_TRUE(rcre_nlp_sent_iter_next(&sit, &sent) && rcre_slice_eq(sent, RCRE_STR("Yes, zero malloc arenas rock.")));
    ASSERT_FALSE(rcre_nlp_sent_iter_next(&sit, &sent));

    /* LexRank + MMR Extractive Summarization */
    RCRE_Slice article = RCRE_STR(
        "The weather outside was mildly cloudy today. "
        "RCRE accelerates sparse matrix multiplication using ARM NEON SIMD kernels and linear arenas. "
        "Linear arenas eliminate heap fragmentation and guarantee deterministic latency for SIMD kernels. "
        "Sparse matrix multiplication and SIMD kernels deliver predictable HPC throughput in pure C23. "
        "Meanwhile, a cat slept quietly on a windowsill."
    );
    RCRE_SummarySent summary[2];
    size_t n_sum = 0;
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_summarize_extractive(article, 2, nullptr, summary, &n_sum, scratch));
    ASSERT_UINT_EQ(0, scratch->offset);
    ASSERT_UINT_EQ(2, n_sum);
    /* Must preserve original document order and pick from the 3 core HPC sentences (indices 1,2,3) */
    ASSERT_TRUE(summary[0].sent_index < summary[1].sent_index);
    ASSERT_TRUE(summary[0].sent_index >= 1 && summary[0].sent_index <= 3);
    ASSERT_TRUE(summary[1].sent_index >= 1 && summary[1].sent_index <= 3);
    /* Zero-copy verification */
    ASSERT_TRUE(summary[0].sentence.data >= article.data &&
                summary[0].sentence.data + summary[0].sentence.len <= article.data + article.len);

    /* TextRank Co-Occurrence Keyword Extraction */
    RCRE_Keyword kws[5];
    size_t n_kw = 0;
    ASSERT_INT_EQ(RCRE_OK, rcre_nlp_extract_keywords(article, 5, 4, kws, &n_kw, arena, scratch));
    ASSERT_UINT_EQ(0, scratch->offset);
    ASSERT_UINT_EQ(5, n_kw);
    ASSERT_TRUE(kws[0].score >= kws[1].score && kws[1].score >= kws[4].score);
    /* Stopwords like "the", "and" must not appear in keywords */
    for (size_t i = 0; i < n_kw; ++i) {
        ASSERT_FALSE(rcre_slice_eq(kws[i].term, RCRE_STR("the")));
        ASSERT_FALSE(rcre_slice_eq(kws[i].term, RCRE_STR("and")));
    }

    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
}

int main(void) {
    rcre_hal_init();
    test_suite_begin("RCRE HPC Natural Language Processing Engine (NLP-REQ-01 to 10)");
    TEST_RUN(test_nlp_tokenizers_and_ngrams);
    TEST_RUN(test_nlp_bm25_and_rrf);
    TEST_RUN(test_nlp_tfidf_and_feature_hashing);
    TEST_RUN(test_nlp_aho_corasick);
    TEST_RUN(test_nlp_edit_distance_and_fuzzy);
    TEST_RUN(test_nlp_dedup_and_rag_chunker);
    TEST_RUN(test_nlp_nmf_topic_modeling);
    TEST_RUN(test_nlp_lda_collapsed_gibbs);
    TEST_RUN(test_nlp_sparse_classifiers_and_cosine);
    TEST_RUN(test_nlp_summarization_and_keywords);
    return test_suite_report();
}
