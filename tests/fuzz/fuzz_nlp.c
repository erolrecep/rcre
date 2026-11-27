/**
 * @file fuzz_nlp.c
 * @brief LibFuzzer & Standalone Differential Fuzzer for RCRE HPC NLP Engine
 *
 * Differentially verifies:
 *   1. Myers 64-bit bit-parallel Levenshtein == textbook scalar DP oracle.
 *   2. Aho-Corasick flat DFA match count == naive substring scan oracle.
 *   3. UTF-8 RAG chunker never splits UTF-8 codepoints and always terminates.
 *   4. Zero ASan/UBSan violations across BM25, TF-IDF, SimHash, MinHash, and tokenizers.
 */

#include "rcre_nlp.h"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fuzz_nlp_abort(const char* msg) {
    fprintf(stderr, "FUZZ_NLP FAILURE: %s\n", msg);
    abort();
}

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (!data || size > 4096U) return 0;

    uint8_t arena_mem[131072];
    uint8_t scratch_mem[65536];
    RCRE_Arena arena, scratch;
    if (rcre_arena_from_buffer(&arena, arena_mem, sizeof(arena_mem)) != RCRE_OK ||
        rcre_arena_from_buffer(&scratch, scratch_mem, sizeof(scratch_mem)) != RCRE_OK) {
        return 0;
    }

    RCRE_Slice s = rcre_slice_from_parts(data, size);

    /* 1. Differential Myers bit-parallel vs scalar Levenshtein */
    size_t mid = size / 2U;
    RCRE_Slice a = rcre_slice_sub(s, 0, mid);
    RCRE_Slice b = rcre_slice_sub(s, mid, size);
    if (a.len <= 160 && b.len <= 160) {
        size_t d_fast = rcre_nlp_levenshtein(a, b, &scratch);
        size_t d_ref  = rcre_nlp_levenshtein_scalar(a, b, &scratch);
        if (d_fast != d_ref) {
            fuzz_nlp_abort("Myers bit-parallel Levenshtein != scalar oracle");
        }
        (void)rcre_nlp_jaro_winkler(a, b, 0.1f, &scratch);
    }

    /* 2. Differential Aho-Corasick DFA vs naive scan */
    if (size >= 8) {
        RCRE_Slice pats[3] = {
            rcre_slice_sub(s, 0, 2),
            rcre_slice_sub(s, 1, 4),
            rcre_slice_sub(s, 3, 6)
        };
        RCRE_ArenaTemp m = rcre_arena_temp_begin(&arena);
        RCRE_AhoCorasick ac;
        if (rcre_ac_build(&ac, pats, 3, false, &arena, &scratch) == RCRE_OK) {
            size_t ac_cnt = rcre_ac_scan(&ac, s, nullptr, nullptr);
            size_t naive_cnt = 0;
            for (size_t p = 0; p < 3; ++p) {
                for (size_t i = 0; i + pats[p].len <= s.len; ++i) {
                    if (memcmp(s.data + i, pats[p].data, pats[p].len) == 0) {
                        naive_cnt++;
                    }
                }
            }
            if (ac_cnt != naive_cnt) {
                fuzz_nlp_abort("Aho-Corasick match count != naive oracle");
            }
            (void)rcre_ac_redact(&ac, s, RCRE_STR("*"), &arena);
        }
        rcre_arena_temp_end(m);
    }

    /* 3. UTF-8 RAG Chunker safety (always valid UTF-8 on valid UTF-8 input) */
    if (rcre_utf8_validate(s, nullptr) && size > 0) {
        RCRE_ArenaTemp m = rcre_arena_temp_begin(&arena);
        RCRE_Slice* chunks = nullptr;
        size_t n_chunks = 0;
        if (rcre_nlp_chunk_utf8(s, 17, 4, &arena, &chunks, &n_chunks) == RCRE_OK) {
            for (size_t i = 0; i < n_chunks; ++i) {
                if (!rcre_utf8_validate(chunks[i], nullptr)) {
                    fuzz_nlp_abort("rcre_nlp_chunk_utf8 broke UTF-8 boundary");
                }
            }
        }
        rcre_arena_temp_end(m);
    }

    /* 4. SimHash, MinHash, BM25, TF-IDF smoke safety */
    (void)rcre_nlp_simhash64(s, nullptr);
    uint64_t sig[16];
    (void)rcre_nlp_minhash_compute(s, 2, sig, 16, nullptr);

    RCRE_Slice mini_docs[2] = { a, b };
    RCRE_ArenaTemp m2 = rcre_arena_temp_begin(&arena);
    RCRE_BM25Index bm25;
    if (rcre_bm25_build(&bm25, mini_docs, 2, nullptr, &arena, &scratch) == RCRE_OK) {
        RCRE_SearchHit hits[2];
        size_t nh = 0;
        (void)rcre_bm25_search(&bm25, a, 2, hits, &nh, &scratch);
    }
    RCRE_TfidfVectorizer tfidf;
    if (rcre_tfidf_fit(&tfidf, mini_docs, 2, nullptr, &arena, &scratch) == RCRE_OK) {
        RCRE_CSRMatrix csr;
        if (rcre_tfidf_transform(&tfidf, mini_docs, 2, &arena, &scratch, &csr) == RCRE_OK && csr.cols > 0) {
            (void)rcre_nlp_csr_row_cosine(&csr, 0, &csr, 1);

            /* 5. Phase 2 fuzzing: Sparse NMF, NB, Softmax, LDA, Summarizer, Keywords */
            RCRE_NmfOpts nopts = rcre_nmf_default_opts(2);
            nopts.max_iter = 4;
            RCRE_NmfModel nmf;
            (void)rcre_nmf_fit(&nmf, &csr, &nopts, &arena, &scratch);

            const uint32_t y2[2] = { 0, 1 };
            RCRE_NaiveBayesModel nb;
            if (rcre_nlp_nb_fit(&nb, &csr, y2, 2, 1.0f, (size & 1U) != 0, &arena, &scratch) == RCRE_OK) {
                uint32_t p2[2];
                float pr[4];
                (void)rcre_nlp_nb_predict(&nb, &csr, p2, pr);
            }
            RCRE_SparseClfOpts sopts = rcre_sparse_clf_default_opts();
            sopts.epochs = 2;
            RCRE_SparseLinearClf sclf;
            if (rcre_nlp_sparse_clf_fit(&sclf, &csr, y2, 2, &sopts, &arena, &scratch) == RCRE_OK) {
                uint32_t p2[2];
                float pr[4];
                (void)rcre_nlp_sparse_clf_predict(&sclf, &csr, p2, pr);
            }
        }
    }

    RCRE_LdaOpts lopts = rcre_lda_default_opts(2);
    lopts.num_iters = 3;
    RCRE_LdaModel lda;
    if (rcre_lda_fit(&lda, mini_docs, 2, &lopts, &arena, &scratch) == RCRE_OK) {
        float th[2];
        (void)rcre_lda_infer(&lda, a, 2, 11ULL, th, &scratch);
    }

    RCRE_SummarySent ssum[2];
    size_t nsum = 0;
    (void)rcre_nlp_summarize_extractive(s, 2, nullptr, ssum, &nsum, &scratch);

    RCRE_Keyword kws[4];
    size_t nkw = 0;
    (void)rcre_nlp_extract_keywords(s, 4, 3, kws, &nkw, &arena, &scratch);
    rcre_arena_temp_end(m2);

    return 0;
}

#ifdef FUZZ_STANDALONE
static uint64_t fuzz_nlp_rng(uint64_t* st) {
    uint64_t x = *st;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *st = x;
    return x;
}

int main(void) {
    rcre_hal_init();
    uint64_t rng = 0x123456789ABCDEF0ULL;
    uint8_t buf[256];

    for (size_t iter = 0; iter < 10000; ++iter) {
        size_t len = (size_t)(fuzz_nlp_rng(&rng) % sizeof(buf));
        int mode = (int)(iter % 3);
        for (size_t i = 0; i < len; ++i) {
            uint64_t r = fuzz_nlp_rng(&rng);
            if (mode == 0) {
                buf[i] = (uint8_t)('a' + (r % 6U)); /* Small alphabet: high AC & edit overlap */
            } else if (mode == 1) {
                buf[i] = ((r & 7U) == 0) ? ' ' : (uint8_t)('a' + (r % 26U));
            } else {
                buf[i] = (uint8_t)(r & 0x7FU);
            }
        }
        LLVMFuzzerTestOneInput(buf, len);
    }
    printf("✅ fuzz_nlp passed 10,000 differential Myers/Aho-Corasick/BM25/TF-IDF iterations!\n");
    return 0;
}
#endif
