/**
 * @file bench_nlp.c
 * @brief HPC Throughput Benchmarks for RCRE NLP Engine (NLP-REQ-01..06)
 */

#include "rcre_nlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double nlp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    rcre_hal_init();
    printf("==================================================================\n");
    printf("  RCRE HPC NLP Engine Benchmark (%s)\n", rcre_hal.backend_name);
    printf("  Host CPU: %s\n", rcre_cpu_get_brand());
    printf("==================================================================\n");

    const size_t corpus_bytes = 16 * 1024 * 1024;
    RCRE_Arena* arena   = rcre_arena_create(64 * 1024 * 1024);
    RCRE_Arena* scratch = rcre_arena_create(16 * 1024 * 1024);
    if (!arena || !scratch) {
        fprintf(stderr, "Failed to create benchmark arenas\n");
        return 1;
    }

    /* Build deterministic 16 MB English-like synthetic corpus */
    static const char* vocab[] = {
        "matrix", "tensor", "arena", "kernel", "vector", "search", "sparse", "memory",
        "cache", "simd", "neon", "avx2", "transformer", "attention", "quantize", "stream",
        "lexical", "bm25", "tfidf", "hash", "token", "utf8", "slice", "builder",
        "parallel", "thread", "slab", "deterministic", "latency", "throughput", "index", "graph"
    };
    uint8_t* text_buf = (uint8_t*)rcre_arena_alloc(arena, corpus_bytes + 1, 64);
    size_t pos = 0;
    uint64_t rng = 0xCAFEBABE12345678ULL;
    while (pos + 24 < corpus_bytes) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        const char* w = vocab[rng & 31U];
        size_t wl = strlen(w);
        memcpy(text_buf + pos, w, wl);
        pos += wl;
        text_buf[pos++] = ((rng >> 8) & 15U) == 0 ? '\n' : ' ';
    }
    RCRE_Slice corpus = rcre_slice_from_parts(text_buf, pos);
    volatile size_t sink = 0;

    /* 1. Zero-copy Word Tokenization */
    double t0 = nlp_now_sec();
    size_t tok_count = 0;
    RCRE_WordIter wit = rcre_nlp_word_iter_init(corpus, nullptr);
    RCRE_Slice tok;
    while (rcre_nlp_word_iter_next(&wit, &tok)) {
        tok_count++;
        sink += tok.len;
    }
    double dt_tok = nlp_now_sec() - t0;
    printf("Zero-Copy Word Tokenize (16MB): %6.2f MB/s (%5.2f M tokens/s, 0 mallocs)\n",
           ((double)corpus.len / 1e6) / dt_tok, ((double)tok_count / 1e6) / dt_tok);

    /* 2. Aho-Corasick 16-pattern Flat DFA Scan */
    RCRE_ArenaTemp m_ac = rcre_arena_temp_begin(arena);
    RCRE_Slice ac_pats[16];
    for (size_t i = 0; i < 16; ++i) ac_pats[i] = rcre_slice_from_cstr(vocab[i * 2]);
    RCRE_AhoCorasick ac;
    (void)rcre_ac_build(&ac, ac_pats, 16, true, arena, scratch);
    t0 = nlp_now_sec();
    size_t ac_matches = rcre_ac_scan(&ac, corpus, nullptr, nullptr);
    double dt_ac = nlp_now_sec() - t0;
    sink += ac_matches;
    printf("Aho-Corasick 16-Pat DFA (16MB): %6.2f MB/s (%zu matches, 1 load/byte)\n",
           ((double)corpus.len / 1e6) / dt_ac, ac_matches);
    rcre_arena_temp_end(m_ac);

    /* 3. Myers 64-Bit Bit-Parallel Levenshtein vs Scalar DP */
    RCRE_Slice s1 = RCRE_STR("deterministic_high_performance_c23_simd_string_distance_bench");
    RCRE_Slice s2 = RCRE_STR("deterministic_high_perf_c23_neon_simd_strng_distance_bench_v2");
    const int lev_iters = 50000;
    t0 = nlp_now_sec();
    for (int i = 0; i < lev_iters; ++i) sink += rcre_nlp_levenshtein_scalar(s1, s2, scratch);
    double dt_lev_s = nlp_now_sec() - t0;

    t0 = nlp_now_sec();
    for (int i = 0; i < lev_iters; ++i) sink += rcre_nlp_levenshtein(s1, s2, scratch);
    double dt_lev_m = nlp_now_sec() - t0;
    printf("Levenshtein (61-byte strings):  Scalar %5.2f M/s | Myers64 %5.2f M/s (%.2fx)\n",
           ((double)lev_iters / 1e6) / dt_lev_s, ((double)lev_iters / 1e6) / dt_lev_m, dt_lev_s / dt_lev_m);

    /* 4. BM25 Index Build & Query Latency (10,000 docs from corpus lines) */
    RCRE_ArenaTemp m_bm = rcre_arena_temp_begin(arena);
    const size_t max_docs = 10000;
    RCRE_Slice* docs = (RCRE_Slice*)rcre_arena_alloc(arena, max_docs * sizeof(RCRE_Slice), 64);
    size_t num_docs = 0;
    RCRE_Slice cur = corpus;
    RCRE_Slice line;
    while (num_docs < max_docs && rcre_split_line_next(&cur, &line)) {
        if (line.len > 0) docs[num_docs++] = line;
    }

    RCRE_BM25Index bm25;
    t0 = nlp_now_sec();
    (void)rcre_bm25_build(&bm25, docs, num_docs, nullptr, arena, scratch);
    double dt_bm_build = nlp_now_sec() - t0;

    RCRE_SearchHit hits[10];
    size_t nh = 0;
    const int q_iters = 500;
    t0 = nlp_now_sec();
    for (int q = 0; q < q_iters; ++q) {
        (void)rcre_bm25_search(&bm25, RCRE_STR("sparse simd neon transformer"), 10, hits, &nh, scratch);
        sink += nh;
    }
    double dt_bm_q = nlp_now_sec() - t0;
    printf("BM25 Index (10k docs):          Build %6.2f ms | Top-10 Query %6.2f us/q\n",
           dt_bm_build * 1e3, (dt_bm_q * 1e6) / (double)q_iters);

    /* 5. TF-IDF -> RCRE_CSRMatrix (10k docs) */
    RCRE_TfidfVectorizer tfidf;
    RCRE_CSRMatrix csr;
    t0 = nlp_now_sec();
    (void)rcre_tfidf_fit(&tfidf, docs, num_docs, nullptr, arena, scratch);
    (void)rcre_tfidf_transform(&tfidf, docs, num_docs, arena, scratch, &csr);
    double dt_tfidf = nlp_now_sec() - t0;
    printf("TF-IDF -> CSR (10k docs):       %6.2f ms (%6.1f k docs/s, nnz=%zu)\n",
           dt_tfidf * 1e3, ((double)num_docs / 1e3) / dt_tfidf, csr.nnz);

    /* 6. Sparse HALS NMF (10k docs x 32 terms, K=8, 20 iters) */
    {
        RCRE_ArenaTemp m_p2 = rcre_arena_temp_begin(arena);
        RCRE_NmfOpts nopts = rcre_nmf_default_opts(8);
        nopts.max_iter = 20;
        RCRE_NmfModel nmf;
        t0 = nlp_now_sec();
        (void)rcre_nmf_fit(&nmf, &csr, &nopts, arena, scratch);
        double dt_nmf = nlp_now_sec() - t0;
        printf("Sparse HALS NMF (10k, K=8):     %6.2f ms (%zu iters, err=%.2f, 0 mallocs)\n",
               dt_nmf * 1e3, nmf.iters_run, nmf.recon_error);
        rcre_arena_temp_end(m_p2);
    }

    /* 7. Collapsed Gibbs LDA (2k docs, K=8, 20 sweeps) */
    {
        RCRE_ArenaTemp m_p2 = rcre_arena_temp_begin(arena);
        size_t lda_n = (num_docs < 2000) ? num_docs : 2000;
        RCRE_LdaOpts lopts = rcre_lda_default_opts(8);
        lopts.num_iters = 20;
        RCRE_LdaModel lda;
        t0 = nlp_now_sec();
        (void)rcre_lda_fit(&lda, docs, lda_n, &lopts, arena, scratch);
        double dt_lda = nlp_now_sec() - t0;
        printf("Collapsed Gibbs LDA (2k, K=8):  %6.2f ms (20 sweeps, logPPL=%.2f)\n",
               dt_lda * 1e3, lda.log_perplexity);
        rcre_arena_temp_end(m_p2);
    }

    /* 8. Native Sparse Multinomial NB & Softmax SGD (10k docs, 4 classes) */
    {
        RCRE_ArenaTemp m_p2 = rcre_arena_temp_begin(arena);
        uint32_t* y = (uint32_t*)rcre_arena_alloc(arena, num_docs * sizeof(uint32_t), 64);
        uint32_t* preds = (uint32_t*)rcre_arena_alloc(arena, num_docs * sizeof(uint32_t), 64);
        for (size_t d = 0; d < num_docs; ++d) y[d] = (uint32_t)(d & 3U);

        RCRE_NaiveBayesModel nb;
        t0 = nlp_now_sec();
        (void)rcre_nlp_nb_fit(&nb, &csr, y, 4, 1.0f, false, arena, scratch);
        double dt_nb_fit = nlp_now_sec() - t0;
        t0 = nlp_now_sec();
        (void)rcre_nlp_nb_predict(&nb, &csr, preds, nullptr);
        double dt_nb_pred = nlp_now_sec() - t0;
        sink += preds[0];
        printf("Sparse Multinomial NB (10k):    Fit %5.2f ms | Predict %5.2f ms (%.1fM docs/s)\n",
               dt_nb_fit * 1e3, dt_nb_pred * 1e3, ((double)num_docs / 1e6) / dt_nb_pred);

        RCRE_SparseClfOpts sopts = rcre_sparse_clf_default_opts();
        sopts.epochs = 10;
        RCRE_SparseLinearClf sclf;
        t0 = nlp_now_sec();
        (void)rcre_nlp_sparse_clf_fit(&sclf, &csr, y, 4, &sopts, arena, scratch);
        double dt_sgd = nlp_now_sec() - t0;
        printf("Sparse Softmax SGD (10k, 10ep): %6.2f ms (%.1f k docs/s/ep)\n",
               dt_sgd * 1e3, ((double)num_docs * 10.0 / 1e3) / dt_sgd);
        rcre_arena_temp_end(m_p2);
    }

    /* 9. LexRank + MMR Extractive Summarizer (64 sentences -> top 3) */
    {
        RCRE_ArenaTemp m_p2 = rcre_arena_temp_begin(arena);
        RCRE_StrBuilder art = rcre_sb_init(arena, 8192);
        for (size_t i = 0; i < 64 && i < num_docs; ++i) {
            (void)rcre_sb_append_slice(&art, docs[i]);
            (void)rcre_sb_append_cstr(&art, ". ");
        }
        RCRE_Slice art_slice = rcre_sb_to_slice(&art);
        RCRE_SummarySent ssum[3];
        size_t nsum = 0;
        const int sum_iters = 50;
        t0 = nlp_now_sec();
        for (int r = 0; r < sum_iters; ++r) {
            (void)rcre_nlp_summarize_extractive(art_slice, 3, nullptr, ssum, &nsum, scratch);
            sink += nsum;
        }
        double dt_sum = (nlp_now_sec() - t0) / (double)sum_iters;
        printf("LexRank+MMR Summarize (64 snt): %6.2f ms / doc (zero-copy slices)\n", dt_sum * 1e3);
        rcre_arena_temp_end(m_p2);
    }

    rcre_arena_temp_end(m_bm);

    /* 10. SimHash64 Dataset Fingerprinting (10k docs) */
    t0 = nlp_now_sec();
    uint64_t xor_fp = 0;
    for (size_t d = 0; d < num_docs; ++d) {
        xor_fp ^= rcre_nlp_simhash64(docs[d], nullptr);
    }
    double dt_sh = nlp_now_sec() - t0;
    sink += (size_t)xor_fp;
    printf("SimHash64 Fingerprint (10k):    %6.2f ms (%6.1f k docs/s, 0 allocs)\n",
           dt_sh * 1e3, ((double)num_docs / 1e3) / dt_sh);

    (void)sink;
    rcre_arena_destroy(scratch);
    rcre_arena_destroy(arena);
    return 0;
}
