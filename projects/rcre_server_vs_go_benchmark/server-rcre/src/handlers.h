#ifndef RCRE_BENCH_HANDLERS_H
#define RCRE_BENCH_HANDLERS_H

#include "rcre.h"
#include "rcre_str.h"
#include "rcre_nlp.h"
#include "http_parser.h"
#include <stddef.h>
#include <stdint.h>

typedef struct AppState {
    RCRE_Arena*    persist_arena;
    RCRE_BM25Index bm25;
    RCRE_GBDTModel gbdt;
    size_t         gbdt_num_features;
} AppState;

/* Per-worker reusable BM25 scratch (O(touched) reset, zero malloc, zero full memset) */
typedef struct WorkerBM25Scratch {
    float*          scores;
    uint32_t*       touched;
    RCRE_SearchHit* hits;
    size_t          num_docs;
} WorkerBM25Scratch;

RCRE_NODISCARD int app_state_init(AppState* app, const char* data_dir);
void app_state_destroy(AppState* app);

RCRE_NODISCARD bool worker_bm25_scratch_init(WorkerBM25Scratch* sc, size_t num_docs);
void worker_bm25_scratch_destroy(WorkerBM25Scratch* sc);

/**
 * @brief Handles a parsed HTTP request and writes the complete HTTP/1.1 wire
 *        response (headers + body) into out_resp (backed by worker buffer).
 *        Zero heap allocations in the hot path; ephemeral work uses scratch.
 * @return Total response bytes written into out_resp->data.
 */
size_t app_handle_request(
    const AppState* app,
    const HttpRequest* req,
    const char* date_hdr_29,
    RCRE_Arena* scratch,
    WorkerBM25Scratch* bm25_sc,
    RCRE_StrBuf* out_resp
);

/**
 * @brief Writes a minimal 400 Bad Request wire response (always Connection: close).
 */
size_t app_write_bad_request(const char* date_hdr_29, RCRE_StrBuf* out_resp);

#endif /* RCRE_BENCH_HANDLERS_H */
