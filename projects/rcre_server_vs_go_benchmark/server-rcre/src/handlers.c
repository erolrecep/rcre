#include "handlers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* Fixture Loading (Startup Only — Persistent RCRE_Arena)                    */
/* ========================================================================= */

static uint8_t* read_entire_file_arena(const char* path, RCRE_Arena* arena, size_t* out_len) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return nullptr;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return nullptr;
    }
    long sz = ftell(fp);
    if (sz < 0) {
        fclose(fp);
        return nullptr;
    }
    rewind(fp);
    size_t len = (size_t)sz;
    uint8_t* buf = (uint8_t*)rcre_arena_alloc(arena, len + 1U, 64);
    if (!buf) {
        fclose(fp);
        return nullptr;
    }
    if (len > 0 && fread(buf, 1, len, fp) != len) {
        fclose(fp);
        return nullptr;
    }
    buf[len] = 0;
    fclose(fp);
    *out_len = len;
    return buf;
}

static uint32_t read_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float read_f32_le(const uint8_t* p) {
    uint32_t u = read_u32_le(p);
    float f = 0.0f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static int load_bm25_corpus(AppState* app, const char* corpus_path) {
    size_t raw_len = 0;
    uint8_t* raw = read_entire_file_arena(corpus_path, app->persist_arena, &raw_len);
    if (!raw) return RCRE_ERR_IO;

    size_t num_docs = 0;
    for (size_t i = 0; i < raw_len; ++i) {
        if (raw[i] == '\n') num_docs++;
    }
    if (raw_len > 0 && raw[raw_len - 1] != '\n') num_docs++;

    RCRE_Arena* tmp = rcre_arena_create(32U * 1024U * 1024U);
    if (!tmp) return RCRE_ERR_OUT_OF_MEMORY;

    RCRE_Slice* docs = (RCRE_Slice*)rcre_arena_alloc(tmp, num_docs * sizeof(RCRE_Slice), 64);
    if (!docs) {
        rcre_arena_destroy(tmp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    size_t d = 0;
    size_t start = 0;
    for (size_t i = 0; i < raw_len; ++i) {
        if (raw[i] == '\n') {
            docs[d++] = rcre_slice_from_parts(raw + start, i - start);
            start = i + 1U;
        }
    }
    if (start < raw_len && d < num_docs) {
        docs[d++] = rcre_slice_from_parts(raw + start, raw_len - start);
    }

    RCRE_BM25Opts opts = rcre_bm25_default_opts();
    int rc = rcre_bm25_build(&app->bm25, docs, d, &opts, app->persist_arena, tmp);
    rcre_arena_destroy(tmp);
    return rc;
}

static int load_gbdt_model(AppState* app, const char* model_path) {
    RCRE_Arena* tmp = rcre_arena_create(8U * 1024U * 1024U);
    if (!tmp) return RCRE_ERR_OUT_OF_MEMORY;

    size_t raw_len = 0;
    uint8_t* raw = read_entire_file_arena(model_path, tmp, &raw_len);
    if (!raw || raw_len < 24U || memcmp(raw, "RGBT", 4) != 0) {
        rcre_arena_destroy(tmp);
        return RCRE_ERR_IO;
    }

    uint32_t num_trees    = read_u32_le(raw + 4);
    uint32_t num_nodes    = read_u32_le(raw + 8);
    uint32_t num_features = read_u32_le(raw + 12);
    float    base_score   = read_f32_le(raw + 16);
    uint32_t is_binary    = read_u32_le(raw + 20);

    size_t expected = 24U + (size_t)num_trees * 4U + (size_t)num_nodes * 24U;
    if (raw_len < expected) {
        rcre_arena_destroy(tmp);
        return RCRE_ERR_IO;
    }

    size_t* roots = (size_t*)rcre_arena_alloc(app->persist_arena, (size_t)num_trees * sizeof(size_t), 64);
    RCRE_TreeNode* nodes = (RCRE_TreeNode*)rcre_arena_alloc(
        app->persist_arena, (size_t)num_nodes * sizeof(RCRE_TreeNode), 64);
    if (!roots || !nodes) {
        rcre_arena_destroy(tmp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    size_t off = 24U;
    for (uint32_t t = 0; t < num_trees; ++t) {
        roots[t] = (size_t)read_u32_le(raw + off);
        off += 4U;
    }
    for (uint32_t i = 0; i < num_nodes; ++i) {
        nodes[i].feature_idx = read_u32_le(raw + off);
        nodes[i].threshold   = read_f32_le(raw + off + 4);
        nodes[i].left_child  = read_u32_le(raw + off + 8);
        nodes[i].right_child = read_u32_le(raw + off + 12);
        nodes[i].leaf_value  = read_f32_le(raw + off + 16);
        nodes[i].is_leaf     = (read_u32_le(raw + off + 20) != 0);
        off += 24U;
    }

    app->gbdt.nodes                    = nodes;
    app->gbdt.tree_roots               = roots;
    app->gbdt.num_trees                = num_trees;
    app->gbdt.num_nodes                = num_nodes;
    app->gbdt.base_score               = base_score;
    app->gbdt.is_binary_classification = (is_binary != 0);
    app->gbdt_num_features             = num_features;

    rcre_arena_destroy(tmp);
    return RCRE_OK;
}

int app_state_init(AppState* app, const char* data_dir) {
    memset(app, 0, sizeof(*app));
    rcre_hal_init();

    app->persist_arena = rcre_arena_create(64U * 1024U * 1024U);
    if (!app->persist_arena) return RCRE_ERR_OUT_OF_MEMORY;

    RCRE_STRBUF_STACK(path_buf, 1024);
    (void)rcre_strbuf_fmt(&path_buf, "%s/corpus_10k.txt", data_dir);
    int rc = load_bm25_corpus(app, rcre_strbuf_cstr(&path_buf));
    if (rc != RCRE_OK) {
        app_state_destroy(app);
        return rc;
    }

    rcre_strbuf_reset(&path_buf);
    (void)rcre_strbuf_fmt(&path_buf, "%s/gbdt_model.bin", data_dir);
    rc = load_gbdt_model(app, rcre_strbuf_cstr(&path_buf));
    if (rc != RCRE_OK) {
        app_state_destroy(app);
        return rc;
    }

    return RCRE_OK;
}

void app_state_destroy(AppState* app) {
    if (app && app->persist_arena) {
        rcre_arena_destroy(app->persist_arena);
        app->persist_arena = nullptr;
    }
}

bool worker_bm25_scratch_init(WorkerBM25Scratch* sc, size_t num_docs) {
    memset(sc, 0, sizeof(*sc));
    if (num_docs == 0) return true;
    sc->scores  = (float*)calloc(num_docs, sizeof(float));
    sc->touched = (uint32_t*)malloc(num_docs * sizeof(uint32_t));
    sc->hits    = (RCRE_SearchHit*)malloc(num_docs * sizeof(RCRE_SearchHit));
    if (!sc->scores || !sc->touched || !sc->hits) {
        worker_bm25_scratch_destroy(sc);
        return false;
    }
    sc->num_docs = num_docs;
    return true;
}

void worker_bm25_scratch_destroy(WorkerBM25Scratch* sc) {
    free(sc->scores);
    free(sc->touched);
    free(sc->hits);
    memset(sc, 0, sizeof(*sc));
}

/* ========================================================================= */
/* Zero-Malloc HTTP Response Framing Helpers (Zero strlen)                   */
/* ========================================================================= */

static size_t write_http_response(
    RCRE_StrBuf* out,
    RCRE_Slice status_and_ct,
    const char* date_hdr_29,
    bool keep_alive,
    const uint8_t* body,
    size_t body_len
) {
    rcre_strbuf_reset(out);
    (void)rcre_strbuf_append(out, status_and_ct);
    (void)rcre_strbuf_append_u64(out, (uint64_t)body_len);
    (void)rcre_strbuf_append(out, RCRE_STR("\r\nDate: "));
    (void)rcre_strbuf_append(out, rcre_slice_from_parts((const uint8_t*)date_hdr_29, 29));
    if (keep_alive) {
        (void)rcre_strbuf_append(out, RCRE_STR("\r\nConnection: keep-alive\r\n\r\n"));
    } else {
        (void)rcre_strbuf_append(out, RCRE_STR("\r\nConnection: close\r\n\r\n"));
    }
    if (body_len > 0) {
        (void)rcre_strbuf_append(out, rcre_slice_from_parts(body, body_len));
    }
    return out->len;
}

#define HDR_200_JSON RCRE_STR("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ")
#define HDR_200_TEXT RCRE_STR("HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: ")
#define HDR_400_JSON RCRE_STR("HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: ")
#define HDR_404_JSON RCRE_STR("HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: ")
#define HDR_405_JSON RCRE_STR("HTTP/1.1 405 Method Not Allowed\r\nContent-Type: application/json\r\nContent-Length: ")
#define HDR_500_JSON RCRE_STR("HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: ")

size_t app_write_bad_request(const char* date_hdr_29, RCRE_StrBuf* out_resp) {
    RCRE_Slice body = RCRE_STR("{\"error\":\"bad_request\"}");
    return write_http_response(out_resp, HDR_400_JSON, date_hdr_29, false, body.data, body.len);
}

/* ========================================================================= */
/* W2: POST /api/v1/validate (NEON UTF-8 + Branchless Codepoint Count)       */
/* ========================================================================= */

static inline uint64_t fnv1a_64(const uint8_t* data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static inline size_t count_valid_utf8_codepoints_simd(const uint8_t* data, size_t len) {
    /* On already-validated UTF-8, a byte starts a codepoint iff (int8_t)b >= -64 */
    size_t cp = 0;
    const int8_t* s = (const int8_t*)data;
    for (size_t i = 0; i < len; ++i) {
        cp += (size_t)(s[i] >= -64);
    }
    return cp;
}

static inline size_t count_lines_fast(const uint8_t* data, size_t len) {
    if (len == 0) return 0;
    size_t lines = 0;
    const uint8_t* p = data;
    const uint8_t* end = data + len;
    while (p < end) {
        const uint8_t* nl = (const uint8_t*)memchr(p, '\n', (size_t)(end - p));
        if (!nl) break;
        lines++;
        p = nl + 1;
    }
    if (data[len - 1] != '\n') lines++;
    return lines;
}

static void append_hex16_lower(RCRE_StrBuf* sb, uint64_t v) {
    static const uint8_t hex[17] = "0123456789abcdef";
    uint8_t tmp[16];
    for (int i = 15; i >= 0; --i) {
        tmp[i] = hex[v & 0xFULL];
        v >>= 4;
    }
    (void)rcre_strbuf_append(sb, rcre_slice_from_parts(tmp, 16));
}

static size_t handle_validate(
    const HttpRequest* req,
    const char* date_hdr_29,
    RCRE_StrBuf* out_resp
) {
    if (req->method != HTTP_METHOD_POST) {
        RCRE_Slice b = RCRE_STR("{\"error\":\"method_not_allowed\"}");
        return write_http_response(out_resp, HDR_405_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
    }

    RCRE_Slice body = req->body;
    bool valid = rcre_utf8_validate(body, nullptr);
    size_t cp = valid ? count_valid_utf8_codepoints_simd(body.data, body.len) : 0U;
    size_t lines = count_lines_fast(body.data, body.len);
    uint64_t h = fnv1a_64(body.data, body.len);

    RCRE_STRBUF_STACK(jbuf, 256);
    (void)rcre_strbuf_append(&jbuf, valid ? RCRE_STR("{\"valid\":true,\"bytes\":")
                                          : RCRE_STR("{\"valid\":false,\"bytes\":"));
    (void)rcre_strbuf_append_u64(&jbuf, (uint64_t)body.len);
    (void)rcre_strbuf_append(&jbuf, RCRE_STR(",\"codepoints\":"));
    (void)rcre_strbuf_append_u64(&jbuf, (uint64_t)cp);
    (void)rcre_strbuf_append(&jbuf, RCRE_STR(",\"lines\":"));
    (void)rcre_strbuf_append_u64(&jbuf, (uint64_t)lines);
    (void)rcre_strbuf_append(&jbuf, RCRE_STR(",\"fnv64\":\""));
    append_hex16_lower(&jbuf, h);
    (void)rcre_strbuf_append(&jbuf, RCRE_STR("\"}"));

    return write_http_response(out_resp, HDR_200_JSON, date_hdr_29, req->keep_alive, jbuf.data, jbuf.len);
}

/* ========================================================================= */
/* W3: GET /api/v1/search?q=...&k=... (O(touched) Reset + Partial Top-K)     */
/* ========================================================================= */

static inline int hex_nibble(uint8_t c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static RCRE_Slice url_decode_into(RCRE_Slice src, uint8_t* dst, size_t dst_cap) {
    size_t out = 0;
    for (size_t i = 0; i < src.len && out < dst_cap; ++i) {
        uint8_t c = src.data[i];
        if (c == '+') {
            dst[out++] = ' ';
        } else if (c == '%' && i + 2 < src.len) {
            int hi = hex_nibble(src.data[i + 1]);
            int lo = hex_nibble(src.data[i + 2]);
            if (hi >= 0 && lo >= 0) {
                dst[out++] = (uint8_t)((hi << 4) | lo);
                i += 2;
            } else {
                dst[out++] = c;
            }
        } else {
            dst[out++] = c;
        }
    }
    return rcre_slice_from_parts(dst, out);
}

static void append_json_escaped(RCRE_StrBuf* sb, RCRE_Slice s) {
    static const uint8_t hex[17] = "0123456789abcdef";
    (void)rcre_strbuf_append_byte(sb, '"');
    for (size_t i = 0; i < s.len; ++i) {
        uint8_t c = s.data[i];
        switch (c) {
            case '"':  (void)rcre_strbuf_append(sb, RCRE_STR("\\\"")); break;
            case '\\': (void)rcre_strbuf_append(sb, RCRE_STR("\\\\")); break;
            case '\n': (void)rcre_strbuf_append(sb, RCRE_STR("\\n"));  break;
            case '\r': (void)rcre_strbuf_append(sb, RCRE_STR("\\r"));  break;
            case '\t': (void)rcre_strbuf_append(sb, RCRE_STR("\\t"));  break;
            default:
                if (c < 0x20U) {
                    uint8_t esc[6] = { '\\', 'u', '0', '0', hex[c >> 4], hex[c & 0xFU] };
                    (void)rcre_strbuf_append(sb, rcre_slice_from_parts(esc, 6));
                } else {
                    (void)rcre_strbuf_append_byte(sb, c);
                }
                break;
        }
    }
    (void)rcre_strbuf_append_byte(sb, '"');
}

/*
 * Persistent-scratch BM25 search over RCRE's CSR RCRE_BM25Index:
 *   - Uses rcre_nlp_word_iter_next + rcre_hal.ascii_to_lower + rcre_strpool_find
 *   - O(touched) score array cleanup (no 40 KB memset per query)
 *   - O(touched * top_k) partial selection (exact algorithmic parity with go-tuned)
 */
static size_t bm25_search_fast(
    const RCRE_BM25Index* index,
    RCRE_Slice query,
    size_t top_k,
    WorkerBM25Scratch* sc,
    RCRE_SearchHit* out_hits
) {
    if (index->num_docs == 0 || index->num_terms == 0 || top_k == 0 || query.len == 0) {
        return 0;
    }

    float* scores = sc->scores;
    uint32_t* touched = sc->touched;
    size_t num_touched = 0;

    const float k1    = index->k1;
    const float k1p1  = k1 + 1.0f;
    const float b     = index->b;
    const float avgdl = (index->avg_doc_len > 0.0f) ? index->avg_doc_len : 1.0f;
    const float k1_1mb = k1 * (1.0f - b);
    const float k1_b_inv_avgdl = k1 * (b / avgdl);

    const uint32_t* doc_lens = index->doc_lens;
    const uint32_t* postings_doc = index->postings_doc;
    const uint32_t* postings_tf  = index->postings_tf;

    uint8_t low_buf[256];
    RCRE_WordIter it = rcre_nlp_word_iter_init(query, &index->tok_opts);
    RCRE_Slice tok;
    while (rcre_nlp_word_iter_next(&it, &tok)) {
        RCRE_Slice norm = tok;
        if (index->lowercase && tok.len <= sizeof(low_buf)) {
            rcre_hal.ascii_to_lower(tok.data, low_buf, tok.len);
            norm = rcre_slice_from_parts(low_buf, tok.len);
        }
        RCRE_StrId tid = rcre_strpool_find(&index->vocab, norm);
        if (tid == RCRE_STR_ID_INVALID) continue;

        float idf_k1p1 = index->term_idf[tid] * k1p1;
        uint32_t p0 = index->term_offsets[tid];
        uint32_t p1 = index->term_offsets[tid + 1U];
        for (uint32_t p = p0; p < p1; ++p) {
            uint32_t d = postings_doc[p];
            float tf = (float)postings_tf[p];
            float dl = (float)doc_lens[d];
            float denom = tf + k1_1mb + k1_b_inv_avgdl * dl;
            float contrib = (idf_k1p1 * tf) / denom;
            float prev = scores[d];
            if (prev == 0.0f && contrib > 0.0f) {
                touched[num_touched++] = d;
            }
            scores[d] = prev + contrib;
        }
    }

    if (num_touched == 0) {
        return 0;
    }

    RCRE_SearchHit* hits = sc->hits;
    for (size_t i = 0; i < num_touched; ++i) {
        uint32_t d = touched[i];
        hits[i].doc_id = d;
        hits[i].score  = scores[d];
        scores[d] = 0.0f; /* O(touched) cleanup */
    }

    size_t k = (num_touched < top_k) ? num_touched : top_k;
    for (size_t i = 0; i < k; ++i) {
        size_t best = i;
        RCRE_SearchHit bh = hits[i];
        for (size_t j = i + 1U; j < num_touched; ++j) {
            RCRE_SearchHit h = hits[j];
            if (h.score > bh.score || (h.score == bh.score && h.doc_id < bh.doc_id)) {
                best = j;
                bh = h;
            }
        }
        if (best != i) {
            hits[best] = hits[i];
            hits[i] = bh;
        }
        out_hits[i] = bh;
    }
    return k;
}

static size_t handle_search(
    const AppState* app,
    const HttpRequest* req,
    const char* date_hdr_29,
    WorkerBM25Scratch* bm25_sc,
    RCRE_StrBuf* out_resp
) {
    if (req->method != HTTP_METHOD_GET) {
        RCRE_Slice b = RCRE_STR("{\"error\":\"method_not_allowed\"}");
        return write_http_response(out_resp, HDR_405_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
    }

    RCRE_Slice raw_q = rcre_slice_empty();
    size_t top_k = 10U;

    RCRE_SplitIter it = rcre_split_byte_init(req->query, '&');
    RCRE_Slice pair;
    while (rcre_split_next(&it, &pair)) {
        if (pair.len < 2) continue;
        const uint8_t* eq = (const uint8_t*)memchr(pair.data, '=', pair.len);
        if (!eq) continue;
        RCRE_Slice key = rcre_slice_from_parts(pair.data, (size_t)(eq - pair.data));
        RCRE_Slice val = rcre_slice_from_parts(eq + 1, pair.len - key.len - 1U);
        if (key.len == 1 && key.data[0] == 'q') {
            raw_q = val;
        } else if (key.len == 1 && key.data[0] == 'k') {
            size_t parsed = 0;
            bool ok = (val.len > 0 && val.len <= 4);
            for (size_t i = 0; i < val.len && ok; ++i) {
                if (val.data[i] < '0' || val.data[i] > '9') {
                    ok = false;
                } else {
                    parsed = parsed * 10U + (size_t)(val.data[i] - '0');
                }
            }
            if (ok) top_k = parsed;
        }
    }

    if (top_k < 1U) top_k = 1U;
    if (top_k > 50U) top_k = 50U;

    uint8_t q_buf[512];
    RCRE_Slice q_decoded = url_decode_into(raw_q, q_buf, sizeof(q_buf));

    RCRE_SearchHit hits[50];
    size_t hit_count = bm25_search_fast(&app->bm25, q_decoded, top_k, bm25_sc, hits);

    RCRE_STRBUF_STACK(jbuf, 4096);
    (void)rcre_strbuf_append(&jbuf, RCRE_STR("{\"query\":"));
    append_json_escaped(&jbuf, q_decoded);
    (void)rcre_strbuf_append(&jbuf, RCRE_STR(",\"count\":"));
    (void)rcre_strbuf_append_u64(&jbuf, (uint64_t)hit_count);
    (void)rcre_strbuf_append(&jbuf, RCRE_STR(",\"hits\":["));
    for (size_t i = 0; i < hit_count; ++i) {
        if (i > 0) (void)rcre_strbuf_append_byte(&jbuf, ',');
        (void)rcre_strbuf_append(&jbuf, RCRE_STR("{\"doc_id\":"));
        (void)rcre_strbuf_append_u64(&jbuf, (uint64_t)hits[i].doc_id);
        (void)rcre_strbuf_append(&jbuf, RCRE_STR(",\"score\":"));
        (void)rcre_strbuf_append_f64(&jbuf, (double)hits[i].score, 4);
        (void)rcre_strbuf_append_byte(&jbuf, '}');
    }
    (void)rcre_strbuf_append(&jbuf, RCRE_STR("]}"));

    return write_http_response(out_resp, HDR_200_JSON, date_hdr_29, req->keep_alive, jbuf.data, jbuf.len);
}

/* ========================================================================= */
/* W4: POST /api/v1/infer (GBDT Batch Prediction)                            */
/* ========================================================================= */

static inline bool is_ws_or_comma(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ',';
}

static bool parse_fast_f32(const uint8_t* s, size_t len, float* out) {
    if (len == 0) return false;
    size_t i = 0;
    bool neg = false;
    if (s[i] == '-') {
        neg = true;
        i++;
    } else if (s[i] == '+') {
        i++;
    }
    if (i >= len) return false;

    uint64_t int_part = 0;
    bool has_digit = false;
    while (i < len && s[i] >= '0' && s[i] <= '9') {
        int_part = int_part * 10ULL + (uint64_t)(s[i] - '0');
        has_digit = true;
        i++;
    }

    uint64_t frac_part = 0;
    uint32_t frac_digits = 0;
    if (i < len && s[i] == '.') {
        i++;
        while (i < len && s[i] >= '0' && s[i] <= '9') {
            if (frac_digits < 9U) {
                frac_part = frac_part * 10ULL + (uint64_t)(s[i] - '0');
                frac_digits++;
            }
            has_digit = true;
            i++;
        }
    }

    if (!has_digit) return false;
    if (i < len) {
        double d = 0.0;
        if (!rcre_slice_parse_f64(rcre_slice_from_parts(s, len), &d)) return false;
        *out = (float)d;
        return true;
    }

    static const double pow10_inv[10] = {
        1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9
    };
    double val = (double)int_part + (double)frac_part * pow10_inv[frac_digits];
    if (neg) val = -val;
    *out = (float)val;
    return true;
}

static size_t handle_infer(
    const AppState* app,
    const HttpRequest* req,
    const char* date_hdr_29,
    RCRE_Arena* scratch,
    RCRE_StrBuf* out_resp
) {
    if (req->method != HTTP_METHOD_POST) {
        RCRE_Slice b = RCRE_STR("{\"error\":\"method_not_allowed\"}");
        return write_http_response(out_resp, HDR_405_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
    }

    size_t rows_key = rcre_slice_find(req->body, RCRE_STR("\"rows\""));
    if (rows_key == RCRE_NPOS) {
        RCRE_Slice b = RCRE_STR("{\"error\":\"bad_request\"}");
        return write_http_response(out_resp, HDR_400_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
    }

    const uint8_t* p = req->body.data;
    size_t n = req->body.len;
    size_t i = rows_key + 6U;
    while (i < n && p[i] != '[') i++;
    if (i >= n) {
        RCRE_Slice b = RCRE_STR("{\"error\":\"bad_request\"}");
        return write_http_response(out_resp, HDR_400_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
    }
    i++;

    constexpr size_t MAX_ROWS = 1024U;
    size_t D = app->gbdt_num_features;
    float* flat = (float*)rcre_arena_alloc(scratch, MAX_ROWS * D * sizeof(float), 64);
    float* preds_buf = (float*)rcre_arena_alloc(scratch, MAX_ROWS * sizeof(float), 64);
    if (!flat || !preds_buf) {
        RCRE_Slice b = RCRE_STR("{\"error\":\"oom\"}");
        return write_http_response(out_resp, HDR_500_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
    }

    size_t n_rows = 0;
    while (i < n) {
        while (i < n && is_ws_or_comma(p[i])) i++;
        if (i >= n || p[i] == ']') break;
        if (p[i] != '[' || n_rows >= MAX_ROWS) {
            RCRE_Slice b = RCRE_STR("{\"error\":\"bad_request\"}");
            return write_http_response(out_resp, HDR_400_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
        }
        i++;
        size_t cols = 0;
        float* row_out = flat + n_rows * D;
        while (i < n) {
            while (i < n && is_ws_or_comma(p[i])) i++;
            if (i >= n) {
                RCRE_Slice b = RCRE_STR("{\"error\":\"bad_request\"}");
                return write_http_response(out_resp, HDR_400_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
            }
            if (p[i] == ']') {
                i++;
                break;
            }
            size_t start = i;
            while (i < n && p[i] != ',' && p[i] != ']' && !is_ws_or_comma(p[i])) i++;
            if (cols >= D) {
                RCRE_Slice b = RCRE_STR("{\"error\":\"bad_request\"}");
                return write_http_response(out_resp, HDR_400_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
            }
            float val = 0.0f;
            if (!parse_fast_f32(p + start, i - start, &val)) {
                RCRE_Slice b = RCRE_STR("{\"error\":\"bad_request\"}");
                return write_http_response(out_resp, HDR_400_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
            }
            row_out[cols++] = val;
        }
        if (cols != D) {
            RCRE_Slice b = RCRE_STR("{\"error\":\"bad_request\"}");
            return write_http_response(out_resp, HDR_400_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
        }
        n_rows++;
    }

    if (n_rows == 0) {
        RCRE_Slice b = RCRE_STR("{\"error\":\"bad_request\"}");
        return write_http_response(out_resp, HDR_400_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
    }

    RCRE_Tensor X = {0};
    X.data        = flat;
    X.ndim        = 2;
    X.shape[0]    = n_rows;
    X.shape[1]    = D;
    X.strides[0]  = (ptrdiff_t)D;
    X.strides[1]  = 1;
    X.dtype       = RCRE_DTYPE_F32;
    X.layout      = RCRE_LAYOUT_TABULAR;
    X.is_view     = true;

    RCRE_Tensor preds = {0};
    preds.data       = preds_buf;
    preds.ndim       = 1;
    preds.shape[0]   = n_rows;
    preds.strides[0] = 1;
    preds.dtype      = RCRE_DTYPE_F32;
    preds.is_view    = true;

    if (rcre_ml_gbdt_predict(&app->gbdt, &X, &preds) != RCRE_OK) {
        RCRE_Slice b = RCRE_STR("{\"error\":\"infer_failed\"}");
        return write_http_response(out_resp, HDR_500_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
    }

    double sum = 0.0;
    for (size_t r = 0; r < n_rows; ++r) {
        sum += (double)preds_buf[r];
    }
    double mean = sum / (double)n_rows;

    uint8_t* j_raw = (uint8_t*)rcre_arena_alloc(scratch, 16384U, 64);
    RCRE_StrBuf jbuf = rcre_strbuf_init(j_raw, 16384U);
    (void)rcre_strbuf_append(&jbuf, RCRE_STR("{\"count\":"));
    (void)rcre_strbuf_append_u64(&jbuf, (uint64_t)n_rows);
    (void)rcre_strbuf_append(&jbuf, RCRE_STR(",\"mean\":"));
    (void)rcre_strbuf_append_f64(&jbuf, mean, 6);
    (void)rcre_strbuf_append(&jbuf, RCRE_STR(",\"predictions\":["));
    for (size_t r = 0; r < n_rows; ++r) {
        if (r > 0) (void)rcre_strbuf_append_byte(&jbuf, ',');
        (void)rcre_strbuf_append_f64(&jbuf, (double)preds_buf[r], 4);
    }
    (void)rcre_strbuf_append(&jbuf, RCRE_STR("]}"));

    return write_http_response(out_resp, HDR_200_JSON, date_hdr_29, req->keep_alive, jbuf.data, jbuf.len);
}

/* ========================================================================= */
/* Top-Level Route Dispatcher                                                */
/* ========================================================================= */

size_t app_handle_request(
    const AppState* app,
    const HttpRequest* req,
    const char* date_hdr_29,
    RCRE_Arena* scratch,
    WorkerBM25Scratch* bm25_sc,
    RCRE_StrBuf* out_resp
) {
    RCRE_Slice p = req->path;

    /* W1: /health (fast path) */
    if (p.len == 7U && memcmp(p.data, "/health", 7) == 0) {
        if (req->method != HTTP_METHOD_GET) {
            RCRE_Slice b = RCRE_STR("{\"error\":\"method_not_allowed\"}");
            return write_http_response(out_resp, HDR_405_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
        }
        static const uint8_t ok_body[2] = { 'O', 'K' };
        return write_http_response(out_resp, HDR_200_TEXT, date_hdr_29, req->keep_alive, ok_body, 2U);
    }

    /* W2: /api/v1/validate */
    if (p.len == 16U && memcmp(p.data, "/api/v1/validate", 16) == 0) {
        return handle_validate(req, date_hdr_29, out_resp);
    }

    /* W3: /api/v1/search */
    if (p.len == 14U && memcmp(p.data, "/api/v1/search", 14) == 0) {
        return handle_search(app, req, date_hdr_29, bm25_sc, out_resp);
    }

    /* W4: /api/v1/infer */
    if (p.len == 13U && memcmp(p.data, "/api/v1/infer", 13) == 0) {
        return handle_infer(app, req, date_hdr_29, scratch, out_resp);
    }

    RCRE_Slice b = RCRE_STR("{\"error\":\"not_found\"}");
    return write_http_response(out_resp, HDR_404_JSON, date_hdr_29, req->keep_alive, b.data, b.len);
}
