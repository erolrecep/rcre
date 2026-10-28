/**
 * @file xgb_io.c
 * @brief CSV loading (header auto-detect, NaN/missing, label column) and
 *        binary model save/load.
 *
 * Model format "RCXG" v1 (little-endian, native floats):
 *   magic[4]  ver u32
 *   task u8   K u32   num_trees u32   num_nodes u32   dim u32
 *   eta f32   base_score f32   best_iteration i32
 *   base_scores f32[16]
 *   val_curve_len i32 + val_curve f32[len]
 *   roots u32[num_trees]
 *   nodes: { feature u32, threshold f32, left u32, right u32,
 *            leaf f32, flags u8 (bit0 is_leaf, bit1 default_left), pad[3] }
 */
#include "xgb_internal.h"

/* ========================================================================= */
/* CSV / whitespace table loading                                            */
/* ========================================================================= */

static int is_missing_token(const char* s) {
    if (s[0] == 0) return 1;
    if (!strcmp(s, "nan") || !strcmp(s, "NaN") || !strcmp(s, "NAN") ||
        !strcmp(s, "na")  || !strcmp(s, "NA")  || !strcmp(s, "null") ||
        !strcmp(s, "NULL") || !strcmp(s, "-"))
        return 1;
    return 0;
}

/* split one row into tokens (comma and/or whitespace separated);
 * returns token count, row is NUL-mangled */
static int row_tokens(char* row, char** toks, int maxtoks) {
    int n = 0;
    char* p = row;
    while (*p && n < maxtoks) {
        while (*p == ' ' || *p == '\t' || *p == ',') p++;
        if (!*p) break;
        toks[n++] = p;
        while (*p && *p != ' ' && *p != '\t' && *p != ',') p++;
        if (*p) { *p = 0; p++; }
    }
    return n;
}

static int parse_float_tok(const char* s, float* out) {
    if (is_missing_token(s)) { *out = NAN; return 1; }
    char* end = NULL;
    float v = strtof(s, &end);
    if (end == s || *end != 0) return 0;
    *out = v;
    return 1;
}

/* count tokens in a row without modifying it (comma/whitespace separated) */
static int count_row_tokens(const char* row, size_t len) {
    int n = 0, in_tok = 0;
    for (size_t i = 0; i < len; i++) {
        char c = row[i];
        if (c == ' ' || c == '\t' || c == ',' || c == '\r') in_tok = 0;
        else if (!in_tok) { n++; in_tok = 1; }
    }
    return n;
}

int xgb_csv_load(const char* path, float** X, float** y,
                 size_t* n, size_t* d, int label_col,
                 RCRE_Arena* arena) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return -1; }
    char* buf = (char*)rcre_arena_alloc(arena, (size_t)sz + 1, 16);
    if (!buf) { fclose(f); return -1; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = 0;

    char* toks[4096];

    /* ---- pass 1 (non-destructive): header detection + census ---- */
    int has_header = 0;
    size_t nrows = 0, ncols = 0;
    {
        char* p = buf;
        int first = 1;
        while (*p) {
            char* e = p;
            while (*e && *e != '\n') e++;
            size_t len = (size_t)(e - p);
            while (len > 0 && p[len - 1] == '\r') len--;
            int nt = count_row_tokens(p, len);
            if (nt > 0) {
                if (first) {
                    /* header? test first token */
                    size_t tl = 0;
                    while (tl < len && p[tl] != ' ' && p[tl] != '\t' && p[tl] != ',') tl++;
                    char tmp[64];
                    if (tl < sizeof tmp - 1) { memcpy(tmp, p, tl); tmp[tl] = 0; }
                    float v;
                    if (tl > 0 && !parse_float_tok(tmp, &v)) has_header = 1;
                }
                if (!(first && has_header)) {
                    if (nrows == 0) ncols = (size_t)nt;
                    if ((size_t)nt == ncols) nrows++;
                }
            }
            first = 0;
            p = (*e ? e + 1 : e);
        }
    }
    if (nrows == 0 || ncols == 0) return -1;
    size_t nd = ncols - (label_col >= 0 && label_col < (int)ncols ? 1 : 0);

    /* ---- pass 2: fill ---- */
    float* Xb = (float*)rcre_arena_alloc(arena, nrows * nd * sizeof(float), 16);
    float* yb = (label_col >= 0 && label_col < (int)ncols)
        ? (float*)rcre_arena_alloc(arena, nrows * sizeof(float), 16) : NULL;
    if (!Xb || (yb == NULL && label_col >= 0 && label_col < (int)ncols)) return -1;

    size_t nr = 0;
    char* p = buf;
    int first = 1;
    while (*p) {
        char* e = p;
        while (*e && *e != '\n') e++;
        int adv = (*e != 0);      /* newline present (buffer NUL means last row) */
        if (*e == '\n') *e = 0;
        int nt = row_tokens(p, toks, 4096);
        if (nt > 0) {
            if (first && has_header) { /* skip */ }
            else if ((size_t)nt == ncols) {
                int oi = 0, ok = 1;
                for (int c = 0; c < nt; c++) {
                    if (c == label_col) continue;
                    float v;
                    if (!parse_float_tok(toks[c], &v)) { ok = 0; break; }
                    Xb[nr * nd + oi++] = v;
                }
                if (ok && label_col >= 0 && label_col < nt) {
                    float v;
                    if (!parse_float_tok(toks[label_col], &v)) ok = 0;
                    else yb[nr] = v;
                }
                if (ok) nr++;
            }
        }
        first = 0;
        p = adv ? e + 1 : e;
    }

    *X = Xb;
    *y = yb;
    *n = nr;
    *d = nd;
    return nr > 0 ? 0 : -1;
}

/* ========================================================================= */
/* Model save / load                                                         */
/* ========================================================================= */

typedef struct {
    char magic[4];
    uint32_t ver;
    uint8_t task;
    uint32_t K, num_trees, num_nodes, dim;
    float eta, base_score;
    int32_t best_iteration;
    float base_scores[16];
    int32_t val_curve_len;
    /* followed by: val_curve f32[len], roots u32[num_trees], nodes[] */
} XGBModelFile;

int xgb_model_save(const XGBModel* m, const char* path) {
    if (!m || !path) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    XGBModelFile hdr;
    memset(&hdr, 0, sizeof hdr);
    memcpy(hdr.magic, "RCXG", 4);
    hdr.ver = 1;
    hdr.task = (uint8_t)m->task;
    hdr.K = (uint32_t)m->num_classes;
    hdr.num_trees = (uint32_t)m->num_trees;
    hdr.num_nodes = (uint32_t)m->num_nodes;
    hdr.dim = (uint32_t)m->dim;
    hdr.eta = m->eta;
    hdr.base_score = m->base_score;
    hdr.best_iteration = m->best_iteration;
    for (int k = 0; k < 16; k++) hdr.base_scores[k] = m->base_scores[k];
    hdr.val_curve_len = m->val_curve_len;

    int rc = 0;
    if (fwrite(&hdr, sizeof hdr, 1, f) != 1) rc = -1;
    if (rc == 0 && m->val_curve_len > 0 &&
        fwrite(m->val_curve, sizeof(float), m->val_curve_len, f) != (size_t)m->val_curve_len)
        rc = -1;
    if (rc == 0 && m->num_trees > 0 &&
        fwrite(m->tree_roots, sizeof(uint32_t), m->num_trees, f) != (size_t)m->num_trees)
        rc = -1;
    for (size_t i = 0; rc == 0 && i < m->num_nodes; i++) {
        const XGBNode* nd = &m->nodes[i];
        uint32_t feature = nd->feature, left = nd->left, right = nd->right;
        float thr = nd->threshold, leaf = nd->leaf_value;
        uint8_t flags = (uint8_t)((nd->is_leaf ? 1 : 0) | (nd->default_left ? 2 : 0));
        uint8_t pad[3] = {0, 0, 0};
        if (fwrite(&feature, 4, 1, f) != 1 ||
            fwrite(&thr, 4, 1, f) != 1 ||
            fwrite(&left, 4, 1, f) != 1 ||
            fwrite(&right, 4, 1, f) != 1 ||
            fwrite(&leaf, 4, 1, f) != 1 ||
            fwrite(&flags, 1, 1, f) != 1 ||
            fwrite(pad, 1, 3, f) != 3)   /* count of 1-byte items */
            rc = -1;
    }
    fclose(f);
    return rc;
}

int xgb_model_load(const char* path, XGBModel* m, RCRE_Arena* arena) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    XGBModelFile hdr;
    if (fread(&hdr, sizeof hdr, 1, f) != 1 ||
        memcmp(hdr.magic, "RCXG", 4) != 0 || hdr.ver != 1) {
        fclose(f);
        return -1;
    }
    memset(m, 0, sizeof(*m));
    m->task = (XGBTask)hdr.task;
    m->num_classes = (int)hdr.K;
    m->num_trees = (int)hdr.num_trees;
    m->num_nodes = hdr.num_nodes;
    m->dim = (int)hdr.dim;
    m->eta = hdr.eta;
    m->base_score = hdr.base_score;
    m->best_iteration = hdr.best_iteration;
    for (int k = 0; k < 16; k++) m->base_scores[k] = hdr.base_scores[k];

    m->tree_roots = (uint32_t*)rcre_arena_alloc(arena, hdr.num_trees * sizeof(uint32_t), 4);
    m->nodes = (XGBNode*)rcre_arena_alloc(arena,
                  hdr.num_nodes ? hdr.num_nodes * sizeof(XGBNode) : 16, 16);
    if (!m->tree_roots || !m->nodes) { fclose(f); return -1; }

    int rc = 0;
    if (hdr.val_curve_len > 0) {
        m->val_curve = (float*)rcre_arena_alloc(arena,
                                 hdr.val_curve_len * sizeof(float), 4);
        m->val_curve_len = (int)hdr.val_curve_len;
        if (!m->val_curve) rc = -1;
        else if (fread(m->val_curve, sizeof(float), hdr.val_curve_len, f)
                 != (size_t)hdr.val_curve_len)
            rc = -1;
    }
    if (rc == 0 && m->num_trees > 0 &&
        fread(m->tree_roots, sizeof(uint32_t), m->num_trees, f) != (size_t)m->num_trees)
        rc = -1;
    for (size_t i = 0; rc == 0 && i < m->num_nodes; i++) {
        XGBNode* nd = &m->nodes[i];
        uint32_t feature, left, right;
        float thr, leaf;
        uint8_t flags, pad[3];
        if (fread(&feature, 4, 1, f) != 1 ||
            fread(&thr, 4, 1, f) != 1 ||
            fread(&left, 4, 1, f) != 1 ||
            fread(&right, 4, 1, f) != 1 ||
            fread(&leaf, 4, 1, f) != 1 ||
            fread(&flags, 1, 1, f) != 1 ||
            fread(pad, 1, 3, f) != 3)   /* count of 1-byte items */
        {
            rc = -1;
            break;
        }
        nd->feature = feature;
        nd->threshold = thr;
        nd->left = left;
        nd->right = right;
        nd->leaf_value = leaf;
        nd->is_leaf = (flags & 1) != 0;
        nd->default_left = (flags & 2) != 0;
    }
    fclose(f);
    return rc;
}
