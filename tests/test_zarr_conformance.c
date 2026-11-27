/* RCRE H5 conformance driver — Zarr v3 (C side of the cross-tooling gate)
 *
 * `make conformance-zarr` orchestrates:
 *   1. tests/ref_zarr_conformance.py write <store>
 *        -> writes <name>_py arrays + manifest.txt + expected hashes
 *   2. this driver (bin/test_zarr_conformance <store>)
 *        -> reads every <name>_py (hashes raw bytes -> c_hashes.json)
 *        -> creates every <name>_c from the identical LCG stream
 *           (hashes the data it wrote -> c_expected.json)
 *   3. tests/ref_zarr_conformance.py verify <store>
 *        -> zarr-python reads the _c arrays and compares hashes,
 *           metadata, golden JSON, NaN fill survival.
 *
 * Data contract (must match ref_zarr_conformance.py EXACTLY):
 *   st = st * 6364136223846793005ull + 1442695040888963407ull   (64-bit)
 *   v  = (st >> 11) & ((1ull << 53) - 1)                        (53-bit int)
 *   f64: v                f32: (float)(v & 0xFFFFFFFFF)
 *   f16: rcre_f16_from_f64((double)(v & 0x3FF))
 *   i64/u64: v            i32: (int32_t)v  u32: (uint32_t)v
 *   i16: (int16_t)v  ...  bool: v & 1
 *
 * Hash: FNV-1a 64 over the little-endian byte stream (offset
 * 0xcbf29ce484222325, prime 0x100000001b3), identical on both sides.
 */
#include "rcre.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;
static const uint64_t FNV_PRIME = 0x100000001b3ull;

static uint64_t fnv1a64(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < len; ++i) h = (h ^ p[i]) * FNV_PRIME;
    return h;
}

/* one manifest line:
 * name ndim d0..dn-1 c0..cn-1 dti fillint fill64bits zstd seed */
typedef struct {
    char name[64];
    int  ndim;
    size_t shape[8];
    size_t chunk[8];
    int   dti;
    int64_t fillint;
    uint64_t fillbits;
    int   zstd;
    uint64_t seed;
} ConSpec;

static int parse_line(char* line, ConSpec* s) {
    if (sscanf(line, "%63s %d", s->name, &s->ndim) != 2) return -1;
    if (s->ndim < 1 || s->ndim > 8) return -1;
    /* re-tokenize the remainder (ndim, dims, chunks, dti, fillint,
     * fillbits, zstd, seed) with strtok */
    char* tok = strtok(line + strlen(s->name) + 1, " \n"); /* ndim */
    if (!tok) return -1;
    int d = 0;
    while (d < s->ndim && (tok = strtok(NULL, " \n"))) s->shape[d++] = (size_t)strtoul(tok, NULL, 10);
    int c = 0;
    while (c < s->ndim && (tok = strtok(NULL, " \n"))) s->chunk[c++] = (size_t)strtoul(tok, NULL, 10);
    if (d != s->ndim || c != s->ndim) return -1;
    if (!(tok = strtok(NULL, " \n"))) return -1; s->dti = atoi(tok);
    if (!(tok = strtok(NULL, " \n"))) return -1; s->fillint = (int64_t)strtoll(tok, NULL, 10);
    if (!(tok = strtok(NULL, " \n"))) return -1; s->fillbits = strtoull(tok, NULL, 16);
    if (!(tok = strtok(NULL, " \n"))) return -1; s->zstd = atoi(tok);
    if (!(tok = strtok(NULL, " \n"))) return -1; s->seed = strtoull(tok, NULL, 16);
    return 0;
}

/* generate the conformance data stream for spec s (see header contract) */
static void con_fill(void* dst, size_t n, const ConSpec* s) {
    uint64_t st = s->seed;
    RCRE_ZarrDType dt = (RCRE_ZarrDType)s->dti;
    switch (dt) {
        case RCRE_ZARR_F64: {
            double* p = (double*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (double)((st >> 11) & ((1ull << 53) - 1));
            }
            break;
        }
        case RCRE_ZARR_F32: {
            float* p = (float*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (float)((st >> 11) & 0xFFFFFFFFFull);
            }
            break;
        }
        case RCRE_ZARR_F16: {
            uint16_t* p = (uint16_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = rcre_f16_from_f64((double)((st >> 11) & 0x3FFull));
            }
            break;
        }
        case RCRE_ZARR_I64: {
            int64_t* p = (int64_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (int64_t)((st >> 11) & ((1ull << 53) - 1));
            }
            break;
        }
        case RCRE_ZARR_I32: {
            int32_t* p = (int32_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (int32_t)((st >> 11) & ((1ull << 53) - 1));
            }
            break;
        }
        case RCRE_ZARR_I16: {
            int16_t* p = (int16_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (int16_t)((st >> 11) & ((1ull << 53) - 1));
            }
            break;
        }
        case RCRE_ZARR_I8: {
            int8_t* p = (int8_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (int8_t)((st >> 11) & ((1ull << 53) - 1));
            }
            break;
        }
        case RCRE_ZARR_U64: {
            uint64_t* p = (uint64_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (st >> 11) & ((1ull << 53) - 1);
            }
            break;
        }
        case RCRE_ZARR_U32: {
            uint32_t* p = (uint32_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (uint32_t)((st >> 11) & ((1ull << 53) - 1));
            }
            break;
        }
        case RCRE_ZARR_U16: {
            uint16_t* p = (uint16_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (uint16_t)((st >> 11) & ((1ull << 53) - 1));
            }
            break;
        }
        case RCRE_ZARR_U8: {
            uint8_t* p = (uint8_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (uint8_t)((st >> 11) & ((1ull << 53) - 1));
            }
            break;
        }
        case RCRE_ZARR_BOOL: {
            uint8_t* p = (uint8_t*)dst;
            for (size_t i = 0; i < n; ++i) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                p[i] = (uint8_t)((st >> 11) & 1ull);
            }
            break;
        }
        default:
            abort();
    }
}

static size_t spec_elems(const ConSpec* s) {
    size_t n = 1;
    for (int d = 0; d < s->ndim; ++d) n *= s->shape[d];
    return n;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <store_dir>\n", argv[0]);
        return 2;
    }
    const char* store = argv[1];
    rcre_hal_init();
    rcre_zarr_zstd_ensure_registered(); /* zstd arrays need the codec up front */

    char mpath[1024];
    snprintf(mpath, sizeof mpath, "%s/manifest.txt", store);
    FILE* mf = fopen(mpath, "r");
    if (!mf) {
        fprintf(stderr, "missing manifest: %s\n", mpath);
        return 2;
    }
    ConSpec specs[64];
    int nspec = 0;
    char line[512];
    while (fgets(line, sizeof line, mf) && nspec < 64) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        if (parse_line(line, &specs[nspec]) != 0) {
            fprintf(stderr, "bad manifest line: %s", line);
            fclose(mf);
            return 2;
        }
        nspec++;
    }
    fclose(mf);
    printf("conformance: %d specs\n", nspec);

    char line_json[4096];
    snprintf(line_json, sizeof line_json, "%s/c_hashes.json", store);
    FILE* hf = fopen(line_json, "w");
    fprintf(hf, "{\n");
    char exp_json[4096];
    snprintf(exp_json, sizeof exp_json, "%s/c_expected.json", store);
    FILE* ef = fopen(exp_json, "w");
    fprintf(ef, "{\n");

    int fails = 0;
    for (int i = 0; i < nspec; ++i) {
        ConSpec* s = &specs[i];
        size_t n = spec_elems(s);

        /* --- read the _py array zarr-python wrote, hash its bytes --- */
        char pname[96];
        snprintf(pname, sizeof pname, "%s_py", s->name);
        RCRE_ZarrArray* a = rcre_zarr_open(store, pname);
        if (!a) {
            fprintf(stderr, "  [FAIL] cannot open %s\n", pname);
            fails++;
            continue;
        }
        size_t isz = rcre_zarr_itemsize(a);
        uint8_t* buf = (uint8_t*)malloc(n * isz);
        if (rcre_zarr_read_array(a, buf, NULL) != RCRE_OK) {
            fprintf(stderr, "  [FAIL] read %s\n", pname);
            fails++;
        } else {
            uint64_t h = fnv1a64(buf, n * isz);
            fprintf(hf, "  \"%s\": %llu%s\n", pname, (unsigned long long)h,
                    i + 1 < nspec ? "," : "");
            printf("  %-16s py->C fnv=%016llx\n", pname,
                   (unsigned long long)h);
        }
        free(buf);
        rcre_zarr_close(a);

        /* --- create the _c array and write our identical stream --- */
        RCRE_ZarrCreate opt = {0};
        opt.ndim = s->ndim;
        opt.shape = s->shape;
        opt.chunk_shape = s->chunk;
        opt.dtype = (RCRE_ZarrDType)s->dti;
        opt.fill_int = s->fillint;
        memcpy(&opt.fill_f64, &s->fillbits, sizeof(uint64_t));
        opt.zstd_level = s->zstd;
        char cname[96];
        snprintf(cname, sizeof cname, "%s_c", s->name);
        RCRE_ZarrArray* c = rcre_zarr_create(store, cname, &opt);
        if (!c) {
            fprintf(stderr, "  [FAIL] create %s\n", cname);
            fails++;
            continue;
        }
        uint8_t* cbuf = (uint8_t*)malloc(n * rcre_zarr_itemsize(c));
        con_fill(cbuf, n, s);
        uint64_t h = fnv1a64(cbuf, n * rcre_zarr_itemsize(c));
        if (rcre_zarr_write_array(c, cbuf, NULL) != RCRE_OK) {
            fprintf(stderr, "  [FAIL] write %s\n", cname);
            fails++;
        } else {
            fprintf(ef, "  \"%s\": %llu%s\n", cname, (unsigned long long)h,
                    i + 1 < nspec ? "," : "");
        }
        free(cbuf);
        rcre_zarr_close(c);
    }
    fprintf(hf, "}\n");
    fprintf(ef, "}\n");
    fclose(hf);
    fclose(ef);

    if (fails == 0) {
        printf("C conformance side OK (%d arrays read + %d arrays written)\n",
               nspec, nspec);
        return 0;
    }
    printf("C conformance side FAILED (%d problems)\n", fails);
    return 1;
}
