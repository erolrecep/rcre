#define _POSIX_C_SOURCE 200809L
/* RCRE H5 benchmark — Zarr v3 chunked arrays (2.2.0)
 *
 * `make bench_h5`. Measured on Apple M5 Max (single process):
 *   - whole-array write / read throughput (MB/s), serial vs pooled(8)
 *   - zstd (level 3) write throughput + on-disk compression ratio
 *   - per-chunk dispatch overhead proxy: small-array round trip
 *
 * The slab-parallel design (one task per chunk, fixed chunk-order
 * reduction) is validated bitwise in tests/test_zarr.c; this file is
 * about throughput bars only.
 */
#include "rcre.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void rmdir_tree(const char* dir) {
    char cmd[512];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
    (void)system(cmd);
}

typedef struct {
    const char* label;
    size_t      n0, n1;   /* shape */
    size_t      c0, c1;   /* chunk */
    int         zstd;     /* 0 = identity, >0 = zstd level */
} Case;

static void run_case(const char* root, const Case* k, int pooled) {
    char name[64];
    snprintf(name, sizeof name, "h5_%s%s%s", k->label, k->zstd ? "_z" : "",
             pooled ? "_p" : "");
    RCRE_ZarrCreate opt = {0};
    size_t shape[2] = {k->n0, k->n1};
    size_t chunk[2] = {k->c0, k->c1};
    opt.ndim = 2;
    opt.shape = shape;
    opt.chunk_shape = chunk;
    opt.dtype = RCRE_ZARR_F64;
    opt.zstd_level = k->zstd;
    RCRE_ZarrArray* a = rcre_zarr_create(root, name, &opt);
    if (!a) {
        printf("  %-28s CREATE FAILED\n", name);
        return;
    }
    size_t total = k->n0 * k->n1;
    double* data = (double*)malloc(total * 8);
    double* out = (double*)malloc(total * 8);
    if (!data || !out) {
        free(data);
        free(out);
        rcre_zarr_close(a);
        return;
    }
    for (size_t i = 0; i < total; ++i) data[i] = (double)(i & 0xFFFF) * 0.5;
    RCRE_ThreadPool* pool = pooled ? rcre_threadpool_create(8) : NULL;

    double t0 = now_s();
    if (rcre_zarr_write_array(a, data, pool) != RCRE_OK) {
        printf("  %-28s WRITE FAILED\n", name);
        free(data);
        free(out);
        if (pool) rcre_threadpool_destroy(pool);
        rcre_zarr_close(a);
        return;
    }
    double tw = now_s() - t0;
    t0 = now_s();
    if (rcre_zarr_read_array(a, out, pool) != RCRE_OK) {
        printf("  %-28s READ FAILED\n", name);
        free(data);
        free(out);
        if (pool) rcre_threadpool_destroy(pool);
        rcre_zarr_close(a);
        return;
    }
    double tr = now_s() - t0;

    double mb = (double)total * 8 / 1e6;
    int bad = 0;
    for (size_t i = 0; i < total; ++i)
        if (out[i] != data[i]) { bad = 1; break; }
    printf("  %-28s write %8.3f ms  read %8.3f ms  %7.1f / %7.1f MB/s  %s\n",
           name, tw * 1e3, tr * 1e3, mb / tw, mb / tr, bad ? "MISMATCH!" : "ok");

    free(data);
    free(out);
    if (pool) rcre_threadpool_destroy(pool);
    rcre_zarr_close(a);
}

int main(void) {
    rcre_hal_init();
    rcre_zarr_zstd_ensure_registered(); /* idempotent; no-op without libzstd */
    char root[256];
    snprintf(root, sizeof root, "/tmp/rcre_bench_h5_%d", (int)getpid());
    rmdir_tree(root);
    if (rcre_zarr_create_group_root(root) != RCRE_OK) {
        printf("bench_h5: cannot create store root\n");
        return 1;
    }
    printf("Zarr v3 I/O (H5) — f64, chunked, %s zstd codec\n\n",
           rcre_zarr_codec_available("zstd") ? "with" : "WITHOUT");

    Case cases[] = {
        { "256x256/32",    256,  256,   32,  32, 0 },
        { "1024x1024/64",  1024, 1024,  64,  64, 0 },
        { "2048x2048/64",  2048, 2048,  64,  64, 0 },
        { "1024x1024/64z", 1024, 1024,  64,  64, 3 },
    };
    for (int i = 0; i < (int)(sizeof cases / sizeof cases[0]); ++i) {
        if (cases[i].zstd && !rcre_zarr_codec_available("zstd")) continue;
        run_case(root, &cases[i], 0);
        run_case(root, &cases[i], 1);
    }
    /* compression ratio probe (zstd only) */
    if (rcre_zarr_codec_available("zstd")) {
        size_t shape[2] = {1024, 1024}, chunk[2] = {64, 64};
        RCRE_ZarrCreate opt = {0};
        opt.ndim = 2;
        opt.shape = shape;
        opt.chunk_shape = chunk;
        opt.dtype = RCRE_ZARR_F64;
        opt.zstd_level = 3;
        RCRE_ZarrArray* a = rcre_zarr_create(root, "ratio_probe", &opt);
        size_t total = 1024 * 1024;
        double* data = (double*)malloc(total * 8);
        for (size_t i = 0; i < total; ++i) data[i] = (double)(i & 0xFF);
        (void)rcre_zarr_write_array(a, data, NULL);
        rcre_zarr_close(a);
        /* sum chunk file sizes via the c/ tree */
        char cdir[300];
        snprintf(cdir, sizeof cdir, "%s/ratio_probe/c", root);
        unsigned long long disk = 0;
        /* walk c/i/j (2 levels) */
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) {
                char p[400];
                struct stat st;
                snprintf(p, sizeof p, "%s/%d/%d", cdir, i, j);
                if (stat(p, &st) == 0) disk += (unsigned long long)st.st_size;
            }
        printf("\n  zstd level 3 on structured f64 data: disk %.2f MB vs raw %.2f MB "
               "(ratio %.2fx)\n", (double)disk / 1e6,
               (double)total * 8 / 1e6,
               (double)total * 8 / (double)(disk ? disk : 1));
        free(data);
    }
    rmdir_tree(root);
    printf("\nbench_h5 done\n");
    return 0;
}
