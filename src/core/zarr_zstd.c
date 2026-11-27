/* RCRE H5 — Zarr v3 "zstd" codec (opt-in TU, 2.2.0)
 *
 * Bridges libzstd into the Zarr v3 codec pipeline. This translation unit is
 * the D1 bridge/extras pattern: it is compiled ONLY when the build defines
 * RCRE_WITH_ZSTD and libzstd is available (the Makefile auto-detects and
 * adds -lzstd; infrastructure-immune — a build without libzstd simply
 * omits this file and rcre_zarr_codec_available("zstd") reports 0).
 *
 * Zarr v3 zstd codec: stored frame = ZSTD_compress(bytes, level). The
 * v3 "checksum" configuration (frame content checksum) is not emitted
 * (checksum: false — matching zarr-python's default) and, if encountered
 * on read, is accepted transparently (ZSTD_decompress validates frames
 * with or without the checksum flag).
 *
 * Registration: this TU installs its ensure implementation into chunked.c
 * through rcre_zarr_zstd_set_ensure_hook() from a C constructor (link-order
 * immune — no weak symbols, no archive-extraction subtleties). The public
 * rcre_zarr_zstd_ensure_registered() therefore lives in chunked.c and is a
 * no-op only when this TU is absent (e.g. the amalgamated single header,
 * which excludes this dependency-bearing TU).
 */
#include "rcre.h"
#include <stdlib.h>
#include <string.h>

#if defined(RCRE_WITH_ZSTD)
  #include <zstd.h>

static int g_zstd_registered = 0;

static int zstd_decode(const uint8_t* in, size_t in_len, uint8_t* out,
                       size_t out_cap) {
    /* ZSTD_getFrameContentSize: the caller allocates the native chunk
     * buffer (out_cap), so this is a bounded decompression. */
    unsigned long long csize = ZSTD_getFrameContentSize(in, in_len);
    if (csize == ZSTD_CONTENTSIZE_UNKNOWN ||
        csize == ZSTD_CONTENTSIZE_ERROR)
        return RCRE_ERR_INVALID_ARG;
    if (csize > out_cap) return RCRE_ERR_INVALID_ARG;
    size_t r = ZSTD_decompress(out, out_cap, in, in_len);
    if (ZSTD_isError(r)) return RCRE_ERR_INVALID_ARG;
    if (r != (size_t)csize) return RCRE_ERR_INVALID_ARG;
    return RCRE_OK;
}

static int zstd_encode(const uint8_t* in, size_t in_len, int level,
                       uint8_t** out, size_t* out_len) {
    /* The metadata records the level; encode with exactly that level so
     * the recorded codec configuration is true. (libzstd: level >= 1.) */
    size_t cap = ZSTD_compressBound(in_len);
    uint8_t* buf = (uint8_t*)malloc(cap);
    if (!buf) return RCRE_ERR_OUT_OF_MEMORY;
    size_t c = ZSTD_compress(buf, cap, in, in_len, level);
    if (ZSTD_isError(c)) {
        free(buf);
        return RCRE_ERR_INVALID_ARG;
    }
    *out = buf;
    *out_len = c;
    return RCRE_OK;
}

static void zstd_ensure_impl(void) {
    if (g_zstd_registered) return;
    g_zstd_registered = 1;
    (void)rcre_zarr_register_codec("zstd", zstd_decode, zstd_encode);
}

/* Install the hook before main() runs. The constructor touches only a
 * function pointer — no allocation, no I/O — so it is safe in any
 * statically or dynamically linked configuration (clang/gcc; this TU is
 * the only part of the library that uses a constructor attribute). */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void zarr_zstd_install_hook(void) {
    rcre_zarr_zstd_set_ensure_hook(zstd_ensure_impl);
}

#endif /* RCRE_WITH_ZSTD */
