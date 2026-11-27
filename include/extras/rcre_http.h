/**
 * @file rcre_http.h
 * @brief Portable Zero-Malloc HTTP/1.1 Server Companion for RCRE (kqueue / epoll)
 *
 * Strictly segregated under include/extras/ so include/rcre.h remains 100%
 * OS-independent and freestanding-compatible.
 *
 * Targets:
 *   - macOS / Apple Silicon / BSD : kqueue + EVFILT_USER + Apple QoS hints
 *   - Linux (x86_64, RPi, Jetson) : epoll + eventfd + MSG_NOSIGNAL
 *
 * Design guarantees:
 *   - Shared-nothing worker-per-core event loop (1-worker folds accept() inline).
 *   - Zero heap allocations on the request hot path:
 *       * Pre-allocated flat connection slab per worker.
 *       * Slab-pooled 16 KB I/O buffers released back to pool when idle.
 *       * Per-worker RCRE_Arena rewound in O(1) after every request.
 *   - Zero-copy HTTP/1.1 parsing into RCRE_Slice views.
 *   - Per-worker 1-second cached RFC-7231 Date header.
 */

#ifndef RCRE_EXTRAS_HTTP_H
#define RCRE_EXTRAS_HTTP_H

#include "rcre.h"
#include "rcre_str.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum RCRE_HttpMethod {
    RCRE_HTTP_UNKNOWN = 0,
    RCRE_HTTP_GET,
    RCRE_HTTP_POST,
    RCRE_HTTP_HEAD,
    RCRE_HTTP_PUT,
    RCRE_HTTP_DELETE,
    RCRE_HTTP_OPTIONS
} RCRE_HttpMethod;

typedef enum RCRE_HttpParseStatus {
    RCRE_HTTP_PARSE_OK         =  0,
    RCRE_HTTP_PARSE_INCOMPLETE =  1,
    RCRE_HTTP_PARSE_ERROR      = -1
} RCRE_HttpParseStatus;

typedef struct RCRE_HttpRequest {
    RCRE_HttpMethod method;
    RCRE_Slice      path;
    RCRE_Slice      query;
    RCRE_Slice      body;
    size_t          content_length;
    size_t          consumed;
    size_t          expected_total;
    bool            keep_alive;
    uint8_t         http_minor;
} RCRE_HttpRequest;

typedef struct RCRE_HttpContext {
    RCRE_Arena* scratch;    /* Per-worker arena, rewound O(1) after each request */
    size_t      worker_id;  /* 0 .. num_workers - 1                              */
    const char* date_hdr;   /* 29-byte cached RFC-7231 GMT Date header           */
    void*       user_ctx;   /* Opaque user pointer from RCRE_HttpServerConfig    */
} RCRE_HttpContext;

/**
 * @brief Request handler callback. Must serialize a complete HTTP/1.1 response
 *        into out_resp (e.g. via rcre_http_write_response) and return out_resp->len.
 */
typedef size_t (*rcre_http_handler_fn)(
    const RCRE_HttpRequest* req,
    RCRE_HttpContext* ctx,
    RCRE_StrBuf* out_resp
);

typedef struct RCRE_HttpServerConfig {
    const char*          host;                /* Default: "127.0.0.1"        */
    uint16_t             port;                /* 0 => OS-assigned ephemeral  */
    size_t               num_workers;         /* 0 => 1 worker               */
    size_t               worker_arena_bytes;  /* 0 => 4 MiB default          */
    size_t               max_conns_per_worker;/* 0 => 4096 default           */
    size_t               max_body_bytes;      /* 0 => 1 MiB default          */
    rcre_http_handler_fn handler;             /* Required request callback   */
    void*                user_ctx;            /* Passed into RCRE_HttpContext*/
} RCRE_HttpServerConfig;

typedef struct RCRE_HttpServer RCRE_HttpServer;

/** Zero-copy incremental HTTP/1.1 parser over a raw socket buffer. */
RCRE_NODISCARD RCRE_HttpParseStatus rcre_http_parse_request(
    const uint8_t* buf,
    size_t len,
    size_t max_body_bytes,
    RCRE_HttpRequest* out_req
);

/** Zero-allocation HTTP/1.1 response framer writing headers + body into out_resp. */
size_t rcre_http_write_response(
    RCRE_StrBuf* out_resp,
    RCRE_Slice status_line,
    RCRE_Slice content_type,
    const char* date_hdr_29,
    bool keep_alive,
    RCRE_Slice body
);

/** Creates and binds the listening socket + worker reactors (does not block). */
RCRE_NODISCARD RCRE_HttpServer* rcre_http_server_create(const RCRE_HttpServerConfig* cfg);

/** Returns the actual bound TCP port (useful when cfg->port == 0). */
RCRE_NODISCARD uint16_t rcre_http_server_port(const RCRE_HttpServer* srv);

/** Runs the server event loop until rcre_http_server_stop() is called. */
RCRE_NODISCARD int rcre_http_server_run(RCRE_HttpServer* srv);

/** Thread-safe, signal-safe request to gracefully stop a running server. */
void rcre_http_server_stop(RCRE_HttpServer* srv);

/** Destroys the server and frees all worker slabs, arenas, and sockets. */
void rcre_http_server_destroy(RCRE_HttpServer* srv);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_EXTRAS_HTTP_H */
