#ifndef RCRE_BENCH_HTTP_PARSER_H
#define RCRE_BENCH_HTTP_PARSER_H

#include "rcre.h"
#include "rcre_str.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum HttpMethod {
    HTTP_METHOD_UNKNOWN = 0,
    HTTP_METHOD_GET,
    HTTP_METHOD_POST,
    HTTP_METHOD_HEAD,
    HTTP_METHOD_PUT,
    HTTP_METHOD_DELETE,
    HTTP_METHOD_OPTIONS
} HttpMethod;

typedef enum HttpParseResult {
    HTTP_PARSE_OK         = 0,
    HTTP_PARSE_INCOMPLETE = 1,
    HTTP_PARSE_ERROR      = -1
} HttpParseResult;

typedef struct HttpRequest {
    HttpMethod method;
    RCRE_Slice path;
    RCRE_Slice query;
    RCRE_Slice body;
    size_t     content_length;
    size_t     consumed;       /* Total bytes consumed from input buffer */
    size_t     expected_total; /* Valid when INCOMPLETE after headers parsed */
    bool       keep_alive;
    uint8_t    http_minor;     /* 0 or 1 */
} HttpRequest;

constexpr size_t HTTP_MAX_HEADER_BYTES = 16384U;
constexpr size_t HTTP_MAX_BODY_BYTES   = 1048576U; /* 1 MiB */

RCRE_NODISCARD HttpParseResult http_parse_request(
    const uint8_t* buf,
    size_t len,
    HttpRequest* out_req
);

#endif /* RCRE_BENCH_HTTP_PARSER_H */
