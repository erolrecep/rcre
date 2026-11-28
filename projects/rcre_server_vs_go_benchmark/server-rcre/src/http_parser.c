#include "http_parser.h"
#include <string.h>

static bool parse_u64_strict(RCRE_Slice s, size_t* out) {
    s = rcre_slice_trim_left_ascii(rcre_slice_trim_right_ascii(s));
    if (s.len == 0 || s.len > 12) {
        return false;
    }
    size_t v = 0;
    for (size_t i = 0; i < s.len; ++i) {
        uint8_t c = s.data[i];
        if (c < '0' || c > '9') {
            return false;
        }
        v = v * 10U + (size_t)(c - '0');
    }
    *out = v;
    return true;
}

static HttpMethod parse_method(RCRE_Slice m) {
    if (m.len == 3 && memcmp(m.data, "GET", 3) == 0) return HTTP_METHOD_GET;
    if (m.len == 4 && memcmp(m.data, "POST", 4) == 0) return HTTP_METHOD_POST;
    if (m.len == 4 && memcmp(m.data, "HEAD", 4) == 0) return HTTP_METHOD_HEAD;
    if (m.len == 3 && memcmp(m.data, "PUT", 3) == 0) return HTTP_METHOD_PUT;
    if (m.len == 6 && memcmp(m.data, "DELETE", 6) == 0) return HTTP_METHOD_DELETE;
    if (m.len == 7 && memcmp(m.data, "OPTIONS", 7) == 0) return HTTP_METHOD_OPTIONS;
    return HTTP_METHOD_UNKNOWN;
}

/* Fast scan for "\r\n\r\n" */
static size_t find_header_end(const uint8_t* buf, size_t len) {
    if (len < 4) return RCRE_NPOS;
    const uint8_t* p = buf;
    const uint8_t* end = buf + len - 3;
    while (p < end) {
        const uint8_t* r = (const uint8_t*)memchr(p, '\r', (size_t)(end - p));
        if (!r) return RCRE_NPOS;
        if (r[1] == '\n' && r[2] == '\r' && r[3] == '\n') {
            return (size_t)(r - buf);
        }
        p = r + 1;
    }
    return RCRE_NPOS;
}

HttpParseResult http_parse_request(const uint8_t* buf, size_t len, HttpRequest* out_req) {
    memset(out_req, 0, sizeof(*out_req));
    if (len == 0) {
        return HTTP_PARSE_INCOMPLETE;
    }

    size_t hdr_end = find_header_end(buf, len);
    if (hdr_end == RCRE_NPOS) {
        if (len > HTTP_MAX_HEADER_BYTES) {
            return HTTP_PARSE_ERROR;
        }
        return HTTP_PARSE_INCOMPLETE;
    }
    if (hdr_end > HTTP_MAX_HEADER_BYTES) {
        return HTTP_PARSE_ERROR;
    }

    size_t body_off = hdr_end + 4U;

    /* Find end of request line */
    const uint8_t* cr = (const uint8_t*)memchr(buf, '\r', hdr_end + 2U);
    if (!cr || (size_t)(cr - buf) + 1U >= hdr_end + 4U || cr[1] != '\n') {
        return HTTP_PARSE_ERROR;
    }
    size_t rline_len = (size_t)(cr - buf);

    /* Parse: METHOD SP TARGET SP HTTP/1.x */
    const uint8_t* sp1 = (const uint8_t*)memchr(buf, ' ', rline_len);
    if (!sp1 || sp1 == buf) {
        return HTTP_PARSE_ERROR;
    }
    size_t m_len = (size_t)(sp1 - buf);
    out_req->method = parse_method(rcre_slice_from_parts(buf, m_len));
    if (out_req->method == HTTP_METHOD_UNKNOWN) {
        return HTTP_PARSE_ERROR;
    }

    const uint8_t* target_start = sp1 + 1;
    size_t rem_len = rline_len - m_len - 1U;
    const uint8_t* sp2 = (const uint8_t*)memchr(target_start, ' ', rem_len);
    if (!sp2 || sp2 == target_start) {
        return HTTP_PARSE_ERROR;
    }
    size_t target_len = (size_t)(sp2 - target_start);
    if (target_start[0] != '/') {
        return HTTP_PARSE_ERROR;
    }

    const uint8_t* ver = sp2 + 1;
    size_t ver_len = (size_t)((buf + rline_len) - ver);
    if (ver_len != 8U || memcmp(ver, "HTTP/1.", 7) != 0 || (ver[7] != '0' && ver[7] != '1')) {
        return HTTP_PARSE_ERROR;
    }
    out_req->http_minor = (uint8_t)(ver[7] - '0');
    out_req->keep_alive = (out_req->http_minor == 1U);

    /* Split target into path and query */
    const uint8_t* qmark = (const uint8_t*)memchr(target_start, '?', target_len);
    if (qmark) {
        size_t plen = (size_t)(qmark - target_start);
        out_req->path  = rcre_slice_from_parts(target_start, plen);
        out_req->query = rcre_slice_from_parts(qmark + 1, target_len - plen - 1U);
    } else {
        out_req->path  = rcre_slice_from_parts(target_start, target_len);
        out_req->query = rcre_slice_empty();
    }

    /* Parse headers */
    size_t pos = rline_len + 2U;
    bool has_cl = false;
    while (pos < hdr_end + 2U) {
        const uint8_t* h_end = (const uint8_t*)memchr(buf + pos, '\r', (hdr_end + 2U) - pos);
        if (!h_end || h_end[1] != '\n') {
            return HTTP_PARSE_ERROR;
        }
        size_t h_len = (size_t)(h_end - (buf + pos));
        if (h_len == 0) {
            break;
        }
        const uint8_t* colon = (const uint8_t*)memchr(buf + pos, ':', h_len);
        if (!colon || colon == buf + pos) {
            return HTTP_PARSE_ERROR;
        }
        RCRE_Slice name = rcre_slice_from_parts(buf + pos, (size_t)(colon - (buf + pos)));
        RCRE_Slice val  = rcre_slice_from_parts(colon + 1, h_len - name.len - 1U);

        /* Fast length dispatch before case-insensitive compare */
        if (name.len == 14U && rcre_slice_eq_ascii_ci(name, RCRE_STR("Content-Length"))) {
            size_t cl = 0;
            if (!parse_u64_strict(val, &cl) || cl > HTTP_MAX_BODY_BYTES) {
                return HTTP_PARSE_ERROR;
            }
            if (has_cl && cl != out_req->content_length) {
                return HTTP_PARSE_ERROR;
            }
            out_req->content_length = cl;
            has_cl = true;
        } else if (name.len == 10U && rcre_slice_eq_ascii_ci(name, RCRE_STR("Connection"))) {
            val = rcre_slice_trim_left_ascii(rcre_slice_trim_right_ascii(val));
            if (rcre_slice_eq_ascii_ci(val, RCRE_STR("close"))) {
                out_req->keep_alive = false;
            } else if (rcre_slice_eq_ascii_ci(val, RCRE_STR("keep-alive"))) {
                out_req->keep_alive = true;
            }
        }

        pos = (size_t)(h_end - buf) + 2U;
    }

    size_t total_needed = body_off + out_req->content_length;
    out_req->expected_total = total_needed;
    if (len < total_needed) {
        return HTTP_PARSE_INCOMPLETE;
    }

    out_req->body = (out_req->content_length > 0)
        ? rcre_slice_from_parts(buf + body_off, out_req->content_length)
        : rcre_slice_empty();
    out_req->consumed = total_needed;
    return HTTP_PARSE_OK;
}
