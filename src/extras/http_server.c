/**
 * @file http_server.c
 * @brief Portable Zero-Malloc HTTP/1.1 Reactor Companion (macOS kqueue / Linux epoll)
 */

#include "extras/rcre_http.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
  #define RCRE_HTTP_USE_KQUEUE 1
  #include <sys/event.h>
  #if defined(__APPLE__)
    #include <pthread/qos.h>
  #endif
#elif defined(__linux__)
  #define RCRE_HTTP_USE_EPOLL 1
  #include <sys/epoll.h>
  #include <sys/eventfd.h>
#else
  #error "rcre_http requires kqueue (macOS/BSD) or epoll (Linux)"
#endif

/* ========================================================================= */
/* Constants & Internal Limits                                               */
/* ========================================================================= */

#define RCRE_HTTP_SLAB_BUF_SIZE     16384U
#define RCRE_HTTP_SLAB_BUFS_WKR     256U
#define RCRE_HTTP_ACCEPT_RING_CAP   4096U  /* Power of two */
#define RCRE_HTTP_EVENT_BATCH       256U
#define RCRE_HTTP_MAX_HEADER_BYTES  16384U
#define RCRE_HTTP_TX_STAGING_CAP    24576U

#if defined(RCRE_HTTP_USE_KQUEUE)
  #define RCRE_HTTP_WAKE_IDENT      ((uintptr_t)0x52435245ULL)
#endif

/* ========================================================================= */
/* Zero-Copy HTTP/1.1 Parser                                                 */
/* ========================================================================= */

static bool http_parse_u64_strict(RCRE_Slice s, size_t* out) {
    s = rcre_slice_trim_left_ascii(rcre_slice_trim_right_ascii(s));
    if (s.len == 0 || s.len > 12U) return false;
    size_t v = 0;
    for (size_t i = 0; i < s.len; ++i) {
        uint8_t c = s.data[i];
        if (c < '0' || c > '9') return false;
        v = v * 10U + (size_t)(c - '0');
    }
    *out = v;
    return true;
}

static RCRE_HttpMethod http_parse_method(RCRE_Slice m) {
    if (m.len == 3 && memcmp(m.data, "GET", 3) == 0)     return RCRE_HTTP_GET;
    if (m.len == 4 && memcmp(m.data, "POST", 4) == 0)    return RCRE_HTTP_POST;
    if (m.len == 4 && memcmp(m.data, "HEAD", 4) == 0)    return RCRE_HTTP_HEAD;
    if (m.len == 3 && memcmp(m.data, "PUT", 3) == 0)     return RCRE_HTTP_PUT;
    if (m.len == 6 && memcmp(m.data, "DELETE", 6) == 0)  return RCRE_HTTP_DELETE;
    if (m.len == 7 && memcmp(m.data, "OPTIONS", 7) == 0) return RCRE_HTTP_OPTIONS;
    return RCRE_HTTP_UNKNOWN;
}

static size_t http_find_header_end(const uint8_t* buf, size_t len) {
    if (len < 4U) return RCRE_NPOS;
    const uint8_t* p = buf;
    const uint8_t* end = buf + len - 3U;
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

RCRE_HttpParseStatus rcre_http_parse_request(
    const uint8_t* buf,
    size_t len,
    size_t max_body_bytes,
    RCRE_HttpRequest* out_req
) {
    if (!out_req) return RCRE_HTTP_PARSE_ERROR;
    memset(out_req, 0, sizeof(*out_req));
    if (!buf || len == 0) return RCRE_HTTP_PARSE_INCOMPLETE;
    if (max_body_bytes == 0) max_body_bytes = 1048576U;

    size_t hdr_end = http_find_header_end(buf, len);
    if (hdr_end == RCRE_NPOS) {
        return (len > RCRE_HTTP_MAX_HEADER_BYTES) ? RCRE_HTTP_PARSE_ERROR : RCRE_HTTP_PARSE_INCOMPLETE;
    }
    if (hdr_end > RCRE_HTTP_MAX_HEADER_BYTES) return RCRE_HTTP_PARSE_ERROR;

    size_t body_off = hdr_end + 4U;

    const uint8_t* cr = (const uint8_t*)memchr(buf, '\r', hdr_end + 2U);
    if (!cr || (size_t)(cr - buf) + 1U >= body_off || cr[1] != '\n') return RCRE_HTTP_PARSE_ERROR;
    size_t rline_len = (size_t)(cr - buf);

    const uint8_t* sp1 = (const uint8_t*)memchr(buf, ' ', rline_len);
    if (!sp1 || sp1 == buf) return RCRE_HTTP_PARSE_ERROR;
    size_t m_len = (size_t)(sp1 - buf);
    out_req->method = http_parse_method(rcre_slice_from_parts(buf, m_len));
    if (out_req->method == RCRE_HTTP_UNKNOWN) return RCRE_HTTP_PARSE_ERROR;

    const uint8_t* target_start = sp1 + 1;
    size_t rem_len = rline_len - m_len - 1U;
    const uint8_t* sp2 = (const uint8_t*)memchr(target_start, ' ', rem_len);
    if (!sp2 || sp2 == target_start || target_start[0] != '/') return RCRE_HTTP_PARSE_ERROR;
    size_t target_len = (size_t)(sp2 - target_start);

    const uint8_t* ver = sp2 + 1;
    size_t ver_len = (size_t)((buf + rline_len) - ver);
    if (ver_len != 8U || memcmp(ver, "HTTP/1.", 7) != 0 || (ver[7] != '0' && ver[7] != '1')) {
        return RCRE_HTTP_PARSE_ERROR;
    }
    out_req->http_minor = (uint8_t)(ver[7] - '0');
    out_req->keep_alive = (out_req->http_minor == 1U);

    const uint8_t* qmark = (const uint8_t*)memchr(target_start, '?', target_len);
    if (qmark) {
        size_t plen = (size_t)(qmark - target_start);
        out_req->path  = rcre_slice_from_parts(target_start, plen);
        out_req->query = rcre_slice_from_parts(qmark + 1, target_len - plen - 1U);
    } else {
        out_req->path  = rcre_slice_from_parts(target_start, target_len);
        out_req->query = rcre_slice_empty();
    }

    size_t pos = rline_len + 2U;
    bool has_cl = false;
    while (pos < hdr_end + 2U) {
        const uint8_t* h_end = (const uint8_t*)memchr(buf + pos, '\r', (hdr_end + 2U) - pos);
        if (!h_end || h_end[1] != '\n') return RCRE_HTTP_PARSE_ERROR;
        size_t h_len = (size_t)(h_end - (buf + pos));
        if (h_len == 0) break;
        const uint8_t* colon = (const uint8_t*)memchr(buf + pos, ':', h_len);
        if (!colon || colon == buf + pos) return RCRE_HTTP_PARSE_ERROR;

        RCRE_Slice name = rcre_slice_from_parts(buf + pos, (size_t)(colon - (buf + pos)));
        RCRE_Slice val  = rcre_slice_from_parts(colon + 1, h_len - name.len - 1U);

        if (name.len == 14U && rcre_slice_eq_ascii_ci(name, RCRE_STR("Content-Length"))) {
            size_t cl = 0;
            if (!http_parse_u64_strict(val, &cl) || cl > max_body_bytes) return RCRE_HTTP_PARSE_ERROR;
            if (has_cl && cl != out_req->content_length) return RCRE_HTTP_PARSE_ERROR;
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
    if (len < total_needed) return RCRE_HTTP_PARSE_INCOMPLETE;

    out_req->body = (out_req->content_length > 0)
        ? rcre_slice_from_parts(buf + body_off, out_req->content_length)
        : rcre_slice_empty();
    out_req->consumed = total_needed;
    return RCRE_HTTP_PARSE_OK;
}

size_t rcre_http_write_response(
    RCRE_StrBuf* out_resp,
    RCRE_Slice status_line,
    RCRE_Slice content_type,
    const char* date_hdr_29,
    bool keep_alive,
    RCRE_Slice body
) {
    if (!out_resp) return 0;
    rcre_strbuf_reset(out_resp);
    (void)rcre_strbuf_append(out_resp, status_line);
    (void)rcre_strbuf_append(out_resp, RCRE_STR("\r\nContent-Type: "));
    (void)rcre_strbuf_append(out_resp, content_type);
    (void)rcre_strbuf_append(out_resp, RCRE_STR("\r\nContent-Length: "));
    (void)rcre_strbuf_append_u64(out_resp, (uint64_t)body.len);
    if (date_hdr_29) {
        (void)rcre_strbuf_append(out_resp, RCRE_STR("\r\nDate: "));
        (void)rcre_strbuf_append(out_resp, rcre_slice_from_parts((const uint8_t*)date_hdr_29, 29));
    }
    if (keep_alive) {
        (void)rcre_strbuf_append(out_resp, RCRE_STR("\r\nConnection: keep-alive\r\n\r\n"));
    } else {
        (void)rcre_strbuf_append(out_resp, RCRE_STR("\r\nConnection: close\r\n\r\n"));
    }
    if (body.len > 0 && body.data) {
        (void)rcre_strbuf_append(out_resp, body);
    }
    return out_resp->len;
}

/* ========================================================================= */
/* Per-Worker Slab Buffer Pool & Lock-Free SPSC FD Ring                      */
/* ========================================================================= */

typedef struct HttpBufSlab {
    uint8_t*  backing;
    uint32_t* free_stack;
    size_t    free_top;
    size_t    count;
} HttpBufSlab;

static bool http_slab_init(HttpBufSlab* slab, size_t count) {
    slab->backing = (uint8_t*)aligned_alloc(64, count * RCRE_HTTP_SLAB_BUF_SIZE);
    slab->free_stack = (uint32_t*)malloc(count * sizeof(uint32_t));
    if (!slab->backing || !slab->free_stack) {
        free(slab->backing);
        free(slab->free_stack);
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        slab->free_stack[i] = (uint32_t)(count - 1U - i);
    }
    slab->free_top = count;
    slab->count = count;
    return true;
}

static void http_slab_destroy(HttpBufSlab* slab) {
    free(slab->backing);
    free(slab->free_stack);
    memset(slab, 0, sizeof(*slab));
}

static inline uint8_t* http_slab_acquire(HttpBufSlab* slab, uint32_t* out_idx) {
    if (slab->free_top == 0) {
        *out_idx = UINT32_MAX;
        return nullptr;
    }
    uint32_t idx = slab->free_stack[--slab->free_top];
    *out_idx = idx;
    return slab->backing + (size_t)idx * RCRE_HTTP_SLAB_BUF_SIZE;
}

static inline void http_slab_release(HttpBufSlab* slab, uint32_t idx) {
    if (idx != UINT32_MAX && slab->free_top < slab->count) {
        slab->free_stack[slab->free_top++] = idx;
    }
}

typedef struct HttpFdRing {
    alignas(64) atomic_size_t head;
    alignas(64) atomic_size_t tail;
    int fds[RCRE_HTTP_ACCEPT_RING_CAP];
} HttpFdRing;

static inline bool http_ring_push(HttpFdRing* r, int fd) {
    size_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    size_t next = (t + 1U) & (RCRE_HTTP_ACCEPT_RING_CAP - 1U);
    if (next == atomic_load_explicit(&r->head, memory_order_acquire)) return false;
    r->fds[t] = fd;
    atomic_store_explicit(&r->tail, next, memory_order_release);
    return true;
}

static inline bool http_ring_pop(HttpFdRing* r, int* out_fd) {
    size_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    if (h == atomic_load_explicit(&r->tail, memory_order_acquire)) return false;
    *out_fd = r->fds[h];
    atomic_store_explicit(&r->head, (h + 1U) & (RCRE_HTTP_ACCEPT_RING_CAP - 1U), memory_order_release);
    return true;
}

/* ========================================================================= */
/* Connection, Worker & Server Structs                                       */
/* ========================================================================= */

typedef struct HttpConn {
    int      fd;
    uint32_t slot_id;

    uint8_t* rbuf;
    uint32_t rbuf_slab_idx;
    size_t   rbuf_len;
    size_t   rbuf_cap;
    bool     rbuf_heap;

    uint8_t* wbuf;
    uint32_t wbuf_slab_idx;
    size_t   wbuf_len;
    size_t   wbuf_sent;
    size_t   wbuf_cap;
    bool     wbuf_heap;

    bool     write_armed;
    bool     close_after_write;
} HttpConn;

typedef struct HttpWorker {
    size_t           id;
    int              poll_fd;
#if defined(RCRE_HTTP_USE_EPOLL)
    int              wake_fd;
#endif
    int              inline_listen_fd;
    RCRE_HttpServer* srv;
    pthread_t        thread;

    HttpFdRing       ring;
    atomic_bool      sleeping;

    RCRE_Arena*      scratch;
    HttpBufSlab      slab;
    HttpConn*        conns;
    uint32_t*        free_slots;
    size_t           free_top;
    size_t           max_conns;
    uint8_t*         tx_staging;

    time_t           cached_sec;
    char             date_hdr[32];
} HttpWorker;

struct RCRE_HttpServer {
    int                  listen_fd;
    uint16_t             bound_port;
    size_t               num_workers;
    size_t               max_body_bytes;
    rcre_http_handler_fn handler;
    void*                user_ctx;
    HttpWorker*          workers;
    atomic_bool          stop;
#if defined(RCRE_HTTP_USE_KQUEUE)
    int                  accept_poll_fd;
#elif defined(RCRE_HTTP_USE_EPOLL)
    int                  accept_poll_fd;
    int                  accept_wake_fd;
#endif
};

static bool http_set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return false;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static void http_configure_client(int fd) {
    (void)http_set_nonblocking(fd);
    int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}

static ssize_t http_sock_write(int fd, const uint8_t* buf, size_t len) {
#if defined(MSG_NOSIGNAL)
    return send(fd, buf, len, MSG_NOSIGNAL);
#else
    return write(fd, buf, len);
#endif
}

static void http_refresh_date(HttpWorker* w, time_t now) {
    if (now == w->cached_sec) return;
    w->cached_sec = now;
    struct tm gmt;
    gmtime_r(&now, &gmt);
    static const char* days[7] = { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" };
    static const char* months[12] = {
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    };
    snprintf(w->date_hdr, sizeof(w->date_hdr),
             "%s, %02d %s %04d %02d:%02d:%02d GMT",
             days[gmt.tm_wday % 7], gmt.tm_mday, months[gmt.tm_mon % 12],
             gmt.tm_year + 1900, gmt.tm_hour, gmt.tm_min, gmt.tm_sec);
}

static void http_conn_release_rbuf(HttpWorker* w, HttpConn* c) {
    if (!c->rbuf) return;
    if (c->rbuf_heap) free(c->rbuf);
    else http_slab_release(&w->slab, c->rbuf_slab_idx);
    c->rbuf = nullptr;
    c->rbuf_slab_idx = UINT32_MAX;
    c->rbuf_len = 0;
    c->rbuf_cap = 0;
    c->rbuf_heap = false;
}

static void http_conn_release_wbuf(HttpWorker* w, HttpConn* c) {
    if (!c->wbuf) return;
    if (c->wbuf_heap) free(c->wbuf);
    else http_slab_release(&w->slab, c->wbuf_slab_idx);
    c->wbuf = nullptr;
    c->wbuf_slab_idx = UINT32_MAX;
    c->wbuf_len = 0;
    c->wbuf_sent = 0;
    c->wbuf_cap = 0;
    c->wbuf_heap = false;
}

static bool http_conn_ensure_rbuf(HttpWorker* w, HttpConn* c, size_t min_cap) {
    if (c->rbuf && c->rbuf_cap >= min_cap) return true;
    if (!c->rbuf && min_cap <= RCRE_HTTP_SLAB_BUF_SIZE) {
        uint32_t idx = UINT32_MAX;
        uint8_t* p = http_slab_acquire(&w->slab, &idx);
        if (p) {
            c->rbuf = p;
            c->rbuf_slab_idx = idx;
            c->rbuf_cap = RCRE_HTTP_SLAB_BUF_SIZE;
            c->rbuf_heap = false;
            return true;
        }
    }
    size_t new_cap = (min_cap < RCRE_HTTP_SLAB_BUF_SIZE) ? RCRE_HTTP_SLAB_BUF_SIZE : min_cap;
    uint8_t* nb = (uint8_t*)malloc(new_cap);
    if (!nb) return false;
    if (c->rbuf && c->rbuf_len > 0) memcpy(nb, c->rbuf, c->rbuf_len);
    size_t old_len = c->rbuf_len;
    http_conn_release_rbuf(w, c);
    c->rbuf = nb;
    c->rbuf_cap = new_cap;
    c->rbuf_len = old_len;
    c->rbuf_heap = true;
    return true;
}

static void http_conn_close(HttpWorker* w, HttpConn* c) {
    if (c->fd < 0) return;
#if defined(RCRE_HTTP_USE_EPOLL)
    (void)epoll_ctl(w->poll_fd, EPOLL_CTL_DEL, c->fd, nullptr);
#endif
    close(c->fd);
    c->fd = -1;
    http_conn_release_rbuf(w, c);
    http_conn_release_wbuf(w, c);
    c->write_armed = false;
    c->close_after_write = false;
    w->free_slots[w->free_top++] = c->slot_id;
}

static void http_worker_register_client(HttpWorker* w, int fd) {
    if (w->free_top == 0) {
        close(fd);
        return;
    }
    uint32_t slot = w->free_slots[--w->free_top];
    HttpConn* c = &w->conns[slot];
    memset(c, 0, sizeof(*c));
    c->fd = fd;
    c->slot_id = slot;
    c->rbuf_slab_idx = UINT32_MAX;
    c->wbuf_slab_idx = UINT32_MAX;

#if defined(RCRE_HTTP_USE_KQUEUE)
    struct kevent ev;
    EV_SET(&ev, (uintptr_t)fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, c);
    if (kevent(w->poll_fd, &ev, 1, nullptr, 0, nullptr) < 0) {
        http_conn_close(w, c);
    }
#elif defined(RCRE_HTTP_USE_EPOLL)
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.events = EPOLLIN | EPOLLRDHUP;
    ev.data.ptr = c;
    if (epoll_ctl(w->poll_fd, EPOLL_CTL_ADD, fd, &ev) < 0) {
        http_conn_close(w, c);
    }
#endif
}

static void http_conn_arm_write(HttpWorker* w, HttpConn* c) {
    if (c->write_armed) return;
#if defined(RCRE_HTTP_USE_KQUEUE)
    struct kevent ev;
    EV_SET(&ev, (uintptr_t)c->fd, EVFILT_WRITE, EV_ADD | EV_ENABLE, 0, 0, c);
    if (kevent(w->poll_fd, &ev, 1, nullptr, 0, nullptr) == 0) {
        c->write_armed = true;
    } else {
        http_conn_close(w, c);
    }
#elif defined(RCRE_HTTP_USE_EPOLL)
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.events = EPOLLIN | EPOLLOUT | EPOLLRDHUP;
    ev.data.ptr = c;
    if (epoll_ctl(w->poll_fd, EPOLL_CTL_MOD, c->fd, &ev) == 0) {
        c->write_armed = true;
    } else {
        http_conn_close(w, c);
    }
#endif
}

static void http_conn_disarm_write(HttpWorker* w, HttpConn* c) {
    if (!c->write_armed) return;
#if defined(RCRE_HTTP_USE_KQUEUE)
    struct kevent ev;
    EV_SET(&ev, (uintptr_t)c->fd, EVFILT_WRITE, EV_DELETE, 0, 0, nullptr);
    (void)kevent(w->poll_fd, &ev, 1, nullptr, 0, nullptr);
#elif defined(RCRE_HTTP_USE_EPOLL)
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.events = EPOLLIN | EPOLLRDHUP;
    ev.data.ptr = c;
    (void)epoll_ctl(w->poll_fd, EPOLL_CTL_MOD, c->fd, &ev);
#endif
    c->write_armed = false;
}

static bool http_conn_send(HttpWorker* w, HttpConn* c, const uint8_t* data, size_t len) {
    if (len == 0) return true;
    if (c->wbuf_len == 0) {
        ssize_t n = http_sock_write(c->fd, data, len);
        if (n == (ssize_t)len) return true;
        size_t written = 0;
        if (n > 0) written = (size_t)n;
        else if (errno != EAGAIN && errno != EWOULDBLOCK) return false;

        size_t rem = len - written;
        uint32_t idx = UINT32_MAX;
        if (rem <= RCRE_HTTP_SLAB_BUF_SIZE) {
            c->wbuf = http_slab_acquire(&w->slab, &idx);
            if (c->wbuf) {
                c->wbuf_slab_idx = idx;
                c->wbuf_cap = RCRE_HTTP_SLAB_BUF_SIZE;
                c->wbuf_heap = false;
            }
        }
        if (!c->wbuf) {
            size_t cap = (rem < RCRE_HTTP_SLAB_BUF_SIZE) ? RCRE_HTTP_SLAB_BUF_SIZE : rem;
            c->wbuf = (uint8_t*)malloc(cap);
            if (!c->wbuf) return false;
            c->wbuf_cap = cap;
            c->wbuf_heap = true;
        }
        memcpy(c->wbuf, data + written, rem);
        c->wbuf_len = rem;
        c->wbuf_sent = 0;
        http_conn_arm_write(w, c);
        return (c->fd >= 0);
    }

    size_t pending = c->wbuf_len - c->wbuf_sent;
    if (c->wbuf_sent > 0 && pending + len <= c->wbuf_cap) {
        memmove(c->wbuf, c->wbuf + c->wbuf_sent, pending);
        c->wbuf_len = pending;
        c->wbuf_sent = 0;
    }
    if (c->wbuf_len + len > c->wbuf_cap) {
        size_t new_cap = c->wbuf_cap * 2U;
        while (new_cap < pending + len) new_cap *= 2U;
        uint8_t* nb = (uint8_t*)malloc(new_cap);
        if (!nb) return false;
        memcpy(nb, c->wbuf + c->wbuf_sent, pending);
        http_conn_release_wbuf(w, c);
        c->wbuf = nb;
        c->wbuf_cap = new_cap;
        c->wbuf_len = pending;
        c->wbuf_sent = 0;
        c->wbuf_heap = true;
    }
    memcpy(c->wbuf + c->wbuf_len, data, len);
    c->wbuf_len += len;
    http_conn_arm_write(w, c);
    return (c->fd >= 0);
}

static void http_conn_on_writable(HttpWorker* w, HttpConn* c) {
    while (c->wbuf_sent < c->wbuf_len) {
        ssize_t n = http_sock_write(c->fd, c->wbuf + c->wbuf_sent, c->wbuf_len - c->wbuf_sent);
        if (n > 0) {
            c->wbuf_sent += (size_t)n;
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return;
        } else {
            http_conn_close(w, c);
            return;
        }
    }
    http_conn_release_wbuf(w, c);
    http_conn_disarm_write(w, c);
    if (c->close_after_write) http_conn_close(w, c);
}

static void http_conn_on_readable(HttpWorker* w, HttpConn* c) {
    for (;;) {
        if (!http_conn_ensure_rbuf(w, c, c->rbuf_len + 4096U)) {
            http_conn_close(w, c);
            return;
        }
        ssize_t n = read(c->fd, c->rbuf + c->rbuf_len, c->rbuf_cap - c->rbuf_len);
        if (n > 0) {
            c->rbuf_len += (size_t)n;
        } else if (n == 0) {
            http_conn_close(w, c);
            return;
        } else {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            http_conn_close(w, c);
            return;
        }

        while (c->rbuf_len > 0) {
            RCRE_HttpRequest req;
            RCRE_HttpParseStatus pr = rcre_http_parse_request(
                c->rbuf, c->rbuf_len, w->srv->max_body_bytes, &req);
            if (pr == RCRE_HTTP_PARSE_INCOMPLETE) {
                if (req.expected_total > c->rbuf_cap) {
                    if (req.expected_total > RCRE_HTTP_MAX_HEADER_BYTES + w->srv->max_body_bytes ||
                        !http_conn_ensure_rbuf(w, c, req.expected_total)) {
                        http_conn_close(w, c);
                        return;
                    }
                }
                break;
            }
            if (pr == RCRE_HTTP_PARSE_ERROR) {
                RCRE_StrBuf out = rcre_strbuf_init(w->tx_staging, RCRE_HTTP_TX_STAGING_CAP);
                size_t out_len = rcre_http_write_response(
                    &out, RCRE_STR("HTTP/1.1 400 Bad Request"), RCRE_STR("application/json"),
                    w->date_hdr, false, RCRE_STR("{\"error\":\"bad_request\"}"));
                (void)http_conn_send(w, c, out.data, out_len);
                if (c->wbuf_len == 0) http_conn_close(w, c);
                else c->close_after_write = true;
                return;
            }

            RCRE_HttpContext ctx = {
                .scratch   = w->scratch,
                .worker_id = w->id,
                .date_hdr  = w->date_hdr,
                .user_ctx  = w->srv->user_ctx
            };
            RCRE_StrBuf out = rcre_strbuf_init(w->tx_staging, RCRE_HTTP_TX_STAGING_CAP);
            size_t out_len = w->srv->handler(&req, &ctx, &out);
            rcre_arena_reset(w->scratch);

            if (req.consumed == c->rbuf_len) {
                c->rbuf_len = 0;
            } else {
                size_t rem = c->rbuf_len - req.consumed;
                memmove(c->rbuf, c->rbuf + req.consumed, rem);
                c->rbuf_len = rem;
            }

            if (!http_conn_send(w, c, out.data, out_len)) {
                http_conn_close(w, c);
                return;
            }
            if (!req.keep_alive) {
                if (c->wbuf_len == 0) http_conn_close(w, c);
                else c->close_after_write = true;
                return;
            }
        }

        if (c->rbuf_len == 0 && (size_t)n < 4096U) break;
    }

    if (c->fd >= 0 && c->rbuf_len == 0 && c->rbuf != nullptr) {
        http_conn_release_rbuf(w, c);
    }
}

static void http_worker_drain_ring(HttpWorker* w) {
    int fd = -1;
    while (http_ring_pop(&w->ring, &fd)) {
        http_worker_register_client(w, fd);
    }
}

static void http_worker_accept_inline(HttpWorker* w) {
    for (;;) {
        int cfd = accept(w->inline_listen_fd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        http_configure_client(cfd);
        http_worker_register_client(w, cfd);
    }
}

static void http_worker_wake(HttpWorker* w) {
#if defined(RCRE_HTTP_USE_KQUEUE)
    struct kevent ev;
    EV_SET(&ev, RCRE_HTTP_WAKE_IDENT, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
    (void)kevent(w->poll_fd, &ev, 1, nullptr, 0, nullptr);
#elif defined(RCRE_HTTP_USE_EPOLL)
    uint64_t one = 1;
    (void)write(w->wake_fd, &one, sizeof(one));
#endif
}

static void* http_worker_main(void* arg) {
    HttpWorker* w = (HttpWorker*)arg;
#if defined(__APPLE__)
    (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    http_refresh_date(w, time(nullptr));

#if defined(RCRE_HTTP_USE_KQUEUE)
    struct kevent events[RCRE_HTTP_EVENT_BATCH];
    struct timespec timeout = { .tv_sec = 0, .tv_nsec = 100000000L };

    while (!atomic_load_explicit(&w->srv->stop, memory_order_acquire)) {
        atomic_store_explicit(&w->sleeping, true, memory_order_release);
        if (atomic_load_explicit(&w->ring.head, memory_order_acquire) !=
            atomic_load_explicit(&w->ring.tail, memory_order_acquire)) {
            atomic_store_explicit(&w->sleeping, false, memory_order_relaxed);
            http_worker_drain_ring(w);
            continue;
        }
        int nev = kevent(w->poll_fd, nullptr, 0, events, (int)RCRE_HTTP_EVENT_BATCH, &timeout);
        atomic_store_explicit(&w->sleeping, false, memory_order_relaxed);
        if (nev < 0) {
            if (errno == EINTR) continue;
            break;
        }
        http_refresh_date(w, time(nullptr));
        for (int i = 0; i < nev; ++i) {
            struct kevent* ev = &events[i];
            if (ev->filter == EVFILT_USER && ev->ident == RCRE_HTTP_WAKE_IDENT) {
                http_worker_drain_ring(w);
                continue;
            }
            if (w->inline_listen_fd >= 0 && (int)ev->ident == w->inline_listen_fd) {
                http_worker_accept_inline(w);
                continue;
            }
            HttpConn* c = (HttpConn*)ev->udata;
            if (!c || c->fd < 0) continue;
            if (ev->flags & EV_ERROR) {
                http_conn_close(w, c);
                continue;
            }
            if (ev->filter == EVFILT_READ) {
                if ((ev->flags & EV_EOF) && ev->data == 0) http_conn_close(w, c);
                else http_conn_on_readable(w, c);
            } else if (ev->filter == EVFILT_WRITE) {
                http_conn_on_writable(w, c);
            }
        }
    }
#elif defined(RCRE_HTTP_USE_EPOLL)
    struct epoll_event events[RCRE_HTTP_EVENT_BATCH];
    while (!atomic_load_explicit(&w->srv->stop, memory_order_acquire)) {
        atomic_store_explicit(&w->sleeping, true, memory_order_release);
        if (atomic_load_explicit(&w->ring.head, memory_order_acquire) !=
            atomic_load_explicit(&w->ring.tail, memory_order_acquire)) {
            atomic_store_explicit(&w->sleeping, false, memory_order_relaxed);
            http_worker_drain_ring(w);
            continue;
        }
        int nev = epoll_wait(w->poll_fd, events, (int)RCRE_HTTP_EVENT_BATCH, 100);
        atomic_store_explicit(&w->sleeping, false, memory_order_relaxed);
        if (nev < 0) {
            if (errno == EINTR) continue;
            break;
        }
        http_refresh_date(w, time(nullptr));
        for (int i = 0; i < nev; ++i) {
            void* ptr = events[i].data.ptr;
            if (ptr == &w->wake_fd) {
                uint64_t val = 0;
                while (read(w->wake_fd, &val, sizeof(val)) > 0) {}
                http_worker_drain_ring(w);
                continue;
            }
            if (ptr == &w->inline_listen_fd) {
                http_worker_accept_inline(w);
                continue;
            }
            HttpConn* c = (HttpConn*)ptr;
            if (!c || c->fd < 0) continue;
            uint32_t mask = events[i].events;
            if (mask & (EPOLLERR | EPOLLHUP)) {
                http_conn_close(w, c);
                continue;
            }
            if (mask & EPOLLIN) {
                http_conn_on_readable(w, c);
            }
            if (c->fd >= 0 && (mask & EPOLLOUT)) {
                http_conn_on_writable(w, c);
            }
        }
    }
#endif
    return nullptr;
}

static bool http_worker_init(
    HttpWorker* w,
    size_t id,
    RCRE_HttpServer* srv,
    size_t arena_bytes,
    size_t max_conns,
    int inline_listen_fd
) {
    memset(w, 0, sizeof(*w));
    w->id = id;
    w->srv = srv;
    w->inline_listen_fd = inline_listen_fd;
    w->max_conns = max_conns;
    w->poll_fd = -1;
#if defined(RCRE_HTTP_USE_EPOLL)
    w->wake_fd = -1;
#endif

#if defined(RCRE_HTTP_USE_KQUEUE)
    w->poll_fd = kqueue();
    if (w->poll_fd < 0) return false;
#elif defined(RCRE_HTTP_USE_EPOLL)
    w->poll_fd = epoll_create1(EPOLL_CLOEXEC);
    w->wake_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (w->poll_fd < 0 || w->wake_fd < 0) return false;
#endif

    w->scratch = rcre_arena_create(arena_bytes);
    if (!w->scratch) return false;
    if (!http_slab_init(&w->slab, RCRE_HTTP_SLAB_BUFS_WKR)) return false;

    w->conns = (HttpConn*)calloc(max_conns, sizeof(HttpConn));
    w->free_slots = (uint32_t*)malloc(max_conns * sizeof(uint32_t));
    w->tx_staging = (uint8_t*)aligned_alloc(64, RCRE_HTTP_TX_STAGING_CAP);
    if (!w->conns || !w->free_slots || !w->tx_staging) return false;

    for (size_t i = 0; i < max_conns; ++i) {
        w->conns[i].fd = -1;
        w->conns[i].slot_id = (uint32_t)i;
        w->free_slots[i] = (uint32_t)(max_conns - 1U - i);
    }
    w->free_top = max_conns;

#if defined(RCRE_HTTP_USE_KQUEUE)
    struct kevent ev[2];
    int nev = 0;
    EV_SET(&ev[nev++], RCRE_HTTP_WAKE_IDENT, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr);
    if (inline_listen_fd >= 0) {
        EV_SET(&ev[nev++], (uintptr_t)inline_listen_fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
    }
    if (kevent(w->poll_fd, ev, nev, nullptr, 0, nullptr) < 0) return false;
#elif defined(RCRE_HTTP_USE_EPOLL)
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.events = EPOLLIN;
    ev.data.ptr = &w->wake_fd;
    if (epoll_ctl(w->poll_fd, EPOLL_CTL_ADD, w->wake_fd, &ev) < 0) return false;
    if (inline_listen_fd >= 0) {
        ev.data.ptr = &w->inline_listen_fd;
        if (epoll_ctl(w->poll_fd, EPOLL_CTL_ADD, inline_listen_fd, &ev) < 0) return false;
    }
#endif
    return true;
}

static void http_worker_destroy(HttpWorker* w) {
    if (w->conns) {
        for (size_t i = 0; i < w->max_conns; ++i) {
            if (w->conns[i].fd >= 0) http_conn_close(w, &w->conns[i]);
        }
    }
    int fd = -1;
    while (http_ring_pop(&w->ring, &fd)) close(fd);
    if (w->poll_fd >= 0) close(w->poll_fd);
#if defined(RCRE_HTTP_USE_EPOLL)
    if (w->wake_fd >= 0) close(w->wake_fd);
#endif
    if (w->scratch) rcre_arena_destroy(w->scratch);
    http_slab_destroy(&w->slab);
    free(w->conns);
    free(w->free_slots);
    free(w->tx_staging);
}

/* ========================================================================= */
/* Public Server Lifecycle                                                   */
/* ========================================================================= */

RCRE_HttpServer* rcre_http_server_create(const RCRE_HttpServerConfig* cfg) {
    if (!cfg || !cfg->handler) return nullptr;
    rcre_hal_init();

    RCRE_HttpServer* srv = (RCRE_HttpServer*)calloc(1, sizeof(RCRE_HttpServer));
    if (!srv) return nullptr;
    srv->listen_fd = -1;
    srv->accept_poll_fd = -1;
#if defined(RCRE_HTTP_USE_EPOLL)
    srv->accept_wake_fd = -1;
#endif
    srv->num_workers    = (cfg->num_workers > 0) ? cfg->num_workers : 1U;
    if (srv->num_workers > 64U) srv->num_workers = 64U;
    srv->max_body_bytes = (cfg->max_body_bytes > 0) ? cfg->max_body_bytes : 1048576U;
    srv->handler        = cfg->handler;
    srv->user_ctx       = cfg->user_ctx;

    size_t arena_bytes = (cfg->worker_arena_bytes > 0) ? cfg->worker_arena_bytes : (4U * 1024U * 1024U);
    size_t max_conns   = (cfg->max_conns_per_worker > 0) ? cfg->max_conns_per_worker : 4096U;
    const char* host   = cfg->host ? cfg->host : "127.0.0.1";

    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        rcre_http_server_destroy(srv);
        return nullptr;
    }
    int one = 1;
    (void)setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#ifdef SO_REUSEPORT
    (void)setsockopt(lfd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
#endif
    (void)http_set_nonblocking(lfd);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(cfg->port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1 ||
        bind(lfd, (struct sockaddr*)&addr, sizeof(addr)) < 0 ||
        listen(lfd, 4096) < 0) {
        close(lfd);
        rcre_http_server_destroy(srv);
        return nullptr;
    }

    struct sockaddr_in bound;
    socklen_t blen = sizeof(bound);
    if (getsockname(lfd, (struct sockaddr*)&bound, &blen) == 0) {
        srv->bound_port = ntohs(bound.sin_port);
    } else {
        srv->bound_port = cfg->port;
    }
    srv->listen_fd = lfd;

    srv->workers = (HttpWorker*)calloc(srv->num_workers, sizeof(HttpWorker));
    if (!srv->workers) {
        rcre_http_server_destroy(srv);
        return nullptr;
    }
    for (size_t i = 0; i < srv->num_workers; ++i) {
        int inline_fd = (srv->num_workers == 1U) ? lfd : -1;
        if (!http_worker_init(&srv->workers[i], i, srv, arena_bytes, max_conns, inline_fd)) {
            rcre_http_server_destroy(srv);
            return nullptr;
        }
    }

    if (srv->num_workers > 1U) {
#if defined(RCRE_HTTP_USE_KQUEUE)
        srv->accept_poll_fd = kqueue();
        if (srv->accept_poll_fd < 0) {
            rcre_http_server_destroy(srv);
            return nullptr;
        }
        struct kevent ev[2];
        EV_SET(&ev[0], (uintptr_t)lfd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
        EV_SET(&ev[1], RCRE_HTTP_WAKE_IDENT, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr);
        if (kevent(srv->accept_poll_fd, ev, 2, nullptr, 0, nullptr) < 0) {
            rcre_http_server_destroy(srv);
            return nullptr;
        }
#elif defined(RCRE_HTTP_USE_EPOLL)
        srv->accept_poll_fd = epoll_create1(EPOLL_CLOEXEC);
        srv->accept_wake_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (srv->accept_poll_fd < 0 || srv->accept_wake_fd < 0) {
            rcre_http_server_destroy(srv);
            return nullptr;
        }
        struct epoll_event ev;
        memset(&ev, 0, sizeof(ev));
        ev.events = EPOLLIN;
        ev.data.fd = lfd;
        if (epoll_ctl(srv->accept_poll_fd, EPOLL_CTL_ADD, lfd, &ev) < 0) {
            rcre_http_server_destroy(srv);
            return nullptr;
        }
        ev.data.fd = srv->accept_wake_fd;
        if (epoll_ctl(srv->accept_poll_fd, EPOLL_CTL_ADD, srv->accept_wake_fd, &ev) < 0) {
            rcre_http_server_destroy(srv);
            return nullptr;
        }
#endif
    }

    return srv;
}

uint16_t rcre_http_server_port(const RCRE_HttpServer* srv) {
    return srv ? srv->bound_port : 0;
}

void rcre_http_server_stop(RCRE_HttpServer* srv) {
    if (!srv) return;
    atomic_store_explicit(&srv->stop, true, memory_order_release);
    if (srv->num_workers == 1U && srv->workers) {
        http_worker_wake(&srv->workers[0]);
    } else if (srv->accept_poll_fd >= 0) {
#if defined(RCRE_HTTP_USE_KQUEUE)
        struct kevent ev;
        EV_SET(&ev, RCRE_HTTP_WAKE_IDENT, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
        (void)kevent(srv->accept_poll_fd, &ev, 1, nullptr, 0, nullptr);
#elif defined(RCRE_HTTP_USE_EPOLL)
        if (srv->accept_wake_fd >= 0) {
            uint64_t one = 1;
            (void)write(srv->accept_wake_fd, &one, sizeof(one));
        }
#endif
    }
}

int rcre_http_server_run(RCRE_HttpServer* srv) {
    if (!srv) return RCRE_ERR_INVALID_ARG;
    signal(SIGPIPE, SIG_IGN);

    if (srv->num_workers == 1U) {
        http_worker_main(&srv->workers[0]);
        return RCRE_OK;
    }

    size_t started = 0;
    for (size_t i = 0; i < srv->num_workers; ++i) {
        if (pthread_create(&srv->workers[i].thread, nullptr, http_worker_main, &srv->workers[i]) != 0) {
            rcre_http_server_stop(srv);
            break;
        }
        started++;
    }

    size_t rr = 0;
#if defined(RCRE_HTTP_USE_KQUEUE)
    struct kevent out_ev[8];
    struct timespec timeout = { .tv_sec = 0, .tv_nsec = 100000000L };
    while (!atomic_load_explicit(&srv->stop, memory_order_acquire)) {
        int nev = kevent(srv->accept_poll_fd, nullptr, 0, out_ev, 8, &timeout);
        if (nev <= 0) continue;
        for (;;) {
            int cfd = accept(srv->listen_fd, nullptr, nullptr);
            if (cfd < 0) {
                if (errno == EINTR) continue;
                break;
            }
            http_configure_client(cfd);
            HttpWorker* target = &srv->workers[rr % srv->num_workers];
            rr++;
            if (!http_ring_push(&target->ring, cfd)) {
                close(cfd);
                continue;
            }
            if (atomic_load_explicit(&target->sleeping, memory_order_acquire)) {
                http_worker_wake(target);
            }
        }
    }
#elif defined(RCRE_HTTP_USE_EPOLL)
    struct epoll_event out_ev[8];
    while (!atomic_load_explicit(&srv->stop, memory_order_acquire)) {
        int nev = epoll_wait(srv->accept_poll_fd, out_ev, 8, 100);
        if (nev <= 0) continue;
        for (;;) {
            int cfd = accept(srv->listen_fd, nullptr, nullptr);
            if (cfd < 0) {
                if (errno == EINTR) continue;
                break;
            }
            http_configure_client(cfd);
            HttpWorker* target = &srv->workers[rr % srv->num_workers];
            rr++;
            if (!http_ring_push(&target->ring, cfd)) {
                close(cfd);
                continue;
            }
            if (atomic_load_explicit(&target->sleeping, memory_order_acquire)) {
                http_worker_wake(target);
            }
        }
    }
#endif

    for (size_t i = 0; i < started; ++i) {
        http_worker_wake(&srv->workers[i]);
    }
    for (size_t i = 0; i < started; ++i) {
        pthread_join(srv->workers[i].thread, nullptr);
    }
    return RCRE_OK;
}

void rcre_http_server_destroy(RCRE_HttpServer* srv) {
    if (!srv) return;
    if (srv->accept_poll_fd >= 0) close(srv->accept_poll_fd);
#if defined(RCRE_HTTP_USE_EPOLL)
    if (srv->accept_wake_fd >= 0) close(srv->accept_wake_fd);
#endif
    if (srv->listen_fd >= 0) close(srv->listen_fd);
    if (srv->workers) {
        for (size_t i = 0; i < srv->num_workers; ++i) {
            http_worker_destroy(&srv->workers[i]);
        }
        free(srv->workers);
    }
    free(srv);
}
