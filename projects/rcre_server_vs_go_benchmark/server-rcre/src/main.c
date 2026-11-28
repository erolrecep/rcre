/**
 * @file main.c
 * @brief High-Performance Zero-Malloc ISO C23 HTTP/1.1 Web Server on RCRE + kqueue
 *
 * Architecture:
 *   - Shared-nothing worker-per-core kqueue reactor (1-worker folds accept inline;
 *     N>1 workers use a dedicated non-blocking acceptor dispatching accepted FDs
 *     round-robin over lock-free SPSC ring buffers + EVFILT_USER wakeups).
 *   - Zero heap allocations on the request hot path:
 *       * Pre-allocated flat connection table per worker.
 *       * Slab-pooled 16 KB I/O buffers attached only while reading/writing
 *         (idle keep-alive connections consume ~96 bytes each!).
 *       * Per-worker 4 MB RCRE_Arena rewound in O(1) after every request.
 *   - Immediate non-blocking write fast-path (EVFILT_WRITE enabled only on EAGAIN).
 *   - Per-worker 1-second cached RFC-7231 Date header (zero locks, zero syscalls/req).
 */

#include "rcre.h"
#include "rcre_str.h"
#include "http_parser.h"
#include "handlers.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

/* ========================================================================= */
/* Configuration Constants                                                   */
/* ========================================================================= */

constexpr size_t SLAB_BUF_SIZE       = 16384U;
constexpr size_t SLAB_BUFS_PER_WKR   = 512U;
constexpr size_t MAX_CONNS_PER_WKR   = 8192U;
constexpr size_t ACCEPT_RING_CAP     = 8192U; /* Must be power of 2 */
constexpr size_t KEVENT_BATCH_SIZE   = 256U;
constexpr size_t WORKER_ARENA_BYTES  = 4U * 1024U * 1024U;
constexpr uintptr_t WAKEUP_IDENT     = 0xBEEF0001ULL;

static atomic_bool g_stop = false;

static void on_signal(int sig) {
    (void)sig;
    atomic_store_explicit(&g_stop, true, memory_order_release);
}

/* ========================================================================= */
/* Per-Worker Slab Buffer Pool (Zero-Malloc I/O Buffers)                     */
/* ========================================================================= */

typedef struct BufSlab {
    uint8_t*  backing;
    uint32_t* free_stack;
    size_t    free_top;
    size_t    count;
} BufSlab;

static bool buf_slab_init(BufSlab* slab, size_t count) {
    slab->backing = (uint8_t*)aligned_alloc(64, count * SLAB_BUF_SIZE);
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

static void buf_slab_destroy(BufSlab* slab) {
    free(slab->backing);
    free(slab->free_stack);
    memset(slab, 0, sizeof(*slab));
}

static inline uint8_t* buf_slab_acquire(BufSlab* slab, uint32_t* out_idx) {
    if (slab->free_top == 0) {
        *out_idx = UINT32_MAX;
        return nullptr;
    }
    uint32_t idx = slab->free_stack[--slab->free_top];
    *out_idx = idx;
    return slab->backing + (size_t)idx * SLAB_BUF_SIZE;
}

static inline void buf_slab_release(BufSlab* slab, uint32_t idx) {
    if (idx != UINT32_MAX && slab->free_top < slab->count) {
        slab->free_stack[slab->free_top++] = idx;
    }
}

/* ========================================================================= */
/* Lock-Free SPSC FD Ring (Acceptor -> Worker)                               */
/* ========================================================================= */

typedef struct FdRing {
    alignas(64) atomic_size_t head;
    alignas(64) atomic_size_t tail;
    int fds[ACCEPT_RING_CAP];
} FdRing;

static inline bool fd_ring_push(FdRing* r, int fd) {
    size_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    size_t next = (t + 1U) & (ACCEPT_RING_CAP - 1U);
    if (next == atomic_load_explicit(&r->head, memory_order_acquire)) {
        return false;
    }
    r->fds[t] = fd;
    atomic_store_explicit(&r->tail, next, memory_order_release);
    return true;
}

static inline bool fd_ring_pop(FdRing* r, int* out_fd) {
    size_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    if (h == atomic_load_explicit(&r->tail, memory_order_acquire)) {
        return false;
    }
    *out_fd = r->fds[h];
    atomic_store_explicit(&r->head, (h + 1U) & (ACCEPT_RING_CAP - 1U), memory_order_release);
    return true;
}

/* ========================================================================= */
/* Connection & Worker State                                                 */
/* ========================================================================= */

typedef struct Conn {
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
} Conn;

typedef struct Worker {
    int             id;
    int             kq;
    int             inline_listen_fd; /* >=0 only when workers == 1 */
    const AppState* app;
    pthread_t       thread;

    FdRing          ring;
    atomic_bool     sleeping;

    RCRE_Arena*       scratch;
    WorkerBM25Scratch bm25_sc;
    BufSlab           slab;
    Conn*           conns;
    uint32_t*       free_slots;
    size_t          free_top;
    uint8_t*        tx_staging; /* Reusable 24 KB staging buffer for immediate write fast path */

    time_t          cached_sec;
    char            date_hdr[32];
} Worker;

static void refresh_date_header(Worker* w, time_t now) {
    if (now == w->cached_sec) return;
    w->cached_sec = now;
    struct tm gmt;
    gmtime_r(&now, &gmt);
    static const char* days[7]   = { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" };
    static const char* months[12] = {
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    };
    snprintf(w->date_hdr, sizeof(w->date_hdr),
             "%s, %02d %s %04d %02d:%02d:%02d GMT",
             days[gmt.tm_wday % 7], gmt.tm_mday, months[gmt.tm_mon % 12],
             gmt.tm_year + 1900, gmt.tm_hour, gmt.tm_min, gmt.tm_sec);
}

static bool set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return false;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static void configure_client_socket(int fd) {
    (void)set_nonblocking(fd);
    int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}

static void conn_release_rbuf(Worker* w, Conn* c) {
    if (!c->rbuf) return;
    if (c->rbuf_heap) {
        free(c->rbuf);
    } else {
        buf_slab_release(&w->slab, c->rbuf_slab_idx);
    }
    c->rbuf = nullptr;
    c->rbuf_slab_idx = UINT32_MAX;
    c->rbuf_len = 0;
    c->rbuf_cap = 0;
    c->rbuf_heap = false;
}

static void conn_release_wbuf(Worker* w, Conn* c) {
    if (!c->wbuf) return;
    if (c->wbuf_heap) {
        free(c->wbuf);
    } else {
        buf_slab_release(&w->slab, c->wbuf_slab_idx);
    }
    c->wbuf = nullptr;
    c->wbuf_slab_idx = UINT32_MAX;
    c->wbuf_len = 0;
    c->wbuf_sent = 0;
    c->wbuf_cap = 0;
    c->wbuf_heap = false;
}

static bool conn_ensure_rbuf(Worker* w, Conn* c, size_t min_cap) {
    if (c->rbuf && c->rbuf_cap >= min_cap) return true;
    if (!c->rbuf && min_cap <= SLAB_BUF_SIZE) {
        uint32_t idx = UINT32_MAX;
        uint8_t* p = buf_slab_acquire(&w->slab, &idx);
        if (p) {
            c->rbuf = p;
            c->rbuf_slab_idx = idx;
            c->rbuf_cap = SLAB_BUF_SIZE;
            c->rbuf_heap = false;
            return true;
        }
    }
    size_t new_cap = (min_cap < SLAB_BUF_SIZE) ? SLAB_BUF_SIZE : min_cap;
    uint8_t* nb = (uint8_t*)malloc(new_cap);
    if (!nb) return false;
    if (c->rbuf && c->rbuf_len > 0) {
        memcpy(nb, c->rbuf, c->rbuf_len);
    }
    size_t old_len = c->rbuf_len;
    conn_release_rbuf(w, c);
    c->rbuf = nb;
    c->rbuf_cap = new_cap;
    c->rbuf_len = old_len;
    c->rbuf_heap = true;
    return true;
}

static void conn_close(Worker* w, Conn* c) {
    if (c->fd < 0) return;
    close(c->fd); /* kqueue automatically removes closed descriptors */
    c->fd = -1;
    conn_release_rbuf(w, c);
    conn_release_wbuf(w, c);
    c->write_armed = false;
    c->close_after_write = false;
    w->free_slots[w->free_top++] = c->slot_id;
}

static void worker_register_client(Worker* w, int fd) {
    if (w->free_top == 0) {
        close(fd);
        return;
    }
    uint32_t slot = w->free_slots[--w->free_top];
    Conn* c = &w->conns[slot];
    memset(c, 0, sizeof(*c));
    c->fd = fd;
    c->slot_id = slot;
    c->rbuf_slab_idx = UINT32_MAX;
    c->wbuf_slab_idx = UINT32_MAX;

    struct kevent ev;
    EV_SET(&ev, (uintptr_t)fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, c);
    if (kevent(w->kq, &ev, 1, nullptr, 0, nullptr) < 0) {
        conn_close(w, c);
    }
}

static void conn_arm_write(Worker* w, Conn* c) {
    if (c->write_armed) return;
    struct kevent ev;
    EV_SET(&ev, (uintptr_t)c->fd, EVFILT_WRITE, EV_ADD | EV_ENABLE, 0, 0, c);
    if (kevent(w->kq, &ev, 1, nullptr, 0, nullptr) == 0) {
        c->write_armed = true;
    } else {
        conn_close(w, c);
    }
}

static void conn_disarm_write(Worker* w, Conn* c) {
    if (!c->write_armed) return;
    struct kevent ev;
    EV_SET(&ev, (uintptr_t)c->fd, EVFILT_WRITE, EV_DELETE, 0, 0, nullptr);
    (void)kevent(w->kq, &ev, 1, nullptr, 0, nullptr);
    c->write_armed = false;
}

/* Queue bytes for writing (fast-path immediate write() when nothing pending) */
static bool conn_send(Worker* w, Conn* c, const uint8_t* data, size_t len) {
    if (len == 0) return true;

    /* Fast path: no buffered unsent bytes yet */
    if (c->wbuf_len == 0) {
        ssize_t n = write(c->fd, data, len);
        if (n == (ssize_t)len) {
            return true;
        }
        size_t written = 0;
        if (n > 0) {
            written = (size_t)n;
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
            return false;
        }
        size_t rem = len - written;
        uint32_t idx = UINT32_MAX;
        if (rem <= SLAB_BUF_SIZE) {
            c->wbuf = buf_slab_acquire(&w->slab, &idx);
            if (c->wbuf) {
                c->wbuf_slab_idx = idx;
                c->wbuf_cap = SLAB_BUF_SIZE;
                c->wbuf_heap = false;
            }
        }
        if (!c->wbuf) {
            size_t cap = (rem < SLAB_BUF_SIZE) ? SLAB_BUF_SIZE : rem;
            c->wbuf = (uint8_t*)malloc(cap);
            if (!c->wbuf) return false;
            c->wbuf_cap = cap;
            c->wbuf_heap = true;
        }
        memcpy(c->wbuf, data + written, rem);
        c->wbuf_len = rem;
        c->wbuf_sent = 0;
        conn_arm_write(w, c);
        return (c->fd >= 0);
    }

    /* Slow path: append to existing pending write buffer */
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
        conn_release_wbuf(w, c);
        c->wbuf = nb;
        c->wbuf_cap = new_cap;
        c->wbuf_len = pending;
        c->wbuf_sent = 0;
        c->wbuf_heap = true;
    }
    memcpy(c->wbuf + c->wbuf_len, data, len);
    c->wbuf_len += len;
    conn_arm_write(w, c);
    return (c->fd >= 0);
}

static void conn_on_writable(Worker* w, Conn* c) {
    while (c->wbuf_sent < c->wbuf_len) {
        ssize_t n = write(c->fd, c->wbuf + c->wbuf_sent, c->wbuf_len - c->wbuf_sent);
        if (n > 0) {
            c->wbuf_sent += (size_t)n;
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return;
        } else {
            conn_close(w, c);
            return;
        }
    }
    conn_release_wbuf(w, c);
    conn_disarm_write(w, c);
    if (c->close_after_write) {
        conn_close(w, c);
    }
}

static void conn_on_readable(Worker* w, Conn* c) {
    for (;;) {
        if (!conn_ensure_rbuf(w, c, c->rbuf_len + 4096U)) {
            conn_close(w, c);
            return;
        }
        ssize_t n = read(c->fd, c->rbuf + c->rbuf_len, c->rbuf_cap - c->rbuf_len);
        if (n > 0) {
            c->rbuf_len += (size_t)n;
        } else if (n == 0) {
            conn_close(w, c);
            return;
        } else {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            conn_close(w, c);
            return;
        }

        /* Parse & serve all complete requests currently in rbuf */
        while (c->rbuf_len > 0) {
            HttpRequest req;
            HttpParseResult pr = http_parse_request(c->rbuf, c->rbuf_len, &req);
            if (pr == HTTP_PARSE_INCOMPLETE) {
                if (req.expected_total > c->rbuf_cap) {
                    if (req.expected_total > HTTP_MAX_HEADER_BYTES + HTTP_MAX_BODY_BYTES ||
                        !conn_ensure_rbuf(w, c, req.expected_total)) {
                        conn_close(w, c);
                        return;
                    }
                }
                break;
            }
            if (pr == HTTP_PARSE_ERROR) {
                RCRE_StrBuf out = rcre_strbuf_init(w->tx_staging, 24576U);
                size_t out_len = app_write_bad_request(w->date_hdr, &out);
                (void)conn_send(w, c, out.data, out_len);
                if (c->wbuf_len == 0) {
                    conn_close(w, c);
                } else {
                    c->close_after_write = true;
                }
                return;
            }

            /* Dispatch complete request with O(1) arena rewind */
            RCRE_StrBuf out = rcre_strbuf_init(w->tx_staging, 24576U);
            size_t out_len = app_handle_request(w->app, &req, w->date_hdr, w->scratch, &w->bm25_sc, &out);
            rcre_arena_reset(w->scratch);

            /* Advance input buffer */
            if (req.consumed == c->rbuf_len) {
                c->rbuf_len = 0;
            } else {
                size_t rem = c->rbuf_len - req.consumed;
                memmove(c->rbuf, c->rbuf + req.consumed, rem);
                c->rbuf_len = rem;
            }

            if (!conn_send(w, c, out.data, out_len)) {
                conn_close(w, c);
                return;
            }
            if (!req.keep_alive) {
                if (c->wbuf_len == 0) {
                    conn_close(w, c);
                } else {
                    c->close_after_write = true;
                }
                return;
            }
        }

        /* If we filled the entire buffer, keep reading immediately; otherwise release idle buffer */
        if (c->rbuf_len == 0 && (size_t)n < 4096U) {
            break;
        }
    }

    /* Return idle read buffer back to slab so dormant Keep-Alive sockets hold ~0 buffer RAM */
    if (c->fd >= 0 && c->rbuf_len == 0 && c->rbuf != nullptr) {
        conn_release_rbuf(w, c);
    }
}

static void worker_drain_accept_ring(Worker* w) {
    int fd = -1;
    while (fd_ring_pop(&w->ring, &fd)) {
        worker_register_client(w, fd);
    }
}

static void worker_accept_inline(Worker* w) {
    for (;;) {
        int cfd = accept(w->inline_listen_fd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        configure_client_socket(cfd);
        worker_register_client(w, cfd);
    }
}

static void* worker_main(void* arg) {
    Worker* w = (Worker*)arg;
#if defined(__APPLE__)
    (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    struct kevent events[KEVENT_BATCH_SIZE];
    struct timespec timeout = { .tv_sec = 0, .tv_nsec = 200000000L }; /* 200 ms */

    refresh_date_header(w, time(nullptr));

    while (!atomic_load_explicit(&g_stop, memory_order_acquire)) {
        atomic_store_explicit(&w->sleeping, true, memory_order_release);
        /* Double-check ring before blocking so we never miss a wakeup */
        if (atomic_load_explicit(&w->ring.head, memory_order_acquire) !=
            atomic_load_explicit(&w->ring.tail, memory_order_acquire)) {
            atomic_store_explicit(&w->sleeping, false, memory_order_relaxed);
            worker_drain_accept_ring(w);
            continue;
        }

        int nev = kevent(w->kq, nullptr, 0, events, (int)KEVENT_BATCH_SIZE, &timeout);
        atomic_store_explicit(&w->sleeping, false, memory_order_relaxed);

        if (nev < 0) {
            if (errno == EINTR) continue;
            break;
        }

        refresh_date_header(w, time(nullptr));

        for (int i = 0; i < nev; ++i) {
            struct kevent* ev = &events[i];
            if (ev->filter == EVFILT_USER && ev->ident == WAKEUP_IDENT) {
                worker_drain_accept_ring(w);
                continue;
            }
            if (w->inline_listen_fd >= 0 && (int)ev->ident == w->inline_listen_fd) {
                worker_accept_inline(w);
                continue;
            }
            Conn* c = (Conn*)ev->udata;
            if (!c || c->fd < 0) continue;

            if (ev->flags & EV_ERROR) {
                conn_close(w, c);
                continue;
            }
            if (ev->filter == EVFILT_READ) {
                if ((ev->flags & EV_EOF) && ev->data == 0) {
                    conn_close(w, c);
                } else {
                    conn_on_readable(w, c);
                }
            } else if (ev->filter == EVFILT_WRITE) {
                conn_on_writable(w, c);
            }
        }
    }

    return nullptr;
}

static void worker_wake(Worker* w) {
    struct kevent ev;
    EV_SET(&ev, WAKEUP_IDENT, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
    (void)kevent(w->kq, &ev, 1, nullptr, 0, nullptr);
}

static bool worker_init(Worker* w, int id, const AppState* app, int inline_listen_fd) {
    memset(w, 0, sizeof(*w));
    w->id = id;
    w->app = app;
    w->inline_listen_fd = inline_listen_fd;
    w->kq = kqueue();
    if (w->kq < 0) return false;

    w->scratch = rcre_arena_create(WORKER_ARENA_BYTES);
    if (!w->scratch) return false;
    if (!worker_bm25_scratch_init(&w->bm25_sc, app->bm25.num_docs)) return false;
    if (!buf_slab_init(&w->slab, SLAB_BUFS_PER_WKR)) return false;

    w->conns = (Conn*)calloc(MAX_CONNS_PER_WKR, sizeof(Conn));
    w->free_slots = (uint32_t*)malloc(MAX_CONNS_PER_WKR * sizeof(uint32_t));
    w->tx_staging = (uint8_t*)aligned_alloc(64, 24576U);
    if (!w->conns || !w->free_slots || !w->tx_staging) return false;

    for (size_t i = 0; i < MAX_CONNS_PER_WKR; ++i) {
        w->conns[i].fd = -1;
        w->conns[i].slot_id = (uint32_t)i;
        w->free_slots[i] = (uint32_t)(MAX_CONNS_PER_WKR - 1U - i);
    }
    w->free_top = MAX_CONNS_PER_WKR;

    struct kevent ev[2];
    int n_ev = 0;
    EV_SET(&ev[n_ev++], WAKEUP_IDENT, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr);
    if (inline_listen_fd >= 0) {
        EV_SET(&ev[n_ev++], (uintptr_t)inline_listen_fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
    }
    if (kevent(w->kq, ev, n_ev, nullptr, 0, nullptr) < 0) return false;

    return true;
}

static void worker_destroy(Worker* w) {
    if (w->conns) {
        for (size_t i = 0; i < MAX_CONNS_PER_WKR; ++i) {
            if (w->conns[i].fd >= 0) {
                conn_close(w, &w->conns[i]);
            }
        }
    }
    int fd = -1;
    while (fd_ring_pop(&w->ring, &fd)) {
        close(fd);
    }
    if (w->kq >= 0) close(w->kq);
    if (w->scratch) rcre_arena_destroy(w->scratch);
    worker_bm25_scratch_destroy(&w->bm25_sc);
    buf_slab_destroy(&w->slab);
    free(w->conns);
    free(w->free_slots);
    free(w->tx_staging);
}

/* ========================================================================= */
/* Listen Socket & Main Entry Point                                          */
/* ========================================================================= */

static int create_listen_socket(const char* host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#ifdef SO_REUSEPORT
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
#endif
    (void)set_nonblocking(fd);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd);
        return -1;
    }
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 4096) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int main(int argc, char** argv) {
    const char* host     = "127.0.0.1";
    int         port     = 18081;
    int         nworkers = 4;
    const char* data_dir = "../spec/data";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            nworkers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        }
    }
    if (nworkers < 1) nworkers = 1;
    if (nworkers > 64) nworkers = 64;

    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    AppState app;
    int rc = app_state_init(&app, data_dir);
    if (rc != RCRE_OK) {
        fprintf(stderr, "[rcre] failed to initialize app state from %s (rc=%d)\n", data_dir, rc);
        return 1;
    }

    int listen_fd = create_listen_socket(host, port);
    if (listen_fd < 0) {
        perror("[rcre] listen");
        app_state_destroy(&app);
        return 1;
    }

    Worker* workers = (Worker*)calloc((size_t)nworkers, sizeof(Worker));
    if (!workers) {
        close(listen_fd);
        app_state_destroy(&app);
        return 1;
    }

    for (int i = 0; i < nworkers; ++i) {
        int inline_fd = (nworkers == 1) ? listen_fd : -1;
        if (!worker_init(&workers[i], i, &app, inline_fd)) {
            fprintf(stderr, "[rcre] failed to init worker %d\n", i);
            return 1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double init_ms = ((double)(t1.tv_sec - t0.tv_sec) * 1000.0) +
                     ((double)(t1.tv_nsec - t0.tv_nsec) / 1e6);

    fprintf(stderr, "[rcre] ready on %s:%d (workers=%d, docs=%u, trees=%zu, init=%.2fms)\n",
            host, port, nworkers, app.bm25.num_docs, app.gbdt.num_trees, init_ms);
    fflush(stderr);

    if (nworkers == 1) {
        /* Exact 1-OS-thread mode: accept + I/O in a single kqueue loop */
        worker_main(&workers[0]);
    } else {
        for (int i = 0; i < nworkers; ++i) {
            if (pthread_create(&workers[i].thread, nullptr, worker_main, &workers[i]) != 0) {
                fprintf(stderr, "[rcre] pthread_create failed\n");
                atomic_store(&g_stop, true);
                break;
            }
        }

        int accept_kq = kqueue();
        struct kevent ev;
        EV_SET(&ev, (uintptr_t)listen_fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
        (void)kevent(accept_kq, &ev, 1, nullptr, 0, nullptr);

        struct kevent out_ev[8];
        struct timespec timeout = { .tv_sec = 0, .tv_nsec = 200000000L };
        size_t rr = 0;

        while (!atomic_load_explicit(&g_stop, memory_order_acquire)) {
            int nev = kevent(accept_kq, nullptr, 0, out_ev, 8, &timeout);
            if (nev <= 0) continue;

            for (;;) {
                int cfd = accept(listen_fd, nullptr, nullptr);
                if (cfd < 0) {
                    if (errno == EINTR) continue;
                    break;
                }
                configure_client_socket(cfd);
                Worker* target = &workers[rr % (size_t)nworkers];
                rr++;
                if (!fd_ring_push(&target->ring, cfd)) {
                    close(cfd);
                    continue;
                }
                if (atomic_load_explicit(&target->sleeping, memory_order_acquire)) {
                    worker_wake(target);
                }
            }
        }

        close(accept_kq);
        for (int i = 0; i < nworkers; ++i) {
            worker_wake(&workers[i]);
        }
        for (int i = 0; i < nworkers; ++i) {
            pthread_join(workers[i].thread, nullptr);
        }
    }

    close(listen_fd);
    for (int i = 0; i < nworkers; ++i) {
        worker_destroy(&workers[i]);
    }
    free(workers);
    app_state_destroy(&app);
    return 0;
}
