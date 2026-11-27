/**
 * @file test_http.c
 * @brief Unit & Multi-Worker Integration Tests for include/extras/rcre_http.h
 *
 * Verifies:
 *   1. Zero-copy HTTP/1.1 parser (GET/POST, query params, Content-Length, partial reads, malformed).
 *   2. Single-worker inline-accept reactor (keep-alive reuse, fragmented TCP sends, 400 error).
 *   3. Multi-worker (4-worker kqueue/epoll) concurrent client stress (TSan & ASan+UBSan clean).
 */

#include "extras/rcre_http.h"
#include "test_framework.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static size_t test_echo_handler(const RCRE_HttpRequest* req, RCRE_HttpContext* ctx, RCRE_StrBuf* out) {
    atomic_size_t* req_counter = (atomic_size_t*)ctx->user_ctx;
    if (req_counter) {
        atomic_fetch_add_explicit(req_counter, 1U, memory_order_relaxed);
    }

    if (rcre_slice_eq(req->path, RCRE_STR("/health"))) {
        return rcre_http_write_response(
            out, RCRE_STR("HTTP/1.1 200 OK"), RCRE_STR("text/plain"),
            ctx->date_hdr, req->keep_alive, RCRE_STR("OK"));
    }
    if (rcre_slice_eq(req->path, RCRE_STR("/echo")) && req->method == RCRE_HTTP_POST) {
        /* Copy body into per-worker arena to exercise O(1) rewind under ASan */
        RCRE_Slice dup = rcre_str_dup(ctx->scratch, req->body);
        return rcre_http_write_response(
            out, RCRE_STR("HTTP/1.1 200 OK"), RCRE_STR("application/octet-stream"),
            ctx->date_hdr, req->keep_alive, dup);
    }
    return rcre_http_write_response(
        out, RCRE_STR("HTTP/1.1 404 Not Found"), RCRE_STR("application/json"),
        ctx->date_hdr, req->keep_alive, RCRE_STR("{\"error\":\"not_found\"}"));
}

static void test_http_parser_unit(void) {
    RCRE_HttpRequest req;

    /* Partial request */
    const char* part = "GET /search?q=neon HTTP/1.1\r\nHost: local";
    ASSERT_INT_EQ(RCRE_HTTP_PARSE_INCOMPLETE,
                  rcre_http_parse_request((const uint8_t*)part, strlen(part), 1024, &req));

    /* Complete GET with query */
    const char* full_get = "GET /search?q=neon&k=5 HTTP/1.1\r\nHost: localhost\r\n\r\n";
    ASSERT_INT_EQ(RCRE_HTTP_PARSE_OK,
                  rcre_http_parse_request((const uint8_t*)full_get, strlen(full_get), 1024, &req));
    ASSERT_INT_EQ(RCRE_HTTP_GET, req.method);
    ASSERT_TRUE(rcre_slice_eq(req.path, RCRE_STR("/search")));
    ASSERT_TRUE(rcre_slice_eq(req.query, RCRE_STR("q=neon&k=5")));
    ASSERT_TRUE(req.keep_alive);
    ASSERT_UINT_EQ(strlen(full_get), req.consumed);

    /* Complete POST with body and Connection: close */
    const char* full_post =
        "POST /echo HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello";
    ASSERT_INT_EQ(RCRE_HTTP_PARSE_OK,
                  rcre_http_parse_request((const uint8_t*)full_post, strlen(full_post), 1024, &req));
    ASSERT_INT_EQ(RCRE_HTTP_POST, req.method);
    ASSERT_FALSE(req.keep_alive);
    ASSERT_TRUE(rcre_slice_eq(req.body, RCRE_STR("hello")));

    /* Malformed request line */
    const char* bad = "GARBAGE\r\n\r\n";
    ASSERT_INT_EQ(RCRE_HTTP_PARSE_ERROR,
                  rcre_http_parse_request((const uint8_t*)bad, strlen(bad), 1024, &req));
}

static void* server_thread_fn(void* arg) {
    RCRE_HttpServer* srv = (RCRE_HttpServer*)arg;
    (void)rcre_http_server_run(srv);
    return nullptr;
}

static int connect_local(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static bool send_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = (const uint8_t*)buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}

static size_t recv_until_contains(int fd, char* buf, size_t cap, const char* needle) {
    size_t total = 0;
    while (total + 1 < cap) {
        ssize_t n = recv(fd, buf + total, cap - 1 - total, 0);
        if (n <= 0) break;
        total += (size_t)n;
        buf[total] = '\0';
        if (strstr(buf, needle) != nullptr) break;
    }
    return total;
}

typedef struct ClientJob {
    uint16_t port;
    int      reqs;
    atomic_int* passed;
} ClientJob;

static void* client_worker_fn(void* arg) {
    ClientJob* job = (ClientJob*)arg;
    int fd = connect_local(job->port);
    if (fd < 0) return nullptr;

    const char* req = "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n";
    char resp[512];
    int ok = 0;
    for (int i = 0; i < job->reqs; ++i) {
        if (!send_all(fd, req, strlen(req))) break;
        size_t n = recv_until_contains(fd, resp, sizeof(resp), "\r\n\r\nOK");
        if (n > 0 && strstr(resp, "200 OK") && strstr(resp, "\r\n\r\nOK")) {
            ok++;
        }
    }
    close(fd);
    atomic_fetch_add(job->passed, ok);
    return nullptr;
}

static void run_server_integration(size_t num_workers) {
    atomic_size_t handled = 0;
    RCRE_HttpServerConfig cfg = {
        .host                 = "127.0.0.1",
        .port                 = 0,
        .num_workers          = num_workers,
        .worker_arena_bytes   = 512 * 1024,
        .max_conns_per_worker = 256,
        .max_body_bytes       = 65536,
        .handler              = test_echo_handler,
        .user_ctx             = &handled
    };

    RCRE_HttpServer* srv = rcre_http_server_create(&cfg);
    ASSERT_NOT_NULL(srv);
    uint16_t port = rcre_http_server_port(srv);
    ASSERT_TRUE(port > 0);

    pthread_t srv_th;
    ASSERT_INT_EQ(0, pthread_create(&srv_th, nullptr, server_thread_fn, srv));

    /* 1. Fragmented GET + keep-alive POST echo on one socket */
    int fd = connect_local(port);
    ASSERT_TRUE(fd >= 0);
    ASSERT_TRUE(send_all(fd, "GET /he", 7));
    ASSERT_TRUE(send_all(fd, "alth HTTP/1.1\r\nHost: x\r\n\r\n", 26));
    char buf[1024];
    size_t n = recv_until_contains(fd, buf, sizeof(buf), "\r\n\r\nOK");
    ASSERT_TRUE(n > 0 && strstr(buf, "200 OK") != nullptr);

    const char* post_req =
        "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 9\r\nConnection: close\r\n\r\nRCRE_HTTP";
    ASSERT_TRUE(send_all(fd, post_req, strlen(post_req)));
    n = recv_until_contains(fd, buf, sizeof(buf), "RCRE_HTTP");
    ASSERT_TRUE(n > 0 && strstr(buf, "200 OK") != nullptr && strstr(buf, "RCRE_HTTP") != nullptr);
    close(fd);

    /* 2. Malformed request yields 400 Bad Request */
    int bad_fd = connect_local(port);
    ASSERT_TRUE(bad_fd >= 0);
    ASSERT_TRUE(send_all(bad_fd, "NOT_HTTP\r\n\r\n", 12));
    n = recv_until_contains(bad_fd, buf, sizeof(buf), "bad_request");
    ASSERT_TRUE(n > 0 && strstr(buf, "400 Bad Request") != nullptr);
    close(bad_fd);

    /* 3. Concurrent multi-client keep-alive burst */
    atomic_int ok_count = 0;
    pthread_t clients[4];
    ClientJob jobs[4];
    for (int i = 0; i < 4; ++i) {
        jobs[i].port = port;
        jobs[i].reqs = 25;
        jobs[i].passed = &ok_count;
        ASSERT_INT_EQ(0, pthread_create(&clients[i], nullptr, client_worker_fn, &jobs[i]));
    }
    for (int i = 0; i < 4; ++i) {
        pthread_join(clients[i], nullptr);
    }
    ASSERT_INT_EQ(100, atomic_load(&ok_count));
    ASSERT_TRUE(atomic_load(&handled) >= 102U);

    rcre_http_server_stop(srv);
    pthread_join(srv_th, nullptr);
    rcre_http_server_destroy(srv);
}

static void test_http_single_worker(void) {
    run_server_integration(1);
}

static void test_http_multi_worker(void) {
    run_server_integration(4);
}

int main(void) {
    test_suite_begin("RCRE Extras HTTP/1.1 Reactor (kqueue/epoll)");
    TEST_RUN(test_http_parser_unit);
    TEST_RUN(test_http_single_worker);
    TEST_RUN(test_http_multi_worker);
    return test_suite_report();
}
