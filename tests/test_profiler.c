/**
 * @file test_profiler.c
 * @brief Unit tests for the RCRE Profiling & Telemetry Subsystem (rcre_prof)
 */

#include "rcre.h"
#include "rcre_prof.h"
#include "test_framework.h"

/* --------------------------- timing primitives -------------------------- */

static void test_timing_primitives(void) {
    uint64_t a = rcre_prof_now_ns();
    uint64_t b = rcre_prof_now_ns();
    ASSERT_TRUE(b >= a);                       /* monotonic */
    ASSERT_TRUE((b - a) < 1000000ULL);         /* two reads < 1 ms apart */

    uint64_t c0 = rcre_prof_cycles();
    uint64_t c1 = rcre_prof_cycles();
    ASSERT_TRUE(c1 >= c0);                     /* cycle counter monotonic */
    ASSERT_TRUE(rcre_prof_cycles_freq() > 0);  /* frequency known on supported hw */

    double dur_ns = rcre_prof_to_ns(1000);
    ASSERT_TRUE(dur_ns > 0.0);
}

/* ------------------------- scope zone mechanics ------------------------- */

static void test_scope_balance_and_duration(void) {
    RCRE_ProfScope s = rcre_prof_scope_begin(NULL, "timing_only");
    volatile double x = 0.0;
    for (int i = 0; i < 100000; i++) x += (double)i; /* burn a little time */
    (void)x;
    uint64_t dur = rcre_prof_scope_end(&s);
    ASSERT_TRUE(dur > 0);
}

static void test_zone_records_events(void) {
    RCRE_ProfContext* ctx = rcre_prof_init(256);
    ASSERT_NOT_NULL(ctx);
    ASSERT_UINT_EQ(0, ctx->count);

    RCRE_PROFILE_SCOPE(ctx, "alpha", z);
    volatile double burn = 0.0;
    for (int i = 0; i < 5000; i++) burn += (double)i;
    (void)burn;
    RCRE_PROFILE_END(z);
    ASSERT_UINT_EQ(1, ctx->count);

    RCRE_ProfEvent* e = &ctx->ring[0];
    ASSERT_UINT_EQ(0, e->kind);                /* X event */
    ASSERT_TRUE(strcmp(e->name, "alpha") == 0);
    ASSERT_TRUE(e->dur_ns > 0);

    /* nested zones */
    RCRE_PROFILE_SCOPE(ctx, "outer", zo);
    RCRE_PROFILE_SCOPE(ctx, "inner", zi);
    RCRE_PROFILE_END(zi);
    RCRE_PROFILE_END(zo);
    ASSERT_UINT_EQ(3, ctx->count);
    ASSERT_INT_EQ(0, ctx->depth);              /* balanced */

    rcre_prof_free(ctx);
}

static void test_ring_wraps_without_crash(void) {
    RCRE_ProfContext* ctx = rcre_prof_init(64);
    ASSERT_NOT_NULL(ctx);
    const int N = 1000;
    for (int i = 0; i < N; i++) {
        RCRE_PROFILE_SCOPE(ctx, "spin", z);
        RCRE_PROFILE_END(z);
    }
    ASSERT_UINT_EQ(1000, ctx->count);   /* monotonic total */
    ASSERT_UINT_EQ((size_t)(N - 64), ctx->dropped);
    rcre_prof_free(ctx);
}

static void test_null_context_is_noop(void) {
    RCRE_ProfScope s = rcre_prof_scope_begin(NULL, "noop");
    uint64_t d = rcre_prof_scope_end(&s);
    ASSERT_TRUE(d >= 0);
    rcre_prof_mark(NULL, "x", 1.0);            /* must not crash */
    rcre_prof_sample_rss(NULL);
    rcre_prof_io_record(NULL, false, 10);
}

/* --------------------------- counter events ----------------------------- */

static void test_counters(void) {
    RCRE_ProfContext* ctx = rcre_prof_init(64);
    rcre_prof_mark(ctx, "GFLOPS", 42.5);
    ASSERT_UINT_EQ(1, ctx->count);
    ASSERT_UINT_EQ(1, ctx->ring[0].kind);
    ASSERT_TRUE(ctx->ring[0].value > 42.4 && ctx->ring[0].value < 42.6);
    rcre_prof_free(ctx);
}

/* ------------------- arena telemetry (Layer A RAM) ---------------------- */

static void test_arena_telemetry(void) {
    RCRE_Arena* arena = rcre_arena_create(1 << 20);
    ASSERT_NOT_NULL(arena);
    ASSERT_UINT_EQ(0, arena->high_watermark);
    ASSERT_UINT_EQ(0, arena->alloc_count);
    ASSERT_UINT_EQ(0, arena->rewind_count);

    void* a = rcre_arena_alloc(arena, 1000, 64);
    ASSERT_NOT_NULL(a);
    ASSERT_UINT_EQ(1, arena->alloc_count);
    ASSERT_TRUE(arena->high_watermark >= 1000);
    size_t hwm1 = arena->high_watermark;

    RCRE_ArenaTemp t = rcre_arena_temp_begin(arena);
    void* b = rcre_arena_alloc(arena, 4000, 64);
    ASSERT_NOT_NULL(b);
    ASSERT_TRUE(arena->high_watermark > hwm1); /* watermark captured the spike */
    rcre_arena_temp_end(t);
    ASSERT_UINT_EQ(1, arena->rewind_count);
    ASSERT_TRUE(arena->high_watermark > hwm1); /* watermark SURVIVES rewind */

    RCRE_ProfContext* ctx = rcre_prof_init(64);
    rcre_prof_sample_arena(ctx, arena);
    ASSERT_UINT_EQ(arena->high_watermark, ctx->arena_hwm);
    ASSERT_UINT_EQ(arena->alloc_count, ctx->arena_allocs);
    ASSERT_UINT_EQ(arena->rewind_count, ctx->arena_rewinds);
    ASSERT_UINT_EQ(1, ctx->count);             /* ARENA_HWM_KB mark from sample_arena */
    rcre_prof_free(ctx);

    rcre_arena_destroy(arena);
}

/* --------------------- OS probes (Layer B RAM / I/O) --------------------- */

static void test_os_probes(void) {
    RCRE_ProfContext* ctx = rcre_prof_init(64);
    ASSERT_TRUE(ctx->rss_start > 0);           /* RSS measurable */

    uint64_t u, s;
    rcre_prof_cpu_time_ns(&u, &s);
    ASSERT_TRUE(u + s > 0);                    /* CPU time measurable */

    uint64_t io_r = 99, io_w = 99;
    int avail = rcre_prof_disk_io_bytes(&io_r, &io_w);
    (void)avail;                               /* platform-dependent; just callable */
    (void)io_r; (void)io_w;

    /* app-level I/O ledger is exact everywhere */
    rcre_prof_io_record(ctx, false, 1024);
    rcre_prof_io_record(ctx, true, 2048);
    ASSERT_UINT_EQ(1024, ctx->io_read_app);
    ASSERT_UINT_EQ(2048, ctx->io_write_app);
    rcre_prof_free(ctx);
}

/* ------------------------------ exporters -------------------------------- */

static void test_perfetto_json_export(void) {
    RCRE_ProfContext* ctx = rcre_prof_init(64);
    RCRE_PROFILE_SCOPE(ctx, "work", z);
    RCRE_PROFILE_END(z);
    rcre_prof_mark(ctx, "RSS_KB", 123.0);
    char path[256];
    snprintf(path, sizeof(path), "/tmp/rcre_prof_test_%d.json", (int)getpid());
    ASSERT_TRUE(rcre_prof_export_perfetto_json(ctx, path));

    FILE* f = fopen(path, "r");
    ASSERT_NOT_NULL(f);
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(buf, "\"ph\": \"X\"") != NULL);
    ASSERT_TRUE(strstr(buf, "\"ph\": \"C\"") != NULL);
    ASSERT_TRUE(strstr(buf, "work") != NULL);
    ASSERT_TRUE(strstr(buf, "\"name\": \"work\", \"cat\": \"Compute\"") != NULL);
    remove(path);
    rcre_prof_free(ctx);
}

static void test_html_export(void) {
    RCRE_ProfContext* ctx = rcre_prof_init(64);
    RCRE_PROFILE_SCOPE(ctx, "op_a", z);
    RCRE_PROFILE_END(z);
    rcre_prof_io_record(ctx, true, 512);
    char path[256];
    snprintf(path, sizeof(path), "/tmp/rcre_prof_test_%d.html", (int)getpid());
    ASSERT_TRUE(rcre_prof_export_html(ctx, path));

    FILE* f = fopen(path, "r");
    ASSERT_NOT_NULL(f);
    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);
    ASSERT_TRUE(strstr(buf, "<!DOCTYPE html>") != NULL);
    ASSERT_TRUE(strstr(buf, "Total consumption") != NULL);
    ASSERT_TRUE(strstr(buf, "CPU") != NULL);
    ASSERT_TRUE(strstr(buf, "RAM (RSS)") != NULL);
    ASSERT_TRUE(strstr(buf, "Storage (engine I/O)") != NULL);
    remove(path);
    rcre_prof_free(ctx);
}

static void test_terminal_summary(void) {
    RCRE_ProfContext* ctx = rcre_prof_init(64);
    RCRE_PROFILE_SCOPE(ctx, "scope_x", z);
    RCRE_PROFILE_END(z);
    /* must not crash; capture nothing (goes to stderr in CI) */
    rcre_prof_print_summary(ctx, stderr);
    rcre_prof_free(ctx);
}

/* ----------------------- overhead budget (P3 gate) ----------------------- */

static void test_scope_overhead_budget(void) {
    /* Stage-1 sandbox target: < 25 ns per scope at -O2/-O3. */
    const int N = 2000000;
    uint64_t t0 = rcre_prof_now_ns();
    for (int i = 0; i < N; i++) {
        RCRE_ProfScope s = rcre_prof_scope_begin(NULL, "zone");
        rcre_prof_scope_end(&s);
    }
    uint64_t t1 = rcre_prof_now_ns();
    double ns_per_scope = (double)(t1 - t0) / (double)N;
    fprintf(stderr, "      (per-scope overhead: %.2f ns)\n", ns_per_scope);
    /* Hard CI gate is generous (100 ns) to avoid flakes; the 25 ns design
     * target is verified by the bench_prof Stage-1 validation. */
    ASSERT_TRUE(ns_per_scope < 100.0);
}

int main(void) {
    test_suite_begin("RCRE Profiling & Telemetry Subsystem (rcre_prof)");
    TEST_RUN(test_timing_primitives);
    TEST_RUN(test_scope_balance_and_duration);
    TEST_RUN(test_zone_records_events);
    TEST_RUN(test_ring_wraps_without_crash);
    TEST_RUN(test_null_context_is_noop);
    TEST_RUN(test_counters);
    TEST_RUN(test_arena_telemetry);
    TEST_RUN(test_os_probes);
    TEST_RUN(test_perfetto_json_export);
    TEST_RUN(test_html_export);
    TEST_RUN(test_terminal_summary);
    TEST_RUN(test_scope_overhead_budget);
    return test_suite_report();
}
