/**
 * @file prof.c
 * @brief RCRE Native Profiler Engine (ISO C23, zero-dependency, cross-platform)
 *
 * Platform probes:
 *   - ARM64 (Apple Silicon / Linux ARM): mrs cntvct_el0 (EL0-accessible on both
 *     macOS and Linux; validated empirically on Apple M5).
 *   - x86-64: RDTSC.
 *   - Fallback: CLOCK_MONOTONIC for both timing sources.
 *   - RSS: macOS MACH_TASK_BASIC_INFO / Linux /proc/self/statm.
 *   - CPU time: getrusage (both platforms).
 *   - Disk I/O (OS layer): Linux /proc/self/io (byte-exact).
 *     macOS under strict -std=c2x: proc_pid_rusage()/rusage_info_v4 are hidden
 *     by __STRICT_ANSI__ in Apple's SDK and getrusage block counters are never
 *     populated — the application-level ledger (rcre_prof_io_record) is the
 *     source of truth there.
 */

#include "rcre_prof.h"
#include "rcre.h"   /* RCRE_Arena definition */

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

/* ========================================================================= */
/* Timing                                                                     */
/* ========================================================================= */

uint64_t rcre_prof_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#if defined(__aarch64__)
uint64_t rcre_prof_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}
uint64_t rcre_prof_cycles_freq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}
#elif defined(__x86_64__) || defined(_M_X64)
uint64_t rcre_prof_cycles(void) {
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
uint64_t rcre_prof_cycles_freq(void) { return 0; } /* calibrate if needed */
#else
uint64_t rcre_prof_cycles(void) { return rcre_prof_now_ns(); }
uint64_t rcre_prof_cycles_freq(void) { return 1000000000ULL; }
#endif

double rcre_prof_to_ns(uint64_t duration) {
    uint64_t f = rcre_prof_cycles_freq();
    return (double)duration * 1e9 / (double)(f ? f : 1000000000ULL);
}

/* ========================================================================= */
/* Scoped zones                                                               */
/* ========================================================================= */

RCRE_ProfScope rcre_prof_scope_begin(RCRE_ProfContext* ctx, const char* name) {
    RCRE_ProfScope s;
    s.ctx = ctx;
    s.name = name ? name : "?";
    s.begin = rcre_prof_cycles();
    s.active = 1;
    if (ctx && ctx->depth < 64) ctx->depth++;
    return s;
}

uint64_t rcre_prof_scope_end(RCRE_ProfScope* scope) {
    uint64_t now_ticks = rcre_prof_cycles();
    uint64_t dur_ticks = now_ticks - scope->begin;
    double dur_ns = rcre_prof_to_ns(dur_ticks);
    scope->active = 0;

    RCRE_ProfContext* ctx = scope->ctx;
    if (ctx) {
        if (ctx->depth > 0) ctx->depth--;
        /* Timeline timestamp: CLOCK_MONOTONIC epoch only (never the cycle
         * counter's epoch — they differ on Apple Silicon). ts = end - dur. */
        uint64_t now_ns = rcre_prof_now_ns();
        uint64_t ts_ns = (dur_ns * 1e-9 < (double)now_ns) ?
                         now_ns - (uint64_t)(dur_ns) : 0;

        size_t slot = ctx->count % ctx->cap;
        if (ctx->count >= ctx->cap) ctx->dropped++; /* ring full: overwrite oldest */
        ctx->count++;                               /* monotonic write cursor */
        RCRE_ProfEvent* e = &ctx->ring[slot];
        const char* name = scope->name;
        size_t n = strlen(name);
        if (n >= 48) n = 47;
        memcpy(e->name, name, n);
        e->name[n] = '\0';
        e->kind = 0;
        e->depth = (uint8_t)(ctx->depth > 0 ? ctx->depth : 0);
        e->ts_ns = ts_ns;
        e->dur_ns = (uint64_t)dur_ns;
        e->value = 0.0;
    }
    return (uint64_t)dur_ns;
}

/* ========================================================================= */
/* OS probes                                                                  */
/* ========================================================================= */

static uint64_t os_rss_bytes(void) {
#if defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (uint64_t)info.resident_size;
    }
    return 0;
#else
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long long total = 0, resident = 0;
    if (fscanf(f, "%llu %llu", &total, &resident) == 2) {
        long ps = sysconf(_SC_PAGESIZE);
        fclose(f);
        return (uint64_t)resident * (uint64_t)ps;
    }
    fclose(f);
    return 0;
#endif
}

uint64_t rcre_prof_rss_bytes(void) { return os_rss_bytes(); }

void rcre_prof_cpu_time_ns(uint64_t* user_ns, uint64_t* sys_ns) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    *user_ns = (uint64_t)ru.ru_utime.tv_sec * 1000000000ULL + (uint64_t)ru.ru_utime.tv_usec * 1000ULL;
    *sys_ns  = (uint64_t)ru.ru_stime.tv_sec * 1000000000ULL + (uint64_t)ru.ru_stime.tv_usec * 1000ULL;
}

int rcre_prof_disk_io_bytes(uint64_t* read_bytes, uint64_t* write_bytes) {
    *read_bytes = 0;
    *write_bytes = 0;
#if !defined(__APPLE__)
    FILE* f = fopen("/proc/self/io", "r");
    if (!f) return 0;
    char line[128];
    while (fgets(line, sizeof(line), f)) {
        unsigned long long v = 0;
        if (sscanf(line, "read_bytes: %llu", &v) == 1)       *read_bytes = v;
        else if (sscanf(line, "write_bytes: %llu", &v) == 1) *write_bytes = v;
    }
    fclose(f);
    return 1;
#else
    (void)0; /* see file header: unavailable under strict C23 on macOS */
    return 0;
#endif
}

/* ========================================================================= */
/* Context lifecycle                                                          */
/* ========================================================================= */

static RCRE_ProfContext* s_global_ctx = NULL;

static size_t next_pow2(size_t v) {
    size_t p = 2;
    while (p < v) p <<= 1;
    return p;
}

RCRE_ProfContext* rcre_prof_init(size_t capacity) {
    RCRE_ProfContext* ctx = (RCRE_ProfContext*)calloc(1, sizeof(RCRE_ProfContext));
    if (!ctx) return NULL;
    ctx->cap = next_pow2(capacity ? capacity : 4096);
    ctx->ring = (RCRE_ProfEvent*)calloc(ctx->cap, sizeof(RCRE_ProfEvent));
    if (!ctx->ring) { free(ctx); return NULL; }
    ctx->t0_ns = rcre_prof_now_ns();
    ctx->rss_start = os_rss_bytes();
    ctx->rss_peak = ctx->rss_start;
    rcre_prof_cpu_time_ns(&ctx->cpu_user_start, &ctx->cpu_sys_start);
    ctx->os_io_available = rcre_prof_disk_io_bytes(&ctx->io_read_start, &ctx->io_write_start);
    return ctx;
}

void rcre_prof_free(RCRE_ProfContext* ctx) {
    if (!ctx) return;
    if (s_global_ctx == ctx) s_global_ctx = NULL;
    free(ctx->ring);
    free(ctx);
}

/* ---- process-wide context (engine-internal scopes) ---- */
RCRE_ProfContext* rcre_prof_global(void) { return s_global_ctx; }
void rcre_prof_set_global(RCRE_ProfContext* ctx) { s_global_ctx = ctx; }

/* ========================================================================= */
/* Telemetry sampling                                                         */
/* ========================================================================= */

void rcre_prof_mark(RCRE_ProfContext* ctx, const char* name, double value) {
    if (!ctx) return;
    size_t slot = ctx->count % ctx->cap;
    if (ctx->count >= ctx->cap) ctx->dropped++;
    ctx->count++;
    RCRE_ProfEvent* e = &ctx->ring[slot];
    size_t n = strlen(name);
    if (n >= 48) n = 47;
    memcpy(e->name, name, n);
    e->name[n] = '\0';
    e->kind = 1;
    e->depth = 0;
    e->ts_ns = rcre_prof_now_ns();
    e->dur_ns = 0;
    e->value = value;
}

void rcre_prof_sample_rss(RCRE_ProfContext* ctx) {
    if (!ctx) return;
    uint64_t rss = os_rss_bytes();
    if (rss > ctx->rss_peak) ctx->rss_peak = rss;
    rcre_prof_mark(ctx, "RSS_KB", (double)rss / 1024.0);
}

void rcre_prof_sample_arena(RCRE_ProfContext* ctx, RCRE_Arena* arena) {
    if (!ctx || !arena) return;
    ctx->arena_hwm = arena->high_watermark;
    ctx->arena_allocs = arena->alloc_count;
    ctx->arena_rewinds = arena->rewind_count;
    rcre_prof_mark(ctx, "ARENA_HWM_KB", (double)arena->high_watermark / 1024.0);
}

void rcre_prof_io_record(RCRE_ProfContext* ctx, bool is_write, uint64_t bytes) {
    if (!ctx) return;
    if (is_write) ctx->io_write_app += bytes;
    else          ctx->io_read_app  += bytes;
}

void rcre_prof_sample_io(RCRE_ProfContext* ctx) {
    if (!ctx) return;
    uint64_t r = 0, w = 0;
    if (rcre_prof_disk_io_bytes(&r, &w)) {
        ctx->io_read_start  = r;   /* refreshed baseline; deltas computed vs init */
        ctx->io_write_start = w;
        rcre_prof_mark(ctx, "IO_READ_MB", (double)r / 1e6);
        rcre_prof_mark(ctx, "IO_WRITE_MB", (double)w / 1e6);
    }
}

/* ========================================================================= */
/* Ring iteration helpers (count is a monotonic total; ring holds the last   */
/* `cap` events — never index ring[] by raw count)                            */
/* ========================================================================= */

static size_t ring_valid(const RCRE_ProfContext* ctx) {
    return ctx->count < ctx->cap ? ctx->count : ctx->cap;
}
static size_t ring_slot(const RCRE_ProfContext* ctx, size_t i) {
    size_t start = (ctx->count <= ctx->cap) ? 0 : (ctx->count % ctx->cap);
    return (start + i) % ctx->cap;
}

/* ========================================================================= */
/* Aggregation helper (per-scope totals)                                      */
/* ========================================================================= */

typedef struct {
    char name[48];
    uint64_t total_ns;
    uint64_t max_ns;
    size_t count;
} RCRE_ProfAgg;

static int agg_cmp(const void* a, const void* b) {
    const RCRE_ProfAgg* x = (const RCRE_ProfAgg*)a;
    const RCRE_ProfAgg* y = (const RCRE_ProfAgg*)b;
    return (y->total_ns > x->total_ns) - (y->total_ns < x->total_ns);
}

static size_t aggregate(const RCRE_ProfContext* ctx, RCRE_ProfAgg* out, size_t max_out) {
    size_t used = 0;
    const size_t n_valid = ring_valid(ctx);
    for (size_t i = 0; i < n_valid; i++) {
        const RCRE_ProfEvent* e = &ctx->ring[ring_slot(ctx, i)];
        if (e->kind != 0) continue;
        size_t slot = (size_t)-1;
        for (size_t k = 0; k < used; k++) {
            if (strcmp(out[k].name, e->name) == 0) { slot = k; break; }
        }
        if (slot == (size_t)-1) {
            if (used >= max_out) continue;
            slot = used++;
            memcpy(out[slot].name, e->name, 48);
            out[slot].total_ns = 0;
            out[slot].max_ns = 0;
            out[slot].count = 0;
        }
        out[slot].total_ns += e->dur_ns;
        out[slot].count++;
        if (e->dur_ns > out[slot].max_ns) out[slot].max_ns = e->dur_ns;
    }
    qsort(out, used, sizeof(RCRE_ProfAgg), agg_cmp);
    return used;
}

/* ========================================================================= */
/* Exporter 1: ANSI terminal summary (TOTAL CPU / RAM / Storage)              */
/* ========================================================================= */

static void bar(const char* label, double value, double max, const char* suffix, FILE* out) {
    int width = 28, filled = (int)(value / (max > 0 ? max : 1.0) * width);
    if (filled > width) filled = width;
    if (filled < 0) filled = 0;
    int use_color = (getenv("NO_COLOR") == NULL) && isatty(fileno(out));
    const char* cbar = use_color ? "\033[1;36m" : "";
    const char* cre  = use_color ? "\033[0m" : "";
    fprintf(out, "  %-*s [", 24, label);
    fputs(cbar, out);
    for (int i = 0; i < width; i++) fputc(i < filled ? '#' : ' ', out);
    fputs(cre, out);
    fprintf(out, "] %s\n", suffix);
}

void rcre_prof_print_summary(RCRE_ProfContext* ctx, FILE* out) {
    if (!ctx) return;
    uint64_t t_end = rcre_prof_now_ns();
    uint64_t rss_end = os_rss_bytes();
    if (rss_end > ctx->rss_peak) ctx->rss_peak = rss_end;
    uint64_t u, s;
    rcre_prof_cpu_time_ns(&u, &s);
    uint64_t io_r_os = 0, io_w_os = 0;
    (void)rcre_prof_disk_io_bytes(&io_r_os, &io_w_os);

    double wall_s = (double)(t_end - ctx->t0_ns) * 1e-9;
    double cpu_s  = ((double)(u - ctx->cpu_user_start) + (double)(s - ctx->cpu_sys_start)) * 1e-9;
    double util   = cpu_s / (wall_s > 0 ? wall_s : 1.0) * 100.0;
    uint64_t rss_delta = rss_end > ctx->rss_start ? rss_end - ctx->rss_start : 0;
    uint64_t io_r = io_r_os > ctx->io_read_start ? io_r_os - ctx->io_read_start : 0;
    uint64_t io_w = io_w_os > ctx->io_write_start ? io_w_os - ctx->io_write_start : 0;

    char suffix[160];

    fprintf(out, "======================================================================\n");
    fprintf(out, "   RCRE Profiling Report (totals + per-scope)\n");
    fprintf(out, "======================================================================\n");

    /* per-scope aggregates */
    RCRE_ProfAgg* agg = (RCRE_ProfAgg*)calloc(256, sizeof(RCRE_ProfAgg));
    if (!agg) return;
    size_t n = aggregate(ctx, agg, 256);
    fprintf(out, "  Scope                                time       count  max\n");
    double max_total = 0;
    for (size_t i = 0; i < n; i++) {
        if (agg[i].total_ns > max_total) max_total = (double)agg[i].total_ns;
        snprintf(suffix, sizeof(suffix), "%8.3f ms  %6zu  %8.3f ms",
                 agg[i].total_ns / 1e6, agg[i].count, agg[i].max_ns / 1e6);
        bar(agg[i].name, (double)agg[i].total_ns, max_total, suffix, out);
    }
    free(agg);

    fprintf(out, "----------------------------------------------------------------------\n");
    fprintf(out, "  TOTAL CONSUMPTION\n");
    snprintf(suffix, sizeof(suffix), "%.3f s", wall_s);
    bar("Wall time", wall_s, wall_s > 0 ? wall_s : 1.0, suffix, out);
    snprintf(suffix, sizeof(suffix), "%.3f s  (%.0f%% of wall, %.1f core-equivalent)",
             cpu_s, util, util / 100.0);
    bar("CPU (user+sys)", cpu_s, wall_s > 0 ? wall_s : 1.0, suffix, out);
    snprintf(suffix, sizeof(suffix), "start %8.1f MB   peak %8.1f MB   end %8.1f MB",
             (double)ctx->rss_start / 1e6, (double)ctx->rss_peak / 1e6, (double)rss_end / 1e6);
    bar("RAM RSS", (double)ctx->rss_peak / 1e6, 100.0, suffix, out);
    snprintf(suffix, sizeof(suffix), "%+.1f MB", (double)(int64_t)rss_delta / 1e6);
    bar("RAM delta", (double)rss_delta / 1e6, 100.0, suffix, out);
    if (ctx->arena_hwm) {
        snprintf(suffix, sizeof(suffix), "HWM %8.1f MB   %zu allocs   %zu rewinds",
                 (double)ctx->arena_hwm / 1e6, ctx->arena_allocs, ctx->arena_rewinds);
        bar("Arena (RCRE_Arena)", (double)ctx->arena_hwm / 1e6,
            (double)ctx->arena_hwm / 1e6 > 0 ? (double)ctx->arena_hwm / 1e6 : 1.0, suffix, out);
    }
    snprintf(suffix, sizeof(suffix), "%.2f MB  (%.2f MB/s)",
             (double)ctx->io_read_app / 1e6, (double)ctx->io_read_app / 1e6 / (wall_s > 0 ? wall_s : 1.0));
    bar("Storage read (engine)", (double)ctx->io_read_app / 1e6, 16.0, suffix, out);
    snprintf(suffix, sizeof(suffix), "%.2f MB  (%.2f MB/s)",
             (double)ctx->io_write_app / 1e6, (double)ctx->io_write_app / 1e6 / (wall_s > 0 ? wall_s : 1.0));
    bar("Storage write (engine)", (double)ctx->io_write_app / 1e6, 16.0, suffix, out);
    if (ctx->os_io_available) {
        snprintf(suffix, sizeof(suffix), "read %.2f MB + write %.2f MB (OS-level)",
                 (double)io_r / 1e6, (double)io_w / 1e6);
        bar("Storage (OS probe)", (double)(io_r + io_w) / 1e6, 16.0, suffix, out);
    } else {
        fprintf(out, "  %-24s [                        ] (OS probe unavailable: macOS strict-C23; engine ledger is source of truth)\n", "Storage (OS probe)");
    }
    fprintf(out, "  Events: %zu valid in ring, %zu total recorded, %zu dropped (ring %zu)\n",
            ring_valid(ctx), ctx->count, ctx->dropped, ctx->cap);
    fprintf(out, "======================================================================\n");
}

/* ========================================================================= */
/* Exporter 2: Chrome Tracing / Perfetto JSON                                 */
/* ========================================================================= */

static void json_escape(const char* s, char* out, size_t cap) {
    size_t o = 0;
    for (size_t i = 0; s[i] && o + 2 < cap; i++) {
        char c = s[i];
        if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = c; }
        else if ((unsigned char)c < 0x20) { o += (size_t)snprintf(out + o, cap - o, "\\u%04x", c); }
        else out[o++] = c;
    }
    out[o] = '\0';
}

bool rcre_prof_export_perfetto_json(RCRE_ProfContext* ctx, const char* filepath) {
    if (!ctx) return false;
    FILE* f = fopen(filepath, "w");
    if (!f) return false;
    uint64_t t0 = ctx->t0_ns;
    fprintf(f, "[\n");
    int first = 1;
    const size_t n_valid = ring_valid(ctx);
    for (size_t i = 0; i < n_valid; i++) {
        const RCRE_ProfEvent* e = &ctx->ring[ring_slot(ctx, i)];
        char esc[96];
        json_escape(e->name, esc, sizeof(esc));
        double ts_us = (double)(e->ts_ns > t0 ? e->ts_ns - t0 : 0) / 1000.0;
        if (!first) fprintf(f, ",\n");
        first = 0;
        if (e->kind == 0) {
            fprintf(f, "  {\"name\": \"%s\", \"cat\": \"Compute\", \"ph\": \"X\", \"ts\": %.3f, "
                       "\"dur\": %.3f, \"pid\": 1, \"tid\": 1, \"args\": {\"depth\": %u}}",
                    esc, ts_us, (double)e->dur_ns / 1000.0, (unsigned)e->depth);
        } else {
            fprintf(f, "  {\"name\": \"%s\", \"cat\": \"Memory\", \"ph\": \"C\", \"ts\": %.3f, "
                       "\"pid\": 1, \"tid\": 1, \"args\": {\"Value\": %.3f}}",
                    esc, ts_us, e->value);
        }
    }
    fprintf(f, "\n]\n");
    fclose(f);
    return true;
}

/* ========================================================================= */
/* Exporter 3: self-contained HTML dashboard (CPU / RAM / Storage totals)     */
/* ========================================================================= */

bool rcre_prof_export_html(RCRE_ProfContext* ctx, const char* filepath) {
    if (!ctx) return false;
    uint64_t t_end = rcre_prof_now_ns();
    uint64_t rss_end = os_rss_bytes();
    if (rss_end > ctx->rss_peak) ctx->rss_peak = rss_end;
    uint64_t u, s;
    rcre_prof_cpu_time_ns(&u, &s);
    double wall_s = (double)(t_end - ctx->t0_ns) * 1e-9;
    double cpu_s  = ((double)(u - ctx->cpu_user_start) + (double)(s - ctx->cpu_sys_start)) * 1e-9;
    double util   = cpu_s / (wall_s > 0 ? wall_s : 1.0) * 100.0;
    uint64_t rss_delta = rss_end > ctx->rss_start ? rss_end - ctx->rss_start : 0;
    double io_r = (double)ctx->io_read_app / 1e6;
    double io_w = (double)ctx->io_write_app / 1e6;

    RCRE_ProfAgg* agg = (RCRE_ProfAgg*)calloc(256, sizeof(RCRE_ProfAgg));
    if (!agg) return false;
    size_t n = aggregate(ctx, agg, 256);

    FILE* f = fopen(filepath, "w");
    if (!f) { free(agg); return false; }

    fprintf(f,
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\n"
        "<title>RCRE Profiling Report</title>\n"
        "<style>\n"
        " body { font-family: -apple-system, 'Segoe UI', Roboto, monospace; background: #0d1117; color: #e6edf3; margin: 24px; }\n"
        " h1 { font-size: 20px; } h2 { font-size: 14px; color: #8b949e; margin: 24px 0 8px; }\n"
        " .cards { display: flex; gap: 16px; flex-wrap: wrap; }\n"
        " .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; min-width: 240px; flex: 1; }\n"
        " .card h3 { margin: 0 0 12px; font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }\n"
        " .big { font-size: 26px; font-weight: 700; }\n"
        " .sub { color: #8b949e; font-size: 12px; margin-top: 4px; }\n"
        " .bar-bg { background: #21262d; border-radius: 4px; height: 10px; margin-top: 10px; overflow: hidden; }\n"
        " .bar-fg { height: 100%%; border-radius: 4px; }\n"
        " .row { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 13px; }\n"
        " .row .lbl { width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }\n"
        " .row .track { flex: 1; background: #21262d; border-radius: 4px; height: 14px; }\n"
        " .row .fill { height: 14px; border-radius: 4px; background: #388bfd; }\n"
        " .row .val { width: 150px; text-align: right; color: #8b949e; }\n"
        " .foot { color: #484f58; font-size: 11px; margin-top: 28px; }\n"
        "</style></head><body>\n"
        "<h1>RCRE Profiling Report</h1>");

    /* --- total-consumption cards --- */
    fprintf(f, "<h2>Total consumption</h2>\n<div class=\"cards\">\n");

    fprintf(f, "<div class=\"card\"><h3>CPU</h3>"
               "<div class=\"big\">%.2f s</div>"
               "<div class=\"sub\">user+sys &middot; %.3f s wall &middot; %.1f core-equivalent</div>"
               "<div class=\"bar-bg\"><div class=\"bar-fg\" style=\"width: %.1f%%; background: #388bfd\"></div></div>"
               "<div class=\"sub\">%.0f%% of wall time</div></div>\n",
            cpu_s, wall_s, util / 100.0, util > 100.0 ? 100.0 : util, util);

    fprintf(f, "<div class=\"card\"><h3>RAM (RSS)</h3>"
               "<div class=\"big\">%.1f MB</div>"
               "<div class=\"sub\">peak &middot; start %.1f MB &rarr; end %.1f MB (&Delta; %+.1f MB)</div>"
               "<div class=\"bar-bg\"><div class=\"bar-fg\" style=\"width: %.1f%%; background: #3fb950\"></div></div>"
               "<div class=\"sub\">arena HWM %.1f MB (%zu allocs, %zu rewinds)</div></div>\n",
            (double)ctx->rss_peak / 1e6, (double)ctx->rss_start / 1e6, (double)rss_end / 1e6,
            (double)(int64_t)rss_delta / 1e6,
            (double)ctx->rss_peak / 1e6 / 1024.0 * 100.0 > 100.0 ? 100.0 : (double)ctx->rss_peak / 1e6 / 1024.0 * 100.0,
            (double)ctx->arena_hwm / 1e6, ctx->arena_allocs, ctx->arena_rewinds);

    fprintf(f, "<div class=\"card\"><h3>Storage (engine I/O)</h3>"
               "<div class=\"big\">%.1f MB</div>"
               "<div class=\"sub\">read %.2f MB (%.1f MB/s) &middot; write %.2f MB (%.1f MB/s)</div>"
               "<div class=\"bar-bg\"><div class=\"bar-fg\" style=\"width: %.1f%%; background: #d29922\"></div></div>"
               "<div class=\"sub\">total %zu events, %zu dropped</div></div>\n",
            io_r + io_w, io_r, io_r / (wall_s > 0 ? wall_s : 1.0), io_w, io_w / (wall_s > 0 ? wall_s : 1.0),
            (io_r + io_w) / 32.0 * 100.0 > 100.0 ? 100.0 : (io_r + io_w) / 32.0 * 100.0,
            ctx->count, ctx->dropped);

    fprintf(f, "</div>\n");

    /* --- per-scope bars --- */
    fprintf(f, "<h2>Scope breakdown (by total time)</h2>\n");
    double max_total = n > 0 ? (double)agg[0].total_ns : 1.0;
    for (size_t i = 0; i < n && i < 64; i++) {
        double w = (double)agg[i].total_ns / max_total * 100.0;
        fprintf(f, "<div class=\"row\"><div class=\"lbl\">%s</div>"
                   "<div class=\"track\"><div class=\"fill\" style=\"width: %.2f%%\"></div></div>"
                   "<div class=\"val\">%.3f ms &middot; %zu&times; &middot; max %.3f ms</div></div>\n",
                agg[i].name, w, agg[i].total_ns / 1e6, agg[i].count, agg[i].max_ns / 1e6);
    }
    fprintf(f, "<div class=\"foot\">Generated by rcre_prof &middot; wall %.3f s &middot; "
               "open alongside ui.perfetto.dev for the timeline view</div>\n", wall_s);
    fprintf(f, "</body></html>\n");
    fclose(f);
    free(agg);
    return true;
}
