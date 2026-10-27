/**
 * @file rcre_prof.h
 * @brief RCRE Native Profiling & Telemetry Subsystem (ISO C23, zero-dependency)
 *
 * Implements the architecture from rcre_profiling_task.md:
 *   Pillar 1 — CPU timing: hardware cycle counter for durations (ARM64
 *              cntvct_el0, x86-64 RDTSC, CLOCK_MONOTONIC fallback);
 *              CLOCK_MONOTONIC for timeline timestamps (epoch-safe:
 *              ts = now - dur, computed at scope end).
 *   Pillar 2 — RAM: dual layer: RCRE_Arena watermarks (byte-exact) + OS RSS/peak.
 *   Pillar 3 — Storage: application-level I/O ledger (exact, cross-platform) +
 *              OS-level bytes where the platform exposes them (Linux /proc/self/io).
 *   Pillar 4 — Export: ANSI terminal summary, Chrome Tracing/Perfetto JSON,
 *              and a self-contained HTML dashboard (CPU / RAM / Storage totals).
 *
 * Design rules (validated in the apps/c-performance Stage-1 sandbox):
 *   - Scope names are POINTERS (callers pass string literals). Copying the
 *     name into the zone struct costs ~12 ns and breaks the < 25 ns per-scope
 *     overhead budget; pointer indirection keeps total overhead at ~7 ns.
 *   - The cycle counter and CLOCK_MONOTONIC have DIFFERENT epochs on Apple
 *     Silicon (virtualized virtual timer). Durations always come from one
 *     timer; timestamps always from CLOCK_MONOTONIC. Never mix them.
 *   - Explicit begin/end (no C23 `auto` / RAII): compiles under clang AND gcc
 *     (the CI matrix builds Linux with GCC, which lacks C23 type inference).
 *   - Passing NULL as the context disables recording with ~1 cycle-read
 *     overhead (safe default for #ifdef-free call sites).
 */

#ifndef RCRE_PROF_H
#define RCRE_PROF_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct RCRE_Arena;

/* ========================================================================= */
/* Timing primitives                                                          */
/* ========================================================================= */

/** Monotonic wall clock in nanoseconds (CLOCK_MONOTONIC epoch). */
uint64_t rcre_prof_now_ns(void);

/** Hardware cycle counter (0 = unsupported platform). */
uint64_t rcre_prof_cycles(void);

/** Cycle counter frequency in Hz (0 = unsupported/unknown). */
uint64_t rcre_prof_cycles_freq(void);

/* ---- OS probes (public for tests and for host-side dashboards) ---- */

/** Process CPU time (user + system) in nanoseconds (getrusage). */
void rcre_prof_cpu_time_ns(uint64_t* user_ns, uint64_t* sys_ns);

/** Process disk I/O bytes since start; returns 0 where the OS hides them
 *  (macOS under strict C23 — use rcre_prof_io_record for engine I/O). */
int  rcre_prof_disk_io_bytes(uint64_t* read_bytes, uint64_t* write_bytes);

/** Process resident set size in bytes. */
uint64_t rcre_prof_rss_bytes(void);

/* ========================================================================= */
/* Scoped zones (handle-based: balance every begin with an end)               */
/* ========================================================================= */

typedef struct {
    struct RCRE_ProfContext* ctx;  /* NULL = timing-only, no recording */
    uint64_t   begin;              /* timer ticks at begin */
    const char* name;              /* pointer; literal must outlive scope end */
    int        active;
} RCRE_ProfScope;

/** Starts a zone. Cost: one cycle-counter read (~0.4 ns on ARM64). */
RCRE_ProfScope rcre_prof_scope_begin(struct RCRE_ProfContext* ctx, const char* name);

/** Ends a zone; returns duration in nanoseconds (emits the X event if ctx). */
uint64_t rcre_prof_scope_end(RCRE_ProfScope* scope);

/** Converts a duration in the active timer unit to nanoseconds. */
double rcre_prof_to_ns(uint64_t duration);

/* ========================================================================= */
/* Profiler context (event ring + live counters)                              */
/* ========================================================================= */

typedef struct RCRE_ProfEvent {
    uint8_t  kind;          /* 0 = scope (X), 1 = counter (C) */
    uint8_t  depth;
    uint16_t _pad;
    char     name[48];
    uint64_t ts_ns;         /* timeline timestamp (CLOCK_MONOTONIC epoch) */
    uint64_t dur_ns;        /* scope duration (0 for counters) */
    double   value;         /* counter value */
} RCRE_ProfEvent;           /* 80 bytes */

typedef struct RCRE_ProfContext {
    RCRE_ProfEvent* ring;
    size_t  cap;
    size_t  count;
    size_t  dropped;
    int     depth;
    uint64_t t0_ns;

    /* OS baselines captured at init (for delta totals) */
    uint64_t rss_start;
    uint64_t rss_peak;
    uint64_t cpu_user_start;
    uint64_t cpu_sys_start;
    uint64_t io_read_start;
    uint64_t io_write_start;
    int      os_io_available;

    /* application-level I/O ledger (engine file ops — exact) */
    uint64_t io_read_app;
    uint64_t io_write_app;

    /* last sampled arena telemetry (byte-exact) */
    size_t   arena_hwm;
    size_t   arena_allocs;
    size_t   arena_rewinds;
} RCRE_ProfContext;

/** Creates a profiler with a ring buffer of `capacity` events (power of 2). */
RCRE_ProfContext* rcre_prof_init(size_t capacity);
void rcre_prof_free(RCRE_ProfContext* ctx);

/** Optional process-wide context so engine-internal call sites (guarded by
 *  RCRE_ENABLE_PROFILING) can record without API changes. NULL by default. */
RCRE_ProfContext* rcre_prof_global(void);
void rcre_prof_set_global(RCRE_ProfContext* ctx);

/** Records a counter (C) event, e.g. GFLOPS, RSS_KB, ARENA_HWM_KB. */
void rcre_prof_mark(RCRE_ProfContext* ctx, const char* name, double value);

/** Samples OS memory (RSS); updates ctx->rss_peak; emits an RSS_KB counter. */
void rcre_prof_sample_rss(RCRE_ProfContext* ctx);

/** Samples arena telemetry (byte-exact watermarks); emits ARENA_HWM_KB. */
void rcre_prof_sample_arena(RCRE_ProfContext* ctx, struct RCRE_Arena* arena);

/** Records engine file I/O (application layer, exact, cross-platform). */
void rcre_prof_io_record(RCRE_ProfContext* ctx, bool is_write, uint64_t bytes);

/** Samples OS-level disk I/O (Linux: /proc/self/io; macOS strict C23: none). */
void rcre_prof_sample_io(RCRE_ProfContext* ctx);

/* ========================================================================= */
/* Exporters                                                                  */
/* ========================================================================= */

/** ANSI terminal report: per-scope aggregates + TOTAL CPU / RAM / Storage. */
void rcre_prof_print_summary(RCRE_ProfContext* ctx, FILE* out);

/** Chrome Tracing / Perfetto JSON (ui.perfetto.dev compatible). */
bool rcre_prof_export_perfetto_json(RCRE_ProfContext* ctx, const char* filepath);

/** Self-contained HTML dashboard (CPU / RAM / Storage cards + scope bars). */
bool rcre_prof_export_html(RCRE_ProfContext* ctx, const char* filepath);

/* ========================================================================= */
/* Ergonomic macros (GCC-safe: no `auto`, no C23 attributes)                  */
/* ========================================================================= */

#define RCRE_PROF_PASTE_(a, b) a ## b
#define RCRE_PROF_PASTE(a, b) RCRE_PROF_PASTE_(a, b)

/**
 * Usage (explicit handle — nestable, GCC-safe, no C23 auto):
 *   RCRE_PROFILE_SCOPE(ctx, "gemm", z);
 *   ... work ...
 *   RCRE_PROFILE_END(z);
 *
 * No-op recording when ctx is NULL (timing stays available via z).
 */
#define RCRE_PROFILE_SCOPE(ctx_, name_, var_) \
    RCRE_ProfScope var_ = rcre_prof_scope_begin((ctx_), (name_))
#define RCRE_PROFILE_END(var_) \
    rcre_prof_scope_end(&(var_))

#ifdef __cplusplus
}
#endif

#endif /* RCRE_PROF_H */
