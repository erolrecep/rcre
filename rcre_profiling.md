# RCRE Profiling & Telemetry Subsystem (`rcre_prof`)

A high-performance, zero-dependency, ISO C23 profiling and hardware telemetry engine engineered specifically for low-latency tensor computation, computer vision, and machine learning runtimes.

---

## 1. Overview & Architecture

Modern systems programming in AI and computer vision demands visibility across four critical dimensions without disturbing the cache hierarchy or stalling pipelines. `rcre_prof` addresses this by delivering **$< 7\text{ ns}$ instrumented scope latency** with zero heap allocations during measurement.

```
+-----------------------------------------------------------------------------------------+
|                                    RCRE Engine                                          |
|   HAL GEMM  |  CV Filters  |  HNSW Vector Search  |  Autograd DAG  |  NanoGPT Inference |
+-----------------------------------------------------------------------------------------+
                                             │
                       RCRE_PROFILE_SCOPE / RCRE_PROFILE_END
                                             ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                 rcre_prof Engine                                        |
|                                                                                         |
|  Pillar 1: Hardware Cycles       Pillar 2: Dual-Layer RAM    Pillar 3: I/O & CPU        |
|  • ARM64 mrs cntvct_el0 (0.4ns)  • Layer A: Arena Watermark  • getrusage (User+Sys CPU) |
|  • x86-64 rdtsc                  • Layer B: OS Physical RSS  • Engine App-Level Ledger  |
|  • Monotonic epoch sync          • Zero-syscall watermark    • Linux /proc/self/io      |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                             │
                        Lockless Circular Event Ring (Power-of-2)
                                             ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                     Exporters                                           |
|   1. ANSI Terminal Table    2. Chrome / Perfetto JSON       3. Self-Contained HTML      |
|      (Totals & Bar Charts)     (Open in ui.perfetto.dev)       (Interactive Dark Theme) |
+─────────────────────────────────────────────────────────────────────────────────────────+
```

### The Four Pillars
1. **Pillar 1: High-Frequency Scoped Event Tracing**
   * **ARM64**: Uses EL0-accessible virtual timer `mrs cntvct_el0` (~0.45 ns per read).
   * **x86-64**: Uses `rdtsc`.
   * **Dual-Epoch Synchronization**: Keeps raw scope durations in cycle counts to avoid timer skew, while computing event timestamps in the `CLOCK_MONOTONIC` epoch via $\text{ts} = \text{now}_{\text{monotonic}} - \text{dur}_{\text{ns}}$.
2. **Pillar 2: Dual-Layer Memory Telemetry**
   * **Layer A (Engine Arena)**: Direct zero-syscall sampling of `RCRE_Arena` (`high_watermark`, `alloc_count`, `rewind_count`). Watermarks survive temporary stack rewinds (`rcre_arena_temp_end`), exposing peak transient usage.
   * **Layer B (OS Physical RSS)**: Mach task info (`MACH_TASK_BASIC_INFO`) on macOS and `/proc/self/statm` on Linux.
3. **Pillar 3: CPU Time & Storage I/O Accounting**
   * **CPU Time**: `getrusage(RUSAGE_SELF, ...)` computing exact user and system time with microsecond resolution.
   * **Engine I/O Ledger**: Cross-platform application-level ledger (`rcre_prof_io_record`) capturing exact bytes read/written during model checkpoints and dataset streaming, bypassing OS-level buffering discrepancies.
   * **OS Disk Probe**: Linux `/proc/self/io` byte-exact counter.
4. **Pillar 4: Zero-Dependency Visualizations & Exporters**
   * Formatted ANSI terminal report with bar charts and core-equivalent utilization.
   * Native Chrome Tracing / Perfetto JSON format (`ui.perfetto.dev` compatible).
   * Standalone, self-contained HTML dashboard with modern dark theme cards.

---

## 2. Quickstart: Running the Profiler

The RCRE repository includes an automated profiling target that exercises every subsystem (Linear Arena, NEON/AVX2 GEMM, 1080p Image Filtering, NanoGPT autoregressive generation, and disk checkpointing).

### 1. Execute the Profiling Workload
From the `apps/rcre` directory, run:

```bash
make profile
```

This compiles the engine with `-DRCRE_ENABLE_PROFILING -O3`, runs the full workload, prints the ANSI terminal summary, and outputs two artifact files into `prof/`:
* `prof/rcre_profile.json`: Chrome Tracing / Perfetto timeline trace.
* `prof/rcre_profile.html`: Standalone HTML dashboard.

### 2. Inspect the Terminal Report
The terminal displays both per-scope breakdown and total resource consumption:

```text
======================================================================
   RCRE Profiling Report (totals + per-scope)
======================================================================
  Scope                                time       count  max
  hal_gemm_512             [############################]   78.514 ms       1    78.514 ms
  gpt_inference_64tok      [#######                     ]   21.269 ms       1    21.269 ms
  gpt_forward_token        [#######                     ]   20.608 ms      64     0.366 ms
  cv_grayscale_1080p       [####                        ]   12.877 ms       1    12.877 ms
  gpt_init                 [###                         ]    8.445 ms       1     8.445 ms
  gpt_sample_token         [                            ]    0.651 ms      64     0.012 ms
  checkpoint_load          [                            ]    0.462 ms       1     0.462 ms
  checkpoint_save          [                            ]    0.380 ms       1     0.380 ms
  arena_setup              [                            ]    0.003 ms       1     0.003 ms
----------------------------------------------------------------------
  TOTAL CONSUMPTION
  Wall time                [############################] 0.123 s
  CPU (user+sys)           [########################### ] 0.122 s  (100% of wall, 1.0 core-equivalent)
  RAM RSS                  [#########                   ] start 7.9 MB   peak 33.6 MB   end 33.6 MB
  RAM delta                [#######                     ] +25.7 MB
  Arena (RCRE_Arena)       [############################] HWM 25.6 MB   719 allocs   64 rewinds
  Storage read (engine)    [######                      ] 3.47 MB  (28.24 MB/s)
  Storage write (engine)   [######                      ] 3.47 MB  (28.24 MB/s)
  Events: 147 valid in ring, 147 total recorded, 0 dropped (ring 8192)
======================================================================
```

### 3. Open Interactive Trace in Perfetto UI
1. Open your browser and navigate to [**ui.perfetto.dev**](https://ui.perfetto.dev).
2. Click **Open trace file** in the left sidebar (or drag-and-drop).
3. Select `apps/rcre/prof/rcre_profile.json`.
4. Use standard navigation:
   * **`W` / `S`**: Zoom in / out.
   * **`A` / `D`**: Pan left / right.
   * Inspect individual token generation slices, GEMM steps, and memory counter tracks (`ARENA_HWM_KB`, `RSS_KB`, `GFLOPS`).

### 4. Open the Self-Contained HTML Dashboard
View the visual dashboard directly in any browser without an internet connection:

```bash
# macOS
open prof/rcre_profile.html

# Linux
xdg-open prof/rcre_profile.html
```

---

## 3. Programmatic C23 API Guide

You can integrate `rcre_prof` into your own applications or custom kernels using the single-header distribution ([`dist/rcre.h`](dist/rcre.h)) or the modular header ([`include/rcre_prof.h`](include/rcre_prof.h)).

### Step 1: Initialize Context
```c
#include "rcre.h"
#include "rcre_prof.h"

int main(void) {
    // Allocate a circular ring buffer with capacity for 4096 events (auto-rounded to power of 2)
    RCRE_ProfContext* prof = rcre_prof_init(4096);
    if (!prof) return 1;

    // Optional: Register as global process-wide context for engine-internal scopes
    rcre_prof_set_global(prof);
    
    // ... code ...
    
    rcre_prof_free(prof);
    return 0;
}
```

### Step 2: Instrument Scopes (Hot Path)
Wrap functions, loops, or critical execution zones with `RCRE_PROFILE_SCOPE` and `RCRE_PROFILE_END`.

```c
void run_inference(RCRE_ProfContext* prof, RCRE_Arena* arena) {
    RCRE_PROFILE_SCOPE(prof, "inference_pipeline", z_pipe);

    // Inner nested scope
    {
        RCRE_PROFILE_SCOPE(prof, "attention_qkv", z_qkv);
        // MatMul kernel
        RCRE_PROFILE_END(z_qkv);
    }

    RCRE_PROFILE_END(z_pipe);
}
```
*Note: Scopes incur only $4 - 7\text{ ns}$ overhead. If `prof` is `NULL`, the macro safely executes as a fast duration probe without recording to the ring.*

### Step 3: Record Continuous Telemetry & Counters
Capture dynamic metrics like throughput, memory watermark, or bandwidth:

```c
// Track linear arena memory utilization
rcre_prof_sample_arena(prof, arena);

// Sample OS-level physical memory (RSS)
rcre_prof_sample_rss(prof);

// Record custom continuous metrics (plotted as counter tracks in Perfetto)
rcre_prof_mark(prof, "GFLOPS", 72.5);
rcre_prof_mark(prof, "TOK_PER_SEC", 3000.0);
```

### Step 4: Record Storage I/O
Record file ingestion or checkpoint operations into the application-level ledger:

```c
// Track disk write
rcre_prof_io_record(prof, /*is_write=*/true, checkpoint_bytes);

// Track disk read
rcre_prof_io_record(prof, /*is_write=*/false, dataset_bytes);
```

### Step 5: Export Telemetry
At the end of your run, generate reports:

```c
// 1. ANSI summary to stdout or stderr
rcre_prof_print_summary(prof, stdout);

// 2. Perfetto JSON trace for ui.perfetto.dev
rcre_prof_export_perfetto_json(prof, "inference_trace.json");

// 3. Standalone HTML dashboard
rcre_prof_export_html(prof, "inference_report.html");
```

---

## 4. Verification & Quality Assurance

The profiling subsystem is continuously tested and verified under strict compiler sanitizers:

### Run Unit Tests
```bash
make bin/test_profiler && ./bin/test_profiler
```
*Tests timing primitives, scope mechanics, ring wrap-around, null safety, counter events, arena telemetry, and verifies that overhead remains strictly within budget.*

### Run AddressSanitizer & UBSan
```bash
make test-sanitizers
```
*Guarantees zero memory leaks, zero heap buffer overflows, and zero undefined behavior.*

### Standalone Header Portability Test
```bash
make test-portable
```
*Extracts `dist/rcre.h` to an isolated temporary directory and compiles a standalone test binary with zero external dependencies.*

---

## 5. Overhead & Performance Comparison

| Profiler | Integration Type | Overhead / Scope | Dependencies | Arena Memory Tracking | Chrome / Perfetto Export |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`rcre_prof`** | **Header/Engine Native (C23)** | **$4.3 - 6.7\text{ ns}$** | **None (Zero)** | **Native (Watermark + Rewinds)** | **Yes (Native JSON)** |
| **Tracy Profiler** | Client/Server Protocol | $15 - 25\text{ ns}$ | C++17, Cap'n Proto, Pthreads | Manual markup | Yes (via Tracy GUI) |
| **Linux `perf`** | Kernel Sampling (PMU) | Variable (Syscall) | Linux only, Root permissions | No (OS pages only) | No (Text / FlameGraph) |
| **Instruments** | OS XPC Profiling | $50 - 200\text{ ns}$ | macOS proprietary GUI | No (Virtual memory only) | No (Trace bundles only) |

`rcre_prof` delivers domain-specific visibility into bare-metal memory arenas and tensor operations while maintaining strict zero-dependency portability across macOS, Linux, and WebAssembly.
