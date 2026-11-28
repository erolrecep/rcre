# RCRE (ISO C23) vs Go (1.27) High-Performance Web Server Benchmark

A rigorous, reproducible, apples-to-apples systems benchmark comparing a zero-malloc ISO C23 `kqueue` HTTP/1.1 web server built on **RCRE** against **Go (`net/http`)** in both **idiomatic** and **tuned zero-allocation** configurations.

## Quick Start

```bash
cd projects/rcre_server_vs_go_benchmark
# (or from apps/rcre: make project_server_bench_verify / make project_server_bench_bench)

# 1. Generate deterministic shared fixtures (10k-doc BM25 corpus, 100-tree GBDT, 4KB UTF-8, 64x16 JSON)
make fixtures

# 2. Build Release (-O3 -flto -mcpu=native), ASan/UBSan, and Go binaries
make build

# 3. Run ASan/UBSan safety gate + HTTP/1.1 conformance + cross-server byte parity checks
make verify

# 4. Run full benchmark matrix (Phases A-D) & generate results/BENCHMARK_REPORT.md
make bench
```

## Architecture Comparison

| Server | Concurrency Model | Memory Model | Core Primitives |
| :--- | :--- | :--- | :--- |
| **`rcre`** (`bin/server-rcre`) | Shared-nothing worker-per-core `kqueue` reactor (1-worker folds `accept()` inline; $N>1$ uses lock-free SPSC FD ring + `EVFILT_USER`) | Zero `malloc` in request hot path; slab-pooled 16 KB I/O buffers released on idle (~96 B/idle socket); per-worker 4 MB `RCRE_Arena` with $\mathcal{O}(1)$ reset | `librcre.a`: `rcre_str.h` (`RCRE_Slice`, `RCRE_StrBuf`, NEON `rcre_utf8_validate`), `rcre_nlp.h` (`RCRE_BM25Index`), `rcre_ml_gbdt_predict` |
| **`go-idiomatic`** (`bin/server-go --mode idiomatic`) | Standard `net/http` goroutine-per-connection (`GOMAXPROCS=N`) | Standard Go GC heap (`io.ReadAll`, `encoding/json`) | Pure Go 1.27 stdlib + CSR BM25 + flat GBDT |
| **`go-tuned`** (`bin/server-go --mode tuned`) | Standard `net/http` (`GOMAXPROCS=N`) | `sync.Pool` read/BM25/GBDT buffers, zero-alloc JSON scanner & writer, $\mathcal{O}(\text{touched})$ BM25 cleanup | Pure Go 1.27 stdlib tuned for minimal GC pressure |

## Workloads

- **W1 (`GET /health`)**: Pure HTTP/1.1 I/O baseline (`text/plain`, `OK`).
- **W2 (`POST /api/v1/validate`)**: 4,096-byte mixed ASCII/UTF-8 payload validation (strict RFC 3629 UTF-8, codepoint count, newline count, FNV-1a 64-bit checksum).
- **W3 (`GET /api/v1/search?q=...&k=10`)**: Stateful read-heavy Okapi BM25 top-10 search over 10,000 indexed documents.
- **W4 (`POST /api/v1/infer`)**: Compute-bound tabular ML inference (64 samples × 16 features through a 100-tree, depth-6 flat GBDT ensemble).
- **W5 (`GET /health` @ $C \in \{64, 256, 1024, 2048\}$)**: High-concurrency keep-alive connection fan-out and RSS scaling stress test.

## Outputs

- Raw JSON metrics (throughput, latency histograms, 100 ms sampled RSS, kernel `wait4()` CPU & context switches): `results/raw_results.json`
- Formatted Markdown comparison report: `results/BENCHMARK_REPORT.md`
