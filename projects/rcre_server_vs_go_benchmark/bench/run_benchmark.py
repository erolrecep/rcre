#!/usr/bin/env python3
"""
Automated, Reproducible Benchmark Harness: RCRE (C23) vs Go (1.27) Web Servers.

Collects:
  - Throughput (RPS, MB/s)
  - Latency percentiles (p50, p90, p99, p99.9, max) via oha JSON output
  - Exact kernel rusage via os.wait4(): user CPU, sys CPU, CPU-s / 1M reqs,
    peak RSS (ru_maxrss), voluntary & involuntary context switches
  - Live 100ms RSS sampling for steady-state memory footprint
  - Cold-start time to first 200 OK + binary size

Writes:
  - results/raw_results.json
  - results/BENCHMARK_REPORT.md
"""

import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
DATA = ROOT / "spec" / "data"
RESULTS_DIR = ROOT / "results"

OHA_BIN = shutil.which("oha") or "/opt/homebrew/bin/oha"
OHA_THREADS = 8  # Fixed client thread budget so client + server never oversubscribe 18 cores

SERVERS = [
    ("rcre", str(BIN / "server-rcre"), []),
    ("go-idiomatic", str(BIN / "server-go"), ["--mode", "idiomatic"]),
    ("go-tuned", str(BIN / "server-go"), ["--mode", "tuned"]),
]


def get_rss_kb(pid: int) -> float | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        if out:
            return float(out)
    except Exception:
        pass
    return None


def wait_ready_timed(url: str, t_start: float, timeout: float = 15.0) -> float:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return (time.monotonic() - t_start) * 1000.0
        except Exception:
            time.sleep(0.005)
    raise RuntimeError(f"Server did not become ready at {url}")


class RSSSampler:
    def __init__(self, pid: int, interval: float = 0.1):
        self.pid = pid
        self.interval = interval
        self.samples_mb: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            kb = get_rss_kb(self.pid)
            if kb is not None:
                self.samples_mb.append(kb / 1024.0)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def run_oha(
    url: str,
    concurrency: int,
    duration_sec: int,
    method: str = "GET",
    body_file: Path | None = None,
    content_type: str | None = None,
    rate_qps: int | None = None,
) -> dict:
    cmd = [
        OHA_BIN,
        "--no-tui",
        "--output-format", "json",
        "--worker-threads", str(OHA_THREADS),
        "-c", str(concurrency),
        "-z", f"{duration_sec}s",
        "-w",
        "-m", method,
    ]
    if body_file is not None:
        cmd.extend(["-D", str(body_file)])
    if content_type is not None:
        cmd.extend(["-T", content_type])
    if rate_qps is not None:
        cmd.extend(["-q", str(rate_qps), "--latency-correction"])
    cmd.append(url)

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"oha failed ({res.returncode}): {res.stderr}")
    return json.loads(res.stdout)


def extract_oha_metrics(data: dict) -> dict:
    summary = data.get("summary", {})
    pct = data.get("latencyPercentiles", {})
    status_dist = data.get("statusCodeDistribution", {})
    err_dist = data.get("errorDistribution", {})
    total_ok = int(status_dist.get("200", 0))
    total_err = sum(int(v) for v in err_dist.values())
    return {
        "rps": float(summary.get("requestsPerSec", 0.0)),
        "mb_per_sec": float(summary.get("sizePerSec", 0.0)) / (1024.0 * 1024.0),
        "total_ok": total_ok,
        "total_err": total_err,
        "p50_ms": float(pct.get("p50", 0.0)) * 1000.0,
        "p90_ms": float(pct.get("p90", 0.0)) * 1000.0,
        "p99_ms": float(pct.get("p99", 0.0)) * 1000.0,
        "p999_ms": float(pct.get("p99.9", 0.0)) * 1000.0,
        "max_ms": float(summary.get("slowest", 0.0)) * 1000.0,
        "mean_ms": float(summary.get("average", 0.0)) * 1000.0,
    }


def benchmark_cell(
    server_name: str,
    server_bin: str,
    extra_args: list[str],
    port: int,
    workers: int,
    workload: dict,
    concurrency: int,
    warmup_sec: int,
    measure_sec: int,
    repeats: int,
    rate_qps: int | None = None,
) -> dict:
    """
    Launches a fresh server process per cell to capture clean startup RSS,
    steady-state RSS, and exact kernel wait4() CPU/RSS/context-switch rusage.
    """
    cmd = [
        server_bin,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--workers", str(workers),
        "--data", str(DATA),
    ] + extra_args

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = f"http://127.0.0.1:{port}"
    try:
        startup_ms = wait_ready_timed(base_url + "/health", t0)
        idle_rss_mb = (get_rss_kb(proc.pid) or 0.0) / 1024.0

        url = base_url + workload["path"]
        # Warmup phase (not counted in measurement)
        _ = run_oha(
            url,
            concurrency=concurrency,
            duration_sec=warmup_sec,
            method=workload["method"],
            body_file=workload.get("body_file"),
            content_type=workload.get("content_type"),
            rate_qps=rate_qps,
        )

        # Measurement repeats with live RSS sampling
        sampler = RSSSampler(proc.pid, interval=0.1)
        sampler.start()
        runs = []
        total_ok_all = 0
        for _ in range(repeats):
            raw = run_oha(
                url,
                concurrency=concurrency,
                duration_sec=measure_sec,
                method=workload["method"],
                body_file=workload.get("body_file"),
                content_type=workload.get("content_type"),
                rate_qps=rate_qps,
            )
            m = extract_oha_metrics(raw)
            runs.append(m)
            total_ok_all += m["total_ok"]
        sampler.stop()

    finally:
        proc.send_signal(signal.SIGTERM)
        _, _, rusage = os.wait4(proc.pid, 0)
        time.sleep(0.5)  # brief thermal/scheduler cooldown between cells

    # Note: on macOS ru_maxrss is in BYTES
    peak_rss_mb = float(rusage.ru_maxrss) / (1024.0 * 1024.0)
    steady_rss_mb = statistics.median(sampler.samples_mb) if sampler.samples_mb else peak_rss_mb
    user_cpu_s = float(rusage.ru_utime)
    sys_cpu_s = float(rusage.ru_stime)
    total_cpu_s = user_cpu_s + sys_cpu_s

    # Approximate total requests served including warmup for CPU/1M req efficiency
    est_total_reqs = total_ok_all * (1.0 + (warmup_sec / max(1, measure_sec * repeats)))
    cpu_sec_per_1m = (total_cpu_s / est_total_reqs) * 1e6 if est_total_reqs > 0 else 0.0

    def med(key: str) -> float:
        return statistics.median(r[key] for r in runs)

    def iqr(key: str) -> float:
        vals = sorted(r[key] for r in runs)
        if len(vals) < 3:
            return vals[-1] - vals[0]
        return vals[-1] - vals[0]

    return {
        "server": server_name,
        "workload": workload["id"],
        "workers": workers,
        "concurrency": concurrency,
        "rate_qps": rate_qps,
        "startup_ms": round(startup_ms, 2),
        "idle_rss_mb": round(idle_rss_mb, 2),
        "steady_rss_mb": round(steady_rss_mb, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "rps_median": round(med("rps"), 1),
        "rps_spread": round(iqr("rps"), 1),
        "mb_per_sec": round(med("mb_per_sec"), 2),
        "p50_ms": round(med("p50_ms"), 3),
        "p90_ms": round(med("p90_ms"), 3),
        "p99_ms": round(med("p99_ms"), 3),
        "p999_ms": round(med("p999_ms"), 3),
        "max_ms": round(med("max_ms"), 3),
        "user_cpu_s": round(user_cpu_s, 3),
        "sys_cpu_s": round(sys_cpu_s, 3),
        "cpu_s_per_1m_req": round(cpu_sec_per_1m, 3),
        "vol_ctx_sw": int(rusage.ru_nvcsw),
        "invol_ctx_sw": int(rusage.ru_nivcsw),
        "total_ok": total_ok_all,
        "total_err": sum(r["total_err"] for r in runs),
    }


def hw_info() -> dict:
    def sysctl(key: str) -> str:
        try:
            return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        except Exception:
            return "unknown"

    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "logical_cores": sysctl("hw.ncpu"),
        "perf_cores": sysctl("hw.perflevel0.physicalcpu"),
        "eff_cores": sysctl("hw.perflevel1.physicalcpu"),
        "ram_gb": round(int(sysctl("hw.memsize") or "0") / (1024**3), 1),
        "os": platform.platform(),
        "go_version": subprocess.check_output(["go", "version"], text=True).strip(),
        "clang_version": subprocess.check_output(["clang", "--version"], text=True).splitlines()[0],
        "oha_version": subprocess.check_output([OHA_BIN, "--version"], text=True).strip(),
        "rcre_bin_kb": round((BIN / "server-rcre").stat().st_size / 1024.0, 1),
        "go_bin_kb": round((BIN / "server-go").stat().st_size / 1024.0, 1),
    }


def generate_markdown_report(results: dict, out_path: Path) -> None:
    hw = results["hardware"]
    lines = [
        "# RCRE (ISO C23) vs Go (1.27) High-Performance Web Server Benchmark Report",
        "",
        "## 1. Environment & Binary Footprint",
        "",
        f"- **Host CPU**: `{hw['cpu']}` ({hw['logical_cores']} cores: {hw['perf_cores']} + {hw['eff_cores']} physical, {hw['ram_gb']} GB unified RAM)",
        f"- **OS**: `{hw['os']}`",
        f"- **Compilers**: `{hw['clang_version']}` (`-std=c2x -O3 -flto -mcpu=native`) vs `{hw['go_version']}`",
        f"- **Load Generator**: `{hw['oha_version']}` (`--worker-threads {OHA_THREADS}`, keep-alive enabled)",
        f"- **Binary Size**: `server-rcre` = **{hw['rcre_bin_kb']:.1f} KB** vs `server-go` = **{hw['go_bin_kb']:.1f} KB** (**{hw['go_bin_kb'] / max(1.0, hw['rcre_bin_kb']):.1f}×** smaller)",
        "",
        "---",
        "",
        "## 2. Core Workload Matrix (4 Workers, C = 128, Median of 3 × 5s Runs)",
        "",
        "| Workload | Server | RPS | Speedup vs Go-Idiom | Speedup vs Go-Tuned | p50 (ms) | p99 (ms) | p99.9 (ms) | Steady RSS (MB) | Peak RSS (MB) | CPU-s / 1M req | Errors |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    core = results["core_matrix"]
    by_w: dict[str, dict[str, dict]] = {}
    for r in core:
        by_w.setdefault(r["workload"], {})[r["server"]] = r

    for wid in ["W1_health", "W2_validate", "W3_search", "W4_infer"]:
        if wid not in by_w:
            continue
        group = by_w[wid]
        gi_rps = group["go-idiomatic"]["rps_median"]
        gt_rps = group["go-tuned"]["rps_median"]
        for sname in ["rcre", "go-tuned", "go-idiomatic"]:
            r = group[sname]
            sp_i = f"**{r['rps_median'] / gi_rps:.2f}×**" if sname == "rcre" else f"{r['rps_median'] / gi_rps:.2f}×"
            sp_t = f"**{r['rps_median'] / gt_rps:.2f}×**" if sname == "rcre" else f"{r['rps_median'] / gt_rps:.2f}×"
            lines.append(
                f"| `{wid}` | **{sname}** | {r['rps_median']:,.1f} | {sp_i} | {sp_t} | "
                f"{r['p50_ms']:.3f} | {r['p99_ms']:.3f} | {r['p999_ms']:.3f} | "
                f"{r['steady_rss_mb']:.1f} | {r['peak_rss_mb']:.1f} | {r['cpu_s_per_1m_req']:.2f} | {r['total_err']} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Multi-Core Scaling Sweep (Workers = 1, 4, 8; C = 128)",
        "",
        "| Workload | Workers | RCRE RPS | Go-Tuned RPS | Go-Idiomatic RPS | RCRE vs Go-Tuned | RCRE RSS (MB) | Go-Tuned RSS (MB) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    scaling = results["scaling"]
    sc_map: dict[tuple[str, int], dict[str, dict]] = {}
    for r in scaling:
        sc_map.setdefault((r["workload"], r["workers"]), {})[r["server"]] = r
    for wid in ["W1_health", "W3_search", "W4_infer"]:
        for w in [1, 4, 8]:
            key = (wid, w)
            if key not in sc_map:
                continue
            g = sc_map[key]
            rc = g["rcre"]
            gt = g["go-tuned"]
            gi = g["go-idiomatic"]
            lines.append(
                f"| `{wid}` | {w} | **{rc['rps_median']:,.1f}** | {gt['rps_median']:,.1f} | {gi['rps_median']:,.1f} | "
                f"**{rc['rps_median'] / max(1.0, gt['rps_median']):.2f}×** | {rc['peak_rss_mb']:.1f} | {gt['peak_rss_mb']:.1f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. High-Concurrency Fan-Out Stress (`W5`: `GET /health`, 4 Workers)",
        "",
        "| Concurrency (C) | Server | RPS | p50 (ms) | p99 (ms) | p99.9 (ms) | Peak RSS (MB) | CPU-s / 1M req | Errors |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    fanout = results["concurrency_fanout"]
    for c in [64, 256, 1024, 2048]:
        for sname in ["rcre", "go-tuned", "go-idiomatic"]:
            matches = [r for r in fanout if r["concurrency"] == c and r["server"] == sname]
            if not matches:
                continue
            r = matches[0]
            lines.append(
                f"| {c} | **{sname}** | {r['rps_median']:,.1f} | {r['p50_ms']:.3f} | {r['p99_ms']:.3f} | "
                f"{r['p999_ms']:.3f} | {r['peak_rss_mb']:.1f} | {r['cpu_s_per_1m_req']:.2f} | {r['total_err']} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Open-Loop Fixed-RPS Tail Latency (`--latency-correction`, 4 Workers)",
        "",
        "| Workload | Target QPS | Server | Achieved RPS | p50 (ms) | p99 (ms) | p99.9 (ms) | max (ms) | Peak RSS (MB) |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in results["open_loop"]:
        lines.append(
            f"| `{r['workload']}` | {r['rate_qps']:,} | **{r['server']}** | {r['rps_median']:,.1f} | "
            f"{r['p50_ms']:.3f} | {r['p99_ms']:.3f} | {r['p999_ms']:.3f} | {r['max_ms']:.3f} | {r['peak_rss_mb']:.1f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Startup Time & Idle Footprint (4 Workers, 10k BM25 + 100-Tree GBDT Loaded)",
        "",
        "| Server | Binary Size (KB) | Cold Start to 200 OK (ms) | Post-Init Idle RSS (MB) |",
        "| :--- | ---: | ---: | ---: |",
    ])
    for sname in ["rcre", "go-tuned", "go-idiomatic"]:
        sample = next((r for r in core if r["server"] == sname and r["workload"] == "W1_health"), None)
        if sample:
            bsz = hw["rcre_bin_kb"] if sname == "rcre" else hw["go_bin_kb"]
            lines.append(
                f"| **{sname}** | {bsz:.1f} | {sample['startup_ms']:.2f} | {sample['idle_rss_mb']:.1f} |"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    workloads = {
        "W1_health": {"id": "W1_health", "method": "GET", "path": "/health"},
        "W2_validate": {
            "id": "W2_validate",
            "method": "POST",
            "path": "/api/v1/validate",
            "body_file": DATA / "payload_validate_4k.txt",
            "content_type": "text/plain; charset=utf-8",
        },
        "W3_search": {
            "id": "W3_search",
            "method": "GET",
            "path": "/api/v1/search?q=kqueue+arena+latency&k=10",
        },
        "W4_infer": {
            "id": "W4_infer",
            "method": "POST",
            "path": "/api/v1/infer",
            "body_file": DATA / "payload_infer_64x16.json",
            "content_type": "application/json",
        },
    }

    all_results: dict = {
        "hardware": hw_info(),
        "core_matrix": [],
        "scaling": [],
        "concurrency_fanout": [],
        "open_loop": [],
    }
    port = 19001

    print("=== Phase A: Core Workload Matrix (4 workers, C=128, 3x5s) ===")
    for wid in ["W1_health", "W2_validate", "W3_search", "W4_infer"]:
        wl = workloads[wid]
        for sname, sbin, sargs in SERVERS:
            r = benchmark_cell(sname, sbin, sargs, port, workers=4, workload=wl,
                               concurrency=128, warmup_sec=2, measure_sec=5, repeats=3)
            port += 1
            all_results["core_matrix"].append(r)
            print(f"  [{wid:11s}] {sname:12s} | RPS={r['rps_median']:10,.1f} | p99={r['p99_ms']:6.3f}ms | RSS={r['peak_rss_mb']:5.1f}MB | CPU/1M={r['cpu_s_per_1m_req']:6.2f}s | err={r['total_err']}", flush=True)

    print("\n=== Phase B: Multi-Core Scaling Sweep (workers=1, 4, 8; C=128, 2x4s) ===")
    for wid in ["W1_health", "W3_search", "W4_infer"]:
        wl = workloads[wid]
        for w in [1, 4, 8]:
            if w == 4:
                # Reuse already-measured 4-worker core matrix entries for consistency
                for r in all_results["core_matrix"]:
                    if r["workload"] == wid:
                        all_results["scaling"].append(r)
                continue
            for sname, sbin, sargs in SERVERS:
                r = benchmark_cell(sname, sbin, sargs, port, workers=w, workload=wl,
                                   concurrency=128, warmup_sec=1, measure_sec=4, repeats=2)
                port += 1
                all_results["scaling"].append(r)
                print(f"  [{wid:11s} w={w}] {sname:12s} | RPS={r['rps_median']:10,.1f} | p99={r['p99_ms']:6.3f}ms | RSS={r['peak_rss_mb']:5.1f}MB", flush=True)

    print("\n=== Phase C: High-Concurrency Fan-Out Stress (W1, 4 workers, C=64..2048) ===")
    for c in [64, 256, 1024, 2048]:
        for sname, sbin, sargs in SERVERS:
            r = benchmark_cell(sname, sbin, sargs, port, workers=4, workload=workloads["W1_health"],
                               concurrency=c, warmup_sec=1, measure_sec=4, repeats=2)
            port += 1
            all_results["concurrency_fanout"].append(r)
            print(f"  [W5 C={c:4d}] {sname:12s} | RPS={r['rps_median']:10,.1f} | p99={r['p99_ms']:6.3f}ms | p99.9={r['p999_ms']:6.3f}ms | RSS={r['peak_rss_mb']:5.1f}MB | err={r['total_err']}", flush=True)

    print("\n=== Phase D: Open-Loop Fixed-RPS Tail Latency (4 workers, --latency-correction) ===")
    # Dynamically calibrate target QPS to 75% of min(rcre, go-tuned) saturation RPS
    # so both tuned servers are tested under heavy, non-saturated open-loop load.
    core_lookup = {(r["workload"], r["server"]): r["rps_median"] for r in all_results["core_matrix"]}
    open_loop_cases = []
    for wid, conc in [("W1_health", 128), ("W2_validate", 128), ("W3_search", 64), ("W4_infer", 64)]:
        base_rps = min(core_lookup[(wid, "rcre")], core_lookup[(wid, "go-tuned")])
        target_qps = max(500, int((base_rps * 0.75) // 500) * 500)
        open_loop_cases.append((wid, target_qps, conc))
    for wid, target_qps, conc in open_loop_cases:
        wl = workloads[wid]
        for sname, sbin, sargs in SERVERS:
            r = benchmark_cell(sname, sbin, sargs, port, workers=4, workload=wl,
                               concurrency=conc, warmup_sec=1, measure_sec=5, repeats=1,
                               rate_qps=target_qps)
            port += 1
            all_results["open_loop"].append(r)
            print(f"  [OL {wid:11s} @{target_qps:6d}] {sname:12s} | RPS={r['rps_median']:9,.1f} | p50={r['p50_ms']:6.3f}ms | p99={r['p99_ms']:6.3f}ms | p99.9={r['p999_ms']:6.3f}ms | max={r['max_ms']:6.3f}ms", flush=True)

    raw_path = RESULTS_DIR / "raw_results.json"
    raw_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    md_path = RESULTS_DIR / "BENCHMARK_REPORT.md"
    generate_markdown_report(all_results, md_path)
    print(f"\nBenchmark complete!\n  JSON:   {raw_path}\n  Report: {md_path}")


if __name__ == "__main__":
    main()
