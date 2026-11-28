#!/usr/bin/env python3
"""
End-to-end verification harness:
  1. Launches server-rcre-asan (ASan + UBSan), server-go (idiomatic), server-go (tuned).
  2. Runs verify_parity.py against all 3 servers simultaneously.
  3. Sends SIGTERM and asserts zero ASan/UBSan errors and clean exit codes.
"""

import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
DATA = ROOT / "spec" / "data"


def wait_ready(url: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"Server did not become ready at {url}")


def main() -> None:
    procs = []
    try:
        configs = [
            ("rcre-asan", [str(BIN / "server-rcre-asan"), "--host", "127.0.0.1", "--port", "18181", "--workers", "4", "--data", str(DATA)], "http://127.0.0.1:18181"),
            ("go-idiomatic", [str(BIN / "server-go"), "--host", "127.0.0.1", "--port", "18182", "--workers", "4", "--data", str(DATA), "--mode", "idiomatic"], "http://127.0.0.1:18182"),
            ("go-tuned", [str(BIN / "server-go"), "--host", "127.0.0.1", "--port", "18183", "--workers", "4", "--data", str(DATA), "--mode", "tuned"], "http://127.0.0.1:18183"),
        ]
        for name, cmd, base_url in configs:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            procs.append((name, p))
            wait_ready(base_url + "/health")
            print(f"[READY] {name} at {base_url}")

        # Run full HTTP/1.1 conformance & byte-exact parity verifier
        verify_cmd = [
            sys.executable,
            str(ROOT / "bench" / "verify_parity.py"),
            "--targets",
            "rcre-asan=http://127.0.0.1:18181",
            "go-idiomatic=http://127.0.0.1:18182",
            "go-tuned=http://127.0.0.1:18183",
        ]
        res = subprocess.run(verify_cmd, check=False)
        if res.returncode != 0:
            raise SystemExit(res.returncode)

        # Also test single-worker (inline accept) mode on rcre-asan!
        p1 = subprocess.Popen(
            [str(BIN / "server-rcre-asan"), "--host", "127.0.0.1", "--port", "18184", "--workers", "1", "--data", str(DATA)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        procs.append(("rcre-asan-1w", p1))
        wait_ready("http://127.0.0.1:18184/health")
        res2 = subprocess.run(
            [sys.executable, str(ROOT / "bench" / "verify_parity.py"), "--targets", "rcre-asan-1w=http://127.0.0.1:18184"],
            check=False,
        )
        if res2.returncode != 0:
            raise SystemExit(res2.returncode)

    finally:
        sanitizer_fail = False
        for name, p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
            try:
                out, err = p.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                p.kill()
                out, err = p.communicate()
            print(f"--- [{name}] exit={p.returncode} ---")
            if err.strip():
                print(err.strip())
            if p.returncode != 0 or "ERROR: AddressSanitizer" in err or "runtime error:" in err:
                sanitizer_fail = True
        if sanitizer_fail:
            print("VERIFICATION FAILED: sanitizer error or non-zero exit!", file=sys.stderr)
            raise SystemExit(1)

    print("\nALL SANITIZER (ASan/UBSan) AND PARITY VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    main()
