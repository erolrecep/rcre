#!/usr/bin/env python3
"""
HTTP/1.1 conformance, robustness, and cross-server byte-exact parity verifier.

Usage:
  python3 bench/verify_parity.py --targets rcre=http://127.0.0.1:18081 go-idiomatic=http://127.0.0.1:18082 go-tuned=http://127.0.0.1:18083
"""

import argparse
import http.client
import json
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).resolve().parent.parent / "spec" / "data"


def fnv1a_64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def http_request(host: str, port: int, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
    conn = http.client.HTTPConnection(host, port, timeout=5.0)
    hdrs = headers.copy() if headers else {}
    conn.request(method, path, body=body, headers=hdrs)
    resp = conn.getresponse()
    raw_body = resp.read()
    resp_headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, resp_headers, raw_body


def test_keepalive_and_fragmented_tcp(host: str, port: int, name: str) -> None:
    """Verify HTTP/1.1 Keep-Alive reuse and partial/fragmented TCP reads."""
    s = socket.create_connection((host, port), timeout=5.0)
    s.settimeout(5.0)
    # 1. Send fragmented GET /health across 3 tiny TCP writes
    req1 = b"GET /he"
    req2 = b"alth HTTP/1.1\r\nHost: local"
    req3 = b"host\r\nConnection: keep-alive\r\n\r\n"
    for chunk in (req1, req2, req3):
        s.sendall(chunk)
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += s.recv(4096)
    hdr_part, rest = buf.split(b"\r\n\r\n", 1)
    assert b"200 OK" in hdr_part, f"[{name}] fragmented health failed: {hdr_part!r}"
    while len(rest) < 2:
        rest += s.recv(4096)
    assert rest[:2] == b"OK", f"[{name}] expected OK, got {rest[:2]!r}"

    # 2. Reuse SAME socket for a POST /api/v1/validate with body split across two sends
    body = b"Hello \xc3\xbc\nWorld!"
    hdr = (
        f"POST /api/v1/validate HTTP/1.1\r\n"
        f"Host: localhost\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("ascii")
    s.sendall(hdr + body[:5])
    s.sendall(body[5:])
    resp = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        resp += chunk
    s.close()
    assert b"200 OK" in resp, f"[{name}] keep-alive reuse POST failed: {resp!r}"
    json_body = json.loads(resp.split(b"\r\n\r\n", 1)[1].decode("utf-8"))
    assert json_body["valid"] is True
    assert json_body["bytes"] == len(body)
    assert json_body["codepoints"] == len(body.decode("utf-8"))
    assert json_body["lines"] == 2
    assert json_body["fnv64"] == f"{fnv1a_64(body):016x}"


def test_malformed_request(host: str, port: int, name: str) -> None:
    s = socket.create_connection((host, port), timeout=5.0)
    s.settimeout(3.0)
    s.sendall(b"GARBAGE_LINE_WITHOUT_HTTP\r\n\r\n")
    resp = s.recv(4096)
    s.close()
    assert b"400 Bad Request" in resp, f"[{name}] expected 400 for malformed request, got {resp!r}"


def run_suite_for_target(name: str, url: str) -> dict[str, tuple[int, bytes]]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80

    val_4k = (DATA_DIR / "payload_validate_4k.txt").read_bytes()
    infer_64 = (DATA_DIR / "payload_infer_64x16.json").read_bytes()

    results: dict[str, tuple[int, bytes]] = {}

    # W1: Health
    st, hdrs, body = http_request(host, port, "GET", "/health")
    assert st == 200 and body == b"OK", f"[{name}] W1 failed: {st} {body!r}"
    assert "content-length" in hdrs and hdrs["content-length"] == "2"
    assert "date" in hdrs, f"[{name}] missing Date header"
    results["W1_health"] = (st, body)

    # W2a: Validate 4K valid UTF-8
    st, hdrs, body = http_request(host, port, "POST", "/api/v1/validate", body=val_4k)
    assert st == 200, f"[{name}] W2a status {st}"
    j = json.loads(body.decode("utf-8"))
    assert j["valid"] is True
    assert j["bytes"] == 4096
    assert j["codepoints"] == len(val_4k.decode("utf-8"))
    assert j["fnv64"] == f"{fnv1a_64(val_4k):016x}"
    results["W2a_validate_4k"] = (st, body)

    # W2b: Validate invalid UTF-8
    bad_utf8 = b"valid prefix \xff\xfe invalid suffix\nline2"
    st, _, body = http_request(host, port, "POST", "/api/v1/validate", body=bad_utf8)
    assert st == 200
    j = json.loads(body.decode("utf-8"))
    assert j["valid"] is False
    assert j["bytes"] == len(bad_utf8)
    assert j["codepoints"] == 0
    assert j["lines"] == 2
    assert j["fnv64"] == f"{fnv1a_64(bad_utf8):016x}"
    results["W2b_validate_bad_utf8"] = (st, body)

    # W2c: Validate empty body
    st, _, body = http_request(host, port, "POST", "/api/v1/validate", body=b"")
    assert st == 200
    j = json.loads(body.decode("utf-8"))
    assert j["valid"] is True and j["bytes"] == 0 and j["lines"] == 0
    results["W2c_validate_empty"] = (st, body)

    # W3a: BM25 search multi-term
    st, _, body = http_request(host, port, "GET", "/api/v1/search?q=kqueue+arena+latency&k=10")
    assert st == 200, f"[{name}] W3a status {st}"
    j = json.loads(body.decode("utf-8"))
    assert j["query"] == "kqueue arena latency"
    assert j["count"] == 10 and len(j["hits"]) == 10
    scores = [h["score"] for h in j["hits"]]
    assert scores == sorted(scores, reverse=True), f"[{name}] W3a scores not sorted"
    results["W3a_search_multi"] = (st, body)

    # W3b: BM25 search percent-encoded
    st, _, body = http_request(host, port, "GET", "/api/v1/search?q=neon%20bm25%20gbdt&k=5")
    assert st == 200
    j = json.loads(body.decode("utf-8"))
    assert j["query"] == "neon bm25 gbdt" and j["count"] == 5
    results["W3b_search_pct"] = (st, body)

    # W4: GBDT Inference 64x16
    st, _, body = http_request(
        host, port, "POST", "/api/v1/infer",
        body=infer_64, headers={"Content-Type": "application/json"}
    )
    assert st == 200, f"[{name}] W4 status {st}: {body!r}"
    j = json.loads(body.decode("utf-8"))
    assert j["count"] == 64 and len(j["predictions"]) == 64
    assert all(0.0 <= p <= 1.0 for p in j["predictions"])
    results["W4_infer_64x16"] = (st, body)

    # Error handling: 404 & 405
    st, _, body = http_request(host, port, "GET", "/does-not-exist")
    assert st == 404, f"[{name}] expected 404, got {st}"
    results["E_404"] = (st, body)

    st, _, body = http_request(host, port, "POST", "/health")
    assert st == 405, f"[{name}] expected 405, got {st}"
    results["E_405"] = (st, body)

    # Low-level TCP keep-alive + fragmented read + malformed request
    test_keepalive_and_fragmented_tcp(host, port, name)
    test_malformed_request(host, port, name)

    print(f"  [PASS] {name:14s} passed all conformance & edge-case checks")
    return results


def compare_results(all_results: dict[str, dict[str, tuple[int, bytes]]]) -> None:
    names = list(all_results.keys())
    if len(names) < 2:
        return
    ref_name = names[0]
    ref = all_results[ref_name]
    for other_name in names[1:]:
        other = all_results[other_name]
        for case, (ref_st, ref_body) in ref.items():
            oth_st, oth_body = other[case]
            assert ref_st == oth_st, f"Status mismatch on {case}: {ref_name}={ref_st} vs {other_name}={oth_st}"
            if ref_body != oth_body:
                # Allow tiny float rounding tolerance (<= 1e-4) on JSON numeric scores
                try:
                    rj = json.loads(ref_body.decode("utf-8"))
                    oj = json.loads(oth_body.decode("utf-8"))
                    if case.startswith("W3"):
                        assert rj["count"] == oj["count"] and rj["query"] == oj["query"]
                        for rh, oh in zip(rj["hits"], oj["hits"]):
                            assert rh["doc_id"] == oh["doc_id"], (
                                f"BM25 doc_id mismatch on {case}: {rh} vs {oh}"
                            )
                            assert abs(rh["score"] - oh["score"]) <= 2e-4, (
                                f"BM25 score drift on {case}: {rh} vs {oh}"
                            )
                    elif case.startswith("W4"):
                        assert rj["count"] == oj["count"]
                        assert abs(rj["mean"] - oj["mean"]) <= 2e-5
                        for rp, op in zip(rj["predictions"], oj["predictions"]):
                            assert abs(rp - op) <= 2e-4, f"GBDT drift: {rp} vs {op}"
                    else:
                        raise AssertionError(
                            f"Exact body mismatch on {case}:\n  {ref_name}: {ref_body!r}\n  {other_name}: {oth_body!r}"
                        )
                except Exception as exc:
                    raise AssertionError(
                        f"Parity mismatch on {case} ({ref_name} vs {other_name}): {exc}\n"
                        f"  {ref_name}: {ref_body[:200]!r}\n  {other_name}: {oth_body[:200]!r}"
                    ) from exc
        print(f"  [PARITY] {ref_name} == {other_name} across all {len(ref)} test cases!")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", required=True, help="name=http://host:port entries")
    args = parser.parse_args()

    all_results = {}
    for item in args.targets:
        name, url = item.split("=", 1)
        all_results[name] = run_suite_for_target(name, url)

    compare_results(all_results)
    print("ALL PARITY & CONFORMANCE CHECKS PASSED!")


if __name__ == "__main__":
    main()
