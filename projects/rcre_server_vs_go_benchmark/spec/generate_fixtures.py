#!/usr/bin/env python3
"""
Generate deterministic shared benchmark fixtures for RCRE vs Go web server suite.

Outputs (under spec/data/):
  - corpus_10k.txt           : 10,000 synthetic technical documents (1 per line)
  - gbdt_model.bin           : 100-tree, depth-6 flat GBDT binary model (16 features)
  - payload_validate_4k.txt  : Exact 4,096-byte valid UTF-8 payload (ASCII + multi-byte)
  - payload_infer_64x16.json : 64x16 float32 feature batch in JSON
  - search_queries.txt       : Multi-term BM25 search URLs
"""

import json
import math
import os
import random
import struct
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data"


def build_vocab(rng: random.Random) -> list[str]:
    core = [
        "kernel", "memory", "arena", "allocator", "cache", "vector", "simd",
        "neon", "avx", "tensor", "matrix", "sparse", "bm25", "search",
        "index", "graph", "hnsw", "gbdt", "tree", "inference", "latency",
        "throughput", "socket", "kqueue", "epoll", "reactor", "goroutine",
        "scheduler", "runtime", "garbage", "collector", "slice", "buffer",
        "stream", "parser", "http", "protocol", "keepalive", "pipeline",
        "worker", "thread", "atomic", "lockfree", "ring", "slab", "page",
        "alignment", "prefetch", "register", "pipeline", "branch", "compute",
        "bandwidth", "roofline", "profile", "telemetry", "benchmark", "server",
        "client", "request", "response", "header", "payload", "json", "utf8",
        "codec", "compress", "quantize", "neural", "attention", "transformer",
        "gradient", "descent", "entropy", "softmax", "sigmoid", "eigen",
        "fourier", "wavelet", "convolution", "filter", "sobel", "resize",
        "bilinear", "homography", "optical", "flow", "cluster", "kmeans",
        "manifold", "umap", "tsne", "pca", "svd", "cholesky", "krylov",
        "poisson", "multigrid", "solver", "preconditioner", "sparse", "csr",
    ]
    prefixes = [
        "micro", "macro", "hyper", "ultra", "meta", "auto", "sync", "async",
        "zero", "fast", "flat", "pack", "tile", "grid", "mesh", "node",
        "edge", "core", "base", "deep", "wide", "thin", "dual", "poly",
    ]
    suffixes = [
        "net", "sys", "ops", "lib", "api", "mod", "ctx", "buf", "idx", "map",
        "set", "vec", "mat", "row", "col", "blk", "seg", "tag", "key", "val",
        "bit", "num", "str", "tok", "doc", "hit", "err", "log", "io", "fs",
    ]
    vocab = list(dict.fromkeys(core))
    for p in prefixes:
        for s in suffixes:
            vocab.append(f"{p}{s}")
    for i in range(600):
        vocab.append(f"term{i:04d}")
    rng.shuffle(vocab)
    # Keep first 1500 unique terms, Zipf-ordered
    return vocab[:1500]


def generate_corpus(path: Path) -> None:
    rng = random.Random(20261126)
    vocab = build_vocab(rng)
    # Precompute Zipf CDF weights
    n_vocab = len(vocab)
    weights = [1.0 / ((i + 5.0) ** 1.05) for i in range(n_vocab)]
    total_w = sum(weights)
    cdf = []
    acc = 0.0
    for w in weights:
        acc += w / total_w
        cdf.append(acc)

    import bisect

    num_docs = 10_000
    lines = []
    for doc_id in range(num_docs):
        doc_len = rng.randint(24, 48)
        words = []
        for _ in range(doc_len):
            u = rng.random()
            idx = bisect.bisect_left(cdf, u)
            if idx >= n_vocab:
                idx = n_vocab - 1
            words.append(vocab[idx])
        # Sprinkle structured domain anchors deterministically so benchmark queries
        # hit realistic posting list lengths (~50-600 docs)
        if doc_id % 19 == 0:
            words[0] = "kqueue"
        if doc_id % 23 == 0:
            words[1] = "arena"
        if doc_id % 29 == 0:
            words[2] = "neon"
        if doc_id % 31 == 0:
            words[3] = "bm25"
        if doc_id % 37 == 0:
            words[4] = "gbdt"
        if doc_id % 41 == 0:
            words[5] = "latency"
        lines.append(" ".join(words))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_gbdt_model(path: Path) -> None:
    rng = random.Random(424242)
    num_trees = 100
    depth = 6  # full binary tree of depth 6 => 2^(6+1)-1 = 127 nodes per tree
    nodes_per_tree = (1 << (depth + 1)) - 1
    num_nodes = num_trees * nodes_per_tree
    num_features = 16
    base_score = 0.0
    is_binary = 1

    tree_roots = []
    nodes = []  # (feature_idx, threshold, left_child, right_child, leaf_value, is_leaf)

    for t in range(num_trees):
        root_idx = t * nodes_per_tree
        tree_roots.append(root_idx)
        for local_idx in range(nodes_per_tree):
            left_local = 2 * local_idx + 1
            right_local = 2 * local_idx + 2
            if left_local >= nodes_per_tree:
                # Leaf node
                leaf_val = round(rng.uniform(-0.18, 0.18), 5)
                nodes.append((0, 0.0, 0, 0, leaf_val, 1))
            else:
                feat = rng.randrange(num_features)
                thresh = round(rng.uniform(-2.0, 2.0), 5)
                nodes.append((
                    feat,
                    thresh,
                    root_idx + left_local,
                    root_idx + right_local,
                    0.0,
                    0,
                ))

    with path.open("wb") as f:
        f.write(struct.pack("<4sIIIfI", b"RGBT", num_trees, num_nodes, num_features, base_score, is_binary))
        for r in tree_roots:
            f.write(struct.pack("<I", r))
        for feat, thresh, lc, rc, lv, is_leaf in nodes:
            f.write(struct.pack("<IfIIfI", feat, thresh, lc, rc, lv, is_leaf))


def generate_validate_payload(path: Path) -> None:
    # Exact 4,096-byte valid UTF-8 document mixing ASCII + multi-byte scripts
    chunks = [
        '{"service":"rcre-bench","region":"us-central","msg":"High-throughput zero-malloc C23 arena reactor on Apple Silicon"}\n',
        '{"locale":"tr","note":"İstanbul çağdaş yazılım mimarisi ve yüksek performanslı bellek yönetimi — ölçüm #1"}\n',
        '{"locale":"ja","note":"超高速イベントループとゼロコピー文字列解析のベンチマーク評価"}\n',
        '{"locale":"math","formula":"∑(i=1..N) αᵢ·xᵢ ≤ ε (10⁻⁵) → O(1) arena rewind ✓"}\n',
        '{"metrics":{"p50_us":42.5,"p99_us":118.2,"rss_mb":14.8,"allocs":0,"simd":"ARM_NEON"}}\n',
    ]
    buf = bytearray()
    i = 0
    while len(buf) < 4096:
        nxt = chunks[i % len(chunks)].encode("utf-8")
        if len(buf) + len(nxt) <= 4096:
            buf.extend(nxt)
        else:
            rem = 4096 - len(buf)
            if rem == 1:
                buf.extend(b"\n")
            else:
                buf.extend(b"A" * (rem - 1) + b"\n")
            break
        i += 1
    assert len(buf) == 4096
    buf.decode("utf-8")  # assert valid UTF-8
    path.write_bytes(bytes(buf))


def generate_infer_payload(path: Path) -> None:
    rng = random.Random(987654)
    rows = [
        [round(rng.uniform(-2.5, 2.5), 4) for _ in range(16)]
        for _ in range(64)
    ]
    text = json.dumps({"rows": rows}, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")


def generate_search_queries(path: Path) -> None:
    queries = [
        "kqueue+arena+latency",
        "neon+bm25+gbdt",
        "kqueue+neon+cache",
        "arena+memory+allocator",
        "bm25+search+index",
        "gbdt+inference+latency",
        "socket+reactor+throughput",
        "vector+simd+tensor",
    ]
    path.write_text("\n".join(queries) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_corpus(OUT_DIR / "corpus_10k.txt")
    generate_gbdt_model(OUT_DIR / "gbdt_model.bin")
    generate_validate_payload(OUT_DIR / "payload_validate_4k.txt")
    generate_infer_payload(OUT_DIR / "payload_infer_64x16.json")
    generate_search_queries(OUT_DIR / "search_queries.txt")
    print(f"Generated fixtures in {OUT_DIR}:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  - {p.name:28s} {p.stat().st_size:10d} bytes")


if __name__ == "__main__":
    main()
