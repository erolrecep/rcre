#!/usr/bin/env python3
"""
RCRE Single-Header Amalgamation Script (Phase 6 Packaging)

Bundles the public C23 header, autograd extensions, and all core kernel implementations
into a single standalone header file: dist/rcre.h.

Usage:
    python3 scripts/amalgamate.py
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE_DIR = ROOT / "include"
SRC_DIR = ROOT / "src"
DIST_DIR = ROOT / "dist"

C_FILES = [
    SRC_DIR / "hal" / "kernels_scalar.c",
    SRC_DIR / "hal" / "kernels_neon.c",
    SRC_DIR / "hal" / "kernels_avx2.c",
    SRC_DIR / "hal" / "kernels_avx512.c",
    SRC_DIR / "hal" / "kernels_wasm.c",
    SRC_DIR / "hal" / "cpu_probe.c",
    SRC_DIR / "hal" / "hal_dispatch.c",
    SRC_DIR / "core" / "arena.c",
    SRC_DIR / "core" / "slice.c",
    SRC_DIR / "core" / "tensor.c",
    SRC_DIR / "core" / "linalg.c",
    SRC_DIR / "core" / "reduce.c",
    SRC_DIR / "core" / "sf.c",
    SRC_DIR / "core" / "rng.c",
    SRC_DIR / "core" / "adapt.c",
    SRC_DIR / "core" / "threadpool.c",
    SRC_DIR / "core" / "prof.c",
    SRC_DIR / "core" / "sparse.c",
    SRC_DIR / "core" / "io.c",
    SRC_DIR / "core" / "df.c",
    SRC_DIR / "cv" / "color.c",
    SRC_DIR / "cv" / "geom.c",
    SRC_DIR / "cv" / "filter.c",
    SRC_DIR / "cv" / "fft.c",
    SRC_DIR / "cv" / "features.c",
    SRC_DIR / "cv" / "homography.c",
    SRC_DIR / "cv" / "optical_flow.c",
    SRC_DIR / "cv" / "nn.c",
    SRC_DIR / "cv" / "codec.c",
    SRC_DIR / "cv" / "onnx_subset.c",
    SRC_DIR / "ml" / "linear.c",
    SRC_DIR / "ml" / "cluster.c",
    SRC_DIR / "ml" / "e_means.c",
    SRC_DIR / "ml" / "gbdt_infer.c",
    SRC_DIR / "ml" / "forest.c",
    SRC_DIR / "ml" / "hnsw.c",
    SRC_DIR / "ml" / "stat.c",
    SRC_DIR / "ml" / "prep.c",
    SRC_DIR / "ml" / "metric.c",
    SRC_DIR / "autograd" / "scalar_autograd.c",
    SRC_DIR / "autograd" / "tensor_autograd.c",
    SRC_DIR / "nn" / "loss.c",
    SRC_DIR / "nn" / "optim.c",
    SRC_DIR / "nn" / "data.c",
    SRC_DIR / "nn" / "vis.c",
    SRC_DIR / "nn" / "vis_ds.c",
    SRC_DIR / "transformer" / "nn.c",
    SRC_DIR / "transformer" / "model.c",
    SRC_DIR / "transformer" / "kv_cache.c",
    SRC_DIR / "transformer" / "tokenizer.c",
    SRC_DIR / "transformer" / "dataset.c",
    SRC_DIR / "transformer" / "optim.c",
    SRC_DIR / "transformer" / "generate.c",
    SRC_DIR / "transformer" / "gguf.c",
    SRC_DIR / "transformer" / "llm.c",
]

def clean_file_content(content: str) -> str:
    """Removes local includes that are redundant in amalgamated file."""
    lines = []
    for line in content.splitlines():
        # P4: also strip PATH-QUALIFIED local includes (e.g. "../core/rcre_rng.h")
        # that previously leaked into dist/rcre.h and broke out-of-tree builds.
        if re.match(r'^\s*#include\s*["<]((?:\.\./)?(?:core|hal|ml|transformer)/)?(rcre\.h|rcre_prof\.h|rcre_autograd\.h|rcre_transformer\.h|rcre_sparse\.h|rcre_io\.h|rcre_llm\.h|rcre_codec\.h|rcre_df\.h|rcre_stat\.h|rcre_prep\.h|rcre_metric\.h|rcre_vis_ds\.h|transformer_internal\.h|rcre_rng\.h|hal_internal\.h|ml_internal\.h|e_means_internal\.h)[">]', line):
            continue
        lines.append(line)
    return "\n".join(lines)

def verify_symbol_uniqueness(c_files):
    """Ensures no un-namespaced colliding static function symbols exist."""
    seen_funcs = {}
    for c_file in c_files:
        text = c_file.read_text(encoding="utf-8")
        for m in re.finditer(r"^(?:static\s+(?:inline\s+)?[\w\s\*]+?)\s+([a-zA-Z0-9_]+)\s*\(", text, re.MULTILINE):
            name = m.group(1)
            seen_funcs.setdefault(name, []).append(c_file.name)
    duplicates = {k: v for k, v in seen_funcs.items() if len(v) > 1}
    if duplicates:
        raise ValueError(f"Duplicate static symbols detected during amalgamation: {duplicates}")

def main():
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DIST_DIR / "rcre.h"

    print(f"📦 Amalgamating RCRE into {out_file}...")

    verify_symbol_uniqueness(C_FILES)

    header_content = (INCLUDE_DIR / "rcre.h").read_text(encoding="utf-8")
    autograd_header = (INCLUDE_DIR / "rcre_autograd.h").read_text(encoding="utf-8")
    autograd_clean = clean_file_content(autograd_header)
    header_content = header_content.replace('#include "rcre_autograd.h"', autograd_clean)

    transformer_header = (INCLUDE_DIR / "rcre_transformer.h").read_text(encoding="utf-8")
    transformer_clean = clean_file_content(transformer_header)
    header_content = header_content.replace('#include "rcre_transformer.h"', transformer_clean)

    io_header = (INCLUDE_DIR / "rcre_io.h").read_text(encoding="utf-8")
    io_clean = clean_file_content(io_header)
    header_content = header_content.replace('#include "rcre_io.h"', io_clean)

    llm_header = (INCLUDE_DIR / "rcre_llm.h").read_text(encoding="utf-8")
    header_content = header_content.replace('#include "rcre_llm.h"', clean_file_content(llm_header))

    codec_header = (INCLUDE_DIR / "rcre_codec.h").read_text(encoding="utf-8")
    header_content = header_content.replace('#include "rcre_codec.h"', clean_file_content(codec_header))

    df_header = (INCLUDE_DIR / "rcre_df.h").read_text(encoding="utf-8")
    header_content = header_content.replace('#include "rcre_df.h"', clean_file_content(df_header))

    stat_header = (INCLUDE_DIR / "rcre_stat.h").read_text(encoding="utf-8")
    header_content = header_content.replace('#include "rcre_stat.h"', clean_file_content(stat_header))

    prep_header = (INCLUDE_DIR / "rcre_prep.h").read_text(encoding="utf-8")
    header_content = header_content.replace('#include "rcre_prep.h"', clean_file_content(prep_header))

    metric_header = (INCLUDE_DIR / "rcre_metric.h").read_text(encoding="utf-8")
    header_content = header_content.replace('#include "rcre_metric.h"', clean_file_content(metric_header))

    vis_ds_header = (INCLUDE_DIR / "rcre_vis_ds.h").read_text(encoding="utf-8")
    header_content = header_content.replace('#include "rcre_vis_ds.h"', clean_file_content(vis_ds_header))

    prof_header = (INCLUDE_DIR / "rcre_prof.h").read_text(encoding="utf-8")
    header_content = header_content + "\n" + "/* --- Profiling & Telemetry Subsystem (rcre_prof) --- */\n" + clean_file_content(prof_header) + "\n"

    sparse_header = (INCLUDE_DIR / "rcre_sparse.h").read_text(encoding="utf-8")
    header_content = header_content + "\n" + "/* --- Sparse Matrix & Pruning Engine (rcre_sparse) --- */\n" + clean_file_content(sparse_header) + "\n"

    hal_internal = (SRC_DIR / "hal" / "hal_internal.h").read_text(encoding="utf-8")
    ml_internal = (SRC_DIR / "ml" / "ml_internal.h").read_text(encoding="utf-8")
    em_internal = (SRC_DIR / "ml" / "e_means_internal.h").read_text(encoding="utf-8")
    rng_internal = (SRC_DIR / "core" / "rcre_rng.h").read_text(encoding="utf-8")
    pcg_table = (SRC_DIR / "core" / "rng_pcg_table.h").read_text(encoding="utf-8")
    sobol_table = (SRC_DIR / "core" / "rng_sobol_table.h").read_text(encoding="utf-8")
    transformer_internal = (SRC_DIR / "transformer" / "transformer_internal.h").read_text(encoding="utf-8")

    out_lines = []
    out_lines.append("/* ========================================================================= */")
    out_lines.append("/* RCRE — Recep's Code Run Everywhere (Amalgamated Single-Header C23)        */")
    out_lines.append("/* License: GNU AGPLv3 / Commercial | Auto-generated by amalgamate.py        */")
    out_lines.append("/* ========================================================================= */")
    out_lines.append("")
    out_lines.append(header_content)
    out_lines.append("")
    out_lines.append("#ifdef RCRE_IMPLEMENTATION")
    out_lines.append("")
    out_lines.append("/* --- Internal Hardware Abstraction Layer Declarations --- */")
    out_lines.append(clean_file_content(hal_internal))
    out_lines.append("")
    out_lines.append("/* --- Internal Machine Learning Mathematical Primitives --- */")
    out_lines.append(clean_file_content(ml_internal))
    out_lines.append("")
    out_lines.append("/* --- e-means State Machine (test-visible internals) --- */")
    out_lines.append(clean_file_content(em_internal))
    out_lines.append("")
    out_lines.append("/* --- Internal Random Number Generation --- */")
    out_lines.append(clean_file_content(rng_internal))
    out_lines.append("")
    out_lines.append("/* --- PCG64 Skip-Ahead Jump Table (H1) --- */")
    out_lines.append(clean_file_content(pcg_table))
    out_lines.append("")
    out_lines.append("/* --- Sobol' Direction Numbers (H1) --- */")
    out_lines.append(clean_file_content(sobol_table))
    out_lines.append("")
    out_lines.append("/* --- Internal Transformer Definitions & Helpers --- */")
    out_lines.append(clean_file_content(transformer_internal))
    out_lines.append("")

    for c_file in C_FILES:
        print(f"  -> Bundling {c_file.relative_to(ROOT)}...")
        out_lines.append(f"/* --- Source: {c_file.name} --- */")
        code = clean_file_content(c_file.read_text(encoding="utf-8"))
        out_lines.append(code)
        out_lines.append("")

    out_lines.append("#endif /* RCRE_IMPLEMENTATION */")
    out_lines.append("")

    full_amalgamated = "\n".join(out_lines)
    out_file.write_text(full_amalgamated, encoding="utf-8")

    file_size_kb = len(full_amalgamated.encode('utf-8')) / 1024
    line_count = len(full_amalgamated.splitlines())
    print(f"✅ Generated single-header release: {out_file} ({file_size_kb:.1f} KB, {line_count} lines)")

if __name__ == "__main__":
    main()
