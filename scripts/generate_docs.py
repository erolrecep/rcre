#!/usr/bin/env python3
"""
RCRE C23 Header API Extractor
Parses include/rcre.h and include/extras/rcre_cam.h to extract functions,
structs, enums, parameters, return codes, and attributes for documentation.
"""

import re
import json
import os
from pathlib import Path

def parse_header(header_path):
    with open(header_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract functions with docstrings
    func_pattern = re.compile(
        r'/\*\*\s*\n([\s\S]*?)\*/\s*\n\s*(?:RCRE_NODISCARD\s+)?([A-Za-z0-9_*]+(?:\s+[A-Za-z0-9_*]+)*)\s+([A-Za-z0-9_]+)\s*\(([\s\S]*?)\);',
        re.MULTILINE
    )

    functions = []
    for match in func_pattern.finditer(content):
        docstring, ret_type, name, params_raw = match.groups()
        
        # Parse docstring tags
        brief = ""
        params_doc = {}
        returns_doc = ""
        
        for line in docstring.split('\n'):
            line = line.strip().lstrip('*').strip()
            if line.startswith('@brief'):
                brief = line.replace('@brief', '').strip()
            elif line.startswith('@param'):
                p_parts = line.replace('@param', '').strip().split(maxsplit=1)
                if len(p_parts) == 2:
                    params_doc[p_parts[0]] = p_parts[1]
                elif len(p_parts) == 1:
                    params_doc[p_parts[0]] = ""
            elif line.startswith('@return'):
                returns_doc = line.replace('@return', '').strip()

        # Parse parameters
        params = []
        clean_params = params_raw.strip().replace('\n', ' ')
        clean_params = re.sub(r'\s+', ' ', clean_params)
        
        if clean_params and clean_params != 'void':
            for p in clean_params.split(','):
                p = p.strip()
                if p:
                    p_name = p.split()[-1].lstrip('*')
                    p_type = p[:p.rfind(p_name)].strip()
                    desc = params_doc.get(p_name, "")
                    params.append({
                        "name": p_name,
                        "type": p_type,
                        "description": desc
                    })

        # Categorize
        category = "Core"
        if name.startswith("rcre_arena"):
            category = "Memory & Arenas"
        elif name.startswith("rcre_tensor"):
            category = "Strided Tensors"
        elif name.startswith("rcre_threadpool"):
            category = "Concurrency"
        elif name.startswith("rcre_cv"):
            category = "Computer Vision"
        elif name.startswith("rcre_ml"):
            category = "Machine Learning"
        elif name.startswith("rcre_hnsw"):
            category = "Vector Search (HNSW)"
        elif name.startswith("rcre_camera"):
            category = "Camera Hardware"
        elif name.startswith("rcre_hal") or name.startswith("rcre_cpu"):
            category = "Hardware Acceleration"

        functions.append({
            "name": name,
            "return_type": ret_type.strip(),
            "brief": brief,
            "params": params,
            "returns": returns_doc,
            "category": category,
            "header": Path(header_path).name
        })

    return functions

def main():
    root = Path(__file__).resolve().parent.parent
    rcre_h = root / "include" / "rcre.h"
    rcre_cam_h = root / "include" / "extras" / "rcre_cam.h"

    all_funcs = []
    if rcre_h.exists():
        all_funcs.extend(parse_header(rcre_h))
    if rcre_cam_h.exists():
        all_funcs.extend(parse_header(rcre_cam_h))

    out_file = root / "web" / "docs" / "api_data.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_funcs, f, indent=2)

    print(f"Extracted {len(all_funcs)} C23 functions into {out_file}")

if __name__ == "__main__":
    main()
