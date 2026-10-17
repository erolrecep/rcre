#!/usr/bin/env python3
"""
RCRE Multi-Agent Orchestrator & Launcher
Allows launching, inspecting, and coordinating specialized AI development agents
for the RCRE (Recep's Code Run Everywhere) project.
"""

import sys
import argparse

AGENTS = {
    0: {
        "name": "Lead System Architect",
        "skill": "rcre-lead-architect",
        "scope": ["include/rcre.h", "CMakeLists.txt", "Makefile", "include/extras/rcre_telemetry.h"],
        "description": "Guards public C23 APIs, ABI stability, build matrix, and architectural boundaries."
    },
    1: {
        "name": "Core Kernel Engineer",
        "skill": "rcre-kernel-engineer",
        "scope": ["src/core/arena.c", "src/core/tensor.c", "src/core/adapt.c"],
        "description": "Implements 64-byte aligned linear arenas, O(1) ShapeTracker tensor math, and input adaptation."
    },
    2: {
        "name": "Hardware Acceleration & SIMD Specialist",
        "skill": "rcre-simd-specialist",
        "scope": ["src/hal/cpu_probe.c", "src/hal/hal_dispatch.c", "src/hal/kernels_*.c"],
        "description": "Writes ARM NEON, AVX2, AVX-512, and WASM SIMD micro-kernels with cache tiling."
    },
    3: {
        "name": "Vision & ML Engineer",
        "skill": "rcre-vision-ml-engineer",
        "scope": ["src/cv/*", "src/ml/*", "include/extras/rcre_cam.h"],
        "description": "Implements classical CV filters, GBDT inference, HNSW search, and hardware video capture."
    },
    4: {
        "name": "QA, Verification & Sanitizer Gatekeeper",
        "skill": "rcre-qa-verifier",
        "scope": ["tests/*", "benchmarks/*", "tests/test_against_scipy_opencv.py"],
        "description": "Runs ASan/UBSan/Valgrind memory checks and automates numerical parity tests vs OpenCV & Scikit-Learn."
    },
    5: {
        "name": "Frontend, WASM & Web Visualizer",
        "skill": "rcre-wasm-visualizer",
        "scope": ["web/*", "web/playground.html", "web/learn/*"],
        "description": "Compiles C23 core to rcre.wasm and builds interactive 3D/2D educational visualizers."
    },
    6: {
        "name": "Technical Writer & Academic Book Author",
        "skill": "rcre-academic-author",
        "scope": ["docs/book/*", "docs/book/book.tex"],
        "description": "Authors publication-grade textbook chapters, mathematical derivations, and architecture tutorials."
    }
}

def list_agents():
    print("=" * 80)
    print("🚀 RCRE (Recep's Code Run Everywhere) - Autonomous Multi-Agent Roster")
    print("=" * 80)
    for agent_id, data in sorted(AGENTS.items()):
        print(f"Agent {agent_id}: [{data['name']}] (Skill: {data['skill']})")
        print(f"  Target Scope : {', '.join(data['scope'])}")
        print(f"  Role         : {data['description']}")
        print("-" * 80)

def print_agent_instructions(agent_id):
    if agent_id not in AGENTS:
        print(f"Error: Unknown Agent ID {agent_id}. Valid IDs are 0 to 6.")
        sys.exit(1)
    
    agent = AGENTS[agent_id]
    print(f"\n--- System Prompt for Agent {agent_id}: {agent['name']} ---")
    print(f"Skill: {agent['skill']}")
    print(f"Target Files: {', '.join(agent['scope'])}")
    print(f"Directives:\n- Adhere strictly to apps/rcre/AGENTS.md project rules.")
    print(f"- Zero heap allocation on hot paths (use RCRE_Arena with 64-byte alignment).")
    print(f"- Enforce ISO C23 standards (-std=c2x/-std=c23).")
    print(f"- Verify all code passes tests and sanitizers before completion.\n")

def main():
    parser = argparse.ArgumentParser(description="RCRE Multi-Agent Orchestrator CLI")
    parser.add_argument("--list", action="store_true", help="List all 7 specialized development agents")
    parser.add_argument("--agent", type=int, choices=range(0, 7), help="Inspect directives for a specific agent (0-6)")
    args = parser.parse_args()

    if args.agent is not None:
        print_agent_instructions(args.agent)
    else:
        list_agents()

if __name__ == "__main__":
    main()
