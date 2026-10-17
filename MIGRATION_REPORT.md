# RCRE Project Handover & Multi-Agent Migration Report
## Project: Recep's Code Run Everywhere (RCRE)

---

## 1. Executive Summary & Purpose

This migration package contains the complete technical architecture, formal system requirements, 24-week phased development roadmap, multi-agent operational strategy, and academic book curriculum for **RCRE (Recep's Code Run Everywhere)**.

Use this report to bootstrap the development environment on your target computer equipped with the multi-agent setup.

---

## 2. Package Manifest & File Inventory

The archive contains the following authoritative design documents:

```
rcre/
├── MIGRATION_REPORT.md             <-- This document (Read first on new machine)
├── rcre.md                         <-- Master Blueprint (High-level architecture, UML, philosophy)
└── docs/
    ├── requirements_spec.md        <-- Formal System Requirements Specification (SRS with IDs)
    ├── development_roadmap.md      <-- Phased milestones (MVP 1-5) & 4-layer test matrix
    ├── multi_agent_strategy.md     <-- Agent roles, Git branching, and CI/CD gatekeeping
    └── book_proposal.md            <-- 16-Chapter academic textbook syllabus & pedagogy
```

---

## 3. New Machine Environment Setup & Prerequisites

Before launching the multi-agent development loop, verify that the target machine has the necessary toolchains installed:

### 3.1 Core C23 Toolchain
* **Compiler**: Clang 18+ or GCC 14+ with full ISO C23 support (`-std=c2x` or `-std=c23`).
  ```bash
  clang --version   # Verify version >= 18.0
  ```
* **Build System**: CMake 3.25+ and Ninja / GNU Make.
* **Sanitizers**: AddressSanitizer (ASan), UndefinedBehaviorSanitizer (UBSan), and Valgrind (on Linux).

### 3.2 Verification & Test Environment (Python 3)
* Install Python 3.10+ with reference libraries for ground-truth parity testing:
  ```bash
  pip install numpy opencv-python scikit-learn pytest
  ```

### 3.3 Optional WebAssembly Toolchain (For Phase 4-5)
* Emscripten SDK (`emcc`) for compiling the C23 core to WebAssembly with SIMD (`-msimd128`).
  ```bash
  emcc --version
  ```

---

## 4. Multi-Agent Launch & Orchestration Guide

On your target computer, configure your multi-agent orchestrator with the following operational roles and initial prompts:

### Agent Roles & Task Allocations

| Agent ID | Assigned Role | Initial Primary Task | Target Files |
| :--- | :--- | :--- | :--- |
| **Agent 0** | **Lead System Architect** | Scaffold CMake/Makefile, define `include/rcre.h` baseline types, enforce C23 conventions. | `CMakeLists.txt`, `Makefile`, `include/rcre.h` |
| **Agent 1** | **Core Kernel Engineer** | Implement the 64-byte aligned linear arena allocator and the $\mathcal{O}(1)$ strided tensor ShapeTracker. | `src/core/arena.c`, `src/core/tensor.c` |
| **Agent 2** | **SIMD Specialist** | Implement CPU feature detection (`cpuid`/`sysctl`) and construct the dynamic HAL dispatch table. | `src/hal/cpu_probe.c`, `src/hal/hal_dispatch.c` |
| **Agent 3** | **Input Adaptor Specialist**| Implement `rcre_tensor_adapt()` for letterboxing, $HWC \leftrightarrow CHW$ layout swapping, and normalization. | `src/core/adapt.c` |
| **Agent 4** | **QA & Test Gatekeeper** | Author unit tests for strided tensors and build the automated Python parity test harness. | `tests/test_tensor_strides.c`, `tests/test_against_scipy_opencv.py` |
| **Agent 5** | **Frontend & WASM Engineer**| Setup the `web/` static site scaffolding and configure the Emscripten WASM build target. | `web/index.html`, `web/playground.html` |
| **Agent 6** | **Book & Academic Author** | Draft Chapter 1 (*The Modern C23 Systems Mindset*) and Chapter 2 (*Memory Hierarchies and Arenas*). | `docs/book/ch01_c23_mindset.md`, `docs/book/ch02_arenas.md` |

---

## 5. Immediate Kickoff Sequence (Phase 0 & 1)

Follow these exact steps upon unzipping the archive on the new machine:

1. **Unzip & Enter Workspace**:
   ```bash
   unzip rcre_project.zip -d ~/rcre
   cd ~/rcre
   ```

2. **Initialize Local Git Repository**:
   ```bash
   git init
   git add .
   git commit -m "feat: bootstrap RCRE architecture and requirements from handover"
   ```

3. **Link to Private GitHub Repository**:
   ```bash
   # Ensure you are authenticated via 'gh auth login'
   gh repo create rcre --private --source=. --remote=origin --push
   ```

4. **Assign Phase 0 Tasks to Agents**:
   * Direct **Agent 0** to generate the root `CMakeLists.txt` and `include/rcre.h`.
   * Direct **Agent 1** to implement `src/core/arena.c` adhering to `MEM-REQ-01` through `MEM-REQ-04` in `requirements_spec.md`.
   * Direct **Agent 4** to execute memory sanitizers (`-fsanitize=address,undefined`) against the new code.
