# RCRE Multi-Agent Engineering & Project Guidelines

This document governs all AI agents and engineers operating within `apps/rcre/`.

---

## 1. Project Mission & Identity
**RCRE ("Recep's Code Run Everywhere")** is a high-performance, single-header, zero-dependency C23 engine unifying modern Computer Vision and Classical Machine Learning under a single strided tensor paradigm with zero-copy views, linear arena allocation, and dynamic SIMD dispatch across Edge, HPC, and WebAssembly.

---

## 2. Universal Technical Laws

### Law 1: Strict ISO C23 Compliance
* Target `-std=c2x` or `-std=c23`.
* Utilize modern standard attributes and features: `[[nodiscard]]`, `constexpr`, `nullptr`, `<stdatomic.h>`, `<stdbit.h>`, `<threads.h>`.
* Plain C ABI only — zero C++ name mangling, enabling zero-copy FFI for Python, Rust, Go, and Zig.

### Law 2: Zero Dynamic Memory in Hot Paths
* **Never** call `malloc()`, `calloc()`, `realloc()`, or `free()` inside inner algorithmic loops or inference routines (`MEM-REQ-02`).
* All transient compute allocations must be serviced by a caller-supplied or thread-local `RCRE_Arena*`.
* All arena buffers must maintain strict **64-byte alignment** (`ptr % 64 == 0`) for AVX-512 and cache line alignment (`MEM-REQ-01`).

### Law 3: Architectural Separation of Concerns
* **Core Compute (`include/rcre.h`)**: Pure mathematical and algorithmic compute. Zero OS-specific library dependencies.
* **Camera Hardware I/O (`include/extras/rcre_cam.h`)**: Isolated companion header for V4L2 (Linux), AVFoundation (macOS), and WebRTC (WASM).
* **Educational Telemetry (`include/extras/rcre_telemetry.h`)**: Step-by-step tracing hooks gated behind `#ifdef RCRE_ENABLE_TELEMETRY`. In production builds, telemetry must incur 0% runtime overhead.

### Law 4: Zero-Copy ShapeTracker Principle
* Slicing, reshaping, transpositions, and channel permutations must be strictly $\mathcal{O}(1)$ metadata updates to `shape`, `strides`, and `data` pointer offset (`TNS-REQ-04`).
* Deep copies only occur when explicitly executing `rcre_tensor_contiguous()` or during explicit adaptation.

### Law 5: Explicit Input Adaptation
* Always use `rcre_tensor_adapt()` (`ADP-REQ-01`) for letterboxing, layout conversion ($HWC \leftrightarrow CHW$), float-casting, and normalization. Never bury hidden preprocessing copies inside inference kernels.

### Law 6: Dynamic SIMD Dispatch without In-Loop Branches
* CPU capabilities must be probed once at startup (`cpuid` or `sysctl`) and cached in the global function table `rcre_hal` (`HAL-REQ-01`).
* Never place conditional CPU checks (e.g., `if (has_avx2)`) inside per-pixel or matrix loops. Dispatch directly through function pointers (`HAL-REQ-02`).

### Law 7: Strict Numerical Parity Gate
* Every algorithm must match Python reference outputs (OpenCV and Scikit-Learn) within tolerance $\epsilon \le 10^{-5}$ (`NFR-05`).
* Sanitizer verification is mandatory: All tests must run cleanly under Clang AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan).

---

## 3. Specialized Agent Roles & Authority Matrix

| Agent Role | Skill Identifier | Authority & Primary File Scope |
| :--- | :--- | :--- |
| **Agent 0: Lead Architect** | `rcre-lead-architect` | `include/rcre.h`, `CMakeLists.txt`, `Makefile`, public API review |
| **Agent 1: Core Kernel Engineer** | `rcre-kernel-engineer` | `src/core/tensor.c`, `src/core/arena.c`, `src/core/adapt.c` |
| **Agent 2: SIMD Specialist** | `rcre-simd-specialist` | `src/hal/*`, `src/hal/kernels_*.c`, CPU feature probes |
| **Agent 3: Vision & ML Engineer** | `rcre-vision-ml-engineer` | `src/cv/*`, `src/ml/*`, `include/extras/rcre_cam.h` |
| **Agent 4: QA Gatekeeper** | `rcre-qa-verifier` | `tests/*`, `benchmarks/*`, sanitizer runs, parity checks |
| **Agent 5: WASM & Web Visualizer** | `rcre-wasm-visualizer` | `web/*`, Emscripten build, Three.js 3D & 2D canvas |
| **Agent 6: Academic Book Author** | `rcre-academic-author` | `docs/book/*`, LaTeX formulas, pedagogical tutorials |

---

## 4. Verification Checkpoints

Before approving or merging any code:
1. `make test` — All unit tests pass.
2. `make test-sanitizers` — Zero leaks, zero buffer overflows under ASan/UBSan.
3. `python3 tests/test_against_scipy_opencv.py` — 100% numerical parity against industry standard baselines.
