# RCRE Multi-Agent Engineering & Repository Operations Strategy
## Autonomous Parallel Acceleration & Secure Repository Management

---

## 1. Repository Strategy & Security Analysis

### 1.1 Private GitHub Repository Evaluation
* **Verdict: Strongly Recommended.**
  * Hosting RCRE in a private GitHub repository provides version control, milestone tracking, automated multi-platform CI/CD (Ubuntu x86_64, macOS Apple Silicon, Ubuntu ARM64), and branch protection.
* **Critical Security Guideline Regarding Credentials**:
  * **DO NOT** paste GitHub Personal Access Tokens (PATs), SSH private keys, or passwords directly into AI chat prompts or commit them to text files.
  * **The Safe Protocol**:
    1. Authenticate locally on your workstation using the official GitHub CLI:
       ```bash
       gh auth login
       ```
    2. Or configure an SSH key pair (`~/.ssh/id_ed25519`) and add the public key to your GitHub account settings.
    3. Initialize and link the local repository without exposing secrets:
       ```bash
       cd /Users/erolrecep/.gemini/antigravity/scratch/rcre
       git init
       git add .
       git commit -m "feat: initial RCRE architectural specification and roadmap"
       gh repo create rcre --private --source=. --remote=origin --push
       ```

---

## 2. Multi-Agent Engineering Architecture

Because RCRE has strict, decoupled module boundaries and automated verification suites, it is an **ideal project for autonomous multi-agent acceleration**. Multiple specialized AI agents can work concurrently on isolated components without merge conflicts.

```
+-----------------------------------------------------------------------------------+
|                           AGENT 0: LEAD ARCHITECT                                 |
|          Oversees rcre.h API integrity, approves PRs, enforces C23 standards     |
+-----------------------------------------------------------------------------------+
          |                               |                               |
          v                               v                               v
+-------------------+           +-------------------+           +-------------------+
| AGENT 1: KERNEL   |           | AGENT 2: SIMD     |           | AGENT 3: VISION   |
| Core Tensors,     |           | ARM NEON, AVX-2,  |           | Filters, Resize,  |
| ShapeTracker,     |           | AVX-512, WASM     |           | CLAHE, Bilateral, |
| Memory Arenas     |           | Micro-kernels     |           | Adaptor Pipeline  |
+-------------------+           +-------------------+           +-------------------+
          |                               |                               |
          +-------------------------------+-------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           AGENT 4: QA & PARITY VERIFIER                           |
|      Runs ASan/UBSan/Valgrind, executes test_against_scipy_opencv.py, checks       |
|      numerical tolerances (epsilon <= 1e-5), and blocks regressions.              |
+-----------------------------------------------------------------------------------+
          |                                                               |
          v                                                               v
+-----------------------------------+           +-----------------------------------+
| AGENT 5: WEB & VISUALIZATION      |           | AGENT 6: BOOK & ACADEMIC AUTHOR   |
| Emscripten WASM Build, Three.js   |           | Drafts textbook chapters, LaTeX   |
| 3D Stride & Stencil Visualizers   |           | derivations, and system tutorials |
+-----------------------------------+           +-----------------------------------+
```

---

## 3. Specialized Agent Role Definitions

### Agent 0: Lead System Architect
* **Primary Scope**: `include/rcre.h`, `Makefile`, `dist/rcre.h`.
* **Responsibilities**:
  * Guards public API boundaries.
  * Enforces C23 syntax (`[[nodiscard]]`, `constexpr`, `nullptr`, `<stdatomic.h>`).
  * Reviews module interfaces before integration.

### Agent 1: Core Kernel Engineer
* **Primary Scope**: `src/core/tensor.c`, `src/core/arena.c`, `src/core/adapt.c`.
* **Responsibilities**:
  * Implements zero-copy $\mathcal{O}(1)$ slicing, transpositions, and reshape math.
  * Guarantees 64-byte alignment across all arena memory blocks.
  * Implements `rcre_tensor_adapt()` with aspect-ratio letterboxing.

### Agent 2: Hardware Acceleration & SIMD Specialist
* **Primary Scope**: `src/hal/*`, `src/hal/kernels_*.c`.
* **Responsibilities**:
  * Writes ARM NEON, AVX-2, AVX-512, and WebAssembly 128-bit SIMD intrinsics.
  * Tunes cache tiling (L1/L2 block sizes) for GEMM and stencil convolutions.
  * Maintains the dynamic CPU detection and dispatch table (`rcre_hal`).

### Agent 3: Computer Vision & ML Engineer
* **Primary Scope**: `src/cv/*`, `src/ml/*`, `include/extras/rcre_cam.h`.
* **Responsibilities**:
  * Implements classical CV filters and modern ML algorithms (GBDT inference, HNSW search).
  * Implements OS video capture drivers (V4L2 for Linux, AVFoundation for macOS).

### Agent 4: QA, Verification & Sanitizer Gatekeeper
* **Primary Scope**: `tests/*`, `benchmarks/*`.
* **Responsibilities**:
  * Executes Clang AddressSanitizer, UndefinedBehaviorSanitizer, and Valgrind memcheck.
  * Runs automated Python parity test suites against OpenCV and Scikit-Learn.
  * Acts as an automated gatekeeper: code cannot be merged without passing all parity checks.

### Agent 5: Frontend, WASM & Web Visualizer
* **Primary Scope**: `web/*`, Emscripten build pipeline.
* **Responsibilities**:
  * Compiles C23 code to `rcre.wasm` with 128-bit SIMD.
  * Builds interactive Three.js 3D tensor stride animations and Canvas2D sliding stencil visualizers.
  * Connects in-browser webcam frames to WASM inference in real time.

### Agent 6: Technical Writer & Academic Book Author
* **Primary Scope**: `docs/book/`, documentation site, tutorials.
* **Responsibilities**:
  * Authors textbook chapters explaining systems programming, mathematical derivations, and architecture.
  * Formats LaTeX formulas and code walkthroughs for publication.

---

## 4. Multi-Agent Execution Protocol

To prevent merge conflicts and coordination overhead, agents follow a strict worktree-isolated Git workflow:

```
feature/tensor-engine    (Agent 1)  -----\
feature/simd-neon        (Agent 2)  ------\   Pull Requests   +------------------+
feature/cv-filters       (Agent 3)  -------> (Sanitizer Gate) -> | main (Stable)  |
feature/wasm-visualizer  (Agent 5)  ------/   (Agent 4 Check) +------------------+
feature/book-drafts      (Agent 6)  -----/
```

1. **Branch Isolation**: Each agent operates in its own dedicated branch or Git worktree.
2. **Objective Fitness Test**: An agent cannot submit work until local tests pass:
   ```bash
   make test && python3 tests/test_against_scipy_opencv.py
   ```
3. **Automated Sanity Pass**: Agent 4 reviews all PRs with ASan, UBSan, and Valgrind.
4. **Integration**: Agent 0 merges verified PRs into `main`.
