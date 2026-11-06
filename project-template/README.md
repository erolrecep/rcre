# RCRE Project Starter Template

> **Get started with RCRE in under 60 seconds.** Zero external package managers, zero build-system complications, pure modern C23, and a single-header distribution.
> 
> ⚡ *Looking for a quick cheat sheet and silicon best practices? See the [Developer Mini Guideline](GUIDELINES.md).*

---

## 1. Overview & Philosophy

Welcome to the **RCRE (Recep's Code Run Everywhere)** project starter template. 

Starting a new high-performance C project often comes with painful friction: configuring complex build systems, linking external shared libraries, battling conflicting compiler flags, and debugging memory allocator crashes.

This template eliminates that complexity:
- **Single-Header Simplicity**: The entire RCRE engine (Dynamic SIMD HAL, Contiguous Memory Arenas, N-D Strided Tensors, Computer Vision Filters, and ML Primitives) is packaged into a self-contained header: [`include/rcre.h`](file:///Users/erolrecep/Developer/codebase/apps/rcre/project-template/include/rcre.h).
- **Zero External Dependencies**: Compiles out of the box with standard C compilers (`clang`, `gcc`, or `cc`) with only `-lm -lpthread`.
- **Zero-Allocation Memory Arena**: Learn the production paradigm that powers modern AI engines—pre-allocate linear memory once, reset in $O(1)$ time, and eliminate heap fragmentation entirely.
- **Hardware-Accelerated Dispatch**: Automatically probes host CPU capabilities (ARM NEON, x86 AVX-512, AVX2, or WASM SIMD128) and dispatches vector micro-kernels dynamically at runtime.
- **Battle-Tested Sanitizers**: Pre-configured test runner with AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan).

---

## 2. Directory Structure

```text
project-template/
├── Makefile                # Turnkey build system (all, run, test, clean)
├── README.md               # Complete newcomer onboarding manual (you are here)
├── GUIDELINES.md           # Developer mini guideline & silicon best practices cheat sheet
├── .gitignore              # Ignores build binaries and object artifacts
├── include/
│   └── rcre.h              # Amalgamated single-header C23 RCRE engine (~14k lines)
├── src/
│   └── main.c              # Pedagogical starter application demonstrating all core APIs
├── tests/
│   └── test_main.c         # Unit test suite verifying alignment, tensors, and math parity
├── bin/
│   └── .gitkeep            # Compiled binaries output directory
└── logs/
    ├── .gitkeep            # Output logs directory
    └── sample_run.log      # Captured telemetry log from sample benchmark run
```

---

## 3. Quick Start: Build & Run in 60 Seconds

### Prerequisites
- A modern C compiler supporting C23/C2x:
  - **macOS**: Apple Clang 15+ or LLVM Clang 16+ (included with Xcode Command Line Tools: `xcode-select --install`).
  - **Linux**: GCC 13+ or Clang 16+ (`sudo apt-get install clang make` or `sudo dnf install clang make`).
  - **Windows**: WSL2 (Ubuntu 22.04+) or MSYS2 Clang64.
- Standard `make`.

### Step 1: Clone or Copy the Template
Copy the `project-template` folder anywhere on your machine to start your new project:
```bash
cp -r /path/to/apps/rcre/project-template ~/my-rcre-app
cd ~/my-rcre-app
```

### Step 2: Build & Run the Sample Application
```bash
make run
```

You will see output similar to:
```text
🔨 Compiling bin/app (-O3)...
cc -std=c2x -Wall -Wextra -Wpedantic -O3 -Iinclude src/main.c -lm -lpthread -o bin/app
✅ Built bin/app successfully.
🚀 Executing bin/app...
===================================================================
   ____   ____ ____  _____
  |  _ \ / ___|  _ \| ____|
  | |_) | |   | |_) |  _|
  |  _ <| |___|  _ <| |___
  |_| \_\\____|_| \_\_____|  Project Starter Template (C23)
===================================================================

[1/5] Initializing Hardware Abstraction Layer (HAL)...
      -> Detected SIMD Acceleration: [ARM NEON 128-bit]

[2/5] Creating 16MB contiguous memory arena...
      -> Arena allocated at: 0x7226400000 (Capacity: 16.00 MB)
      -> Alignment check: PASS (64-byte AVX-512 / Cache-Line Aligned) (address % 64 == 0)

[3/5] Allocating 1080p RGB tensor (1920x1080x3, U8)...
      -> Image synthesized. Arena memory used: 5.93 MB / 16.00 MB

[4/5] Executing SIMD-accelerated Rec.601 RGB-to-Grayscale conversion...
      -> Processed 100 frames of 1080p (2.07 MP)
      -> Average Latency: 0.306 ms per frame
      -> Throughput:      6787.56 Megapixels / second

[5/5] Arena lifecycle and memory reset...
      -> Peak Arena Offset: 8294400 bytes (7.91 MB)
      -> Arena Remaining:   8482816 bytes (8.09 MB)
      -> rcre_arena_reset() executed: current offset = 0 bytes (O(1) reset)
      -> rcre_arena_destroy() complete. Zero memory leaks.

===================================================================
🚀 Template execution successful! You are ready to build with RCRE.
===================================================================
```

### Step 3: Run the Sanitized Test Suite
```bash
make test
```
This builds and executes `tests/test_main.c` under `-fsanitize=address,undefined`:
```text
===================================================================
🧪 Running RCRE Starter Project Verification Tests
===================================================================
  [RUN] test_hal_initialization...
  [PASS] test_hal_initialization
  [RUN] test_arena_alignment_and_rewind...
  [PASS] test_arena_alignment_and_rewind
  [RUN] test_tensor_creation...
  [PASS] test_tensor_creation
  [RUN] test_cv_rgb_to_gray_parity...
  [PASS] test_cv_rgb_to_gray_parity

-------------------------------------------------------------------
Test Results: 4/4 passed (ALL TESTS PASSED ✅)
-------------------------------------------------------------------
```

---

## 4. Deep-Dive: Understanding the Starter Files

### 4.1 `include/rcre.h` (The Single-Header Engine)
Like the famous `stb` libraries or `miniaudio`, RCRE distributes as a unified single header. 
In **exactly one** `.c` translation unit (by default, `src/main.c`), define `RCRE_IMPLEMENTATION` before including the header:

```c
#define RCRE_IMPLEMENTATION
#include "rcre.h"
```

In any other translation units (if you add `src/filter.c` or `src/model.c`), simply include the header without `#define RCRE_IMPLEMENTATION`:
```c
#include "rcre.h"
```

### 4.2 `src/main.c` (The 5 Core Steps)

[`src/main.c`](file:///Users/erolrecep/Developer/codebase/apps/rcre/project-template/src/main.c) illustrates the standard lifecycle of any RCRE application:

1. **Hardware Detection**:
   ```c
   rcre_hal_init();
   printf("Detected SIMD: %s\n", rcre_hal.backend_name);
   ```
   At startup, RCRE probes CPU registers (via `CPUID` on x86, or sysctl / HWCAP on ARM). The function dispatch table `rcre_hal` is automatically wired to the fastest available vector instructions (e.g. AVX-512, AVX2+FMA, NEON, or scalar fallback).

2. **Linear Arena Allocation**:
   ```c
   RCRE_Arena* arena = rcre_arena_create(16 * 1024 * 1024); // 16 MB
   ```
   Instead of calling `malloc()` millions of times, RCRE allocates a single large memory block aligned to 64 bytes (the x86/ARM L1 cache line and AVX-512 register width).

3. **Strided N-D Tensor Creation**:
   ```c
   size_t rgb_shape[3] = {1080, 1920, 3};
   RCRE_Tensor rgb = rcre_tensor_create(rgb_shape, 3, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
   ```
   Tensors are sub-allocated linearly from the arena in $O(1)$ time with zero OS syscall overhead.

4. **SIMD Computation**:
   ```c
   size_t gray_shape[2] = {1080, 1920};
   RCRE_Tensor gray = rcre_tensor_create(gray_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);
   rcre_cv_rgb_to_gray(&rgb, &gray);
   ```
   The engine executes vector micro-kernels achieving gigabytes-per-second memory bandwidth and multi-gigapixel throughput.

5. **$O(1)$ Reset & Teardown**:
   ```c
   rcre_arena_reset(arena);   // Instantly rewinds offset to 0 for next frame/request
   rcre_arena_destroy(arena); // Releases the 16MB block back to the OS
   ```

### 4.3 `tests/test_main.c` (Verification Gatekeeper)
[`tests/test_main.c`](file:///Users/erolrecep/Developer/codebase/apps/rcre/project-template/tests/test_main.c) provides a lightweight, zero-dependency C23 unit testing harness:
- **`test_hal_initialization`**: Asserts that dynamic function pointers are bound.
- **`test_arena_alignment_and_rewind`**: Asserts that every sub-allocation is strictly 64-byte aligned and verifies instant offset reset.
- **`test_tensor_creation`**: Asserts dimensions, strides, and memory filling.
- **`test_cv_rgb_to_gray_parity`**: Mathematically checks Rec.601 color formula:
  $$Y = 0.299 R + 0.587 G + 0.114 B$$
  Verifies that SIMD integer arithmetic matches floating-point ground truth within $\pm 1$ LSB rounding tolerance.

### 4.4 `Makefile` (Clean, Portable Commands)
[`Makefile`](file:///Users/erolrecep/Developer/codebase/apps/rcre/project-template/Makefile) defines clear developer workflows:
- `make`: Builds the optimized binary `bin/app` (`-O3 -std=c2x`).
- `make run`: Runs `bin/app` and pipes execution metrics to [`logs/sample_run.log`](file:///Users/erolrecep/Developer/codebase/apps/rcre/project-template/logs/sample_run.log).
- `make test`: Builds `bin/test_runner` with AddressSanitizer and UndefinedBehaviorSanitizer (`-fsanitize=address,undefined -g`) and runs tests.
- `make clean`: Deletes all binaries and intermediate object files.
- `make help`: Prints an interactive cheat sheet.

### 4.5 `logs/sample_run.log` (Telemetry Record)
Every run of `make run` logs real-time diagnostics:
- SIMD acceleration backend detected.
- Memory arena pointers and alignment verification.
- Peak memory offset used vs. remaining capacity.
- Throughput in Megapixels / second and average latency per frame in milliseconds.

---

## 5. How to Build Your Own Application

### Example: Adding a 2D Gaussian Blur Filter
Open [`src/main.c`](file:///Users/erolrecep/Developer/codebase/apps/rcre/project-template/src/main.c) and allocate a blurred output tensor:

```c
size_t blur_shape[2] = {height, width};
RCRE_Tensor blurred = rcre_tensor_create(blur_shape, 2, RCRE_DTYPE_U8, RCRE_LAYOUT_HWC, arena);

// Execute 5x5 separable Gaussian blur with sigma=1.0
int status = rcre_cv_gaussian_blur(&gray, &blurred, 5, 1.0f, arena);
if (status == RCRE_OK) {
    printf("Gaussian blur applied successfully!\n");
}
```

### Example: Matrix Operations & Linear Algebra
You can create float tensors and execute matrix multiplications or linear algebra:

```c
size_t a_shape[2] = {128, 256};
size_t b_shape[2] = {256, 64};
size_t c_shape[2] = {128, 64};

RCRE_Tensor A = rcre_tensor_create(a_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
RCRE_Tensor B = rcre_tensor_create(b_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);
RCRE_Tensor C = rcre_tensor_create(c_shape, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_HWC, arena);

rcre_tensor_fill(&A, 1.0f);
rcre_tensor_fill(&B, 2.0f);

// Hardware-accelerated GEMM: C = alpha * (A @ B) + beta * C
rcre_hal.gemm_f32(128, 64, 256, 1.0f, (float*)A.data, 256, (float*)B.data, 64, 0.0f, (float*)C.data, 64);
```

---

## 6. The Three Golden Rules of RCRE

When writing production systems in RCRE, keep these three rules in mind:

1. **Never Call `malloc()` in Inner Loops**:
   All allocations should come from an `RCRE_Arena`. Pre-allocate your arena at startup, sub-allocate during processing, and reset (`rcre_arena_reset()`) at frame/batch boundaries. This eliminates heap fragmentation and latency spikes.

2. **Always Respect 64-Byte Alignment**:
   SIMD vector load instructions (`vmovaps`, `vld1q_f32`) execute significantly faster and prevent bus alignment faults when pointers are 64-byte aligned. RCRE arenas enforce this automatically.

3. **Run Sanitizers Frequently**:
   Before deploying your code, run `make test` or compile with `-fsanitize=address,undefined`. AddressSanitizer will immediately pinpoint any out-of-bounds reads or buffer overflows.

---

## 7. Next Steps & Recommended Reading

Now that you have your project up and running:
- Explore the **RCRE Book**: Learn the mathematical foundations of strided tensors, SIMD kernels, and cache memory hierarchies.
- Inspect the **Production Projects**: Check `apps/rcre/projects/` (or Appendix A in the RCRE book) to see how RCRE powers real-world face expression analysis, document super-resolution, PCA compression, HNSW vector search, and gradient boosting trees.
- Happy hacking with C23!

