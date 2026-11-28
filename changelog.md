# Changelog

All notable changes to the RCRE (Recep's Code Run Everywhere) project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.7.1] - 2026-11-28

### Added
- **GraphicsMagick-Exact Resampling Tier (`src/cv/resize_gm.c`)**:
  - Implemented 16 continuous windowed filter kernels (`Point`, `Box`, `Triangle`, `Hermite`, `Hanning`, `Hamming`, `Blackman`, `Gaussian`, `Quadratic`, `Cubic`, `Catrom`, `Mitchell`, `Lanczos`, `Bessel`, `Sinc`) ported from GraphicsMagick 1.3.49 (`magick/resize.c`).
  - Added `RCRE_CV_Interp` enum and public API `rcre_cv_resize_gm(const RCRE_Tensor* src, RCRE_Tensor* dst, RCRE_CV_Interp interp, RCRE_Arena* arena)` in `include/rcre.h`.
  - Added separable 2-pass execution with pass-ordering heuristic ($H \to V$ vs $V \to H$) minimizing intermediate pixel work:
    $$W_{\text{out}} \cdot (H_{\text{in}} + H_{\text{out}}) > H_{\text{out}} \cdot (W_{\text{in}} + W_{\text{out}})$$
  - Added continuous subpixel mapping preserving aspect ratios and eliminating boundary drift.
  - Native multi-channel support for 2D `[H, W]`, 3D planar `[C, H, W]`, and 3D interleaved `[H, W, C]` float tensors.
  - Linear scratchpad arena integration (`RCRE_Arena`) for temporary tap weights and intermediate pass buffers with zero-malloc fallback.
- **Monorepo Cross-Project Integration Targets**:
  - Added `project_rcvc_sync`, `project_rcvc_test`, `project_rcvc_san`, and `project_rcvc_all` targets to `apps/rcre/Makefile`.
  - Added `test-rcre`, `test-rcvc`, and `test-systems` targets to monorepo root `Makefile`.
- **Amalgamated Single-Header Distribution**:
  - Regenerated `dist/rcre.h` with `resize_gm.c` bundled: **52,666 lines, 1.96 MB**.
- **Documentation**:
  - Added [docs/cv_resampling.md](docs/cv_resampling.md) detailing the mathematical model, filter window formulations, and pass ordering optimizations.
- **Turnkey Production Projects (Project 14 & 15)**:
  - Migrated `rcre_server_vs_go_benchmark` into `projects/rcre_server_vs_go_benchmark/` as **`rcre-server-bench`** (Project 14).
  - Added **`rcre-cam-stream`** (Project 15) in `projects/rcre_cam_stream/`: turnkey zero-copy camera stream studio with lock-free SPSC frame ring buffer (`rcre_ringbuf.h`), background capture worker, OpenGL 3.3 fragment shader filters, interactive toolbar, telemetry HUD, and full-resolution snapshot codec.
  - Added `project_cam_stream_build`, `project_cam_stream_demo`, `project_cam_stream_run`, `project_cam_stream_san`, and `project_cam_stream_clean` to `Makefile`.
  - Added comprehensive architectural documentation in `projects/README.md` (Sections 14 and 15).
- **Roadmap / Planned (Project 16)**:
  - Added specification and roadmap TODO in `projects/README.md` for `rcre-cv-ops` (`projects/rcre_cv_ops/`), a planned turnkey C23 CLI utility providing command-line access to continuous image operations, GM-exact filters, e-means palette quantization, and terminal display backends migrating from the retired `rcvc` project.

---

## [2.7.0] - 2026-11-27

### Added
- **Generative & Vision Deep Learning Suite (`include/rcre_gen.h`, `src/gen/`)**:
  - **Convolutional / Denoising Autoencoder** (`RCRE_ConvAE`): Encoder-decoder with strided and transposed spatial convolutions.
  - **Convolutional Variational Autoencoder** (`RCRE_ConvVAE` / $\beta$-VAE): Reparameterized latent sampling with accumulating analytical Gaussian KL divergence.
  - **Time-Conditioned Tiny U-Net & Diffusion Scheduler** (`RCRE_TinyUNet`, `RCRE_DiffusionScheduler`): Discrete DDPM ancestral and deterministic $\eta$-DDIM sampling with linear and cosine noise schedules.
  - **Vision Transformer & Diffusion Transformer with AdaLN-Zero** (`RCRE_ViT`, `RCRE_DiT`): Patchification, learnable/sinusoidal position embeddings, multi-head self-attention with causal/bidirectional masking, adaptive layer normalization (AdaLN-Zero), and class conditioning.
  - **25+ Reverse-Mode Tensor Tape Autograd VJP Operations**: Added LayerNorm, GroupNorm, MHSA, patchify/unpatchify, FiLM, AdaLN, gate, avgpool, embedding lookup, and reparameterization backward gradients to `src/autograd/tensor_autograd.c`.
  - **Model Lifecycle & Quantization**: Magnitude weight pruning, lossless RCRZ compression (`.rgc1`), post-training quantization (F16, BF16, Q8_0, Q4_0 $\to$ `.rgq1`), and `.safetensors` checkpoint serialization.
  - **Project 13: `rcre_gen_studio`**: Turnkey interactive CLI, training automation, Docker deployment packager, and zero-dependency POSIX HTTP JSON inference server.
  - **Unit Test Suite (`tests/test_gen.c`)**: 9 test suites and 37,598 assertions validating central-difference gradient checks, schedule values, and same-seed bit-identical sampling.

---

## [2.6.0] - 2026-11-26

### Added
- **HPC Wave: MPI Shim, Distributed Operators, 3D Poisson 256³, pow2 FFT (`docs/hpc.md` §12)**:
  - 12-function MPI shim over self-registering backends with `loopback` conformance oracle.
  - Distributed row-block SpMV (+ all-to-all gather) and halo-exchange 5-point stencil.
  - 3D structured-grid PDE layer (7-point stencil + separable DST-I 3D Poisson solver) with 256³ full run verified bit-reproducible across threadpool sizes.
  - Optimized in-place power-of-two F64 FFT (`rcre_fft_pow2_f64`).
  - Library-wide `-ffp-contract=off` compilation flag ensuring bit-level determinism across platforms.

---

## [2.5.0] - 2026-11-25

### Added
- **Optimization Polish: L-BFGS, FISTA, CD-LASSO, SPSA (`docs/hpc.md` §11)**:
  - F64 two-loop L-BFGS with Armijo backtracking and box constraints verified against SciPy L-BFGS-B.
  - Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) and cyclic coordinate descent for LASSO.
  - Simultaneous Perturbation Stochastic Approximation (SPSA) zeroth-order optimizer.

---

## [2.4.0] - 2026-11-24

### Added
- **Domain Layer: PDE, N-Body, MCMC & Streaming Statistics (`docs/hpc.md` §10)**:
  - 2D Poisson solver (separable DST-I spectral + multigrid V-cycle).
  - 2D Heat FTCS and 1D Burgers MUSCL-upwind with Shu-Osher and Hopf-Lax reference solutions.
  - 2D Barnes-Hut N-body simulation with velocity-Verlet integration.
  - Metropolis-Hastings (MH), Hamiltonian Monte Carlo (HMC), and No-U-Turn Sampler (NUTS).
  - Online streaming statistics with Chan-merge Welford moments, $P^2$ quantiles, and moving-block bootstrap.

---

## [2.3.0] - 2026-11-23

### Added
- **GPU Bridge: Context ABI + CUDA Backend (`docs/hpc.md` §9)**:
  - Unified GPU execution context ABI with CPU reference backend oracle and optional CUDA backend.
  - Philox-4x32-10 pseudo-random number generator bit-identical across CPU and GPU.
  - GPU-accelerated GEMM, SpMV, AXPY, and DOT operators.

---

## [2.2.0] - 2026-11-22

### Added
- **Out-of-Core Zarr v3 Chunked Arrays (`docs/hpc.md` §8)**:
  - Canonical Zarr v3 storage format support on POSIX filesystem across 12 NumPy-compatible dtypes.
  - Slab-parallel chunked I/O, strided `write_block`, and deterministic `reduce_sum`.
  - Conformance test suite verified against `zarr-python`.

---

## [2.1.0] - 2026-11-21

### Added
- **H0 Scientific Substrate (`docs/hpc.md` §1-§4)**:
  - Dispatched F64 BLAS operations (`gemm`, `gemv`, `dot`, `axpy`, `scale`, `sum`) across NEON, AVX2, and AVX-512.
  - Kahan-Neumaier compensated floating-point summations.
  - Special functions library (`erf`, `erfc`, `lgamma`, `digamma`, `logsumexp`, `expit`, `norm_cdf`).

---

## [2.0.0] - 2026-11-15

### Added
- **Core C23 Engine Architecture**:
  - ISO C23 standard compliance (`clang -std=c2x` / `gcc -std=c23`).
  - 64-byte aligned linear scratchpad memory arena allocator (`RCRE_Arena`).
  - Strided N-D tensor abstraction (`RCRE_Tensor`) with $O(1)$ ShapeTracker zero-copy views.
  - Dynamic Hardware Abstraction Layer (HAL) with runtime CPU feature probe.
