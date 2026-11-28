# `rcre_gen_studio` — Generative & Vision AI Studio (Pure ISO C23)

Turnkey zero-dependency studio covering the **complete ML lifecycle (`train → compress → quantize → infer → deploy / serve`)** across RCRE's five generative & vision architectures (`include/rcre_gen.h`):

1. **Convolutional & Denoising Autoencoder (`ae` / `RCRE_ConvAE`)**
2. **Convolutional Variational Autoencoder (`vae` / `RCRE_ConvVAE` / $\beta$-VAE)**
3. **Diffusion Tiny U-Net (`diffusion` / `RCRE_DiffSchedule` + `RCRE_TinyUNet`, DDPM & DDIM)**
4. **Vision Transformer (`vit` / `RCRE_ViT`)**
5. **Diffusion Transformer (`dit` / `RCRE_DiT` with AdaLN-Zero)**

## Lifecycle Capabilities (All 5 Models)

- **Train**: Pure C23 `RCRE_TensorTape` + Decoupled AdamW + gradient clipping → `.safetensors` F32 checkpoints.
- **Compress**: Unstructured magnitude pruning (`rcre_gen_prune_magnitude`) + lossless 0-ULP SIMD byte-shuffled RCRZ archives (`.rgc1`).
- **Quantize**: Post-training quantization to `F16` (`~2x`), `BF16` (`~2x`), `Q8_0` (`~3.76x`), and `Q4_0` (`~7.11x`) into `.rgq1` artifacts with exact SQNR/RMSE reporting.
- **Infer**: Auto-detecting loader (`rcre_gen_load_any`) supporting `.safetensors`, `.rgc1`, and `.rgq1`, with latency benchmarks, PSNR/accuracy reporting, ASCII previews, and lossless PNG sprite grids.
- **Deploy & Serve**: Self-contained single-header edge packager (`scripts/deploy_package.sh` → `rcre.h` + `deploy_infer.c` + quantized weights + `Dockerfile` + `run.sh`) and a zero-dependency POSIX HTTP JSON server (`rcre_gen_studio serve`).

## Quick Start

```bash
# Run full lifecycle demo (train -> compress -> quantize -> infer) on all 5 models:
make -C projects/rcre_gen_studio demo

# Run automated lifecycle pipeline script for all 5 models (exports F32, RGC1, F16, Q8_0, Q4_0):
make -C projects/rcre_gen_studio pipeline

# Package a standalone single-header edge deployment bundle (+ Dockerfile):
make -C projects/rcre_gen_studio deploy MODEL=dit QTYPE=q8_0

# Launch zero-dep HTTP JSON inference server:
make -C projects/rcre_gen_studio serve MODEL=vit QTYPE=q8_0 PORT=8090

# Run unit tests & AddressSanitizer/UBSan verification:
make -C projects/rcre_gen_studio test
make -C projects/rcre_gen_studio san
```

📖 For full CLI flags, quantization metrics, Docker instructions, and C API details, see [**DEPLOYMENT.md**](DEPLOYMENT.md).
