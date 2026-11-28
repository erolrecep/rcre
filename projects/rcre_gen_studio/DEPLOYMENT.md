# Deployment & Lifecycle Runbook — RCRE Generative & Vision Models

Complete end-to-end instructions for **training**, **compressing** (magnitude pruning + lossless byte-shuffled RCRZ), **quantizing** (`F16`, `BF16`, `Q8_0`, `Q4_0`), **running inference**, and **deploying** all five RCRE generative & vision architectures (`ae`, `vae`, `diffusion`, `vit`, `dit`).

---

## 1. Quick Automated Pipeline (All 5 Models)

Run the full lifecycle (`train → compress → quantize → infer`) across every model with one command:

```bash
cd apps/rcre/projects/rcre_gen_studio

# Run all 5 models (ae, vae, diffusion, vit, dit):
make pipeline

# Or target a single model with custom steps and sparsity:
make pipeline MODEL=dit STEPS=80 SPARSITY=0.60
```

Each model writes artifacts under `build/pipeline_artifacts/<model>/`:
- `<model>_f32.safetensors` — Trained IEEE-754 F32 checkpoint (HuggingFace `.safetensors` format)
- `<model>_pruned.rgc1` — Magnitude-pruned + SIMD byte-shuffled lossless RCRZ archive (`0-ULP` bit-exact)
- `<model>_f16.rgq1` — IEEE FP16 quantized checkpoint (`~2.0x` smaller)
- `<model>_q8_0.rgq1` — Block-32 INT8 + FP16 scale quantized checkpoint (`~3.76x` smaller)
- `<model>_q4_0.rgq1` — Block-32 INT4 + FP16 scale quantized checkpoint (`~7.11x` smaller)
- `<model>_q8_preview.png` — Lossless PNG grid of quantized inference outputs

---

## 2. Step-by-Step CLI Reference

All five commands accept `--model <ae|vae|diffusion|vit|dit>`:

### Step A — Train (`train`)
Trains the model from scratch using `RCRE_TensorTape` + Decoupled AdamW + global L2 gradient clipping and exports `.safetensors`:

```bash
./build/rcre_gen_studio train --model ae        --steps 60 --out build/ae.safetensors
./build/rcre_gen_studio train --model vae       --steps 60 --out build/vae.safetensors
./build/rcre_gen_studio train --model diffusion --steps 60 --out build/diffusion.safetensors
./build/rcre_gen_studio train --model vit       --steps 60 --out build/vit.safetensors
./build/rcre_gen_studio train --model dit       --steps 60 --out build/dit.safetensors
```

### Step B — Compress (`compress`: Magnitude Pruning + RCRZ)
Applies unstructured magnitude pruning (preserving 1D bias/LayerNorm/GroupNorm parameters by default) and writes a lossless byte-shuffled `"RGC1"` RCRZ archive:

```bash
./build/rcre_gen_studio compress --model dit \
    --weights build/dit.safetensors \
    --sparsity 0.50 \
    --out build/dit_pruned50.rgc1
```

### Step C — Quantize (`quantize`: `f16` | `bf16` | `q8_0` | `q4_0`)
Quantizes any F32 or pruned checkpoint into a compact `"RGQ1"` artifact and reports exact compression ratio, RMSE, MaxAbs error, and SQNR (dB):

```bash
# Block-32 INT8 (~3.76x smaller, ~40+ dB SQNR — recommended production default)
./build/rcre_gen_studio quantize --model dit --weights build/dit.safetensors --qtype q8_0 --out build/dit_q8.rgq1

# Block-32 INT4 (~7.11x smaller — ultra-compact edge footprint)
./build/rcre_gen_studio quantize --model dit --weights build/dit.safetensors --qtype q4_0 --out build/dit_q4.rgq1

# Half precision (2.0x smaller, ~65+ dB SQNR)
./build/rcre_gen_studio quantize --model dit --weights build/dit.safetensors --qtype f16  --out build/dit_f16.rgq1
```

### Step D — Inference & Evaluation (`infer`)
`rcre_gen_load_any()` auto-detects `.safetensors`, `.rgc1`, and `.rgq1` (`F16`/`BF16`/`Q8_0`/`Q4_0`):

```bash
./build/rcre_gen_studio infer --model dit --weights build/dit_q8.rgq1 --ddim-steps 10 --ascii --out-png build/dit_out.png
./build/rcre_gen_studio infer --model vit --weights build/vit_q8.rgq1
./build/rcre_gen_studio infer --model ae  --weights build/ae_pruned50.rgc1 --ascii --out-png build/ae_out.png
```

---

## 3. Deployment Options

### Option 1 — Standalone Single-Header Edge Binary (Zero Dependencies)
Packages `dist/rcre.h`, `deploy/deploy_infer.c`, your chosen quantized weights, a precompiled native binary `rcre_infer`, `run.sh`, and a multi-stage `Dockerfile`:

```bash
# Package & smoke-test a standalone Q8_0 DiT edge deployment bundle:
make deploy MODEL=dit QTYPE=q8_0

# Package any of the 5 models:
./scripts/deploy_package.sh ae        build/pipeline_artifacts/ae/ae_q8_0.rgq1               build/deploy_ae
./scripts/deploy_package.sh vae       build/pipeline_artifacts/vae/vae_q8_0.rgq1             build/deploy_vae
./scripts/deploy_package.sh diffusion build/pipeline_artifacts/diffusion/diffusion_q8_0.rgq1 build/deploy_diffusion
./scripts/deploy_package.sh vit       build/pipeline_artifacts/vit/vit_q8_0.rgq1             build/deploy_vit
./scripts/deploy_package.sh dit       build/pipeline_artifacts/dit/dit_q8_0.rgq1             build/deploy_dit
```

To cross-compile or build the bundle on any target machine with only a C23 compiler:

```bash
cd build/deploy_dit
cc -std=c2x -O3 -ffp-contract=off deploy_infer.c -lm -lpthread -o rcre_infer
./run.sh --out-png sample.png
```

### Option 2 — Docker Container Deployment
Every generated bundle includes a minimal multi-stage `Dockerfile`:

```bash
cd build/deploy_dit
docker build -t rcre-dit-edge .
docker run --rm rcre-dit-edge
```

### Option 3 — Zero-Dependency HTTP JSON Microservice (`serve`)
Serve any of the 5 models locally over loopback POSIX sockets with zero external libraries:

```bash
# Start HTTP service on http://127.0.0.1:8090
make serve MODEL=vit QTYPE=q8_0 PORT=8090

# Query endpoints from another terminal:
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/v1/model
curl -X POST http://127.0.0.1:8090/v1/infer
```

---

## 4. C23 Library API Summary (`include/rcre_gen.h`)

All lifecycle operations are available directly from C23 (or via `dist/rcre.h`):
- **Save/Load F32**: `rcre_gen_save_safetensors`, `rcre_gen_load_safetensors`
- **Magnitude Pruning**: `rcre_gen_prune_magnitude`
- **Lossless RCRZ Archive (`"RGC1"`)**: `rcre_gen_save_compressed`, `rcre_gen_load_compressed`
- **Quantization (`"RGQ1"`: F16 / BF16 / Q8_0 / Q4_0)**: `rcre_gen_save_quantized`, `rcre_gen_load_quantized`
- **Auto-Detect Loader**: `rcre_gen_load_any`
