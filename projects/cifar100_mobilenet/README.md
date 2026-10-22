# RCRE MobileNet CIFAR-100 Inference & Training Engine

Zero-heap, sub-4ms MobileNet inference engine and PyTorch training pipeline on CIFAR-100 ($32 \times 32$ image classification) implemented in ISO C23 for **RCRE (Recep's Code Run Everywhere)**.

---

## 🌟 Key Features

1. **Pure ISO C23 Zero-Heap Allocations**:
   - Uses linear memory arenas (`RCRE_Arena`) for persistent model parameters and dynamic activation memory.
   - Zero `malloc` calls in the critical inference path.
2. **Zero-Cost Edge BatchNorm Folding**:
   - `rcre_nn_fuse_conv_batchnorm` mathematically absorbs all scale ($\gamma$), shift ($\beta$), running mean ($\mu$), and variance ($\sigma^2$) directly into Conv2D and Depthwise Conv2D weights and biases at export/load time.
   - Eliminates BatchNorm computation overhead completely during inference ($0$ latency impact).
3. **Hardware-Accelerated SIMD Micro-Kernels**:
   - ARM NEON (Apple Silicon / Raspberry Pi) & x86 AVX2 acceleration via the RCRE Hardware Abstraction Layer (`rcre_hal`).
   - Pointwise $1 \times 1$ convolutions executed via SIMD `axpy_f32` vectors.
   - Fully connected linear layers accelerated via hardware dot product `dot_f32`.
4. **Blazing Fast Performance**:
   - **~3.24 ms mean latency** (~308 FPS) on Apple Silicon M5 Max.
   - 589,220 parameters (2.25 MB binary footprint).

---

## 🏗️ Architecture Specification

The network is adapted specifically for CIFAR-100 ($32 \times 32 \times 3$ input resolution):

| Layer | Type | Input Shape | Output Shape | Kernel / Stride | Activation |
|---|---|---|---|---|---|
| 0 | Stem Conv2D + Bias | $[3, 32, 32]$ | $[32, 32, 32]$ | $3 \times 3$, stride 1, pad 1 | ReLU6 |
| 1 | Depthwise + Pointwise | $[32, 32, 32]$ | $[64, 32, 32]$ | DW: $3 \times 3$, s1 \| PW: $1 \times 1$ | ReLU6 |
| 2 | Depthwise + Pointwise | $[64, 32, 32]$ | $[128, 16, 16]$ | DW: $3 \times 3$, s2 \| PW: $1 \times 1$ | ReLU6 |
| 3 | Depthwise + Pointwise | $[128, 16, 16]$ | $[128, 16, 16]$ | DW: $3 \times 3$, s1 \| PW: $1 \times 1$ | ReLU6 |
| 4 | Depthwise + Pointwise | $[128, 16, 16]$ | $[256, 8, 8]$ | DW: $3 \times 3$, s2 \| PW: $1 \times 1$ | ReLU6 |
| 5 | Depthwise + Pointwise | $[256, 8, 8]$ | $[256, 8, 8]$ | DW: $3 \times 3$, s1 \| PW: $1 \times 1$ | ReLU6 |
| 6 | Depthwise + Pointwise | $[256, 8, 8]$ | $[512, 4, 4]$ | DW: $3 \times 3$, s2 \| PW: $1 \times 1$ | ReLU6 |
| 7 | Depthwise + Pointwise | $[512, 4, 4]$ | $[512, 4, 4]$ | DW: $3 \times 3$, s1 \| PW: $1 \times 1$ | ReLU6 |
| 8 | Global Average Pool | $[512, 4, 4]$ | $[512, 1, 1]$ | Spatial Reduction ($4 \times 4$) | Linear |
| 9 | Linear Projection + Softmax | $[512]$ | $[100]$ | Dense Matrix Vector | Softmax |

---

## 🚀 Quickstart & Commands

From `apps/rcre`:

```bash
# Compile and run the demo on 10 CIFAR-100 test samples
make -C projects/cifar100_mobilenet demo

# Run the 50-iteration latency and throughput benchmark
make -C projects/cifar100_mobilenet benchmark

# Classify a single custom float32 binary sample
./projects/cifar100_mobilenet/bin/rcre-mobilenet --input projects/cifar100_mobilenet/data/sample_0.bin

# Train the network using PyTorch & re-export fused weights
make -C projects/cifar100_mobilenet train
```

---

## 📦 File Layout

```
projects/cifar100_mobilenet/
├── cifar100_labels.h        # 100 CIFAR-100 class names & 20 superclass mappings
├── main.c                   # C23 zero-allocation MobileNet CLI application
├── Makefile                 # Build rules linking with librcre.a
├── mobilenet_cifar100.bin   # Fused binary weights (RCRE_MOB v1)
├── data/
│   ├── sample_0.bin .. 9.bin# Contiguous float32 [3, 32, 32] normalized test samples
│   └── samples.txt          # Ground-truth manifest & PyTorch top-1 comparison
└── train/
    ├── model.py             # PyTorch MobileNet CIFAR-100 model & weight exporter
    └── train_cifar100.py    # Training & evaluation pipeline
```
