# 🎨 RCRE PCA Image Compression & Quantization Engine

> **A Pure C23, Zero-Dependency Dimensionality Reduction & Image Codec**  
> Built natively on top of RCRE Core Tensor Math (`src/core/*`) and Machine Learning Kernels (`src/ml/cluster.c`).  
> Zero Python, Zero libjpeg, Zero OpenCV, Zero external dependencies.

---

## 📖 Overview

**RCRE PCA Compression** is a standalone, ultra-fast image compression engine that leverages **Principal Component Analysis (PCA)** and **8-Bit Scalar Uniform Quantization** on spatial 2D patches.

By projecting localized image patches onto their primary eigen-vectors and quantizing projection coefficients:
- **High Compression Ratios**: Achieves **$8.8\times$ to $25\times$ compression** ($88\%$ to $96\%$ reduction in payload).
- **High Reconstruction Fidelity**: Yields **$33.5\text{ dB}$ to $58.0\text{ dB}$ PSNR** (Peak Signal-to-Noise Ratio).
- **Sub-Millisecond Decoding**: Decompression executes in **$< 0.35\text{ ms}$** via SIMD-accelerated linear combination (`rcre_ml_pca_inverse_transform`).
- **Deterministic Linear Memory**: Operates with $0$ heap allocations in hot paths using a high-speed linear memory arena (`RCRE_Arena`).

---

## 📐 Mathematical Formulation

### 1. Spatial Tiling
Given an image $I \in \mathbb{R}^{H \times W \times C}$, we partition it into non-overlapping $B \times B$ spatial patches (e.g., $B=8, C=3$).  
Each patch is unrolled into a feature vector of dimension:
$$D = B \times B \times C = 8 \times 8 \times 3 = 192$$
For an image with $N$ patches, we form the patch data matrix:
$$X \in \mathbb{R}^{N \times D}$$

### 2. Centering & Covariance Analysis
$$\mu = \frac{1}{N} \sum_{i=1}^N X_i \in \mathbb{R}^D, \quad X_c = X - \mu$$
$$\Sigma = \frac{1}{N - 1} X_c^T X_c \in \mathbb{R}^{D \times D}$$

### 3. Principal Eigenvector Extraction (Power Iteration & Deflation)
We extract the top $K$ principal components ($K \ll D$, e.g., $K=8$ or $K=12$):
$$W \in \mathbb{R}^{K \times D}$$
Each eigenvector satisfies $\Sigma w_k = \lambda_k w_k$ with $\|w_k\|_2 = 1$.

### 4. Projection & 8-Bit Scalar Uniform Quantization
$$Z = X_c W^T \in \mathbb{R}^{N \times K}$$
For each component column $j \in [0, K-1]$:
$$q_{ij} = \text{round}\left( 255 \times \frac{z_{ij} - z_j^{\min}}{z_j^{\max} - z_j^{\min}} \right) \in [0, 255]$$
Only the 8-bit quantized values $q_{ij}$, scales $(z_j^{\min}, z_j^{\max})$, mean $\mu$, and components $W$ are stored in the binary bitstream (`.rpca`).

### 5. Inverse PCA Reconstruction
$$\hat{Z}_{ij} = z_j^{\min} + \left(\frac{q_{ij}}{255}\right) \times (z_j^{\max} - z_j^{\min})$$
$$\hat{X}_c = \hat{Z} W \in \mathbb{R}^{N \times D}, \quad \hat{X} = \hat{X}_c + \mu$$
Reconstructed patches are reassembled into the final output image canvas $\hat{I}$.

---

## 📊 Rate-Distortion Benchmark (256x256 RGB, $B=8$)

| $K$ Comp | Patch Dim ($D$) | Compressed Size | Compression Ratio | Bitstream Savings | MSE | PSNR (dB) | Encode Latency | Decode Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | 192 | 4.4 KB | **44.56x** | **97.8%** | 931.6 | 18.44 dB | 3.30 ms | 0.28 ms |
| **4** | 192 | 8.0 KB | **24.54x** | **95.9%** | 29.0 | **33.50 dB** | 3.96 ms | 0.20 ms |
| **8** | 192 | 15.2 KB | **12.92x** | **92.3%** | 0.10 | **57.99 dB** | 4.90 ms | 0.25 ms |
| **12** | 192 | 22.4 KB | **8.77x** | **88.6%** | 0.10 | **58.01 dB** | 6.12 ms | 0.28 ms |
| **16** | 192 | 29.6 KB | **6.64x** | **84.9%** | 0.10 | **58.06 dB** | 7.04 ms | 0.31 ms |

---

## ⚡ Quickstart & CLI Usage

### Build
```bash
make -C apps/rcre/projects/pca_compression all
```

### Run Automated End-to-End Demo
```bash
./bin/rcre-pca --demo
```

### Run Rate-Distortion Benchmark
```bash
./bin/rcre-pca --benchmark [input_image.bmp] -b 8
```

### Compress an Image
```bash
./bin/rcre-pca --compress photo.bmp photo.rpca -k 12 -b 8
```

### Decompress an Image
```bash
./bin/rcre-pca --decompress photo.rpca reconstructed.bmp
```

---

## 🌐 Interactive In-Browser PCA Web Playground

You can test this exact PCA image compression codec live directly in your browser without any installation:

* **Live Demo**: [https://erolrecep.github.io/rcre/playground.html#pca-lab-section](https://erolrecep.github.io/rcre/playground.html#pca-lab-section)
* **Local Web Server**:
  ```bash
  make up_web_app
  # Opens http://localhost:8080/playground.html
  ```

### Web Studio Features
1. **Live Camera & Synthetic Test Patterns**: Compress video feeds in real time or test custom synthetic colorbars and grid patterns.
2. **📸 Snapshot Capture with Filesystem Storage**: Take high-resolution photos and save them directly to disk using the **File System Access API** (PNG, JPEG, BMP).
3. **Interactive Split-Comparison Slider**: Drag the divider across the viewport to compare original vs. reconstructed images with pixel-level precision.
4. **Error Heatmap Visualization**: Inspect magnified ($\times 5$) reconstruction residuals mapped to an inferno pseudo-colormap.
5. **Learned Eigen-Bases Gallery**: Visualize the learned orthogonal spatial basis patches ($W_k$) and their individual variance contributions.
6. **Bitstream Filesystem Export**: Save reconstructed photos or export standalone `.rpca` binary files that can be loaded and decompressed directly with the CLI tool (`./bin/rcre-pca --decompress`).

