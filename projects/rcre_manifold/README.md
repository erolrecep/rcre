# RCRE Manifold (`rcre-manifold`)
## High-Dimensional Manifold Learning, Dimensionality Reduction & Untangling Engine

A pure ISO C23 zero-dependency dimensionality reduction library providing **UMAP**, **Barnes-Hut t-SNE**, **PaCMAP** (Wang et al. 2021), and orthogonal **PCA**.

Designed for bare-metal systems, browser-based WebAssembly simulations, and real-time visualization of high-dimensional datasets.

---

## 🚀 Key Mathematical Capabilities

### 1. UMAP (Uniform Manifold Approximation and Projection)
- **Fuzzy Simplicial Set Topology**: Local Riemannian metric approximation via smooth $\sigma_i$ / $\rho_i$ bisection:
  $$\sum_{j=1}^k \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right) = \log_2(k)$$
- **Symmetric Graph Union**: $p_{ij} = p_{i|j} + p_{j|i} - p_{i|j} p_{j|i}$ assembled into a compact Compressed Sparse Row (CSR) structure.
- **Cross-Entropy SGD with Negative Sampling**:
  - Low-D Cauchy/exponential curve: $q_{ij} = \frac{1}{1 + a d_{ij}^{2b}}$.
  - Attractive gradient pulling connected neighbors together.
  - Repulsive gradient pushing randomly sampled negative vertices apart.

### 2. Barnes-Hut t-SNE
- **Perplexity Binary Search**: Shannon entropy calibration $H(P_i) = \log_2(\text{Perp})$ for adaptive high-dimensional Gaussian variances $\sigma_i$.
- **$\mathcal{O}(N \log N)$ 2D Quadtree**: Multipole spatial tree evaluation with opening angle threshold $\theta = 0.5$ for low-dimensional Cauchy repulsive forces.
- **Optimization Schedules**: Early exaggeration ($\times 12$ for first 200 epochs) to untangle global modes, followed by heavy-ball momentum updates ($\beta = 0.5 \to 0.8$).

### 3. PaCMAP (Pairwise Controlled Manifold Approximation)
- **3-Tier Sampling Strategy**:
  1. *Neighbor Pairs* ($k=10$): Local neighborhood preservation.
  2. *Mid-Near Pairs* (sampled from rank $k \dots 6k$): Intermediate geometric structure.
  3. *Further Pairs* (sampled uniformly): Global cluster topology preservation.
- **3-Stage Dynamic Annealing**:
  - Stage 1 (Epochs 0–100): High mid-near attraction ($w_{\text{MN}} = 2.0$) establishing global layout.
  - Stage 2 (Epochs 100–200): Linear decay of mid-near forces.
  - Stage 3 (Epochs 200–350): Local neighborhood refinement.

### 4. Interactive Step-by-Step API
- Step-wise execution via `rcre_manifold_step()` allowing 60 FPS animation frames in browser WebAssembly and terminal loops.

---

## 📁 Directory Architecture

```
rcre_manifold/
├── include/
│   └── rcre_manifold.h       # Public C23 API declarations
├── src/
│   ├── manifold.c            # UMAP, Barnes-Hut t-SNE, PaCMAP, PCA engines
│   └── manifold_internal.h   # Internal CSR graphs, Quadtree, and PRNG
├── examples/
│   ├── manifold_demo.c       # 10-D Gaussian mixture ASCII terminal demo
│   └── manifold_animate.c    # Interactive timeline animation & teaching visualizer
├── tests/
│   └── test_manifold.c       # Mathematical invariant and cluster separation tests
├── web/
│   └── index.html            # Interactive Canvas2D / WebGL untangling laboratory
└── Makefile                  # Build and test orchestration
```

---

## ⚡ Quick Start & Verification

```bash
# Run unit tests and mathematical invariant verification:
make test

# Run AddressSanitizer and UndefinedBehaviorSanitizer:
make san

# Run the 10-D ASCII scatter visualizer demo:
make demo
```

---

## 🎬 Terminal Timeline Animation & Teaching Laboratory

For lectures, algorithmic demonstrations, and terminal screencasts, `manifold_animate` provides a live 60 FPS animated HUD rendering the evolution of cluster separation, learning rate annealing, and negative sampling repulsion across epochs.

```bash
# 1. Animate UMAP untangling 10D Gaussian mixture:
make animate

# 2. Animate 3D Swiss Roll unrolling into a 2D plane:
make animate-swiss

# 3. Animate 3D Trefoil Knot topological loop untangling:
make animate-trefoil

# 4. Animate PaCMAP 3-Stage Dynamic Annealing:
make animate-pacmap

# 5. Interactive lecture stepping mode (advance on Enter keypress):
./build/manifold_animate --step --umap
```

### Command Line Options
| Option | Description |
| :--- | :--- |
| `--umap` | Run UMAP algorithm (*default*) |
| `--pacmap` | Run PaCMAP 3-tier algorithm |
| `--tsne` | Run Barnes-Hut t-SNE algorithm |
| `--pca` | Run Orthogonal PCA baseline |
| `--blobs` | 10D Gaussian Blobs (*default*) |
| `--swiss` | 3D Swiss Roll unrolling |
| `--trefoil` | 3D Trefoil Knot untangling |
| `--step` | Interactive step mode (press Enter to advance epoch-by-epoch) |
| `--fast` | Fast animation playback |
| `--slow` | Slower speed for in-depth classroom commentary |

---

## 🌐 In-Browser Untangling Lab

Open `web/index.html` in any modern web browser to interactively untangle 3D Swiss rolls, Trefoil knots, 10D Gaussians, and concentric rings in real time.
