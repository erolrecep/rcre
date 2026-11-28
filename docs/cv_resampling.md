# RCRE Continuous Computer Vision Resampling & GM-Exact Filters

**Module**: `src/cv/resize_gm.c`, `src/cv/geom.c`  
**Public API**: `include/rcre.h` (`rcre_cv_resize_nn`, `rcre_cv_resize_bilinear`, `rcre_cv_resize_gm`)  
**Standard**: ISO C23 (`-std=c2x` / `-std=c23`), Zero Heap Allocations in Loops  

---

## 1. Overview & Mathematical Model

Image resizing in RCRE is continuous and subpixel-accurate. Unlike naive integer-truncated implementations that suffer from boundary drift and aspect distortion, RCRE aligns continuous pixel centers with subpixel coordinates:

$$\text{center} = \left(x + 0.5\right) \cdot \frac{L_{\text{src}}}{L_{\text{dst}}}$$

$$\text{invf} = \frac{L_{\text{src}}}{L_{\text{dst}}}, \qquad s_{\text{pre}} = \max\left(\text{invf}, 1.0\right)$$

When downscaling ($L_{\text{dst}} < L_{\text{src}}$), the filter support window is dynamically expanded by $s_{\text{pre}} = 1/f$ to avoid aliasing and preserve high-frequency energy:

$$\text{support} = s_{\text{pre}} \cdot \text{filter\_support}$$

$$\text{scale} = \begin{cases} 1.0 & \text{if } \text{support} \le 0.5 \\ \frac{1.0}{s_{\text{pre}}} & \text{otherwise} \end{cases}$$

---

## 2. The 16 Continuous Windowed Filter Kernels

RCRE implements GraphicsMagick 1.3.49's bit-for-bit filter window formulations across 16 continuous kernels:

| Interp Enum | Filter Name | Support | Definition / Window Formulation |
|---|---|---|---|
| `RCRE_CV_INTERP_GM_POINT` | Point | 0.0 | Nearest box tap |
| `RCRE_CV_INTERP_GM_BOX` | Box | 0.5 | Rectangular average: $1.0$ if $|x| < 0.5$, else $0.0$ |
| `RCRE_CV_INTERP_GM_TRIANGLE`| Triangle / Tent | 1.0 | Piecewise linear: $1.0 - |x|$ for $|x| < 1.0$ |
| `RCRE_CV_INTERP_GM_HERMITE` | Hermite | 1.0 | Cubic smoothstep: $(2|x| - 3)x^2 + 1$ for $|x| < 1.0$ |
| `RCRE_CV_INTERP_GM_HANNING` | Hanning | 1.0 | Cosine window: $0.5 + 0.5 \cos(\pi x)$ |
| `RCRE_CV_INTERP_GM_HAMMING` | Hamming | 1.0 | Raised cosine: $0.54 + 0.46 \cos(\pi x)$ |
| `RCRE_CV_INTERP_GM_BLACKMAN` | Blackman | 1.0 | Exact Blackman: $0.42 + 0.5\cos(\pi x) + 0.08\cos(2\pi x)$ |
| `RCRE_CV_INTERP_GM_GAUSSIAN` | Gaussian | 1.25 | Exponential: $\exp(-2x^2) \sqrt{2/\pi}$ |
| `RCRE_CV_INTERP_GM_QUADRATIC`| Quadratic | 1.5 | Bell B-spline: 3-piece quadratic |
| `RCRE_CV_INTERP_GM_CUBIC` | Cubic | 2.0 | B-spline: 4-piece cubic |
| `RCRE_CV_INTERP_GM_CATROM` | Catmull-Rom | 2.0 | Interpolating cubic spline |
| `RCRE_CV_INTERP_GM_MITCHELL` | Mitchell-Netravali| 2.0 | $B=1/3, C=1/3$ balanced blur/ringing cubic |
| `RCRE_CV_INTERP_GM_LANCZOS` | Lanczos3 | 3.0 | Sinc-windowed sinc: $\text{sinc}(x) \cdot \text{sinc}(x/3)$ |
| `RCRE_CV_INTERP_GM_BESSEL` | Bessel | 3.2383 | Blackman-windowed $J_1(\pi x)/(2x)$ via Cephes polynomial |
| `RCRE_CV_INTERP_GM_SINC` | Sinc | 4.0 | Blackman-windowed sinc: $\text{Blackman}(x/4) \cdot \text{sinc}(\pi x)$ |

Tap weights for each output coordinate $x$ are normalized across valid input samples $[start, stop)$:

$$w_i = \frac{f(\text{scale} \cdot (start + i + 0.5 - \text{center}))}{\sum_{j} w_j}$$

---

## 3. Separable 2D Pass Ordering Optimization

Because 2D convolution with continuous separable kernels factors into 1D horizontal and 1D vertical passes ($K(x, y) = K(x) \cdot K(y)$), the order of passes determines the number of intermediate pixels evaluated. RCRE automatically executes the pass ordering that minimizes work:

$$\text{Pass Order Heuristic}: \quad \text{Horizontal First if } \quad W_{\text{out}} \cdot (H_{\text{in}} + H_{\text{out}}) > H_{\text{out}} \cdot (W_{\text{in}} + W_{\text{out}})$$

- **Horizontal Pass**: Filters along contiguous memory rows (maximizing hardware cacheline reuse and streaming prefetches).
- **Vertical Pass**: Filters across column strides with pre-gathered pointer tables.

---

## 4. Memory Discipline & Arena Integration

To satisfy RCRE's zero-heap discipline, `rcre_cv_resize_gm` accepts an optional `RCRE_Arena*` pointer:
- When `arena != NULL`, all temporary tap tables and intermediate pass buffers are linear-allocated from the arena and automatically reclaimed at the caller's scope with zero fragmentation and zero OS syscalls.
- When `arena == NULL`, the function falls back to standard `malloc` and `free` with guaranteed zero leaks under AddressSanitizer.

---

## 5. Tensor Format & Channel Support

`rcre_cv_resize_gm` natively operates on `RCRE_Tensor*` with:
- **DType**: `RCRE_DTYPE_F32`
- **Ranks**:
  - 2D: `[H, W]` (single channel / grayscale)
  - 3D Planar (`RCRE_LAYOUT_CHW`): `[C, H, W]` (ideal for Deep Learning / CNNs / ViTs)
  - 3D Interleaved (`RCRE_LAYOUT_HWC`): `[H, W, C]` (ideal for OpenCV / Camera pipelines)

Each channel is transformed independently, ensuring zero cross-channel color bleeding.
