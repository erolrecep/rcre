# RCRE Galton Board (Quincunx) — High-Performance Flow & CLT Engine

An architectural, multi-runtime demonstration of the **Galton Board (Bean Machine / Quincunx)**, modeling the emergence of the **Gaussian Normal Distribution** from discrete **Bernoulli Trials** (Central Limit Theorem & De Moivre–Laplace Theorem).

Implemented across two cohesive tiers in this repository:
1. **ISO C23 Engine & WebAssembly-Ready Visualizer**: Ultra-high-throughput Monte-Carlo engine in strict C23 with live ANSI terminal visualization, zero-dependency HTML5 Canvas2D / Web Audio visualizer, and hardware-accelerated bit-level intrinsics (**750+ Million drops/sec**).
2. **Python 3.15 FastAPI Service**: Modern asynchronous service built on Python 3.15.2, featuring REST APIs, WebSocket real-time streaming, strict Pydantic v2 schemas, and full statistical moment verification. Also mirrored as a standalone package under `_python_/galtonboard_fastapi/`.

---

## Mathematical & Physical Foundations

### 1. The Galton Board Lattice
A triangular pin lattice of $N$ rows indexed $r \in \{0, \dots, N-1\}$. At each row $r$, there are $r + 1$ pegs spaced by $(d_x, d_y)$:

$$x_{r, c} = x_{\text{center}} + \left(c - \frac{r}{2}\right) d_x, \quad y_{r, c} = y_{\text{top}} + r \cdot d_y$$

A marble enters the top funnel and undergoes $N$ independent collisions. At each peg, it deflects right with probability $p$ and left with probability $1 - p$.

### 2. Discrete Binomial Distribution
The cumulative horizontal displacement corresponds to the total number of right bounces $K = \sum_{i=1}^N B_i$, where $B_i \sim \text{Bernoulli}(p)$:

$$P(K = k) = \binom{N}{k} p^k (1 - p)^{N - k}, \quad k \in \{0, 1, \dots, N\}$$

To prevent numerical overflow for large $N$, combinations are computed in log-space using standard `lgamma`:

$$\ln \binom{N}{k} = \ln \Gamma(N + 1) - \ln \Gamma(k + 1) - \ln \Gamma(N - k + 1)$$

$$\ln P(K = k) = \ln \binom{N}{k} + k \ln p + (N - k) \ln(1 - p)$$

### 3. De Moivre–Laplace Theorem & Central Limit Theorem
As $N \to \infty$, the discrete Binomial PMF converges asymptotically to the continuous Gaussian Normal density $\mathcal{N}(\mu, \sigma^2)$:

$$\mu = N p, \quad \sigma^2 = N p (1 - p)$$

$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2\right)$$

### 4. Statistical Parity & Moments
The engine continuously computes sample moments and conducts a **Pearson Chi-Square Goodness-of-Fit Test**:

$$\chi^2 = \sum_{k=0}^N \frac{(O_k - E_k)^2}{E_k}, \quad E_k = M \cdot P(K = k)$$

- **Sample Mean**: $\bar{x} = \frac{1}{M}\sum k \cdot O_k$
- **Sample Variance**: $s^2 = \frac{1}{M - 1}\sum (k - \bar{x})^2 O_k$
- **Sample Skewness**: $\gamma_1 = \frac{1}{M s^3}\sum (k - \bar{x})^3 O_k \to \frac{1 - 2p}{\sqrt{N p (1 - p)}}$
- **Excess Kurtosis**: $\gamma_2 = \frac{1}{M s^4}\sum (k - \bar{x})^4 O_k - 3.0 \to \frac{1 - 6p(1 - p)}{N p (1 - p)}$

---

## Directory Structure

```
apps/rcre-galtonboard/
├── Makefile                     # Root build system (C23, sanitizers, benchmarks, python)
├── README.md                    # System architecture & mathematical derivation
├── include/
│   ├── rcre_galton.h            # Clean C23 public API & data structures
│   └── rcre.h                   # RCRE unified single-header framework
├── src/
│   ├── galton_core.c            # xoshiro256** PRNG, SIMD/popcount kernels, CLT stats, export
│   ├── galton_ansi.c            # Live ANSI terminal renderer & ASCII histogram
│   └── main.c                   # CLI runner & 20M benchmark harness
├── tests/
│   └── test_galton.c            # C23 test runner (ASan + UBSan enabled)
├── web/
│   ├── index.html               # Interactive Canvas2D + Web Audio + CLT visualizer
│   └── galton_wasm.c            # WebAssembly C export interfaces
└── python/
    ├── pyproject.toml           # Python 3.15 uv project manifest
    ├── Makefile                 # Python test & run targets
    ├── README.md                # Python FastAPI documentation
    ├── app/
    │   ├── main.py              # FastAPI app, REST routes & WebSockets
    │   ├── engine.py            # Python 3.15 simulation engine & moments
    │   ├── models.py            # Pydantic v2 schemas
    │   └── streaming.py         # WebSocket broadcast controller
    └── tests/
        ├── test_api.py          # FastAPI REST & WebSocket integration tests
        └── test_engine.py       # CLT numerical verification tests

_python_/galtonboard_fastapi/    # Dedicated standalone mirror for monorepo Python ecosystem
```

---

## Quick Start & Verification

### 1. Build and Run the C23 Engine
```bash
cd apps/rcre-galtonboard

# Compile release binary and shared library
make

# Run standard simulation (1,000,000 balls, 16 rows)
make run

# Run live ANSI terminal visualization
make animate

# Run ultra-high-throughput benchmark (20 Million balls)
make benchmark
```

#### Benchmark Telemetry
On Apple Silicon M-series (Single Core):
```
Executing Monte-Carlo Bernoulli trials...
[Simulation Completed in 0.0265 seconds]
  Throughput          : 754.15 Million drops/sec (1.33 ns/drop)

--- STATISTICAL VERIFICATION REPORT ---
  Total Balls (M)     : 20000000
  Pin Levels (N)      : 16  (Bins: 17)
  Right Bias (p)      : 0.5000
  Empirical Mean (x̄)  : 7.999663  (Target: 8.000000, Error: -0.000337)
  Empirical Variance  : 3.999587  (Target: 4.000000, Error: -0.000413)
  Empirical Std Dev   : 1.999897  (Target: 2.000000, Error: -0.000103)
  Sample Skewness     : +0.000029  (Target: +0.000000)
  Excess Kurtosis     : -0.124764  (Target: -0.125000)
  Chi-Square Test (χ²): 13.8712  (dof: 16, p-value: 0.6083)
  Null Hypothesis H₀  : ACCEPTED
```

### 2. Run C Sanitizer Test Suite (ASan & UBSan)
```bash
make test
```
Verifies PRNG determinism, exact Binomial PMF values, Central Limit Theorem convergence, skewed distributions ($p=0.25$), and file exports under zero memory leaks or undefined behaviors.

### 3. Open Interactive Web Visualizer
Open `web/index.html` directly in any web browser, or serve it locally:
```bash
# Using Python built-in HTTP server:
python3 -m http.server 8080 --directory web/
```
Features:
- **60 FPS Canvas Physics**: Elastic pin collisions with restitution $e$, gravity, sub-pixel motion.
- **Web Audio Synthesis**: Synthesized pentatonic chime tones on peg hits scaling with row depth.
- **Dual Distribution Curves**: Live Amber dotted line (Binomial PMF) + Cyan glowing curve (Gaussian PDF).
- **Interactive Controls**: Sliders for rows ($6$ to $24$), bias ($0.10$ to $0.90$), drop rate, and bounciness.
- **One-Click Export**: Instant CSV and JSON export.

### 4. Run Python 3.15 FastAPI Service
The Python 3.15 service is powered by `uv` and includes full WebSocket streaming:

```bash
# Launch FastAPI server on port 8000
make python-run

# Run automated Python pytest suite (9 tests)
make python-test
```

#### Key Endpoints:
- `GET /` — Serves the interactive Web visualizer.
- `GET /health` — Environment healthcheck (reports Python `3.15.2`).
- `POST /api/v1/simulate` — High-speed Monte-Carlo batch run with JSON moments & probabilities.
- `GET /api/v1/stats` — Current accumulated board statistics.
- `POST /api/v1/reset` — Reset counters.
- `GET /api/v1/export/csv` — Export CSV.
- `GET /api/v1/export/json` — Export JSON.
- `WS /ws/stream` — Real-time bidirectional streaming channel.

---

## Standalone Python Mirror
In addition to the main project directory, the Python implementation is also preserved as a standalone project at:
```
_python_/galtonboard_fastapi/
```
To run tests directly in the Python folder:
```bash
cd _python_/galtonboard_fastapi
uv run --extra dev --python /opt/homebrew/bin/python3.15 pytest tests/ -v
```
