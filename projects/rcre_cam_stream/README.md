# Project 15: `rcre-cam-stream` (Real-Time Camera Stream & Video Studio)

A turnkey, high-performance camera streaming application and live video studio built on **RCRE** and pure ISO C23.

---

## 🌟 Architecture & Key Features

* **Lock-Free Thread Decoupling**: Background capture worker feeding a wait-free / lock-free single-producer single-consumer (SPSC) circular triple-buffer (`rcre_ringbuf.h` backed by `RCRE_Arena`). The render and processing pipeline always consumes the newest frame without blocking on camera exposure.
* **Multi-Platform Hardware Capture**:
  * **macOS**: Native AVFoundation ingestion with zero-copy CVPixelBuffer conversion.
  * **Linux**: Video4Linux2 (V4L2) kernel MMAP buffer streaming.
  * **Synthetic Dummy Mode**: Built-in animated test pattern generator for automated CI, Docker, and headless server environments.
* **Hardware-Accelerated OpenGL 3.3 Core Renderer**:
  * Real-time GPU fragment shaders for instant image filtering:
    * **Normal** (Clean Passthrough)
    * **Grayscale** (Luminance preserving)
    * **High Contrast** (Dynamic stretch)
    * **Sepia** (Warm tone matrix)
    * **Color Invert** (Negative image)
  * **Mirror Mode**: Instant horizontal selfie flipping (`M` key or toolbar button).
  * **Freeze Frame**: Pause and resume live video without dropping the active window.
* **Interactive Control Toolbar & Telemetry HUD**:
  * Floating bottom toolbar with hover animations and click interactions.
  * Real-time metrics HUD: Live FPS counter, frame latency in milliseconds, resolution, and pixel format.
  * Toast status popups for mode changes.
* **Full-Resolution Snapshot Codec**:
  * Instant screen capture to `snapshots/webcam-snapshot-YYYYMMDD-HHMMSS.bmp` with realistic camera shutter flash animation.

---

## 🚀 Quick Start

```bash
# Build the application
make build

# Launch the live camera studio (opens SDL3 / OpenGL window)
make run

# Run automated headless verification (synthetic camera, 60 frames)
make demo

# Run full AddressSanitizer and UndefinedBehaviorSanitizer verification
make san
```

---

## ⌨️ Interactive Keyboard Controls

| Key | Action |
| :--- | :--- |
| **`M`** | Toggle Mirror / Selfie mode (flips image horizontally) |
| **`Space`** or **`P`** | Pause / Resume live camera feed |
| **`C`** or **`Enter`** | Capture full-resolution photo to `snapshots/*.bmp` with shutter flash |
| **`Tab`** | Cycle through GPU shader filters (Normal $\to$ Grayscale $\to$ Contrast $\to$ Sepia $\to$ Invert) |
| **`S`** or **`F3`** | Toggle on-screen telemetry stats HUD (FPS, latency, resolution) |
| **`F`** | Toggle Fullscreen display mode |
| **`ESC`** or **`Q`** | Quit application |

---

## 🛠️ CLI Options

```
Usage: rcre_cam_stream [OPTIONS]

Options:
  --width <W>       Capture width in pixels (default: 1280)
  --height <H>      Capture height in pixels (default: 720)
  --fps <FPS>       Target framerate (default: 30)
  --dummy           Use synthetic animated test pattern (ideal for headless / CI)
  --headless        Run in benchmark mode without creating an SDL3 window
  --frames <N>      Exit automatically after N frames (default: 0 = unlimited)
  --device <N>      Camera device index (default: 0)
  -h, --help        Show this help message
```
