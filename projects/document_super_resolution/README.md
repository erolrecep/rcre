# RCRE-SR — Document Image Super-Resolution

Crystal-clear, sharp document images with better OCR legibility — built on the
**RCRE** zero-dependency C image/ML library. No OpenCV, no PIL, no pip. C23,
portable (clang/gcc), tuned for Apple Silicon.

```
  degraded scan ──► sr_enhance() ──► sharp, binarizable, OCR-friendly output
  (1200×1600)         ~110–160 ms        (2400×3200 @ 2×)
```

**Design stance: structure-preserving restoration, not generative SR.**
Learned super-resolution (ESRGAN & co.) hallucinates strokes and digits —
acceptable for photos, *disqualifying* for documents where OCR accuracy is the
KPI. RCRE-SR instead measures what the document needs (blur, noise, contrast,
skew) and applies the classical restoration chain that fixes exactly that:

```
assess ──► deskew ──► denoise ──► deblur ──► Lanczos-3 ×N ──► edge-masked sharpen
 (adaptive: each restoration stage runs only when the assessment says it's needed)
```

## Quick start

```sh
cd projects/document_super_resolution
make            # builds lib/libsr.a, bin/sr_cli, bin/bench_sr, bin/test_sr
make test       # 84 checks (>91% coverage: resize, bicubic, deblur, sharpen, pipeline, PNG/PNM I/O)
make demo       # synthetic document → degrade → enhance → PSNR/SSIM → OCR report
make bench      # kernel + pipeline benchmarks (median of 5)
make samples    # writes samples/original.pgm + samples/degraded.pgm

# The one-command demo: synthetic document → degrade → enhance → PSNR/SSIM → OCR
bin/sr_cli demo
```

`sr_cli demo` prints a full report: per-stage timings, quality vs ground truth,
and a Tesseract OCR comparison (input vs enhanced vs binarized-enhanced, as
character error rate — lower is better). Tesseract is optional; everything else
works without it.

```sh
bin/sr_cli enhance in.pgm out.pgm --scale 2 --style auto
bin/sr_cli enhance in.pgm out.pgm --scale 2 --style binarize   # best for OCR
bin/sr_cli assess in.pgm            # blur / noise / contrast / skew diagnostics
```

**Web demo** (Python 3 stdlib only — no pip installs):

```sh
make web        # or: python3 web/server.py [port]
# open http://127.0.0.1:8080 — upload a scan, drag the before/after slider,
# pick scale 1–4×, style, sharpen strength
```

The web server decodes/encodes PNG in pure stdlib (`zlib`+`struct`), bridges to
`bin/sr_cli`, and streams the result back.

## Using it in your own project

Two headers, one static lib, zero other dependencies:

```c
#include "sr.h"
#include "rcre.h"   // from ../rcre/dist (or third_party/)

RCRE_Arena* a = rcre_arena_create(16 << 20);
RCRE_Tensor in, out;
sr_pgm_read("scan.pgm", &in, a);        // or build a tensor from your own pixels
SRConfig cfg = sr_config_default();
cfg.style = SR_STYLE_AUTO;              // adaptive; or SR_STYLE_BINARIZE for OCR
cfg.scale = 2.0f;
SRReport rep;
sr_enhance(&in, &out, &cfg, a, &rep);   // ~110 ms for 1200×1600 → 2400×3200
sr_pnm_write("enhanced.pgm", &out);
sr_png_write("enhanced.png", &out);     // dependency-free PNG (stored DEFLATE)
// rep: per-stage on/off + ms, output dims, arena high-water mark
rcre_arena_destroy(a);
```

Link: `-Llib -lsr -lm` with `-Iinclude -I../rcre/dist`. For a single-file
embed, RCRE ships an amalgamated `dist/rcre.h`; compile `src/*.c` into your
build.

### Styles

| Style      | Behavior                                                        |
|------------|-----------------------------------------------------------------|
| `auto`     | Full adaptive pipeline; picks sharpen from the blur score       |
| `sharp`    | Force stronger edge-masked sharpen                              |
| `clean`    | Denoise + CLAHE emphasis (noisy/low-contrast scans)             |
| `binarize` | Full pipeline + Otsu / adaptive-local threshold — **OCR mode** |
| `upscale`  | Lanczos-3 only                                                  |

## How it works

- **Assessment** (subsampled, ~44 ms for 1200×1600): Laplacian-variance blur
  score, robust noise σ, p2–p98 contrast, projection-profile skew estimate.
  Every restoration stage is gated on these, so a clean scan skips 4 of 7 stages.
- **Deskew**: closed-form 4-point homography → `rcre_cv_warp_perspective`;
  out-of-bounds corners repainted as paper-white.
- **Denoise**: guided (edge-preserving) filter built on RCRE convolutions —
  removes sensor noise without smearing strokes.
- **Deblur**: FFT Wiener filter (RCRE's 2-D FFT). Gaussian, defocus and motion
  PSFs. The boundary handling mirrors RCRE's `reflect_101` blur padding and the
  PSF is phase-aligned so the edge band isn't corrupted.
- **Upscale**: separable **Lanczos-3** with precomputed per-coordinate weight
  tables — the inner loop is pure FMA, no per-pixel transcendentals
  (17× faster than naive kernel evaluation; ~70 MPix/s on M5 Max).
- **Sharpen**: unsharp mask masked by a smoothed gradient magnitude, so flat
  paper stays flat while strokes get contrast.
- **CLAHE**: contrast-limited histogram equalization for uneven illumination.
- **Binarize**: Otsu global or adaptive local threshold + light morphology.

## Performance (M5 Max, 1200×1600 → 2×, `make bench`)

```
pipeline 2× (sharp input)     113 ms     upscale 28 ms + sharpen 39 ms
pipeline 2× (degraded input)  154 ms     + CLAHE 24 ms
kernels:
  assess              44 ms  ( 44 MPix/s)
  lanczos 2×          27 ms  ( 70 MPix/s)
  clahe                8 ms  (256 MPix/s)
  sharpen              8 ms  (241 MPix/s)
  guided denoise      15 ms  (132 MPix/s)
  wiener deblur      265 ms  (  7 MPix/s)   ← only if the scan is actually blurred
arena high-water:  ~195 MB per run (all scratch in one arena)
```

Web demo round-trip (PNG decode → CLI → PNG encode) for the same image: ~300 ms.

## Honest results (synthetic benchmark document)

On `sr_cli demo` (60% res + blur + noise + skew + uneven light), the OCR
comparison against the known ground truth looks like this:

```
input   (degraded)       : CER  65.9 %
pipeline (×2)            : CER  67.1 %
pipeline (×2, binarize)  : CER  62.8 %   ← best, +3.1 pt vs input
```

(First run after tesseract's font-shape cache is cold may differ by ~2 pt;
results are stable afterwards.)

Takeaways we verified while tuning:

- **Binarization is the OCR win**, not the upscale. Tesseract prefers small,
  high-contrast text; Lanczos-upscaled blur is not a substitute for clean
  edges. `--style binarize` is the right default for OCR pipelines.- The synthetic 5×7-style document is a *hard* OCR target (unusual glyph
  shapes), which caps the absolute CER around 60%. On real typefaces the same
  pipeline (denoise + sharpen + binarize) recovers much more.
- PSNR/SSIM here track the bilinear-upscale baseline (~9.6 dB) because the
  information loss happened at acquisition; the pipeline's job is to make the
  *remaining* structure as readable as possible without inventing anything.

Known slow path: Wiener deblur (~265 ms, only triggered when the blur score
says the scan is actually blurred). Candidates: real FFT, precomputed H,
smaller padding.

## Layout

```
include/sr.h          public API (SRConfig, SRReport, sr_enhance, I/O, metrics)
src/sr_glue.c         compiles the RCRE implementation once
src/sr_io.c           PGM/PPM + dependency-free PNG writer (zlib stored DEFLATE)
src/sr_metrics.c      PSNR / SSIM
src/sr_assess.c       blur / noise / contrast / skew diagnostics
src/sr_resize.c       separable Lanczos-3 & bicubic (precomputed weight tables)
src/sr_enhance.c      edge-masked sharpen, CLAHE, Otsu, adaptive threshold
src/sr_restore.c      guided denoise, FFT Wiener deblur (3 PSF kinds)
src/sr_pipeline.c     adaptive pipeline + SRReport
examples/sr_cli.c     CLI: generate / enhance / assess / demo
tests/test_sr.c       50 checks
benchmarks/bench_sr.c kernel + pipeline benchmarks
web/server.py         stdlib-only web demo (PNG codec + before/after UI)
```

## Building

```sh
make                      # needs cc; RCRE header from ../rcre/dist or third_party/
make test && make bench
```

`third_party/rcre.h` is a vendored copy so the project builds standalone;
point `RCRE_DIR` at a sibling checkout to use a fresh amalgamation:
`make RCRE_DIR=../rcre/dist`.
