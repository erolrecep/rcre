/**
 * @file wavelet_codec.h
 * @brief RCRE Multi-Resolution Integer Wavelet Image Codec (LeGall 5/3)
 *
 * Implements a reversible, integer-to-integer 2D Discrete Wavelet Transform (DWT)
 * lifting scheme with dead-zone quantization and run-length entropy coding.
 *
 * Features:
 *  - 100% reversible bit-exact lossless or lossy multi-scale compression
 *  - Eliminates JPEG block boundary artifacts
 *  - Strictly arena-resident memory management (0 heap allocations in loops)
 *  - Portable ISO C23 and ARM NEON / AVX2 acceleration
 *
 * Part of Recep's Code Run Everywhere (RCRE).
 */

#ifndef RCRE_WAVELET_CODEC_H
#define RCRE_WAVELET_CODEC_H

#include "rcre.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RCRE_RDWT_MAGIC "RDWT"
#define RCRE_RDWT_VERSION 1

/**
 * @brief Standalone RDWT File Header (32 bytes packed)
 */
typedef struct [[gnu::packed]] {
    char     magic[4];       /* "RDWT" (0x52 0x44 0x57 0x54)               */
    uint16_t version;        /* Format version (1)                         */
    uint16_t width;          /* Image width (pixels)                       */
    uint16_t height;         /* Image height (pixels)                      */
    uint8_t  channels;       /* Channel count (1 for Gray, 3 for RGB)      */
    uint8_t  levels;         /* DWT decomposition levels (typically 3)     */
    uint16_t quant_step;     /* Dead-zone quantization step (1 = lossless) */
    uint32_t payload_bytes;  /* Compressed payload size in bytes           */
    uint8_t  reserved[16];   /* Reserved alignment padding                 */
} RCRE_WaveletHeader;

/**
 * @brief Compression Telemetry and Quality Metrics
 */
typedef struct {
    size_t original_bytes;
    size_t compressed_bytes;
    float  compression_ratio;
    float  bits_per_pixel;
    float  mse;
    float  psnr_db;
    double encode_time_ms;
    double decode_time_ms;
} RCRE_WaveletMetrics;

/* ========================================================================= */
/* Core 2D Integer Wavelet Lifting Transform Primitives                      */
/* ========================================================================= */

/**
 * @brief Performs in-place forward 2D LeGall 5/3 integer DWT on a single channel.
 * @param buffer Contiguous 2D signed 32-bit integer array [height * width].
 * @param width Channel width.
 * @param height Channel height.
 * @param levels Number of multi-resolution decomposition pyramid levels.
 * @param arena Transient memory arena for scratch line buffers.
 * @return RCRE_OK on success.
 */
RCRE_NODISCARD int rcre_wavelet_forward_2d(
    int32_t* buffer,
    size_t width,
    size_t height,
    size_t levels,
    RCRE_Arena* arena
);

/**
 * @brief Performs in-place inverse 2D LeGall 5/3 integer DWT on a single channel.
 * @param buffer Contiguous 2D signed 32-bit integer array [height * width].
 * @param width Channel width.
 * @param height Channel height.
 * @param levels Number of multi-resolution decomposition pyramid levels.
 * @param arena Transient memory arena for scratch line buffers.
 * @return RCRE_OK on success.
 */
RCRE_NODISCARD int rcre_wavelet_inverse_2d(
    int32_t* buffer,
    size_t width,
    size_t height,
    size_t levels,
    RCRE_Arena* arena
);

/* ========================================================================= */
/* End-to-End Image Encoding & Decoding                                      */
/* ========================================================================= */

/**
 * @brief Encodes an RGB or Grayscale image tensor into a compressed RDWT bitstream.
 * @param image Input image tensor [H, W, C] (U8 or F32).
 * @param levels Number of DWT pyramid decomposition levels (e.g., 3).
 * @param quant_step Quantization step (1 = lossless, 2..64 = lossy).
 * @param arena Memory arena to allocate compressed buffer.
 * @param out_bytes Pointer to receive compressed byte count.
 * @return Pointer to arena-allocated compressed bitstream, or NULL on error.
 */
RCRE_NODISCARD uint8_t* rcre_wavelet_encode(
    const RCRE_Tensor* image,
    size_t levels,
    uint16_t quant_step,
    RCRE_Arena* arena,
    size_t* out_bytes
);

/**
 * @brief Decodes a compressed RDWT bitstream into a reconstructed image tensor.
 * @param bitstream Compressed buffer starting with RCRE_WaveletHeader.
 * @param bitstream_size Size of bitstream in bytes.
 * @param arena Memory arena for decoded tensor.
 * @param out_image Pointer to receive reconstructed image tensor [H, W, C] (U8).
 * @return RCRE_OK on success.
 */
RCRE_NODISCARD int rcre_wavelet_decode(
    const uint8_t* bitstream,
    size_t bitstream_size,
    RCRE_Arena* arena,
    RCRE_Tensor* out_image
);

/**
 * @brief Computes Peak Signal-to-Noise Ratio (PSNR) between original and reconstructed images.
 */
RCRE_NODISCARD float rcre_wavelet_compute_psnr(
    const RCRE_Tensor* original,
    const RCRE_Tensor* reconstructed,
    float* out_mse
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_WAVELET_CODEC_H */

