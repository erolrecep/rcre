/**
 * @file rcre_codec.h
 * @brief Zero-Dependency Image Codecs (PNG / JPEG / BMP / PPM) & Minimal ONNX Protobuf Runner
 *
 * Part of Recep's Code Run Everywhere (RCRE) v2.0.0.
 * Strict ISO C23 compliant, zero external dependencies (uses rcre_io_mmap_open / write_atomic).
 */

#ifndef RCRE_CODEC_H
#define RCRE_CODEC_H

#include "rcre.h"
#include "rcre_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Zero-Dependency Raster Image Codecs (PPM, BMP, PNG, Baseline JPEG)     */
/* ========================================================================= */

/** Outputs [H, W, C] RCRE_DTYPE_U8 tensor (C=1,3, or 4) in arena. */
RCRE_NODISCARD int rcre_codec_load_ppm(const char* filepath, RCRE_Arena* arena, RCRE_Tensor* out_img);
RCRE_NODISCARD int rcre_codec_save_ppm(const RCRE_Tensor* img, const char* filepath);

RCRE_NODISCARD int rcre_codec_load_bmp(const char* filepath, RCRE_Arena* arena, RCRE_Tensor* out_img);
RCRE_NODISCARD int rcre_codec_save_bmp(const RCRE_Tensor* img, const char* filepath);

RCRE_NODISCARD int rcre_codec_load_png_from_buffer(const void* buf, size_t size, RCRE_Arena* arena, RCRE_Tensor* out_img);
RCRE_NODISCARD int rcre_codec_load_png(const char* filepath, RCRE_Arena* arena, RCRE_Tensor* out_img);
RCRE_NODISCARD int rcre_codec_save_png(const RCRE_Tensor* img, const char* filepath);

RCRE_NODISCARD int rcre_codec_load_jpeg_from_buffer(const void* buf, size_t size, RCRE_Arena* arena, RCRE_Tensor* out_img);
RCRE_NODISCARD int rcre_codec_load_jpeg(const char* filepath, RCRE_Arena* arena, RCRE_Tensor* out_img);
RCRE_NODISCARD int rcre_codec_save_jpeg(const RCRE_Tensor* img, const char* filepath, int quality);

/* ========================================================================= */
/* 2. Minimal Zero-Dependency ONNX Protobuf Inference Subset                 */
/* ========================================================================= */

#define RCRE_ONNX_MAX_NODES   128
#define RCRE_ONNX_MAX_TENSORS 256

typedef enum {
    RCRE_ONNX_OP_UNKNOWN = 0,
    RCRE_ONNX_OP_RELU,
    RCRE_ONNX_OP_GEMM,
    RCRE_ONNX_OP_MATMUL,
    RCRE_ONNX_OP_ADD,
    RCRE_ONNX_OP_SOFTMAX,
    RCRE_ONNX_OP_CONV,
    RCRE_ONNX_OP_MAXPOOL,
    RCRE_ONNX_OP_GLOBAL_AVG_POOL
} RCRE_ONNXOpType;

typedef struct {
    char*  name;
    float* data;
    size_t dims[4];
    uint8_t ndim;
    size_t numel;
} RCRE_ONNXTensor;

typedef struct {
    RCRE_ONNXOpType op;
    char            inputs[4][64];
    size_t          n_inputs;
    char            output[64];
    int             stride;
    int             pad;
    int             kernel;
    int             trans_b;
} RCRE_ONNXNode;

typedef struct {
    char            input_name[64];
    char            output_name[64];
    RCRE_ONNXTensor initializers[RCRE_ONNX_MAX_TENSORS];
    size_t          n_init;
    RCRE_ONNXNode   nodes[RCRE_ONNX_MAX_NODES];
    size_t          n_nodes;
} RCRE_ONNXModel;

RCRE_NODISCARD int rcre_onnx_load_buffer(const void* buf, size_t size, RCRE_ONNXModel* out_model);
RCRE_NODISCARD int rcre_onnx_run(const RCRE_ONNXModel* model, const RCRE_Tensor* input,
                                 RCRE_Arena* arena, RCRE_Tensor* output);
void rcre_onnx_free(RCRE_ONNXModel* model);

/** Serializes a 2-layer ONNX protobuf linear/ReLU/Gemm graph into caller buffer for testing. */
RCRE_NODISCARD size_t rcre_onnx_build_mlp_protobuf(
    uint8_t* dst,
    size_t max_bytes,
    const float* w1, const float* b1, size_t in_dim, size_t hid_dim,
    const float* w2, const float* b2, size_t out_dim
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_CODEC_H */
