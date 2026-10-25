/**
 * @file rcre.h
 * @brief RCRE: Recep's Code Run Everywhere - Unified C23 CV & ML Engine
 * @version 1.0.0
 * 
 * Strict ISO C23 compliant, zero external dependencies.
 */

#ifndef RCRE_H
#define RCRE_H

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Version & Compatibility Macros                                            */
/* ========================================================================= */

#define RCRE_VERSION_MAJOR 1
#define RCRE_VERSION_MINOR 0
#define RCRE_VERSION_PATCH 0

#ifndef RCRE_NODISCARD
  #if defined(__has_c_attribute) && __has_c_attribute(nodiscard)
    #define RCRE_NODISCARD [[nodiscard]]
  #elif defined(__GNUC__) || defined(__clang__)
    #define RCRE_NODISCARD __attribute__((warn_unused_result))
  #else
    #define RCRE_NODISCARD
  #endif
#endif

#ifndef constexpr
  #if !defined(__cpp_constexpr) && (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 202311L)
    #define constexpr static const
  #endif
#endif

/* ========================================================================= */
/* Debug Hooks                                                               */
/* ========================================================================= */

/* Define RCRE_DEBUG before including rcre.h (or compile with -DDEBUG) to
 * enable loud diagnostics: out-of-range label indices in loss functions are
 * reported to stderr (and still clamped) instead of failing silently. */
#ifndef RCRE_DEBUG
  #if defined(DEBUG) || defined(_DEBUG)
    #define RCRE_DEBUG 1
  #endif
#endif

/* ========================================================================= */
/* Status & Error Codes                                                      */
/* ========================================================================= */

typedef enum {
    RCRE_OK                  =  0,
    RCRE_ERR_INVALID_ARG     = -1,
    RCRE_ERR_OUT_OF_MEMORY   = -2,
    RCRE_ERR_SHAPE_MISMATCH  = -3,
    RCRE_ERR_UNSUPPORTED_OP  = -4,
    RCRE_ERR_TIMEOUT         = -5,
    RCRE_ERR_IO              = -6
} RCRE_Status;

/* ========================================================================= */
/* Data Types & Layouts                                                      */
/* ========================================================================= */

typedef enum {
    RCRE_DTYPE_U8  = 0, /* 8-bit unsigned integer (image pixels, quantized bins) */
    RCRE_DTYPE_I32 = 1, /* 32-bit signed integer (class labels, indices)          */
    RCRE_DTYPE_F32 = 3  /* 32-bit single precision float                           */
} RCRE_DType;

typedef enum {
    RCRE_LAYOUT_UNKNOWN = 0,
    RCRE_LAYOUT_HWC     = 1, /* Interleaved (e.g., RGBRGB... used in OpenCV/Cameras) */
    RCRE_LAYOUT_CHW     = 2, /* Planar (e.g., RRR...GGG...BBB... used in NNs/HPC)   */
    RCRE_LAYOUT_TABULAR = 3  /* 2D Matrix [Samples, Features]                        */
} RCRE_Layout;

/* ========================================================================= */
/* Linear Arena Allocator (MEM-REQ-01 to MEM-REQ-04)                         */
/* ========================================================================= */

typedef struct RCRE_Arena {
    uint8_t* buffer;
    size_t   capacity;
    size_t   offset;
} RCRE_Arena;

typedef struct {
    RCRE_Arena* arena;
    size_t      saved_offset;
} RCRE_ArenaTemp;

typedef struct RCRE_GraphCtx {
    RCRE_Arena* persistent_arena; /* Model weights, persistent states */
    RCRE_Arena* tape_arena;       /* Ephemeral forward/backward computational graph tape */
} RCRE_GraphCtx;

/**
 * @brief Creates a contiguous linear memory arena.
 * @param capacity Maximum allocation capacity in bytes.
 * @return Pointer to newly initialized RCRE_Arena, or NULL on failure.
 */
RCRE_NODISCARD RCRE_Arena* rcre_arena_create(size_t capacity);

/**
 * @brief Allocates an aligned buffer from the linear scratchpad arena.
 * @param arena Active scratchpad arena.
 * @param size Number of bytes requested.
 * @param align Alignment boundary (must be power of two, defaults to 64).
 * @return Aligned memory pointer, or NULL if arena capacity exceeded.
 *
 * Note: the returned pointer is aligned UP to `align`, so the allocation
 * consumes `size + padding` (padding < align). Requesting exactly the free
 * remainder can therefore fail even though `size <= free_capacity`.
 */
RCRE_NODISCARD void* rcre_arena_alloc(RCRE_Arena* arena, size_t size, size_t align);

/**
 * @brief Resets the arena offset back to zero in O(1) time without freeing OS memory.
 */
void rcre_arena_reset(RCRE_Arena* arena);

/**
 * @brief Releases the arena memory buffer back to the operating system.
 */
void rcre_arena_destroy(RCRE_Arena* arena);

/**
 * @brief Saves current arena allocation offset for scoped scratchpad rewinding.
 */
RCRE_NODISCARD RCRE_ArenaTemp rcre_arena_temp_begin(RCRE_Arena* arena);

/**
 * @brief Restores arena offset back to the checkpoint saved in temp.
 */
void rcre_arena_temp_end(RCRE_ArenaTemp temp);

/**
 * @brief Creates a dual-arena execution context (persistent weights + ephemeral tape).
 */
RCRE_NODISCARD RCRE_GraphCtx* rcre_graph_ctx_create(size_t persistent_cap, size_t tape_cap);

/**
 * @brief Resets the ephemeral tape arena back to zero offset.
 */
void rcre_graph_ctx_reset_tape(RCRE_GraphCtx* ctx);

/**
 * @brief Destroys both arenas and frees the graph context.
 */
void rcre_graph_ctx_destroy(RCRE_GraphCtx* ctx);

/* ========================================================================= */
/* Tensor Abstraction & Zero-Copy Views (TNS-REQ-01 to TNS-REQ-05)           */
/* ========================================================================= */

typedef struct RCRE_Tensor {
    void*         data;            /* Pointer to element (0, 0, ...)               */
    float*        grad;            /* Backprop buffer (training); NULL otherwise   */
    size_t        shape[4];        /* Dimensions (rank 1 to 4)                     */
    ptrdiff_t     strides[4];      /* Step in elements per dimension               */
    size_t        numel;           /* Cached element count (set at construction;   */
                                   /* shape/strides are immutable afterwards)      */
    uint8_t       ndim;            /* Rank (1 to 4)                                */
    RCRE_DType    dtype;
    RCRE_Layout   layout;
    bool          is_view;         /* true: data owned elsewhere (slice/permute/…) */
    const char*   label;           /* Optional name (visualization/debug); NULL ok */
} RCRE_Tensor;

/**
 * @brief Typed accessor for F32 tensor data (single source of the float cast
 *        used by the autograd/nn training layer and user training code).
 */
RCRE_NODISCARD static inline float* rcre_tensor_f32_data(const RCRE_Tensor* t) {
    return (float*)t->data;
}

/**
 * @brief Allocates a new contiguous tensor inside caller-supplied arena.
 */
RCRE_NODISCARD RCRE_Tensor rcre_tensor_create(const size_t* shape, uint8_t ndim, 
                                             RCRE_DType dtype, RCRE_Layout layout, 
                                             RCRE_Arena* arena);

/**
 * @brief Wraps an existing external memory buffer into an RCRE_Tensor view.
 */
RCRE_NODISCARD RCRE_Tensor rcre_tensor_from_buffer(void* data, const size_t* shape, 
                                                 uint8_t ndim, RCRE_DType dtype, 
                                                 RCRE_Layout layout);

/**
 * @brief O(1) slicing view operation: updates offsets and shape without copying.
 */
RCRE_NODISCARD RCRE_Tensor rcre_tensor_slice(const RCRE_Tensor* src, 
                                            const size_t* starts, 
                                            const size_t* ends);

/**
 * @brief O(1) reshape view operation (requires contiguous source tensor).
 */
RCRE_NODISCARD RCRE_Tensor rcre_tensor_reshape(const RCRE_Tensor* src, 
                                              const size_t* new_shape, 
                                              uint8_t new_ndim);

/**
 * @brief O(1) axis permutation view operation.
 */
RCRE_NODISCARD RCRE_Tensor rcre_tensor_permute(const RCRE_Tensor* src, 
                                              const uint8_t* order);

/**
 * @brief O(1) two-axis transpose view operation.
 */
RCRE_NODISCARD RCRE_Tensor rcre_tensor_transpose(const RCRE_Tensor* src, 
                                                uint8_t dim0, uint8_t dim1);

/**
 * @brief Evaluates whether tensor memory is row-major contiguous.
 */
RCRE_NODISCARD bool rcre_tensor_is_contiguous(const RCRE_Tensor* t);

/**
 * @brief Computes total number of logical elements in the tensor.
 */
RCRE_NODISCARD size_t rcre_tensor_numel(const RCRE_Tensor* t);

/**
 * @brief Computes total size in bytes for contiguous representation.
 */
RCRE_NODISCARD size_t rcre_tensor_nbytes(const RCRE_Tensor* t);

/**
 * @brief Returns byte size of a given RCRE_DType.
 */
RCRE_NODISCARD size_t rcre_dtype_size(RCRE_DType dtype);

/* ========================================================================= */
/* Input Recognition & Adaptation Pipeline (ADP-REQ-01 to ADP-REQ-03)        */
/* ========================================================================= */

typedef struct RCRE_PreprocessSpec {
    size_t      target_shape[4]; /* e.g., [1, 3, 224, 224]                          */
    RCRE_Layout target_layout;   /* HWC or CHW                                      */
    RCRE_DType  target_dtype;    /* U8, F32                                         */
    float       mean[4];         /* Channel-wise normalization: (x - mean) / std    */
    float       std[4];
    bool        preserve_aspect; /* If true, applies letterboxing                   */
    float       pad_value;       /* Padding fill value (e.g., 0.0f or 114.0f)       */
} RCRE_PreprocessSpec;

/**
 * @brief Explicit, deterministic input preprocessing and layout standardization.
 */
RCRE_NODISCARD int rcre_tensor_adapt(const RCRE_Tensor* src, 
                                    const RCRE_PreprocessSpec* spec, 
                                    RCRE_Arena* arena, 
                                    RCRE_Tensor* dst);

/* ========================================================================= */
/* Concurrency & Threadpool Engine (THR-REQ-01 to THR-REQ-03)                */
/* Architectural Design: Unix Philosophy (Keep Kernel Primitives Pure &      */
/* Single-Threaded). RCRE kernel operators (CV filters, FFT, CNN, ML) are    */
/* strictly deterministic, re-entrant, and single-threaded for universal    */
/* portability across bare-metal, embedded microcontrollers, and WASM.       */
/* The threadpool is an opt-in pipeline concurrency engine for orchestrating */
/* outer batch loops, multi-stream camera feeds, or tile partitioning.       */
/* ========================================================================= */

typedef struct RCRE_ThreadPool RCRE_ThreadPool;

typedef void (*rcre_task_fn)(void* arg, size_t task_id);
typedef void (*rcre_tile_fn)(void* arg, size_t y_start, size_t y_end, size_t x_start, size_t x_end);

/**
 * @brief Creates a C23 / POSIX work-stealing threadpool with persistent workers.
 * @param num_threads Number of worker threads (0 = auto-detect hardware concurrency).
 */
RCRE_NODISCARD RCRE_ThreadPool* rcre_threadpool_create(size_t num_threads);

/**
 * @brief Executes a 1D parallel-for loop across threadpool workers.
 */
void rcre_threadpool_parallel_for(RCRE_ThreadPool* pool, size_t count, rcre_task_fn task, void* arg);

/**
 * @brief Partitions 2D spatial dimensions into cache-friendly rectangular tiles.
 */
void rcre_threadpool_parallel_for_2d(RCRE_ThreadPool* pool, size_t height, size_t width, 
                                     size_t tile_h, size_t tile_w, 
                                     rcre_tile_fn tile_fn, void* arg);

/**
 * @brief Shuts down worker threads and destroys the threadpool.
 */
void rcre_threadpool_destroy(RCRE_ThreadPool* pool);

/* ========================================================================= */
/* Computer Vision Reference Primitives (CV-REQ-01 to CV-REQ-07)             */
/* ========================================================================= */

/**
 * @brief Converts 3-channel RGB image to 1-channel Grayscale using Rec.601 luma.
 */
RCRE_NODISCARD int rcre_cv_rgb_to_gray(const RCRE_Tensor* src, RCRE_Tensor* dst);

/**
 * @brief Swaps color channels between RGB and BGR in-place or into destination.
 */
RCRE_NODISCARD int rcre_cv_bgr_to_rgb(const RCRE_Tensor* src, RCRE_Tensor* dst);

/**
 * @brief Converts 3-channel RGB [0, 1] image to HSV [H in [0, 1), S in [0, 1], V in [0, 1]].
 */
RCRE_NODISCARD int rcre_cv_rgb_to_hsv(const RCRE_Tensor* src, RCRE_Tensor* dst);

/**
 * @brief Converts 3-channel HSV image back to RGB [0, 1].
 */
RCRE_NODISCARD int rcre_cv_hsv_to_rgb(const RCRE_Tensor* src, RCRE_Tensor* dst);

/**
 * @brief Scales a single channel in an image by a multiplier factor.
 */
RCRE_NODISCARD int rcre_cv_scale_channel(RCRE_Tensor* tensor, size_t channel_idx, float scale);

/**
 * @brief Shifts a single channel in an image by a constant offset.
 */
RCRE_NODISCARD int rcre_cv_shift_channel(RCRE_Tensor* tensor, size_t channel_idx, float shift);

/**
 * @brief Clamps tensor values in-place into [min_val, max_val].
 */
RCRE_NODISCARD int rcre_cv_clamp(RCRE_Tensor* tensor, float min_val, float max_val);

/**
 * @brief Resizes an image using Nearest-Neighbor interpolation with continuous subpixel coordinates.
 */
RCRE_NODISCARD int rcre_cv_resize_nn(const RCRE_Tensor* src, RCRE_Tensor* dst);

/**
 * @brief Resizes an image using 4-point Bilinear interpolation with continuous subpixel coordinates.
 */
RCRE_NODISCARD int rcre_cv_resize_bilinear(const RCRE_Tensor* src, RCRE_Tensor* dst);

/**
 * @brief Separable 2D Gaussian blur with reflection padding.
 */
RCRE_NODISCARD int rcre_cv_gaussian_blur(const RCRE_Tensor* src, RCRE_Tensor* dst, 
                                         float sigma, RCRE_Arena* arena);

/**
 * @brief Computes 3x3 horizontal (dx) and vertical (dy) Sobel spatial derivatives.
 */
RCRE_NODISCARD int rcre_cv_sobel(const RCRE_Tensor* src, RCRE_Tensor* dst_dx, 
                                 RCRE_Tensor* dst_dy, RCRE_Arena* arena);

/**
 * @brief General 2D spatial convolution across channels with edge clamping.
 */
RCRE_NODISCARD int rcre_cv_convolve_2d(const RCRE_Tensor* src, const RCRE_Tensor* kernel,
                                      RCRE_Tensor* dst, bool preserve_channels, RCRE_Arena* arena);

/**
 * @brief Constructs a 2D Box blur filter kernel tensor of size [size, size, 1].
 */
RCRE_NODISCARD int rcre_cv_filter_box(RCRE_Tensor* kernel, int size, RCRE_Arena* arena);

/**
 * @brief Constructs a 3x3 Sharpen filter kernel tensor.
 */
RCRE_NODISCARD int rcre_cv_filter_sharpen(RCRE_Tensor* kernel, RCRE_Arena* arena);

/**
 * @brief Constructs a 3x3 Emboss filter kernel tensor.
 */
RCRE_NODISCARD int rcre_cv_filter_emboss(RCRE_Tensor* kernel, RCRE_Arena* arena);

/**
 * @brief Constructs a 3x3 High-Pass filter kernel tensor.
 */
RCRE_NODISCARD int rcre_cv_filter_highpass(RCRE_Tensor* kernel, RCRE_Arena* arena);

/* --- Feature Detection, Descriptors & Matching --- */

typedef struct RCRE_CV_Point {
    float x;
    float y;
} RCRE_CV_Point;

typedef struct RCRE_CV_Descriptor {
    RCRE_CV_Point pt;
    int len;
    float* data;
} RCRE_CV_Descriptor;

typedef struct RCRE_CV_Match {
    RCRE_CV_Point p;
    RCRE_CV_Point q;
    int idx_a;
    int idx_b;
    float distance;
} RCRE_CV_Match;

/**
 * @brief Computes Harris cornerness response matrix R = det(S) - k * trace(S)^2.
 */
RCRE_NODISCARD int rcre_cv_harris_response(const RCRE_Tensor* src, RCRE_Tensor* dst_r,
                                          float sigma, float k, RCRE_Arena* arena);

/**
 * @brief Performs 2D Non-Maximum Suppression (NMS) over a window radius.
 */
RCRE_NODISCARD int rcre_cv_nms_2d(const RCRE_Tensor* src, RCRE_Tensor* dst,
                                 int window_radius, float threshold);

/**
 * @brief Detects Harris corners and extracts local gradient patch descriptors.
 */
RCRE_NODISCARD int rcre_cv_harris_detect(const RCRE_Tensor* src, float sigma, float threshold,
                                        int nms_radius, RCRE_CV_Descriptor** out_desc,
                                        int* out_count, RCRE_Arena* arena);

/**
 * @brief Matches feature descriptors using Lowe's ratio test and mutual uniqueness.
 */
RCRE_NODISCARD int rcre_cv_match_descriptors(const RCRE_CV_Descriptor* a, int na,
                                            const RCRE_CV_Descriptor* b, int nb,
                                            float ratio_thresh, RCRE_CV_Match** out_matches,
                                            int* out_count, RCRE_Arena* arena);

/**
 * @brief Draws crosshair markers on an image for detected keypoints.
 */
void rcre_cv_draw_markers(RCRE_Tensor* image, const RCRE_CV_Point* points, int count,
                          float r, float g, float b);

/* --- Homography, RANSAC & Panorama Stitching --- */

/**
 * @brief Computes 3x3 projective homography matrix H via Direct Linear Transformation (DLT).
 */
RCRE_NODISCARD int rcre_cv_compute_homography_dlt(const RCRE_CV_Match* matches, int n,
                                                 float H[3][3], RCRE_Arena* arena);

/**
 * @brief Computes optimal robust homography using RANSAC inlier consensus.
 */
RCRE_NODISCARD int rcre_cv_find_homography_ransac(RCRE_CV_Match* matches, int n,
                                                 float inlier_thresh, int iters, int cutoff,
                                                 float out_H[3][3], int* out_inliers,
                                                 RCRE_Arena* arena);

/**
 * @brief Warps an image using backward projective mapping with bilinear interpolation.
 */
RCRE_NODISCARD int rcre_cv_warp_perspective(const RCRE_Tensor* src, const float H[3][3],
                                           RCRE_Tensor* dst, RCRE_Arena* arena);

/**
 * @brief Projects an image onto a cylinder with focal length f.
 */
RCRE_NODISCARD int rcre_cv_cylindrical_project(const RCRE_Tensor* src, float f,
                                              RCRE_Tensor* dst, RCRE_Arena* arena);

/**
 * @brief End-to-end two-image panorama stitching pipeline.
 */
RCRE_NODISCARD int rcre_cv_stitch_panorama(const RCRE_Tensor* a, const RCRE_Tensor* b,
                                          float f_cylindrical, float sigma, float thresh,
                                          int nms, float inlier_thresh, int iters,
                                          RCRE_Tensor* dst, RCRE_Arena* arena);

/* --- Integral Images & Lucas-Kanade Optical Flow --- */

/**
 * @brief Computes 2D Summed-Area Table (Integral Image) for O(1) box queries.
 */
RCRE_NODISCARD int rcre_cv_integral_image(const RCRE_Tensor* src, RCRE_Tensor* dst_integral);

/**
 * @brief Applies constant-time box filter on an image using its precomputed integral table.
 */
RCRE_NODISCARD int rcre_cv_box_filter_integral(const RCRE_Tensor* integral, RCRE_Tensor* dst,
                                              int box_radius);

/**
 * @brief Computes Lucas-Kanade dense/strided optical flow velocity field [H/s, W/s, 2].
 */
RCRE_NODISCARD int rcre_cv_optical_flow_lk(const RCRE_Tensor* curr, const RCRE_Tensor* prev,
                                          RCRE_Tensor* dst_velocity, int window_size,
                                          int stride, float max_v, RCRE_Arena* arena);

/**
 * @brief Overlays optical flow velocity vectors onto an image.
 */
void rcre_cv_draw_flow(RCRE_Tensor* canvas, const RCRE_Tensor* velocity, int stride, float scale);


/* ========================================================================= */
/* Fast Fourier Transform & Spectral Image Processing (FFT-REQ-01 to 04)     */
/* ========================================================================= */

typedef struct RCRE_Complex {
    float re;
    float im;
} RCRE_Complex;

/**
 * @brief 1D In-Place Radix-2 Cooley-Tukey Fast Fourier Transform (Forward).
 * @param data Array of complex numbers of length N (N must be a power of 2).
 * @param n Length of the transform (must satisfy (n & (n - 1)) == 0).
 * @param arena Optional scratchpad arena.
 */
RCRE_NODISCARD int rcre_fft_1d(RCRE_Complex* data, size_t n, RCRE_Arena* arena);

/**
 * @brief 1D In-Place Radix-2 Inverse Fast Fourier Transform with 1/N normalization.
 */
RCRE_NODISCARD int rcre_ifft_1d(RCRE_Complex* data, size_t n, RCRE_Arena* arena);

/**
 * @brief 2D Fast Fourier Transform on an image or matrix (Forward).
 * @param input Real input tensor of shape [H, W] (F32).
 * @param output_re Real part output tensor of shape [H, W] (F32).
 * @param output_im Imaginary part output tensor of shape [H, W] (F32).
 * @param arena Scratchpad arena for intermediate row/column buffers.
 */
RCRE_NODISCARD int rcre_fft_2d(const RCRE_Tensor* input,
                               RCRE_Tensor* output_re,
                               RCRE_Tensor* output_im,
                               RCRE_Arena* arena);

/**
 * @brief 2D Inverse Fast Fourier Transform (Complex frequency domain -> Real spatial domain).
 * @param input_re Real frequency tensor of shape [H, W] (F32).
 * @param input_im Imaginary frequency tensor of shape [H, W] (F32).
 * @param output Reconstructed real spatial tensor of shape [H, W] (F32).
 * @param arena Scratchpad arena for intermediate row/column buffers.
 */
RCRE_NODISCARD int rcre_ifft_2d(const RCRE_Tensor* input_re,
                                const RCRE_Tensor* input_im,
                                RCRE_Tensor* output,
                                RCRE_Arena* arena);

/**
 * @brief Shifts zero-frequency component to center of 2D spectrum [H/2, W/2].
 */
RCRE_NODISCARD int rcre_fftshift_2d(RCRE_Tensor* tensor_re, RCRE_Tensor* tensor_im);

/**
 * @brief Inverse shift: moves centered DC component back to top-left corner [0, 0].
 */
RCRE_NODISCARD int rcre_ifftshift_2d(RCRE_Tensor* tensor_re, RCRE_Tensor* tensor_im);

/**
 * @brief Frequency-domain 2D filtering (Ideal or Gaussian, Low-pass or High-pass).
 * @param spectrum_re Real part of centered 2D spectrum.
 * @param spectrum_im Imaginary part of centered 2D spectrum.
 * @param cutoff_radius Cutoff frequency radius D_0 in pixels.
 * @param is_highpass If true, creates high-pass filter; if false, low-pass filter.
 * @param is_gaussian If true, applies smooth Gaussian attenuation; if false, sharp brickwall.
 */
RCRE_NODISCARD int rcre_fft_filter_2d(RCRE_Tensor* spectrum_re,
                                      RCRE_Tensor* spectrum_im,
                                      float cutoff_radius,
                                      bool is_highpass,
                                      bool is_gaussian);

/* ========================================================================= */
/* Convolutional Neural Network (CNN) Inference Primitives (NN-REQ-01 to 05) */
/* ========================================================================= */

typedef struct RCRE_Conv2DParams {
    size_t stride_h;
    size_t stride_w;
    size_t pad_h;
    size_t pad_w;
} RCRE_Conv2DParams;

/**
 * @brief Standard 2D Spatial Convolution: [C_in, H_in, W_in] x [C_out, C_in, K_h, K_w] -> [C_out, H_out, W_out].
 */
RCRE_NODISCARD int rcre_nn_conv2d(const RCRE_Tensor* input,
                                  const RCRE_Tensor* weights,
                                  const RCRE_Tensor* bias,
                                  const RCRE_Conv2DParams* params,
                                  RCRE_Tensor* output,
                                  RCRE_Arena* arena);

/**
 * @brief Depthwise 2D Convolution (channel-multiplier = 1):
 *        Convolves each input channel independently with a spatial filter:
 *        [C, H_in, W_in] x [C, 1, K_h, K_w] -> [C, H_out, W_out].
 */
RCRE_NODISCARD int rcre_nn_depthwise_conv2d(const RCRE_Tensor* input,
                                            const RCRE_Tensor* weights,
                                            const RCRE_Tensor* bias,
                                            const RCRE_Conv2DParams* params,
                                            RCRE_Tensor* output,
                                            RCRE_Arena* arena);

/**
 * @brief Pointwise 1x1 Convolution / Channel Projection:
 *        [C_in, H, W] x [C_out, C_in, 1, 1] -> [C_out, H, W].
 */
RCRE_NODISCARD int rcre_nn_pointwise_conv2d(const RCRE_Tensor* input,
                                            const RCRE_Tensor* weights,
                                            const RCRE_Tensor* bias,
                                            RCRE_Tensor* output,
                                            RCRE_Arena* arena);

/**
 * @brief Fused 2D Batch Normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta.
 */
RCRE_NODISCARD int rcre_nn_batchnorm2d(const RCRE_Tensor* input,
                                      const RCRE_Tensor* gamma,
                                      const RCRE_Tensor* beta,
                                      const RCRE_Tensor* mean,
                                      const RCRE_Tensor* var,
                                      float eps,
                                      RCRE_Tensor* output);

/**
 * @brief Fold BatchNorm scale & shift directly into Conv2D weights and bias (Zero-Cost Edge BN).
 */
RCRE_NODISCARD int rcre_nn_fuse_conv_batchnorm(RCRE_Tensor* conv_weight,
                                               RCRE_Tensor* conv_bias,
                                               const RCRE_Tensor* gamma,
                                               const RCRE_Tensor* beta,
                                               const RCRE_Tensor* mean,
                                               const RCRE_Tensor* var,
                                               float eps);

/**
 * @brief Piecewise ReLU6 activation: y = min(max(x, 0), 6.0).
 */
RCRE_NODISCARD int rcre_nn_relu6(const RCRE_Tensor* input, RCRE_Tensor* output);

/**
 * @brief Global Average Pooling 2D: [C, H, W] -> [C, 1, 1].
 */
RCRE_NODISCARD int rcre_nn_global_avgpool(const RCRE_Tensor* input, RCRE_Tensor* output);

/**
 * @brief Fully connected linear projection: output = input * W^T + bias.
 */
RCRE_NODISCARD int rcre_nn_linear(const RCRE_Tensor* input,
                                  const RCRE_Tensor* weights,
                                  const RCRE_Tensor* bias,
                                  RCRE_Tensor* output,
                                  RCRE_Arena* arena);

/**
 * @brief Numerically stable Softmax activation.
 */
RCRE_NODISCARD int rcre_nn_softmax(const RCRE_Tensor* input, RCRE_Tensor* output);

/* ========================================================================= */
/* Classical Machine Learning Primitives (ML-REQ-01)                         */
/* ========================================================================= */

/**
 * @brief Fits an Ordinary Least Squares (OLS) regression model: w = (X^T X)^-1 X^T y.
 * @param X Matrix of shape [N, D] (F32).
 * @param y Vector of shape [N, 1] (F32).
 * @param weights Output vector of shape [D, 1] (F32).
 */
RCRE_NODISCARD int rcre_ml_ols_fit(const RCRE_Tensor* X, const RCRE_Tensor* y, 
                                   RCRE_Tensor* weights, RCRE_Arena* arena);

/**
 * @brief Fits Logistic Regression via Gradient Descent with Cross-Entropy Loss.
 */
RCRE_NODISCARD int rcre_ml_logistic_fit(const RCRE_Tensor* X, const RCRE_Tensor* y, 
                                        RCRE_Tensor* weights, float lr, 
                                        size_t epochs, RCRE_Arena* arena);

/**
 * @brief Fits Linear Soft-Margin SVM via Pegasos Subgradient Descent.
 */
RCRE_NODISCARD int rcre_ml_pegasos_svm_fit(const RCRE_Tensor* X, const RCRE_Tensor* y, 
                                           RCRE_Tensor* weights, float lambda_reg, 
                                           size_t epochs, RCRE_Arena* arena);

/**
 * @brief Computes linear model inference: y_pred = X * weights.
 */
RCRE_NODISCARD int rcre_ml_predict_linear(const RCRE_Tensor* X, const RCRE_Tensor* weights, 
                                          RCRE_Tensor* preds);

/* ========================================================================= */
/* Clustering & Dimensionality Reduction (ML-REQ-02)                         */
/* ========================================================================= */

typedef struct RCRE_KMeansParams {
    size_t   max_iter;    /* Maximum iterations (e.g. 100) */
    size_t   n_init;      /* Number of independent initializations (e.g. 10) */
    float    tol;         /* Centroid shift convergence tolerance (e.g. 1e-4f) */
    uint64_t seed;        /* Random seed for PRNG reproducibility */
} RCRE_KMeansParams;

/**
 * @brief Fits k-Means clustering via standard settings.
 */
RCRE_NODISCARD int rcre_ml_kmeans_fit(const RCRE_Tensor* X, size_t k, size_t max_iter,
                                      RCRE_Tensor* centroids, RCRE_Tensor* labels,
                                      RCRE_Arena* arena);

/**
 * @brief Fits k-Means clustering with full control over initialization, tolerance, and multi-start.
 * @param X Matrix of shape [N, D] (F32).
 * @param k Number of clusters.
 * @param params Advanced parameters (max_iter, n_init, tol, seed).
 * @param centroids Output matrix of shape [K, D] (F32).
 * @param labels Optional output vector of shape [N, 1] (I32 or F32).
 * @param out_inertia Optional pointer to output the within-cluster sum of squared errors (WCSS).
 * @param arena Scratchpad arena for transient allocations.
 */
RCRE_NODISCARD int rcre_ml_kmeans_fit_ex(const RCRE_Tensor* X, size_t k,
                                         const RCRE_KMeansParams* params,
                                         RCRE_Tensor* centroids, RCRE_Tensor* labels,
                                         float* out_inertia, RCRE_Arena* arena);

/**
 * @brief Predicts closest cluster centroid index for input samples.
 */
RCRE_NODISCARD int rcre_ml_kmeans_predict(const RCRE_Tensor* X, const RCRE_Tensor* centroids,
                                          RCRE_Tensor* labels);

typedef struct RCRE_DBSCANParams {
    float  eps;          /* Neighborhood radius distance epsilon */
    size_t min_samples;  /* Minimum points within eps to form a core point */
} RCRE_DBSCANParams;

/**
 * @brief Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
 * @param X Matrix of shape [N, D] (F32).
 * @param params DBSCAN configuration (eps, min_samples).
 * @param labels Output vector of shape [N, 1] (I32 or F32). Noise outliers are assigned -1.
 * @param out_num_clusters Optional pointer to receive total number of discovered clusters.
 * @param arena Scratchpad arena for transient allocations.
 */
RCRE_NODISCARD int rcre_ml_dbscan_fit(const RCRE_Tensor* X,
                                      const RCRE_DBSCANParams* params,
                                      RCRE_Tensor* labels,
                                      size_t* out_num_clusters,
                                      RCRE_Arena* arena);

/**
 * @brief Computes top K Principal Components via covariance power iteration.
 * @param X Input matrix [N, D] (F32).
 * @param components Output matrix [K, D] representing top K principal eigenvectors.
 * @param mean Output vector [D, 1] representing feature means.
 */
RCRE_NODISCARD int rcre_ml_pca_fit(const RCRE_Tensor* X, size_t k, 
                                   RCRE_Tensor* components, RCRE_Tensor* mean,
                                   RCRE_Arena* arena);

/**
 * @brief Projects data onto fitted principal components: X_proj = (X - mean) * components^T.
 */
RCRE_NODISCARD int rcre_ml_pca_transform(const RCRE_Tensor* X, const RCRE_Tensor* components,
                                         const RCRE_Tensor* mean, RCRE_Tensor* X_proj);

/**
 * @brief Reconstructs data from principal component projections: X_recon = X_proj * components + mean.
 */
RCRE_NODISCARD int rcre_ml_pca_inverse_transform(const RCRE_Tensor* X_proj, const RCRE_Tensor* components,
                                                 const RCRE_Tensor* mean, RCRE_Tensor* X_recon);


/* ========================================================================= */
/* Modern Tabular Ensemble: Flat-Array GBDT Inference (ML-REQ-03)            */
/* ========================================================================= */

typedef struct RCRE_TreeNode {
    uint32_t feature_idx;     /* Feature index to split on */
    float    threshold;       /* Numerical split threshold */
    uint32_t left_child;      /* Node index for x <= threshold */
    uint32_t right_child;     /* Node index for x > threshold */
    float    leaf_value;      /* Prediction value if is_leaf is true */
    bool     is_leaf;
} RCRE_TreeNode;

typedef struct RCRE_GBDTModel {
    const RCRE_TreeNode* nodes;     /* Flat array of nodes for all trees */
    const size_t*        tree_roots;/* Starting node index for each tree */
    size_t               num_trees;
    size_t               num_nodes;
    float                base_score;
    bool                 is_binary_classification;
} RCRE_GBDTModel;

/**
 * @brief Fast zero-allocation GBDT ensemble inference over flat-array nodes.
 */
RCRE_NODISCARD int rcre_ml_gbdt_predict(const RCRE_GBDTModel* model, 
                                        const RCRE_Tensor* X, 
                                        RCRE_Tensor* preds);

/* ========================================================================= */
/* Similarity & Fast Vector Search: HNSW Graph Index (ML-REQ-04)             */
/* ========================================================================= */

typedef struct RCRE_HNSWIndex RCRE_HNSWIndex;

typedef enum {
    RCRE_METRIC_L2     = 0,
    RCRE_METRIC_COSINE = 1
} RCRE_MetricType;

/**
 * @brief Allocates an HNSW vector index inside caller scratchpad arena.
 */
RCRE_NODISCARD RCRE_HNSWIndex* rcre_hnsw_create(size_t dim, size_t max_elements, 
                                                size_t M, size_t ef_construction, 
                                                RCRE_MetricType metric, 
                                                RCRE_Arena* arena);

/**
 * @brief Inserts a vector with an external ID into the HNSW graph.
 */
RCRE_NODISCARD int rcre_hnsw_insert(RCRE_HNSWIndex* index, const float* vector, size_t id, 
                                    RCRE_Arena* scratch);

/**
 * @brief Searches for k nearest neighbors to a query vector.
 */
RCRE_NODISCARD int rcre_hnsw_search(const RCRE_HNSWIndex* index, const float* query, 
                                    size_t k, size_t* out_ids, float* out_distances, 
                                    size_t ef_search, RCRE_Arena* scratch);

/* ========================================================================= */
/* Hardware Abstraction Layer & SIMD Dispatch (HAL-REQ-01 to HAL-REQ-04)     */
/* ========================================================================= */

typedef enum {
    RCRE_CPU_FEAT_SCALAR      = (1 << 0),
    RCRE_CPU_FEAT_NEON        = (1 << 1),
    RCRE_CPU_FEAT_AVX2        = (1 << 2),
    RCRE_CPU_FEAT_FMA         = (1 << 3),
    RCRE_CPU_FEAT_AVX512F     = (1 << 4),
    RCRE_CPU_FEAT_WASM_SIMD   = (1 << 5)
} RCRE_CpuFeatures;

typedef void (*rcre_gemm_fn)(size_t M, size_t N, size_t K, float alpha, 
                             const float* A, size_t lda, 
                             const float* B, size_t ldb, 
                             float beta, float* C, size_t ldc);

typedef float (*rcre_dot_fn)(const float* x, const float* y, size_t n);

typedef void (*rcre_axpy_fn)(size_t n, float alpha, const float* x, float* y);

typedef void (*rcre_rgb_to_gray_u8_fn)(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

typedef struct RCRE_HalDispatch {
    const char*            backend_name;
    uint32_t               cpu_features;
    rcre_gemm_fn           gemm_f32;
    rcre_dot_fn            dot_f32;
    rcre_axpy_fn           axpy_f32;
    rcre_rgb_to_gray_u8_fn rgb_to_gray_u8;
} RCRE_HalDispatch;

extern RCRE_HalDispatch rcre_hal;

/**
 * @brief Probes host hardware and initializes dynamic SIMD dispatch table.
 */
void rcre_hal_init(void);

/**
 * @brief Returns bitmask of detected CPU SIMD features.
 */
uint32_t rcre_cpu_get_features(void);

/**
 * @brief Returns human-readable CPU brand/model string.
 */
const char* rcre_cpu_get_brand(void);

/* ========================================================================= */
/* Autograd & Neural Network Extensions                                      */
/* ========================================================================= */
#ifndef RCRE_NO_AUTOGRAD
#include "rcre_autograd.h"
#endif

/* ========================================================================= */
/* Generative AI & Autoregressive Transformer Architecture                    */
/* ========================================================================= */
#ifndef RCRE_NO_TRANSFORMER
#include "rcre_transformer.h"
#endif

#ifdef __cplusplus
}
#endif

#endif /* RCRE_H */
