/**
 * @file rcre.h
 * @brief RCRE: Recep's Code Run Everywhere - Unified C23 CV & ML Engine
 * @version 1.0.0
 * 
 * Strict ISO C23 compliant, zero external dependencies.
 */

#ifndef RCRE_H
#define RCRE_H

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

/* ========================================================================= */
/* Status & Error Codes                                                      */
/* ========================================================================= */

typedef enum {
    RCRE_OK                  =  0,
    RCRE_ERR_INVALID_ARG     = -1,
    RCRE_ERR_OUT_OF_MEMORY   = -2,
    RCRE_ERR_SHAPE_MISMATCH  = -3,
    RCRE_ERR_NOT_CONTIGUOUS  = -4,
    RCRE_ERR_UNSUPPORTED_OP  = -5
} RCRE_Status;

/* ========================================================================= */
/* Data Types & Layouts                                                      */
/* ========================================================================= */

typedef enum {
    RCRE_DTYPE_U8  = 0, /* 8-bit unsigned integer (image pixels, quantized bins) */
    RCRE_DTYPE_I32 = 1, /* 32-bit signed integer (class labels, indices)          */
    RCRE_DTYPE_F16 = 2, /* 16-bit half precision float                             */
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

/* ========================================================================= */
/* Storage & Tensor Abstractions (TNS-REQ-01 to TNS-REQ-05)                  */
/* ========================================================================= */

typedef struct RCRE_Storage {
    void*         raw_data;
    size_t        size_bytes;
    atomic_size_t ref_count;
    bool          is_arena_allocated;
} RCRE_Storage;

typedef struct RCRE_Tensor {
    RCRE_Storage* storage;
    void*         data;            /* Pointer to element (0, 0, ...) inside storage */
    size_t        shape[4];        /* Dimensions (rank 1 to 4)                     */
    ptrdiff_t     strides[4];      /* Step in elements per dimension               */
    uint8_t       ndim;            /* Rank (1 to 4)                                */
    RCRE_DType    dtype;
    RCRE_Layout   layout;
} RCRE_Tensor;

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
    RCRE_DType  target_dtype;    /* U8, F32, F16                                    */
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
/* Computer Vision Reference Primitives (CV-REQ-01 to CV-REQ-03)             */
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
 * @brief Separable 2D Gaussian blur with reflection padding.
 */
RCRE_NODISCARD int rcre_cv_gaussian_blur(const RCRE_Tensor* src, RCRE_Tensor* dst, 
                                         float sigma, RCRE_Arena* arena);

/**
 * @brief Computes 3x3 horizontal (dx) and vertical (dy) Sobel spatial derivatives.
 */
RCRE_NODISCARD int rcre_cv_sobel(const RCRE_Tensor* src, RCRE_Tensor* dst_dx, 
                                 RCRE_Tensor* dst_dy, RCRE_Arena* arena);

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

#ifdef __cplusplus
}
#endif

#endif /* RCRE_H */
