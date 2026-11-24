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

#define RCRE_VERSION_MAJOR 2
#define RCRE_VERSION_MINOR 2
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

#define RCRE_ABI_VERSION 0x020000U
RCRE_NODISCARD uint32_t rcre_abi_version(void);

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
    RCRE_DTYPE_U8   = 0, /* 8-bit unsigned integer (image pixels, quantized bins) */
    RCRE_DTYPE_I32  = 1, /* 32-bit signed integer (class labels, indices)          */
    RCRE_DTYPE_I64  = 2, /* 64-bit signed integer (token IDs, Arrow/NumPy int64)   */
    RCRE_DTYPE_F32  = 3, /* 32-bit single precision float                         */
    RCRE_DTYPE_F16  = 4, /* 16-bit IEEE 754-2008 half precision float             */
    RCRE_DTYPE_BF16 = 5, /* 16-bit Google Brain bfloat16 format                   */
    RCRE_DTYPE_I8   = 6, /* 8-bit signed quantized integer                        */
    RCRE_DTYPE_F64  = 7  /* 64-bit IEEE-754 double precision (HPC scientific, H0) */
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
    /* --- profiling telemetry (appended at end; zero-initialized) --- */
    size_t   high_watermark; /* peak offset reached since creation */
    size_t   alloc_count;    /* sub-allocations served */
    size_t   rewind_count;   /* scoped rewinds (temp_end/reset) executed */
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
 * @brief Initializes a stack/static RCRE_Arena over caller-supplied memory
 *        (zero heap allocations; ideal for freestanding/embedded targets).
 * @param arena Caller-owned RCRE_Arena struct to initialize.
 * @param backing_buffer Pre-allocated memory buffer (64-byte alignment recommended).
 * @param capacity Byte capacity of backing_buffer.
 * @return RCRE_OK on success, RCRE_ERR_INVALID_ARG on NULL/zero capacity.
 */
RCRE_NODISCARD int rcre_arena_from_buffer(RCRE_Arena* arena, void* backing_buffer, size_t capacity);

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
/* Zero-Copy Byte Slice & Arena String Builder (SLC-REQ-01 to SLC-REQ-03)    */
/* ========================================================================= */

#define RCRE_NPOS ((size_t)-1)

typedef struct RCRE_Slice {
    const uint8_t* data;
    size_t         len;
} RCRE_Slice;

typedef struct RCRE_SlicePair {
    RCRE_Slice head;
    RCRE_Slice tail;
    bool       found;
} RCRE_SlicePair;

typedef struct RCRE_StrBuilder {
    uint8_t*    buf;
    size_t      len;
    size_t      cap;
    RCRE_Arena* arena;
    bool        overflow;
} RCRE_StrBuilder;

RCRE_NODISCARD RCRE_Slice rcre_slice_empty(void);
RCRE_NODISCARD RCRE_Slice rcre_slice_from_parts(const void* data, size_t len);
RCRE_NODISCARD RCRE_Slice rcre_slice_from_cstr(const char* cstr);
RCRE_NODISCARD RCRE_Slice rcre_slice_sub(RCRE_Slice s, size_t start, size_t end);
RCRE_NODISCARD bool rcre_slice_eq(RCRE_Slice a, RCRE_Slice b);
RCRE_NODISCARD bool rcre_slice_eq_cstr(RCRE_Slice a, const char* cstr);
RCRE_NODISCARD bool rcre_slice_starts_with(RCRE_Slice s, RCRE_Slice prefix);
RCRE_NODISCARD bool rcre_slice_ends_with(RCRE_Slice s, RCRE_Slice suffix);
RCRE_NODISCARD RCRE_Slice rcre_slice_trim_ascii(RCRE_Slice s);
RCRE_NODISCARD size_t rcre_slice_find_byte(RCRE_Slice s, uint8_t needle);
RCRE_NODISCARD size_t rcre_slice_rfind_byte(RCRE_Slice s, uint8_t needle);
RCRE_NODISCARD size_t rcre_slice_count_byte(RCRE_Slice s, uint8_t needle);
RCRE_NODISCARD RCRE_SlicePair rcre_slice_split_once_byte(RCRE_Slice s, uint8_t delim);
RCRE_NODISCARD RCRE_SlicePair rcre_slice_split_once(RCRE_Slice s, RCRE_Slice delim);
RCRE_NODISCARD bool rcre_slice_parse_i64(RCRE_Slice s, int64_t* out_val);
RCRE_NODISCARD bool rcre_slice_parse_u64(RCRE_Slice s, uint64_t* out_val);
RCRE_NODISCARD char* rcre_arena_cstr_dup(RCRE_Arena* arena, RCRE_Slice s);

RCRE_NODISCARD RCRE_StrBuilder rcre_sb_init(RCRE_Arena* arena, size_t initial_cap);
bool rcre_sb_append_slice(RCRE_StrBuilder* sb, RCRE_Slice s);
bool rcre_sb_append_cstr(RCRE_StrBuilder* sb, const char* cstr);
bool rcre_sb_append_byte(RCRE_StrBuilder* sb, uint8_t b);
bool rcre_sb_append_u64(RCRE_StrBuilder* sb, uint64_t val);
bool rcre_sb_append_i64(RCRE_StrBuilder* sb, int64_t val);
RCRE_NODISCARD RCRE_Slice rcre_sb_to_slice(const RCRE_StrBuilder* sb);
RCRE_NODISCARD char* rcre_sb_to_cstr(RCRE_StrBuilder* sb);

/* ========================================================================= */
/* Tensor Abstraction & Zero-Copy Views (TNS-REQ-01 to TNS-REQ-05)           */
/* ========================================================================= */

#define RCRE_MAX_DIMS 8

typedef struct RCRE_Tensor {
    void*         data;                 /* Pointer to element (0, 0, ...)               */
    float*        grad;                 /* Backprop buffer (training); NULL otherwise   */
    size_t        shape[RCRE_MAX_DIMS]; /* Dimensions (rank 1 to 8)                     */
    ptrdiff_t     strides[RCRE_MAX_DIMS];/* Step in elements per dimension              */
    size_t        numel;                /* Cached element count (set at construction;   */
                                        /* shape/strides are immutable afterwards)      */
    uint8_t       ndim;                 /* Rank (1 to 8)                                */
    RCRE_DType    dtype;
    RCRE_Layout   layout;
    bool          is_view;              /* true: data owned elsewhere (slice/permute/…) */
    const char*   label;                /* Optional name (visualization/debug); NULL ok */
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

/**
 * @brief Computes NumPy-compatible broadcast output shape for two tensors.
 * @return RCRE_OK if shapes are broadcast-compatible, RCRE_ERR_SHAPE_MISMATCH otherwise.
 */
RCRE_NODISCARD int rcre_tensor_broadcast_shape(
    const size_t* shape_a, uint8_t ndim_a,
    const size_t* shape_b, uint8_t ndim_b,
    size_t out_shape[RCRE_MAX_DIMS], uint8_t* out_ndim
);

/**
 * @brief Creates a zero-copy strided broadcast view (0-stride along expanded axes).
 */
RCRE_NODISCARD RCRE_Tensor rcre_tensor_broadcast_to(
    const RCRE_Tensor* src,
    const size_t* target_shape,
    uint8_t target_ndim
);

/**
 * @brief Broadcasted elementwise addition: dst = a + b (F32).
 */
RCRE_NODISCARD int rcre_tensor_add_broadcast(
    const RCRE_Tensor* a, const RCRE_Tensor* b,
    RCRE_Arena* arena, RCRE_Tensor* dst
);

/**
 * @brief Broadcasted elementwise multiplication: dst = a * b (F32).
 */
RCRE_NODISCARD int rcre_tensor_mul_broadcast(
    const RCRE_Tensor* a, const RCRE_Tensor* b,
    RCRE_Arena* arena, RCRE_Tensor* dst
);

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

typedef enum {
    RCRE_QOS_DEFAULT    = 0, /* Inherit caller OS scheduling class                */
    RCRE_QOS_PERF       = 1, /* Steer onto Performance (P) cores / high priority  */
    RCRE_QOS_EFFICIENCY = 2, /* Steer onto Efficiency (E) cores / utility class   */
    RCRE_QOS_BACKGROUND = 3  /* Lowest background thermal/power scheduling class  */
} RCRE_QoSClass;

typedef void (*rcre_task_fn)(void* arg, size_t task_id);
typedef void (*rcre_tile_fn)(void* arg, size_t y_start, size_t y_end, size_t x_start, size_t x_end);

/**
 * @brief Creates a C23 / POSIX work-stealing threadpool with persistent workers.
 * @param num_threads Number of worker threads (0 = auto-detect effective quota/concurrency).
 */
RCRE_NODISCARD RCRE_ThreadPool* rcre_threadpool_create(size_t num_threads);

/**
 * @brief Creates a work-stealing threadpool with explicit OS Quality-of-Service / core steering.
 * @param num_threads Number of worker threads (0 = auto-detect matching QoS tier or quota).
 * @param qos OS scheduling / asymmetric core steering hint.
 */
RCRE_NODISCARD RCRE_ThreadPool* rcre_threadpool_create_ex(size_t num_threads, RCRE_QoSClass qos);

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

typedef struct RCRE_ORBKeypoint {
    float    x;
    float    y;
    float    angle;   /* Dominant orientation in radians [-pi, pi] */
    uint32_t score;   /* FAST corner score */
    uint32_t bits[8]; /* 256-bit rotated BRIEF descriptor */
} RCRE_ORBKeypoint;

/** Detects FAST-9 corners on a grayscale/RGB F32 or U8 image and computes intensity-centroid orientation. */
RCRE_NODISCARD int rcre_cv_fast_detect(const RCRE_Tensor* gray_f32, float threshold,
                                       int max_keypoints, RCRE_ORBKeypoint** out_kps,
                                       int* out_count, RCRE_Arena* arena);

/** Computes 256-bit rotated BRIEF binary descriptors at detected ORB keypoints. */
RCRE_NODISCARD int rcre_cv_orb_extract(const RCRE_Tensor* gray_f32, RCRE_ORBKeypoint* kps, int count);

/** Matches two sets of ORB binary descriptors using Hamming distance (popcount) + ratio filter. */
RCRE_NODISCARD int rcre_cv_orb_match(const RCRE_ORBKeypoint* a, int na,
                                     const RCRE_ORBKeypoint* b, int nb,
                                     int max_hamming, RCRE_CV_Match** out_matches,
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

/** Fits L1+L2 ElasticNet sparse linear model via coordinate descent. */
RCRE_NODISCARD int rcre_ml_sparse_linear_fit(const RCRE_Tensor* X, const RCRE_Tensor* y,
                                             float l1_reg, float l2_reg, size_t max_iter,
                                             RCRE_Tensor* weights);

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

/* ========================================================================= */
/* e-means — mini-batch EMA k-means (ML-REQ-02, sibling of rcre_ml_kmeans_*) */
/* ========================================================================= */

/**
 * @brief e-means for RCRE — mini-batch EMA k-means with scaling-law recipes.
 *
 * Implements Algorithm 1 of "e-means: outperforming k-means (in one-tenth
 * the epochs)" (Bessa et al., arXiv 2611.09501, 2026): Euclidean, FP32
 * model state, CPU, single process. Lloyd k-means remains available as
 * `rcre_ml_kmeans_fit_ex` — the two are complementary (random-init EMA at a
 * fixed epoch budget vs k-means++ Lloyd with convergence tolerance).
 *
 * Exactness contract (projects/rcre_emeans/docs/exactness.md):
 *  - pooled runs are bit-identical run-to-run and across pool sizes (the
 *    slab partition depends only on the batch size);
 *  - the serial run (single slab) is a separately quality-guaranteed
 *    configuration: final MSE within 0.5% and >= 99% label agreement vs a
 *    pooled run at the same seed (F32 partial-sum regrouping feeds back
 *    chaotically through the EMA system — 1 ULP is NOT the contract);
 *  - batch sums: FP32, unbuffered, row order; objective and finalization
 *    sums: FP64 (documented improvement over the reference's FP32);
 *  - state machine: bit-exact port of the reference's FP32 operation order
 *    (gated against the Python oracle in projects/rcre_emeans/ref/).
 */

/** Batch-size law selector. AUTO resolves by dimension: quality for
 *  d <= 16, throughput above (the reference's "auto" behavior). */
typedef enum {
    RCRE_EM_BATCH_AUTO = 0,
    RCRE_EM_BATCH_THROUGHPUT = 1,
    RCRE_EM_BATCH_QUALITY = 2,
} RCRE_EMBatchLaw;

/**
 * Resolved training plan (the reference's "recipe"). Exposed for
 * observability: the auto-tuning is a formula, and the user can see what
 * it produced for their (n, k, T).
 */
typedef struct RCRE_EMMeansRecipe {
    size_t batch_size;       /* resolved statistical batch B            */
    float  alpha0;           /* resolved peak EMA step (FP32)           */
    double alpha0_full;      /* peak step, unrounded double (Eq. 5)     */
    float  nepochs;          /* T as requested (fractional ok)          */
    size_t epoch_rows[16];   /* rows per epoch (final is always n)      */
    size_t n_epochs;         /* number of epochs (len of epoch_rows)    */
    size_t steps_per_pass;   /* ceil(n / B)                             */
    size_t total_steps;      /* M: total EMA updates over the run       */
    float  unused_threshold; /* tau (0 = revival off)                   */
    float  split_epsilon;    /* donor-split perturbation delta (1e-4)   */
    float  init_usage;       /* N_j at init (2^-16)                     */
    int    batch_law;        /* resolved law (RCRE_EMBatchLaw or -1 = manual) */
    uint64_t seed;           /* seed used for shuffle/revival streams   */
} RCRE_EMMeansRecipe;

typedef struct RCRE_EMMeansParams {
    float    nepochs;              /* T (default 5); fractional ok, <= 16 */
    size_t   batch_size;           /* 0 = batch law (default)             */
    float    alpha0;               /* <= 0 = step law (default); NaN err  */
    int      batch_law;            /* RCRE_EMBatchLaw (default AUTO)      */
    float    unused_threshold;     /* < 0 = auto (0.1, off for d <= 2);   */
                                   /* 0 = revival off; > 0 fixed tau      */
    bool     shuffle;              /* fresh per-epoch shuffle (default true) */
    const RCRE_Tensor* init_centroids; /* [k,d] F32 or NULL = random rows */
    size_t   max_points_per_centroid;  /* 0 = no cap; else train on a     */
                                       /* random sorted subset of         */
                                       /* cap*k rows (reference parity)   */
    uint64_t seed;                 /* 0 = 1234 (reference default)        */
    bool     capture_labels;       /* fill labels with last-epoch passes  */
                                   /* (default true)                      */
} RCRE_EMMeansParams;

/**
 * @brief Returns the default parameter struct: T = 5, batch and alpha0 from
 *        the scaling laws, AUTO revival threshold (-1), shuffle on, random
 *        init, label capture on, reference seed (1234). Override what you
 *        need. (A zeroed struct means "batch law + step law + revival off";
 *        use this constructor for the reference-default behavior.)
 */
RCRE_EMMeansParams rcre_ml_emeans_params_default(void);

/**
 * @brief Fills the e-means training plan without running training.
 * @param X [n,d] F32 input (only n and d are needed; data may be NULL).
 * @param k Number of clusters.
 * @param params NULL = all defaults.
 * @param out Resolved recipe (batch B, alpha0, epoch rows, total steps).
 * @returns RCRE_OK or RCRE_ERR_INVALID_ARG.
 */
RCRE_NODISCARD int rcre_ml_emeans_recipe(const RCRE_Tensor* X, size_t k,
                                         const RCRE_EMMeansParams* params,
                                         RCRE_EMMeansRecipe* out);

/**
 * @brief Trains e-means on X with all-default parameters (T = nepochs).
 *        Serial execution; B/alpha0 from the scaling laws.
 * @param X [n,d] F32.
 * @param k Number of clusters (1 <= k <= n).
 * @param nepochs Epoch budget T (fractional supported, > 0, <= 16).
 * @param centroids Output [k,d] F32 (last-epoch-mean finalization).
 * @param labels Output [n,1] I32: last-epoch nearest-centroid id, or NULL.
 *               Rows not visited in the final epoch (fractional T < 1) get -1.
 * @param arena Scratch arena (permutation, batch stats, labels).
 * @returns RCRE_OK or an RCRE_ERR_* code.
 */
RCRE_NODISCARD int rcre_ml_emeans_fit(const RCRE_Tensor* X, size_t k, float nepochs,
                                      RCRE_Tensor* centroids, RCRE_Tensor* labels,
                                      RCRE_Arena* arena);

/**
 * @brief Trains e-means with full control (see RCRE_EMMeansParams).
 * @param X [n,d] F32 (n >= 1, d >= 1, n < 2^32).
 * @param k Number of clusters (1 <= k <= n).
 * @param params NULL = all defaults.
 * @param centroids Output [k,d] F32.
 * @param labels Output [n,1] I32 (F32 also accepted), or NULL.
 * @param out_inertia Optional: moving-pass mean squared distance of the
 *                    last epoch (FP64 internally; the exact final MSE needs
 *                    an rcre_ml_emeans_assign pass — the centroids still
 *                    move during the last epoch).
 * @param out_n_splits Optional: total dead-cluster revival splits performed.
 * @param arena Scratch arena.
 * @param pool Optional RCRE_ThreadPool (NULL = serial). Pooled runs are
 *             bit-identical across pool sizes; serial vs pooled is a
 *             quality-equivalence guarantee (see the header contract above).
 * @returns RCRE_OK or an RCRE_ERR_* code.
 */
RCRE_NODISCARD int rcre_ml_emeans_fit_ex(const RCRE_Tensor* X, size_t k,
                                         const RCRE_EMMeansParams* params,
                                         RCRE_Tensor* centroids, RCRE_Tensor* labels,
                                         float* out_inertia, int* out_n_splits,
                                         RCRE_Arena* arena,
                                         RCRE_ThreadPool* pool);

/**
 * @brief Exact nearest-centroid assignment of X against trained centroids.
 *        Dual kernel: per-row true distance for k*d <= 16384, HAL GEMM
 *        above. Costs roughly one epoch of FLOPs; use training labels
 *        instead when the budget is tight (see the reference's README note).
 * @param labels Output [n,1] I32 (F32 also accepted).
 * @param out_mse Optional: mean squared distance to the assigned centroid
 *                (FP64).
 * @returns RCRE_OK or an RCRE_ERR_* code.
 */
RCRE_NODISCARD int rcre_ml_emeans_assign(const RCRE_Tensor* X,
                                         const RCRE_Tensor* centroids,
                                         RCRE_Tensor* labels, float* out_mse,
                                         RCRE_Arena* arena);

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

typedef struct RCRE_RFNode {
    int32_t feature_idx;
    float   threshold;
    int32_t left;
    int32_t right;
    float   leaf_prob;
} RCRE_RFNode;

typedef struct RCRE_RandomForest {
    RCRE_RFNode* nodes;
    size_t*      tree_roots;
    size_t       num_trees;
    size_t       num_nodes;
    size_t       num_features;
    size_t       max_depth;
} RCRE_RandomForest;

RCRE_NODISCARD int rcre_ml_rf_fit(const RCRE_Tensor* X, const RCRE_Tensor* y,
                                  size_t num_trees, size_t max_depth, uint64_t seed,
                                  RCRE_Arena* arena, RCRE_RandomForest* out_rf);
RCRE_NODISCARD int rcre_ml_rf_predict(const RCRE_RandomForest* rf, const RCRE_Tensor* X, RCRE_Tensor* out_probs);

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
    RCRE_CPU_FEAT_WASM_SIMD   = (1 << 5),
    RCRE_CPU_FEAT_ARM_SVE2    = (1 << 6),
    RCRE_CPU_FEAT_ARM_DOTPROD = (1 << 7)
} RCRE_CpuFeatures;

typedef void (*rcre_gemm_fn)(size_t M, size_t N, size_t K, float alpha, 
                             const float* A, size_t lda, 
                             const float* B, size_t ldb, 
                             float beta, float* C, size_t ldc);

typedef void (*rcre_gemv_fn)(size_t M, size_t K, float alpha,
                             const float* A, size_t lda,
                             const float* x, float beta, float* y);

typedef float (*rcre_dot_fn)(const float* x, const float* y, size_t n);

typedef int32_t (*rcre_dot_i8_fn)(const int8_t* x, const int8_t* y, size_t n);

typedef void (*rcre_gemm_i8_fn)(size_t M, size_t N, size_t K,
                                const int8_t* A, size_t lda,
                                const int8_t* B, size_t ldb,
                                int32_t* C, size_t ldc);

typedef void (*rcre_gemm_bf16_fn)(size_t M, size_t N, size_t K, float alpha,
                                  const uint16_t* A, size_t lda,
                                  const uint16_t* B, size_t ldb,
                                  float beta, float* C, size_t ldc);

typedef void (*rcre_axpy_fn)(size_t n, float alpha, const float* x, float* y);

/* ---- H0 (2.1.0): 64-bit doubles + the HPC op matrix ---------------------
 * F64 is the scientific substrate: every algorithm that needs double
 * precision calls the dispatched f64 op and transparently gets scalar /
 * NEON (4-wide) / AVX2 (4-wide) / AVX-512 (8-wide) on the host. f32 gets
 * the same gemv/scale/sum cells so the op matrix is complete. */
typedef void (*rcre_gemm_f64_fn)(size_t M, size_t N, size_t K, double alpha,
                                 const double* A, size_t lda,
                                 const double* B, size_t ldb,
                                 double beta, double* C, size_t ldc);
typedef double (*rcre_dot_f64_fn)(const double* x, const double* y, size_t n);
typedef void (*rcre_axpy_f64_fn)(size_t n, double alpha, const double* x, double* y);
typedef void (*rcre_gemv_f64_fn)(size_t M, size_t N, double alpha,
                                 const double* A, size_t lda, const double* x,
                                 double beta, double* y);
typedef void (*rcre_scale_f32_fn)(size_t n, float alpha, float* x);
typedef void (*rcre_scale_f64_fn)(size_t n, double alpha, double* x);
typedef float (*rcre_sum_f32_fn)(const float* x, size_t n);
typedef double (*rcre_sum_f64_fn)(const double* x, size_t n);

typedef void (*rcre_rgb_to_gray_u8_fn)(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

typedef size_t (*rcre_find_byte_fn)(const uint8_t* hay, size_t len, uint8_t needle);

typedef size_t (*rcre_count_byte_fn)(const uint8_t* hay, size_t len, uint8_t needle);

typedef struct RCRE_CpuTopology {
    uint32_t logical_cores;         /* Host total online logical processors            */
    uint32_t perf_cores;            /* Asymmetric Performance (P) cores (Apple/hybrid) */
    uint32_t eff_cores;             /* Asymmetric Efficiency (E) cores (Apple/hybrid)  */
    uint32_t effective_quota_cores; /* Effective cgroup v2 / scheduler allowed cores   */
    uint64_t total_ram_bytes;       /* Total physical/unified memory in bytes          */
} RCRE_CpuTopology;

typedef struct RCRE_HalDispatch {
    const char*            backend_name;
    uint32_t               cpu_features;
    rcre_gemm_fn           gemm_f32;
    rcre_gemv_fn           gemv_f32;
    rcre_dot_fn            dot_f32;
    rcre_dot_i8_fn         dot_i8;
    rcre_gemm_i8_fn        gemm_i8;
    rcre_gemm_bf16_fn      gemm_bf16;
    rcre_axpy_fn           axpy_f32;
    rcre_rgb_to_gray_u8_fn rgb_to_gray_u8;
    rcre_find_byte_fn      find_byte;
    rcre_count_byte_fn     count_byte;
    /* H0 (2.1.0): HPC op matrix — f64 substrate + f32 completeness.
     * (gemv_f32 reuses main's v2.0 cell above — identical signature.) */
    rcre_gemm_f64_fn       gemm_f64;
    rcre_dot_f64_fn        dot_f64;
    rcre_axpy_f64_fn       axpy_f64;
    rcre_gemv_f64_fn       gemv_f64;
    rcre_scale_f32_fn      scale_f32;
    rcre_scale_f64_fn      scale_f64;
    rcre_sum_f32_fn        sum_f32;
    rcre_sum_f64_fn        sum_f64;
} RCRE_HalDispatch;

extern RCRE_HalDispatch rcre_hal;

/**
 * @brief Opt-in parallel SGEMM partitioned over RCRE_ThreadPool row slabs.
 *        Falls back to single-threaded rcre_hal.gemm_f32 when pool is NULL or small.
 */
void rcre_hal_gemm_f32_mt(RCRE_ThreadPool* pool,
                          size_t M, size_t N, size_t K, float alpha,
                          const float* A, size_t lda,
                          const float* B, size_t ldb,
                          float beta, float* C, size_t ldc);

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

/**
 * @brief Returns probed asymmetric core counts, cgroup CPU quota, and physical RAM.
 */
RCRE_CpuTopology rcre_cpu_get_topology(void);

/**
 * @brief Queries current OS memory pressure level (0 = normal, 1 = warn, 2 = critical).
 */
int rcre_cpu_get_memory_pressure(void);

/* ========================================================================= */
/* Apache Arrow C Data Interface (Standard Interoperability Structs)         */
/* ========================================================================= */

typedef struct RCRE_ArrowSchema {
    const char*                format;      /* "f" = F32, "i" = I32, "C" = U8 */
    const char*                name;
    const char*                metadata;
    int64_t                    flags;
    int64_t                    n_children;
    struct RCRE_ArrowSchema**  children;
    struct RCRE_ArrowSchema*   dictionary;
    void (*release)(struct RCRE_ArrowSchema*);
    void*                      private_data;
} RCRE_ArrowSchema;

typedef struct RCRE_ArrowArray {
    int64_t                   length;
    int64_t                   null_count;
    int64_t                   offset;
    int64_t                   n_buffers;
    int64_t                   n_children;
    const void**              buffers;     /* buffers[0]=validity, buffers[1]=data */
    struct RCRE_ArrowArray**  children;
    struct RCRE_ArrowArray*   dictionary;
    void (*release)(struct RCRE_ArrowArray*);
    void*                     private_data;
} RCRE_ArrowArray;

/* ========================================================================= */
/* HPC Numerical Layer (H0, 2.1.0)                                          */
/* ========================================================================= */

/**
 * @file section HPC Numerical Layer
 * @brief Double-precision substrate, compensated reductions, and the special
 *        function library that HPC scientific code is built on.
 *
 * Design (docs/hpc.md):
 *  - F64 is the scientific substrate: every op below exists for f32 AND f64,
 *    dispatched through `rcre_hal` (scalar / NEON / AVX2 / AVX-512).
 *  - Determinism: the dispatched reductions are single-threaded and therefore
 *    bitwise reproducible; the threadpool slab-reduction helpers below carry
 *    the library-wide determinism policy (fixed-order combine, e-means slab
 *    contract) with an explicit opt-in fast mode.
 *  - Exactness: `rcre_sum_neumaier_*` is the high-accuracy reference
 *    (<= 2 ULP error, order-dependent but deterministic); plain reductions
 *    are the performance path. Both are always available.
 */

/* ------------------------------- dispatched linalg ----------------------- */

/** C = alpha*A*B + beta*C (row-major, leading dims in elements), f32. */
void rcre_linalg_gemm_f32(size_t M, size_t N, size_t K, float alpha,
                          const float* A, size_t lda, const float* B, size_t ldb,
                          float beta, float* C, size_t ldc);
/** C = alpha*A*B + beta*C, f64 — the scientific workhorse. */
void rcre_linalg_gemm_f64(size_t M, size_t N, size_t K, double alpha,
                          const double* A, size_t lda, const double* B, size_t ldb,
                          double beta, double* C, size_t ldc);
/**
 * @brief BLIS-style blocked GEMM, f64 (H2, 2.2.0).
 *
 * C = alpha*A*B + beta*C, all row-major. One-sided (A-panel) blocking:
 * the loop order is (i-block, kk-block, j-block) so the A micropanel
 * A(i0:i0+BM, kk0:kk0+BK) stays hot while sweeping j; B and C blocks
 * are BM×BN tiles. The micro-panel calls go through the HAL gemm_f64
 * (ISA-dispatched), so the blocked structure composes with the vector
 * kernels. Deterministic (fixed loop order) — but the K accumulation
 * order differs from the unblocked rcre_linalg_gemm_f64, so results
 * may differ in a few ulps: both are in the float determinism class.
 */
void rcre_linalg_gemm_blocked_f64(size_t M, size_t N, size_t K, double alpha,
                                  const double* A, size_t lda, const double* B, size_t ldb,
                                  double beta, double* C, size_t ldc);
/** y = alpha*A*x + beta*y, f32 (A is MxN row-major). */
void rcre_linalg_gemv_f32(size_t M, size_t N, float alpha,
                          const float* A, size_t lda, const float* x,
                          float beta, float* y);
/** y = alpha*A*x + beta*y, f64. */
void rcre_linalg_gemv_f64(size_t M, size_t N, double alpha,
                          const double* A, size_t lda, const double* x,
                          double beta, double* y);
/** x = alpha*x + y (in-place on x), f32. */
void rcre_linalg_axpy_f32(size_t n, float alpha, const float* x, float* y);
/** x = alpha*x + y (in-place on x), f64. */
void rcre_linalg_axpy_f64(size_t n, double alpha, const double* x, double* y);
/** x *= alpha, f32. */
void rcre_linalg_scale_f32(size_t n, float alpha, float* x);
/** x *= alpha, f64. */
void rcre_linalg_scale_f64(size_t n, double alpha, double* x);
/** dot product x.y, f32. */
float  rcre_linalg_dot_f32(const float* x, const float* y, size_t n);
/** dot product x.y, f64. */
double rcre_linalg_dot_f64(const double* x, const double* y, size_t n);
/** sum(x), f32. */
float  rcre_linalg_sum_f32(const float* x, size_t n);
/** sum(x), f64. */
double rcre_linalg_sum_f64(const double* x, size_t n);

/**
 * @brief Triangular solve, f64 (H2, 2.2.0). Solve A*X = alpha*X in
 * place. A is n×n row-major triangular; uplo selects the triangle,
 * trans solves with A^T, diag may be unit (skips the division). X is
 * n×nrhs ROW-MAJOR (ldx = row stride; the RCRE convention — note
 * BLAS trsm is column-major). The H3 LU/QR machinery builds on this
 * primitive. Returns 0 on success, -1 on a zero diagonal (non-invertible).
 */
int rcre_linalg_trsm_f64(int uplo, int trans, int diag, double alpha,
                         const double* A, size_t lda,
                         double* X, size_t ldx, size_t n, size_t nrhs);
/** Single-vector triangular solve: A*x = alpha*x in place. */
int rcre_linalg_trsv_f64(int uplo, int trans, int diag, double alpha,
                         const double* A, size_t lda, double* x, size_t n);

/**
 * @brief General-size 1-D FFT, F64 (H2, 2.2.0).
 *
 * Any n >= 1: mixed-radix Cooley-Tukey for n with a small prime
 * factor (2,3,5,7), Bluestein chirp-z otherwise (primes, products of
 * large primes). Convention: forward X_k = sum_j x_j exp(-2*pi*i*j*k/n);
 * invert uses + and divides by n (numpy convention).
 * in/out may alias (copies are made). 0 on success, -1 on n == 0.
 */
RCRE_NODISCARD int rcre_fft_1d_f64(const double* in_re, const double* in_im,
                                   double* out_re, double* out_im,
                                   size_t n, int invert);

/* ------------------- direct factorizations (H3, 2.2.0) ------------------ */

/**
 * @brief LU factorization with partial pivoting, F64 (H3, 2.2.0).
 *
 * A (n×n row-major, in/out) becomes the packed LU: U in the upper
 * triangle, the unit multipliers of L below (L implicit, unit diagonal).
 * piv[k] = row swapped with k at step k; applying the piv[k] row-swaps
 * in increasing k order to a vector reproduces the row permutation P
 * (A = P·L·U). 0 on success, -1 if exactly singular (zero pivot).
 */
RCRE_NODISCARD int rcre_linalg_lu_factor_f64(double* A, size_t lda, size_t n,
                                             size_t* piv);
/**
 * @brief Solve A·X = B from a packed LU factor (getrs-class).
 * B (n×nrhs row-major, in/out); the row permutation is applied to B,
 * then the unit-lower forward solve and the upper backward solve.
 */
RCRE_NODISCARD int rcre_linalg_lu_solve_f64(const double* A, size_t lda,
                                            const size_t* piv, size_t n,
                                            double* B, size_t ldx, size_t nrhs);
/**
 * @brief One-shot dense solve A·X = B (A not modified; factor + solve).
 * -1 if A is singular (B may be modified in that case).
 */
RCRE_NODISCARD int rcre_linalg_solve_f64(const double* A, size_t lda, size_t n,
                                         double* B, size_t ldx, size_t nrhs);
/**
 * @brief In-place Cholesky factorization of an SPD matrix, F64 (H3).
 * A (n×n row-major, in/out) -> L (lower = 1) or L^T (lower = 0) packed
 * in the requested triangle including the diagonal.
 * 0 on success, -1 if a pivot is <= 0 (not positive definite).
 */
RCRE_NODISCARD int rcre_linalg_cholesky_f64(double* A, size_t lda, size_t n,
                                            int lower);
/**
 * @brief Solve A·X = B from a Cholesky factor (row-major, multi-RHS).
 * F is the packed L (lower = 1) or L^T (lower = 0); B in/out.
 */
RCRE_NODISCARD int rcre_linalg_cholesky_solve_f64(const double* F, size_t lda,
                                                  size_t n, double* B, size_t ldx,
                                                  size_t nrhs, int lower);
/**
 * @brief In-place Householder QR, F64 (H3): A (m×n row-major, m >= n,
 * in/out) -> R in the upper triangle; Householder vectors v_k (length
 * m-k, scaled with v_k[0] implicit = 1) in A[k+1:m, k]; tau (length
 * min(m,n), out) with H_k = I - tau_k·v_k·v_k^T. Q·R = A with
 * Q = H_0 H_1 ... H_{n-1} (m×m; first n columns span range(A)).
 */
RCRE_NODISCARD int rcre_linalg_qr_factor_f64(double* A, size_t lda, size_t m,
                                             size_t n, double* tau);
/**
 * @brief Explicit Q (m×m row-major out) from a packed QR factorization.
 * Q = H_0 ... H_{kmax-1}; each H_k updates ALL columns (rows k..m-1).
 */
void rcre_linalg_qr_q_f64(const double* A, size_t lda, size_t m, size_t n,
                          const double* tau, double* Q, size_t ldq);
/**
 * @brief Least squares min ||A·x - b|| per column of B via QR (H3).
 * A already factored (qr_factor_f64, m >= n, nonsingular R).
 * B (m×nrhs, in/out) becomes [x (rows 0..n-1); residual (rows n..m-1)].
 */
RCRE_NODISCARD int rcre_linalg_qr_lsolve_f64(const double* A, size_t lda,
                                             size_t m, size_t n, const double* tau,
                                             double* B, size_t ldx, size_t nrhs);
/**
 * @brief Symmetric eigendecomposition, F64 (H3, dsyev-class).
 *
 * A (n×n symmetric row-major, in/out — overwritten) -> w (n, out):
 * eigenvalues in ASCENDING order; if compute_vectors != 0, V (n×n
 * row-major, ldv, out): column j = unit eigenvector for w[j] (so
 * A·V = V·diag(w), Q-orthogonal V). Householder tridiagonalization +
 * Wilkinson-shift implicit QL. Deterministic. 0 on success, -1 on
 * QL non-convergence (practically unreachable for well-scaled input).
 */
RCRE_NODISCARD int rcre_linalg_syev_f64(double* A, size_t lda, size_t n,
                                        int compute_vectors, double* w,
                                        double* V, size_t ldv);

/* -------------------- sparse CSR + Krylov solvers (H4, 2.2.0) ----------- */

/**
 * @brief Compressed sparse row (CSR) matrix view, F64 (H4, 2.2.0).
 *
 * A plain C view: row_ptr (rows+1), col_idx (nnz, ascending per row),
 * values (nnz). The caller owns the three arrays (or gets them from
 * rcre_csr_f64_from_dense and frees them all together). Row-major
 * canonical layout; the Krylov and preconditioner APIs treat the
 * matrix as immutable.
 */
typedef struct RCRE_CSR_F64 {
    size_t rows;
    size_t cols;
    size_t nnz;
    size_t* row_ptr;
    size_t* col_idx;
    double* values;
} RCRE_CSR_F64;

/** Same as RCRE_CSR_F64 with F32 values. */
typedef struct RCRE_CSR_F32 {
    size_t rows;
    size_t cols;
    size_t nnz;
    size_t* row_ptr;
    size_t* col_idx;
    float* values;
} RCRE_CSR_F32;

/**
 * @brief Krylov convergence callback (H4).
 *
 * Called after each residual update with the CURRENT iteration count
 * and the current relative residual (true residual norm over the
 * reference norm, accumulated in double even on the F32 path).
 * Returning from the callback cannot abort the solve (documented:
 * the callback is telemetry only — profiling integration).
 */
typedef void (*RCRE_KrylovCallback)(size_t iter, double rel_res, void* user);

/** Preconditioner selection for the Krylov drivers (H4). */
enum {
    RCRE_PC_NONE = 0,           /* no preconditioning (pc_data ignored) */
    RCRE_PC_JACOBI = 1,         /* diagonal: pc_data = inv_diag (n)     */
    RCRE_PC_WEIGHTED_JACOBI = 2,/* d*Jacobi: pc_data = array w[i] = omega/|A_ii| */
    RCRE_PC_ILU0 = 3            /* sparse LU, no fill: pc_data = the RCRE_CSR factor */
};

/**
 * @brief Dense (row-major) to CSR conversion, F64 (H4).
 *
 * Keeps entries with |a| > threshold (threshold = 0 keeps every
 * nonzero). Allocates row_ptr (rows+1), col_idx (nnz), values (nnz);
 * on success the caller owns ALL THREE (free each). 0 on success, -1
 * on NULL output or allocation failure (*nnz out, = 0 on failure).
 */
RCRE_NODISCARD int rcre_csr_f64_from_dense(const double* A, size_t rows,
                                           size_t cols, double threshold,
                                           size_t** row_ptr,
                                           size_t** col_idx,
                                           double** values, size_t* nnz);

/** F32 twin of rcre_csr_f64_from_dense. */
RCRE_NODISCARD int rcre_csr_f32_from_dense(const float* A, size_t rows,
                                           size_t cols, float threshold,
                                           size_t** row_ptr,
                                           size_t** col_idx,
                                           float** values, size_t* nnz);

/** @brief SpMV: y = A·x (OVERWRITES y), F64 (H4). */
void rcre_spmv_f64(const RCRE_CSR_F64* A, const double* x, double* y);

/** F32 twin of rcre_spmv_f64. */
void rcre_spmv_f32(const RCRE_CSR_F32* A, const float* x, float* y);

/**
 * @brief Jacobi preconditioner: inv_diag[i] = 1/A[i,i] (H4).
 * 0 on success, -1 on NULL or a zero diagonal entry.
 */
RCRE_NODISCARD int rcre_pc_jacobi_f64(const RCRE_CSR_F64* A, double* inv_diag);

/** F32 twin of rcre_pc_jacobi_f64. */
RCRE_NODISCARD int rcre_pc_jacobi_f32(const RCRE_CSR_F32* A, float* inv_diag);

/**
 * @brief Incomplete LU factorization, zero fill (ILU(0)), F64 (H4).
 *
 * Factors A = L·U following EXACTLY A's sparsity pattern (unit-
 * diagonal L implicit). The factor is returned in `fac`: fac->row_ptr
 * and fac->col_idx are NEW copies of A's pattern, fac->values holds
 * L (strictly lower) and U (diagonal + upper). A is not modified.
 * 0 on success (caller owns all three fac arrays), -1 on NULL, a
 * missing/zero diagonal, or a zero pivot during the sweep.
 */
RCRE_NODISCARD int rcre_pc_ilu0_f64(const RCRE_CSR_F64* A, RCRE_CSR_F64* fac);

/** F32 twin of rcre_pc_ilu0_f64. */
RCRE_NODISCARD int rcre_pc_ilu0_f32(const RCRE_CSR_F32* A, RCRE_CSR_F32* fac);

/**
 * @brief Apply an ILU(0) preconditioner: z = (L·U)⁻¹·r (H4).
 * Two sparse triangular solves (forward with unit L, then backward).
 */
void rcre_pc_ilu0_solve_f64(const RCRE_CSR_F64* fac, const double* r, double* z);

/** F32 twin of rcre_pc_ilu0_solve_f64. */
void rcre_pc_ilu0_solve_f32(const RCRE_CSR_F32* fac, const float* r, float* z);

/**
 * @brief Preconditioned conjugate gradients for SPD A·x = b, F64 (H4).
 *
 * x is the initial guess AND the output (x += ... in place). tol is
 * the target on the TRUE relative residual ||b - A·x|| / ||b||.
 * max_iter bounds the outer iterations. pc selects the preconditioner
 * (see RCRE_PC_*); pc_data is the inv_diag buffer or the ILU(0)
 * factor built by the corresponding rcre_pc_* call. fcg != 0 selects
 * FCG (apply M⁻¹ to the residual only at the first step; the plain
 * call is preconditioned CG with M⁻¹ on r every step). cb is optional
 * telemetry (NULL allowed), called after each iteration with the
 * current relative residual.
 *
 * Returns the number of outer iterations performed (0 when x0
 * already satisfies tol), -1 when tol was not reached within
 * max_iter (x still holds the last iterate). Deterministic: fixed
 * loop order, no threading, bitwise reproducible run-to-run.
 */
RCRE_NODISCARD int rcre_krylov_pcg_f64(const RCRE_CSR_F64* A, const double* b,
                                       double* x, double tol,
                                       size_t max_iter, int pc,
                                       const void* pc_data, int fcg,
                                       RCRE_KrylovCallback cb, void* user);

/** F32 twin of rcre_krylov_pcg_f64 (tol still on the double-accumulated relative residual). */
RCRE_NODISCARD int rcre_krylov_pcg_f32(const RCRE_CSR_F32* A, const float* b,
                                       float* x, double tol,
                                       size_t max_iter, int pc,
                                       const void* pc_data, int fcg,
                                       RCRE_KrylovCallback cb, void* user);

/**
 * @brief BiCGStab for general (non-symmetric) A·x = b, F64 (H4).
 * Same conventions as rcre_krylov_pcg_f64 (x in/out, tol on the true
 * relative residual, pc/pc_data, optional telemetry callback).
 * Returns the number of outer iterations or -1.
 */
RCRE_NODISCARD int rcre_krylov_bicgstab_f64(const RCRE_CSR_F64* A,
                                            const double* b, double* x,
                                            double tol, size_t max_iter,
                                            int pc, const void* pc_data,
                                            RCRE_KrylovCallback cb,
                                            void* user);

/** F32 twin of rcre_krylov_bicgstab_f64. */
RCRE_NODISCARD int rcre_krylov_bicgstab_f32(const RCRE_CSR_F32* A,
                                            const float* b, float* x,
                                            double tol, size_t max_iter,
                                            int pc, const void* pc_data,
                                            RCRE_KrylovCallback cb,
                                            void* user);


/**
 * @brief Restored GMRES(m) with Householder QR on the Hessenberg,
 * F64 (H4). Same conventions as rcre_krylov_pcg_f64; m is the inner
 * restart length (1 <= m < n; use m = n-1 for the full method).
 * Arnoldi with one MGS re-orthogonalization pass per step (the
 * measured-orthogonality bar is gated in the tests). Convergence
 * check uses the Arnoldi residual (exact in exact arithmetic: it
 * equals the true residual norm) and the final accepted iterate is
 * re-checked against the true residual before returning success.
 * Returns the total number of matrix-vector products (sum over all
 * restart cycles) or -1.
 */
RCRE_NODISCARD int rcre_krylov_gmres_f64(const RCRE_CSR_F64* A,
                                         const double* b, double* x,
                                         double tol, size_t max_iter,
                                         size_t m, int pc,
                                         const void* pc_data,
                                         RCRE_KrylovCallback cb, void* user);

/** F32 twin of rcre_krylov_gmres_f64. */
RCRE_NODISCARD int rcre_krylov_gmres_f32(const RCRE_CSR_F32* A,
                                         const float* b, float* x,
                                         double tol, size_t max_iter,
                                         size_t m, int pc,
                                         const void* pc_data,
                                         RCRE_KrylovCallback cb,
                                         void* user);

/**
 * @brief Classic geometric multigrid V-cycle, 2D Poisson (Dirichlet),
 *        structured tensor-product grid, F64 (H4).
 *
 * One residual-correction V-cycle for the h^2-scaled 5-point Laplacian
 * (A u)_{i,j} = 4u_{i,j} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1} on
 * the interior grid 0..nx-1 x 0..ny-1 with zero boundary values (fold
 * boundary data into f). Anatomy per level: pre/post red-black
 * Gauss-Seidel sweeps (pre/post counts); defect r = f - A u; full-weighting
 * restriction of the defect; bilinear prolongation of the coarse
 * correction; exact dense (Gaussian) solve at the coarsest level. x is the
 * approximation, updated in place. Measured behavior (docs/hpc.md):
 * broadband residuals reach 1e-10 in ~20-60 cycles on 32^2..512^2, while a
 * pure smoothest-mode error decays slowly (rate -> 1 with n; fixed
 * 2+2 smoothing is the limit, not a bug) — for smooth-error-dominated
 * solves use rcre_krylov_pcg + RCRE_PC_ILU0. Returns 1 on success, -1 on
 * invalid input/alloc failure.
 */
RCRE_NODISCARD int rcre_mg_vcycle_poisson2d_f64(size_t nx, size_t ny,
                                                const double* f, double* x,
                                                int pre, int post);
RCRE_NODISCARD int rcre_mg_vcycle_poisson2d_f32(size_t nx, size_t ny,
                                                const float* f, float* x,
                                                int pre, int post);

/**
 * @brief Geometric multigrid FAS cycle, 2D Poisson (Dirichlet), structured
 *        tensor-product grid, F64 (H4).
 *
 * One Full Approximation Scheme cycle for the same grid/stencil as
 * rcre_mg_vcycle_poisson2d_f64: pre/post red-black Gauss-Seidel sweeps;
 * defect r = f - A u; exact dense correction at the coarsest level; the
 * coarse level solves against the full restricted RHS R f starting from
 * the full-weighting of u, and the fine update is
 * u += P (u_c_after - u_c_before). FAS corrects against the full RHS at
 * every level (not just the residual), which changes the smoothest-mode
 * behavior compared with the V-cycle; both are provided so users can pick
 * per problem. Returns 1 on success, -1 on invalid input/alloc failure.
 */
RCRE_NODISCARD int rcre_mg_fas_poisson2d_f64(size_t nx, size_t ny,
                                             const double* f, double* x,
                                             int pre, int post);
RCRE_NODISCARD int rcre_mg_fas_poisson2d_f32(size_t nx, size_t ny,
                                             const float* f, float* x,
                                             int pre, int post);

/**
 * @brief Repeat V-cycles until the true relative residual ||f - A x||/||f||
 *        <= tol, F64. Same grid convention as rcre_mg_vcycle_poisson2d_f64.
 * Cycles use 2 pre + 2 post red-black GS sweeps. Returns the number of
 * cycles applied, 0 if already converged, -1 on failure (max_cycles is a
 * hard cap). The callback (telemetry only) receives (cycle, rel_res).
 */
RCRE_NODISCARD int rcre_mg_solve_poisson2d_f64(size_t nx, size_t ny,
                                               const double* f, double* x,
                                               double tol, int max_cycles,
                                               RCRE_KrylovCallback cb,
                                               void* user);
RCRE_NODISCARD int rcre_mg_solve_poisson2d_f32(size_t nx, size_t ny,
                                               const float* f, float* x,
                                               double tol, int max_cycles,
                                               RCRE_KrylovCallback cb,
                                               void* user);

/* --------------------------- compensated reductions ---------------------- */

/**
 * @brief Neumaier-compensated sum (high-accuracy, deterministic).
 *
 * Two-accumulator Kahan-Neumaier scheme: error bound ~2*eps*|sum| +
 * O(eps^2) independent of n, in contrast to the naive O(n*eps) of plain
 * sequential accumulation. This is the library's reference reduction —
 * the e-means FP64 objective sums, solver residual checks, and the H4
 * convergence gates are all built on it. Order-dependent (like any
 * deterministic reduction) but bitwise reproducible run-to-run.
 */
double rcre_sum_neumaier_f64(const double* x, size_t n);
float  rcre_sum_neumaier_f32(const float* x, size_t n);

/**
 * @brief Neumaier-compensated dot product (high-accuracy, deterministic).
 */
double rcre_dot_neumaier_f64(const double* x, const double* y, size_t n);
float  rcre_dot_neumaier_f32(const float* x, const float* y, size_t n);

/**
 * @brief Two-way (recursive) split sum — the fast error-floor reference.
 *
 * Splits the array in halves recursively (depth ~log2(n)) so each FP add
 * combines two equal-magnitude partials; error ~log2(n) ulps for
 * well-scaled data (measured: bit-exact on 10^6 * (1e-6)), and exactly
 * -1 ulp on the adversarial 1e20 + 2^20*1 layout (vs ~-64 ulps naive,
 * 0 for Neumaier). Deterministic — also the slab-combine model for the
 * threadpool reductions (each slab sums Neumaier-ly, partials combine
 * in fixed slab order).
 */
double rcre_sum_pairwise_f64(const double* x, size_t n);

/* ------------------------------ special functions ------------------------ */

/**
 * @file section Special Functions (H0)
 * @brief C99-style special functions for statistics and physics.
 *
 * Accuracy gates (tests/test_linalg.c, against published values):
 *   - erf/erfc:  <= 1.2e-15 absolute (Cody rational + Chebyshev tails)
 *   - lgamma:    <= 1 ULP for x > 0.5 (Lanczos g=7, 9 coefficients),
 *                reflection below
 *   - digamma:   <= 2 ULP for x > 4 (asymptotic + recurrence),
 *                reflection-based below
 *   - logsumexp / expit: algebraically stable (no overflow in exp)
 */

/** Error function erf(x). |erf| <= 1 for all x; erf(-x) = -erf(x). */
double rcre_sf_erf_f64(double x);
float  rcre_sf_erf_f32(float x);
/** Complementary error function erfc(x) = 1 - erf(x) (tail-accurate). */
double rcre_sf_erfc_f64(double x);
/** Natural log of the absolute Gamma function: lgamma(x) = ln|Gamma(x)|. */
double rcre_sf_lgamma_f64(double x);
/** Digamma psi(x) = d/dx ln Gamma(x). Pole at 0, -1, -2, ... */
double rcre_sf_digamma_f64(double x);
/**
 * @brief Numerically stable log-sum-exp: log(sum(exp(x_i))).
 * Shifts by the max, so no overflow for large |x|.
 */
double rcre_sf_logsumexp_f64(const double* x, size_t n);
float  rcre_sf_logsumexp_f32(const float* x, size_t n);
/** Sigmoid expit(x) = 1/(1+exp(-x)), branch-free stable form. */
double rcre_sf_expit_f64(double x);
float  rcre_sf_expit_f32(float x);

/**
 * @brief Standard normal CDF Phi(x) = 0.5*erfc(-x/sqrt(2)).
 * The single most-used special function in statistics; provided as a
 * named op so tests can gate it directly.
 */
double rcre_sf_norm_cdf_f64(double x);

/* ========================================================================= */
/* H1 (2.2.0) — Reproducible Random & Quasi-Random Streams                  */
/* =========================================================================
 *
 * Design (docs/hpc.md):
 *  - PCG64 is the counter-based workhorse: 128-bit LCG state, stream is a
 *    pure function of its seed; bit-exact conformance with numpy PCG64
 *    (the MIT-licensed reference) is checked in tests/rng_fixtures/.
 *  - Substreams follow the slab determinism contract: worker i derives
 *    its stream from (base, i), so pooled runs are bit-identical across
 *    pool sizes and machines.
 *  - Sobol' (d = 1..20, new Joe-Kuo direction numbers) and Halton
 *    (bases 2,3,5) are the quasi-Monte Carlo paths; both are stateless
 *    random-access (point n is a pure function of n), which is what
 *    out-of-core and slab-parallel code needs.
 */

/* PCG64 state (128-bit state + 128-bit odd increment). */
typedef struct RCRE_RNG {
    uint64_t s_hi, s_lo;   /* 128-bit LCG state: hi << 64 | lo */
    uint64_t i_hi, i_lo;   /* 128-bit increment (odd): hi << 64 | lo */
} RCRE_RNG;

/**
 * @brief Seed from four 64-bit parts — numpy-compatible.
 *
 * With s0..s3 = SeedSequence(seed).generate_state(4), this produces the
 * exact state/inc of np.random.PCG64(seed); the raw stream is bit-identical
 * to random_raw. See tests/rng_fixtures/ for the checked-in oracle.
 */
void rcre_rng_pcg_init_parts(RCRE_RNG* r,
                             uint64_t s0, uint64_t s1,
                             uint64_t s2, uint64_t s3);

/** @brief Seed from a single 64-bit seed (RCRE-native derivation, splitmix64). */
void rcre_rng_pcg_init(RCRE_RNG* r, uint64_t seed);

/** @brief Next raw 64-bit output (step + XSL-RR output). */
uint64_t rcre_rng_pcg_next_u64(RCRE_RNG* r);
/** @brief Next 32 bits (top word of a raw draw). */
uint32_t rcre_rng_pcg_next_u32(RCRE_RNG* r);
/** @brief Next uniform double in [0,1): (u >> 11) * 2^-53 (numpy mapping). */
double rcre_rng_pcg_next_f64(RCRE_RNG* r);
/** @brief Next uniform float in [0,1): top 24 bits. */
float rcre_rng_pcg_next_f32(RCRE_RNG* r);

/**
 * @brief Deterministic substream for worker `id` of base RNG.
 *
 * Slab determinism contract: the substream depends only on (base,
 * id) — never on pool size or core count — so pooled results are
 * bit-identical across machines. Distinct ids give statistically
 * independent streams (chi-square gated in tests).
 */
void rcre_rng_pcg_substream(const RCRE_RNG* base, uint64_t id, RCRE_RNG* out);

/** @brief Fast-forward the state by n steps (O(popcount(n)) via the jump table). */
void rcre_rng_pcg_skip(RCRE_RNG* r, uint64_t n);

/* Quasi-random (low-discrepancy) state — point counters for the
 * stateless-at API; stateless code should call *_at directly. */
typedef struct RCRE_QRNG {
    uint64_t n;   /* number of points already emitted (0-based next index) */
    int      d;   /* active dimension count (clamped per sequence) */
} RCRE_QRNG;

#define RCRE_QRNG_SOBOL_DIM  20   /* shipped Joe-Kuo direction numbers */
#define RCRE_QRNG_HALTON_DIM 3    /* bases 2, 3, 5 */

/** @brief Sobol' point x_n (n >= 0, x_0 = 0), d components, bit-exact
 *  scipy 1.18.1 qmc.Sobol(scramble=False) for n < 2^20 (shipped depth). */
void rcre_qrng_sobol_at(uint64_t n, int d, double* out);
/** @brief Stateful iteration: init, then point() advances n. */
void rcre_qrng_sobol_init(RCRE_QRNG* q, int d);
void rcre_qrng_sobol_point(RCRE_QRNG* q, double* out);

/** @brief Halton point (n is 0-based; point n+1), k components. */
void rcre_qrng_halton_at(uint64_t n, int k, double* out);
void rcre_qrng_halton_init(RCRE_QRNG* q, int k);
void rcre_qrng_halton_point(RCRE_QRNG* q, double* out);

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

#ifndef RCRE_BUILDING_FROM_SUBHEADER
/* ========================================================================= */
/* High-Performance Scientific Data I/O & Zero-Copy Ecosystem Bridges         */
/* ========================================================================= */
#ifndef RCRE_NO_IO
#include "rcre_io.h"
#endif

/* ========================================================================= */
/* Modern LLM Architecture, GGUF Reader, Q8_0/Q4_0 Quantization & LoRA       */
/* ========================================================================= */
#ifndef RCRE_NO_LLM
#include "rcre_llm.h"
#endif

/* ========================================================================= */
/* Zero-Dependency Image Codecs & Minimal ONNX Protobuf Inference            */
/* ========================================================================= */
#ifndef RCRE_NO_CODEC
#include "rcre_codec.h"
#endif
#endif /* !RCRE_BUILDING_FROM_SUBHEADER */

/* ========================================================================= */
/* High-Performance Columnar DataFrame Engine (rcre_df)                      */
/* ========================================================================= */
#ifndef RCRE_NO_DF
#include "rcre_df.h"
#endif

/* ========================================================================= */
/* Exploratory Data Analysis & Statistics (rcre_stat)                        */
/* ========================================================================= */
#ifndef RCRE_NO_STAT
#include "rcre_stat.h"
#endif

/* ========================================================================= */
/* Feature Engineering & Data Preprocessing (rcre_prep)                      */
/* ========================================================================= */
#ifndef RCRE_NO_PREP
#include "rcre_prep.h"
#endif

/* ========================================================================= */
/* Model Evaluation Metrics & Validation Protocol (rcre_metric)              */
/* ========================================================================= */
#ifndef RCRE_NO_METRIC
#include "rcre_metric.h"
#endif

/* ========================================================================= */
/* Data Science Terminal & ASCII Visualizations (rcre_vis_ds)                */
/* ========================================================================= */
#ifndef RCRE_NO_VIS_DS
#include "rcre_vis_ds.h"
#endif

#ifdef __cplusplus
}
#endif

#endif /* RCRE_H */
