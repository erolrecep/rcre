/**
 * @file rcre_io.h
 * @brief RCRE High-Performance Scientific Data I/O, Zero-Copy Memory Mapping,
 *        SIMD Tabular Ingestion, Standard Tensor Containers (.npy / .safetensors),
 *        SIMD Byte-Shuffle Block Compression, Out-of-Core Ping-Pong Streaming,
 *        and Zero-Copy Cross-Framework ABI Bridges (DLPack v1.0 / Apache Arrow).
 *
 * Part of Recep's Code Run Everywhere (RCRE).
 * Strict ISO C23 compliant, zero external dependencies.
 */

#ifndef RCRE_IO_H
#define RCRE_IO_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Virtual Memory Mapping & OS Page-Cache Control (IO-REQ-01)             */
/* ========================================================================= */

typedef enum {
    RCRE_IO_HINT_DEFAULT    = 0, /* Standard OS page cache behavior               */
    RCRE_IO_HINT_SEQUENTIAL = 1, /* Aggressive sequential prefetch (MADV_WILLNEED)*/
    RCRE_IO_HINT_RANDOM     = 2, /* Random access (disable readahead thrashing)   */
    RCRE_IO_HINT_STREAMING  = 3  /* Out-of-core streaming (evict/bypass cache)    */
} RCRE_IOHint;

typedef struct RCRE_MappedFile {
    const uint8_t* data;         /* Page-aligned (4KB/16KB) read-only base ptr    */
    size_t         size;         /* Exact file size in bytes                      */
    intptr_t       os_handle;    /* Underlying OS descriptor/handle (-1 if mem)   */
    RCRE_IOHint    hint;         /* Active kernel paging hint                     */
    bool           is_mapped;    /* True if backed by OS mmap / heap fallback     */
    bool           is_heap_buf;  /* True if fallback heap buffer (non-mmap OS)    */
} RCRE_MappedFile;

/**
 * @brief Maps a file from disk into virtual memory with zero heap allocations.
 * @param filepath Source file path on disk.
 * @param hint OS page-cache advisory policy (sequential, random, streaming).
 * @param out_map Output mapped file descriptor.
 * @return RCRE_OK on success, RCRE_ERR_INVALID_ARG or RCRE_ERR_IO on failure.
 */
RCRE_NODISCARD int rcre_io_mmap_open(
    const char* filepath,
    RCRE_IOHint hint,
    RCRE_MappedFile* out_map
);

/**
 * @brief Wraps an existing in-memory buffer (e.g., WASM ArrayBuffer) as a mapped file.
 * @param buf Read-only memory pointer.
 * @param size Byte length of buffer.
 * @param out_map Output mapped file descriptor.
 * @return RCRE_OK on success, RCRE_ERR_INVALID_ARG on NULL/empty buffer.
 */
RCRE_NODISCARD int rcre_io_mmap_from_buffer(
    const void* buf,
    size_t size,
    RCRE_MappedFile* out_map
);

/**
 * @brief Releases the virtual memory mapping and closes the file descriptor.
 * @param map Mapped file descriptor to close.
 */
void rcre_io_mmap_close(RCRE_MappedFile* map);

/**
 * @brief Returns a zero-copy RCRE_Slice spanning the entire mapped file.
 * @param map Active mapped file descriptor.
 * @return RCRE_Slice viewing the mapped bytes, or empty slice if invalid.
 */
RCRE_NODISCARD RCRE_Slice rcre_io_mmap_slice(const RCRE_MappedFile* map);

/**
 * @brief Advises the OS kernel on upcoming access patterns for a byte sub-range.
 * @param map Active mapped file descriptor.
 * @param offset Starting byte offset within the file.
 * @param length Number of bytes in the sub-range.
 * @param hint Desired paging policy for the sub-range.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_madvise_range(
    const RCRE_MappedFile* map,
    size_t offset,
    size_t length,
    RCRE_IOHint hint
);

/**
 * @brief Atomically writes a contiguous byte buffer to disk via temp-file rename.
 * @param filepath Destination file path.
 * @param data Source byte buffer.
 * @param size Number of bytes to write.
 * @return RCRE_OK on success, RCRE_ERR_IO on failure.
 */
RCRE_NODISCARD int rcre_io_write_atomic(
    const char* filepath,
    const void* data,
    size_t size
);

/* ========================================================================= */
/* 2. Fast Floating-Point & SIMD Tabular CSV/TSV Ingestion (IO-REQ-02)       */
/* ========================================================================= */

/**
 * @brief Parses an IEEE-754 single-precision float from a byte slice using
 *        the Clinger / Eisel-Lemire fast path (zero heap, locale-independent).
 * @param s Input byte slice containing ASCII floating-point representation.
 * @param out_val Output parsed float32 value.
 * @return true if valid float parsed, false on malformed input.
 */
RCRE_NODISCARD bool rcre_slice_parse_f32(RCRE_Slice s, float* out_val);

typedef struct RCRE_CSVOptions {
    uint8_t          delimiter;      /* ',' or '\t', or 0 for auto (',' or '\t') */
    bool             skip_header;    /* Skip first row if true                   */
    bool             strict_numbers; /* Reject non-numeric tokens if true        */
    size_t           max_rows;       /* 0 = read all rows                        */
    RCRE_ThreadPool* pool;           /* Optional threadpool for parallel parsing */
} RCRE_CSVOptions;

/**
 * @brief High-throughput memory-mapped & SIMD-accelerated CSV/TSV loader.
 * @param filepath Path to CSV/TSV file.
 * @param opts Parsing options (delimiter, skip_header, max_rows, threadpool).
 * @param arena Arena to allocate the output 2D F32 tensor.
 * @param out_tensor Output 2D F32 tensor [rows, cols].
 * @return RCRE_OK on success, RCRE_ERR_IO or RCRE_ERR_OUT_OF_MEMORY on failure.
 */
RCRE_NODISCARD int rcre_io_load_csv_ex(
    const char* filepath,
    const RCRE_CSVOptions* opts,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
);

/**
 * @brief Parses a CSV/TSV directly from an in-memory RCRE_Slice (zero file I/O).
 * @param csv_slice In-memory byte slice containing CSV/TSV text.
 * @param opts Parsing options.
 * @param arena Arena to allocate the output 2D F32 tensor.
 * @param out_tensor Output 2D F32 tensor [rows, cols].
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_parse_csv_slice(
    RCRE_Slice csv_slice,
    const RCRE_CSVOptions* opts,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
);

/* ========================================================================= */
/* 3. NumPy (.npy v1.0 / v2.0) Serialization & Zero-Copy Views (IO-REQ-03)   */
/* ========================================================================= */

/**
 * @brief Serializes an RCRE_Tensor to a NumPy .npy file with strict 64-byte
 *        header padding so the payload is always 64-byte aligned on disk.
 * @param tensor Contiguous 1D-4D tensor (U8, I32, or F32) to serialize.
 * @param filepath Destination .npy file path.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_save_npy(
    const RCRE_Tensor* tensor,
    const char* filepath
);

/**
 * @brief Loads a NumPy .npy file (v1.0 or v2.0) into a 64-byte aligned arena buffer.
 * @param filepath Source .npy file path.
 * @param arena Destination memory arena.
 * @param out_tensor Output tensor allocated inside arena.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_load_npy(
    const char* filepath,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
);

/**
 * @brief Creates a zero-copy RCRE_Tensor view (is_view = true) directly over
 *        an open RCRE_MappedFile containing a 64-byte aligned .npy file.
 * @param map Open memory-mapped .npy file (must outlive the returned view).
 * @param out_view Output zero-copy tensor view.
 * @return RCRE_OK on success, RCRE_ERR_INVALID_ARG if payload is not 64B-aligned.
 */
RCRE_NODISCARD int rcre_io_view_npy(
    const RCRE_MappedFile* map,
    RCRE_Tensor* out_view
);

/* ========================================================================= */
/* 4. Multi-Tensor Safetensors (.safetensors) Container Engine (IO-REQ-04)   */
/* ========================================================================= */

#define RCRE_SAFETENSORS_MAX_TENSORS 256

typedef struct RCRE_SafeTensorEntry {
    RCRE_Slice  name;          /* Zero-copy slice into JSON header               */
    RCRE_DType  dtype;         /* U8, I32, or F32                                */
    uint8_t     ndim;          /* Rank (1 to 4)                                  */
    size_t      shape[4];      /* Dimensions                                     */
    size_t      data_offset;   /* Relative byte offset from data section start   */
    size_t      data_nbytes;   /* Validated byte length of payload               */
} RCRE_SafeTensorEntry;

typedef struct RCRE_SafeTensorsArchive {
    RCRE_MappedFile      map;
    size_t               header_nbytes;    /* JSON header size N                 */
    size_t               data_base_offset; /* 8 + N (64-byte aligned by RCRE)    */
    size_t               num_tensors;
    RCRE_SafeTensorEntry entries[RCRE_SAFETENSORS_MAX_TENSORS];
} RCRE_SafeTensorsArchive;

/**
 * @brief Opens and validates a .safetensors archive via zero-copy mmap.
 * @param filepath Path to .safetensors file.
 * @param out_archive Output parsed archive handle.
 * @return RCRE_OK on success, RCRE_ERR_IO on corrupt/invalid archive.
 */
RCRE_NODISCARD int rcre_io_safetensors_open(
    const char* filepath,
    RCRE_SafeTensorsArchive* out_archive
);

/**
 * @brief Opens and validates a .safetensors archive from an in-memory buffer.
 * @param buf Pointer to .safetensors bytes in memory.
 * @param size Total byte length of buffer.
 * @param out_archive Output parsed archive handle.
 * @return RCRE_OK on success, RCRE_ERR_IO on corrupt/invalid archive.
 */
RCRE_NODISCARD int rcre_io_safetensors_open_from_buffer(
    const void* buf,
    size_t size,
    RCRE_SafeTensorsArchive* out_archive
);

/**
 * @brief Closes a mapped .safetensors archive and releases OS virtual pages.
 * @param archive Archive handle to close.
 */
void rcre_io_safetensors_close(RCRE_SafeTensorsArchive* archive);

/**
 * @brief Retrieves a tensor by name from an open .safetensors archive.
 *        Returns a zero-copy view (is_view = true) whenever the tensor payload
 *        is 64-byte aligned in memory; otherwise copies into fallback_arena.
 * @param archive Open .safetensors archive.
 * @param name Null-terminated tensor name to look up.
 * @param fallback_arena Optional arena used only if file offset is not 64B-aligned.
 * @param out_tensor Output tensor (zero-copy view or arena-copied).
 * @return RCRE_OK on success, RCRE_ERR_INVALID_ARG if not found or unaligned without arena.
 */
RCRE_NODISCARD int rcre_io_safetensors_get_tensor(
    const RCRE_SafeTensorsArchive* archive,
    const char* name,
    RCRE_Arena* fallback_arena,
    RCRE_Tensor* out_tensor
);

/**
 * @brief Saves an array of named RCRE_Tensors to a 64-byte aligned .safetensors file.
 * @param filepath Destination .safetensors file path.
 * @param names Array of tensor name strings [num_tensors].
 * @param tensors Array of contiguous RCRE_Tensors [num_tensors].
 * @param num_tensors Number of tensors to serialize (1 to 256).
 * @param scratch Transient scratch arena for header construction.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_safetensors_save(
    const char* filepath,
    const char* const* names,
    const RCRE_Tensor* tensors,
    size_t num_tensors,
    RCRE_Arena* scratch
);

/* ========================================================================= */
/* 5. SIMD Byte-Shuffle Codec & Out-of-Core Ping-Pong Streamer (IO-REQ-05)   */
/* ========================================================================= */

/**
 * @brief Applies a SIMD-accelerated byte-shuffle transpose on 32-bit elements
 *        (F32/I32), grouping byte 0, byte 1, byte 2, and byte 3 into contiguous
 *        streams to maximize downstream compression ratio and throughput.
 * @param src Source array of num_elements 32-bit values (4 * num_elements bytes).
 * @param dst Destination buffer (must not alias src, 4 * num_elements bytes).
 * @param num_elements Number of 32-bit elements.
 */
void rcre_io_byte_shuffle_32(const void* src, void* dst, size_t num_elements);

/**
 * @brief Reconstructs interleaved 32-bit elements from a byte-shuffled buffer.
 * @param src Byte-shuffled source buffer (4 * num_elements bytes).
 * @param dst Destination array of num_elements 32-bit values.
 * @param num_elements Number of 32-bit elements.
 */
void rcre_io_byte_unshuffle_32(const void* src, void* dst, size_t num_elements);

/**
 * @brief Compresses a contiguous 32-bit tensor (F32/I32) using SIMD byte-shuffle
 *        followed by a zero-dependency LZ4-class run/match block compressor.
 * @param tensor Contiguous F32 or I32 source tensor.
 * @param arena Destination arena to allocate compressed byte buffer.
 * @param out_slice Output slice containing the self-describing compressed frame.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_compress_tensor(
    const RCRE_Tensor* tensor,
    RCRE_Arena* arena,
    RCRE_Slice* out_slice
);

/**
 * @brief Decompresses a byte-shuffled compressed tensor frame into an arena.
 * @param compressed_frame Slice produced by rcre_io_compress_tensor.
 * @param arena Arena to allocate the reconstructed 64-byte aligned tensor.
 * @param out_tensor Output reconstructed tensor (0-ULP bit-exact).
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_decompress_tensor(
    RCRE_Slice compressed_frame,
    RCRE_Arena* arena,
    RCRE_Tensor* out_tensor
);

typedef struct RCRE_TensorStream {
    RCRE_MappedFile map;
    RCRE_DType      dtype;
    RCRE_Layout     layout;
    uint8_t         ndim;
    size_t          full_shape[4];
    size_t          row_elements;    /* Product of shape[1..ndim-1]              */
    size_t          row_bytes;       /* row_elements * sizeof(dtype)             */
    size_t          total_rows;      /* full_shape[0]                            */
    size_t          batch_rows;      /* Maximum rows per streamed batch          */
    size_t          current_row;     /* Next row cursor                          */
    size_t          data_offset;     /* Byte offset to row 0 in file             */
    uint8_t*        ping_pong_buf[2];/* Dual 64B-aligned arena buffers           */
    uint8_t         active_slot;     /* Toggles 0 <-> 1 on each batch            */
} RCRE_TensorStream;

/**
 * @brief Opens an out-of-core ping-pong batch streamer over a .npy tensor file.
 * @param filepath Source .npy file path.
 * @param batch_rows Number of outer-dimension rows per batch.
 * @param arena Arena from which two ping-pong batch buffers are sub-allocated.
 * @param out_stream Output initialized tensor stream.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_stream_open_npy(
    const char* filepath,
    size_t batch_rows,
    RCRE_Arena* arena,
    RCRE_TensorStream* out_stream
);

/**
 * @brief Fetches the next batch from an open RCRE_TensorStream into a 64-byte
 *        aligned ping-pong buffer and advises the OS kernel to evict consumed pages.
 * @param stream Active tensor stream.
 * @param out_batch Output tensor view wrapping the active ping-pong buffer.
 * @return true if a batch was read, false when end-of-stream is reached.
 */
RCRE_NODISCARD bool rcre_io_stream_next(
    RCRE_TensorStream* stream,
    RCRE_Tensor* out_batch
);

/**
 * @brief Rewinds an RCRE_TensorStream back to row 0 for a new training epoch.
 * @param stream Active tensor stream.
 */
void rcre_io_stream_reset(RCRE_TensorStream* stream);

/**
 * @brief Closes an RCRE_TensorStream and unmaps its underlying file.
 * @param stream Tensor stream to close.
 */
void rcre_io_stream_close(RCRE_TensorStream* stream);

/* ========================================================================= */
/* 6. Zero-Copy Cross-Framework C ABI Bridges: DLPack & Arrow (IO-REQ-06)    */
/* ========================================================================= */

typedef enum {
    RCRE_DL_DEVICE_CPU  = 1,
    RCRE_DL_DEVICE_CUDA = 2
} RCRE_DLDeviceType;

typedef enum {
    RCRE_DL_INT   = 0,
    RCRE_DL_UINT  = 1,
    RCRE_DL_FLOAT = 2
} RCRE_DLDataTypeCode;

typedef struct RCRE_DLTensor {
    void*     data;
    int32_t   device_type; /* RCRE_DL_DEVICE_CPU (1)                          */
    int32_t   device_id;   /* 0                                               */
    int32_t   ndim;        /* 1 to 4                                          */
    uint8_t   dtype_code;  /* RCRE_DL_INT (0), RCRE_DL_UINT (1), FLOAT (2)    */
    uint8_t   dtype_bits;  /* 8 or 32                                         */
    uint16_t  dtype_lanes; /* 1                                               */
    int64_t*  shape;       /* Pointer to int64_t[ndim]                        */
    int64_t*  strides;     /* Pointer to int64_t[ndim] in elements            */
    uint64_t  byte_offset; /* 0                                               */
} RCRE_DLTensor;

/**
 * @brief Exports an RCRE_Tensor to a zero-copy DLPack DLTensor descriptor.
 * @param src Source RCRE_Tensor.
 * @param meta_arena Arena to allocate the 64-bit shape/stride metadata arrays.
 * @param out_dl Output DLPack tensor struct sharing src->data zero-copy.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_to_dlpack(
    const RCRE_Tensor* src,
    RCRE_Arena* meta_arena,
    RCRE_DLTensor* out_dl
);

/**
 * @brief Imports a CPU DLPack DLTensor into a zero-copy RCRE_Tensor view.
 * @param dl Source DLPack tensor descriptor (device_type must be CPU).
 * @param out_view Output zero-copy RCRE_Tensor view (is_view = true).
 * @return RCRE_OK on success, RCRE_ERR_INVALID_ARG or UNSUPPORTED_OP on mismatch.
 */
RCRE_NODISCARD int rcre_io_from_dlpack(
    const RCRE_DLTensor* dl,
    RCRE_Tensor* out_view
);

/* Apache Arrow C Data Interface (ABI-compatible plain C structs) */
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

/**
 * @brief Exports a contiguous 1D RCRE_Tensor to the Apache Arrow C Data Interface.
 * @param src Contiguous 1D RCRE_Tensor (U8, I32, or F32).
 * @param meta_arena Arena to allocate the 2-element buffers pointer array.
 * @param out_schema Output ArrowSchema descriptor.
 * @param out_array Output ArrowArray descriptor pointing directly to src->data.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_to_arrow(
    const RCRE_Tensor* src,
    RCRE_Arena* meta_arena,
    RCRE_ArrowSchema* out_schema,
    RCRE_ArrowArray* out_array
);

/**
 * @brief Imports a flat numeric column from the Apache Arrow C Data Interface
 *        into a zero-copy 1D RCRE_Tensor view.
 * @param schema ArrowSchema describing the column type ("f", "i", or "C").
 * @param array ArrowArray holding the column buffer (null_count must be 0).
 * @param out_view Output zero-copy 1D RCRE_Tensor view.
 * @return RCRE_OK on success, error code otherwise.
 */
RCRE_NODISCARD int rcre_io_from_arrow(
    const RCRE_ArrowSchema* schema,
    const RCRE_ArrowArray* array,
    RCRE_Tensor* out_view
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_IO_H */
