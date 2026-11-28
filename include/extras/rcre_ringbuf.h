/**
 * @file rcre_ringbuf.h
 * @brief Lock-Free SPSC Frame Ring Buffer for High-Performance Video Streaming
 *
 * Implements a wait-free / lock-free single-producer single-consumer circular
 * triple-buffer in pure ISO C23 using <stdatomic.h>.
 * Memory is pre-allocated from an RCRE_Arena (zero malloc in the hot path).
 */

#ifndef RCRE_RINGBUF_H
#define RCRE_RINGBUF_H

#include "rcre.h"
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RCRE_RingBuf RCRE_RingBuf;

/**
 * @brief Creates a lock-free SPSC frame ring buffer backed by an arena.
 * @param slots Number of buffer slots (minimum 3 for triple buffering).
 * @param frame_bytes Maximum payload size per slot in bytes.
 * @param arena Arena allocator for ring buffer structure and slot payloads.
 * @return Pointer to RCRE_RingBuf, or NULL on allocation failure.
 */
RCRE_NODISCARD RCRE_RingBuf* rcre_ringbuf_create(size_t slots, size_t frame_bytes, RCRE_Arena* arena);

/**
 * @brief Atomically pushes a new frame into the ring buffer.
 * Overwrites the oldest unconsumed slot if the consumer is slower than capture.
 * @param rb Ring buffer handle.
 * @param data Frame payload pointer.
 * @param size Byte count of data.
 * @param width Frame width in pixels.
 * @param height Frame height in pixels.
 * @param format User/backend format identifier (e.g. 0=RGB24, 1=NV12, 2=YUYV).
 * @return 0 on success, negative on error.
 */
int rcre_ringbuf_push(RCRE_RingBuf* rb, const void* data, size_t size, size_t width, size_t height, int format);

/**
 * @brief Atomically pulls the latest frame into caller-supplied destination.
 * @param rb Ring buffer handle.
 * @param out_data Destination buffer (capacity >= max_bytes).
 * @param max_bytes Buffer capacity.
 * @param out_size Received frame byte count.
 * @param out_width Received width.
 * @param out_height Received height.
 * @param out_format Received format identifier.
 * @return 1 if a fresh frame was copied, 0 if no new frame is available.
 */
int rcre_ringbuf_pull_latest(RCRE_RingBuf* rb, void* out_data, size_t max_bytes, size_t* out_size, size_t* out_width, size_t* out_height, int* out_format);

/**
 * @brief Returns total frames successfully pushed since initialization.
 */
uint64_t rcre_ringbuf_pushed_count(const RCRE_RingBuf* rb);

/**
 * @brief Returns total frames successfully pulled since initialization.
 */
uint64_t rcre_ringbuf_pulled_count(const RCRE_RingBuf* rb);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_RINGBUF_H */
