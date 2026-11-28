/**
 * @file rcre_ringbuf.c
 * @brief Lock-Free SPSC Frame Ring Buffer Implementation
 */

#include "extras/rcre_ringbuf.h"
#include <string.h>

typedef struct {
    uint8_t* data;
    size_t   size;
    size_t   width;
    size_t   height;
    int      format;
    uint64_t seq;
} RCRE_RingSlot;

struct RCRE_RingBuf {
    RCRE_RingSlot* slots;
    size_t         nslots;
    size_t         slot_capacity;
    atomic_int     latest_idx;       /* Slot index containing freshest completed frame */
    atomic_int     reading_idx;      /* Slot index currently being read by consumer */
    atomic_uint_fast64_t seq_pushed; /* Total frames pushed by producer */
    atomic_uint_fast64_t seq_pulled; /* Total frames pulled by consumer */
    uint64_t       last_seen_seq;    /* Consumer local sequence tracker */
    int            write_idx;        /* Producer private next candidate slot */
};

RCRE_RingBuf* rcre_ringbuf_create(size_t slots, size_t frame_bytes, RCRE_Arena* arena) {
    if (!arena || slots < 2 || frame_bytes == 0) return NULL;

    RCRE_RingBuf* rb = (RCRE_RingBuf*)rcre_arena_alloc(arena, sizeof(RCRE_RingBuf), 16);
    if (!rb) return NULL;

    rb->nslots = slots;
    rb->slot_capacity = frame_bytes;
    rb->slots = (RCRE_RingSlot*)rcre_arena_alloc(arena, sizeof(RCRE_RingSlot) * slots, 16);
    if (!rb->slots) return NULL;

    for (size_t i = 0; i < slots; ++i) {
        rb->slots[i].data = (uint8_t*)rcre_arena_alloc(arena, frame_bytes, 64);
        if (!rb->slots[i].data) return NULL;
        rb->slots[i].size = 0;
        rb->slots[i].width = 0;
        rb->slots[i].height = 0;
        rb->slots[i].format = 0;
        rb->slots[i].seq = 0;
    }

    atomic_init(&rb->latest_idx, -1);
    atomic_init(&rb->reading_idx, -1);
    atomic_init(&rb->seq_pushed, 0);
    atomic_init(&rb->seq_pulled, 0);
    rb->last_seen_seq = 0;
    rb->write_idx = 0;

    return rb;
}

int rcre_ringbuf_push(RCRE_RingBuf* rb, const void* data, size_t size, size_t width, size_t height, int format) {
    if (!rb || !data || size == 0 || size > rb->slot_capacity) {
        return RCRE_ERR_INVALID_ARG;
    }

    /* Choose a candidate slot that the consumer is NOT currently actively reading */
    int reading = atomic_load_explicit(&rb->reading_idx, memory_order_acquire);
    int target = rb->write_idx;
    if (target == reading) {
        target = (target + 1) % (int)rb->nslots;
    }

    RCRE_RingSlot* s = &rb->slots[target];
    memcpy(s->data, data, size);
    s->size = size;
    s->width = width;
    s->height = height;
    s->format = format;

    uint64_t next_seq = atomic_load_explicit(&rb->seq_pushed, memory_order_relaxed) + 1;
    s->seq = next_seq;

    /* Publish slot to consumer atomically with release ordering */
    atomic_store_explicit(&rb->latest_idx, target, memory_order_release);
    atomic_store_explicit(&rb->seq_pushed, next_seq, memory_order_release);

    /* Advance producer candidate index */
    rb->write_idx = (target + 1) % (int)rb->nslots;
    return RCRE_OK;
}

int rcre_ringbuf_pull_latest(RCRE_RingBuf* rb, void* out_data, size_t max_bytes, size_t* out_size, size_t* out_width, size_t* out_height, int* out_format) {
    if (!rb || !out_data) return 0;

    int idx = atomic_load_explicit(&rb->latest_idx, memory_order_acquire);
    if (idx < 0) return 0; /* No frames published yet */

    /* Mark slot as actively being read so producer avoids overwriting it */
    atomic_store_explicit(&rb->reading_idx, idx, memory_order_release);

    RCRE_RingSlot* s = &rb->slots[idx];
    if (s->seq <= rb->last_seen_seq) {
        /* No new frame since last pull */
        atomic_store_explicit(&rb->reading_idx, -1, memory_order_release);
        return 0;
    }

    size_t copy_bytes = s->size < max_bytes ? s->size : max_bytes;
    memcpy(out_data, s->data, copy_bytes);

    if (out_size)   *out_size = s->size;
    if (out_width)  *out_width = s->width;
    if (out_height) *out_height = s->height;
    if (out_format) *out_format = s->format;

    rb->last_seen_seq = s->seq;
    atomic_fetch_add_explicit(&rb->seq_pulled, 1, memory_order_relaxed);

    /* Release consumer lock on slot */
    atomic_store_explicit(&rb->reading_idx, -1, memory_order_release);
    return 1;
}

uint64_t rcre_ringbuf_pushed_count(const RCRE_RingBuf* rb) {
    if (!rb) return 0;
    return atomic_load_explicit(&rb->seq_pushed, memory_order_relaxed);
}

uint64_t rcre_ringbuf_pulled_count(const RCRE_RingBuf* rb) {
    if (!rb) return 0;
    return atomic_load_explicit(&rb->seq_pulled, memory_order_relaxed);
}
