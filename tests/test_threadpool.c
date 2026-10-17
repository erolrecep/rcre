/**
 * @file test_threadpool.c
 * @brief Unit tests for RCRE Multicore Threadpool & Tile Partitioning
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdatomic.h>

typedef struct {
    int* array;
    atomic_size_t executed_count;
} TaskPayload;

static void increment_task(void* arg, size_t task_id) {
    TaskPayload* p = (TaskPayload*)arg;
    p->array[task_id] = (int)(task_id * 3 + 7);
    atomic_fetch_add(&p->executed_count, 1);
}

typedef struct {
    uint8_t* grid;
    size_t width;
    size_t height;
    atomic_size_t pixel_visits;
} TilePayload;

static void tile_task(void* arg, size_t y_start, size_t y_end, size_t x_start, size_t x_end) {
    TilePayload* p = (TilePayload*)arg;
    for (size_t y = y_start; y < y_end; ++y) {
        for (size_t x = x_start; x < x_end; ++x) {
            p->grid[y * p->width + x] += 1;
            atomic_fetch_add(&p->pixel_visits, 1);
        }
    }
}

static void test_parallel_for_1d(void) {
    printf("  -> Testing parallel_for 1D dispatch...\n");
    RCRE_ThreadPool* pool = rcre_threadpool_create(4);
    assert(pool != NULL);

    const size_t N = 10000;
    int* data = (int*)calloc(N, sizeof(int));
    assert(data != NULL);

    TaskPayload payload = {
        .array = data,
        .executed_count = 0
    };

    rcre_threadpool_parallel_for(pool, N, increment_task, &payload);

    assert(atomic_load(&payload.executed_count) == N);
    for (size_t i = 0; i < N; ++i) {
        assert(data[i] == (int)(i * 3 + 7));
    }

    free(data);
    rcre_threadpool_destroy(pool);
    printf("     ✓ 10,000 tasks dispatched and verified across 4 threads\n");
}

static void test_parallel_for_2d_tiling(void) {
    printf("  -> Testing parallel_for_2d tile partitioning...\n");
    RCRE_ThreadPool* pool = rcre_threadpool_create(0); /* Auto-detect hardware concurrency */
    assert(pool != NULL);

    const size_t H = 105; /* Non-multiple of tile size */
    const size_t W = 197;
    const size_t tile_h = 16;
    const size_t tile_w = 32;

    uint8_t* grid = (uint8_t*)calloc(H * W, sizeof(uint8_t));
    assert(grid != NULL);

    TilePayload payload = {
        .grid = grid,
        .width = W,
        .height = H,
        .pixel_visits = 0
    };

    rcre_threadpool_parallel_for_2d(pool, H, W, tile_h, tile_w, tile_task, &payload);

    assert(atomic_load(&payload.pixel_visits) == (H * W));

    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            assert(grid[y * W + x] == 1); /* Exact single visit per pixel */
        }
    }

    free(grid);
    rcre_threadpool_destroy(pool);
    printf("     ✓ 2D non-uniform grid (%zux%zu) partitioned and covered with 0 overlaps\n", H, W);
}

static void test_single_threaded_fallback(void) {
    printf("  -> Testing single-threaded fallback when pool is NULL...\n");
    const size_t N = 50;
    int data[50] = {0};

    TaskPayload payload = {
        .array = data,
        .executed_count = 0
    };

    rcre_threadpool_parallel_for(NULL, N, increment_task, &payload);

    assert(atomic_load(&payload.executed_count) == N);
    for (size_t i = 0; i < N; ++i) {
        assert(data[i] == (int)(i * 3 + 7));
    }
    printf("     ✓ Single-threaded fallback executed without failure\n");
}

int main(void) {
    printf("🧪 Running RCRE Multicore Threadpool Tests (THR-REQ-01 to THR-REQ-03)...\n");
    test_parallel_for_1d();
    test_parallel_for_2d_tiling();
    test_single_threaded_fallback();
    printf("✅ All Threadpool Tests Passed Successfully!\n");
    return 0;
}
