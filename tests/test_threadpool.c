/**
 * @file test_threadpool.c
 * @brief Unit tests for RCRE Multicore Threadpool & Tile Partitioning
 */

#include "rcre.h"
#include "test_framework.h"
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
    RCRE_ThreadPool* pool = rcre_threadpool_create(4);
    ASSERT_NOT_NULL(pool);

    const size_t N = 10000;
    int* data = (int*)calloc(N, sizeof(int));
    ASSERT_NOT_NULL(data);

    TaskPayload payload = {
        .array = data,
        .executed_count = 0
    };

    rcre_threadpool_parallel_for(pool, N, increment_task, &payload);

    ASSERT_UINT_EQ(N, atomic_load(&payload.executed_count));
    for (size_t i = 0; i < N; ++i) {
        ASSERT_INT_EQ((int)(i * 3 + 7), data[i]);
    }

    rcre_threadpool_destroy(pool);
    free(data);
}

static void test_parallel_for_2d_tiling(void) {
    RCRE_ThreadPool* pool = rcre_threadpool_create(0); /* Auto-detect hardware concurrency */
    ASSERT_NOT_NULL(pool);

    const size_t H = 105; /* Non-multiple of tile size */
    const size_t W = 197;
    const size_t tile_h = 16;
    const size_t tile_w = 32;

    uint8_t* grid = (uint8_t*)calloc(H * W, sizeof(uint8_t));
    ASSERT_NOT_NULL(grid);

    TilePayload payload = {
        .grid = grid,
        .width = W,
        .height = H,
        .pixel_visits = 0
    };

    rcre_threadpool_parallel_for_2d(pool, H, W, tile_h, tile_w, tile_task, &payload);

    ASSERT_UINT_EQ(H * W, atomic_load(&payload.pixel_visits));

    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            ASSERT_UINT_EQ(1, grid[y * W + x]); /* Exact single visit per pixel */
        }
    }

    rcre_threadpool_destroy(pool);
    free(grid);
}

static void test_single_threaded_fallback(void) {
    const size_t N = 50;
    int data[50] = {0};

    TaskPayload payload = {
        .array = data,
        .executed_count = 0
    };

    rcre_threadpool_parallel_for(NULL, N, increment_task, &payload);

    ASSERT_UINT_EQ(N, atomic_load(&payload.executed_count));
    for (size_t i = 0; i < N; ++i) {
        ASSERT_INT_EQ((int)(i * 3 + 7), data[i]);
    }
}

static void test_qos_steered_threadpools(void) {
    const RCRE_QoSClass classes[] = {
        RCRE_QOS_PERF,
        RCRE_QOS_EFFICIENCY,
        RCRE_QOS_BACKGROUND
    };
    const size_t N = 1024;
    int* data = (int*)calloc(N, sizeof(int));
    ASSERT_NOT_NULL(data);

    for (size_t c = 0; c < sizeof(classes) / sizeof(classes[0]); ++c) {
        memset(data, 0, N * sizeof(int));
        RCRE_ThreadPool* pool = rcre_threadpool_create_ex(0, classes[c]);
        ASSERT_NOT_NULL(pool);

        TaskPayload payload = { .array = data, .executed_count = 0 };
        rcre_threadpool_parallel_for(pool, N, increment_task, &payload);
        ASSERT_UINT_EQ(N, atomic_load(&payload.executed_count));
        rcre_threadpool_destroy(pool);
    }
    free(data);
}

int main(void) {
    test_suite_begin("RCRE Multicore Threadpool & Work Stealing (THR-REQ-01 to 03)");
    TEST_RUN(test_parallel_for_1d);
    TEST_RUN(test_parallel_for_2d_tiling);
    TEST_RUN(test_single_threaded_fallback);
    TEST_RUN(test_qos_steered_threadpools);
    return test_suite_report();
}
