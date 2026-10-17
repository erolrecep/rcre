/**
 * @file threadpool.c
 * @brief Lock-Free Atomic Task Scheduler & Multicore Threadpool
 * 
 * Satisfies THR-REQ-01, THR-REQ-02, and THR-REQ-03.
 * Uses atomic fetch-and-add task distribution and pthread/C23 thread workers.
 */

#include "rcre.h"
#include <stdlib.h>
#include <stdatomic.h>
#include <unistd.h>
#include <pthread.h>

typedef struct RCRE_WorkBatch {
    rcre_task_fn task_fn;
    void*        arg;
    size_t       total_tasks;
    atomic_size_t next_task;
    atomic_size_t completed_tasks;
} RCRE_WorkBatch;

struct RCRE_ThreadPool {
    pthread_t*      threads;
    size_t          num_threads;
    pthread_mutex_t lock;
    pthread_cond_t  work_cond;
    pthread_cond_t  done_cond;
    RCRE_WorkBatch* current_batch;
    bool            stop;
};

static void* worker_routine(void* arg) {
    RCRE_ThreadPool* pool = (RCRE_ThreadPool*)arg;

    while (true) {
        pthread_mutex_lock(&pool->lock);
        while (!pool->stop && (!pool->current_batch || 
               atomic_load(&pool->current_batch->next_task) >= pool->current_batch->total_tasks)) {
            pthread_cond_wait(&pool->work_cond, &pool->lock);
        }

        if (pool->stop) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        RCRE_WorkBatch* batch = pool->current_batch;
        pthread_mutex_unlock(&pool->lock);

        /* Work-stealing atomic loop */
        while (true) {
            size_t task_id = atomic_fetch_add(&batch->next_task, 1);
            if (task_id >= batch->total_tasks) {
                break;
            }

            batch->task_fn(batch->arg, task_id);
            size_t completed = atomic_fetch_add(&batch->completed_tasks, 1) + 1;

            if (completed == batch->total_tasks) {
                pthread_mutex_lock(&pool->lock);
                pthread_cond_broadcast(&pool->done_cond);
                pthread_mutex_unlock(&pool->lock);
            }
        }
    }
    return NULL;
}

RCRE_NODISCARD RCRE_ThreadPool* rcre_threadpool_create(size_t num_threads) {
    if (num_threads == 0) {
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        num_threads = (nprocs > 0) ? (size_t)nprocs : 4;
    }

    RCRE_ThreadPool* pool = (RCRE_ThreadPool*)malloc(sizeof(RCRE_ThreadPool));
    if (!pool) return NULL;

    pool->num_threads = num_threads;
    pool->stop = false;
    pool->current_batch = NULL;

    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->work_cond, NULL);
    pthread_cond_init(&pool->done_cond, NULL);

    pool->threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    if (!pool->threads) {
        free(pool);
        return NULL;
    }

    for (size_t i = 0; i < num_threads; ++i) {
        if (pthread_create(&pool->threads[i], NULL, worker_routine, pool) != 0) {
            pool->num_threads = i;
            rcre_threadpool_destroy(pool);
            return NULL;
        }
    }

    return pool;
}

void rcre_threadpool_parallel_for(RCRE_ThreadPool* pool, size_t count, rcre_task_fn task, void* arg) {
    if (count == 0 || !task) return;

    if (!pool || pool->num_threads <= 1) {
        /* Run serially on single-threaded fallback */
        for (size_t i = 0; i < count; ++i) {
            task(arg, i);
        }
        return;
    }

    RCRE_WorkBatch batch;
    batch.task_fn = task;
    batch.arg = arg;
    batch.total_tasks = count;
    atomic_init(&batch.next_task, 0);
    atomic_init(&batch.completed_tasks, 0);

    pthread_mutex_lock(&pool->lock);
    pool->current_batch = &batch;
    pthread_cond_broadcast(&pool->work_cond);

    /* Main thread participates in task execution */
    pthread_mutex_unlock(&pool->lock);

    while (true) {
        size_t task_id = atomic_fetch_add(&batch.next_task, 1);
        if (task_id >= count) {
            break;
        }
        task(arg, task_id);
        size_t completed = atomic_fetch_add(&batch.completed_tasks, 1) + 1;
        if (completed == count) {
            pthread_mutex_lock(&pool->lock);
            pthread_cond_broadcast(&pool->done_cond);
            pthread_mutex_unlock(&pool->lock);
        }
    }

    /* Wait for batch to finish */
    pthread_mutex_lock(&pool->lock);
    while (atomic_load(&batch.completed_tasks) < count) {
        pthread_cond_wait(&pool->done_cond, &pool->lock);
    }
    pool->current_batch = NULL;
    pthread_mutex_unlock(&pool->lock);
}

typedef struct {
    rcre_tile_fn tile_fn;
    void*        arg;
    size_t       height;
    size_t       width;
    size_t       tile_h;
    size_t       tile_w;
    size_t       tiles_x;
} TileAdapterArg;

static void tile_task_adapter(void* arg, size_t task_id) {
    TileAdapterArg* t = (TileAdapterArg*)arg;
    size_t ty = task_id / t->tiles_x;
    size_t tx = task_id % t->tiles_x;

    size_t y_start = ty * t->tile_h;
    size_t y_end = y_start + t->tile_h;
    if (y_end > t->height) y_end = t->height;

    size_t x_start = tx * t->tile_w;
    size_t x_end = x_start + t->tile_w;
    if (x_end > t->width) x_end = t->width;

    t->tile_fn(t->arg, y_start, y_end, x_start, x_end);
}

void rcre_threadpool_parallel_for_2d(RCRE_ThreadPool* pool, size_t height, size_t width, 
                                     size_t tile_h, size_t tile_w, 
                                     rcre_tile_fn tile_fn, void* arg) {
    if (height == 0 || width == 0 || tile_h == 0 || tile_w == 0 || !tile_fn) {
        return;
    }

    size_t tiles_y = (height + tile_h - 1) / tile_h;
    size_t tiles_x = (width + tile_w - 1) / tile_w;
    size_t total_tiles = tiles_y * tiles_x;

    TileAdapterArg adapt_arg = {
        .tile_fn = tile_fn,
        .arg = arg,
        .height = height,
        .width = width,
        .tile_h = tile_h,
        .tile_w = tile_w,
        .tiles_x = tiles_x
    };

    rcre_threadpool_parallel_for(pool, total_tiles, tile_task_adapter, &adapt_arg);
}

void rcre_threadpool_destroy(RCRE_ThreadPool* pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->lock);
    pool->stop = true;
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->lock);

    for (size_t i = 0; i < pool->num_threads; ++i) {
        pthread_join(pool->threads[i], NULL);
    }

    free(pool->threads);
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->work_cond);
    pthread_cond_destroy(&pool->done_cond);
    free(pool);
}
