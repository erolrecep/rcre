/**
 * @file threadpool.c
 * @brief Lock-Free Atomic Task Scheduler & Multicore Threadpool
 * 
 * Satisfies THR-REQ-01, THR-REQ-02, and THR-REQ-03.
 * Uses atomic fetch-and-add task distribution and pthread/C23 thread workers.
 *
 * ARCHITECTURAL DESIGN NOTE (Unix Philosophy — Keep Primitives Single-Threaded):
 * Core RCRE computational kernels (CV filters, FFT, CNN layers, ML models) are
 * intentionally designed to be purely deterministic, re-entrant, and single-threaded.
 * This guarantees 100% portability to bare-metal embedded microcontrollers and WASM
 * runtimes where OS threads are absent or require SharedArrayBuffer.
 * This threadpool engine is an opt-in pipeline concurrency tool for orchestrating
 * outer loops, multi-camera capture streams, or high-level spatial tile decomposition.
 */

#include "rcre.h"
#include <stdlib.h>
#include <stdatomic.h>
#include <unistd.h>
#include <pthread.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#elif defined(__linux__)
#include <sched.h>
#endif

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
    RCRE_QoSClass   qos;
    pthread_mutex_t lock;
    pthread_cond_t  work_cond;
    pthread_cond_t  done_cond;
    RCRE_WorkBatch  batch;
    size_t          active_workers;
    bool            has_batch;
    bool            stop;
};

static void tp_apply_worker_qos(RCRE_QoSClass qos) {
#if defined(__APPLE__)
    switch (qos) {
        case RCRE_QOS_PERF:
            (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
            break;
        case RCRE_QOS_EFFICIENCY:
            (void)pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
            break;
        case RCRE_QOS_BACKGROUND:
            (void)pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
            break;
        case RCRE_QOS_DEFAULT:
        default:
            break;
    }
#elif defined(__linux__)
    struct sched_param sp = { .sched_priority = 0 };
    if (qos == RCRE_QOS_BACKGROUND) {
  #if defined(SCHED_IDLE)
        (void)pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp);
  #endif
    } else if (qos == RCRE_QOS_EFFICIENCY) {
  #if defined(SCHED_BATCH)
        (void)pthread_setschedparam(pthread_self(), SCHED_BATCH, &sp);
  #endif
    }
#else
    (void)qos;
#endif
}

static void* worker_routine(void* arg) {
    RCRE_ThreadPool* pool = (RCRE_ThreadPool*)arg;
    tp_apply_worker_qos(pool->qos);

    while (true) {
        pthread_mutex_lock(&pool->lock);
        while (!pool->stop && (!pool->has_batch ||
               atomic_load(&pool->batch.next_task) >= pool->batch.total_tasks)) {
            pthread_cond_wait(&pool->work_cond, &pool->lock);
        }

        if (pool->stop) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        pool->active_workers++;
        size_t total = pool->batch.total_tasks;
        rcre_task_fn fn = pool->batch.task_fn;
        void* fn_arg = pool->batch.arg;
        pthread_mutex_unlock(&pool->lock);

        /* Work-stealing atomic loop */
        while (true) {
            size_t task_id = atomic_fetch_add(&pool->batch.next_task, 1);
            if (task_id >= total) {
                break;
            }

            fn(fn_arg, task_id);
            atomic_fetch_add(&pool->batch.completed_tasks, 1);
        }

        pthread_mutex_lock(&pool->lock);
        pool->active_workers--;
        if (pool->active_workers == 0 &&
            atomic_load(&pool->batch.completed_tasks) >= pool->batch.total_tasks) {
            pthread_cond_broadcast(&pool->done_cond);
        }
        pthread_mutex_unlock(&pool->lock);
    }
    return NULL;
}

RCRE_NODISCARD RCRE_ThreadPool* rcre_threadpool_create_ex(size_t num_threads, RCRE_QoSClass qos) {
    if (num_threads == 0) {
        RCRE_CpuTopology topo = rcre_cpu_get_topology();
        if (qos == RCRE_QOS_PERF && topo.perf_cores > 0) {
            num_threads = (topo.perf_cores < topo.effective_quota_cores)
                          ? topo.perf_cores : topo.effective_quota_cores;
        } else if (qos == RCRE_QOS_EFFICIENCY && topo.eff_cores > 0) {
            num_threads = (topo.eff_cores < topo.effective_quota_cores)
                          ? topo.eff_cores : topo.effective_quota_cores;
        } else {
            num_threads = (topo.effective_quota_cores > 0) ? topo.effective_quota_cores : 4;
        }
    }

    RCRE_ThreadPool* pool = (RCRE_ThreadPool*)malloc(sizeof(RCRE_ThreadPool));
    if (!pool) return NULL;

    pool->num_threads = num_threads;
    pool->qos = qos;
    pool->active_workers = 0;
    pool->has_batch = false;
    pool->stop = false;

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

RCRE_NODISCARD RCRE_ThreadPool* rcre_threadpool_create(size_t num_threads) {
    return rcre_threadpool_create_ex(num_threads, RCRE_QOS_DEFAULT);
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

    pthread_mutex_lock(&pool->lock);
    pool->batch.task_fn = task;
    pool->batch.arg = arg;
    pool->batch.total_tasks = count;
    atomic_init(&pool->batch.next_task, 0);
    atomic_init(&pool->batch.completed_tasks, 0);
    pool->has_batch = true;
    pthread_cond_broadcast(&pool->work_cond);

    /* Main thread participates in task execution */
    pthread_mutex_unlock(&pool->lock);

    while (true) {
        size_t task_id = atomic_fetch_add(&pool->batch.next_task, 1);
        if (task_id >= count) {
            break;
        }
        task(arg, task_id);
        atomic_fetch_add(&pool->batch.completed_tasks, 1);
    }

    /* Wait for all tasks AND all in-flight worker drains to finish */
    pthread_mutex_lock(&pool->lock);
    while (atomic_load(&pool->batch.completed_tasks) < count || pool->active_workers > 0) {
        pthread_cond_wait(&pool->done_cond, &pool->lock);
    }
    pool->has_batch = false;
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
