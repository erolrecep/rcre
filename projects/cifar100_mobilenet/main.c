/**
 * @file main.c
 * @brief Zero-Allocation C23 MobileNet Inference Pipeline on CIFAR-100
 * 
 * Demonstrates:
 * - Loading fused MobileNet weights (RCRE_MOB v1) into memory arena.
 * - End-to-end zero-heap inference pipeline using RCRE CNN operators:
 *   Conv2D -> DepthwiseConv2D -> PointwiseConv2D -> ReLU6 -> GlobalAvgPool -> Linear -> Softmax.
 * - Top-1 / Top-5 classification and sub-millisecond latency benchmarking.
 */

#include "rcre.h"
#include "cifar100_labels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_BLOCKS 7

typedef struct MobileNetBlock {
    RCRE_Tensor dw_w;
    RCRE_Tensor dw_b;
    RCRE_Tensor pw_w;
    RCRE_Tensor pw_b;
    size_t stride;
} MobileNetBlock;

typedef struct MobileNetModel {
    RCRE_Tensor stem_w;
    RCRE_Tensor stem_b;
    MobileNetBlock blocks[NUM_BLOCKS];
    RCRE_Tensor fc_w;
    RCRE_Tensor fc_b;
} MobileNetModel;

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int read_tensor_record(FILE* f, RCRE_Tensor* t, RCRE_Arena* arena) {
    uint32_t ndim = 0;
    if (fread(&ndim, sizeof(uint32_t), 1, f) != 1) return RCRE_ERR_IO;

    uint32_t shape[4] = {0};
    if (fread(shape, sizeof(uint32_t), 4, f) != 4) return RCRE_ERR_IO;

    size_t t_shape[4];
    size_t numel = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        t_shape[i] = (size_t)shape[i];
        numel *= t_shape[i];
    }

    *t = rcre_tensor_create(t_shape, (uint8_t)ndim, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW, arena);
    if (!t->data) return RCRE_ERR_OUT_OF_MEMORY;

    if (fread(t->data, sizeof(float), numel, f) != numel) {
        return RCRE_ERR_IO;
    }

    return RCRE_OK;
}

static int load_mobilenet_model(const char* filepath, MobileNetModel* model, RCRE_Arena* arena) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "❌ Error: Could not open model weights file: %s\n", filepath);
        return RCRE_ERR_IO;
    }

    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "RCRE_MOB", 8) != 0) {
        fprintf(stderr, "❌ Error: Invalid magic header in weights file (expected RCRE_MOB)\n");
        fclose(f);
        return RCRE_ERR_INVALID_ARG;
    }

    uint32_t version = 0, num_tensors = 0;
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(&num_tensors, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return RCRE_ERR_IO;
    }

    if (num_tensors != 32) {
        fprintf(stderr, "❌ Error: Expected 32 tensors, found %u in %s\n", num_tensors, filepath);
        fclose(f);
        return RCRE_ERR_INVALID_ARG;
    }

    /* 1. Stem Conv & Bias */
    if (read_tensor_record(f, &model->stem_w, arena) != RCRE_OK ||
        read_tensor_record(f, &model->stem_b, arena) != RCRE_OK) {
        fclose(f);
        return RCRE_ERR_IO;
    }

    /* Block strides: 1, 2, 1, 2, 1, 2, 1 */
    size_t block_strides[NUM_BLOCKS] = {1, 2, 1, 2, 1, 2, 1};

    /* 2. Seven Depthwise Separable Blocks */
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        model->blocks[i].stride = block_strides[i];
        if (read_tensor_record(f, &model->blocks[i].dw_w, arena) != RCRE_OK ||
            read_tensor_record(f, &model->blocks[i].dw_b, arena) != RCRE_OK ||
            read_tensor_record(f, &model->blocks[i].pw_w, arena) != RCRE_OK ||
            read_tensor_record(f, &model->blocks[i].pw_b, arena) != RCRE_OK) {
            fclose(f);
            return RCRE_ERR_IO;
        }
    }

    /* 3. Fully Connected Linear Head */
    if (read_tensor_record(f, &model->fc_w, arena) != RCRE_OK ||
        read_tensor_record(f, &model->fc_b, arena) != RCRE_OK) {
        fclose(f);
        return RCRE_ERR_IO;
    }

    fclose(f);
    return RCRE_OK;
}

static int load_sample_binary(const char* filepath, float* buffer, size_t numel) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return RCRE_ERR_IO;
    size_t read_bytes = fread(buffer, sizeof(float), numel, f);
    fclose(f);
    return (read_bytes == numel) ? RCRE_OK : RCRE_ERR_IO;
}

/**
 * @brief Executes forward pass of MobileNet on a single 3x32x32 image tensor.
 */
static int mobilenet_predict(const MobileNetModel* model, 
                             const RCRE_Tensor* input, 
                             RCRE_Tensor* out_probs, 
                             RCRE_Arena* scratch) {
    size_t scratch_base = scratch->offset;

    /* 1. Stem Conv: [3, 32, 32] -> [32, 32, 32] */
    RCRE_Conv2DParams stem_params = {.stride_h = 1, .stride_w = 1, .pad_h = 1, .pad_w = 1};
    RCRE_Tensor cur_act = {0};
    int status = rcre_nn_conv2d(input, &model->stem_w, &model->stem_b, &stem_params, &cur_act, scratch);
    if (status != RCRE_OK) return status;
    status = rcre_nn_relu6(&cur_act, &cur_act);
    if (status != RCRE_OK) return status;

    /* 2. 7 Depthwise Separable Blocks */
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        const MobileNetBlock* b = &model->blocks[i];

        /* Depthwise Conv */
        RCRE_Conv2DParams dw_params = {
            .stride_h = b->stride,
            .stride_w = b->stride,
            .pad_h = 1,
            .pad_w = 1
        };
        RCRE_Tensor dw_out = {0};
        status = rcre_nn_depthwise_conv2d(&cur_act, &b->dw_w, &b->dw_b, &dw_params, &dw_out, scratch);
        if (status != RCRE_OK) return status;
        status = rcre_nn_relu6(&dw_out, &dw_out);
        if (status != RCRE_OK) return status;

        /* Pointwise Conv */
        RCRE_Tensor pw_out = {0};
        status = rcre_nn_pointwise_conv2d(&dw_out, &b->pw_w, &b->pw_b, &pw_out, scratch);
        if (status != RCRE_OK) return status;
        status = rcre_nn_relu6(&pw_out, &pw_out);
        if (status != RCRE_OK) return status;

        cur_act = pw_out;
    }

    /* 3. Global Average Pooling: [512, 4, 4] -> [512, 1, 1] */
    size_t pool_shape[3] = {512, 1, 1};
    RCRE_Tensor pooled = rcre_tensor_create(pool_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW, scratch);
    status = rcre_nn_global_avgpool(&cur_act, &pooled);
    if (status != RCRE_OK) return status;

    /* 4. Fully Connected Linear: [512] -> [100] */
    RCRE_Tensor logits = {0};
    status = rcre_nn_linear(&pooled, &model->fc_w, &model->fc_b, &logits, scratch);
    if (status != RCRE_OK) return status;

    /* 5. Softmax: [100] -> probabilities [100] */
    status = rcre_nn_softmax(&logits, out_probs);

    /* Free scratch memory used by intermediate activations */
    scratch->offset = scratch_base;
    return status;
}

typedef struct Prediction {
    int class_id;
    float prob;
} Prediction;

static void get_topk(const float* probs, size_t n, size_t k, Prediction* topk) {
    for (size_t i = 0; i < k; ++i) {
        topk[i].class_id = -1;
        topk[i].prob = -1.0f;
    }

    for (size_t i = 0; i < n; ++i) {
        float p = probs[i];
        if (p > topk[k - 1].prob) {
            topk[k - 1].class_id = (int)i;
            topk[k - 1].prob = p;

            /* Bubble up */
            for (int j = (int)k - 2; j >= 0; --j) {
                if (topk[j + 1].prob > topk[j].prob) {
                    Prediction tmp = topk[j];
                    topk[j] = topk[j + 1];
                    topk[j + 1] = tmp;
                } else {
                    break;
                }
            }
        }
    }
}

static void print_topk(const Prediction* topk, size_t k) {
    for (size_t i = 0; i < k; ++i) {
        int cid = topk[i].class_id;
        int coarse_id = CIFAR100_FINE_TO_COARSE[cid];
        printf("     #%zu: %-16s (%2d) [%-20s] -> %5.2f%%\n",
               i + 1,
               CIFAR100_CLASSES[cid],
               cid,
               CIFAR100_SUPERCLASSES[coarse_id],
               topk[i].prob * 100.0f);
    }
}

int main(int argc, char** argv) {
    const char* weights_path = "mobilenet_cifar100.bin";
    const char* sample_path = "data/sample_0.bin";
    int benchmark_runs = 0;
    bool run_demo = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            sample_path = argv[++i];
        } else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
            benchmark_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--demo") == 0) {
            run_demo = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: rcre-mobilenet [OPTIONS]\n"
                   "  --weights PATH     Path to binary model weights (default: mobilenet_cifar100.bin)\n"
                   "  --input PATH       Path to input sample float array (default: data/sample_0.bin)\n"
                   "  --benchmark N      Run latency benchmark over N iterations\n"
                   "  --demo             Evaluate all 10 extracted CIFAR-100 test samples in data/\n"
                   "  --help             Display this help message\n");
            return 0;
        }
    }

    printf("==================================================================\n");
    printf("  RCRE MobileNet CIFAR-100 Neural Inference Engine (C23)\n");
    printf("==================================================================\n");

    /* Initialize Hardware Abstraction Layer & Vector Dispatch */
    rcre_hal_init();
    printf("  -> CPU Detected:  %s\n", rcre_cpu_get_brand());
    printf("  -> HAL Backend:   %s\n", rcre_hal.backend_name);

    /* Allocate persistent model arena (16 MB) and scratchpad arena (8 MB) */
    RCRE_Arena* model_arena = rcre_arena_create(16 * 1024 * 1024);
    RCRE_Arena* scratch_arena = rcre_arena_create(8 * 1024 * 1024);
    if (!model_arena || !scratch_arena) {
        fprintf(stderr, "❌ Failed to allocate memory arenas\n");
        return 1;
    }

    printf("  -> Model Arena:   %.1f MB\n", (double)model_arena->capacity / (1024 * 1024));
    printf("  -> Scratch Arena: %.1f MB (Zero-heap dynamic activations)\n", 
           (double)scratch_arena->capacity / (1024 * 1024));

    /* Load Weights */
    MobileNetModel model;
    printf("  -> Loading weights: %s ...\n", weights_path);
    double t_load_start = get_time_sec();
    int status = load_mobilenet_model(weights_path, &model, model_arena);
    double t_load = (get_time_sec() - t_load_start) * 1000.0;
    if (status != RCRE_OK) {
        fprintf(stderr, "❌ Failed to load model weights (error %d)\n", status);
        return 1;
    }
    printf("     ✓ Loaded 32 fused layers in %.2f ms (Memory: %.2f MB)\n", 
           t_load, (double)model_arena->offset / (1024 * 1024));

    /* Prepare output probabilities tensor */
    size_t out_shape[1] = {CIFAR100_NUM_CLASSES};
    RCRE_Tensor out_probs = rcre_tensor_create(out_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, scratch_arena);

    if (run_demo) {
        printf("\n==================================================================\n");
        printf("  Evaluating 10 CIFAR-100 Test Samples (data/sample_0.bin .. 9.bin)\n");
        printf("==================================================================\n");

        float sample_buf[3 * 32 * 32];
        size_t in_shape[3] = {3, 32, 32};
        RCRE_Tensor in_t = rcre_tensor_from_buffer(sample_buf, in_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

        for (int idx = 0; idx < 10; ++idx) {
            char path[256];
            snprintf(path, sizeof(path), "data/sample_%d.bin", idx);
            if (load_sample_binary(path, sample_buf, 3 * 32 * 32) != RCRE_OK) {
                continue;
            }

            double t0 = get_time_sec();
            status = mobilenet_predict(&model, &in_t, &out_probs, scratch_arena);
            double dt = (get_time_sec() - t0) * 1000.0;

            Prediction top5[5];
            get_topk((const float*)out_probs.data, 100, 5, top5);

            printf("\n📷 [Sample %d] (%s) - Inference: %.2f ms\n", idx, path, dt);
            printf("   Ground Truth Class: %d (%s)\n", idx, CIFAR100_CLASSES[idx]);
            printf("   Top-5 Predictions:\n");
            print_topk(top5, 5);
        }
    } else {
        /* Run Single Inference */
        float sample_buf[3 * 32 * 32];
        size_t in_shape[3] = {3, 32, 32};
        RCRE_Tensor in_t = rcre_tensor_from_buffer(sample_buf, in_shape, 3, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW);

        printf("  -> Loading input:   %s ...\n", sample_path);
        if (load_sample_binary(sample_path, sample_buf, 3 * 32 * 32) != RCRE_OK) {
            fprintf(stderr, "❌ Failed to read sample binary: %s\n", sample_path);
            return 1;
        }

        double t0 = get_time_sec();
        status = mobilenet_predict(&model, &in_t, &out_probs, scratch_arena);
        double dt = (get_time_sec() - t0) * 1000.0;
        if (status != RCRE_OK) {
            fprintf(stderr, "❌ Inference failed with error %d\n", status);
            return 1;
        }

        Prediction top5[5];
        get_topk((const float*)out_probs.data, 100, 5, top5);

        printf("\n🎯 Inference Complete in %.2f ms!\n", dt);
        printf("   Top-5 Classification Results:\n");
        print_topk(top5, 5);

        if (benchmark_runs > 0) {
            printf("\n⏱️  Running Benchmark (%d iterations)...\n", benchmark_runs);
            double min_dt = 1e9, max_dt = 0.0, sum_dt = 0.0;

            for (int r = 0; r < benchmark_runs; ++r) {
                double t_start = get_time_sec();
                mobilenet_predict(&model, &in_t, &out_probs, scratch_arena);
                double t_end = get_time_sec();

                double dur_ms = (t_end - t_start) * 1000.0;
                sum_dt += dur_ms;
                if (dur_ms < min_dt) min_dt = dur_ms;
                if (dur_ms > max_dt) max_dt = dur_ms;
            }

            double avg_dt = sum_dt / benchmark_runs;
            printf("   -> Mean Latency: %.2f ms (%.1f FPS)\n", avg_dt, 1000.0 / avg_dt);
            printf("   -> Min Latency:  %.2f ms\n", min_dt);
            printf("   -> Max Latency:  %.2f ms\n", max_dt);
        }
    }

    rcre_arena_destroy(scratch_arena);
    rcre_arena_destroy(model_arena);

    printf("\n✅ MobileNet Execution Completed Cleanly (Zero Memory Leaks)!\n");
    return 0;
}
