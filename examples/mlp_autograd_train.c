/**
 * @file mlp_autograd_train.c
 * @brief End-to-End Deep MLP Training with Dynamic Tape Autograd & AdamW
 * 
 * Part of RCRE Unified Engine Showcase.
 * Demonstrates:
 *  - 64-byte aligned linear memory arena (RCRE_Arena)
 *  - Multi-layer Perceptron (2 -> 16 -> 1) forward pass with dynamic tape recording
 *  - Vector-Jacobian Product (VJP) reverse-mode automatic differentiation
 *  - Decoupled weight decay AdamW optimization
 *  - In-terminal ANSI training progression and loss visualization
 */

#include "rcre.h"
#include "rcre_autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_SAMPLES 200
#define BATCH_SIZE 32
#define EPOCHS 100

int main(void) {
    printf("\033[1;36m==================================================================\033[0m\n");
    printf("\033[1;36m🧠 RCRE Unified Engine: Deep MLP Training with Tensor Autograd    \033[0m\n");
    printf("\033[1;36m==================================================================\033[0m\n\n");

    /* 1. Initialize HAL dynamic dispatch */
    rcre_hal_init();
    printf("⚡ Hardware Backend: \033[1;32m%s\033[0m\n", rcre_hal.backend_name);

    /* 2. Create memory arenas: 8MB persistent for weights, 16MB ephemeral for tape */
    RCRE_GraphCtx* ctx = rcre_graph_ctx_create(8 * 1024 * 1024, 16 * 1024 * 1024);
    if (!ctx) {
        fprintf(stderr, "Failed to create graph context\n");
        return 1;
    }

    uint64_t rng = 20261024;

    /* 3. Generate non-linear synthetic regression dataset: y = sin(x1) + cos(x2) */
    RCRE_Tensor* X = rcre_tensor_alloc_2d(ctx->persistent_arena, N_SAMPLES, 2);
    RCRE_Tensor* y = rcre_tensor_alloc_2d(ctx->persistent_arena, N_SAMPLES, 1);

    for (size_t i = 0; i < N_SAMPLES; ++i) {
        float x1 = ((float)(i % 20) / 10.0f) - 1.0f;
        float x2 = ((float)(i / 20) / 5.0f) - 1.0f;
        rcre_tensor_f32_data(X)[i * 2 + 0] = x1;
        rcre_tensor_f32_data(X)[i * 2 + 1] = x2;
        rcre_tensor_f32_data(y)[i] = sinf(x1 * 3.14159f) + 0.5f * cosf(x2 * 3.14159f);
    }

    /* 4. Architecture: Layer 1: [2 -> 16], Layer 2: [16 -> 1] */
    RCRE_Tensor* W1 = rcre_tensor_alloc_2d(ctx->persistent_arena, 2, 16);
    RCRE_Tensor* b1 = rcre_tensor_alloc_1d(ctx->persistent_arena, 16);
    RCRE_Tensor* W2 = rcre_tensor_alloc_2d(ctx->persistent_arena, 16, 1);
    RCRE_Tensor* b2 = rcre_tensor_alloc_1d(ctx->persistent_arena, 1);

    rcre_tensor_init_xavier(W1, 2, 16, &rng);
    rcre_tensor_init_xavier(W2, 16, 1, &rng);
    rcre_tensor_zeros(b1);
    rcre_tensor_zeros(b2);

    /* 5. Initialize AdamW optimizer */
    RCRE_Tensor* model_params[4] = { W1, b1, W2, b2 };
    RCRE_TensorOptim* opt = rcre_tensor_optim_adamw_create(ctx->persistent_arena, model_params, 4,
                                                           0.02f, 0.9f, 0.999f, 1e-8f, 1e-4f);

    /* 6. Create tensor autograd tape on ephemeral arena */
    RCRE_TensorTape* tape = rcre_tensor_tape_create(ctx->tape_arena, 64);

    /* 7. Training loop */
    float loss_history[EPOCHS];
    printf("🚀 Training Deep MLP (%zu samples, %d epochs, batch_size=%d)...\n\n", (size_t)N_SAMPLES, EPOCHS, BATCH_SIZE);

    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        size_t n_batches = 0;

        for (size_t start = 0; start < N_SAMPLES; start += BATCH_SIZE) {
            size_t cur_b = (start + BATCH_SIZE <= N_SAMPLES) ? BATCH_SIZE : (N_SAMPLES - start);

            /* Scoped scratchpad checkpoint */
            RCRE_ArenaTemp temp = rcre_arena_temp_begin(ctx->tape_arena);

            /* View slice for mini-batch (stack struct, borrows X/y buffers;
             * must carry dtype F32 for the autograd ops' dtype guard) */
            RCRE_Tensor batch_x = {
                .data = rcre_tensor_f32_data(X) + start * 2,
                .shape = { cur_b, 2 },
                .strides = { 2, 1 },
                .ndim = 2,
                .numel = cur_b * 2,
                .dtype = RCRE_DTYPE_F32,
                .layout = RCRE_LAYOUT_TABULAR,
                .is_view = true
            };
            RCRE_Tensor batch_y = {
                .data = rcre_tensor_f32_data(y) + start * 1,
                .shape = { cur_b, 1 },
                .strides = { 1, 1 },
                .ndim = 2,
                .numel = cur_b * 1,
                .dtype = RCRE_DTYPE_F32,
                .layout = RCRE_LAYOUT_TABULAR,
                .is_view = true
            };

            rcre_tensor_tape_reset(tape);
            rcre_tensor_optim_zero_grad(opt);

            /* Forward Pass: H = ReLU(X * W1 + b1) */
            RCRE_Tensor* z1 = rcre_autograd_matmul(tape, &batch_x, W1);
            RCRE_Tensor* a1 = rcre_autograd_add_bias(tape, z1, b1);
            RCRE_Tensor* h = rcre_autograd_relu(tape, a1);

            /* Output: preds = H * W2 + b2 */
            RCRE_Tensor* z2 = rcre_autograd_matmul(tape, h, W2);
            RCRE_Tensor* preds = rcre_autograd_add_bias(tape, z2, b2);

            /* Compute MSE Loss and seed preds->grad */
            float loss = rcre_loss_mse(preds, &batch_y, preds);
            epoch_loss += loss;
            n_batches++;

            /* Backpropagation: Vector-Jacobian Product traversal */
            rcre_tensor_tape_backward(tape);

            /* AdamW weight update step */
            rcre_tensor_optim_step(opt);

            /* Scoped memory rewind */
            rcre_arena_temp_end(temp);
        }

        loss_history[epoch] = epoch_loss / (float)n_batches;
        if ((epoch + 1) % 20 == 0 || epoch == 0) {
            printf("  [Epoch %3zu/%d] Loss: %8.5f\n", epoch + 1, EPOCHS, loss_history[epoch]);
        }
    }

    /* 8. Render terminal loss curve visualization */
    rcre_vis_loss_history(loss_history, EPOCHS, "RCRE Deep MLP Convergence (AdamW)");

    printf("\033[1;32m✅ Training concluded successfully!\033[0m\n");
    printf("   Initial Loss: %.5f  -->  Final Loss: %.5f\n\n", loss_history[0], loss_history[EPOCHS - 1]);

    /* Clean teardown */
    rcre_tensor_tape_destroy(tape);
    rcre_graph_ctx_destroy(ctx);
    return 0;
}
