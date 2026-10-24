/**
 * @file linear.c
 * @brief Classical Linear Models (OLS, Logistic Regression, Pegasos SVM)
 * 
 * Satisfies ML-REQ-01.
 */

#include "rcre.h"
#include "ml_internal.h"
#include <math.h>
#include <string.h>

/* Solve A * x = b via Gaussian elimination with partial pivoting */
static bool solve_linear_system(float* A, float* b, float* x, size_t n) {
    /* Forward elimination */
    for (size_t i = 0; i < n; ++i) {
        size_t max_row = i;
        float max_val = fabsf(A[i * n + i]);

        for (size_t r = i + 1; r < n; ++r) {
            float val = fabsf(A[r * n + i]);
            if (val > max_val) {
                max_val = val;
                max_row = r;
            }
        }

        if (max_val < 1e-12f) {
            return false; /* Singular or nearly singular */
        }

        /* Swap rows in A and b */
        if (max_row != i) {
            for (size_t c = 0; c < n; ++c) {
                float tmp = A[i * n + c];
                A[i * n + c] = A[max_row * n + c];
                A[max_row * n + c] = tmp;
            }
            float tmp_b = b[i];
            b[i] = b[max_row];
            b[max_row] = tmp_b;
        }

        /* Pivot */
        float pivot = A[i * n + i];
        for (size_t r = i + 1; r < n; ++r) {
            float factor = A[r * n + i] / pivot;
            for (size_t c = i; c < n; ++c) {
                A[r * n + c] -= factor * A[i * n + c];
            }
            b[r] -= factor * b[i];
        }
    }

    /* Back substitution */
    for (int i = (int)n - 1; i >= 0; --i) {
        float sum = b[i];
        for (size_t c = (size_t)i + 1; c < n; ++c) {
            sum -= A[(size_t)i * n + c] * x[c];
        }
        x[i] = sum / A[(size_t)i * n + (size_t)i];
    }

    return true;
}

RCRE_NODISCARD int rcre_ml_ols_fit(const RCRE_Tensor* X, const RCRE_Tensor* y, 
                                   RCRE_Tensor* weights, RCRE_Arena* arena) {
    if (!X || !X->data || !y || !y->data || !weights || !weights->data || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    if (X->ndim != 2 || y->ndim < 1 || weights->ndim < 1) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t N = X->shape[0];
    size_t D = X->shape[1];

    if (weights->shape[0] != D || y->shape[0] != N) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* X_data = (const float*)X->data;
    const float* y_data = (const float*)y->data;
    float* w_data = (float*)weights->data;

    /* Allocate Gram matrix XtX (D x D) and Xty (D x 1) from arena */
    float* XtX = (float*)rcre_arena_alloc(arena, D * D * sizeof(float), 64);
    float* Xty = (float*)rcre_arena_alloc(arena, D * sizeof(float), 64);
    if (!XtX || !Xty) {
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    memset(XtX, 0, D * D * sizeof(float));
    memset(Xty, 0, D * sizeof(float));

    /* Compute XtX and Xty */
    for (size_t i = 0; i < N; ++i) {
        const float* row = &X_data[i * D];
        float y_i = y_data[i];

        for (size_t r = 0; r < D; ++r) {
            float x_ir = row[r];
            Xty[r] += x_ir * y_i;
            for (size_t c = r; c < D; ++c) {
                XtX[r * D + c] += x_ir * row[c];
            }
        }
    }

    /* Symmetric fill & small ridge stabilization */
    for (size_t r = 0; r < D; ++r) {
        XtX[r * D + r] += 1e-6f;
        for (size_t c = r + 1; c < D; ++c) {
            XtX[c * D + r] = XtX[r * D + c];
        }
    }

    if (!solve_linear_system(XtX, Xty, w_data, D)) {
        return RCRE_ERR_UNSUPPORTED_OP;
    }

    return RCRE_OK;
}



RCRE_NODISCARD int rcre_ml_logistic_fit(const RCRE_Tensor* X, const RCRE_Tensor* y, 
                                        RCRE_Tensor* weights, float lr, 
                                        size_t epochs, RCRE_Arena* arena) {
    if (!X || !X->data || !y || !y->data || !weights || !weights->data || !arena) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t N = X->shape[0];
    size_t D = X->shape[1];

    const float* X_data = (const float*)X->data;
    const float* y_data = (const float*)y->data;
    float* w_data = (float*)weights->data;

    float* grad = (float*)rcre_arena_alloc(arena, D * sizeof(float), 64);
    if (!grad) return RCRE_ERR_OUT_OF_MEMORY;

    /* Initialize weights to zero */
    memset(w_data, 0, D * sizeof(float));

    float inv_N = 1.0f / (float)N;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        memset(grad, 0, D * sizeof(float));

        for (size_t i = 0; i < N; ++i) {
            const float* row = &X_data[i * D];
            float z = rcre_hal.dot_f32(row, w_data, D);

            float error = rcre_fast_sigmoidf(z) - y_data[i];
            rcre_hal.axpy_f32(D, error, row, grad);
        }

        /* Parameter update: w = w - (lr / N) * grad */
        rcre_hal.axpy_f32(D, -lr * inv_N, grad, w_data);
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_ml_pegasos_svm_fit(const RCRE_Tensor* X, const RCRE_Tensor* y, 
                                           RCRE_Tensor* weights, float lambda_reg, 
                                           size_t epochs, RCRE_Arena* arena) {
    (void)arena;
    if (!X || !X->data || !y || !y->data || !weights || !weights->data || lambda_reg <= 0.0f) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t N = X->shape[0];
    size_t D = X->shape[1];

    const float* X_data = (const float*)X->data;
    const float* y_data = (const float*)y->data;
    float* w_data = (float*)weights->data;

    memset(w_data, 0, D * sizeof(float));
    size_t t = 1;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < N; ++i) {
            float eta = 1.0f / (lambda_reg * (float)t);
            t++;

            const float* row = &X_data[i * D];
            float dot = rcre_hal.dot_f32(row, w_data, D);

            float y_i = (y_data[i] > 0.0f) ? 1.0f : -1.0f;
            float margin = y_i * dot;

            if (margin < 1.0f) {
                for (size_t d = 0; d < D; ++d) {
                    w_data[d] = (1.0f - eta * lambda_reg) * w_data[d] + eta * y_i * row[d];
                }
            } else {
                for (size_t d = 0; d < D; ++d) {
                    w_data[d] = (1.0f - eta * lambda_reg) * w_data[d];
                }
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_ml_predict_linear(const RCRE_Tensor* X, const RCRE_Tensor* weights, 
                                          RCRE_Tensor* preds) {
    if (!X || !X->data || !weights || !weights->data || !preds || !preds->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t N = X->shape[0];
    size_t D = X->shape[1];

    const float* X_data = (const float*)X->data;
    const float* w_data = (const float*)weights->data;
    float* p_data = (float*)preds->data;

    for (size_t i = 0; i < N; ++i) {
        p_data[i] = rcre_hal.dot_f32(&X_data[i * D], w_data, D);
    }

    return RCRE_OK;
}
