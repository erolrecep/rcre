/**
 * @file nn.c
 * @brief RCRE Transformer Neural Network Kernels & Mathematical Primitives (ISO C23)
 */

#include "transformer_internal.h"

/* ========================================================================= */
/* Transpose-Capable GEMM Driver                                             */
/* ========================================================================= */

void rcre_transformer_gemm(
    bool transA, bool transB,
    size_t M, size_t N, size_t K,
    float alpha,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    float beta,
    float *C, size_t ldc
) {
    /* Fast path: Direct HAL dispatch when no transpositions are requested */
    if (!transA && !transB) {
        rcre_hal.gemm_f32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    /* Initialize or scale C */
    if (beta == 0.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    if (alpha == 0.0f) return;

    /* Cache-blocked GEMM: 64x64x64 tiles */
    const size_t MB = 64;
    const size_t NB = 64;
    const size_t KB = 64;

    for (size_t ii = 0; ii < M; ii += MB) {
        size_t m_end = (ii + MB < M) ? ii + MB : M;
        for (size_t kk = 0; kk < K; kk += KB) {
            size_t k_end = (kk + KB < K) ? kk + KB : K;
            for (size_t jj = 0; jj < N; jj += NB) {
                size_t j_end = (jj + NB < N) ? jj + NB : N;

                for (size_t i = ii; i < m_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        float a_val = alpha * (transA ? A[k * lda + i] : A[i * lda + k]);
                        for (size_t j = jj; j < j_end; ++j) {
                            float b_val = transB ? B[j * ldb + k] : B[k * ldb + j];
                            C[i * ldc + j] += a_val * b_val;
                        }
                    }
                }
            }
        }
    }
}

/* ========================================================================= */
/* 1. Linear Projection (Dense Layer)                                        */
/* ========================================================================= */

void rcre_transformer_linear_forward(float *out, const float *x, const float *w, const float *b, int N, int Din, int Dout) {
    if (b != nullptr) {
        for (int n = 0; n < N; n++) {
            memcpy(out + n * Dout, b, sizeof(float) * Dout);
        }
    }
    rcre_transformer_gemm(
        false, false,
        (size_t)N, (size_t)Dout, (size_t)Din,
        1.0f,
        x, (size_t)Din,
        w, (size_t)Dout,
        (b != nullptr) ? 1.0f : 0.0f,
        out, (size_t)Dout
    );
}

void rcre_transformer_linear_backward(float *dx, float *dw, float *db, const float *dout, const float *x, const float *w, int N, int Din, int Dout) {
    /* 1. dx = dout * W^T */
    if (dx != nullptr) {
        rcre_transformer_gemm(
            false, true,
            (size_t)N, (size_t)Din, (size_t)Dout,
            1.0f,
            dout, (size_t)Dout,
            w, (size_t)Dout,
            0.0f,
            dx, (size_t)Din
        );
    }

    /* 2. dw += X^T * dout (accumulate) */
    if (dw != nullptr) {
        rcre_transformer_gemm(
            true, false,
            (size_t)Din, (size_t)Dout, (size_t)N,
            1.0f,
            x, (size_t)Din,
            dout, (size_t)Dout,
            1.0f, /* Accumulate into existing dw! */
            dw, (size_t)Dout
        );
    }

    /* 3. db += sum(dout, axis=0) */
    if (db != nullptr) {
        for (int j = 0; j < Dout; j++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                sum += dout[n * Dout + j];
            }
            db[j] += sum;
        }
    }
}

/* ========================================================================= */
/* 2. GELU Non-Linearity                                                     */
/* ========================================================================= */

static const float SQRT_2_OVER_PI = 0.7978845608028654f;
static const float GELU_COEF = 0.044715f;

float rcre_transformer_gelu_scalar(float x) {
    float cube = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * cube);
    return 0.5f * x * (1.0f + tanhf(inner));
}

float rcre_transformer_gelu_scalar_backward(float x, float dout) {
    float cube = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * cube);
    float t = tanhf(inner);
    float dt_dinner = 1.0f - t * t;
    float dinner_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x * x);
    float dgelu_dx = 0.5f * (1.0f + t) + 0.5f * x * dt_dinner * dinner_dx;
    return dout * dgelu_dx;
}

void rcre_transformer_gelu_forward(float *out, const float *x, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = rcre_transformer_gelu_scalar(x[i]);
    }
}

void rcre_transformer_gelu_backward(float *dx, const float *x, const float *dout, int N) {
    for (int i = 0; i < N; i++) {
        dx[i] = rcre_transformer_gelu_scalar_backward(x[i], dout[i]);
    }
}

/* ========================================================================= */
/* 3. Layer Normalization                                                    */
/* ========================================================================= */

static const float LN_EPS = 1e-5f;

void rcre_transformer_layernorm_forward(float *out, float *mean, float *rstd, const float *x, const float *weight, const float *bias, int N, int C) {
    for (int n = 0; n < N; n++) {
        const float *x_row = x + n * C;
        float *out_row = out + n * C;

        /* Mean */
        float sum = 0.0f;
        for (int i = 0; i < C; i++) {
            sum += x_row[i];
        }
        float m = sum / (float)C;

        /* Variance */
        float var_sum = 0.0f;
        for (int i = 0; i < C; i++) {
            float diff = x_row[i] - m;
            var_sum += diff * diff;
        }
        float r = 1.0f / sqrtf(var_sum / (float)C + LN_EPS);

        mean[n] = m;
        rstd[n] = r;

        /* Normalize & Scale */
        for (int i = 0; i < C; i++) {
            float x_hat = (x_row[i] - m) * r;
            out_row[i] = x_hat * weight[i] + bias[i];
        }
    }
}

void rcre_transformer_layernorm_backward(float *dx, float *dweight, float *dbias, const float *dout, const float *x, const float *mean, const float *rstd, const float *weight, int N, int C) {
    for (int n = 0; n < N; n++) {
        const float *dout_row = dout + n * C;
        const float *x_row = x + n * C;
        float *dx_row = dx + n * C;
        float m = mean[n];
        float r = rstd[n];

        /* Dot-products: sum(dout * gamma) and sum(dout * gamma * x_hat) */
        float sum_dhat = 0.0f;
        float sum_dhat_xhat = 0.0f;
        for (int i = 0; i < C; i++) {
            float x_hat = (x_row[i] - m) * r;
            float dhat = dout_row[i] * weight[i];
            sum_dhat += dhat;
            sum_dhat_xhat += dhat * x_hat;

            if (dweight != nullptr) dweight[i] += dout_row[i] * x_hat;
            if (dbias != nullptr) dbias[i] += dout_row[i];
        }

        /* Input gradient dx */
        float inv_C = 1.0f / (float)C;
        for (int i = 0; i < C; i++) {
            float x_hat = (x_row[i] - m) * r;
            float dhat = dout_row[i] * weight[i];
            dx_row[i] = r * (dhat - inv_C * sum_dhat - inv_C * x_hat * sum_dhat_xhat);
        }
    }
}

/* ========================================================================= */
/* 4. Softmax                                                                */
/* ========================================================================= */

void rcre_transformer_softmax_forward(float *out, const float *inp, int N, int C) {
    for (int n = 0; n < N; n++) {
        const float *inp_row = inp + n * C;
        float *out_row = out + n * C;

        /* Max for numerical stability */
        float max_val = inp_row[0];
        for (int i = 1; i < C; i++) {
            if (inp_row[i] > max_val) max_val = inp_row[i];
        }

        /* Exponentiate & sum */
        float sum = 0.0f;
        for (int i = 0; i < C; i++) {
            float e = expf(inp_row[i] - max_val);
            out_row[i] = e;
            sum += e;
        }

        float inv_sum = 1.0f / sum;
        for (int i = 0; i < C; i++) {
            out_row[i] *= inv_sum;
        }
    }
}

void rcre_transformer_softmax_backward(float *dinp, const float *dout, const float *out, int N, int C) {
    for (int n = 0; n < N; n++) {
        const float *dout_row = dout + n * C;
        const float *out_row = out + n * C;
        float *dinp_row = dinp + n * C;

        float sum = 0.0f;
        for (int i = 0; i < C; i++) {
            sum += dout_row[i] * out_row[i];
        }

        for (int i = 0; i < C; i++) {
            dinp_row[i] = out_row[i] * (dout_row[i] - sum);
        }
    }
}

/* ========================================================================= */
/* 5. Causal Multi-Head Self-Attention                                       */
/* ========================================================================= */

void rcre_transformer_attention_forward(
    float *out, float *att_scores,
    const float *q, const float *k, const float *v,
    int B, int T, int C, int H
) {
    int d_k = C / H;
    float scale = 1.0f / sqrtf((float)d_k);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int tq = 0; tq < T; tq++) {
                const float *q_row = q + IDX4(b, h, tq, 0, H, T, d_k);
                float *att_row = att_scores + IDX_ATTN(b, h, tq, 0, H, T);

                /* 1. Raw dot-product scores for causal positions: tk <= tq */
                float max_val = -1e9f;
                for (int tk = 0; tk <= tq; tk++) {
                    const float *k_row = k + IDX4(b, h, tk, 0, H, T, d_k);
                    float dot = 0.0f;
                    for (int d = 0; d < d_k; d++) {
                        dot += q_row[d] * k_row[d];
                    }
                    dot *= scale;
                    att_row[tk] = dot;
                    if (dot > max_val) max_val = dot;
                }

                /* Causal mask */
                for (int tk = tq + 1; tk < T; tk++) {
                    att_row[tk] = 0.0f;
                }

                /* 2. Softmax over tk <= tq */
                float sum = 0.0f;
                for (int tk = 0; tk <= tq; tk++) {
                    float e = expf(att_row[tk] - max_val);
                    att_row[tk] = e;
                    sum += e;
                }
                float inv_sum = 1.0f / sum;
                for (int tk = 0; tk <= tq; tk++) {
                    att_row[tk] *= inv_sum;
                }

                /* 3. Weighted sum of V: out = att_scores * V */
                float *out_head_row = out + IDX4(b, h, tq, 0, H, T, d_k);
                for (int d = 0; d < d_k; d++) {
                    float val = 0.0f;
                    for (int tk = 0; tk <= tq; tk++) {
                        const float *v_row = v + IDX4(b, h, tk, 0, H, T, d_k);
                        val += att_row[tk] * v_row[d];
                    }
                    out_head_row[d] = val;
                }
            }
        }
    }
}

void rcre_transformer_attention_backward(
    float *dq, float *dk, float *dv,
    float *datt_scores,
    const float *dout, const float *att_scores,
    const float *q, const float *k, const float *v,
    int B, int T, int C, int H
) {
    int d_k = C / H;
    float scale = 1.0f / sqrtf((float)d_k);

    memset(dq, 0, sizeof(float) * B * H * T * d_k);
    memset(dk, 0, sizeof(float) * B * H * T * d_k);
    memset(dv, 0, sizeof(float) * B * H * T * d_k);
    memset(datt_scores, 0, sizeof(float) * B * H * T * T);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int tq = 0; tq < T; tq++) {
                const float *dout_head_row = dout + IDX4(b, h, tq, 0, H, T, d_k);
                const float *att_row = att_scores + IDX_ATTN(b, h, tq, 0, H, T);
                float *datt_row = datt_scores + IDX_ATTN(b, h, tq, 0, H, T);

                /* 1. Gradient w.r.t. att_scores and w.r.t. V */
                for (int tk = 0; tk <= tq; tk++) {
                    const float *v_row = v + IDX4(b, h, tk, 0, H, T, d_k);
                    float *dv_row = dv + IDX4(b, h, tk, 0, H, T, d_k);
                    float d_att = 0.0f;
                    for (int d = 0; d < d_k; d++) {
                        d_att += dout_head_row[d] * v_row[d];
                        dv_row[d] += att_row[tk] * dout_head_row[d];
                    }
                    datt_row[tk] = d_att;
                }

                /* 2. Gradient through causal softmax */
                float sum_datt_att = 0.0f;
                for (int tk = 0; tk <= tq; tk++) {
                    sum_datt_att += datt_row[tk] * att_row[tk];
                }

                /* 3. Gradient w.r.t. Q and K */
                float *dq_row = dq + IDX4(b, h, tq, 0, H, T, d_k);
                for (int tk = 0; tk <= tq; tk++) {
                    float dscore = att_row[tk] * (datt_row[tk] - sum_datt_att) * scale;
                    const float *k_row = k + IDX4(b, h, tk, 0, H, T, d_k);
                    float *dk_row = dk + IDX4(b, h, tk, 0, H, T, d_k);
                    const float *q_row = q + IDX4(b, h, tq, 0, H, T, d_k);

                    for (int d = 0; d < d_k; d++) {
                        dq_row[d] += dscore * k_row[d];
                        dk_row[d] += dscore * q_row[d];
                    }
                }
            }
        }
    }
}

/* ========================================================================= */
/* 6. Cross-Entropy Loss with Softmax over Vocabulary                         */
/* ========================================================================= */

float rcre_transformer_cross_entropy_forward(float *probs, const float *logits, const int *targets, int N, int V) {
    rcre_transformer_softmax_forward(probs, logits, N, V);

    float total_loss = 0.0f;
    for (int n = 0; n < N; n++) {
        int target = targets[n];
        if (target < 0 || target >= V) {
            /* v0.2.2-style clamp policy: clamp instead of crash; callers
             * with invalid labels get a warning under RCRE_DEBUG. */
#ifdef RCRE_DEBUG
            {
                fprintf(stderr, "[rcre][warn] cross_entropy: target %d out of [0,%d); clamped\n", target, V);
            }
#endif
            target = (target < 0) ? 0 : V - 1;
        }
        float p = probs[n * V + target];
        total_loss += -logf(p + 1e-12f);
    }
    return total_loss / (float)N;
}

void rcre_transformer_cross_entropy_backward(float *dlogits, const float *probs, const int *targets, int N, int V) {
    float inv_N = 1.0f / (float)N;
    for (int n = 0; n < N; n++) {
        int target = targets[n];
        for (int v = 0; v < V; v++) {
            float p = probs[n * V + v];
            if (v == target) {
                dlogits[n * V + v] = (p - 1.0f) * inv_N;
            } else {
                dlogits[n * V + v] = p * inv_N;
            }
        }
    }
}
