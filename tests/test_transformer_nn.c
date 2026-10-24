/**
 * @file test_transformer_nn.c
 * @brief Finite-Difference Numerical Verification for Transformer Kernels (ISO C23)
 */

#include "rcre.h"
#include "test_framework.h"

static bool is_close(float analytical, float numerical, float atol, float rtol) {
    float diff = fabsf(analytical - numerical);
    return diff <= (atol + rtol * fabsf(numerical));
}

static void test_linear_grad(void) {
    int N = 3, Din = 4, Dout = 2;
    float x[12] = {0.1f, -0.2f, 0.5f, 0.9f, -0.4f, 0.3f, 0.7f, -0.1f, 0.2f, 0.6f, -0.5f, 0.3f};
    float w[8]  = {0.4f, -0.3f, 0.2f, 0.5f, -0.1f, 0.8f, 0.6f, -0.7f};
    float b[2]  = {0.1f, -0.2f};
    float dout[6] = {0.5f, -0.4f, 0.3f, 0.2f, -0.1f, 0.6f};

    float dx[12], dw[8] = {0}, db[2] = {0};
    rcre_transformer_linear_backward(dx, dw, db, dout, x, w, N, Din, Dout);

    float eps = 1e-4f;
    float atol = 2e-4f, rtol = 1e-2f;

    /* Verify dw */
    for (int i = 0; i < Din * Dout; i++) {
        float orig = w[i];
        float out_pos[6], out_neg[6];

        w[i] = orig + eps;
        rcre_transformer_linear_forward(out_pos, x, w, b, N, Din, Dout);

        w[i] = orig - eps;
        rcre_transformer_linear_forward(out_neg, x, w, b, N, Din, Dout);

        w[i] = orig;

        float num_grad = 0.0f;
        for (int k = 0; k < 6; k++) {
            num_grad += dout[k] * (out_pos[k] - out_neg[k]) / (2.0f * eps);
        }
        ASSERT_TRUE(is_close(dw[i], num_grad, atol, rtol));
    }

    /* Verify db */
    for (int j = 0; j < Dout; j++) {
        float orig = b[j];
        float out_pos[6], out_neg[6];

        b[j] = orig + eps;
        rcre_transformer_linear_forward(out_pos, x, w, b, N, Din, Dout);

        b[j] = orig - eps;
        rcre_transformer_linear_forward(out_neg, x, w, b, N, Din, Dout);

        b[j] = orig;

        float num_grad = 0.0f;
        for (int k = 0; k < 6; k++) {
            num_grad += dout[k] * (out_pos[k] - out_neg[k]) / (2.0f * eps);
        }
        ASSERT_TRUE(is_close(db[j], num_grad, atol, rtol));
    }

    /* Verify dx */
    for (int i = 0; i < N * Din; i++) {
        float orig = x[i];
        float out_pos[6], out_neg[6];

        x[i] = orig + eps;
        rcre_transformer_linear_forward(out_pos, x, w, b, N, Din, Dout);

        x[i] = orig - eps;
        rcre_transformer_linear_forward(out_neg, x, w, b, N, Din, Dout);

        x[i] = orig;

        float num_grad = 0.0f;
        for (int k = 0; k < 6; k++) {
            num_grad += dout[k] * (out_pos[k] - out_neg[k]) / (2.0f * eps);
        }
        ASSERT_TRUE(is_close(dx[i], num_grad, atol, rtol));
    }
}

static void test_gelu_grad(void) {
    int N = 5;
    float x[5] = {-1.5f, -0.2f, 0.0f, 0.7f, 2.1f};
    float dout[5] = {0.3f, -0.5f, 0.8f, -0.2f, 0.4f};
    float dx[5];
    rcre_transformer_gelu_backward(dx, x, dout, N);

    float eps = 1e-4f;
    float atol = 2e-4f, rtol = 1e-2f;

    for (int i = 0; i < N; i++) {
        float f_pos = rcre_transformer_gelu_scalar(x[i] + eps);
        float f_neg = rcre_transformer_gelu_scalar(x[i] - eps);
        float num_grad = dout[i] * (f_pos - f_neg) / (2.0f * eps);
        ASSERT_TRUE(is_close(dx[i], num_grad, atol, rtol));
    }
}

static void test_layernorm_grad(void) {
    int N = 2, C = 4;
    float x[8] = {1.2f, -0.5f, 0.8f, -1.1f,  0.3f, 2.1f, -0.7f, 0.4f};
    float weight[4] = {1.1f, 0.9f, 1.05f, 0.95f};
    float bias[4]   = {0.1f, -0.05f, 0.02f, -0.1f};
    float dout[8]   = {0.4f, -0.2f, 0.3f, 0.1f, -0.5f, 0.6f, -0.3f, 0.2f};

    float out[8], mean[2], rstd[2];
    rcre_transformer_layernorm_forward(out, mean, rstd, x, weight, bias, N, C);

    float dx[8], dweight[4] = {0}, dbias[4] = {0};
    rcre_transformer_layernorm_backward(dx, dweight, dbias, dout, x, mean, rstd, weight, N, C);

    float eps = 1e-4f;
    float atol = 5e-4f, rtol = 2e-2f;

    /* Verify dweight */
    for (int i = 0; i < C; i++) {
        float orig = weight[i];
        float out_pos[8], out_neg[8], m_tmp[2], r_tmp[2];

        weight[i] = orig + eps;
        rcre_transformer_layernorm_forward(out_pos, m_tmp, r_tmp, x, weight, bias, N, C);

        weight[i] = orig - eps;
        rcre_transformer_layernorm_forward(out_neg, m_tmp, r_tmp, x, weight, bias, N, C);

        weight[i] = orig;

        float num_grad = 0.0f;
        for (int k = 0; k < 8; k++) {
            num_grad += dout[k] * (out_pos[k] - out_neg[k]) / (2.0f * eps);
        }
        ASSERT_TRUE(is_close(dweight[i], num_grad, atol, rtol));
    }

    /* Verify dbias */
    for (int i = 0; i < C; i++) {
        float orig = bias[i];
        float out_pos[8], out_neg[8], m_tmp[2], r_tmp[2];

        bias[i] = orig + eps;
        rcre_transformer_layernorm_forward(out_pos, m_tmp, r_tmp, x, weight, bias, N, C);

        bias[i] = orig - eps;
        rcre_transformer_layernorm_forward(out_neg, m_tmp, r_tmp, x, weight, bias, N, C);

        bias[i] = orig;

        float num_grad = 0.0f;
        for (int k = 0; k < 8; k++) {
            num_grad += dout[k] * (out_pos[k] - out_neg[k]) / (2.0f * eps);
        }
        ASSERT_TRUE(is_close(dbias[i], num_grad, atol, rtol));
    }

    /* Verify dx */
    for (int i = 0; i < N * C; i++) {
        float orig = x[i];
        float out_pos[8], out_neg[8], m_tmp[2], r_tmp[2];

        x[i] = orig + eps;
        rcre_transformer_layernorm_forward(out_pos, m_tmp, r_tmp, x, weight, bias, N, C);

        x[i] = orig - eps;
        rcre_transformer_layernorm_forward(out_neg, m_tmp, r_tmp, x, weight, bias, N, C);

        x[i] = orig;

        float num_grad = 0.0f;
        for (int k = 0; k < 8; k++) {
            num_grad += dout[k] * (out_pos[k] - out_neg[k]) / (2.0f * eps);
        }
        ASSERT_TRUE(is_close(dx[i], num_grad, atol, rtol));
    }
}

static void test_softmax_grad(void) {
    int N = 2, C = 3;
    float inp[6] = {0.5f, 1.2f, -0.3f, -1.0f, 0.0f, 2.0f};
    float dout[6] = {0.1f, -0.4f, 0.3f, 0.2f, 0.5f, -0.7f};
    float out[6], dinp[6];

    rcre_transformer_softmax_forward(out, inp, N, C);
    rcre_transformer_softmax_backward(dinp, dout, out, N, C);

    float eps = 1e-4f;
    float atol = 2e-4f, rtol = 1e-2f;

    for (int i = 0; i < N * C; i++) {
        float orig = inp[i];
        float out_pos[6], out_neg[6];

        inp[i] = orig + eps;
        rcre_transformer_softmax_forward(out_pos, inp, N, C);

        inp[i] = orig - eps;
        rcre_transformer_softmax_forward(out_neg, inp, N, C);

        inp[i] = orig;

        float num_grad = 0.0f;
        for (int k = 0; k < 6; k++) {
            num_grad += dout[k] * (out_pos[k] - out_neg[k]) / (2.0f * eps);
        }
        ASSERT_TRUE(is_close(dinp[i], num_grad, atol, rtol));
    }
}

static void test_attention_grad(void) {
    int B = 1, T = 3, C = 4, H = 2;
    int total = B * T * C;

    float q[12] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f, 0.9f, -1.0f, 1.1f, -1.2f};
    float k[12] = {0.2f, 0.3f, -0.4f, 0.5f, 0.6f, -0.7f, 0.8f, -0.9f, 1.0f, -1.1f, 1.2f, -1.3f};
    float v[12] = {-0.1f, 0.2f, 0.5f, -0.3f, 0.4f, 0.1f, -0.6f, 0.7f, -0.8f, 0.9f, -0.2f, 0.3f};
    float dout[12] = {0.3f, -0.1f, 0.4f, 0.2f, -0.5f, 0.6f, 0.1f, -0.2f, 0.7f, -0.4f, 0.3f, 0.5f};

    float out[12], att_scores[1 * 2 * 3 * 3];
    rcre_transformer_attention_forward(out, att_scores, q, k, v, B, T, C, H);

    float dq[12], dk[12], dv[12], datt_scores[18];
    rcre_transformer_attention_backward(dq, dk, dv, datt_scores, dout, att_scores, q, k, v, B, T, C, H);

    float eps = 1e-4f;
    float atol = 5e-4f, rtol = 2e-2f;

    /* Verify dv */
    for (int i = 0; i < total; i++) {
        float orig = v[i];
        float out_pos[12], out_neg[12], att_tmp[18];

        v[i] = orig + eps;
        rcre_transformer_attention_forward(out_pos, att_tmp, q, k, v, B, T, C, H);

        v[i] = orig - eps;
        rcre_transformer_attention_forward(out_neg, att_tmp, q, k, v, B, T, C, H);

        v[i] = orig;

        float num_grad = 0.0f;
        for (int j = 0; j < total; j++) {
            num_grad += dout[j] * (out_pos[j] - out_neg[j]) / (2.0f * eps);
        }
        ASSERT_TRUE(is_close(dv[i], num_grad, atol, rtol));
    }

    /* Verify dq */
    for (int i = 0; i < total; i++) {
        float orig = q[i];
        float out_pos[12], out_neg[12], att_tmp[18];

        q[i] = orig + eps;
        rcre_transformer_attention_forward(out_pos, att_tmp, q, k, v, B, T, C, H);

        q[i] = orig - eps;
        rcre_transformer_attention_forward(out_neg, att_tmp, q, k, v, B, T, C, H);

        q[i] = orig;

        float num_grad = 0.0f;
        for (int j = 0; j < total; j++) {
            num_grad += dout[j] * (out_pos[j] - out_neg[j]) / (2.0f * eps);
        }
        ASSERT_TRUE(is_close(dq[i], num_grad, atol, rtol));
    }
}

static void test_cross_entropy_grad(void) {
    int N = 2, V = 4;
    float logits[8] = {1.0f, 2.0f, -0.5f, 0.2f,  -1.0f, 0.5f, 2.5f, 1.2f};
    int targets[2] = {1, 2};

    float probs[8];
    float loss = rcre_transformer_cross_entropy_forward(probs, logits, targets, N, V);
    ASSERT_TRUE(loss > 0.0f);

    float dlogits[8];
    rcre_transformer_cross_entropy_backward(dlogits, probs, targets, N, V);

    float eps = 1e-4f;
    float atol = 5e-4f, rtol = 2e-2f;

    for (int i = 0; i < N * V; i++) {
        float orig = logits[i];
        float p_pos[8], p_neg[8];

        logits[i] = orig + eps;
        float l_pos = rcre_transformer_cross_entropy_forward(p_pos, logits, targets, N, V);

        logits[i] = orig - eps;
        float l_neg = rcre_transformer_cross_entropy_forward(p_neg, logits, targets, N, V);

        logits[i] = orig;

        float num_grad = (l_pos - l_neg) / (2.0f * eps);
        ASSERT_TRUE(is_close(dlogits[i], num_grad, atol, rtol));
    }
}

int main(void) {
    test_suite_begin("RCRE Transformer Analytical & Finite-Difference Gradient Verification");
    TEST_RUN(test_linear_grad);
    TEST_RUN(test_gelu_grad);
    TEST_RUN(test_layernorm_grad);
    TEST_RUN(test_softmax_grad);
    TEST_RUN(test_attention_grad);
    TEST_RUN(test_cross_entropy_grad);
    return test_suite_report();
}
