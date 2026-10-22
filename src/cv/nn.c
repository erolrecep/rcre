/**
 * @file nn.c
 * @brief Convolutional Neural Network (CNN) Inference Primitives
 * 
 * Satisfies NN-REQ-01 through NN-REQ-05.
 * Implements pure C23 zero-allocation inference operators:
 * - 2D Spatial Convolution (Conv2D)
 * - Depthwise Separable 2D Convolution (Depthwise Conv2D)
 * - 1x1 Pointwise Convolution (Channel Projection)
 * - 2D Batch Normalization & In-Place BN Folding (Zero-Cost Edge BN)
 * - ReLU6, Global Average Pooling, Linear (FC), and Numerically Stable Softmax
 */

#include "rcre.h"
#include <math.h>
#include <string.h>
#include <float.h>

static void ensure_hal_init(void) {
    if (!rcre_hal.dot_f32 || !rcre_hal.axpy_f32) {
        rcre_hal_init();
    }
}

/* ========================================================================= */
/* 2D Spatial Convolution                                                    */
/* ========================================================================= */

RCRE_NODISCARD int rcre_nn_conv2d(const RCRE_Tensor* input,
                                  const RCRE_Tensor* weights,
                                  const RCRE_Tensor* bias,
                                  const RCRE_Conv2DParams* params,
                                  RCRE_Tensor* output,
                                  RCRE_Arena* arena) {
    if (!input || !input->data || !weights || !weights->data || !output) {
        return RCRE_ERR_INVALID_ARG;
    }
    ensure_hal_init();

    size_t batch_size = 1;
    size_t in_c = 0, in_h = 0, in_w = 0;

    if (input->ndim == 3) {
        in_c = input->shape[0];
        in_h = input->shape[1];
        in_w = input->shape[2];
    } else if (input->ndim == 4) {
        batch_size = input->shape[0];
        in_c = input->shape[1];
        in_h = input->shape[2];
        in_w = input->shape[3];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (weights->ndim != 4) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t out_c = weights->shape[0];
    if (weights->shape[1] != in_c) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }
    size_t k_h = weights->shape[2];
    size_t k_w = weights->shape[3];

    size_t stride_h = (params && params->stride_h > 0) ? params->stride_h : 1;
    size_t stride_w = (params && params->stride_w > 0) ? params->stride_w : 1;
    size_t pad_h = params ? params->pad_h : 0;
    size_t pad_w = params ? params->pad_w : 0;

    if (in_h + 2 * pad_h < k_h || in_w + 2 * pad_w < k_w) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    size_t out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    size_t out_shape[4];
    uint8_t out_ndim = input->ndim;
    if (input->ndim == 3) {
        out_shape[0] = out_c;
        out_shape[1] = out_h;
        out_shape[2] = out_w;
    } else {
        out_shape[0] = batch_size;
        out_shape[1] = out_c;
        out_shape[2] = out_h;
        out_shape[3] = out_w;
    }

    if (!output->data) {
        if (!arena) return RCRE_ERR_INVALID_ARG;
        *output = rcre_tensor_create(out_shape, out_ndim, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW, arena);
        if (!output->data) return RCRE_ERR_OUT_OF_MEMORY;
    } else {
        if (output->ndim != out_ndim) return RCRE_ERR_SHAPE_MISMATCH;
        for (uint8_t i = 0; i < out_ndim; ++i) {
            if (output->shape[i] != out_shape[i]) return RCRE_ERR_SHAPE_MISMATCH;
        }
    }

    const float* in_ptr = (const float*)input->data;
    const float* w_ptr = (const float*)weights->data;
    const float* b_ptr = (bias && bias->data) ? (const float*)bias->data : nullptr;
    float* out_ptr = (float*)output->data;

    size_t in_channel_stride = in_h * in_w;
    size_t in_batch_stride = in_c * in_channel_stride;
    size_t out_channel_stride = out_h * out_w;
    size_t out_batch_stride = out_c * out_channel_stride;
    size_t kernel_spatial = k_h * k_w;
    size_t filter_stride = in_c * kernel_spatial;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* cur_in = in_ptr + b * in_batch_stride;
        float* cur_out = out_ptr + b * out_batch_stride;

        for (size_t oc = 0; oc < out_c; ++oc) {
            float bias_val = b_ptr ? b_ptr[oc] : 0.0f;
            const float* filter_weights = w_ptr + oc * filter_stride;
            float* oc_out = cur_out + oc * out_channel_stride;

            for (size_t oh = 0; oh < out_h; ++oh) {
                int64_t ih_base = (int64_t)(oh * stride_h) - (int64_t)pad_h;
                for (size_t ow = 0; ow < out_w; ++ow) {
                    int64_t iw_base = (int64_t)(ow * stride_w) - (int64_t)pad_w;
                    float sum = bias_val;

                    for (size_t ic = 0; ic < in_c; ++ic) {
                        const float* ic_in = cur_in + ic * in_channel_stride;
                        const float* ic_w = filter_weights + ic * kernel_spatial;

                        for (size_t kh = 0; kh < k_h; ++kh) {
                            int64_t ih = ih_base + (int64_t)kh;
                            if (ih < 0 || ih >= (int64_t)in_h) continue;

                            for (size_t kw = 0; kw < k_w; ++kw) {
                                int64_t iw = iw_base + (int64_t)kw;
                                if (iw < 0 || iw >= (int64_t)in_w) continue;

                                float x_val = ic_in[ih * (int64_t)in_w + iw];
                                float w_val = ic_w[kh * k_w + kw];
                                sum += x_val * w_val;
                            }
                        }
                    }
                    oc_out[oh * out_w + ow] = sum;
                }
            }
        }
    }

    return RCRE_OK;
}

/* ========================================================================= */
/* Depthwise 2D Convolution                                                  */
/* ========================================================================= */

RCRE_NODISCARD int rcre_nn_depthwise_conv2d(const RCRE_Tensor* input,
                                            const RCRE_Tensor* weights,
                                            const RCRE_Tensor* bias,
                                            const RCRE_Conv2DParams* params,
                                            RCRE_Tensor* output,
                                            RCRE_Arena* arena) {
    if (!input || !input->data || !weights || !weights->data || !output) {
        return RCRE_ERR_INVALID_ARG;
    }
    ensure_hal_init();

    size_t batch_size = 1;
    size_t in_c = 0, in_h = 0, in_w = 0;

    if (input->ndim == 3) {
        in_c = input->shape[0];
        in_h = input->shape[1];
        in_w = input->shape[2];
    } else if (input->ndim == 4) {
        batch_size = input->shape[0];
        in_c = input->shape[1];
        in_h = input->shape[2];
        in_w = input->shape[3];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t k_h = 0, k_w = 0;
    if (weights->ndim == 4) {
        if (weights->shape[0] != in_c || weights->shape[1] != 1) {
            return RCRE_ERR_SHAPE_MISMATCH;
        }
        k_h = weights->shape[2];
        k_w = weights->shape[3];
    } else if (weights->ndim == 3) {
        if (weights->shape[0] != in_c) {
            return RCRE_ERR_SHAPE_MISMATCH;
        }
        k_h = weights->shape[1];
        k_w = weights->shape[2];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t stride_h = (params && params->stride_h > 0) ? params->stride_h : 1;
    size_t stride_w = (params && params->stride_w > 0) ? params->stride_w : 1;
    size_t pad_h = params ? params->pad_h : 0;
    size_t pad_w = params ? params->pad_w : 0;

    if (in_h + 2 * pad_h < k_h || in_w + 2 * pad_w < k_w) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    size_t out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    size_t out_shape[4];
    uint8_t out_ndim = input->ndim;
    if (input->ndim == 3) {
        out_shape[0] = in_c;
        out_shape[1] = out_h;
        out_shape[2] = out_w;
    } else {
        out_shape[0] = batch_size;
        out_shape[1] = in_c;
        out_shape[2] = out_h;
        out_shape[3] = out_w;
    }

    if (!output->data) {
        if (!arena) return RCRE_ERR_INVALID_ARG;
        *output = rcre_tensor_create(out_shape, out_ndim, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW, arena);
        if (!output->data) return RCRE_ERR_OUT_OF_MEMORY;
    } else {
        if (output->ndim != out_ndim) return RCRE_ERR_SHAPE_MISMATCH;
        for (uint8_t i = 0; i < out_ndim; ++i) {
            if (output->shape[i] != out_shape[i]) return RCRE_ERR_SHAPE_MISMATCH;
        }
    }

    const float* in_ptr = (const float*)input->data;
    const float* w_ptr = (const float*)weights->data;
    const float* b_ptr = (bias && bias->data) ? (const float*)bias->data : nullptr;
    float* out_ptr = (float*)output->data;

    size_t in_channel_stride = in_h * in_w;
    size_t in_batch_stride = in_c * in_channel_stride;
    size_t out_channel_stride = out_h * out_w;
    size_t out_batch_stride = in_c * out_channel_stride;
    size_t kernel_spatial = k_h * k_w;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* cur_in = in_ptr + b * in_batch_stride;
        float* cur_out = out_ptr + b * out_batch_stride;

        for (size_t c = 0; c < in_c; ++c) {
            float bias_val = b_ptr ? b_ptr[c] : 0.0f;
            const float* c_w = w_ptr + c * kernel_spatial;
            const float* c_in = cur_in + c * in_channel_stride;
            float* c_out = cur_out + c * out_channel_stride;

            for (size_t oh = 0; oh < out_h; ++oh) {
                int64_t ih_base = (int64_t)(oh * stride_h) - (int64_t)pad_h;
                for (size_t ow = 0; ow < out_w; ++ow) {
                    int64_t iw_base = (int64_t)(ow * stride_w) - (int64_t)pad_w;
                    float sum = bias_val;

                    for (size_t kh = 0; kh < k_h; ++kh) {
                        int64_t ih = ih_base + (int64_t)kh;
                        if (ih < 0 || ih >= (int64_t)in_h) continue;

                        for (size_t kw = 0; kw < k_w; ++kw) {
                            int64_t iw = iw_base + (int64_t)kw;
                            if (iw < 0 || iw >= (int64_t)in_w) continue;

                            float x_val = c_in[ih * (int64_t)in_w + iw];
                            float w_val = c_w[kh * k_w + kw];
                            sum += x_val * w_val;
                        }
                    }
                    c_out[oh * out_w + ow] = sum;
                }
            }
        }
    }

    return RCRE_OK;
}

/* ========================================================================= */
/* Pointwise 1x1 Convolution                                                 */
/* ========================================================================= */

RCRE_NODISCARD int rcre_nn_pointwise_conv2d(const RCRE_Tensor* input,
                                            const RCRE_Tensor* weights,
                                            const RCRE_Tensor* bias,
                                            RCRE_Tensor* output,
                                            RCRE_Arena* arena) {
    if (!input || !input->data || !weights || !weights->data || !output) {
        return RCRE_ERR_INVALID_ARG;
    }
    ensure_hal_init();

    size_t batch_size = 1;
    size_t in_c = 0, h = 0, w = 0;

    if (input->ndim == 3) {
        in_c = input->shape[0];
        h = input->shape[1];
        w = input->shape[2];
    } else if (input->ndim == 4) {
        batch_size = input->shape[0];
        in_c = input->shape[1];
        h = input->shape[2];
        w = input->shape[3];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t out_c = weights->shape[0];
    if (weights->ndim == 2) {
        if (weights->shape[1] != in_c) return RCRE_ERR_SHAPE_MISMATCH;
    } else if (weights->ndim == 4) {
        if (weights->shape[1] != in_c || weights->shape[2] != 1 || weights->shape[3] != 1) {
            return RCRE_ERR_SHAPE_MISMATCH;
        }
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t out_shape[4];
    uint8_t out_ndim = input->ndim;
    if (input->ndim == 3) {
        out_shape[0] = out_c;
        out_shape[1] = h;
        out_shape[2] = w;
    } else {
        out_shape[0] = batch_size;
        out_shape[1] = out_c;
        out_shape[2] = h;
        out_shape[3] = w;
    }

    if (!output->data) {
        if (!arena) return RCRE_ERR_INVALID_ARG;
        *output = rcre_tensor_create(out_shape, out_ndim, RCRE_DTYPE_F32, RCRE_LAYOUT_CHW, arena);
        if (!output->data) return RCRE_ERR_OUT_OF_MEMORY;
    } else {
        if (output->ndim != out_ndim) return RCRE_ERR_SHAPE_MISMATCH;
        for (uint8_t i = 0; i < out_ndim; ++i) {
            if (output->shape[i] != out_shape[i]) return RCRE_ERR_SHAPE_MISMATCH;
        }
    }

    const float* in_ptr = (const float*)input->data;
    const float* w_ptr = (const float*)weights->data;
    const float* b_ptr = (bias && bias->data) ? (const float*)bias->data : nullptr;
    float* out_ptr = (float*)output->data;

    size_t spatial_size = h * w;
    size_t in_batch_stride = in_c * spatial_size;
    size_t out_batch_stride = out_c * spatial_size;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* cur_in = in_ptr + b * in_batch_stride;
        float* cur_out = out_ptr + b * out_batch_stride;

        for (size_t oc = 0; oc < out_c; ++oc) {
            float bias_val = b_ptr ? b_ptr[oc] : 0.0f;
            float* oc_out = cur_out + oc * spatial_size;

            /* Initialize spatial slice with bias */
            for (size_t s = 0; s < spatial_size; ++s) {
                oc_out[s] = bias_val;
            }

            /* Accumulate channel projection: oc_out += w[oc, ic] * cur_in[ic, :] */
            const float* oc_weights = w_ptr + oc * in_c;
            for (size_t ic = 0; ic < in_c; ++ic) {
                float weight = oc_weights[ic];
                const float* ic_in = cur_in + ic * spatial_size;
                rcre_hal.axpy_f32(spatial_size, weight, ic_in, oc_out);
            }
        }
    }

    return RCRE_OK;
}

/* ========================================================================= */
/* Batch Normalization 2D & BN Folding                                      */
/* ========================================================================= */

RCRE_NODISCARD int rcre_nn_batchnorm2d(const RCRE_Tensor* input,
                                      const RCRE_Tensor* gamma,
                                      const RCRE_Tensor* beta,
                                      const RCRE_Tensor* mean,
                                      const RCRE_Tensor* var,
                                      float eps,
                                      RCRE_Tensor* output) {
    if (!input || !input->data || !gamma || !gamma->data || 
        !beta || !beta->data || !mean || !mean->data || !var || !var->data || !output || !output->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (eps < 0.0f) eps = 1e-5f;

    size_t batch_size = 1;
    size_t num_channels = 0, h = 0, w = 0;

    if (input->ndim == 3) {
        num_channels = input->shape[0];
        h = input->shape[1];
        w = input->shape[2];
    } else if (input->ndim == 4) {
        batch_size = input->shape[0];
        num_channels = input->shape[1];
        h = input->shape[2];
        w = input->shape[3];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (gamma->shape[0] != num_channels || beta->shape[0] != num_channels ||
        mean->shape[0] != num_channels || var->shape[0] != num_channels) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* in_ptr = (const float*)input->data;
    const float* g_ptr = (const float*)gamma->data;
    const float* b_ptr = (const float*)beta->data;
    const float* m_ptr = (const float*)mean->data;
    const float* v_ptr = (const float*)var->data;
    float* out_ptr = (float*)output->data;

    size_t spatial_size = h * w;
    size_t batch_stride = num_channels * spatial_size;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* b_in = in_ptr + b * batch_stride;
        float* b_out = out_ptr + b * batch_stride;

        for (size_t c = 0; c < num_channels; ++c) {
            float g = g_ptr[c];
            float beta_val = b_ptr[c];
            float m = m_ptr[c];
            float v = v_ptr[c];

            float inv_std = 1.0f / sqrtf(v + eps);
            float scale = g * inv_std;
            float shift = beta_val - m * scale;

            const float* c_in = b_in + c * spatial_size;
            float* c_out = b_out + c * spatial_size;

            for (size_t s = 0; s < spatial_size; ++s) {
                c_out[s] = c_in[s] * scale + shift;
            }
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_nn_fuse_conv_batchnorm(RCRE_Tensor* conv_weight,
                                               RCRE_Tensor* conv_bias,
                                               const RCRE_Tensor* gamma,
                                               const RCRE_Tensor* beta,
                                               const RCRE_Tensor* mean,
                                               const RCRE_Tensor* var,
                                               float eps) {
    if (!conv_weight || !conv_weight->data || !gamma || !gamma->data || 
        !beta || !beta->data || !mean || !mean->data || !var || !var->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    if (eps < 0.0f) eps = 1e-5f;

    size_t out_c = conv_weight->shape[0];
    if (gamma->shape[0] != out_c || beta->shape[0] != out_c ||
        mean->shape[0] != out_c || var->shape[0] != out_c) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t elements_per_filter = 1;
    for (uint8_t i = 1; i < conv_weight->ndim; ++i) {
        elements_per_filter *= conv_weight->shape[i];
    }

    float* w_ptr = (float*)conv_weight->data;
    float* b_ptr = (conv_bias && conv_bias->data) ? (float*)conv_bias->data : nullptr;
    const float* g_ptr = (const float*)gamma->data;
    const float* beta_ptr = (const float*)beta->data;
    const float* m_ptr = (const float*)mean->data;
    const float* v_ptr = (const float*)var->data;

    for (size_t oc = 0; oc < out_c; ++oc) {
        float inv_std = 1.0f / sqrtf(v_ptr[oc] + eps);
        float scale = g_ptr[oc] * inv_std;
        float shift = beta_ptr[oc] - m_ptr[oc] * scale;

        /* Scale weights */
        float* filter_w = w_ptr + oc * elements_per_filter;
        for (size_t i = 0; i < elements_per_filter; ++i) {
            filter_w[i] *= scale;
        }

        /* Adjust or initialize bias */
        if (b_ptr) {
            b_ptr[oc] = b_ptr[oc] * scale + shift;
        }
    }

    return RCRE_OK;
}

/* ========================================================================= */
/* Activation & Pooling Layers                                               */
/* ========================================================================= */

RCRE_NODISCARD int rcre_nn_relu6(const RCRE_Tensor* input, RCRE_Tensor* output) {
    if (!input || !input->data || !output || !output->data) {
        return RCRE_ERR_INVALID_ARG;
    }
    size_t n = rcre_tensor_numel(input);
    if (rcre_tensor_numel(output) != n) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* in = (const float*)input->data;
    float* out = (float*)output->data;

    for (size_t i = 0; i < n; ++i) {
        float v = in[i];
        if (v < 0.0f) v = 0.0f;
        else if (v > 6.0f) v = 6.0f;
        out[i] = v;
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_nn_global_avgpool(const RCRE_Tensor* input, RCRE_Tensor* output) {
    if (!input || !input->data || !output || !output->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t batch_size = 1;
    size_t channels = 0, h = 0, w = 0;

    if (input->ndim == 3) {
        channels = input->shape[0];
        h = input->shape[1];
        w = input->shape[2];
    } else if (input->ndim == 4) {
        batch_size = input->shape[0];
        channels = input->shape[1];
        h = input->shape[2];
        w = input->shape[3];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t spatial_size = h * w;
    if (spatial_size == 0) return RCRE_ERR_INVALID_ARG;
    float inv_spatial = 1.0f / (float)spatial_size;

    const float* in_ptr = (const float*)input->data;
    float* out_ptr = (float*)output->data;

    size_t in_batch_stride = channels * spatial_size;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* b_in = in_ptr + b * in_batch_stride;
        float* b_out = out_ptr + b * channels;

        for (size_t c = 0; c < channels; ++c) {
            const float* c_in = b_in + c * spatial_size;
            float sum = 0.0f;
            for (size_t s = 0; s < spatial_size; ++s) {
                sum += c_in[s];
            }
            b_out[c] = sum * inv_spatial;
        }
    }

    return RCRE_OK;
}

/* ========================================================================= */
/* Fully Connected Linear Layer & Softmax                                    */
/* ========================================================================= */

RCRE_NODISCARD int rcre_nn_linear(const RCRE_Tensor* input,
                                  const RCRE_Tensor* weights,
                                  const RCRE_Tensor* bias,
                                  RCRE_Tensor* output,
                                  RCRE_Arena* arena) {
    if (!input || !input->data || !weights || !weights->data || !output) {
        return RCRE_ERR_INVALID_ARG;
    }
    ensure_hal_init();

    size_t batch_size = 1;
    size_t in_features = 0;

    if (input->ndim == 1) {
        in_features = input->shape[0];
    } else if (input->ndim == 2) {
        batch_size = input->shape[0];
        in_features = input->shape[1];
    } else if (input->ndim == 4 && input->shape[2] == 1 && input->shape[3] == 1) {
        batch_size = input->shape[0];
        in_features = input->shape[1];
    } else if (input->ndim == 3 && input->shape[1] == 1 && input->shape[2] == 1) {
        in_features = input->shape[0];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (weights->ndim != 2 || weights->shape[1] != in_features) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }
    size_t out_features = weights->shape[0];

    if (bias && bias->data && bias->shape[0] != out_features) {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    size_t out_shape[2];
    uint8_t out_ndim = (input->ndim == 1 || (input->ndim == 3 && input->shape[1] == 1 && input->shape[2] == 1)) ? 1 : 2;
    if (out_ndim == 1) {
        out_shape[0] = out_features;
    } else {
        out_shape[0] = batch_size;
        out_shape[1] = out_features;
    }

    if (!output->data) {
        if (!arena) return RCRE_ERR_INVALID_ARG;
        *output = rcre_tensor_create(out_shape, out_ndim, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
        if (!output->data) return RCRE_ERR_OUT_OF_MEMORY;
    }

    const float* in_ptr = (const float*)input->data;
    const float* w_ptr = (const float*)weights->data;
    const float* b_ptr = (bias && bias->data) ? (const float*)bias->data : nullptr;
    float* out_ptr = (float*)output->data;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* cur_in = in_ptr + b * in_features;
        float* cur_out = out_ptr + b * out_features;

        for (size_t j = 0; j < out_features; ++j) {
            float b_val = b_ptr ? b_ptr[j] : 0.0f;
            const float* w_row = w_ptr + j * in_features;
            float dot = rcre_hal.dot_f32(cur_in, w_row, in_features);
            cur_out[j] = b_val + dot;
        }
    }

    return RCRE_OK;
}

RCRE_NODISCARD int rcre_nn_softmax(const RCRE_Tensor* input, RCRE_Tensor* output) {
    if (!input || !input->data || !output || !output->data) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t batch_size = 1;
    size_t num_classes = 0;

    if (input->ndim == 1) {
        num_classes = input->shape[0];
    } else if (input->ndim == 2) {
        batch_size = input->shape[0];
        num_classes = input->shape[1];
    } else {
        return RCRE_ERR_SHAPE_MISMATCH;
    }

    if (num_classes == 0) return RCRE_ERR_INVALID_ARG;

    const float* in_ptr = (const float*)input->data;
    float* out_ptr = (float*)output->data;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* cur_in = in_ptr + b * num_classes;
        float* cur_out = out_ptr + b * num_classes;

        /* Find maximum for numerical stability */
        float max_val = cur_in[0];
        for (size_t i = 1; i < num_classes; ++i) {
            if (cur_in[i] > max_val) {
                max_val = cur_in[i];
            }
        }

        /* Exponentiate and sum */
        float sum_exp = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            float e = expf(cur_in[i] - max_val);
            cur_out[i] = e;
            sum_exp += e;
        }

        /* Normalize */
        float inv_sum = 1.0f / sum_exp;
        for (size_t i = 0; i < num_classes; ++i) {
            cur_out[i] *= inv_sum;
        }
    }

    return RCRE_OK;
}
