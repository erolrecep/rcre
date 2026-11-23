/**
 * @file onnx_subset.c
 * @brief Minimal Zero-Dependency ONNX Protobuf Parser & Graph Inference Engine (ISO C23)
 */

#include "rcre_codec.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t* p;
    size_t         len;
    size_t         pos;
} RCRE_PBCursor;

static bool onnx_read_varint(RCRE_PBCursor* c, uint64_t* out) {
    uint64_t v = 0;
    int shift = 0;
    while (c->pos < c->len && shift < 64) {
        uint8_t b = c->p[c->pos++];
        v |= ((uint64_t)(b & 0x7FU) << shift);
        if ((b & 0x80U) == 0) {
            *out = v;
            return true;
        }
        shift += 7;
    }
    return false;
}

static bool onnx_skip_field(RCRE_PBCursor* c, uint32_t wire_type) {
    uint64_t v = 0;
    switch (wire_type) {
        case 0: return onnx_read_varint(c, &v);
        case 1: if (c->len - c->pos < 8) return false; c->pos += 8; return true;
        case 2:
            if (!onnx_read_varint(c, &v) || v > (uint64_t)(c->len - c->pos)) return false;
            c->pos += (size_t)v;
            return true;
        case 5: if (c->len - c->pos < 4) return false; c->pos += 4; return true;
        default: return false;
    }
}

static RCRE_ONNXOpType onnx_parse_op(const char* s, size_t len) {
    if (len == 4 && memcmp(s, "Relu", 4) == 0) return RCRE_ONNX_OP_RELU;
    if (len == 4 && memcmp(s, "Gemm", 4) == 0) return RCRE_ONNX_OP_GEMM;
    if (len == 6 && memcmp(s, "MatMul", 6) == 0) return RCRE_ONNX_OP_MATMUL;
    if (len == 3 && memcmp(s, "Add", 3) == 0) return RCRE_ONNX_OP_ADD;
    if (len == 7 && memcmp(s, "Softmax", 7) == 0) return RCRE_ONNX_OP_SOFTMAX;
    if (len == 4 && memcmp(s, "Conv", 4) == 0) return RCRE_ONNX_OP_CONV;
    if (len == 7 && memcmp(s, "MaxPool", 7) == 0) return RCRE_ONNX_OP_MAXPOOL;
    if (len == 17 && memcmp(s, "GlobalAveragePool", 17) == 0) return RCRE_ONNX_OP_GLOBAL_AVG_POOL;
    return RCRE_ONNX_OP_UNKNOWN;
}

static void onnx_copy_str(char* dst, size_t cap, const uint8_t* s, size_t len) {
    size_t n = (len < cap - 1U) ? len : (cap - 1U);
    if (n > 0) memcpy(dst, s, n);
    dst[n] = '\0';
}

static bool onnx_parse_tensor_proto(const uint8_t* buf, size_t len, RCRE_ONNXTensor* out) {
    memset(out, 0, sizeof(*out));
    RCRE_PBCursor c = { buf, len, 0 };
    const uint8_t* raw_data = nullptr;
    size_t raw_len = 0;
    size_t float_cap = 0, float_n = 0;
    float* float_arr = nullptr;

    while (c.pos < c.len) {
        uint64_t tag = 0;
        if (!onnx_read_varint(&c, &tag)) { free(float_arr); return false; }
        uint32_t field = (uint32_t)(tag >> 3);
        uint32_t wire  = (uint32_t)(tag & 7U);

        if (field == 1 && wire == 0) { /* dims repeated int64 */
            uint64_t d = 0;
            if (!onnx_read_varint(&c, &d)) { free(float_arr); return false; }
            if (out->ndim < 4) out->dims[out->ndim++] = (size_t)d;
        } else if (field == 1 && wire == 2) { /* packed dims */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) { free(float_arr); return false; }
            RCRE_PBCursor sub = { c.p + c.pos, (size_t)blen, 0 };
            c.pos += (size_t)blen;
            while (sub.pos < sub.len) {
                uint64_t d = 0;
                if (!onnx_read_varint(&sub, &d)) break;
                if (out->ndim < 4) out->dims[out->ndim++] = (size_t)d;
            }
        } else if (field == 4 && wire == 5) { /* float_data single */
            if (c.len - c.pos < 4) { free(float_arr); return false; }
            float v = 0.0f;
            memcpy(&v, c.p + c.pos, 4);
            c.pos += 4;
            if (float_n >= float_cap) {
                float_cap = float_cap ? float_cap * 2U : 16U;
                float* nb = (float*)realloc(float_arr, float_cap * sizeof(float));
                if (!nb) { free(float_arr); return false; }
                float_arr = nb;
            }
            float_arr[float_n++] = v;
        } else if (field == 4 && wire == 2) { /* packed float_data */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) { free(float_arr); return false; }
            size_t cnt = (size_t)blen / 4U;
            float* nb = (float*)realloc(float_arr, (float_n + cnt) * sizeof(float));
            if (!nb) { free(float_arr); return false; }
            float_arr = nb;
            memcpy(float_arr + float_n, c.p + c.pos, cnt * 4U);
            float_n += cnt;
            c.pos += (size_t)blen;
        } else if (field == 8 && wire == 2) { /* name */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) { free(float_arr); return false; }
            free(out->name);
            out->name = (char*)malloc((size_t)blen + 1U);
            if (!out->name) { free(float_arr); return false; }
            onnx_copy_str(out->name, (size_t)blen + 1U, c.p + c.pos, (size_t)blen);
            c.pos += (size_t)blen;
        } else if (field == 9 && wire == 2) { /* raw_data float bytes */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) { free(float_arr); return false; }
            raw_data = c.p + c.pos;
            raw_len  = (size_t)blen;
            c.pos   += (size_t)blen;
        } else {
            if (!onnx_skip_field(&c, wire)) { free(float_arr); return false; }
        }
    }

    size_t numel = (out->ndim > 0) ? 1 : 0;
    for (uint8_t d = 0; d < out->ndim; ++d) numel *= out->dims[d];
    out->numel = numel;

    if (raw_data && raw_len >= numel * 4U && numel > 0) {
        free(float_arr);
        out->data = (float*)malloc(numel * sizeof(float));
        if (!out->data) return false;
        memcpy(out->data, raw_data, numel * 4U);
    } else if (float_arr && float_n > 0) {
        out->data = float_arr;
        if (out->numel == 0) {
            out->numel = float_n;
            out->ndim = 1;
            out->dims[0] = float_n;
        }
    } else {
        free(float_arr);
    }
    return true;
}

static bool onnx_parse_node_proto(const uint8_t* buf, size_t len, RCRE_ONNXNode* out) {
    memset(out, 0, sizeof(*out));
    out->stride = 1;
    out->kernel = 1;
    out->trans_b = 1; /* common ONNX Gemm default when set */
    RCRE_PBCursor c = { buf, len, 0 };

    while (c.pos < c.len) {
        uint64_t tag = 0;
        if (!onnx_read_varint(&c, &tag)) return false;
        uint32_t field = (uint32_t)(tag >> 3);
        uint32_t wire  = (uint32_t)(tag & 7U);

        if (field == 1 && wire == 2) { /* input repeated string */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) return false;
            if (out->n_inputs < 4) onnx_copy_str(out->inputs[out->n_inputs++], 64, c.p + c.pos, (size_t)blen);
            c.pos += (size_t)blen;
        } else if (field == 2 && wire == 2) { /* output repeated string */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) return false;
            if (out->output[0] == '\0') onnx_copy_str(out->output, 64, c.p + c.pos, (size_t)blen);
            c.pos += (size_t)blen;
        } else if (field == 4 && wire == 2) { /* op_type */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) return false;
            out->op = onnx_parse_op((const char*)(c.p + c.pos), (size_t)blen);
            c.pos += (size_t)blen;
        } else if (field == 5 && wire == 2) { /* AttributeProto */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) return false;
            RCRE_PBCursor ac = { c.p + c.pos, (size_t)blen, 0 };
            c.pos += (size_t)blen;
            char aname[32] = {0};
            int64_t ival = 0;
            int64_t first_int = 0;
            while (ac.pos < ac.len) {
                uint64_t atag = 0;
                if (!onnx_read_varint(&ac, &atag)) break;
                uint32_t af = (uint32_t)(atag >> 3), aw = (uint32_t)(atag & 7U);
                if (af == 1 && aw == 2) {
                    uint64_t sl = 0;
                    if (onnx_read_varint(&ac, &sl) && sl <= ac.len - ac.pos) {
                        onnx_copy_str(aname, sizeof(aname), ac.p + ac.pos, (size_t)sl);
                        ac.pos += (size_t)sl;
                    }
                } else if (af == 3 && aw == 0) {
                    uint64_t u = 0; (void)onnx_read_varint(&ac, &u); ival = (int64_t)u;
                } else if (af == 7 && aw == 0) {
                    uint64_t u = 0; (void)onnx_read_varint(&ac, &u); if (first_int == 0) first_int = (int64_t)u;
                } else {
                    if (!onnx_skip_field(&ac, aw)) break;
                }
            }
            if (strcmp(aname, "transB") == 0) out->trans_b = (int)ival;
            else if (strcmp(aname, "strides") == 0 && first_int > 0) out->stride = (int)first_int;
            else if (strcmp(aname, "pads") == 0) out->pad = (int)first_int;
            else if (strcmp(aname, "kernel_shape") == 0 && first_int > 0) out->kernel = (int)first_int;
        } else {
            if (!onnx_skip_field(&c, wire)) return false;
        }
    }
    return true;
}

static bool onnx_parse_graph_proto(const uint8_t* buf, size_t len, RCRE_ONNXModel* m) {
    RCRE_PBCursor c = { buf, len, 0 };
    while (c.pos < c.len) {
        uint64_t tag = 0;
        if (!onnx_read_varint(&c, &tag)) return false;
        uint32_t field = (uint32_t)(tag >> 3), wire = (uint32_t)(tag & 7U);
        if (field == 1 && wire == 2) { /* node */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) return false;
            if (m->n_nodes < RCRE_ONNX_MAX_NODES) {
                if (!onnx_parse_node_proto(c.p + c.pos, (size_t)blen, &m->nodes[m->n_nodes++])) return false;
            }
            c.pos += (size_t)blen;
        } else if (field == 5 && wire == 2) { /* initializer TensorProto */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) return false;
            if (m->n_init < RCRE_ONNX_MAX_TENSORS) {
                if (!onnx_parse_tensor_proto(c.p + c.pos, (size_t)blen, &m->initializers[m->n_init++])) return false;
            }
            c.pos += (size_t)blen;
        } else if (field == 11 && wire == 2) { /* input ValueInfoProto */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) return false;
            RCRE_PBCursor vc = { c.p + c.pos, (size_t)blen, 0 };
            uint64_t vtag = 0;
            if (onnx_read_varint(&vc, &vtag) && (vtag >> 3) == 1 && (vtag & 7U) == 2) {
                uint64_t sl = 0;
                if (onnx_read_varint(&vc, &sl) && sl <= vc.len - vc.pos && m->input_name[0] == '\0') {
                    onnx_copy_str(m->input_name, sizeof(m->input_name), vc.p + vc.pos, (size_t)sl);
                }
            }
            c.pos += (size_t)blen;
        } else if (field == 12 && wire == 2) { /* output ValueInfoProto */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos) return false;
            RCRE_PBCursor vc = { c.p + c.pos, (size_t)blen, 0 };
            uint64_t vtag = 0;
            if (onnx_read_varint(&vc, &vtag) && (vtag >> 3) == 1 && (vtag & 7U) == 2) {
                uint64_t sl = 0;
                if (onnx_read_varint(&vc, &sl) && sl <= vc.len - vc.pos && m->output_name[0] == '\0') {
                    onnx_copy_str(m->output_name, sizeof(m->output_name), vc.p + vc.pos, (size_t)sl);
                }
            }
            c.pos += (size_t)blen;
        } else {
            if (!onnx_skip_field(&c, wire)) return false;
        }
    }
    return true;
}

RCRE_NODISCARD int rcre_onnx_load_buffer(const void* buf, size_t size, RCRE_ONNXModel* out_model) {
    if (!buf || size == 0 || !out_model) return RCRE_ERR_INVALID_ARG;
    memset(out_model, 0, sizeof(*out_model));
    RCRE_PBCursor c = { (const uint8_t*)buf, size, 0 };
    bool parsed_graph = false;

    while (c.pos < c.len) {
        uint64_t tag = 0;
        if (!onnx_read_varint(&c, &tag)) { rcre_onnx_free(out_model); return RCRE_ERR_IO; }
        uint32_t field = (uint32_t)(tag >> 3), wire = (uint32_t)(tag & 7U);
        if (field == 7 && wire == 2) { /* graph */
            uint64_t blen = 0;
            if (!onnx_read_varint(&c, &blen) || blen > c.len - c.pos ||
                !onnx_parse_graph_proto(c.p + c.pos, (size_t)blen, out_model)) {
                rcre_onnx_free(out_model);
                return RCRE_ERR_IO;
            }
            parsed_graph = true;
            c.pos += (size_t)blen;
        } else {
            if (!onnx_skip_field(&c, wire)) { rcre_onnx_free(out_model); return RCRE_ERR_IO; }
        }
    }
    if (!parsed_graph || out_model->n_nodes == 0) {
        rcre_onnx_free(out_model);
        return RCRE_ERR_IO;
    }
    if (out_model->output_name[0] == '\0') {
        snprintf(out_model->output_name, 64, "%s", out_model->nodes[out_model->n_nodes - 1U].output);
    }
    return RCRE_OK;
}

void rcre_onnx_free(RCRE_ONNXModel* model) {
    if (!model) return;
    for (size_t i = 0; i < model->n_init; ++i) {
        free(model->initializers[i].name);
        free(model->initializers[i].data);
    }
    memset(model, 0, sizeof(*model));
}

typedef struct {
    const char* name;
    RCRE_Tensor tensor;
} RCRE_ONNXSlot;

static const RCRE_Tensor* onnx_find_slot(const RCRE_ONNXSlot* slots, size_t n, const char* name) {
    for (size_t i = 0; i < n; ++i) {
        if (slots[i].name && strcmp(slots[i].name, name) == 0) return &slots[i].tensor;
    }
    return nullptr;
}

RCRE_NODISCARD int rcre_onnx_run(const RCRE_ONNXModel* model, const RCRE_Tensor* input,
                                 RCRE_Arena* arena, RCRE_Tensor* output) {
    if (!model || !input || !input->data || !arena || !output || input->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_ONNXSlot slots[RCRE_ONNX_MAX_TENSORS + RCRE_ONNX_MAX_NODES + 2];
    size_t n_slots = 0;

    const char* in_name = model->input_name[0] ? model->input_name : model->nodes[0].inputs[0];
    slots[n_slots++] = (RCRE_ONNXSlot){ .name = in_name, .tensor = *input };

    for (size_t i = 0; i < model->n_init; ++i) {
        const RCRE_ONNXTensor* it = &model->initializers[i];
        if (!it->name || !it->data || it->ndim == 0) continue;
        slots[n_slots++] = (RCRE_ONNXSlot){
            .name   = it->name,
            .tensor = rcre_tensor_from_buffer(it->data, it->dims, it->ndim, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR)
        };
    }

    for (size_t n = 0; n < model->n_nodes; ++n) {
        const RCRE_ONNXNode* nd = &model->nodes[n];
        const RCRE_Tensor* A = onnx_find_slot(slots, n_slots, nd->inputs[0]);
        if (!A) return RCRE_ERR_IO;
        RCRE_Tensor res = {0};

        if (nd->op == RCRE_ONNX_OP_RELU) {
            res = rcre_tensor_create(A->shape, A->ndim, RCRE_DTYPE_F32, A->layout, arena);
            if (!res.data) return RCRE_ERR_OUT_OF_MEMORY;
            const float* src = (const float*)A->data;
            float* dst = (float*)res.data;
            size_t num = rcre_tensor_numel(A);
            for (size_t i = 0; i < num; ++i) dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
        } else if (nd->op == RCRE_ONNX_OP_SOFTMAX) {
            res = rcre_tensor_create(A->shape, A->ndim, RCRE_DTYPE_F32, A->layout, arena);
            if (!res.data) return RCRE_ERR_OUT_OF_MEMORY;
            size_t cols = A->shape[A->ndim - 1U];
            size_t rows = rcre_tensor_numel(A) / cols;
            const float* src = (const float*)A->data;
            float* dst = (float*)res.data;
            for (size_t r = 0; r < rows; ++r) {
                float mx = src[r * cols];
                for (size_t c = 1; c < cols; ++c) if (src[r * cols + c] > mx) mx = src[r * cols + c];
                float s = 0.0f;
                for (size_t c = 0; c < cols; ++c) {
                    dst[r * cols + c] = expf(src[r * cols + c] - mx);
                    s += dst[r * cols + c];
                }
                for (size_t c = 0; c < cols; ++c) dst[r * cols + c] /= s;
            }
        } else if (nd->op == RCRE_ONNX_OP_ADD) {
            const RCRE_Tensor* B = (nd->n_inputs >= 2) ? onnx_find_slot(slots, n_slots, nd->inputs[1]) : nullptr;
            if (!B) return RCRE_ERR_IO;
            if (rcre_tensor_add_broadcast(A, B, arena, &res) != RCRE_OK) return RCRE_ERR_SHAPE_MISMATCH;
        } else if (nd->op == RCRE_ONNX_OP_MATMUL || nd->op == RCRE_ONNX_OP_GEMM) {
            const RCRE_Tensor* B = (nd->n_inputs >= 2) ? onnx_find_slot(slots, n_slots, nd->inputs[1]) : nullptr;
            const RCRE_Tensor* C = (nd->op == RCRE_ONNX_OP_GEMM && nd->n_inputs >= 3)
                                       ? onnx_find_slot(slots, n_slots, nd->inputs[2])
                                       : nullptr;
            if (!B || A->ndim != 2 || B->ndim != 2) return RCRE_ERR_SHAPE_MISMATCH;
            size_t M = A->shape[0], K = A->shape[1];
            size_t N = (nd->op == RCRE_ONNX_OP_GEMM && nd->trans_b) ? B->shape[0] : B->shape[1];
            size_t B_k = (nd->op == RCRE_ONNX_OP_GEMM && nd->trans_b) ? B->shape[1] : B->shape[0];
            if (K != B_k) return RCRE_ERR_SHAPE_MISMATCH;
            size_t sh[2] = { M, N };
            res = rcre_tensor_create(sh, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
            if (!res.data) return RCRE_ERR_OUT_OF_MEMORY;
            const float* ad = (const float*)A->data;
            const float* bd = (const float*)B->data;
            float* od = (float*)res.data;
            if (nd->op == RCRE_ONNX_OP_GEMM && nd->trans_b) {
                for (size_t i = 0; i < M; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        od[i * N + j] = rcre_hal.dot_f32(ad + i * K, bd + j * K, K);
                    }
                }
            } else {
                rcre_hal.gemm_f32(M, N, K, 1.0f, ad, K, bd, N, 0.0f, od, N);
            }
            if (C && C->data) {
                const float* cd = (const float*)C->data;
                size_t c_num = rcre_tensor_numel(C);
                for (size_t i = 0; i < M; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        od[i * N + j] += cd[(c_num == N) ? j : ((i * N + j) % c_num)];
                    }
                }
            }
        } else if (nd->op == RCRE_ONNX_OP_CONV) {
            const RCRE_Tensor* W = (nd->n_inputs >= 2) ? onnx_find_slot(slots, n_slots, nd->inputs[1]) : nullptr;
            const RCRE_Tensor* Bias = (nd->n_inputs >= 3) ? onnx_find_slot(slots, n_slots, nd->inputs[2]) : nullptr;
            if (!W || A->ndim != 3 || W->ndim != 4) return RCRE_ERR_SHAPE_MISMATCH;
            RCRE_Conv2DParams p = { (size_t)nd->stride, (size_t)nd->stride, (size_t)nd->pad, (size_t)nd->pad };
            if (rcre_nn_conv2d(A, W, Bias, &p, &res, arena) != RCRE_OK) return RCRE_ERR_IO;
        } else if (nd->op == RCRE_ONNX_OP_GLOBAL_AVG_POOL) {
            if (A->ndim != 3 && A->ndim != 4) return RCRE_ERR_SHAPE_MISMATCH;
            size_t C_ch = (A->ndim == 4) ? A->shape[1] : A->shape[0];
            size_t H = (A->ndim == 4) ? A->shape[2] : A->shape[1];
            size_t W = (A->ndim == 4) ? A->shape[3] : A->shape[2];
            size_t sh[2] = { 1, C_ch };
            res = rcre_tensor_create(sh, 2, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, arena);
            if (!res.data) return RCRE_ERR_OUT_OF_MEMORY;
            const float* src = (const float*)A->data;
            float* dst = (float*)res.data;
            for (size_t c = 0; c < C_ch; ++c) {
                float s = 0.0f;
                for (size_t i = 0; i < H * W; ++i) s += src[c * (H * W) + i];
                dst[c] = s / (float)(H * W);
            }
        } else {
            return RCRE_ERR_UNSUPPORTED_OP;
        }

        slots[n_slots++] = (RCRE_ONNXSlot){ .name = nd->output, .tensor = res };
    }

    const RCRE_Tensor* out_slot = onnx_find_slot(slots, n_slots, model->output_name);
    if (!out_slot) return RCRE_ERR_IO;
    *output = *out_slot;
    return RCRE_OK;
}

typedef struct {
    uint8_t* buf;
    size_t   pos;
    size_t   cap;
} RCRE_PBEnc;

static void onnx_enc_varint(RCRE_PBEnc* e, uint64_t v) {
    do {
        uint8_t b = (uint8_t)(v & 0x7FU);
        v >>= 7U;
        if (v != 0) b |= 0x80U;
        if (e->pos < e->cap) e->buf[e->pos] = b;
        e->pos++;
    } while (v != 0);
}

static void onnx_enc_bytes(RCRE_PBEnc* e, uint32_t field, const void* data, size_t len) {
    onnx_enc_varint(e, ((uint64_t)field << 3U) | 2U);
    onnx_enc_varint(e, (uint64_t)len);
    if (e->pos + len <= e->cap && len > 0) memcpy(e->buf + e->pos, data, len);
    e->pos += len;
}

static void onnx_enc_tensor(uint8_t* tmp, size_t* out_len, const char* name,
                            const float* w, size_t d0, size_t d1, uint8_t ndim) {
    RCRE_PBEnc e = { tmp, 0, 8192 };
    onnx_enc_varint(&e, (1ULL << 3) | 0ULL); onnx_enc_varint(&e, (uint64_t)d0);
    if (ndim == 2) {
        onnx_enc_varint(&e, (1ULL << 3) | 0ULL); onnx_enc_varint(&e, (uint64_t)d1);
    }
    onnx_enc_bytes(&e, 8, name, strlen(name));
    size_t n = (ndim == 2) ? (d0 * d1) : d0;
    onnx_enc_bytes(&e, 9, w, n * sizeof(float));
    *out_len = e.pos;
}

RCRE_NODISCARD size_t rcre_onnx_build_mlp_protobuf(
    uint8_t* dst,
    size_t max_bytes,
    const float* w1, const float* b1, size_t in_dim, size_t hid_dim,
    const float* w2, const float* b2, size_t out_dim
) {
    uint8_t gbuf[16384];
    RCRE_PBEnc g = { gbuf, 0, sizeof(gbuf) };

    uint8_t sub[8192];
    size_t slen = 0;

    /* Node 1: Gemm(x, w1, b1) -> h0 (transB=0) */
    {
        RCRE_PBEnc n = { sub, 0, sizeof(sub) };
        onnx_enc_bytes(&n, 1, "x", 1);
        onnx_enc_bytes(&n, 1, "w1", 2);
        onnx_enc_bytes(&n, 1, "b1", 2);
        onnx_enc_bytes(&n, 2, "h0", 2);
        onnx_enc_bytes(&n, 4, "Gemm", 4);
        uint8_t attr[32];
        RCRE_PBEnc a = { attr, 0, sizeof(attr) };
        onnx_enc_bytes(&a, 1, "transB", 6);
        onnx_enc_varint(&a, (3ULL << 3) | 0ULL); onnx_enc_varint(&a, 0ULL);
        onnx_enc_bytes(&n, 5, attr, a.pos);
        onnx_enc_bytes(&g, 1, sub, n.pos);
    }
    /* Node 2: Relu(h0) -> h1 */
    {
        RCRE_PBEnc n = { sub, 0, sizeof(sub) };
        onnx_enc_bytes(&n, 1, "h0", 2);
        onnx_enc_bytes(&n, 2, "h1", 2);
        onnx_enc_bytes(&n, 4, "Relu", 4);
        onnx_enc_bytes(&g, 1, sub, n.pos);
    }
    /* Node 3: Gemm(h1, w2, b2) -> y (transB=0) */
    {
        RCRE_PBEnc n = { sub, 0, sizeof(sub) };
        onnx_enc_bytes(&n, 1, "h1", 2);
        onnx_enc_bytes(&n, 1, "w2", 2);
        onnx_enc_bytes(&n, 1, "b2", 2);
        onnx_enc_bytes(&n, 2, "y", 1);
        onnx_enc_bytes(&n, 4, "Gemm", 4);
        uint8_t attr[32];
        RCRE_PBEnc a = { attr, 0, sizeof(attr) };
        onnx_enc_bytes(&a, 1, "transB", 6);
        onnx_enc_varint(&a, (3ULL << 3) | 0ULL); onnx_enc_varint(&a, 0ULL);
        onnx_enc_bytes(&n, 5, attr, a.pos);
        onnx_enc_bytes(&g, 1, sub, n.pos);
    }

    onnx_enc_tensor(sub, &slen, "w1", w1, in_dim, hid_dim, 2); onnx_enc_bytes(&g, 5, sub, slen);
    onnx_enc_tensor(sub, &slen, "b1", b1, hid_dim, 1, 1);     onnx_enc_bytes(&g, 5, sub, slen);
    onnx_enc_tensor(sub, &slen, "w2", w2, hid_dim, out_dim, 2); onnx_enc_bytes(&g, 5, sub, slen);
    onnx_enc_tensor(sub, &slen, "b2", b2, out_dim, 1, 1);     onnx_enc_bytes(&g, 5, sub, slen);

    /* input "x", output "y" */
    {
        RCRE_PBEnc v = { sub, 0, sizeof(sub) };
        onnx_enc_bytes(&v, 1, "x", 1);
        onnx_enc_bytes(&g, 11, sub, v.pos);
    }
    {
        RCRE_PBEnc v = { sub, 0, sizeof(sub) };
        onnx_enc_bytes(&v, 1, "y", 1);
        onnx_enc_bytes(&g, 12, sub, v.pos);
    }

    RCRE_PBEnc m = { dst, 0, max_bytes };
    onnx_enc_bytes(&m, 7, gbuf, g.pos);
    return (m.pos <= max_bytes) ? m.pos : 0;
}
