/**
 * @file tensor.c
 * @brief Strided N-Dimensional Tensor Engine & ShapeTracker
 * 
 * Satisfies TNS-REQ-01 through TNS-REQ-05.
 */

#include "rcre.h"
#include <string.h>

RCRE_NODISCARD size_t rcre_dtype_size(RCRE_DType dtype) {
    switch (dtype) {
        case RCRE_DTYPE_U8:   return 1;
        case RCRE_DTYPE_I8:   return 1;
        case RCRE_DTYPE_F16:  return 2;
        case RCRE_DTYPE_BF16: return 2;
        case RCRE_DTYPE_I32:    return 4;
        case RCRE_DTYPE_F32:    return 4;
        case RCRE_DTYPE_STR_ID: return 4;
        case RCRE_DTYPE_I64:    return 8;
        case RCRE_DTYPE_F64:  return 8;
        default:              return 0;
    }
}

RCRE_NODISCARD size_t rcre_tensor_numel(const RCRE_Tensor* t) {
    if (!t || t->ndim == 0) return 0;
    size_t count = 1;
    for (uint8_t i = 0; i < t->ndim; ++i) {
        count *= t->shape[i];
    }
    return count;
}

RCRE_NODISCARD size_t rcre_tensor_nbytes(const RCRE_Tensor* t) {
    return rcre_tensor_numel(t) * rcre_dtype_size(t->dtype);
}

RCRE_NODISCARD bool rcre_tensor_is_contiguous(const RCRE_Tensor* t) {
    if (!t || t->ndim == 0) return false;
    
    ptrdiff_t expected_stride = 1;
    for (int i = (int)t->ndim - 1; i >= 0; --i) {
        if (t->shape[i] == 0) return true;
        if (t->shape[i] > 1) {
            if (t->strides[i] != expected_stride) {
                return false;
            }
            expected_stride *= (ptrdiff_t)t->shape[i];
        }
    }
    return true;
}

static void compute_contiguous_strides(const size_t* shape, uint8_t ndim, ptrdiff_t* strides) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * (ptrdiff_t)shape[i + 1];
    }
}

RCRE_NODISCARD RCRE_Tensor rcre_tensor_create(const size_t* shape, uint8_t ndim, 
                                             RCRE_DType dtype, RCRE_Layout layout, 
                                             RCRE_Arena* arena) {
    RCRE_Tensor t = {0};
    if (ndim == 0 || ndim > RCRE_MAX_DIMS || !arena) {
        return t;
    }

    t.ndim = ndim;
    t.dtype = dtype;
    t.layout = layout;
    memcpy(t.shape, shape, ndim * sizeof(size_t));
    compute_contiguous_strides(shape, ndim, t.strides);
    t.numel = rcre_tensor_numel(&t);
    t.is_view = false; /* owns its arena allocation */

    size_t total_bytes = rcre_tensor_nbytes(&t);
    t.data = rcre_arena_alloc(arena, total_bytes, 64);
    if (!t.data) {
        memset(&t, 0, sizeof(RCRE_Tensor));
        return t;
    }

    return t;
}

RCRE_NODISCARD RCRE_Tensor rcre_tensor_from_buffer(void* data, const size_t* shape, 
                                                 uint8_t ndim, RCRE_DType dtype, 
                                                 RCRE_Layout layout) {
    RCRE_Tensor t = {0};
    if (!data || ndim == 0 || ndim > RCRE_MAX_DIMS) {
        return t;
    }

    t.data = data;
    t.ndim = ndim;
    t.dtype = dtype;
    t.layout = layout;
    memcpy(t.shape, shape, ndim * sizeof(size_t));
    compute_contiguous_strides(shape, ndim, t.strides);
    t.numel = rcre_tensor_numel(&t);
    t.is_view = true;  /* borrows caller-owned memory */

    return t;
}

RCRE_NODISCARD RCRE_Tensor rcre_tensor_slice(const RCRE_Tensor* src, 
                                            const size_t* starts, 
                                            const size_t* ends) {
    RCRE_Tensor dst = {0};
    if (!src || !src->data || !starts || !ends) {
        return dst;
    }

    dst = *src;
    size_t offset_elements = 0;
    size_t elem_size = rcre_dtype_size(src->dtype);

    for (uint8_t i = 0; i < src->ndim; ++i) {
        if (starts[i] > ends[i] || ends[i] > src->shape[i]) {
            memset(&dst, 0, sizeof(RCRE_Tensor));
            return dst; /* Invalid slice bounds */
        }
        dst.shape[i] = ends[i] - starts[i];
        offset_elements += starts[i] * (size_t)src->strides[i];
    }

    dst.data = (uint8_t*)src->data + (offset_elements * elem_size);
    dst.numel = rcre_tensor_numel(&dst);
    dst.grad = nullptr; /* parent grad buffer is indexed in parent coordinates */
    dst.is_view = true;
    return dst;
}

RCRE_NODISCARD RCRE_Tensor rcre_tensor_reshape(const RCRE_Tensor* src, 
                                              const size_t* new_shape, 
                                              uint8_t new_ndim) {
    RCRE_Tensor dst = {0};
    if (!src || !src->data || !new_shape || new_ndim == 0 || new_ndim > RCRE_MAX_DIMS) {
        return dst;
    }

    /* Reshape requires memory to be contiguous */
    if (!rcre_tensor_is_contiguous(src)) {
        return dst;
    }

    /* Verify element counts match */
    size_t new_numel = 1;
    for (uint8_t i = 0; i < new_ndim; ++i) {
        new_numel *= new_shape[i];
    }

    if (new_numel != rcre_tensor_numel(src)) {
        return dst; /* Incompatible volume */
    }

    dst = *src;
    dst.ndim = new_ndim;
    memcpy(dst.shape, new_shape, new_ndim * sizeof(size_t));
    compute_contiguous_strides(new_shape, new_ndim, dst.strides);
    dst.grad = nullptr; /* layout changed: parent grad is not addressable */
    dst.is_view = true;

    return dst;
}

RCRE_NODISCARD RCRE_Tensor rcre_tensor_permute(const RCRE_Tensor* src, 
                                              const uint8_t* order) {
    RCRE_Tensor dst = {0};
    if (!src || !src->data || !order) {
        return dst;
    }

    /* Verify order is a valid permutation */
    bool seen[RCRE_MAX_DIMS] = {false};
    for (uint8_t i = 0; i < src->ndim; ++i) {
        if (order[i] >= src->ndim || seen[order[i]]) {
            return dst;
        }
        seen[order[i]] = true;
    }

    dst = *src;
    for (uint8_t i = 0; i < src->ndim; ++i) {
        dst.shape[i] = src->shape[order[i]];
        dst.strides[i] = src->strides[order[i]];
    }
    dst.grad = nullptr; /* strides changed: parent grad is not addressable */
    dst.is_view = true;

    return dst;
}

RCRE_NODISCARD RCRE_Tensor rcre_tensor_transpose(const RCRE_Tensor* src, 
                                                uint8_t dim0, uint8_t dim1) {
    RCRE_Tensor dst = {0};
    if (!src || !src->data || dim0 >= src->ndim || dim1 >= src->ndim) {
        return dst;
    }

    dst = *src;
    size_t temp_shape = dst.shape[dim0];
    dst.shape[dim0] = dst.shape[dim1];
    dst.shape[dim1] = temp_shape;

    ptrdiff_t temp_stride = dst.strides[dim0];
    dst.strides[dim0] = dst.strides[dim1];
    dst.strides[dim1] = temp_stride;
    dst.grad = nullptr; /* strides changed: parent grad is not addressable */
    dst.is_view = true;

    return dst;
}

/* ========================================================================= */
/* NumPy-Style Broadcasting (up to RCRE_MAX_DIMS = 8)                        */
/* ========================================================================= */

RCRE_NODISCARD int rcre_tensor_broadcast_shape(
    const size_t* shape_a, uint8_t ndim_a,
    const size_t* shape_b, uint8_t ndim_b,
    size_t out_shape[RCRE_MAX_DIMS], uint8_t* out_ndim
) {
    if (!shape_a || !shape_b || !out_shape || !out_ndim ||
        ndim_a == 0 || ndim_b == 0 || ndim_a > RCRE_MAX_DIMS || ndim_b > RCRE_MAX_DIMS) {
        return RCRE_ERR_INVALID_ARG;
    }

    uint8_t nd = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    memset(out_shape, 0, sizeof(size_t) * RCRE_MAX_DIMS);

    for (uint8_t k = 0; k < nd; ++k) {
        int ia = (int)ndim_a - 1 - (int)k;
        int ib = (int)ndim_b - 1 - (int)k;
        size_t da = (ia >= 0) ? shape_a[ia] : 1;
        size_t db = (ib >= 0) ? shape_b[ib] : 1;
        if (da != db && da != 1 && db != 1) {
            return RCRE_ERR_SHAPE_MISMATCH;
        }
        out_shape[nd - 1 - k] = (da > db) ? da : db;
    }
    *out_ndim = nd;
    return RCRE_OK;
}

RCRE_NODISCARD RCRE_Tensor rcre_tensor_broadcast_to(
    const RCRE_Tensor* src,
    const size_t* target_shape,
    uint8_t target_ndim
) {
    RCRE_Tensor dst = {0};
    if (!src || !src->data || !target_shape ||
        src->ndim == 0 || src->ndim > RCRE_MAX_DIMS ||
        target_ndim < src->ndim || target_ndim > RCRE_MAX_DIMS) {
        return dst;
    }

    dst.data    = src->data;
    dst.dtype   = src->dtype;
    dst.layout  = src->layout;
    dst.ndim    = target_ndim;
    dst.is_view = true;
    dst.grad    = nullptr;

    uint8_t pad = target_ndim - src->ndim;
    size_t numel = 1;
    for (uint8_t i = 0; i < target_ndim; ++i) {
        size_t td = target_shape[i];
        if (td == 0) {
            memset(&dst, 0, sizeof(RCRE_Tensor));
            return dst;
        }
        dst.shape[i] = td;
        numel *= td;
        if (i < pad) {
            dst.strides[i] = 0;
        } else {
            size_t sd = src->shape[i - pad];
            if (sd == td) {
                dst.strides[i] = src->strides[i - pad];
            } else if (sd == 1) {
                dst.strides[i] = 0;
            } else {
                memset(&dst, 0, sizeof(RCRE_Tensor));
                return dst;
            }
        }
    }
    dst.numel = numel;
    return dst;
}

static int rcre_tensor_binary_broadcast(
    const RCRE_Tensor* a, const RCRE_Tensor* b,
    RCRE_Arena* arena, RCRE_Tensor* dst, bool is_mul
) {
    if (!a || !b || !a->data || !b->data || !arena || !dst ||
        a->dtype != RCRE_DTYPE_F32 || b->dtype != RCRE_DTYPE_F32) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t out_shape[RCRE_MAX_DIMS] = {0};
    uint8_t out_ndim = 0;
    int rc = rcre_tensor_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim);
    if (rc != RCRE_OK) return rc;

    RCRE_Tensor va = rcre_tensor_broadcast_to(a, out_shape, out_ndim);
    RCRE_Tensor vb = rcre_tensor_broadcast_to(b, out_shape, out_ndim);
    if (!va.data || !vb.data) return RCRE_ERR_SHAPE_MISMATCH;

    *dst = rcre_tensor_create(out_shape, out_ndim, RCRE_DTYPE_F32, a->layout, arena);
    if (!dst->data) return RCRE_ERR_OUT_OF_MEMORY;

    const float* pa = (const float*)va.data;
    const float* pb = (const float*)vb.data;
    float* pc = (float*)dst->data;
    size_t n = dst->numel;

    size_t coords[RCRE_MAX_DIMS] = {0};
    ptrdiff_t off_a = 0;
    ptrdiff_t off_b = 0;

    for (size_t idx = 0; idx < n; ++idx) {
        pc[idx] = is_mul ? (pa[off_a] * pb[off_b]) : (pa[off_a] + pb[off_b]);
        for (int d = (int)out_ndim - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < out_shape[d]) {
                off_a += va.strides[d];
                off_b += vb.strides[d];
                break;
            }
            off_a -= (ptrdiff_t)(coords[d] - 1) * va.strides[d];
            off_b -= (ptrdiff_t)(coords[d] - 1) * vb.strides[d];
            coords[d] = 0;
        }
    }
    return RCRE_OK;
}

RCRE_NODISCARD int rcre_tensor_add_broadcast(
    const RCRE_Tensor* a, const RCRE_Tensor* b,
    RCRE_Arena* arena, RCRE_Tensor* dst
) {
    return rcre_tensor_binary_broadcast(a, b, arena, dst, false);
}

RCRE_NODISCARD int rcre_tensor_mul_broadcast(
    const RCRE_Tensor* a, const RCRE_Tensor* b,
    RCRE_Arena* arena, RCRE_Tensor* dst
) {
    return rcre_tensor_binary_broadcast(a, b, arena, dst, true);
}
