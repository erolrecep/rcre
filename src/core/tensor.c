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
        case RCRE_DTYPE_U8:  return 1;
        case RCRE_DTYPE_I32: return 4;
        case RCRE_DTYPE_F32: return 4;
        default:             return 0;
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
    if (ndim == 0 || ndim > 4 || !arena) {
        return t;
    }

    t.ndim = ndim;
    t.dtype = dtype;
    t.layout = layout;
    memcpy(t.shape, shape, ndim * sizeof(size_t));
    compute_contiguous_strides(shape, ndim, t.strides);

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
    if (!data || ndim == 0 || ndim > 4) {
        return t;
    }

    t.data = data;
    t.ndim = ndim;
    t.dtype = dtype;
    t.layout = layout;
    memcpy(t.shape, shape, ndim * sizeof(size_t));
    compute_contiguous_strides(shape, ndim, t.strides);

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
    return dst;
}

RCRE_NODISCARD RCRE_Tensor rcre_tensor_reshape(const RCRE_Tensor* src, 
                                              const size_t* new_shape, 
                                              uint8_t new_ndim) {
    RCRE_Tensor dst = {0};
    if (!src || !src->data || !new_shape || new_ndim == 0 || new_ndim > 4) {
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

    return dst;
}

RCRE_NODISCARD RCRE_Tensor rcre_tensor_permute(const RCRE_Tensor* src, 
                                              const uint8_t* order) {
    RCRE_Tensor dst = {0};
    if (!src || !src->data || !order) {
        return dst;
    }

    /* Verify order is a valid permutation */
    bool seen[4] = {false};
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

    return dst;
}
